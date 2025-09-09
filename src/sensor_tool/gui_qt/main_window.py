"""
Main application window for PySide6 implementation.

This module contains the main window class that serves as the primary interface
for SensorSphere, featuring dockable panels, standard menus,
and integrated functionality.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QToolBar,
)

# Import enhanced backend functionality
from .models.qt_adapter import QtSensorRepository

# Import font management system
from .utils.font_manager import create_styled_font, get_font_manager
from .widgets.chart_widget import ChartWidget
from .widgets.comparison_table import SensorComparisonTable
from .widgets.detail_panel import SensorDetailWidget

# Import Qt-specific widgets
from .widgets.filter_panel import AdvancedFilterWidget

logger = logging.getLogger(__name__)


class SensorComparisonMainWindow(QMainWindow):
    """
    Main application window with standard interface and docking panels.

    Features:
    - Menu bar with File, View, Tools, Help menus
    - Toolbar with common actions
    - Status bar with sensor count and operation status
    - Dockable filter panel (left)
    - Central comparison table
    - Dockable detail panel (right)
    - Modern styling and layout management
    """

    # Signals for inter-component communication
    sensors_loaded = Signal(list)  # Emitted when sensors are loaded
    filter_applied = Signal(dict)  # Emitted when filters are applied
    sensor_selected = Signal(str)  # Emitted when a sensor is selected

    def __init__(self) -> None:
        """Initialize main window with dockable panels and standard interface."""
        super().__init__()

        # Initialize enhanced repository
        self.qt_repository = QtSensorRepository("sensors")
        self.sensors_data = []
        self.filtered_sensors = []
        self.selected_sensors = []
        self.is_loading = False
        self.is_initial_load = (
            True  # Track if this is initial data load vs filtered results
        )

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize font management
        self.font_manager = get_font_manager()

        # Initialize UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_status_bar()
        self.setup_docking()

        # Connect signals
        self.connect_signals()

        # Load initial data
        self.load_sensor_data()

    def setup_ui(self) -> None:
        """Initialize main window geometry and basic UI properties."""
        self.setWindowTitle("SensorSphere")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)

        # Central widget will be set up in setup_docking()

    def setup_menu_bar(self) -> None:
        """Create comprehensive menu bar with File, View, Tools, and Help menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Load sensors action
        load_action = QAction("&Load Sensors...", self)
        load_action.setShortcut(QKeySequence.Open)
        load_action.setStatusTip("Load sensor data from files")
        load_action.triggered.connect(self.load_sensor_files)
        file_menu.addAction(load_action)

        # Refresh action
        refresh_action = QAction("&Refresh Data", self)
        refresh_action.setShortcut(QKeySequence.Refresh)
        refresh_action.setStatusTip("Refresh sensor data from disk")
        refresh_action.triggered.connect(self.refresh_sensor_data)
        file_menu.addAction(refresh_action)

        file_menu.addSeparator()

        # Export actions
        export_menu = file_menu.addMenu("&Export")

        export_csv_action = QAction("Export as &CSV...", self)
        export_csv_action.triggered.connect(lambda: self.export_data("csv"))
        export_menu.addAction(export_csv_action)

        export_json_action = QAction("Export as &JSON...", self)
        export_json_action.triggered.connect(lambda: self.export_data("json"))
        export_menu.addAction(export_json_action)

        export_yaml_action = QAction("Export as &YAML...", self)
        export_yaml_action.triggered.connect(lambda: self.export_data("yaml"))
        export_menu.addAction(export_yaml_action)

        export_menu.addSeparator()

        # PDF Export action
        export_pdf_action = QAction("Export as &PDF Report...", self)
        export_pdf_action.setShortcut("Ctrl+P")
        export_pdf_action.triggered.connect(self.export_pdf_report)
        export_menu.addAction(export_pdf_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Ctrl+C exit action for terminal-like behavior
        ctrl_c_exit_action = QAction("Force Exit", self)
        ctrl_c_exit_action.setShortcut("Ctrl+C")
        ctrl_c_exit_action.setStatusTip("Force exit the application (Ctrl+C)")
        ctrl_c_exit_action.triggered.connect(self.force_close)
        # Don't add to menu - this is just for keyboard shortcut
        self.addAction(ctrl_c_exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Panel visibility toggles (will be connected to dock widgets)
        self.filter_panel_action = QAction("&Filter Panel", self)
        self.filter_panel_action.setCheckable(True)
        self.filter_panel_action.setChecked(True)
        view_menu.addAction(self.filter_panel_action)

        self.detail_panel_action = QAction("&Detail Panel", self)
        self.detail_panel_action.setCheckable(True)
        self.detail_panel_action.setChecked(True)
        view_menu.addAction(self.detail_panel_action)

        self.chart_panel_action = QAction("&Chart Panel", self)
        self.chart_panel_action.setCheckable(True)
        self.chart_panel_action.setChecked(True)
        view_menu.addAction(self.chart_panel_action)

        view_menu.addSeparator()

        # Theme toggle
        self.theme_toggle_action = QAction("Toggle &Dark Mode", self)
        self.theme_toggle_action.setShortcut("Ctrl+Shift+T")
        self.theme_toggle_action.setStatusTip("Toggle between light and dark themes")
        self.theme_toggle_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.theme_toggle_action)

        view_menu.addSeparator()

        # Reset layout action
        reset_layout_action = QAction("&Reset Layout", self)
        reset_layout_action.setStatusTip("Reset window layout to default")
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        # Validate sensors action
        validate_action = QAction("&Validate Sensor Data", self)
        validate_action.setStatusTip("Validate all sensor data files")
        validate_action.triggered.connect(self.validate_sensor_data)
        tools_menu.addAction(validate_action)

        # Settings action
        settings_action = QAction("&Settings...", self)
        settings_action.setStatusTip("Open application settings")
        settings_action.triggered.connect(self.open_settings)
        tools_menu.addAction(settings_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        # User manual action
        manual_action = QAction("&User Manual", self)
        manual_action.setShortcut(QKeySequence.HelpContents)
        manual_action.triggered.connect(self.open_user_manual)
        help_menu.addAction(manual_action)

        # GitHub issue action
        github_action = QAction("Report &Issue on GitHub", self)
        github_action.setStatusTip("Report a bug or request a feature on GitHub")
        github_action.triggered.connect(self.open_github_issue)
        help_menu.addAction(github_action)

        help_menu.addSeparator()

        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_toolbar(self) -> None:
        """Create toolbar with common actions for quick access."""
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)

        # Set initial toolbar font - use larger toolbar style
        toolbar_font = create_styled_font("toolbar", "medium")
        toolbar.setFont(toolbar_font)

        # Load sensors button
        load_action = QAction("Load", self)
        load_action.setStatusTip("Load sensor database from YAML files")
        load_action.setToolTip(
            "Reload sensor database from disk - useful when sensor files are updated"
        )
        load_action.triggered.connect(self.load_sensor_files)
        toolbar.addAction(load_action)

        # Refresh button
        refresh_action = QAction("Refresh", self)
        refresh_action.setStatusTip("Refresh current sensor view and filters")
        refresh_action.setToolTip("Refresh the current view and reapply filters")
        refresh_action.triggered.connect(self.refresh_sensor_data)
        toolbar.addAction(refresh_action)

        toolbar.addSeparator()

        # Export button
        export_action = QAction("Export", self)
        export_action.setStatusTip("Export sensor comparison data and charts")
        export_action.setToolTip(
            "Export selected sensors to CSV, PDF report, or chart images"
        )
        export_action.triggered.connect(self.show_export_dialog)
        toolbar.addAction(export_action)

        toolbar.addSeparator()

        # Settings button
        settings_action = QAction("Settings", self)
        settings_action.setStatusTip("Open application settings and preferences")
        settings_action.setToolTip(
            "Configure display options, export defaults, and application behavior"
        )
        settings_action.triggered.connect(self.open_settings)
        toolbar.addAction(settings_action)

        toolbar.addSeparator()

        # Theme toggle button - quick access for dark/light mode switching
        self.theme_toggle_toolbar_action = QAction(
            "🌙", self
        )  # Moon icon for dark mode toggle
        self.theme_toggle_toolbar_action.setStatusTip(
            "Toggle between light and dark themes"
        )
        self.theme_toggle_toolbar_action.setToolTip(
            "Switch between light and dark mode for better visibility in different lighting conditions"
        )
        self.theme_toggle_toolbar_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(self.theme_toggle_toolbar_action)

        # GitHub issues button - quick access to report issues
        github_toolbar_action = QAction("🐛", self)  # Bug icon for GitHub issues
        github_toolbar_action.setStatusTip(
            "Report bugs, suggest features, or get help on GitHub"
        )
        github_toolbar_action.setToolTip(
            "Click to open GitHub issues page for bug reports, feature requests, and support"
        )
        github_toolbar_action.triggered.connect(self.open_github_issue)
        toolbar.addAction(github_toolbar_action)

    def setup_status_bar(self) -> None:
        """Create status bar with sensor count, progress indicator, and messages."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Sensor count label
        self.sensor_count_label = QLabel("Sensors: 0")
        self.status_bar.addWidget(self.sensor_count_label)

        # Progress bar for long operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Status message
        self.status_bar.showMessage("Ready")

    def setup_docking(self) -> None:
        """Configure dockable panels for filters, details, and charts around central table."""
        # Create central comparison table
        self.comparison_table = SensorComparisonTable()
        self.setCentralWidget(self.comparison_table)

        # Create filter panel dock
        self.filter_dock = QDockWidget("Filter Panel", self)
        self.filter_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        # Set dock title font
        self.filter_dock.setFont(create_styled_font("dock_title", "medium"))

        # Advanced filter widget
        self.filter_widget = AdvancedFilterWidget()
        self.filter_dock.setWidget(self.filter_widget)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.filter_dock)

        # Create detail panel dock
        self.detail_dock = QDockWidget("Sensor Details", self)
        self.detail_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.detail_dock.setFont(create_styled_font("dock_title", "medium"))

        # Sensor detail widget
        self.detail_widget = SensorDetailWidget()
        self.detail_dock.setWidget(self.detail_widget)

        self.addDockWidget(Qt.RightDockWidgetArea, self.detail_dock)

        # Create chart panel dock
        self.chart_dock = QDockWidget("Chart Panel", self)
        self.chart_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.chart_dock.setFont(create_styled_font("dock_title", "medium"))

        # Charting widget
        self.chart_widget = ChartWidget()
        self.chart_dock.setWidget(self.chart_widget)

        self.addDockWidget(Qt.BottomDockWidgetArea, self.chart_dock)

        # Enable dock features for better user control
        self.filter_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.detail_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.chart_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )

        # Set initial dock sizes with standard proportions
        self.resizeDocks([self.filter_dock], [300], Qt.Horizontal)
        self.resizeDocks([self.detail_dock], [400], Qt.Horizontal)

        # Calculate standard chart panel height (40% of window height minus toolbar/menubar)
        # Assume initial window height of 900px, subtract 150px for menu/toolbar/status
        initial_chart_height = int((900 - 150) * 0.4)  # 40% of available vertical space
        self.resizeDocks([self.chart_dock], [initial_chart_height], Qt.Vertical)

        # Apply standard splitter styling for better UX
        self.setup_standard_splitters()

        # Connect dock visibility to menu actions
        self.filter_panel_action.toggled.connect(self.filter_dock.setVisible)
        self.detail_panel_action.toggled.connect(self.detail_dock.setVisible)
        self.chart_panel_action.toggled.connect(self.chart_dock.setVisible)

        self.filter_dock.visibilityChanged.connect(self.filter_panel_action.setChecked)
        self.detail_dock.visibilityChanged.connect(self.detail_panel_action.setChecked)
        self.chart_dock.visibilityChanged.connect(self.chart_panel_action.setChecked)

    def setup_standard_splitters(self) -> None:
        """Apply standard styling to splitter handles for better UX."""
        from .utils.theme_manager import get_theme_manager

        theme_manager = get_theme_manager()
        colors = theme_manager.get_stylesheet_colors()

        # Modern splitter styling with clear visual indicators
        splitter_style = f"""
            QMainWindow::separator {{
                background: {colors['surface']};
                width: 8px;
                height: 8px;
                border: 1px solid {colors['border']};
                border-radius: 2px;
            }}

            QMainWindow::separator:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 {colors['surface']},
                    stop: 0.5 {colors['accent']},
                    stop: 1 {colors['surface']});
                border: 1px solid {colors['accent']};
            }}

            QMainWindow::separator:horizontal {{
                width: 8px;
                margin: 0px 2px;
            }}

            QMainWindow::separator:vertical {{
                height: 8px;
                margin: 2px 0px;
            }}

            QMainWindow::separator:horizontal:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {colors['surface']},
                    stop: 0.3 {colors['accent']},
                    stop: 0.7 {colors['accent']},
                    stop: 1 {colors['surface']});
            }}

            QMainWindow::separator:vertical:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {colors['surface']},
                    stop: 0.3 {colors['accent']},
                    stop: 0.7 {colors['accent']},
                    stop: 1 {colors['surface']});
            }}
        """

        # Apply the styling to the main window
        current_style = self.styleSheet()
        self.setStyleSheet(current_style + splitter_style)

    def connect_signals(self) -> None:
        """Connect internal signals for component communication."""
        try:
            # Connect internal signals
            self.sensors_loaded.connect(self.on_sensors_loaded)
            self.filter_applied.connect(self.on_filter_applied)
            self.sensor_selected.connect(self.on_sensor_selected)

            # Connect enhanced repository signals with error handling
            if self.qt_repository:
                try:
                    self.qt_repository.sensors_loaded.connect(
                        self.on_repository_sensors_loaded
                    )
                    self.qt_repository.operation_started.connect(
                        self.on_repository_operation_started
                    )
                    self.qt_repository.operation_progress.connect(
                        self.on_repository_progress
                    )
                    self.qt_repository.operation_finished.connect(
                        self.on_repository_operation_finished
                    )
                    self.qt_repository.operation_error.connect(self.on_repository_error)
                except AttributeError as e:
                    self.logger.error(f"Repository signal connection error: {e}")

            # Connect widget signals with error handling
            try:
                if hasattr(self, "filter_widget") and self.filter_widget:
                    self.filter_widget.filter_changed.connect(
                        self.on_widget_filter_changed
                    )
                    self.filter_widget.clear_filters.connect(
                        self.on_widget_filters_cleared
                    )
            except (AttributeError, RuntimeError) as e:
                self.logger.error(f"Filter widget signal connection error: {e}")

            try:
                if hasattr(self, "comparison_table") and self.comparison_table:
                    self.comparison_table.sensor_selected.connect(
                        self.on_widget_sensor_selected
                    )
                    self.comparison_table.selection_changed.connect(
                        self.on_widget_selection_changed
                    )
                    self.comparison_table.selection_changed.connect(
                        self.on_chart_selection_changed
                    )
            except (AttributeError, RuntimeError) as e:
                self.logger.error(f"Comparison table signal connection error: {e}")

            try:
                if hasattr(self, "detail_widget") and self.detail_widget:
                    self.detail_widget.export_requested.connect(
                        self.on_detail_export_requested
                    )
            except (AttributeError, RuntimeError) as e:
                self.logger.error(f"Detail widget signal connection error: {e}")

            # Connect font management signals
            try:
                if hasattr(self, "font_manager") and self.font_manager:
                    self.font_manager.fonts_updated.connect(self.on_fonts_updated)
            except (AttributeError, RuntimeError) as e:
                self.logger.error(f"Font manager signal connection error: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error connecting signals: {e}")
            # Don't raise here as this would break application initialization

    def load_sensor_data(self) -> None:
        """Load sensor data using enhanced repository."""
        if self.is_loading:
            self.logger.warning("Data loading already in progress")
            return

        try:
            # Validate repository state before loading
            if not self.qt_repository:
                error_msg = "Sensor repository not initialized"
                self.logger.error(error_msg)
                self.status_bar.showMessage("Repository not initialized")
                QMessageBox.critical(self, "Initialization Error", error_msg)
                return

            # Start async loading via Qt repository
            self.qt_repository.load_sensors_async(force_reload=False)
            self.logger.info("Initiated sensor data loading")

        except AttributeError as e:
            error_msg = f"Repository interface error: {e}"
            self.logger.error(error_msg)
            self.status_bar.showMessage("Repository interface error")
            QMessageBox.critical(self, "Interface Error", error_msg)
        except RuntimeError as e:
            error_msg = f"Runtime error during sensor loading: {e}"
            self.logger.error(error_msg)
            self.status_bar.showMessage("Runtime error during loading")
            QMessageBox.critical(self, "Runtime Error", error_msg)
        except Exception as e:
            error_msg = f"Failed to start sensor data loading: {e}"
            self.logger.error(error_msg)
            self.status_bar.showMessage("Failed to load sensor data")
            QMessageBox.critical(
                self, "Error", f"Failed to load sensor data:\n{str(e)}"
            )

    def on_sensors_loaded(self, sensors: List[Dict[str, Any]]) -> None:
        """Handle sensors loaded signal."""
        self.sensor_count_label.setText(f"Sensors: {len(sensors)}")

    def on_filter_applied(self, criteria: Dict[str, Any]) -> None:
        """Handle filter applied signal (internal)."""
        # This method handles internal filter_applied signals
        # Widget filters are handled by on_widget_filter_changed
        if self.is_loading:
            self.logger.warning("Cannot filter while loading data")
            return

        try:
            # Update filtered sensors for backward compatibility
            self.filtered_sensors = [
                s
                for s in self.sensors_data
                if self._matches_filter_criteria(s, criteria)
            ]

            # Update comparison table with filtered data
            self.comparison_table.update_sensors_data(self.filtered_sensors)

        except Exception as e:
            self.logger.error(f"Failed to apply filters: {e}")
            QMessageBox.warning(
                self, "Filter Error", f"Failed to apply filters:\n{str(e)}"
            )

    def on_sensor_selected(self, sensor_id: str) -> None:
        """Handle sensor selection signal."""
        self.status_bar.showMessage(f"Selected sensor: {sensor_id}")

    # Enhanced repository signal handlers
    def on_repository_sensors_loaded(self, sensors: List[Dict[str, Any]]) -> None:
        """Handle sensors loaded from enhanced repository."""
        self.sensors_data = sensors
        self.filtered_sensors = sensors.copy()

        # Update widgets with new data
        self.comparison_table.update_sensors_data(sensors)

        # Only update price range on initial load, not for filtered results
        # This prevents the circular update that resets user's price selections
        self.filter_widget.update_filter_options(
            sensors, update_price_range=self.is_initial_load
        )

        # Emit internal signal for any connected widgets
        self.sensors_loaded.emit(sensors)

        # After initial load, subsequent loads are filtered results
        if self.is_initial_load:
            self.is_initial_load = False

        self.logger.info(f"Enhanced repository loaded {len(sensors)} sensors")

    def on_repository_operation_started(self, operation: str) -> None:
        """Handle repository operation started."""
        self.is_loading = True
        self.status_bar.showMessage(operation)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

    def on_repository_progress(self, message: str, percent: int) -> None:
        """Handle repository operation progress."""
        self.status_bar.showMessage(message)
        if percent >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percent)

    def on_repository_operation_finished(self) -> None:
        """Handle repository operation finished."""
        self.is_loading = False
        self.progress_bar.setVisible(False)

        # Update sensor count in status bar
        sensor_count = len(self.sensors_data)
        filtered_count = len(self.filtered_sensors)

        if sensor_count == filtered_count:
            self.sensor_count_label.setText(f"Sensors: {sensor_count}")
            self.status_bar.showMessage(f"Loaded {sensor_count} sensors")
        else:
            self.sensor_count_label.setText(
                f"Sensors: {filtered_count} (of {sensor_count})"
            )
            self.status_bar.showMessage(f"Filtered to {filtered_count} sensors")

    def on_repository_error(self, error_message: str) -> None:
        """Handle repository operation error."""
        self.is_loading = False
        self.progress_bar.setVisible(False)

        self.status_bar.showMessage(f"Error: {error_message}")
        self.logger.error(f"Repository error: {error_message}")

        QMessageBox.critical(
            self, "Repository Error", f"Failed to load sensor data:\n{error_message}"
        )

    # Widget signal handlers
    def on_widget_filter_changed(self, filter_criteria: Dict[str, Any]) -> None:
        """Handle filter changes from filter widget."""
        if self.is_loading:
            return

        try:
            # Validate filter criteria
            if not isinstance(filter_criteria, dict):
                self.logger.warning("Invalid filter criteria - expected dictionary")
                return

            # Apply filters via repository
            if self.qt_repository:
                self.qt_repository.filter_sensors_async(**filter_criteria)
            else:
                self.logger.error("Repository not available for filtering")
                self.status_bar.showMessage("Repository not available")

        except TypeError as e:
            error_msg = f"Invalid filter parameters: {e}"
            self.logger.error(error_msg)
            self.status_bar.showMessage("Invalid filter parameters")
        except AttributeError as e:
            error_msg = f"Repository interface error during filtering: {e}"
            self.logger.error(error_msg)
            self.status_bar.showMessage("Repository interface error")
        except Exception as e:
            error_msg = f"Failed to apply widget filters: {e}"
            self.logger.error(error_msg)
            self.status_bar.showMessage(f"Filter error: {e}")
            # Show user-friendly message for filter errors
            QMessageBox.warning(
                self,
                "Filter Error",
                f"Unable to apply filters. Please try again.\n\nDetails: {str(e)}",
            )

    def on_widget_filters_cleared(self) -> None:
        """Handle filter clearing from filter widget."""
        # Set flag to force full range update on clear
        self.is_initial_load = True
        # Reload all sensors (no filters)
        self.qt_repository.load_sensors_async(force_reload=False)

    def on_widget_sensor_selected(self, sensor_id: str) -> None:
        """Handle sensor selection from comparison table."""
        # Find sensor data
        sensor_data = None
        for sensor in self.sensors_data:
            if sensor.get("sensor_id") == sensor_id:
                sensor_data = sensor
                break

        if sensor_data:
            # Update detail panel
            self.detail_widget.show_sensor_details(sensor_data)

            # Emit internal signal
            self.sensor_selected.emit(sensor_id)
        else:
            self.logger.warning(f"Sensor data not found for ID: {sensor_id}")

    def on_widget_selection_changed(self, selected_sensor_ids: List[str]) -> None:
        """Handle selection changes from comparison table."""
        self.selected_sensors = selected_sensor_ids.copy()

        # Update detail panel based on selection
        if len(selected_sensor_ids) == 1:
            # Show details for single selected sensor
            self.on_widget_sensor_selected(selected_sensor_ids[0])
        elif len(selected_sensor_ids) == 0:
            # Clear detail panel
            self.detail_widget.show_empty_state()
        else:
            # Multiple selection - could show comparison view
            self.detail_widget.show_empty_state()

    def on_chart_selection_changed(self, selected_sensor_ids: List[str]) -> None:
        """Handle selection changes for chart updates."""
        if len(selected_sensor_ids) >= 2:
            # Get full sensor data for selected sensors
            selected_data = []
            for sensor_id in selected_sensor_ids:
                # Find sensor in current filtered data
                for sensor in getattr(self, "filtered_sensors", self.sensors_data):
                    if sensor.get("sensor_id") == sensor_id:
                        selected_data.append(sensor)
                        break

            if selected_data:
                self.chart_widget.update_sensors_data(selected_data)
                logger.info(f"Updated chart widget with {len(selected_data)} sensors")

    def on_detail_export_requested(self, sensor_id: str) -> None:
        """Handle export request from detail panel."""
        QMessageBox.information(
            self,
            "Export Sensor",
            f"Export functionality for sensor '{sensor_id}' will be implemented.",
        )

    def _matches_filter_criteria(
        self, sensor: Dict[str, Any], criteria: Dict[str, Any]
    ) -> bool:
        """Check if sensor matches filter criteria (helper method)."""
        try:
            for key, value in criteria.items():
                sensor_value = sensor.get(key)

                if key == "manufacturer" and sensor_value:
                    if value.lower() not in sensor_value.lower():
                        return False
                elif key == "sensor_type" and sensor_value:
                    if value.lower() not in sensor_value.lower():
                        return False
                elif key == "model_search":
                    # Enhanced search across multiple fields for better user experience
                    search_term = value.lower()
                    model = sensor.get("model", "").lower()
                    manufacturer = sensor.get("manufacturer", "").lower()
                    sensor_id = sensor.get("sensor_id", "").lower()

                    # Search fails if term not found in any of these fields
                    if not any(
                        [
                            search_term in model,
                            search_term in manufacturer,
                            search_term in sensor_id,
                        ]
                    ):
                        return False
                elif key == "ros_compatibility" and isinstance(value, list):
                    sensor_ros = sensor.get("ros_compatibility", [])
                    if not any(ros_version in sensor_ros for ros_version in value):
                        return False

            return True
        except Exception:
            return False

    # Menu action handlers
    def load_sensor_files(self) -> None:
        """Handle load sensor files action."""
        QMessageBox.information(
            self, "Load Sensors", "Load sensor files functionality will be implemented."
        )

    def refresh_sensor_data(self) -> None:
        """Handle refresh sensor data action."""
        if self.is_loading:
            self.logger.warning("Data loading already in progress")
            return

        try:
            # Set flag to force full range update on refresh
            self.is_initial_load = True
            # Force reload from enhanced repository
            self.qt_repository.load_sensors_async(force_reload=True)

        except Exception as e:
            self.logger.error(f"Failed to refresh sensor data: {e}")
            self.status_bar.showMessage("Failed to refresh sensor data")
            QMessageBox.critical(
                self, "Error", f"Failed to refresh sensor data:\n{str(e)}"
            )

    def export_data(self, format_type: str) -> None:
        """Handle export data actions."""
        QMessageBox.information(
            self,
            "Export",
            f"Export to {format_type.upper()} functionality will be implemented.",
        )

    def export_pdf_report(self) -> None:
        """Export PDF report using the integrated PDF export widget."""
        if len(self.selected_sensors) < 2:
            QMessageBox.warning(
                self,
                "PDF Export",
                "Please select at least 2 sensors in the comparison table to generate a PDF report.",
            )
            return

        # Switch to PDF Export tab and auto-configure
        self.chart_widget.tab_widget.setCurrentWidget(
            self.chart_widget.pdf_export_widget
        )

        # Show informational message
        QMessageBox.information(
            self,
            "PDF Export",
            f"PDF Export tab activated with {len(self.selected_sensors)} selected sensors.\n\n"
            "Configure your export options in the PDF Export tab and click 'Generate PDF Report'.",
        )

    def show_export_dialog(self) -> None:
        """Show the export dialog."""
        QMessageBox.information(self, "Export", "Export dialog will be implemented.")

    def validate_sensor_data(self) -> None:
        """Handle validate sensor data action."""
        try:
            # Get repository stats for validation info
            stats = self.qt_repository.get_repository_stats()

            cache_stats = stats.get("cache", {})
            perf_stats = stats.get("performance", {})

            validation_info = [
                "Repository Status:",
                f"• Sensors loaded: {cache_stats.get('sensor_count', 0)}",
                f"• Files tracked: {cache_stats.get('file_count', 0)}",
                f"• Memory usage: {cache_stats.get('memory_usage_mb', 0):.2f} MB",
                "",
                "Performance Stats:",
                f"• Total loads: {perf_stats.get('total_loads', 0)}",
                f"• Cache hits: {perf_stats.get('cache_hits', 0)}",
                f"• File reads: {perf_stats.get('file_reads', 0)}",
                f"• Errors: {perf_stats.get('errors', 0)}",
            ]

            QMessageBox.information(
                self, "Repository Validation", "\n".join(validation_info)
            )

        except Exception as e:
            self.logger.error(f"Failed to get validation info: {e}")
            QMessageBox.warning(
                self, "Validation Error", f"Failed to get validation info:\n{str(e)}"
            )

    def open_settings(self) -> None:
        """Handle open settings action."""
        QMessageBox.information(
            self, "Settings", "Settings dialog will be implemented."
        )

    def open_user_manual(self) -> None:
        """Handle open user manual action."""
        QMessageBox.information(self, "Help", "User manual will be implemented.")

    def open_github_issue(self) -> None:
        """Open GitHub issues page in default browser."""
        import webbrowser

        github_url = "https://github.com/Sahil-cmd/sensor-sphere/issues/new"
        try:
            webbrowser.open(github_url)
            self.status_bar.showMessage("Opened GitHub issues page in browser")
        except Exception as e:
            self.logger.error(f"Failed to open GitHub URL: {e}")
            QMessageBox.information(
                self,
                "GitHub Issues",
                f"Please visit: {github_url}\n\nTo report issues or request features.",
            )

    def show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About",
            "SensorSphere\n"
            "The Robot Sensor Hub and Selection Engine\n\n"
            "Version 1.0.0\n"
            "Built with PySide6 and Python",
        )

    def toggle_theme(self) -> None:
        """Toggle between light and dark themes."""
        from .utils.theme_manager import get_theme_manager

        theme_manager = get_theme_manager()
        theme_manager.toggle_theme()

        # Update UI elements based on current theme
        is_dark = theme_manager.is_dark_mode

        # Update menu text
        self.theme_toggle_action.setText(
            "Toggle &Light Mode" if is_dark else "Toggle &Dark Mode"
        )

        # Update toolbar button icon and tooltip
        self.theme_toggle_toolbar_action.setText(
            "☀️" if is_dark else "🌙"
        )  # Sun for light mode, Moon for dark mode
        self.theme_toggle_toolbar_action.setStatusTip(
            f"Switch to {'light' if is_dark else 'dark'} theme (Currently: {'dark' if is_dark else 'light'} mode)"
        )

        # Re-apply theme to ensure all components are updated
        theme_manager.apply_modern_theme_to_app()

        # Reapply standard splitter styling with new theme colors
        self.setup_standard_splitters()

        # Update fonts to match new theme
        self.on_fonts_updated()

        self.status_bar.showMessage(
            f"Switched to {'dark' if is_dark else 'light'} theme"
        )

    def reset_layout(self) -> None:
        """Reset window layout to default."""
        # Reset dock positions
        self.addDockWidget(Qt.LeftDockWidgetArea, self.filter_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.detail_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.chart_dock)

        # Show all docks
        self.filter_dock.setVisible(True)
        self.detail_dock.setVisible(True)
        self.chart_dock.setVisible(True)

        self.status_bar.showMessage("Layout reset to default")

    def resizeEvent(self, event: QEvent) -> None:
        """Handle window resize events to update font scaling and panel proportions."""
        super().resizeEvent(event)

        # Update font scaling based on new window size
        new_size = event.size()
        self.font_manager.update_scale_factor(new_size.width(), new_size.height())

        # Dynamically adjust chart panel height to maintain 40% proportion
        if hasattr(self, "chart_dock") and self.chart_dock:
            # Calculate 40% of available vertical space (excluding menu/toolbar/status ~150px)
            available_height = new_size.height() - 150
            target_chart_height = int(available_height * 0.4)

            # Only resize if the difference is significant (avoid constant micro-adjustments)
            current_height = self.chart_dock.height()
            if abs(current_height - target_chart_height) > 20:
                self.resizeDocks([self.chart_dock], [target_chart_height], Qt.Vertical)

    def on_fonts_updated(self) -> None:
        """Handle font updates by refreshing all widgets."""
        # Update fonts for all child widgets
        self.update_widget_fonts()

        # Force layout update
        self.update()

    def update_widget_fonts(self) -> None:
        """Update fonts for all widgets in the application."""
        try:
            # Update menu bar fonts
            menu_bar = self.menuBar()
            menu_font = create_styled_font("menu", "normal")
            menu_bar.setFont(menu_font)

            # Update dock widget titles with proper sizing
            if hasattr(self, "filter_dock"):
                self.filter_dock.setFont(create_styled_font("dock_title", "medium"))
            if hasattr(self, "detail_dock"):
                self.detail_dock.setFont(create_styled_font("dock_title", "medium"))
            if hasattr(self, "chart_dock"):
                self.chart_dock.setFont(create_styled_font("dock_title", "medium"))

            # Update main toolbar fonts for better visibility
            toolbar = self.findChild(QToolBar, "Main Toolbar")
            if toolbar:
                toolbar_font = create_styled_font("toolbar", "medium")
                toolbar.setFont(toolbar_font)
                # Also update all actions in the toolbar
                for action in toolbar.actions():
                    if hasattr(action, "setFont"):
                        action.setFont(toolbar_font)

            # Update comparison table fonts
            if hasattr(self, "comparison_table"):
                self.comparison_table.update_fonts()

            # Update detail panel fonts
            if hasattr(self, "detail_widget"):
                self.detail_widget.update_fonts()

            # Update filter panel fonts
            if hasattr(self, "filter_widget"):
                self.filter_widget.update_fonts()

        except Exception as e:
            self.logger.warning(f"Failed to update some widget fonts: {e}")

    def force_close(self) -> None:
        """Force close the application (Ctrl+C behavior)."""
        self.logger.info("Force close requested via Ctrl+C")
        # Cancel any running operations immediately
        if self.qt_repository.is_operation_running():
            self.qt_repository.cancel_operation()

        # Force quit the application
        from PySide6.QtWidgets import QApplication

        QApplication.instance().quit()

    def closeEvent(self, event: QEvent) -> None:
        """Handle application close event with confirmation dialog."""
        try:
            # Show confirmation dialog
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Are you sure you want to exit SensorSphere?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            # Cancel any running repository operations
            if self.qt_repository and hasattr(
                self.qt_repository, "is_operation_running"
            ):
                try:
                    if self.qt_repository.is_operation_running():
                        self.qt_repository.cancel_operation()
                        self.logger.info("Cancelled running repository operations")
                except Exception as e:
                    self.logger.warning(f"Error cancelling repository operations: {e}")

            self.logger.info("Application closing")
            event.accept()

        except Exception as e:
            self.logger.error(f"Error during application close: {e}")
            # Accept event even if cleanup fails to avoid hanging
            event.accept()


def main():
    """Main entry point for PySide6 GUI."""
    import signal
    import sys

    from PySide6.QtWidgets import QApplication

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("sensor_tool_qt.log"), logging.StreamHandler()],
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SensorSphere")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("Robotics Tools")

    # Apply modern standard theme
    from .utils.theme_manager import get_theme_manager

    theme_manager = get_theme_manager()
    theme_manager.apply_modern_theme_to_app()

    # Create and show main window
    window = SensorComparisonMainWindow()

    # Initialize toolbar button icon based on current theme
    is_dark = theme_manager.is_dark_mode
    window.theme_toggle_toolbar_action.setText("☀️" if is_dark else "🌙")
    window.theme_toggle_toolbar_action.setStatusTip(
        f"Switch to {'light' if is_dark else 'dark'} theme (Currently: {'dark' if is_dark else 'light'} mode)"
    )

    window.show()

    # Setup Ctrl+C signal handling for graceful termination
    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) for graceful application termination."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        window.close()
        app.quit()

    # Install signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Setup timer to allow Python to process signals
    # This is needed because Qt event loop blocks signal processing
    timer = QTimer()
    timer.start(500)  # Check for signals every 500ms
    timer.timeout.connect(lambda: None)  # Just wake up the event loop

    # Start event loop
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
        window.close()
        app.quit()


if __name__ == "__main__":
    main()

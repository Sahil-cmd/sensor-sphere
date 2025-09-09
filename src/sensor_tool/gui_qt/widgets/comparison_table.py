"""
Comparison Table Widget

Modern sensor comparison table with sorting, filtering, and selection capabilities.
This is a foundational implementation that will be enhanced with rich formatting,
side-by-side comparison, and advanced table features.
"""

import logging
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..dialogs.comparison_dialog import ComparisonDialog
from ..utils.font_manager import create_styled_font
from ..utils.fuzzy_matcher import create_fuzzy_matcher
from ..utils.natural_language_query import ParsedQuery, create_nlp_parser

logger = logging.getLogger(__name__)

# Custom role for sorting numerical data
SORT_ROLE = Qt.UserRole + 1


class SortableTableWidgetItem(QTableWidgetItem):
    """Custom table widget item that sorts by SORT_ROLE data instead of display text."""

    def __lt__(self, other):
        """Override comparison for sorting."""
        try:
            # Get sort values from SORT_ROLE
            self_value = self.data(SORT_ROLE)
            other_value = other.data(SORT_ROLE)

            # Handle None values
            if self_value is None and other_value is None:
                return False
            elif self_value is None:
                return True  # None sorts first
            elif other_value is None:
                return False

            # Compare values
            return self_value < other_value
        except (TypeError, ValueError):
            # Fall back to string comparison if comparison fails
            return super().__lt__(other)


class SensorComparisonTable(QWidget):
    """
    Modern sensor comparison table widget.

    Features:
    - Sortable columns
    - Row selection for detailed comparison
    - Modern styling
    - Export capabilities
    - Real-time search and filtering

    Signals:
        sensor_selected: Emitted when a sensor is selected
        selection_changed: Emitted when selection changes
    """

    # Signals
    sensor_selected = Signal(str)  # sensor_id
    selection_changed = Signal(list)  # List of selected sensor_ids

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data storage
        self.sensors_data: List[Dict[str, Any]] = []
        self.filtered_data: List[Dict[str, Any]] = []
        self.selected_sensors: List[str] = []

        # Enhanced natural language processing
        self.nlp_parser = create_nlp_parser()
        self.fuzzy_matcher = create_fuzzy_matcher()
        self.last_parsed_query: Optional[ParsedQuery] = None

        # Table configuration
        self.column_config = self._get_default_columns()
        self.column_visibility = {
            col["key"]: True for col in self.column_config
        }  # All visible by default

        # Setup UI
        self.setup_ui()
        self.setup_table()
        self.connect_signals()

    def setup_ui(self):
        """Setup the comparison table UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with controls
        header_layout = QHBoxLayout()

        # Table title and sensor count
        self.title_label = QLabel("Sensor Comparison Panel")
        self.title_label.setFont(
            create_styled_font("dock_title", "bold")
        )  # Match dock titles size with bold weight

        self.count_label = QLabel("0 sensors")
        self.count_label.setFont(create_styled_font("caption"))
        self.count_label.setStyleSheet("color: #666666;")

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.count_label)

        layout.addLayout(header_layout)

        # Enhanced natural language search
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Try: 'stereolabs cameras', 'sensors with fps>30', or 'ROS 2 under $500'"
        )
        search_layout.addWidget(self.search_input)

        # Search feedback label
        self.search_feedback_label = QLabel("")
        self.search_feedback_label.setStyleSheet(
            "color: #666; font-size: 11px; font-style: italic;"
        )
        self.search_feedback_label.setWordWrap(True)

        # View options
        self.show_selected_only = QCheckBox("Show selected only")
        self.show_selected_only.setToolTip(
            "Filter table to show only sensors you have selected for comparison"
        )
        search_layout.addWidget(self.show_selected_only)

        # Column configuration button
        self.configure_columns_btn = QPushButton("Configure Columns")
        self.configure_columns_btn.setToolTip(
            "Customize which sensor specifications are displayed in the table"
        )
        self.configure_columns_btn.clicked.connect(self.show_column_configuration)
        search_layout.addWidget(self.configure_columns_btn)

        layout.addLayout(search_layout)

        # Add search feedback label
        layout.addWidget(self.search_feedback_label)

        # Main table
        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Selection summary
        selection_layout = QHBoxLayout()

        self.selection_label = QLabel("No sensors selected")

        # Create buttons with consistent styling and size
        self.unselect_all_button = QPushButton("Unselect All")
        self.unselect_all_button.setEnabled(False)
        self.unselect_all_button.setFont(create_styled_font("button"))
        self.unselect_all_button.setMinimumHeight(32)  # Consistent height
        self.unselect_all_button.setToolTip("Clear all sensor selections in the table")

        self.compare_button = QPushButton("Compare Selected")
        self.compare_button.setEnabled(False)
        self.compare_button.setFont(create_styled_font("button"))
        self.compare_button.setMinimumHeight(32)  # Consistent height
        self.compare_button.setToolTip(
            "Open detailed side-by-side comparison for selected sensors"
        )

        self.export_button = QPushButton("Export Table")
        self.export_button.setFont(create_styled_font("button"))
        self.export_button.setMinimumHeight(32)  # Consistent height
        self.export_button.setToolTip(
            "Export current table data to CSV or Excel format"
        )

        selection_layout.addWidget(self.selection_label)
        selection_layout.addStretch()
        selection_layout.addWidget(self.unselect_all_button)
        selection_layout.addWidget(self.compare_button)
        selection_layout.addWidget(self.export_button)

        layout.addLayout(selection_layout)

    def setup_table(self):
        """Configure the table widget."""
        # Set selection behavior - standard interaction model
        # ExtendedSelection provides: Single click = select one, Ctrl+Click = toggle, Shift+Click = range
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setAlternatingRowColors(True)

        # Enable sorting
        self.table.setSortingEnabled(True)

        # Configure headers with last column stretching
        self.table.horizontalHeader().setStretchLastSection(
            True
        )  # Last column takes remaining space
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.verticalHeader().setVisible(False)

        # Enable horizontal scrolling with mouse/trackpad
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)

        # Install event filter for enhanced horizontal scrolling
        self.table.installEventFilter(self)
        self.table.setTabKeyNavigation(True)

        # Connect header click events for sort indicators
        self.table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)
        self.current_sort_column = -1
        self.current_sort_order = Qt.AscendingOrder

        # Enable context menu for header
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(
            self.show_header_context_menu
        )

        # Apply standard styling
        self.apply_table_styling()

    def eventFilter(self, source, event):
        """Event filter to handle horizontal scrolling with mouse/trackpad."""
        if source == self.table and event.type() == QEvent.Wheel:
            wheel_event = event

            # Check for horizontal scrolling (Shift+Wheel or trackpad horizontal gesture)
            modifiers = wheel_event.modifiers()
            angle_delta = wheel_event.angleDelta()

            # Handle horizontal scrolling
            if modifiers & Qt.ShiftModifier or abs(angle_delta.x()) > abs(
                angle_delta.y()
            ):
                # Horizontal scroll requested
                horizontal_scroll = self.table.horizontalScrollBar()

                # Determine scroll direction and amount
                if angle_delta.x() != 0:
                    # Direct horizontal wheel movement (trackpad)
                    scroll_amount = (
                        -angle_delta.x() // 8
                    )  # Convert to reasonable scroll amount
                else:
                    # Shift+Wheel for horizontal scroll (mouse)
                    scroll_amount = -angle_delta.y() // 8

                # Apply horizontal scroll
                new_value = horizontal_scroll.value() + scroll_amount
                new_value = max(
                    horizontal_scroll.minimum(),
                    min(horizontal_scroll.maximum(), new_value),
                )
                horizontal_scroll.setValue(new_value)

                return True  # Event handled

        return super().eventFilter(source, event)

    def apply_table_styling(self):
        """Apply standard styling to the table."""
        self.table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: white;
                alternate-background-color: #f8f9fa;
                selection-background-color: #0078d4;
                selection-color: white;
                color: #333333;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
                color: #333333;
            }
            QTableWidget::item:hover {
                background-color: #e3f2fd;
                color: #333333;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QHeaderView::section {
                background-color: #f1f3f4;
                padding: 8px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
                color: #333333;
            }
            QHeaderView::section:hover {
                background-color: #e8eaed;
                color: #333333;
            }
        """
        )

    def connect_signals(self):
        """Connect widget signals."""
        # Table selection changes
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)

        # Search and filtering
        self.search_input.textChanged.connect(self.on_search_changed)
        self.show_selected_only.toggled.connect(self.update_table_display)

        # Action buttons
        self.unselect_all_button.clicked.connect(self.on_unselect_all_clicked)
        self.compare_button.clicked.connect(self.on_compare_clicked)
        self.export_button.clicked.connect(self.on_export_clicked)

        # Delayed search timer
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.apply_search_filter)

    def _get_default_columns(self) -> List[Dict[str, Any]]:
        """Get default column configuration with sensor-agnostic headers and enhanced columns."""
        return [
            {
                "key": "manufacturer",
                "title": "Manufacturer",
                "width": 120,
                "sortable": True,
                "sort_type": "string",
                "min_width": 100,
                "essential": True,
            },
            {
                "key": "model",
                "title": "Model",
                "width": 180,
                "sortable": True,
                "sort_type": "string",
                "min_width": 130,
                "essential": True,
            },
            {
                "key": "sensor_type",
                "title": "Type",
                "width": 90,
                "sortable": True,
                "sort_type": "string",
                "min_width": 80,
            },
            {
                "key": "resolution_display",
                "title": "Resolution/Channels",
                "width": 120,
                "sortable": True,
                "sort_type": "string",
                "min_width": 110,
            },  # Pixels for cameras, Channels for LiDAR
            {
                "key": "frame_rate_display",
                "title": "Frame Rate",
                "width": 100,
                "sortable": True,
                "sort_type": "numeric",
                "min_width": 95,
            },  # Auto-detects FPS vs Hz
            {
                "key": "fov_display",
                "title": "FOV",
                "width": 130,
                "sortable": True,
                "sort_type": "string",
                "min_width": 110,
            },  # Field of View
            {
                "key": "range_display",
                "title": "Range",
                "width": 130,
                "sortable": True,
                "sort_type": "string",
                "min_width": 110,
            },  # Operating range
            {
                "key": "angular_resolution_display",
                "title": "Angular Res.",
                "width": 110,
                "sortable": True,
                "sort_type": "string",
                "min_width": 100,
            },  # Critical for LiDAR precision
            {
                "key": "environmental_rating",
                "title": "IP Rating",
                "width": 90,
                "sortable": True,
                "sort_type": "string",
                "min_width": 80,
            },  # IP68, IP67 etc
            {
                "key": "returns_display",
                "title": "Returns",
                "width": 80,
                "sortable": True,
                "sort_type": "string",
                "min_width": 70,
            },  # Single/Dual/Multi-echo
            {
                "key": "latency_display",
                "title": "Latency",
                "width": 90,
                "sortable": True,
                "sort_type": "numeric",
                "min_width": 80,
            },
            {
                "key": "power_display",
                "title": "Power",
                "width": 90,
                "sortable": True,
                "sort_type": "numeric",
                "min_width": 75,
            },
            {
                "key": "size_weight_display",
                "title": "Size/Weight",
                "width": 140,
                "sortable": True,
                "sort_type": "string",
                "min_width": 120,
            },
            {
                "key": "platform_support_display",
                "title": "Platforms",
                "width": 120,
                "sortable": True,
                "sort_type": "string",
                "min_width": 100,
            },
            {
                "key": "ros_support",
                "title": "ROS",
                "width": 50,
                "sortable": True,
                "sort_type": "string",
                "min_width": 45,
            },
            {
                "key": "price_display",
                "title": "Price",
                "width": 130,
                "sortable": True,
                "sort_type": "numeric",
                "min_width": 120,
            },
            {
                "key": "communication_interface",
                "title": "Interface",
                "width": 140,
                "sortable": True,
                "sort_type": "string",
                "min_width": 120,
            },
            # Radar-specific columns
            {
                "key": "points_per_second_display",
                "title": "Points/Sec",
                "width": 100,
                "sortable": True,
                "sort_type": "numeric",
                "min_width": 90,
            },
            {
                "key": "operational_principle",
                "title": "Principle",
                "width": 120,
                "sortable": True,
                "sort_type": "string",
                "min_width": 100,
            },
            {
                "key": "channels_display",
                "title": "Channels",
                "width": 80,
                "sortable": True,
                "sort_type": "numeric",
                "min_width": 70,
            },
            {
                "key": "range_accuracy",
                "title": "Range Acc.",
                "width": 100,
                "sortable": True,
                "sort_type": "string",
                "min_width": 90,
            },
            # IMU-specific columns
            {
                "key": "sampling_rate_display",
                "title": "Sample Rate",
                "width": 100,
                "sortable": True,
                "sort_type": "numeric",
                "min_width": 90,
            },
            {
                "key": "mtbf",
                "title": "MTBF",
                "width": 120,
                "sortable": True,
                "sort_type": "string",
                "min_width": 100,
            },
            {
                "key": "input_voltage",
                "title": "Input Voltage",
                "width": 100,
                "sortable": True,
                "sort_type": "string",
                "min_width": 90,
            },
            {
                "key": "current_draw",
                "title": "Current",
                "width": 80,
                "sortable": True,
                "sort_type": "string",
                "min_width": 70,
            },
        ]

    def update_sensors_data(self, sensors: List[Dict[str, Any]]):
        """Update the table with new sensor data."""
        self.sensors_data = sensors.copy()
        self.filtered_data = sensors.copy()

        # Update table display
        self.update_table_display()

        # Update count label
        self.count_label.setText(f"{len(sensors)} sensors")

    def update_table_display(self):
        """Update the table display based on current filters."""
        display_data = self.get_display_data()

        # Get visible columns only
        visible_columns = [
            col
            for col in self.column_config
            if self.column_visibility.get(col["key"], True)
        ]

        # Set table dimensions
        self.table.setRowCount(len(display_data))
        self.table.setColumnCount(len(visible_columns))

        # Set headers
        headers = [col["title"] for col in visible_columns]
        self.table.setHorizontalHeaderLabels(headers)

        # Set column widths with intelligent content-based calculation
        self.optimize_column_widths(display_data, visible_columns)

        # Populate data
        for row_idx, sensor in enumerate(display_data):
            for col_idx, col_config in enumerate(visible_columns):
                key = col_config["key"]
                value = sensor.get(key, "N/A")

                # Format value based on type
                display_value = self.format_cell_value(value, key)

                # Create table item with custom sorting capability
                item = SortableTableWidgetItem(str(display_value))

                # Always store sensor_id for selection logic
                item.setData(Qt.UserRole, sensor.get("sensor_id", ""))

                # Store numerical values for proper sorting
                sort_type = col_config.get("sort_type", "string")
                if sort_type == "numeric":
                    numeric_value = self.extract_numeric_for_sorting(display_value, key)
                    item.setData(SORT_ROLE, numeric_value)
                else:
                    # For string columns, store the string value for sorting
                    item.setData(SORT_ROLE, str(display_value))

                # Set item properties
                if key in [
                    "price_display",
                    "frame_rate_display",
                    "resolution_display",
                    "latency_display",
                    "power_display",
                ]:
                    item.setTextAlignment(Qt.AlignCenter)

                self.table.setItem(row_idx, col_idx, item)

        # Restore selection if possible
        self.restore_selection()

    def get_display_data(self) -> List[Dict[str, Any]]:
        """Get data to display based on current filters."""
        display_data = self.filtered_data.copy()

        # Apply show selected only filter
        if self.show_selected_only.isChecked():
            display_data = [
                s for s in display_data if s.get("sensor_id") in self.selected_sensors
            ]

        return display_data

    def format_cell_value(self, value: Any, key: str) -> str:
        """Format cell value for display."""
        if value is None or value == "":
            return "N/A"

        if key == "ros_support":
            return "✓" if value else "✗"
        elif key == "price_display" and isinstance(value, str):
            return value
        elif isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        else:
            return str(value)

    def extract_numeric_for_sorting(self, display_value: str, key: str) -> float:
        """Extract numerical value from formatted display string for sorting."""
        import re

        # Handle N/A and empty values
        if not display_value or display_value.strip() in [
            "N/A",
            "Contact manufacturer",
            "Contact ...",
            "",
        ]:
            return -1.0  # Sort N/A values to bottom

        try:
            if key == "frame_rate_display":
                # Extract number from "90.0 FPS" -> 90.0
                match = re.search(r"(\d+\.?\d*)", str(display_value))
                if match:
                    return float(match.group(1))
            elif key == "price_display":
                # Extract number from "$249" or "$11999" -> 249, 11999
                if display_value.startswith("$"):
                    price_str = display_value[1:]  # Remove $ symbol
                    # Remove any commas
                    price_str = price_str.replace(",", "")
                    match = re.search(r"(\d+\.?\d*)", price_str)
                    if match:
                        return float(match.group(1))
            elif key == "latency_display":
                # Extract number from "33ms" or "16.7ms" -> 33.0, 16.7
                match = re.search(r"(\d+\.?\d*)", str(display_value))
                if match:
                    return float(match.group(1))
            elif key == "power_display":
                # Extract number from "2.5W" or "15W" -> 2.5, 15.0
                match = re.search(r"(\d+\.?\d*)", str(display_value))
                if match:
                    return float(match.group(1))
        except (ValueError, AttributeError):
            pass

        return -1.0  # Default for unparseable values

    def optimize_column_widths(
        self,
        display_data: List[Dict[str, Any]],
        visible_columns: List[Dict[str, Any]] = None,
    ):
        """Calculate minimum required widths for all columns, with last column also stretching to fill remaining space."""
        if visible_columns is None:
            visible_columns = [
                col
                for col in self.column_config
                if self.column_visibility.get(col["key"], True)
            ]

        if not display_data:
            # Fallback to default widths if no data
            for i, col_config in enumerate(visible_columns):
                self.table.setColumnWidth(i, col_config["width"])
            return

        # Calculate minimum required widths for all columns (including the last)
        # The last column will stretch beyond its minimum width due to setStretchLastSection(True)

        for col_idx, col_config in enumerate(visible_columns):

            key = col_config["key"]
            title = col_config["title"]
            min_width = col_config.get("min_width", 80)

            # Calculate header width requirement using font metrics
            table_font = self.table.font()
            font_metrics = QFontMetrics(table_font)
            header_width = (
                font_metrics.horizontalAdvance(title) + 30
            )  # Text width + padding for sort arrows

            # Calculate content width requirement
            content_width = self._calculate_content_width(display_data, key)

            # Determine minimum required width (content + small padding)
            required_width = max(header_width, content_width, min_width)

            # Add small padding for readability (no artificial caps)
            final_width = required_width + 10  # Just 10px padding beyond minimum

            # Set the calculated minimum width
            self.table.setColumnWidth(col_idx, int(final_width))

            # Log column width optimization details
            logger.debug(
                f"Column '{title}': header={header_width}px, content={content_width}px, "
                f"required={required_width}px, final={final_width}px"
            )

    def _calculate_content_width(
        self, display_data: List[Dict[str, Any]], key: str
    ) -> int:
        """Calculate required width for content in a specific column using actual Qt font metrics."""
        max_content_width = 0

        # Get the table font for accurate measurements
        table_font = self.table.font()
        font_metrics = QFontMetrics(table_font)

        # Intelligent sampling for better width calculation
        # For small datasets, use all data
        # For larger datasets, sample from beginning, middle, and end to get variety
        if len(display_data) <= 15:
            sample_data = display_data
        else:
            # Sample from different parts of the data for better representation
            sample_size = min(15, len(display_data))
            step = len(display_data) // sample_size
            sample_data = [display_data[i * step] for i in range(sample_size)]
            # Always include first and last items for edge cases
            if display_data[0] not in sample_data:
                sample_data.insert(0, display_data[0])
            if display_data[-1] not in sample_data:
                sample_data.append(display_data[-1])

        for sensor in sample_data:
            value = sensor.get(key, "N/A")
            display_value = self.format_cell_value(value, key)

            # Use Qt's actual font metrics for precise width calculation
            text_width = font_metrics.horizontalAdvance(str(display_value))

            # Add small padding for cell margins and readability
            content_width = text_width + 16  # 8px padding on each side

            max_content_width = max(max_content_width, content_width)

            # Debug logging for troubleshooting
            logger.debug(
                f"Content width calc - {key}: '{display_value}' = {text_width}px + 16px = {content_width}px"
            )

        return int(max_content_width)

    def auto_fit_columns(self):
        """Manually trigger column auto-sizing to fit current content."""
        if hasattr(self, "sensors_data") and self.sensors_data:
            display_data = self.get_display_data()
            self.optimize_column_widths(display_data)
            logger.info("Column widths auto-fitted to content")

    def auto_fit_column(self, column_index: int):
        """Auto-fit a specific column to its content."""
        if not hasattr(self, "sensors_data") or not self.sensors_data:
            return

        display_data = self.get_display_data()
        visible_columns = [
            col
            for col in self.column_config
            if self.column_visibility.get(col["key"], True)
        ]

        if 0 <= column_index < len(visible_columns):
            col_config = visible_columns[column_index]
            key = col_config["key"]
            title = col_config["title"]
            min_width = col_config.get("min_width", 80)

            # Calculate optimal width for this column
            table_font = self.table.font()
            font_metrics = QFontMetrics(table_font)
            header_width = font_metrics.horizontalAdvance(title) + 30
            content_width = self._calculate_content_width(display_data, key)
            required_width = max(header_width, content_width, min_width)
            final_width = required_width + 10

            self.table.setColumnWidth(column_index, int(final_width))
            logger.debug(
                f"Auto-fitted column {column_index} ('{title}') to {final_width}px"
            )

    def show_header_context_menu(self, position):
        """Show context menu for table header with column management options."""
        header = self.table.horizontalHeader()
        logical_index = header.logicalIndexAt(position)

        if logical_index >= 0:
            menu = QMenu()

            # Auto-fit this column
            auto_fit_action = menu.addAction("Auto-fit Column")
            auto_fit_action.triggered.connect(
                lambda: self.auto_fit_column(logical_index)
            )

            menu.addSeparator()

            # Auto-fit all columns
            auto_fit_all_action = menu.addAction("Auto-fit All Columns")
            auto_fit_all_action.triggered.connect(self.auto_fit_columns)

            menu.addSeparator()

            # Column configuration
            config_action = menu.addAction("Configure Columns...")
            config_action.triggered.connect(self.show_column_configuration)

            # Show menu at cursor position
            menu.exec_(header.mapToGlobal(position))

    # Note: Removed complex resizeEvent method since Qt's setStretchLastSection(True)
    # automatically handles window resizing by stretching the last column

    def on_header_clicked(self, logical_index: int):
        """Handle header click for sort indicators."""
        # Determine sort order
        if self.current_sort_column == logical_index:
            # Same column clicked, toggle sort order
            self.current_sort_order = (
                Qt.DescendingOrder
                if self.current_sort_order == Qt.AscendingOrder
                else Qt.AscendingOrder
            )
        else:
            # Different column clicked, start with ascending
            self.current_sort_order = Qt.AscendingOrder

        self.current_sort_column = logical_index

        # Perform the sort
        self.table.sortItems(logical_index, self.current_sort_order)

        # Update visual indicators
        self.update_header_sort_indicators()

    def update_header_sort_indicators(self):
        """Update visual sort indicators in column headers."""
        for i in range(len(self.column_config)):
            col_config = self.column_config[i]
            title = col_config["title"]

            if i == self.current_sort_column:
                # Add sort indicator to current sorted column
                if self.current_sort_order == Qt.AscendingOrder:
                    indicator = " ↑"
                else:
                    indicator = " ↓"
                header_text = title + indicator
            else:
                header_text = title

            # Update header label
            self.table.horizontalHeaderItem(i).setText(header_text)

    def on_selection_changed(self):
        """Handle table selection changes."""
        selected_items = self.table.selectedItems()

        # Get unique sensor IDs from selection
        selected_ids = set()
        for item in selected_items:
            sensor_id = item.data(Qt.UserRole)
            if sensor_id:
                selected_ids.add(sensor_id)

        self.selected_sensors = list(selected_ids)

        # Update UI
        count = len(self.selected_sensors)
        if count == 0:
            self.selection_label.setText("No sensors selected")
            self.unselect_all_button.setEnabled(False)
            self.compare_button.setEnabled(False)
        elif count == 1:
            self.selection_label.setText("1 sensor selected")
            self.unselect_all_button.setEnabled(True)
            self.compare_button.setEnabled(False)
        else:
            self.selection_label.setText(f"{count} sensors selected")
            self.unselect_all_button.setEnabled(True)
            self.compare_button.setEnabled(True)

        # Emit signal
        self.selection_changed.emit(self.selected_sensors)

    def on_cell_double_clicked(self, row: int, column: int):
        """Handle cell double-click."""
        item = self.table.item(row, 0)  # Get first column item for sensor ID
        if item:
            sensor_id = item.data(Qt.UserRole)
            if sensor_id:
                self.sensor_selected.emit(sensor_id)

    def on_search_changed(self, text: str):
        """Handle search text changes with debouncing."""
        self.search_timer.stop()
        self.search_timer.start(300)  # 300ms delay

    def apply_search_filter(self):
        """Apply enhanced search filter with fuzzy matching and natural language processing."""
        search_text = self.search_input.text().strip()

        # Clear feedback initially
        self.search_feedback_label.setText("")

        if not search_text:
            self.filtered_data = self.sensors_data.copy()
            self.last_parsed_query = None
        else:
            try:
                # Apply fuzzy corrections to the search text
                corrected_query = self._enhance_query_with_fuzzy_matching(search_text)

                # Parse query using natural language processor
                self.last_parsed_query = self.nlp_parser.parse_query(corrected_query)

                # Build feedback with corrections and parsing
                feedback_parts = []

                if corrected_query != search_text:
                    feedback_parts.append(
                        f"Auto-corrected: '{search_text}' → '{corrected_query}'"
                    )

                parsing_feedback = self.nlp_parser.get_parsing_explanation(
                    self.last_parsed_query
                )
                feedback_parts.append(parsing_feedback)

                self.search_feedback_label.setText(" | ".join(feedback_parts))

                # Apply filters based on confidence
                if (
                    self.last_parsed_query.confidence >= 0.3
                    and self.last_parsed_query.filters
                ):
                    # High confidence - use structured filtering
                    self.filtered_data = self._apply_structured_filters(
                        self.last_parsed_query
                    )
                else:
                    # Low confidence - fallback to enhanced text search
                    self.filtered_data = self._apply_enhanced_text_search(
                        corrected_query
                    )

            except Exception as e:
                # Error in NLP parsing - fallback to enhanced text search
                logger.warning(f"Enhanced search error: {e}")
                enhanced_query = self._enhance_query_with_fuzzy_matching(search_text)
                self.filtered_data = self._apply_enhanced_text_search(enhanced_query)
                self.search_feedback_label.setText(
                    "Using enhanced text search with typo tolerance"
                )

        # Update display
        self.update_table_display()

    def _apply_structured_filters(
        self, parsed_query: ParsedQuery
    ) -> List[Dict[str, Any]]:
        """Apply structured filters from parsed query with intelligent OR logic for application context."""
        filtered_sensors = self.sensors_data.copy()

        # Check for special "show all" filter
        show_all_filter = next(
            (f for f in parsed_query.filters if f.field == "__show_all__"), None
        )
        if show_all_filter:
            # Return all sensors when "all sensors" query is detected
            return filtered_sensors

        # Separate semantic filters (application + qualitative) from strict filters
        # Semantic filters use OR logic within the query context
        semantic_filter_fields = [
            "use_cases",
            "tags",
            "latency",
            "frame_rate",
            "weight",
            "power_consumption",
            "price_avg",
            "max_range",
            "min_range",
        ]
        semantic_filters = [
            f for f in parsed_query.filters if f.field in semantic_filter_fields
        ]
        strict_filters = [
            f for f in parsed_query.filters if f.field not in semantic_filter_fields
        ]

        # Apply strict filters with AND logic (traditional way)
        for filter_obj in strict_filters:
            field = filter_obj.field
            operator = filter_obj.operator
            value = filter_obj.value

            filtered_sensors = [
                sensor
                for sensor in filtered_sensors
                if self._sensor_matches_filter(sensor, field, operator, value)
            ]

        # Apply semantic filters with OR logic (if sensor matches ANY semantic filter, include it)
        if semantic_filters:
            semantic_matched_sensors = []
            for sensor in filtered_sensors:
                matches_any_semantic_filter = False
                for filter_obj in semantic_filters:
                    if self._sensor_matches_filter(
                        sensor, filter_obj.field, filter_obj.operator, filter_obj.value
                    ):
                        matches_any_semantic_filter = True
                        break

                if matches_any_semantic_filter:
                    semantic_matched_sensors.append(sensor)

            filtered_sensors = semantic_matched_sensors

        # Apply text search on remaining terms if any
        if parsed_query.text_search_terms:
            text_query = " ".join(parsed_query.text_search_terms)
            filtered_sensors = self._apply_text_search_to_list(
                text_query, filtered_sensors
            )

        return filtered_sensors

    def _sensor_matches_filter(
        self, sensor: Dict[str, Any], field: str, operator, value
    ) -> bool:
        """Check if sensor matches a specific filter."""
        from ..utils.natural_language_query import OperatorType

        # Get sensor field value
        sensor_value = self._extract_sensor_value(sensor, field)
        if sensor_value is None:
            return False

        try:
            if operator == OperatorType.GREATER_THAN:
                sensor_numeric = self._extract_numeric_from_value(sensor_value)
                if sensor_numeric is not None:
                    return sensor_numeric > float(value)
            elif operator == OperatorType.GREATER_EQUAL:
                sensor_numeric = self._extract_numeric_from_value(sensor_value)
                if sensor_numeric is not None:
                    return sensor_numeric >= float(value)
            elif operator == OperatorType.LESS_THAN:
                sensor_numeric = self._extract_numeric_from_value(sensor_value)
                if sensor_numeric is not None:
                    return sensor_numeric < float(value)
            elif operator == OperatorType.LESS_EQUAL:
                sensor_numeric = self._extract_numeric_from_value(sensor_value)
                if sensor_numeric is not None:
                    return sensor_numeric <= float(value)
            elif operator == OperatorType.EQUALS:
                sensor_numeric = self._extract_numeric_from_value(sensor_value)
                if sensor_numeric is not None:
                    return sensor_numeric == float(value)
            elif operator == OperatorType.CONTAINS:
                # Handle list fields (use_cases, tags, ros_compatibility)
                if isinstance(sensor_value, list):
                    search_value = str(value).lower()
                    for item in sensor_value:
                        if search_value in str(item).lower():
                            return True
                    return False
                else:
                    return str(value).lower() in str(sensor_value).lower()
            elif operator == OperatorType.BETWEEN:
                if isinstance(value, list) and len(value) == 2:
                    sensor_numeric = self._extract_numeric_from_value(sensor_value)
                    if sensor_numeric is not None:
                        return value[0] <= sensor_numeric <= value[1]

        except (ValueError, TypeError):
            # For non-numeric fields, use string comparison
            if operator == OperatorType.CONTAINS:
                # Handle list fields (use_cases, tags, ros_compatibility)
                if isinstance(sensor_value, list):
                    search_value = str(value).lower()
                    for item in sensor_value:
                        if search_value in str(item).lower():
                            return True
                    return False
                else:
                    return str(value).lower() in str(sensor_value).lower()
            elif operator == OperatorType.EQUALS:
                return str(sensor_value).lower() == str(value).lower()

        return False

    def _extract_numeric_from_value(self, value) -> float:
        """Extract numeric value from string, numeric, or mixed formats."""
        import math
        import re

        if value is None:
            return None

        # Handle NaN values
        if isinstance(value, float) and math.isnan(value):
            return None

        # If already numeric, return as-is
        if isinstance(value, (int, float)) and not math.isnan(value):
            return float(value)

        # If string, extract number
        if isinstance(value, str):
            # Remove common units and extract first number
            match = re.search(r"(\d+(?:\.\d+)?)", value.lower())
            if match:
                return float(match.group(1))

        return None

    def _extract_sensor_value(self, sensor: Dict[str, Any], field: str) -> Any:
        """
        Extract field value from sensor data, handling mixed formats and conversions.

        This method handles the inconsistent data formats found in sensor YAML files:
        - String values with units: "800 ms" → 800.0
        - None/NaN values → None (filtered out)
        - Nested dictionary values
        - Different field naming conventions
        """
        import math
        import re

        # Helper function to extract numeric value from mixed formats
        def extract_numeric_value(raw_value):
            """Extract numeric value from string, numeric, or mixed formats."""
            if raw_value is None:
                return None

            # Handle NaN values
            if isinstance(raw_value, float) and math.isnan(raw_value):
                return None

            # If already numeric, return as-is
            if isinstance(raw_value, (int, float)) and not math.isnan(raw_value):
                return float(raw_value)

            # If string, extract number
            if isinstance(raw_value, str):
                # Remove common units and extract first number
                match = re.search(r"(\d+(?:\.\d+)?)", raw_value.lower())
                if match:
                    return float(match.group(1))

            return None

        # Helper function to extract price values
        def extract_price_value(price_range, price_type="avg"):
            """Extract price value from various price_range formats."""
            if price_range is None:
                return None

            if isinstance(price_range, str):
                if "contact" in price_range.lower():
                    return None
                # Try to extract number from string
                match = re.search(r"(\d+(?:\.\d+)?)", price_range)
                if match:
                    return float(match.group(1))
                return None

            if isinstance(price_range, dict):
                if price_type == "min" and "min_price" in price_range:
                    return price_range["min_price"]
                elif price_type == "max" and "max_price" in price_range:
                    return price_range["max_price"]
                elif price_type == "avg":
                    if "avg" in price_range:
                        return price_range["avg"]
                    elif "min_price" in price_range and "max_price" in price_range:
                        min_p, max_p = (
                            price_range["min_price"],
                            price_range["max_price"],
                        )
                        if min_p is not None and max_p is not None:
                            return (min_p + max_p) / 2

            return None

        # Field mappings and extraction logic
        try:
            # Direct field access with mixed format handling
            if field in sensor:
                raw_value = sensor[field]

                # Handle numeric fields that might have units
                if field in ["latency", "frame_rate", "weight", "power_consumption"]:
                    return extract_numeric_value(raw_value)

                # Handle string fields
                elif field in ["sensor_type", "manufacturer", "model"]:
                    return raw_value if raw_value and str(raw_value).strip() else None

                # Handle other fields as-is
                else:
                    return raw_value if raw_value is not None else None

            # Handle special field mappings
            if field == "price_avg":
                return extract_price_value(sensor.get("price_range"), "avg")

            elif field == "price_min":
                return extract_price_value(sensor.get("price_range"), "min")

            elif field == "price_max":
                return extract_price_value(sensor.get("price_range"), "max")

            elif field == "resolution_rgb_pixels":
                resolution = sensor.get("resolution", {})
                if isinstance(resolution, dict) and "rgb" in resolution:
                    rgb = resolution["rgb"]
                    if isinstance(rgb, dict) and "width" in rgb and "height" in rgb:
                        return rgb["width"] * rgb["height"]

            elif field == "field_of_view_horizontal":
                fov = sensor.get("field_of_view", {})
                if isinstance(fov, dict):
                    return fov.get("horizontal")

            elif field == "max_range":
                # Try max_range first, then nested range.max
                max_range = sensor.get("max_range")
                if max_range is not None:
                    return extract_numeric_value(max_range)

                range_data = sensor.get("range", {})
                if isinstance(range_data, dict):
                    return extract_numeric_value(range_data.get("max"))

            elif field == "min_range":
                # Try min_range first, then nested range.min
                min_range = sensor.get("min_range")
                if min_range is not None:
                    return extract_numeric_value(min_range)

                range_data = sensor.get("range", {})
                if isinstance(range_data, dict):
                    return extract_numeric_value(range_data.get("min"))

            elif field == "ros_compatibility":
                # Handle ROS compatibility list
                ros_compat = sensor.get("ros_compatibility", [])
                if isinstance(ros_compat, list):
                    return ros_compat
                elif ros_compat:
                    return [str(ros_compat)]
                return []

            elif field in ["use_cases", "tags"]:
                # Handle use_cases and tags lists
                field_value = sensor.get(field, [])
                if isinstance(field_value, list):
                    return field_value
                elif field_value:
                    return [str(field_value)]
                return []

            # Default case - try direct access with None/NaN handling
            raw_value = sensor.get(field)
            if raw_value is None:
                return None
            if isinstance(raw_value, float) and math.isnan(raw_value):
                return None

            return raw_value

        except Exception as e:
            # Log the error but don't break the search
            logger.warning(
                f"Error extracting field '{field}' from sensor '{sensor.get('sensor_id', 'unknown')}': {e}"
            )
            return None

    def _enhance_query_with_fuzzy_matching(self, query: str) -> str:
        """Apply fuzzy matching enhancements to the search query."""
        try:
            # Apply typo corrections
            corrected = self.fuzzy_matcher.correct_query(query)

            # Expand abbreviations
            expanded = self.fuzzy_matcher.expand_abbreviations(corrected)

            # Normalize units (handle spaces, symbols, etc.)
            normalized = self.fuzzy_matcher.normalize_units(expanded)

            return normalized

        except Exception as e:
            logger.warning(f"Fuzzy matching error: {e}")
            return query

    def _apply_enhanced_text_search(self, search_text: str) -> List[Dict[str, Any]]:
        """Apply enhanced text search with fuzzy matching alternatives."""
        try:
            # Get alternative query variations for better matching
            alternatives = self.fuzzy_matcher.get_alternative_queries(
                search_text, max_alternatives=5
            )

            # Start with original query
            results = set()

            # Try each alternative and combine results
            for alt_query in alternatives:
                alt_results = self._apply_text_search_to_list(
                    alt_query, self.sensors_data
                )
                for result in alt_results:
                    results.add(result["sensor_id"])  # Use sensor_id to deduplicate

            # Convert back to list format
            unique_results = []
            for sensor in self.sensors_data:
                if sensor["sensor_id"] in results:
                    unique_results.append(sensor)

            return unique_results

        except Exception as e:
            logger.warning(f"Enhanced text search error: {e}")
            # Fallback to basic text search
            return self._apply_text_search(search_text)

    def _apply_text_search(self, search_text: str) -> List[Dict[str, Any]]:
        """Apply traditional text search across multiple fields."""
        return self._apply_text_search_to_list(search_text, self.sensors_data)

    def _apply_text_search_to_list(
        self, search_text: str, sensor_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply text search to a specific list of sensors."""
        search_lower = search_text.lower()
        search_fields = [
            "manufacturer",
            "model",
            "sensor_type",
            "communication_interface",
            "platform_support_display",
        ]

        filtered_data = []
        for sensor in sensor_list:
            match = False
            for field in search_fields:
                value = sensor.get(field, "")
            if search_lower in str(value).lower():
                match = True
                break

            if match:
                filtered_data.append(sensor)

        return filtered_data

    def restore_selection(self):
        """Restore previous selection after table update."""
        if not self.selected_sensors:
            return

        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                sensor_id = item.data(Qt.UserRole)
                if sensor_id in self.selected_sensors:
                    self.table.selectRow(row)

    def on_compare_clicked(self):
        """Handle compare button click to show side-by-side comparison."""
        if len(self.selected_sensors) >= 2:
            # Get full sensor data for selected sensors
            selected_data = []
            for sensor_id in self.selected_sensors:
                # Find sensor in our data
                for sensor in self.sensors_data:
                    if (
                        sensor.get("sensor_id") == sensor_id
                        or sensor.get("model") == sensor_id
                    ):
                        selected_data.append(sensor)
                        break

            if selected_data:
                # Open comparison dialog
                dialog = ComparisonDialog(selected_data, self)
                dialog.exec()
            else:
                QMessageBox.warning(
                    self,
                    "Comparison Error",
                    "Could not retrieve data for selected sensors.",
                )

    def on_unselect_all_clicked(self):
        """Handle unselect all button click to clear all selections."""
        # Clear table selection
        self.table.clearSelection()

        # Clear internal selection tracking
        self.selected_sensors.clear()

        # Update UI state
        self.selection_label.setText("No sensors selected")
        self.unselect_all_button.setEnabled(False)
        self.compare_button.setEnabled(False)

        # Emit signal to notify other components
        self.selection_changed.emit(self.selected_sensors)

        logger.info("All sensor selections cleared")

    def on_export_clicked(self):
        """Handle export button click."""
        # TODO: Implement export functionality
        QMessageBox.information(
            self, "Export Table", "Table export functionality will be implemented."
        )

    def clear_selection(self):
        """Clear current selection."""
        self.table.clearSelection()
        self.selected_sensors = []
        self.selection_changed.emit([])

    def select_sensors(self, sensor_ids: List[str]):
        """Programmatically select sensors by ID."""
        self.selected_sensors = sensor_ids.copy()

        # Update table selection
        self.table.clearSelection()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.UserRole) in sensor_ids:
                self.table.selectRow(row)

        self.on_selection_changed()

    def get_selected_sensors(self) -> List[str]:
        """Get list of currently selected sensor IDs."""
        return self.selected_sensors.copy()

    def show_column_configuration(self):
        """Show dialog to configure visible columns."""
        dialog = ColumnConfigurationDialog(
            self.column_config, self.column_visibility, self
        )
        if dialog.exec() == QDialog.Accepted:
            # Update column visibility based on dialog results
            self.column_visibility = dialog.get_column_visibility()
            # Refresh table display with new column configuration
            self.update_table_display()
            logger.info("Column configuration updated")

    def update_fonts(self):
        """Update fonts for dynamic scaling."""
        try:
            # Update header fonts
            self.title_label.setFont(create_styled_font("dock_title", "bold"))
            self.count_label.setFont(create_styled_font("caption"))

            # Update search label
            search_labels = self.findChildren(QLabel)
            for label in search_labels:
                if label.text() == "Search:":
                    label.setFont(create_styled_font("body"))

            # Update button fonts
            self.compare_button.setFont(create_styled_font("body"))
            self.export_button.setFont(create_styled_font("body"))
            self.selection_label.setFont(create_styled_font("body"))

            # Update table font (this affects all cells)
            table_font = create_styled_font("table_cell")
            self.table.setFont(table_font)

            # Update table header fonts with style sheet to ensure they stand out
            header_font = create_styled_font("table_header", "bold")
            header_font_size = header_font.pointSize()

            # Apply enhanced header styling
            header_stylesheet = f"""
                QHeaderView::section {{
                    background-color: #f1f3f4;
                    padding: 10px 8px;
                    border: 1px solid #d0d0d0;
                    font-weight: bold;
                    font-size: {header_font_size}pt;
                    color: #333333;
                    text-align: center;
                }}
                QHeaderView::section:hover {{
                    background-color: #e8eaed;
                    color: #333333;
                }}
            """
            self.table.horizontalHeader().setStyleSheet(header_stylesheet)

            # Update search input font
            self.search_input.setFont(create_styled_font("body"))

            # Force table to refresh its display
            self.table.viewport().update()

            # Auto-fit columns after font changes for optimal display
            QTimer.singleShot(
                100, self.auto_fit_columns
            )  # Small delay to ensure font is applied

        except Exception as e:
            # Fallback to basic font update if dynamic system fails
            logger.warning(f"Font update failed: {e}")
            basic_font = QFont()
            basic_font.setPointSize(10)
            self.setFont(basic_font)

    def keyPressEvent(self, event):
        """Handle keyboard navigation for standard table interaction."""
        key = event.key()
        modifiers = event.modifiers()

        # Handle Ctrl+A for select all
        if key == Qt.Key_A and modifiers == Qt.ControlModifier:
            self.table.selectAll()
            self.on_selection_changed()
            event.accept()
            return

        # Handle Escape to clear selection
        elif key == Qt.Key_Escape:
            self.clear_selection()
            event.accept()
            return

        # Handle Enter/Return to open comparison dialog
        elif key in (Qt.Key_Enter, Qt.Key_Return):
            selected_sensors = self.get_selected_sensors()
            if selected_sensors:
                self.on_compare_clicked()
            event.accept()
            return

        # Handle Delete to unselect all
        elif key == Qt.Key_Delete:
            self.on_unselect_all_clicked()
            event.accept()
            return

        # Handle F3 to focus search
        elif key == Qt.Key_F3:
            self.search_input.setFocus()
            self.search_input.selectAll()
            event.accept()
            return

        # Handle F5 to refresh/reload data
        elif key == Qt.Key_F5:
            # Trigger a refresh of the table display
            self.update_table_display()
            event.accept()
            return

        # Handle Ctrl+R to auto-fit all columns
        elif key == Qt.Key_R and modifiers == Qt.ControlModifier:
            self.auto_fit_columns()
            event.accept()
            return

        # Let the table handle other navigation keys (arrows, Page Up/Down, Home/End)
        # These are handled natively by QTableWidget
        super().keyPressEvent(event)


class ColumnConfigurationDialog(QDialog):
    """Dialog for configuring visible table columns."""

    def __init__(
        self,
        column_config: List[Dict[str, Any]],
        column_visibility: Dict[str, bool],
        parent=None,
    ):
        super().__init__(parent)
        self.column_config = column_config
        self.column_visibility = column_visibility.copy()

        self.setWindowTitle("Configure Table Columns")
        self.setModal(True)
        self.resize(400, 300)  # Back to standard height

        self.setup_ui()

    def setup_ui(self):
        """Setup the dialog UI."""
        # Get theme manager colors at the start
        from ..utils.theme_manager import get_theme_manager

        theme_manager = get_theme_manager()
        colors = theme_manager.get_stylesheet_colors()

        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel("Select which columns to display in the table:")
        instructions.setFont(create_styled_font("body"))
        layout.addWidget(instructions)

        # Essential columns info with theme-aware colors
        essential_info = QLabel(
            "* Essential columns (Manufacturer, Model) cannot be hidden - required for sensor identification"
        )
        essential_info.setFont(create_styled_font("caption"))
        essential_info.setStyleSheet(
            f"color: {colors['text_secondary']}; font-style: italic; margin: 5px;"
        )
        layout.addWidget(essential_info)

        # Column checkboxes
        self.column_checkboxes = {}

        for col_config in self.column_config:
            key = col_config["key"]
            title = col_config["title"]
            is_essential = col_config.get("essential", False)

            checkbox = QCheckBox(title)
            checkbox.setChecked(self.column_visibility.get(key, True))
            checkbox.setFont(create_styled_font("body"))

            # Handle essential columns
            if is_essential:
                checkbox.setEnabled(False)  # Disable the checkbox
                checkbox.setChecked(True)  # Always checked
                checkbox.setToolTip(
                    f"{title} is an essential column and cannot be hidden"
                )
                # Add visual indicator for essential columns with theme-aware colors
                checkbox.setText(f"{title} *")
                checkbox.setStyleSheet(
                    f"color: {colors['text_primary']}; font-weight: bold;"
                )

            self.column_checkboxes[key] = checkbox
            layout.addWidget(checkbox)

        # Add some spacing
        layout.addStretch()

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset,
            parent=self,
        )

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Reset).clicked.connect(
            self.reset_to_defaults
        )

        layout.addWidget(button_box)

        # Apply theme-aware standard styling
        dialog_style = f"""
            QDialog {{
                background-color: {colors['surface']};
                color: {colors['text_primary']};
                border: 1px solid {colors['border']};
            }}
            QCheckBox {{
                padding: 5px;
                margin: 2px;
                color: {colors['text_primary']};
            }}
            QCheckBox:hover {{
                background-color: {colors['hover']};
            }}
            QCheckBox:disabled {{
                color: {colors['text_primary']};
            }}
        """
        self.setStyleSheet(dialog_style)

    def reset_to_defaults(self):
        """Reset all columns to visible (default state)."""
        for checkbox in self.column_checkboxes.values():
            checkbox.setChecked(True)

    def get_column_visibility(self) -> Dict[str, bool]:
        """Get the current column visibility configuration."""
        visibility = {}
        for key, checkbox in self.column_checkboxes.items():
            # Find the column config to check if it's essential
            col_config = next(
                (col for col in self.column_config if col["key"] == key), None
            )
            is_essential = col_config.get("essential", False) if col_config else False

            # Essential columns are always visible
            if is_essential:
                visibility[key] = True
            else:
                visibility[key] = checkbox.isChecked()

        return visibility

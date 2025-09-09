"""
Filter Panel Widget

Provides advanced filtering controls for sensor comparison.
This is a placeholder implementation that will be enhanced with autocomplete,
multiple filters, and real-time filtering capabilities.
"""

from typing import Any, Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..utils.font_manager import create_styled_font
from .dual_range_slider import DualRangeSlider
from .fuzzy_completer import FuzzySearchComboBox, FuzzySearchLineEdit


class AdvancedFilterWidget(QWidget):
    """
    Advanced filtering widget with multiple filter types.

    Signals:
        filter_changed: Emitted when any filter criteria changes
        clear_filters: Emitted when filters are cleared
    """

    # Signals
    filter_changed = Signal(dict)  # Emitted with filter criteria
    clear_filters = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize filter state
        self.current_filters = {}

        # Setup UI
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Setup the filter panel UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create scroll area for filters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Main content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # Basic Info Filters
        content_layout.addWidget(self.create_basic_filters_group())

        # Technical Specs Filters
        content_layout.addWidget(self.create_tech_specs_group())

        # ROS Integration Filters
        content_layout.addWidget(self.create_ros_filters_group())

        # Price and Availability Filters
        content_layout.addWidget(self.create_price_filters_group())

        # Add stretch to push everything to top
        content_layout.addStretch()

        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)

        # Add scroll area to main layout
        layout.addWidget(scroll_area)

        # Action buttons
        layout.addWidget(self.create_action_buttons())

    def create_basic_filters_group(self) -> QGroupBox:
        """Create basic info filters group with fuzzy matching auto-complete."""
        group = QGroupBox("Basic Information")
        group.setFont(create_styled_font("section_header", "medium"))
        layout = QVBoxLayout(group)

        # Manufacturer filter with fuzzy matching
        mfg_layout = QHBoxLayout()
        mfg_label = QLabel("Manufacturer:")
        mfg_label.setFont(create_styled_font("field_label", "medium"))
        mfg_layout.addWidget(mfg_label)
        self.manufacturer_combo = FuzzySearchComboBox(include_any=True)
        self.manufacturer_combo.setToolTip(
            "Filter sensors by manufacturer (e.g., Intel, ZED, Mech-Mind) - supports fuzzy search"
        )
        # Initial items - will be updated dynamically from sensor data
        self.manufacturer_combo.addItems(
            ["Intel", "ZED", "Mech-Mind", "IDS", "Zivid", "Stereolabs"]
        )
        mfg_layout.addWidget(self.manufacturer_combo)
        layout.addLayout(mfg_layout)

        # Sensor type filter with fuzzy matching
        type_layout = QHBoxLayout()
        type_label = QLabel("Sensor Type:")
        type_label.setFont(create_styled_font("field_label", "medium"))
        type_layout.addWidget(type_label)
        self.sensor_type_combo = FuzzySearchComboBox(include_any=True)
        self.sensor_type_combo.setToolTip(
            "Filter by sensor technology type (RGB, Depth, Stereo, LiDAR, etc.) - critical for robotics application matching"
        )
        self.sensor_type_combo.addItems(
            [
                "RGB Camera",
                "Depth Camera",
                "Stereo Camera",
                "LiDAR",
                "ToF Camera",
                "Structured Light",
            ]
        )
        type_layout.addWidget(self.sensor_type_combo)
        layout.addLayout(type_layout)

        # Model search with fuzzy matching suggestions
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setFont(create_styled_font("field_label", "medium"))
        model_layout.addWidget(model_label)
        self.model_search = FuzzySearchLineEdit()
        self.model_search.setPlaceholderText("Search by model name...")
        self.model_search.setToolTip(
            "Search for specific sensor models (e.g., D435i, ZED 2, Mech-Eye) with fuzzy matching"
        )
        # Model suggestions will be set dynamically from sensor data
        model_layout.addWidget(self.model_search)
        layout.addLayout(model_layout)

        return group

    def create_tech_specs_group(self) -> QGroupBox:
        """Create technical specifications filters group."""
        group = QGroupBox("Technical Specifications")
        group.setFont(create_styled_font("section_header", "medium"))
        layout = QVBoxLayout(group)

        # Resolution filters
        res_group = QGroupBox("Resolution")
        res_layout = QVBoxLayout(res_group)

        # Min resolution
        min_res_layout = QHBoxLayout()
        min_res_label = QLabel("Min Resolution:")
        min_res_label.setFont(create_styled_font("field_label", "medium"))
        min_res_layout.addWidget(min_res_label)
        self.min_resolution_spin = QSpinBox()
        self.min_resolution_spin.setFont(create_styled_font("input_text"))
        self.min_resolution_spin.setRange(0, 8000)
        self.min_resolution_spin.setSuffix(" pixels")
        self.min_resolution_spin.setValue(0)
        min_res_layout.addWidget(self.min_resolution_spin)
        res_layout.addLayout(min_res_layout)

        # Frame rate filters
        fps_group = QGroupBox("Frame Rate")
        fps_layout = QVBoxLayout(fps_group)

        min_fps_layout = QHBoxLayout()
        min_fps_label = QLabel("Min FPS:")
        min_fps_label.setFont(create_styled_font("field_label", "medium"))
        min_fps_layout.addWidget(min_fps_label)
        self.min_fps_spin = QDoubleSpinBox()
        self.min_fps_spin.setFont(create_styled_font("input_text"))
        self.min_fps_spin.setRange(0.0, 1000.0)
        self.min_fps_spin.setSuffix(" FPS")
        self.min_fps_spin.setValue(0.0)
        min_fps_layout.addWidget(self.min_fps_spin)
        fps_layout.addLayout(min_fps_layout)

        layout.addWidget(res_group)
        layout.addWidget(fps_group)

        return group

    def create_ros_filters_group(self) -> QGroupBox:
        """Create ROS integration filters group."""
        group = QGroupBox("ROS Integration")
        group.setFont(create_styled_font("section_header", "medium"))
        layout = QVBoxLayout(group)

        # ROS compatibility checkboxes
        self.ros1_check = QCheckBox("ROS 1")
        self.ros2_check = QCheckBox("ROS 2")

        layout.addWidget(self.ros1_check)
        layout.addWidget(self.ros2_check)

        return group

    def create_price_filters_group(self) -> QGroupBox:
        """Create price range filters group with enhanced dual-range slider."""
        group = QGroupBox("Price Range")
        group.setFont(create_styled_font("section_header", "medium"))
        layout = QVBoxLayout(group)

        # Enhanced dual-range price filter
        price_label = QLabel("Price Range (USD)")
        price_label.setStyleSheet("font-weight: bold; margin-bottom: 8px;")
        layout.addWidget(price_label)

        # Get actual price range from data (defaults to reasonable robotics sensor range)
        # These values will be updated dynamically when sensor data is loaded
        self.price_filter = DualRangeSlider(
            min_value=0,  # Will be updated with actual min price from data
            max_value=15000,  # Will be updated with actual max price from data
            current_min=0,
            current_max=15000,
        )
        self.price_filter.setToolTip(
            "Drag handles to set price range, use text inputs for precise values, "
            "or click quick filter buttons for common budget categories"
        )
        layout.addWidget(self.price_filter)

        return group

    def create_action_buttons(self) -> QWidget:
        """Create action buttons widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Apply filters button
        self.apply_button = QPushButton("Apply Filters")
        self.apply_button.setFont(create_styled_font("button", "medium"))
        self.apply_button.setToolTip(
            "Apply current filter settings to narrow down sensor results based on your criteria"
        )

        # Use theme manager for consistent button styling
        from ..utils.theme_manager import get_theme_manager

        theme_manager = get_theme_manager()
        self.apply_button.setStyleSheet(
            theme_manager.create_button_stylesheet("primary")
        )

        # Clear filters button
        self.clear_button = QPushButton("Clear All")
        self.clear_button.setFont(create_styled_font("button", "medium"))
        self.clear_button.setToolTip(
            "Remove all filter settings and show all sensors in the database"
        )
        self.clear_button.setStyleSheet(
            theme_manager.create_button_stylesheet("secondary")
        )

        layout.addWidget(self.apply_button)
        layout.addWidget(self.clear_button)

        return widget

    def connect_signals(self):
        """Connect widget signals."""
        # Connect all filter controls to update method
        self.manufacturer_combo.currentTextChanged.connect(self.on_filter_changed)
        self.sensor_type_combo.currentTextChanged.connect(self.on_filter_changed)
        self.model_search.textChanged.connect(self.on_filter_changed)
        self.min_resolution_spin.valueChanged.connect(self.on_filter_changed)
        self.min_fps_spin.valueChanged.connect(self.on_filter_changed)
        self.ros1_check.toggled.connect(self.on_filter_changed)
        self.ros2_check.toggled.connect(self.on_filter_changed)

        # Connect enhanced price filter signals
        self.price_filter.range_changed.connect(self.on_price_range_changed)
        self.price_filter.filter_disabled.connect(self.on_price_filter_disabled)
        self.price_filter.include_unknown_changed.connect(
            self.on_include_unknown_changed
        )

        # Connect action buttons
        self.apply_button.clicked.connect(self.apply_current_filters)
        self.clear_button.clicked.connect(self.clear_all_filters)

    def on_filter_changed(self):
        """Handle filter value changes."""
        self.update_current_filters()

    def on_price_range_changed(self, min_price: int, max_price: int):
        """Handle price range changes from dual range slider."""
        self.update_current_filters()

    def on_price_filter_disabled(self):
        """Handle price filter being disabled (Any Price selected)."""
        self.update_current_filters()

    def on_include_unknown_changed(self, include_unknown: bool):
        """Handle include unknown prices checkbox toggle."""
        self.update_current_filters()

    def update_current_filters(self):
        """Update current filters dictionary."""
        self.current_filters = {}

        # Basic filters - use value() method for FuzzySearchComboBox
        mfg_value = self.manufacturer_combo.value()
        if mfg_value:  # value() returns None for "Any"
            self.current_filters["manufacturer"] = mfg_value

        type_value = self.sensor_type_combo.value()
        if type_value:  # value() returns None for "Any"
            self.current_filters["sensor_type"] = type_value

        if self.model_search.text().strip():
            self.current_filters["model_search"] = self.model_search.text().strip()

        # Technical specs
        if self.min_resolution_spin.value() > 0:
            self.current_filters["min_resolution"] = self.min_resolution_spin.value()

        if self.min_fps_spin.value() > 0:
            self.current_filters["min_frame_rate"] = self.min_fps_spin.value()

        # ROS filters
        ros_versions = []
        if self.ros1_check.isChecked():
            ros_versions.append("ROS1")
        if self.ros2_check.isChecked():
            ros_versions.append("ROS2")
        if ros_versions:
            self.current_filters["ros_compatibility"] = ros_versions

        # Enhanced price filters with dual range support
        min_price, max_price = self.price_filter.get_range()
        if min_price is not None and max_price is not None:
            # Only add to filters if values are not at full range
            if min_price > self.price_filter.min_value:
                self.current_filters["min_price"] = min_price
            if max_price < self.price_filter.max_value:
                self.current_filters["max_price"] = max_price
            # Include unknown prices preference
            self.current_filters["include_unknown_prices"] = (
                self.price_filter.get_include_unknown()
            )

    def apply_current_filters(self):
        """Apply current filters."""
        self.update_current_filters()

        import logging

        logger = logging.getLogger(__name__)

        self.filter_changed.emit(self.current_filters)

    def clear_all_filters(self):
        """Clear all filter values."""
        # Reset all controls to default values
        self.manufacturer_combo.setValue(None)  # Sets to "Any"
        self.sensor_type_combo.setValue(None)  # Sets to "Any"
        self.model_search.clear()
        self.min_resolution_spin.setValue(0)
        self.min_fps_spin.setValue(0.0)
        self.ros1_check.setChecked(False)
        self.ros2_check.setChecked(False)
        self.price_filter.reset()  # Reset enhanced price filter to "Any Price"

        # Clear current filters and emit signal
        self.current_filters = {}
        self.clear_filters.emit()

    def update_filter_options(
        self, sensors_data: List[Dict[str, Any]], update_price_range: bool = True
    ):
        """Update filter options based on available sensor data with fuzzy matching support."""
        if not sensors_data:
            return

        # Extract unique values from sensor data
        manufacturers = set()
        sensor_types = set()
        models = set()

        for sensor in sensors_data:
            if "manufacturer" in sensor:
                manufacturers.add(sensor["manufacturer"])
            if "sensor_type" in sensor:
                sensor_types.add(sensor["sensor_type"])
            if "model" in sensor:
                models.add(sensor["model"])

        # Update manufacturer combo with fuzzy matching support
        current_mfg = self.manufacturer_combo.currentText()
        self.manufacturer_combo.clear()
        self.manufacturer_combo.setItems(sorted(manufacturers))

        # Update sensor type combo with fuzzy matching support
        current_type = self.sensor_type_combo.currentText()
        self.sensor_type_combo.clear()
        self.sensor_type_combo.setItems(sorted(sensor_types))

        # Update model search suggestions for fuzzy matching
        self.model_search.setSuggestions(sorted(models))

        # Only update price filter range for initial data load, not for filtered results
        # This prevents circular updates that reset user's price selections
        if update_price_range:
            self._update_price_filter_range(sensors_data)

        # Restore previous selections if still valid
        if current_mfg and current_mfg != "Any":
            if current_mfg in manufacturers:
                self.manufacturer_combo.setCurrentText(current_mfg)

        if current_type and current_type != "Any":
            if current_type in sensor_types:
                self.sensor_type_combo.setCurrentText(current_type)

    def get_current_filters(self) -> Dict[str, Any]:
        """Get current filter criteria."""
        self.update_current_filters()
        return self.current_filters.copy()

    def _update_price_filter_range(self, sensors_data: List[Dict[str, Any]]):
        """Update price filter range based on actual sensor data."""
        if not sensors_data:
            return

        # Extract price data from sensors
        prices = []
        for sensor in sensors_data:
            if "price_range" in sensor:
                price_data = sensor["price_range"]

                # Handle different price data formats
                if isinstance(price_data, dict):
                    if "avg" in price_data:
                        prices.append(price_data["avg"])
                    elif "min_price" in price_data and "max_price" in price_data:
                        avg_price = (
                            price_data["min_price"] + price_data["max_price"]
                        ) / 2
                        prices.append(avg_price)
                elif isinstance(price_data, (int, float)):
                    prices.append(price_data)

        if prices:
            min_price = int(min(prices))
            max_price = int(max(prices))

            # Add some padding to the range for better UX
            min_price = max(0, min_price - 100)  # Don't go below 0
            max_price = max_price + 500  # Add padding for high-end

            # Update the price filter widget with actual data range
            current_min, current_max = self.price_filter.get_range()

            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                f"Price filter range update requested: ${min_price:,} - ${max_price:,}, current user selection: ({current_min}, {current_max})"
            )

            # Create new price filter with updated range
            self.price_filter.min_value = min_price
            self.price_filter.max_value = max_price

            # Update the range labels
            self.price_filter.min_range_label.setText(f"${min_price:,}")
            self.price_filter.max_range_label.setText(f"${max_price:,}")

            # Update slider ranges while preserving current user selection
            self.price_filter.range_slider.setRange(min_price, max_price)

            # Ensure current user selection is preserved within the new bounds
            preserved_min = max(min_price, min(current_min, max_price))
            preserved_max = max(min_price, min(current_max, max_price))

            # Update the slider values to preserve user selection
            if hasattr(self.price_filter.range_slider, "setValue"):
                try:
                    if hasattr(self.price_filter.range_slider, "value") and callable(
                        getattr(self.price_filter.range_slider, "value")
                    ):
                        # This is superqt.QRangeSlider
                        self.price_filter.range_slider.setValue(
                            (preserved_min, preserved_max)
                        )
                        logger.debug(
                            f"Preserved user selection: ({preserved_min}, {preserved_max}) within bounds ({min_price}, {max_price})"
                        )
                    else:
                        # This is fallback QSlider
                        self.price_filter.range_slider.setValue(preserved_max)
                        logger.debug(
                            f"Preserved max value: {preserved_max} within bounds ({min_price}, {max_price})"
                        )
                except Exception as e:
                    logger.warning(f"Failed to preserve slider values: {e}")

            # Update the internal current values
            self.price_filter.current_min = preserved_min
            self.price_filter.current_max = preserved_max

            # Update input validators
            self.price_filter.min_input.setValidator(
                QIntValidator(min_price, max_price)
            )
            self.price_filter.max_input.setValidator(
                QIntValidator(min_price, max_price)
            )

            # Reset to full range if this is initial setup
            if current_min is None and current_max is None:
                self.price_filter.current_min = min_price
                self.price_filter.current_max = max_price
                self.price_filter._update_slider_values()
                self.price_filter.update_display()
                logger.debug(
                    f"Price filter initialized to full range: ${min_price:,} - ${max_price:,}"
                )
            else:
                logger.debug(
                    f"Preserving user price selection: ${current_min:,} - ${current_max:,}"
                )

    def update_fonts(self):
        """Update fonts for dynamic scaling."""
        try:
            # Update all group box titles (these are section headers)
            group_boxes = self.findChildren(QGroupBox)
            for group_box in group_boxes:
                group_box.setFont(create_styled_font("section_header", "medium"))

            # Update labels
            labels = self.findChildren(QLabel)
            for label in labels:
                label.setFont(create_styled_font("body"))

            # Update fuzzy search combo boxes
            fuzzy_combos = self.findChildren(FuzzySearchComboBox)
            for combo in fuzzy_combos:
                combo.setFont(create_styled_font("body"))

            # Update fuzzy search line edits
            fuzzy_edits = self.findChildren(FuzzySearchLineEdit)
            for edit in fuzzy_edits:
                edit.setFont(create_styled_font("body"))

            # Update regular input widgets (if any remain)
            for widget_type in [QLineEdit, QComboBox, QPushButton, QCheckBox]:
                widgets = self.findChildren(widget_type)
                for widget in widgets:
                    # Skip fuzzy widgets as they're already handled
                    if not isinstance(
                        widget, (FuzzySearchComboBox, FuzzySearchLineEdit)
                    ):
                        widget.setFont(create_styled_font("body"))

            # Update spin boxes
            spin_boxes = self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox)
            for spin_box in spin_boxes:
                spin_box.setFont(create_styled_font("body"))

            # Force update
            self.update()

        except Exception as e:
            # Fallback to basic font update if dynamic system fails
            logger.warning(f"Font update failed: {e}")
            from PySide6.QtGui import QFont

            basic_font = QFont()
            basic_font.setPointSize(10)
            self.setFont(basic_font)

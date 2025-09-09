"""
Side-by-Side Sensor Comparison Dialog

Provides a standard comparison view for multiple sensors with
highlighted differences and synchronized scrolling.
"""

import logging
from typing import Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..utils.font_manager import create_styled_font

logger = logging.getLogger(__name__)


class ComparisonDialog(QDialog):
    """
    Modern side-by-side sensor comparison dialog.

    Features:
    - Side-by-side comparison of up to 5 sensors
    - Highlighted differences
    - Category grouping (Basic, Technical, ROS, etc.)
    - Export comparison results
    - Synchronized scrolling
    """

    def __init__(self, sensors: List[Dict[str, Any]], parent=None):
        super().__init__(parent)

        if not sensors:
            logger.warning("ComparisonDialog created with no sensors")
            return

        # Allow more sensors but warn about usability
        max_sensors = 8  # Increased limit with horizontal scrolling
        if len(sensors) > max_sensors:
            logger.warning(
                f"Limiting comparison to {max_sensors} sensors (received {len(sensors)})"
            )
            self.sensors = sensors[:max_sensors]
        else:
            self.sensors = sensors

        logger.info(f"ComparisonDialog initialized with {len(self.sensors)} sensors")

        self.setup_ui()
        self.populate_comparison()

    def setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle(f"Sensor Comparison - {len(self.sensors)} sensors")
        self.setModal(True)

        # Calculate dynamic dialog size based on number of sensors
        screen_size = self.screen().availableGeometry()

        # Dynamic width calculation
        property_column_width = 200
        min_sensor_column_width = 220  # Minimum readable width per sensor
        optimal_sensor_column_width = 280  # Optimal width per sensor

        # Calculate required width for optimal viewing
        optimal_width = (
            property_column_width
            + (len(self.sensors) * optimal_sensor_column_width)
            + 100
        )  # +100 for margins/scrollbar
        min_width = (
            property_column_width + (len(self.sensors) * min_sensor_column_width) + 100
        )

        # Respect screen boundaries but prioritize content visibility
        max_screen_width = int(screen_size.width() * 0.95)  # Use up to 95% of screen
        absolute_min_width = 800  # Never go below this width

        if optimal_width <= max_screen_width:
            dialog_width = max(optimal_width, absolute_min_width)
        elif min_width <= max_screen_width:
            dialog_width = max_screen_width
        else:
            # For very large sensor counts, use full screen width
            dialog_width = max_screen_width

        # Set minimum dialog width to ensure usability
        self.setMinimumWidth(absolute_min_width)

        # Height calculation - more rows needed for detailed comparison
        dialog_height = min(screen_size.height() * 0.9, 1000)  # Use more vertical space

        self.resize(dialog_width, dialog_height)

        # Store calculated widths for table setup
        self.property_column_width = property_column_width
        self.available_sensor_width = (
            dialog_width - property_column_width - 120
        )  # Account for margins and scrollbar

        # Main layout
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()

        title_label = QLabel(f"Comparing {len(self.sensors)} Sensors")
        title_label.setFont(create_styled_font("h2", "bold"))
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Options
        self.highlight_differences = QCheckBox("Highlight Differences")
        self.highlight_differences.setChecked(True)
        self.highlight_differences.toggled.connect(self.update_highlighting)
        header_layout.addWidget(self.highlight_differences)

        self.view_mode = QComboBox()
        self.view_mode.addItems(
            ["All Properties", "Key Specs Only", "Differences Only"]
        )
        self.view_mode.currentTextChanged.connect(self.update_view_mode)
        header_layout.addWidget(self.view_mode)

        layout.addLayout(header_layout)

        # Main comparison table
        self.comparison_table = QTableWidget()
        self.setup_comparison_table()
        layout.addWidget(self.comparison_table)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        export_button = QPushButton("Export Comparison")
        export_button.clicked.connect(self.export_comparison)
        button_layout.addWidget(export_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setDefault(True)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

    def setup_comparison_table(self):
        """Configure the comparison table."""
        # Set up columns (property name + one column per sensor)
        self.comparison_table.setColumnCount(len(self.sensors) + 1)

        # Set headers
        headers = ["Property"]
        for sensor in self.sensors:
            model = sensor.get("model", "Unknown")
            manufacturer = sensor.get("manufacturer", "")
            header = f"{manufacturer}\n{model}" if manufacturer else model
            headers.append(header)

        self.comparison_table.setHorizontalHeaderLabels(headers)

        # Configure table properties
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setSelectionMode(QTableWidget.NoSelection)
        self.comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Dynamic column width calculation
        self.comparison_table.setColumnWidth(
            0, self.property_column_width
        )  # Property name column

        # Calculate sensor column widths to use available space optimally
        sensor_column_width = max(
            200, self.available_sensor_width // len(self.sensors)
        )  # Minimum 200px per sensor

        for i in range(1, len(self.sensors) + 1):
            self.comparison_table.setColumnWidth(i, sensor_column_width)

        # Configure headers for better readability
        header = self.comparison_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setMinimumHeight(60)  # More space for two-line headers
        header.setStretchLastSection(True)  # Last column takes any remaining space

        # Enable horizontal scrolling if needed for many sensors
        self.comparison_table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)

        # Apply styling
        self.apply_table_styling()

    def apply_table_styling(self):
        """Apply standard styling to the comparison table."""
        self.comparison_table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: white;
                alternate-background-color: #f8f9fa;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                color: #000000;
            }
            QTableWidget::item {
                padding: 8px;
                border-right: 1px solid #e0e0e0;
                color: #000000;
                background-color: white;
            }
            QHeaderView::section {
                background-color: #f1f3f4;
                padding: 8px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
                color: #333333;
                text-align: center;
            }
            QTableWidget::item:alternate {
                background-color: #f8f9fa;
                color: #000000;
            }
        """
        )

    def populate_comparison(self):
        """Populate the comparison table with sensor data."""

        # Define comparison categories and properties
        categories = [
            {
                "name": "Basic Information",
                "properties": [
                    ("manufacturer", "Manufacturer"),
                    ("model", "Model"),
                    ("sensor_type", "Sensor Type"),
                    ("technology", "Technology"),
                ],
            },
            {
                "name": "Image Specifications",
                "properties": [
                    ("resolution_display", "Resolution"),
                    ("frame_rate_display", "Frame Rate"),
                    ("pixel_size", "Pixel Size"),
                    ("sensor_format", "Sensor Format"),
                ],
            },
            {
                "name": "Optical Properties",
                "properties": [
                    ("field_of_view", "Field of View"),
                    ("min_range", "Min Range"),
                    ("max_range", "Max Range"),
                    ("accuracy", "Accuracy"),
                    ("depth_resolution", "Depth Resolution"),
                ],
            },
            {
                "name": "Physical Properties",
                "properties": [
                    ("dimensions", "Dimensions"),
                    ("weight", "Weight"),
                    ("power_consumption", "Power"),
                    ("operating_temperature", "Operating Temp"),
                ],
            },
            {
                "name": "Connectivity",
                "properties": [
                    ("communication_interface", "Interface"),
                    ("data_rate", "Data Rate"),
                    ("sync_support", "Sync Support"),
                ],
            },
            {
                "name": "ROS Integration",
                "properties": [
                    ("ros_support", "ROS Support"),
                    ("ros1_driver", "ROS1 Driver"),
                    ("ros2_driver", "ROS2 Driver"),
                ],
            },
            {
                "name": "Commercial",
                "properties": [
                    ("price_display", "Price"),
                    ("availability", "Availability"),
                    ("warranty", "Warranty"),
                ],
            },
        ]

        row = 0
        for category in categories:
            # Add category header row
            self.comparison_table.insertRow(row)

            # Category header in first column
            category_item = QTableWidgetItem(category["name"])
            # Use standard font instead of custom font manager
            header_font = QFont("Arial", 11)
            header_font.setWeight(QFont.Weight.Bold)
            category_item.setFont(header_font)
            category_item.setBackground(QBrush(QColor(230, 235, 240)))
            category_item.setForeground(QBrush(QColor(0, 0, 0)))  # Black text
            self.comparison_table.setItem(row, 0, category_item)

            # Span the category header across all sensor columns
            for col in range(1, len(self.sensors) + 1):
                empty_item = QTableWidgetItem("")
                empty_item.setBackground(QBrush(QColor(230, 235, 240)))
                self.comparison_table.setItem(row, col, empty_item)

            row += 1

            # Add properties for this category
            for prop_key, prop_display in category["properties"]:
                self.comparison_table.insertRow(row)

                # Property name
                prop_item = QTableWidgetItem(prop_display)
                # Use standard font instead of custom font manager
                standard_font = QFont("Arial", 10)
                standard_font.setWeight(QFont.Weight.Medium)
                prop_item.setFont(standard_font)
                # Ensure text is visible
                prop_item.setForeground(QBrush(QColor(0, 0, 0)))  # Black text
                self.comparison_table.setItem(row, 0, prop_item)

                # Values for each sensor
                values = []
                for col, sensor in enumerate(self.sensors, 1):
                    value = self.get_sensor_property(sensor, prop_key)
                    value_item = QTableWidgetItem(value)
                    # Use standard font instead of custom font manager
                    value_font = QFont("Arial", 10)
                    value_item.setFont(value_font)
                    # Ensure text is visible
                    value_item.setForeground(QBrush(QColor(0, 0, 0)))  # Black text
                    values.append(value)
                    self.comparison_table.setItem(row, col, value_item)

                # Highlight differences if enabled
                if self.highlight_differences.isChecked():
                    self.highlight_row_differences(row, values)

                row += 1

        # Adjust row heights for better readability
        for i in range(self.comparison_table.rowCount()):
            self.comparison_table.setRowHeight(i, 35)

        # Ensure table is properly displayed
        self.comparison_table.resizeColumnsToContents()
        self.comparison_table.resizeRowsToContents()
        self.comparison_table.show()
        self.comparison_table.setVisible(True)

        # Force layout update
        self.layout().update()
        self.update()

    def get_sensor_property(self, sensor: Dict[str, Any], property_key: str) -> str:
        """
        Get a formatted property value from sensor data.

        Handles special formatting for display properties and nested values.
        """
        # Test basic properties first
        if property_key in ["manufacturer", "model", "sensor_type"]:
            value = sensor.get(property_key, "N/A")
            return str(value)
        # Special handling for display properties
        if property_key == "resolution_display":
            rgb_res = sensor.get("resolution_rgb", {})
            depth_res = sensor.get("resolution_depth", {})

            resolutions = []
            if rgb_res:
                width = rgb_res.get("width", 0)
                height = rgb_res.get("height", 0)
                if width and height:
                    mp = (width * height) / 1_000_000
                    resolutions.append(f"RGB: {width}x{height} ({mp:.1f}MP)")

            if depth_res:
                width = depth_res.get("width", 0)
                height = depth_res.get("height", 0)
                if width and height:
                    resolutions.append(f"Depth: {width}x{height}")

            return "\n".join(resolutions) if resolutions else "N/A"

        elif property_key == "frame_rate_display":
            frame_rate = sensor.get("frame_rate")
            if isinstance(frame_rate, dict):
                max_rate = frame_rate.get("max", 0)
                return f"{max_rate} FPS" if max_rate else "N/A"
            elif frame_rate:
                return f"{frame_rate} FPS"
            return "N/A"

        elif property_key == "price_display":
            price = sensor.get("price_range", {})
            if isinstance(price, dict):
                avg = price.get("avg")
                if avg:
                    return f"${avg:,}"
                min_p = price.get("min")
                max_p = price.get("max")
                if min_p and max_p:
                    return f"${min_p:,}-${max_p:,}"
            return "N/A"

        elif property_key == "ros_support":
            ros = sensor.get("ros_compatibility", [])
            if ros:
                return ", ".join(ros)
            return "None"

        elif property_key == "field_of_view":
            fov = sensor.get("field_of_view", {})
            if isinstance(fov, dict):
                h = fov.get("horizontal")
                v = fov.get("vertical")
                if h and v:
                    return f"H: {h}°, V: {v}°"
            return "N/A"

        elif property_key == "dimensions":
            dims = sensor.get("dimensions", {})
            # The YAML uses 'size' instead of 'dimensions'
            if not dims:
                dims = sensor.get("size", {})
            if isinstance(dims, dict):
                length = dims.get("length")
                w = dims.get("width")
                h = dims.get("height")
                if length and w and h:
                    return f"{length}x{w}x{h}mm"
            return "N/A"

        # Default handling
        value = sensor.get(property_key)

        if value is None:
            return "N/A"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        elif isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            # Try to format dict nicely
            if "min" in value and "max" in value:
                return f"{value['min']}-{value['max']}"
            return str(value)
        else:
            return str(value)

    def highlight_row_differences(self, row: int, values: List[str]):
        """
        Highlight cells in a row that have different values.

        Args:
            row: Row index
            values: List of values for each sensor
        """
        # Check if all values are the same
        unique_values = set(values)

        if len(unique_values) > 1:
            # Values differ - highlight cells with less common values
            value_counts = {}
            for v in values:
                value_counts[v] = value_counts.get(v, 0) + 1

            # Find the most common value
            most_common = max(value_counts, key=value_counts.get)
            most_common_count = value_counts[most_common]

            # Highlight cells that differ from the most common
            for col, value in enumerate(values, 1):
                item = self.comparison_table.item(row, col)
                if item:
                    if value != most_common:
                        # Different value - highlight in light yellow
                        item.setBackground(QBrush(QColor(255, 253, 184)))
                    elif most_common_count == 1:
                        # All values are unique - highlight all in very light yellow
                        item.setBackground(QBrush(QColor(255, 255, 224)))

    def update_highlighting(self):
        """Update difference highlighting based on checkbox state."""
        # Re-populate to apply/remove highlighting
        self.comparison_table.clearContents()
        self.comparison_table.setRowCount(0)
        self.populate_comparison()

    def update_view_mode(self, mode: str):
        """Update the view based on selected mode."""
        # TODO: Implement view mode filtering
        # For now, just repopulate
        self.comparison_table.clearContents()
        self.comparison_table.setRowCount(0)
        self.populate_comparison()

    def resizeEvent(self, event):
        """Handle dialog resize to adjust column widths dynamically."""
        super().resizeEvent(event)

        if (
            hasattr(self, "comparison_table")
            and self.comparison_table.columnCount() > 1
        ):
            # Recalculate available width for sensor columns
            new_available_width = self.width() - self.property_column_width - 120

            # Ensure minimum width per sensor column
            sensor_column_width = max(180, new_available_width // len(self.sensors))

            # Update sensor column widths
            for i in range(1, len(self.sensors) + 1):
                self.comparison_table.setColumnWidth(i, sensor_column_width)

    def export_comparison(self):
        """Export the comparison to a file."""
        # TODO: Implement export functionality
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(
            self, "Export", "Export functionality will save comparison as CSV/PDF"
        )

    def update_fonts(self):
        """Update fonts for dynamic scaling."""
        # Update title
        for widget in self.findChildren(QLabel):
            if widget.text().startswith("Comparing"):
                widget.setFont(create_styled_font("h2", "bold"))

        # Update table fonts if needed
        for row in range(self.comparison_table.rowCount()):
            for col in range(self.comparison_table.columnCount()):
                item = self.comparison_table.item(row, col)
                if item:
                    # Check if it's a category header
                    if col == 0 and item.font().bold():
                        item.setFont(create_styled_font("h4", "bold"))
                    else:
                        item.setFont(create_styled_font("body"))

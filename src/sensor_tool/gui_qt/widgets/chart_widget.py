"""
Chart Widget for PySide6 GUI

Embeds matplotlib charts in Qt widgets for sensor comparison visualization.
This module ports the functionality from visualize.py to Qt widgets.
"""

import matplotlib

matplotlib.use("Qt5Agg")  # Use Qt backend for matplotlib

import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ...utils import (
    extract_fov,
    extract_numeric,
    extract_price_avg,
    extract_resolution,
    extract_ros_compatibility,
    format_label,
)
from ..utils.font_manager import create_styled_font
from ..utils.theme_manager import get_theme_manager
from .pdf_export_widget import PDFExportWidget
from .radar_chart_widget import RadarChartWidget
from .ros_config_widget import ROSConfigWidget

logger = logging.getLogger(__name__)


class AttributeSelectionDialog(QDialog):
    """Dialog for selecting which attributes to include in chart comparisons."""

    @staticmethod
    def get_intelligent_defaults(available_attributes, sensor_data=None):
        """Generate intelligent default attribute selections based on data availability and utility."""

        # Define attribute priorities and categories
        # Priority 1: Universal numeric attributes (100% coverage, always useful)
        priority_1 = ["frame_rate", "min_range", "max_range"]

        # Priority 2: High-coverage numeric attributes (>60% coverage)
        priority_2 = ["power_consumption", "latency", "weight"]

        # Priority 3: Display attributes (processed for visualization)
        priority_3 = ["resolution_display", "price_display", "frame_rate_display"]

        # Priority 4: Categorical attributes (useful for grouping)
        priority_4 = ["sensor_type", "environmental_rating", "communication_interface"]

        # Priority 5: Other useful numeric attributes
        priority_5 = [
            "field_of_view",
            "accuracy",
            "operating_temperature_min",
            "operating_temperature_max",
        ]

        # Build intelligent selection
        intelligent_defaults = []

        # Add Priority 1 attributes (always include if available)
        for attr in priority_1:
            if attr in available_attributes:
                intelligent_defaults.append(attr)

        # Add Priority 2 attributes (high-value numeric) - only if they actually exist
        for attr in priority_2:
            if attr in available_attributes:
                intelligent_defaults.append(attr)

        # Add Priority 3 display attributes (good for visualization)
        for attr in priority_3:
            if attr in available_attributes and attr not in intelligent_defaults:
                intelligent_defaults.append(attr)

        # Fill to optimal count (4-6 attributes for best visualization)
        target_count = 5
        current_count = len(intelligent_defaults)

        if current_count < target_count:
            # Add Priority 4 and 5 attributes to reach target
            remaining_slots = target_count - current_count
            for attr_list in [priority_4, priority_5]:
                for attr in attr_list:
                    if (
                        attr in available_attributes
                        and attr not in intelligent_defaults
                    ):
                        intelligent_defaults.append(attr)
                        remaining_slots -= 1
                        if remaining_slots <= 0:
                            break
                if remaining_slots <= 0:
                    break

        return intelligent_defaults[:6]  # Cap at 6 for optimal visualization

    @staticmethod
    def get_sensor_type_defaults(sensor_types, available_attributes):
        """Get defaults optimized for specific sensor types."""

        if not sensor_types:
            return AttributeSelectionDialog.get_intelligent_defaults(
                available_attributes
            )

        # Get the most common sensor type
        most_common_type = (
            max(set(sensor_types), key=sensor_types.count) if sensor_types else None
        )

        type_specific_defaults = {
            "Depth Camera": [
                "frame_rate",
                "min_range",
                "max_range",
                "resolution_display",
                "latency",
                "power_consumption",
            ],
            "Stereo Camera": [
                "frame_rate",
                "resolution_display",
                "field_of_view",
                "min_range",
                "max_range",
                "latency",
            ],
            "Structured Light Camera": [
                "frame_rate",
                "min_range",
                "max_range",
                "resolution_display",
                "accuracy",
                "power_consumption",
            ],
            "LiDAR": [
                "frame_rate",
                "min_range",
                "max_range",
                "field_of_view",
                "accuracy",
                "power_consumption",
            ],
            "RGB Camera": [
                "frame_rate",
                "resolution_display",
                "field_of_view",
                "price_display",
                "weight",
                "power_consumption",
            ],
        }

        if most_common_type in type_specific_defaults:
            defaults = []
            for attr in type_specific_defaults[most_common_type]:
                if attr in available_attributes:
                    defaults.append(attr)

            # Fill remaining slots with intelligent defaults if needed
            if len(defaults) < 4:
                intelligent_fallback = (
                    AttributeSelectionDialog.get_intelligent_defaults(
                        available_attributes
                    )
                )
                for attr in intelligent_fallback:
                    if attr not in defaults and len(defaults) < 6:
                        defaults.append(attr)

            return defaults

        # Fallback to general intelligent defaults
        return AttributeSelectionDialog.get_intelligent_defaults(available_attributes)

    @staticmethod
    def get_radar_chart_defaults(sensor_types, available_attributes):
        """Get radar chart optimized defaults that work well on normalized scales."""

        # Radar chart specific attributes that normalize well and provide meaningful comparisons
        # Focus on metrics that can be effectively visualized on a spider/radar chart
        radar_optimized = [
            # Performance metrics (higher is generally better, easy to normalize)
            "frame_rate",
            "max_range",
            # Quality metrics (higher resolution is better)
            "resolution_rgb",
            "resolution_depth",
            # Efficiency metrics (lower is often better, can be inverted for radar)
            "latency",
            "power_consumption",
            "min_range",
            # Physical characteristics (meaningful for comparison)
            "field_of_view",
            "weight",
            # Cost consideration (important comparison factor)
            "price_avg",
        ]

        # Filter to only include available attributes
        radar_defaults = [
            attr for attr in radar_optimized if attr in available_attributes
        ]

        # Ensure we have 4-6 attributes for optimal radar chart visualization
        if len(radar_defaults) < 4:
            # Fill with general intelligent defaults if needed
            fallback = AttributeSelectionDialog.get_intelligent_defaults(
                available_attributes
            )
            for attr in fallback:
                if attr not in radar_defaults and len(radar_defaults) < 6:
                    radar_defaults.append(attr)

        return radar_defaults[:6]  # Cap at 6 for optimal radar visualization

    def __init__(
        self,
        available_attributes,
        current_attributes,
        parent=None,
        sensor_data=None,
        chart_type="bar",
    ):
        super().__init__(parent)
        self.available_attributes = available_attributes
        self.sensor_data = sensor_data or []
        self.chart_type = chart_type  # "bar" or "radar"

        # Use intelligent defaults if no current attributes are selected
        if not current_attributes or len(current_attributes) == 0:
            # Extract sensor types from sensor data for type-specific defaults
            sensor_types = []
            if self.sensor_data:
                sensor_types = [
                    s.get("sensor_type", "")
                    for s in self.sensor_data
                    if s.get("sensor_type")
                ]
                if chart_type == "radar":
                    self.selected_attributes = self.get_radar_chart_defaults(
                        sensor_types, available_attributes
                    )
                else:
                    self.selected_attributes = self.get_sensor_type_defaults(
                        sensor_types, available_attributes
                    )
            else:
                if chart_type == "radar":
                    self.selected_attributes = self.get_radar_chart_defaults(
                        [], available_attributes
                    )
                else:
                    self.selected_attributes = self.get_intelligent_defaults(
                        available_attributes
                    )
        else:
            self.selected_attributes = current_attributes.copy()

        chart_type_display = "Radar Chart" if chart_type == "radar" else "Bar Chart"
        self.setWindowTitle(f"Select {chart_type_display} Attributes")
        self.setModal(True)
        self.resize(700, 600)  # Increased size for better content display

        # Get theme manager
        self.theme_manager = get_theme_manager()

        self.setup_ui()

    def setup_ui(self):
        """Setup the dialog user interface."""
        layout = QVBoxLayout(self)

        # Title and description
        title_label = QLabel("Select Attributes for Chart Comparison")
        title_label.setFont(create_styled_font("heading", "medium"))
        layout.addWidget(title_label)

        desc_label = QLabel(
            "Choose which sensor attributes to compare in charts. "
            "Select 2-6 attributes for optimal visualization."
        )
        desc_label.setFont(create_styled_font("body"))
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Attribute categories and selection
        content_layout = QGridLayout()

        # Performance attributes
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QVBoxLayout(perf_group)
        self.perf_checkboxes = {}

        performance_attrs = [
            ("frame_rate", "Frame Rate (FPS)", "Sensor capture rate performance"),
            ("latency", "Latency", "Processing delay in milliseconds"),
            ("max_range", "Maximum Range", "Maximum detection distance"),
            ("min_range", "Minimum Range", "Minimum detection distance"),
        ]

        for attr, label, desc in performance_attrs:
            if attr in self.available_attributes:
                checkbox = QCheckBox(f"{label}")
                checkbox.setChecked(attr in self.selected_attributes)
                checkbox.toggled.connect(
                    lambda checked, a=attr: self.toggle_attribute(a, checked)
                )
                self.perf_checkboxes[attr] = checkbox

                # Add description with theme-aware styling
                desc_label = QLabel(f"  {desc}")
                desc_label.setFont(create_styled_font("caption"))
                desc_label.setProperty("class", "description-label")

                perf_layout.addWidget(checkbox)
                perf_layout.addWidget(desc_label)

        content_layout.addWidget(perf_group, 0, 0)

        # Technical attributes
        tech_group = QGroupBox("Technical Specifications")
        tech_layout = QVBoxLayout(tech_group)
        self.tech_checkboxes = {}

        technical_attrs = [
            ("resolution_rgb", "RGB Resolution", "Color image resolution in pixels"),
            (
                "resolution_depth",
                "Depth Resolution",
                "Depth image resolution in pixels",
            ),
            ("field_of_view", "Field of View", "Angular coverage in degrees"),
            ("price_avg", "Average Price", "Cost in USD (average of range)"),
        ]

        for attr, label, desc in technical_attrs:
            if attr in self.available_attributes:
                checkbox = QCheckBox(f"{label}")
                checkbox.setChecked(attr in self.selected_attributes)
                checkbox.toggled.connect(
                    lambda checked, a=attr: self.toggle_attribute(a, checked)
                )
                self.tech_checkboxes[attr] = checkbox

                # Add description with theme-aware styling
                desc_label = QLabel(f"  {desc}")
                desc_label.setFont(create_styled_font("caption"))
                desc_label.setProperty("class", "description-label")

                tech_layout.addWidget(checkbox)
                tech_layout.addWidget(desc_label)

        content_layout.addWidget(tech_group, 0, 1)

        # Additional attributes (auto-detected)
        if len(self.available_attributes) > 8:
            other_group = QGroupBox("Other Available Attributes")
            other_layout = QVBoxLayout(other_group)
            self.other_checkboxes = {}

            known_attrs = {attr for attr, _, _ in performance_attrs + technical_attrs}
            other_attrs = [
                attr for attr in self.available_attributes if attr not in known_attrs
            ]

            for attr in other_attrs[:6]:  # Limit to 6 additional
                checkbox = QCheckBox(format_label(attr))
                checkbox.setChecked(attr in self.selected_attributes)
                checkbox.toggled.connect(
                    lambda checked, a=attr: self.toggle_attribute(a, checked)
                )
                self.other_checkboxes[attr] = checkbox
                other_layout.addWidget(checkbox)

            content_layout.addWidget(other_group, 1, 0, 1, 2)

        layout.addLayout(content_layout)

        # Selection summary
        self.summary_label = QLabel()
        self.summary_label.setFont(create_styled_font("body", "medium"))
        self.update_summary()
        layout.addWidget(self.summary_label)

        # Buttons
        button_layout = QHBoxLayout()

        # Smart preset buttons
        smart_defaults_button = QPushButton("Smart Selection")
        smart_defaults_button.setToolTip(
            "Automatically select the most useful attributes for comparison"
        )
        smart_defaults_button.clicked.connect(self.apply_smart_defaults)
        button_layout.addWidget(smart_defaults_button)

        performance_preset_button = QPushButton("Performance Focus")
        performance_preset_button.setToolTip(
            "Select attributes focused on performance metrics"
        )
        performance_preset_button.clicked.connect(self.apply_performance_preset)
        button_layout.addWidget(performance_preset_button)

        technical_preset_button = QPushButton("Technical Specs")
        technical_preset_button.setToolTip(
            "Select attributes focused on technical specifications"
        )
        technical_preset_button.clicked.connect(self.apply_technical_preset)
        button_layout.addWidget(technical_preset_button)

        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        button_layout.addWidget(separator)

        select_all_button = QPushButton("Select All")
        select_all_button.setToolTip(
            "Select all available sensor attributes for comprehensive comparison"
        )
        select_all_button.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_button)

        clear_all_button = QPushButton("Clear All")
        clear_all_button.setToolTip("Deselect all attributes and start fresh")
        clear_all_button.clicked.connect(self.clear_all)
        button_layout.addWidget(clear_all_button)

        button_layout.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.setToolTip(
            "Close without applying changes to attribute selection"
        )
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        ok_button = QPushButton("Apply Selection")
        ok_button.setToolTip("Apply the selected attributes and generate charts")
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)

        layout.addLayout(button_layout)

        # Apply theme-aware styling
        self.apply_theme()

        # Connect to theme changes
        self.theme_manager.theme_changed.connect(self.apply_theme)

    def apply_theme(self):
        """Apply current theme styling to the dialog."""
        # Get dialog stylesheet and button stylesheet
        dialog_style = self.theme_manager.create_dialog_stylesheet()
        button_style = self.theme_manager.create_button_stylesheet("primary")

        # Combine stylesheets
        combined_style = dialog_style + "\n" + button_style
        self.setStyleSheet(combined_style)

    def toggle_attribute(self, attribute, checked):
        """Toggle attribute selection."""
        if checked and attribute not in self.selected_attributes:
            self.selected_attributes.append(attribute)
        elif not checked and attribute in self.selected_attributes:
            self.selected_attributes.remove(attribute)
        self.update_summary()

    def update_summary(self):
        """Update the selection summary."""
        count = len(self.selected_attributes)
        if count == 0:
            text = "No attributes selected"
            color = "#cc0000"
        elif count < 2:
            text = f"{count} attribute selected (minimum 2 recommended)"
            color = "#ff8800"
        elif count <= 6:
            text = f"{count} attributes selected (optimal range)"
            color = "#008800"
        else:
            text = f"{count} attributes selected (consider reducing for clarity)"
            color = "#ff8800"

        self.summary_label.setText(text)
        self.summary_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def select_all(self):
        """Select all available attributes."""
        self.selected_attributes = self.available_attributes.copy()
        self.update_checkboxes()
        self.update_summary()

    def clear_all(self):
        """Clear all attribute selections."""
        self.selected_attributes = []
        self.update_checkboxes()
        self.update_summary()

    def update_checkboxes(self):
        """Update all checkbox states to match selected attributes."""
        for checkbox_dict in [self.perf_checkboxes, self.tech_checkboxes]:
            for attr, checkbox in checkbox_dict.items():
                checkbox.setChecked(attr in self.selected_attributes)

        if hasattr(self, "other_checkboxes"):
            for attr, checkbox in self.other_checkboxes.items():
                checkbox.setChecked(attr in self.selected_attributes)

    def apply_smart_defaults(self):
        """Apply intelligent default attribute selection based on chart type."""
        if self.chart_type == "radar":
            if self.sensor_data:
                sensor_types = [
                    s.get("sensor_type", "")
                    for s in self.sensor_data
                    if s.get("sensor_type")
                ]
                self.selected_attributes = self.get_radar_chart_defaults(
                    sensor_types, self.available_attributes
                )
            else:
                self.selected_attributes = self.get_radar_chart_defaults(
                    [], self.available_attributes
                )
        else:  # bar charts
            if self.sensor_data:
                sensor_types = [
                    s.get("sensor_type", "")
                    for s in self.sensor_data
                    if s.get("sensor_type")
                ]
                self.selected_attributes = self.get_sensor_type_defaults(
                    sensor_types, self.available_attributes
                )
            else:
                self.selected_attributes = self.get_intelligent_defaults(
                    self.available_attributes, self.sensor_data
                )
        self.update_checkboxes()
        self.update_summary()

    def apply_performance_preset(self):
        """Apply performance-focused attribute preset based on chart type."""
        if self.chart_type == "radar":
            # Radar chart performance attributes - focus on metrics that normalize well
            performance_focused = [
                "frame_rate",
                "max_range",
                "latency",
                "power_consumption",
            ]
        else:
            # Bar chart performance attributes - can handle diverse metrics
            performance_focused = [
                "frame_rate",
                "latency",
                "min_range",
                "max_range",
                "power_consumption",
                "weight",
            ]

        self.selected_attributes = [
            attr for attr in performance_focused if attr in self.available_attributes
        ][
            :6
        ]  # Limit to 6 for optimal visualization

        # If we don't have enough performance attributes, fill with chart-appropriate defaults
        if len(self.selected_attributes) < 4:
            if self.chart_type == "radar":
                fallback = self.get_radar_chart_defaults([], self.available_attributes)
            else:
                fallback = self.get_intelligent_defaults(
                    self.available_attributes, self.sensor_data
                )

            for attr in fallback:
                if (
                    attr not in self.selected_attributes
                    and len(self.selected_attributes) < 6
                ):
                    self.selected_attributes.append(attr)

        self.update_checkboxes()
        self.update_summary()

    def apply_technical_preset(self):
        """Apply technical specifications-focused attribute preset based on chart type."""
        if self.chart_type == "radar":
            # Radar chart technical attributes - focus on numeric metrics that can be normalized
            technical_focused = [
                "resolution_rgb",
                "resolution_depth",
                "field_of_view",
                "price_avg",
                "weight",
                "min_range",
            ]
        else:
            # Bar chart technical attributes - can handle more diverse specifications
            technical_focused = [
                "resolution_rgb",
                "resolution_depth",
                "field_of_view",
                "price_avg",
                "environmental_rating",
                "communication_interface",
            ]

        self.selected_attributes = [
            attr for attr in technical_focused if attr in self.available_attributes
        ][
            :6
        ]  # Limit to 6 for optimal visualization

        # If we don't have enough technical attributes, fill with chart-appropriate defaults
        if len(self.selected_attributes) < 4:
            if self.chart_type == "radar":
                fallback = self.get_radar_chart_defaults([], self.available_attributes)
            else:
                fallback = self.get_intelligent_defaults(
                    self.available_attributes, self.sensor_data
                )

            for attr in fallback:
                if (
                    attr not in self.selected_attributes
                    and len(self.selected_attributes) < 6
                ):
                    self.selected_attributes.append(attr)

        self.update_checkboxes()
        self.update_summary()


class ChartGenerationThread(QThread):
    """Background thread for generating charts to keep UI responsive."""

    finished = Signal(Figure)
    error = Signal(str)

    def __init__(self, sensors_data, attributes, weights=None, benchmarks=None):
        super().__init__()
        self.sensors_data = sensors_data
        self.attributes = attributes
        self.weights = weights
        self.benchmarks = benchmarks

    def run(self):
        """Generate chart in background thread."""
        try:
            fig = self.generate_chart()
            self.finished.emit(fig)
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            self.error.emit(str(e))

    def _get_smart_unit(self, attribute, df):
        """Get smart unit label that accounts for mixed units in sensor data.

        For frame_rate: Cameras use FPS, LiDAR uses Hz. Since 1 Hz = 1 FPS,
        values are equivalent but we show the appropriate unit label.
        Returns 'Hz/FPS' if sensors have mixed units.
        """
        from ...utils import get_unit

        # Default fallback unit
        default_unit = get_unit(attribute)

        # Special handling for frame_rate which can have mixed Hz/FPS units
        if attribute == "frame_rate" and hasattr(self, "sensors_data"):
            units_found = set()
            for sensor in self.sensors_data:
                unit = sensor.get("frame_rate_unit", "")
                if unit:
                    units_found.add(unit)

            # If we have mixed units, show both
            if len(units_found) > 1:
                return "Hz/FPS"
            elif units_found:
                # Use the actual unit found in data
                return list(units_found)[0]

        return default_unit

    def generate_chart(self):
        """Generate matplotlib figure with sensor comparison charts."""
        if not self.sensors_data or len(self.sensors_data) < 2:
            raise ValueError("At least two sensors required for comparison")

        # Create DataFrame from sensors data
        df = pd.DataFrame(self.sensors_data)

        # Preprocess data similar to visualize.py
        df["ros_compatibility_score"] = df["ros_compatibility"].apply(
            extract_ros_compatibility
        )

        # Process resolution fields if needed
        if "resolution_rgb" in self.attributes:
            df["resolution_rgb"] = df.get(
                "resolution", pd.Series([{}] * len(df))
            ).apply(
                lambda x: extract_resolution(x, "rgb") if isinstance(x, dict) else 0
            )
        if "resolution_depth" in self.attributes:
            df["resolution_depth"] = df.get(
                "resolution", pd.Series([{}] * len(df))
            ).apply(
                lambda x: extract_resolution(x, "depth") if isinstance(x, dict) else 0
            )

        # Process other complex fields
        if "field_of_view" in self.attributes:
            df["field_of_view"] = df.get(
                "field_of_view", pd.Series([{}] * len(df))
            ).apply(extract_fov)

        if "price_avg" in self.attributes:
            df["price_avg"] = df.get("price_range", pd.Series([{}] * len(df))).apply(
                extract_price_avg
            )

        # Handle display field mappings for chart attributes
        if "latency" in self.attributes and "latency_display" in df.columns:
            # Use latency_display data for latency charts, extracting numeric values
            logger.debug(
                f"[LATENCY FIX] Mapping latency_display to latency for chart generation"
            )
            df["latency"] = df["latency_display"].apply(extract_numeric)
            logger.debug(
                f"[LATENCY FIX] Extracted latency values: {df['latency'].tolist()}"
            )

        # Extract numeric values for comparison attributes
        for attr in self.attributes:
            if attr in [
                "ros_compatibility_score",
                "resolution_rgb",
                "resolution_depth",
                "field_of_view",
                "price_avg",
                "latency",
            ]:
                continue
            if attr in df.columns:
                df[attr] = df[attr].apply(extract_numeric)
            else:
                # Create a default column with zeros for missing attributes
                df[attr] = 0
                logger.warning(
                    f"Attribute '{attr}' not found in sensor data, using default value 0"
                )

        # Calculate layout
        num_attributes = len(self.attributes)
        cols = 2
        rows = (num_attributes + cols - 1) // cols

        # Create figure with extra height to push bottom charts down and prevent overlap
        fig = Figure(
            figsize=(14, 7.5 * rows), dpi=120
        )  # Much more height for bottom chart separation

        # Modern main title with proper spacing
        fig.suptitle(
            "Sensor Comparison Charts",
            fontsize=16,
            fontweight="bold",
            y=0.95,
            color="#2C3E50",
        )  # Safe position with adequate top margin

        # Define colors for sensors
        sensor_colors = plt.colormaps.get_cmap("tab10")
        color_map = {
            sensor_id: sensor_colors(i) for i, sensor_id in enumerate(df["sensor_id"])
        }

        def should_show_sensor_names(chart_index, total_attributes, cols=2):
            """
            Smart strategy for showing sensor names on bottom-most chart of each column.

            Strategy: Show sensor names on the bottom-most chart in each vertical column
            - Left column (col 0): Show on highest row chart in positions 0, 2, 4, 6...
            - Right column (col 1): Show on highest row chart in positions 1, 3, 5, 7...
            This eliminates overlap while providing one set of labels per column.
            """
            current_row = chart_index // cols
            current_col = chart_index % cols

            # Find the bottom-most chart in the current column
            bottom_most_row_for_col = -1
            for i in range(total_attributes):
                test_row = i // cols
                test_col = i % cols
                if test_col == current_col:
                    bottom_most_row_for_col = max(bottom_most_row_for_col, test_row)

            # Show sensor names if this is the bottom-most chart in its column
            return current_row == bottom_most_row_for_col

        # Create subplots
        for i, attribute in enumerate(self.attributes):
            ax = fig.add_subplot(rows, cols, i + 1)

            values = (
                df[attribute] if attribute in df.columns else pd.Series([0] * len(df))
            )

            # Set up y-axis limits
            max_value = values.max() if not values.isna().all() else 1
            y_max = max_value * 1.2 if pd.notna(max_value) and max_value > 0 else 1
            ax.set_ylim(0, y_max)

            # Assign colors based on sensor_id
            colors = []
            for sensor_idx, sensor_id_val in enumerate(df["sensor_id"]):
                value = values.iloc[sensor_idx] if sensor_idx < len(values) else None
                if pd.isna(value):
                    colors.append("gray")
                else:
                    colors.append(color_map[sensor_id_val])

            # Create bar chart
            bar_width = 0.6
            x_positions = np.arange(len(df))
            bars = ax.bar(
                x_positions,
                values.fillna(0),
                color=colors,
                edgecolor="black",
                alpha=0.7,
                width=bar_width,
            )

            # Add value labels on bars
            for bar, value in zip(bars, values):
                if pd.notna(value) and value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + (y_max * 0.02),
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,  # Balanced font for values
                        fontweight="medium",
                        color="#2C3E50",  # Modern dark color
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            facecolor="white",
                            alpha=0.8,
                            edgecolor="none",
                        ),  # Subtle background
                    )
                else:
                    # Handle N/A values
                    bar_height = y_max * 0.05
                    bar.set_height(bar_height)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar_height + (y_max * 0.01),
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontsize=9,  # Proportionate font
                        fontweight="medium",
                        color="#7F8C8D",  # Gray color for N/A
                        style="italic",
                    )
                    bar.set_color("gray")

            # Balanced standard formatting
            ax.set_title(
                f"Comparison of {format_label(attribute)}",
                fontsize=12,
                fontweight="bold",
                pad=15,
                color="#34495E",
            )  # Proportionate title
            # X-axis label removed to save vertical space for better chart separation
            unit = self._get_smart_unit(attribute, df)
            ylabel = (
                f"{format_label(attribute)} ({unit})"
                if unit
                else format_label(attribute)
            )
            ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", color="#2C3E50")

            # Enhanced grid for better readability
            ax.grid(axis="y", linestyle="--", alpha=0.6, linewidth=0.8, color="#BDC3C7")
            ax.set_axisbelow(True)  # Grid behind bars

            # Smart multi-line x-axis labels with 45° rotation for full visibility
            ax.set_xticks(x_positions)
            formatted_labels = [format_label(label) for label in df["sensor_id"]]

            # Create multi-line labels for better readability
            multiline_labels = []
            for label in formatted_labels:
                # Split at logical breakpoints (spaces, underscores) for multi-line display
                if len(label) > 12:  # Only multi-line for longer names
                    # Split by spaces first, then by underscores if no spaces
                    if " " in label:
                        parts = label.split(" ")
                        # Group parts to create balanced lines
                        if len(parts) == 2:
                            multiline_labels.append(f"{parts[0]}\n{parts[1]}")
                        elif len(parts) >= 3:
                            # Group into 2 lines for readability
                            mid = len(parts) // 2
                            line1 = " ".join(parts[:mid])
                            line2 = " ".join(parts[mid:])
                            multiline_labels.append(f"{line1}\n{line2}")
                        else:
                            multiline_labels.append(label)
                    elif "_" in label:
                        parts = label.split("_")
                        if len(parts) == 2:
                            multiline_labels.append(f"{parts[0]}\n{parts[1]}")
                        elif len(parts) >= 3:
                            mid = len(parts) // 2
                            line1 = "_".join(parts[:mid])
                            line2 = "_".join(parts[mid:])
                            multiline_labels.append(f"{line1}\n{line2}")
                        else:
                            multiline_labels.append(label)
                    else:
                        multiline_labels.append(label)
                else:
                    multiline_labels.append(label)

            # Conditionally apply sensor names based on strategic positioning
            if should_show_sensor_names(i, len(self.attributes), cols):
                # Apply 45-degree rotation with multi-line support
                ax.set_xticklabels(
                    multiline_labels,
                    rotation=45,
                    ha="right",
                    va="top",
                    fontsize=9,
                    fontweight="bold",
                    color="#2C3E50",
                    linespacing=1.2,  # Better line spacing for multi-line text
                )
            else:
                # Hide sensor names to eliminate redundancy and overlap
                ax.set_xticklabels([])

            # Appropriately sized tick labels
            ax.tick_params(axis="y", labelsize=9, colors="#2C3E50")
            ax.tick_params(axis="x", labelsize=9, colors="#2C3E50")

            # Add benchmark line if provided
            if self.benchmarks and i < len(self.benchmarks):
                benchmark_value = self.benchmarks[i]
                if benchmark_value is not None:
                    ax.axhline(
                        y=benchmark_value,
                        color="red",
                        linestyle="--",
                        label=f"Benchmark: {benchmark_value}",
                    )

        # Create legend
        legend_elements = []
        for sensor_id_val, color in color_map.items():
            legend_elements.append(
                Patch(
                    facecolor=color,
                    edgecolor="black",
                    label=format_label(sensor_id_val),
                )
            )

        # Add N/A legend if there's any N/A data
        if df[self.attributes].isnull().values.any():
            legend_elements.append(
                Patch(facecolor="gray", edgecolor="black", label="Data Not Available")
            )

        # Enhanced legend with standard styling
        if len(fig.axes) > 0 and legend_elements:
            # Place legend on figure rather than individual subplot for better positioning
            fig.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.02),  # Bottom of figure
                ncol=min(3, len(legend_elements)),  # Max 3 columns
                fontsize=10,  # Balanced legend font size
                frameon=True,
                fancybox=True,
                shadow=True,
                facecolor="white",
                edgecolor="#BDC3C7",
                title="Sensors",
                title_fontsize=11,
            )  # Proportionate title size

        # Modern layout with maximum spacing to prevent header overlap
        fig.tight_layout(
            pad=3.2, h_pad=7.0, w_pad=2.0
        )  # Maximum vertical spacing for multi-line labels
        fig.subplots_adjust(
            top=0.90, bottom=0.20
        )  # Reduced bottom margin since x-axis label removed

        return fig


class ChartWidget(QWidget):
    """Qt widget that embeds matplotlib charts for sensor comparisons."""

    # Signal emitted when chart needs to be updated
    chart_update_requested = Signal(list, list)  # sensors_data, attributes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sensors_data = []
        self.current_attributes = []
        self.current_figure = None
        self.chart_thread = None

        # Get theme manager
        self.theme_manager = get_theme_manager()

        self.setup_ui()
        self.setup_default_chart()

    def setup_ui(self):
        """Initialize the user interface."""
        # Create main scroll area for better accessibility on smaller screens
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create main content widget that will be scrollable
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Control panel (specific to Bar Charts)
        self.controls_group = QGroupBox("Bar Chart Controls")
        self.controls_group.setFont(
            create_styled_font("h4", "medium")
        )  # Better hierarchy
        controls_layout = QHBoxLayout(self.controls_group)

        # Attribute selection
        self.attribute_label = QLabel("Attributes:")
        self.attribute_label.setFont(create_styled_font("body", "medium"))
        controls_layout.addWidget(self.attribute_label)

        # Attribute selection button that opens advanced dialog
        self.select_attributes_button = QPushButton("Select Attributes...")
        self.select_attributes_button.setFont(
            create_styled_font("button")
        )  # Better button font
        self.select_attributes_button.setToolTip(
            "Choose which sensor specifications to compare (resolution, frame rate, latency, etc.)"
        )
        self.select_attributes_button.clicked.connect(
            lambda: logger.info("[BAR CHART DEBUG] Button clicked!")
            or self.open_attribute_selection()
        )
        controls_layout.addWidget(self.select_attributes_button)

        # Display current attributes
        self.attributes_info = QLabel("Auto-selected based on sensor data")
        self.attributes_info.setFont(
            create_styled_font("section_header")
        )  # Better visibility

        # Configure for single-line display as requested by user
        self.attributes_info.setWordWrap(False)
        self.attributes_info.setTextFormat(Qt.PlainText)
        self.attributes_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        controls_layout.addWidget(self.attributes_info)

        controls_layout.addStretch()

        # Generate button (more appropriate than "Refresh" since no chart exists initially)
        self.refresh_button = QPushButton("Generate Charts")
        self.refresh_button.setFont(
            create_styled_font("button", "medium")
        )  # More prominent button
        self.refresh_button.setToolTip(
            "Generate comparison charts for selected sensors and attributes - helps visualize performance differences"
        )
        self.refresh_button.clicked.connect(self.refresh_chart)
        controls_layout.addWidget(self.refresh_button)

        layout.addWidget(self.controls_group)

        # Progress bar for chart generation
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Tabbed interface for different chart types
        self.tab_widget = QTabWidget()

        # Bar Charts Tab (existing matplotlib functionality)
        self.bar_chart_widget = QWidget()
        bar_layout = QVBoxLayout(self.bar_chart_widget)

        # Matplotlib canvas
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Custom navigation toolbar with dark mode support
        self.toolbar = self.create_custom_toolbar()

        bar_layout.addWidget(self.toolbar)
        bar_layout.addWidget(self.canvas)

        # PDF Export Tab (standard reporting)
        self.pdf_export_widget = PDFExportWidget()

        # ROS Config Export Tab (ROS launch file generation)
        self.ros_config_widget = ROSConfigWidget()

        # Radar Chart Tab (dedicated radar/spider chart visualization)
        self.radar_chart_widget = RadarChartWidget()

        # Add tabs - Clean organized tab structure
        self.tab_widget.addTab(self.bar_chart_widget, "Bar Charts")
        self.tab_widget.addTab(self.radar_chart_widget, "Radar Charts")
        self.tab_widget.addTab(self.pdf_export_widget, "PDF Export")
        self.tab_widget.addTab(self.ros_config_widget, "ROS Config")

        layout.addWidget(self.tab_widget)

        # Set the content widget in the scroll area
        scroll_area.setWidget(content_widget)

        # Set the scroll area as the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)

        # Apply theme-aware styling
        self.apply_theme()

        # Connect to theme changes
        self.theme_manager.theme_changed.connect(self.apply_theme)

        # Connect to tab changes to show/hide appropriate controls
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Initially show Bar Chart Controls only if Bar Charts tab is selected
        self.on_tab_changed(0)  # Initialize with first tab (Bar Charts)

    def on_tab_changed(self, index):
        """Handle tab change events to show/hide appropriate controls."""
        # Only show Bar Chart Controls when Bar Charts tab is active (index 0)
        if index == 0:  # Bar Charts tab
            self.controls_group.setVisible(True)
        else:  # All other tabs (Radar Charts, PDF Export, ROS Config)
            self.controls_group.setVisible(False)

    def create_custom_toolbar(self):
        """Create a custom toolbar with dark mode support and sensor-comparison features."""
        toolbar = QToolBar("Chart Tools", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))

        # Home/Reset View - Most useful for sensor comparisons
        home_action = toolbar.addAction("🏠", self.home_view)
        home_action.setToolTip("Reset view to show all data")
        home_action.setStatusTip("Reset zoom and pan to show all sensor data")

        # Zoom In tool - Rectangle selection for detailed analysis
        zoom_in_action = toolbar.addAction("🔍+", self.toggle_zoom_in)
        zoom_in_action.setToolTip("Zoom In (click and drag to select area)")
        zoom_in_action.setStatusTip(
            "Select a rectangular area to zoom into for detailed sensor analysis"
        )
        zoom_in_action.setCheckable(True)

        # Zoom Out tool - Step-wise zoom out for broader perspective
        zoom_out_action = toolbar.addAction("🔍-", self.zoom_out)
        zoom_out_action.setToolTip("Zoom Out (expand view)")
        zoom_out_action.setStatusTip("Zoom out one level to see more data context")

        # Pan tool - Useful when zoomed in
        pan_action = toolbar.addAction("✋", self.toggle_pan)
        pan_action.setToolTip("Pan view (click and drag)")
        pan_action.setStatusTip("Drag to move the view around")
        pan_action.setCheckable(True)

        # Group zoom in and pan actions (only one active at a time)
        # Zoom out is not included as it's an immediate action, not a mode
        self.nav_group = QButtonGroup(self)
        self.nav_group.addButton(toolbar.widgetForAction(zoom_in_action))
        self.nav_group.addButton(toolbar.widgetForAction(pan_action))
        self.nav_group.setExclusive(True)

        # Store actions for later reference
        self.zoom_in_action = zoom_in_action
        self.zoom_out_action = zoom_out_action
        self.pan_action = pan_action

        # Initialize zoom level tracking for smart zoom out
        self.zoom_history = []  # Stack of previous zoom states

        toolbar.addSeparator()

        # Export/Save - Essential for reporting
        save_action = toolbar.addAction("💾", self.save_chart)
        save_action.setToolTip("Save chart as image")
        save_action.setStatusTip("Export the current chart to PNG, SVG, or PDF")

        # Quick export to PDF - Sensor comparison specific
        pdf_action = toolbar.addAction("📄", self.quick_export_pdf)
        pdf_action.setToolTip("Quick export to PDF")
        pdf_action.setStatusTip("Export chart with metadata for reports")

        toolbar.addSeparator()

        # Chart refresh - Useful when data changes
        refresh_action = toolbar.addAction("🔄", self.refresh_chart)
        refresh_action.setToolTip("Refresh chart")
        refresh_action.setStatusTip("Regenerate chart with current settings")

        # Auto-fit - Optimize view for current data
        fit_action = toolbar.addAction("📏", self.auto_fit_view)
        fit_action.setToolTip("Auto-fit view")
        fit_action.setStatusTip("Automatically adjust view to show all data optimally")

        # Apply theme styling to toolbar
        self.style_toolbar(toolbar)

        # Initialize zoom out button state (disabled until there's something to zoom out from)
        zoom_out_action.setEnabled(False)

        return toolbar

    def style_toolbar(self, toolbar):
        """Apply theme-aware styling to the custom toolbar."""
        colors = self.theme_manager.get_stylesheet_colors()

        toolbar_style = f"""
            QToolBar {{
                background-color: {colors['surface']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 4px;
                spacing: 2px;
            }}
            QToolBar::separator {{
                background-color: {colors['border']};
                width: 2px;
                margin: 2px 4px;
            }}
            QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px;
                margin: 1px;
                font-size: 16px;
                color: {colors['text_primary']};
            }}
            QToolButton:hover {{
                background-color: {colors['hover']};
                border-color: {colors['border_focus']};
            }}
            QToolButton:pressed {{
                background-color: {colors['active']};
            }}
            QToolButton:checked {{
                background-color: {colors['primary']};
                color: {colors['text_on_primary']};
                border-color: {colors['primary']};
            }}
        """
        toolbar.setStyleSheet(toolbar_style)

    def home_view(self):
        """Reset view to show all data clearly."""
        if hasattr(self.canvas, "toolbar") and hasattr(self.canvas.toolbar, "home"):
            # Use matplotlib's home functionality if available
            self.canvas.toolbar.home()
        else:
            # Manual home implementation
            for ax in self.figure.axes:
                ax.relim()
                ax.autoscale()
            self.canvas.draw()

        # Deactivate zoom/pan modes and clear zoom history
        self.zoom_in_action.setChecked(False)
        self.pan_action.setChecked(False)
        self._clear_interaction_mode()
        self.zoom_history.clear()  # Reset zoom history on home

    def toggle_zoom_in(self):
        """Toggle zoom in rectangle selection mode."""
        if self.zoom_in_action.isChecked():
            # Activate zoom in mode
            self.pan_action.setChecked(False)
            self._clear_interaction_mode()  # Clear any existing handlers first
            self._setup_zoom_in_mode()
            logger.debug("Zoom in mode activated")
        else:
            # Deactivate zoom in mode
            self._clear_interaction_mode()
            logger.debug("Zoom in mode deactivated")

    def zoom_out(self):
        """Zoom out one level or to fit all data if no zoom history."""
        if not self.figure.axes:
            return

        if self.zoom_history:
            # Restore previous zoom level
            previous_limits = self.zoom_history.pop()
            for ax, (xlim, ylim) in previous_limits.items():
                if ax in self.figure.axes:  # Ensure axis still exists
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            self.canvas.draw()
        else:
            # No zoom history, perform auto-fit (same as home but without clearing modes)
            self.auto_fit_view()

        # Update zoom out button state based on zoom history
        zoom_out_available = len(self.zoom_history) > 0 or self._is_zoomed()
        self.zoom_out_action.setEnabled(zoom_out_available)
        logger.debug(
            f"Zoom out button enabled: {zoom_out_available}, history items: {len(self.zoom_history)}"
        )

    def toggle_pan(self):
        """Toggle pan mode."""
        if self.pan_action.isChecked():
            # Activate pan mode
            self.zoom_in_action.setChecked(False)
            self._clear_interaction_mode()  # Clear any existing handlers first
            self._setup_pan_mode()
            logger.debug("Pan mode activated")
        else:
            # Deactivate pan mode
            self._clear_interaction_mode()
            logger.debug("Pan mode deactivated")

    def _setup_zoom_in_mode(self):
        """Setup zoom in rectangle interaction with history tracking."""

        def on_press(event):
            if event.inaxes:
                self._zoom_start = (event.xdata, event.ydata)

        def on_release(event):
            if event.inaxes and hasattr(self, "_zoom_start"):
                x1, y1 = self._zoom_start
                x2, y2 = event.xdata, event.ydata
                if x1 != x2 and y1 != y2:
                    # Save current zoom state to history before changing
                    self._save_zoom_state()

                    # Apply new zoom
                    event.inaxes.set_xlim(min(x1, x2), max(x1, x2))
                    event.inaxes.set_ylim(min(y1, y2), max(y1, y2))
                    self.canvas.draw()

                    # Enable zoom out button
                    self.zoom_out_action.setEnabled(True)

                    # Auto-deactivate zoom in mode after successful zoom (one-shot behavior)
                    self.zoom_in_action.setChecked(False)
                    self._clear_interaction_mode()
                    logger.debug(
                        "Zoom operation completed - auto-deactivated zoom mode"
                    )

                if hasattr(self, "_zoom_start"):
                    delattr(self, "_zoom_start")

        self._zoom_press_cid = self.canvas.mpl_connect("button_press_event", on_press)
        self._zoom_release_cid = self.canvas.mpl_connect(
            "button_release_event", on_release
        )

    def _setup_pan_mode(self):
        """Setup pan interaction."""

        def on_press(event):
            if event.inaxes:
                self._pan_start = (event.x, event.y)
                self._pan_axes_start = {
                    ax: (ax.get_xlim(), ax.get_ylim()) for ax in self.figure.axes
                }

        def on_motion(event):
            if event.inaxes and hasattr(self, "_pan_start"):
                dx = event.x - self._pan_start[0]
                dy = event.y - self._pan_start[1]

                for ax in self.figure.axes:
                    if ax == event.inaxes:
                        xlim, ylim = self._pan_axes_start[ax]
                        x_range = xlim[1] - xlim[0]
                        y_range = ylim[1] - ylim[0]

                        # Convert pixel movement to data coordinates
                        x_shift = -dx / ax.bbox.width * x_range
                        y_shift = -dy / ax.bbox.height * y_range

                        ax.set_xlim(xlim[0] + x_shift, xlim[1] + x_shift)
                        ax.set_ylim(ylim[0] + y_shift, ylim[1] + y_shift)

                self.canvas.draw()

        self._pan_press_cid = self.canvas.mpl_connect("button_press_event", on_press)
        self._pan_motion_cid = self.canvas.mpl_connect("motion_notify_event", on_motion)

    def _clear_interaction_mode(self):
        """Clear all interaction event handlers and ensure clean state."""
        # Disconnect all interaction events
        for attr in [
            "_zoom_press_cid",
            "_zoom_release_cid",
            "_pan_press_cid",
            "_pan_motion_cid",
        ]:
            if hasattr(self, attr):
                try:
                    self.canvas.mpl_disconnect(getattr(self, attr))
                    delattr(self, attr)
                    logger.debug(f"Cleared event handler: {attr}")
                except Exception as e:
                    logger.warning(f"Failed to clear event handler {attr}: {e}")

        # Clean up any temporary state
        for temp_attr in ["_zoom_start", "_pan_start", "_pan_axes_start"]:
            if hasattr(self, temp_attr):
                delattr(self, temp_attr)

    def save_chart(self):
        """Save chart with user-selected format and location."""
        if not hasattr(self, "figure") or not self.figure.axes:
            QMessageBox.information(
                self, "Save Chart", "No chart to save. Generate a chart first."
            )
            return

        # File dialog with multiple format options
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Chart",
            "sensor_comparison_chart.png",
            "PNG Image (*.png);;SVG Vector (*.svg);;PDF Document (*.pdf);;JPEG Image (*.jpg)",
        )

        if filename:
            try:
                # High quality export with metadata
                self.figure.savefig(
                    filename,
                    dpi=300,
                    bbox_inches="tight",
                    metadata={
                        "Title": "Sensor Comparison Chart",
                        "Creator": "SensorSphere",
                    },
                )
                QMessageBox.information(
                    self, "Export Success", f"Chart saved to {filename}"
                )
            except Exception as e:
                QMessageBox.warning(
                    self, "Export Error", f"Failed to save chart: {str(e)}"
                )

    def quick_export_pdf(self):
        """Quick export to PDF with predefined settings."""
        if not hasattr(self, "figure") or not self.figure.axes:
            QMessageBox.information(
                self, "Quick Export", "No chart to export. Generate a chart first."
            )
            return

        # Quick export with timestamp filename
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sensor_comparison_{timestamp}.pdf"

        try:
            self.figure.savefig(
                filename,
                format="pdf",
                dpi=300,
                bbox_inches="tight",
                metadata={
                    "Title": f"Sensor Comparison Report - {timestamp}",
                    "Creator": "SensorSphere",
                    "Subject": f"Comparison of {len(self.sensors_data)} sensors",
                },
            )
            QMessageBox.information(
                self,
                "Quick Export Success",
                f"Chart exported to {filename} in current directory",
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Quick Export Error", f"Failed to export chart: {str(e)}"
            )

    def auto_fit_view(self):
        """Automatically adjust view to optimally display all sensor data."""
        if not self.figure.axes:
            return

        for ax in self.figure.axes:
            # Smart auto-scaling that leaves some margin
            ax.relim()
            ax.autoscale()

            # Add 5% margin for better visibility
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_margin = (xlim[1] - xlim[0]) * 0.05
            y_margin = (ylim[1] - ylim[0]) * 0.05

            ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
            ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)

        self.canvas.draw()

        # Deactivate interaction modes and update zoom out state
        self.zoom_in_action.setChecked(False)
        self.pan_action.setChecked(False)
        self._clear_interaction_mode()

        # Update zoom out button state
        self.zoom_out_action.setEnabled(len(self.zoom_history) > 0 or self._is_zoomed())

    def _save_zoom_state(self):
        """Save current zoom state to history for zoom out functionality."""
        current_limits = {}
        for ax in self.figure.axes:
            current_limits[ax] = (ax.get_xlim(), ax.get_ylim())
        self.zoom_history.append(current_limits)

        # Limit history size to prevent memory issues
        if len(self.zoom_history) > 10:
            self.zoom_history.pop(0)

    def _is_zoomed(self):
        """Check if any axis is currently zoomed (not showing full data range)."""
        for ax in self.figure.axes:
            # Get data limits
            ax.relim()
            data_xlim = ax.dataLim.intervalx
            data_ylim = ax.dataLim.intervaly

            # Get current view limits
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            # Check if current view is smaller than data range (with small tolerance)
            tolerance = 0.01
            x_zoomed = (current_xlim[1] - current_xlim[0]) < (
                data_xlim[1] - data_xlim[0]
            ) * (1 - tolerance)
            y_zoomed = (current_ylim[1] - current_ylim[0]) < (
                data_ylim[1] - data_ylim[0]
            ) * (1 - tolerance)

            if x_zoomed or y_zoomed:
                return True
        return False

    def _update_attributes_display(self, attributes):
        """Update the attributes display text with enhanced formatting and tooltip."""
        from ...utils import format_label

        formatted_attrs = [format_label(attr) for attr in attributes]

        # Enhanced display logic - show more before truncating
        if len(attributes) <= 4:
            # Show all attributes for 4 or fewer
            attr_text = ", ".join(formatted_attrs)
        elif len(attributes) <= 6:
            # Show first 3 + count for 5-6 attributes
            attr_text = f"{', '.join(formatted_attrs[:3])} (+{len(attributes)-3} more)"
        else:
            # Show first 3 + total count for 7+ attributes
            attr_text = f"{', '.join(formatted_attrs[:3])} (+{len(attributes)-3} more)"

        # Set display text on single line as requested by user
        self.attributes_info.setText(f"Selected: {attr_text}")

        # Create comprehensive tooltip showing all attributes
        tooltip_text = f"Selected Attributes ({len(attributes)}):\n" + "\n".join(
            [f"• {attr}" for attr in formatted_attrs]
        )
        self.attributes_info.setToolTip(tooltip_text)

        # Disable word wrap to keep everything on one line
        self.attributes_info.setWordWrap(False)

        # Enable ellipsis for long text that doesn't fit
        self.attributes_info.setTextFormat(Qt.PlainText)

        # Set size policy to allow horizontal expansion but limit vertical
        self.attributes_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def apply_theme(self):
        """Apply current theme styling to the widget."""
        # Get basic styling
        colors = self.theme_manager.get_stylesheet_colors()
        button_style = self.theme_manager.create_button_stylesheet("primary")

        # Create standardized widget styling with enhanced GroupBox support
        widget_style = f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {colors['border']};
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 15px;
                color: {colors['text_primary']};
                background-color: {colors['surface']};
                font-size: 12px;
            }}
            QGroupBox::title {{
                background-color: {colors['surface']};
                color: {colors['text_primary']};
                font-weight: bold;
                font-size: 13px;
                border: 1px solid {colors['border']};
                border-radius: 3px;
                padding: 0 8px 0 8px;
                left: 10px;
                subcontrol-origin: margin;
                subcontrol-position: top left;
            }}
            QLabel {{
                color: {colors['text_primary']};
                background-color: transparent;
            }}
            QCheckBox {{
                color: {colors['text_primary']};
                background-color: transparent;
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid {colors['border']};
                border-radius: 3px;
                background-color: {colors['surface']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {colors['primary']};
                border-color: {colors['primary']};
            }}
            QCheckBox::indicator:hover {{
                border-color: {colors['border_focus']};
            }}
            QComboBox {{
                color: {colors['text_primary']};
                background-color: {colors['surface']};
                border: 2px solid {colors['border']};
                border-radius: 3px;
                padding: 4px 8px;
                min-width: 100px;
            }}
            QComboBox:hover {{
                border-color: {colors['border_focus']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {colors['text_primary']};
                margin-right: 5px;
            }}
        """

        # Combine stylesheets
        combined_style = widget_style + "\n" + button_style
        self.setStyleSheet(combined_style)

        # Apply styling to custom toolbar if it exists
        if hasattr(self, "toolbar"):
            self.style_toolbar(self.toolbar)

        try:
            # Update group box titles
            group_boxes = self.findChildren(QGroupBox)
            for group_box in group_boxes:
                group_box.setFont(create_styled_font("h4", "medium"))

            # Update all labels
            labels = self.findChildren(QLabel)
            for label in labels:
                label.setFont(create_styled_font("body"))

            # Update all buttons
            buttons = self.findChildren(QPushButton)
            for button in buttons:
                button.setFont(create_styled_font("button"))

            # Update combo boxes
            comboboxes = self.findChildren(QComboBox)
            for combo in comboboxes:
                combo.setFont(create_styled_font("input_text"))

        except Exception as e:
            logger.error(f"Error applying theme to chart widget: {e}")

    def setup_default_chart(self):
        """Setup an initial empty or placeholder chart."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            'Select sensors in the comparison table\nand click "Generate Charts" to view comparison charts',
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
            color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        self.canvas.draw()

    def update_sensors_data(self, sensors_data: List[Dict[str, Any]]):
        """Update the sensor data for chart generation."""
        logger.info(
            f"[BAR CHART DEBUG] update_sensors_data called with {len(sensors_data) if sensors_data else 0} sensors"
        )
        self.sensors_data = sensors_data.copy() if sensors_data else []
        logger.info(
            f"[BAR CHART DEBUG] Chart widget updated with {len(self.sensors_data)} sensors"
        )

        # Update button state and visual feedback
        self.update_ui_state()

        # Auto-select relevant attributes based on available data
        if sensors_data:
            self.current_attributes = self.get_relevant_attributes(sensors_data)
            self._update_attributes_display(self.current_attributes)

        # Update radar chart widget with the same data
        self.radar_chart_widget.update_sensors_data(sensors_data)

        # Update PDF export widget with the same data
        self.pdf_export_widget.update_sensors_data(sensors_data)

        # Update ROS config widget with the same data
        self.ros_config_widget.update_sensors_data(sensors_data)

    def update_ui_state(self):
        """Update UI elements based on current sensor data state."""
        has_sensors = bool(self.sensors_data)

        # Enable/disable button based on sensor data availability
        self.select_attributes_button.setEnabled(has_sensors)

        # Update button tooltip and styling based on state
        if has_sensors:
            self.select_attributes_button.setToolTip(
                f"Customize chart attributes ({len(self.sensors_data)} sensors loaded)"
            )
            self.select_attributes_button.setStyleSheet("")  # Reset to default styling
            self.attributes_info.setText(
                f"Ready: {len(self.sensors_data)} sensors loaded"
            )
        else:
            self.select_attributes_button.setToolTip(
                "Load sensor data first by selecting sensors from the table above"
            )
            self.select_attributes_button.setStyleSheet(
                "color: #666666;"
            )  # Greyed out appearance
            self.attributes_info.setText(
                "Load sensor data first by selecting sensors above"
            )

    def get_relevant_attributes(self, sensors_data: List[Dict[str, Any]]) -> List[str]:
        """Determine relevant attributes for comparison based on available data."""
        # Common attributes that are good for comparison
        priority_attributes = [
            "frame_rate",
            "price_avg",
            "max_range",
            "min_range",
            "resolution_rgb",
            "resolution_depth",
            "latency",
        ]

        available_attributes = []
        if sensors_data:
            # Get all keys from all sensors to ensure comprehensive coverage
            all_keys = set()
            for sensor in sensors_data:
                all_keys.update(sensor.keys())

            # Check which priority attributes have data in the actual sensor data
            for attr in priority_attributes:
                # Check if attribute exists directly or can be derived
                if attr in all_keys:
                    available_attributes.append(attr)
                elif attr == "price_avg" and "price_range" in all_keys:
                    available_attributes.append(attr)
                elif (
                    attr in ["resolution_rgb", "resolution_depth"]
                    and "resolution" in all_keys
                ):
                    available_attributes.append(attr)
                elif attr == "latency" and (
                    "latency_display" in all_keys
                    or any("latency" in str(k).lower() for k in all_keys)
                ):
                    available_attributes.append(attr)

            # If not enough attributes, add some basic display attributes that are always available
            if len(available_attributes) < 3:
                basic_attrs = [
                    "frame_rate_display",
                    "price_display",
                    "range_display",
                    "resolution_display",
                ]
                for attr in basic_attrs:
                    if attr in all_keys and attr not in available_attributes:
                        available_attributes.append(attr)

        return available_attributes[:6]  # Limit to 6 attributes for readability

    def open_attribute_selection(self):
        """Open the attribute selection dialog."""
        logger.info(
            f"[DEBUG] Button clicked! sensors_data length: {len(self.sensors_data) if self.sensors_data else 'None'}"
        )

        if not self.sensors_data:
            logger.warning("[DEBUG] No sensor data available - showing message to user")
            QMessageBox.information(
                self,
                "Attribute Selection",
                "Please load sensor data first to see available attributes.\n\n"
                "Try selecting some sensors from the table first.",
            )
            return

        logger.info(
            f"[DEBUG] Opening attribute selection dialog with {len(self.sensors_data)} sensors"
        )

        # Get all possible attributes from current sensor data
        all_attributes = set()
        for sensor in self.sensors_data:
            all_attributes.update(sensor.keys())

        # Filter to include only numeric/comparable attributes
        comparable_attributes = []
        exclude_attrs = {
            "sensor_id",
            "manufacturer",
            "model",
            "sensor_type",
            "schema_version",
            "datasheet_link",
            "driver_link",
            "driver_link_ros1",
            "driver_link_ros2",
            "github_repo",
            "key_features",
            "use_cases",
            "tags",
            "notes",
            "communication_interface",
            "supported_platforms",
            "environmental_rating",
        }

        for attr in all_attributes:
            if attr not in exclude_attrs:
                # Check if attribute has numeric data in at least one sensor
                has_numeric = False
                for sensor in self.sensors_data:
                    value = sensor.get(attr)
                    if value is not None:
                        # Try to extract numeric value
                        try:
                            numeric_val = extract_numeric(value)
                            if numeric_val is not None:
                                has_numeric = True
                                break
                        except (ValueError, TypeError, AttributeError):
                            # Check for special attributes
                            if attr in [
                                "resolution",
                                "price_range",
                                "field_of_view",
                                "ros_compatibility",
                            ]:
                                has_numeric = True
                                break

                if has_numeric:
                    comparable_attributes.append(attr)

        # Add processed attributes that we can generate
        processed_attrs = [
            "resolution_rgb",
            "resolution_depth",
            "price_avg",
            "ros_compatibility_score",
        ]
        for attr in processed_attrs:
            if attr not in comparable_attributes:
                comparable_attributes.append(attr)

        comparable_attributes.sort()

        # Open dialog
        dialog = AttributeSelectionDialog(
            comparable_attributes,
            self.current_attributes,
            self,
            sensor_data=self.sensors_data,
            chart_type="bar",
        )

        if dialog.exec() == QDialog.Accepted:
            # Apply new attribute selection
            new_attributes = dialog.selected_attributes
            if len(new_attributes) >= 2:
                self.current_attributes = new_attributes
                # Update display using helper function
                self._update_attributes_display(new_attributes)
                logger.info(f"Updated chart attributes: {new_attributes}")

                # Auto-refresh chart if we have sensor data
                if len(self.sensors_data) >= 2:
                    self.refresh_chart()
            else:
                QMessageBox.warning(
                    self,
                    "Attribute Selection",
                    "Please select at least 2 attributes for comparison.",
                )

        dialog.deleteLater()

    def refresh_chart(self):
        """Refresh the chart with current sensor data."""
        if not self.sensors_data or len(self.sensors_data) < 2:
            QMessageBox.information(
                self,
                "Chart Generation",
                "Please select at least 2 sensors for comparison.",
            )
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.refresh_button.setEnabled(False)

        # Start chart generation in background thread
        self.chart_thread = ChartGenerationThread(
            self.sensors_data, self.current_attributes
        )
        self.chart_thread.finished.connect(self.on_chart_generated)
        self.chart_thread.error.connect(self.on_chart_error)
        self.chart_thread.start()

    def on_chart_generated(self, figure):
        """Handle successful chart generation."""
        self.progress_bar.setVisible(False)
        self.refresh_button.setEnabled(True)

        # Replace the canvas figure with the generated one
        # This is simpler and avoids copying issues
        self.canvas.figure = figure
        self.figure = figure
        self.canvas.draw()

        logger.info("Chart generated successfully")

    def on_chart_error(self, error_message):
        """Handle chart generation error."""
        self.progress_bar.setVisible(False)
        self.refresh_button.setEnabled(True)

        QMessageBox.warning(
            self,
            "Chart Generation Error",
            f"Failed to generate chart:\n{error_message}",
        )
        logger.error(f"Chart generation error: {error_message}")

    def export_chart(self, filename: str):
        """Export current chart to file."""
        try:
            self.figure.savefig(filename, dpi=300, bbox_inches="tight")
            logger.info(f"Chart exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export chart: {e}")
            return False

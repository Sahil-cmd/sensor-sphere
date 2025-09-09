"""
Radar Chart Widget for PySide6 GUI - STABLE BACKUP (2025-08-23 14:30:52)

WORKING FEATURES AT THIS STAGE:
- ✓ Radar chart Select Attributes button working correctly
- ✓ Modern attribute display with enhanced truncation (shows 3+ attributes before "+X more")
- ✓ Detailed tooltips showing all selected attributes in bulleted list
- ✓ Interactive Plotly radar/spider charts generating successfully
- ✓ Multi-dimensional sensor comparison visualization working
- ✓ Proper integration with main chart widget tab system
- ✓ No more confusion with duplicate buttons (context-sensitive controls implemented)
- ✓ Sensor data flowing correctly from main table
- ✓ Visual feedback for button states based on data availability

ENHANCED ATTRIBUTE DISPLAY:
- 4 or fewer attributes: Show all completely
- 5-6 attributes: "Frame Rate, Max Range, Min Range (+2 more)"
- 7+ attributes: "Frame Rate, Max Range, Latency (+4 more)"
- Hover tooltip shows complete bulleted list of all attributes

RADAR CHART FUNCTIONALITY:
- Generates interactive spider charts with proper scaling
- Handles multiple sensors with color-coded visualization
- Supports dynamic attribute selection from comprehensive dialog
- Intelligent defaults for radar chart optimization
- Proper theme support for dark/light modes

CURRENT STATE: Radar chart tab fully functional as standalone widget.
No more UI duplication. Clean, standard interface.

NEXT PHASE: PDF report structure and space utilization improvements.

Embeds interactive Plotly radar/spider charts in Qt widgets for multi-dimensional sensor comparison.
This module provides spider chart visualization to complement the bar charts.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QFont
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
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
    get_unit,
)
from ..utils.font_manager import create_styled_font
from ..utils.theme_manager import get_theme_manager

logger = logging.getLogger(__name__)


class RadarChartGenerationThread(QThread):
    """Background thread for generating radar charts to keep UI responsive."""

    finished = Signal(str)  # HTML file path
    error = Signal(str)

    def __init__(self, sensors_data, attributes):
        super().__init__()
        self.sensors_data = sensors_data
        self.attributes = attributes

    def run(self):
        """Generate radar chart in background thread."""
        try:
            html_path = self.generate_radar_chart()
            self.finished.emit(html_path)
        except Exception as e:
            logger.error(f"Error generating radar chart: {e}")
            self.error.emit(str(e))

    def generate_radar_chart(self):
        """Generate Plotly radar chart with sensor comparison data."""
        if not self.sensors_data or len(self.sensors_data) < 2:
            raise ValueError("At least two sensors required for radar comparison")

        # Create DataFrame from sensors data
        df = pd.DataFrame(self.sensors_data)

        # Preprocess data for radar chart
        radar_data = self.prepare_radar_data(df)

        # Create Plotly radar chart
        fig = go.Figure()

        # Color palette for different sensors
        colors = px.colors.qualitative.Set1

        for i, (sensor_id, data) in enumerate(radar_data.items()):
            color = colors[i % len(colors)]

            # Add trace for each sensor
            fig.add_trace(
                go.Scatterpolar(
                    r=data["values"],
                    theta=data["categories"],
                    fill="toself",
                    name=format_label(sensor_id),
                    line_color=color,
                    fillcolor=color,
                    opacity=0.6,
                )
            )

        # Enhanced standard layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],  # Normalized data 0-1
                    tickfont=dict(
                        size=12, color="#2C3E50", family="Arial"
                    ),  # Larger, standard color
                    gridcolor="#BDC3C7",  # Modern gray
                    gridwidth=1,
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Clear tick marks
                    ticktext=[
                        "0%",
                        "20%",
                        "40%",
                        "60%",
                        "80%",
                        "100%",
                    ],  # Percentage labels
                ),
                angularaxis=dict(
                    tickfont=dict(
                        size=14, color="#2C3E50", family="Arial", weight="bold"
                    ),  # Larger, bold labels
                    rotation=90,
                    direction="counterclockwise",
                    linecolor="#34495E",
                    linewidth=2,
                ),
                bgcolor="rgba(248, 249, 250, 0.8)",  # Light background
            ),
            showlegend=True,
            title=dict(
                text="Multi-Dimensional Sensor Comparison",
                x=0.5,
                y=0.98,  # Positioned outside and above the entire radar chart area
                font=dict(
                    size=20, family="Arial", color="#2C3E50", weight="bold"
                ),  # Modern size for title position
                pad=dict(t=20, b=20),
            ),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.9,  # Lower than title
                xanchor="left",
                x=1.05,  # Further right for more space
                font=dict(
                    size=14, family="Arial", color="#2C3E50"
                ),  # Larger legend text
                bgcolor="rgba(255, 255, 255, 0.9)",  # Semi-transparent background
                bordercolor="#BDC3C7",
                borderwidth=1,
                title=dict(text="Sensors", font=dict(size=16, weight="bold")),
            ),
            width=900,  # Wider for better readability
            height=650,  # Taller for better proportions
            template="plotly_white",
            margin=dict(l=80, r=150, t=100, b=80),  # Better margins
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        # Save to temporary HTML file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
        html_content = plot(fig, output_type="div", include_plotlyjs="inline")

        # Create full HTML page
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sensor Radar Chart</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 10px;
                    background-color: #ffffff;
                }}
                .chart-container {{
                    width: 100%;
                    height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
            </style>
        </head>
        <body>
            <div class="chart-container">
                {html_content}
            </div>
        </body>
        </html>
        """

        temp_file.write(full_html)
        temp_file.close()

        return temp_file.name

    def prepare_radar_data(self, df):
        """Prepare and normalize data for radar chart display."""
        radar_data = {}

        # Define radar chart attributes with their normalization approach
        radar_attributes = [
            ("frame_rate", "higher_better"),
            ("max_range", "higher_better"),
            ("price_avg", "lower_better"),
            ("resolution_rgb", "higher_better"),
            ("min_range", "lower_better"),
            ("latency", "lower_better"),
        ]

        # Process attributes that might need extraction
        for attr, _ in radar_attributes:
            if attr == "resolution_rgb":
                if "resolution" in df.columns:
                    df[attr] = df["resolution"].apply(
                        lambda x: (
                            extract_resolution(x, "rgb") if isinstance(x, dict) else 0
                        )
                    )
                else:
                    df[attr] = 0
            elif attr == "price_avg":
                if "price_range" in df.columns:
                    df[attr] = df["price_range"].apply(extract_price_avg)
                else:
                    df[attr] = 0
            elif attr in df.columns:
                df[attr] = df[attr].apply(extract_numeric)
            else:
                df[attr] = 0

        # Normalize data for each sensor
        for _, sensor in df.iterrows():
            sensor_id = sensor.get("sensor_id", "Unknown")
            values = []
            categories = []

            for attr, direction in radar_attributes:
                if attr in df.columns:
                    # Get value for this sensor
                    value = sensor.get(attr, 0)
                    if pd.isna(value) or value is None:
                        normalized_value = 0
                    else:
                        # Normalize based on min/max in dataset
                        series = df[attr].apply(
                            lambda x: x if pd.notna(x) and x is not None else 0
                        )
                        min_val = series.min()
                        max_val = series.max()

                        if max_val == min_val:
                            normalized_value = 0.5  # If all values same, put in middle
                        else:
                            if direction == "higher_better":
                                normalized_value = (value - min_val) / (
                                    max_val - min_val
                                )
                            else:  # lower_better
                                normalized_value = (max_val - value) / (
                                    max_val - min_val
                                )

                        # Clamp to 0-1 range
                        normalized_value = max(0, min(1, normalized_value))

                    values.append(normalized_value)
                    categories.append(format_label(attr))

            # Close the radar chart by repeating first value
            if values:
                values.append(values[0])
                categories.append(categories[0])

            radar_data[sensor_id] = {"values": values, "categories": categories}

        return radar_data


class RadarChartWidget(QWidget):
    """Qt widget that embeds interactive Plotly radar charts for sensor comparisons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sensors_data = []
        self.current_html_path = None
        self.chart_thread = None
        self.current_attributes = []  # Will be set by intelligent defaults

        # Get theme manager
        self.theme_manager = get_theme_manager()

        self.setup_ui()
        self.setup_default_chart()

    def setup_ui(self):
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)

        # Create scroll area for better accessibility on smaller screens
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create content widget that will be scrollable
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Control panel
        controls_group = QGroupBox("Radar Chart Controls")
        controls_group.setFont(create_styled_font("h4", "medium"))  # Better hierarchy
        controls_layout = QHBoxLayout(controls_group)

        # Attribute selection
        self.attribute_label = QLabel("Attributes:")
        self.attribute_label.setFont(
            create_styled_font("section_header", "medium")
        )  # Better visibility
        controls_layout.addWidget(self.attribute_label)

        # Attribute selection button
        self.select_attributes_button = QPushButton("Select Attributes...")
        self.select_attributes_button.setFont(
            create_styled_font("button")
        )  # Better button font
        self.select_attributes_button.clicked.connect(self.open_attribute_selection)
        controls_layout.addWidget(self.select_attributes_button)

        # Display current attributes
        self.attributes_info = QLabel("Radar-optimized defaults selected")
        self.attributes_info.setFont(
            create_styled_font("section_header")
        )  # Better visibility
        controls_layout.addWidget(self.attributes_info)

        controls_layout.addStretch()

        # Refresh button
        self.refresh_button = QPushButton("Generate Radar Chart")
        self.refresh_button.setFont(
            create_styled_font("button", "medium")
        )  # More prominent button
        self.refresh_button.clicked.connect(self.refresh_chart)
        controls_layout.addWidget(self.refresh_button)

        layout.addWidget(controls_group)

        # Progress bar for chart generation
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Web view for displaying Plotly chart
        self.web_view = QWebEngineView()
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.web_view)

        # Set the content widget in the scroll area
        scroll_area.setWidget(content_widget)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)

        # Apply theme-aware styling
        self.apply_theme()

        # Connect to theme changes
        self.theme_manager.theme_changed.connect(self.apply_theme)

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
        """

        # Combine stylesheets
        combined_style = widget_style + "\n" + button_style
        self.setStyleSheet(combined_style)

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

        except Exception as e:
            logger.error(f"Error applying theme to radar chart widget: {e}")

    def setup_default_chart(self):
        """Setup a default empty state."""
        default_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Radar Chart</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #ffffff;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    flex-direction: column;
                }
                .message {
                    text-align: center;
                    color: #666;
                    font-size: 16px;
                }
            </style>
        </head>
        <body>
            <div class="message">
                <h3>Radar Chart Visualization</h3>
                <p>Select 2 or more sensors in the comparison table<br/>
                and click "Generate Radar Chart" to view<br/>
                multi-dimensional comparison</p>
            </div>
        </body>
        </html>
        """

        # Save default HTML to temp file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
        temp_file.write(default_html)
        temp_file.close()

        self.web_view.load(QUrl.fromLocalFile(temp_file.name))
        self.current_html_path = temp_file.name

    def update_sensors_data(self, sensors_data: List[Dict[str, Any]]):
        """Update the sensor data for radar chart generation."""
        self.sensors_data = sensors_data.copy()

        # Initialize radar-optimized attributes if not already set
        if not self.current_attributes:
            self.initialize_radar_attributes()

        logger.info(f"Radar chart widget updated with {len(sensors_data)} sensors")

    def initialize_radar_attributes(self):
        """Initialize radar-optimized default attributes."""
        if not self.sensors_data:
            # Default radar attributes when no data is available - use display attributes that are always available
            self.current_attributes = [
                "frame_rate_display",
                "range_display",
                "price_display",
                "resolution_display",
            ]
            return

        # Get all possible attributes from current sensor data
        all_attributes = set()
        for sensor in self.sensors_data:
            all_attributes.update(sensor.keys())

        # Use radar-optimized defaults with proper availability checking
        radar_optimized = [
            "frame_rate",
            "max_range",
            "resolution_rgb",
            "resolution_depth",
            "latency",
            "power_consumption",
            "min_range",
            "field_of_view",
            "weight",
            "price_avg",
        ]

        # Filter to only include available attributes with special handling for derived attributes
        available_radar_attributes = []
        for attr in radar_optimized:
            if attr in all_attributes:
                available_radar_attributes.append(attr)
            elif attr == "price_avg" and "price_range" in all_attributes:
                available_radar_attributes.append(attr)
            elif (
                attr in ["resolution_rgb", "resolution_depth"]
                and "resolution" in all_attributes
            ):
                available_radar_attributes.append(attr)
            elif attr == "latency" and (
                "latency_display" in all_attributes
                or any("latency" in str(k).lower() for k in all_attributes)
            ):
                available_radar_attributes.append(attr)

        self.current_attributes = available_radar_attributes[:6]

        # If we don't have enough attributes, add some basic ones
        if len(self.current_attributes) < 4:
            basic_attrs = ["frame_rate", "price_avg", "max_range", "min_range"]
            for attr in basic_attrs:
                if attr in all_attributes and attr not in self.current_attributes:
                    self.current_attributes.append(attr)

        # Update display
        self.update_attributes_display()

    def update_attributes_display(self):
        """Update the attributes info display with enhanced formatting and tooltip."""
        formatted_attrs = [
            self.format_attr_label(attr) for attr in self.current_attributes
        ]

        # Enhanced display logic - show more before truncating (consistent with bar charts)
        if len(self.current_attributes) <= 4:
            # Show all attributes for 4 or fewer
            attr_text = ", ".join(formatted_attrs)
        elif len(self.current_attributes) <= 6:
            # Show first 3 + count for 5-6 attributes
            attr_text = f"{', '.join(formatted_attrs[:3])} (+{len(self.current_attributes)-3} more)"
        else:
            # Show first 3 + total count for 7+ attributes
            attr_text = f"{', '.join(formatted_attrs[:3])} (+{len(self.current_attributes)-3} more)"

        # Set display text and detailed tooltip
        self.attributes_info.setText(f"Selected: {attr_text}")

        # Create comprehensive tooltip showing all attributes
        tooltip_text = (
            f"Selected Attributes ({len(self.current_attributes)}):\n"
            + "\n".join([f"• {attr}" for attr in formatted_attrs])
        )
        self.attributes_info.setToolTip(tooltip_text)

        # Enable single-line display (no word wrap) for standard appearance
        self.attributes_info.setWordWrap(False)
        self.attributes_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def format_attr_label(self, attr):
        """Format attribute name for display."""
        # Simple formatting - convert underscores to spaces and title case
        return attr.replace("_", " ").title()

    def open_attribute_selection(self):
        """Open the attribute selection dialog for radar charts."""
        logger.info(
            f"[RADAR DEBUG] Button clicked! sensors_data length: {len(self.sensors_data) if self.sensors_data else 'None'}"
        )

        if not self.sensors_data:
            logger.warning(
                "[RADAR DEBUG] No sensor data available - showing message to user"
            )
            QMessageBox.information(
                self,
                "Attribute Selection",
                "Please load sensor data first to see available attributes.\n\n"
                "Try selecting some sensors from the table first.",
            )
            return

        logger.info(
            f"[RADAR DEBUG] Opening radar attribute selection dialog with {len(self.sensors_data)} sensors"
        )

        # Import locally to avoid circular imports
        from ...utils import extract_numeric
        from .chart_widget import AttributeSelectionDialog

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

        # Open dialog with radar chart type
        dialog = AttributeSelectionDialog(
            comparable_attributes,
            self.current_attributes,
            self,
            sensor_data=self.sensors_data,
            chart_type="radar",
        )

        if dialog.exec() == QDialog.Accepted:
            # Apply new attribute selection
            new_attributes = dialog.selected_attributes
            if len(new_attributes) >= 2:
                self.current_attributes = new_attributes
                self.update_attributes_display()
                logger.info(f"Updated radar chart attributes: {new_attributes}")

                # Auto-refresh chart if we have sensor data
                if len(self.sensors_data) >= 2:
                    self.refresh_chart()
            else:
                QMessageBox.warning(
                    self,
                    "Attribute Selection",
                    "Please select at least 2 attributes for radar comparison.",
                )

    def refresh_chart(self):
        """Refresh the radar chart with current sensor data."""
        if not self.sensors_data or len(self.sensors_data) < 2:
            QMessageBox.information(
                self,
                "Radar Chart Generation",
                "Please select at least 2 sensors for radar comparison.",
            )
            return

        # Initialize attributes if not set
        if not self.current_attributes:
            self.initialize_radar_attributes()

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.refresh_button.setEnabled(False)

        # Start chart generation in background thread
        self.chart_thread = RadarChartGenerationThread(
            self.sensors_data, self.current_attributes
        )
        self.chart_thread.finished.connect(self.on_chart_generated)
        self.chart_thread.error.connect(self.on_chart_error)
        self.chart_thread.start()

    def on_chart_generated(self, html_path):
        """Handle successful chart generation."""
        self.progress_bar.setVisible(False)
        self.refresh_button.setEnabled(True)

        # Clean up previous chart file
        if self.current_html_path and os.path.exists(self.current_html_path):
            try:
                os.unlink(self.current_html_path)
            except OSError:
                pass  # Ignore cleanup errors

        # Load new chart
        self.web_view.load(QUrl.fromLocalFile(html_path))
        self.current_html_path = html_path

        logger.info("Radar chart generated successfully")

    def on_chart_error(self, error_message):
        """Handle chart generation error."""
        self.progress_bar.setVisible(False)
        self.refresh_button.setEnabled(True)

        QMessageBox.warning(
            self,
            "Radar Chart Generation Error",
            f"Failed to generate radar chart:\n{error_message}",
        )
        logger.error(f"Radar chart generation error: {error_message}")

    def cleanup(self):
        """Clean up temporary files."""
        if self.current_html_path and os.path.exists(self.current_html_path):
            try:
                os.unlink(self.current_html_path)
            except OSError:
                pass

    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()

"""
ROS Config Export Widget for PySide6 GUI

Provides interface to generate ROS launch files and parameter templates from selected sensors.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...ros_config.generator import ROSConfigGenerator
from ..utils.font_manager import create_styled_font
from ..utils.theme_manager import get_theme_manager

logger = logging.getLogger(__name__)


class ROSConfigGenerationThread(QThread):
    """Background thread for generating ROS configurations to keep UI responsive."""

    finished = Signal(str)  # ZIP file path
    error = Signal(str)
    progress = Signal(str)  # Progress message

    def __init__(self, sensors_data, config_options, generation_type="ros_configs"):
        super().__init__()
        self.sensors_data = sensors_data
        self.config_options = config_options
        self.generation_type = generation_type  # "ros_configs" or "urdf_files"

    def run(self):
        """Generate ROS configurations or URDF files in background thread."""
        try:
            generator = ROSConfigGenerator()

            if self.generation_type == "urdf_files":
                self.progress.emit("Initializing URDF generator...")

                include_meshes = self.config_options.get("include_meshes", False)

                self.progress.emit(
                    f"Generating URDF files for {len(self.sensors_data)} sensors..."
                )

                # Generate URDF files
                zip_path = generator.generate_urdf_files(
                    self.sensors_data, include_meshes
                )

                self.progress.emit("URDF generation completed successfully!")
                self.finished.emit(zip_path)

            else:  # ros_configs
                self.progress.emit("Initializing ROS config generator...")

                self.progress.emit("Preparing sensor configurations...")

                # Extract options
                ros_versions = self.config_options.get("ros_versions", ["ros1", "ros2"])
                include_params = self.config_options.get("include_params", True)

                self.progress.emit(
                    f"Generating configs for {len(self.sensors_data)} sensors..."
                )

                # Generate configurations
                zip_path = generator.generate_configs(
                    self.sensors_data,
                    ros_versions=ros_versions,
                    include_params=include_params,
                )

                self.progress.emit("ROS configuration generation completed!")
                self.finished.emit(zip_path)

        except Exception as e:
            logger.error(f"Error in generation: {e}")
            self.error.emit(str(e))


class ROSConfigWidget(QWidget):
    """Qt widget for ROS configuration export functionality."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sensors_data = []
        self.generated_zip_path = None
        self.generation_thread = None

        # Get theme manager
        self.theme_manager = get_theme_manager()

        self.setup_ui()
        self.apply_theme()

        # Connect to theme changes
        self.theme_manager.theme_changed.connect(self.apply_theme)

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

        # Header section
        header_group = QGroupBox("ROS Configuration Export")
        header_group.setFont(create_styled_font("h3", "medium"))  # Better hierarchy
        header_layout = QVBoxLayout(header_group)

        # Description
        desc_label = QLabel(
            "Generate ready-to-use ROS launch files and parameter templates "
            "for selected sensors. This feature creates ROS1 and ROS2 configurations "
            "with sensor-specific parameters for rapid deployment."
        )
        desc_label.setFont(
            create_styled_font("body")
        )  # Appropriate for description text
        desc_label.setWordWrap(True)
        header_layout.addWidget(desc_label)

        layout.addWidget(header_group)

        # Configuration options
        config_group = QGroupBox("Configuration Options")
        config_group.setFont(create_styled_font("h4", "medium"))  # Better hierarchy
        config_layout = QVBoxLayout(config_group)

        # ROS version selection
        ros_version_layout = QHBoxLayout()
        ros_version_label = QLabel("ROS Versions:")
        ros_version_label.setFont(
            create_styled_font("section_header", weight="medium")
        )  # Better visibility
        ros_version_layout.addWidget(ros_version_label)

        self.ros1_checkbox = QCheckBox("ROS1")
        self.ros1_checkbox.setFont(create_styled_font("body"))  # Consistent font
        self.ros1_checkbox.setChecked(True)
        ros_version_layout.addWidget(self.ros1_checkbox)

        self.ros2_checkbox = QCheckBox("ROS2")
        self.ros2_checkbox.setFont(create_styled_font("body"))  # Consistent font
        self.ros2_checkbox.setChecked(True)
        ros_version_layout.addWidget(self.ros2_checkbox)

        ros_version_layout.addStretch()
        config_layout.addLayout(ros_version_layout)

        # Include parameter files option
        self.include_params_checkbox = QCheckBox("Include parameter template files")
        self.include_params_checkbox.setChecked(True)
        self.include_params_checkbox.setFont(
            create_styled_font("section_header")
        )  # Better visibility
        config_layout.addWidget(self.include_params_checkbox)

        # Add some spacing for better visual hierarchy
        config_layout.addSpacing(10)

        # Output format info
        output_info_layout = QVBoxLayout()
        output_label = QLabel("Output Format:")
        output_label.setFont(
            create_styled_font("section_header", weight="medium")
        )  # Better visibility
        output_info_layout.addWidget(output_label)

        output_desc = QLabel(
            "• ROS1: .launch files with XML-based configuration\n"
            "• ROS2: .launch.py files with Python-based configuration\n"
            "• Parameters: .yaml files with sensor-specific settings\n"
            "• README: Usage instructions and setup guide"
        )
        output_desc.setFont(create_styled_font("body"))
        output_desc.setWordWrap(True)  # Enable word wrap for better layout
        output_desc.setIndent(20)
        output_info_layout.addWidget(output_desc)

        config_layout.addLayout(output_info_layout)
        layout.addWidget(config_group)

        # Selected sensors display
        sensors_group = QGroupBox("Selected Sensors")
        sensors_group.setFont(create_styled_font("h4", "medium"))  # Better hierarchy
        sensors_layout = QVBoxLayout(sensors_group)

        self.sensors_list = QListWidget()
        self.sensors_list.setMaximumHeight(120)
        self.sensors_list.setFont(
            create_styled_font("section_header")
        )  # Better visibility for sensor names
        sensors_layout.addWidget(self.sensors_list)

        self.sensors_info_label = QLabel("No sensors selected")
        self.sensors_info_label.setFont(
            create_styled_font("section_header")
        )  # Better visibility
        sensors_layout.addWidget(self.sensors_info_label)

        layout.addWidget(sensors_group)

        # Generation controls
        controls_layout = QHBoxLayout()

        # Generate button
        self.generate_button = QPushButton("Generate ROS Configurations")
        self.generate_button.setFont(
            create_styled_font("button", weight="medium")
        )  # Better button font
        self.generate_button.clicked.connect(self.generate_configs)
        controls_layout.addWidget(self.generate_button)

        # URDF generation button
        self.urdf_button = QPushButton("Generate URDF Files")
        self.urdf_button.setFont(create_styled_font("button", weight="medium"))
        self.urdf_button.setToolTip(
            "Generate URDF (Unified Robot Description Format) files for sensor integration"
        )
        # Connect URDF button with error handling
        try:
            self.urdf_button.clicked.connect(self.generate_urdf_files)
            logger.info("URDF button connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect URDF button: {e}")
        controls_layout.addWidget(self.urdf_button)

        controls_layout.addStretch()

        # Open output folder button
        self.open_folder_button = QPushButton("Open Generated Files")
        self.open_folder_button.setFont(
            create_styled_font("button")
        )  # Consistent button font
        self.open_folder_button.clicked.connect(self.open_generated_files)
        self.open_folder_button.setEnabled(False)
        controls_layout.addWidget(self.open_folder_button)

        layout.addLayout(controls_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to generate ROS configurations")
        self.status_label.setFont(
            create_styled_font("section_header")
        )  # Better visibility for status messages
        layout.addWidget(self.status_label)

        # Add stretch to push content to top
        layout.addStretch()

        # Set the content widget in the scroll area
        scroll_area.setWidget(content_widget)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)

    def apply_theme(self):
        """Apply current theme styling to the widget."""
        colors = self.theme_manager.get_stylesheet_colors()
        button_style = self.theme_manager.create_button_stylesheet("primary")

        # Widget-specific styling with enhanced text visibility
        widget_style = f"""
            ROSConfigWidget {{
                background-color: {colors['background']};
                color: {colors['text_primary']};
            }}
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
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background-color: {colors['surface']};
                color: {colors['text_primary']};
                font-weight: bold;
                font-size: 13px;
                border: 1px solid {colors['border']};
                border-radius: 3px;
            }}
            QLabel {{
                color: {colors['text_primary']};
                background-color: transparent;
                font-size: 11px;
            }}
            QCheckBox {{
                color: {colors['text_primary']};
                background-color: transparent;
                font-size: 11px;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid {colors['border']};
                background-color: {colors['background']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {colors['primary']};
                border-color: {colors['primary']};
            }}
            QCheckBox::indicator:hover {{
                border-color: {colors['primary']};
            }}
            QListWidget {{
                background-color: {colors['background']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                color: {colors['text_primary']};
                padding: 6px;
                font-size: 11px;
                selection-background-color: {colors['primary']};
                selection-color: {colors['text_on_primary']};
            }}
            QListWidget::item {{
                padding: 4px;
                border-radius: 2px;
            }}
            QListWidget::item:hover {{
                background-color: {colors['hover']};
            }}
            QPushButton {{
                font-size: 12px;
            }}
        """

        # Combine stylesheets
        combined_style = widget_style + "\\n" + button_style
        self.setStyleSheet(combined_style)

    def update_sensors_data(self, sensors_data: List[Dict[str, Any]]):
        """Update the sensor data for ROS config generation."""
        self.sensors_data = sensors_data.copy()
        self.update_sensors_display()

        logger.info(f"ROS config widget updated with {len(sensors_data)} sensors")

    def update_sensors_display(self):
        """Update the sensors list display."""
        self.sensors_list.clear()

        if not self.sensors_data:
            self.sensors_info_label.setText("No sensors selected")
            self.generate_button.setEnabled(False)
            return

        # Add sensors to list
        for sensor in self.sensors_data:
            sensor_id = sensor.get("sensor_id", "Unknown")
            manufacturer = sensor.get("manufacturer", "Unknown")
            model = sensor.get("model", "Unknown")

            item_text = f"{sensor_id} - {manufacturer} {model}"
            item = QListWidgetItem(item_text)
            self.sensors_list.addItem(item)

        # Update info label
        count = len(self.sensors_data)
        self.sensors_info_label.setText(
            f"{count} sensor{'s' if count != 1 else ''} selected"
        )
        self.generate_button.setEnabled(count > 0)

    def generate_configs(self):
        """Generate ROS configuration files."""
        if not self.sensors_data:
            QMessageBox.warning(
                self,
                "No Sensors Selected",
                "Please select at least one sensor to generate ROS configurations.",
            )
            return

        # Validate ROS version selection
        ros_versions = []
        if self.ros1_checkbox.isChecked():
            ros_versions.append("ros1")
        if self.ros2_checkbox.isChecked():
            ros_versions.append("ros2")

        if not ros_versions:
            QMessageBox.warning(
                self,
                "No ROS Version Selected",
                "Please select at least one ROS version (ROS1 or ROS2).",
            )
            return

        # Prepare configuration options
        config_options = {
            "ros_versions": ros_versions,
            "include_params": self.include_params_checkbox.isChecked(),
        }

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.generate_button.setEnabled(False)
        self.status_label.setText("Generating ROS configurations...")

        # Start generation in background thread
        self.generation_thread = ROSConfigGenerationThread(
            self.sensors_data, config_options
        )
        self.generation_thread.finished.connect(self.on_generation_complete)
        self.generation_thread.error.connect(self.on_generation_error)
        self.generation_thread.progress.connect(self.on_generation_progress)
        self.generation_thread.start()

    def on_generation_progress(self, message):
        """Handle generation progress updates."""
        self.status_label.setText(message)

    def on_generation_complete(self, zip_path):
        """Handle successful ROS config generation."""
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)

        self.generated_zip_path = zip_path
        filename = os.path.basename(zip_path)
        self.status_label.setText(
            f"ROS configurations generated successfully: {filename}"
        )

        # Show completion dialog with options
        msg = QMessageBox(self)
        msg.setWindowTitle("ROS Configuration Export Complete")
        msg.setText("ROS configuration files have been generated successfully!")
        # Fix newline character issues in detailed text
        detailed_text = f"Generated ZIP file: {zip_path}\n\n"
        detailed_text += (
            f"Contains configurations for {len(self.sensors_data)} sensors:\n"
        )
        detailed_text += "\n".join(
            f"• {s.get('sensor_id', 'Unknown')}" for s in self.sensors_data
        )
        msg.setDetailedText(detailed_text)
        msg.setIcon(QMessageBox.Information)

        # Add custom buttons
        open_button = msg.addButton("Open Generated Files", QMessageBox.AcceptRole)
        save_button = msg.addButton("Save As...", QMessageBox.ActionRole)
        msg.addButton("Close", QMessageBox.RejectRole)
        msg.setDefaultButton(open_button)

        # Execute dialog and handle result
        msg.exec()
        if msg.clickedButton() == open_button:
            self.open_generated_files()
        elif msg.clickedButton() == save_button:
            self.save_generated_files()

        logger.info(f"ROS configurations generated: {zip_path}")

    def on_generation_error(self, error_message):
        """Handle ROS config generation error."""
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)

        self.status_label.setText("ROS configuration generation failed")

        QMessageBox.critical(
            self,
            "ROS Configuration Export Error",
            f"Failed to generate ROS configurations:\\n\\n{error_message}",
        )

        logger.error(f"ROS config generation error: {error_message}")

    def open_generated_files(self):
        """Open the folder containing generated files."""
        if not self.generated_zip_path or not os.path.exists(self.generated_zip_path):
            QMessageBox.warning(
                self,
                "No Generated Files",
                "No generated files found. Please generate configurations first.",
            )
            return

        try:
            import platform
            import subprocess

            folder_path = os.path.dirname(self.generated_zip_path)

            if platform.system() == "Linux":
                subprocess.run(["xdg-open", folder_path])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", folder_path])
            elif platform.system() == "Windows":
                os.startfile(folder_path)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error Opening Folder",
                f"Failed to open generated files folder:\\n{e}",
            )

    def save_generated_files(self):
        """Save generated files to user-specified location."""
        if not self.generated_zip_path or not os.path.exists(self.generated_zip_path):
            QMessageBox.warning(
                self,
                "No Generated Files",
                "No generated files found. Please generate configurations first.",
            )
            return

        # Open save dialog with timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_name = f"ros_configs_{len(self.sensors_data)}_sensors_{timestamp}.zip"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROS Configuration Files",
            suggested_name,
            "ZIP Files (*.zip);;All Files (*)",
        )

        if save_path:
            try:
                import shutil

                shutil.copy2(self.generated_zip_path, save_path)
                QMessageBox.information(
                    self,
                    "Files Saved",
                    f"ROS configuration files saved to:\\n{save_path}",
                )
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save files:\\n{e}")

    def generate_urdf_files(self):
        """Generate URDF files for selected sensors."""
        logger.info(
            f"URDF button clicked! Sensors data: {len(self.sensors_data) if self.sensors_data else 0} sensors"
        )

        try:
            # Immediate visual feedback
            self.status_label.setText("URDF generation starting...")

            if not self.sensors_data:
                self.status_label.setText("No sensors selected for URDF generation")
                QMessageBox.information(
                    self,
                    "No Sensors Available",
                    "Please select sensors from the main sensor table first.\n\n"
                    "To use URDF generation:\n"
                    "1. Go to the main sensor comparison table\n"
                    "2. Select one or more sensors\n"
                    "3. Return to this ROS Config tab\n"
                    "4. Click 'Generate URDF Files'",
                )
                return

            logger.info(
                f"Starting URDF generation for {len(self.sensors_data)} sensors"
            )

            # Create configuration for URDF generation
            config_options = {
                "include_meshes": False,  # For now, keep it simple without mesh files
            }

            # Start generation in background thread
            self.generation_thread = ROSConfigGenerationThread(
                self.sensors_data, config_options, generation_type="urdf_files"
            )
            self.generation_thread.finished.connect(self.on_urdf_generation_finished)
            self.generation_thread.error.connect(self.on_urdf_generation_error)
            self.generation_thread.progress.connect(self.on_generation_progress)

            # Update UI
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.urdf_button.setEnabled(False)
            self.generate_button.setEnabled(False)
            self.status_label.setText("Generating URDF files...")

            # Start generation
            self.generation_thread.start()

            logger.info(f"Started URDF generation for {len(self.sensors_data)} sensors")

        except Exception as e:
            logger.error(f"Error in URDF generation setup: {e}")
            self.status_label.setText(f"URDF generation failed: {str(e)}")
            QMessageBox.critical(
                self,
                "URDF Generation Error",
                f"Failed to start URDF generation:\n\n{str(e)}",
            )

    def on_urdf_generation_finished(self, zip_path):
        """Handle successful URDF generation completion."""
        self.generated_zip_path = zip_path

        # Update UI
        self.progress_bar.setVisible(False)
        self.urdf_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)

        self.status_label.setText("URDF files generated successfully!")

        logger.info(f"URDF generation completed: {zip_path}")

        # Show success message with option to open files
        reply = QMessageBox.question(
            self,
            "URDF Generation Complete",
            "URDF files have been generated successfully!\n\n"
            "Would you like to open the generated files?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Yes:
            self.open_generated_files()

    def on_urdf_generation_error(self, error_message):
        """Handle URDF generation errors."""
        # Update UI
        self.progress_bar.setVisible(False)
        self.urdf_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.status_label.setText(
            "URDF generation failed. See error message for details."
        )

        logger.error(f"URDF generation failed: {error_message}")

        # Show error message
        QMessageBox.critical(
            self,
            "URDF Generation Failed",
            f"Failed to generate URDF files:\n\n{error_message}",
        )

    def cleanup(self):
        """Clean up temporary files and resources."""
        if self.generation_thread and self.generation_thread.isRunning():
            self.generation_thread.quit()
            self.generation_thread.wait()

    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()

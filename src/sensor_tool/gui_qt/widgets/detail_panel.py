"""
Sensor Detail Panel Widget

Rich detail display for selected sensors with HTML formatting, images, and links.
This is a foundational implementation that will be enhanced with rich HTML display,
clickable links, and integrated image viewing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..utils.font_manager import create_styled_font
from ..utils.image_cache import LazyImageLabel, get_image_cache_manager

logger = logging.getLogger(__name__)


class SensorDetailWidget(QWidget):
    """
    Rich sensor detail display widget.

    Features:
    - Tabbed interface for different detail sections
    - Rich text formatting for specifications
    - Clickable links to datasheets and repositories
    - Image display capabilities (placeholder)
    - Export individual sensor data

    Signals:
        link_clicked: Emitted when a link is clicked
        export_requested: Emitted when export is requested
    """

    # Signals
    link_clicked = Signal(str)  # URL
    export_requested = Signal(str)  # sensor_id

    def __init__(self, parent=None):
        super().__init__(parent)

        # Current sensor data
        self.current_sensor: Optional[Dict[str, Any]] = None

        # Initialize image cache manager for lazy loading
        self.image_manager = get_image_cache_manager()

        # Setup UI
        self.setup_ui()
        self.connect_signals()

        # Show empty state initially
        self.show_empty_state()

    def setup_ui(self):
        """Setup the detail panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with sensor name and actions
        header_layout = QHBoxLayout()

        self.sensor_title = QLabel("No sensor selected")
        self.sensor_title.setFont(create_styled_font("h2", "bold"))
        self.sensor_title.setWordWrap(True)

        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)
        self.export_button.setMaximumWidth(80)

        header_layout.addWidget(self.sensor_title)
        header_layout.addStretch()
        header_layout.addWidget(self.export_button)

        layout.addLayout(header_layout)

        # Main content area with tabs
        self.tab_widget = QTabWidget()

        # Overview tab
        self.overview_tab = self.create_overview_tab()
        self.tab_widget.addTab(self.overview_tab, "Overview")

        # Specifications tab
        self.specs_tab = self.create_specifications_tab()
        self.tab_widget.addTab(self.specs_tab, "Specifications")

        # ROS Integration tab
        self.ros_tab = self.create_ros_tab()
        self.tab_widget.addTab(self.ros_tab, "ROS")

        # Links & Resources tab
        self.links_tab = self.create_links_tab()
        self.tab_widget.addTab(self.links_tab, "Resources")

        layout.addWidget(self.tab_widget)

    def create_overview_tab(self) -> QWidget:
        """Create the overview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # Sensor image section (prominent placement at top)
        image_group = QGroupBox("Sensor Image")
        image_layout = QVBoxLayout(image_group)

        # Create lazy loading image label with optimal size for detail viewing
        self.sensor_image_label = LazyImageLabel(cache_manager=self.image_manager)
        self.sensor_image_label.setMinimumSize(QSize(280, 200))
        self.sensor_image_label.setMaximumSize(
            QSize(350, 260)
        )  # Larger to show full sensor without cropping
        # Proper scaling to maintain aspect ratio and prevent cropping
        self.sensor_image_label.setScaledContents(True)
        # Ensure proper center alignment
        self.sensor_image_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.sensor_image_label.setStyleSheet(
            """
            QLabel {
                border: 1px solid #cccccc;
                border-radius: 8px;
                background-color: #f8f8f8;
                padding: 8px;
                margin: 4px;
            }
        """
        )

        image_layout.addWidget(self.sensor_image_label)
        # Ensure the layout centers the image widget properly
        image_layout.setAlignment(self.sensor_image_label, Qt.AlignCenter)
        image_layout.setContentsMargins(10, 10, 10, 10)

        # Basic info group
        basic_group = QGroupBox("Basic Information")
        basic_layout = QGridLayout(basic_group)

        self.manufacturer_label = QLabel("-")
        self.model_label = QLabel("-")
        self.type_label = QLabel("-")
        self.price_label = QLabel("-")

        basic_layout.addWidget(QLabel("Manufacturer:"), 0, 0)
        basic_layout.addWidget(self.manufacturer_label, 0, 1)
        basic_layout.addWidget(QLabel("Model:"), 1, 0)
        basic_layout.addWidget(self.model_label, 1, 1)
        basic_layout.addWidget(QLabel("Type:"), 2, 0)
        basic_layout.addWidget(self.type_label, 2, 1)
        basic_layout.addWidget(QLabel("Price:"), 3, 0)
        basic_layout.addWidget(self.price_label, 3, 1)

        # Key features group
        features_group = QGroupBox("Key Features")
        features_layout = QVBoxLayout(features_group)

        self.features_text = QTextEdit()
        self.features_text.setMaximumHeight(120)
        self.features_text.setReadOnly(True)
        features_layout.addWidget(self.features_text)

        # Use cases group
        use_cases_group = QGroupBox("Use Cases")
        use_cases_layout = QVBoxLayout(use_cases_group)

        self.use_cases_text = QTextEdit()
        self.use_cases_text.setMaximumHeight(100)
        self.use_cases_text.setReadOnly(True)
        use_cases_layout.addWidget(self.use_cases_text)

        content_layout.addWidget(image_group, alignment=Qt.AlignCenter)
        content_layout.addWidget(basic_group)
        content_layout.addWidget(features_group)
        content_layout.addWidget(use_cases_group)
        content_layout.addStretch()

        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

        return widget

    def create_specifications_tab(self) -> QWidget:
        """Create the specifications tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Scroll area for specifications
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # Performance specs
        perf_group = QGroupBox("Performance")
        perf_layout = QGridLayout(perf_group)

        self.resolution_label = QLabel("-")
        self.frame_rate_label = QLabel("-")
        self.latency_label = QLabel("-")
        self.range_label = QLabel("-")
        self.fov_label = QLabel("-")

        perf_layout.addWidget(QLabel("Resolution:"), 0, 0)
        perf_layout.addWidget(self.resolution_label, 0, 1)
        perf_layout.addWidget(QLabel("Frame Rate:"), 1, 0)
        perf_layout.addWidget(self.frame_rate_label, 1, 1)
        perf_layout.addWidget(QLabel("Latency:"), 2, 0)
        perf_layout.addWidget(self.latency_label, 2, 1)
        perf_layout.addWidget(QLabel("Range:"), 3, 0)
        perf_layout.addWidget(self.range_label, 3, 1)
        perf_layout.addWidget(QLabel("Field of View:"), 4, 0)
        perf_layout.addWidget(self.fov_label, 4, 1)

        # Physical specs
        physical_group = QGroupBox("Physical Properties")
        physical_layout = QGridLayout(physical_group)

        self.size_weight_label = QLabel("-")
        self.power_label = QLabel("-")
        self.interface_label = QLabel("-")
        self.platform_label = QLabel("-")

        physical_layout.addWidget(QLabel("Size/Weight:"), 0, 0)
        physical_layout.addWidget(self.size_weight_label, 0, 1)
        physical_layout.addWidget(QLabel("Power:"), 1, 0)
        physical_layout.addWidget(self.power_label, 1, 1)
        physical_layout.addWidget(QLabel("Interface:"), 2, 0)
        physical_layout.addWidget(self.interface_label, 2, 1)
        physical_layout.addWidget(QLabel("Platforms:"), 3, 0)
        physical_layout.addWidget(self.platform_label, 3, 1)

        content_layout.addWidget(perf_group)
        content_layout.addWidget(physical_group)
        content_layout.addStretch()

        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

        return widget

    def create_ros_tab(self) -> QWidget:
        """Create the ROS integration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ROS compatibility info
        compat_group = QGroupBox("ROS Compatibility")
        compat_layout = QVBoxLayout(compat_group)

        self.ros_compat_label = QLabel("-")
        self.ros_compat_label.setWordWrap(True)
        compat_layout.addWidget(self.ros_compat_label)

        # Driver links
        drivers_group = QGroupBox("Driver Information")
        drivers_layout = QVBoxLayout(drivers_group)

        self.ros1_driver_label = QLabel("-")
        self.ros1_driver_label.setOpenExternalLinks(True)
        self.ros1_driver_label.setWordWrap(True)

        self.ros2_driver_label = QLabel("-")
        self.ros2_driver_label.setOpenExternalLinks(True)
        self.ros2_driver_label.setWordWrap(True)

        drivers_layout.addWidget(QLabel("ROS 1 Driver:"))
        drivers_layout.addWidget(self.ros1_driver_label)
        drivers_layout.addWidget(QLabel("ROS 2 Driver:"))
        drivers_layout.addWidget(self.ros2_driver_label)

        layout.addWidget(compat_group)
        layout.addWidget(drivers_group)
        layout.addStretch()

        return widget

    def create_links_tab(self) -> QWidget:
        """Create the links and resources tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Official links
        official_group = QGroupBox("Official Resources")
        official_layout = QVBoxLayout(official_group)

        # Add website/product page link
        self.website_label = QLabel("-")
        self.website_label.setOpenExternalLinks(True)
        self.website_label.setWordWrap(True)

        self.datasheet_label = QLabel("-")
        self.datasheet_label.setOpenExternalLinks(True)
        self.datasheet_label.setWordWrap(True)

        self.github_label = QLabel("-")
        self.github_label.setOpenExternalLinks(True)
        self.github_label.setWordWrap(True)

        official_layout.addWidget(QLabel("Website:"))
        official_layout.addWidget(self.website_label)
        official_layout.addWidget(QLabel("Datasheet:"))
        official_layout.addWidget(self.datasheet_label)
        official_layout.addWidget(QLabel("GitHub Repository:"))
        official_layout.addWidget(self.github_label)

        # Additional info
        additional_group = QGroupBox("Additional Information")
        additional_layout = QVBoxLayout(additional_group)

        self.notes_text = QTextEdit()
        self.notes_text.setReadOnly(True)
        self.notes_text.setMaximumHeight(150)
        additional_layout.addWidget(self.notes_text)

        layout.addWidget(official_group)
        layout.addWidget(additional_group)
        layout.addStretch()

        return widget

    def connect_signals(self):
        """Connect widget signals."""
        self.export_button.clicked.connect(self.on_export_clicked)

    def show_sensor_details(self, sensor_data: Dict[str, Any]):
        """Display details for the specified sensor."""
        self.current_sensor = sensor_data

        # Update header
        display_name = f"{sensor_data.get('manufacturer', 'Unknown')} {sensor_data.get('model', 'Unknown')}"
        self.sensor_title.setText(display_name)
        self.export_button.setEnabled(True)

        # Update overview tab
        self.update_overview_tab(sensor_data)

        # Update specifications tab
        self.update_specifications_tab(sensor_data)

        # Update ROS tab
        self.update_ros_tab(sensor_data)

        # Update links tab
        self.update_links_tab(sensor_data)

    def update_overview_tab(self, sensor: Dict[str, Any]):
        """Update the overview tab with sensor data."""
        self.manufacturer_label.setText(sensor.get("manufacturer", "-"))
        self.model_label.setText(sensor.get("model", "-"))
        self.type_label.setText(sensor.get("sensor_type", "-"))

        # Format price
        price_info = sensor.get("price_range", {})
        if isinstance(price_info, dict) and "min_price" in price_info:
            min_price = price_info["min_price"]
            max_price = price_info.get("max_price", min_price)
            currency = price_info.get("currency", "USD")

            if min_price == max_price:
                price_text = f"{currency} {min_price:,.0f}"
            else:
                price_text = f"{currency} {min_price:,.0f} - {max_price:,.0f}"
        else:
            price_text = "-"

        self.price_label.setText(price_text)

        # Key features - Clean up and format properly
        features = sensor.get("key_features", [])
        if features:
            # Clean and format each feature
            cleaned_features = []
            for feature in features:
                if isinstance(feature, str):
                    # Remove any unwanted characters and strip whitespace
                    cleaned_feature = (
                        feature.strip().replace("\\n", " ").replace("\n", " ")
                    )
                    if cleaned_feature:
                        cleaned_features.append(cleaned_feature)

            if cleaned_features:
                features_text = "\n".join(
                    f"• {feature}" for feature in cleaned_features
                )
            else:
                features_text = "No features listed"
        else:
            features_text = "No features listed"
        self.features_text.setPlainText(features_text)

        # Use cases - Clean up and format properly
        use_cases = sensor.get("use_cases", [])
        if use_cases:
            # Clean and format each use case
            cleaned_use_cases = []
            for use_case in use_cases:
                if isinstance(use_case, str):
                    # Remove any unwanted characters and strip whitespace
                    cleaned_use_case = (
                        use_case.strip().replace("\\n", " ").replace("\n", " ")
                    )
                    if cleaned_use_case:
                        cleaned_use_cases.append(cleaned_use_case)

            if cleaned_use_cases:
                use_cases_text = "\n".join(
                    f"• {use_case}" for use_case in cleaned_use_cases
                )
            else:
                use_cases_text = "No use cases listed"
        else:
            use_cases_text = "No use cases listed"
        self.use_cases_text.setPlainText(use_cases_text)

        # Load sensor image using intelligent fallback strategy
        self._load_sensor_image(sensor)

    def update_specifications_tab(self, sensor: Dict[str, Any]):
        """Update the specifications tab with sensor data using display fields."""
        # Use the formatted display fields from Qt adapter for consistency

        # Resolution - use display format
        resolution_display = sensor.get("resolution_display", "-")
        self.resolution_label.setText(resolution_display)

        # Frame rate - use display format
        frame_rate_display = sensor.get("frame_rate_display", "-")
        self.frame_rate_label.setText(frame_rate_display)

        # Latency - use display format
        latency_display = sensor.get("latency_display", "-")
        self.latency_label.setText(latency_display)

        # Range - use display format
        range_display = sensor.get("range_display", "-")
        self.range_label.setText(range_display)

        # Field of view - use display format
        fov_display = sensor.get("fov_display", "-")
        self.fov_label.setText(fov_display)

        # Physical properties
        # Size/Weight - use combined display format
        size_weight_display = sensor.get("size_weight_display", "-")
        self.size_weight_label.setText(size_weight_display)

        # Power - use display format
        power_display = sensor.get("power_display", "-")
        self.power_label.setText(power_display)

        # Interface - use existing field
        interface = sensor.get("communication_interface", "-")
        self.interface_label.setText(interface)

        # Platform support - use display format
        platform_display = sensor.get("platform_support_display", "-")
        self.platform_label.setText(platform_display)

    def update_ros_tab(self, sensor: Dict[str, Any]):
        """Update the ROS tab with sensor data."""
        ros_compat = sensor.get("ros_compatibility", [])
        if ros_compat:
            compat_text = "Compatible with: " + ", ".join(ros_compat)
        else:
            compat_text = "ROS compatibility not specified"

        self.ros_compat_label.setText(compat_text)

        # Driver links
        ros1_link = sensor.get("driver_link_ros1", "")
        if ros1_link:
            self.ros1_driver_label.setText(
                f'<a href="{ros1_link}">View ROS 1 Driver</a>'
            )
        else:
            self.ros1_driver_label.setText("Not available")

        ros2_link = sensor.get("driver_link_ros2", "")
        if ros2_link:
            self.ros2_driver_label.setText(
                f'<a href="{ros2_link}">View ROS 2 Driver</a>'
            )
        else:
            self.ros2_driver_label.setText("Not available")

    def update_links_tab(self, sensor: Dict[str, Any]):
        """Update the links tab with sensor data."""
        # Website/Product page link
        website_link = sensor.get("product_page", "")
        if website_link:
            self.website_label.setText(
                f'<a href="{website_link}">View Product Page</a>'
            )
        else:
            self.website_label.setText("Not available")

        # Datasheet link
        datasheet_link = sensor.get("datasheet_link", "")
        if datasheet_link:
            self.datasheet_label.setText(
                f'<a href="{datasheet_link}">View Datasheet</a>'
            )
        else:
            self.datasheet_label.setText("Not available")

        # GitHub link
        github_link = sensor.get("github_repo", "")
        if github_link:
            self.github_label.setText(f'<a href="{github_link}">View Repository</a>')
        else:
            self.github_label.setText("Not available")

        # Notes
        notes = sensor.get("notes", "")
        self.notes_text.setPlainText(
            notes if notes else "No additional notes available"
        )

    def show_empty_state(self):
        """Show empty state when no sensor is selected."""
        self.sensor_title.setText("Select a sensor to view details")
        self.sensor_title.setFont(
            create_styled_font("empty_state", "medium")
        )  # Use proper empty_state style
        self.export_button.setEnabled(False)

        # Clear all labels and text fields
        labels_to_clear = [
            self.manufacturer_label,
            self.model_label,
            self.type_label,
            self.price_label,
            self.resolution_label,
            self.frame_rate_label,
            self.latency_label,
            self.range_label,
            self.fov_label,
            self.size_weight_label,
            self.power_label,
            self.interface_label,
            self.platform_label,
            self.ros_compat_label,
            self.ros1_driver_label,
            self.ros2_driver_label,
            self.datasheet_label,
            self.github_label,
        ]

        for label in labels_to_clear:
            label.setText("-")

        # Clear text areas
        self.features_text.clear()
        self.use_cases_text.clear()
        self.notes_text.clear()

    def on_export_clicked(self):
        """Handle export button click."""
        if self.current_sensor:
            sensor_id = self.current_sensor.get("sensor_id", "unknown")
            self.export_requested.emit(sensor_id)
        else:
            QMessageBox.information(self, "Export", "No sensor selected for export.")

    def _get_sensor_image_path(self, sensor: Dict[str, Any]) -> Optional[str]:
        """
        Resolve sensor image path using intelligent fallback strategy.
        Priority: Local assets → Remote URLs → None
        """
        sensor_id = sensor.get("sensor_id", "")

        # Strategy 1: Try local assets first (fastest and most reliable)
        local_image_path = self._find_local_image(sensor_id)
        logger.debug(f"Local image search for {sensor_id}: {local_image_path}")
        if local_image_path:
            logger.info(f"Using local image for {sensor_id}: {local_image_path}")
            return local_image_path

        # Strategy 2: Use remote URL from sensor data
        remote_url = sensor.get("sensor_image", "")
        if remote_url:
            # Handle both string URLs and Pydantic HttpUrl objects
            url_str = str(remote_url)
            if url_str.startswith(("http://", "https://")):
                logger.debug(f"Using remote image for {sensor_id}: {url_str}")
                return url_str

        # Strategy 3: No image available
        logger.debug(f"No image available for sensor {sensor_id}")
        return None

    def _find_local_image(self, sensor_id: str) -> Optional[str]:
        """
        Find local image file for sensor ID using intelligent naming patterns.
        Maps sensor IDs to local image files in assets/sensor_images/
        """
        # Find the project root by going up from the current file location
        # detail_panel.py -> widgets -> gui_qt -> sensor_tool -> src -> project_root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        assets_dir = project_root / "assets" / "sensor_images"

        if not assets_dir.exists():
            return None

        # Create intelligent mapping patterns for sensor ID to filename
        # Handle common naming patterns used in the assets directory
        possible_patterns = [
            # Direct mapping patterns
            f"{sensor_id}.jpg",
            f"{sensor_id}.jpeg",
            f"{sensor_id}.png",
            # Remove underscores and try dashes
            f"{sensor_id.replace('_', '-')}.jpg",
            f"{sensor_id.replace('_', '-')}.jpeg",
            f"{sensor_id.replace('_', '-')}.png",
            # Remove manufacturer prefix patterns (common in naming)
            # e.g., intel_realsense_d435i -> realsense-d435i or d435i
            *self._generate_shortened_patterns(sensor_id),
        ]

        # Try each pattern
        for pattern in possible_patterns:
            image_path = assets_dir / pattern
            if image_path.exists():
                logger.info(f"Found local image for {sensor_id}: {image_path}")
                return str(image_path)

        return None

    def _generate_shortened_patterns(self, sensor_id: str) -> List[str]:
        """Generate comprehensive naming patterns from sensor ID."""
        patterns = []
        parts = sensor_id.split("_")
        extensions = [".jpg", ".jpeg", ".png"]

        if len(parts) > 1:
            # Try without first part (manufacturer)
            shortened = "_".join(parts[1:])
            for ext in extensions:
                patterns.extend(
                    [
                        f"{shortened}{ext}",
                        f"{shortened.replace('_', '-')}{ext}",
                        f"{shortened.replace('_', '')}{ext}",  # No separators
                    ]
                )

            # Try just model part (last parts)
            if len(parts) > 2:
                model_part = "_".join(parts[2:])
                for ext in extensions:
                    patterns.extend(
                        [
                            f"{model_part}{ext}",
                            f"{model_part.replace('_', '-')}{ext}",
                            f"{model_part.replace('_', '')}{ext}",
                        ]
                    )

        # Additional patterns for LiDAR sensors specifically
        sensor_lower = sensor_id.lower()
        for ext in extensions:
            # Handle common LiDAR naming patterns
            if "lrs" in sensor_lower:
                # SICK LRS patterns: sick_lrs4000 -> lrs-4000
                if "lrs4000" in sensor_lower:
                    patterns.append(f"lrs-4000{ext}")
                elif "lrs1000" in sensor_lower:
                    patterns.append(f"lms-1000{ext}")  # Sometimes named lms
                    patterns.append(f"lrs-1000{ext}")

            # RPLiDAR patterns: slamtec_rplidar_a1 -> rplidar-a1
            if "rplidar" in sensor_lower:
                if "a1" in sensor_lower:
                    patterns.append(f"rplidar-a1{ext}")
                elif "a2" in sensor_lower:
                    patterns.append(f"rplidar-a2{ext}")
                elif "s1" in sensor_lower:
                    patterns.append(f"rplidar-s1{ext}")

            # Velodyne patterns
            if "vlp" in sensor_lower:
                if "16" in sensor_lower:
                    patterns.append(f"vlp-16{ext}")
                elif "32" in sensor_lower and "c" in sensor_lower:
                    patterns.append(f"vlp-32c{ext}")

            # Ouster patterns: ouster_os0 -> os0
            if "ouster" in sensor_lower:
                if "os0" in sensor_lower:
                    patterns.append(f"os0{ext}")
                elif "os1" in sensor_lower:
                    patterns.append(f"os1{ext}")
                elif "os2" in sensor_lower:
                    patterns.append(f"os2{ext}")

            # Hesai patterns: hesai_pandarqt -> pandar-qt
            if "pandar" in sensor_lower:
                if "qt" in sensor_lower:
                    patterns.append(f"pandar-qt{ext}")
                elif "xt16" in sensor_lower:
                    patterns.append(f"pandar-xt16{ext}")

        return patterns

    def _load_sensor_image(self, sensor: Dict[str, Any]):
        """
        Load sensor image using lazy loading and intelligent fallback.
        This integrates with the Phase 3 image caching system for optimal performance.
        """
        image_path = self._get_sensor_image_path(sensor)

        if image_path:
            # Use larger size with proper aspect ratio to avoid cropping
            # Set a larger max size but let the image scale proportionally
            target_size = QSize(320, 240)  # 4:3 ratio, more space for full sensor view
            self.sensor_image_label.set_image_path(image_path, target_size)
            logger.info(f"Loading sensor image: {image_path}")
        else:
            # No image available, show placeholder
            self.sensor_image_label.set_image_path(None)
            logger.debug(
                f"No image found for sensor {sensor.get('sensor_id', 'unknown')}"
            )

    def update_fonts(self):
        """Update fonts for dynamic scaling."""
        try:
            # Update main title font - use different font based on content
            title_text = self.sensor_title.text()
            if "Select a sensor" in title_text or "No sensor selected" in title_text:
                self.sensor_title.setFont(create_styled_font("empty_state", "medium"))
            else:
                self.sensor_title.setFont(create_styled_font("h2", "bold"))

            # Update export button
            self.export_button.setFont(create_styled_font("button"))

            # Update all QLabel children with appropriate hierarchy
            labels = self.findChildren(QLabel)
            for label in labels:
                # Skip the main title as it's handled above
                if label == self.sensor_title:
                    continue

                # Use different font sizes based on widget name patterns
                widget_name = label.objectName() if label.objectName() else ""
                label_text = label.text()

                if any(
                    x in label_text.lower()
                    for x in ["specifications", "overview", "features"]
                ):
                    # Section headers
                    label.setFont(create_styled_font("h4", "medium"))
                elif any(x in widget_name.lower() for x in ["title", "header"]):
                    # Widget titles
                    label.setFont(create_styled_font("h5", "medium"))
                else:
                    # Regular labels
                    label.setFont(create_styled_font("body"))

            # Update group box titles
            group_boxes = self.findChildren(QGroupBox)
            for group_box in group_boxes:
                group_box.setFont(create_styled_font("h4", "medium"))

            # Update text areas
            text_areas = self.findChildren(QTextEdit)
            for text_area in text_areas:
                text_area.setFont(create_styled_font("body"))

            # Update tab widget font
            if hasattr(self, "tab_widget"):
                tab_font = create_styled_font("body", "medium")
                self.tab_widget.setFont(tab_font)

                # Update tab bar specifically for better visibility
                tab_bar = self.tab_widget.tabBar()
                tab_bar.setFont(tab_font)

            # Force update
            self.update()

        except Exception as e:
            logger.error(f"Error updating detail panel fonts: {e}")

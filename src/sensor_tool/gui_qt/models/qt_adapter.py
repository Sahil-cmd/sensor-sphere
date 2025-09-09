"""
Qt Integration Adapter for Enhanced Backend

Provides Qt-compatible interface to enhanced SensorRepository and Pydantic models.
Handles async operations and converts data to formats suitable for Qt widgets.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import yaml
from PySide6.QtCore import QObject, QThread, Signal

from sensor_tool.models.repository import SensorRepository, get_default_repository
from sensor_tool.models.sensor_v2 import SensorV2

logger = logging.getLogger(__name__)


class RepositoryWorker(QThread):
    """Background thread worker for non-blocking repository operations."""

    # Signals for operation results
    sensors_loaded = Signal(list)  # List of sensor dictionaries
    operation_progress = Signal(str, int)  # (message, progress_percent)
    operation_error = Signal(str)  # Error message
    operation_completed = Signal()

    def __init__(self, repository: SensorRepository) -> None:
        """Initialize worker with repository instance.

        Args:
            repository: SensorRepository instance for data operations.
        """
        super().__init__()
        self.repository = repository
        self.operation = None
        self.operation_kwargs = {}

    def set_operation(self, operation: str, **kwargs: Any) -> None:
        """Configure operation to execute in background thread.

        Args:
            operation: Operation name ('load_all', 'filter_sensors', 'get_sensor').
            **kwargs: Operation-specific parameters.
        """
        self.operation = operation
        self.operation_kwargs = kwargs

    def run(self) -> None:
        """Execute configured operation in background thread with error handling."""
        try:
            if self.operation == "load_all":
                self._load_all_sensors()
            elif self.operation == "filter_sensors":
                self._filter_sensors()
            elif self.operation == "get_sensor":
                self._get_single_sensor()
            else:
                raise ValueError(f"Unknown operation: {self.operation}")

        except Exception as e:
            logger.error(f"Worker operation failed: {e}")
            self.operation_error.emit(str(e))
        finally:
            self.operation_completed.emit()

    def _load_all_sensors(self) -> None:
        """Load all sensors from repository with progress reporting."""
        self.operation_progress.emit("Loading sensor data...", 10)

        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            force_reload = self.operation_kwargs.get("force_reload", False)
            sensors = loop.run_until_complete(
                self.repository.load_all_sensors(force_reload=force_reload)
            )

            self.operation_progress.emit("Converting data for display...", 80)

            # Convert to enhanced Qt display format with proper data formatting
            sensor_dicts = [
                QtDataAdapter.sensor_to_display_dict(sensor) for sensor in sensors
            ]

            self.operation_progress.emit("Data loaded successfully", 100)
            self.sensors_loaded.emit(sensor_dicts)

        finally:
            loop.close()

    def _filter_sensors(self) -> None:
        """Apply filter criteria to sensors with progress reporting."""
        self.operation_progress.emit("Applying filters...", 10)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            criteria = self.operation_kwargs.get("criteria", {})
            filtered_sensors = loop.run_until_complete(
                self.repository.filter_sensors(**criteria)
            )

            self.operation_progress.emit("Converting filtered data...", 80)

            # Convert to enhanced Qt display format with proper data formatting
            sensor_dicts = [
                QtDataAdapter.sensor_to_display_dict(sensor)
                for sensor in filtered_sensors
            ]

            self.operation_progress.emit("Filtering completed", 100)
            self.sensors_loaded.emit(sensor_dicts)

        finally:
            loop.close()

    def _get_single_sensor(self) -> None:
        """Retrieve specific sensor by ID with error handling."""
        sensor_id = self.operation_kwargs.get("sensor_id")
        if not sensor_id:
            raise ValueError("sensor_id required for get_sensor operation")

        self.operation_progress.emit(f"Loading sensor {sensor_id}...", 10)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            sensor = loop.run_until_complete(
                self.repository.get_sensor_by_id(sensor_id)
            )

            if sensor:
                sensor_dict = QtDataAdapter.sensor_to_display_dict(sensor)
                self.operation_progress.emit("Sensor loaded", 100)
                self.sensors_loaded.emit([sensor_dict])
            else:
                self.operation_error.emit(f"Sensor '{sensor_id}' not found")

        finally:
            loop.close()


class QtSensorRepository(QObject):
    """
    Qt-compatible interface to enhanced SensorRepository.

    Provides signals/slots interface and manages background operations
    for non-blocking GUI interactions.
    """

    # Signals
    sensors_loaded = Signal(list)  # Emitted when sensors are loaded
    operation_started = Signal(str)  # Emitted when operation starts
    operation_progress = Signal(str, int)  # Progress updates (message, percent)
    operation_finished = Signal()  # Emitted when operation completes
    operation_error = Signal(str)  # Emitted on errors

    def __init__(
        self, sensors_directory: str = "sensors", parent: Optional[QObject] = None
    ) -> None:
        """Initialize Qt repository with sensor data directory.

        Args:
            sensors_directory: Directory containing sensor YAML files.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)

        # Initialize repository
        self.repository = SensorRepository(sensors_directory)

        # Background worker
        self.worker = None

        # Cache for current sensor data
        self._current_sensors: List[Dict[str, Any]] = []
        self._repository_stats = {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def load_sensors_async(self, force_reload: bool = False) -> None:
        """Start background loading of all sensors.

        Args:
            force_reload: If True, bypass cache and reload from files.
        """
        if self.worker and self.worker.isRunning():
            self.logger.warning("Operation already in progress")
            return

        self.operation_started.emit("Loading sensors")

        # Create and start worker
        self.worker = RepositoryWorker(self.repository)
        self.worker.sensors_loaded.connect(self._on_sensors_loaded)
        self.worker.operation_progress.connect(self.operation_progress)
        self.worker.operation_error.connect(self.operation_error)
        self.worker.operation_completed.connect(self.operation_finished)

        self.worker.set_operation("load_all", force_reload=force_reload)
        self.worker.start()

    def filter_sensors_async(self, **criteria: Any) -> None:
        """Start background filtering of sensors by criteria.

        Args:
            **criteria: Filter parameters (manufacturer, sensor_type, etc.).
        """
        if self.worker and self.worker.isRunning():
            self.logger.warning("Operation already in progress")
            return

        self.operation_started.emit("Filtering sensors")

        # Create and start worker
        self.worker = RepositoryWorker(self.repository)
        self.worker.sensors_loaded.connect(self._on_sensors_loaded)
        self.worker.operation_progress.connect(self.operation_progress)
        self.worker.operation_error.connect(self.operation_error)
        self.worker.operation_completed.connect(self.operation_finished)

        self.worker.set_operation("filter_sensors", criteria=criteria)
        self.worker.start()

    def get_sensor_async(self, sensor_id: str) -> None:
        """Start background retrieval of specific sensor.

        Args:
            sensor_id: Unique sensor identifier.
        """
        if self.worker and self.worker.isRunning():
            self.logger.warning("Operation already in progress")
            return

        self.operation_started.emit(f"Loading sensor {sensor_id}")

        # Create and start worker
        self.worker = RepositoryWorker(self.repository)
        self.worker.sensors_loaded.connect(self._on_sensors_loaded)
        self.worker.operation_progress.connect(self.operation_progress)
        self.worker.operation_error.connect(self.operation_error)
        self.worker.operation_completed.connect(self.operation_finished)

        self.worker.set_operation("get_sensor", sensor_id=sensor_id)
        self.worker.start()

    def get_current_sensors(self) -> List[Dict[str, Any]]:
        """Get copy of currently cached sensor data.

        Returns:
            List of sensor dictionaries from last successful operation.
        """
        return self._current_sensors.copy()

    def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository performance and cache statistics.

        Returns:
            Dictionary with cache stats, performance metrics, and errors.
        """
        try:
            return self.repository.get_stats()
        except Exception as e:
            self.logger.error(f"Failed to get repository stats: {e}")
            return {}

    def is_operation_running(self) -> bool:
        """Check if background worker thread is active.

        Returns:
            True if operation is currently running.
        """
        return self.worker is not None and self.worker.isRunning()

    def cancel_operation(self) -> None:
        """Terminate background operation and cleanup worker thread."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(5000)  # Wait up to 5 seconds
            self.operation_finished.emit()

    def _on_sensors_loaded(self, sensors: List[Dict[str, Any]]) -> None:
        """Update cache and emit signal when worker completes loading.

        Args:
            sensors: List of sensor dictionaries from worker.
        """
        self._current_sensors = sensors
        self.sensors_loaded.emit(sensors)

        # Update stats
        self._repository_stats = self.get_repository_stats()

        self.logger.info(f"Loaded {len(sensors)} sensors via Qt adapter")


class QtDataAdapter:
    """Static utility class for converting Pydantic models to Qt display formats."""

    @staticmethod
    def sensor_to_display_dict(sensor: SensorV2) -> Dict[str, Any]:
        """Convert Pydantic sensor model to Qt-optimized display dictionary.

        Args:
            sensor: SensorV2 Pydantic model instance.

        Returns:
            Dictionary with formatted display fields for Qt widgets.
        """
        display_dict = sensor.to_dict_v1_compatible()

        # Add computed fields useful for Qt widgets
        display_dict["display_name"] = f"{sensor.manufacturer} {sensor.model}"
        display_dict["type_category"] = sensor.sensor_type.value
        display_dict["ros_support"] = (
            sensor.ros_integration.has_ros_support if sensor.ros_integration else False
        )

        # Try to load raw YAML data as fallback for missing fields
        raw_data = QtDataAdapter._load_raw_sensor_data(sensor.sensor_id)

        # Enhanced price display with proper currency handling
        price_display = "N/A"
        price_numeric_usd = None  # For sorting

        # Enhanced currency symbols mapping
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CNY": "¥",
            "KRW": "₩",
        }

        if sensor.price_range and sensor.price_range.avg_price:
            currency = sensor.price_range.currency or "USD"
            symbol = currency_symbols.get(currency, currency + " ")
            price_display = f"{symbol}{sensor.price_range.avg_price:.0f}"
            price_numeric_usd = sensor.price_range.avg_price  # Assume USD for now
        elif sensor.price_range and sensor.price_range.min_price:
            currency = sensor.price_range.currency or "USD"
            symbol = currency_symbols.get(currency, currency + " ")
            if sensor.price_range.max_price:
                if sensor.price_range.min_price == sensor.price_range.max_price:
                    price_display = f"{symbol}{sensor.price_range.min_price:.0f}"
                else:
                    price_display = f"{symbol}{sensor.price_range.min_price:.0f}–{sensor.price_range.max_price:.0f}"
                price_numeric_usd = (
                    sensor.price_range.min_price + sensor.price_range.max_price
                ) / 2
            else:
                price_display = f"{symbol}{sensor.price_range.min_price:.0f}+"
                price_numeric_usd = sensor.price_range.min_price
        elif display_dict.get("price_range"):
            # Try to extract from price_range structure
            pr = display_dict["price_range"]
            if isinstance(pr, dict):
                min_price = pr.get("min_price")
                max_price = pr.get("max_price")
                avg_price = pr.get("avg_price")
                currency = pr.get("currency", "USD")
                symbol = currency_symbols.get(currency, currency + " ")

                if avg_price:
                    price_display = f"{symbol}{avg_price:.0f}"
                    price_numeric_usd = avg_price
                elif min_price and max_price:
                    if min_price == max_price:
                        price_display = f"{symbol}{min_price:.0f}"
                        price_numeric_usd = min_price
                    else:
                        price_display = f"{symbol}{min_price:.0f}–{max_price:.0f}"
                        price_numeric_usd = (min_price + max_price) / 2
                elif min_price:
                    price_display = f"{symbol}{min_price:.0f}+"
                    price_numeric_usd = min_price
            elif isinstance(pr, str) and pr != "N/A":
                # Handle string price like "Contact manufacturer"
                price_display = pr
                # price_numeric_usd remains None for string prices
        elif raw_data and raw_data.get("price_range"):
            # Fallback to raw YAML price_range
            raw_price = raw_data["price_range"]
            if isinstance(raw_price, dict):
                min_price = raw_price.get("min_price")
                max_price = raw_price.get("max_price")
                avg_price = raw_price.get("avg_price")
                currency = raw_price.get("currency", "USD")
                symbol = currency_symbols.get(currency, currency + " ")

                if avg_price:
                    price_display = f"{symbol}{avg_price:.0f}"
                    price_numeric_usd = avg_price
                elif min_price and max_price:
                    if min_price == max_price:
                        price_display = f"{symbol}{min_price:.0f}"
                        price_numeric_usd = min_price
                    else:
                        price_display = f"{symbol}{min_price:.0f}–{max_price:.0f}"
                        price_numeric_usd = (min_price + max_price) / 2
                elif min_price:
                    price_display = f"{symbol}{min_price:.0f}+"
                    price_numeric_usd = min_price
            elif isinstance(raw_price, str) and raw_price != "N/A":
                price_display = raw_price

        display_dict["price_display"] = price_display
        display_dict["price_numeric_usd"] = price_numeric_usd  # For sorting

        # Enhanced resolution display - try multiple sources
        resolution_display = "N/A"
        if sensor.resolution:
            # Try RGB first, then depth
            if "rgb" in sensor.resolution and sensor.resolution["rgb"]:
                rgb_res = sensor.resolution["rgb"]
                resolution_display = f"{rgb_res.width}x{rgb_res.height}"
            elif "depth" in sensor.resolution and sensor.resolution["depth"]:
                depth_res = sensor.resolution["depth"]
                resolution_display = f"{depth_res.width}x{depth_res.height}"
        elif display_dict.get("resolution"):
            # Try to extract from resolution structure
            res = display_dict["resolution"]
            if isinstance(res, dict):
                if "rgb" in res and isinstance(res["rgb"], dict):
                    rgb = res["rgb"]
                    resolution_display = (
                        f"{rgb.get('width', '?')}x{rgb.get('height', '?')}"
                    )
                elif "depth" in res and isinstance(res["depth"], dict):
                    depth = res["depth"]
                    resolution_display = (
                        f"{depth.get('width', '?')}x{depth.get('height', '?')}"
                    )
                elif "rgb" in res and isinstance(res["rgb"], str):
                    # Handle string format like "1440x1080"
                    resolution_display = res["rgb"]
        elif raw_data and raw_data.get("resolution"):
            # Fallback to raw YAML resolution
            raw_res = raw_data["resolution"]
            if isinstance(raw_res, dict):
                # Try RGB first, then depth
                if "rgb" in raw_res:
                    if isinstance(raw_res["rgb"], dict):
                        rgb = raw_res["rgb"]
                        resolution_display = (
                            f"{rgb.get('width', '?')}x{rgb.get('height', '?')}"
                        )
                    elif isinstance(raw_res["rgb"], str):
                        resolution_display = raw_res["rgb"]
                elif "depth" in raw_res:
                    if isinstance(raw_res["depth"], dict):
                        depth = raw_res["depth"]
                        resolution_display = (
                            f"{depth.get('width', '?')}x{depth.get('height', '?')}"
                        )
                    elif isinstance(raw_res["depth"], str):
                        resolution_display = raw_res["depth"]
            elif isinstance(raw_res, str):
                # Handle string formats like "2208x1242 (max)" or "1920x1200 (side-by-side)"
                # Extract just the resolution part before any parentheses
                import re

                match = re.match(r"(\d+x\d+)", raw_res)
                if match:
                    resolution_display = match.group(1)
                else:
                    resolution_display = raw_res

        # Universal LiDAR resolution extraction (manufacturer-agnostic)
        # Always try LiDAR extractor for LiDAR sensors, regardless of initial resolution_display
        if raw_data:
            sensor_type = display_dict.get("sensor_type", "").lower()
            if "lidar" in sensor_type:
                # Use universal extractor for LiDAR resolution/channels
                lidar_resolution = QtDataAdapter._extract_lidar_resolution(raw_data)
                if lidar_resolution != "N/A":
                    resolution_display = lidar_resolution

        display_dict["resolution_display"] = resolution_display

        # Enhanced frame rate display with unit awareness
        frame_rate_display = "N/A"
        frame_rate_unit = "FPS"  # Default unit

        # Check for LiDAR sensors and their specific unit
        sensor_type = display_dict.get("sensor_type", "").lower()
        if "lidar" in sensor_type and raw_data and raw_data.get("frame_rate_unit"):
            frame_rate_unit = raw_data["frame_rate_unit"]

        if sensor.performance and sensor.performance.frame_rate is not None:
            frame_rate_display = (
                f"{sensor.performance.frame_rate:.1f} {frame_rate_unit}"
            )
        elif display_dict.get("frame_rate"):
            # Handle direct frame_rate value
            fr = display_dict["frame_rate"]
            if isinstance(fr, (int, float)):
                frame_rate_display = f"{fr:.1f} {frame_rate_unit}"
            elif isinstance(fr, str) and ("FPS" in fr or "Hz" in fr):
                # Already formatted
                frame_rate_display = fr
        elif raw_data and raw_data.get("frame_rate"):
            # Fallback to raw YAML frame_rate
            raw_fr = raw_data["frame_rate"]
            if isinstance(raw_fr, str):
                frame_rate_display = raw_fr
            elif isinstance(raw_fr, (int, float)):
                frame_rate_display = f"{raw_fr:.1f} {frame_rate_unit}"
        elif raw_data and raw_data.get("sample_frequency") and "lidar" in sensor_type:
            # For LiDAR sensors, show sample frequency as additional info
            sample_freq = raw_data["sample_frequency"]
            sample_unit = raw_data.get("sample_frequency_unit", "Hz")
            if isinstance(sample_freq, (int, float)):
                frame_rate_display = f"Sample: {sample_freq:.0f} {sample_unit}"
            elif isinstance(sample_freq, str):
                frame_rate_display = f"Sample: {sample_freq}"
        display_dict["frame_rate_display"] = frame_rate_display

        # Enhanced use cases display
        use_cases_display = "N/A"
        if display_dict.get("use_cases") and isinstance(
            display_dict["use_cases"], list
        ):
            # Convert list to comma-separated string, limit to first 3 items for readability
            use_cases_list = display_dict["use_cases"][:3]  # Take first 3 use cases
            use_cases_display = ", ".join(use_cases_list)
            # Add "..." if there are more use cases
            if len(display_dict["use_cases"]) > 3:
                use_cases_display += "..."
        elif raw_data and raw_data.get("use_cases"):
            # Fallback to raw YAML use_cases
            raw_use_cases = raw_data["use_cases"]
            if isinstance(raw_use_cases, list) and raw_use_cases:
                use_cases_list = raw_use_cases[:3]  # Take first 3 use cases
                use_cases_display = ", ".join(use_cases_list)
                if len(raw_use_cases) > 3:
                    use_cases_display += "..."
            elif isinstance(raw_use_cases, str) and raw_use_cases != "N/A":
                use_cases_display = raw_use_cases
        display_dict["use_cases_display"] = use_cases_display

        # Enhanced FOV display
        fov_display = "N/A"
        if sensor.field_of_view:
            if hasattr(sensor.field_of_view, "horizontal") and hasattr(
                sensor.field_of_view, "vertical"
            ):
                h_fov = sensor.field_of_view.horizontal
                v_fov = sensor.field_of_view.vertical
                if h_fov and v_fov:
                    fov_display = f"{h_fov}°×{v_fov}° (H×V)"
                elif h_fov:
                    fov_display = f"{h_fov}° (H)"
        elif raw_data and raw_data.get("field_of_view"):
            raw_fov = raw_data["field_of_view"]
            if isinstance(raw_fov, dict):
                h_fov = raw_fov.get("horizontal")
                v_fov = raw_fov.get("vertical")
                if h_fov and v_fov:
                    fov_display = f"{h_fov}°×{v_fov}° (H×V)"
                elif h_fov:
                    fov_display = f"{h_fov}° (H)"
            elif isinstance(raw_fov, str) and raw_fov != "N/A":
                fov_display = raw_fov
        display_dict["fov_display"] = fov_display

        # Enhanced range display
        range_display = "N/A"
        min_range = None
        max_range = None

        if sensor.performance:
            min_range = sensor.performance.min_range
            max_range = sensor.performance.max_range

        # If not found in performance object, check raw YAML data for top-level min_range/max_range
        if min_range is None and raw_data and raw_data.get("min_range") is not None:
            min_range = raw_data["min_range"]
        if max_range is None and raw_data and raw_data.get("max_range") is not None:
            max_range = raw_data["max_range"]

        if min_range is not None and max_range is not None:
            range_display = f"{min_range}m - {max_range}m"
        elif min_range is not None:
            range_display = f"≥{min_range}m"
        elif max_range is not None:
            range_display = f"≤{max_range}m"
        elif raw_data and raw_data.get("range"):
            # Fallback to range dict structure
            raw_range = raw_data["range"]
            if isinstance(raw_range, dict):
                min_range = raw_range.get("min")
                max_range = raw_range.get("max")
                if min_range is not None and max_range is not None:
                    range_display = f"{min_range}m - {max_range}m"
                elif min_range is not None:
                    range_display = f"≥{min_range}m"
                elif max_range is not None:
                    range_display = f"≤{max_range}m"
            elif isinstance(raw_range, str) and raw_range != "N/A":
                range_display = raw_range
        display_dict["range_display"] = range_display

        # Preserve raw min_range and max_range values for chart generation
        display_dict["min_range"] = min_range
        display_dict["max_range"] = max_range

        # Enhanced latency display with capture_time and data_latency handling
        latency_display = "N/A"
        if sensor.performance and sensor.performance.latency is not None:
            latency_display = f"{sensor.performance.latency}ms"
        elif raw_data and raw_data.get("latency"):
            raw_latency = raw_data["latency"]
            if isinstance(raw_latency, (int, float)):
                latency_display = f"{raw_latency}ms"
            elif isinstance(raw_latency, str) and raw_latency != "N/A":
                if "ms" not in raw_latency.lower():
                    # Add ms unit if not present
                    import re

                    match = re.search(r"(\d+\.?\d*)", raw_latency)
                    if match:
                        latency_display = f"{match.group(1)}ms"
                else:
                    latency_display = raw_latency
        elif raw_data and raw_data.get("data_latency") and "lidar" in sensor_type:
            # Handle LiDAR data_latency field
            data_latency = raw_data["data_latency"]
            if isinstance(data_latency, str) and data_latency != "N/A":
                latency_display = data_latency
        elif raw_data and raw_data.get("capture_time"):
            # Handle capture time as alternative to latency
            capture_time = raw_data["capture_time"]
            if isinstance(capture_time, str):
                # Convert seconds to milliseconds if needed
                import re

                if "s" in capture_time and "ms" not in capture_time:
                    # Assume it's in seconds, convert to ms
                    match = re.search(r"(\d+\.?\d*)", capture_time)
                    if match:
                        seconds = float(match.group(1))
                        latency_display = f"{seconds*1000:.0f}ms (Capture time)"
                else:
                    latency_display = f"{capture_time} (Capture time)"
            elif isinstance(capture_time, (int, float)):
                # Assume it's in seconds if > 10, otherwise ms
                if capture_time > 10:
                    latency_display = f"{capture_time*1000:.0f}ms (Capture time)"
                else:
                    latency_display = f"{capture_time}ms (Capture time)"
        display_dict["latency_display"] = latency_display

        # Enhanced power display
        power_display = "N/A"
        if sensor.performance and sensor.performance.power_consumption is not None:
            power_display = f"{sensor.performance.power_consumption}W"
        elif raw_data and raw_data.get("power_consumption"):
            raw_power = raw_data["power_consumption"]
            if isinstance(raw_power, (int, float)):
                power_display = f"{raw_power}W"
            elif isinstance(raw_power, str) and raw_power != "N/A":
                if "w" not in raw_power.lower():
                    # Add W unit if not present
                    import re

                    match = re.search(r"(\d+\.?\d*)", raw_power)
                    if match:
                        power_display = f"{match.group(1)}W"
                else:
                    power_display = raw_power
        display_dict["power_display"] = power_display

        # Enhanced size/weight display - Fixed to handle actual YAML structure
        size_weight_display = "N/A"
        if raw_data:
            # Look for size/weight in raw data
            size_info = raw_data.get("size") or raw_data.get("dimensions")
            weight_info = raw_data.get("weight") or raw_data.get("mass")
            weight_unit = raw_data.get("weight_unit", "g")

            size_str = ""
            weight_str = ""

            if size_info:
                if isinstance(size_info, dict):
                    # Handle structured size data - FIXED to handle all formats
                    length = size_info.get("length")
                    width = size_info.get("width")
                    height = size_info.get("height")
                    diameter = size_info.get("diameter")
                    # Also check for depth as alternative
                    depth = size_info.get("depth")
                    unit = size_info.get("unit", "mm")

                    # Handle different size formats
                    if length and width and height:
                        # Rectangular: length×width×height
                        size_str = f"{length}×{width}×{height}{unit}"
                    elif width and height and depth:
                        # Alternative: width×height×depth
                        size_str = f"{width}×{height}×{depth}{unit}"
                    elif diameter and height:
                        # Cylindrical: ⌀diameter×height
                        size_str = f"⌀{diameter}×{height}{unit}"
                    elif diameter:
                        # Diameter only: ⌀diameter
                        size_str = f"⌀{diameter}{unit}"
                    elif length and width:
                        # 2D: length×width
                        size_str = f"{length}×{width}{unit}"
                    elif height and width:
                        # Height×width format
                        size_str = f"{height}×{width}{unit}"
                elif isinstance(size_info, str) and size_info != "N/A":
                    size_str = size_info

            if weight_info:
                if isinstance(weight_info, (int, float)):
                    # Handle unit conversion (kg to g)
                    if weight_unit == "kg":
                        weight_info = weight_info * 1000  # Convert kg to g
                        weight_str = f"{weight_info:.0f}g"
                    else:
                        weight_str = f"{weight_info}g"
                elif isinstance(weight_info, str) and weight_info != "N/A":
                    weight_str = weight_info

            # Combine size and weight
            if size_str and weight_str:
                size_weight_display = f"{size_str}, {weight_str}"
            elif size_str:
                size_weight_display = size_str
            elif weight_str:
                size_weight_display = weight_str
        display_dict["size_weight_display"] = size_weight_display

        # Enhanced platform support display
        platform_support_display = "N/A"
        if sensor.supported_platforms:
            # Handle list of platforms
            if isinstance(sensor.supported_platforms, list):
                platform_support_display = ", ".join(sensor.supported_platforms)
            else:
                platform_support_display = str(sensor.supported_platforms)
        elif raw_data and raw_data.get("supported_platforms"):
            raw_platforms = raw_data["supported_platforms"]
            if isinstance(raw_platforms, list):
                platform_support_display = ", ".join(raw_platforms)
            elif isinstance(raw_platforms, str) and raw_platforms != "N/A":
                platform_support_display = raw_platforms
        # Also check for 'platforms' field as alternative
        elif raw_data and raw_data.get("platforms"):
            raw_platforms = raw_data["platforms"]
            if isinstance(raw_platforms, list):
                platform_support_display = ", ".join(raw_platforms)
            elif isinstance(raw_platforms, str) and raw_platforms != "N/A":
                platform_support_display = raw_platforms
        display_dict["platform_support_display"] = platform_support_display

        # Add product_page (website link) support
        product_page = "N/A"
        if raw_data and raw_data.get("product_page"):
            product_page = raw_data["product_page"]
        elif raw_data and raw_data.get("website_link"):
            product_page = raw_data["website_link"]
        elif raw_data and raw_data.get("website"):
            product_page = raw_data["website"]
        display_dict["product_page"] = product_page

        # Universal angular resolution extraction (manufacturer-agnostic)
        angular_resolution_display = QtDataAdapter._extract_angular_resolution(raw_data)
        display_dict["angular_resolution_display"] = angular_resolution_display

        # Universal returns extraction (manufacturer-agnostic)
        returns_display = QtDataAdapter._extract_returns_info(raw_data)
        display_dict["returns_display"] = returns_display

        # Radar-specific display fields
        display_dict["points_per_second_display"] = (
            QtDataAdapter._extract_points_per_second(raw_data)
        )
        display_dict["channels_display"] = QtDataAdapter._extract_channels_info(
            raw_data
        )

        # IMU-specific display fields
        display_dict["sampling_rate_display"] = QtDataAdapter._extract_sampling_rate(
            raw_data
        )

        return display_dict

    @staticmethod
    def _extract_lidar_resolution(raw_data: Optional[Dict[str, Any]]) -> str:
        """Extract LiDAR resolution/channels from various manufacturer formats.

        Args:
            raw_data: Raw YAML sensor data dictionary.

        Returns:
            Formatted resolution string (e.g., '64 channels', '0.1° resolution').
        """
        if not raw_data:
            return "N/A"

        # Try multiple field names and formats for channel information
        channel_fields = ["channels", "channel_count", "num_channels", "beams"]
        for field_name in channel_fields:
            value = raw_data.get(field_name)
            if value is not None:
                # Format 1: Dict with options array (Ouster style)
                if isinstance(value, dict) and "options" in value:
                    options = value["options"]
                    if isinstance(options, list):
                        if len(options) == 1:
                            return f"{options[0]} channels"
                        else:
                            return f"{'/'.join(map(str, options))} channels"
                # Format 2: Direct number (Hesai style)
                elif isinstance(value, (int, float)):
                    return f"{int(value)} channels"
                # Format 3: String representation
                elif isinstance(value, str) and value != "N/A":
                    return f"{value} channels"

        # Try horizontal resolution as fallback
        horizontal_res = raw_data.get("horizontal_resolution")
        if horizontal_res:
            if isinstance(horizontal_res, dict) and "options" in horizontal_res:
                options = horizontal_res["options"]
                if isinstance(options, list):
                    return f"{'/'.join(map(str, options))} points"
            elif isinstance(horizontal_res, (int, float)):
                return f"{horizontal_res} points"

        # For 2D LiDAR, use angular resolution as identifier
        angular_res = raw_data.get("angular_resolution")
        if angular_res:
            if isinstance(angular_res, dict):
                if angular_res.get("typical"):
                    return f"{angular_res['typical']}° resolution"
                elif angular_res.get("horizontal"):
                    if isinstance(angular_res["horizontal"], list):
                        return f"{'/'.join(map(str, angular_res['horizontal']))}° resolution"
                    else:
                        return f"{angular_res['horizontal']}° resolution"

        return "N/A"

    @staticmethod
    def _extract_angular_resolution(raw_data: Optional[Dict[str, Any]]) -> str:
        """Extract angular resolution from various manufacturer formats.

        Args:
            raw_data: Raw YAML sensor data dictionary.

        Returns:
            Formatted angular resolution string or 'N/A' if not found.
        """
        if not raw_data:
            return "N/A"

        # Try angular_resolution first (most common)
        angular_res = raw_data.get("angular_resolution")
        if angular_res and isinstance(angular_res, dict):
            # Multi-value horizontal (SICK)
            if angular_res.get("horizontal"):
                horizontal = angular_res["horizontal"]
                if isinstance(horizontal, list):
                    return f"{'/'.join(map(str, horizontal))}°"
                else:
                    return f"{horizontal}°"
            # Typical value (RPLIDAR)
            elif angular_res.get("typical"):
                return f"{angular_res['typical']}°"
            # Vertical + Horizontal range (Velodyne)
            elif angular_res.get("vertical") and angular_res.get("horizontal_min"):
                v_res = angular_res["vertical"]
                h_min = angular_res["horizontal_min"]
                h_max = angular_res.get("horizontal_max", h_min)
                if h_min == h_max:
                    return f"V:{v_res}° H:{h_min}°"
                else:
                    return f"V:{v_res}° H:{h_min}-{h_max}°"
            # Vertical + Horizontal (Hesai)
            elif angular_res.get("vertical") and angular_res.get("horizontal"):
                v_res = angular_res["vertical"]
                h_res = angular_res["horizontal"]
                return f"V:{v_res}° H:{h_res}°"

        # Try angular_sampling_accuracy (Ouster)
        angular_acc = raw_data.get("angular_sampling_accuracy")
        if angular_acc and isinstance(angular_acc, dict):
            h_acc = angular_acc.get("horizontal")
            v_acc = angular_acc.get("vertical")
            if h_acc and v_acc and h_acc == v_acc:
                return f"{h_acc}"
            elif h_acc:
                return f"H: {h_acc}"

        return "N/A"

    @staticmethod
    def _extract_returns_info(raw_data: Optional[Dict[str, Any]]) -> str:
        """Extract returns/echo information from various formats.

        Args:
            raw_data: Raw YAML sensor data dictionary.

        Returns:
            Formatted returns string (e.g., 'Single', 'Dual', '3 echoes').
        """
        if not raw_data:
            return "N/A"

        # Try multiple field names for returns/echo information
        return_fields = ["returns", "echoes", "multi_echo", "return_type", "echo_count"]
        for field_name in return_fields:
            value = raw_data.get(field_name)
            if value is not None:
                # String format (most common)
                if isinstance(value, str) and value.strip() and value != "N/A":
                    return value.strip()
                # Numeric format
                elif isinstance(value, (int, float)) and value > 0:
                    echo_count = int(value)
                    if echo_count == 1:
                        return "Single"
                    elif echo_count == 2:
                        return "Dual"
                    else:
                        return f"{echo_count} echoes"
                # Boolean format (for multi_echo field)
                elif isinstance(value, bool):
                    return "Multi-echo" if value else "Single"

        return "N/A"

    @staticmethod
    def _extract_points_per_second(raw_data: Optional[Dict[str, Any]]) -> str:
        """Extract radar points per second information from various formats.

        Args:
            raw_data: Raw YAML sensor data dictionary.

        Returns:
            Formatted points per second string (e.g., '100K points/sec') or 'N/A' if not found.
        """
        if not raw_data:
            return "N/A"

        # Try multiple field names for points per second information
        point_fields = [
            "points_per_second",
            "point_rate",
            "data_points",
            "measurement_rate",
        ]
        for field_name in point_fields:
            value = raw_data.get(field_name)
            if value is not None:
                if isinstance(value, (int, float)):
                    # Format large numbers with K/M suffixes
                    if value >= 1000000:
                        return f"{value/1000000:.1f}M points/sec"
                    elif value >= 1000:
                        return f"{value/1000:.0f}K points/sec"
                    else:
                        return f"{int(value)} points/sec"
                elif isinstance(value, str) and value.strip() and value != "N/A":
                    return value.strip()

        return "N/A"

    @staticmethod
    def _extract_channels_info(raw_data: Optional[Dict[str, Any]]) -> str:
        """Extract radar channels information from various formats.

        Args:
            raw_data: Raw YAML sensor data dictionary.

        Returns:
            Formatted channels string (e.g., '128 channels', '64 TX/RX') or 'N/A' if not found.
        """
        if not raw_data:
            return "N/A"

        # Try multiple field names for channel information
        channel_fields = [
            "channels",
            "channel_count",
            "antennas",
            "tx_rx_channels",
            "radar_channels",
        ]
        for field_name in channel_fields:
            value = raw_data.get(field_name)
            if value is not None:
                if isinstance(value, (int, float)):
                    return f"{int(value)} channels"
                elif isinstance(value, str) and value.strip() and value != "N/A":
                    # Check if it already contains 'channel' word
                    if "channel" not in value.lower():
                        return f"{value.strip()} channels"
                    else:
                        return value.strip()
                elif isinstance(value, dict):
                    # Handle structured channel data
                    if "tx" in value and "rx" in value:
                        tx = value["tx"]
                        rx = value["rx"]
                        return f"{tx}TX/{rx}RX"
                    elif "total" in value:
                        return f"{value['total']} channels"

        return "N/A"

    @staticmethod
    def _extract_sampling_rate(raw_data: Optional[Dict[str, Any]]) -> str:
        """Extract IMU sampling rate information from various formats.

        Args:
            raw_data: Raw YAML sensor data dictionary.

        Returns:
            Formatted sampling rate string (e.g., '800 Hz', '1 kHz') or 'N/A' if not found.
        """
        if not raw_data:
            return "N/A"

        # Try multiple field names for sampling rate
        rate_fields = [
            "sampling_rate",
            "sample_rate",
            "output_data_rate",
            "data_rate",
            "update_rate",
        ]
        for field_name in rate_fields:
            value = raw_data.get(field_name)
            unit = raw_data.get(f"{field_name}_unit", "Hz")

            if value is not None:
                if isinstance(value, (int, float)):
                    # Format with appropriate unit
                    if value >= 1000 and unit == "Hz":
                        return f"{value/1000:.1f} kHz"
                    else:
                        return f"{int(value)} {unit}"
                elif isinstance(value, str) and value.strip() and value != "N/A":
                    # Check if it already contains unit
                    if any(u in value.lower() for u in ["hz", "khz", "mhz"]):
                        return value.strip()
                    else:
                        return f"{value.strip()} {unit}"

        return "N/A"

    @staticmethod
    def _load_raw_sensor_data(sensor_id: str) -> Optional[Dict[str, Any]]:
        """Load raw YAML data as fallback when structured data is incomplete.

        Args:
            sensor_id: Unique sensor identifier.

        Returns:
            Raw YAML dictionary or None if not found.
        """
        try:
            # Get the default repository to find the file
            repo = get_default_repository()
            file_path = repo._find_file_for_sensor(sensor_id)

            if file_path and file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f)
                return raw_data

        except Exception as e:
            logger.warning(f"Failed to load raw YAML for sensor {sensor_id}: {e}")

        return None

    @staticmethod
    def sensors_to_table_data(sensors: List[SensorV2]) -> List[Dict[str, Any]]:
        """Convert sensor models to Qt table-compatible format.

        Args:
            sensors: List of SensorV2 model instances.

        Returns:
            List of display dictionaries for Qt table widgets.
        """
        return [QtDataAdapter.sensor_to_display_dict(sensor) for sensor in sensors]

    @staticmethod
    def extract_unique_values(sensors: List[Dict[str, Any]], field: str) -> List[str]:
        """Extract unique field values for populating filter combo boxes.

        Args:
            sensors: List of sensor dictionaries.
            field: Field name to extract values from.

        Returns:
            Sorted list of unique string values.
        """
        values = set()
        for sensor in sensors:
            value = sensor.get(field)
            if value:
                if isinstance(value, list):
                    values.update(str(v) for v in value)
                else:
                    values.add(str(value))
        return sorted(list(values))

    @staticmethod
    def get_filter_suggestions(sensors: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate filter suggestions for common sensor fields.

        Args:
            sensors: List of sensor dictionaries to analyze.

        Returns:
            Dictionary mapping field names to lists of unique values.
        """
        if not sensors:
            return {}

        return {
            "manufacturer": QtDataAdapter.extract_unique_values(
                sensors, "manufacturer"
            ),
            "sensor_type": QtDataAdapter.extract_unique_values(sensors, "sensor_type"),
            "ros_compatibility": QtDataAdapter.extract_unique_values(
                sensors, "ros_compatibility"
            ),
            "communication_interface": QtDataAdapter.extract_unique_values(
                sensors, "communication_interface"
            ),
            "supported_platforms": QtDataAdapter.extract_unique_values(
                sensors, "supported_platforms"
            ),
        }


def create_qt_repository(sensors_directory: str = "sensors") -> QtSensorRepository:
    """Create configured Qt repository instance.

    Args:
        sensors_directory: Directory containing sensor YAML files.

    Returns:
        Configured QtSensorRepository instance.
    """
    return QtSensorRepository(sensors_directory)

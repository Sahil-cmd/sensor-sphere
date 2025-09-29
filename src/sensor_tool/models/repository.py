"""
Sensor Repository with Smart Caching
Data repository with intelligent caching, async operations, and performance monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiofiles
import yaml

from .sensor_v2 import SensorV2

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Performance monitoring with configurable response time targets."""

    @staticmethod
    def measure_time(target_ms: int = 500):
        """Decorator to measure and warn about slow operations.

        Args:
            target_ms: Warning threshold in milliseconds.
        """

        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):

                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.perf_counter()
                    result = await func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    if elapsed_ms > target_ms:
                        logger.warning(
                            f"{func.__name__} took {elapsed_ms:.2f}ms (target: {target_ms}ms)"
                        )
                    else:
                        logger.debug(f"{func.__name__} completed in {elapsed_ms:.2f}ms")

                    return result

                return async_wrapper
            else:

                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.perf_counter()
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    if elapsed_ms > target_ms:
                        logger.warning(
                            f"{func.__name__} took {elapsed_ms:.2f}ms (target: {target_ms}ms)"
                        )
                    else:
                        logger.debug(f"{func.__name__} completed in {elapsed_ms:.2f}ms")

                    return result

                return sync_wrapper

        return decorator


class SensorCache:
    """High-performance sensor cache with intelligent file change detection."""

    def __init__(self) -> None:
        self._sensors: Dict[str, SensorV2] = {}
        self._file_timestamps: Dict[str, float] = {}
        self._last_scan_time: Optional[datetime] = None

    def get(self, sensor_id: str) -> Optional[SensorV2]:
        """Retrieve cached sensor by ID.

        Args:
            sensor_id: Unique sensor identifier.

        Returns:
            Cached sensor instance or None if not found.
        """
        return self._sensors.get(sensor_id)

    def get_all(self) -> List[SensorV2]:
        """Get all cached sensors as a list.

        Returns:
            List of all cached sensor instances.
        """
        return list(self._sensors.values())

    def put(self, sensor: SensorV2, file_path: Optional[str] = None) -> None:
        """Cache sensor with optional file change tracking.

        Args:
            sensor: Sensor instance to cache.
            file_path: Source file path for timestamp tracking.

        Raises:
            ValueError: If sensor or sensor_id is invalid
        """
        if not sensor:
            raise ValueError("Cannot cache None sensor")
        if not sensor.sensor_id:
            raise ValueError("Cannot cache sensor without sensor_id")

        try:
            self._sensors[sensor.sensor_id] = sensor

            if file_path:
                try:
                    file_path_obj = Path(file_path)
                    if file_path_obj.exists():
                        self._file_timestamps[file_path] = file_path_obj.stat().st_mtime
                    else:
                        logger.warning(
                            f"File path does not exist for timestamp tracking: {file_path}"
                        )
                except (OSError, IOError) as e:
                    logger.warning(f"Could not get timestamp for {file_path}: {e}")
                except Exception as e:
                    logger.error(
                        f"Unexpected error getting timestamp for {file_path}: {e}"
                    )
        except Exception as e:
            logger.error(f"Failed to cache sensor {sensor.sensor_id}: {e}")
            raise

    def is_file_changed(self, file_path: str) -> bool:
        """Check if file has changed since last cache update.

        Args:
            file_path: File path to check.

        Returns:
            True if file has been modified since last cache.
        """
        try:
            current_mtime = Path(file_path).stat().st_mtime
            cached_mtime = self._file_timestamps.get(file_path, 0)
            return current_mtime > cached_mtime
        except OSError:
            return True  # Assume changed if we can't check

    def clear(self) -> None:
        """Clear all cached sensors and file timestamps."""
        self._sensors.clear()
        self._file_timestamps.clear()
        self._last_scan_time = None

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics.

        Returns:
            Dictionary with sensor count, file count, memory usage, and last scan time.
        """
        return {
            "sensor_count": len(self._sensors),
            "file_count": len(self._file_timestamps),
            "last_scan": (
                self._last_scan_time.isoformat() if self._last_scan_time else None
            ),
            "memory_usage_mb": sum(
                len(str(sensor.model_dump())) for sensor in self._sensors.values()
            )
            / 1024
            / 1024,
        }


class SensorRepository:
    """
    Modern sensor data repository with intelligent caching and async operations.

    Features:
    - Smart file change detection
    - Async I/O for non-blocking operations
    - Performance monitoring (<500ms target)
    - Backward compatibility with v1.0 YAML files
    - Error resilience and logging
    """

    def __init__(self, sensors_directory: str = "sensors") -> None:
        """Initialize repository with sensor data directory.

        Args:
            sensors_directory: Directory containing sensor YAML files.
        """
        self.sensors_directory = Path(sensors_directory)
        self.cache = SensorCache()
        self._loading_lock = asyncio.Lock()

        # Configure logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Performance tracking
        self._load_stats = {
            "total_loads": 0,
            "cache_hits": 0,
            "file_reads": 0,
            "errors": 0,
        }

        # File mapping cache for sensor lookup
        self._sensor_id_to_file_map: Dict[str, Path] = {}

    @PerformanceMonitor.measure_time(target_ms=500)
    async def load_all_sensors(self, force_reload: bool = False) -> List[SensorV2]:
        """
        Load all sensors with smart caching (target <500ms).

        Args:
            force_reload: If True, ignore cache and reload all files

        Returns:
            List of SensorV2 instances
        """
        async with self._loading_lock:
            self._load_stats["total_loads"] += 1

            if not force_reload:
                cached_sensors = self.cache.get_all()
                if cached_sensors and not await self._any_files_changed():
                    self._load_stats["cache_hits"] += 1
                    self.logger.debug(f"Cache hit: {len(cached_sensors)} sensors")
                    return cached_sensors

            # Load changed/new files
            changed_files = (
                await self._get_changed_files()
                if not force_reload
                else await self._get_all_yaml_files()
            )

            # Load files in parallel for better performance
            if changed_files:
                sensors = await self._load_multiple_files(changed_files)

                # Update cache
                for sensor in sensors:
                    file_path = self._find_file_for_sensor(sensor.sensor_id)
                    if file_path:
                        self.cache.put(sensor, str(file_path))

            all_sensors = self.cache.get_all()
            self.logger.info(
                f"Loaded {len(all_sensors)} sensors ({len(changed_files)} files read)"
            )
            return all_sensors

    @PerformanceMonitor.measure_time(target_ms=300)
    def filter_sensors_sync(self, **criteria) -> List[SensorV2]:
        """
        Synchronous sensor filtering with <300ms target.
        For use with existing CLI/GUI code that expects sync operations.
        """
        sensors = self.cache.get_all()
        if not sensors:
            # If no cache, do synchronous load
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                sensors = loop.run_until_complete(self.load_all_sensors())
            except RuntimeError:
                # No event loop, create one
                sensors = asyncio.run(self.load_all_sensors())

        return self._apply_filters(sensors, criteria)

    async def filter_sensors(self, **criteria) -> List[SensorV2]:
        """Filter sensors by multiple criteria with async performance.

        Args:
            **criteria: Filter criteria (manufacturer, sensor_type, ros_compatibility, price ranges, etc.)

        Returns:
            Filtered list of sensors matching all criteria.
        """
        sensors = await self.load_all_sensors()
        return self._apply_filters(sensors, criteria)

    def _apply_filters(
        self, sensors: List[SensorV2], criteria: Dict[str, Any]
    ) -> List[SensorV2]:
        """Apply multiple filter criteria to sensor collection.

        Args:
            sensors: List of sensors to filter.
            criteria: Dictionary of filter criteria and values.

        Returns:
            Filtered sensor list matching all criteria.
        """
        filtered = []

        for sensor in sensors:
            if self._matches_criteria(sensor, criteria):
                filtered.append(sensor)

        self.logger.debug(f"Filtered {len(sensors)} sensors to {len(filtered)} matches")
        return filtered

    def _matches_criteria(self, sensor: SensorV2, criteria: Dict[str, Any]) -> bool:
        """Check if individual sensor matches all filter criteria.

        Args:
            sensor: Sensor to evaluate.
            criteria: Filter criteria dictionary.

        Returns:
            True if sensor matches all criteria.
        """

        # Manufacturer filter
        if "manufacturer" in criteria:
            if criteria["manufacturer"].lower() not in sensor.manufacturer.lower():
                return False

        # Sensor type filter
        if "sensor_type" in criteria:
            if criteria["sensor_type"].lower() not in sensor.sensor_type.value.lower():
                return False

        # Model search filter - searches across model, manufacturer, and sensor_id
        if "model_search" in criteria:
            search_term = criteria["model_search"].lower()
            model_match = search_term in sensor.model.lower()
            manufacturer_match = search_term in sensor.manufacturer.lower()
            sensor_id_match = search_term in sensor.sensor_id.lower()

            if not any([model_match, manufacturer_match, sensor_id_match]):
                return False

        # ROS compatibility filter
        if "ros_compatibility" in criteria:
            if not sensor.ros_integration:
                return False
            sensor_ros_versions = [
                v.value for v in sensor.ros_integration.ros_compatibility
            ]
            criteria_ros = criteria["ros_compatibility"]

            # Handle both list and single value criteria
            if isinstance(criteria_ros, list):
                # Check if sensor supports ANY of the requested ROS versions
                if not any(
                    ros_version in sensor_ros_versions for ros_version in criteria_ros
                ):
                    return False
            else:
                # Single value criteria
                if criteria_ros not in sensor_ros_versions:
                    return False

        # Price range filters
        if "min_price" in criteria or "max_price" in criteria:
            sensor_price = sensor.price_usd
            include_unknown = criteria.get(
                "include_unknown_prices", True
            )  # Default: include unknown

            # Debug logging for price filtering
            logger.debug(
                f"Price filter check for {sensor.sensor_id}: price_usd={sensor_price}, "
                f"range=({criteria.get('min_price', 'None')}, {criteria.get('max_price', 'None')}), "
                f"include_unknown={include_unknown}"
            )

            if sensor_price is None:
                # Include sensors with missing prices if include_unknown is True
                result = include_unknown
                logger.debug(
                    f"  → {sensor.sensor_id}: price unknown, include_unknown={include_unknown} → {result}"
                )
                return result

            if "min_price" in criteria and sensor_price < criteria["min_price"]:
                logger.debug(
                    f"  → {sensor.sensor_id}: price {sensor_price} < min {criteria['min_price']} → excluded"
                )
                return False
            if "max_price" in criteria and sensor_price > criteria["max_price"]:
                logger.debug(
                    f"  → {sensor.sensor_id}: price {sensor_price} > max {criteria['max_price']} → excluded"
                )
                return False

            logger.debug(
                f"  → {sensor.sensor_id}: price {sensor_price} within range → included"
            )

        # Resolution filters
        if "min_resolution" in criteria or "max_resolution" in criteria:
            max_pixels = sensor.max_resolution_pixels
            if max_pixels == 0:
                return False

            if "min_resolution" in criteria and max_pixels < criteria["min_resolution"]:
                return False
            if "max_resolution" in criteria and max_pixels > criteria["max_resolution"]:
                return False

        # Frame rate filters
        if "min_frame_rate" in criteria or "max_frame_rate" in criteria:
            fps = sensor.frame_rate_fps
            if fps is None:
                return False

            if "min_frame_rate" in criteria and fps < criteria["min_frame_rate"]:
                return False
            if "max_frame_rate" in criteria and fps > criteria["max_frame_rate"]:
                return False

        return True

    async def get_sensor_by_id(self, sensor_id: str) -> Optional[SensorV2]:
        """Retrieve sensor by unique identifier.

        Args:
            sensor_id: Unique sensor identifier.

        Returns:
            Sensor instance or None if not found.
        """
        # Check cache first
        cached = self.cache.get(sensor_id)
        if cached:
            return cached

        # Load all sensors if not in cache
        await self.load_all_sensors()
        return self.cache.get(sensor_id)

    async def _get_all_yaml_files(self) -> List[Path]:
        """Scan directory for all YAML sensor files.

        Returns:
            Sorted list of YAML file paths.

        Raises:
            FileNotFoundError: If sensors directory doesn't exist
            PermissionError: If directory access is denied
        """
        try:
            if not self.sensors_directory.exists():
                error_msg = f"Sensors directory not found: {self.sensors_directory}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            if not self.sensors_directory.is_dir():
                error_msg = f"Sensors path is not a directory: {self.sensors_directory}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            yaml_files = []
            try:
                for pattern in ["*.yaml", "*.yml"]:
                    yaml_files.extend(self.sensors_directory.rglob(pattern))
            except (PermissionError, OSError) as e:
                error_msg = f"Permission denied or I/O error scanning directory {self.sensors_directory}: {e}"
                self.logger.error(error_msg)
                raise PermissionError(error_msg) from e

            self.logger.debug(
                f"Found {len(yaml_files)} YAML files in {self.sensors_directory}"
            )
            return sorted(yaml_files)

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError, PermissionError)):
                raise
            error_msg = (
                f"Unexpected error scanning directory {self.sensors_directory}: {e}"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def _get_changed_files(self) -> List[Path]:
        """Identify files modified since last cache update.

        Returns:
            List of changed file paths requiring reload.
        """
        all_files = await self._get_all_yaml_files()
        changed_files = []

        for file_path in all_files:
            if self.cache.is_file_changed(str(file_path)):
                changed_files.append(file_path)

        return changed_files

    async def _any_files_changed(self) -> bool:
        """Fast check for any file modifications.

        Returns:
            True if any sensor files have been modified.
        """
        changed = await self._get_changed_files()
        return len(changed) > 0

    async def _load_multiple_files(self, file_paths: List[Path]) -> List[SensorV2]:
        """Load multiple sensor files concurrently for performance.

        Args:
            file_paths: List of YAML files to load.

        Returns:
            List of successfully loaded sensors.
        """
        tasks = [self._load_single_file(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sensors = []
        for result in results:
            if isinstance(result, SensorV2):
                sensors.append(result)
            elif isinstance(result, Exception):
                self._load_stats["errors"] += 1
                self.logger.error(f"Failed to load sensor file: {result}")

        return sensors

    async def _load_single_file(self, file_path: Path) -> Optional[SensorV2]:
        """Load and parse single sensor YAML file.

        Args:
            file_path: Path to sensor YAML file.

        Returns:
            Parsed sensor instance or None on error.

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If sensor data is invalid
            IOError: If file I/O fails
        """
        try:
            self._load_stats["file_reads"] += 1

            # Validate file exists and is readable
            if not file_path.exists():
                raise FileNotFoundError(f"Sensor file not found: {file_path}")
            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")

            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
            except (OSError, IOError) as e:
                raise IOError(f"Failed to read file {file_path}: {e}") from e
            except UnicodeDecodeError as e:
                raise ValueError(f"File encoding error in {file_path}: {e}") from e

            if not content or content.strip() == "":
                raise ValueError(f"Empty sensor file: {file_path}")

            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"YAML parsing error in {file_path}: {e}") from e

            if data is None:
                raise ValueError(f"YAML file contains no data: {file_path}")
            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid sensor data format in {file_path} - expected dictionary"
                )

            # Convert from v1.0 format to v2.0 with validation
            try:
                sensor = SensorV2.from_dict_v1(data)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse sensor data from {file_path}: {e}"
                ) from e

            if not sensor.sensor_id:
                raise ValueError(f"Sensor missing sensor_id in {file_path}")

            # Build file mapping cache for sensor lookup
            self._sensor_id_to_file_map[sensor.sensor_id] = file_path

            self.logger.debug(
                f"Loaded sensor: {sensor.sensor_id} from {file_path.name}"
            )
            return sensor

        except Exception as e:
            # Log the error but let the caller handle it
            if isinstance(e, (FileNotFoundError, yaml.YAMLError, ValueError, IOError)):
                self.logger.warning(f"Failed to load {file_path}: {e}")
            else:
                self.logger.error(f"Unexpected error loading {file_path}: {e}")
            raise  # Re-raise for gather() to handle

    def _find_file_for_sensor(self, sensor_id: str) -> Optional[Path]:
        """Locate YAML file for sensor using cached mapping.

        Args:
            sensor_id: Unique sensor identifier.

        Returns:
            File path for sensor or None if not found.
        """
        # Use the file mapping cache first (fast O(1) lookup)
        if sensor_id in self._sensor_id_to_file_map:
            return self._sensor_id_to_file_map[sensor_id]

        # If cache is empty, build the complete mapping first
        if not self._sensor_id_to_file_map:
            self._build_file_mapping_cache()

        # Try cache again after building
        if sensor_id in self._sensor_id_to_file_map:
            return self._sensor_id_to_file_map[sensor_id]

        return None

    def _build_file_mapping_cache(self) -> None:
        """Build sensor ID to file path mapping for fast lookups."""

        self._sensor_id_to_file_map.clear()

        for file_path in self.sensors_directory.rglob("*.yaml"):
            try:
                # Load YAML to get sensor_id
                with open(file_path, "r", encoding="utf-8") as f:
                    import yaml

                    data = yaml.safe_load(f)

                if data and "sensor_id" in data:
                    sensor_id = data["sensor_id"]
                    self._sensor_id_to_file_map[sensor_id] = file_path

            except Exception as e:
                self.logger.warning(
                    f"Failed to read {file_path} for cache building: {e}"
                )

                # Fallback to filename-based mapping
                file_stem = file_path.stem
                self._sensor_id_to_file_map[file_stem] = file_path

    # Backward compatibility methods for existing code
    def load_sensor_data_sync(self) -> List[Dict[str, Any]]:
        """
        Load sensor data in v1.0 dictionary format for backward compatibility.
        Used by existing CLI/GUI components.

        Returns:
            List of sensor dictionaries in v1.0 format

        Raises:
            RuntimeError: If loading fails due to system errors
        """
        sensors = []
        try:
            # Check if we have cached data first
            cached_sensors = self.cache.get_all()
            if not cached_sensors:
                # Load without using asyncio.run to avoid event loop conflicts
                import asyncio

                try:
                    # Try to get existing event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, can't use asyncio.run
                        self.logger.warning(
                            "Cannot load sensors synchronously from within async context"
                        )
                        return []
                    else:
                        cached_sensors = loop.run_until_complete(
                            self.load_all_sensors()
                        )
                except RuntimeError:
                    # No event loop, create one
                    try:
                        cached_sensors = asyncio.run(self.load_all_sensors())
                    except Exception as async_error:
                        error_msg = (
                            f"Failed to load sensors in new event loop: {async_error}"
                        )
                        self.logger.error(error_msg)
                        raise RuntimeError(error_msg) from async_error
                except Exception as e:
                    error_msg = f"Failed to load sensors using existing event loop: {e}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

            # Convert to v1.0 format with error handling
            try:
                sensors = []
                for sensor in cached_sensors:
                    try:
                        sensor_dict = sensor.to_dict_v1_compatible()
                        sensors.append(sensor_dict)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to convert sensor {sensor.sensor_id} to v1 format: {e}"
                        )
                        continue

                self.logger.info(
                    f"Successfully converted {len(sensors)} sensors to v1 format"
                )

            except Exception as e:
                error_msg = f"Failed to convert sensors to v1.0 format: {e}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            error_msg = f"Unexpected error loading sensors in sync mode: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        return sensors

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive repository performance statistics.

        Returns:
            Dictionary with cache stats, performance metrics, and directory info.
        """
        cache_stats = self.cache.get_stats()
        return {
            "cache": cache_stats,
            "performance": self._load_stats.copy(),
            "sensors_directory": str(self.sensors_directory),
            "directory_exists": self.sensors_directory.exists(),
        }


# Backward compatibility: Create a global repository instance
# This allows existing code to import and use the repository easily
_default_repository = None


def get_default_repository() -> SensorRepository:
    """Get or create the default repository singleton.

    Returns:
        Global repository instance for backward compatibility.
    """
    global _default_repository
    if _default_repository is None:
        _default_repository = SensorRepository()
    return _default_repository

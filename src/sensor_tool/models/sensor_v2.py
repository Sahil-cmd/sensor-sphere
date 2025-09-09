"""
Sensor Models v2.0
Pydantic models for sensor data validation and type safety.
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, model_validator, validator


def parse_numeric_value(value: Any) -> Optional[float]:
    """Parse numeric value from various string formats or return numeric value directly.

    This utility function extracts numeric values from strings containing units
    or other text, enabling flexible data parsing from sensor specifications.

    Args:
        value: Input value to parse. Can be int, float, str, or Any type.
               String examples: "1.25 FPS", "800 ms", "30 W", "-5.5 dB".

    Returns:
        Parsed numeric value as float, or None if parsing fails.

    Example:
        >>> parse_numeric_value("1.25 FPS")
        1.25
        >>> parse_numeric_value(42)
        42.0
        >>> parse_numeric_value("invalid")
        None
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Extract first numeric value from string
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

    return None


class SensorType(str, Enum):
    """Standardized sensor type enumeration for consistent classification.

    Defines the supported sensor categories used throughout the SensorSphere
    application for filtering, validation, and display purposes.

    Values:
        RGB_CAMERA: Standard color cameras
        DEPTH_CAMERA: Depth-sensing cameras (structured light, ToF, etc.)
        STEREO_CAMERA: Dual-camera stereo vision systems
        LIDAR: Light Detection and Ranging sensors
        RADAR: Radio Detection and Ranging sensors
        IMU: Inertial Measurement Units
        INFRARED_CAMERA: Infrared/thermal imaging cameras
        THERMAL_CAMERA: Thermal imaging sensors
        TOF_CAMERA: Time-of-Flight cameras
        STRUCTURED_LIGHT: Structured light projection systems
    """

    RGB_CAMERA = "RGB Camera"
    DEPTH_CAMERA = "Depth Camera"
    STEREO_CAMERA = "Stereo Camera"
    LIDAR = "LiDAR"
    RADAR = "Radar"
    IMU = "IMU"
    INFRARED_CAMERA = "Infrared Camera"
    THERMAL_CAMERA = "Thermal Camera"
    TOF_CAMERA = "Time-of-Flight Camera"
    STRUCTURED_LIGHT = "Structured Light Camera"


class Currency(str, Enum):
    """Supported currency codes for international pricing.

    ISO 4217 currency codes used for sensor pricing information.
    Enables multi-currency support and standardized price comparisons.

    Values:
        USD: United States Dollar
        EUR: Euro
        SGD: Singapore Dollar
        GBP: British Pound Sterling
        INR: Indian Rupee
    """

    USD = "USD"
    EUR = "EUR"
    SGD = "SGD"
    GBP = "GBP"
    INR = "INR"


class ROSVersion(str, Enum):
    """Robot Operating System (ROS) version enumeration.

    Defines supported ROS versions for driver compatibility tracking.
    Used to assess sensor integration capabilities with robotic systems.

    Values:
        ROS1: ROS 1 (Noetic, Melodic, etc.)
        ROS2: ROS 2 (Humble, Galactic, Foxy, etc.)
    """

    ROS1 = "ROS1"
    ROS2 = "ROS2"


class Resolution(BaseModel):
    """Image resolution specification with validation and computed properties.

    Represents pixel dimensions for various sensor modalities (RGB, depth, IR)
    with automatic calculation of derived metrics like total pixels and aspect ratio.

    Attributes:
        width: Resolution width in pixels (must be > 0)
        height: Resolution height in pixels (must be > 0)

    Example:
        >>> res = Resolution(width=1920, height=1080)
        >>> res.total_pixels
        2073600
        >>> res.aspect_ratio
        1.777...
    """

    width: int = Field(..., gt=0, description="Resolution width in pixels")
    height: int = Field(..., gt=0, description="Resolution height in pixels")

    @property
    def total_pixels(self) -> int:
        """Calculate total pixel count for resolution comparison.

        Returns:
            Total number of pixels (width × height).

        Example:
            >>> Resolution(width=1920, height=1080).total_pixels
            2073600
        """
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio for display format analysis.

        Returns:
            Aspect ratio as width/height (e.g., 1.777 for 16:9).

        Example:
            >>> Resolution(width=1920, height=1080).aspect_ratio
            1.7777777777777777
        """
        return self.width / self.height

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


class FieldOfView(BaseModel):
    """Field of view angles specification with validation.

    Represents the angular coverage of a sensor in horizontal, vertical,
    and diagonal dimensions. Supports partial specifications where only
    some angles are known.

    Attributes:
        horizontal: Horizontal field of view in degrees (0-360°)
        vertical: Vertical field of view in degrees (0-360°)
        diagonal: Diagonal field of view in degrees (0-360°)

    Note:
        All angles are optional, allowing for incomplete specifications
        from manufacturer datasheets.
    """

    horizontal: Optional[float] = Field(
        None, ge=0, le=360, description="Horizontal FOV in degrees"
    )
    vertical: Optional[float] = Field(
        None, ge=0, le=360, description="Vertical FOV in degrees"
    )
    diagonal: Optional[float] = Field(
        None, ge=0, le=360, description="Diagonal FOV in degrees"
    )

    @property
    def calculated_diagonal(self) -> Optional[float]:
        """Calculate diagonal field of view from horizontal and vertical angles.

        Uses Pythagorean theorem to compute diagonal FOV when both horizontal
        and vertical angles are available, otherwise returns the explicitly
        set diagonal value.

        Returns:
            Calculated or explicit diagonal FOV in degrees, or None if unavailable.

        Note:
            The calculation assumes rectangular field of view geometry.
        """
        if self.horizontal is not None and self.vertical is not None:
            return (self.horizontal**2 + self.vertical**2) ** 0.5
        return self.diagonal


class PriceRange(BaseModel):
    """Price range specification with currency support and validation.

    Represents sensor pricing information as a range with currency specification.
    Supports single-point pricing (min or max only) and provides computed
    average pricing for comparison purposes.

    Attributes:
        min_price: Minimum price in specified currency (≥ 0)
        max_price: Maximum price in specified currency (≥ 0, ≥ min_price)
        currency: Currency code (defaults to USD)

    Raises:
        ValueError: If max_price is less than min_price.

    Example:
        >>> price = PriceRange(min_price=500.0, max_price=800.0, currency=Currency.USD)
        >>> price.avg_price
        650.0
    """

    min_price: Optional[float] = Field(None, ge=0, description="Minimum price")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price")
    currency: Currency = Field(Currency.USD, description="Currency code")

    @property
    def avg_price(self) -> Optional[float]:
        """Calculate average price for comparison purposes.

        Returns:
            Average of min_price and max_price if both available,
            otherwise returns the single available price, or None.

        Example:
            >>> PriceRange(min_price=500.0, max_price=800.0).avg_price
            650.0
            >>> PriceRange(min_price=500.0).avg_price
            500.0
        """
        if self.min_price is not None and self.max_price is not None:
            return (self.min_price + self.max_price) / 2
        return self.min_price or self.max_price

    @validator("max_price")
    def max_price_must_be_greater_than_min(
        cls, v: Optional[float], values: Dict[str, Any]
    ) -> Optional[float]:
        if v is not None and "min_price" in values and values["min_price"] is not None:
            if v < values["min_price"]:
                raise ValueError("max_price must be greater than or equal to min_price")
        return v


class ROSIntegration(BaseModel):
    """ROS (Robot Operating System) integration specifications.

    Comprehensive ROS compatibility information including supported versions,
    driver repositories, and metadata for robotic system integration assessment.

    Attributes:
        ros_compatibility: List of supported ROS versions (ROS1, ROS2)
        driver_link_ros1: URL to ROS1 driver repository
        driver_link_ros2: URL to ROS2 driver repository
        driver_link: General driver repository URL
        package_name: ROS package name for installation
        maintainer: Package maintainer information
        last_commit_date: Last driver update timestamp

    Example:
        >>> ros = ROSIntegration(
        ...     ros_compatibility=[ROSVersion.ROS1, ROSVersion.ROS2],
        ...     package_name="realsense2_camera"
        ... )
        >>> ros.has_ros_support
        True
    """

    ros_compatibility: List[ROSVersion] = Field(
        default_factory=list, description="Supported ROS versions"
    )
    driver_link_ros1: Optional[HttpUrl] = Field(
        None, description="ROS1 driver repository URL"
    )
    driver_link_ros2: Optional[HttpUrl] = Field(
        None, description="ROS2 driver repository URL"
    )
    driver_link: Optional[HttpUrl] = Field(
        None, description="General driver repository URL"
    )

    # Enhanced ROS metadata (for future expansion)
    package_name: Optional[str] = Field(None, description="ROS package name")
    maintainer: Optional[str] = Field(None, description="Package maintainer")
    last_commit_date: Optional[datetime] = Field(None, description="Last driver update")

    @property
    def ros_compatibility_score(self) -> int:
        """Calculate numerical ROS compatibility score for comparison.

        Scoring system:
        - ROS1 support: +1 point
        - ROS2 support: +2 points (weighted higher for modern systems)

        Returns:
            Compatibility score (0-3):
            - 0: No ROS support
            - 1: ROS1 only
            - 2: ROS2 only
            - 3: Both ROS1 and ROS2
        """
        score = 0
        if ROSVersion.ROS1 in self.ros_compatibility:
            score += 1
        if ROSVersion.ROS2 in self.ros_compatibility:
            score += 2
        return score

    @property
    def has_ros_support(self) -> bool:
        """Check if sensor has any ROS integration support.

        Returns:
            True if sensor supports at least one ROS version, False otherwise.

        Example:
            >>> ros = ROSIntegration(ros_compatibility=[ROSVersion.ROS2])
            >>> ros.has_ros_support
            True
        """
        return len(self.ros_compatibility) > 0


class PerformanceMetrics(BaseModel):
    """Sensor performance metrics with validation and units.

    Comprehensive performance specifications including timing, range,
    and power characteristics for sensor evaluation and comparison.

    Attributes:
        frame_rate: Frame rate in FPS (cameras) or Hz (LiDAR), must be > 0
        frame_rate_unit: Unit specification (FPS for cameras, Hz for LiDAR)
        latency: Processing latency in milliseconds, must be ≥ 0
        min_range: Minimum sensing range in meters, must be ≥ 0
        max_range: Maximum sensing range in meters, must be > min_range
        power_consumption: Power consumption in watts, must be ≥ 0

    Raises:
        ValueError: If max_range is not greater than min_range.

    Example:
        >>> perf = PerformanceMetrics(
        ...     frame_rate=30.0,
        ...     latency=50.0,
        ...     min_range=0.2,
        ...     max_range=10.0,
        ...     power_consumption=2.5
        ... )
    """

    frame_rate: Optional[float] = Field(
        None, gt=0, description="Frame rate in FPS or Hz"
    )
    frame_rate_unit: Optional[str] = Field(
        "FPS", description="Frame rate unit (FPS for cameras, Hz for LiDAR)"
    )
    latency: Optional[float] = Field(None, ge=0, description="Latency in milliseconds")
    min_range: Optional[float] = Field(
        None, ge=0, description="Minimum range in meters"
    )
    max_range: Optional[float] = Field(
        None, gt=0, description="Maximum range in meters"
    )
    power_consumption: Optional[float] = Field(
        None, ge=0, description="Power consumption in watts"
    )

    @validator("max_range")
    def max_range_must_be_greater_than_min(
        cls, v: Optional[float], values: Dict[str, Any]
    ) -> Optional[float]:
        if v is not None and "min_range" in values and values["min_range"] is not None:
            if v <= values["min_range"]:
                raise ValueError("max_range must be greater than min_range")
        return v


class PhysicalSpecs(BaseModel):
    """Physical sensor characteristics and dimensions.

    Specifies physical properties for mechanical integration planning
    and system design considerations.

    Attributes:
        weight: Sensor weight in grams, must be > 0
        length: Length dimension in millimeters, must be > 0
        width: Width dimension in millimeters, must be > 0
        height: Height dimension in millimeters, must be > 0

    Note:
        All dimensions are optional to accommodate incomplete specifications.
        Volume calculation requires all three dimensions.

    Example:
        >>> specs = PhysicalSpecs(
        ...     weight=175.0,
        ...     length=90.0,
        ...     width=25.0,
        ...     height=25.0
        ... )
        >>> specs.volume_mm3
        56250.0
    """

    weight: Optional[float] = Field(None, gt=0, description="Weight in grams")
    length: Optional[float] = Field(None, gt=0, description="Length in mm")
    width: Optional[float] = Field(None, gt=0, description="Width in mm")
    height: Optional[float] = Field(None, gt=0, description="Height in mm")

    @property
    def volume_mm3(self) -> Optional[float]:
        """Calculate sensor volume for space planning.

        Returns:
            Volume in cubic millimeters (length × width × height),
            or None if any dimension is missing.

        Example:
            >>> specs = PhysicalSpecs(length=90.0, width=25.0, height=25.0)
            >>> specs.volume_mm3
            56250.0
        """
        if all(dim is not None for dim in [self.length, self.width, self.height]):
            return self.length * self.width * self.height
        return None


class AngularResolution(BaseModel):
    """LiDAR angular resolution specification with multiple measurement types.

    Defines the angular resolution characteristics for LiDAR sensors,
    supporting various specification formats found in manufacturer datasheets.

    Attributes:
        vertical: Vertical angular resolution in degrees
        horizontal: Horizontal angular resolution in degrees
        horizontal_min: Minimum horizontal angular resolution in degrees
        horizontal_max: Maximum horizontal angular resolution in degrees
        typical: Typical angular resolution in degrees
        unit: Unit of measurement (defaults to "degrees")
        note: Additional clarifying information about the resolution

    Note:
        LiDAR specifications often provide ranges or conditional values,
        hence the flexible attribute structure.
    """

    vertical: Optional[float] = Field(
        None, description="Vertical angular resolution in degrees"
    )
    horizontal: Optional[float] = Field(
        None, description="Horizontal angular resolution in degrees"
    )
    horizontal_min: Optional[float] = Field(
        None, description="Minimum horizontal angular resolution in degrees"
    )
    horizontal_max: Optional[float] = Field(
        None, description="Maximum horizontal angular resolution in degrees"
    )
    typical: Optional[float] = Field(
        None, description="Typical angular resolution in degrees"
    )
    unit: Optional[str] = Field("degrees", description="Unit for angular resolution")
    note: Optional[str] = Field(
        None, description="Additional notes about angular resolution"
    )


class LaserSpecification(BaseModel):
    """Comprehensive laser characteristics for LiDAR sensors.

    Detailed laser specifications including safety, optical, and technical
    parameters essential for regulatory compliance and system integration.

    Attributes:
        wavelength: Laser wavelength specification (e.g., '905 nm', '1550 nm')
        laser_class: Safety classification (e.g., 'Class 1', 'Class 1M')
        type: Laser type description (e.g., 'Infrared laser', 'VCSEL')
        power: Optical power specification (e.g., '<5 mW', '10-50 mW')
        safety: Safety rating description (e.g., 'Eye-safe', 'Skin-safe')
        modulation: Modulation technique (e.g., 'Modulated pulse', 'CW')
        beam_diameter: Laser beam diameter specification (e.g., '9.5 mm')
        beam_divergence: Beam divergence angle (e.g., '0.18° (FWHM)')
        light_source: General light source description

    Note:
        All specifications are string-based to accommodate the variety
        of formats used in manufacturer documentation.
    """

    wavelength: Optional[str] = Field(
        None, description="Laser wavelength (e.g., '905 nm')"
    )
    laser_class: Optional[str] = Field(
        None, description="Laser safety class (e.g., 'Class 1')"
    )
    type: Optional[str] = Field(
        None, description="Type of laser (e.g., 'Infrared laser')"
    )
    power: Optional[str] = Field(None, description="Laser power (e.g., '<5 mW')")
    safety: Optional[str] = Field(None, description="Safety rating (e.g., 'Eye-safe')")
    modulation: Optional[str] = Field(
        None, description="Modulation type (e.g., 'Modulated pulse')"
    )
    beam_diameter: Optional[str] = Field(
        None, description="Beam diameter (e.g., '9.5 mm')"
    )
    beam_divergence: Optional[str] = Field(
        None, description="Beam divergence (e.g., '0.18° (FWHM)')"
    )
    light_source: Optional[str] = Field(None, description="Light source description")


class LiDARSpecs(BaseModel):
    """Comprehensive LiDAR sensor specifications and capabilities.

    Detailed technical specifications specific to LiDAR sensors, including
    scanning patterns, laser characteristics, data processing, and performance metrics.

    Attributes:
        channels: Number of laser channels/beams, must be > 0
        angular_resolution: Angular resolution specifications
        range_accuracy: Range measurement accuracy specification
        range_precision: Distance-dependent precision specifications
        returns: Return capability (e.g., 'Single/Dual', 'Multi-echo')
        points_per_second: Point cloud generation rates by mode
        sample_frequency: Sampling frequency specifications
        scanning_method: Scanning technique description
        operational_principle: Core measurement principle
        measurement_principle: Specific measurement technology
        laser_specification: Detailed laser characteristics
        response_time: Response time in milliseconds, must be ≥ 0
        data_latency: Data processing latency specification
        timestamp_resolution: Timestamp precision specification
        false_positive_rate: False detection rate specification
        multi_sensor_immunity: Multi-sensor interference resistance
        interference_rejection: General interference rejection capability
        time_synchronization: Supported synchronization methods

    Example:
        >>> lidar = LiDARSpecs(
        ...     channels=64,
        ...     returns="Dual",
        ...     response_time=100.0
        ... )
        >>> lidar.is_multi_channel
        True
    """

    channels: Optional[int] = Field(None, gt=0, description="Number of laser channels")
    angular_resolution: Optional[AngularResolution] = Field(
        None, description="Angular resolution specification"
    )

    # Range and accuracy
    range_accuracy: Optional[str] = Field(
        None, description="Range measurement accuracy"
    )
    range_precision: Optional[Dict[str, str]] = Field(
        None, description="Range precision at different distances"
    )

    # Returns and point cloud
    returns: Optional[str] = Field(
        None, description="Return type (e.g., 'Single/Dual', '3 (multi-echo)')"
    )
    points_per_second: Optional[Dict[str, int]] = Field(
        None, description="Point cloud generation rates"
    )

    # Technical specifications
    sample_frequency: Optional[Dict[str, Union[int, str]]] = Field(
        None, description="Sampling frequency specification"
    )
    scanning_method: Optional[str] = Field(None, description="Scanning methodology")
    operational_principle: Optional[str] = Field(
        None, description="Measurement principle"
    )
    measurement_principle: Optional[str] = Field(
        None, description="Specific measurement technology"
    )

    # Laser details
    laser_specification: Optional[LaserSpecification] = Field(
        None, description="Laser specifications"
    )

    # Performance characteristics
    response_time: Optional[float] = Field(
        None, ge=0, description="Response time in milliseconds"
    )
    data_latency: Optional[str] = Field(None, description="Data processing latency")
    timestamp_resolution: Optional[str] = Field(None, description="Timestamp precision")
    false_positive_rate: Optional[str] = Field(
        None, description="False positive detection rate"
    )

    # Interference and synchronization
    multi_sensor_immunity: Optional[bool] = Field(
        None, description="Multi-sensor interference immunity"
    )
    interference_rejection: Optional[bool] = Field(
        None, description="Interference rejection capability"
    )
    time_synchronization: Optional[List[str]] = Field(
        default_factory=list, description="Time sync capabilities"
    )

    @property
    def is_multi_channel(self) -> bool:
        """Determine if LiDAR has multiple laser channels.

        Returns:
            True if sensor has more than one laser channel, False otherwise.

        Note:
            Multi-channel LiDARs typically provide higher vertical resolution
            and better performance for autonomous vehicle applications.
        """
        return self.channels is not None and self.channels > 1

    @property
    def has_multi_return(self) -> bool:
        """Check if LiDAR supports multiple return detection.

        Analyzes the returns specification to determine if the sensor
        can detect multiple reflections per laser pulse, useful for
        seeing through vegetation or detecting transparent objects.

        Returns:
            True if sensor supports dual, triple, or multi-return detection,
            False for single-return only sensors.

        Example:
            >>> specs = LiDARSpecs(returns="Dual")
            >>> specs.has_multi_return
            True
        """
        if not self.returns:
            return False
        return any(
            term in self.returns.lower()
            for term in ["dual", "multi", "triple", "2", "3"]
        )


class SensorV2(BaseModel):
    """
    Enhanced Sensor Model v2.0

    Modern-grade sensor specification with comprehensive validation,
    type safety, and computed properties.
    """

    # Core identification (backward compatible with v1.0)
    schema_version: str = Field("2.0", description="Schema version")
    sensor_id: str = Field(
        ..., pattern=r"^[a-zA-Z0-9_-]+$", description="Unique sensor identifier"
    )
    sensor_type: SensorType = Field(..., description="Type of sensor")
    manufacturer: str = Field(..., min_length=1, description="Manufacturer name")
    model: str = Field(..., min_length=1, description="Model designation")

    # Technical specifications
    resolution: Optional[Dict[str, Resolution]] = Field(
        default_factory=dict,
        description="Resolution specifications for different modalities (rgb, depth, ir)",
    )
    field_of_view: Optional[FieldOfView] = Field(
        None, description="Field of view specification"
    )
    performance: Optional[PerformanceMetrics] = Field(
        None, description="Performance characteristics"
    )
    physical: Optional[PhysicalSpecs] = Field(
        None, description="Physical characteristics"
    )
    lidar_specs: Optional[LiDARSpecs] = Field(
        None, description="LiDAR-specific specifications"
    )

    # Pricing and availability
    price_range: Optional[PriceRange] = Field(None, description="Price information")

    # ROS integration
    ros_integration: Optional[ROSIntegration] = Field(
        None, description="ROS integration details"
    )

    # Documentation and resources
    datasheet_link: Optional[HttpUrl] = Field(
        None, description="Link to official datasheet"
    )
    github_repo: Optional[HttpUrl] = Field(None, description="GitHub repository")
    sensor_image: Optional[HttpUrl] = Field(None, description="Product image URL")

    # Metadata
    key_features: List[str] = Field(
        default_factory=list, description="Key product features"
    )
    use_cases: List[str] = Field(default_factory=list, description="Common use cases")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    notes: Optional[str] = Field(None, description="Additional notes")

    # Enhanced metadata for standard use
    environmental_rating: Optional[str] = Field(
        None, description="IP rating or environmental spec"
    )
    supported_platforms: List[str] = Field(
        default_factory=list, description="Supported OS platforms"
    )
    communication_interface: Optional[str] = Field(
        None, description="Communication interface"
    )

    # Validation and provenance (for future use)
    validation_status: str = Field("draft", description="Validation status")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"  # Prevent unexpected fields
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            HttpUrl: str,
        }
        json_schema_extra = {
            "example": {
                "schema_version": "2.0",
                "sensor_id": "intel_realsense_d435i",
                "sensor_type": "Depth Camera",
                "manufacturer": "Intel",
                "model": "RealSense D435i",
                "resolution": {
                    "rgb": {"width": 1920, "height": 1080},
                    "depth": {"width": 1280, "height": 720},
                },
                "field_of_view": {
                    "horizontal": 87.0,
                    "vertical": 58.0,
                    "diagonal": 91.2,
                },
                "performance": {
                    "frame_rate": 90.0,
                    "latency": 50.0,
                    "min_range": 0.2,
                    "max_range": 10.0,
                    "power_consumption": 1.5,
                },
            }
        }

    @validator("sensor_id")
    def sensor_id_must_be_lowercase(cls, v: str) -> str:
        """Validate and enforce sensor ID naming conventions.

        Ensures sensor IDs use consistent lowercase format with underscores
        and hyphens only, enabling reliable file naming and URL generation.

        Args:
            v: The sensor_id value to validate.

        Returns:
            Validated sensor_id string.

        Raises:
            ValueError: If sensor_id contains invalid characters or uppercase letters.

        Example:
            >>> # Valid IDs
            >>> "intel_realsense_d435i"  # OK
            >>> "velodyne-vlp16"         # OK
            >>>
            >>> # Invalid IDs (will raise ValueError)
            >>> "Intel_RealSense"        # Uppercase letters
            >>> "sensor id with spaces"  # Spaces
        """
        if not re.match(r"^[a-z0-9_-]+$", v):
            raise ValueError(
                "sensor_id must be lowercase alphanumeric with underscores/hyphens only"
            )
        return v

    @validator("resolution", pre=True)
    def parse_resolution_formats(
        cls, v: Union[Dict[str, Any], str, None]
    ) -> Dict[str, Resolution]:
        """Parse and normalize resolution data from various input formats.

        Handles multiple resolution input formats commonly found in sensor
        specifications, converting them to a standardized structure.

        Args:
            v: Resolution data in various formats:
               - Dict[str, Resolution]: Already structured (pass-through)
               - Dict[str, dict]: Nested dict with width/height
               - Dict[str, str]: String values like "1440x1080"
               - str: Single resolution like "2208x1242 (max)"
               - None: No resolution data

        Returns:
            Dictionary mapping modality names (rgb, depth, ir) to Resolution objects.
            Empty dict if input is None or unparseable.

        Example:
            >>> # String format
            >>> parse_resolution_formats("1920x1080")
            {'rgb': Resolution(width=1920, height=1080)}

            >>> # Nested dict format
            >>> parse_resolution_formats({
            ...     "rgb": {"width": 1920, "height": 1080},
            ...     "depth": "640x480"
            ... })
            {'rgb': Resolution(width=1920, height=1080),
             'depth': Resolution(width=640, height=480)}
        """
        if v is None:
            return {}

        # Already structured format
        if isinstance(v, dict):
            # Handle nested structured format - convert string values to Resolution objects
            result = {}
            for modality, res_data in v.items():
                if isinstance(res_data, str):
                    # Parse string format like "1440x1080"
                    match = re.match(r"(\d+)x(\d+)", res_data)
                    if match:
                        result[modality] = Resolution(
                            width=int(match.group(1)), height=int(match.group(2))
                        )
                elif (
                    isinstance(res_data, dict)
                    and "width" in res_data
                    and "height" in res_data
                ):
                    result[modality] = Resolution(**res_data)
            return result

        # String format like "2208x1242 (max)" or "1920x1200 (side-by-side)"
        elif isinstance(v, str):
            match = re.match(r"(\d+)x(\d+)", v)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
                return {"rgb": Resolution(width=width, height=height)}

        return {}

    @model_validator(mode="after")
    def validate_ros_integration_consistency(self) -> "SensorV2":
        """Validate consistency between ROS compatibility claims and driver links.

        Performs cross-validation to ensure that declared ROS version support
        is backed by appropriate driver repository links, improving data quality.

        Returns:
            Self (SensorV2 instance) after validation.

        Note:
            This validator logs warnings for inconsistencies but does not raise
            exceptions, allowing for incomplete data while flagging issues.

        Validation Rules:
            - If ROS1 compatibility is claimed, expect driver_link_ros1 or driver_link
            - If ROS2 compatibility is claimed, expect driver_link_ros2 or driver_link
            - Missing links generate warnings but don't block validation
        """
        if self.ros_integration and self.ros_integration.ros_compatibility:
            ros_versions = self.ros_integration.ros_compatibility

            # Check that driver links match declared compatibility
            if ROSVersion.ROS1 in ros_versions and not any(
                [
                    self.ros_integration.driver_link_ros1,
                    self.ros_integration.driver_link,
                ]
            ):
                # Warning: Could add logging here
                pass

            if ROSVersion.ROS2 in ros_versions and not any(
                [
                    self.ros_integration.driver_link_ros2,
                    self.ros_integration.driver_link,
                ]
            ):
                # Warning: Could add logging here
                pass

        return self

    # Computed properties for easy access
    @property
    def rgb_resolution(self) -> Optional[Resolution]:
        """Get RGB camera resolution specification.

        Returns:
            RGB resolution object if available, None otherwise.

        Example:
            >>> sensor.rgb_resolution.width if sensor.rgb_resolution else 0
            1920
        """
        return self.resolution.get("rgb") if self.resolution else None

    @property
    def depth_resolution(self) -> Optional[Resolution]:
        """Get depth camera resolution specification.

        Returns:
            Depth resolution object if available, None otherwise.

        Example:
            >>> sensor.depth_resolution.total_pixels if sensor.depth_resolution else 0
            921600
        """
        return self.resolution.get("depth") if self.resolution else None

    @property
    def max_resolution_pixels(self) -> int:
        """Get highest resolution across all sensor modalities.

        Compares all available modalities (RGB, depth, IR, etc.) and returns
        the highest pixel count for resolution-based filtering and comparison.

        Returns:
            Maximum pixel count across all modalities, or 0 if no resolutions specified.

        Example:
            >>> sensor.max_resolution_pixels
            2073600  # 1920x1080 RGB if higher than depth resolution
        """
        if not self.resolution:
            return 0
        return max(res.total_pixels for res in self.resolution.values())

    @property
    def price_usd(self) -> Optional[float]:
        """Get average price in USD for comparison purposes.

        Currently returns the average price from price_range without currency
        conversion. Future versions may implement multi-currency conversion.

        Returns:
            Average price in original currency units, or None if no pricing available.

        Note:
            Assumes price_range currency is USD or equivalent for comparison.
            Actual currency conversion will be implemented in future versions.
        """
        if self.price_range:
            return self.price_range.avg_price
        return None

    @property
    def ros_score(self) -> int:
        """Get numerical ROS compatibility score for filtering and ranking.

        Returns:
            ROS compatibility score (0-3), or 0 if no ROS integration specified.

        See ROSIntegration.ros_compatibility_score for scoring details.
        """
        if self.ros_integration:
            return self.ros_integration.ros_compatibility_score
        return 0

    @property
    def is_outdoor_suitable(self) -> bool:
        """Heuristic assessment of outdoor deployment suitability.

        Analyzes sensor tags to determine if the sensor is likely suitable
        for outdoor environments based on manufacturer descriptions.

        Returns:
            True if sensor appears suitable for outdoor use, False otherwise.

        Note:
            This is a heuristic based on tag analysis. Always verify with
            manufacturer specifications for critical applications.

        Tags indicating outdoor suitability:
            - "outdoor": Explicitly rated for outdoor use
            - "weather": Weather resistance mentioned
            - "rugged": Ruggedized construction
        """
        return any(tag in ["outdoor", "weather", "rugged"] for tag in self.tags)

    @property
    def frame_rate_fps(self) -> Optional[float]:
        """Get sensor frame rate for performance comparison.

        Returns:
            Frame rate value from performance metrics, or None if not specified.

        Note:
            Units may vary (FPS for cameras, Hz for LiDAR). Check
            performance.frame_rate_unit for unit specification.
        """
        if self.performance:
            return self.performance.frame_rate
        return None

    # LiDAR-specific computed properties
    @property
    def is_lidar(self) -> bool:
        """Check if this is a LiDAR sensor."""
        return self.sensor_type == SensorType.LIDAR

    @property
    def lidar_channels(self) -> Optional[int]:
        """Get LiDAR channel count."""
        if self.lidar_specs:
            return self.lidar_specs.channels
        return None

    @property
    def has_multi_return(self) -> bool:
        """Check if LiDAR supports multiple returns."""
        if self.lidar_specs:
            return self.lidar_specs.has_multi_return
        return False

    @property
    def is_multi_channel_lidar(self) -> bool:
        """Check if this is a multi-channel LiDAR."""
        if self.lidar_specs:
            return self.lidar_specs.is_multi_channel
        return False

    @property
    def display_name(self) -> str:
        """Get formatted display name."""
        return f"{self.manufacturer} {self.model}"

    def to_dict_v1_compatible(self) -> Dict[str, Any]:
        """
        Convert to v1.0 compatible dictionary for backward compatibility.
        Used by existing CLI and GUI components.
        """
        result = {
            "schema_version": "1.0",  # Report as v1.0 for compatibility
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "manufacturer": self.manufacturer,
            "model": self.model,
        }

        # Convert resolution format
        if self.resolution:
            result["resolution"] = {}
            for modality, res in self.resolution.items():
                result["resolution"][modality] = {
                    "width": res.width,
                    "height": res.height,
                }

        # Convert field of view
        if self.field_of_view:
            fov_dict = {}
            if self.field_of_view.horizontal is not None:
                fov_dict["horizontal"] = self.field_of_view.horizontal
            if self.field_of_view.vertical is not None:
                fov_dict["vertical"] = self.field_of_view.vertical
            if self.field_of_view.diagonal is not None:
                fov_dict["diagonal"] = self.field_of_view.diagonal
            if fov_dict:
                result["field_of_view"] = fov_dict

        # Convert performance metrics to individual fields
        if self.performance:
            if self.performance.frame_rate is not None:
                result["frame_rate"] = self.performance.frame_rate
                result["frame_rate_unit"] = "FPS"
            if self.performance.latency is not None:
                result["latency"] = self.performance.latency
                result["latency_unit"] = "ms"
            if self.performance.min_range is not None:
                result["min_range"] = self.performance.min_range
            if self.performance.max_range is not None:
                result["max_range"] = self.performance.max_range
            if self.performance.power_consumption is not None:
                result["power_consumption"] = self.performance.power_consumption
                result["power_consumption_unit"] = "W"

        # Convert ROS integration
        if self.ros_integration and self.ros_integration.ros_compatibility:
            result["ros_compatibility"] = [
                v.value for v in self.ros_integration.ros_compatibility
            ]
            if self.ros_integration.driver_link_ros1:
                result["driver_link_ros1"] = str(self.ros_integration.driver_link_ros1)
            if self.ros_integration.driver_link_ros2:
                result["driver_link_ros2"] = str(self.ros_integration.driver_link_ros2)
            if self.ros_integration.driver_link:
                result["driver_link"] = str(self.ros_integration.driver_link)

        # Convert price range
        if self.price_range:
            price_dict = {"currency": self.price_range.currency.value}
            if self.price_range.min_price is not None:
                price_dict["min_price"] = self.price_range.min_price
            if self.price_range.max_price is not None:
                price_dict["max_price"] = self.price_range.max_price
            result["price_range"] = price_dict

        # Convert physical specs
        if self.physical:
            if self.physical.weight is not None:
                result["weight"] = self.physical.weight
                result["weight_unit"] = "g"

            if all(
                dim is not None
                for dim in [
                    self.physical.length,
                    self.physical.width,
                    self.physical.height,
                ]
            ):
                result["size"] = {
                    "length": self.physical.length,
                    "width": self.physical.width,
                    "height": self.physical.height,
                    "unit": "mm",
                }

        # Copy simple fields
        for field in [
            "key_features",
            "use_cases",
            "tags",
            "notes",
            "environmental_rating",
            "supported_platforms",
            "communication_interface",
        ]:
            value = getattr(self, field, None)
            if value:
                result[field] = value

        # Copy URLs as strings
        if self.datasheet_link:
            result["datasheet_link"] = str(self.datasheet_link)
        if self.github_repo:
            result["github_repo"] = str(self.github_repo)
        if self.sensor_image:
            result["sensor_image"] = str(self.sensor_image)

        return result

    @classmethod
    def from_dict_v1(cls, data: Dict[str, Any]) -> "SensorV2":
        """
        Create SensorV2 instance from v1.0 dictionary format.
        Enables seamless migration from existing YAML files.
        """
        # Convert resolution format
        resolution = {}
        if "resolution" in data and isinstance(data["resolution"], dict):
            for modality, res_data in data["resolution"].items():
                if (
                    isinstance(res_data, dict)
                    and "width" in res_data
                    and "height" in res_data
                ):
                    resolution[modality] = Resolution(
                        width=res_data["width"], height=res_data["height"]
                    )

        # Convert field of view
        field_of_view = None
        if "field_of_view" in data and isinstance(data["field_of_view"], dict):
            fov_data = data["field_of_view"]
            field_of_view = FieldOfView(
                horizontal=fov_data.get("horizontal"),
                vertical=fov_data.get("vertical"),
                diagonal=fov_data.get("diagonal"),
            )

        # Convert performance metrics with intelligent parsing
        performance = None
        perf_fields = {}
        if "frame_rate" in data:
            perf_fields["frame_rate"] = parse_numeric_value(data["frame_rate"])
        if "latency" in data:
            perf_fields["latency"] = parse_numeric_value(data["latency"])
        if "min_range" in data:
            perf_fields["min_range"] = parse_numeric_value(data["min_range"])
        if "max_range" in data:
            perf_fields["max_range"] = parse_numeric_value(data["max_range"])
        if "power_consumption" in data:
            perf_fields["power_consumption"] = parse_numeric_value(
                data["power_consumption"]
            )

        if perf_fields:
            performance = PerformanceMetrics(**perf_fields)

        # Convert physical specs with intelligent parsing
        physical = None
        phys_fields = {}
        if "weight" in data:
            phys_fields["weight"] = parse_numeric_value(data["weight"])
        if "size" in data and isinstance(data["size"], dict):
            size_data = data["size"]
            phys_fields.update(
                {
                    "length": parse_numeric_value(size_data.get("length")),
                    "width": parse_numeric_value(size_data.get("width")),
                    "height": parse_numeric_value(size_data.get("height")),
                }
            )

        if phys_fields:
            physical = PhysicalSpecs(**phys_fields)

        # Convert ROS integration
        ros_integration = None
        if "ros_compatibility" in data:
            ros_compat = []
            for version in data["ros_compatibility"]:
                if version in ["ROS1", "ROS2"]:
                    ros_compat.append(ROSVersion(version))

            ros_fields = {"ros_compatibility": ros_compat}
            if "driver_link_ros1" in data:
                ros_fields["driver_link_ros1"] = data["driver_link_ros1"]
            if "driver_link_ros2" in data:
                ros_fields["driver_link_ros2"] = data["driver_link_ros2"]
            if "driver_link" in data:
                ros_fields["driver_link"] = data["driver_link"]

            if ros_compat or any(
                link for link in ros_fields.values() if isinstance(link, str)
            ):
                ros_integration = ROSIntegration(**ros_fields)

        # Convert price range with intelligent parsing
        price_range = None
        if "price_range" in data and isinstance(data["price_range"], dict):
            price_data = data["price_range"]
            price_fields = {}
            if "min_price" in price_data:
                price_fields["min_price"] = parse_numeric_value(price_data["min_price"])
            if "max_price" in price_data:
                price_fields["max_price"] = parse_numeric_value(price_data["max_price"])
            if "currency" in price_data:
                try:
                    price_fields["currency"] = Currency(price_data["currency"])
                except ValueError:
                    price_fields["currency"] = Currency.USD  # Default fallback

            if price_fields:
                price_range = PriceRange(**price_fields)

        # Convert sensor type
        sensor_type = SensorType.DEPTH_CAMERA  # Default
        if "sensor_type" in data:
            try:
                sensor_type = SensorType(data["sensor_type"])
            except ValueError:
                # Handle unknown sensor types gracefully
                sensor_type = SensorType.DEPTH_CAMERA

        return cls(
            schema_version="2.0",
            sensor_id=data["sensor_id"],
            sensor_type=sensor_type,
            manufacturer=data["manufacturer"],
            model=data["model"],
            resolution=resolution if resolution else None,
            field_of_view=field_of_view,
            performance=performance,
            physical=physical,
            ros_integration=ros_integration,
            price_range=price_range,
            datasheet_link=data.get("datasheet_link"),
            github_repo=data.get("github_repo"),
            sensor_image=data.get("sensor_image"),
            key_features=data.get("key_features", []),
            use_cases=data.get("use_cases", []),
            tags=data.get("tags", []),
            notes=data.get("notes"),
            environmental_rating=data.get("environmental_rating"),
            supported_platforms=data.get("supported_platforms", []),
            communication_interface=data.get("communication_interface"),
        )


# Export key classes for easy imports
__all__ = [
    "SensorV2",
    "ROSIntegration",
    "PerformanceMetrics",
    "SensorType",
    "Resolution",
    "FieldOfView",
    "PriceRange",
    "PhysicalSpecs",
]

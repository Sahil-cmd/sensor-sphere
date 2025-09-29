"""
Natural Language Query Parser for Sensor Database

Implements Tier 1 pattern-based natural language processing for sensor queries.
Converts queries like "Intel depth cameras under $500" into structured filters.

This is a pragmatic, lightweight implementation that provides immediate value
without complex AI dependencies.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class OperatorType(Enum):
    """Query operators for comparisons."""

    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    EQUALS = "="
    NOT_EQUALS = "!="
    CONTAINS = "contains"
    BETWEEN = "between"


@dataclass
class ParsedFilter:
    """Represents a single parsed filter criterion."""

    field: str
    operator: OperatorType
    value: Union[str, float, int, List]
    confidence: float = 1.0


@dataclass
class ParsedQuery:
    """Structured representation of parsed natural language query."""

    filters: List[ParsedFilter]
    text_search_terms: List[str]
    original_query: str
    confidence: float = 1.0
    error_message: Optional[str] = None


class NaturalLanguageQueryParser:
    """
    Converts natural language queries into structured pandas filters.

    Supports queries like:
    - "Intel depth cameras under $500"
    - "sensors with fps > 30 and latency < 50ms"
    - "ROS2 stereo cameras between $200 and $1000"
    - "high resolution cameras for outdoor use"
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_mappings()

    def _setup_mappings(self):
        """Initialize field mappings and patterns."""

        # Map natural language terms to database fields
        self.field_mappings = {
            # Performance terms
            "fps": "frame_rate",
            "frame rate": "frame_rate",
            "framerate": "frame_rate",
            "frames per second": "frame_rate",
            "hz": "frame_rate",
            # Latency terms
            "latency": "latency",
            "delay": "latency",
            "lag": "latency",
            "response time": "latency",
            # Price terms
            "price": "price_avg",
            "cost": "price_avg",
            "budget": "price_avg",
            "expensive": "price_avg",
            "cheap": "price_avg",
            "under": "price_avg",  # "under $500"
            # Resolution terms
            "resolution": "resolution_rgb_pixels",
            "pixels": "resolution_rgb_pixels",
            "megapixels": "resolution_rgb_pixels",
            "mp": "resolution_rgb_pixels",
            "high resolution": "resolution_rgb_pixels",
            "high res": "resolution_rgb_pixels",
            # Range terms
            "range": "max_range",
            "distance": "max_range",
            "reach": "max_range",
            "max range": "max_range",
            "maximum range": "max_range",
            "detection range": "max_range",
            # Field of view
            "fov": "field_of_view_horizontal",
            "field of view": "field_of_view_horizontal",
            "viewing angle": "field_of_view_horizontal",
            "horizontal fov": "field_of_view_horizontal",
            # Weight terms
            "weight": "weight",
            "mass": "weight",
            "heavy": "weight",
            "light": "weight",
            # Power terms
            "power": "power_consumption",
            "power consumption": "power_consumption",
            "watts": "power_consumption",
            "w": "power_consumption",
            "energy": "power_consumption",
            # Platform/OS terms
            "platform": "supported_platforms",
            "os": "supported_platforms",
            "operating system": "supported_platforms",
            "runs on": "supported_platforms",
            "supports": "supported_platforms",
            "compatible with": "supported_platforms",
            # Basic info
            "manufacturer": "manufacturer",
            "brand": "manufacturer",
            "company": "manufacturer",
            "made by": "manufacturer",
            "from": "manufacturer",
            "type": "sensor_type",
            "camera": "sensor_type",
            "sensor": "sensor_type",
            "model": "model",
        }

        # Operator patterns (order matters - more specific first)
        self.operator_patterns = [
            (
                r"\b(greater than or equal to|at least|minimum|min|>=)\s*",
                OperatorType.GREATER_EQUAL,
            ),
            (
                r"\b(less than or equal to|at most|maximum|max|<=)\s*",
                OperatorType.LESS_EQUAL,
            ),
            (
                r"\b(greater than|more than|above|over|higher than|exceeds|>)\s*",
                OperatorType.GREATER_THAN,
            ),
            (
                r"\b(less than|under|below|cheaper than|lower than|<)\s*",
                OperatorType.LESS_THAN,
            ),
            (r"\b(exactly|equal to|equals|=)\s*", OperatorType.EQUALS),
            (r"\b(not equal to|not equals|!=)\s*", OperatorType.NOT_EQUALS),
            (r"\b(between)\s+", OperatorType.BETWEEN),
            (r"\b(contains?|includes?|has|with)\s*", OperatorType.CONTAINS),
        ]

        # Common manufacturer aliases (case-insensitive matching)
        self.manufacturer_aliases = {
            "Intel": [
                "intel",
                "realsense",
                "real sense",
                "d435i",
                "d455",
                "t265",
                "d435",
                "d415",
            ],
            "StereoLabs": [
                "stereolabs",
                "zed",
                "stereo labs",
                "stereolab",
                "zed2",
                "zed-x",
                "zed2i",
                "zed 2",
                "zed x",
            ],
            "Microsoft": ["microsoft", "kinect"],
            "ASUS": ["asus", "xtion"],
            "Orbbec": ["orbbec", "astra"],
            "Mech-Mind Robotics": [
                "mech-mind",
                "mech mind",
                "mechmind",
                "mech eye",
                "mech-eye",
            ],
            "IDS Imaging": ["ids", "ensenso", "ids imaging"],
            "Zivid": ["zivid"],
            "Basler": ["basler", "ace", "aca", "aca1300", "aca1920"],
            "Photoneo": ["photoneo", "phoxi", "motioncam", "motion cam", "3d scanner"],
            "Roboception": [
                "roboception",
                "rc",
                "rc_visard",
                "rc visard",
                "rc-visard",
                "rcvisard",
            ],
            "SICK": ["sick", "lms", "lrs", "lms1000", "lrs4000", "lms1104c"],
            "Slamtec": [
                "slamtec",
                "rplidar",
                "rp lidar",
                "rplidar a1",
                "rplidar a2",
                "rplidar s1",
            ],
            "VectorNav": ["vectornav", "vector nav", "vn", "vn100", "vn-100", "ahrs"],
            "Velodyne": [
                "velodyne",
                "vlp",
                "vlp16",
                "vlp-16",
                "vlp32",
                "vlp-32",
                "puck",
                "ultra puck",
            ],
            "Ouster": ["ouster", "os", "os0", "os1", "os2"],
            "smartmicro": ["smartmicro", "smart micro", "drvegrd"],
            "Hesai": [
                "hesai",
                "pandar",
                "pandarqt",
                "pandarxt",
                "pandar qt",
                "pandar xt",
            ],
        }

        # Sensor type patterns (with plurals and variations)
        self.sensor_type_patterns = {
            "Depth Camera": [
                "depth camera",
                "depth cameras",
                "depth sensor",
                "depth sensors",
                "3d camera",
                "3d cameras",
                "rgbd",
                "rgbd camera",
                "rgbd cameras",
                "depth cam",
            ],
            "RGB Camera": [
                "rgb camera",
                "rgb cameras",
                "color camera",
                "color cameras",
                "colour camera",
                "colour cameras",
                "webcam",
                "webcams",
                "rgb cam",
            ],
            "Stereo Camera": [
                "stereo camera",
                "stereo cameras",
                "stereo vision",
                "stereo cam",
                "stereo cams",
                "stereoscopic camera",
                "stereoscopic cameras",
            ],
            "Structured Light Camera": [
                "structured light camera",
                "structured light cameras",
                "structured light",
                "structured-light camera",
                "structured-light cameras",
            ],
            "LiDAR": [
                "lidar",
                "lidars",
                "laser scanner",
                "laser scanners",
                "3d laser",
                "3d lasers",
                "light detection",
                "laser sensor",
                "laser sensors",
            ],
            "Thermal Camera": [
                "thermal camera",
                "thermal cameras",
                "infrared camera",
                "infrared cameras",
                "heat sensor",
                "heat sensors",
                "thermal cam",
                "thermal cams",
                "ir camera",
                "ir cameras",
            ],
            "ToF Camera": [
                "tof camera",
                "tof cameras",
                "time of flight",
                "time-of-flight",
                "tof sensor",
                "tof sensors",
                "time of flight camera",
                "time of flight cameras",
            ],
            "IMU": [
                "imu",
                "imus",
                "ahrs",
                "inertial",
                "inertial measurement",
                "inertial measurement unit",
                "gyroscope",
                "accelerometer",
                "magnetometer",
                "orientation sensor",
                "attitude sensor",
                "motion sensor",
                "6dof",
                "9dof",
                "6 dof",
                "9 dof",
            ],
            "Radar": [
                "radar",
                "radars",
                "automotive radar",
                "mmwave",
                "mm wave",
                "77ghz",
                "79ghz",
                "24ghz",
                "77 ghz",
                "79 ghz",
                "24 ghz",
                "millimeter wave",
                "range sensor",
                "doppler radar",
                "fmcw radar",
                "short range radar",
                "long range radar",
                "srr",
                "lrr",
            ],
        }

        # ROS version patterns (with space variations)
        self.ros_patterns = {
            "ROS1": ["ros1", "ros 1", "ros-1", "kinetic", "melodic", "noetic"],
            "ROS2": [
                "ros2",
                "ros 2",
                "ros-2",
                "foxy",
                "galactic",
                "humble",
                "iron",
                "rolling",
                "jazzy",
            ],
            "ROS": ["ros compatible", "ros-compatible", "supports ros"],
        }

        # Application context keywords and platforms
        self.application_keywords = {
            "indoor": ["indoor", "inside", "interior"],
            "outdoor": [
                "outdoor",
                "outside",
                "exterior",
                "weatherproof",
                "ip65",
                "ip66",
                "ip67",
            ],
            "mobile robot": [
                "mobile robot",
                "navigation",
                "autonomous vehicle",
                "agv",
                "amr",
            ],
            "manipulation": [
                "manipulation",
                "grasping",
                "pick and place",
                "robot arm",
                "gripper",
            ],
            "precision": ["precision", "accurate", "high accuracy", "precise"],
            "warehouse": ["warehouse", "logistics", "inventory", "automation"],
            "bin picking": ["bin picking", "pick and place", "picking"],
            "navigation": [
                "navigation",
                "autonomous navigation",
                "mobile navigation",
                "slam",
                "mapping",
            ],
            "quality assurance": ["quality assurance", "quality control", "qc", "qa"],
            "inspection": ["inspection", "visual inspection", "robotic inspection"],
            "slam": ["slam", "mapping", "localization", "simultaneous localization"],
            "industrial": ["industrial", "manufacturing", "factory", "production"],
        }

        # Platform/OS patterns
        self.platform_patterns = {
            "Windows": ["windows", "win", "microsoft windows"],
            "Linux": ["linux", "ubuntu", "debian", "ros"],
            "Mac": ["mac", "macos", "osx", "apple"],
            "ARM": ["arm", "raspberry", "raspberry pi", "jetson", "nvidia jetson"],
        }

    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse natural language query into structured filters.

        Args:
            query: Natural language query string

        Returns:
            ParsedQuery with extracted filters and metadata
        """
        self.logger.debug(f"Parsing query: '{query}'")

        try:
            # Normalize query
            normalized_query = self._normalize_query(query)

            # Extract filters
            filters = []
            remaining_text = normalized_query

            # Extract numeric comparisons
            numeric_filters, remaining_text = self._extract_numeric_filters(
                remaining_text
            )
            filters.extend(numeric_filters)

            # Extract manufacturer filters
            manufacturer_filters, remaining_text = self._extract_manufacturer_filters(
                remaining_text
            )
            filters.extend(manufacturer_filters)

            # Extract sensor type filters
            type_filters, remaining_text = self._extract_sensor_type_filters(
                remaining_text
            )
            filters.extend(type_filters)

            # Extract ROS compatibility filters
            ros_filters, remaining_text = self._extract_ros_filters(remaining_text)
            filters.extend(ros_filters)

            # Extract platform/OS filters
            platform_filters, remaining_text = self._extract_platform_filters(
                remaining_text
            )
            filters.extend(platform_filters)

            # Extract semantic qualitative term filters FIRST (more specific)
            qualitative_filters, remaining_text = self._extract_qualitative_terms(
                remaining_text
            )
            filters.extend(qualitative_filters)

            # Extract application context filters SECOND (more general)
            application_filters, remaining_text = self._extract_application_filters(
                remaining_text
            )
            filters.extend(application_filters)

            # Check for generic terms and "only" keywords
            generic_filters, remaining_text = self._extract_generic_terms(
                remaining_text
            )
            filters.extend(generic_filters)

            # Extract implicit numeric values (e.g., "10.0 fps")
            implicit_filters, remaining_text = self._extract_implicit_numeric(
                remaining_text
            )
            filters.extend(implicit_filters)

            # Remaining words become text search terms
            text_terms = self._extract_text_search_terms(remaining_text)

            # Calculate overall confidence
            confidence = self._calculate_confidence(filters, text_terms, query)

            result = ParsedQuery(
                filters=filters,
                text_search_terms=text_terms,
                original_query=query,
                confidence=confidence,
            )

            self.logger.debug(
                f"Parsed into {len(filters)} filters and {len(text_terms)} text terms"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error parsing query '{query}': {e}")
            return ParsedQuery(
                filters=[],
                text_search_terms=[query.split()],  # Fallback to simple word splitting
                original_query=query,
                confidence=0.1,
                error_message=str(e),
            )

    def _normalize_query(self, query: str) -> str:
        """Normalize query for better parsing."""
        # Convert to lowercase
        normalized = query.lower().strip()

        # Check for "all" queries first
        if normalized in ["all", "all sensors", "show all", "list all", "everything"]:
            return "all_sensors"

        # Handle currency symbols - keep numbers intact
        normalized = re.sub(r"\$(\d+)", r"\1 dollar", normalized)

        # Handle common abbreviations
        normalized = re.sub(r"\bms\b", " millisecond", normalized)
        normalized = re.sub(r"\bgm?\b", " gram", normalized)  # g or gm to gram
        normalized = re.sub(r"\bkg\b", " kilogram", normalized)
        normalized = re.sub(r"\bw\b(?!\w)", " watts", normalized)  # W to watts
        normalized = re.sub(r"\bm\b(?!\w)", " meter", normalized)  # m to meter
        normalized = re.sub(r"\bcm\b", " centimeter", normalized)
        normalized = re.sub(r"\b°\b", " degree", normalized)  # degree symbol

        # Clean up whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized

    def _extract_numeric_filters(self, text: str) -> Tuple[List[ParsedFilter], str]:
        """Extract numeric comparison filters."""
        filters = []
        remaining = text

        # More specific patterns for better matching

        # Pattern 1: "under $500", "over $1000", etc.
        price_patterns = [
            (r"\b(under|below)\s+(\d+)\s*dollar", OperatorType.LESS_THAN, "price_avg"),
            (
                r"\b(over|above|more than)\s+(\d+)\s*dollar",
                OperatorType.GREATER_THAN,
                "price_avg",
            ),
        ]

        for pattern, operator, field in price_patterns:
            matches = re.finditer(pattern, remaining, re.IGNORECASE)
            for match in matches:
                number = float(match.group(2))
                filter_obj = ParsedFilter(
                    field=field, operator=operator, value=number, confidence=0.9
                )
                filters.append(filter_obj)
                remaining = remaining.replace(match.group(0), " ", 1)

        # Pattern 2: Extended patterns with all operators
        numeric_patterns = [
            # Frame rate patterns (with unit handling)
            (
                r"\b(fps|frame\s*rate|framerate)\s*"
                r"(>|above|higher\s+than|more\s+than)\s*(\d+(?:\.\d+)?)\s*"
                r"(fps|hz|hertz)?",
                OperatorType.GREATER_THAN,
                "frame_rate",
            ),
            (
                r"\b(fps|frame\s*rate|framerate)\s*"
                r"(<|below|lower\s+than|less\s+than)\s*(\d+(?:\.\d+)?)\s*"
                r"(fps|hz|hertz)?",
                OperatorType.LESS_THAN,
                "frame_rate",
            ),
            (
                r"\b(fps|frame\s*rate|framerate)\s*"
                r"(=|equals|exactly)\s*(\d+(?:\.\d+)?)\s*(fps|hz|hertz)?",
                OperatorType.EQUALS,
                "frame_rate",
            ),
            # Latency patterns (with unit handling)
            (
                r"\b(latency|delay)\s*(<|below|under)\s*(\d+(?:\.\d+)?)\s*"
                r"(ms|millisecond|milliseconds|microsecond|microseconds|"
                r"nanosecond|nanoseconds|second|seconds|sec)?",
                OperatorType.LESS_THAN,
                "latency",
            ),
            (
                r"\b(latency|delay)\s*(>|above|over)\s*(\d+(?:\.\d+)?)\s*"
                r"(ms|millisecond|milliseconds|microsecond|microseconds|"
                r"nanosecond|nanoseconds|second|seconds|sec)?",
                OperatorType.GREATER_THAN,
                "latency",
            ),
            (
                r"\b(latency|delay)\s*(=|equals|exactly)\s*(\d+(?:\.\d+)?)\s*"
                r"(ms|millisecond|milliseconds|microsecond|microseconds|"
                r"nanosecond|nanoseconds|second|seconds|sec)?",
                OperatorType.EQUALS,
                "latency",
            ),
            # Range patterns (with unit handling)
            (
                r"\b(range|distance|detection\s+range)\s*"
                r"(>|above|over)\s*(\d+(?:\.\d+)?)\s*(meter|meters|m)?",
                OperatorType.GREATER_THAN,
                "max_range",
            ),
            (
                r"\b(range|distance|detection\s+range)\s*"
                r"(<|below|under)\s*(\d+(?:\.\d+)?)\s*(meter|meters|m)?",
                OperatorType.LESS_THAN,
                "max_range",
            ),
            # Weight patterns
            (
                r"\b(weight|mass)\s*"
                r"(>|above|more\s+than|heavier\s+than)\s*(\d+(?:\.\d+)?)\s*"
                r"(gram|grams|g|kilogram|kilograms|kg)?",
                OperatorType.GREATER_THAN,
                "weight",
            ),
            (
                r"\b(weight|mass)\s*"
                r"(<|below|less\s+than|lighter\s+than)\s*(\d+(?:\.\d+)?)\s*"
                r"(gram|grams|g|kilogram|kilograms|kg)?",
                OperatorType.LESS_THAN,
                "weight",
            ),
            # Power patterns
            (
                r"\b(power|power\s+consumption|watts)\s*"
                r"(>|above|more\s+than)\s*(\d+(?:\.\d+)?)\s*(watts|w)?",
                OperatorType.GREATER_THAN,
                "power_consumption",
            ),
            (
                r"\b(power|power\s+consumption|watts)\s*"
                r"(<|below|less\s+than)\s*(\d+(?:\.\d+)?)\s*(watts|w)?",
                OperatorType.LESS_THAN,
                "power_consumption",
            ),
            # FOV patterns
            (
                r"\b(fov|field\s+of\s+view)\s*"
                r"(>|above|wider\s+than)\s*(\d+(?:\.\d+)?)\s*(degree|degrees|°)?",
                OperatorType.GREATER_THAN,
                "field_of_view_horizontal",
            ),
            (
                r"\b(fov|field\s+of\s+view)\s*"
                r"(<|below|narrower\s+than)\s*(\d+(?:\.\d+)?)\s*(degree|degrees|°)?",
                OperatorType.LESS_THAN,
                "field_of_view_horizontal",
            ),
        ]

        for pattern, operator, field in numeric_patterns:
            matches = re.finditer(pattern, remaining, re.IGNORECASE)
            for match in matches:
                # Extract number from appropriate group (usually group 3)
                number_str = (
                    match.group(3) if len(match.groups()) >= 3 else match.group(2)
                )
                number = float(number_str)

                # Handle unit conversions if needed
                if field == "weight" and len(match.groups()) >= 4:
                    unit = match.group(4) if match.group(4) else "gram"
                    if "kilogram" in unit or unit == "kg":
                        number *= 1000  # Convert kg to grams

                filter_obj = ParsedFilter(
                    field=field, operator=operator, value=number, confidence=0.9
                )
                filters.append(filter_obj)
                remaining = remaining.replace(match.group(0), " ", 1)

        # Pattern 3: "less than X", "more than X" with context
        contextual_patterns = [
            (
                r"\b(latency|delay).*?(less\s+than|under)\s+(\d+)",
                OperatorType.LESS_THAN,
                "latency",
            ),
            (
                r"\b(fps|frame\s*rate).*?(more\s+than|over)\s+(\d+)",
                OperatorType.GREATER_THAN,
                "frame_rate",
            ),
        ]

        for pattern, operator, field in contextual_patterns:
            matches = re.finditer(pattern, remaining, re.IGNORECASE)
            for match in matches:
                number = float(match.group(3))
                filter_obj = ParsedFilter(
                    field=field, operator=operator, value=number, confidence=0.8
                )
                filters.append(filter_obj)
                remaining = remaining.replace(match.group(0), " ", 1)

        # Pattern 4: "between X and Y"
        between_pattern = r"\bbetween\s+(\d+)\s*dollar\s+and\s+(\d+)\s*dollar"
        between_matches = re.finditer(between_pattern, remaining, re.IGNORECASE)

        for match in between_matches:
            min_val = float(match.group(1))
            max_val = float(match.group(2))
            filter_obj = ParsedFilter(
                field="price_avg",
                operator=OperatorType.BETWEEN,
                value=[min_val, max_val],
                confidence=0.95,
            )
            filters.append(filter_obj)
            remaining = remaining.replace(match.group(0), " ", 1)

        return filters, remaining

    def _extract_manufacturer_filters(
        self, text: str
    ) -> Tuple[List[ParsedFilter], str]:
        """Extract manufacturer filters."""
        filters = []
        remaining = text.lower()  # Ensure case-insensitive matching

        for manufacturer, aliases in self.manufacturer_aliases.items():
            for alias in aliases:
                if alias.lower() in remaining:
                    filter_obj = ParsedFilter(
                        field="manufacturer",
                        operator=OperatorType.CONTAINS,
                        value=manufacturer,  # Already properly cased in dict
                        confidence=0.9,
                    )
                    filters.append(filter_obj)
                    remaining = remaining.replace(alias.lower(), " ", 1)
                    break  # Only match once per manufacturer

        return filters, remaining

    def _extract_sensor_type_filters(self, text: str) -> Tuple[List[ParsedFilter], str]:
        """Extract sensor type filters."""
        filters = []
        remaining = text

        for sensor_type, patterns in self.sensor_type_patterns.items():
            for pattern in patterns:
                if pattern in remaining:
                    filter_obj = ParsedFilter(
                        field="sensor_type",
                        operator=OperatorType.CONTAINS,
                        value=sensor_type,  # Already properly cased in patterns dict
                        confidence=0.9,
                    )
                    filters.append(filter_obj)
                    remaining = remaining.replace(pattern, " ", 1)
                    break  # Only match once per sensor type

        return filters, remaining

    def _extract_ros_filters(self, text: str) -> Tuple[List[ParsedFilter], str]:
        """Extract ROS compatibility filters."""
        filters = []
        remaining = text

        for ros_version, patterns in self.ros_patterns.items():
            for pattern in patterns:
                if pattern in remaining:
                    filter_obj = ParsedFilter(
                        field="ros_compatibility",
                        operator=OperatorType.CONTAINS,
                        value=ros_version,
                        confidence=0.9,
                    )
                    filters.append(filter_obj)
                    remaining = remaining.replace(pattern, " ", 1)
                    break  # Only match once per ROS version

        return filters, remaining

    def _extract_platform_filters(self, text: str) -> Tuple[List[ParsedFilter], str]:
        """Extract platform/OS compatibility filters."""
        filters = []
        remaining = text

        for platform, patterns in self.platform_patterns.items():
            for pattern in patterns:
                # Check for various phrasings
                platform_phrases = [
                    f"{pattern}",
                    f"on {pattern}",
                    f"runs on {pattern}",
                    f"supports {pattern}",
                    f"{pattern} compatible",
                    f"for {pattern}",
                ]

                for phrase in platform_phrases:
                    if phrase in remaining:
                        filter_obj = ParsedFilter(
                            field="supported_platforms",
                            operator=OperatorType.CONTAINS,
                            value=platform,
                            confidence=0.85,
                        )
                        filters.append(filter_obj)
                        remaining = remaining.replace(phrase, " ", 1)
                        break  # Only match once per platform

        return filters, remaining

    def _extract_application_filters(self, text: str) -> Tuple[List[ParsedFilter], str]:
        """Extract application context and use case filters."""
        filters = []
        remaining = text

        # Map application contexts to sensor use_cases and tags
        application_mappings = {
            "bin picking": {
                "use_cases": [
                    "Bin Picking",
                    "Pick and Place",
                    "Grasping",
                    "Manipulation",
                ],
                "tags": ["manipulation", "precision", "industrial"],
            },
            "navigation": {
                "use_cases": [
                    "Navigation",
                    "SLAM",
                    "Mapping",
                    "Autonomous Navigation",
                    "Mobile Robot",
                ],
                "tags": ["navigation", "mobile", "slam", "autonomous"],
            },
            "quality assurance": {
                "use_cases": [
                    "Quality Assurance",
                    "Quality Control",
                    "Inspection",
                    "Manufacturing",
                ],
                "tags": ["industrial", "inspection", "manufacturing", "quality"],
            },
            "warehouse": {
                "use_cases": [
                    "Warehouse Automation",
                    "Logistics",
                    "Inventory",
                    "Material Handling",
                ],
                "tags": ["warehouse", "logistics", "automation", "industrial"],
            },
            "outdoor": {
                "tags": ["outdoor", "weatherproof", "rugged", "harsh environment"]
            },
            "manipulation": {
                "use_cases": [
                    "Manipulation",
                    "Grasping",
                    "Pick and Place",
                    "Robot Arm",
                ],
                "tags": ["manipulation", "precision", "gripper", "industrial"],
            },
            "inspection": {
                "use_cases": [
                    "Inspection",
                    "Quality Assurance",
                    "Robotic Inspection",
                    "Visual Inspection",
                ],
                "tags": ["inspection", "quality", "industrial", "precision"],
            },
            "slam": {
                "use_cases": ["SLAM", "Mapping", "Navigation", "Localization"],
                "tags": ["slam", "navigation", "mapping", "localization"],
            },
        }

        # Track which applications we've already processed to avoid duplicates
        processed_applications = set()

        for application, keywords in self.application_keywords.items():
            # Skip if we've already processed this application
            if application in processed_applications:
                continue

            for keyword in keywords:
                # Check for various phrasings
                application_phrases = [
                    f"{keyword}",
                    f"for {keyword}",
                    f"{keyword} application",
                    f"{keyword} use case",
                    f"{keyword} sensors",
                    f"{keyword} applications",
                ]

                for phrase in application_phrases:
                    if phrase in remaining:
                        processed_applications.add(application)

                        # Create filter for use_cases field if mapping exists
                        if application in application_mappings:
                            mapping = application_mappings[application]

                            # Create use_cases filter if mapping has use_cases
                            if "use_cases" in mapping:
                                for use_case in mapping["use_cases"]:
                                    filter_obj = ParsedFilter(
                                        field="use_cases",
                                        operator=OperatorType.CONTAINS,
                                        value=use_case,
                                        confidence=0.80,
                                    )
                                    filters.append(filter_obj)

                            # Create tags filter if mapping has tags
                            if "tags" in mapping:
                                for tag in mapping["tags"]:
                                    filter_obj = ParsedFilter(
                                        field="tags",
                                        operator=OperatorType.CONTAINS,
                                        value=tag,
                                        confidence=0.75,
                                    )
                                    filters.append(filter_obj)
                        else:
                            # Generic application filter
                            filter_obj = ParsedFilter(
                                field="use_cases",
                                operator=OperatorType.CONTAINS,
                                value=keyword,
                                confidence=0.70,
                            )
                            filters.append(filter_obj)

                        remaining = remaining.replace(phrase, " ", 1)
                        break  # Only match once per keyword

                # If we processed this application, break to next application
                if application in processed_applications:
                    break

        return filters, remaining

    def _extract_qualitative_terms(self, text: str) -> Tuple[List[ParsedFilter], str]:
        """Extract qualitative terms and map them to quantitative filters."""
        filters = []
        remaining = text

        qualitative_mappings = {
            # Precision/Accuracy terms
            "high precision": [
                (
                    "latency",
                    OperatorType.LESS_THAN,
                    50.0,
                    0.85,
                ),  # < 50ms for high precision
                ("tags", OperatorType.CONTAINS, "precision", 0.80),  # Has precision tag
            ],
            "precise": [
                ("latency", OperatorType.LESS_THAN, 100.0, 0.80),
                ("tags", OperatorType.CONTAINS, "precision", 0.75),
            ],
            "accurate": [
                ("latency", OperatorType.LESS_THAN, 100.0, 0.80),
                ("tags", OperatorType.CONTAINS, "precision", 0.75),
            ],
            "precision": [
                ("tags", OperatorType.CONTAINS, "precision", 0.85),
                ("latency", OperatorType.LESS_THAN, 100.0, 0.70),
            ],
            # Speed/Latency terms
            "low latency": [
                ("latency", OperatorType.LESS_THAN, 100.0, 0.90),
            ],
            "fast": [
                ("latency", OperatorType.LESS_THAN, 50.0, 0.85),
                ("frame_rate", OperatorType.GREATER_THAN, 30.0, 0.80),
            ],
            "real time": [
                ("latency", OperatorType.LESS_THAN, 30.0, 0.90),
                ("frame_rate", OperatorType.GREATER_THAN, 15.0, 0.85),
            ],
            "high speed": [
                ("frame_rate", OperatorType.GREATER_THAN, 60.0, 0.85),
                ("latency", OperatorType.LESS_THAN, 30.0, 0.80),
            ],
            # Frame rate terms
            "high frame rate": [
                ("frame_rate", OperatorType.GREATER_THAN, 60.0, 0.90),
            ],
            "fast frame rate": [
                ("frame_rate", OperatorType.GREATER_THAN, 30.0, 0.85),
            ],
            # Size/Weight terms
            "compact": [
                ("weight", OperatorType.LESS_THAN, 500.0, 0.80),  # < 500g
            ],
            "lightweight": [
                ("weight", OperatorType.LESS_THAN, 300.0, 0.85),  # < 300g
            ],
            "small": [
                ("weight", OperatorType.LESS_THAN, 200.0, 0.75),
            ],
            "portable": [
                ("weight", OperatorType.LESS_THAN, 1000.0, 0.70),  # < 1kg
            ],
            # Power terms
            "low power": [
                ("power_consumption", OperatorType.LESS_THAN, 10.0, 0.85),  # < 10W
            ],
            "energy efficient": [
                ("power_consumption", OperatorType.LESS_THAN, 15.0, 0.80),
            ],
            # Cost terms
            "affordable": [
                ("price_avg", OperatorType.LESS_THAN, 500.0, 0.75),  # < $500
            ],
            "budget": [
                ("price_avg", OperatorType.LESS_THAN, 300.0, 0.80),  # < $300
            ],
            "cheap": [
                ("price_avg", OperatorType.LESS_THAN, 200.0, 0.85),  # < $200
            ],
            "expensive": [
                ("price_avg", OperatorType.GREATER_THAN, 2000.0, 0.75),  # > $2000
            ],
            "premium": [
                ("price_avg", OperatorType.GREATER_THAN, 1500.0, 0.80),  # > $1500
            ],
            # Range terms
            "long range": [
                ("max_range", OperatorType.GREATER_THAN, 10.0, 0.85),  # > 10m
            ],
            "short range": [
                ("max_range", OperatorType.LESS_THAN, 2.0, 0.80),  # < 2m
            ],
            "close range": [
                ("min_range", OperatorType.LESS_THAN, 0.5, 0.75),  # < 0.5m min
                ("max_range", OperatorType.LESS_THAN, 5.0, 0.70),  # < 5m max
            ],
            # Quality terms
            "industrial grade": [
                ("tags", OperatorType.CONTAINS, "industrial", 0.90),
            ],
            "standard": [
                ("price_avg", OperatorType.GREATER_THAN, 1000.0, 0.70),
                ("tags", OperatorType.CONTAINS, "industrial", 0.75),
            ],
            "consumer grade": [
                ("price_avg", OperatorType.LESS_THAN, 800.0, 0.70),
            ],
            "commercial": [
                ("price_avg", OperatorType.BETWEEN, [800.0, 3000.0], 0.75),
            ],
        }

        # Check for qualitative terms in the text
        for term, mappings in qualitative_mappings.items():
            # Check various phrasings
            term_phrases = [
                f"{term}",
                f"{term} sensors",
                f"{term} cameras",
                f"sensors with {term}",
                f"cameras with {term}",
                f"highly {term}",
                f"very {term}",
            ]

            for phrase in term_phrases:
                if phrase in remaining:
                    # Add all filters for this qualitative term
                    for field, operator, value, confidence in mappings:
                        filter_obj = ParsedFilter(
                            field=field,
                            operator=operator,
                            value=value,
                            confidence=confidence,
                        )
                        filters.append(filter_obj)

                    remaining = remaining.replace(phrase, " ", 1)
                    break  # Only match once per term

        return filters, remaining

    def _extract_generic_terms(self, text: str) -> Tuple[List[ParsedFilter], str]:
        """Extract generic terms like 'all sensors', 'only cameras', etc."""
        filters = []
        remaining = text.strip()

        # Handle "all sensors" - should return everything
        all_sensors_patterns = [
            r"\ball_sensors\b",  # Normalized form from _normalize_query
            r"\ball\s+sensors?\b",
            r"\bevery\s+sensors?\b",
            r"\bshow\s+all\s+sensors?\b",
            r"\blist\s+all\s+sensors?\b",
        ]

        for pattern in all_sensors_patterns:
            if re.search(pattern, remaining, re.IGNORECASE):
                # Create a special "show all" filter with high confidence
                filter_obj = ParsedFilter(
                    field="__show_all__",  # Special marker field
                    operator=OperatorType.CONTAINS,
                    value="*",
                    confidence=0.9,
                )
                filters.append(filter_obj)
                remaining = re.sub(pattern, " ", remaining, flags=re.IGNORECASE).strip()
                break

        # Handle "only" keyword for specific types
        only_patterns = [
            (r"\bonly\s+(cameras?)\b", "Camera", 0.9),
            (r"\bonly\s+(depth\s+cameras?)\b", "Depth Camera", 0.9),
            (r"\bonly\s+(stereo\s+cameras?)\b", "Stereo Camera", 0.9),
            (r"\bonly\s+(rgb\s+cameras?)\b", "RGB Camera", 0.9),
            (r"\bonly\s+(lidars?)\b", "LiDAR", 0.9),
            (r"\bonly\s+(tof\s+sensors?)\b", "ToF", 0.9),
            (r"\bjust\s+(cameras?)\b", "Camera", 0.85),
            (r"\bjust\s+(lidars?)\b", "LiDAR", 0.85),
            (r"\bshow\s+me\s+(cameras?)\b", "Camera", 0.8),
            (r"\bshow\s+me\s+(lidars?)\b", "LiDAR", 0.8),
        ]

        for pattern, sensor_type, confidence in only_patterns:
            match = re.search(pattern, remaining, re.IGNORECASE)
            if match:
                filter_obj = ParsedFilter(
                    field="sensor_type",
                    operator=OperatorType.CONTAINS,
                    value=sensor_type,
                    confidence=confidence,
                )
                filters.append(filter_obj)
                remaining = re.sub(pattern, " ", remaining, flags=re.IGNORECASE).strip()
                break  # Only match first pattern to avoid duplicates

        # Handle generic type mentions (lower confidence)
        generic_patterns = [
            (r"\bcameras?\s+only\b", "Camera", 0.8),
            (r"\blidars?\s+only\b", "LiDAR", 0.8),
            (r"\b(?:all\s+)?cameras?\b", "Camera", 0.6),
            (r"\b(?:all\s+)?depth\s+cameras?\b", "Depth Camera", 0.7),
            (r"\b(?:all\s+)?stereo\s+cameras?\b", "Stereo Camera", 0.7),
            (r"\b(?:all\s+)?rgb\s+cameras?\b", "RGB Camera", 0.7),
            (r"\b(?:all\s+)?lidars?\b", "LiDAR", 0.7),
        ]

        for pattern, sensor_type, confidence in generic_patterns:
            if not any(
                f.field == "sensor_type" and sensor_type in str(f.value)
                for f in filters
            ):
                match = re.search(pattern, remaining, re.IGNORECASE)
                if match:
                    filter_obj = ParsedFilter(
                        field="sensor_type",
                        operator=OperatorType.CONTAINS,
                        value=sensor_type,
                        confidence=confidence,
                    )
                    filters.append(filter_obj)
                    # Remove the term from remaining text
                    remaining = re.sub(
                        pattern, " ", remaining, flags=re.IGNORECASE
                    ).strip()

        return filters, remaining

    def _extract_implicit_numeric(self, text: str) -> Tuple[List[ParsedFilter], str]:
        """Extract implicit numeric values (e.g., '10.0 fps' means '= 10.0 fps')."""
        filters = []
        remaining = text

        # Patterns for implicit equals (just number followed by unit)
        implicit_patterns = [
            # FPS patterns
            (r"\b(\d+(?:\.\d+)?)\s*(fps|hz)\b", "frame_rate"),
            # Power patterns
            (r"\b(\d+(?:\.\d+)?)\s*(watts|w)\b", "power_consumption"),
            # Weight patterns (handle unit conversion)
            (r"\b(\d+(?:\.\d+)?)\s*(gram|grams|g)\b", "weight", 1.0),
            (r"\b(\d+(?:\.\d+)?)\s*(kilogram|kilograms|kg)\b", "weight", 1000.0),
            # Range patterns
            (r"\b(\d+(?:\.\d+)?)\s*(meter|meters|m)\s+range\b", "max_range"),
        ]

        for pattern_info in implicit_patterns:
            if len(pattern_info) == 2:
                pattern, field = pattern_info
                multiplier = 1.0
            else:
                pattern, field, multiplier = pattern_info

            matches = re.finditer(pattern, remaining, re.IGNORECASE)
            for match in matches:
                number = float(match.group(1)) * multiplier
                filter_obj = ParsedFilter(
                    field=field,
                    operator=OperatorType.EQUALS,
                    value=number,
                    confidence=0.75,  # Lower confidence for implicit
                )
                filters.append(filter_obj)
                remaining = remaining.replace(match.group(0), " ", 1)

        return filters, remaining

    def _map_field_text(self, field_text: str) -> Optional[str]:
        """Map natural language field text to database field."""
        field_text = field_text.strip().lower()

        # Direct mapping
        if field_text in self.field_mappings:
            return self.field_mappings[field_text]

        # Partial matching for compound terms
        for term, field in self.field_mappings.items():
            if term in field_text or field_text in term:
                return field

        return None

    def _extract_text_search_terms(self, remaining_text: str) -> List[str]:
        """Extract remaining words as text search terms."""
        # Clean up the text
        cleaned = re.sub(r"\s+", " ", remaining_text.strip())

        # Remove very short words, common stop words, and generic sensor terms
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "with",
            "for",
            "to",
            "of",
            "in",
            "on",
            "at",
            "and",
            "or",
            "but",
            "that",
            "this",
            "it",
            "as",
            "by",
            "from",
        }

        # Generic terms that don't add search value (should be handled by type filters)
        generic_terms = {
            "sensor",
            "sensors",
            "camera",
            "cameras",
            "lidar",
            "lidars",
            "device",
            "devices",
            "unit",
            "units",
            "system",
            "systems",
            "equipment",
            "hardware",
            "component",
            "components",
            # Unit terms that might leak through
            "millisecond",
            "milliseconds",
            "ms",
            "second",
            "seconds",
            "sec",
            "gram",
            "grams",
            "kilogram",
            "kilograms",
            "kg",
            "g",
            "watt",
            "watts",
            "w",
            "degree",
            "degrees",
            "deg",
            "meter",
            "meters",
            "metre",
            "metres",
            "m",
            "cm",
            "mm",
            "fps",
            "hz",
            "hertz",
        }

        words = []
        for word in cleaned.split():
            word = word.strip().lower()
            if len(word) > 2 and word not in stop_words and word not in generic_terms:
                words.append(word)

        return words

    def _calculate_confidence(
        self, filters: List[ParsedFilter], text_terms: List[str], original_query: str
    ) -> float:
        """Calculate confidence score for the parsing."""
        if not filters and not text_terms:
            return 0.0

        # Base confidence from filters
        if filters:
            filter_confidence = sum(f.confidence for f in filters) / len(filters)
        else:
            filter_confidence = 0.0

        # Penalty for too many unmatched text terms
        query_words = len(original_query.split())
        unmatched_ratio = len(text_terms) / max(query_words, 1)

        if unmatched_ratio > 0.7:  # More than 70% unmatched
            confidence = filter_confidence * 0.5
        elif unmatched_ratio > 0.5:  # More than 50% unmatched
            confidence = filter_confidence * 0.7
        else:
            confidence = filter_confidence * 0.9

        return min(confidence, 1.0)

    def convert_to_pandas_filters(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        Convert parsed query to pandas DataFrame filter criteria.

        Args:
            parsed_query: ParsedQuery object

        Returns:
            Dictionary compatible with existing filter system
        """
        pandas_filters = {}

        for filter_obj in parsed_query.filters:
            field = filter_obj.field
            operator = filter_obj.operator
            value = filter_obj.value

            if operator == OperatorType.GREATER_THAN:
                pandas_filters[f"min_{field}"] = value
            elif operator == OperatorType.GREATER_EQUAL:
                pandas_filters[f"min_{field}"] = value
            elif operator == OperatorType.LESS_THAN:
                pandas_filters[f"max_{field}"] = value
            elif operator == OperatorType.LESS_EQUAL:
                pandas_filters[f"max_{field}"] = value
            elif operator == OperatorType.EQUALS:
                pandas_filters[field] = value
            elif operator == OperatorType.CONTAINS:
                if field == "manufacturer":
                    pandas_filters["manufacturer"] = value
                elif field == "sensor_type":
                    pandas_filters["sensor_type"] = value
                elif field == "ros_compatibility":
                    # Handle ROS compatibility as list
                    if "ros_compatibility" not in pandas_filters:
                        pandas_filters["ros_compatibility"] = []
                    pandas_filters["ros_compatibility"].append(value)
                else:
                    pandas_filters[f"{field}_contains"] = value
            elif operator == OperatorType.BETWEEN:
                if isinstance(value, list) and len(value) == 2:
                    pandas_filters[f"min_{field}"] = value[0]
                    pandas_filters[f"max_{field}"] = value[1]

        # Add text search terms if any
        if parsed_query.text_search_terms:
            pandas_filters["text_search"] = " ".join(parsed_query.text_search_terms)

        return pandas_filters

    def get_parsing_explanation(self, parsed_query: ParsedQuery) -> str:
        """Generate human-readable explanation of how query was parsed."""
        if parsed_query.error_message:
            return f"Error parsing query: {parsed_query.error_message}"

        if not parsed_query.filters and not parsed_query.text_search_terms:
            return "No filters detected. Using basic text search."

        explanations = []

        # Explain filters
        for filter_obj in parsed_query.filters:
            field_display = filter_obj.field.replace("_", " ").title()

            if filter_obj.operator == OperatorType.GREATER_THAN:
                explanations.append(f"{field_display} > {filter_obj.value}")
            elif filter_obj.operator == OperatorType.LESS_THAN:
                explanations.append(f"{field_display} < {filter_obj.value}")
            elif filter_obj.operator == OperatorType.CONTAINS:
                explanations.append(f"{field_display} contains '{filter_obj.value}'")
            elif filter_obj.operator == OperatorType.BETWEEN:
                explanations.append(
                    f"{field_display} between {filter_obj.value[0]} and "
                    f"{filter_obj.value[1]}"
                )
            else:
                explanations.append(
                    f"{field_display} {filter_obj.operator.value} {filter_obj.value}"
                )

        # Explain text search
        if parsed_query.text_search_terms:
            explanations.append(
                f"Text search: {', '.join(parsed_query.text_search_terms)}"
            )

        result = "Parsed as: " + "; ".join(explanations)

        # Add confidence indicator
        if parsed_query.confidence < 0.5:
            result += " (Low confidence - may use basic text search)"
        elif parsed_query.confidence < 0.8:
            result += " (Medium confidence)"

        return result


# Factory function for easy usage
def create_nlp_parser() -> NaturalLanguageQueryParser:
    """Create and return configured NLP parser instance."""
    return NaturalLanguageQueryParser()

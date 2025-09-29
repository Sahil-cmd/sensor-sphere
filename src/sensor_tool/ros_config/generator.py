"""
ROS Configuration Generator

Generates ROS launch files and parameter templates from sensor data.
"""

import logging
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import extract_numeric

logger = logging.getLogger(__name__)


class ROSConfigGenerator:
    """Generates ROS launch files and parameter templates for sensors."""

    # Sensor type to template mapping
    SENSOR_TYPE_TEMPLATES = {
        "Depth Camera": {
            "ros1": "basic_depth_camera.launch",
            "ros2": "basic_depth_camera.launch.py",
            "params": "depth_camera_params.yaml",
        },
        "Stereo Camera": {
            "ros1": "basic_stereo_camera.launch",
            "ros2": "basic_stereo_camera.launch.py",
            "params": "depth_camera_params.yaml",
        },
        "Structured Light Camera": {
            "ros1": "basic_structured_light.launch",
            "ros2": "basic_depth_camera.launch.py",  # fallback
            "params": "depth_camera_params.yaml",
        },
    }

    # Driver package mappings for known sensors
    DRIVER_MAPPINGS = {
        "intel_realsense": {
            "ros1": {
                "package": "realsense2_camera",
                "include": "$(find realsense2_camera)/launch/rs_camera.launch",
            },
            "ros2": {
                "package": "realsense2_camera",
                "executable": "realsense2_camera_node",
            },
        },
        "stereolabs_zed": {
            "ros1": {
                "package": "zed_wrapper",
                "include": "$(find zed_wrapper)/launch/zed.launch",
            },
            "ros2": {"package": "zed_wrapper", "executable": "zed_wrapper"},
        },
        "ids_ensenso": {
            "ros1": {
                "package": "ensenso_driver",
                "include": "$(find ensenso_driver)/launch/ensenso.launch",
            },
            "ros2": {"package": "ensenso_driver", "executable": "ensenso_node"},
        },
        "mech_mind": {
            "ros1": {
                "package": "mecheye_ros_interface",
                "include": "$(find mecheye_ros_interface)/launch/start_camera.launch",
            },
            "ros2": {"package": "mecheye_ros_interface", "executable": "camera_node"},
        },
        "zivid": {
            "ros1": {
                "package": "zivid_camera",
                "include": "$(find zivid_camera)/launch/zivid_camera.launch",
            },
            "ros2": {"package": "zivid_camera", "executable": "zivid_camera"},
        },
    }

    def __init__(self) -> None:
        """Initialize the ROS config generator."""
        self.templates_dir = Path(__file__).parent / "templates"
        logger.info(
            f"ROS Config Generator initialized with templates at: {self.templates_dir}"
        )

    def generate_configs(
        self,
        sensors_data: List[Dict[str, Any]],
        ros_versions: Optional[List[str]] = None,
        include_params: bool = True,
    ) -> str:
        """
        Generate ROS configuration files for selected sensors.

        Args:
            sensors_data: List of sensor dictionaries
            ros_versions: List of ROS versions to generate ('ros1', 'ros2')
            include_params: Whether to include parameter files

        Returns:
            Path to generated ZIP file containing all configs
        """
        if ros_versions is None:
            ros_versions = ["ros1", "ros2"]

        logger.info(
            f"Generating ROS configs for {len(sensors_data)} sensors, versions: {ros_versions}"
        )

        # Create temporary directory for generated files
        temp_dir = tempfile.mkdtemp(prefix="ros_configs_")
        generated_files = []

        try:
            for sensor in sensors_data:
                sensor_files = self._generate_sensor_configs(
                    sensor, ros_versions, include_params, temp_dir
                )
                generated_files.extend(sensor_files)

            # Create README with usage instructions
            readme_path = self._generate_readme(sensors_data, temp_dir)
            generated_files.append(readme_path)

            # Create ZIP archive
            zip_path = self._create_zip_archive(generated_files, temp_dir)

            logger.info(f"Generated ROS config ZIP: {zip_path}")
            return zip_path

        except Exception as e:
            logger.error(f"Failed to generate ROS configs: {e}")
            raise

    def _generate_sensor_configs(
        self,
        sensor: Dict[str, Any],
        ros_versions: List[str],
        include_params: bool,
        output_dir: str,
    ) -> List[str]:
        """Generate configuration files for a single sensor."""
        generated_files = []
        sensor_id = sensor.get("sensor_id", "unknown_sensor")
        sensor_type = sensor.get("sensor_type", "Depth Camera")

        logger.info(f"Generating configs for {sensor_id} ({sensor_type})")

        # Get template info for sensor type
        template_info = self.SENSOR_TYPE_TEMPLATES.get(
            sensor_type, self.SENSOR_TYPE_TEMPLATES["Depth Camera"]  # fallback
        )

        # Prepare template variables
        template_vars = self._prepare_template_variables(sensor)

        # Generate launch files for each ROS version
        for ros_version in ros_versions:
            if ros_version in template_info:
                launch_file = self._generate_launch_file(
                    sensor,
                    ros_version,
                    template_info[ros_version],
                    template_vars,
                    output_dir,
                )
                if launch_file:
                    generated_files.append(launch_file)

        # Generate parameter file
        if include_params and "params" in template_info:
            param_file = self._generate_param_file(
                sensor, template_info["params"], template_vars, output_dir
            )
            if param_file:
                generated_files.append(param_file)

        return generated_files

    def _prepare_template_variables(self, sensor: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variables for template substitution."""
        sensor_id = sensor.get("sensor_id", "unknown_sensor")
        manufacturer = sensor.get("manufacturer", "Unknown")
        model = sensor.get("model", "Unknown")

        # Extract resolution data safely
        resolution = sensor.get("resolution", {})
        rgb_res = {}
        depth_res = {}

        if isinstance(resolution, dict):
            rgb_res = (
                resolution.get("rgb", {})
                if isinstance(resolution.get("rgb"), dict)
                else {}
            )
            depth_res = (
                resolution.get("depth", {})
                if isinstance(resolution.get("depth"), dict)
                else {}
            )

        # Default values and safe extraction
        rgb_width = 640
        rgb_height = 480
        depth_width = 640
        depth_height = 480

        if rgb_res and isinstance(rgb_res, dict):
            rgb_width = rgb_res.get("width", 640)
            rgb_height = rgb_res.get("height", 480)

        if depth_res and isinstance(depth_res, dict):
            depth_width = depth_res.get("width", rgb_width)
            depth_height = depth_res.get("height", rgb_height)

        # Extract other parameters
        frame_rate = extract_numeric(sensor.get("frame_rate", 30))
        min_range = extract_numeric(sensor.get("min_range", 0.1))
        max_range = extract_numeric(sensor.get("max_range", 10.0))

        # Determine connection type
        comm_interface = sensor.get("communication_interface", "")
        connection_type = "usb"
        if "ethernet" in str(comm_interface).lower():
            connection_type = "ethernet"
        elif "usb" in str(comm_interface).lower():
            connection_type = "usb"

        # Get driver information
        driver_info = self._get_driver_info(sensor)

        variables = {
            "sensor_id": sensor_id,
            "manufacturer": manufacturer,
            "model": model,
            "sensor_description": f"{manufacturer} {model} ({sensor_id})",
            "rgb_width": str(rgb_width),
            "rgb_height": str(rgb_height),
            "depth_width": str(depth_width),
            "depth_height": str(depth_height),
            "frame_rate": str(int(frame_rate)),
            "min_range": str(min_range),
            "max_range": str(max_range),
            "connection_type": connection_type,
            "driver_package": driver_info.get("package", "unknown_driver"),
            "driver_executable": driver_info.get("executable", "unknown_node"),
            "driver_include_file": driver_info.get(
                "include", "$(find unknown_driver)/launch/sensor.launch"
            ),
        }

        logger.debug(f"Template variables for {sensor_id}: {variables}")
        return variables

    def _get_driver_info(self, sensor: Dict[str, Any]) -> Dict[str, str]:
        """Get driver package information for sensor."""
        sensor_id = sensor.get("sensor_id", "").lower()
        manufacturer = sensor.get("manufacturer", "").lower()

        # Try to match based on sensor ID or manufacturer
        for key, mapping in self.DRIVER_MAPPINGS.items():
            if (
                key in sensor_id
                or key.split("_")[0] in manufacturer
                or any(part in manufacturer for part in key.split("_"))
            ):
                return mapping.get("ros1", {})  # Default to ros1 mapping

        # Generic fallback
        return {
            "package": f'{manufacturer.replace(" ", "_").lower()}_driver',
            "include": f'$(find {manufacturer.replace(" ", "_").lower()}_driver)/launch/sensor.launch',
            "executable": "camera_node",
        }

    def _generate_launch_file(
        self,
        sensor: Dict[str, Any],
        ros_version: str,
        template_name: str,
        template_vars: Dict[str, Any],
        output_dir: str,
    ) -> Optional[str]:
        """Generate launch file from template."""
        try:
            # Load template
            template_path = self.templates_dir / ros_version / template_name
            if not template_path.exists():
                logger.warning(f"Template not found: {template_path}")
                return None

            with open(template_path, "r") as f:
                template_content = f.read()

            # Update driver info for specific ROS version
            if ros_version in ["ros1", "ros2"]:
                driver_key = self._get_driver_key(sensor)
                if driver_key and driver_key in self.DRIVER_MAPPINGS:
                    driver_mapping = self.DRIVER_MAPPINGS[driver_key].get(
                        ros_version, {}
                    )
                    template_vars.update(
                        {
                            "driver_package": driver_mapping.get(
                                "package", template_vars["driver_package"]
                            ),
                            "driver_executable": driver_mapping.get(
                                "executable", template_vars["driver_executable"]
                            ),
                            "driver_include_file": driver_mapping.get(
                                "include", template_vars["driver_include_file"]
                            ),
                        }
                    )

            # Substitute template variables
            generated_content = template_content.format(**template_vars)

            # Create output file
            sensor_id = sensor.get("sensor_id", "unknown_sensor")
            if ros_version == "ros1":
                output_filename = f"{sensor_id}.launch"
            else:
                output_filename = f"{sensor_id}.launch.py"

            output_path = os.path.join(output_dir, ros_version, output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w") as f:
                f.write(generated_content)

            logger.info(f"Generated {ros_version} launch file: {output_path}")
            return output_path

        except Exception as e:
            logger.error(
                f"Failed to generate launch file for {sensor.get('sensor_id')}: {e}"
            )
            return None

    def _get_driver_key(self, sensor: Dict[str, Any]) -> Optional[str]:
        """Get the driver key for sensor from DRIVER_MAPPINGS."""
        sensor_id = sensor.get("sensor_id", "").lower()
        manufacturer = sensor.get("manufacturer", "").lower()

        for key in self.DRIVER_MAPPINGS.keys():
            if (
                key in sensor_id
                or key.split("_")[0] in manufacturer
                or any(part in manufacturer for part in key.split("_"))
            ):
                return key
        return None

    def _generate_param_file(
        self,
        sensor: Dict[str, Any],
        template_name: str,
        template_vars: Dict[str, Any],
        output_dir: str,
    ) -> Optional[str]:
        """Generate parameter file from template."""
        try:
            # Load template
            template_path = self.templates_dir / "parameters" / template_name
            if not template_path.exists():
                logger.warning(f"Parameter template not found: {template_path}")
                return None

            with open(template_path, "r") as f:
                template_content = f.read()

            # Substitute template variables
            generated_content = template_content.format(**template_vars)

            # Create output file
            sensor_id = sensor.get("sensor_id", "unknown_sensor")
            output_filename = f"{sensor_id}_params.yaml"
            output_path = os.path.join(output_dir, "parameters", output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w") as f:
                f.write(generated_content)

            logger.info(f"Generated parameter file: {output_path}")
            return output_path

        except Exception as e:
            logger.error(
                f"Failed to generate parameter file for {sensor.get('sensor_id')}: {e}"
            )
            return None

    def _generate_readme(
        self, sensors_data: List[Dict[str, Any]], output_dir: str
    ) -> str:
        """Generate README with usage instructions."""
        sensor_list = [s.get("sensor_id", "unknown") for s in sensors_data]

        readme_content = f"""# ROS Configuration Files
Generated by Sensor Comparison Tool
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Included Sensors
{chr(10).join(f'- {sensor_id}' for sensor_id in sensor_list)}

## Directory Structure
- `ros1/`: ROS1 launch files (.launch)
- `ros2/`: ROS2 launch files (.launch.py)
- `parameters/`: Parameter configuration files (.yaml)

## Usage Instructions

### ROS1 Launch Files
```bash
# Basic usage
roslaunch ros1/<sensor_id>.launch

# With custom parameters
roslaunch ros1/<sensor_id>.launch serial_no:=YOUR_SERIAL fps:=30

# With parameter file
roslaunch ros1/<sensor_id>.launch load_params:=true param_file:=parameters/<sensor_id>_params.yaml
```

### ROS2 Launch Files
```bash
# Basic usage
ros2 launch ros2/<sensor_id>.launch.py

# With custom parameters
ros2 launch ros2/<sensor_id>.launch.py serial_no:=YOUR_SERIAL fps:=30

# Load parameter file
ros2 param load /<sensor_id> parameters/<sensor_id>_params.yaml
```

## Common Parameters
- `serial_no`: Sensor serial number (leave empty for auto-detection)
- `fps`: Frame rate (frames per second)
- `enable_pointcloud`: Enable point cloud generation
- `publish_tf`: Publish TF transforms
- `tf_prefix`: Prefix for TF frames

## Notes
1. **Driver Installation**: Ensure the appropriate ROS drivers are installed for your sensors
2. **Serial Numbers**: Update serial numbers in launch files or parameter files for specific sensor identification
3. **Calibration**: Update calibration parameters in parameter files if you have custom calibration
4. **Multi-sensor Setup**: For multiple sensors, use different namespaces and TF prefixes

## Driver Repositories
Make sure to install the required ROS drivers:

{chr(10).join(self._get_driver_installation_notes(sensors_data))}

## Support
- Check sensor datasheets for specific parameter ranges
- Refer to driver documentation for advanced configuration
- Update parameter files based on your specific use case requirements

Generated by Sensor Comparison Tool - https://github.com/your-repo/sensor-comparison-tool
"""

        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)

        return readme_path

    def _get_driver_installation_notes(
        self, sensors_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate driver installation notes for sensors."""
        notes = []
        seen_drivers = set()

        for sensor in sensors_data:
            driver_key = self._get_driver_key(sensor)
            if driver_key and driver_key not in seen_drivers:
                seen_drivers.add(driver_key)

                if driver_key == "intel_realsense":
                    notes.append(
                        "- Intel RealSense: `sudo apt install ros-$ROS_DISTRO-realsense2-camera`"
                    )
                elif driver_key == "stereolabs_zed":
                    notes.append(
                        "- Stereolabs ZED: Install ZED SDK and `ros-$ROS_DISTRO-zed-wrapper`"
                    )
                elif driver_key == "ids_ensenso":
                    notes.append(
                        "- IDS Ensenso: Install Ensenso SDK and compile ensenso_driver from source"
                    )
                elif driver_key == "mech_mind":
                    notes.append(
                        "- Mech-Mind: Clone and build mecheye_ros_interface from GitHub"
                    )
                elif driver_key == "zivid":
                    notes.append(
                        "- Zivid: Install Zivid SDK and compile zivid_camera from source"
                    )

        return (
            notes
            if notes
            else [
                "- Please refer to manufacturer documentation for driver installation"
            ]
        )

    def _create_zip_archive(self, file_paths: List[str], base_dir: str) -> str:
        """Create ZIP archive of generated files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"ros_configs_{timestamp}.zip"
        zip_path = os.path.join(base_dir, zip_filename)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    # Create archive path relative to base directory
                    archive_path = os.path.relpath(file_path, base_dir)
                    zipf.write(file_path, archive_path)

        logger.info(f"Created ZIP archive: {zip_path}")
        return zip_path

    def generate_urdf_files(
        self, sensors_data: List[Dict[str, Any]], include_meshes: bool = False
    ) -> str:
        """Generate URDF files for sensors.

        Args:
            sensors_data: List of sensor dictionaries
            include_meshes: Whether to include mesh references (requires mesh files)

        Returns:
            Path to generated ZIP file containing URDF files
        """
        logger.info(f"Generating URDF files for {len(sensors_data)} sensors")

        # Create temporary directory for generated files
        temp_dir = tempfile.mkdtemp(prefix="urdf_files_")
        generated_files = []

        try:
            for sensor in sensors_data:
                urdf_files = self._generate_sensor_urdf(
                    sensor, include_meshes, temp_dir
                )
                generated_files.extend(urdf_files)

            # Generate combined robot URDF if multiple sensors
            if len(sensors_data) > 1:
                combined_urdf = self._generate_combined_urdf(
                    sensors_data, include_meshes, temp_dir
                )
                if combined_urdf:
                    generated_files.append(combined_urdf)

            # Create README for URDF usage
            readme_path = self._generate_urdf_readme(sensors_data, temp_dir)
            generated_files.append(readme_path)

            # Create ZIP archive
            zip_path = self._create_zip_archive(generated_files, temp_dir)

            logger.info(f"Generated URDF ZIP: {zip_path}")
            return zip_path

        except Exception as e:
            logger.error(f"Failed to generate URDF files: {e}")
            raise

    def _generate_sensor_urdf(
        self, sensor: Dict[str, Any], include_meshes: bool, output_dir: str
    ) -> List[str]:
        """Generate URDF file for a single sensor."""
        sensor_id = sensor.get("sensor_id", "unknown_sensor")

        # Create filename
        urdf_filename = f"{sensor_id.lower()}.urdf"
        urdf_path = os.path.join(output_dir, urdf_filename)

        # Generate URDF content based on sensor type
        urdf_content = self._generate_urdf_content(sensor, include_meshes)

        # Write URDF file
        with open(urdf_path, "w") as f:
            f.write(urdf_content)

        logger.info(f"Generated URDF: {urdf_path}")
        return [urdf_path]

    def _generate_urdf_content(
        self, sensor: Dict[str, Any], include_meshes: bool
    ) -> str:
        """Generate URDF XML content for a sensor."""
        sensor_id = sensor.get("sensor_id", "unknown_sensor").lower()
        manufacturer = sensor.get("manufacturer", "Unknown")
        model = sensor.get("model", "Unknown")
        sensor_type = sensor.get("sensor_type", "Unknown")

        # Extract physical dimensions
        size = sensor.get("size", {})
        length = (
            extract_numeric(size.get("length", 0.1)) / 1000
            if size.get("unit") == "mm"
            else extract_numeric(size.get("length", 0.1))
        )
        width = (
            extract_numeric(size.get("width", 0.05)) / 1000
            if size.get("unit") == "mm"
            else extract_numeric(size.get("width", 0.05))
        )
        height = (
            extract_numeric(size.get("height", 0.03)) / 1000
            if size.get("unit") == "mm"
            else extract_numeric(size.get("height", 0.03))
        )

        weight = extract_numeric(sensor.get("weight", 0.1))
        if sensor.get("weight_unit") == "g":
            weight = weight / 1000  # Convert grams to kg

        # Calculate inertia (simple box approximation)
        mass = weight if weight > 0 else 0.1
        ixx = (mass / 12) * (width**2 + height**2)
        iyy = (mass / 12) * (length**2 + height**2)
        izz = (mass / 12) * (length**2 + width**2)

        # Generate visual and collision geometry
        if include_meshes:
            visual_geometry = f"""<mesh filename="package://{sensor_id}_description/meshes/{sensor_id}.dae" scale="1 1 1"/>"""
            collision_geometry = f"""<mesh filename="package://{sensor_id}_description/meshes/{sensor_id}_collision.dae" scale="1 1 1"/>"""
        else:
            visual_geometry = f"""<box size="{length:.4f} {width:.4f} {height:.4f}"/>"""
            collision_geometry = (
                f"""<box size="{length:.4f} {width:.4f} {height:.4f}"/>"""
            )

        # Sensor-specific frames and properties
        sensor_frames = self._generate_sensor_frames(sensor)

        urdf_template = f"""<?xml version="1.0"?>
<robot name="{sensor_id}" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- {manufacturer} {model} ({sensor_type}) -->
  <!-- Generated by SensorSphere on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->

  <!-- Base link -->
  <link name="{sensor_id}_base_link"/>

  <!-- Main sensor body -->
  <link name="{sensor_id}_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        {visual_geometry}
      </geometry>
      <material name="{sensor_id}_material">
        <color rgba="0.2 0.2 0.2 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        {collision_geometry}
      </geometry>
    </collision>

    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.4f}"/>
      <inertia ixx="{ixx:.8f}" ixy="0.0" ixz="0.0"
               iyy="{iyy:.8f}" iyz="0.0"
               izz="{izz:.8f}"/>
    </inertial>
  </link>

  <!-- Joint connecting base to sensor body -->
  <joint name="{sensor_id}_joint" type="fixed">
    <parent link="{sensor_id}_base_link"/>
    <child link="{sensor_id}_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

{sensor_frames}

  <!-- Gazebo properties -->
  <gazebo reference="{sensor_id}_link">
    <material>Gazebo/DarkGrey</material>
  </gazebo>

</robot>
"""

        return urdf_template

    def _generate_sensor_frames(self, sensor: Dict[str, Any]) -> str:
        """Generate sensor-specific frames based on sensor type."""
        sensor_id = sensor.get("sensor_id", "unknown_sensor").lower()
        sensor_type = sensor.get("sensor_type", "Unknown")

        frames = []

        if sensor_type in {
            "RGB Camera",
            "Depth Camera",
            "Stereo Camera",
            "Infrared Camera",
            "Thermal Camera",
            "Time-of-Flight Camera",
            "Structured Light Camera",
        }:
            # Camera-specific frames
            frames.append(
                f"""  <!-- Camera optical frame (ROS camera convention: x=right, y=down, z=forward) -->
  <link name="{sensor_id}_optical_frame"/>

  <joint name="{sensor_id}_optical_joint" type="fixed">
    <parent link="{sensor_id}_link"/>
    <child link="{sensor_id}_optical_frame"/>
    <origin xyz="0 0 0" rpy="-1.5708 0 -1.5708"/>
  </joint>"""
            )

            if sensor_type in {
                "Depth Camera",
                "Stereo Camera",
                "Time-of-Flight Camera",
                "Structured Light Camera",
            }:
                # Depth camera frames
                frames.append(
                    f"""
  <!-- Depth optical frame -->
  <link name="{sensor_id}_depth_optical_frame"/>

  <joint name="{sensor_id}_depth_optical_joint" type="fixed">
    <parent link="{sensor_id}_link"/>
    <child link="{sensor_id}_depth_optical_frame"/>
    <origin xyz="0 0 0" rpy="-1.5708 0 -1.5708"/>
  </joint>"""
                )

        elif sensor_type == "LiDAR":
            # LiDAR-specific frame (typically rotates around Z-axis)
            frames.append(
                f"""  <!-- LiDAR scanning frame -->
  <link name="{sensor_id}_scan_frame"/>

  <joint name="{sensor_id}_scan_joint" type="fixed">
    <parent link="{sensor_id}_link"/>
    <child link="{sensor_id}_scan_frame"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>"""
            )

        elif sensor_type in {"IMU", "Gyroscope", "Accelerometer", "Magnetometer"}:
            # IMU frame (aligned with sensor body)
            frames.append(
                f"""  <!-- IMU frame -->
  <link name="{sensor_id}_imu_frame"/>

  <joint name="{sensor_id}_imu_joint" type="fixed">
    <parent link="{sensor_id}_link"/>
    <child link="{sensor_id}_imu_frame"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>"""
            )

        return "\n".join(frames) if frames else ""

    def _generate_combined_urdf(
        self, sensors_data: List[Dict[str, Any]], include_meshes: bool, output_dir: str
    ) -> Optional[str]:
        """Generate combined URDF with multiple sensors on a robot platform."""
        urdf_filename = "multi_sensor_robot.urdf"
        urdf_path = os.path.join(output_dir, urdf_filename)

        robot_name = "multi_sensor_platform"

        # Start URDF with robot platform base
        urdf_content = f"""<?xml version="1.0"?>
<robot name="{robot_name}" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Multi-sensor robot platform -->
  <!-- Generated by SensorSphere on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->

  <!-- Robot base link -->
  <link name="base_link"/>

  <!-- Robot platform (simple box) -->
  <link name="platform_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.05"/>
      </geometry>
      <material name="platform_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.05"/>
      </geometry>
    </collision>

    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0"
               iyy="0.02" iyz="0.0"
               izz="0.025"/>
    </inertial>
  </link>

  <joint name="base_to_platform" type="fixed">
    <parent link="base_link"/>
    <child link="platform_link"/>
    <origin xyz="0 0 0.025" rpy="0 0 0"/>
  </joint>

"""

        # Add each sensor to the platform
        for i, sensor in enumerate(sensors_data):
            # Position sensors around the platform
            x_offset = 0.1 * (i % 3 - 1)  # -0.1, 0, 0.1
            y_offset = 0.08 * ((i // 3) % 2 - 0.5)  # -0.04, 0.04
            z_offset = 0.03  # Above platform

            # Generate individual sensor URDF content (simplified for inclusion)
            sensor_urdf_snippet = self._generate_sensor_urdf_snippet(
                sensor, x_offset, y_offset, z_offset
            )
            urdf_content += sensor_urdf_snippet

        urdf_content += "</robot>\n"

        # Write combined URDF
        with open(urdf_path, "w") as f:
            f.write(urdf_content)

        logger.info(f"Generated combined URDF: {urdf_path}")
        return urdf_path

    def _generate_sensor_urdf_snippet(
        self, sensor: Dict[str, Any], x_offset: float, y_offset: float, z_offset: float
    ) -> str:
        """Generate URDF snippet for a sensor to be included in combined URDF."""
        sensor_id = sensor.get("sensor_id", "unknown_sensor").lower()
        sensor_type = sensor.get("sensor_type", "Unknown")

        # Extract dimensions
        size = sensor.get("size", {})
        length = (
            extract_numeric(size.get("length", 0.1)) / 1000
            if size.get("unit") == "mm"
            else extract_numeric(size.get("length", 0.1))
        )
        width = (
            extract_numeric(size.get("width", 0.05)) / 1000
            if size.get("unit") == "mm"
            else extract_numeric(size.get("width", 0.05))
        )
        height = (
            extract_numeric(size.get("height", 0.03)) / 1000
            if size.get("unit") == "mm"
            else extract_numeric(size.get("height", 0.03))
        )

        weight = extract_numeric(sensor.get("weight", 0.1))
        if sensor.get("weight_unit") == "g":
            weight = weight / 1000

        mass = weight if weight > 0 else 0.1

        snippet = f"""  <!-- {sensor_id} sensor -->
  <link name="{sensor_id}_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{length:.4f} {width:.4f} {height:.4f}"/>
      </geometry>
      <material name="{sensor_id}_material">
        <color rgba="0.2 0.2 0.8 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{length:.4f} {width:.4f} {height:.4f}"/>
      </geometry>
    </collision>

    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.4f}"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0"
               iyy="0.001" iyz="0.0"
               izz="0.001"/>
    </inertial>
  </link>

  <joint name="platform_to_{sensor_id}" type="fixed">
    <parent link="platform_link"/>
    <child link="{sensor_id}_link"/>
    <origin xyz="{x_offset:.3f} {y_offset:.3f} {z_offset:.3f}" rpy="0 0 0"/>
  </joint>

"""

        # Add sensor-specific frames
        if sensor_type in {"RGB Camera", "Depth Camera", "Stereo Camera"}:
            snippet += f"""  <link name="{sensor_id}_optical_frame"/>
  <joint name="{sensor_id}_optical_joint" type="fixed">
    <parent link="{sensor_id}_link"/>
    <child link="{sensor_id}_optical_frame"/>
    <origin xyz="0 0 0" rpy="-1.5708 0 -1.5708"/>
  </joint>

"""

        return snippet

    def _generate_urdf_readme(
        self, sensors_data: List[Dict[str, Any]], output_dir: str
    ) -> str:
        """Generate README for URDF files."""
        readme_path = os.path.join(output_dir, "README_URDF.md")

        sensor_list = "\n".join(
            [
                f"- {sensor.get('manufacturer', 'Unknown')} {sensor.get('model', 'Unknown')} ({sensor.get('sensor_id', 'unknown')})"
                for sensor in sensors_data
            ]
        )

        readme_content = f"""# URDF Files for Robotics Sensors

Generated by SensorSphere on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Sensors Included

{sensor_list}

## Files Description

- Individual sensor URDF files: `<sensor_id>.urdf`
- Combined robot URDF (if multiple sensors): `multi_sensor_robot.urdf`

## Usage

### Individual Sensor URDFs

Each sensor has its own URDF file with:
- Physical properties (dimensions, mass, inertia)
- Visual and collision geometry
- Sensor-specific frames (optical, depth, etc.)
- Gazebo properties

### ROS Integration

1. **Copy URDF to your robot description package:**
   ```bash
   cp *.urdf ~/catkin_ws/src/your_robot_description/urdf/
   ```

2. **Include in your main robot URDF:**
   ```xml
   <xacro:include filename="$(find your_package)/urdf/sensor_name.urdf"/>
   ```

3. **Launch with robot_state_publisher:**
   ```xml
   <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher">
     <param name="robot_description" command="$(find xacro)/xacro --inorder $(find your_package)/urdf/robot.urdf.xacro"/>
   </node>
   ```

### Frame Conventions

- **Camera sensors:** Follow ROS camera convention (x=right, y=down, z=forward)
- **LiDAR sensors:** Scanning frame aligned with sensor body
- **IMU sensors:** Frame aligned with sensor body

### Customization

- Modify joint origins to position sensors correctly on your robot
- Adjust visual materials and colors as needed
- Add mesh files for more accurate visual representation
- Update mass and inertia properties based on actual measurements

### Gazebo Simulation

These URDFs include basic Gazebo properties. For full simulation:

1. Add appropriate Gazebo plugins for sensor functionality
2. Configure sensor-specific parameters
3. Ensure proper collision and physics properties

## Notes

- Dimensions and mass are based on manufacturer specifications
- Inertia calculations use simple box approximations
- Visual geometry uses boxes unless mesh files are available
- All measurements are in SI units (meters, kilograms)

## Support

For issues or questions about these URDF files, please refer to:
- ROS URDF documentation: http://wiki.ros.org/urdf
- Sensor manufacturer documentation for exact specifications
- SensorSphere GitHub repository for tool-specific issues
"""

        with open(readme_path, "w") as f:
            f.write(readme_content)

        return readme_path

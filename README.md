# **SensorSphere: Streamlining Sensor Selection**

## **Robot Sensor Hub and Selection Engine**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![ROS Compatible](https://img.shields.io/badge/ROS-1%20%26%202-brightgreen.svg)](https://www.ros.org/)

SensorSphere is a desktop application designed to streamline sensor selection and comparison workflows for robotics engineers and researchers. It provides an intuitive interface for evaluating sensors across multiple dimensions with comparative analysis and export capabilities.


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Sensor Database](#sensor-database)
- [Usage Guide](#usage-guide)
- [Technical Architecture](#technical-architecture)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

### Core Capabilities

- **Sensor Database**: 30+ validated sensors across cameras, LiDAR, and other sensor types (more to be added soon)
- **Advanced Filtering**: Multi-criteria filtering by price, ROS compatibility, technical specifications
- **Interactive Comparisons**: Side-by-side analysis with sortable tables and visual charts
- **Multi-Dimensional Analysis**: Radar charts for performance visualization across key attributes
- **Professional Export**: PDF reports, CSV data export, and customizable chart generation

### ROS Integration

- **ROS1/ROS2 Compatibility**: Generate launch files and parameter templates
- **Driver Validation**: Links to ROS packages and drivers
- **Configuration Templates**: Auto-generated configs for common robotics workflows

### User Experience

- **Modern GUI**: Professional PySide6/Qt interface with dockable panels
- **Theme Support**: Dark/light mode with system integration
- **Responsive Design**: Adaptive layouts for different screen sizes
- **Offline Operation**: No internet required

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Linux (Ubuntu 22.04+)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Sahil-cmd/sensor-sphere.git
   cd sensor-sphere
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv .test-venv
   source .test-venv/bin/activate
   ```

3. **Install SensorSphere**
   ```bash
   pip install -e .
   ```

## Quick Start

### Launch Application

```bash
# Activate virtual environment
source .test-venv/bin/activate

# Launch GUI - after installation, launch SensorSphere using any of these intuitive commands:
sensorsphere        # Main command
sensor-sphere       # Alternative
sphere              # Quick shortcut
```


### Basic Workflow

1. **Browse Sensors**: Use the Filter Panel to narrow down sensor options
2. **Compare**: Select multiple sensors in the comparison table
3. **Analyze**: View detailed specifications in the Sensor Details panel
4. **Visualize**: Generate bar charts and radar plots in the Chart Panel
5. **Export**: Create PDF reports or export data for further analysis

### Example: Finding a Depth Camera for Mobile Robot

1. Filter by sensor type: "Depth Camera"
2. Set price range: $200-$800
3. Filter by ROS compatibility: "ROS2"
4. Compare results
5. Export comparison report

## Sensor Database

### Data Sources

All sensor specifications are sourced from:
- Manufacturer datasheets
- Verified ROS driver documentation
- Public technical specifications

### Data Quality

- **Validation**: All entries validated against YAML schema
- **Consistency**: Standardized units and measurement criteria
- **Completeness**: Minimum required fields enforced

## Usage Guide

### Interface Overview

**Main Window Components:**
- **Filter Panel** (Left): Search and filter controls
- **Comparison Table** (Center): Sensor list with sortable columns
- **Sensor Details** (Right): Detailed specifications for selected sensor
- **Chart Panel** (Bottom): Visualization and analysis tools

### Advanced Features

#### Multi-Attribute Analysis
- Select sensors and attributes for radar chart comparison
- Attribute selection based on sensor types

#### Professional Export
- Customizable PDF reports with executive summaries
- Technical analysis with robotics-specific insights
- Chart export in multiple formats

#### ROS Configuration Generation
- Launch file templates for ROS1 and ROS2
- Parameter files with documented settings

## Technical Architecture

### Core Components

```
src/sensor_tool/
├── gui_qt/                 # PySide6 GUI application
│   ├── main_window.py     # Main application window
│   ├── widgets/           # UI components
│   ├── utils/             # Theme and font management
│   └── models/            # Qt data adapters
├── models/                # Data models and validation
│   ├── sensor_v2.py      # Enhanced Pydantic models
│   └── repository.py     # Data access layer
├── data_loader.py         # YAML data loading
├── validate_sensors.py   # Schema validation
└── utils/                 # Utility functions
```

### Data Architecture

- **Storage**: YAML files with hierarchical organization
- **Validation**: Yamale schema with comprehensive type checking
- **Models**: Pydantic v2 for runtime validation and serialization
- **Caching**: Smart caching for improved performance

### Dependencies

**Core Dependencies:**
- PySide6: GUI framework
- Pandas: Data manipulation
- Matplotlib: Chart generation
- Plotly: Interactive visualizations
- Pydantic: Data validation
- PyYAML: Configuration parsing

## Contributing

We welcome contributions from the robotics community! Here's how to get involved:

1. **Fork the repository** on GitHub
2. **Create a feature branch** (`git checkout -b feature/your-feature`)
3. **Make your changes** - add sensors, fix bugs, implement features
4. **Test your changes** with the existing sensor database
5. **Commit with clear messages** (`git commit -m "feat: add sensor XYZ"`)
6. **Push your branch** (`git push origin feature/your-feature`)
7. **Open a pull request** with detailed description

## Reporting Issues

Found a bug or have a feature request? 

- **Via GitHub**: Visit [sensorsphere/issues](https://github.com/Sahil-cmd/sensor-sphere/issues)
- **Via GUI**: Use Help → "Report Issue on GitHub" in SensorSphere
- **Include**: Steps to reproduce, expected vs actual behavior, system info

## Troubleshooting

### Common Issues

**GUI doesn't start:**
```bash
# Check PySide6 installation
pip install --upgrade PySide6==6.7.3

# Verify display settings
echo $DISPLAY  # Linux
```
```bash
# Check file permissions
ls -la sensors/
```

**Chart generation fails:**
```bash
# Install missing dependencies
pip install matplotlib plotly kaleido

# Check memory usage
free -h  # Linux
```

## Development Setup

### For Contributors

```bash
# Clone repository
git clone https://github.com/Sahil-cmd/sensor-sphere.git
cd sensorsphere

# Setup development environment
python3 -m venv .test-venv
source .test-venv/bin/activate

# Install with development dependencies
pip install -r requirements.txt
pip install black flake8 pytest pytest-qt

# Format code
black src/
flake8 src/
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Attribution

Sensor specifications are sourced from publicly available manufacturer documentation. All trademarks and product names are property of their respective owners.

---


**Built by robotics engineers, for robotics engineers.**


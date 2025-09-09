"""
PySide6 GUI Implementation for SensorSphere

This package contains the PySide6-based graphical user interface implementation
for the robot sensor hub and selection engine.

Modules:
    main_window: Main application window with docking panels
    widgets: Custom Qt widgets for sensor comparison
    dialogs: Dialog windows for various operations
    models: Data models and Qt model/view classes
    utils: Qt-specific utility functions and helpers
"""

from .main_window import SensorComparisonMainWindow

__all__ = ["SensorComparisonMainWindow"]

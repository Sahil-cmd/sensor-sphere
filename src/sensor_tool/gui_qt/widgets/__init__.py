"""
Qt Widgets Package for Sensor Comparison Tool

Contains custom PySide6 widgets for the sensor comparison interface.
"""

from .comparison_table import SensorComparisonTable
from .detail_panel import SensorDetailWidget
from .filter_panel import AdvancedFilterWidget

__all__ = ["AdvancedFilterWidget", "SensorComparisonTable", "SensorDetailWidget"]

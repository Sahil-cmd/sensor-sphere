"""
Font Management System for Modern Qt Application

This module provides a centralized font management system with dynamic scaling
based on window size and DPI. It ensures consistent typography throughout
the application and adapts to different screen sizes and resolutions.
"""

import logging
import math
from typing import Dict, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


class FontManager(QObject):
    """
    Centralized font management with dynamic scaling capabilities.

    Features:
    - Consistent font hierarchy (H1-H6, Body, Caption, etc.)
    - Dynamic scaling based on window size
    - DPI awareness
    - Modern typography standards

    Signals:
        fonts_updated: Emitted when font scaling changes
    """

    fonts_updated = Signal()

    # Base font sizes (at standard 1920x1080 resolution) - Modern hierarchy
    BASE_SIZES = {
        "h1": 20,  # Main titles
        "h2": 18,  # Section headers
        "h3": 15,  # Subsection headers/dock titles (reduced from 16)
        "h4": 16,  # Widget group titles (increased from 14)
        "h5": 14,  # Secondary labels (increased from 13)
        "h6": 13,  # Small labels (increased from 12)
        "body": 14,  # Regular text (increased from 12)
        "caption": 12,  # Fine print (increased from 11)
        "tiny": 11,  # Very small text (increased from 10)
        "menu": 13,  # Menu bar text (increased from 12)
        "toolbar": 14,  # Toolbar buttons (increased from 13)
        "empty_state": 17,  # Empty state messages (increased from 14)
        "section_header": 16,  # Filter panel section headers (increased from 14)
        "table_header": 13,  # Table headers (reverted to original)
        "table_cell": 12,  # Table cells (reverted to original)
        "dialog_title": 17,  # Dialog titles (increased from 16)
        "button": 13,  # Button text (increased from 12)
        "field_label": 15,  # Form field labels (increased from 13)
        "input_text": 14,  # Input field text (increased from 12)
        "control_header": 16,  # Control panel headers (Bar Chart Controls, etc.)
        "control_label": 14,  # Labels within control panels
        "panel_title": 15,  # Panel titles (matches h3 for consistency)
        "dock_title": 14,  # Dock widget titles (10-15% smaller than h3=16)
    }

    # Font weights
    WEIGHTS = {
        "light": QFont.Light,
        "normal": QFont.Normal,
        "medium": QFont.Medium,
        "bold": QFont.Bold,
        "black": QFont.Black,
    }

    def __init__(self):
        super().__init__()
        self.scale_factor = 1.0
        self.base_window_width = (
            1600  # Reference window width - updated for modern displays
        )
        self.base_window_height = 1000  # Reference window height - better aspect ratio
        self.fonts: Dict[str, QFont] = {}
        self._init_fonts()

    def _init_fonts(self):
        """Initialize the font hierarchy."""
        app = QApplication.instance()
        base_font = app.font() if app else QFont()

        for size_name, base_size in self.BASE_SIZES.items():
            font = QFont(base_font.family())
            font.setPointSize(int(base_size * self.scale_factor))

            # Apply weight based on standard hierarchy
            if size_name in ["h1", "h2", "dialog_title"]:
                font.setWeight(self.WEIGHTS["bold"])
            elif size_name in [
                "h3",
                "h4",
                "section_header",
                "empty_state",
                "field_label",
                "control_header",
                "panel_title",
                "dock_title",
            ]:
                font.setWeight(self.WEIGHTS["medium"])
            elif size_name in ["table_header"]:
                font.setWeight(self.WEIGHTS["medium"])  # Headers stand out in tables
            else:
                font.setWeight(self.WEIGHTS["normal"])

            self.fonts[size_name] = font

    def update_scale_factor(self, window_width: int, window_height: int):
        """
        Update the scale factor based on window dimensions.

        Args:
            window_width: Current window width
            window_height: Current window height
        """
        # Calculate scale factor based on window size
        width_scale = window_width / self.base_window_width
        height_scale = window_height / self.base_window_height

        # Use geometric mean for balanced scaling, with more conservative bounds
        raw_scale_factor = math.sqrt(width_scale * height_scale)

        # Enhanced scaling algorithm with better responsiveness and wider range
        if raw_scale_factor > 1.0:
            # For larger windows, scale more responsively
            excess_scale = raw_scale_factor - 1.0
            if excess_scale <= 0.4:  # Moderate scaling up to 140%
                dampened_scale = (
                    1.0 + excess_scale * 0.75
                )  # 75% of the scaling - more responsive
            else:  # Still responsive for very large displays
                dampened_scale = (
                    1.0 + 0.3 + (excess_scale - 0.4) * 0.4
                )  # 30% + 40% of additional
        else:
            # For smaller windows, scale down more gently to maintain readability
            scale_reduction = 1.0 - raw_scale_factor
            dampened_scale = (
                1.0 - scale_reduction * 0.4
            )  # 40% of the reduction - gentler

        # Expanded bounds for better scalability across more screen sizes
        new_scale_factor = max(
            0.8, min(1.8, dampened_scale)
        )  # Wider range: 80% to 180%

        # Only update if change is significant (avoid constant updates)
        if abs(new_scale_factor - self.scale_factor) > 0.03:
            old_scale = self.scale_factor
            self.scale_factor = new_scale_factor
            self._init_fonts()
            self.fonts_updated.emit()

            # Debug logging for font scaling
            logger.debug(
                f"Font scaling updated: {old_scale:.2f} -> {new_scale_factor:.2f} "
                f"(window: {window_width}x{window_height}, raw: {raw_scale_factor:.2f})"
            )

    def get_font(self, style: str, weight: Optional[str] = None) -> QFont:
        """
        Get a font for the specified style and weight.

        Args:
            style: Font style (h1, h2, h3, h4, h5, h6, body, caption, tiny)
            weight: Font weight override (light, normal, medium, bold, black)

        Returns:
            QFont object with appropriate styling
        """
        if style not in self.fonts:
            style = "body"  # Fallback to body text

        font = QFont(self.fonts[style])

        if weight and weight in self.WEIGHTS:
            font.setWeight(self.WEIGHTS[weight])

        return font

    def get_metrics(self, style: str) -> QFontMetrics:
        """Get font metrics for the specified style."""
        return QFontMetrics(self.get_font(style))

    def calculate_text_width(self, text: str, style: str) -> int:
        """Calculate the width of text for the given style."""
        metrics = self.get_metrics(style)
        return metrics.horizontalAdvance(text)

    def calculate_text_height(self, style: str) -> int:
        """Calculate the height of text for the given style."""
        metrics = self.get_metrics(style)
        return metrics.height()

    def get_scale_factor(self) -> float:
        """Get the current scale factor."""
        return self.scale_factor


# Global font manager instance
_font_manager: Optional[FontManager] = None


def get_font_manager() -> FontManager:
    """Get the global font manager instance."""
    global _font_manager
    if _font_manager is None:
        _font_manager = FontManager()
    return _font_manager


def create_styled_font(style: str, weight: Optional[str] = None) -> QFont:
    """Convenience function to create a styled font."""
    return get_font_manager().get_font(style, weight)


def update_fonts_for_window_size(width: int, height: int):
    """Update fonts for the given window size."""
    get_font_manager().update_scale_factor(width, height)

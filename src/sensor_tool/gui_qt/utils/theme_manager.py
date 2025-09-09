"""
Theme Manager for PySide6 GUI

Provides dark/light mode theming similar to CustomTkinter's approach.
Ensures text visibility across different themes.
"""

from enum import Enum
from typing import Any, Dict

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"


class ThemeManager(QObject):
    """Manages application-wide theme switching between light and dark modes."""

    theme_changed = Signal(str)  # Emits theme name when changed

    def __init__(self):
        super().__init__()
        self._current_theme = ThemeMode.LIGHT
        self._themes = self._define_themes()

    def _define_themes(self) -> Dict[ThemeMode, Dict[str, Any]]:
        """Define modern standard color schemes for light and dark themes."""
        return {
            ThemeMode.LIGHT: {
                # Base colors - Clean, modern neutrals
                "background": "#ffffff",
                "surface": "#f8f9fb",
                "surface_elevated": "#ffffff",
                "surface_container": "#f3f4f6",
                # Brand colors - Modern blue gradient
                "primary": "#2563eb",  # Modern blue
                "primary_light": "#3b82f6",  # Lighter shade
                "primary_dark": "#1d4ed8",  # Darker shade
                "primary_gradient_start": "#2563eb",
                "primary_gradient_end": "#1d4ed8",
                # Accent colors
                "accent": "#0ea5e9",  # Sky blue
                "accent_light": "#38bdf8",
                "secondary": "#64748b",  # Slate
                "secondary_light": "#94a3b8",
                # Text hierarchy - High contrast for accessibility
                "text_primary": "#0f172a",  # Near black
                "text_secondary": "#334155",  # Dark slate
                "text_tertiary": "#64748b",  # Medium slate
                "text_quaternary": "#94a3b8",  # Light slate
                "text_on_primary": "#ffffff",
                "text_on_surface": "#0f172a",
                "text_disabled": "#cbd5e1",
                # Interactive elements - Subtle, modern
                "border": "#e2e8f0",
                "border_light": "#f1f5f9",
                "border_strong": "#cbd5e1",
                "border_focus": "#2563eb",
                "border_error": "#ef4444",
                # State colors
                "hover": "#f1f5f9",
                "hover_strong": "#e2e8f0",
                "active": "#ddd6fe",
                "selected": "#eff6ff",
                "focus": "#dbeafe",
                # Status colors - Modern, accessible
                "success": "#059669",
                "success_light": "#d1fae5",
                "warning": "#d97706",
                "warning_light": "#fef3c7",
                "error": "#dc2626",
                "error_light": "#fee2e2",
                "info": "#0284c7",
                "info_light": "#e0f2fe",
                # Shadow and elevation
                "shadow_light": "rgba(0, 0, 0, 0.05)",
                "shadow_medium": "rgba(0, 0, 0, 0.1)",
                "shadow_strong": "rgba(0, 0, 0, 0.15)",
                "overlay": "rgba(0, 0, 0, 0.5)",
            },
            ThemeMode.DARK: {
                # Base colors - Rich, deep neutrals
                "background": "#0f172a",  # Deep navy
                "surface": "#1e293b",  # Slate 800
                "surface_elevated": "#334155",  # Slate 700
                "surface_container": "#1e293b",
                # Brand colors - Vibrant blues for dark mode
                "primary": "#3b82f6",  # Blue 500
                "primary_light": "#60a5fa",  # Blue 400
                "primary_dark": "#2563eb",  # Blue 600
                "primary_gradient_start": "#3b82f6",
                "primary_gradient_end": "#1d4ed8",
                # Accent colors
                "accent": "#0ea5e9",  # Sky 500
                "accent_light": "#38bdf8",  # Sky 400
                "secondary": "#94a3b8",  # Slate 400
                "secondary_light": "#cbd5e1",  # Slate 300
                # Text hierarchy - Optimized for dark backgrounds
                "text_primary": "#f8fafc",  # Slate 50
                "text_secondary": "#e2e8f0",  # Slate 200
                "text_tertiary": "#cbd5e1",  # Slate 300
                "text_quaternary": "#94a3b8",  # Slate 400
                "text_on_primary": "#ffffff",
                "text_on_surface": "#f8fafc",
                "text_disabled": "#64748b",
                # Interactive elements - Subtle but visible
                "border": "#374151",  # Gray 700
                "border_light": "#4b5563",  # Gray 600
                "border_strong": "#6b7280",  # Gray 500
                "border_focus": "#3b82f6",
                "border_error": "#ef4444",
                # State colors
                "hover": "#374151",
                "hover_strong": "#4b5563",
                "active": "#581c87",  # Purple 900
                "selected": "#1e3a8a",  # Blue 900
                "focus": "#1e40af",  # Blue 800
                # Status colors - High contrast for dark mode
                "success": "#10b981",  # Emerald 500
                "success_light": "#064e3b",  # Emerald 900
                "warning": "#f59e0b",  # Amber 500
                "warning_light": "#78350f",  # Amber 900
                "error": "#ef4444",  # Red 500
                "error_light": "#7f1d1d",  # Red 900
                "info": "#06b6d4",  # Cyan 500
                "info_light": "#083344",  # Cyan 900
                # Shadow and elevation
                "shadow_light": "rgba(0, 0, 0, 0.2)",
                "shadow_medium": "rgba(0, 0, 0, 0.3)",
                "shadow_strong": "rgba(0, 0, 0, 0.4)",
                "overlay": "rgba(0, 0, 0, 0.7)",
            },
        }

    @property
    def current_theme(self) -> ThemeMode:
        """Get current theme mode."""
        return self._current_theme

    @property
    def is_dark_mode(self) -> bool:
        """Check if current theme is dark mode."""
        return self._current_theme == ThemeMode.DARK

    def get_color(self, color_name: str) -> str:
        """Get color value for current theme."""
        return self._themes[self._current_theme].get(color_name, "#000000")

    def set_theme(self, theme: ThemeMode):
        """Set the application theme."""
        if theme != self._current_theme:
            self._current_theme = theme
            self._apply_theme_to_application()
            self.theme_changed.emit(theme.value)

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        new_theme = (
            ThemeMode.DARK
            if self._current_theme == ThemeMode.LIGHT
            else ThemeMode.LIGHT
        )
        self.set_theme(new_theme)

    def _apply_theme_to_application(self):
        """Apply theme colors to the QApplication palette."""
        app = QApplication.instance()
        if not app:
            return

        palette = QPalette()
        colors = self._themes[self._current_theme]

        # Set palette colors
        palette.setColor(QPalette.Window, QColor(colors["background"]))
        palette.setColor(QPalette.WindowText, QColor(colors["text_primary"]))
        palette.setColor(QPalette.Base, QColor(colors["surface"]))
        palette.setColor(QPalette.AlternateBase, QColor(colors["hover"]))
        palette.setColor(QPalette.Text, QColor(colors["text_primary"]))
        palette.setColor(QPalette.Button, QColor(colors["surface"]))
        palette.setColor(QPalette.ButtonText, QColor(colors["text_primary"]))
        palette.setColor(QPalette.Highlight, QColor(colors["primary"]))
        palette.setColor(QPalette.HighlightedText, QColor(colors["text_on_primary"]))

        app.setPalette(palette)

    def get_stylesheet_colors(self) -> Dict[str, str]:
        """Get all colors for current theme as a dictionary for stylesheet formatting."""
        return self._themes[self._current_theme].copy()

    def create_button_stylesheet(self, variant: str = "primary") -> str:
        """Create modern, standard button stylesheets with hover effects."""
        colors = self.get_stylesheet_colors()

        if variant == "primary":
            return f"""
                QPushButton {{
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 {colors['primary']},
                        stop: 1 {colors['primary_dark']}
                    );
                    color: {colors['text_on_primary']};
                    border: none;
                    border-radius: 8px;
                    padding: 10px 18px;
                    font-weight: 500;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 {colors['primary_light']},
                        stop: 1 {colors['primary']}
                    );
                }}
                QPushButton:pressed {{
                    background: {colors['primary_dark']};
                }}
                QPushButton:disabled {{
                    background-color: {colors['text_disabled']};
                    color: {colors['text_quaternary']};
                }}
            """
        elif variant == "secondary":
            return f"""
                QPushButton {{
                    background-color: {colors['surface']};
                    color: {colors['text_secondary']};
                    border: 2px solid {colors['border_strong']};
                    border-radius: 8px;
                    padding: 10px 18px;
                    font-weight: 500;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {colors['hover_strong']};
                    border-color: {colors['primary']};
                    color: {colors['text_primary']};
                }}
                QPushButton:pressed {{
                    background-color: {colors['active']};
                }}
                QPushButton:disabled {{
                    background-color: {colors['text_disabled']};
                    color: {colors['text_quaternary']};
                    border-color: {colors['text_disabled']};
                }}
            """
        elif variant == "accent":
            return f"""
                QPushButton {{
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 {colors['accent']},
                        stop: 1 {colors['accent']}
                    );
                    color: {colors['text_on_primary']};
                    border: none;
                    border-radius: 8px;
                    padding: 10px 18px;
                    font-weight: 500;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background: {colors['accent_light']};
                }}
                QPushButton:pressed {{
                    background: {colors['accent']};
                }}
            """
        elif variant == "danger":
            return f"""
                QPushButton {{
                    background-color: {colors['error']};
                    color: {colors['text_on_primary']};
                    border: none;
                    border-radius: 8px;
                    padding: 10px 18px;
                    font-weight: 500;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: #ef4444;
                }}
                QPushButton:pressed {{
                    background-color: #dc2626;
                }}
            """
        else:  # default
            return f"""
                QPushButton {{
                    background-color: {colors['surface']};
                    color: {colors['text_primary']};
                    border: 1px solid {colors['border']};
                    border-radius: 8px;
                    padding: 10px 18px;
                    font-weight: 500;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {colors['hover']};
                    border-color: {colors['border_focus']};
                }}
                QPushButton:pressed {{
                    background-color: {colors['active']};
                }}
                QPushButton:disabled {{
                    background-color: {colors['text_disabled']};
                    color: {colors['text_quaternary']};
                    border-color: {colors['text_disabled']};
                }}
            """

    def create_card_stylesheet(self) -> str:
        """Create modern card styling with elevation and shadows."""
        colors = self.get_stylesheet_colors()

        return f"""
            .card {{
                background-color: {colors['surface_elevated']};
                border: 1px solid {colors['border_light']};
                border-radius: 12px;
                padding: 20px;
                margin: 8px;
            }}
            QGroupBox {{
                background-color: {colors['surface_elevated']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
                padding-top: 20px;
                margin-top: 8px;
                color: {colors['text_primary']};
                font-weight: 600;
                font-size: 14px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                color: {colors['text_primary']};
                background-color: {colors['surface_elevated']};
            }}
        """

    def create_input_stylesheet(self) -> str:
        """Create modern input field styling."""
        colors = self.get_stylesheet_colors()

        return f"""
            QLineEdit, QTextEdit, QPlainTextEdit {{
                background-color: {colors['surface']};
                border: 2px solid {colors['border']};
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 14px;
                color: {colors['text_primary']};
                selection-background-color: {colors['selected']};
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border-color: {colors['border_focus']};
                background-color: {colors['background']};
            }}
            QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover {{
                border-color: {colors['border_strong']};
            }}
            QComboBox {{
                background-color: {colors['surface']};
                border: 2px solid {colors['border']};
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 14px;
                color: {colors['text_primary']};
                min-width: 120px;
            }}
            QComboBox:hover {{
                border-color: {colors['border_strong']};
            }}
            QComboBox:focus {{
                border-color: {colors['border_focus']};
            }}
            QComboBox::drop-down {{
                border: none;
                background: transparent;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 8px solid {colors['text_secondary']};
                margin: 0 8px;
            }}
            QComboBox::down-arrow:hover {{
                border-top: 8px solid {colors['text_primary']};
            }}
            QComboBox::down-arrow:pressed {{
                margin-top: 1px;
            }}
        """

    def create_table_stylesheet(self) -> str:
        """Create modern table styling with hover effects."""
        colors = self.get_stylesheet_colors()

        return f"""
            QTableWidget {{
                background-color: {colors['background']};
                alternate-background-color: {colors['surface']};
                gridline-color: {colors['border_light']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                selection-background-color: {colors['selected']};
                selection-color: {colors['text_primary']};
            }}
            QTableWidget::item {{
                padding: 12px 8px;
                border-bottom: 1px solid {colors['border_light']};
                color: {colors['text_primary']};
            }}
            QTableWidget::item:hover {{
                background-color: {colors['hover']};
            }}
            QTableWidget::item:selected {{
                background-color: {colors['selected']};
                color: {colors['text_primary']};
            }}
            QHeaderView::section {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {colors['surface_elevated']},
                    stop: 1 {colors['surface']}
                );
                padding: 12px 8px;
                border: 1px solid {colors['border']};
                border-radius: 0px;
                font-weight: 600;
                color: {colors['text_primary']};
                text-align: left;
            }}
            QHeaderView::section:hover {{
                background-color: {colors['hover_strong']};
            }}
        """

    def create_dialog_stylesheet(self) -> str:
        """Create modern dialog styling with standard appearance."""
        colors = self.get_stylesheet_colors()

        return f"""
            QDialog {{
                background-color: {colors['background']};
                color: {colors['text_primary']};
                border-radius: 12px;
            }}
            QLabel {{
                color: {colors['text_primary']};
                background-color: transparent;
                font-size: 14px;
            }}
            QCheckBox {{
                spacing: 8px;
                padding: 4px;
                color: {colors['text_primary']};
                font-size: 14px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {colors['border_strong']};
                border-radius: 4px;
                background-color: {colors['background']};
            }}
            QCheckBox::indicator:hover {{
                border-color: {colors['border_focus']};
                background-color: {colors['hover']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {colors['primary']};
                border-color: {colors['primary']};
                image: none;
            }}
            QRadioButton {{
                spacing: 8px;
                padding: 4px;
                color: {colors['text_primary']};
                font-size: 14px;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {colors['border_strong']};
                border-radius: 10px;
                background-color: {colors['background']};
            }}
            QRadioButton::indicator:hover {{
                border-color: {colors['border_focus']};
                background-color: {colors['hover']};
            }}
            QRadioButton::indicator:checked {{
                background-color: {colors['primary']};
                border-color: {colors['primary']};
            }}
        """

    def create_navigation_stylesheet(self) -> str:
        """Create modern navigation and menu styling."""
        colors = self.get_stylesheet_colors()

        return f"""
            QMenuBar {{
                background-color: {colors['surface']};
                color: {colors['text_primary']};
                border-bottom: 1px solid {colors['border']};
                font-size: 14px;
                padding: 4px;
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 8px 12px;
                margin: 2px;
                border-radius: 6px;
            }}
            QMenuBar::item:selected {{
                background-color: {colors['hover']};
                color: {colors['text_primary']};
            }}
            QMenuBar::item:pressed {{
                background-color: {colors['active']};
            }}
            QMenu {{
                background-color: {colors['surface_elevated']};
                color: {colors['text_primary']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }}
            QMenu::item {{
                background-color: transparent;
                padding: 10px 16px;
                border-radius: 6px;
                margin: 2px;
            }}
            QMenu::item:selected {{
                background-color: {colors['hover_strong']};
                color: {colors['text_primary']};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {colors['border']};
                margin: 6px 8px;
            }}
            QToolBar {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {colors['surface_elevated']},
                    stop: 1 {colors['surface']}
                );
                border: none;
                border-bottom: 1px solid {colors['border']};
                padding: 4px;
                spacing: 2px;
            }}
            QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 8px;
                margin: 2px;
                color: {colors['text_primary']};
                font-size: 14px;
            }}
            QToolButton:hover {{
                background-color: {colors['hover']};
                border-color: {colors['border']};
            }}
            QToolButton:pressed {{
                background-color: {colors['active']};
            }}
        """

    def create_status_bar_stylesheet(self) -> str:
        """Create modern status bar styling."""
        colors = self.get_stylesheet_colors()

        return f"""
            QStatusBar {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {colors['surface']},
                    stop: 1 {colors['surface_container']}
                );
                border-top: 1px solid {colors['border']};
                color: {colors['text_secondary']};
                font-size: 13px;
                padding: 2px;
            }}
            QProgressBar {{
                background-color: {colors['surface_container']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
                text-align: center;
                font-size: 12px;
                color: {colors['text_primary']};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {colors['primary_light']},
                    stop: 1 {colors['primary']}
                );
                border-radius: 4px;
                margin: 1px;
            }}
        """

    def get_comprehensive_stylesheet(self) -> str:
        """Get comprehensive stylesheet combining all modern components."""
        return f"""
            {self.create_card_stylesheet()}
            {self.create_input_stylesheet()}
            {self.create_table_stylesheet()}
            {self.create_dialog_stylesheet()}
            {self.create_navigation_stylesheet()}
            {self.create_status_bar_stylesheet()}
        """

    def apply_modern_theme_to_app(self) -> None:
        """Apply modern theme styling to the entire application."""
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            # Apply base palette
            self._apply_theme_to_application()
            # Apply comprehensive modern styling
            app.setStyleSheet(self.get_comprehensive_stylesheet())


# Global theme manager instance
_theme_manager = None


def get_theme_manager() -> ThemeManager:
    """Get the global theme manager instance."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager

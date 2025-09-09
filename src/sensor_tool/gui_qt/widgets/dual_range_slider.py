"""
Dual Range Slider Widget

A dual-handle range slider for price filtering in sensor comparison.
Supports light and dark themes with precise value control.
Uses superqt's QRangeSlider for dual-handle functionality.
"""

import logging
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from superqt import QRangeSlider

    SUPERQT_AVAILABLE = True
except ImportError:
    SUPERQT_AVAILABLE = False
    # Fallback to basic QSlider if superqt is not available
    from PySide6.QtWidgets import QSlider

logger = logging.getLogger(__name__)


class DualRangeSlider(QWidget):
    """
    Dual-range slider for price filtering.

    Features:
    - Dual handles for min/max range selection
    - Text input fields for precise values
    - Quick filter buttons for common ranges
    - Theme-aware styling
    - Real-time updates with debouncing

    Signals:
        range_changed(min_val, max_val): Emitted when range changes
        filter_disabled(): Emitted when "Any Price" is selected
    """

    range_changed = Signal(int, int)  # min_price, max_price
    filter_disabled = Signal()
    include_unknown_changed = Signal(bool)  # include_unknown_prices

    def __init__(
        self,
        min_value: int = 0,
        max_value: int = 15000,
        current_min: int = 0,
        current_max: int = 15000,
        parent=None,
    ):
        super().__init__(parent)

        # Range parameters
        self.min_value = min_value
        self.max_value = max_value
        self.current_min = current_min
        self.current_max = current_max

        # Debounce timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.emit_range_change)
        self.debounce_delay = 300  # ms

        self.setup_ui()
        self.connect_signals()
        self.update_display()

        # Validate functionality after setup
        if hasattr(self, "range_slider"):
            self.validate_slider_functionality()

    def resizeEvent(self, event):
        """Handle widget resize to maintain proper slider alignment."""
        super().resizeEvent(event)
        # No manual resizing needed for QRangeSlider - it handles this automatically

    def setup_ui(self):
        """Setup the dual range slider UI with standard styling."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header with "Any Price" checkbox
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.any_price_checkbox = QCheckBox("Any Price")
        self.any_price_checkbox.setToolTip(
            "Disable price filtering to show all sensors"
        )
        header_layout.addWidget(self.any_price_checkbox)

        # Include unknown prices checkbox
        self.include_unknown_checkbox = QCheckBox("Include Unknown Prices")
        self.include_unknown_checkbox.setToolTip(
            "Show sensors even if price data is not available"
        )
        self.include_unknown_checkbox.setChecked(
            True
        )  # Default: include unknown prices
        header_layout.addWidget(self.include_unknown_checkbox)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Range display labels
        range_display_layout = QHBoxLayout()
        range_display_layout.setContentsMargins(0, 0, 0, 0)

        self.min_range_label = QLabel(f"${self.min_value:,}")
        self.min_range_label.setAlignment(Qt.AlignLeft)
        self.min_range_label.setStyleSheet("color: #64748b; font-size: 10px;")

        self.max_range_label = QLabel(f"${self.max_value:,}")
        self.max_range_label.setAlignment(Qt.AlignRight)
        self.max_range_label.setStyleSheet("color: #64748b; font-size: 10px;")

        range_display_layout.addWidget(self.min_range_label)
        range_display_layout.addStretch()
        range_display_layout.addWidget(self.max_range_label)

        layout.addLayout(range_display_layout)

        # Modern dual-handle range slider
        slider_container = QFrame()
        slider_container.setMinimumHeight(60)
        slider_container.setStyleSheet("QFrame { background: transparent; }")
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setContentsMargins(10, 10, 10, 10)

        if SUPERQT_AVAILABLE:
            # Use superqt's QRangeSlider for reliable dual-handle functionality
            logger.info("Using superqt QRangeSlider for dual-handle price filtering")
            self.range_slider = QRangeSlider(Qt.Horizontal)
            self.range_slider.setRange(self.min_value, self.max_value)
            self.range_slider.setValue((self.current_min, self.current_max))
            self.range_slider.setMinimumHeight(30)

            # Apply standard styling
            self.range_slider.setStyleSheet(self.get_qrange_slider_stylesheet())

            slider_layout.addWidget(self.range_slider)
        else:
            # Fallback: Use a single slider for max value (simpler but functional)
            logger.warning("superqt not available, using fallback single slider")

            from PySide6.QtWidgets import QSlider

            self.range_slider = QSlider(Qt.Horizontal)
            self.range_slider.setRange(self.min_value, self.max_value)
            self.range_slider.setValue(self.current_max)
            self.range_slider.setMinimumHeight(30)
            self.range_slider.setStyleSheet(self.get_fallback_slider_stylesheet())

            slider_layout.addWidget(self.range_slider)

        layout.addWidget(slider_container)

        # Current values display and input
        values_layout = QHBoxLayout()
        values_layout.setContentsMargins(0, 0, 0, 0)

        # Min input
        min_input_layout = QHBoxLayout()
        min_input_layout.addWidget(QLabel("Min:"))
        self.min_input = QLineEdit()
        self.min_input.setValidator(QIntValidator(self.min_value, self.max_value))
        self.min_input.setMaximumWidth(100)
        self.min_input.setPlaceholderText(f"${self.min_value:,}")
        min_input_layout.addWidget(self.min_input)

        # Separator
        separator_label = QLabel("—")
        separator_label.setAlignment(Qt.AlignCenter)
        separator_label.setStyleSheet("color: #64748b; font-weight: bold;")

        # Max input
        max_input_layout = QHBoxLayout()
        max_input_layout.addWidget(QLabel("Max:"))
        self.max_input = QLineEdit()
        self.max_input.setValidator(QIntValidator(self.min_value, self.max_value))
        self.max_input.setMaximumWidth(100)
        self.max_input.setPlaceholderText(f"${self.max_value:,}")
        max_input_layout.addWidget(self.max_input)

        values_layout.addLayout(min_input_layout)
        values_layout.addWidget(separator_label)
        values_layout.addLayout(max_input_layout)
        values_layout.addStretch()

        layout.addLayout(values_layout)

        # Quick filter buttons for common robotics budget ranges
        quick_filters_layout = QHBoxLayout()
        quick_filters_layout.setContentsMargins(0, 0, 0, 0)

        self.budget_btn = QPushButton("Budget\n< $1K")
        self.budget_btn.setMinimumWidth(110)
        self.budget_btn.setMinimumHeight(45)
        self.budget_btn.setStyleSheet(self.get_quick_filter_stylesheet())
        self.budget_btn.setToolTip("Budget sensors under $1,000")

        self.mid_range_btn = QPushButton("Mid-range\n$1K-5K")
        self.mid_range_btn.setMinimumWidth(110)
        self.mid_range_btn.setMinimumHeight(45)
        self.mid_range_btn.setStyleSheet(self.get_quick_filter_stylesheet())
        self.mid_range_btn.setToolTip("Mid-range sensors $1,000-$5,000")

        self.premium_btn = QPushButton("Premium\n> $5K")
        self.premium_btn.setMinimumWidth(110)
        self.premium_btn.setMinimumHeight(45)
        self.premium_btn.setStyleSheet(self.get_quick_filter_stylesheet())
        self.premium_btn.setToolTip("Premium sensors over $5,000")

        quick_filters_layout.addWidget(self.budget_btn)
        quick_filters_layout.addSpacing(8)
        quick_filters_layout.addWidget(self.mid_range_btn)
        quick_filters_layout.addSpacing(8)
        quick_filters_layout.addWidget(self.premium_btn)
        quick_filters_layout.addStretch()

        layout.addLayout(quick_filters_layout)

    def get_qrange_slider_stylesheet(self) -> str:
        """Get theme-aware stylesheet for superqt QRangeSlider."""
        return """
        QRangeSlider::groove:horizontal {
            border: 1px solid #cbd5e1;
            height: 8px;
            background: #f1f5f9;
            margin: 2px 0;
            border-radius: 4px;
        }
        QRangeSlider::handle:horizontal {
            background: #3b82f6;
            border: 2px solid #ffffff;
            width: 20px;
            height: 20px;
            margin: -7px 0;
            border-radius: 10px;
        }
        QRangeSlider::handle:horizontal:hover {
            background: #2563eb;
            border-color: #f8fafc;
        }
        QRangeSlider::handle:horizontal:pressed {
            background: #1d4ed8;
        }
        QRangeSlider::sub-page:horizontal {
            background: #3b82f6;
            border: 1px solid #2563eb;
            height: 8px;
            border-radius: 4px;
        }
        """

    def get_fallback_slider_stylesheet(self) -> str:
        """Get stylesheet for fallback single slider."""
        return """
        QSlider::groove:horizontal {
            border: 1px solid #cbd5e1;
            height: 8px;
            background: #f1f5f9;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #3b82f6;
            border: 2px solid #ffffff;
            width: 20px;
            height: 20px;
            margin: -7px 0;
            border-radius: 10px;
        }
        QSlider::handle:horizontal:hover {
            background: #2563eb;
        }
        QSlider::sub-page:horizontal {
            background: #3b82f6;
            border: 1px solid #2563eb;
            height: 8px;
            border-radius: 4px;
        }
        """

    def get_quick_filter_stylesheet(self) -> str:
        """Get stylesheet for quick filter buttons with improved readability."""
        return """
        QPushButton {
            background-color: #f8f9fb;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 600;
            color: #334155;
            text-align: center;
        }
        QPushButton:hover {
            background-color: #e2e8f0;
            border-color: #cbd5e1;
            color: #1e293b;
        }
        QPushButton:pressed {
            background-color: #d1d5db;
            border-color: #94a3b8;
        }
        """

    def connect_signals(self):
        """Connect widget signals."""
        # Range slider signals
        if SUPERQT_AVAILABLE:
            self.range_slider.valueChanged.connect(self.on_range_slider_changed)
        else:
            self.range_slider.valueChanged.connect(self.on_fallback_slider_changed)

        # Input field signals
        self.min_input.textChanged.connect(self.on_min_input_changed)
        self.max_input.textChanged.connect(self.on_max_input_changed)

        # Quick filter buttons
        self.budget_btn.clicked.connect(lambda: self.set_range(self.min_value, 1000))
        self.mid_range_btn.clicked.connect(lambda: self.set_range(1000, 5000))
        self.premium_btn.clicked.connect(lambda: self.set_range(5000, self.max_value))

        # Any price checkbox
        self.any_price_checkbox.toggled.connect(self.on_any_price_toggled)

        # Include unknown prices checkbox
        self.include_unknown_checkbox.toggled.connect(self.on_include_unknown_toggled)

    def on_range_slider_changed(self, value):
        """Handle QRangeSlider value change."""
        if isinstance(value, tuple) and len(value) == 2:
            self.current_min, self.current_max = value
            self.update_display()
            self.schedule_update()

    def on_fallback_slider_changed(self, value):
        """Handle fallback single slider value change."""
        # For fallback, we only control max value
        self.current_max = value
        self.update_display()
        self.schedule_update()

    def on_min_input_changed(self, text):
        """Handle min input field change."""
        try:
            value = (
                int(text.replace("$", "").replace(",", "")) if text else self.min_value
            )
            value = max(self.min_value, min(value, self.max_value))

            if value <= self.current_max:
                self.current_min = value
                self._update_slider_values()
                self.schedule_update()
        except ValueError:
            pass  # Invalid input, ignore

    def on_max_input_changed(self, text):
        """Handle max input field change."""
        try:
            value = (
                int(text.replace("$", "").replace(",", "")) if text else self.max_value
            )
            value = max(self.min_value, min(value, self.max_value))

            if value >= self.current_min:
                self.current_max = value
                self._update_slider_values()
                self.schedule_update()
        except ValueError:
            pass  # Invalid input, ignore

    def _update_slider_values(self):
        """Update slider values based on current min/max."""
        if SUPERQT_AVAILABLE:
            self.range_slider.setValue((self.current_min, self.current_max))
        else:
            self.range_slider.setValue(self.current_max)

    def on_any_price_toggled(self, checked):
        """Handle Any Price checkbox toggle."""
        self.set_widgets_enabled(not checked)

        if checked:
            self.filter_disabled.emit()
        else:
            self.emit_range_change()

    def on_include_unknown_toggled(self, checked):
        """Handle Include Unknown Prices checkbox toggle."""
        self.include_unknown_changed.emit(checked)

    def set_widgets_enabled(self, enabled: bool):
        """Enable/disable all input widgets."""
        self.range_slider.setEnabled(enabled)
        self.min_input.setEnabled(enabled)
        self.max_input.setEnabled(enabled)
        self.budget_btn.setEnabled(enabled)
        self.mid_range_btn.setEnabled(enabled)
        self.premium_btn.setEnabled(enabled)

    def set_range(self, min_val: int, max_val: int):
        """Set the range programmatically."""
        self.current_min = max(self.min_value, min_val)
        self.current_max = min(self.max_value, max_val)

        self._update_slider_values()
        self.update_display()
        self.emit_range_change()

    def update_display(self):
        """Update input field displays."""
        self.min_input.setText(f"${self.current_min:,}")
        self.max_input.setText(f"${self.current_max:,}")

    def schedule_update(self):
        """Schedule a debounced update."""
        self.update_timer.start(self.debounce_delay)

    def emit_range_change(self):
        """Emit the range changed signal."""
        if not self.any_price_checkbox.isChecked():
            self.range_changed.emit(self.current_min, self.current_max)
            # Validation: Ensure min <= max
            if self.current_min > self.current_max:
                logger.warning(
                    f"Invalid range: min ({self.current_min}) > max ({self.current_max})"
                )

    def get_range(self) -> Tuple[Optional[int], Optional[int]]:
        """Get current range values."""
        if self.any_price_checkbox.isChecked():
            return None, None
        return self.current_min, self.current_max

    def get_include_unknown(self) -> bool:
        """Get whether to include sensors with unknown prices."""
        return self.include_unknown_checkbox.isChecked()

    def reset(self):
        """Reset to default state."""
        self.any_price_checkbox.setChecked(True)
        self.current_min = self.min_value
        self.current_max = self.max_value
        self._update_slider_values()
        self.update_display()

    def validate_slider_functionality(self) -> bool:
        """Validate that the slider handles are working correctly."""
        if not SUPERQT_AVAILABLE:
            logger.warning(
                "Dual range slider validation: superqt not available, using fallback"
            )
            return False

        # Test if we can get and set values properly
        try:
            current_value = self.range_slider.value()
            if isinstance(current_value, tuple) and len(current_value) == 2:
                logger.info(
                    f"Dual range slider validation: Both handles working, current value: {current_value}"
                )
                return True
            else:
                logger.error(
                    f"Dual range slider validation: Invalid value format: {current_value}"
                )
                return False
        except Exception as e:
            logger.error(f"Dual range slider validation failed: {e}")
            return False

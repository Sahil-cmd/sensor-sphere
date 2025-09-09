"""
Fuzzy Matching Auto-Complete Widget

Provides intelligent auto-complete functionality with fuzzy matching support
for enhanced user experience in filtering and searching.
"""

import difflib
from typing import List, Optional

from PySide6.QtCore import QSortFilterProxyModel, QStringListModel, Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QLineEdit,
    QListView,
    QStyledItemDelegate,
)

from ..utils.font_manager import create_styled_font


class FuzzyFilterProxyModel(QSortFilterProxyModel):
    """
    Custom proxy model that implements fuzzy matching for filtering.

    Uses difflib.SequenceMatcher for intelligent fuzzy matching that:
    - Handles typos and slight misspellings
    - Supports partial matching
    - Ranks results by similarity score
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.fuzzy_threshold = (
            0.5  # Minimum similarity score (0.0 to 1.0) - more lenient for typos
        )
        self._filter_string = ""

    def setFilterString(self, filter_string: str):
        """Set the filter string for fuzzy matching."""
        self._filter_string = filter_string.lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        """
        Override to implement fuzzy matching logic.

        Returns True if the row should be included based on fuzzy match score.
        """
        if not self._filter_string:
            return True  # Show all items if no filter

        # Get the text from the source model
        index = self.sourceModel().index(source_row, 0, source_parent)
        text = self.sourceModel().data(index, Qt.DisplayRole)

        if not text:
            return False

        text_lower = text.lower()

        # Exact match or contains
        if self._filter_string in text_lower:
            return True

        # Check if filter matches any word start
        words = text_lower.split()
        for word in words:
            if word.startswith(self._filter_string):
                return True

        # Also check for matching first letters of words (acronym matching)
        acronym = "".join(word[0] for word in words if word)
        if self._filter_string in acronym:
            return True

        # Fuzzy matching using SequenceMatcher for the whole text
        similarity = difflib.SequenceMatcher(
            None, self._filter_string, text_lower
        ).ratio()
        if similarity >= self.fuzzy_threshold:
            return True

        # Also try fuzzy matching against individual words for better results
        for word in words:
            word_similarity = difflib.SequenceMatcher(
                None, self._filter_string, word
            ).ratio()
            if (
                word_similarity >= self.fuzzy_threshold + 0.1
            ):  # Slightly higher threshold for word matching
                return True

        return False

    def lessThan(self, left, right) -> bool:
        """
        Override to sort by relevance score when filtering.

        Items that better match the filter appear first.
        """
        if not self._filter_string:
            # Default alphabetical sorting when no filter
            left_data = self.sourceModel().data(left, Qt.DisplayRole)
            right_data = self.sourceModel().data(right, Qt.DisplayRole)
            return left_data < right_data

        left_text = self.sourceModel().data(left, Qt.DisplayRole).lower()
        right_text = self.sourceModel().data(right, Qt.DisplayRole).lower()

        # Calculate similarity scores
        left_score = self._calculate_relevance_score(left_text)
        right_score = self._calculate_relevance_score(right_text)

        # Higher scores should appear first (reverse sort)
        return left_score > right_score

    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate relevance score for sorting."""
        # Exact match gets highest score
        if self._filter_string == text:
            return 2.0

        # Starts with filter gets high score
        if text.startswith(self._filter_string):
            return 1.5

        # Contains filter gets medium score
        if self._filter_string in text:
            return 1.0

        # Fuzzy match score
        return difflib.SequenceMatcher(None, self._filter_string, text).ratio()


class FuzzyCompleter(QCompleter):
    """
    Custom QCompleter with fuzzy matching support.

    Enhances the standard QCompleter with intelligent fuzzy matching
    for better user experience.
    """

    def __init__(self, items: List[str], parent=None):
        super().__init__(parent)

        # Set up the model and proxy
        self.source_model = QStringListModel(items)
        self.proxy_model = FuzzyFilterProxyModel()
        self.proxy_model.setSourceModel(self.source_model)
        self.proxy_model.setDynamicSortFilter(True)

        # Configure completer
        self.setModel(self.proxy_model)
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.setCompletionMode(QCompleter.PopupCompletion)
        self.setFilterMode(Qt.MatchContains)

        # Style the popup
        popup = self.popup()
        if popup:
            popup.setFont(create_styled_font("body"))

    def setFilterString(self, text: str):
        """Update the filter string for fuzzy matching."""
        self.proxy_model.setFilterString(text)

    def updateItems(self, items: List[str]):
        """Update the list of items for completion."""
        self.source_model.setStringList(items)


class FuzzySearchComboBox(QComboBox):
    """
    Enhanced QComboBox with fuzzy matching auto-complete.

    Features:
    - Intelligent fuzzy matching for typos and partial matches
    - Ranked results by relevance
    - Smooth keyboard navigation
    - "Any" option support for clearing selection

    Signals:
        fuzzy_match_selected: Emitted when a fuzzy match is selected
    """

    fuzzy_match_selected = Signal(str)

    def __init__(self, parent=None, include_any=True):
        super().__init__(parent)

        self.include_any = include_any
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)

        # Initialize with proper font
        self.setFont(create_styled_font("body"))

        # Apply proper dropdown arrow styling
        self.setStyleSheet(self._get_combo_box_stylesheet())

        # Store original items
        self._items = []

        # Set up fuzzy completer
        self.fuzzy_completer = None
        self._setup_completer()

        # Connect signals
        self.currentTextChanged.connect(self._on_text_changed)

    def _get_combo_box_stylesheet(self) -> str:
        """Get stylesheet for combobox with proper dropdown arrow."""
        return """
        QComboBox {
            border: 1px solid #d1d5db;
            border-radius: 6px;
            padding: 8px 30px 8px 12px;
            background-color: white;
            selection-background-color: #e7f3ff;
            min-height: 18px;
        }
        QComboBox:hover {
            border-color: #94a3b8;
            background-color: #f8fafc;
        }
        QComboBox:focus {
            border-color: #3b82f6;
            outline: none;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 25px;
            border-left-width: 1px;
            border-left-color: #d1d5db;
            border-left-style: solid;
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
            background-color: #f9fafb;
        }
        QComboBox::drop-down:hover {
            background-color: #f3f4f6;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #6b7280;
            width: 0;
            height: 0;
            margin: 2px;
        }
        QComboBox::down-arrow:hover {
            border-top-color: #374151;
        }
        QComboBox QAbstractItemView {
            border: 1px solid #d1d5db;
            background-color: white;
            selection-background-color: #e7f3ff;
            selection-color: #1f2937;
            border-radius: 6px;
            padding: 4px;
        }
        QComboBox QAbstractItemView::item {
            padding: 6px 12px;
            border-radius: 4px;
            min-height: 20px;
        }
        QComboBox QAbstractItemView::item:hover {
            background-color: #f3f4f6;
        }
        QComboBox QAbstractItemView::item:selected {
            background-color: #e7f3ff;
            color: #1f2937;
        }
        """

    def _setup_completer(self):
        """Set up the fuzzy matching completer."""
        if self._items:
            all_items = ["Any"] + self._items if self.include_any else self._items
            self.fuzzy_completer = FuzzyCompleter(all_items, self)
            self.setCompleter(self.fuzzy_completer)

            # Connect completer activation
            self.fuzzy_completer.activated.connect(self._on_completer_activated)

    def addItems(self, items: List[str]):
        """Override to store items and set up fuzzy matching."""
        # Store items without "Any"
        self._items = [item for item in items if item != "Any"]

        # Add to combo box
        if self.include_any and "Any" not in items:
            super().addItem("Any")
        super().addItems(self._items)

        # Set up completer with all items
        self._setup_completer()

    def setItems(self, items: List[str]):
        """Set items, replacing existing ones."""
        self.clear()
        self.addItems(items)

    def _on_text_changed(self, text: str):
        """Handle text changes for fuzzy filtering."""
        if self.fuzzy_completer and text and text != "Any":
            self.fuzzy_completer.setFilterString(text)

    def _on_completer_activated(self, text: str):
        """Handle completer selection."""
        self.setCurrentText(text)
        self.fuzzy_match_selected.emit(text)

    def keyPressEvent(self, event: QKeyEvent):
        """
        Override to improve keyboard navigation.

        - Tab/Enter confirms current selection
        - Escape clears to "Any"
        """
        if event.key() == Qt.Key_Escape and self.include_any:
            self.setCurrentText("Any")
            event.accept()
        else:
            super().keyPressEvent(event)

    def value(self) -> Optional[str]:
        """
        Get the current value, returning None for "Any".

        Returns:
            Current text or None if "Any" is selected
        """
        text = self.currentText()
        return None if text == "Any" else text

    def setValue(self, value: Optional[str]):
        """
        Set the current value.

        Args:
            value: Text to set, or None/"Any" to clear
        """
        if value is None or value == "Any":
            if self.include_any:
                self.setCurrentText("Any")
            else:
                self.setCurrentText("")
        else:
            self.setCurrentText(value)


class FuzzySearchLineEdit(QLineEdit):
    """
    Enhanced QLineEdit with fuzzy matching auto-complete.

    Features:
    - Intelligent fuzzy matching
    - Inline completion suggestions
    - Smooth keyboard navigation

    Signals:
        fuzzy_match_selected: Emitted when a fuzzy match is selected
    """

    fuzzy_match_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize with proper font
        self.setFont(create_styled_font("body"))

        # Store suggestion items
        self._items = []

        # Set up fuzzy completer
        self.fuzzy_completer = None
        self._setup_completer()

    def _setup_completer(self):
        """Set up the fuzzy matching completer."""
        if self._items:
            self.fuzzy_completer = FuzzyCompleter(self._items, self)
            self.setCompleter(self.fuzzy_completer)

            # Connect completer activation
            self.fuzzy_completer.activated.connect(self._on_completer_activated)

            # Update filter on text change
            self.textChanged.connect(self._on_text_changed)

    def setSuggestions(self, items: List[str]):
        """Set the list of suggestions for auto-complete."""
        self._items = items
        self._setup_completer()

    def _on_text_changed(self, text: str):
        """Handle text changes for fuzzy filtering."""
        if self.fuzzy_completer and text:
            self.fuzzy_completer.setFilterString(text)

    def _on_completer_activated(self, text: str):
        """Handle completer selection."""
        self.setText(text)
        self.fuzzy_match_selected.emit(text)

    def keyPressEvent(self, event: QKeyEvent):
        """
        Override to improve keyboard navigation.

        - Escape clears the field
        """
        if event.key() == Qt.Key_Escape:
            self.clear()
            event.accept()
        else:
            super().keyPressEvent(event)

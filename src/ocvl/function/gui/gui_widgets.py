"""
widgets.py — Base reusable form widgets.

Replaces the widget portion of constructors.py. Complex dialogs
(FormatEditorWidget, ColorMapSelector, etc.) live in dialogs.py.
"""

import os

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QGridLayout, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QPushButton, QRadioButton, QSizePolicy,
    QTextEdit, QToolButton, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_numeric(text: str):
    """Return float/int if parseable, else the original string."""
    try:
        return int(text) if '.' not in text else float(text)
    except (ValueError, TypeError):
        return text


# ---------------------------------------------------------------------------
# OptionalField
# ---------------------------------------------------------------------------

class OptionalField(QWidget):
    """Wraps any widget with an enable/disable checkbox."""

    def __init__(self, widget, default_checked=True):
        super().__init__()
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(default_checked)

        self.inner_widget = widget
        self.inner_widget.setEnabled(default_checked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.inner_widget)
        layout.setAlignment(Qt.AlignLeft)

        self.checkbox.toggled.connect(self.inner_widget.setEnabled)

    def is_checked(self):
        return self.checkbox.isChecked()

    def get_widget(self):
        return self.inner_widget

    @property
    def field_widget(self):
        return self.inner_widget


# ---------------------------------------------------------------------------
# CollapsibleSection
# ---------------------------------------------------------------------------

# Accent colors per nesting depth (dark-theme friendly, muted)
_DEPTH_COLORS = [
    "#2d6a9f",   # depth 0 — steel blue
    "#2d8a6a",   # depth 1 — teal
    "#7a4d9f",   # depth 2 — purple
    "#9f7a2d",   # depth 3 — amber
    "#9f2d2d",   # depth 4 — rust (deep nesting, rare)
]

def _depth_color(depth: int) -> str:
    return _DEPTH_COLORS[depth % len(_DEPTH_COLORS)]


class CollapsibleSection(QWidget):
    """A collapsible section with a toggle-arrow header and left accent border."""

    def __init__(self, title="", default=None, depth=0, parent=None):
        super().__init__(parent)
        self._title = title
        self._depth = depth
        color = _depth_color(depth)

        self.toggle_button = QToolButton(self)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setStyleSheet(
            f"QToolButton {{ border: none; text-align: left; padding: 5px;"
            f" color: {color}; font-weight: bold; }}"
            f"QToolButton:disabled {{ color: gray; }}"
        )
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setIconSize(QSize(12, 12))
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.clicked.connect(self._toggle_content)

        # Header: [▶ Title] ... (add field / remove buttons injected by SectionWithAddButton)
        self.header = QWidget()
        header_layout = QHBoxLayout(self.header)
        header_layout.setSpacing(8)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self.toggle_button)
        header_layout.setAlignment(self.toggle_button, Qt.AlignLeft)

        # Content area: fixed padding from the accent line, consistent across all depths.
        # Depth-based indentation is applied to the CollapsibleSection container instead.
        self.content_area = QWidget()
        self.content_area.setObjectName("content_area")
        self.content_area.setVisible(False)
        self.content_area.setStyleSheet(
            f"QWidget#content_area {{"
            f" border-left: 3px solid {color};"
            f" padding-left: 10px;"
            f"}}"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        # Left margin increases with depth so nested sections indent visually,
        # but the spacing between the line and content stays fixed at 10px.
        layout.setContentsMargins(depth * 12, 0, 0, 0)
        layout.addWidget(self.header)
        layout.addWidget(self.content_area)

    def depth(self) -> int:
        return self._depth

    def title(self):
        return self._title

    def header_layout(self):
        return self.header.layout()

    def _toggle_content(self):
        visible = self.toggle_button.isChecked()
        self.content_area.setVisible(visible)
        self.toggle_button.setArrowType(Qt.DownArrow if visible else Qt.RightArrow)

    def set_content_layout(self, layout):
        if self.content_area.layout():
            old = self.content_area.layout()
            while old.count():
                item = old.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        self.content_area.setLayout(layout)

    def is_enabled(self):
        # Sections are always included when present — no checkbox to gate them.
        return True


# ---------------------------------------------------------------------------
# FreetextBox
# ---------------------------------------------------------------------------

class FreetextBox(QWidget):
    """Single- or multi-line free-text input."""

    def __init__(self, title=None, multi_line=False, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if multi_line:
            self.text_input = QTextEdit(self)
            self.text_input.setPlaceholderText(title or "")
        else:
            self.text_input = QLineEdit(self)
            self.text_input.setPlaceholderText(title or "")

        layout.addWidget(self.text_input)

    def set_text(self, text):
        self.text_input.setText(str(text) if text is not None else "")

    def get_text(self):
        text = (
            self.text_input.toPlainText()
            if isinstance(self.text_input, QTextEdit)
            else self.text_input.text()
        )
        return text if text else (self.text_input.placeholderText() or "")

    # Alias so widgets with get_value also work
    def get_value(self):
        return self.get_text()

    def set_value(self, val):
        self.set_text(val)


# ---------------------------------------------------------------------------
# freeFloat / freeInt
# ---------------------------------------------------------------------------

class freeFloat(QWidget):
    """Validated float input."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.textbox = QLineEdit(self)
        self.textbox.setValidator(QDoubleValidator(-999.9, 999.9, 2))
        layout.addWidget(self.textbox)

    def set_text(self, text):
        self.textbox.setText(str(text) if text is not None else "")

    def get_text(self):
        text = self.textbox.text()
        return None if not text or text == "null" else text

    def get_value(self):
        return self.get_text()

    def set_value(self, val):
        self.set_text(val)


class freeInt(QWidget):
    """Validated integer input."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.textbox = QLineEdit(self)
        self.textbox.setValidator(QIntValidator(-999, 999))
        layout.addWidget(self.textbox)

    def set_text(self, text):
        self.textbox.setText(str(text) if text is not None else "")

    def get_text(self):
        text = self.textbox.text()
        return None if not text or text == "null" else text

    def get_value(self):
        return self.get_text()

    def set_value(self, val):
        self.set_text(val)


# ---------------------------------------------------------------------------
# TrueFalseSelector
# ---------------------------------------------------------------------------

class TrueFalseSelector(QWidget):
    """Simple checkbox representing a boolean value."""

    def __init__(self, default_value=None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.checkbox = QCheckBox("")
        self.checkbox.setChecked(bool(default_value))
        layout.addWidget(self.checkbox)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def get_value(self):
        return self.checkbox.isChecked()

    def set_value(self, val):
        self.checkbox.setChecked(bool(val))


# ---------------------------------------------------------------------------
# AffineRigidSelector
# ---------------------------------------------------------------------------

class AffineRigidSelector(QWidget):
    """Radio-button pair for affine/rigid selection."""

    def __init__(self, default_value=None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setSpacing(30)

        self.true_button = QRadioButton("Affine")
        self.false_button = QRadioButton("Rigid")

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.true_button, 1)
        self.button_group.addButton(self.false_button, 0)

        if default_value:
            self.true_button.setChecked(True)
        else:
            self.false_button.setChecked(True)

        layout.addWidget(self.true_button)
        layout.addWidget(self.false_button)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def get_value(self):
        return "affine" if self.true_button.isChecked() else "rigid"

    def set_value(self, val):
        if val == "affine":
            self.true_button.setChecked(True)
        else:
            self.false_button.setChecked(True)


# ---------------------------------------------------------------------------
# DropdownMenu
# ---------------------------------------------------------------------------

class DropdownMenu(QWidget):
    """Thin wrapper around QComboBox with get/set helpers."""

    def __init__(self, default="", options=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.comboBox = QComboBox(self)
        self.comboBox.setFixedWidth(100)

        if isinstance(options, str):
            options = [options]

        self.comboBox.addItem(default)
        if options:
            for option in options:
                self.comboBox.addItem(option)

        layout.addWidget(self.comboBox)

    def get_value(self):
        return self.comboBox.currentText()

    def set_value(self, val):
        if val is None:
            return
        if self.comboBox.findText(str(val)) == -1:
            self.comboBox.addItem(str(val))
        self.comboBox.setCurrentText(str(val))

    def update_options(self, options, keep_selected=True):
        current_val = self.get_value() if keep_selected else None
        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        for option in options:
            self.comboBox.addItem(option)
        if keep_selected and current_val in options:
            self.comboBox.setCurrentText(current_val)
        self.comboBox.blockSignals(False)


# ---------------------------------------------------------------------------
# AlignmentModalitySelector
# ---------------------------------------------------------------------------

class AlignmentModalitySelector(QWidget):
    """Combo box that populates from a ListEditorWidget."""

    def __init__(self, modalities_list_creator, default_value="null", parent=None):
        super().__init__(parent)
        self.modalities_list_creator = modalities_list_creator
        self.default_value = default_value

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.comboBox = QComboBox(self)
        self.comboBox.setFixedWidth(120)
        self.update_options()
        layout.addWidget(self.comboBox)

        self.modalities_list_creator.edit_button.clicked.connect(self.update_options)

    def update_options(self):
        current_text = self.comboBox.currentText()
        self.comboBox.clear()
        self.comboBox.addItem("null")
        for modality in (self.modalities_list_creator.get_list() or []):
            self.comboBox.addItem(modality)
        if current_text in [self.comboBox.itemText(i) for i in range(self.comboBox.count())]:
            self.comboBox.setCurrentText(current_text)
        else:
            self.comboBox.setCurrentText(self.default_value)

    def get_value(self):
        text = self.comboBox.currentText()
        return None if text == "null" else text

    def set_value(self, value):
        self.comboBox.setCurrentText(value if value else "null")


# ---------------------------------------------------------------------------
# ListEditorWidget + ListEditorDialog
# ---------------------------------------------------------------------------

class ListEditorWidget(QWidget):
    itemsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("null")
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.edit_button = QPushButton("Edit List")
        self.edit_button.clicked.connect(self._show_editor)

        layout.addWidget(self.label)
        layout.addWidget(self.edit_button)

        self.items = []

    def _show_editor(self):
        dialog = ListEditorDialog(self.items, self)
        if dialog.exec() == QDialog.Accepted:
            self.items = dialog.get_items()
            self._update_label()
            self.itemsChanged.emit()

    def _update_label(self):
        self.label.setText(', '.join(self.items) if self.items else "null")

    def get_list(self):
        return self.items if self.items else None

    def set_value(self, val):
        self.items = list(val) if val else []
        self._update_label()
        self.itemsChanged.emit()

    # Alias
    def get_value(self):
        return self.get_list()


class ListEditorDialog(QDialog):
    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit List")
        self.setModal(True)

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.addItems(items)

        controls = QHBoxLayout()
        self.new_item_input = QLineEdit()
        self.new_item_input.setPlaceholderText("New item")
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_item)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_item)
        controls.addWidget(self.new_item_input)
        controls.addWidget(add_btn)
        controls.addWidget(remove_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(self.list_widget)
        layout.addLayout(controls)
        layout.addWidget(buttons)

    def _add_item(self):
        text = self.new_item_input.text().strip()
        if text:
            self.list_widget.addItem(text)
            self.new_item_input.clear()
            self.list_widget.scrollToBottom()

    def _remove_item(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)

    def get_items(self):
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]


# ---------------------------------------------------------------------------
# OpenFolder
# ---------------------------------------------------------------------------

class OpenFolder(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setSpacing(30)
        layout.setAlignment(Qt.AlignLeft)

        self.textbox = FreetextBox("null")
        self.textbox.setFixedWidth(250)
        self.button = QPushButton("Select Output Folder")
        self.button.setFixedWidth(150)
        self.button.clicked.connect(self._select_folder)

        layout.addWidget(self.textbox)
        layout.addWidget(self.button)

        self.folder_path = ""

    def _select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder", "", QFileDialog.ShowDirsOnly)
        if path:
            self.folder_path = path
            self.textbox.set_text(path)
        else:
            self.folder_path = ""
            self.textbox.set_text("null")

    def get_text(self):
        t = self.textbox.get_text()
        return None if t == "null" else t

    def set_text(self, val):
        self.textbox.set_text(val)

    def get_value(self):
        return self.get_text()

    def set_value(self, val):
        self.set_text(val)


# ---------------------------------------------------------------------------
# rangeSelector
# ---------------------------------------------------------------------------

class rangeSelector(QWidget):
    def __init__(self, def_min=None, def_max=None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        validator = QDoubleValidator(-999.99, 999.99, 2)

        self.min_box = QLineEdit(self)
        self.min_box.setFixedWidth(125)
        self.min_box.setValidator(validator)

        self.max_box = QLineEdit(self)
        self.max_box.setFixedWidth(125)
        self.max_box.setValidator(validator)

        layout.addWidget(self.min_box)
        layout.addWidget(self.max_box)
        layout.setAlignment(Qt.AlignLeft)

    def get_value(self):
        return [float(self.min_box.text()), float(self.max_box.text())]

    def set_value(self, value):
        if isinstance(value, str):
            parts = value.strip('[]').split(',')
            min_val, max_val = float(parts[0].strip()), float(parts[1].strip())
        else:
            min_val, max_val = float(value[0]), float(value[1])
        self.min_box.setText(str(min_val))
        self.max_box.setText(str(max_val))


# ---------------------------------------------------------------------------
# ColorMapSelector + ColorMapSelectorDialog
# ---------------------------------------------------------------------------

class ColorMapSelector(QWidget):
    def __init__(self, parent=None, cmap_def=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        self.chosen_label = QLabel(cmap_def or "None")
        self.chosen_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.chosen_image = QLabel()
        self.chosen_image.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.chosen_image.setFixedSize(180, 25)

        self.select_button = QPushButton("Select Colormap")
        self.select_button.clicked.connect(self._show_selector)

        layout.addWidget(self.chosen_label)
        layout.addWidget(self.chosen_image)
        layout.addWidget(self.select_button)
        layout.addStretch()

        self.setMaximumHeight(35)
        self.current_selection = cmap_def
        if cmap_def:
            self._update_selection(cmap_def)

    def _show_selector(self):
        dialog = ColorMapSelectorDialog(self.current_selection, self)
        if dialog.exec() == QDialog.Accepted:
            selection = dialog.get_selection()
            if selection:
                self.current_selection = selection
                self._update_selection(selection)

    def _update_selection(self, selection):
        self.chosen_label.setText(selection)
        image_path = f"images/colormap_options/{selection.lower()}.png"
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            self.chosen_image.setPixmap(
                pixmap.scaled(self.chosen_image.width(), self.chosen_image.height(),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def get_value(self):
        return self.current_selection.lower() if self.current_selection else None

    def set_value(self, val):
        self.current_selection = val
        if val:
            self._update_selection(val)
        else:
            self.chosen_label.setText("None")
            self.chosen_image.clear()


class ColorMapSelectorDialog(QDialog):
    COLORMAPS = ["Cividis", "Inferno", "Magma", "Plasma", "Viridis"]

    def __init__(self, current_selection=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Colormap")
        self.setModal(True)
        self.setFixedSize(550, 200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(8)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        from PySide6.QtGui import QFontMetrics
        max_width = max(QFontMetrics(self.font()).horizontalAdvance(c) for c in self.COLORMAPS) + 25

        grid = QGridLayout()
        grid.setHorizontalSpacing(15)
        grid.setVerticalSpacing(5)

        for row, cmap in enumerate(self.COLORMAPS):
            radio = QRadioButton(cmap)
            radio.setFixedWidth(max_width)
            if cmap == current_selection:
                radio.setChecked(True)
            self.button_group.addButton(radio)

            image_label = QLabel()
            image_path = f"images/colormap_options/{cmap.lower()}.png"
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                image_label.setPixmap(pixmap.scaled(300, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                image_label.setFixedSize(300, 30)

            grid.addWidget(radio, row, 0)
            grid.addWidget(image_label, row, 1)

        layout.addLayout(grid)
        layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selection(self):
        btn = self.button_group.checkedButton()
        return btn.text() if btn else None
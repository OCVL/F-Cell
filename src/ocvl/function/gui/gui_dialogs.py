"""
dialogs.py — Complex format-editor widgets.

Replaces the dialog/compound-widget portion of constructors.py.
Base widgets live in widgets.py.
"""

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMenu,
    QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# FormatElementsEditor  (the big drag-and-drop dialog)
# ---------------------------------------------------------------------------

class FormatElementsEditor(QDialog):
    """Dialog for building filename format strings from predefined elements."""

    copyRequested = Signal(str, str, str)

    _BASE_ELEMENTS = [
        "Day", "Eye", "FOV_Height", "FOV_Width",
        "IDnum", "LocX", "LocY", "Modality", "Month",
        "VidNum", "Year",
    ]
    _TOOLTIPS = {
        ":.1": "Fixed-length specifier (e.g. Year:.4 = always 4 chars).",
        "Day": "Calendar day (numerical).",
        "Eye": "Eye: OD/OS for right/left eye.",
        "FOV_Height": "Imaging field-of-view height.",
        "FOV_Width": "Imaging field-of-view width.",
        "IDnum": "Unique identifier for the image.",
        "LocX": "X-coordinate location.",
        "LocY": "Y-coordinate location.",
        "Modality": "Imaging modality (e.g., OCT, Fundus).",
        "Month": "Calendar month (numerical).",
        "VidNum": "Video number identifier.",
        "Year": "Calendar year (numerical).",
        "QueryLoc": "Query location identifier.",
    }
    _EXTENSIONS = {
        "image":    [".tif", ".png", ".jpg", ".mat", ".npy"],
        "video":    [".avi", ".mov", ".mat", ".npy"],
        "mask":     [".avi", ".mov", ".mat", ".npy"],
        "meta":     [".txt", ".json", ".xml", ".csv", ".log"],
        "queryloc": [".txt", ".csv", ".json", ".dat"],
    }

    def __init__(self, current_format=None, parent=None, type=None,
                 section_name=None, format_key=None,
                 enable_copy=True, show_extensions=True):
        super().__init__(parent)
        self.setWindowTitle("Format Editor")
        self.setGeometry(600, 600, 650, 500)

        self.type = type
        self.section_name = section_name
        self.format_key = format_key
        self.show_extensions = show_extensions

        self.original_elements = self._BASE_ELEMENTS.copy()
        if type == "queryloc":
            self.original_elements.append("QueryLoc")
        self.available_elements = self.original_elements.copy()

        self.existing_extension = None

        # ---- extension combo ----
        self.file_type_combo = None
        if show_extensions:
            self.file_type_combo = QComboBox()
            ext_options = self._EXTENSIONS.get(type or "", [".txt", ".dat", ".log"])
            self.file_type_combo.addItems(ext_options)
            self.file_type_combo.currentTextChanged.connect(self._update_preview)

        # ---- copy button ----
        self.copy_button = QPushButton("Copy to All in Section")
        self.copy_button.clicked.connect(self._copy_to_all)

        # ========== layout ==========
        window_layout = QVBoxLayout(self)

        # Preview row
        preview_layout = QHBoxLayout()
        preview_label = QLabel("Preview:")
        preview_label.setStyleSheet("font-weight: bold; padding: 6px;")
        self.preview_display = QLabel("")
        preview_layout.addWidget(preview_label)
        preview_layout.addWidget(self.preview_display)
        if self.file_type_combo is not None:
            preview_layout.addWidget(self.file_type_combo)
        if enable_copy:
            preview_layout.addWidget(self.copy_button)
        preview_layout.setAlignment(Qt.AlignLeft)
        window_layout.addLayout(preview_layout)

        # Main body: Available | Buttons | Selected
        body_layout = QHBoxLayout()
        window_layout.addLayout(body_layout)

        # Left: available
        left = QWidget()
        left_layout = QVBoxLayout(left)
        lbl = QLabel("Available Elements")
        lbl.setStyleSheet("font-weight: bold; padding: 6px;")
        left_layout.addWidget(lbl)
        self.available_list = QListWidget()
        self.available_list.addItems(self.available_elements)
        self.available_list.setMouseTracking(True)
        self.available_list.itemEntered.connect(self._update_tooltip)
        self.available_list.itemDoubleClicked.connect(self._dbl_click_available)
        self.available_list.leaveEvent = lambda e: self._clear_tooltip()
        left_layout.addWidget(self.available_list)
        body_layout.addWidget(left)

        # Center: transfer buttons
        mid = QWidget()
        mid_layout = QVBoxLayout(mid)
        mid_layout.setAlignment(Qt.AlignCenter)
        mid_layout.addStretch()
        self.add_btn = QPushButton("Add >>")
        self.remove_btn = QPushButton("<< Remove")
        self.width_btn = QPushButton("Set Width")
        for btn in (self.add_btn, self.remove_btn, self.width_btn):
            btn.setMinimumWidth(120)
            mid_layout.addWidget(btn)
        mid_layout.addStretch()
        body_layout.addWidget(mid)

        # Right: selected + side buttons
        right = QWidget()
        right_main = QHBoxLayout(right)

        right_list_panel = QWidget()
        right_list_layout = QVBoxLayout(right_list_panel)
        lbl2 = QLabel("Selected Format Elements")
        lbl2.setStyleSheet("font-weight: bold; padding: 6px;")
        right_list_layout.addWidget(lbl2)
        self.selected_list = QListWidget()
        self.selected_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.selected_list.customContextMenuRequested.connect(self._show_context_menu)
        self.selected_list.setMouseTracking(True)
        self.selected_list.itemEntered.connect(self._update_tooltip)
        self.selected_list.itemDoubleClicked.connect(self._dbl_click_selected)
        self.selected_list.leaveEvent = lambda e: self._clear_tooltip()
        right_list_layout.addWidget(self.selected_list)
        right_main.addWidget(right_list_panel)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setAlignment(Qt.AlignCenter)
        side_layout.addStretch()
        self.up_btn = QPushButton("Move Up")
        self.down_btn = QPushButton("Move Down")
        self.text_btn = QPushButton("Add Text")
        self.clear_btn = QPushButton("Clear All")
        for btn in (self.up_btn, self.down_btn, self.text_btn, self.clear_btn):
            btn.setMinimumWidth(120)
            side_layout.addWidget(btn)
        side_layout.addStretch()
        right_main.addWidget(side)
        body_layout.addWidget(right)

        body_layout.setStretchFactor(left, 1)
        body_layout.setStretchFactor(mid, 0)
        body_layout.setStretchFactor(right, 1)

        # Tooltip label
        self.tooltip_label = QLabel("Hover over an available element to see its description")
        self.tooltip_label.setWordWrap(True)
        self.tooltip_label.setStyleSheet(
            "QLabel { border: 1px solid #ccc; border-radius: 6px; background-color: #f0f0f0;"
            " padding: 6px; font-style: italic; color: #333; min-height: 40px; }"
        )
        window_layout.addWidget(self.tooltip_label)

        # Accept / Cancel
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        accept_btn = QPushButton("Accept")
        cancel_btn = QPushButton("Cancel")
        accept_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(accept_btn)
        btn_row.addWidget(cancel_btn)
        window_layout.addLayout(btn_row)

        # Connections
        self.add_btn.clicked.connect(self._add_selected_element)
        self.remove_btn.clicked.connect(self._remove_selected_element)
        self.width_btn.clicked.connect(self._set_element_width)
        self.up_btn.clicked.connect(self._move_up)
        self.down_btn.clicked.connect(self._move_down)
        self.text_btn.clicked.connect(self._add_separator)
        self.clear_btn.clicked.connect(self._clear_all)

        self.selected_list.model().rowsInserted.connect(self._update_preview)
        self.selected_list.model().rowsRemoved.connect(self._update_preview)
        self.selected_list.model().dataChanged.connect(self._update_preview)

        if current_format:
            self._parse_format(current_format)
        self._update_preview()

    # ---- internal item helpers ----

    def _add_to_selected(self, internal, display=None):
        item = QListWidgetItem()
        item.setData(Qt.UserRole, internal)
        item.setText(display if display is not None else internal)
        self.selected_list.addItem(item)

    @staticmethod
    def _internal(item):
        v = item.data(Qt.UserRole)
        return v if v else item.text()

    def _refresh_available(self):
        self.available_list.clear()
        self.available_list.addItems(sorted(set(self.available_elements)))

    # ---- parsing ----

    def _parse_format(self, format_string):
        # Strip trailing extension
        ext = self._detect_extension(format_string)
        if ext:
            self.existing_extension = ext
            self._add_existing_extension_to_dropdown(ext)
            format_string = format_string[: -len(ext)]

        remaining = format_string
        while remaining:
            start = remaining.find('{')
            if start == -1:
                if remaining:
                    self._add_to_selected(f"{{Added Text: {remaining}}}", remaining)
                break
            if start > 0:
                self._add_to_selected(f"{{Added Text: {remaining[:start]}}}", remaining[:start])
            end = remaining.find('}', start)
            if end == -1:
                self._add_to_selected(f"{{Added Text: {remaining}}}", remaining)
                break
            element = remaining[start + 1:end]
            if element in self.available_elements or element == ":.1":
                self._add_to_selected(f"{{{element}}}")
                if element in self.available_elements:
                    self.available_elements.remove(element)
                self._refresh_available()
            else:
                # Unknown element – treat as literal text with braces
                self._add_to_selected(f"{{Added Text: {{{element}}}}}", f"{{{element}}}")
            remaining = remaining[end + 1:]

    # ---- element manipulation ----

    def _add_selected_element(self):
        item = self.available_list.currentItem()
        if not item:
            QMessageBox.information(self, "No Selection", "Select an element from the Available list first.")
            return
        element = item.text()
        self._add_to_selected(f"{{{element}}}")
        self.available_list.takeItem(self.available_list.currentRow())
        self.available_elements.remove(element)
        self._update_preview()

    def _remove_selected_element(self):
        item = self.selected_list.currentItem()
        if not item:
            QMessageBox.information(self, "No Selection", "Select an element from the Selected list first.")
            return
        self._return_to_available(item)
        self.selected_list.takeItem(self.selected_list.currentRow())
        self._update_preview()

    def _return_to_available(self, item):
        text = self._internal(item)
        if text.startswith("{") and text.endswith("}") and not text.startswith("{Added Text:"):
            element = text[1:-1]
            base = element.split(':')[0] if ':' in element else element
            if base in self.original_elements:
                self.available_elements.append(base)
            elif element == ":.1":
                self.available_elements.append(element)
            self.available_elements = sorted(set(self.available_elements))
            self._refresh_available()

    def _dbl_click_available(self, item):
        element = item.text()
        self._add_to_selected(f"{{{element}}}")
        self.available_list.takeItem(self.available_list.row(item))
        self.available_elements.remove(element)
        self._update_preview()

    def _dbl_click_selected(self, item):
        self._return_to_available(item)
        self.selected_list.takeItem(self.selected_list.row(item))
        self._update_preview()

    def _move_up(self):
        row = self.selected_list.currentRow()
        if row <= 0:
            return
        item = self.selected_list.takeItem(row)
        self.selected_list.insertItem(row - 1, item)
        self.selected_list.setCurrentRow(row - 1)
        self._update_preview()

    def _move_down(self):
        row = self.selected_list.currentRow()
        if row < 0 or row >= self.selected_list.count() - 1:
            return
        item = self.selected_list.takeItem(row)
        self.selected_list.insertItem(row + 1, item)
        self.selected_list.setCurrentRow(row + 1)
        self._update_preview()

    def _add_separator(self):
        text, ok = QInputDialog.getText(self, "Add Text", "Enter text:")
        if ok:
            row = self.selected_list.currentRow()
            self._add_to_selected(f"{{Added Text: {text}}}", text)
            if row >= 0:
                self.selected_list.insertItem(row + 1, self.selected_list.takeItem(self.selected_list.count() - 1))
                self.selected_list.setCurrentRow(row + 1)
            self._update_preview()

    def _clear_all(self):
        for i in range(self.selected_list.count()):
            self._return_to_available(self.selected_list.item(i))
        self.selected_list.clear()
        self._update_preview()

    def _set_element_width(self):
        item = self.selected_list.currentItem()
        if not item:
            QMessageBox.information(self, "No Selection", "Select a format element first.")
            return
        text = self._internal(item)
        if not (text.startswith("{") and text.endswith("}")) or text.startswith("{Added Text:"):
            QMessageBox.information(self, "Invalid", "Width can only be set for format elements.")
            return
        element = text[1:-1]
        element_name = element.split(':')[0] if ':' in element else element
        width, ok = QInputDialog.getInt(self, "Set Width", f"Width for {element_name}:", 4, 1, 20)
        if ok:
            new_internal = f"{{{element_name}:{width}}}"
            item.setData(Qt.UserRole, new_internal)
            item.setText(new_internal)
            self._update_preview()

    # ---- preview ----

    def _build_format_string(self):
        parts = []
        for i in range(self.selected_list.count()):
            text = self._internal(self.selected_list.item(i))
            if text.startswith("{Added Text: ") and text.endswith("}"):
                parts.append(text[13:-1])
            else:
                parts.append(text)
        result = "".join(parts)
        if self.file_type_combo is not None:
            ext = self.file_type_combo.currentText()
            if ext and not self._detect_extension(result):
                result += ext
        return result

    def _update_preview(self):
        self.preview_display.setText(self._build_format_string())

    def get_format_string(self):
        return self._build_format_string()

    def get_formatted_preview(self):
        return self.preview_display.text()

    # ---- tooltip ----

    def _update_tooltip(self, item):
        if not item:
            self._clear_tooltip()
            return
        internal = item.data(Qt.UserRole) or item.text()
        if internal.startswith("{Added Text:"):
            self.tooltip_label.setText("Static text that always appears in the filename.")
            return
        if internal.startswith("{") and internal.endswith("}"):
            element = internal[1:-1].split(':')[0]
        else:
            element = internal
        tip = self._TOOLTIPS.get(element, "No description available.")
        self.tooltip_label.setText(f"{element}: {tip}")

    def _clear_tooltip(self):
        self.tooltip_label.setText("Hover over an element to see its description")

    # ---- context menu ----

    def _show_context_menu(self, position):
        item = self.selected_list.itemAt(position)
        if not item:
            return
        menu = QMenu()
        edit_action = QAction("Edit", self)
        remove_action = QAction("Remove", self)
        menu.addAction(edit_action)
        menu.addAction(remove_action)
        action = menu.exec(self.selected_list.mapToGlobal(position))
        if action == edit_action:
            text = self._internal(item)
            if text.startswith("{Added Text: ") and text.endswith("}"):
                old = text[13:-1]
                new, ok = QInputDialog.getText(self, "Edit Text", "Edit:", text=old)
                if ok:
                    item.setData(Qt.UserRole, f"{{Added Text: {new}}}")
                    item.setText(new)
        elif action == remove_action:
            self._remove_selected_element()
        self._update_preview()

    # ---- copy to all ----

    def _copy_to_all(self):
        reply = QMessageBox.question(
            self, "Confirm Copy",
            "Copy this format to all others in this section?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.copyRequested.emit(self.section_name, self.format_key, self.get_format_string())
            self.accept()

    # ---- extension helpers ----

    @staticmethod
    def _detect_extension(format_string):
        if not format_string:
            return None
        known = ['.tif', '.png', '.jpg', '.jpeg', '.avi', '.mov', '.mat', '.npy',
                 '.txt', '.json', '.xml', '.csv', '.log', '.dat']
        for ext in known:
            if format_string.lower().endswith(ext):
                return ext
        m = re.search(r'\.([a-zA-Z0-9]{2,5})$', format_string)
        return ('.' + m.group(1)) if m else None

    def _add_existing_extension_to_dropdown(self, extension):
        if self.file_type_combo is None:
            return
        existing = [self.file_type_combo.itemText(i) for i in range(self.file_type_combo.count())]
        if extension and extension not in existing:
            self.file_type_combo.insertItem(0, extension)
            self.file_type_combo.insertSeparator(1)
            self.file_type_combo.setCurrentText(extension)


# ---------------------------------------------------------------------------
# FormatEditorWidget  (inline widget that opens FormatElementsEditor)
# ---------------------------------------------------------------------------

class FormatEditorWidget(QWidget):
    """Preview label + Edit button; opens FormatElementsEditor on click."""

    formatChanged = Signal(str)
    copyToAllRequested = Signal(str, str, str)

    def __init__(self, label_text, default_format="", parent=None,
                 type=None, section_name=None, format_key=None):
        super().__init__(parent)
        self.label_text = label_text
        self.default_format = default_format
        self.current_format = default_format
        self.return_text = default_format
        self.type = type
        self.section_name = section_name
        self.format_key = format_key

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.format_label = QLabel(default_format)
        self.edit_button = QPushButton("Edit...")
        self.edit_button.clicked.connect(self._open_editor)

        layout.addWidget(self.format_label)
        layout.addWidget(self.edit_button)

    def _open_editor(self):
        dialog = FormatElementsEditor(
            self.current_format, self, self.type,
            self.section_name, self.format_key,
        )
        dialog.copyRequested.connect(self._relay_copy)
        if dialog.exec() == QDialog.Accepted:
            self.return_text = dialog.get_format_string()
            self.current_format = dialog.get_formatted_preview()
            self.format_label.setText(self.current_format)
            self.formatChanged.emit(self.return_text)

    def _relay_copy(self, section, key, fmt):
        self.copyToAllRequested.emit(section, key, fmt)

    def get_value(self):
        return self.return_text

    def set_value(self, val):
        self.return_text = val or ""
        self.current_format = val or ""
        self.format_label.setText(self.current_format)
        if val:
            self.formatChanged.emit(val)


# ---------------------------------------------------------------------------
# GroupByFormatEditorWidget
# ---------------------------------------------------------------------------

class GroupByFormatEditorWidget(FormatEditorWidget):
    """Format editor whose available elements are derived from other format strings."""

    def __init__(self, image_format, video_format, mask_format,
                 label_text, default_format="", parent=None):
        super().__init__(label_text, default_format, parent)
        self.image_format = image_format or None
        self.video_format = video_format or None
        self.mask_format = mask_format or None
        self.available_elements = self._get_dynamic_elements()

        if self.current_format == "null":
            self.current_format = ""
            self.format_label.setText("")

    def _get_dynamic_elements(self):
        elements = set()
        for fmt in (self.image_format, self.video_format, self.mask_format):
            if fmt and fmt != "null":
                for m in re.finditer(r"{(.*?)}", fmt):
                    token = m.group(1)
                    if not token.startswith("Added Text:"):
                        elements.add(token)
        elements.discard('')
        return sorted(elements)

    def update_format_sources(self, image_format, video_format, mask_format):
        self.image_format = image_format or None
        self.video_format = video_format or None
        self.mask_format = mask_format or None
        self.available_elements = self._get_dynamic_elements()

        current_fields = re.findall(r"{(.*?)}", self.current_format) if self.current_format else []
        if any(f not in self.available_elements for f in current_fields):
            self.current_format = ""
            self.return_text = "null"
            self.format_label.setText("")
            self.formatChanged.emit("null")

    def _open_editor(self):
        self.available_elements = self._get_dynamic_elements()
        current = None if self.current_format == "null" else self.current_format
        dialog = FormatElementsEditor(current, self, enable_copy=False, show_extensions=False)
        dialog.original_elements = self.available_elements.copy()
        dialog.available_elements = self.available_elements.copy()
        dialog.available_list.clear()
        dialog.available_list.addItems(dialog.available_elements)

        if dialog.exec() == QDialog.Accepted:
            new_fmt = dialog.get_format_string()
            self.return_text = "null" if not new_fmt else new_fmt
            self.current_format = self.return_text
            self.format_label.setText("" if not new_fmt else new_fmt)
            self.formatChanged.emit(self.return_text)

    def set_value(self, val):
        if not val or val == "null":
            self.current_format = "null"
            self.return_text = "null"
            self.format_label.setText("")
        else:
            self.current_format = val
            self.return_text = val
            self.format_label.setText(val)


# ---------------------------------------------------------------------------
# SaveasExtensionsEditorWidget + SaveasExtensionsEditor
# ---------------------------------------------------------------------------

class SaveasExtensionsEditor(QDialog):
    """Dialog for selecting file-save extensions (e.g. .png, .tiff)."""

    _OPTIONS = [".png", ".tiff", ".svg", ".ps", ".pgf", ".pgm"]

    def __init__(self, current_format=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save-As Extensions Editor")
        self.setGeometry(500, 500, 750, 400)

        self.available_elements = self._OPTIONS.copy()

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Available Extensions"))
        self.available_list = QListWidget()
        self.available_list.addItems(self.available_elements)
        left_layout.addWidget(self.available_list)

        # Center
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addStretch()
        add_btn = QPushButton("Add >>")
        add_btn.clicked.connect(self._add)
        remove_btn = QPushButton("<< Remove")
        remove_btn.clicked.connect(self._remove)
        center_layout.addWidget(add_btn)
        center_layout.addWidget(remove_btn)
        center_layout.addStretch()

        # Right
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Selected Extensions"))
        self.selected_list = QListWidget()
        self.selected_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.selected_list.customContextMenuRequested.connect(self._context_menu)
        right_layout.addWidget(self.selected_list)

        # Preview
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Preview:"))
        self.preview_text = QLabel("")
        preview_layout.addWidget(self.preview_text)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)

        main_v = QVBoxLayout()
        main_v.addWidget(splitter)
        main_v.addLayout(preview_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_v.addWidget(buttons)

        # Replace the horizontal layout with vertical
        while layout.count():
            layout.takeAt(0)
        for i in range(main_v.count()):
            item = main_v.itemAt(i)
            if item.widget():
                layout.addWidget(item.widget())

        if current_format:
            self._load(current_format)

    def _load(self, fmt):
        for ext in self._OPTIONS:
            if ext in fmt:
                self.selected_list.addItem(ext)
        self._update_preview()

    def _add(self):
        item = self.available_list.currentItem()
        if item:
            self.selected_list.addItem(item.text())
            self.available_list.takeItem(self.available_list.currentRow())
            self._update_preview()

    def _remove(self):
        row = self.selected_list.currentRow()
        if row >= 0:
            self.selected_list.takeItem(row)
            self._update_preview()

    def _update_preview(self):
        items = [self.selected_list.item(i).text() for i in range(self.selected_list.count())]
        self.preview_text.setText(str(items))

    def _context_menu(self, position):
        item = self.selected_list.itemAt(position)
        if not item:
            return
        menu = QMenu()
        remove_action = QAction("Remove", self)
        menu.addAction(remove_action)
        action = menu.exec(self.selected_list.mapToGlobal(position))
        if action == remove_action:
            self.selected_list.takeItem(self.selected_list.row(item))
            self._update_preview()

    def get_format_string(self):
        return self.preview_text.text()


class SaveasExtensionsEditorWidget(QWidget):
    """Inline widget opening SaveasExtensionsEditor."""

    formatChanged = Signal(str)

    def __init__(self, label_text, default_format="", parent=None):
        super().__init__(parent)
        self.label_text = label_text
        self.default_format = default_format
        self.current_format = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.format_label = QLabel(default_format)
        self.edit_button = QPushButton("Edit...")
        self.edit_button.clicked.connect(self._open_editor)

        layout.addWidget(self.format_label)
        layout.addWidget(self.edit_button)

    def _open_editor(self):
        dialog = SaveasExtensionsEditor(self.current_format, self)
        if dialog.exec() == QDialog.Accepted:
            self.current_format = dialog.get_format_string()
            self.format_label.setText(self.current_format)
            self.formatChanged.emit(self.current_format)

    def get_value(self):
        fmt = self.format_label.text()
        if not fmt or fmt == "[]":
            return []
        if fmt.startswith('[') and fmt.endswith(']'):
            fmt = fmt[1:-1]
        if fmt.strip():
            result = []
            for elem in fmt.split(","):
                elem = elem.strip().strip('"\'')
                if elem:
                    result.append(elem)
            return result
        return []

    def set_value(self, format_string):
        self.current_format = format_string or ""
        self.format_label.setText(self.current_format)

    def reset_to_default(self):
        self.set_value(self.default_format)
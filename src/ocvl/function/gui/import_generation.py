""" MASTER_GUI Format
{
    "type":        "<widget_type>",  # required, indicates field type
    "required":    true/false,       # prevents remove button from appearing
    "description": "...",            # shown in Add picker
    "save":        true/false,       # store in saved_widgets for dependency wiring
    "format_type": "image"|...       # for formatEditor widgets (image, video, mask, etc.)
}
"""

from src.ocvl.function.gui.gui_widgets import (
    AffineRigidSelector, CollapsibleSection,
    ColorMapSelector, DropdownMenu, FreetextBox, freeFloat, freeInt,
    ListEditorWidget, OpenFolder, OptionalField, rangeSelector, TrueFalseSelector,
    _depth_color,
)
from src.ocvl.function.gui.gui_dialogs import (
    FormatEditorWidget, GroupByFormatEditorWidget, SaveasExtensionsEditorWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_label(key: str) -> str:
    return key.replace('_', ' ').title()


def extract_widget_type(field_def):
    if isinstance(field_def, dict):
        return field_def.get("type")
    return field_def


def _is_subsection(tmpl_val) -> bool:
    if not isinstance(tmpl_val, dict):
        return False
    type_val = tmpl_val.get("type")
    # If "type" is absent or is itself a dict, this is a subsection
    return type_val is None or isinstance(type_val, dict)


def _is_required(tmpl_val) -> bool:
    return bool(tmpl_val.get("required", False)) if isinstance(tmpl_val, dict) else False


def coerce_value(value, widget_type: str):
    """Convert raw widget output to the correct Python/JSON type."""
    if value is None:
        return None

    if isinstance(value, str):
        low = value.lower()
        if low == "null" or low == "":
            return None
        if low == "true":
            return True
        if low == "false":
            return False
        # freeText fields must stay as strings — never coerce to numbers.
        # version, description, and any other text field could look numeric.
        if widget_type == "freeText":
            return value
        try:
            if widget_type == "freeInt":
                return int(value)
            if widget_type == "freeFloat":
                return float(value)
            return int(value) if '.' not in value else float(value)
        except (ValueError, TypeError):
            return value

    if widget_type == "freeInt" and isinstance(value, (int, float)):
        return int(value)
    if widget_type == "freeFloat" and isinstance(value, (int, float)):
        return float(value)
    if widget_type == "trueFalse":
        return bool(value)
    if widget_type == "null":
        return None

    return value


# ---------------------------------------------------------------------------
# _REMOVE_BTN_STYLE  — shared style for all remove buttons
# ---------------------------------------------------------------------------

_REMOVE_BTN_STYLE = (
    "QPushButton { color: #cc0000; border: none; padding: 0px 4px;"
    " font-size: 11px; background: transparent; }"
    "QPushButton:hover { color: #ff0000; text-decoration: underline; }"
)


# ---------------------------------------------------------------------------
# FieldRow  — label + field widget + optional remove button (no checkbox)
# ---------------------------------------------------------------------------

class FieldRow(QWidget):
    """
    [Label:]  [field_widget]  [✕]   (✕ absent for required fields)

    The remove button calls a parent-supplied callback so the owning
    SectionWithAddButton can return the field to the available pool.
    """

    def __init__(self, key: str, label_text: str, field_widget: QWidget,
                 required: bool = False,
                 on_remove=None,
                 parent=None):
        super().__init__(parent)
        self._key = key
        self._field_widget = field_widget
        self._required = required
        self._on_remove = on_remove

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignLeft)

        self._label = QLabel(label_text + ':')
        layout.addWidget(self._label)
        layout.addWidget(field_widget)

        if not required and on_remove is not None:
            self._remove_btn = QPushButton("✕")
            self._remove_btn.setStyleSheet(_REMOVE_BTN_STYLE)
            self._remove_btn.setToolTip(f"Remove '{label_text}' from this section")
            self._remove_btn.setFixedWidth(20)
            self._remove_btn.clicked.connect(self._do_remove)
            layout.addWidget(self._remove_btn)

    def _do_remove(self):
        if self._on_remove:
            self._on_remove(self._key, self)

    @property
    def key(self) -> str:
        return self._key

    @property
    def field_widget(self) -> QWidget:
        return self._field_widget

    @property
    def label_text(self) -> str:
        return self._label.text().rstrip(':')



# ---------------------------------------------------------------------------
# _PickerDialog  — base for field and section pickers
# ---------------------------------------------------------------------------

class _PickerDialog(QDialog):
    """Base list-picker used by AddFieldDialog and AddSectionDialog."""

    def __init__(self, available: dict, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(420, 300)
        self._available = available

        layout = QVBoxLayout(self)
        header = QLabel("Select items to add:")
        header.setStyleSheet("font-weight: bold; margin-bottom: 4px;")
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self._list)

        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet(
            "color: #555; font-style: italic; padding: 4px;"
            "border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9;"
        )
        self._desc_label.setMinimumHeight(36)
        layout.addWidget(self._desc_label)
        self._list.currentItemChanged.connect(self._on_selection_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self, items: dict):
        for key, tmpl_val in items.items():
            label = format_label(key)
            description = tmpl_val.get("description", "") if isinstance(tmpl_val, dict) else ""
            display = f"{label}  —  {description}" if description else label
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, key)
            self._list.addItem(item)

    def _on_selection_changed(self, current, _previous):
        if not current:
            self._desc_label.setText("")
            return
        key = current.data(Qt.UserRole)
        tmpl_val = self._available.get(key, {})
        desc = tmpl_val.get("description", "") if isinstance(tmpl_val, dict) else ""
        self._desc_label.setText(desc or "No description available.")

    def selected_keys(self) -> list:
        return [item.data(Qt.UserRole) for item in self._list.selectedItems()]


class AddFieldDialog(_PickerDialog):
    """Picker showing only leaf fields (non-subsections)."""
    def __init__(self, available: dict, parent=None):
        super().__init__(available, "Add Field", parent)
        self._populate({k: v for k, v in available.items() if not _is_subsection(v)})


class AddSectionDialog(_PickerDialog):
    """Picker showing only subsections."""
    def __init__(self, available: dict, parent=None):
        super().__init__(available, "Add Section", parent)
        self._populate({k: v for k, v in available.items() if _is_subsection(v)})


# ---------------------------------------------------------------------------
# SectionWithAddButton
# ---------------------------------------------------------------------------

class SectionWithAddButton(QWidget):
    """
    Wraps a CollapsibleSection with:

    Header row  : [checkbox] [▶ Title]  [+ Add Field]  [✕ Remove]
                   The "+ Add Field" button opens a picker for leaf fields
                   only. Shown only when leaf fields are available in the pool.

    Content footer: [+ Add Section]
                   At the bottom of the content area. Opens a picker for
                   subsections only. Shown only when subsections are available.

    depth drives accent color and propagates to child sections.
    """

    def __init__(self, section: CollapsibleSection,
                 available: dict,
                 template: dict,
                 template_key_order: list,
                 parent_name: str,
                 saved_widgets: dict,
                 is_required: bool = False,
                 on_remove=None,
                 section_key: str = "",
                 depth: int = 0,
                 parent=None):
        super().__init__(parent)
        self._section = section
        self._available = available
        self._template = template
        self._template_key_order = template_key_order
        self._parent_name = parent_name
        self._saved_widgets = saved_widgets
        self._is_required = is_required
        self._on_remove = on_remove
        self._section_key = section_key
        self._depth = depth

        color = _depth_color(depth)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(section)

        # ---- Header injections ----
        hdr = section.header_layout()

        # "+ Add Field" immediately after the toggle — left-aligned
        self._add_field_btn = QPushButton("+ Add Field")
        self._add_field_btn.setStyleSheet(
            f"QPushButton {{ color: {color}; border: 1px solid {color};"
            f" border-radius: 3px; padding: 2px 7px; font-size: 11px; background: transparent; }}"
            f"QPushButton:hover {{ background: {color}22; }}"
            f"QPushButton:disabled {{ color: #555; border-color: #555; }}"
        )
        self._add_field_btn.setFixedHeight(22)
        self._add_field_btn.clicked.connect(self._open_field_picker)
        self._add_field_btn.setVisible(self._has_available_fields())
        hdr.addWidget(self._add_field_btn, alignment=Qt.AlignVCenter)

        # Stretch pushes Remove to the far right
        hdr.addStretch()

        # "✕ Remove" right-aligned
        if not is_required and on_remove is not None:
            remove_btn = QPushButton("✕ Remove Section")
            remove_btn.setStyleSheet(_REMOVE_BTN_STYLE)
            remove_btn.setToolTip(f"Remove the '{format_label(section_key)}' section")
            remove_btn.setFixedHeight(22)
            remove_btn.clicked.connect(self._do_remove)
            hdr.addWidget(remove_btn, alignment=Qt.AlignVCenter)

        # ---- Content footer — "+ Add Section" for subsections only ----
        content_layout = section.content_area.layout()
        if content_layout:
            self._section_line = QFrame()
            self._section_line.setFrameShape(QFrame.HLine)
            self._section_line.setStyleSheet(f"color: {color};")
            content_layout.addWidget(self._section_line)

            self._add_section_btn = QPushButton("+ Add Section")
            self._add_section_btn.setStyleSheet(
                f"QPushButton {{ color: {color}; border: none; text-align: left;"
                f" padding: 4px 8px; font-size: 12px; }}"
                f"QPushButton:hover {{ text-decoration: underline; }}"
            )
            self._add_section_btn.clicked.connect(self._open_section_picker)
            content_layout.addWidget(self._add_section_btn)

            self._update_section_btn_visibility()

    # ------------------------------------------------------------------ #
    # Visibility helpers
    # ------------------------------------------------------------------ #

    def _has_available_fields(self) -> bool:
        return any(not _is_subsection(v) for v in self._available.values())

    def _has_available_sections(self) -> bool:
        return any(_is_subsection(v) for v in self._available.values())

    def _update_field_btn_visibility(self):
        self._add_field_btn.setVisible(self._has_available_fields())

    def _update_section_btn_visibility(self):
        has = self._has_available_sections()
        self._add_section_btn.setVisible(has)
        self._section_line.setVisible(has)

    # ------------------------------------------------------------------ #
    # Pickers
    # ------------------------------------------------------------------ #

    def _open_field_picker(self):
        field_pool = {k: v for k, v in self._available.items() if not _is_subsection(v)}
        if not field_pool:
            return
        dialog = AddFieldDialog(field_pool, self)
        if dialog.exec() != QDialog.Accepted:
            return
        self._insert_items(dialog.selected_keys())

    def _open_section_picker(self):
        section_pool = {k: v for k, v in self._available.items() if _is_subsection(v)}
        if not section_pool:
            return
        dialog = AddSectionDialog(section_pool, self)
        if dialog.exec() != QDialog.Accepted:
            return
        self._insert_items(dialog.selected_keys())

    def _insert_items(self, keys: list):
        if not keys:
            return
        content_layout = self._section.content_area.layout()
        # Footer is: [...items..., section_line, add_section_btn]
        footer_count = 2

        for key in keys:
            tmpl_val = self._available.pop(key, None)
            if tmpl_val is None:
                continue
            widget = self._build_item(key, tmpl_val)
            if widget is None:
                continue
            insert_pos = self._find_insert_pos(key, content_layout, footer_count)
            content_layout.insertWidget(insert_pos, widget)

        self._update_field_btn_visibility()
        self._update_section_btn_visibility()

    # ------------------------------------------------------------------ #
    # Shared helpers (unchanged from before)
    # ------------------------------------------------------------------ #

    @property
    def collapsable(self) -> CollapsibleSection:
        return self._section

    def _do_remove(self):
        if self._on_remove:
            self._on_remove(self._section_key, self)

    def _find_insert_pos(self, new_key: str, content_layout, footer_count: int) -> int:
        rendered_keys = []
        for i in range(content_layout.count() - footer_count):
            item = content_layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, FieldRow):
                rendered_keys.append(w.key)
            elif isinstance(w, SectionWithAddButton):
                rendered_keys.append(w._section_key)
        last_pos = 0
        for tmpl_key in self._template_key_order:
            if tmpl_key == new_key:
                break
            if tmpl_key in rendered_keys:
                last_pos = rendered_keys.index(tmpl_key) + 1
        return last_pos

    def _build_item(self, key: str, tmpl_val):
        if _is_subsection(tmpl_val):
            return self._build_subsection(key, tmpl_val)
        return _build_field_row(
            key=key,
            tmpl_val=tmpl_val,
            data_val=None,
            parent_name=self._parent_name,
            saved_widgets=self._saved_widgets,
            on_remove=self._return_field_to_pool,
        )

    def _build_subsection(self, key: str, tmpl_val: dict):
        current_parent = f"{self._parent_name}_{key}" if self._parent_name else key
        child_depth = self._depth + 1
        inner = build_form_from_template(tmpl_val, {}, current_parent, self._saved_widgets, child_depth)
        section = CollapsibleSection(title=format_label(key) + ":", default=True, depth=child_depth)
        section.set_content_layout(inner.layout())
        available = getattr(inner, '_available_fields', {})
        return SectionWithAddButton(
            section=section,
            available=available,
            template=tmpl_val,
            template_key_order=list(tmpl_val.keys()),
            parent_name=current_parent,
            saved_widgets=self._saved_widgets,
            is_required=False,
            on_remove=self._return_field_to_pool,
            section_key=key,
            depth=child_depth,
        )

    def _return_field_to_pool(self, key: str, widget: QWidget):
        content_layout = self._section.content_area.layout()
        idx = content_layout.indexOf(widget)
        if idx >= 0:
            content_layout.takeAt(idx)
            widget.setParent(None)
            widget.deleteLater()
        self._available[key] = self._template.get(key, {})
        self._update_field_btn_visibility()
        self._update_section_btn_visibility()


# ---------------------------------------------------------------------------
# Widget factory
# ---------------------------------------------------------------------------

def _create_format_editor(field_key: str, widget_spec: dict):
    fmt_type = (widget_spec or {}).get("format_type")
    return FormatEditorWidget(label_text=format_label(field_key), default_format="", type=fmt_type)


WIDGET_FACTORY = {
    "freeText":                      lambda config=None: FreetextBox(),
    "freeFloat":                     lambda config=None: freeFloat(),
    "freeInt":                       lambda config=None: freeInt(),
    "trueFalse":                     lambda config=None: TrueFalseSelector(),
    "comboBox":                      lambda config=None: DropdownMenu(default="null"),
    "outputSubfolderMethodComboBox": lambda config=None: DropdownMenu(options=["DateTime", "Date", "Sequential"]),
    "shapeComboBox":                 lambda config=None: DropdownMenu(default="null", options=["disk", "box"]),
    "summaryComboBox":               lambda config=None: DropdownMenu(default="null", options=["mean", "median"]),
    "typeComboBox":                  lambda config=None: DropdownMenu(default="null", options=["stim-relative", "absolute"]),
    "unitsComboBox":                 lambda config=None: DropdownMenu(default="null", options=["time", "frames"]),
    "normComboBox":                  lambda config=None: DropdownMenu(default="score", options=["mean", "median", "none"]),
    "standardizationMethodComboBox": lambda config=None: DropdownMenu(
                                         default="null",
                                         options=["mean_stddev", "stddev", "linear_stddev",
                                                  "linear_vast", "relative_change", "none"]),
    "summaryMethodComboBox":         lambda config=None: DropdownMenu(default="null", options=["rms", "stddev", "var", "avg"]),
    "controlComboBox":               lambda config=None: DropdownMenu(default="null", options=["none", "subtraction", "division"]),
    "listEditor":                    lambda config=None: ListEditorWidget(),
    "openFolder":                    lambda config=None: OpenFolder(),
    "formatEditor":                  lambda key, spec=None: _create_format_editor(key, spec or {}),
    "groupbyEditor":                 lambda config=None: GroupByFormatEditorWidget(None, None, None, "Group By"),
    "cmapSelector":                  lambda config=None: ColorMapSelector(),
    "affineRigidSelector":           lambda config=None: AffineRigidSelector(),
    "saveasSelector":                lambda config=None: SaveasExtensionsEditorWidget("Save as"),
    "rangeSelector":                 lambda config=None: rangeSelector(),
    "null":                          lambda config=None: QLabel("null"),
}


# ---------------------------------------------------------------------------
# _build_field_row  — builds a single FieldRow for a leaf field
# ---------------------------------------------------------------------------

def _build_field_row(key: str, tmpl_val, data_val,
                     parent_name: str, saved_widgets: dict,
                     on_remove=None) -> "FieldRow | None":
    """
    Build a FieldRow for a leaf field. Returns None if no widget can be made.
    Subsection dicts (no "type" key) are rejected here — they must go through
    the subsection path in build_form_from_template.
    """
    # Guard: reject subsection dicts — they have no "type"
    if _is_subsection(tmpl_val):
        return None

    if isinstance(tmpl_val, dict):
        widget_type = tmpl_val.get("type")
        is_req      = bool(tmpl_val.get("required", False))
        save_widget = bool(tmpl_val.get("save", False))
    else:
        widget_type = tmpl_val
        is_req      = False
        save_widget = False

    # Infer widget type from data value when template doesn't specify
    if not widget_type or widget_type not in WIDGET_FACTORY:
        if isinstance(data_val, bool):
            widget_type = "trueFalse"
        elif isinstance(data_val, (int, float)):
            widget_type = "freeText"
        elif isinstance(data_val, list):
            widget_type = "listEditor"
        elif data_val is None:
            widget_type = "null"
        else:
            widget_type = "freeText"

    widget_constructor = WIDGET_FACTORY.get(widget_type)
    if not widget_constructor:
        return None

    if widget_type == "formatEditor":
        field_widget = widget_constructor(key, tmpl_val if isinstance(tmpl_val, dict) else {})
    else:
        field_widget = widget_constructor()

    if isinstance(field_widget, FormatEditorWidget):
        field_widget.section_name = parent_name
        field_widget.format_key = key
        field_widget.copyToAllRequested.connect(
            lambda s, k, v, sw=saved_widgets: propagate_advanced_copy(sw, s, k, v)
        )

    if data_val is not None:
        _set_widget_value(field_widget, data_val, widget_type)

    if save_widget and parent_name:
        saved_widgets[f"{parent_name}_{key}"] = field_widget

    return FieldRow(
        key=key,
        label_text=format_label(key),
        field_widget=field_widget,
        required=is_req,
        on_remove=on_remove if not is_req else None,
    )


def _set_widget_value(field_widget, val, widget_type: str):
    """Push an imported value into a field widget. Skips None so defaults are preserved."""
    if val is None:
        return   # leave the widget at its default rather than pushing "null"
    if hasattr(field_widget, "set_value"):
        if isinstance(val, bool):
            field_widget.set_value(val)
        elif isinstance(val, list) and isinstance(field_widget, ListEditorWidget):
            field_widget.set_value(val)
        elif isinstance(val, (list, dict)):
            field_widget.set_value(str(val))
        else:
            field_widget.set_value(str(val))
    elif hasattr(field_widget, "set_text"):
        field_widget.set_text(str(val))


# ---------------------------------------------------------------------------
# _TopLevelAddBar  — add bar for the outermost form level
# ---------------------------------------------------------------------------

class _TopLevelAddBar(QWidget):
    """
    A persistent "+ Add Section" bar at the bottom of the top-level form.
    Handles only top-level sections (preanalysis, analysis, raw, etc.).
    Top-level leaf fields (version, description) are always required so
    they never end up in this pool in practice.
    """

    def __init__(self, available: dict, template: dict,
                 template_key_order: list, parent_name: str,
                 saved_widgets: dict, form_layout, parent=None):
        super().__init__(parent)
        self._available = available
        self._template = template
        self._template_key_order = template_key_order
        self._parent_name = parent_name
        self._saved_widgets = saved_widgets
        self._form_layout = form_layout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 4)
        layout.setSpacing(2)

        self._line = QFrame()
        self._line.setFrameShape(QFrame.HLine)
        self._line.setStyleSheet("color: #444;")
        layout.addWidget(self._line)

        self._btn = QPushButton("+ Add Section")
        self._btn.setStyleSheet(
            "QPushButton { color: #007ACC; border: none; text-align: left;"
            " padding: 4px 8px; font-size: 12px; }"
            "QPushButton:hover { text-decoration: underline; }"
        )
        self._btn.clicked.connect(self._open_picker)
        layout.addWidget(self._btn)

        # Only show when there are sections to add
        has_sections = any(_is_subsection(v) for v in available.values())
        self.setVisible(has_sections)

    def _refresh(self):
        has_sections = any(_is_subsection(v) for v in self._available.values())
        self.setVisible(has_sections)

    def _open_picker(self):
        section_pool = {k: v for k, v in self._available.items() if _is_subsection(v)}
        if not section_pool:
            return
        dialog = AddSectionDialog(section_pool, self)
        if dialog.exec() != QDialog.Accepted:
            return
        keys = dialog.selected_keys()
        if not keys:
            return

        bar_idx = self._form_layout.indexOf(self)
        for key in keys:
            tmpl_val = self._available.pop(key, None)
            if tmpl_val is None:
                continue
            widget = self._build_item(key, tmpl_val)
            if widget is None:
                continue
            insert_pos = self._find_insert_pos(key, bar_idx)
            self._form_layout.insertWidget(insert_pos, widget)

        self._refresh()

    def _find_insert_pos(self, new_key: str, bar_idx: int) -> int:
        rendered_keys = []
        for i in range(bar_idx):
            item = self._form_layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, SectionWithAddButton):
                rendered_keys.append(w._section_key)
            elif isinstance(w, FieldRow):
                rendered_keys.append(w.key)
        last_pos = 0
        for tmpl_key in self._template_key_order:
            if tmpl_key == new_key:
                break
            if tmpl_key in rendered_keys:
                last_pos = rendered_keys.index(tmpl_key) + 1
        return last_pos

    def _return_item_to_pool(self, key: str, widget: QWidget):
        idx = self._form_layout.indexOf(widget)
        if idx >= 0:
            self._form_layout.takeAt(idx)
            widget.setParent(None)
            widget.deleteLater()
        self._available[key] = self._template.get(key, {})
        self._refresh()

    def _build_item(self, key: str, tmpl_val):
        if _is_subsection(tmpl_val):
            current_parent = f"{self._parent_name}_{key}" if self._parent_name else key
            inner = build_form_from_template(tmpl_val, {}, current_parent, self._saved_widgets, depth=1)
            section = CollapsibleSection(title=format_label(key) + ":", default=True, depth=0)
            section.set_content_layout(inner.layout())
            sub_available = getattr(inner, '_available_fields', {})
            return SectionWithAddButton(
                section=section,
                available=sub_available,
                template=tmpl_val,
                template_key_order=list(tmpl_val.keys()),
                parent_name=current_parent,
                saved_widgets=self._saved_widgets,
                is_required=False,
                on_remove=self._return_item_to_pool,
                section_key=key,
                depth=0,
            )
        return _build_field_row(
            key=key,
            tmpl_val=tmpl_val,
            data_val=None,
            parent_name=self._parent_name,
            saved_widgets=self._saved_widgets,
            on_remove=self._return_item_to_pool,
        )


# ---------------------------------------------------------------------------
# Form builder
# ---------------------------------------------------------------------------

def build_form_from_template(template: dict, data: dict,
                              parent_name: str = "",
                              saved_widgets: dict = None,
                              depth: int = 0) -> QWidget:
    """
    Build a form from *template*, rendering only fields/sections present in
    *data* (or marked required). Each CollapsibleSection is wrapped in a
    SectionWithAddButton giving access to the remaining items.
    """
    if saved_widgets is None:
        saved_widgets = {}

    form_container = QWidget()
    form_layout = QVBoxLayout(form_container)
    form_layout.setSpacing(20)
    form_layout.setContentsMargins(15, 15, 15, 15)

    # Collect the pool of absent optional items up front.
    available_fields: dict = {}
    for key, tmpl_val in template.items():
        req = _is_required(tmpl_val)
        present = isinstance(data, dict) and key in data
        if not present and not req:
            available_fields[key] = tmpl_val

    def _return_to_pool(key: str, widget: QWidget):
        """Remove a widget from the top-level layout and return its key to the pool."""
        idx = form_layout.indexOf(widget)
        if idx >= 0:
            form_layout.takeAt(idx)
            widget.setParent(None)
            widget.deleteLater()
        available_fields[key] = template.get(key, {})
        if depth == 0:
            top_bar._refresh()

    for key, tmpl_val in template.items():
        req     = _is_required(tmpl_val)
        present = isinstance(data, dict) and key in data

        if not present and not req:
            continue   # in the pool, not rendered yet

        # ---- Subsection ----
        if _is_subsection(tmpl_val):
            sub_data = data.get(key, {}) if isinstance(data, dict) else {}
            current_parent = f"{parent_name}_{key}" if parent_name else key

            inner = build_form_from_template(tmpl_val, sub_data, current_parent, saved_widgets, depth + 1)

            if inner.layout().count() > 0:
                section = CollapsibleSection(
                    title=format_label(key) + ":",
                    default=True,
                    depth=depth,
                )
                section.set_content_layout(inner.layout())
                sub_available = getattr(inner, '_available_fields', {})
                wrapped = SectionWithAddButton(
                    section=section,
                    available=sub_available,
                    template=tmpl_val,
                    template_key_order=list(tmpl_val.keys()),
                    parent_name=current_parent,
                    saved_widgets=saved_widgets,
                    is_required=req,
                    on_remove=_return_to_pool if not req else None,
                    section_key=key,
                    depth=depth,
                )
                form_layout.addWidget(wrapped)

            if key == "pipeline_params" and parent_name == "preanalysis":
                setup_preanalysis_dependencies(saved_widgets, parent_name)
            elif key == "analysis_params" and parent_name == "analysis":
                setup_analysis_dependencies(saved_widgets, parent_name)

            continue

        # ---- Leaf field ----
        row = _build_field_row(
            key=key,
            tmpl_val=tmpl_val,
            data_val=data.get(key) if isinstance(data, dict) else None,
            parent_name=parent_name,
            saved_widgets=saved_widgets,
            on_remove=_return_to_pool if not req else None,
        )
        if row is not None:
            form_layout.addWidget(row)

    # Top-level add bar — only at the outermost form level (depth 0).
    # Recursive calls manage their footer via SectionWithAddButton instead.
    if depth == 0:
        top_bar = _TopLevelAddBar(
            available=available_fields,
            template=template,
            template_key_order=list(template.keys()),
            parent_name=parent_name,
            saved_widgets=saved_widgets,
            form_layout=form_layout,
        )
        form_layout.addWidget(top_bar)

    form_container._available_fields = available_fields   # type: ignore[attr-defined]
    return form_container


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------

def propagate_advanced_copy(saved_widgets, section, source_key, format_string):
    for saved_key, widget in saved_widgets.items():
        if not isinstance(widget, FormatEditorWidget):
            continue
        if widget.section_name == section and widget.format_key != source_key:
            widget.set_value(format_string)


def setup_preanalysis_dependencies(saved_widgets, parent_name):
    image_format = saved_widgets.get(f"{parent_name}_image_format")
    video_format = saved_widgets.get(f"{parent_name}_video_format")
    mask_format  = saved_widgets.get(f"{parent_name}_mask_format")
    modalities   = saved_widgets.get(f"{parent_name}_pipeline_params_modalities")

    format_widgets = [w for w in [image_format, video_format, mask_format] if w is not None]

    def has_modality_token():
        return any(
            isinstance(w.get_value(), str) and "{Modality}" in w.get_value()
            for w in format_widgets if hasattr(w, 'get_value')
        )

    def update_modalities_enabled():
        if modalities:
            enabled = has_modality_token()
            modalities.setEnabled(enabled)
            if not enabled:
                modalities.set_value([])

    for w in format_widgets:
        if hasattr(w, 'formatChanged'):
            w.formatChanged.connect(update_modalities_enabled)

    update_modalities_enabled()

    alignment_ref = saved_widgets.get(f"{parent_name}_pipeline_params_alignment_reference_modality")

    def update_alignment_options():
        if modalities and alignment_ref and hasattr(alignment_ref, 'update_options'):
            alignment_ref.update_options(modalities.get_list() or [])

    if modalities:
        modalities.itemsChanged.connect(update_alignment_options)

    update_alignment_options()

    groupby = saved_widgets.get(f"{parent_name}_pipeline_params_group_by")

    def update_groupby_sources():
        if groupby and hasattr(groupby, 'update_format_sources'):
            groupby.update_format_sources(
                image_format.get_value() if image_format else "",
                video_format.get_value() if video_format else "",
                mask_format.get_value()  if mask_format  else "",
            )

    for w in format_widgets:
        if hasattr(w, 'formatChanged'):
            w.formatChanged.connect(update_groupby_sources)

    update_groupby_sources()


def setup_analysis_dependencies(saved_widgets, parent_name):
    image_format    = saved_widgets.get(f"{parent_name}_image_format")
    video_format    = saved_widgets.get(f"{parent_name}_video_format")
    queryloc_format = saved_widgets.get(f"{parent_name}_queryloc_format")
    modalities      = saved_widgets.get(f"{parent_name}_analysis_params_modalities")

    format_widgets = [w for w in [image_format, video_format, queryloc_format] if w is not None]

    def has_modality_token():
        return any(
            isinstance(w.get_value(), str) and "{Modality}" in w.get_value()
            for w in format_widgets if hasattr(w, 'get_value')
        )

    def update_modalities_enabled():
        if modalities:
            enabled = has_modality_token()
            modalities.setEnabled(enabled)
            if not enabled:
                modalities.set_value([])

    for w in format_widgets:
        if hasattr(w, 'formatChanged'):
            w.formatChanged.connect(update_modalities_enabled)

    update_modalities_enabled()


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def generate_json(form_container: QWidget, template: dict,
                  skip_disabled: bool = True) -> dict:
    """
    Walk the form widget tree and build a JSON-serialisable dict.
    All present FieldRows and enabled CollapsibleSections are included.
    """

    def walk_layout(layout, template_for_layout: dict) -> dict:
        result = {}
        if not layout:
            return result

        # template_for_layout may be None if a parent section had no template entry
        if not isinstance(template_for_layout, dict):
            template_for_layout = {}

        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if not widget:
                continue

            # SectionWithAddButton — unwrap to inner CollapsibleSection
            if isinstance(widget, SectionWithAddButton):
                widget = widget.collapsable

            # CollapsibleSection → recurse
            if isinstance(widget, CollapsibleSection):
                if skip_disabled and not widget.is_enabled():
                    continue

                section_key = widget.title().replace(':', '').replace(' ', '_').lower()
                content_layout = widget.content_area.layout()
                if not content_layout:
                    continue

                # Use `or {}` so a None value in the template is treated as empty
                section_template = template_for_layout.get(section_key) or {}
                section_data = walk_layout(content_layout, section_template)
                if section_data:
                    result[section_key] = section_data
                continue

            # FieldRow → single leaf field (always included — no checkbox)
            if isinstance(widget, FieldRow):
                key = widget.key
                field_widget = widget.field_widget

                widget_type_def = template_for_layout.get(key)
                widget_type = extract_widget_type(widget_type_def) if widget_type_def else None

                if not widget_type or not isinstance(widget_type, str) \
                        or widget_type not in WIDGET_FACTORY:
                    continue

                raw = _read_widget_value(field_widget)
                result[key] = coerce_value(raw, widget_type)
                continue

            # Legacy plain row — kept for backwards compat
            row_layout = widget.layout()
            if not row_layout or row_layout.count() < 2:
                continue

            label_widget = row_layout.itemAt(0).widget()
            field_widget  = row_layout.itemAt(1).widget()

            if not isinstance(label_widget, QLabel):
                continue

            key = label_widget.text().replace(':', '').replace(' ', '_').lower()

            if isinstance(field_widget, OptionalField):
                if skip_disabled and not field_widget.is_checked():
                    continue
                field_widget = field_widget.field_widget

            widget_type_def = template_for_layout.get(key)
            widget_type = extract_widget_type(widget_type_def) if widget_type_def else None

            if not widget_type or widget_type not in WIDGET_FACTORY:
                continue

            raw = _read_widget_value(field_widget)
            result[key] = coerce_value(raw, widget_type)

        return result

    return walk_layout(form_container.layout(), template)


def _read_widget_value(field_widget):
    """Extract the current value from any field widget."""
    if hasattr(field_widget, 'get_value'):
        return field_widget.get_value()
    if hasattr(field_widget, 'get_text'):
        return field_widget.get_text()
    if hasattr(field_widget, 'get_list'):
        return field_widget.get_list()
    if hasattr(field_widget, 'currentText'):
        return field_widget.currentText()
    if hasattr(field_widget, 'text'):
        return field_widget.text()
    if hasattr(field_widget, 'isChecked'):
        return field_widget.isChecked()
    if isinstance(field_widget, QLabel):
        return field_widget.text()
    return None
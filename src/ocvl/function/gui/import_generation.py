"""
import_generation.py — Form builder and JSON extractor for advanced/import modes.

Design
------
- build_form_from_template iterates the *template* for structure but only
  renders fields/sections present in *data* or marked "required".
- Absent optional items (both leaf fields and subsections) are collected into
  a pool attached to each section via SectionWithAddButton.
- The add button label says "+ Add Field" or "+ Add Section" depending on
  what's in the pool. If both kinds are present it says "+ Add Field / Section".
- Items added from the pool are inserted at the position matching their order
  in the master template, keeping the form consistent.
- Every non-required FieldRow has a small "✕" remove button on the right.
  Removing a field returns it to the section's available pool.
- Every non-required SectionWithAddButton has a small "✕ Remove Section"
  button next to its header. Removing a section returns it to the parent pool.
- generate_json skips nothing by presence — it just walks whatever is
  currently in the layout.

Template leaf format
--------------------
{
    "type":        "<widget_type>",  # required
    "required":    true/false,       # prevents remove button from appearing
    "description": "...",            # shown in Add picker
    "save":        true/false,       # store in saved_widgets for dependency wiring
    "format_type": "image"|...       # for formatEditor widgets
}
"""

import re

from src.ocvl.function.gui.gui_widgets import (
    AffineRigidSelector, AlignmentModalitySelector, CollapsibleSection,
    ColorMapSelector, DropdownMenu, FreetextBox, freeFloat, freeInt,
    ListEditorWidget, OpenFolder, OptionalField, rangeSelector, TrueFalseSelector,
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
    """
    True when tmpl_val represents a nested section, not a leaf field.

    A leaf field has "type" as a plain string, e.g. {"type": "freeText"}.
    A subsection either has no "type" key at all, or has "type" as a nested
    dict (meaning "type" is itself a child field named "type", not a widget
    type specifier). Examples from master_JSON.json:
        {"type": "freeText"}           → leaf   (type is a string)
        {"type": {"type": "freeText"}} → section (type is a dict — it's a field)
        {"framestamps": {...}}         → section (no "type" key at all)
    """
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
        if low == "null":
            return None
        if low == "true":
            return True
        if low == "false":
            return False
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
        layout.setSpacing(10)
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
# AddItemDialog  — picker for fields AND subsections not yet in a section
# ---------------------------------------------------------------------------

class AddItemDialog(QDialog):
    """
    Lists available leaf fields and subsections that can be added.
    Each item shows the human-readable label and optional description.
    Multi-select supported.
    """

    def __init__(self, available: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Field / Section")
        self.setMinimumSize(440, 320)
        self._available = available

        layout = QVBoxLayout(self)

        header = QLabel("Select items to add:")
        header.setStyleSheet("font-weight: bold; margin-bottom: 4px;")
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.ExtendedSelection)

        for key, tmpl_val in available.items():
            label = format_label(key)
            is_sub = _is_subsection(tmpl_val)
            description = ""
            if isinstance(tmpl_val, dict):
                description = tmpl_val.get("description", "")

            tag = " [Section]" if is_sub else ""
            display = f"{label}{tag}"
            if description:
                display += f"  —  {description}"

            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, key)
            self._list.addItem(item)

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


# ---------------------------------------------------------------------------
# SectionWithAddButton
# ---------------------------------------------------------------------------

class SectionWithAddButton(QWidget):
    """
    Wraps a CollapsibleSection with:
    - A "+ Add Field / Section" button inside the content area (footer)
    - An optional "✕ Remove Section" button next to the section header

    Parameters
    ----------
    section          : CollapsibleSection
    available        : dict { key: tmpl_val } — pool of addable items
    template         : full subsection template (for building new widgets)
    template_key_order: list of keys in master template order
    parent_name      : dot-path key prefix
    saved_widgets    : shared dependency-wiring dict
    is_required      : if True, no remove button shown
    on_remove        : callback(key, widget) called when this section is removed
    section_key      : this section's own key in its parent template
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Optional "✕ Remove Section" button in its own header row
        if not is_required and on_remove is not None:
            header_row = QWidget()
            header_layout = QHBoxLayout(header_row)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.addWidget(section, stretch=1)
            remove_btn = QPushButton("✕ Remove Section")
            remove_btn.setStyleSheet(_REMOVE_BTN_STYLE)
            remove_btn.setToolTip(f"Remove the '{format_label(section_key)}' section")
            remove_btn.clicked.connect(self._do_remove)
            header_layout.addWidget(remove_btn, alignment=Qt.AlignTop)
            outer.addWidget(header_row)
        else:
            outer.addWidget(section)

        # "+ Add ..." button inside the section's content area
        self._add_btn = QPushButton(self._add_btn_label())
        self._add_btn.setStyleSheet(
            "QPushButton { color: #007ACC; border: none; text-align: left;"
            " padding: 4px 8px; font-size: 12px; }"
            "QPushButton:hover { text-decoration: underline; }"
        )
        self._add_btn.clicked.connect(self._open_picker)
        self._add_btn.setVisible(bool(available))

        content_layout = section.content_area.layout()
        if content_layout:
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setStyleSheet("color: #ddd;")
            content_layout.addWidget(line)
            content_layout.addWidget(self._add_btn)

        section.enable_checkbox.toggled.connect(
            lambda checked: self._add_btn.setVisible(checked and bool(self._available))
        )

    def _add_btn_label(self) -> str:
        has_fields = any(not _is_subsection(v) for v in self._available.values())
        has_sects  = any(_is_subsection(v) for v in self._available.values())
        if has_fields and has_sects:
            return "+ Add Field / Section"
        if has_sects:
            return "+ Add Section"
        return "+ Add Field"

    @property
    def collapsable(self) -> CollapsibleSection:
        return self._section

    def _do_remove(self):
        if self._on_remove:
            self._on_remove(self._section_key, self)

    def _open_picker(self):
        if not self._available:
            return
        dialog = AddItemDialog(self._available, self)
        if dialog.exec() != QDialog.Accepted:
            return
        keys = dialog.selected_keys()
        if not keys:
            return

        content_layout = self._section.content_area.layout()
        # The layout has: [...existing items..., separator, add_btn]
        # We want to insert before the separator (last 2 items).
        footer_count = 2

        for key in keys:
            tmpl_val = self._available.pop(key, None)
            if tmpl_val is None:
                continue

            widget = self._build_item(key, tmpl_val)
            if widget is None:
                continue

            # Find the correct insertion position based on master template order
            insert_pos = self._find_insert_pos(key, content_layout, footer_count)
            content_layout.insertWidget(insert_pos, widget)

        self._add_btn.setText(self._add_btn_label())
        self._add_btn.setVisible(bool(self._available))

    def _find_insert_pos(self, new_key: str, content_layout, footer_count: int) -> int:
        """
        Return the layout index where new_key should be inserted to maintain
        master template ordering.
        """
        # Build an ordered list of keys currently rendered in the layout
        rendered_keys = []
        for i in range(content_layout.count() - footer_count):
            item = content_layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, FieldRow):
                rendered_keys.append(w.key)
            elif isinstance(w, SectionWithAddButton):
                rendered_keys.append(w._section_key)

        # Walk the master template order and find where new_key slots in
        last_pos = 0
        for tmpl_key in self._template_key_order:
            if tmpl_key == new_key:
                break
            if tmpl_key in rendered_keys:
                last_pos = rendered_keys.index(tmpl_key) + 1

        return last_pos

    def _build_item(self, key: str, tmpl_val):
        """Build the appropriate widget for a field or subsection."""
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
        """Build a nested SectionWithAddButton for a subsection being added."""
        current_parent = f"{self._parent_name}_{key}" if self._parent_name else key
        inner = build_form_from_template(tmpl_val, {}, current_parent, self._saved_widgets)
        section = CollapsibleSection(title=format_label(key) + ":", default=True)
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
        )

    def _return_field_to_pool(self, key: str, widget: QWidget):
        """Remove a field/section widget from the layout and return key to pool."""
        content_layout = self._section.content_area.layout()
        idx = content_layout.indexOf(widget)
        if idx >= 0:
            content_layout.takeAt(idx)
            widget.setParent(None)
            widget.deleteLater()

        # Put the key back in the pool using original template value
        self._available[key] = self._template.get(key, {})
        self._add_btn.setText(self._add_btn_label())
        self._add_btn.setVisible(True)


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
    """Push an imported value into a field widget."""
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
# Form builder
# ---------------------------------------------------------------------------

def build_form_from_template(template: dict, data: dict,
                              parent_name: str = "",
                              saved_widgets: dict = None) -> QWidget:
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

    # Collect the pool of absent optional items before rendering anything,
    # so the SectionWithAddButton gets the full pool from the start.
    available_fields: dict = {}
    for key, tmpl_val in template.items():
        req = _is_required(tmpl_val)
        present = isinstance(data, dict) and key in data
        if not present and not req:
            available_fields[key] = tmpl_val

    # Callback passed down to FieldRows/SectionWithAddButtons so they can
    # return themselves to the available pool and be re-addable.
    # We bind it after the SectionWithAddButton is created (see below).
    parent_section_ref: list = [None]  # mutable container for late binding

    def _return_to_pool(key: str, widget: QWidget):
        content_layout = form_layout  # top-level: items sit directly in form_layout
        idx = content_layout.indexOf(widget)
        if idx >= 0:
            content_layout.takeAt(idx)
            widget.setParent(None)
            widget.deleteLater()
        available_fields[key] = template.get(key, {})
        if parent_section_ref[0] is not None:
            parent_section_ref[0]._available = available_fields
            parent_section_ref[0]._add_btn.setText(parent_section_ref[0]._add_btn_label())
            parent_section_ref[0]._add_btn.setVisible(True)

    for key, tmpl_val in template.items():
        req     = _is_required(tmpl_val)
        present = isinstance(data, dict) and key in data

        if not present and not req:
            continue   # goes into the pool, not rendered yet

        # ---- Subsection ----
        if _is_subsection(tmpl_val):
            sub_data = data.get(key, {}) if isinstance(data, dict) else {}
            current_parent = f"{parent_name}_{key}" if parent_name else key

            inner = build_form_from_template(tmpl_val, sub_data, current_parent, saved_widgets)

            if inner.layout().count() > 0:
                section = CollapsibleSection(
                    title=format_label(key) + ":",
                    default=True,
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

                section_template = template_for_layout.get(section_key, {})
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
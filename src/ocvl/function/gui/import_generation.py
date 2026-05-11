"""
import_generation.py — Form builder and JSON extractor for advanced/import modes.

Changes from the original:
- Dead QLabel stub entries removed from WIDGET_FACTORY (they were never reached).
- Value coercion extracted into a standalone coerce_value() function.
- format_label / extract_widget_type helpers unchanged.
- build_form_from_template / generate_json logic unchanged.
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
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget
from PySide6.QtCore import Qt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_label(key: str) -> str:
    return key.replace('_', ' ').title()


def extract_widget_type(field_def):
    if isinstance(field_def, dict):
        return field_def.get("type")
    return field_def


def coerce_value(value, widget_type: str):
    """
    Convert a raw widget value to the appropriate Python type for JSON output.

    Widgets return their natural Python types where possible, but some
    (FreetextBox, freeInt, freeFloat) always return strings, so we convert
    them here based on the declared widget_type.
    """
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
        # Numeric coercion for text-based widgets
        try:
            if widget_type == "freeInt":
                return int(value)
            if widget_type == "freeFloat":
                return float(value)
            # Generic: try int then float
            return int(value) if '.' not in value else float(value)
        except (ValueError, TypeError):
            return value  # keep as string

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
# Widget factory
# ---------------------------------------------------------------------------

def _create_format_editor(field_key: str, widget_spec: dict):
    fmt_type = (widget_spec or {}).get("format_type")
    return FormatEditorWidget(label_text=format_label(field_key), default_format="", type=fmt_type)


WIDGET_FACTORY = {
    # Text / numeric
    "freeText":   lambda config=None: FreetextBox(),
    "freeFloat":  lambda config=None: freeFloat(),
    "freeInt":    lambda config=None: freeInt(),
    # Boolean
    "trueFalse":  lambda config=None: TrueFalseSelector(),
    # Dropdowns
    "comboBox":                       lambda config=None: DropdownMenu(default="null"),
    "outputSubfolderMethodComboBox":  lambda config=None: DropdownMenu(options=["DateTime", "Date", "Sequential"]),
    "shapeComboBox":                  lambda config=None: DropdownMenu(default="null", options=["disk", "box"]),
    "summaryComboBox":                lambda config=None: DropdownMenu(default="null", options=["mean", "median"]),
    "typeComboBox":                   lambda config=None: DropdownMenu(default="null", options=["stim-relative", "absolute"]),
    "unitsComboBox":                  lambda config=None: DropdownMenu(default="null", options=["time", "frames"]),
    "normComboBox":                   lambda config=None: DropdownMenu(default="score", options=["mean", "median", "none"]),
    "standardizationMethodComboBox":  lambda config=None: DropdownMenu(
                                          default="null",
                                          options=["mean_stddev", "stddev", "linear_stddev",
                                                   "linear_vast", "relative_change", "none"]),
    "summaryMethodComboBox":          lambda config=None: DropdownMenu(default="null", options=["rms", "stddev", "var", "avg"]),
    "controlComboBox":                lambda config=None: DropdownMenu(default="null", options=["none", "subtraction", "division"]),
    # Compound widgets
    "listEditor":         lambda config=None: ListEditorWidget(),
    "openFolder":         lambda config=None: OpenFolder(),
    "formatEditor":       lambda key, spec=None: _create_format_editor(key, spec or {}),
    "groupbyEditor":      lambda config=None: GroupByFormatEditorWidget(None, None, None, "Group By"),
    "cmapSelector":       lambda config=None: ColorMapSelector(),
    "affineRigidSelector":lambda config=None: AffineRigidSelector(),
    "saveasSelector":     lambda config=None: SaveasExtensionsEditorWidget("Save as"),
    "rangeSelector":      lambda config=None: rangeSelector(),
    "null":               lambda config=None: QLabel("null"),
}


# ---------------------------------------------------------------------------
# Form builder
# ---------------------------------------------------------------------------

def build_form_from_template(template: dict, data: dict, adv=False,
                              parent_name="", saved_widgets=None) -> QWidget:
    if saved_widgets is None:
        saved_widgets = {}

    form_container = QWidget()
    form_layout = QVBoxLayout(form_container)
    form_layout.setSpacing(30)
    form_layout.setContentsMargins(15, 15, 15, 15)

    for key, val in data.items():
        if isinstance(val, dict):
            template_for_key = template.get(key, {})
            current_parent = f"{parent_name}_{key}" if parent_name else key
            inner_widget = build_form_from_template(template_for_key, val, adv, current_parent, saved_widgets)

            if inner_widget.layout().count() > 0:
                section = CollapsibleSection(title=format_label(key) + ":", default=True)
                section.set_content_layout(inner_widget.layout())
                form_layout.addWidget(section)

            if key == "pipeline_params" and parent_name == "preanalysis":
                setup_preanalysis_dependencies(saved_widgets, parent_name)
            elif key == "analysis_params" and parent_name == "analysis":
                setup_analysis_dependencies(saved_widgets, parent_name)

            continue

        widget_def = template.get(key, {})
        if isinstance(widget_def, dict):
            widget_type = widget_def.get("type")
            save_widget = widget_def.get("save", False)
            dependencies = widget_def.get("dependencies")
        else:
            widget_type = widget_def
            save_widget = False
            dependencies = None

        # Fall back to type-inference when no widget type declared
        if not widget_type or widget_type not in WIDGET_FACTORY:
            if isinstance(val, bool):
                widget_type = "trueFalse"
            elif isinstance(val, (int, float)):
                widget_type = "freeText"
            elif isinstance(val, list):
                widget_type = "listEditor"
            elif val is None:
                widget_type = "null"
            else:
                widget_type = "freeText"

        widget_constructor = WIDGET_FACTORY.get(widget_type)
        if not widget_constructor:
            continue

        if widget_type == "formatEditor":
            field_widget = widget_constructor(key, widget_def if isinstance(widget_def, dict) else {})
        else:
            field_widget = widget_constructor()

        if isinstance(field_widget, FormatEditorWidget):
            field_widget.section_name = parent_name
            field_widget.format_key = key
            field_widget.copyToAllRequested.connect(
                lambda s, k, v, sw=saved_widgets: propagate_advanced_copy(sw, s, k, v)
            )

        # Set initial value
        if val is not None:
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

        if save_widget and parent_name:
            saved_widgets[f"{parent_name}_{key}"] = field_widget

        # Build row
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(15)
        row_layout.setAlignment(Qt.AlignLeft)
        row_layout.addWidget(QLabel(format_label(key) + ':'))
        row_layout.addWidget(field_widget)
        row_widget.setLayout(row_layout)

        if adv:
            form_layout.addWidget(OptionalField(row_widget))
        else:
            form_layout.addWidget(row_widget)

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

    # Alignment reference dropdown
    alignment_ref = saved_widgets.get(f"{parent_name}_pipeline_params_alignment_reference_modality")

    def update_alignment_options():
        if modalities and alignment_ref and hasattr(alignment_ref, 'update_options'):
            values = modalities.get_list() or []
            alignment_ref.update_options(values)

    if modalities:
        modalities.itemsChanged.connect(update_alignment_options)

    update_alignment_options()

    # Group-by widget
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

def generate_json(form_container, template, skip_disabled=True):
    """
    Walk the form widget tree and build a JSON-serialisable dict.
    """

    def walk_layout(layout, template_for_layout):
        result = {}
        if not layout:
            return result

        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if not widget:
                continue

            # Collapsible section → nested object
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

            # Regular row widget
            row_layout = widget.layout()
            if not row_layout or row_layout.count() < 2:
                continue

            label_widget = row_layout.itemAt(0).widget()
            field_widget = row_layout.itemAt(1).widget()

            if not isinstance(label_widget, QLabel):
                continue

            key = label_widget.text().replace(':', '').replace(' ', '_').lower()

            # Unwrap OptionalField
            if isinstance(field_widget, OptionalField):
                if skip_disabled and not field_widget.is_checked():
                    continue
                field_widget = field_widget.field_widget

            widget_type_def = template_for_layout.get(key)
            widget_type = extract_widget_type(widget_type_def) if widget_type_def else None

            if not widget_type or not isinstance(widget_type, str) or widget_type not in WIDGET_FACTORY:
                continue

            # Pull raw value
            raw = None
            if hasattr(field_widget, 'get_value'):
                raw = field_widget.get_value()
            elif hasattr(field_widget, 'get_text'):
                raw = field_widget.get_text()
            elif hasattr(field_widget, 'get_list'):
                raw = field_widget.get_list()
            elif hasattr(field_widget, 'currentText'):
                raw = field_widget.currentText()
            elif hasattr(field_widget, 'text'):
                raw = field_widget.text()
            elif hasattr(field_widget, 'isChecked'):
                raw = field_widget.isChecked()
            elif isinstance(field_widget, QLabel):
                raw = field_widget.text()

            result[key] = coerce_value(raw, widget_type)

        return result

    return walk_layout(form_container.layout(), template)
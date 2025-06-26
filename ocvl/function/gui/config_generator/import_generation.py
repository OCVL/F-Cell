import json
import sys
import re

from constructors import *
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget, QWizardPage, QScrollArea, QMainWindow, \
    QApplication
from PySide6.QtCore import Qt

def format_label(key: str) -> str:
    return key.replace('_', ' ').title()

def extract_widget_type(field_def):
    if isinstance(field_def, dict):
        return field_def.get("type")
    return field_def

WIDGET_FACTORY = {
    "freeText": lambda: FreetextBox(),
    "freeNumber": lambda: freeNumber(),  # or use a QSpinBox/DoubleSpinBox if you make one
    "trueFalse": lambda: TrueFalseSelector(),
    "comboBox": lambda: DropdownMenu(default="null"),
    "outputSubfolderMethodComboBox": lambda: DropdownMenu(options=["DateTime", "Date", "Sequential"]),
    "shapeComboBox": lambda: DropdownMenu(default="null", options=["disk", "box"]),
    "summaryComboBox": lambda: DropdownMenu(default="null", options=["mean", "median"]),
    "typeComboBox": lambda: DropdownMenu(default="null", options=["stim-relative", "absolute"]),
    "unitsComboBox": lambda: DropdownMenu(default="null", options=["time", "frames"]),
    "standardizationMethodComboBox": lambda: DropdownMenu(default="null",
                                                          options=["mean_stddev", "stddev", "linear_stddev",
                                                                   "linear_vast", "relative_change", "none"]),
    "summaryMethodComboBox": lambda: DropdownMenu(default="null", options=["rms", "stddev", "var", "avg"]),
    "controlComboBox": lambda: DropdownMenu(default="null", options=["none", "subtraction", "division"]),
    "listEditor": lambda: ListEditorWidget(),
    "openFolder": lambda: OpenFolder(),
    "formatEditor": lambda: FormatEditorWidget("Format"),
    "groupbyEditor": lambda: GroupByFormatEditorWidget(None, None, None, "Group By"),
    "formatEditorQueryloc": lambda: FormatEditorWidget("Format", queryloc=True),
    "cmapSelector": lambda: ColorMapSelector(),
    "affineRigidSelector": lambda: AffineRigidSelector(),
    "saveasSelector": lambda: SaveasExtensionsEditorWidget("Save as"),
    "rangeSelector": lambda: rangeSelector(),
    "null": lambda: QLabel("null"),
}

def build_form_from_template(template: dict, data: dict, adv=False, parent_name="", saved_widgets=None) -> QWidget:
    if saved_widgets is None:
        saved_widgets = {}

    form_container = QWidget()
    form_layout = QVBoxLayout(form_container)
    form_layout.setSpacing(30)
    form_layout.setContentsMargins(15, 15, 15, 15)

    for key, val in data.items():
        if isinstance(val, dict):
            # For nested objects
            template_for_key = template.get(key, {})
            current_parent = f"{parent_name}_{key}" if parent_name else key
            inner_widget = build_form_from_template(template_for_key, val, adv, current_parent, saved_widgets)

            if inner_widget.layout().count() > 0:
                section = CollapsibleSection(title=format_label(key) + ":", default=True)
                section.set_content_layout(inner_widget.layout())
                form_layout.addWidget(section)

            # Handle dependencies after building the section
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

        if not widget_type or widget_type not in WIDGET_FACTORY:
            continue

        widget_constructor = WIDGET_FACTORY.get(widget_type)
        field_widget = widget_constructor()

        if val is not None:
            if hasattr(field_widget, "set_text"):
                field_widget.set_text(str(val))
            elif hasattr(field_widget, "set_value"):
                if isinstance(field_widget, ListEditorWidget) and isinstance(val, list):
                    field_widget.set_value(val)
                elif isinstance(val, (list, dict)) and not isinstance(field_widget, ListEditorWidget):
                    field_widget.set_value(str(val))
                elif isinstance(val, bool):
                    field_widget.set_value(val)
                else:
                    field_widget.set_value(str(val))

        # Save widget if marked for saving
        if save_widget and parent_name:
            saved_key = f"{parent_name}_{key}"
            saved_widgets[saved_key] = field_widget

        # Build row layout
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(15)
        row_layout.setAlignment(Qt.AlignLeft)

        row_layout.addWidget(QLabel(format_label(key) + ':'))
        row_layout.addWidget(field_widget)
        row_widget.setLayout(row_layout)
        if adv:
            optional_widget = OptionalField(row_widget)
            form_layout.addWidget(optional_widget)
        else:
            form_layout.addWidget(row_widget)

    return form_container


def setup_preanalysis_dependencies(saved_widgets, parent_name):
    image_format = saved_widgets.get(f"{parent_name}_image_format")
    video_format = saved_widgets.get(f"{parent_name}_video_format")
    mask_format = saved_widgets.get(f"{parent_name}_mask_format")
    modalities = saved_widgets.get(f"{parent_name}_pipeline_params_modalities")

    def check_modality_in_formats():
        format_widgets = [w for w in [image_format, video_format, mask_format] if w is not None]

        for widget in format_widgets:
            if hasattr(widget, 'get_value'):
                format_str = widget.get_value()
                if isinstance(format_str, str) and "{Modality}" in format_str:
                    return True
        return False

    def update_modalities_enabled():
        if modalities:
            has_modality = check_modality_in_formats()
            modalities.setEnabled(has_modality)
            if not has_modality:
                modalities.set_value([])

    def handle_format_change():
        update_modalities_enabled()

    # Connect signals
    connected = 0
    for widget in [image_format, video_format, mask_format]:
        if widget and hasattr(widget, 'formatChanged'):
            widget.formatChanged.connect(handle_format_change)
            connected += 1
    # Initial update
    update_modalities_enabled()

    alignment_ref = saved_widgets.get(f"{parent_name}_pipeline_params_alignment_reference_modality")

    def update_alignment_options():
        """Update dropdown options based on modalities list"""
        if modalities and alignment_ref:
            values = modalities.get_list()
            if isinstance(values, list):
                alignment_ref.update_options(values)

    # Connect to modalities value change
    if modalities:
        modalities.itemsChanged.connect(update_alignment_options)

    # Also trigger once on load
    update_alignment_options()



    groupby = saved_widgets.get(f"{parent_name}_pipeline_params_group_by")

    def extract_elements_from_formats():
        """Extract available elements in format strings like {SubjectID}, {Session}"""
        format_widgets = [w for w in [image_format, video_format, mask_format] if w is not None]
        elements = set()
        for widget in format_widgets:
            if hasattr(widget, 'get_value'):
                fmt = widget.get_value()
                if isinstance(fmt, str):
                    matches = re.findall(r"{(.*?)}", fmt)
                    elements.update(matches)
        return sorted(elements)

    def update_groupby_sources():
        if groupby and hasattr(groupby, 'update_format_sources'):
            fmt_image = image_format.get_value() if image_format else ""
            fmt_video = video_format.get_value() if video_format else ""
            fmt_mask = mask_format.get_value() if mask_format else ""
            groupby.update_format_sources(fmt_image, fmt_video, fmt_mask)

    for widget in [image_format, video_format, mask_format]:
        if widget and hasattr(widget, 'formatChanged'):
            widget.formatChanged.connect(update_groupby_sources)

    update_groupby_sources()

def setup_analysis_dependencies(saved_widgets, parent_name):
    """Set up all dependencies for analysis section"""
    image_format = saved_widgets.get(f"{parent_name}_image_format")
    video_format = saved_widgets.get(f"{parent_name}_video_format")
    queryloc_format = saved_widgets.get(f"{parent_name}_queryloc_format")
    modalities = saved_widgets.get(f"{parent_name}_analysis_params_modalities")

    def check_modality_in_formats():
        format_widgets = [w for w in [image_format, video_format, queryloc_format] if w is not None]
        for widget in format_widgets:
            if hasattr(widget, 'get_value'):
                format_str = widget.get_value()
                if isinstance(format_str, str) and "{Modality}" in format_str:
                    return True
        return False

    def update_modalities_enabled():
        if modalities:
            has_modality = check_modality_in_formats()
            modalities.setEnabled(has_modality)
            if not has_modality:
                modalities.set_value([])

    # Connect signals
    connected = 0
    for widget in [image_format, video_format, queryloc_format]:
        if widget and hasattr(widget, 'formatChanged'):
            widget.formatChanged.connect(update_modalities_enabled)
            connected += 1
    # Initial update
    update_modalities_enabled()

def generate_json(form_container, template):
    result = {}
    form_layout = form_container.layout()
    if not form_layout:
        return result

    for i in range(form_layout.count()):
        item = form_layout.itemAt(i)
        widget = item.widget()
        if not widget:
            continue

        # Handle collapsible sections (nested objects)
        if isinstance(widget, CollapsibleSection):
            if isinstance(widget, CollapsibleSection):
                if not widget.is_enabled():  # Skip disabled sections
                    continue

                section_title = widget.title().replace(':', '').replace(' ', '_').lower()

                content_layout = widget.content_area.layout()
                if not content_layout:
                    continue

                content_widget = QWidget()
                content_widget.setLayout(content_layout)

                template_for_section = template.get(section_title, {})

                section_data = generate_json(content_widget, template_for_section)
                if section_data:
                    result[section_title] = section_data
                continue

        # Handle regular form rows
        if isinstance(widget, QWidget):
            row_layout = widget.layout()
            if not row_layout or row_layout.count() < 2:
                continue

            # The first item is the label, second is the widget (or OptionalField wrapper)
            label_widget = row_layout.itemAt(0).widget()
            field_widget = row_layout.itemAt(1).widget()

            if not isinstance(label_widget, QLabel):
                continue

            # Get the original key from the label
            label_text = label_widget.text().replace(':', '')
            key = label_text.replace(' ', '_').lower()

            # Handle OptionalField wrapper if present
            if isinstance(field_widget, OptionalField):
                if not field_widget.is_checked():
                    continue  # Skip if the field is disabled
                field_widget = field_widget.field_widget

            # Get the widget type from template to determine how to get the value
            widget_type_def = template.get(key)
            widget_type = extract_widget_type(widget_type_def) if widget_type_def else None

            # Skip if we don't know how to handle this widget type
            if not widget_type or not isinstance(widget_type, str) or widget_type not in WIDGET_FACTORY:
                continue

            # Get the value from the widget based on its type
            value = None
            if hasattr(field_widget, 'get_value'):
                value = field_widget.get_value()
            elif hasattr(field_widget, 'get_text'):
                value = field_widget.get_text()
            elif hasattr(field_widget, 'get_list'):
                value = field_widget.get_list()
            elif isinstance(field_widget, QLabel):
                value = field_widget.text()


            # Convert string values to appropriate types if needed
            if value is not None:
                if widget_type in ["freeNumber"]:
                    try:
                        if '.' in str(value):
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string if conversion fails
                elif widget_type == "trueFalse":
                    value = bool(value)
                elif widget_type == "null":
                    value = None
                elif isinstance(value, str):
                    # Handle special string cases
                    if value.lower() == "null":
                        value = None
                    elif value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False

            # Only add to result if we got a value (including None)
            result[key] = value

    return result
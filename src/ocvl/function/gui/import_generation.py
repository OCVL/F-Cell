from src.ocvl.function.gui.constructors import *
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget
from PySide6.QtCore import Qt

def format_label(key: str) -> str:
    return key.replace('_', ' ').title()

def extract_widget_type(field_def):
    if isinstance(field_def, dict):
        return field_def.get("type")
    return field_def

def create_format_editor_widget_from_spec(field_key: str, widget_spec: dict):
    """Build a FormatEditorWidget with its type coming from the template's 'format_type'."""
    fmt_type = (widget_spec or {}).get("format_type")
    # Label shown left of the widget row uses format_label(key) elsewhere.
    # The FormatEditorWidget itself shows the format string, so label text isn't critical here.
    # We still pass something readable as label_text:
    return FormatEditorWidget(label_text=format_label(field_key), default_format="", type=fmt_type)


WIDGET_FACTORY = {
    # Main fields
    "freeText": lambda config=None: FreetextBox(),
    "freeFloat": lambda config=None: freeFloat(),  # or use a QSpinBox/DoubleSpinBox if you make one
    "freeInt": lambda config=None: freeInt(),
    "trueFalse": lambda config=None: TrueFalseSelector(),
    "comboBox": lambda config=None: DropdownMenu(default="null"),
    "outputSubfolderMethodComboBox": lambda config=None: DropdownMenu(options=["DateTime", "Date", "Sequential"]),
    "shapeComboBox": lambda config=None: DropdownMenu(default="null", options=["disk", "box"]),
    "summaryComboBox": lambda config=None: DropdownMenu(default="null", options=["mean", "median"]),
    "typeComboBox": lambda config=None: DropdownMenu(default="null", options=["stim-relative", "absolute"]),
    "unitsComboBox": lambda config=None: DropdownMenu(default="null", options=["time", "frames"]),
    "normComboBox": lambda config=None: DropdownMenu(default="score", options=["mean", "median", "none"]),
    "standardizationMethodComboBox": lambda config=None: DropdownMenu(default="null",
                                                          options=["mean_stddev", "stddev", "linear_stddev",
                                                                   "linear_vast", "relative_change", "none"]),
    "summaryMethodComboBox": lambda config=None: DropdownMenu(default="null", options=["rms", "stddev", "var", "avg"]),
    "controlComboBox": lambda config=None: DropdownMenu(default="null", options=["none", "subtraction", "division"]),
    "listEditor": lambda config=None: ListEditorWidget(),
    "openFolder": lambda config=None: OpenFolder(),
    "formatEditor": lambda key, spec=None: create_format_editor_widget_from_spec(key, spec or {}),
    "groupbyEditor": lambda config=None: GroupByFormatEditorWidget(None, None, None, "Group By"),
    "cmapSelector": lambda config=None: ColorMapSelector(),
    "affineRigidSelector": lambda config=None: AffineRigidSelector(),
    "saveasSelector": lambda config=None: SaveasExtensionsEditorWidget("Save as"),
    "rangeSelector": lambda config=None: rangeSelector(),
    "null": lambda config=None: QLabel("null"),

    # Subfields
    "text_file": lambda config=None: QLabel("text_file"),  # For metadata type
    "folder": lambda config=None: QLabel("folder"),  # For control location
    "score": lambda config=None: QLabel("score"),  # For normalization method
    "mean_sub": lambda config=None: QLabel("mean_sub"),  # For standardization method
    "auto": lambda config=None: QLabel("auto"),  # For radius
    "disk": lambda config=None: QLabel("disk"),  # For shape
    "mean": lambda config=None: QLabel("mean"),  # For summary
    "rms": lambda config=None: QLabel("rms"),  # For summary method
    "subtraction": lambda config=None: QLabel("subtraction"),  # For control
    "stim-relative": lambda config=None: QLabel("stim-relative"),  # For type
    "time": lambda config=None: QLabel("time"),  # For units
    "viridis": lambda config=None: QLabel("viridis"),  # For cmap
    "plasma": lambda config=None: QLabel("plasma"),  # For cmap
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
            # If no widget type is defined, create a default widget based on value type
            if isinstance(val, bool):
                widget_type = "trueFalse"
            elif isinstance(val, (int, float)):
                widget_type = "freeText"  # Use freeText for numbers
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

        # Set the value based on the actual data type
        if val is not None:
            if hasattr(field_widget, "set_value"):
                # Handle different value types appropriately
                if isinstance(val, bool):
                    field_widget.set_value(val)
                elif isinstance(val, (int, float)):
                    # For numeric values, convert to string for widgets that expect text
                    field_widget.set_value(str(val))
                elif isinstance(val, list) and isinstance(field_widget, ListEditorWidget):
                    field_widget.set_value(val)
                elif isinstance(val, (list, dict)):
                    # For complex types, convert to string representation
                    field_widget.set_value(str(val))
                else:
                    field_widget.set_value(str(val))
            elif hasattr(field_widget, "set_text"):
                field_widget.set_text(str(val))

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

def propagate_advanced_copy(saved_widgets, section, source_key, format_string):
    for saved_key, widget in saved_widgets.items():
        if not isinstance(widget, FormatEditorWidget):
            continue
        if widget.section_name == section and widget.format_key != source_key:
            widget.set_value(format_string)

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

def generate_json(form_container, template, skip_disabled=True):
    """
    Build JSON from the form without ever re-parenting layouts.
    (Re-parenting was breaking collapsibles after Review -> Back.)
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

            # ---- Collapsible section (nested object) ----
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

            # ---- Regular row widget ----
            row_layout = widget.layout()
            if not row_layout or row_layout.count() < 2:
                continue

            label_widget = row_layout.itemAt(0).widget()
            field_widget = row_layout.itemAt(1).widget()

            if not isinstance(label_widget, QLabel):
                continue

            label_text = label_widget.text().replace(':', '')
            key = label_text.replace(' ', '_').lower()

            # OptionalField wrapper
            if isinstance(field_widget, OptionalField):
                if skip_disabled and not field_widget.is_checked():
                    continue
                field_widget = field_widget.field_widget

            widget_type_def = template_for_layout.get(key)
            widget_type = extract_widget_type(widget_type_def) if widget_type_def else None

            if not widget_type or not isinstance(widget_type, str) or widget_type not in WIDGET_FACTORY:
                continue

            # Pull value
            value = None
            if hasattr(field_widget, 'get_value'):
                value = field_widget.get_value()
            elif hasattr(field_widget, 'get_text'):
                value = field_widget.get_text()
            elif hasattr(field_widget, 'get_list'):
                value = field_widget.get_list()
            elif hasattr(field_widget, 'currentText'):
                value = field_widget.currentText()
            elif hasattr(field_widget, 'text'):
                value = field_widget.text()
            elif hasattr(field_widget, 'isChecked'):
                value = field_widget.isChecked()
            elif isinstance(field_widget, QLabel):
                value = field_widget.text()

            # Convert string types where appropriate
            if value is not None:
                if isinstance(value, str):
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except (ValueError, TypeError):
                        if value.lower() == "null":
                            value = None
                        elif value.lower() == "true":
                            value = True
                        elif value.lower() == "false":
                            value = False
                elif widget_type == "freeInt" and isinstance(value, (int, float)):
                    value = int(value)
                elif widget_type == "freeFloat" and isinstance(value, (int, float)):
                    value = float(value)
                elif widget_type == "trueFalse":
                    value = bool(value)
                elif widget_type == "null":
                    value = None

            result[key] = value

        return result

    return walk_layout(form_container.layout(), template)
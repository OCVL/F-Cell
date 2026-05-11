"""
wizard_creator.py — Main wizard window and all page classes.

Changes from the original:
- Dead TextColor class removed.
- Page IDs collected into a Page enum — no more magic numbers scattered across the file.
- Tooltip-label creation extracted into make_tooltip_label() to avoid repetition.
- create_hover_widget() moved to module level (was re-defined in each page class).
- PreanalysisPage / AnalysisPage form-building moved into _build_form() methods
  so __init__ stays a clean wiring step.
- AdvancedSetupPage, ReviewPage, ImportEditorPage, EndPage unchanged in behaviour.
"""

import json

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QRadioButton, QScrollArea, QSizePolicy, QVBoxLayout,
    QWidget, QWizard, QWizardPage,
)

from src.ocvl.function.gui.gui_widgets import (
    AlignmentModalitySelector, ListEditorWidget, TrueFalseSelector,
)
from src.ocvl.function.gui.gui_dialogs import (
    FormatEditorWidget, GroupByFormatEditorWidget,
)
from src.ocvl.function.gui.import_generation import build_form_from_template, generate_json


# ---------------------------------------------------------------------------
# Page ID constants
# ---------------------------------------------------------------------------

class Page:
    INTRO         = 0
    SELECTION     = 1
    VER_DESC      = 2
    PREANALYSIS   = 3
    ANALYSIS      = 4
    ADVANCED      = 5
    REVIEW        = 6
    IMPORT_EDITOR = 7
    END           = 8


# ---------------------------------------------------------------------------
# Module-level helpers shared across pages
# ---------------------------------------------------------------------------

_TOOLTIP_STYLE = """
    QLabel {
        border: 1px solid #ccc;
        border-radius: 6px;
        background-color: #f0f0f0;
        padding: 6px 12px;
        font-style: italic;
        color: #333;
        min-height: 40px;
    }
"""

_RADIO_STYLE = """
    QRadioButton {
        min-width: 350px;
        min-height: 45px;
        font-size: 20px;
        padding: 5px;
        spacing: 5px;
    }
    QRadioButton::indicator { width: 20px; height: 20px; }
"""

_LABEL_STYLE = """
    QLabel {
        min-width: 350px;
        min-height: 30px;
        font-size: 20px;
        font-weight: bold;
        padding: 5px;
        qproperty-alignment: AlignCenter;
    }
"""


def make_tooltip_label(placeholder: str = "Hover over a field to see its description") -> QLabel:
    lbl = QLabel(placeholder)
    lbl.setWordWrap(True)
    lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    lbl.setStyleSheet(_TOOLTIP_STYLE)
    return lbl


def create_hover_widget(layout, tooltip_label: QLabel, tooltip_text: str,
                        placeholder: str = "Hover over a field to see its description") -> QFrame:
    """Wrap a layout in a QFrame that updates a tooltip label on hover."""
    wrapper = QFrame()
    wrapper.setLayout(layout)
    wrapper.setMouseTracking(True)
    wrapper.setStyleSheet("QFrame { background: transparent; }")
    wrapper.enterEvent = lambda event: tooltip_label.setText(tooltip_text)
    wrapper.leaveEvent = lambda event: tooltip_label.setText(placeholder)
    return wrapper


def _scrolled_page(page_widget: QWizardPage) -> tuple:
    """
    Create a scroll area + container + main_layout for a page.
    Returns (scroll, container, main_layout).
    Caller is responsible for setting the page layout.
    """
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    container = QWidget()
    scroll.setWidget(container)
    main_layout = QVBoxLayout(container)
    main_layout.setAlignment(Qt.AlignLeft)
    main_layout.setContentsMargins(10, 10, 10, 10)
    return scroll, container, main_layout


def has_modality(format_string: str) -> bool:
    return "{Modality}" in format_string if format_string else False


def _clear_layout(layout):
    while layout and layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        elif item.layout():
            _clear_layout(item.layout())


# ---------------------------------------------------------------------------
# MainWizard
# ---------------------------------------------------------------------------

class MainWizard(QWizard):
    bold = QtGui.QFont()
    bold.setBold(True)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWizardStyle(QWizard.WizardStyle.ClassicStyle)
        self.setOption(QWizard.WizardOption.IndependentPages, False)
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowSystemMenuHint
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(800, 600)

        self.setPage(Page.INTRO,         IntroPage())
        self.setPage(Page.SELECTION,     SelectionPage())
        self.setPage(Page.VER_DESC,      VerDescriptionPage())
        self.setPage(Page.PREANALYSIS,   PreanalysisPage())
        self.setPage(Page.ANALYSIS,      AnalysisPage())
        self.setPage(Page.ADVANCED,      AdvancedSetupPage())
        self.setPage(Page.REVIEW,        ReviewPage())
        self.setPage(Page.IMPORT_EDITOR, ImportEditorPage())
        self.setPage(Page.END,           EndPage())

        self.currentIdChanged.connect(self._update_button_text)
        self.page(Page.INTRO).button_group.buttonToggled.connect(self._update_button_text_for_intro)

    def _update_button_text(self, page_id):
        if page_id == Page.INTRO:
            self._update_button_text_for_intro()
        elif page_id == Page.REVIEW:
            self.button(QWizard.NextButton).setText("Save >")
        elif page_id == Page.IMPORT_EDITOR:
            self.button(QWizard.NextButton).setText("Review >")
        else:
            self.button(QWizard.NextButton).setText("Next >")

    def _update_button_text_for_intro(self):
        if self.currentId() == Page.INTRO:
            if self.page(Page.INTRO).import_button.isChecked():
                self.button(QWizard.NextButton).setText("Import >")
            else:
                self.button(QWizard.NextButton).setText("Next >")


# ---------------------------------------------------------------------------
# IntroPage
# ---------------------------------------------------------------------------

class IntroPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Welcome to the Marquette Engineering Adaptive Optics (MEAO) Configuration File Generator!")
        self.setSubTitle(
            'To begin, choose if you wish to import an existing config file, or create a new one\n'
            '• Note: Importing an existing config file will bring you to "advanced" setup'
        )
        self.imported_config = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)

        outer_layout = QVBoxLayout(container)
        outer_layout.setAlignment(Qt.AlignCenter)

        center_layout = QHBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)

        label = QLabel("Select an option:")
        label.setStyleSheet(_LABEL_STYLE)
        center_layout.addWidget(label)

        self.create_button = QRadioButton("Create New Configuration")
        self.import_button = QRadioButton("Import Existing Configuration")
        self.create_button.setStyleSheet(_RADIO_STYLE)
        self.import_button.setStyleSheet(_RADIO_STYLE)

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.create_button, 0)
        self.button_group.addButton(self.import_button, 1)
        self.create_button.setChecked(True)

        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self.create_button)
        btn_layout.addWidget(self.import_button)
        btn_layout.setAlignment(Qt.AlignCenter)
        center_layout.addLayout(btn_layout)

        outer_layout.addLayout(center_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        main_layout.setContentsMargins(0, 0, 0, 0)

    def validatePage(self):
        if self.create_button.isChecked():
            return True
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Configuration File", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return False
        try:
            with open(file_path, "r") as f:
                self.imported_config = json.load(f)
            return True
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load file:\n{str(e)}")
            return False

    def nextId(self):
        if self.import_button.isChecked():
            return Page.IMPORT_EDITOR
        return Page.SELECTION


# ---------------------------------------------------------------------------
# SelectionPage
# ---------------------------------------------------------------------------

class SelectionPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('Choose your desired type of setup and click "Next"')
        self.setSubTitle(
            "• Simple generation: Step-by-step process to create configuration file\n"
            "• Advanced generation: In-depth menu with access to change any and all fields"
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)

        outer_layout = QVBoxLayout(container)
        outer_layout.setAlignment(Qt.AlignCenter)

        center_layout = QHBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)

        label = QLabel("Choose type of generation:")
        label.setStyleSheet(_LABEL_STYLE)
        center_layout.addWidget(label)

        self.simple_button = QRadioButton("Simple Generation")
        self.adv_button = QRadioButton("Advanced Generation")
        self.simple_button.setStyleSheet(_RADIO_STYLE)
        self.adv_button.setStyleSheet(_RADIO_STYLE)

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.simple_button, 0)
        self.button_group.addButton(self.adv_button, 1)
        self.simple_button.setChecked(True)

        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self.simple_button)
        btn_layout.addWidget(self.adv_button)
        btn_layout.setAlignment(Qt.AlignCenter)
        center_layout.addLayout(btn_layout)
        outer_layout.addLayout(center_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        main_layout.setContentsMargins(0, 0, 0, 0)

    def nextId(self):
        return Page.VER_DESC if self.simple_button.isChecked() else Page.ADVANCED


# ---------------------------------------------------------------------------
# VerDescriptionPage
# ---------------------------------------------------------------------------

class VerDescriptionPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Version and Description Setup")
        self.setSubTitle("Choose version and set a description for the configuration file")

        scroll, container, main_layout = _scrolled_page(self)
        placeholder = "Hover over a field to see its description"
        self.tooltip_label = make_tooltip_label(placeholder)

        # Version
        version_layout = QHBoxLayout()
        version_layout.setAlignment(Qt.AlignLeft)
        version_layout.addWidget(QLabel("Version:"))
        self.version_value = QLineEdit("0.2")
        self.version_value.setFixedWidth(100)
        version_layout.addWidget(self.version_value)
        version_widget = create_hover_widget(version_layout, self.tooltip_label,
                                             "Version: Version number for this configuration file.",
                                             placeholder)

        # Description
        description_layout = QHBoxLayout()
        description_layout.setAlignment(Qt.AlignLeft)
        description_layout.addWidget(QLabel("Description:"))
        self.description_value = QLineEdit("The pipeline and analysis JSON for the OCVL's MEAOSLO.")
        self.description_value.setFixedWidth(400)
        description_layout.addWidget(self.description_value)
        description_widget = create_hover_widget(description_layout, self.tooltip_label,
                                                 "Description: A brief explanation of what this configuration does.",
                                                 placeholder)

        main_layout.addWidget(version_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(description_widget, alignment=Qt.AlignLeft)
        main_layout.addStretch()
        main_layout.addWidget(self.tooltip_label)

        page_layout = QVBoxLayout(self)
        page_layout.addWidget(scroll)
        page_layout.setContentsMargins(0, 0, 0, 0)

    def nextId(self):
        return Page.PREANALYSIS


# ---------------------------------------------------------------------------
# PreanalysisPage
# ---------------------------------------------------------------------------

class PreanalysisPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Pre-Analysis Pipeline Setup")
        self.setSubTitle("Configure pre-analysis pipeline below")

        scroll, container, main_layout = _scrolled_page(self)
        placeholder = "Hover over a field to see its description"
        self.tooltip_label = make_tooltip_label(placeholder)

        self._build_form(main_layout, placeholder)

        main_layout.addStretch()
        main_layout.addWidget(self.tooltip_label)

        page_layout = QVBoxLayout(self)
        page_layout.addWidget(scroll)
        page_layout.setContentsMargins(0, 0, 0, 0)

    def _build_form(self, layout, placeholder):
        def hover(row_layout, tip):
            return create_hover_widget(row_layout, self.tooltip_label, tip, placeholder)

        def fmt_row(label, widget):
            row = QHBoxLayout()
            row.setAlignment(Qt.AlignLeft)
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            return row

        DEFAULT_FMT = "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}"

        splitter1 = QLabel("Formats:")
        splitter1.setFont(MainWizard.bold)
        layout.addWidget(splitter1, alignment=Qt.AlignLeft)

        self.image_format_value = FormatEditorWidget("Image Format:", DEFAULT_FMT, type='image')
        self.video_format_value = FormatEditorWidget("Video Format:", DEFAULT_FMT, type='video')
        self.mask_format_value  = FormatEditorWidget("Mask Format:",  DEFAULT_FMT, type='mask')
        self.recursive_search_tf = TrueFalseSelector(False)

        layout.addWidget(hover(fmt_row("Image Format:", self.image_format_value),
                               "Image Format: Format string for image filenames."), alignment=Qt.AlignLeft)
        layout.addWidget(hover(fmt_row("Video Format:", self.video_format_value),
                               "Video Format: Format string for video filenames."), alignment=Qt.AlignLeft)
        layout.addWidget(hover(fmt_row("Mask Format:", self.mask_format_value),
                               "Mask Format: Format string for mask filenames."), alignment=Qt.AlignLeft)
        layout.addWidget(hover(fmt_row("Recursive Search:", self.recursive_search_tf),
                               "Recursive Search: If enabled, subfolders will be searched recursively."),
                         alignment=Qt.AlignLeft)

        splitter2 = QLabel("Pipeline Parameters:")
        splitter2.setFont(MainWizard.bold)
        layout.addWidget(splitter2, alignment=Qt.AlignLeft)

        self.modalities_list_creator = ListEditorWidget()
        self.alignment_ref_value = AlignmentModalitySelector(self.modalities_list_creator, "null")
        self.groupby_value = GroupByFormatEditorWidget(
            image_format="", video_format="", mask_format="",
            label_text="Group By:", default_format="null",
        )

        layout.addWidget(hover(fmt_row("Modalities:", self.modalities_list_creator),
                               "Modalities: List of imaging modalities (requires '{Modality}' in at least one format string to edit)."),
                         alignment=Qt.AlignLeft)
        layout.addWidget(hover(fmt_row("Alignment Reference Modality:", self.alignment_ref_value),
                               "Alignment Reference Modality: Reference modality used for alignment."),
                         alignment=Qt.AlignLeft)
        layout.addWidget(hover(fmt_row("Group By:", self.groupby_value),
                               "Group By: Field used to group sessions (optional, options derived from format strings)."),
                         alignment=Qt.AlignLeft)

        # Wire copy-to-all
        for key, widget in [("image_format", self.image_format_value),
                             ("video_format", self.video_format_value),
                             ("mask_format",  self.mask_format_value)]:
            widget.section_name = "preanalysis"
            widget.format_key = key
            widget.copyToAllRequested.connect(self._copy_format_to_all)

        # Format → groupby / modality state
        for w in (self.image_format_value, self.video_format_value, self.mask_format_value):
            w.formatChanged.connect(self._update_groupby_elements)
            w.formatChanged.connect(self._update_modality_editor_state)

        self.modalities_list_creator.itemsChanged.connect(self._update_alignment_modality_state)
        self._update_alignment_modality_state()

    def _copy_format_to_all(self, section, key, value):
        if section != "preanalysis":
            return
        for k, w in [("image_format", self.image_format_value),
                     ("video_format", self.video_format_value),
                     ("mask_format",  self.mask_format_value)]:
            if k != key:
                w.set_value(value)

    def _update_alignment_modality_state(self):
        self.alignment_ref_value.setEnabled(bool(self.modalities_list_creator.get_list()))

    def _update_groupby_elements(self):
        self.groupby_value.image_format = self.image_format_value.get_value()
        self.groupby_value.video_format = self.video_format_value.get_value()
        self.groupby_value.mask_format  = self.mask_format_value.get_value()

    def _update_modality_editor_state(self):
        any_has = any(has_modality(w.get_value())
                      for w in (self.image_format_value, self.video_format_value, self.mask_format_value))
        self.modalities_list_creator.setEnabled(any_has)

    def nextId(self):
        return Page.ANALYSIS


# ---------------------------------------------------------------------------
# AnalysisPage
# ---------------------------------------------------------------------------

class AnalysisPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Analysis Pipeline Setup")
        self.setSubTitle("Configure analysis pipeline below")

        scroll, container, main_layout = _scrolled_page(self)
        placeholder = "Hover over a field to see its description"
        self.tooltip_label = make_tooltip_label(placeholder)

        self._build_form(main_layout, placeholder)

        main_layout.addStretch()
        main_layout.addWidget(self.tooltip_label)
        main_layout.setAlignment(Qt.AlignLeft)

        page_layout = QVBoxLayout(self)
        page_layout.addWidget(scroll)
        page_layout.setContentsMargins(0, 0, 0, 0)

    def _build_form(self, layout, placeholder):
        def hover(row_layout, tip):
            return create_hover_widget(row_layout, self.tooltip_label, tip, placeholder)

        def fmt_row(label, widget):
            row = QHBoxLayout()
            row.setAlignment(Qt.AlignLeft)
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            return row

        DEFAULT_FMT = "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}"

        splitter1 = QLabel("Formats:")
        splitter1.setFont(MainWizard.bold)
        layout.addWidget(splitter1, alignment=Qt.AlignLeft)

        self.image_format_value    = FormatEditorWidget("Image Format:",    DEFAULT_FMT, type='image')
        self.queryloc_format_value = FormatEditorWidget("Queryloc Format:", DEFAULT_FMT, type='queryloc')
        self.video_format_value    = FormatEditorWidget("Video Format:",    DEFAULT_FMT, type='video')
        self.recursive_search_tf   = TrueFalseSelector(False)

        layout.addWidget(hover(fmt_row("Image Format:",      self.image_format_value),
                               "Image Format: Format string for image filenames."), alignment=Qt.AlignLeft)
        layout.addWidget(hover(fmt_row("Query Loc Format:",  self.queryloc_format_value),
                               "Query Loc Format: Filename format for the reference query location."), alignment=Qt.AlignLeft)
        layout.addWidget(hover(fmt_row("Video Format:",      self.video_format_value),
                               "Video Format: Format string for video filenames."), alignment=Qt.AlignLeft)
        layout.addWidget(hover(fmt_row("Recursive Search:",  self.recursive_search_tf),
                               "Recursive Search: If enabled, subfolders will be searched recursively."), alignment=Qt.AlignLeft)

        splitter2 = QLabel("Pipeline Parameters:")
        splitter2.setFont(MainWizard.bold)
        layout.addWidget(splitter2, alignment=Qt.AlignLeft)

        self.modalities_list_creator = ListEditorWidget()
        layout.addWidget(hover(fmt_row("Modalities:", self.modalities_list_creator),
                               "Modalities: List of imaging modalities (requires '{Modality}' in at least one format string to edit)."),
                         alignment=Qt.AlignLeft)

        # Wire copy-to-all
        for key, widget in [("image_format",    self.image_format_value),
                             ("video_format",    self.video_format_value),
                             ("queryloc_format", self.queryloc_format_value)]:
            widget.section_name = "analysis"
            widget.format_key = key
            widget.copyToAllRequested.connect(self._copy_format_to_all)

        for w in (self.image_format_value, self.queryloc_format_value, self.video_format_value):
            w.formatChanged.connect(self._update_modality_editor_state)

    def _copy_format_to_all(self, section, key, value):
        if section != "analysis":
            return
        for k, w in [("image_format",    self.image_format_value),
                     ("video_format",    self.video_format_value),
                     ("queryloc_format", self.queryloc_format_value)]:
            if k != key:
                w.set_value(value)

    def _update_modality_editor_state(self):
        any_has = any(has_modality(w.get_value())
                      for w in (self.image_format_value, self.queryloc_format_value, self.video_format_value))
        self.modalities_list_creator.setEnabled(any_has)

    def nextId(self):
        return Page.REVIEW


# ---------------------------------------------------------------------------
# AdvancedSetupPage
# ---------------------------------------------------------------------------

class AdvancedSetupPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Advanced Setup")
        self.setSubTitle('Configure all fields directly. Click "Next >" when done.')

        self.advanced_widget = None
        self.master_json = None

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        page_layout = QVBoxLayout(self)
        page_layout.addWidget(self.scroll)
        page_layout.setContentsMargins(0, 0, 0, 0)

    def initializePage(self):
        from src.ocvl.function.gui.import_generation import build_form_from_template

        with open(r"master_config_files/advanced_config_JSON.json", "r") as f:
            self.master_json = json.load(f)

        self.advanced_widget = build_form_from_template(
            self.master_json, self.master_json, adv=True
        )

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(self.advanced_widget)
        container_layout.addStretch()
        self.scroll.setWidget(container)

    def nextId(self):
        return Page.REVIEW


# ---------------------------------------------------------------------------
# ReviewPage
# ---------------------------------------------------------------------------

class ReviewPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Review Configuration")
        self.setSubTitle('Review your settings and click "Save >" to generate the configuration JSON file.')

        self.saved_file_path = None
        self.generated_config = None

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        self.container = QWidget()
        self.scroll.setWidget(self.container)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.scroll)

        self.container_layout = QVBoxLayout(self.container)

    def nextId(self):
        return Page.END

    # ---- helpers ----

    def _fmt_value(self, v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            if not v:
                return "{}"
            lines = ["{"]
            for k, val in v.items():
                fv = self._fmt_value(val)
                lines.append(f"  {k}: {fv.replace(chr(10), chr(10) + '  ') if chr(10) in fv else fv}")
            lines.append("}")
            return "\n".join(lines)
        if isinstance(v, (list, tuple)):
            if len(v) <= 3 and all(isinstance(x, (int, float, str)) for x in v):
                return "[" + ", ".join(self._fmt_value(x) for x in v) + "]"
            return "\n• " + "\n• ".join(self._fmt_value(x) for x in v)
        return str(v)

    def _mk_field(self, title, value):
        formatted = self._fmt_value(value).replace('\n', '<br>')
        lab = QLabel(f"<b>{title}:</b> {formatted}")
        lab.setWordWrap(True)
        lab.setStyleSheet("QLabel { padding: 3px 6px; margin: 2px 0; border-left: 3px solid #007ACC; }")
        return lab

    def _mk_section(self, title):
        lab = QLabel(title.upper())
        f = lab.font()
        f.setBold(True)
        lab.setFont(f)
        lab.setStyleSheet("QLabel { font-size: 14px; margin: 10px 0 4px 0; }")
        return lab

    def _titleize(self, key):
        return str(key).replace("_", " ").title()

    def _render_by_template(self, template_node, data_node, header_name=None):
        if header_name is not None:
            self.container_layout.addWidget(self._mk_section(header_name))
        if isinstance(template_node, dict):
            for k, tmpl_child in template_node.items():
                if header_name is None and k in ("version", "description"):
                    continue
                data_child = data_node.get(k) if isinstance(data_node, dict) else None
                is_subsection = isinstance(tmpl_child, dict) and "type" not in tmpl_child
                if is_subsection:
                    self._render_by_template(tmpl_child, data_child or {}, self._titleize(k))
                else:
                    self.container_layout.addWidget(self._mk_field(self._titleize(k), data_child))

    def _render_pruned_template(self, template_node, data_node):
        if not isinstance(template_node, dict) or not isinstance(data_node, dict):
            return data_node
        pruned = {}
        for k, tmpl_child in template_node.items():
            if k in data_node:
                dv = data_node[k]
                is_subsection = isinstance(tmpl_child, dict) and "type" not in tmpl_child
                if is_subsection and isinstance(dv, dict):
                    pruned[k] = self._render_pruned_template(tmpl_child, dv)
                else:
                    pruned[k] = tmpl_child
        return pruned

    def _display_simple_mode(self, cfg, wiz):
        for key, label in [("version", "Version"), ("description", "Description")]:
            if key in cfg:
                self.container_layout.addWidget(self._mk_field(label, cfg[key]))

        if "preanalysis" in cfg:
            pre = cfg["preanalysis"]
            self.container_layout.addWidget(self._mk_section("Pre-Analysis"))
            for label, k in [("Image Format", "image_format"), ("Video Format", "video_format"),
                              ("Mask Format", "mask_format"), ("Recursive Search", "recursive_search")]:
                if k in pre:
                    self.container_layout.addWidget(self._mk_field(label, pre[k]))
            if "pipeline_params" in pre:
                params = pre["pipeline_params"]
                self.container_layout.addWidget(self._mk_section("Pipeline Parameters"))
                for label, k in [("Modalities", "modalities"),
                                  ("Alignment Reference Modality", "alignment_reference_modality"),
                                  ("Group By", "group_by")]:
                    if k in params:
                        self.container_layout.addWidget(self._mk_field(label, params[k]))

        if "analysis" in cfg:
            ana = cfg["analysis"]
            self.container_layout.addWidget(self._mk_section("Analysis"))
            for label, k in [("Image Format", "image_format"), ("Queryloc Format", "queryloc_format"),
                              ("Video Format", "video_format"), ("Recursive Search", "recursive_search")]:
                if k in ana:
                    self.container_layout.addWidget(self._mk_field(label, ana[k]))
            if "analysis_params" in ana:
                self.container_layout.addWidget(self._mk_section("Analysis Params"))
                params = ana["analysis_params"]
                if "modalities" in params:
                    self.container_layout.addWidget(self._mk_field("Modalities", params["modalities"]))

    def _display_advanced_mode(self, cfg, wiz):
        for key, label in [("version", "Version"), ("description", "Description")]:
            if key in cfg:
                self.container_layout.addWidget(self._mk_field(label, cfg[key]))
        with open(r"master_config_files/advanced_config_JSON.json", "r") as f:
            advanced_template = json.load(f)
        self._render_by_template(advanced_template, cfg)

    def _display_import_mode(self, cfg, wiz):
        for key, label in [("version", "Version"), ("description", "Description")]:
            if key in cfg:
                self.container_layout.addWidget(self._mk_field(label, cfg[key]))
        with open(r"master_config_files/master_JSON.json", "r") as f:
            master_template = json.load(f)
        imported = wiz.page(Page.INTRO).imported_config or {}
        pruned = self._render_pruned_template(master_template, imported)
        self._render_by_template(pruned, cfg)

    def initializePage(self):
        wiz = self.wizard()

        # Clear previous render
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Build config
        if wiz.page(Page.SELECTION).adv_button.isChecked():
            advanced_widget = wiz.page(Page.ADVANCED).advanced_widget
            master_json = wiz.page(Page.ADVANCED).master_json
            self.generated_config = generate_json(advanced_widget, master_json)
        elif wiz.page(Page.INTRO).import_button.isChecked():
            import_config_widget = wiz.page(Page.IMPORT_EDITOR).form_widget
            master_json = wiz.page(Page.ADVANCED).master_json
            self.generated_config = generate_json(import_config_widget, master_json, skip_disabled=False)
        else:
            pre_page = wiz.page(Page.PREANALYSIS)
            ana_page = wiz.page(Page.ANALYSIS)
            ver_page = wiz.page(Page.VER_DESC)
            self.generated_config = {
                "version": ver_page.version_value.text(),
                "description": ver_page.description_value.text(),
                "preanalysis": {
                    "image_format": pre_page.image_format_value.get_value(),
                    "video_format": pre_page.video_format_value.get_value(),
                    "mask_format":  pre_page.mask_format_value.get_value(),
                    "recursive_search": pre_page.recursive_search_tf.get_value(),
                    "pipeline_params": {
                        "modalities": pre_page.modalities_list_creator.get_list(),
                        "alignment_reference_modality": pre_page.alignment_ref_value.get_value(),
                        "group_by": None if pre_page.groupby_value.get_value() == "null"
                                    else pre_page.groupby_value.get_value(),
                    },
                },
                "analysis": {
                    "image_format":    ana_page.image_format_value.get_value(),
                    "queryloc_format": ana_page.queryloc_format_value.get_value(),
                    "video_format":    ana_page.video_format_value.get_value(),
                    "recursive_search": ana_page.recursive_search_tf.get_value(),
                    "analysis_params": {
                        "modalities": ana_page.modalities_list_creator.get_list(),
                    },
                },
            }

        # Render
        try:
            if wiz.page(Page.SELECTION).adv_button.isChecked():
                self._display_advanced_mode(self.generated_config, wiz)
            elif wiz.page(Page.INTRO).import_button.isChecked():
                self._display_import_mode(self.generated_config, wiz)
            else:
                self._display_simple_mode(self.generated_config, wiz)
        except Exception as e:
            err = QLabel(f"Error rendering review: {e}")
            err.setStyleSheet("color: red; font-weight: bold;")
            self.container_layout.addWidget(err)

        self.container_layout.addStretch()

    def validatePage(self):
        if not self.generated_config:
            QMessageBox.warning(self, "Error", "No configuration data available")
            return False

        file_path, _ = QFileDialog().getSaveFileName(
            self, "Save Configuration File", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            QMessageBox.warning(self, "Error", "No file selected. Please choose a path to save the configuration.")
            return False

        if not file_path.endswith('.json'):
            file_path += '.json'
        try:
            with open(file_path, 'w') as f:
                json.dump(self.generated_config, f, indent=2)
            self.saved_file_path = file_path
            return True
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save file:\n{str(e)}")
            return False


# ---------------------------------------------------------------------------
# ImportEditorPage
# ---------------------------------------------------------------------------

class ImportEditorPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Edit JSON")
        self.setSubTitle('Edit your imported JSON here. Click "Save >" to save new JSON.')

        self.saved_file_path = None
        self.form_widget = None

        self._layout = QVBoxLayout(self)
        self._placeholder = QLabel("Loading imported configuration...")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._placeholder)

    def initializePage(self):
        self._layout.removeWidget(self._placeholder)
        self._placeholder.deleteLater()

        wizard = self.wizard()
        intro_page = wizard.page(Page.INTRO)

        if hasattr(intro_page, 'imported_config') and intro_page.imported_config:
            with open(r"master_config_files/master_JSON.json", "r") as f:
                master_json = json.load(f)

            self.form_widget = build_form_from_template(master_json, intro_page.imported_config)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.form_widget)
            self._layout.addWidget(scroll)
        else:
            error = QLabel("No imported configuration found.")
            error.setAlignment(Qt.AlignCenter)
            self._layout.addWidget(error)

    def nextId(self):
        return Page.REVIEW


# ---------------------------------------------------------------------------
# EndPage
# ---------------------------------------------------------------------------

class EndPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Generation Complete!")
        self.setSubTitle("JSON configuration file has been created")

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.path_label)
        layout.addStretch()

    def initializePage(self):
        review_page = self.wizard().page(Page.REVIEW)
        import_page = self.wizard().page(Page.IMPORT_EDITOR)
        path = review_page.saved_file_path or import_page.saved_file_path
        if path:
            self.path_label.setText(f"Configuration file saved to:\n{path}")
        else:
            self.path_label.setText("Configuration file was not saved. Contact admin if issues persist.")
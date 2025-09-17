import json
import sys

from PySide6 import QtGui
from PySide6.QtGui import QFont, QMovie
from PySide6.QtWidgets import (QApplication, QWizard, QWizardPage,
                               QLabel, QLineEdit, QVBoxLayout,
                               QCheckBox, QComboBox, QHBoxLayout, QRadioButton, QButtonGroup, QSizePolicy, QScrollArea,
                               QWidget, QFileDialog, QMessageBox, QFrame)
from PySide6.QtCore import Qt, QSize, Signal
from advancedconfig import create_advanced_setup_widget, description_layer
import constructors
import advancedconfig
from import_generation import *

bold = QtGui.QFont()
bold.setBold(True)

class TextColor:
    BOLD_START = '\033[1m'
    END = '\033[0m'
    UNDERLINE = '\033[4m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'

def has_modality(format_string):
    return "{Modality}" in format_string if format_string else False

class HoverWidget(QWidget):
    def __init__(self, tooltip_text, tooltip_label, placeholder_text="Hover over a field to see its description", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tooltip_text = tooltip_text
        self.tooltip_label = tooltip_label
        self.placeholder_text = placeholder_text
        self.setMouseTracking(True)

    def enterEvent(self, event):
        self.tooltip_label.setText(self.tooltip_text)

    def leaveEvent(self, event):
        self.tooltip_label.setText(self.placeholder_text)

class MainWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWizardStyle(QWizard.WizardStyle.ClassicStyle)
        self.setOption(QWizard.WizardOption.IndependentPages, False)

        # Clear existing flags and set new ones
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowSystemMenuHint
        )

        # Ensure the wizard can be resized
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(800, 600)  # Set a reasonable minimum size

        self.setPage(0, IntroPage())
        self.setPage(1, SelectionPage())
        self.setPage(2, VerDescriptionPage())
        self.setPage(3, PreanalysisPage())
        self.setPage(4, AnalysisPage())
        self.setPage(5, AdvancedSetupPage())
        self.setPage(6, ReviewPage())
        self.setPage(7, ImportEditorPage())
        self.setPage(8, EndPage())

        self.currentIdChanged.connect(self.update_button_text)

        intro_page = self.page(0)
        intro_page.button_group.buttonToggled.connect(self.update_button_text_for_intro)

    def update_button_text(self, id):
        if id == 0:
            # On intro page, check which option is selected
            self.update_button_text_for_intro()
        elif id == 6 | 7:
            self.button(QWizard.NextButton).setText("Save >")
        else:
            self.button(QWizard.NextButton).setText("Next >")

    def update_button_text_for_intro(self):
        # Only update if we're currently on the intro page
        if self.currentId() == 0:
            intro_page = self.page(0)
            if intro_page.import_button.isChecked():
                self.button(QWizard.NextButton).setText("Import >")
            else:
                self.button(QWizard.NextButton).setText("Next >")


class IntroPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Welcome to the MEAO Configuration File Generator!")
        self.setSubTitle('To begin, choose if you wish to import an existing config file, or create a new one\n'
                         '• Note: Importing an existing config file will bring you to "advanced" setup')

        self.imported_config = None  # To store the imported config

        # Create scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # Create container widget
        container = QWidget()
        scroll.setWidget(container)

        # Outer layout to center the content
        outer_layout = QVBoxLayout(container)
        outer_layout.setAlignment(Qt.AlignCenter)  # Center everything

        # Inner layout: label and buttons side by side
        center_layout = QHBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)

        # Label
        label = QLabel("Select an option:")
        center_layout.addWidget(label)

        self.create_button = QRadioButton("Create New Configuration")
        self.import_button = QRadioButton("Import Existing Configuration")

        # Style to make buttons bigger
        radio_style = """
            QRadioButton {
                min-width: 350px;
                min-height: 45px;
                font-size: 20px;
                padding: 5px;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
            }
        """
        label_style = """
            QLabel {
                min-width: 350px;
                min-height: 30px;
                font-size: 20px;
                font-weight: bold;
                padding: 5px;
                qproperty-alignment: AlignCenter;
            }
        """

        label.setStyleSheet(label_style)
        self.create_button.setStyleSheet(radio_style)
        self.import_button.setStyleSheet(radio_style)

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.create_button, 0)
        self.button_group.addButton(self.import_button, 1)
        self.create_button.setChecked(True)

        # Layout for buttons stacked vertically
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.create_button)
        button_layout.addWidget(self.import_button)
        button_layout.setAlignment(Qt.AlignCenter)

        center_layout.addLayout(button_layout)

        # Add to outer layout
        outer_layout.addLayout(center_layout)

        # Set main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        main_layout.setContentsMargins(0, 0, 0, 0)

    def nextId(self):
        if self.create_button.isChecked():
            return 1
        elif self.import_button.isChecked():

            # Show file dialog to select JSON file
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self,
                "Open Configuration File",
                "",
                "JSON Files (*.json);;All Files (*)"
            )

            if file_path:
                try:
                    with open(file_path, 'r') as f:
                        self.imported_config = json.load(f)
                    return 7  # Go to advanced setup page
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load file:\n{str(e)}")
                    return -1  # Stay on current page
            return -1  # Stay on current page if no file selected
        return 2

class SelectionPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('Choose your desired type of setup and click "Next"')
        self.setSubTitle("• Simple generation: Step-by-step process to create configuration file\n"
                         "• Advanced generation: In-depth menu with access to change any and all fields in the configuration file in one step")

        # Create scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # Create container widget
        container = QWidget()
        scroll.setWidget(container)

        # Outer layout to center the content
        outer_layout = QVBoxLayout(container)
        outer_layout.setAlignment(Qt.AlignCenter)  # Center everything

        center_layout = QHBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)

        placeholder = "Hover over an option to see details"

        # Tooltip label
        self.tooltip_label = QLabel(placeholder)
        self.tooltip_label.setWordWrap(True)
        self.tooltip_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tooltip_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tooltip_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: #f0f0f0;
                padding: 6px 12px;
                font-style: italic;
                color: #333;
                min-height: 40px;
            }
        """)

        def create_hover_widget(layout, tooltip_text):
            wrapper = QFrame()
            wrapper.setLayout(layout)
            wrapper.setMouseTracking(True)
            wrapper.setStyleSheet("QFrame { background: transparent; }")
            wrapper.enterEvent = lambda event: self.tooltip_label.setText(tooltip_text)
            wrapper.leaveEvent = lambda event: self.tooltip_label.setText(placeholder)
            return wrapper


        # Label
        label = QLabel("Choose type of generation:")
        center_layout.addWidget(label)

        # Radio buttons
        self.simple_button = QRadioButton("Simple Generation")
        self.adv_button = QRadioButton("Advanced Generation")

        radio_style = """
            QRadioButton {
                min-width: 350px;
                min-height: 45px;
                font-size: 20px;
                padding: 5px;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
            }
        """
        label_style = """
            QLabel {
                min-width: 350px;
                min-height: 30px;
                font-size: 20px;
                font-weight: bold;
                padding: 5px;
                qproperty-alignment: AlignCenter;
            }
        """

        self.simple_button.setStyleSheet(radio_style)
        self.adv_button.setStyleSheet(radio_style)
        label.setStyleSheet(label_style)

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.simple_button, 0)
        self.button_group.addButton(self.adv_button, 1)
        self.simple_button.setChecked(True)

        # Layout for buttons stacked vertically
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.simple_button)
        button_layout.addWidget(self.adv_button)
        button_layout.setAlignment(Qt.AlignCenter)

        center_layout.addLayout(button_layout)


        # Add to outer layout
        outer_layout.addLayout(center_layout)

        # Set main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        main_layout.setContentsMargins(0, 0, 0, 0)

    def nextId(self):
        if self.simple_button.isChecked():
            return 2
        elif self.adv_button.isChecked():
            return 5
        return 2

class VerDescriptionPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Version and Description Setup")
        self.setSubTitle("Choose version and set a description for the configuration file")

        placeholder = "Hover over a field to see its description"

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        scroll.setWidget(container)

        main_layout = QVBoxLayout(container)
        main_layout.setAlignment(Qt.AlignLeft)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Tooltip label
        self.tooltip_label = QLabel(placeholder)
        self.tooltip_label.setWordWrap(True)
        self.tooltip_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tooltip_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tooltip_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: #f0f0f0;
                padding: 6px 12px;
                font-style: italic;
                color: #333;
                min-height: 40px;
            }
        """)

        # --- Hover wrapper ---
        def create_hover_widget(layout, tooltip_text):
            wrapper = QFrame()
            wrapper.setLayout(layout)
            wrapper.setMouseTracking(True)
            wrapper.setStyleSheet("QFrame { background: transparent; }")
            wrapper.enterEvent = lambda event: self.tooltip_label.setText(tooltip_text)
            wrapper.leaveEvent = lambda event: self.tooltip_label.setText(placeholder)
            return wrapper

        # Version
        version_layout = QHBoxLayout()
        version_layout.setAlignment(Qt.AlignLeft)
        version_label = QLabel("Version:")
        self.version_value = QLineEdit("0.2")
        self.version_value.setFixedWidth(100)
        version_layout.addWidget(version_label)
        version_layout.addWidget(self.version_value)
        version_widget = create_hover_widget(version_layout, "Version: The version number of the configuration.")

        # Description
        description_layout = QHBoxLayout()
        description_layout.setAlignment(Qt.AlignLeft)
        description_label = QLabel("Description:")
        self.description_value = QLineEdit("The pipeline and analysis JSON for the OCVL's MEAOSLO.")
        self.description_value.setFixedWidth(400)
        description_layout.addWidget(description_label)
        description_layout.addWidget(self.description_value)
        description_widget = create_hover_widget(description_layout, "Description: A brief explanation of what this configuration does.")

        # Add to layout
        main_layout.addWidget(version_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(description_widget, alignment=Qt.AlignLeft)
        main_layout.addStretch()
        main_layout.addWidget(self.tooltip_label)

        # Final page layout
        page_layout = QVBoxLayout(self)
        page_layout.addWidget(scroll)
        page_layout.setContentsMargins(0, 0, 0, 0)

    def nextId(self):
        return 3


class PreanalysisPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Pre-Analysis Pipeline Setup")
        self.setSubTitle("Configure pre-analysis pipeline below")

        placeholder = "Hover over a field to see its description"

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        scroll.setWidget(container)

        main_layout = QVBoxLayout(container)
        main_layout.setAlignment(Qt.AlignLeft)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Tooltip label
        self.tooltip_label = QLabel(placeholder)
        self.tooltip_label.setWordWrap(True)
        self.tooltip_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tooltip_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tooltip_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: #f0f0f0;
                padding: 6px 12px;
                font-style: italic;
                color: #333;
                min-height: 40px;
            }
        """)

        # --- Helper hover wrapper ---
        def create_hover_widget(layout, tooltip_text):
            wrapper = QFrame()
            wrapper.setLayout(layout)
            wrapper.setMouseTracking(True)
            wrapper.setStyleSheet("QFrame { background: transparent; }")
            wrapper.enterEvent = lambda event: self.tooltip_label.setText(tooltip_text)
            wrapper.leaveEvent = lambda event: self.tooltip_label.setText(placeholder)
            return wrapper

        splitter1 = QLabel("Formats:")
        splitter1.setFont(bold)

        # Image Format
        image_layout = QHBoxLayout()
        image_layout.setAlignment(Qt.AlignLeft)
        image_label = QLabel("Image Format:")
        self.image_format_value = constructors.FormatEditorWidget("Image Format:", "{IDnum}_{Year}{Month}{Day}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}.tif", type='image')
        image_layout.addWidget(image_label)
        image_layout.addWidget(self.image_format_value)
        image_widget = create_hover_widget(image_layout, "Image Format: Format string for image filenames.")

        # Video Format
        video_layout = QHBoxLayout()
        video_layout.setAlignment(Qt.AlignLeft)
        video_label = QLabel("Video Format:")
        self.video_format_value = constructors.FormatEditorWidget("Video Format:", "{IDnum}_{Year}{Month}{Day}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}.avi", type='video')
        video_layout.addWidget(video_label)
        video_layout.addWidget(self.video_format_value)
        video_widget = create_hover_widget(video_layout, "Video Format: Format string for video filenames.")

        # Mask Format
        mask_layout = QHBoxLayout()
        mask_layout.setAlignment(Qt.AlignLeft)
        mask_label = QLabel("Mask Format:")
        self.mask_format_value = constructors.FormatEditorWidget("Mask Format:", "{IDnum}_{Year}{Month}{Day}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}.avi", type='mask')
        mask_layout.addWidget(mask_label)
        mask_layout.addWidget(self.mask_format_value)
        mask_widget = create_hover_widget(mask_layout, "Mask Format: Format string for mask filenames.")

        # Recursive Search
        recursive_layout = QHBoxLayout()
        recursive_layout.setAlignment(Qt.AlignLeft)
        recursive_label = QLabel("Recursive Search:")
        self.recursive_search_tf = constructors.TrueFalseSelector(False)
        recursive_layout.addWidget(recursive_label)
        recursive_layout.addWidget(self.recursive_search_tf)
        recursive_widget = create_hover_widget(recursive_layout, "Recursive Search: If enabled, subfolders will be searched recursively.")

        splitter2 = QLabel("Pipeline Parameters:")
        splitter2.setFont(bold)

        # Modalities
        self.modalities_layout = QHBoxLayout()
        self.modalities_layout.setAlignment(Qt.AlignLeft)
        modalities_label = QLabel("Modalities:")
        self.modalities_list_creator = constructors.ListEditorWidget()
        self.modalities_layout.addWidget(modalities_label)
        self.modalities_layout.addWidget(self.modalities_list_creator)
        modalities_widget = create_hover_widget(self.modalities_layout, "Modalities: List of imaging modalities (requires '{Modality}' in at least one format string to edit).")

        # Alignment Reference
        alignment_layout = QHBoxLayout()
        alignment_layout.setAlignment(Qt.AlignLeft)
        alignment_label = QLabel("Alignment Reference Modality:")
        self.alignment_ref_value = constructors.AlignmentModalitySelector(self.modalities_list_creator, "null")
        alignment_layout.addWidget(alignment_label)
        alignment_layout.addWidget(self.alignment_ref_value)
        alignment_widget = create_hover_widget(alignment_layout, "Alignment Reference Modality: Reference modality used for alignment (Note at least one modality must be entered to edit).")

        # Group By
        groupby_layout = QHBoxLayout()
        groupby_layout.setAlignment(Qt.AlignLeft)
        groupby_label = QLabel("Group By:")
        self.groupby_value = constructors.GroupByFormatEditorWidget(
            image_format="",
            video_format="",
            mask_format="",
            label_text="Group By:",
            default_format="null"
        )
        groupby_layout.addWidget(groupby_label)
        groupby_layout.addWidget(self.groupby_value)
        groupby_widget = create_hover_widget(groupby_layout, "Group By: Field used to group sessions across selected elements (optional, available options include those in format strings).")

        for key, widget in [("image_format", self.image_format_value),
                            ("video_format", self.video_format_value),
                            ("mask_format", self.mask_format_value)]:
            widget.section_name = "preanalysis"
            widget.format_key = key
            widget.copyToAllRequested.connect(self.copy_format_to_all)

        # Add all widgets to layout
        main_layout.addWidget(splitter1, alignment=Qt.AlignLeft)
        main_layout.addWidget(image_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(video_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(mask_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(recursive_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(splitter2, alignment=Qt.AlignLeft)
        main_layout.addWidget(modalities_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(alignment_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(groupby_widget, alignment=Qt.AlignLeft)
        main_layout.addStretch()
        main_layout.addWidget(self.tooltip_label)

        # Connections
        self.image_format_value.formatChanged.connect(self.update_groupby_elements)
        self.video_format_value.formatChanged.connect(self.update_groupby_elements)
        self.mask_format_value.formatChanged.connect(self.update_groupby_elements)

        self.image_format_value.formatChanged.connect(self.update_modality_editor_state)
        self.video_format_value.formatChanged.connect(self.update_modality_editor_state)
        self.mask_format_value.formatChanged.connect(self.update_modality_editor_state)

        self.modalities_list_creator.itemsChanged.connect(self.update_alignment_modality_state)
        self.update_alignment_modality_state()

        page_layout = QVBoxLayout(self)
        page_layout.addWidget(scroll)
        page_layout.setContentsMargins(0, 0, 0, 0)

    def copy_format_to_all(self, section, key, value):
        if section != "preanalysis":
            return
        for k, widget in [("image_format", self.image_format_value),
                          ("video_format", self.video_format_value),
                          ("mask_format", self.mask_format_value)]:
            if k != key:
                widget.set_value(value)

    def update_alignment_modality_state(self):
        modalities_empty = not bool(self.modalities_list_creator.get_list())
        self.alignment_ref_value.setEnabled(not modalities_empty)

    def update_groupby_elements(self):
        self.groupby_value.image_format = self.image_format_value.get_value()
        self.groupby_value.video_format = self.video_format_value.get_value()
        self.groupby_value.mask_format = self.mask_format_value.get_value()

    def update_modality_editor_state(self):
        image_has_modality = has_modality(self.image_format_value.get_value())
        video_has_modality = has_modality(self.video_format_value.get_value())
        mask_has_modality = has_modality(self.mask_format_value.get_value())
        any_has_modality = image_has_modality or video_has_modality or mask_has_modality
        self.modalities_list_creator.setEnabled(any_has_modality)

    def nextId(self):
        return 4


class AnalysisPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Analysis Pipeline Setup")
        self.setSubTitle('Configure analysis pipeline below')

        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QFrame, QSizePolicy

        bold = QFont()
        bold.setBold(True)

        placeholder = "Hover over a field to see its description"

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        scroll.setWidget(container)

        main_layout = QVBoxLayout(container)
        main_layout.setAlignment(Qt.AlignLeft)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Tooltip label at the bottom
        self.tooltip_label = QLabel(placeholder)
        self.tooltip_label.setWordWrap(True)
        self.tooltip_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tooltip_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tooltip_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: #f0f0f0;
                padding: 6px 12px;
                font-style: italic;
                color: #333;
                min-height: 40px;
            }
        """)

        # Helper function to wrap a layout and set hover events
        def create_hover_widget(layout, tooltip_text):
            wrapper = QFrame()
            wrapper.setLayout(layout)
            wrapper.setMouseTracking(True)
            wrapper.setStyleSheet("QFrame { background: transparent; }")
            wrapper.enterEvent = lambda event: self.tooltip_label.setText(tooltip_text)
            wrapper.leaveEvent = lambda event: self.tooltip_label.setText("Hover over a field to see its description")
            return wrapper

        splitter1 = QLabel("Formats:")
        splitter1.setFont(bold)

        image_layout = QHBoxLayout()
        image_layout.setAlignment(Qt.AlignLeft)
        image_label = QLabel("Image Format:")
        self.image_format_value = constructors.FormatEditorWidget("Image Format:", "{IDnum}_{Year}{Month}{Day}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}.tif", type='image')
        image_layout.addWidget(image_label)
        image_layout.addWidget(self.image_format_value)
        image_widget = create_hover_widget(image_layout, "Image Format: Format string for image filenames.")

        queryloc_layout = QHBoxLayout()
        queryloc_layout.setAlignment(Qt.AlignLeft)
        queryloc_label = QLabel("Query Loc Format:")
        self.queryloc_format_value = constructors.FormatEditorWidget("Queryloc Format:", "{IDnum}_{Year}{Month}{Day}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}.csv", type='queryloc')
        queryloc_layout.addWidget(queryloc_label)
        queryloc_layout.addWidget(self.queryloc_format_value)
        queryloc_widget = create_hover_widget(queryloc_layout, "Query Loc Format: Filename format used to locate the reference location (query).")

        video_layout = QHBoxLayout()
        video_layout.setAlignment(Qt.AlignLeft)
        video_label = QLabel("Video Format:")
        self.video_format_value = constructors.FormatEditorWidget("Video Format:", "{IDnum}_{Year}{Month}{Day}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}.avi", type='video')
        video_layout.addWidget(video_label)
        video_layout.addWidget(self.video_format_value)
        video_widget = create_hover_widget(video_layout, "Video Format: Format string for video filenames.")

        recursive_layout = QHBoxLayout()
        recursive_layout.setAlignment(Qt.AlignLeft)
        recursive_label = QLabel("Recursive Search:")
        self.recursive_search_tf = constructors.TrueFalseSelector(False)
        recursive_layout.addWidget(recursive_label)
        recursive_layout.addWidget(self.recursive_search_tf)
        recursive_widget = create_hover_widget(recursive_layout, "Recursive Search: If enabled, subfolders will be searched recursively.")

        splitter2 = QLabel("Pipeline Parameters:")
        splitter2.setFont(bold)

        self.modalities_layout = QHBoxLayout()
        self.modalities_layout.setAlignment(Qt.AlignLeft)
        modalities_label = QLabel("Modalities:")
        self.modalities_list_creator = constructors.ListEditorWidget()
        self.modalities_layout.addWidget(modalities_label)
        self.modalities_layout.addWidget(self.modalities_list_creator)
        modalities_widget = create_hover_widget(self.modalities_layout, "Modalities: List of imaging modalities (requires '{Modality}' in at least one format string to edit).")

        for key, widget in [("image_format", self.image_format_value),
                            ("video_format", self.video_format_value),
                            ("queryloc_format", self.queryloc_format_value)]:
            widget.section_name = "analysis"
            widget.format_key = key
            widget.copyToAllRequested.connect(self.copy_format_to_all)

        # Add all widgets to main layout
        main_layout.addWidget(splitter1, alignment=Qt.AlignLeft)
        main_layout.addWidget(image_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(queryloc_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(video_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(recursive_widget, alignment=Qt.AlignLeft)
        main_layout.addWidget(splitter2, alignment=Qt.AlignLeft)
        main_layout.addWidget(modalities_widget, alignment=Qt.AlignLeft)
        main_layout.addStretch()
        main_layout.addWidget(self.tooltip_label)
        main_layout.setAlignment(Qt.AlignLeft)

        # Connections
        self.image_format_value.formatChanged.connect(self.update_modality_editor_state)
        self.queryloc_format_value.formatChanged.connect(self.update_modality_editor_state)
        self.video_format_value.formatChanged.connect(self.update_modality_editor_state)

        # Set main layout with scroll area
        page_layout = QVBoxLayout(self)
        page_layout.addWidget(scroll)
        page_layout.setContentsMargins(0, 0, 0, 0)

    def copy_format_to_all(self, section, key, value):
        if section != "analysis":
            return
        for k, widget in [("image_format", self.image_format_value),
                          ("video_format", self.video_format_value),
                          ("qyueryloc_format", self.queryloc_format_value)]:
            if k != key:
                widget.set_value(value)

    def update_modality_editor_state(self):
        image_has_modality = has_modality(self.image_format_value.get_value())
        queryloc_has_modality = has_modality(self.queryloc_format_value.get_value())
        video_has_modality = has_modality(self.video_format_value.get_value())
        any_has_modality = image_has_modality or queryloc_has_modality or video_has_modality
        self.modalities_list_creator.setEnabled(any_has_modality)


    def nextId(self):
        return 6

class AdvancedSetupPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Advanced Setup")
        self.setSubTitle(
            'You can edit any configuration field from here. For a more simple setup, go back and select "Simple Setup"\n'
            'Tip: Expand window for better visibility')

        with open("master_config_files/advanced_config_JSON.json", "r") as f:
            advanced_config_json = json.load(f)
        with open("master_config_files/master_JSON.json", "r") as f:
            self.master_json = json.load(f)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        layout = QVBoxLayout(self)

        # Store the advanced widget as an instance attribute
        self.advanced_widget = build_form_from_template(self.master_json, advanced_config_json, adv=False)

        scroll_area.setWidget(self.advanced_widget)

        layout.addWidget(scroll_area)
        layout.setAlignment(Qt.AlignJustify)

    def nextId(self):
        return 6

class ReviewPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Review Changes")
        self.setSubTitle('Review your settings and click "Save >" to generate the configuration JSON file.')

        self.saved_file_path = None
        self.generated_config = None

        self.layout = QVBoxLayout(self)
        self.label = QLabel()
        self.label.setWordWrap(True)

        self.label2 = QLabel()
        self.label2.setWordWrap(True)

        self.label3 = QLabel()
        self.label3.setWordWrap(True)

        self.label4 = QLabel()
        self.label4.setWordWrap(True)

        self.label5 = QLabel()
        self.label5.setWordWrap(True)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.label3)
        self.layout.addWidget(self.label4)
        self.layout.addWidget(self.label5)

        self.layout.addStretch()

    def nextId(self):
        return 8

    def initializePage(self):
        wizard = self.wizard()

        if wizard.page(1).adv_button.isChecked():
            self.label.setText(
                "Advanced setup selected.\n\nPlease review your inputs on the previous page before continuing.")
            self.generated_config = None  # Don't regenerate config in advanced mode
            return

        # Simple mode: generate config with only simple-mode configurable elements
        config = {
            "version": wizard.page(2).version_value.text(),
            "description": wizard.page(2).description_value.text(),
            "preanalysis": {
                "image_format": wizard.page(3).image_format_value.get_value(),
                "video_format": wizard.page(3).video_format_value.get_value(),
                "mask_format": wizard.page(3).mask_format_value.get_value(),
                "recursive_search" : wizard.page(3).recursive_search_tf.get_value(),
                "pipeline_params": {
                    "modalities": wizard.page(3).modalities_list_creator.get_list(),
                    "alignment_reference_modality" : wizard.page(3).alignment_ref_value.get_value(),
                    "group_by": None if wizard.page(3).groupby_value.get_value() == "null" else wizard.page(3).groupby_value.get_value(),
                }
            },
            "analysis": {
                "image_format": wizard.page(4).image_format_value.get_value(),
                "queryloc_format": wizard.page(4).queryloc_format_value.get_value(),
                "video_format": wizard.page(4).video_format_value.get_value(),
                "recursive_search": wizard.page(4).recursive_search_tf.get_value(),
                "analysis_params": {
                    "modalities": wizard.page(4).modalities_list_creator.get_list()
                }
            }
        }

        self.generated_config = config

        # Show summary to user
        summary_text1 = f"Version: {config['version']}\n" \
                       f"Description: {config['description']}\n"

        summary_text2 = f"Image Format: {config['preanalysis']['image_format']}\n" \
                       f"Video Format: {config['preanalysis']['video_format']}\n" \
                       f"Mask Format: {config['preanalysis']['mask_format']}\n" \
                       f"Recursive Search: {config['preanalysis']['recursive_search']}\n" \
                       f"Modalities: {config['preanalysis']['pipeline_params']['modalities']}\n" \
                       f"Alignment Reference Modality : {config['preanalysis']['pipeline_params']['alignment_reference_modality']}\n" \
                       f"Group By: {config['preanalysis']['pipeline_params']['group_by']}\n"

        summary_text3 = f"Image Format: {config['analysis']['image_format']}\n" \
                       f"QueryLoc Format: {config['analysis']['queryloc_format']}\n" \
                       f"Video Format: {config['analysis']['video_format']}\n" \
                       f"Recursive Search: {config['analysis']['recursive_search']}\n" \
                       f"Modalities: {config['analysis']['analysis_params']['modalities']}"

        self.label.setText(
            "Please review the following configurations before clicking 'Save >':\n\n" + summary_text1)
        self.label2.setText('Pre-Analysis')
        self.label2.setFont(bold)
        self.label3.setText(summary_text2)
        self.label4.setText('Analysis')
        self.label4.setFont(bold)
        self.label5.setText(summary_text3)

    def validatePage(self):
        wizard = self.wizard()

        if wizard.page(1).adv_button.isChecked():
            # Advanced mode: generate config from widgets directly
            advanced_widget = wizard.page(5).advanced_widget
            master_json = wizard.page(5).master_json
            config = generate_json(advanced_widget, master_json)
        else:
            # Simple mode: use the generated config from initializePage
            config = self.generated_config
            if not config:
                self.label.setText("No configuration data available")
                return False

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            wizard,
            "Save Configuration File",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            if not file_path.endswith('.json'):
                file_path += '.json'

            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.saved_file_path = file_path
                return True
            except Exception as e:
                self.label.setText(f"Failed to save file:\n{str(e)}")
                return False
        else:
            self.label.setText("No file selected. Please choose a path to save the configuration.")
            return False


class ImportEditorPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Edit JSON")
        self.setSubTitle('Edit your imported JSON here. Click "Save >" to save new JSON.')

        self.saved_file_path = None
        self.form_widget = None

        # Create the main layout
        self.layout = QVBoxLayout(self)

        # Create a placeholder widget that will be replaced in initializePage
        self.placeholder_label = QLabel("Loading imported configuration...")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.placeholder_label)

    def initializePage(self):
        # Clear the placeholder
        self.layout.removeWidget(self.placeholder_label)
        self.placeholder_label.deleteLater()

        wizard = self.wizard()
        intro_page = wizard.page(0)

        if hasattr(intro_page, 'imported_config') and intro_page.imported_config:
            with open("config_files/master_JSON.json", "r") as f:
                master_json = json.load(f)

            self.form_widget = build_form_from_template(master_json, intro_page.imported_config)

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(self.form_widget)

            self.layout.addWidget(scroll_area)
        else:
            error_label = QLabel("No imported configuration found.")
            error_label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(error_label)

    def validatePage(self):
        if not self.form_widget:
            QMessageBox.warning(self, "Error", "No configuration to save")
            return False

        # Get the master template
        with open("config_files/master_JSON.json", "r") as f:
            master_json = json.load(f)

        wizard = self.wizard()
        intro_page = wizard.page(0)

        # Generate the JSON from the form
        config = generate_json(self.form_widget, master_json)

        if not config:
            QMessageBox.warning(self, "Error", "No configuration data to save")
            return False

        # Show save file dialog
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Configuration File",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            if not file_path.endswith('.json'):
                file_path += '.json'

            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.saved_file_path = file_path
                return True
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save file:\n{str(e)}")
                return False

        QMessageBox.warning(self, "Error", "No file selected")
        return False

    def nextId(self):
        return 8

class EndPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Generation Complete!")
        self.setSubTitle("JSON configuration file has been created")

        # GIF setup
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        movie = QMovie("images/spincat.gif")
        self.image.setMovie(movie)
        movie.start()

        # Path label
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setAlignment(Qt.AlignCenter)

        # Center the GIF using an HBox layout
        image_layout = QHBoxLayout()
        image_layout.addStretch()
        image_layout.addWidget(self.image)
        image_layout.addStretch()

        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.path_label)
        layout.addLayout(image_layout)
        layout.addStretch()

    def initializePage(self):
        finalize_page = self.wizard().page(6)
        import_page = self.wizard().page(7)
        if finalize_page.saved_file_path:
            self.path_label.setText(f"Configuration file saved to:\n{finalize_page.saved_file_path}")
        elif import_page.saved_file_path:
            self.path_label.setText(f"Configuration file saved to:\n{import_page.saved_file_path}")
        else:
            self.path_label.setText("Configuration file was not saved. Contact admin if issues persist.")
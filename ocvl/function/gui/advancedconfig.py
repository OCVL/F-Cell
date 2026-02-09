from optparse import Option

from PySide6.QtWidgets import (QWidget, QVBoxLayout,
                               QLabel, QPushButton, QHBoxLayout,
                               QScrollArea, QMainWindow, QFileDialog, QMessageBox, QCheckBox)
from PySide6.QtCore import Qt
import json, constructors

from constructors import OptionalField


class version_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Version", True)
        collapsable_layout = QHBoxLayout()
        collapsable_layout.setSpacing(30)

        label = QLabel("Version:")
        self.version_value = constructors.FreetextBox("0.2")

        collapsable_layout.addWidget(label)
        collapsable_layout.addWidget(self.version_value)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return self.version_value.get_text()

class description_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Description", True)
        collapsable_layout = QHBoxLayout()
        collapsable_layout.setSpacing(30)

        label = QLabel("Description:")
        self.description_value = constructors.FreetextBox("The pipeline and analysis JSON for the OCVL's MEAOSLO.")

        collapsable_layout.addWidget(label)
        collapsable_layout.addWidget(self.description_value)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return self.description_value.get_text()

class raw_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Raw", False)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        video_format_layout = QHBoxLayout()
        video_format_layout.setSpacing(30)
        video_format_label = QLabel("Video Format:")
        self.video_format_value = constructors.FormatEditorWidget("Video Format:", "{IDnum}_{Year4}{Month}{Day}_{VidNum}_{Modality}")
        self.video_format_optional = constructors.OptionalField(self.video_format_value)

        video_format_layout.addWidget(video_format_label)
        video_format_layout.addWidget(self.video_format_optional)
        video_format_layout.setAlignment(Qt.AlignLeft)

        collapsable_layout.addLayout(video_format_layout)
        self.raw_metadata_layer = raw_metadata_layer()
        collapsable_layout.addWidget(self.raw_metadata_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}
        if self.video_format_optional.is_checked():
            result["video_format"] = self.video_format_optional.get_widget().get_value()

        # No need to wrap this — it already uses is_enabled()
        metadata_val = self.raw_metadata_layer.get_value()
        if metadata_val is not None:
            result["metadata"] = metadata_val

        return result if result else None

class raw_metadata_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Metadata", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        # --- Type field ---
        type_layout = QHBoxLayout()
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_value = constructors.FreetextBox("text_file")
        self.type_optional = constructors.OptionalField(self.type_value)

        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_optional)

        # --- Metadata Format field ---
        format_layout = QHBoxLayout()
        format_layout.setSpacing(30)
        metadata_format_label = QLabel("Metadata Format:")
        self.metadata_format_value = constructors.FormatEditorWidget(
            "Metadata Format:", "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}"
        )
        self.metadata_format_optional = constructors.OptionalField(self.metadata_format_value)

        format_layout.addWidget(metadata_format_label)
        format_layout.addWidget(self.metadata_format_optional)
        format_layout.setAlignment(Qt.AlignLeft)

        # --- Add to layout ---
        collapsable_layout.addLayout(type_layout)
        collapsable_layout.addLayout(format_layout)

        self.raw_metadata_fieldstoload_layer = raw_metadata_fieldstoload_layer()
        collapsable_layout.addWidget(self.raw_metadata_fieldstoload_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.type_optional.is_checked():
            result["type"] = self.type_optional.get_widget().get_text()

        if self.metadata_format_optional.is_checked():
            result["metadata_format"] = self.metadata_format_optional.get_widget().get_value()

        fields_val = self.raw_metadata_fieldstoload_layer.get_value()
        if fields_val is not None:
            result["fields_to_load"] = fields_val

        return result if result else None


class raw_metadata_fieldstoload_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Fields to Load", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        # --- Timestamps field with checkbox ---
        timestamps_layout = QHBoxLayout()
        timestamps_layout.setSpacing(30)
        timestamps_label = QLabel("Timestamps:")
        self.timestamps_value = constructors.FreetextBox("Timestamp_us")
        self.timestamps_optional = constructors.OptionalField(self.timestamps_value)

        timestamps_layout.addWidget(timestamps_label)
        timestamps_layout.addWidget(self.timestamps_optional)

        # --- Stimulus Train field with checkbox ---
        stimulus_layout = QHBoxLayout()
        stimulus_layout.setSpacing(30)
        stimulus_label = QLabel("Stimulus Train:")
        self.stimulus_value = constructors.FreetextBox("StimulusOn")
        self.stimulus_optional = constructors.OptionalField(self.stimulus_value)

        stimulus_layout.addWidget(stimulus_label)
        stimulus_layout.addWidget(self.stimulus_optional)

        # Add both layouts
        collapsable_layout.addLayout(timestamps_layout)
        collapsable_layout.addLayout(stimulus_layout)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.timestamps_optional.is_checked():
            result["timestamps"] = self.timestamps_optional.get_widget().get_text()

        if self.stimulus_optional.is_checked():
            result["stimulus_train"] = self.stimulus_optional.get_widget().get_text()

        return result if result else None


class preanalysis_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Pre-analysis", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        # --- Image Format with checkbox ---
        image_format_widget = QWidget()
        image_format_layout = QHBoxLayout(image_format_widget)
        image_format_layout.setSpacing(30)
        image_format_label = QLabel("Image Format:")
        self.image_format_value = constructors.FormatEditorWidget("Image Format:", "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}")
        image_format_layout.addWidget(image_format_label)
        image_format_layout.addWidget(self.image_format_value)
        image_format_layout.setAlignment(Qt.AlignLeft)
        self.image_format_optional = constructors.OptionalField(image_format_widget)

        # --- Video Format with checkbox ---
        video_format_widget = QWidget()
        video_format_layout = QHBoxLayout(video_format_widget)
        video_format_layout.setSpacing(30)
        video_format_label = QLabel("Video Format:")
        self.video_format_value = constructors.FormatEditorWidget("Video Format:", "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}")
        video_format_layout.addWidget(video_format_label)
        video_format_layout.addWidget(self.video_format_value)
        video_format_layout.setAlignment(Qt.AlignLeft)
        self.video_format_optional = constructors.OptionalField(video_format_widget)

        # --- Mask Format with checkbox ---
        mask_format_widget = QWidget()
        mask_format_layout = QHBoxLayout(mask_format_widget)
        mask_format_layout.setSpacing(30)
        mask_format_label = QLabel("Mask Format:")
        self.mask_format_value = constructors.FormatEditorWidget("Mask Format:", "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}")
        mask_format_layout.addWidget(mask_format_label)
        mask_format_layout.addWidget(self.mask_format_value)
        mask_format_layout.setAlignment(Qt.AlignLeft)
        self.mask_format_optional = constructors.OptionalField(mask_format_widget)

        # --- Recursive Search ---
        recursive_search_widget = QWidget()
        recursive_search_layout = QHBoxLayout(recursive_search_widget)
        recursive_search_layout.setSpacing(30)
        recursive_search_label = QLabel("Recursive Search:")
        self.recursive_search_tf = constructors.TrueFalseSelector(False)
        recursive_search_layout.addWidget(recursive_search_label)
        recursive_search_layout.addWidget(self.recursive_search_tf)
        recursive_search_layout.setAlignment(Qt.AlignLeft)
        self.recursive_search_optional = constructors.OptionalField(recursive_search_widget)

        # --- Collapsable children ---
        collapsable_layout.addWidget(self.image_format_optional)
        collapsable_layout.addWidget(self.video_format_optional)
        collapsable_layout.addWidget(self.mask_format_optional)

        self.preanalysis_metadata_layer = preanalysis_metadata_layer()
        collapsable_layout.addWidget(self.preanalysis_metadata_layer)

        collapsable_layout.addWidget(self.recursive_search_optional)

        self.preanalysis_pipeline_params_layer = preanalysis_pipeline_params_layer()
        collapsable_layout.addWidget(self.preanalysis_pipeline_params_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

        # Connect format changes to update groupby
        self.image_format_value.formatChanged.connect(self.update_groupby_elements)
        self.video_format_value.formatChanged.connect(self.update_groupby_elements)
        self.mask_format_value.formatChanged.connect(self.update_groupby_elements)

    def update_groupby_elements(self):
        """Update the groupby widget's available elements when formats change"""
        self.preanalysis_pipeline_params_layer.update_format_references(
            image_format=self.image_format_value.get_value(),
            video_format=self.video_format_value.get_value(),
            mask_format=self.mask_format_value.get_value()
        )

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.image_format_optional.is_checked():
            result["image_format"] = self.image_format_value.get_value()
        if self.video_format_optional.is_checked():
            result["video_format"] = self.video_format_value.get_value()
        if self.mask_format_optional.is_checked():
            result["mask_format"] = self.mask_format_value.get_value()
        metadata_val = self.preanalysis_metadata_layer.get_value()
        if metadata_val is not None:
            result["metadata"] = metadata_val
        if self.recursive_search_optional.is_checked():
            result["recursive_search"] = self.recursive_search_tf.get_value()
        pipeline_val = self.preanalysis_pipeline_params_layer.get_value()
        if pipeline_val is not None:
            result["pipeline_params"] = pipeline_val

        return result if result else None

class preanalysis_metadata_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Metadata", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        type_widget = QWidget()
        type_layout = QHBoxLayout(type_widget)
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_value = constructors.FreetextBox("text_file")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_value)
        self.type_optional = OptionalField(type_widget)

        format_widget = QWidget()
        format_layout = QHBoxLayout(format_widget)
        format_layout.setSpacing(30)
        metadata_format_label = QLabel("Metadata Format:")
        self.metadata_format_value = constructors.FormatEditorWidget("Metadata Format:", "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}")
        format_layout.addWidget(metadata_format_label)
        format_layout.addWidget(self.metadata_format_value)
        format_layout.setAlignment(Qt.AlignLeft)
        self.format_optional = OptionalField(format_widget)

        collapsable_layout.addWidget(self.type_optional)
        collapsable_layout.addWidget(self.format_optional)
        self.preanalysis_metadata_fieldstoload_layer = preanalysis_metadata_fieldstoload_layer()
        collapsable_layout.addWidget(self.preanalysis_metadata_fieldstoload_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.type_optional.is_checked():
            result["type"] = self.type_value.get_text()
        if self.format_optional.is_checked():
            result["metadata_format"] = self.metadata_format_value.get_value()
        fieldstoload_val = self.preanalysis_metadata_fieldstoload_layer.get_value()
        if fieldstoload_val is not None:
            result["fields_to_load"] = fieldstoload_val

        return result if result else None


class preanalysis_metadata_fieldstoload_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Fields to Load", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        framestamps_widget = QWidget()
        framestamps_layout = QHBoxLayout(framestamps_widget)
        framestamps_layout.setSpacing(30)
        framestamps_label = QLabel("Framestamps:")
        self.framestamps_value = constructors.FreetextBox("OriginalFrameNumber")
        framestamps_layout.addWidget(framestamps_label)
        framestamps_layout.addWidget(self.framestamps_value)
        self.framestamps_optional = OptionalField(framestamps_widget)

        collapsable_layout.addWidget(self.framestamps_optional)
        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.framestamps_optional.is_checked():
            result["framestamps"] = self.framestamps_value.get_text()

        return result if result else None

class preanalysis_pipeline_params_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Pipeline Params", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        gausblur_widget = QWidget()
        gausblur_layout = QHBoxLayout(gausblur_widget)
        gausblur_layout.setSpacing(30)
        gausblur_label = QLabel("Gaus Blur:")
        self.gausblur_value = constructors.FreetextBox("0.0")
        gausblur_layout.addWidget(gausblur_label)
        gausblur_layout.addWidget(self.gausblur_value)
        self.gausblur_optional = OptionalField(gausblur_widget)

        modalities_widget = QWidget()
        modalities_layout = QHBoxLayout(modalities_widget)
        modalities_layout.setSpacing(30)
        modalities_label = QLabel("Modalities:")
        self.modalities_list_creator = constructors.ListEditorWidget()
        modalities_layout.addWidget(modalities_label)
        modalities_layout.addWidget(self.modalities_list_creator)
        modalities_layout.setAlignment(Qt.AlignLeft)
        self.modalities_optional = OptionalField(modalities_widget)

        alignment_ref_widget = QWidget()
        alignment_ref_layout = QHBoxLayout(alignment_ref_widget)
        alignment_ref_layout.setSpacing(30)
        alignment_ref_label = QLabel("Alignment Reference Modality:")
        self.alignment_ref_value = constructors.AlignmentModalitySelector(
            self.modalities_list_creator,
            "null"
        )
        alignment_ref_layout.addWidget(alignment_ref_label)
        alignment_ref_layout.addWidget(self.alignment_ref_value)
        alignment_ref_layout.setAlignment(Qt.AlignLeft)
        self.alignment_ref_optional = OptionalField(alignment_ref_widget)

        output_folder_widget = QWidget()
        output_folder_layout = QHBoxLayout(output_folder_widget)
        output_folder_layout.setSpacing(30)
        output_folder_label = QLabel("Output Folder:")
        self.output_folder_value = constructors.OpenFolder()
        output_folder_layout.addWidget(output_folder_label)
        output_folder_layout.addWidget(self.output_folder_value)
        output_folder_layout.setAlignment(Qt.AlignLeft)
        self.output_folder_optional = OptionalField(output_folder_widget)

        groupby_widget = QWidget()
        groupby_layout = QHBoxLayout(groupby_widget)
        groupby_layout.setSpacing(30)
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
        groupby_layout.setAlignment(Qt.AlignLeft)
        self.groupby_optional = OptionalField(groupby_widget)

        correct_torsion_widget = QWidget()
        correct_torsion_layout = QHBoxLayout(correct_torsion_widget)
        correct_torsion_layout.setSpacing(30)
        correct_torsion_label = QLabel("Correct Torsion:")
        self.correct_torsion_tf = constructors.TrueFalseSelector(True)
        correct_torsion_layout.addWidget(correct_torsion_label)
        correct_torsion_layout.addWidget(self.correct_torsion_tf)
        self.correct_torsion_optional = OptionalField(correct_torsion_widget)

        intra_stack_xform_widget = QWidget()
        intra_stack_xform_layout = QHBoxLayout(intra_stack_xform_widget)
        intra_stack_xform_layout.setSpacing(30)
        intra_stack_xform_label = QLabel("Intra Stack Xform:")
        self.intra_stack_xform_tf = constructors.AffineRigidSelector(True)
        intra_stack_xform_layout.addWidget(intra_stack_xform_label)
        intra_stack_xform_layout.addWidget(self.intra_stack_xform_tf)
        self.intra_stack_xform_optional = OptionalField(intra_stack_xform_widget)

        inter_stack_xform_widget = QWidget()
        inter_stack_xform_layout = QHBoxLayout(inter_stack_xform_widget)
        inter_stack_xform_layout.setSpacing(30)
        inter_stack_xform_label = QLabel("Inter Stack Xform:")
        self.inter_stack_xform_tf = constructors.AffineRigidSelector(True)
        inter_stack_xform_layout.addWidget(inter_stack_xform_label)
        inter_stack_xform_layout.addWidget(self.inter_stack_xform_tf)
        self.inter_stack_xform_optional = OptionalField(inter_stack_xform_widget)

        flat_field_widget = QWidget()
        flat_field_layout = QHBoxLayout(flat_field_widget)
        flat_field_layout.setSpacing(30)
        flat_field_label = QLabel("Flat Field:")
        self.flat_field_tf = constructors.TrueFalseSelector(False)
        flat_field_layout.addWidget(flat_field_label)
        flat_field_layout.addWidget(self.flat_field_tf)
        self.flat_field_optional = OptionalField(flat_field_widget)

        collapsable_layout.addWidget(self.gausblur_optional)
        self.preanalysis_pipeline_params_maskroi_layer = preanalysis_pipeline_params_maskroi_layer()
        collapsable_layout.addWidget(self.preanalysis_pipeline_params_maskroi_layer)
        collapsable_layout.addWidget(self.modalities_optional)
        collapsable_layout.addWidget(self.alignment_ref_optional)
        collapsable_layout.addWidget(self.output_folder_optional)
        collapsable_layout.addWidget(self.groupby_optional)
        collapsable_layout.addWidget(self.correct_torsion_optional)
        collapsable_layout.addWidget(self.intra_stack_xform_optional)
        collapsable_layout.addWidget(self.inter_stack_xform_optional)
        collapsable_layout.addWidget(self.flat_field_optional)
        self.preanalysis_pipeline_params_trim_layer = preanalysis_pipeline_params_trim_layer()
        collapsable_layout.addWidget(self.preanalysis_pipeline_params_trim_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def update_format_references(self, image_format, video_format, mask_format):
        """Update the format references in the groupby widget"""
        self.groupby_value.image_format = image_format
        self.groupby_value.video_format = video_format
        self.groupby_value.mask_format = mask_format

        # Force update of available elements
        self.groupby_value.available_elements = self.groupby_value.get_available_elements()

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.gausblur_optional.is_checked():
            result["gaus_blur"] = parse_value(self.gausblur_value.get_text())
        mask_roi_val = self.preanalysis_pipeline_params_maskroi_layer.get_value()
        if mask_roi_val is not None:
            result["mask_roi"] = mask_roi_val
        if self.modalities_optional.is_checked():
            result["modalities"] = self.modalities_list_creator.get_list()
        if self.alignment_ref_optional.is_checked():
            result["alignment_reference_modality"] = None if self.alignment_ref_value.get_value() == "null" else self.alignment_ref_value.get_value()
        if self.output_folder_optional.is_checked():
            result["output_folder"] = self.output_folder_value.get_text()
        if self.groupby_optional.is_checked():
            result['group_by'] = None if self.groupby_value.get_value() == "null" else self.groupby_value.get_value()
        if self.correct_torsion_optional.is_checked():
            result["correct_torsion"] = self.correct_torsion_tf.get_value()
        if self.intra_stack_xform_optional.is_checked():
            result["intra_stack_xform"] = self.intra_stack_xform_tf.get_value()
        if self.inter_stack_xform_optional.is_checked():
            result["inter_stack_xform"] = self.inter_stack_xform_tf.get_value()
        if self.flat_field_optional.is_checked():
            result["flat_field"] = self.flat_field_tf.get_value()

        trim_val = self.preanalysis_pipeline_params_trim_layer.get_value()
        if trim_val is not None:
            result["trim"] = trim_val

        return result if result else None


class preanalysis_pipeline_params_maskroi_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Mask ROI", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        r_widget = QWidget()
        r_layout = QHBoxLayout(r_widget)
        r_layout.setSpacing(30)
        r_label = QLabel("Starting Row:")
        self.r_value = constructors.FreetextBox("0")
        r_layout.addWidget(r_label)
        r_layout.addWidget(self.r_value)
        self.r_optional = OptionalField(r_widget)

        c_widget = QWidget()
        c_layout = QHBoxLayout(c_widget)
        c_layout.setSpacing(30)
        c_label = QLabel("Starting Column:")
        self.c_value = constructors.FreetextBox("0")
        c_layout.addWidget(c_label)
        c_layout.addWidget(self.c_value)
        self.c_optional = OptionalField(c_widget)

        width_widget = QWidget()
        width_layout = QHBoxLayout(width_widget)
        width_layout.setSpacing(30)
        width_label = QLabel("Width:")
        self.width_value = constructors.FreetextBox("-1")
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.width_value)
        self.width_optional = OptionalField(width_widget)

        height_widget = QWidget()
        height_layout = QHBoxLayout(height_widget)
        height_layout.setSpacing(30)
        height_label = QLabel("Height:")
        self.height_value = constructors.FreetextBox("-1")
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_value)
        self.height_optional = OptionalField(height_widget)

        collapsable_layout.addWidget(self.r_optional)
        collapsable_layout.addWidget(self.c_optional)
        collapsable_layout.addWidget(self.width_optional)
        collapsable_layout.addWidget(self.height_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.r_optional.is_checked():
            result["r"] = parse_value(self.r_value.get_text())
        if self.c_optional.is_checked():
            result["c"] = parse_value(self.c_value.get_text())
        if self.width_optional.is_checked():
            result["width"] = parse_value(self.width_value.get_text())
        if self.height_optional.is_checked():
            result["height"] = parse_value(self.height_value.get_text())

        return result if result else None

class preanalysis_pipeline_params_trim_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Trimming", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        start_idx_widget = QWidget()
        start_idx_layout = QHBoxLayout(start_idx_widget)
        start_idx_layout.setSpacing(30)
        start_idx_label = QLabel("Start Index:")
        self.start_idx_value = constructors.FreetextBox("0.0")
        start_idx_layout.addWidget(start_idx_label)
        start_idx_layout.addWidget(self.start_idx_value)
        self.start_idx_optional = OptionalField(start_idx_widget)

        end_idx_widget = QWidget()
        end_idx_layout = QHBoxLayout(end_idx_widget)
        end_idx_layout.setSpacing(30)
        end_idx_label = QLabel("End Index:")
        self.end_idx_value = constructors.FreetextBox("-1.0")
        end_idx_layout.addWidget(end_idx_label)
        end_idx_layout.addWidget(self.end_idx_value)
        self.end_idx_optional = OptionalField(end_idx_widget)

        collapsable_layout.addWidget(self.start_idx_optional)
        collapsable_layout.addWidget(self.end_idx_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.start_idx_optional.is_checked():
            result["start_idx"] = parse_value(self.start_idx_value.get_text())
        if self.end_idx_optional.is_checked():
            result["end_idx"] = parse_value(self.end_idx_value.get_text())

        return result if result else None

class analysis_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Analysis", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        image_format_widget = QWidget()
        image_format_layout = QHBoxLayout(image_format_widget)
        image_format_layout.setSpacing(30)
        image_format_label = QLabel("Image Format:")
        self.image_format_value = constructors.FormatEditorWidget("Image Format:", "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}")
        image_format_layout.addWidget(image_format_label)
        image_format_layout.addWidget(self.image_format_value)
        image_format_layout.setAlignment(Qt.AlignLeft)
        self.image_format_optional = constructors.OptionalField(image_format_widget)

        queryloc_format_widget = QWidget()
        queryloc_format_layout = QHBoxLayout(queryloc_format_widget)
        queryloc_format_layout.setSpacing(30)
        queryloc_format_label = QLabel("Query Loc Format:")
        self.queryloc_format_value = constructors.FormatEditorWidget("Video Format:", "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}", queryloc=True)
        queryloc_format_layout.addWidget(queryloc_format_label)
        queryloc_format_layout.addWidget(self.queryloc_format_value)
        queryloc_format_layout.setAlignment(Qt.AlignLeft)
        self.queryloc_format_optional = OptionalField(queryloc_format_widget)

        video_format_widget = QWidget()
        video_format_layout = QHBoxLayout(video_format_widget)
        video_format_layout.setSpacing(30)
        video_format_label = QLabel("Video Format:")
        self.video_format_value = constructors.FormatEditorWidget("Video Format:", "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}")
        video_format_layout.addWidget(video_format_label)
        video_format_layout.addWidget(self.video_format_value)
        video_format_layout.setAlignment(Qt.AlignLeft)
        self.video_format_optional = constructors.OptionalField(video_format_widget)

        recursive_search_widget = QWidget()
        recursive_search_layout = QHBoxLayout(recursive_search_widget)
        recursive_search_layout.setSpacing(30)
        recursive_search_label = QLabel("Recursive Search:")
        self.recursive_search_tf = constructors.TrueFalseSelector(False)
        recursive_search_layout.addWidget(recursive_search_label)
        recursive_search_layout.addWidget(self.recursive_search_tf)
        recursive_search_layout.setAlignment(Qt.AlignLeft)
        self.recursive_search_optional = constructors.OptionalField(recursive_search_widget)

        collapsable_layout.addWidget(self.image_format_optional)
        collapsable_layout.addWidget(self.queryloc_format_optional)
        collapsable_layout.addWidget(self.video_format_optional)

        self.analysis_metadata_layer = analysis_metadata_layer()
        collapsable_layout.addWidget(self.analysis_metadata_layer)

        collapsable_layout.addWidget(self.recursive_search_optional)

        self.analysis_control_layer = analysis_control_layer()
        collapsable_layout.addWidget(self.analysis_control_layer)

        self.analysis_analysis_params_layer = analysis_analysis_params_layer()
        collapsable_layout.addWidget(self.analysis_analysis_params_layer)

        self.analysis_display_params_layer = analysis_display_params_layer()
        collapsable_layout.addWidget(self.analysis_display_params_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.image_format_optional.is_checked():
            result["image_format"] = self.image_format_value.get_value()
        if self.queryloc_format_optional.is_checked():
            result["queryloc_format"] = self.queryloc_format_value.get_value()
        if self.video_format_optional.is_checked():
            result["video_format"] = self.video_format_value.get_value()
        metadata_val = self.analysis_metadata_layer.get_value()
        if metadata_val is not None:
            result["metadata"] = metadata_val
        if self.recursive_search_optional.is_checked():
            result["recursive_search"] = self.recursive_search_tf.get_value()
        control_val = self.analysis_control_layer.get_value()
        if control_val is not None:
            result["control"] = control_val
        analysis_params_val = self.analysis_analysis_params_layer.get_value()
        if analysis_params_val is not None:
            result["analysis_params"] = analysis_params_val
        display_params_val = self.analysis_display_params_layer.get_value()
        if display_params_val is not None:
            result["display_params"] = display_params_val

        return result if result else None

class analysis_metadata_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Metadata", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        type_widget = QWidget()
        type_layout = QHBoxLayout(type_widget)
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_value = constructors.FreetextBox("text_file")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_value)
        self.type_optional = OptionalField(type_widget)

        format_widget = QWidget()
        format_layout = QHBoxLayout(format_widget)
        format_layout.setSpacing(30)
        metadata_format_label = QLabel("Metadata Format:")
        self.metadata_format_value = constructors.FormatEditorWidget("Metadata Format:", "{IDnum}_{Year}{Month}{Day}_{VidNum}_{Modality}")
        format_layout.addWidget(metadata_format_label)
        format_layout.addWidget(self.metadata_format_value)
        format_layout.setAlignment(Qt.AlignLeft)
        self.format_optional = OptionalField(format_widget)

        collapsable_layout.addWidget(self.type_optional)
        collapsable_layout.addWidget(self.format_optional)

        self.raw_analysis_fieldstoload_layer = raw_analysis_fieldstoload_layer()
        collapsable_layout.addWidget(self.raw_analysis_fieldstoload_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.type_optional.is_checked():
            result["type"] = self.type_value.get_text()
        if self.format_optional.is_checked():
            result["metadata_format"] = self.metadata_format_value.get_value()
        fieldstoload_val = self.raw_analysis_fieldstoload_layer.get_value()
        if fieldstoload_val is not None:
            result["fields_to_load"] = fieldstoload_val

        return result if result else None

class raw_analysis_fieldstoload_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Fields to Load", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        framestamps_widget = QWidget()
        framestamps_layout = QHBoxLayout(framestamps_widget)
        framestamps_layout.setSpacing(30)
        framestamps_label = QLabel("Framestamps:")
        self.framestamps_value = constructors.FreetextBox("OriginalFrameNumber")
        framestamps_layout.addWidget(framestamps_label)
        framestamps_layout.addWidget(self.framestamps_value)
        self.framestamps_optional = OptionalField(framestamps_widget)

        stimsequences_widget = QWidget()
        stimsequences_layout = QHBoxLayout(stimsequences_widget)
        stimsequences_layout.setSpacing(30)
        stimsequences_label = QLabel("Stimulus Sequence:")
        self.stimsequences_value = constructors.FreetextBox("StimulusOn")
        stimsequences_layout.addWidget(stimsequences_label)
        stimsequences_layout.addWidget(self.stimsequences_value)
        self.stimsequences_optional = OptionalField(stimsequences_widget)

        collapsable_layout.addWidget(self.framestamps_optional)
        collapsable_layout.addWidget(self.stimsequences_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.framestamps_optional.is_checked():
            result["framestamps"] = self.framestamps_value.get_text()
        if self.stimsequences_optional.is_checked():
            result["stimulus_sequence"] = self.stimsequences_value.get_text()

        return result if result else None

class analysis_control_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Control", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        location_widget = QWidget()
        location_layout = QHBoxLayout(location_widget)
        location_layout.setSpacing(30)
        location_label = QLabel("Location:")
        self.location_value = constructors.FreetextBox("folder")
        location_layout.addWidget(location_label)
        location_layout.addWidget(self.location_value)
        self.location_optional = OptionalField(location_widget)

        folder_name_widget = QWidget()
        folder_name_layout = QHBoxLayout(folder_name_widget)
        folder_name_layout.setSpacing(30)
        folder_name_label = QLabel("Folder Name:")
        self.folder_name_value = constructors.FreetextBox("control")
        folder_name_layout.addWidget(folder_name_label)
        folder_name_layout.addWidget(self.folder_name_value)
        self.folder_name_optional = OptionalField(folder_name_widget)

        collapsable_layout.addWidget(self.location_optional)
        collapsable_layout.addWidget(self.folder_name_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.location_optional.is_checked():
            result["location"] = self.location_value.get_text()
        if self.folder_name_optional.is_checked():
            result["folder_name"] = self.folder_name_value.get_text()

        return result if result else None

class analysis_analysis_params_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Analysis Params", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        modalities_widget = QWidget()
        modalities_layout = QHBoxLayout(modalities_widget)
        modalities_layout.setSpacing(30)
        modalities_label = QLabel("Modalities:")
        self.modalities_list_creator = constructors.ListEditorWidget()
        modalities_layout.addWidget(modalities_label)
        modalities_layout.addWidget(self.modalities_list_creator)
        modalities_layout.setAlignment(Qt.AlignLeft)
        self.modalities_optional = OptionalField(modalities_widget)

        output_folder_widget = QWidget()
        output_folder_layout = QHBoxLayout(output_folder_widget)
        output_folder_layout.setSpacing(30)
        output_folder_label = QLabel("Output Folder:")
        self.output_folder_value = constructors.OpenFolder()
        output_folder_layout.addWidget(output_folder_label)
        output_folder_layout.addWidget(self.output_folder_value)
        output_folder_layout.setAlignment(Qt.AlignLeft)
        self.output_folder_optional = OptionalField(output_folder_widget)

        output_subfolder_widget = QWidget()
        output_subfolder_layout = QHBoxLayout(output_subfolder_widget)
        output_subfolder_layout.setSpacing(30)
        output_subfolder_label = QLabel("Output Subfolder:")
        self.output_subfolder_tf = constructors.TrueFalseSelector(True)
        output_subfolder_layout.addWidget(output_subfolder_label)
        output_subfolder_layout.addWidget(self.output_subfolder_tf)
        output_subfolder_layout.setAlignment(Qt.AlignLeft)
        self.output_subfolder_optional = OptionalField(output_subfolder_widget)

        output_subfolder_method_widget = QWidget()
        output_subfolder_method_layout = QHBoxLayout(output_subfolder_method_widget)
        output_subfolder_method_layout.setSpacing(30)
        output_subfolder_method_label = QLabel("Output Subfolder Method:")
        self.output_subfolder_method_value = constructors.DropdownMenu(default="DateTime", options=["Date", "Sequential"])
        output_subfolder_method_layout.addWidget(output_subfolder_method_label)
        output_subfolder_method_layout.addWidget(self.output_subfolder_method_value)
        output_subfolder_method_layout.setAlignment(Qt.AlignLeft)
        self.output_subfolder_method_optional = OptionalField(output_subfolder_method_widget)

        gausblur_widget = QWidget()
        gausblur_layout = QHBoxLayout(gausblur_widget)
        gausblur_layout.setSpacing(30)
        gausblur_label = QLabel("Gaus Blur:")
        self.gausblur_value = constructors.FreetextBox("0.0")
        gausblur_layout.addWidget(gausblur_label)
        gausblur_layout.addWidget(self.gausblur_value)
        self.gausblur_optional = OptionalField(gausblur_widget)

        collapsable_layout.addWidget(self.modalities_optional)
        collapsable_layout.addWidget(self.output_folder_optional)
        collapsable_layout.addWidget(self.output_subfolder_optional)
        collapsable_layout.addWidget(self.output_subfolder_method_optional)

        self.analysis_analysis_params_normalization_layer = analysis_analysis_params_normalization_layer()
        collapsable_layout.addWidget(self.analysis_analysis_params_normalization_layer)

        self.analysis_analysis_params_segmentation_layer = analysis_analysis_params_segmentation_layer()
        collapsable_layout.addWidget(self.analysis_analysis_params_segmentation_layer)

        self.analysis_analysis_params_exclusion_criteria_layer = analysis_analysis_params_exclusion_criteria_layer()
        collapsable_layout.addWidget(self.analysis_analysis_params_exclusion_criteria_layer)

        self.analysis_analysis_params_standardization_layer = analysis_analysis_params_standardization_layer()
        collapsable_layout.addWidget(self.analysis_analysis_params_standardization_layer)

        self.analysis_analysis_params_summary_layer = analysis_analysis_params_summary_layer()
        collapsable_layout.addWidget(self.analysis_analysis_params_summary_layer)

        collapsable_layout.addWidget(self.gausblur_optional)

        self.analysis_analysis_params_maskroi_layer = analysis_analysis_params_maskroi_layer()
        collapsable_layout.addWidget(self.analysis_analysis_params_maskroi_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        results = {}

        if self.modalities_optional.is_checked():
            results["modalities"] = self.modalities_list_creator.get_list()
        if self.output_folder_optional.is_checked():
            results["output_folder"] = self.output_folder_value.get_text()
        if self.output_subfolder_optional.is_checked():
            results["output_subfolder"] = self.output_subfolder_tf.get_value()
        if self.output_subfolder_method_optional.is_checked():
            results["output_subfolder_method"] = self.output_subfolder_method_value.get_value()
        normalization_val = self.analysis_analysis_params_normalization_layer.get_value()
        if normalization_val is not None:
            results["normalization"] = normalization_val
        segmentation_val = self.analysis_analysis_params_segmentation_layer.get_value()
        if segmentation_val is not None:
            results["segmentation"] = segmentation_val
        exclusion_criteria_val = self.analysis_analysis_params_exclusion_criteria_layer.get_value()
        if exclusion_criteria_val is not None:
            results["exclusion_criteria"] = exclusion_criteria_val
        standardization_val = self.analysis_analysis_params_standardization_layer.get_value()
        if standardization_val is not None:
            results["standardization"] = standardization_val
        summary_val = self.analysis_analysis_params_summary_layer.get_value()
        if summary_val is not None:
            results["summary"] = summary_val
        if self.gausblur_optional.is_checked():
            results["gaus_blur"] = parse_value(self.gausblur_value.get_text())
        mask_roi_val = self.analysis_analysis_params_maskroi_layer.get_value()
        if mask_roi_val is not None:
            results["mask_roi"] = mask_roi_val

        return results if results else None


class analysis_analysis_params_normalization_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Normalization", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        method_widget = QWidget()
        method_layout = QHBoxLayout(method_widget)
        method_layout.setSpacing(10)
        method_label = QLabel("Method:")
        self.method_value = constructors.DropdownMenu("score", ["mean", "median", "none"])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_value)
        method_layout.setAlignment(Qt.AlignLeft)
        self.method_optional = OptionalField(method_widget)

        rescaled_widget = QWidget()
        rescaled_layout = QHBoxLayout(rescaled_widget)
        rescaled_layout.setSpacing(30)
        rescaled_label = QLabel("Rescaled:")
        self.rescaled_tf = constructors.TrueFalseSelector(True)
        rescaled_layout.addWidget(rescaled_label)
        rescaled_layout.addWidget(self.rescaled_tf)
        self.rescaled_optional = OptionalField(rescaled_widget)

        rescaled_mean_widget = QWidget()
        rescaled_mean_layout = QHBoxLayout(rescaled_mean_widget)
        rescaled_layout.setSpacing(30)
        rescaled_mean_label = QLabel("Rescaled Mean:")
        self.rescaled_mean_value = constructors.FreetextBox("70")
        rescaled_mean_layout.addWidget(rescaled_mean_label)
        rescaled_mean_layout.addWidget(self.rescaled_mean_value)
        self.rescaled_mean_optional = OptionalField(rescaled_mean_widget)

        rescaled_std_widget = QWidget()
        rescaled_std_layout = QHBoxLayout(rescaled_std_widget)
        rescaled_layout.setSpacing(30)
        rescaled_std_label = QLabel("Rescaled STDDEV:")
        self.rescaled_std_value = constructors.FreetextBox("35")
        rescaled_std_layout.addWidget(rescaled_std_label)
        rescaled_std_layout.addWidget(self.rescaled_std_value)
        self.rescaled_std_optional = OptionalField(rescaled_std_widget)

        collapsable_layout.addWidget(self.method_optional)
        collapsable_layout.addWidget(self.rescaled_optional)
        collapsable_layout.addWidget(self.rescaled_mean_optional)
        collapsable_layout.addWidget(self.rescaled_std_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.method_optional.is_checked():
            result["method"] = self.method_value.get_value()
        if self.rescaled_optional.is_checked():
            result["rescaled"] = self.rescaled_tf.get_value()
        if self.rescaled_mean_optional.is_checked():
            result["rescale_mean"] = parse_value(self.rescaled_mean_value.get_text())
        if self.rescaled_std_optional.is_checked():
            result["rescale_stddev"] = parse_value(self.rescaled_std_value.get_text())

        return result if result else None

class analysis_analysis_params_segmentation_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Segmentation", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        refine_to_ref_widget = QWidget()
        refine_to_ref_layout = QHBoxLayout(refine_to_ref_widget)
        refine_to_ref_layout.setSpacing(30)
        refine_to_ref_label = QLabel("Refine to Ref:")
        self.refine_to_ref_tf = constructors.TrueFalseSelector(False)
        refine_to_ref_layout.addWidget(refine_to_ref_label)
        refine_to_ref_layout.addWidget(self.refine_to_ref_tf)
        self.refine_to_ref_optional = OptionalField(refine_to_ref_widget)

        refine_to_vid_widget = QWidget()
        refine_to_vid_layout = QHBoxLayout(refine_to_vid_widget)
        refine_to_vid_layout.setSpacing(30)
        refine_to_vid_label = QLabel("Refine to Vid:")
        self.refine_to_vid_tf = constructors.TrueFalseSelector(False)
        refine_to_vid_layout.addWidget(refine_to_vid_label)
        refine_to_vid_layout.addWidget(self.refine_to_vid_tf)
        self.refine_to_vid_optional = OptionalField(refine_to_vid_widget)

        radius_widget = QWidget()
        radius_layout = QHBoxLayout(radius_widget)
        radius_layout.setSpacing(30)
        radius_label = QLabel("Radius:")
        self.radius_value = constructors.FreetextBox("auto")
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(self.radius_value)
        self.radius_optional = OptionalField(radius_widget)

        shape_widget = QWidget()
        shape_layout = QHBoxLayout(shape_widget)
        shape_layout.setSpacing(30)
        shape_label = QLabel("Shape:")
        self.shape_value = constructors.DropdownMenu("disk", "box")
        shape_layout.addWidget(shape_label)
        shape_layout.addWidget(self.shape_value)
        shape_layout.setAlignment(Qt.AlignLeft)
        self.shape_optional = OptionalField(shape_widget)

        summary_widget = QWidget()
        summary_layout = QHBoxLayout(summary_widget)
        summary_layout.setSpacing(30)
        summary_label = QLabel("Summary:")
        self.summary_value = constructors.DropdownMenu("mean", "median")
        summary_layout.addWidget(summary_label)
        summary_layout.addWidget(self.summary_value)
        summary_layout.setAlignment(Qt.AlignLeft)
        self.summary_optional = OptionalField(summary_widget)

        pixelwise_widget = QWidget()
        pixelwise_layout = QHBoxLayout(pixelwise_widget)
        pixelwise_layout.setSpacing(30)
        pixelwise_label = QLabel("Pixelwise:")
        self.pixelwise_tf = constructors.TrueFalseSelector(False)
        pixelwise_layout.addWidget(pixelwise_label)
        pixelwise_layout.addWidget(self.pixelwise_tf)
        self.pixelwise_optional = OptionalField(pixelwise_widget)

        collapsable_layout.addWidget(self.refine_to_ref_optional)
        collapsable_layout.addWidget(self.refine_to_vid_optional)
        collapsable_layout.addWidget(self.radius_optional)
        collapsable_layout.addWidget(self.shape_optional)
        collapsable_layout.addWidget(self.summary_optional)
        collapsable_layout.addWidget(self.pixelwise_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.refine_to_ref_optional.is_checked():
            result["refine_to_ref"] = self.refine_to_ref_tf.get_value()
        if self.refine_to_vid_optional.is_checked():
            result["refine_to_vid"] = self.refine_to_vid_tf.get_value()
        if self.radius_optional.is_checked():
            result["radius"] = parse_value(self.radius_value.get_text())
        if self.shape_optional.is_checked():
            result["shape"] = self.shape_value.get_value()
        if self.summary_optional.is_checked():
            result["summary"] = self.summary_value.get_value()
        if self.pixelwise_optional.is_checked():
            result["pixelwise"] = self.pixelwise_tf.get_value()

        return result if result else None

class analysis_analysis_params_exclusion_criteria_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Exclusion Criteria", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        type_widget = QWidget()
        type_layout = QHBoxLayout(type_widget)
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_value = constructors.DropdownMenu("stim-relative", "absolute")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_value)
        type_layout.setAlignment(Qt.AlignLeft)
        self.type_optional = OptionalField(type_widget)

        units_widget = QWidget()
        units_layout = QHBoxLayout(units_widget)
        units_layout.setSpacing(30)
        units_label = QLabel("Units:")
        self.units_value = constructors.DropdownMenu("time", "frames")
        units_layout.addWidget(units_label)
        units_layout.addWidget(self.units_value)
        units_layout.setAlignment(Qt.AlignLeft)
        self.units_optional = OptionalField(units_widget)

        start_widget = QWidget()
        start_layout = QHBoxLayout(start_widget)
        start_layout.setSpacing(30)
        start_label = QLabel("Start:")
        self.start_value = constructors.FreetextBox("-1")
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_value)
        self.start_optional = OptionalField(start_widget)

        stop_widget = QWidget()
        stop_layout = QHBoxLayout(stop_widget)
        stop_layout.setSpacing(30)
        stop_label = QLabel("Stop:")
        self.stop_value = constructors.FreetextBox("0")
        stop_layout.addWidget(stop_label)
        stop_layout.addWidget(self.stop_value)
        self.stop_optional = OptionalField(stop_widget)

        fraction_widget = QWidget()
        fraction_layout = QHBoxLayout(fraction_widget)
        fraction_layout.setSpacing(30)
        fraction_label = QLabel("Fraction:")
        self.fraction_value = constructors.FreetextBox("0.3")
        fraction_layout.addWidget(fraction_label)
        fraction_layout.addWidget(self.fraction_value)
        self.fraction_optional = OptionalField(fraction_widget)

        collapsable_layout.addWidget(self.type_optional)
        collapsable_layout.addWidget(self.units_optional)
        collapsable_layout.addWidget(self.start_optional)
        collapsable_layout.addWidget(self.stop_optional)
        collapsable_layout.addWidget(self.fraction_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.type_optional.is_checked():
            result["type"] = self.type_value.get_value()
        if self.units_optional.is_checked():
            result["units"] = self.units_value.get_value()
        if self.start_optional.is_checked():
            result["start"] = parse_value(self.start_value.get_text())
        if self.stop_optional.is_checked():
            result["stop"] = parse_value(self.stop_value.get_text())
        if self.fraction_optional.is_checked():
            result["fraction"] = parse_value(self.fraction_value.get_text())

        return result if result else None

class analysis_analysis_params_standardization_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Standardization", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        method_widget = QWidget()
        method_layout = QHBoxLayout(method_widget)
        method_layout.setSpacing(30)
        method_label = QLabel("Method:")
        self.method_value = constructors.DropdownMenu("mean_sub", ["stddev", "linear_stddev", "linear_vast", "relative_change", "none"])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_value)
        method_layout.setAlignment(Qt.AlignLeft)
        self.method_optional = OptionalField(method_widget)

        type_widget = QWidget()
        type_layout = QHBoxLayout(type_widget)
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_value = constructors.DropdownMenu("stim-relative", "absolute")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_value)
        type_layout.setAlignment(Qt.AlignLeft)
        self.type_optional = OptionalField(type_widget)

        units_widget = QWidget()
        units_layout = QHBoxLayout(units_widget)
        units_layout.setSpacing(30)
        units_label = QLabel("Units:")
        self.units_value = constructors.DropdownMenu("time", "frames")
        units_layout.addWidget(units_label)
        units_layout.addWidget(self.units_value)
        units_layout.setAlignment(Qt.AlignLeft)
        self.units_optional = OptionalField(units_widget)

        start_widget = QWidget()
        start_layout = QHBoxLayout(start_widget)
        start_layout.setSpacing(30)
        start_label = QLabel("Start:")
        self.start_value = constructors.FreetextBox("-1")
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_value)
        self.start_optional = OptionalField(start_widget)

        collapsable_layout.addWidget(self.method_optional)
        collapsable_layout.addWidget(self.type_optional)
        collapsable_layout.addWidget(self.units_optional)
        collapsable_layout.addWidget(self.start_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}
        if self.method_optional.is_checked():
            result["method"] = self.method_value.get_value()
        if self.type_optional.is_checked():
            result["type"] = self.type_value.get_value()
        if self.units_optional.is_checked():
            result["units"] = self.units_value.get_value()
        if self.start_optional.is_checked():
            result["start"] = parse_value(self.start_value.get_text())

        return result if result else None

class analysis_analysis_params_summary_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Summary", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        method_widget = QWidget()
        method_layout = QHBoxLayout(method_widget)
        method_layout.setSpacing(30)
        method_label = QLabel("Method:")
        self.method_value = constructors.DropdownMenu("rms", ["std", "var", "avg"])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_value)
        method_layout.setAlignment(Qt.AlignLeft)
        self.method_optional = OptionalField(method_widget)

        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setSpacing(30)
        control_label = QLabel("Control:")
        self.control_value = constructors.DropdownMenu("none", ["subtraction", "division"])
        control_layout.addWidget(control_label)
        control_layout.addWidget(self.control_value)
        control_layout.setAlignment(Qt.AlignLeft)
        self.control_optional = OptionalField(control_widget)

        indiv_cutoff_widget = QWidget()
        indiv_cutoff_layout = QHBoxLayout(indiv_cutoff_widget)
        indiv_cutoff_layout.setSpacing(30)
        indiv_cutoff_label = QLabel("Individual Cutoff:")
        self.indiv_cutoff_value = constructors.FreetextBox("5")
        indiv_cutoff_layout.addWidget(indiv_cutoff_label)
        indiv_cutoff_layout.addWidget(self.indiv_cutoff_value)
        self.indiv_cutoff_optional = OptionalField(indiv_cutoff_widget)

        collapsable_layout.addWidget(self.method_optional)
        collapsable_layout.addWidget(self.control_optional)

        self.analysis_analysis_params_summary_metrics_layer = analysis_analysis_params_summary_metrics_layer()
        collapsable_layout.addWidget(self.analysis_analysis_params_summary_metrics_layer)

        collapsable_layout.addWidget(self.indiv_cutoff_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.method_optional.is_checked():
            result["method"] = self.method_value.get_value()
        if self.control_optional.is_checked():
            result["control"] = self.control_value.get_value()
        metrics_val = self.analysis_analysis_params_summary_metrics_layer.get_value()
        if metrics_val is not None:
            result["metrics"] = metrics_val
        if self.indiv_cutoff_optional.is_checked():
            result["indiv_cutoff"] = parse_value(self.indiv_cutoff_value.get_text())

        return result if result else None

class analysis_analysis_params_summary_metrics_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Metrics", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        type_widget = QWidget()
        type_layout = QHBoxLayout(type_widget)
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_list_creator = constructors.ListEditorWidget()
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_list_creator)
        type_layout.setAlignment(Qt.AlignLeft)
        self.type_optional = OptionalField(type_widget)

        measure_widget = QWidget()
        measured_layout = QHBoxLayout(measure_widget)
        measured_layout.setSpacing(30)
        measured_label = QLabel("Measured:")
        self.measured_value = constructors.FreetextBox("stim-relative")
        measured_layout.addWidget(measured_label)
        measured_layout.addWidget(self.measured_value)
        self.measured_optional = OptionalField(measure_widget)

        units_widget = QWidget()
        units_layout = QHBoxLayout(units_widget)
        units_layout.setSpacing(30)
        units_label = QLabel("Units:")
        self.units_value = constructors.FreetextBox("time")
        units_layout.addWidget(units_label)
        units_layout.addWidget(self.units_value)
        self.units_optional = OptionalField(units_widget)

        prestim_widget = QWidget()
        prestim_layout = QHBoxLayout(prestim_widget)
        prestim_layout.setSpacing(30)
        prestim_label = QLabel("Prestim:")
        self.prestim_value = constructors.FreetextBox("-1, 0")
        prestim_layout.addWidget(prestim_label)
        prestim_layout.addWidget(self.prestim_value)
        self.prestim_optional = OptionalField(prestim_widget)

        poststim_widget = QWidget()
        poststim_layout = QHBoxLayout(poststim_widget)
        poststim_layout.setSpacing(30)
        poststim_label = QLabel("Poststim:")
        self.poststim_value = constructors.FreetextBox("0, 1")
        poststim_layout.addWidget(poststim_label)
        poststim_layout.addWidget(self.poststim_value)
        self.poststim_optional = OptionalField(poststim_widget)

        collapsable_layout.addWidget(self.type_optional)
        collapsable_layout.addWidget(self.measured_optional)
        collapsable_layout.addWidget(self.units_optional)
        collapsable_layout.addWidget(self.prestim_optional)
        collapsable_layout.addWidget(self.poststim_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        def parse_range(text):
            try:
                return [float(x.strip()) for x in text.split(",")]
            except:
                return []

        result = {}

        if self.type_optional.is_checked():
            result["type"] = self.type_list_creator.get_list()
        if self.measured_optional.is_checked():
            result["measured"] = self.measured_value.get_text()
        if self.units_optional.is_checked():
            result["units"] = self.units_value.get_text()
        if self.prestim_optional.is_checked():
            result["prestim"] = parse_range(self.prestim_value.get_text())
        if self.poststim_optional.is_checked():
            result["poststim"] = parse_range(self.poststim_value.get_text())

        return result if result else None

class analysis_analysis_params_maskroi_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("MaskROI", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        r_widget = QWidget()
        r_layout = QHBoxLayout(r_widget)
        r_layout.setSpacing(30)
        r_label = QLabel("R:")
        self.r_value = constructors.FreetextBox("0")
        r_layout.addWidget(r_label)
        r_layout.addWidget(self.r_value)
        self.r_optional = OptionalField(r_widget)

        c_widget = QWidget()
        c_layout = QHBoxLayout(c_widget)
        c_layout.setSpacing(30)
        c_label = QLabel("C:")
        self.c_value = constructors.FreetextBox("0")
        c_layout.addWidget(c_label)
        c_layout.addWidget(self.c_value)
        self.c_optional = OptionalField(c_widget)

        width_widget = QWidget()
        width_layout = QHBoxLayout(width_widget)
        width_layout.setSpacing(30)
        width_label = QLabel("Width:")
        self.width_value = constructors.FreetextBox("-1")
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.width_value)
        self.width_optional = OptionalField(width_widget)

        height_widget = QWidget()
        height_layout = QHBoxLayout(height_widget)
        height_layout.setSpacing(30)
        height_label = QLabel("Height:")
        self.height_value = constructors.FreetextBox("-1")
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_value)
        self.height_optional = OptionalField(height_widget)

        collapsable_layout.addWidget(self.r_optional)
        collapsable_layout.addWidget(self.c_optional)
        collapsable_layout.addWidget(self.width_optional)
        collapsable_layout.addWidget(self.height_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.r_optional.is_checked():
            result["r"] = parse_value(self.r_value.get_text())
        if self.c_optional.is_checked():
            result["c"] = parse_value(self.c_value.get_text())
        if self.width_optional.is_checked():
            result["width"] = parse_value(self.width_value.get_text())
        if self.height_optional.is_checked():
            result["height"] = parse_value(self.height_value.get_text())

        return result if result else None

class analysis_display_params_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Display Parameters", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        saveas_widget = QWidget()
        saveas_layout = QHBoxLayout(saveas_widget)
        saveas_layout.setSpacing(30)
        saveas_label = QLabel("Save as:")
        self.saveas_list_creator = constructors.SaveasExtensionsEditorWidget("Save as:", "[]")
        saveas_layout.addWidget(saveas_label)
        saveas_layout.addWidget(self.saveas_list_creator)
        saveas_layout.setAlignment(Qt.AlignLeft)
        self.saveas_optional = OptionalField(saveas_widget)

        pause_per_folder_widget = QWidget()
        pause_per_folder_layout = QHBoxLayout(pause_per_folder_widget)
        pause_per_folder_layout.setSpacing(30)
        pause_per_folder_label = QLabel("Pause per Folder:")
        self.pause_per_folder_tf = constructors.TrueFalseSelector(True)
        pause_per_folder_layout.addWidget(pause_per_folder_label)
        pause_per_folder_layout.addWidget(self.pause_per_folder_tf)
        self.pause_per_folder_optional = OptionalField(pause_per_folder_widget)

        self.analysis_display_params_debug_layer = analysis_display_params_debug_layer()
        collapsable_layout.addWidget(self.analysis_display_params_debug_layer)

        self.analysis_display_params_pop_summary_overlap_layer = analysis_display_params_pop_summary_overlap_layer()
        collapsable_layout.addWidget(self.analysis_display_params_pop_summary_overlap_layer)

        self.analysis_display_params_pop_summary_seq_layer = analysis_display_params_pop_summary_seq_layer()
        collapsable_layout.addWidget(self.analysis_display_params_pop_summary_seq_layer)

        self.analysis_display_params_indiv_summary_overlap_layer = analysis_display_params_indiv_summary_overlap_layer()
        collapsable_layout.addWidget(self.analysis_display_params_indiv_summary_overlap_layer)

        self.analysis_display_params_indiv_summary_layer = analysis_display_params_indiv_summary_layer()
        collapsable_layout.addWidget(self.analysis_display_params_indiv_summary_layer)

        collapsable_layout.addWidget(self.saveas_optional)
        collapsable_layout.addWidget(self.pause_per_folder_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        debug_val = self.analysis_display_params_debug_layer.get_value()
        if debug_val is not None:
            result["debug"] = debug_val
        pop_summary_overlap_val = self.analysis_display_params_pop_summary_overlap_layer.get_value()
        if pop_summary_overlap_val is not None:
            result["pop_summary_overlap"] = pop_summary_overlap_val
        pop_summary_seq_val = self.analysis_display_params_pop_summary_seq_layer.get_value()
        if pop_summary_seq_val is not None:
            result["pop_summary_seq"] = pop_summary_seq_val
        indiv_summary_overlap_val = self.analysis_display_params_indiv_summary_overlap_layer.get_value()
        if indiv_summary_overlap_val is not None:
            result["indiv_summary_overlap"] = indiv_summary_overlap_val
        indiv_summary_val = self.analysis_display_params_indiv_summary_layer.get_value()
        if indiv_summary_val is not None:
            result["indiv_summary"] = indiv_summary_val
        if self.saveas_optional.is_checked():
            result["saveas"] = self.saveas_list_creator.get_value()
        if self.pause_per_folder_optional.is_checked():
            result["pause_per_folder"] =self.pause_per_folder_tf.get_value()

        return result if result else None

class analysis_display_params_pop_summary_overlap_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Pop Summary Overlap", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        stimulus_widget = QWidget()
        stimulus_layout = QHBoxLayout(stimulus_widget)
        stimulus_layout.setSpacing(30)
        stimulus_label = QLabel("Stimulus:")
        self.stimulus_tf = constructors.TrueFalseSelector(True)
        stimulus_layout.addWidget(stimulus_label)
        stimulus_layout.addWidget(self.stimulus_tf)
        self.stimulus_optional = OptionalField(stimulus_widget)

        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setSpacing(30)
        control_label = QLabel("Control:")
        self.control_tf = constructors.TrueFalseSelector(True)
        control_layout.addWidget(control_label)
        control_layout.addWidget(self.control_tf)
        self.control_optional = OptionalField(control_widget)

        relative_widget = QWidget()
        relative_layout = QHBoxLayout(relative_widget)
        relative_layout.setSpacing(30)
        relative_label = QLabel("Relative:")
        self.relative_tf = constructors.TrueFalseSelector(True)
        relative_layout.addWidget(relative_label)
        relative_layout.addWidget(self.relative_tf)
        self.relative_optional = OptionalField(relative_widget)

        pooled_widget = QWidget()
        pooled_layout = QHBoxLayout(pooled_widget)
        pooled_layout.setSpacing(30)
        pooled_label = QLabel("Pooled:")
        self.pooled_tf = constructors.TrueFalseSelector(True)
        pooled_layout.addWidget(pooled_label)
        pooled_layout.addWidget(self.pooled_tf)
        self.pooled_optional = OptionalField(pooled_widget)

        collapsable_layout.addWidget(self.stimulus_optional)
        collapsable_layout.addWidget(self.control_optional)
        collapsable_layout.addWidget(self.relative_optional)
        collapsable_layout.addWidget(self.pooled_optional)
        self.axis_layer = axes_base_layer(parent=None, xmin_def=0, xmax_def=4, ymin_def=-5, ymax_def=60, cmap_def="plasma", legend_def=True)
        collapsable_layout.addWidget(self.axis_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.stimulus_optional.is_checked():
            result["stimulus"] = self.stimulus_tf.get_value()
        if self.control_optional.is_checked():
            result["control"] = self.control_tf.get_value()
        if self.relative_optional.is_checked():
            result["relative"] = self.relative_tf.get_value()
        if self.pooled_optional.is_checked():
            result["pooled"] = self.pooled_tf.get_value()
        axis_val = self.axis_layer.get_value()
        if axis_val is not None:
            result["axes"] = axis_val

        return result if result else None

class analysis_display_params_pop_summary_seq_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Pop Summary Sequence", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        stimulus_widget = QWidget()
        stimulus_layout = QHBoxLayout(stimulus_widget)
        stimulus_layout.setSpacing(30)
        stimulus_label = QLabel("Stimulus:")
        self.stimulus_tf = constructors.TrueFalseSelector(True)
        stimulus_layout.addWidget(stimulus_label)
        stimulus_layout.addWidget(self.stimulus_tf)
        self.stimulus_optional = OptionalField(stimulus_widget)

        relative_widget = QWidget()
        relative_layout = QHBoxLayout(relative_widget)
        relative_layout.setSpacing(30)
        relative_label = QLabel("Relative:")
        self.relative_tf = constructors.TrueFalseSelector(True)
        relative_layout.addWidget(relative_label)
        relative_layout.addWidget(self.relative_tf)
        self.relative_optional = OptionalField(relative_widget)

        num_in_seq_widget = QWidget()
        num_in_seq_layout = QHBoxLayout(num_in_seq_widget)
        num_in_seq_layout.setSpacing(30)
        num_in_seq_label = QLabel("Number in Sequence:")
        self.num_in_seq_value = constructors.FreetextBox("8")
        num_in_seq_layout.addWidget(num_in_seq_label)
        num_in_seq_layout.addWidget(self.num_in_seq_value)
        self.num_in_seq_optional = OptionalField(num_in_seq_widget)

        collapsable_layout.addWidget(self.stimulus_optional)
        collapsable_layout.addWidget(self.relative_optional)
        collapsable_layout.addWidget(self.num_in_seq_optional)
        self.axis_layer = axes_base_layer(parent=None, xmin_def=0, xmax_def=4, ymin_def=0, ymax_def=60, cmap_def="plasma", legend_def=True)
        collapsable_layout.addWidget(self.axis_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.stimulus_optional.is_checked():
            result["stimulus"] = self.stimulus_tf.get_value()
        if self.relative_optional.is_checked():
            result["relative"] = self.relative_tf.get_value()
        if self.num_in_seq_optional.is_checked():
            result["num_in_seq"] = parse_value(self.num_in_seq_value.get_text())
        axes_val = self.axis_layer.get_value()
        if axes_val is not None:
            result["axes"] = axes_val

        return result if result else None

class analysis_display_params_indiv_summary_overlap_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Individual Summary Overlap", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        stimulus_widget = QWidget()
        stimulus_layout = QHBoxLayout(stimulus_widget)
        stimulus_layout.setSpacing(30)
        stimulus_label = QLabel("Stimulus:")
        self.stimulus_tf = constructors.TrueFalseSelector(True)
        stimulus_layout.addWidget(stimulus_label)
        stimulus_layout.addWidget(self.stimulus_tf)
        self.stimulus_optional = OptionalField(stimulus_widget)

        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setSpacing(30)
        control_label = QLabel("Control:")
        self.control_tf = constructors.TrueFalseSelector(True)
        control_layout.addWidget(control_label)
        control_layout.addWidget(self.control_tf)
        self.control_optional = OptionalField(control_widget)

        relative_widget = QWidget()
        relative_layout = QHBoxLayout(relative_widget)
        relative_layout.setSpacing(30)
        relative_label = QLabel("Relative:")
        self.relative_tf = constructors.TrueFalseSelector(True)
        relative_layout.addWidget(relative_label)
        relative_layout.addWidget(self.relative_tf)
        self.relative_optional = OptionalField(relative_widget)

        collapsable_layout.addWidget(self.stimulus_optional)
        collapsable_layout.addWidget(self.control_optional)
        collapsable_layout.addWidget(self.relative_optional)
        self.axes_layer = analysis_display_params_indiv_summary_overlap_layer_axis_layer()
        collapsable_layout.addWidget(self.axes_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.stimulus_optional.is_checked():
            result["stimulus"] = self.stimulus_tf.get_value()
        if self.control_optional.is_checked():
            result["control"] = self.control_tf.get_value()
        if self.relative_optional.is_checked():
            result["relative"] = self.relative_tf.get_value()
        axis_val = self.axes_layer.get_value()
        if axis_val is not None:
            result["axes"] = axis_val

        return result if result else None

class analysis_display_params_indiv_summary_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Individual Summary", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        histogram_widget = QWidget()
        histogram_layout = QHBoxLayout(histogram_widget)
        histogram_layout.setSpacing(30)
        histogram_label = QLabel("Histogram:")
        self.histogram_tf = constructors.TrueFalseSelector(True)
        histogram_layout.addWidget(histogram_label)
        histogram_layout.addWidget(self.histogram_tf)
        self.histogram_optional = OptionalField(histogram_widget)

        cumulative_histogram_widget = QWidget()
        cumulative_histogram_layout = QHBoxLayout(cumulative_histogram_widget)
        cumulative_histogram_layout.setSpacing(30)
        cumulative_histogram_label = QLabel("Cumulative Histogram:")
        self.cumulative_histogram_tf = constructors.TrueFalseSelector(True)
        cumulative_histogram_layout.addWidget(cumulative_histogram_label)
        cumulative_histogram_layout.addWidget(self.cumulative_histogram_tf)
        self.cumulative_histogram_optional = OptionalField(cumulative_histogram_widget)

        map_overlay_widget = QWidget()
        map_overlay_layout = QHBoxLayout(map_overlay_widget)
        map_overlay_layout.setSpacing(30)
        map_overlay_label = QLabel("Map Overlay:")
        self.map_overlay_tf = constructors.TrueFalseSelector(True)
        map_overlay_layout.addWidget(map_overlay_label)
        map_overlay_layout.addWidget(self.map_overlay_tf)
        self.map_overlay_optional = OptionalField(map_overlay_widget)

        collapsable_layout.addWidget(self.histogram_optional)
        collapsable_layout.addWidget(self.cumulative_histogram_optional)
        collapsable_layout.addWidget(self.map_overlay_optional)
        self.axes_layer = analysis_display_params_indiv_summary_axes_layer()
        collapsable_layout.addWidget(self.axes_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.histogram_optional.is_checked():
            result["histogram"] = self.histogram_tf.get_value()
        if self.cumulative_histogram_optional.is_checked():
            result["cumulative_histogram"] = self.cumulative_histogram_tf.get_value()
        if self.map_overlay_optional.is_checked():
            result["map_overlay"] = self.map_overlay_tf.get_value()
        axes_val = self.axes_layer.get_value()
        if axes_val is not None:
            result["axes"] = axes_val

        return result if result else None

class axes_base_layer(QWidget):
    def __init__(self, parent=None, xmin_def = None, xmax_def = None, ymin_def = None, ymax_def = None, cmap_def = None, legend_def = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Axes", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        xmin_widget = QWidget()
        xmin_layout = QHBoxLayout(xmin_widget)
        xmin_layout.setSpacing(30)
        xmin_label = QLabel("XMin:")
        self.xmin_value = constructors.FreetextBox(f"{xmin_def}")
        xmin_layout.addWidget(xmin_label)
        xmin_layout.addWidget(self.xmin_value)
        self.xmin_optional = OptionalField(xmin_widget)

        xmax_widget = QWidget()
        xmax_layout = QHBoxLayout(xmax_widget)
        xmax_layout.setSpacing(30)
        xmax_label = QLabel("XMax:")
        self.xmax_value = constructors.FreetextBox(f"{xmax_def}")
        xmax_layout.addWidget(xmax_label)
        xmax_layout.addWidget(self.xmax_value)
        self.xmax_optional = OptionalField(xmax_widget)

        ymin_widget = QWidget()
        ymin_layout = QHBoxLayout(ymin_widget)
        ymin_layout.setSpacing(30)
        ymin_label = QLabel("YMin:")
        self.ymin_value = constructors.FreetextBox(f"{ymin_def}")
        ymin_layout.addWidget(ymin_label)
        ymin_layout.addWidget(self.ymin_value)
        self.ymin_optional = OptionalField(ymin_widget)

        ymax_widget = QWidget()
        ymax_layout = QHBoxLayout(ymax_widget)
        ymax_layout.setSpacing(30)
        ymax_label = QLabel("YMax:")
        self.ymax_value = constructors.FreetextBox(f"{ymax_def}")
        ymax_layout.addWidget(ymax_label)
        ymax_layout.addWidget(self.ymax_value)
        self.ymax_optional = OptionalField(ymax_widget)

        colormap_widget = QWidget()
        colormap_layout = QHBoxLayout(colormap_widget)
        colormap_layout.setSpacing(30)
        colormap_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        colormap_label = QLabel("Colormap:")
        colormap_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.colormap_selector = constructors.ColorMapSelector(cmap_def=cmap_def)
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormap_selector)
        colormap_layout.addStretch()
        self.colormap_optional = OptionalField(colormap_widget)

        legend_widget = QWidget()
        legend_layout = QHBoxLayout(legend_widget)
        legend_layout.setSpacing(30)
        legend_label = QLabel("Legend:")
        self.legend_tf = constructors.TrueFalseSelector(legend_def)
        legend_layout.addWidget(legend_label)
        legend_layout.addWidget(self.legend_tf)
        self.legend_optional = OptionalField(legend_widget)

        collapsable_layout.addWidget(self.xmin_optional)
        collapsable_layout.addWidget(self.xmax_optional)
        collapsable_layout.addWidget(self.ymin_optional)
        collapsable_layout.addWidget(self.ymax_optional)
        collapsable_layout.addWidget(self.colormap_optional)
        collapsable_layout.addWidget(self.legend_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.xmin_optional.is_checked():
            result["xmin"] = parse_value(self.xmin_value.get_text())
        if self.xmax_optional.is_checked():
            result["xmax"] = parse_value(self.xmax_value.get_text())
        if self.ymin_optional.is_checked():
            result["ymin"] = parse_value(self.ymin_value.get_text())
        if self.ymax_optional.is_checked():
            result["ymax"] = parse_value(self.ymax_value.get_text())
        if self.colormap_optional.is_checked():
            result["cmap"] = self.colormap_selector.get_value()
        if self.legend_optional.is_checked():
            result["legend"] = self.legend_tf.get_value()

        return result if result else None

class analysis_display_params_indiv_summary_overlap_layer_axis_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Axes", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        xmin_widget = QWidget()
        xmin_layout = QHBoxLayout(xmin_widget)
        xmin_layout.setSpacing(30)
        xmin_label = QLabel("XMin:")
        self.xmin_value = constructors.FreetextBox("null")
        xmin_layout.addWidget(xmin_label)
        xmin_layout.addWidget(self.xmin_value)
        self.xmin_optional = OptionalField(xmin_widget)

        xmax_widget = QWidget()
        xmax_layout = QHBoxLayout(xmax_widget)
        xmax_layout.setSpacing(30)
        xmax_label = QLabel("XMax:")
        self.xmax_value = constructors.FreetextBox("null")
        xmax_layout.addWidget(xmax_label)
        xmax_layout.addWidget(self.xmax_value)
        self.xmax_optional = OptionalField(xmax_widget)

        ymin_widget = QWidget()
        ymin_layout = QHBoxLayout(ymin_widget)
        ymin_layout.setSpacing(30)
        ymin_label = QLabel("YMin:")
        self.ymin_value = constructors.FreetextBox("null")
        ymin_layout.addWidget(ymin_label)
        ymin_layout.addWidget(self.ymin_value)
        self.ymin_optional = OptionalField(ymin_widget)

        ymax_widget = QWidget()
        ymax_layout = QHBoxLayout(ymax_widget)
        ymax_layout.setSpacing(30)
        ymax_label = QLabel("YMax:")
        self.ymax_value = constructors.FreetextBox("null")
        ymax_layout.addWidget(ymax_label)
        ymax_layout.addWidget(self.ymax_value)
        self.ymax_optional = OptionalField(ymax_widget)

        colormap_widget = QWidget()
        colormap_layout = QHBoxLayout(colormap_widget)
        colormap_layout.setSpacing(30)
        colormap_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        colormap_label = QLabel("Colormap:")
        colormap_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.colormap_selector = constructors.ColorMapSelector(cmap_def="plasma")
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormap_selector)
        colormap_layout.addStretch()
        self.colormap_optional = OptionalField(colormap_widget)

        collapsable_layout.addWidget(self.xmin_optional)
        collapsable_layout.addWidget(self.xmax_optional)
        collapsable_layout.addWidget(self.ymin_optional)
        collapsable_layout.addWidget(self.ymax_optional)
        collapsable_layout.addWidget(self.colormap_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.xmin_optional.is_checked():
            result["xmin"] = parse_value(self.xmin_value.get_text())
        if self.xmax_optional.is_checked():
            result["xmax"] = parse_value(self.xmax_value.get_text())
        if self.ymin_optional.is_checked():
            result["ymin"] = parse_value(self.ymin_value.get_text())
        if self.ymax_optional.is_checked():
            result["ymax"] = parse_value(self.ymax_value.get_text())
        if self.colormap_optional.is_checked():
            result["cmap"] = self.colormap_selector.get_value()

        return result if result else None

class analysis_display_params_indiv_summary_axes_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Axes", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        xmin_widget = QWidget()
        xmin_layout = QHBoxLayout(xmin_widget)
        xmin_layout.setSpacing(30)
        xmin_label = QLabel("XMin:")
        self.xmin_value = constructors.FreetextBox("null")
        xmin_layout.addWidget(xmin_label)
        xmin_layout.addWidget(self.xmin_value)
        self.xmin_optional = OptionalField(xmin_widget)

        xstep_widget = QWidget()
        xstep_layout = QHBoxLayout(xstep_widget)
        xstep_layout.setSpacing(30)
        xstep_label = QLabel("XStep:")
        self.xstep_value = constructors.FreetextBox("null")
        xstep_layout.addWidget(xstep_label)
        xstep_layout.addWidget(self.xstep_value)
        self.xstep_optional = OptionalField(xstep_widget)

        xmax_widget = QWidget()
        xmax_layout = QHBoxLayout(xmax_widget)
        xmax_layout.setSpacing(30)
        xmax_label = QLabel("XMax:")
        self.xmax_value = constructors.FreetextBox("null")
        xmax_layout.addWidget(xmax_label)
        xmax_layout.addWidget(self.xmax_value)
        self.xmax_optional = OptionalField(xmax_widget)

        colormap_widget = QWidget()
        colormap_layout = QHBoxLayout(colormap_widget)
        colormap_layout.setSpacing(30)
        colormap_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        colormap_label = QLabel("Colormap:")
        colormap_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.colormap_selector = constructors.ColorMapSelector(cmap_def="plasma")
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormap_selector)
        colormap_layout.addStretch()
        self.colormap_optional = OptionalField(colormap_widget)

        legend_widget = QWidget()
        legend_layout = QHBoxLayout(legend_widget)
        legend_layout.setSpacing(30)
        legend_label = QLabel("Legend:")
        self.legend_tf = constructors.TrueFalseSelector(True)
        legend_layout.addWidget(legend_label)
        legend_layout.addWidget(self.legend_tf)
        self.legend_optional = OptionalField(legend_widget)

        collapsable_layout.addWidget(self.xmin_optional)
        collapsable_layout.addWidget(self.xstep_optional)
        collapsable_layout.addWidget(self.xmax_optional)
        collapsable_layout.addWidget(self.colormap_optional)
        collapsable_layout.addWidget(self.legend_optional)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def parse_value(val):
            return float(val) if is_number(val) else val

        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.xmin_optional.is_checked():
            result["xmin"] = None if  self.xmin_value == "null" else parse_value(self.xmin_value.get_text())
        if self.xstep_optional.is_checked():
            result["xstep"] = None if self.xstep_value == "null" else parse_value(self.xstep_value.get_text())
        if self.xmax_optional.is_checked():
            result["xmax"] = None if  self.xmax_value == "null" else parse_value(self.xmax_value.get_text())
        if self.colormap_optional.is_checked():
            result["cmap"] = self.colormap_selector.get_value()
        if self.legend_optional.is_checked():
            result["legend"] = self.legend_tf.get_value()

        return result if result else None

class analysis_display_params_debug_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Debug", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        output_norm_video_widget = QWidget()
        output_norm_video_layout = QHBoxLayout(output_norm_video_widget)
        output_norm_video_layout.setSpacing(30)
        output_norm_video_label = QLabel("output_norm_video:")
        self.output_norm_video_tf = constructors.TrueFalseSelector(False)
        output_norm_video_layout.addWidget(output_norm_video_label)
        output_norm_video_layout.addWidget(self.output_norm_video_tf)
        self.output_norm_video_optional = OptionalField(output_norm_video_widget)

        plot_refine_to_ref_widget = QWidget()
        plot_refine_to_ref_layout = QHBoxLayout(plot_refine_to_ref_widget)
        plot_refine_to_ref_layout.setSpacing(30)
        plot_refine_to_ref_label = QLabel("plot_refine_to_ref:")
        self.plot_refine_to_ref_tf = constructors.TrueFalseSelector(False)
        plot_refine_to_ref_layout.addWidget(plot_refine_to_ref_label)
        plot_refine_to_ref_layout.addWidget(self.plot_refine_to_ref_tf)
        self.plot_refine_to_ref_optional = OptionalField(plot_refine_to_ref_widget)

        plot_refine_to_vid_widget = QWidget()
        plot_refine_to_vid_layout = QHBoxLayout(plot_refine_to_vid_widget)
        plot_refine_to_vid_layout.setSpacing(30)
        plot_refine_to_vid_label = QLabel("plot_refine_to_vid:")
        self.plot_refine_to_vid_tf = constructors.TrueFalseSelector(False)
        plot_refine_to_vid_layout.addWidget(plot_refine_to_vid_label)
        plot_refine_to_vid_layout.addWidget(self.plot_refine_to_vid_tf)
        self.plot_refine_to_vid_optional = OptionalField(plot_refine_to_vid_widget)

        plot_pop_extracted_orgs_widget = QWidget()
        plot_pop_extracted_orgs_layout = QHBoxLayout(plot_pop_extracted_orgs_widget)
        plot_pop_extracted_orgs_layout.setSpacing(30)
        plot_pop_extracted_orgs_label = QLabel("plot_pop_extracted_orgs:")
        self.plot_pop_extracted_orgs_tf = constructors.TrueFalseSelector(False)
        plot_pop_extracted_orgs_layout.addWidget(plot_pop_extracted_orgs_label)
        plot_pop_extracted_orgs_layout.addWidget(self.plot_pop_extracted_orgs_tf)
        self.plot_pop_extracted_orgs_optional = OptionalField(plot_pop_extracted_orgs_widget)

        plot_pop_stdize_orgs_widget = QWidget()
        plot_pop_stdize_orgs_layout = QHBoxLayout(plot_pop_stdize_orgs_widget)
        plot_pop_stdize_orgs_layout.setSpacing(30)
        plot_pop_stdize_orgs_label = QLabel("plot_pop_stdize_orgs:")
        self.plot_pop_stdize_orgs_tf = constructors.TrueFalseSelector(False)
        plot_pop_stdize_orgs_layout.addWidget(plot_pop_stdize_orgs_label)
        plot_pop_stdize_orgs_layout.addWidget(self.plot_pop_stdize_orgs_tf)
        self.plot_pop_stdize_orgs_optional = OptionalField(plot_pop_stdize_orgs_widget)

        plot_indiv_stdize_orgs_widget = QWidget()
        plot_indiv_stdize_orgs_layout = QHBoxLayout(plot_indiv_stdize_orgs_widget)
        plot_indiv_stdize_orgs_layout.setSpacing(30)
        plot_indiv_stdize_orgs_label = QLabel("plot_indiv_stdize_orgs:")
        self.plot_indiv_stdize_orgs_tf = constructors.TrueFalseSelector(False)
        plot_indiv_stdize_orgs_layout.addWidget(plot_indiv_stdize_orgs_label)
        plot_indiv_stdize_orgs_layout.addWidget(self.plot_indiv_stdize_orgs_tf)
        self.plot_indiv_stdize_orgs_optional = OptionalField(plot_indiv_stdize_orgs_widget)

        output_indiv_stdize_orgs_widget = QWidget()
        output_indiv_stdize_orgs_layout = QHBoxLayout(output_indiv_stdize_orgs_widget)
        output_indiv_stdize_orgs_layout.setSpacing(30)
        output_indiv_stdize_orgs_label = QLabel("output_indiv_stdize_orgs:")
        self.output_indiv_stdize_orgs_tf = constructors.TrueFalseSelector(False)
        output_indiv_stdize_orgs_layout.addWidget(output_indiv_stdize_orgs_label)
        output_indiv_stdize_orgs_layout.addWidget(self.output_indiv_stdize_orgs_tf)
        self.output_indiv_stdize_orgs_optional = OptionalField(output_indiv_stdize_orgs_widget)

        stimulus_widget = QWidget()
        stimulus_layout = QHBoxLayout(stimulus_widget)
        stimulus_layout.setSpacing(30)
        stimulus_label = QLabel("stimulus:")
        self.stimulus_tf = constructors.TrueFalseSelector(True)
        stimulus_layout.addWidget(stimulus_label)
        stimulus_layout.addWidget(self.stimulus_tf)
        self.stimulus_optional = OptionalField(stimulus_widget)

        controls_widget = QWidget()
        control_layout = QHBoxLayout(controls_widget)
        control_layout.setSpacing(30)
        control_label = QLabel("control:")
        self.control_tf = constructors.TrueFalseSelector(True)
        control_layout.addWidget(control_label)
        control_layout.addWidget(self.control_tf)
        self.control_optional = OptionalField(controls_widget)

        collapsable_layout.addWidget(self.output_norm_video_optional)
        collapsable_layout.addWidget(self.plot_refine_to_ref_optional)
        collapsable_layout.addWidget(self.plot_refine_to_vid_optional)
        collapsable_layout.addWidget(self.plot_pop_extracted_orgs_optional)
        collapsable_layout.addWidget(self.plot_pop_stdize_orgs_optional)
        collapsable_layout.addWidget(self.plot_indiv_stdize_orgs_optional)
        collapsable_layout.addWidget(self.plot_indiv_stdize_orgs_optional)
        collapsable_layout.addWidget(self.output_indiv_stdize_orgs_optional)
        collapsable_layout.addWidget(self.stimulus_optional)
        collapsable_layout.addWidget(self.control_optional)
        self.axis_layer = axes_base_layer(parent=None, xmin_def=0, xmax_def=6, ymin_def=-255, ymax_def=255, cmap_def="viridis", legend_def=True)
        collapsable_layout.addWidget(self.axis_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None

        result = {}

        if self.output_norm_video_optional.is_checked():
            result["output_norm_video"] = self.output_norm_video_tf.get_value()
        if self.plot_refine_to_ref_optional.is_checked():
            result["plot_refine_to_ref"] = self.plot_refine_to_ref_tf.get_value()
        if self.plot_refine_to_vid_optional.is_checked():
            result["plot_refine_to_vid"] = self.plot_refine_to_vid_tf.get_value()
        if self.plot_pop_extracted_orgs_optional.is_checked():
            result["plot_pop_extracted_orgs"] = self.plot_pop_extracted_orgs_tf.get_value()
        if self.plot_pop_stdize_orgs_optional.is_checked():
            result["plot_pop_stdize_orgs"] = self.plot_pop_stdize_orgs_tf.get_value()
        if self.plot_indiv_stdize_orgs_optional.is_checked():
            result["plot_indiv_stdize_orgs"] = self.plot_indiv_stdize_orgs_tf.get_value()
        if self.output_indiv_stdize_orgs_optional.is_checked():
            result["output_indiv_stdize_orgs"] = self.output_indiv_stdize_orgs_tf.get_value()
        if self.stimulus_optional.is_checked():
            result["stimulus"] = self.stimulus_tf.get_value()
        if self.control_optional.is_checked():
            result["control"] = self.control_tf.get_value()
        axes_val = self.axis_layer.get_value()
        if axes_val is not None:
            result["axes_val"] = axes_val

        return result if result else None

def create_advanced_setup_widget(parent=None):
    """Creates and returns the advanced setup widget/layout for use in a wizard page."""
    # Create scroll area setup
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)

    # Create container widget
    main_container = QWidget()
    scroll.setWidget(main_container)

    # Add main 5 layouts here (Version, Description, Raw, preanalysis, analysis)
    main_layout = QVBoxLayout(main_container)
    main_layout.setSpacing(30)

    version_layer_inst = version_layer()
    main_layout.addWidget(version_layer_inst)

    description_layer_inst = description_layer()
    main_layout.addWidget(description_layer_inst)

    raw_layer_inst = raw_layer()
    #main_layout.addWidget(raw_layer_inst)

    preanalysis_layer_inst = preanalysis_layer()
    main_layout.addWidget(preanalysis_layer_inst)

    analysis_layer_inst = analysis_layer()
    main_layout.addWidget(analysis_layer_inst)

    main_layout.addStretch()

    return scroll  # Return the scroll area containing the widget

def generate_json(version_layer, description_layer, raw_layer, preanalysis_layer, analysis_layer, parent_window):
    """Handles JSON generation and saving."""
    config = {}

    # Only add sections that are enabled and not None
    version_value = version_layer.get_value()
    if version_value is not None:
        config["version"] = version_value

    description_value = description_layer.get_value()
    if description_value is not None:
        config["description"] = description_value

    raw_value = raw_layer.get_value()
    if raw_value is not None:
        config["raw"] = raw_value

    preanalysis_value = preanalysis_layer.get_value()
    if preanalysis_value is not None:
        config["preanalysis"] = preanalysis_value

    analysis_value = analysis_layer.get_value()
    if analysis_value is not None:
        config["analysis"] = analysis_value

    if not config:  # Check if config is empty
        QMessageBox.warning(parent_window, "Warning", "No configuration data to save. Please enable at least one section.")
        return

    # Show save file dialog
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getSaveFileName(
        parent_window,
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
            QMessageBox.information(parent_window, "Success", "Configuration file saved successfully!")
        except Exception as e:
            QMessageBox.critical(parent_window, "Error", f"Failed to save file:\n{str(e)}")

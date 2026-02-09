from PySide6.QtWidgets import (QWidget)


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
        self.description_value = constructors.FreetextBox("A cunningly created pipeline and analysis JSON.")

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
        self.collapsable = constructors.CollapsibleSection("Raw", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        video_format_layout = QHBoxLayout()
        video_format_layout.setSpacing(30)
        video_format_label = QLabel("Video Format:")
        self.video_format_value = constructors.FormatEditorWidget("Video Format:", "{IDnum}_{Year:4}{Month:2}{Day:2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}.avi")
        video_format_layout.addWidget(video_format_label)
        video_format_layout.addWidget(self.video_format_value)
        video_format_layout.setAlignment(Qt.AlignLeft)

        collapsable_layout.addLayout(video_format_layout)
        self.raw_metadata_layer = raw_metadata_layer()
        collapsable_layout.addWidget(self.raw_metadata_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "video_format": self.video_format_value.get_value(),
            "metadata": self.raw_metadata_layer.get_value()
        }

class raw_metadata_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Metadata", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        type_layout = QHBoxLayout()
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_value = constructors.FreetextBox("text_file")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_value)

        format_layout = QHBoxLayout()
        format_layout.setSpacing(30)
        metadata_format_label = QLabel("Metadata Format:")
        self.metadata_format_value = constructors.FormatEditorWidget("Metadata Format:", "{IDnum}_{Year:4}{Month:2}{Day:2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}.csv")
        format_layout.addWidget(metadata_format_label)
        format_layout.addWidget(self.metadata_format_value)
        format_layout.setAlignment(Qt.AlignLeft)

        collapsable_layout.addLayout(type_layout)
        collapsable_layout.addLayout(format_layout)
        self.raw_metadata_fieldstoload_layer = raw_metadata_fieldstoload_layer()
        collapsable_layout.addWidget(self.raw_metadata_fieldstoload_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "type": self.type_value.get_text(),
            "metadata_format": self.metadata_format_value.get_value(),
            "fields_to_load": self.raw_metadata_fieldstoload_layer.get_value()
        }

class raw_metadata_fieldstoload_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Fields to Load", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        timestamps_layout = QHBoxLayout()
        timestamps_layout.setSpacing(30)
        timestamps_label = QLabel("Timestamps:")
        self.timestamps_value = constructors.FreetextBox("Timestamp_us")
        timestamps_layout.addWidget(timestamps_label)
        timestamps_layout.addWidget(self.timestamps_value)

        stimulus_layout = QHBoxLayout()
        stimulus_layout.setSpacing(30)
        stimulus_label = QLabel("Stimulus Train:")
        self.stimulus_value = constructors.FreetextBox("StimulusOn")
        stimulus_layout.addWidget(stimulus_label)
        stimulus_layout.addWidget(self.stimulus_value)

        collapsable_layout.addLayout(timestamps_layout)
        collapsable_layout.addLayout(stimulus_layout)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "timestamps": self.timestamps_value.get_text(),
            "stimulus_train": self.stimulus_value.get_text(),
        }

class preanalysis_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("preanalysis", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        image_format_layout = QHBoxLayout()
        image_format_layout.setSpacing(30)
        image_format_label = QLabel("Image Format:")
        self.image_format_value = constructors.FormatEditorWidget("Image Format:", "{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}{:.1}_extract_reg_avg.tif")
        image_format_layout.addWidget(image_format_label)
        image_format_layout.addWidget(self.image_format_value)
        image_format_layout.setAlignment(Qt.AlignLeft)

        video_format_layout = QHBoxLayout()
        video_format_layout.setSpacing(30)
        video_format_label = QLabel("Video Format:")
        self.video_format_value = constructors.FormatEditorWidget("Video Format:", "{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}{:.1}_extract_reg_cropped.avi")
        video_format_layout.addWidget(video_format_label)
        video_format_layout.addWidget(self.video_format_value)
        video_format_layout.setAlignment(Qt.AlignLeft)

        mask_format_layout = QHBoxLayout()
        mask_format_layout.setSpacing(30)
        mask_format_label = QLabel("Mask Format:")
        self.mask_format_value = constructors.FormatEditorWidget("Mask Format:", "{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}{:.1}_extract_reg_cropped_mask.avi")
        mask_format_layout.addWidget(mask_format_label)
        mask_format_layout.addWidget(self.mask_format_value)
        mask_format_layout.setAlignment(Qt.AlignLeft)

        recursive_search_layout = QHBoxLayout()
        recursive_search_layout.setSpacing(30)
        recursive_search_label = QLabel("Recursive Search:")
        self.recursive_search_tf = constructors.TrueFalseSelector(False)
        recursive_search_layout.addWidget(recursive_search_label)
        recursive_search_layout.addWidget(self.recursive_search_tf)

        collapsable_layout.addLayout(image_format_layout)
        collapsable_layout.addLayout(video_format_layout)
        collapsable_layout.addLayout(mask_format_layout)
        self.preanalysis_metadata_layer = preanalysis_metadata_layer()
        collapsable_layout.addWidget(self.preanalysis_metadata_layer)
        collapsable_layout.addLayout(recursive_search_layout)
        self.preanalysis_pipeline_params_layer = preanalysis_pipeline_params_layer()
        collapsable_layout.addWidget(self.preanalysis_pipeline_params_layer)

        # Connect format changes to update groupby
        self.image_format_value.formatChanged.connect(self.update_groupby_elements)
        self.video_format_value.formatChanged.connect(self.update_groupby_elements)
        self.mask_format_value.formatChanged.connect(self.update_groupby_elements)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def update_groupby_elements(self):
        """Update the groupby widget's available elements when formats change"""
        image_format = self.image_format_value.get_value()
        video_format = self.video_format_value.get_value()
        mask_format = self.mask_format_value.get_value()

        self.preanalysis_pipeline_params_layer.update_format_references(
            image_format=image_format,
            video_format=video_format,
            mask_format=mask_format
        )

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "image_format": self.image_format_value.get_value(),
            "video_format": self.video_format_value.get_value(),
            "mask_format": self.mask_format_value.get_value(),
            "metadata" : self.preanalysis_metadata_layer.get_value(),
            "recursive_search": self.recursive_search_tf.get_value(),
            "pipeline_params" : self.preanalysis_pipeline_params_layer.get_value(),
        }

class preanalysis_metadata_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Metadata", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        type_layout = QHBoxLayout()
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_value = constructors.FreetextBox("text_file")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_value)

        format_layout = QHBoxLayout()
        format_layout.setSpacing(30)
        metadata_format_label = QLabel("Metadata Format:")
        self.metadata_format_value = constructors.FormatEditorWidget("Metadata Format:", "{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}{:.1}_extract_reg_cropped.csv")
        format_layout.addWidget(metadata_format_label)
        format_layout.addWidget(self.metadata_format_value)
        format_layout.setAlignment(Qt.AlignLeft)

        collapsable_layout.addLayout(type_layout)
        collapsable_layout.addLayout(format_layout)
        self.preanalysis_metadata_fieldstoload_layer = preanalysis_metadata_fieldstoload_layer()
        collapsable_layout.addWidget(self.preanalysis_metadata_fieldstoload_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "type": self.type_value.get_text(),
            "metadata_format": self.metadata_format_value.get_value(),
            "fields_to_load": self.preanalysis_metadata_fieldstoload_layer.get_value(),
        }

class preanalysis_metadata_fieldstoload_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Fields to Load", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        framestamps_layout = QHBoxLayout()
        framestamps_layout.setSpacing(30)
        framestamps_label = QLabel("Framestamps:")
        self.framestamps_value = constructors.FreetextBox("OriginalFrameNumber")
        framestamps_layout.addWidget(framestamps_label)
        framestamps_layout.addWidget(self.framestamps_value)

        collapsable_layout.addLayout(framestamps_layout)
        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "framestamps": self.framestamps_value.get_text(),
        }

class preanalysis_pipeline_params_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Pipeline Params", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        gausblur_layout = QHBoxLayout()
        gausblur_layout.setSpacing(30)
        gausblur_label = QLabel("Gaus Blur:")
        self.gausblur_value = constructors.FreetextBox("0.0")
        gausblur_layout.addWidget(gausblur_label)
        gausblur_layout.addWidget(self.gausblur_value)

        modalities_layout = QHBoxLayout()
        modalities_layout.setSpacing(30)
        modalities_label = QLabel("Modalities:")
        self.modalities_list_creator = constructors.ListEditorWidget()
        modalities_layout.addWidget(modalities_label)
        modalities_layout.addWidget(self.modalities_list_creator)
        modalities_layout.setAlignment(Qt.AlignLeft)

        alignment_ref_layout = QHBoxLayout()
        alignment_ref_layout.setSpacing(30)
        alignment_ref_label = QLabel("Alignment Reference Modality:")
        self.alignment_ref_value = constructors.AlignmentModalitySelector(
            self.modalities_list_creator,
            "null"
        )
        alignment_ref_layout.addWidget(alignment_ref_label)
        alignment_ref_layout.addWidget(self.alignment_ref_value)
        alignment_ref_layout.setAlignment(Qt.AlignLeft)

        output_folder_layout = QHBoxLayout()
        output_folder_layout.setSpacing(30)
        output_folder_label = QLabel("Output Folder:")
        self.output_folder_value = constructors.OpenFolder()
        output_folder_layout.addWidget(output_folder_label)
        output_folder_layout.addWidget(self.output_folder_value)
        output_folder_layout.setAlignment(Qt.AlignLeft)

        groupby_layout = QHBoxLayout()
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

        correct_torsion_layout = QHBoxLayout()
        correct_torsion_layout.setSpacing(30)
        correct_torsion_label = QLabel("Correct Torsion:")
        self.correct_torsion_tf = constructors.TrueFalseSelector(True)
        correct_torsion_layout.addWidget(correct_torsion_label)
        correct_torsion_layout.addWidget(self.correct_torsion_tf)

        intra_stack_xform_layout = QHBoxLayout()
        intra_stack_xform_layout.setSpacing(30)
        intra_stack_xform_label = QLabel("Intra Stack Xform:")
        self.intra_stack_xform_tf = constructors.AffineRigidSelector(True)
        intra_stack_xform_layout.addWidget(intra_stack_xform_label)
        intra_stack_xform_layout.addWidget(self.intra_stack_xform_tf)

        inter_stack_xform_layout = QHBoxLayout()
        inter_stack_xform_layout.setSpacing(30)
        inter_stack_xform_label = QLabel("Inter Stack Xform:")
        self.inter_stack_xform_tf = constructors.AffineRigidSelector(True)
        inter_stack_xform_layout.addWidget(inter_stack_xform_label)
        inter_stack_xform_layout.addWidget(self.inter_stack_xform_tf)

        flat_field_layout = QHBoxLayout()
        flat_field_layout.setSpacing(30)
        flat_field_label = QLabel("Flat Field:")
        self.flat_field_tf = constructors.TrueFalseSelector(False)
        flat_field_layout.addWidget(flat_field_label)
        flat_field_layout.addWidget(self.flat_field_tf)

        collapsable_layout.addLayout(gausblur_layout)
        self.preanalysis_pipeline_params_maskroi_layer = preanalysis_pipeline_params_maskroi_layer()
        collapsable_layout.addWidget(self.preanalysis_pipeline_params_maskroi_layer)
        collapsable_layout.addLayout(modalities_layout)
        collapsable_layout.addLayout(alignment_ref_layout)
        collapsable_layout.addLayout(output_folder_layout)
        collapsable_layout.addLayout(groupby_layout)
        collapsable_layout.addLayout(correct_torsion_layout)
        collapsable_layout.addLayout(intra_stack_xform_layout)
        collapsable_layout.addLayout(inter_stack_xform_layout)
        collapsable_layout.addLayout(flat_field_layout)
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
        return {
            "gaus_blur" : parse_value(self.gausblur_value.get_text()),
            "mask_roi" : self.preanalysis_pipeline_params_maskroi_layer.get_value(),
            "modalities": self.modalities_list_creator.get_list(),
            "alignment_reference_modality": None if self.alignment_ref_value.get_value() == "null" else self.alignment_ref_value.get_value(),
            "output_folder": self.output_folder_value.get_text(),
            "group_by": None if self.groupby_value.get_value() == "null" else self.groupby_value.get_value(),
            "correct_torsion": self.correct_torsion_tf.get_value(),
            "intra_stack_xform": self.intra_stack_xform_tf.get_value(),
            "inter_stack_xform": self.inter_stack_xform_tf.get_value(),
            "flat_field": self.flat_field_tf.get_value(),
            "trim": self.preanalysis_pipeline_params_trim_layer.get_value()
        }

class preanalysis_pipeline_params_maskroi_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Mask ROI", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        r_layout = QHBoxLayout()
        r_layout.setSpacing(30)
        r_label = QLabel("Starting Row:")
        self.r_value = constructors.FreetextBox("0")
        r_layout.addWidget(r_label)
        r_layout.addWidget(self.r_value)

        c_layout = QHBoxLayout()
        c_layout.setSpacing(30)
        c_label = QLabel("Starting Column:")
        self.c_value = constructors.FreetextBox("0")
        c_layout.addWidget(c_label)
        c_layout.addWidget(self.c_value)

        width_layout = QHBoxLayout()
        width_layout.setSpacing(30)
        width_label = QLabel("Width:")
        self.width_value = constructors.FreetextBox("-1")
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.width_value)

        height_layout = QHBoxLayout()
        height_layout.setSpacing(30)
        height_label = QLabel("Height:")
        self.height_value = constructors.FreetextBox("-1")
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_value)

        collapsable_layout.addLayout(r_layout)
        collapsable_layout.addLayout(c_layout)
        collapsable_layout.addLayout(width_layout)
        collapsable_layout.addLayout(height_layout)

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
        return {
            "r" : parse_value(self.r_value.get_text()),
            "c" : parse_value(self.c_value.get_text()),
            "width" : parse_value(self.width_value.get_text()),
            "height" : parse_value(self.height_value.get_text()),
        }

class preanalysis_pipeline_params_trim_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Trimming", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        start_idx_layout = QHBoxLayout()
        start_idx_layout.setSpacing(30)
        start_idx_label = QLabel("Start Index:")
        self.start_idx_value = constructors.FreetextBox("0.0")
        start_idx_layout.addWidget(start_idx_label)
        start_idx_layout.addWidget(self.start_idx_value)

        end_idx_layout = QHBoxLayout()
        end_idx_layout.setSpacing(30)
        end_idx_label = QLabel("End Index:")
        self.end_idx_value = constructors.FreetextBox("-1.0")
        end_idx_layout.addWidget(end_idx_label)
        end_idx_layout.addWidget(self.end_idx_value)

        collapsable_layout.addLayout(start_idx_layout)
        collapsable_layout.addLayout(end_idx_layout)

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
        return {
            "start_idx": parse_value(self.start_idx_value.get_text()),
            "end_idx": parse_value(self.end_idx_value.get_text()),
        }


class analysis_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("analysis", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        image_format_layout = QHBoxLayout()
        image_format_layout.setSpacing(30)
        image_format_label = QLabel("Image Format:")
        self.image_format_value = constructors.FormatEditorWidget("Image Format:", "{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}_ALL_ACQ_AVG.tif")
        image_format_layout.addWidget(image_format_label)
        image_format_layout.addWidget(self.image_format_value)
        image_format_layout.setAlignment(Qt.AlignLeft)

        queryloc_format_layout = QHBoxLayout()
        queryloc_format_layout.setSpacing(30)
        queryloc_format_label = QLabel("Query Loc Format:")
        self.queryloc_format_value = constructors.FormatEditorWidget("Video Format:", "{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}_ALL_ACQ_AVG_{QueryLoc:s?}coords.csv")
        queryloc_format_layout.addWidget(queryloc_format_label)
        queryloc_format_layout.addWidget(self.queryloc_format_value)
        queryloc_format_layout.setAlignment(Qt.AlignLeft)

        video_format_layout = QHBoxLayout()
        video_format_layout.setSpacing(30)
        video_format_label = QLabel("Video Format:")
        self.video_format_value = constructors.FormatEditorWidget("Mask Format:", "{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}_piped.avi")
        video_format_layout.addWidget(video_format_label)
        video_format_layout.addWidget(self.video_format_value)
        video_format_layout.setAlignment(Qt.AlignLeft)

        recursive_search_layout = QHBoxLayout()
        recursive_search_layout.setSpacing(30)
        recursive_search_label = QLabel("Recursive Search:")
        self.recursive_search_tf = constructors.TrueFalseSelector(True)
        recursive_search_layout.addWidget(recursive_search_label)
        recursive_search_layout.addWidget(self.recursive_search_tf)

        collapsable_layout.addLayout(image_format_layout)
        collapsable_layout.addLayout(queryloc_format_layout)
        collapsable_layout.addLayout(video_format_layout)

        self.analysis_metadata_layer = analysis_metadata_layer()
        collapsable_layout.addWidget(self.analysis_metadata_layer)

        collapsable_layout.addLayout(recursive_search_layout)

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
        return {
            "image_format" : self.image_format_value.get_value(),
            "queryloc_format" : self.queryloc_format_value.get_value(),
            "video_format" : self.video_format_value.get_value(),
            "metadata" : self.analysis_metadata_layer.get_value(),
            "recursive_search" : self.recursive_search_tf.get_value(),
            "control" : self.analysis_control_layer.get_value(),
            "analysis_params" : self.analysis_analysis_params_layer.get_value(),
            "display_params" : self.analysis_display_params_layer.get_value(),
        }

class analysis_metadata_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Metadata", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        type_layout = QHBoxLayout()
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_value = constructors.FreetextBox("text_file")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_value)

        format_layout = QHBoxLayout()
        format_layout.setSpacing(30)
        metadata_format_label = QLabel("Metadata Format:")
        self.metadata_format_value = constructors.FormatEditorWidget("Metadata Format:", "{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}_piped.csv")
        format_layout.addWidget(metadata_format_label)
        format_layout.addWidget(self.metadata_format_value)
        format_layout.setAlignment(Qt.AlignLeft)

        collapsable_layout.addLayout(type_layout)
        collapsable_layout.addLayout(format_layout)

        self.raw_analysis_fieldstoload_layer = raw_analysis_fieldstoload_layer()
        collapsable_layout.addWidget(self.raw_analysis_fieldstoload_layer)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "type" : self.type_value.get_text(),
            "metadata_format" : self.metadata_format_value.get_value(),
            "fields_to_load" : self.raw_analysis_fieldstoload_layer.get_value(),
        }

class raw_analysis_fieldstoload_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Fields to Load", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        framestamps_layout = QHBoxLayout()
        framestamps_layout.setSpacing(30)
        framestamps_label = QLabel("Framestamps:")
        self.framestamps_value = constructors.FreetextBox("FrameStamps")
        framestamps_layout.addWidget(framestamps_label)
        framestamps_layout.addWidget(self.framestamps_value)

        stimsequences_layout = QHBoxLayout()
        stimsequences_layout.setSpacing(30)
        stimsequences_label = QLabel("Stimulus Sequence:")
        self.stimsequences_value = constructors.FreetextBox("StimulusOn")
        stimsequences_layout.addWidget(stimsequences_label)
        stimsequences_layout.addWidget(self.stimsequences_value)

        collapsable_layout.addLayout(framestamps_layout)
        collapsable_layout.addLayout(stimsequences_layout)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "framestamps" : self.framestamps_value.get_text(),
            "stimulus_sequence" : self.stimsequences_value.get_text(),
        }

class analysis_control_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Control", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        location_layout = QHBoxLayout()
        location_layout.setSpacing(30)
        location_label = QLabel("Location:")
        self.location_value = constructors.FreetextBox("folder")
        location_layout.addWidget(location_label)
        location_layout.addWidget(self.location_value)

        folder_name_layout = QHBoxLayout()
        folder_name_layout.setSpacing(30)
        folder_name_label = QLabel("Folder Name:")
        self.folder_name_value = constructors.FreetextBox("control")
        folder_name_layout.addWidget(folder_name_label)
        folder_name_layout.addWidget(self.folder_name_value)

        collapsable_layout.addLayout(location_layout)
        collapsable_layout.addLayout(folder_name_layout)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "location" : self.location_value.get_text(),
            "folder_name" : self.folder_name_value.get_text(),
        }

class analysis_analysis_params_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Analysis Params", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        modalities_layout = QHBoxLayout()
        modalities_layout.setSpacing(30)
        modalities_label = QLabel("Modalities:")
        self.modalities_list_creator = constructors.ListEditorWidget()
        modalities_layout.addWidget(modalities_label)
        modalities_layout.addWidget(self.modalities_list_creator)
        modalities_layout.setAlignment(Qt.AlignLeft)

        output_folder_layout = QHBoxLayout()
        output_folder_layout.setSpacing(30)
        output_folder_label = QLabel("Output Folder:")
        self.output_folder_value = constructors.OpenFolder()
        output_folder_layout.addWidget(output_folder_label)
        output_folder_layout.addWidget(self.output_folder_value)
        output_folder_layout.setAlignment(Qt.AlignLeft)

        gausblur_layout = QHBoxLayout()
        gausblur_layout.setSpacing(30)
        gausblur_label = QLabel("Gaus Blur:")
        self.gausblur_value = constructors.FreetextBox("0.0")
        gausblur_layout.addWidget(gausblur_label)
        gausblur_layout.addWidget(self.gausblur_value)

        collapsable_layout.addLayout(modalities_layout)
        collapsable_layout.addLayout(output_folder_layout)

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

        collapsable_layout.addLayout(gausblur_layout)

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
        return {
            "modalities" : self.modalities_list_creator.get_list(),
            "output_folder" : self.output_folder_value.get_text(),
            "normalization" : self.analysis_analysis_params_normalization_layer.get_value(),
            "segmentation" : self.analysis_analysis_params_segmentation_layer.get_value(),
            "exclusion_criteria" : self.analysis_analysis_params_exclusion_criteria_layer.get_value(),
            "standardization" : self.analysis_analysis_params_standardization_layer.get_value(),
            "summary" : self.analysis_analysis_params_summary_layer.get_value(),
            "gaus_blur" : parse_value(self.gausblur_value.get_text()),
            "mask_roi" : self.analysis_analysis_params_maskroi_layer.get_value(),
        }

class analysis_analysis_params_normalization_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Normalization", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        method_layout = QHBoxLayout()
        method_layout.setSpacing(10)
        method_label = QLabel("Method:")
        self.method_value = constructors.DropdownMenu("score", ["mean", "median", "none"])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_value)
        method_layout.setAlignment(Qt.AlignLeft)

        rescaled_layout = QHBoxLayout()
        rescaled_layout.setSpacing(30)
        rescaled_label = QLabel("Rescaled:")
        self.rescaled_tf = constructors.TrueFalseSelector(True)
        rescaled_layout.addWidget(rescaled_label)
        rescaled_layout.addWidget(self.rescaled_tf)

        rescaled_mean_layout = QHBoxLayout()
        rescaled_layout.setSpacing(30)
        rescaled_mean_label = QLabel("Rescaled Mean:")
        self.rescaled_mean_value = constructors.FreetextBox("70")
        rescaled_mean_layout.addWidget(rescaled_mean_label)
        rescaled_mean_layout.addWidget(self.rescaled_mean_value)

        rescaled_std_layout = QHBoxLayout()
        rescaled_layout.setSpacing(30)
        rescaled_std_label = QLabel("Rescaled STD:")
        self.rescaled_std_value = constructors.FreetextBox("35")
        rescaled_std_layout.addWidget(rescaled_std_label)
        rescaled_std_layout.addWidget(self.rescaled_std_value)

        collapsable_layout.addLayout(method_layout)
        collapsable_layout.addLayout(rescaled_layout)
        collapsable_layout.addLayout(rescaled_mean_layout)
        collapsable_layout.addLayout(rescaled_std_layout)

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
        return {
            "method" : self.method_value.get_value(),
            "rescaled" : self.rescaled_tf.get_value(),
            "rescale_mean" : parse_value(self.rescaled_mean_value.get_text()),
            "rescale_std" : parse_value(self.rescaled_std_value.get_text()),
        }

class analysis_analysis_params_segmentation_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Segmentation", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        refine_to_ref_layout = QHBoxLayout()
        refine_to_ref_layout.setSpacing(30)
        refine_to_ref_label = QLabel("Refine to Ref:")
        self.refine_to_ref_tf = constructors.TrueFalseSelector(False)
        refine_to_ref_layout.addWidget(refine_to_ref_label)
        refine_to_ref_layout.addWidget(self.refine_to_ref_tf)

        refine_to_vid_layout = QHBoxLayout()
        refine_to_vid_layout.setSpacing(30)
        refine_to_vid_label = QLabel("Refine to Vid:")
        self.refine_to_vid_tf = constructors.TrueFalseSelector(False)
        refine_to_vid_layout.addWidget(refine_to_vid_label)
        refine_to_vid_layout.addWidget(self.refine_to_vid_tf)

        radius_layout = QHBoxLayout()
        radius_layout.setSpacing(30)
        radius_label = QLabel("Radius:")
        self.radius_value = constructors.FreetextBox("auto")
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(self.radius_value)

        shape_layout = QHBoxLayout()
        shape_layout.setSpacing(30)
        shape_label = QLabel("Shape:")
        self.shape_value = constructors.DropdownMenu("disk", "box")
        shape_layout.addWidget(shape_label)
        shape_layout.addWidget(self.shape_value)
        shape_layout.setAlignment(Qt.AlignLeft)

        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(30)
        summary_label = QLabel("Summary:")
        self.summary_value = constructors.DropdownMenu("mean", "median")
        summary_layout.addWidget(summary_label)
        summary_layout.addWidget(self.summary_value)
        summary_layout.setAlignment(Qt.AlignLeft)

        collapsable_layout.addLayout(refine_to_ref_layout)
        collapsable_layout.addLayout(refine_to_vid_layout)
        collapsable_layout.addLayout(radius_layout)
        collapsable_layout.addLayout(shape_layout)
        collapsable_layout.addLayout(summary_layout)

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
        return {
            "refine_to_ref" : self.refine_to_ref_tf.get_value(),
            "refine_to_vid" : self.refine_to_vid_tf.get_value(),
            "radius" : parse_value(self.radius_value.get_text()),
            "shape" : self.shape_value.get_value(),
            "summary" : self.summary_value.get_value(),
        }

class analysis_analysis_params_exclusion_criteria_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Exclusion Criteria", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        type_layout = QHBoxLayout()
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_value = constructors.DropdownMenu("stim-relative", "absolute")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_value)
        type_layout.setAlignment(Qt.AlignLeft)

        units_layout = QHBoxLayout()
        units_layout.setSpacing(30)
        units_label = QLabel("Units:")
        self.units_value = constructors.DropdownMenu("time", "frames")
        units_layout.addWidget(units_label)
        units_layout.addWidget(self.units_value)
        units_layout.setAlignment(Qt.AlignLeft)

        start_layout = QHBoxLayout()
        start_layout.setSpacing(30)
        start_label = QLabel("Start:")
        self.start_value = constructors.FreetextBox("-1")
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_value)

        stop_layout = QHBoxLayout()
        stop_layout.setSpacing(30)
        stop_label = QLabel("Stop:")
        self.stop_value = constructors.FreetextBox("0")
        stop_layout.addWidget(stop_label)
        stop_layout.addWidget(self.stop_value)

        fraction_layout = QHBoxLayout()
        fraction_layout.setSpacing(30)
        fraction_label = QLabel("Fraction:")
        self.fraction_value = constructors.FreetextBox("0.3")
        fraction_layout.addWidget(fraction_label)
        fraction_layout.addWidget(self.fraction_value)

        collapsable_layout.addLayout(type_layout)
        collapsable_layout.addLayout(units_layout)
        collapsable_layout.addLayout(start_layout)
        collapsable_layout.addLayout(stop_layout)
        collapsable_layout.addLayout(fraction_layout)

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
        return {
            "type" : self.type_value.get_value(),
            "units" : self.units_value.get_value(),
            "start" : parse_value(self.start_value.get_text()),
            "stop" : parse_value(self.stop_value.get_text()),
            "fraction" : parse_value(self.fraction_value.get_text()),
        }

class analysis_analysis_params_standardization_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Standardization", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        method_layout = QHBoxLayout()
        method_layout.setSpacing(30)
        method_label = QLabel("Method:")
        self.method_value = constructors.DropdownMenu("mean_sub", ["std", "linear_std", "linear_vast", "relative_change", "none"])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_value)
        method_layout.setAlignment(Qt.AlignLeft)

        type_layout = QHBoxLayout()
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_value = constructors.DropdownMenu("stim-relative", "absolute")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_value)
        type_layout.setAlignment(Qt.AlignLeft)

        units_layout = QHBoxLayout()
        units_layout.setSpacing(30)
        units_label = QLabel("Units:")
        self.units_value = constructors.DropdownMenu("time", "frames")
        units_layout.addWidget(units_label)
        units_layout.addWidget(self.units_value)
        units_layout.setAlignment(Qt.AlignLeft)

        start_layout = QHBoxLayout()
        start_layout.setSpacing(30)
        start_label = QLabel("Start:")
        self.start_value = constructors.FreetextBox("-1")
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_value)

        stop_layout = QHBoxLayout()
        stop_layout.setSpacing(30)
        stop_label = QLabel("Stop:")
        self.stop_value = constructors.FreetextBox("0")
        stop_layout.addWidget(stop_label)
        stop_layout.addWidget(self.stop_value)

        collapsable_layout.addLayout(method_layout)
        collapsable_layout.addLayout(type_layout)
        collapsable_layout.addLayout(units_layout)
        collapsable_layout.addLayout(start_layout)
        collapsable_layout.addLayout(stop_layout)

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
        return {
            "method" : self.method_value.get_value(),
            "type" : self.type_value.get_value(),
            "units" : self.units_value.get_value(),
            "start" : parse_value(self.start_value.get_text()),
            "stop" : parse_value(self.stop_value.get_text())
        }

class analysis_analysis_params_summary_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Summary", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        method_layout = QHBoxLayout()
        method_layout.setSpacing(30)
        method_label = QLabel("Method:")
        self.method_value = constructors.DropdownMenu("rms", ["std", "var", "avg"])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_value)
        method_layout.setAlignment(Qt.AlignLeft)

        windowsize_layout = QHBoxLayout()
        windowsize_layout.setSpacing(30)
        windowsize_label = QLabel("Window Size:")
        self.windowsize_value = constructors.FreetextBox("1")
        windowsize_layout.addWidget(windowsize_label)
        windowsize_layout.addWidget(self.windowsize_value)

        control_layout = QHBoxLayout()
        control_layout.setSpacing(30)
        control_label = QLabel("Control:")
        self.control_value = constructors.DropdownMenu("none", ["subtraction", "division"])
        control_layout.addWidget(control_label)
        control_layout.addWidget(self.control_value)
        control_layout.setAlignment(Qt.AlignLeft)

        indiv_cutoff_layout = QHBoxLayout()
        indiv_cutoff_layout.setSpacing(30)
        indiv_cutoff_label = QLabel("Individual Cutoff:")
        self.indiv_cutoff_value = constructors.FreetextBox("5")
        indiv_cutoff_layout.addWidget(indiv_cutoff_label)
        indiv_cutoff_layout.addWidget(self.indiv_cutoff_value)

        collapsable_layout.addLayout(method_layout)
        collapsable_layout.addLayout(windowsize_layout)
        collapsable_layout.addLayout(control_layout)

        self.analysis_analysis_params_summary_metrics_layer = analysis_analysis_params_summary_metrics_layer()
        collapsable_layout.addWidget(self.analysis_analysis_params_summary_metrics_layer)

        collapsable_layout.addLayout(indiv_cutoff_layout)

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
        return {
            "method" : self.method_value.get_value(),
            "window_size" : parse_value(self.windowsize_value.get_text()),
            "control" : self.control_value.get_value(),
            "metrics" : self.analysis_analysis_params_summary_metrics_layer.get_value(),
            "indiv_cutoff" : parse_value(self.indiv_cutoff_value.get_text()),
        }

class analysis_analysis_params_summary_metrics_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Metrics", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        type_layout = QHBoxLayout()
        type_layout.setSpacing(30)
        type_label = QLabel("Type:")
        self.type_list_creator = constructors.ListEditorWidget()
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_list_creator)
        type_layout.setAlignment(Qt.AlignLeft)

        measured_layout = QHBoxLayout()
        measured_layout.setSpacing(30)
        measured_label = QLabel("Measured:")
        self.measured_value = constructors.FreetextBox("stim-relative")
        measured_layout.addWidget(measured_label)
        measured_layout.addWidget(self.measured_value)

        units_layout = QHBoxLayout()
        units_layout.setSpacing(30)
        units_label = QLabel("Units:")
        self.units_value = constructors.FreetextBox("time")
        units_layout.addWidget(units_label)
        units_layout.addWidget(self.units_value)

        prestim_layout = QHBoxLayout()
        prestim_layout.setSpacing(30)
        prestim_label = QLabel("Prestim:")
        self.prestim_value = constructors.FreetextBox("-1, 0")
        prestim_layout.addWidget(prestim_label)
        prestim_layout.addWidget(self.prestim_value)

        poststim_layout = QHBoxLayout()
        poststim_layout.setSpacing(30)
        poststim_label = QLabel("Poststim:")
        self.poststim_value = constructors.FreetextBox("0, 1")
        poststim_layout.addWidget(poststim_label)
        poststim_layout.addWidget(self.poststim_value)

        collapsable_layout.addLayout(type_layout)
        collapsable_layout.addLayout(measured_layout)
        collapsable_layout.addLayout(units_layout)
        collapsable_layout.addLayout(prestim_layout)
        collapsable_layout.addLayout(poststim_layout)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "type" : self.type_list_creator.get_list(),
            "measured" : self.measured_value.get_text(),
            "units" : self.units_value.get_text(),
            "prestim" : f"[{self.prestim_value.get_text()}]",
            "poststim" : f"[{self.poststim_value.get_text()}]",
        }

class analysis_analysis_params_maskroi_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("MaskROI", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        r_layout = QHBoxLayout()
        r_layout.setSpacing(30)
        r_label = QLabel("R:")
        self.r_value = constructors.FreetextBox("0")
        r_layout.addWidget(r_label)
        r_layout.addWidget(self.r_value)

        c_layout = QHBoxLayout()
        c_layout.setSpacing(30)
        c_label = QLabel("C:")
        self.c_value = constructors.FreetextBox("0")
        c_layout.addWidget(c_label)
        c_layout.addWidget(self.c_value)

        width_layout = QHBoxLayout()
        width_layout.setSpacing(30)
        width_label = QLabel("Width:")
        self.width_value = constructors.FreetextBox("-1")
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.width_value)

        height_layout = QHBoxLayout()
        height_layout.setSpacing(30)
        height_label = QLabel("Height:")
        self.height_value = constructors.FreetextBox("-1")
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_value)

        collapsable_layout.addLayout(r_layout)
        collapsable_layout.addLayout(c_layout)
        collapsable_layout.addLayout(width_layout)
        collapsable_layout.addLayout(height_layout)

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
        return {
            "r" : parse_value(self.r_value.get_text()),
            "c" : parse_value(self.c_value.get_text()),
            "width" : parse_value(self.width_value.get_text()),
            "height" : parse_value(self.height_value.get_text()),
        }

class analysis_display_params_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Display Parameters", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        saveas_layout = QHBoxLayout()
        saveas_layout.setSpacing(30)
        saveas_label = QLabel("Save as:")
        self.saveas_list_creator = constructors.SaveasExtensionsEditorWidget("Save as:", "[]")
        saveas_layout.addWidget(saveas_label)
        saveas_layout.addWidget(self.saveas_list_creator)
        saveas_layout.setAlignment(Qt.AlignLeft)

        pause_per_folder_layout = QHBoxLayout()
        pause_per_folder_layout.setSpacing(30)
        pause_per_folder_label = QLabel("Pause per Folder:")
        self.pause_per_folder_tf = constructors.TrueFalseSelector(True)
        pause_per_folder_layout.addWidget(pause_per_folder_label)
        pause_per_folder_layout.addWidget(self.pause_per_folder_tf)

        self.analysis_display_params_pop_summary_overlap_layer = analysis_display_params_pop_summary_overlap_layer()
        collapsable_layout.addWidget(self.analysis_display_params_pop_summary_overlap_layer)

        self.analysis_display_params_pop_summary_seq_layer = analysis_display_params_pop_summary_seq_layer()
        collapsable_layout.addWidget(self.analysis_display_params_pop_summary_seq_layer)

        self.analysis_display_params_indiv_summary_layer = analysis_display_params_indiv_summary_layer()
        collapsable_layout.addWidget(self.analysis_display_params_indiv_summary_layer)

        collapsable_layout.addLayout(saveas_layout)
        collapsable_layout.addLayout(pause_per_folder_layout)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "pop_summary_overlap" : self.analysis_display_params_pop_summary_overlap_layer.get_value(),
            "pop_summary_seq" : self.analysis_display_params_pop_summary_seq_layer.get_value(),
            "indiv_summary" : self.analysis_display_params_indiv_summary_layer.get_value(),
            "saveas" : self.saveas_list_creator.get_value(),
            "pause_per_folder" : self.pause_per_folder_tf.get_value(),
        }

class analysis_display_params_pop_summary_overlap_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Pop Summary Overlap", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        stimulus_layout = QHBoxLayout()
        stimulus_layout.setSpacing(30)
        stimulus_label = QLabel("Stimulus:")
        self.stimulus_tf = constructors.TrueFalseSelector(True)
        stimulus_layout.addWidget(stimulus_label)
        stimulus_layout.addWidget(self.stimulus_tf)

        control_layout = QHBoxLayout()
        control_layout.setSpacing(30)
        control_label = QLabel("Control:")
        self.control_tf = constructors.TrueFalseSelector(True)
        control_layout.addWidget(control_label)
        control_layout.addWidget(self.control_tf)

        relative_layout = QHBoxLayout()
        relative_layout.setSpacing(30)
        relative_label = QLabel("Relative:")
        self.relative_tf = constructors.TrueFalseSelector(True)
        relative_layout.addWidget(relative_label)
        relative_layout.addWidget(self.relative_tf)

        collapsable_layout.addLayout(stimulus_layout)
        collapsable_layout.addLayout(control_layout)
        collapsable_layout.addLayout(relative_layout)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "stimulus" : self.stimulus_tf.get_value(),
            "control" : self.control_tf.get_value(),
            "relative" : self.relative_tf.get_value(),
        }

class analysis_display_params_pop_summary_seq_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Pop Summary Sequence", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        stimulus_layout = QHBoxLayout()
        stimulus_layout.setSpacing(30)
        stimulus_label = QLabel("Stimulus:")
        self.stimulus_tf = constructors.TrueFalseSelector(False)
        stimulus_layout.addWidget(stimulus_label)
        stimulus_layout.addWidget(self.stimulus_tf)

        relative_layout = QHBoxLayout()
        relative_layout.setSpacing(30)
        relative_label = QLabel("Relative:")
        self.relative_tf = constructors.TrueFalseSelector(False)
        relative_layout.addWidget(relative_label)
        relative_layout.addWidget(self.relative_tf)

        num_in_seq_layout = QHBoxLayout()
        num_in_seq_layout.setSpacing(30)
        num_in_seq_label = QLabel("Number in Sequence:")
        self.num_in_seq_value = constructors.FreetextBox("8")
        num_in_seq_layout.addWidget(num_in_seq_label)
        num_in_seq_layout.addWidget(self.num_in_seq_value)

        collapsable_layout.addLayout(stimulus_layout)
        collapsable_layout.addLayout(relative_layout)
        collapsable_layout.addLayout(num_in_seq_layout)

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
        return {
            "stimulus" : self.stimulus_tf.get_value(),
            "relative" : self.relative_tf.get_value(),
            "num_in_seq" : parse_value(self.num_in_seq_value.get_text()),
        }

class analysis_display_params_indiv_summary_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.collapsable = constructors.CollapsibleSection("Individual Summary", True)
        collapsable_layout = QVBoxLayout()
        collapsable_layout.setSpacing(30)

        overlap_layout = QHBoxLayout()
        overlap_layout.setSpacing(30)
        overlap_label = QLabel("Overlap:")
        self.overlap_tf = constructors.TrueFalseSelector(False)
        overlap_layout.addWidget(overlap_label)
        overlap_layout.addWidget(self.overlap_tf)

        histogram_layout = QHBoxLayout()
        histogram_layout.setSpacing(30)
        histogram_label = QLabel("Histogram:")
        self.histogram_tf = constructors.TrueFalseSelector(False)
        histogram_layout.addWidget(histogram_label)
        histogram_layout.addWidget(self.histogram_tf)

        cumulative_histogram_layout = QHBoxLayout()
        cumulative_histogram_layout.setSpacing(30)
        cumulative_histogram_label = QLabel("Cumulative Histogram:")
        self.cumulative_histogram_tf = constructors.TrueFalseSelector(False)
        cumulative_histogram_layout.addWidget(cumulative_histogram_label)
        cumulative_histogram_layout.addWidget(self.cumulative_histogram_tf)

        map_overlay_layout = QHBoxLayout()
        map_overlay_layout.setSpacing(30)
        map_overlay_label = QLabel("Map Overlay:")
        self.map_overlay_tf = constructors.TrueFalseSelector(False)
        map_overlay_layout.addWidget(map_overlay_label)
        map_overlay_layout.addWidget(self.map_overlay_tf)

        collapsable_layout.addLayout(overlap_layout)
        collapsable_layout.addLayout(histogram_layout)
        collapsable_layout.addLayout(cumulative_histogram_layout)
        collapsable_layout.addLayout(map_overlay_layout)

        self.collapsable.set_content_layout(collapsable_layout)
        layout.addWidget(self.collapsable)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "overlap" : self.overlap_tf.get_value(),
            "histogram" : self.histogram_tf.get_value(),
            "cumulative_histogram" : self.cumulative_histogram_tf.get_value(),
            "map_overlay" : self.map_overlay_tf.get_value(),
        }

from PySide6.QtWidgets import (QWidget, QVBoxLayout,
                               QLabel, QPushButton, QHBoxLayout,
                               QScrollArea, QMainWindow, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt
import json, constructors

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
    main_layout.addWidget(raw_layer_inst)

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

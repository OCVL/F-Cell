"""
config_layers.py - Holds configuration layers (simple or advanced)
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.ocvl.function.gui.gui_widgets import (
    AffineRigidSelector, AlignmentModalitySelector, CollapsibleSection,
    ColorMapSelector, DropdownMenu, FreetextBox, ListEditorWidget,
    OpenFolder, TrueFalseSelector,
)
from src.ocvl.function.gui.gui_dialogs import (
    FormatEditorWidget, GroupByFormatEditorWidget, SaveasExtensionsEditorWidget,
)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _parse_numeric(text):
    """Convert a string to float/int where possible, else return as-is."""
    try:
        return int(text) if '.' not in text else float(text)
    except (ValueError, TypeError):
        return text


def _row(label_text: str, widget, spacing: int = 30, align=Qt.AlignLeft) -> QHBoxLayout:
    """Build a label + widget row layout."""
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    layout.addWidget(QLabel(label_text))
    layout.addWidget(widget)
    layout.setAlignment(align)
    return layout


def _section(title: str, fields: list, expanded: bool = True) -> "SectionWidget":
    """
    Create a CollapsibleSection containing a vertical stack of field rows.

    Parameters
    ----------
    title  : section header text
    fields : list of (label_str, widget) tuples or bare QLayout/QWidget objects
    expanded : whether the checkbox starts enabled
    """
    collapsable = CollapsibleSection(title, expanded)
    inner = QVBoxLayout()
    inner.setSpacing(30)
    for item in fields:
        if isinstance(item, tuple):
            label_text, widget = item
            inner.addLayout(_row(label_text, widget))
        elif isinstance(item, QWidget):
            inner.addWidget(item)
        else:  # QLayout
            inner.addLayout(item)
    collapsable.set_content_layout(inner)

    wrapper = QWidget()
    wrapper_layout = QVBoxLayout(wrapper)
    wrapper_layout.addWidget(collapsable)
    wrapper.collapsable = collapsable   # expose for is_enabled()
    return wrapper


# ---------------------------------------------------------------------------
# version_layer / description_layer
# ---------------------------------------------------------------------------

class version_layer(QWidget):
    def __init__(self, default="0.2", parent=None):
        super().__init__(parent)
        self.version_value = FreetextBox(default)
        w = _section("Version", [("Version:", self.version_value)])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        return None if not self.collapsable.is_enabled() else self.version_value.get_text()


class description_layer(QWidget):
    def __init__(self, default="A cunningly created pipeline and analysis JSON.", parent=None):
        super().__init__(parent)
        self.description_value = FreetextBox(default)
        w = _section("Description", [("Description:", self.description_value)])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        return None if not self.collapsable.is_enabled() else self.description_value.get_text()


# ---------------------------------------------------------------------------
# raw_metadata_fieldstoload_layer
# ---------------------------------------------------------------------------

class raw_metadata_fieldstoload_layer(QWidget):
    def __init__(self, timestamp_default="Timestamp_us", stimulus_default="StimulusOn", parent=None):
        super().__init__(parent)
        self.timestamps_value = FreetextBox(timestamp_default)
        self.stimulus_value = FreetextBox(stimulus_default)

        w = _section("Fields to Load", [
            ("Timestamps:", self.timestamps_value),
            ("Stimulus Train:", self.stimulus_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "timestamps": self.timestamps_value.get_text(),
            "stimulus_train": self.stimulus_value.get_text(),
        }


# ---------------------------------------------------------------------------
# raw_metadata_layer
# ---------------------------------------------------------------------------

class raw_metadata_layer(QWidget):
    def __init__(self,
                 type_default="text_file",
                 format_default="{IDnum}_{Year:4}{Month:2}{Day:2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}.csv",
                 parent=None):
        super().__init__(parent)
        self.type_value = FreetextBox(type_default)
        self.metadata_format_value = FormatEditorWidget("Metadata Format:", format_default)
        self.fields_layer = raw_metadata_fieldstoload_layer()

        w = _section("Metadata", [
            ("Type:", self.type_value),
            ("Metadata Format:", self.metadata_format_value),
            self.fields_layer,
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "type": self.type_value.get_text(),
            "metadata_format": self.metadata_format_value.get_value(),
            "fields_to_load": self.fields_layer.get_value(),
        }


# ---------------------------------------------------------------------------
# raw_layer
# ---------------------------------------------------------------------------

class raw_layer(QWidget):
    def __init__(self,
                 video_format_default="{IDnum}_{Year4}{Month}{Day}_{VidNum}_{Modality}",
                 expanded=True,
                 parent=None):
        super().__init__(parent)
        self.video_format_value = FormatEditorWidget("Video Format:", video_format_default)
        self.raw_metadata_layer = raw_metadata_layer()

        w = _section("Raw", [
            ("Video Format:", self.video_format_value),
            self.raw_metadata_layer,
        ], expanded=expanded)
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "video_format": self.video_format_value.get_value(),
            "metadata": self.raw_metadata_layer.get_value(),
        }


# ---------------------------------------------------------------------------
# preanalysis_metadata_fieldstoload_layer
# ---------------------------------------------------------------------------

class preanalysis_metadata_fieldstoload_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.framestamps_value = FreetextBox("OriginalFrameNumber")

        w = _section("Fields to Load", [("Framestamps:", self.framestamps_value)])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {"framestamps": self.framestamps_value.get_text()}


# ---------------------------------------------------------------------------
# preanalysis_metadata_layer
# ---------------------------------------------------------------------------

class preanalysis_metadata_layer(QWidget):
    def __init__(self,
                 format_default="{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}{:.1}_extract_reg_cropped.csv",
                 parent=None):
        super().__init__(parent)
        self.type_value = FreetextBox("text_file")
        self.metadata_format_value = FormatEditorWidget("Metadata Format:", format_default)
        self.fields_layer = preanalysis_metadata_fieldstoload_layer()

        w = _section("Metadata", [
            ("Type:", self.type_value),
            ("Metadata Format:", self.metadata_format_value),
            self.fields_layer,
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "type": self.type_value.get_text(),
            "metadata_format": self.metadata_format_value.get_value(),
            "fields_to_load": self.fields_layer.get_value(),
        }


# ---------------------------------------------------------------------------
# preanalysis_pipeline_params_maskroi_layer
# ---------------------------------------------------------------------------

class preanalysis_pipeline_params_maskroi_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.r_value = FreetextBox("0")
        self.c_value = FreetextBox("0")
        self.width_value = FreetextBox("-1")
        self.height_value = FreetextBox("-1")

        w = _section("Mask ROI", [
            ("Starting Row:", self.r_value),
            ("Starting Column:", self.c_value),
            ("Width:", self.width_value),
            ("Height:", self.height_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "r": _parse_numeric(self.r_value.get_text()),
            "c": _parse_numeric(self.c_value.get_text()),
            "width": _parse_numeric(self.width_value.get_text()),
            "height": _parse_numeric(self.height_value.get_text()),
        }


# ---------------------------------------------------------------------------
# preanalysis_pipeline_params_trim_layer
# ---------------------------------------------------------------------------

class preanalysis_pipeline_params_trim_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_idx_value = FreetextBox("0.0")
        self.end_idx_value = FreetextBox("-1.0")

        w = _section("Trimming", [
            ("Start Index:", self.start_idx_value),
            ("End Index:", self.end_idx_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "start_idx": _parse_numeric(self.start_idx_value.get_text()),
            "end_idx": _parse_numeric(self.end_idx_value.get_text()),
        }


# ---------------------------------------------------------------------------
# preanalysis_pipeline_params_layer
# ---------------------------------------------------------------------------

class preanalysis_pipeline_params_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gausblur_value = FreetextBox("0.0")
        self.modalities_list_creator = ListEditorWidget()
        self.alignment_ref_value = AlignmentModalitySelector(
            self.modalities_list_creator, "null"
        )
        self.output_folder_value = OpenFolder()
        self.groupby_value = GroupByFormatEditorWidget(
            image_format="", video_format="", mask_format="",
            label_text="Group By:", default_format="null",
        )
        self.correct_torsion_tf = TrueFalseSelector(True)
        self.intra_stack_xform_tf = AffineRigidSelector(True)
        self.inter_stack_xform_tf = AffineRigidSelector(True)
        self.flat_field_tf = TrueFalseSelector(False)
        self.maskroi_layer = preanalysis_pipeline_params_maskroi_layer()
        self.trim_layer = preanalysis_pipeline_params_trim_layer()

        w = _section("Pipeline Params", [
            ("Gaus Blur:", self.gausblur_value),
            self.maskroi_layer,
            ("Modalities:", self.modalities_list_creator),
            ("Alignment Reference Modality:", self.alignment_ref_value),
            ("Output Folder:", self.output_folder_value),
            ("Group By:", self.groupby_value),
            ("Correct Torsion:", self.correct_torsion_tf),
            ("Intra Stack Xform:", self.intra_stack_xform_tf),
            ("Inter Stack Xform:", self.inter_stack_xform_tf),
            ("Flat Field:", self.flat_field_tf),
            self.trim_layer,
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def update_format_references(self, image_format, video_format, mask_format):
        self.groupby_value.image_format = image_format
        self.groupby_value.video_format = video_format
        self.groupby_value.mask_format = mask_format
        self.groupby_value.available_elements = self.groupby_value.get_available_elements()

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        gb = self.groupby_value.get_value()
        return {
            "gaus_blur": _parse_numeric(self.gausblur_value.get_text()),
            "mask_roi": self.maskroi_layer.get_value(),
            "modalities": self.modalities_list_creator.get_list(),
            "alignment_reference_modality": None if self.alignment_ref_value.get_value() == "null"
                                            else self.alignment_ref_value.get_value(),
            "output_folder": self.output_folder_value.get_text(),
            "group_by": None if gb == "null" else gb,
            "correct_torsion": self.correct_torsion_tf.get_value(),
            "intra_stack_xform": self.intra_stack_xform_tf.get_value(),
            "inter_stack_xform": self.inter_stack_xform_tf.get_value(),
            "flat_field": self.flat_field_tf.get_value(),
            "trim": self.trim_layer.get_value(),
        }


# ---------------------------------------------------------------------------
# preanalysis_layer
# ---------------------------------------------------------------------------

class preanalysis_layer(QWidget):
    def __init__(self,
                 image_fmt="{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}{:.1}_extract_reg_avg.tif",
                 video_fmt="{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}{:.1}_extract_reg_cropped.avi",
                 mask_fmt="{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}{:.1}_extract_reg_cropped_mask.avi",
                 parent=None):
        super().__init__(parent)
        self.image_format_value = FormatEditorWidget("Image Format:", image_fmt)
        self.video_format_value = FormatEditorWidget("Video Format:", video_fmt)
        self.mask_format_value = FormatEditorWidget("Mask Format:", mask_fmt)
        self.recursive_search_tf = TrueFalseSelector(False)
        self.preanalysis_metadata_layer = preanalysis_metadata_layer()
        self.preanalysis_pipeline_params_layer = preanalysis_pipeline_params_layer()

        for fmt_widget in (self.image_format_value, self.video_format_value, self.mask_format_value):
            fmt_widget.formatChanged.connect(self._update_groupby_elements)

        w = _section("preanalysis", [
            ("Image Format:", self.image_format_value),
            ("Video Format:", self.video_format_value),
            ("Mask Format:", self.mask_format_value),
            self.preanalysis_metadata_layer,
            ("Recursive Search:", self.recursive_search_tf),
            self.preanalysis_pipeline_params_layer,
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def _update_groupby_elements(self):
        self.preanalysis_pipeline_params_layer.update_format_references(
            image_format=self.image_format_value.get_value(),
            video_format=self.video_format_value.get_value(),
            mask_format=self.mask_format_value.get_value(),
        )

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "image_format": self.image_format_value.get_value(),
            "video_format": self.video_format_value.get_value(),
            "mask_format": self.mask_format_value.get_value(),
            "metadata": self.preanalysis_metadata_layer.get_value(),
            "recursive_search": self.recursive_search_tf.get_value(),
            "pipeline_params": self.preanalysis_pipeline_params_layer.get_value(),
        }


# ---------------------------------------------------------------------------
# analysis sub-layers
# ---------------------------------------------------------------------------

class raw_analysis_fieldstoload_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.framestamps_value = FreetextBox("FrameStamps")
        self.stimsequences_value = FreetextBox("StimulusOn")

        w = _section("Fields to Load", [
            ("Framestamps:", self.framestamps_value),
            ("Stimulus Sequence:", self.stimsequences_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "framestamps": self.framestamps_value.get_text(),
            "stimulus_sequence": self.stimsequences_value.get_text(),
        }


class analysis_metadata_layer(QWidget):
    def __init__(self,
                 format_default="{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}_piped.csv",
                 parent=None):
        super().__init__(parent)
        self.type_value = FreetextBox("text_file")
        self.metadata_format_value = FormatEditorWidget("Metadata Format:", format_default)
        self.fields_layer = raw_analysis_fieldstoload_layer()

        w = _section("Metadata", [
            ("Type:", self.type_value),
            ("Metadata Format:", self.metadata_format_value),
            self.fields_layer,
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "type": self.type_value.get_text(),
            "metadata_format": self.metadata_format_value.get_value(),
            "fields_to_load": self.fields_layer.get_value(),
        }


class analysis_control_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.location_value = FreetextBox("folder")
        self.folder_name_value = FreetextBox("control")

        w = _section("Control", [
            ("Location:", self.location_value),
            ("Folder Name:", self.folder_name_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "location": self.location_value.get_text(),
            "folder_name": self.folder_name_value.get_text(),
        }


class analysis_analysis_params_normalization_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.method_value = DropdownMenu("score", ["mean", "median", "none"])
        self.rescaled_tf = TrueFalseSelector(True)
        self.rescaled_mean_value = FreetextBox("70")
        self.rescaled_std_value = FreetextBox("35")

        w = _section("Normalization", [
            ("Method:", self.method_value),
            ("Rescaled:", self.rescaled_tf),
            ("Rescaled Mean:", self.rescaled_mean_value),
            ("Rescaled STD:", self.rescaled_std_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "method": self.method_value.get_value(),
            "rescaled": self.rescaled_tf.get_value(),
            "rescale_mean": _parse_numeric(self.rescaled_mean_value.get_text()),
            "rescale_std": _parse_numeric(self.rescaled_std_value.get_text()),
        }


class analysis_analysis_params_segmentation_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.refine_to_ref_tf = TrueFalseSelector(False)
        self.refine_to_vid_tf = TrueFalseSelector(False)
        self.radius_value = FreetextBox("auto")
        self.shape_value = DropdownMenu("disk", "box")
        self.summary_value = DropdownMenu("mean", "median")

        w = _section("Segmentation", [
            ("Refine to Ref:", self.refine_to_ref_tf),
            ("Refine to Vid:", self.refine_to_vid_tf),
            ("Radius:", self.radius_value),
            ("Shape:", self.shape_value),
            ("Summary:", self.summary_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "refine_to_ref": self.refine_to_ref_tf.get_value(),
            "refine_to_vid": self.refine_to_vid_tf.get_value(),
            "radius": _parse_numeric(self.radius_value.get_text()),
            "shape": self.shape_value.get_value(),
            "summary": self.summary_value.get_value(),
        }


class analysis_analysis_params_exclusion_criteria_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.type_value = DropdownMenu("stim-relative", "absolute")
        self.units_value = DropdownMenu("time", "frames")
        self.start_value = FreetextBox("-1")
        self.stop_value = FreetextBox("0")
        self.fraction_value = FreetextBox("0.3")

        w = _section("Exclusion Criteria", [
            ("Type:", self.type_value),
            ("Units:", self.units_value),
            ("Start:", self.start_value),
            ("Stop:", self.stop_value),
            ("Fraction:", self.fraction_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "type": self.type_value.get_value(),
            "units": self.units_value.get_value(),
            "start": _parse_numeric(self.start_value.get_text()),
            "stop": _parse_numeric(self.stop_value.get_text()),
            "fraction": _parse_numeric(self.fraction_value.get_text()),
        }


class analysis_analysis_params_standardization_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.method_value = DropdownMenu("mean_sub", ["std", "linear_std", "linear_vast", "relative_change", "none"])
        self.type_value = DropdownMenu("stim-relative", "absolute")
        self.units_value = DropdownMenu("time", "frames")
        self.start_value = FreetextBox("-1")
        self.stop_value = FreetextBox("0")

        w = _section("Standardization", [
            ("Method:", self.method_value),
            ("Type:", self.type_value),
            ("Units:", self.units_value),
            ("Start:", self.start_value),
            ("Stop:", self.stop_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "method": self.method_value.get_value(),
            "type": self.type_value.get_value(),
            "units": self.units_value.get_value(),
            "start": _parse_numeric(self.start_value.get_text()),
            "stop": _parse_numeric(self.stop_value.get_text()),
        }


class analysis_analysis_params_summary_metrics_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.type_list_creator = ListEditorWidget()
        self.measured_value = FreetextBox("stim-relative")
        self.units_value = FreetextBox("time")
        self.prestim_value = FreetextBox("-1, 0")
        self.poststim_value = FreetextBox("0, 1")

        w = _section("Metrics", [
            ("Type:", self.type_list_creator),
            ("Measured:", self.measured_value),
            ("Units:", self.units_value),
            ("Prestim:", self.prestim_value),
            ("Poststim:", self.poststim_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "type": self.type_list_creator.get_list(),
            "measured": self.measured_value.get_text(),
            "units": self.units_value.get_text(),
            "prestim": self.prestim_value.get_text(),
            "poststim": self.poststim_value.get_text(),
        }


class analysis_analysis_params_summary_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.method_value = DropdownMenu("rms", ["std", "var", "avg"])
        self.windowsize_value = FreetextBox("1")
        self.control_value = DropdownMenu("none", ["subtraction", "division"])
        self.indiv_cutoff_value = FreetextBox("5")
        self.metrics_layer = analysis_analysis_params_summary_metrics_layer()

        w = _section("Summary", [
            ("Method:", self.method_value),
            ("Window Size:", self.windowsize_value),
            ("Control:", self.control_value),
            self.metrics_layer,
            ("Individual Cutoff:", self.indiv_cutoff_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "method": self.method_value.get_value(),
            "window_size": _parse_numeric(self.windowsize_value.get_text()),
            "control": self.control_value.get_value(),
            "metrics": self.metrics_layer.get_value(),
            "indiv_cutoff": _parse_numeric(self.indiv_cutoff_value.get_text()),
        }


class analysis_analysis_params_maskroi_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.r_value = FreetextBox("0")
        self.c_value = FreetextBox("0")
        self.width_value = FreetextBox("-1")
        self.height_value = FreetextBox("-1")

        w = _section("Mask ROI", [
            ("Starting Row:", self.r_value),
            ("Starting Column:", self.c_value),
            ("Width:", self.width_value),
            ("Height:", self.height_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "r": _parse_numeric(self.r_value.get_text()),
            "c": _parse_numeric(self.c_value.get_text()),
            "width": _parse_numeric(self.width_value.get_text()),
            "height": _parse_numeric(self.height_value.get_text()),
        }


class analysis_analysis_params_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.modalities_list_creator = ListEditorWidget()
        self.output_folder_value = OpenFolder()
        self.gausblur_value = FreetextBox("0.0")
        self.normalization_layer = analysis_analysis_params_normalization_layer()
        self.segmentation_layer = analysis_analysis_params_segmentation_layer()
        self.exclusion_criteria_layer = analysis_analysis_params_exclusion_criteria_layer()
        self.standardization_layer = analysis_analysis_params_standardization_layer()
        self.summary_layer = analysis_analysis_params_summary_layer()
        self.maskroi_layer = analysis_analysis_params_maskroi_layer()

        w = _section("Analysis Params", [
            ("Modalities:", self.modalities_list_creator),
            ("Output Folder:", self.output_folder_value),
            self.normalization_layer,
            self.segmentation_layer,
            self.exclusion_criteria_layer,
            self.standardization_layer,
            self.summary_layer,
            ("Gaus Blur:", self.gausblur_value),
            self.maskroi_layer,
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "modalities": self.modalities_list_creator.get_list(),
            "output_folder": self.output_folder_value.get_text(),
            "normalization": self.normalization_layer.get_value(),
            "segmentation": self.segmentation_layer.get_value(),
            "exclusion_criteria": self.exclusion_criteria_layer.get_value(),
            "standardization": self.standardization_layer.get_value(),
            "summary": self.summary_layer.get_value(),
            "gaus_blur": _parse_numeric(self.gausblur_value.get_text()),
            "mask_roi": self.maskroi_layer.get_value(),
        }


# display params sub-layers
class analysis_display_params_pop_summary_seq_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stimulus_tf = TrueFalseSelector(False)
        self.relative_tf = TrueFalseSelector(False)
        self.num_in_seq_value = FreetextBox("8")

        w = _section("Pop Summary Sequence", [
            ("Stimulus:", self.stimulus_tf),
            ("Relative:", self.relative_tf),
            ("Number in Sequence:", self.num_in_seq_value),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "stimulus": self.stimulus_tf.get_value(),
            "relative": self.relative_tf.get_value(),
            "num_in_seq": _parse_numeric(self.num_in_seq_value.get_text()),
        }


class analysis_display_params_indiv_summary_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.overlap_tf = TrueFalseSelector(False)
        self.histogram_tf = TrueFalseSelector(False)
        self.cumulative_histogram_tf = TrueFalseSelector(False)
        self.map_overlay_tf = TrueFalseSelector(False)

        w = _section("Individual Summary", [
            ("Overlap:", self.overlap_tf),
            ("Histogram:", self.histogram_tf),
            ("Cumulative Histogram:", self.cumulative_histogram_tf),
            ("Map Overlay:", self.map_overlay_tf),
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "overlap": self.overlap_tf.get_value(),
            "histogram": self.histogram_tf.get_value(),
            "cumulative_histogram": self.cumulative_histogram_tf.get_value(),
            "map_overlay": self.map_overlay_tf.get_value(),
        }


class analysis_display_params_layer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmap_selector = ColorMapSelector()
        self.save_as_value = SaveasExtensionsEditorWidget("Save as")
        self.pop_summary_seq_layer = analysis_display_params_pop_summary_seq_layer()
        self.indiv_summary_layer = analysis_display_params_indiv_summary_layer()

        w = _section("Display Params", [
            ("Color Map:", self.cmap_selector),
            ("Save As:", self.save_as_value),
            self.pop_summary_seq_layer,
            self.indiv_summary_layer,
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "cmap": self.cmap_selector.get_value(),
            "save_as": self.save_as_value.get_value(),
            "pop_summary_seq": self.pop_summary_seq_layer.get_value(),
            "indiv_summary": self.indiv_summary_layer.get_value(),
        }


# ---------------------------------------------------------------------------
# analysis_layer
# ---------------------------------------------------------------------------

class analysis_layer(QWidget):
    def __init__(self,
                 image_fmt="{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}_ALL_ACQ_AVG.tif",
                 queryloc_fmt="{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}_ALL_ACQ_AVG_{QueryLoc:s?}coords.csv",
                 video_fmt="{IDnum}_{Year:.4}{Month:.2}{Day:.2}_{Eye}_({LocX},{LocY})_{FOV_Width}x{FOV_Height}_{VidNum}_{Modality}_ALL_ACQ_AVG.avi",
                 parent=None):
        super().__init__(parent)
        self.image_format_value = FormatEditorWidget("Image Format:", image_fmt)
        self.queryloc_format_value = FormatEditorWidget("Query Loc Format:", queryloc_fmt)
        self.video_format_value = FormatEditorWidget("Video Format:", video_fmt)
        self.recursive_search_tf = TrueFalseSelector(False)
        self.analysis_metadata_layer = analysis_metadata_layer()
        self.analysis_control_layer = analysis_control_layer()
        self.analysis_analysis_params_layer = analysis_analysis_params_layer()
        self.analysis_display_params_layer = analysis_display_params_layer()

        w = _section("analysis", [
            ("Image Format:", self.image_format_value),
            ("Query Loc Format:", self.queryloc_format_value),
            ("Video Format:", self.video_format_value),
            self.analysis_metadata_layer,
            ("Recursive Search:", self.recursive_search_tf),
            self.analysis_control_layer,
            self.analysis_analysis_params_layer,
            self.analysis_display_params_layer,
        ])
        self.collapsable = w.collapsable
        QVBoxLayout(self).addWidget(w)

    def get_value(self):
        if not self.collapsable.is_enabled():
            return None
        return {
            "image_format": self.image_format_value.get_value(),
            "queryloc_format": self.queryloc_format_value.get_value(),
            "video_format": self.video_format_value.get_value(),
            "metadata": self.analysis_metadata_layer.get_value(),
            "recursive_search": self.recursive_search_tf.get_value(),
            "control": self.analysis_control_layer.get_value(),
            "analysis_params": self.analysis_analysis_params_layer.get_value(),
            "display_params": self.analysis_display_params_layer.get_value(),
        }
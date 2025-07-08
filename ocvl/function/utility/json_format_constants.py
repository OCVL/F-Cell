from enum import StrEnum

class ConfigFields(StrEnum):
    VERSION = "version",
    DESCRIPTION = "description"

class DataFormatType(StrEnum):
    FORMAT_TYPE = "Data_Type"
    IMAGE = "image_format"
    VIDEO = "video_format"
    MASK = "mask_format"
    METADATA = "metadata_format"
    QUERYLOC = "queryloc_format"
    RECURSIVE = "recursive_search"

# Will probably want to make a list of these corresponding to the version of the json files.
# This verson assumes that we don't have any naming collisions; e.g. these constants aren't used in the filenames.
class DataTags(StrEnum):
    DATA_ID = "IDnum"
    VIDEO_ID = "VidNum"
    YEAR = "Year" # YYYY
    MONTH = "Month" # MM
    DAY = "Day" # DD
    HOUR = "Hour"
    MINUTE = "Minute"
    SECOND = "Second"
    EYE = "Eye"
    RET_LOC_X = "LocX"
    RET_LOC_Y = "LocY"
    RET_LOC_Z = "LocZ"
    FOV_WIDTH = "FOV_Width"
    FOV_HEIGHT = "FOV_Height"
    FOV_DEPTH = "FOV_Depth"
    MODALITY = "Modality"
    QUERYLOC = "QueryLoc"

class MetaTags(StrEnum):
    METATAG = "metadata"
    TYPE = "type"
    FIELDS_OF_INTEREST = "fields_to_load"
    TIMESTAMPS = "timestamps"
    FRAMESTAMPS = "framestamps"
    STIMULUS_SEQ = "stimulus_sequence"
    FRAMERATE = "framerate"
    QUERY_LOCATIONS = "query_locations"
    VIDEO = "video"
    MASK = "mask"


class AcquisiTags(StrEnum):
    DATASET = "Dataset"
    DATA_PATH = "Data_Path"
    OUTPUT_PATH = "Output_Path"
    VIDEO_PATH = "Video_Path"
    IMAGE_PATH = "Image_Path"
    QUERYLOC_PATH = "QueryLocation_Path"
    STIMSEQ_PATH = "Stimulus_Sequence_Path"
    MODALITY = "Modality"
    PREFIX = "Output_Prefix"
    BASE_PATH = "Base_Path"
    MASK_PATH = "Mask_Path"
    META_PATH = "Metadata_Path"
    STIM_PRESENT = "Stimulus_Present"

class PreAnalysisPipeline(StrEnum):
    NAME = "preanalysis"
    PARAMS = "pipeline_params"
    GAUSSIAN_BLUR = "gaus_blur"
    MASK_ROI = "mask_roi"
    TRIM = "trim"
    FLAT_FIELD = "flat_field"
    MODALITIES = "modalities"
    ALIGNMENT_REF_MODE = "alignment_reference_modality"
    CORRECT_TORSION = "correct_torsion"
    CUSTOM = "custom"
    OUTPUT_FOLDER = "output_folder"
    GROUP_BY = "group_by"
    INTRA_STACK_XFORM = "intra_stack_xform"
    INTER_STACK_XFORM = "inter_stack_xform"

class Analysis(StrEnum):
    NAME = "analysis"
    PARAMS = "analysis_params"
    FLAT_FIELD = "flat_field"
    GAUSSIAN_BLUR = "gaus_blur"
    OUTPUT_FOLDER = "output_folder"
    OUTPUT_SUBFOLDER = "output_subfolder"
    OUTPUT_SUBFOLDER_METHOD = "output_subfolder_method"
    MODALITIES = "modalities"

class DebugParams(StrEnum):
    NAME = "debug"
    OUTPUT_NORM_VIDEO = "output_norm_video"
    PLOT_REFINE_TO_REF = "plot_refine_to_ref"
    PLOT_REFINE_TO_VID = "plot_refine_to_vid"
    PLOT_POP_EXTRACTED_ORGS = "plot_pop_extracted_orgs"
    PLOT_POP_STANDARDIZED_ORGS = "plot_pop_stdize_orgs"
    PLOT_INDIV_STANDARDIZED_ORGS = "plot_indiv_stdize_orgs"
    OUTPUT_INDIV_STANDARDIZED_ORGS = "output_indiv_stdize_orgs"
    OUTPUT_SUMPOP_ORGS = "output_sum_pop_orgs"
    CELL_VIABILITY = "cell_viability"

class NormParams(StrEnum):
    NAME = "normalization"
    NORM_METHOD = "method"
    NORM_RESCALE = "rescaled"
    NORM_MEAN = "rescale_mean"
    NORM_STD = "rescale_std"

class SegmentParams(StrEnum):
    NAME = "segmentation"
    REFINE_TO_REF = "refine_to_ref"
    REFINE_TO_VID = "refine_to_vid"
    RADIUS = "radius"
    SHAPE = "shape"
    SUMMARY = "mean"
    PIXELWISE = "pixelwise"

class ExclusionParams(StrEnum):
    NAME = "exclusion_criteria"
    TYPE = "type"
    UNITS = "units"
    START = "start"
    STOP = "stop"
    FRACTION = "fraction"

class STDParams(StrEnum):
    NAME = "standardization"
    METHOD = "method"
    TYPE = "type"
    UNITS = "units"
    START = "start"
    STOP = "stop"

class SummaryParams(StrEnum):
    NAME = "summary"
    METHOD = "method"
    WINDOW_SIZE = "window_size"
    CONTROL = "control"
    METRICS = "metrics"
    TYPE = "type"
    MEASURED_TO = "measured"
    UNITS = "units"
    PRESTIM = "prestim"
    POSTSTIM = "poststim"
    INDIV_CUTOFF = "indiv_cutoff"

class ControlParams(StrEnum):
    NAME = "control"
    LOCATION = "location"
    FOLDER_NAME = "folder_name"

class MetricTags(StrEnum):
    AUR = "Response Area Under Curve"
    LOG_AMPLITUDE = "Log Amplitude"
    AMPLITUDE = "Amplitude"
    AMP_IMPLICIT_TIME = "Amplitude Implicit Time"
    HALFAMP_IMPLICIT_TIME = "Half Amplitude Implicit Time"
    RECOVERY_PERCENT = "Recovery Fraction"

class DisplayParams(StrEnum):
    NAME = "display_params"
    POP_SUMMARY_OVERLAP = "pop_summary_overlap"
    POP_SUMMARY_SEQ = "pop_summary_seq"
    POP_SUMMARY_METRICS = "pop_summary_metrics"
    INDIV_SUMMARY_OVERLAP = "indiv_summary_overlap"
    INDIV_SUMMARY = "indiv_summary"
    HISTOGRAM = "histogram"
    CUMULATIVE_HISTOGRAM = "cumulative_histogram"
    MAP_OVERLAY = "map_overlay"
    ORG_VIDEO = "org_video"
    DISP_STIMULUS = "stimulus"
    DISP_CONTROL = "control"
    DISP_RELATIVE = "relative"
    DISP_POOLED = "pooled"
    SAVEAS = "saveas"
    PAUSE_PER_FOLDER = "pause_per_folder"
    NUM_IN_SEQ = "num_in_seq"
    AXES = "axes"
    XMIN = "xmin"
    XSTEP = "xstep"
    XMAX = "xmax"
    NBINS = "nbins"
    YMIN = "ymin"
    YMAX = "ymax"
    CMAP = "cmap"
    CMIN = "cmin"
    CMAX = "cmax"
    LEGEND = "legend"

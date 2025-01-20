from enum import StrEnum


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

class AcquisiTags(StrEnum):
    DATASET = "Dataset"
    DATA_PATH = "Data_Path"
    OUTPUT_PATH = "Output_Path",
    VIDEO_PATH = "Video_Path",
    IMAGE_PATH = "Image_Path",
    QUERYLOC_PATH = "QueryLocation_Path",
    STIMSEQ_PATH = "Stimulus_Sequence_Path",
    MODALITY = "Modality",
    PREFIX = "Output_Prefix",
    BASE_PATH = "Base_Path",
    MASK_PATH = "Mask_Path",
    META_PATH = "Metadata_Path",
    STIM_PRESENT = "Stimulus_Present"

class PipelineParams(StrEnum):
    GAUSSIAN_BLUR = "gaus_blur",
    MASK_ROI = "mask_roi",
    TRIM = "trim",
    MODALITIES = "modalities",
    CORRECT_TORSION = "correct_torsion",
    CUSTOM = "custom"
    OUTPUT_FOLDER = "output_folder"
    GROUP_BY = "group_by"

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
    RADIUS = "auto"
    SHAPE = "disk"
    SUMMARY = "mean"

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


class ControlParams(StrEnum):
    NAME = "control"
    LOCATION = "location"
    FOLDER_NAME = "folder_name"

class MetricTags(StrEnum):
    AUR = "Response Area Under Curve"
    AMPLITUDE = "Amplitude"
    IMPLICT_TIME = "Implict Time"
    RECOVERY_PERCENT = "Recovery Percent"
    RECOVERY = "Recovery"

class DisplayParams(StrEnum):
    NAME = "display_params"
    POP_SUMMARY_OVERLAP = "pop_summary_overlap"
    POP_SUMMARY_SEQ = "pop_summary_seq"
    POP_SUMMARY_METRICS = "pop_summary_metrics"
    INDIV_SUMMARY = "indiv_summary"
    DISP_STIMULUS = "stimulus"
    DISP_CONTROL = "control"
    DISP_RELATIVE = "relative"
    SAVEAS = "saveas"
    PAUSE_PER_FOLDER = "pause_per_folder"
    NUM_IN_SEQ = "num_in_seq"

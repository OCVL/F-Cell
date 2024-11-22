from enum import StrEnum


class FormatTypes(StrEnum):
    FORMAT = "format"
    IMAGE = "image_format"
    VIDEO = "video_format"
    MASK = "mask_format"
    METADATA = "metadata_format"
    QUERYLOC = "queryloc_format"

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
    FORMAT_TYPE = "FormatType"

class MetaTags(StrEnum):
    METATAG = "metadata"
    TYPE = "type"
    FIELDS_OF_INTEREST = "fields_to_load"
    TIMESTAMPS = "timestamps"
    FRAMESTAMPS = "framestamps"
    STIMULUS_SEQ = "stimulus_sequence"
    FRAMERATE = "framerate"
    QUERY_LOC = "query_locations"

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

class PipelineParams(StrEnum):
    GAUSSIAN_BLUR = "gaus_blur",
    MASK_ROI = "mask_roi",
    TRIM = "trim",
    MODALITIES = "modalities",
    CORRECT_TORSION = "correct_torsion",
    CUSTOM = "custom"
    OUTPUT_FOLDER = "output_folder"
    GROUP_BY = "group_by"


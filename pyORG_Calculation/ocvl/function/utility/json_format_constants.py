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


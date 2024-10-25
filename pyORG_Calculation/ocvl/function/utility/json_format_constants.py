from enum import Enum


# Will probably want to make a list of these corresponding to the version of the json files.
# This verson assumes that we don't have any naming collisions; e.g. these constants aren't used in the filenames.
class DataFormat(Enum):
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
    FOV = "FOV"
    MODALITY = "Modality"

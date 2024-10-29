import glob
import os
import warnings
from enum import Enum, StrEnum
from logging import warning

import cv2
import numpy as np
import pandas as pd

from ocvl.function.preprocessing.improc import optimizer_stack_align
from ocvl.function.utility.json_format_constants import DataTags, MetaTags
from ocvl.function.utility.resources import load_video, save_video


class PipeStages(Enum):
    RAW = 0,
    PROCESSED = 1,
    PIPELINED = 2,
    ANALYSIS_READY = 3

class AcquisiTags(StrEnum):
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


def initialize_and_load_dataset(video_path, mask_path=None, extra_metadata_path=None, dataset_metadata=None):

    mask_data = None
    metadata = dataset_metadata
    metadata[AcquisiTags.VIDEO_PATH] = dataset_metadata[AcquisiTags.DATA_PATH]
    metadata[AcquisiTags.MASK_PATH] = mask_path

    if video_path.exists():
        resource = load_video(video_path)
        video_data = resource.data
        metadata[MetaTags.FRAMERATE] = resource.metadict.get("framerate")
    else:
        warning("Video path does not exist at: "+str(video_path))
        return None

    if mask_path:
        if mask_path.exists():
            mask_res = load_video(mask_path)
            mask_data = mask_res.data / mask_res.data.max()
            mask_data[mask_data < 0] = 0
            mask_data[mask_data > 1] = 1
            # Mask our video data correspondingly.
            video_data = (video_data * mask_data)
        else:
            warning("Mask path does not exist at: "+str(mask_path))

    if extra_metadata_path:
        extra_metadata = pd.read_csv(extra_metadata_path, encoding="utf-8-sig")


    return Dataset(video_data, mask_data, metadata)

class Dataset:
    def __init__(self, video_data=None, mask_data=None, framestamps=None, query_locations=None,
                 stimseq=None, metadata=None, stage=PipeStages.PROCESSED):

        # Paths to the data used here.
        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

        # Information about the dataset
        self.stage = stage
        self.framerate = -1
        self.num_frames = -1
        self.width = -1
        self.height = -1
        self.framestamps = framestamps
        self.reference_frame_idx = []
        self.stimtrain_frame_stamps = stimseq

        # The data are roughly grouped by the following:
        # Base data
        self.coord_data = query_locations
        self.z_proj_image_data = np.empty([1])
        # Video data (processed or pipelined)
        self.video_data = video_data
        self.mask_data = mask_data

        if video_data:
            self.num_frames = video_data.shape[-1]
            self.width = video_data.shape[1]
            self.height = video_data.shape[0]

        self.framerate = self.metadata[MetaTags.FRAMERATE]

        self.stimtrain_path = self.metadata[AcquisiTags.STIMSEQ_PATH]
        self.video_path = self.metadata[AcquisiTags.VIDEO_PATH]
        self.mask_path = self.metadata.get(AcquisiTags.MASK_PATH)
        self.base_path = self.metadata[AcquisiTags.BASE_PATH]

        self.prefix = self.metadata[AcquisiTags.PREFIX]

        if self.video_path:
            # If we don't have supplied definitions of the base path of the dataset or the filename prefix,
            # then guess.
            if not self.base_path:
                self.base_path = os.path.dirname(os.path.realpath(self.video_path))
            if not self.prefix:
                self.prefix = os.path.basename(os.path.realpath(self.video_path))[0:-4]

            self.image_path = self.metadata[AcquisiTags.IMAGE_PATH]
            # If we don't have supplied definitions of the image associated with this dataset,
            # then guess.
            if self.image_path is None:
                imname = None
                if stage is PipeStages.PROCESSED:
                    for filename in glob.glob(os.path.join(self.base_path, self.prefix + ".tif")):
                        imname = filename
                elif stage is PipeStages.PIPELINED:
                    # First look for an image associated with this dataset
                    for filename in glob.glob(os.path.join(self.base_path, self.prefix + ".tif")):
                        imname = filename
                    # If we don't have an image specific to this dataset, search for the all acq avg from our pipeline script
                    if not imname:
                        for filename in glob.glob(os.path.join(self.base_path, "*_ALL_ACQ_AVG.tif")):
                            # print(filename)
                            imname = filename
                else:
                    imname = self.video_path[0:-3] + ".tif"

                if not imname:
                    warnings.warn("Unable to detect viable average image file. Dataset functionality may be limited.")
                    self.image_path = None
                else:
                    self.image_path = os.path.join(self.base_path, imname)

            self.coord_path = self.metadata[AcquisiTags.QUERYLOC_PATH]
            # If we don't have query locations associated with this dataset, then try and find them out.
            if not self.coord_path:
                coordname = None
                if stage is PipeStages.PROCESSED:
                    for filename in glob.glob(os.path.join(self.base_path, self.prefix + "_coords.csv")):
                        coordname = filename
                elif stage is PipeStages.PIPELINED:
                    # First look for an image associated with this dataset
                    for filename in glob.glob(os.path.join(self.base_path, self.prefix + "_coords.csv")):
                        coordname = filename

                    # If we don't have an image specific to this dataset, search for the all acq avg
                    if not coordname:
                        for filename in glob.glob(os.path.join(self.base_path, "*_ALL_ACQ_AVG_coords.csv")):
                            coordname = filename

                    if not coordname:
                        warnings.warn("Unable to detect viable coordinate file for pipelined dataset at: "+ self.base_path)
                else:
                    coordname = self.prefix  + "_coords.csv"

                self.coord_path = os.path.join(self.base_path, coordname)

    def clear_video_data(self):
        del self.video_data
        del self.mask_data

    def load_data(self, force_reload=False):

        # Go down the line, loading data that doesn't already exist in this dataset.
        if (not self.video_data or force_reload) and os.path.exists(self.video_path):
            resource = load_video(self.video_path)

            self.video_data = resource.data

            self.framerate = resource.metadict["framerate"]
            self.metadata_data = resource.metadict
            self.width = resource.data.shape[1]
            self.height = resource.data.shape[0]
            self.num_frames = resource.data.shape[-1]

        if (not self.mask_data or force_reload) and os.path.exists(self.mask_path):
            mask_res = load_video(self.mask_path)
            self.mask_data = mask_res.data / mask_res.data.max()
            self.mask_data[self.mask_data < 0] = 0
            self.mask_data[self.mask_data > 1] = 1
            # Mask our video data correspondingly.
            self.video_data = (self.video_data * self.mask_data)

        if (not self.coord_data or force_reload) and os.path.exists(self.coord_path):
            self.coord_data = pd.read_csv(self.coord_path, delimiter=',', header=None,
                                          encoding="utf-8-sig").to_numpy()

        if (not self.z_proj_image_data or force_reload) and os.path.exists(self.image_path) :
            self.z_proj_image_data = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if (not self.stimtrain_frame_stamps or force_reload) and os.path.exists(self.stimtrain_path):
            self.stimtrain_frame_stamps = np.cumsum(np.squeeze(pd.read_csv(self.stimtrain_path, delimiter=',', header=None,
                                                                           encoding="utf-8-sig").to_numpy()))
        else:
            self.stimtrain_frame_stamps = 0

    def save_data(self, suffix):
        save_video(self.video_path[0:-4]+suffix+".avi", self.video_data, self.framerate)




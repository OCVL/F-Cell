import glob
import os
import warnings
from enum import Enum

import cv2
import numpy as np
import pandas as pd

from ocvl.function.preprocessing.improc import optimizer_stack_align
from ocvl.function.utility.resources import load_video, save_video


class PipeStages(Enum):
    RAW = 0,
    PROCESSED = 1,
    PIPELINED = 2,
    ANALYSIS_READY = 3

class Metadata(Enum):
    OUTPUT_PATH = 0,
    VIDEO_PATH = 1,
    IMAGE_PATH = 2,
    QUERYLOC_PATH = 3,
    STIMSEQ_PATH = 4,
    MODALITY = 5,
    PREFIX = 6,
    BASE_PATH = 7,
    MASK_PATH = 8,
    FRAMERATE = 9

class Dataset:
    def __init__(self, video_data=None, timestamps=None, query_locations=None,
                 stimseq=None, metadata=None, stage=PipeStages.PROCESSED):

        # Paths to the data used here.
        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

        if not video_data:
            self.num_frames = video_data.shape[-1]
            self.width = video_data.shape[1]
            self.height = video_data.shape[0]
            self.video_data = video_data

        self.framerate = self.metadata[Metadata.FRAMERATE]

        self.stimtrain_path = self.metadata[Metadata.STIMSEQ_PATH]
        self.video_path = self.metadata[Metadata.VIDEO_PATH]
        self.mask_path = self.metadata.get(Metadata.MASK_PATH, self.video_path[0:-4] + "_mask" + self.video_path[-4:])
        self.base_path = self.metadata[Metadata.BASE_PATH]

        self.prefix = self.metadata[Metadata.PREFIX]

        if self.video_path:
            # If we don't have supplied definitions of the base path of the dataset or the filename prefix,
            # then guess.
            if not self.base_path:
                self.base_path = os.path.dirname(os.path.realpath(self.video_path))
            if not self.prefix:
                self.prefix = os.path.basename(os.path.realpath(self.video_path))[0:-4]

            self.image_path = self.metadata[Metadata.IMAGE_PATH]
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

            self.coord_path = self.metadata[Metadata.QUERYLOC_PATH]
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



        # Information about the dataset
        self.stage = stage
        self.framerate = -1
        self.num_frames = -1
        self.width = -1
        self.height = -1
        self.framestamps = np.empty([1])
        self.reference_frame_idx = []
        self.stimtrain_frame_stamps = np.empty([1])

        # The data are roughly grouped by the following:
        # Base data
        self.coord_data = np.empty([1])
        self.image_data = np.empty([1])
        # Video data (processed or pipelined)
        self.video_data = np.empty([1])
        self.mask_data = np.empty([1])


    def clear_video_data(self):
        print("Deleting video data from "+self.video_path)
        del self.video_data
        del self.mask_data

    def load_data(self):
        if self.stage is PipeStages.RAW:
            self.load_raw_data()
        elif self.stage is PipeStages.PROCESSED:
            self.load_processed_data()
        elif self.stage is PipeStages.PIPELINED:
            self.load_pipelined_data()
        elif self.stage is PipeStages.ANALYSIS_READY:
            self.load_analysis_ready_data()

    def load_raw_data(self):
        resource = load_video(self.video_path)

        self.video_data = resource.data

        self.framerate = resource.metadict["framerate"]
        self.metadata_data = resource.metadict
        self.width = resource.data.shape[1]
        self.height = resource.data.shape[0]
        self.num_frames = resource.data.shape[-1]

        if self.coord_path:
            self.coord_data = pd.read_csv(self.coord_path, delimiter=',', header=None,
                                          encoding="utf-8-sig").to_numpy()

    def load_pipelined_data(self):
        if self.stage is PipeStages.PIPELINED:

            resource = load_video(self.video_path)

            self.video_data = resource.data

            self.framerate = resource.metadict["framerate"]
            self.metadata_data = resource.metadict
            self.width = resource.data.shape[1]
            self.height = resource.data.shape[0]
            self.num_frames = resource.data.shape[-1]

            if os.path.exists(self.mask_path):
                mask_res = load_video(self.mask_path)


            if self.coord_path:
                self.coord_data = pd.read_csv(self.coord_path, delimiter=',', header=None,
                                              encoding="utf-8-sig").to_numpy()
            if self.framestamp_path:
                # Load our text data.
                self.framestamps = pd.read_csv(self.framestamp_path, delimiter=',', header=None,
                                               encoding="utf-8-sig").to_numpy()

            if self.stimtrain_path:
                self.stimtrain_frame_stamps = np.cumsum(np.squeeze(pd.read_csv(self.stimtrain_path, delimiter=',', header=None,
                                                          encoding="utf-8-sig").to_numpy()))
            else:
                self.stimtrain_frame_stamps = self.num_frames-1
        else:
            warnings.warn("Dataset is not currently set as pipelined. Cannot load data.")

    def load_processed_data(self, force=False):
        # Establish our unpipelined filenames
        if self.stage is not PipeStages.RAW or force:
            resource = load_video(self.video_path)

            self.video_data = resource.data

            self.framerate = resource.metadict["framerate"]
            self.metadata_data = resource.metadict
            self.width = resource.data.shape[1]
            self.height = resource.data.shape[0]
            self.num_frames = resource.data.shape[-1]

            self.video_data, xforms, inliers = optimizer_stack_align(self.video_data,
                                                                         reference_idx=self.reference_frame_idx,
                                                                         dropthresh=0.0)

            print( "Keeping " +str(np.sum(inliers))+ " of " +str(self.num_frames)+"...")

            # Update everything with what's an inlier now.
            self.ref_video_data = self.ref_video_data[..., inliers]
            self.framestamps = self.framestamps[inliers]
            self.video_data = self.video_data[..., inliers]
            self.mask_data = self.mask_data[..., inliers]

            (rows, cols) = self.video_data.shape[0:2]

            for f in range(self.num_frames):
                if xforms[f] is not None:
                    self.video_data[..., f] = cv2.warpAffine(self.video_data[..., f], xforms[f],
                                                             (cols, rows),
                                                             flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                    self.mask_data[..., f] = cv2.warpAffine(self.mask_data[..., f], xforms[f],
                                                            (cols, rows),
                                                            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)

            self.num_frames = self.video_data.shape[-1]

    def save_data(self, suffix):
        save_video(self.video_path[0:-4]+suffix+".avi", self.video_data, self.framerate)



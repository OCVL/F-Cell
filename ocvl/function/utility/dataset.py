import glob
import json
import os
import warnings
from enum import Enum, StrEnum
from logging import warning
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
from colorama import Fore
from scipy.ndimage import gaussian_filter

from ocvl.function.preprocessing.improc import optimizer_stack_align, dewarp_2D_data, flat_field, weighted_z_projection
from ocvl.function.utility.format_parser import FormatParser
from ocvl.function.utility.json_format_constants import DataTags, MetaTags, DataFormatType, AcquisiTags, Pipeline, \
    ControlParams
from ocvl.function.utility.resources import load_video, save_video, save_tiff_stack

stimseq_fName = None

class PipeStages(Enum):
    RAW = 0,
    PROCESSED = 1,
    PIPELINED = 2,
    ANALYSIS_READY = 3

def load_metadata(metadata_params, ext_metadata):
    meta_fields = {}
    if metadata_params is not None:
        for metatag in MetaTags:
            thetag = metadata_params.get(metatag)
            if thetag is not None \
                    and thetag is not MetaTags.METATAG \
                    and thetag is not MetaTags.TYPE:
                meta_fields[metatag] = metadata_params.get(metatag)

    # Load our externally sourced metadata
    if not ext_metadata.empty and metadata_params is not None:
        if ext_metadata.at[ext_metadata.index[0], AcquisiTags.DATA_PATH].exists():
            metadata_path = ext_metadata.at[ext_metadata.index[0], AcquisiTags.DATA_PATH]
            metatype = metadata_params.get(MetaTags.TYPE)
            loadfields = metadata_params.get(MetaTags.FIELDS_OF_INTEREST)

            if metatype == "text_file":
                dat_metadata = pd.read_csv(ext_metadata.at[ext_metadata.index[0], AcquisiTags.DATA_PATH],
                                           encoding="utf-8-sig", skipinitialspace=True)

                for field, column in loadfields.items():
                    met_dat = dat_metadata.get(column, pd.Series())
                    if not met_dat.empty:
                        meta_fields[field] = met_dat.to_numpy()
            elif metatype == "database":
                pass
            elif metatype == "mat_file":
                pass

        else:
            metadata_path = None
    else:
        metadata_path = None

    return meta_fields, metadata_path

def parse_file_metadata(config_json_path, pName, group="processed"):
    with open(config_json_path, 'r') as config_json_path:
        dat_form = json.load(config_json_path)

        allFilesColumns = [AcquisiTags.DATASET, AcquisiTags.DATA_PATH, DataFormatType.FORMAT_TYPE]
        allFilesColumns.extend([d.value for d in DataTags])

        dat_format = dat_form.get(group)
        if dat_format is not None:

            im_form = dat_format.get(DataFormatType.IMAGE)
            vid_form = dat_format.get(DataFormatType.VIDEO)
            mask_form = dat_format.get(DataFormatType.MASK)
            query_form = dat_format.get(DataFormatType.QUERYLOC)

            metadata_form = None
            metadata_params = None
            if dat_format.get(MetaTags.METATAG) is not None:
                metadata_params = dat_format.get(MetaTags.METATAG)
                metadata_form = metadata_params.get(DataFormatType.METADATA)

            all_ext = ()
            # Grab our extensions, make sure to check them all.
            all_ext = all_ext + (vid_form[vid_form.rfind(".", -5, -1):],) if vid_form else all_ext
            all_ext = all_ext + (mask_form[mask_form.rfind(".", -5, -1):],) if mask_form and mask_form[
                                                                                             mask_form.rfind(".", -5,
                                                                                                             -1):] not in all_ext else all_ext
            all_ext = all_ext + (im_form[im_form.rfind(".", -5, -1):],) if im_form and im_form[im_form.rfind(".", -5,
                                                                                                             -1):] not in all_ext else all_ext
            all_ext = all_ext + (query_form[query_form.rfind(".", -5, -1):],) if query_form and query_form[
                                                                                                query_form.rfind(".",
                                                                                                                 -5,
                                                                                                                 -1):] not in all_ext else all_ext
            all_ext = all_ext + (metadata_form[metadata_form.rfind(".", -5, -1):],) if metadata_form and metadata_form[
                                                                                                         metadata_form.rfind(
                                                                                                             ".", -5,
                                                                                                             -1):] not in all_ext else all_ext

            # Construct the parser we'll use for each of these forms
            parser = FormatParser(vid_form, mask_form, im_form, metadata_form, query_form)

            # If our control data comes from the file structure, then prep to check for it in case its in the config.
            control_params = dat_format.get(ControlParams.NAME, None)
            if control_params is not None:
                control_loc = control_params.get(ControlParams.LOCATION, "folder")
                if control_loc == "folder":
                    control_folder = control_params.get(ControlParams.FOLDER_NAME, "control")


            # Parse out the locations and filenames, store them in a hash table by location.
            searchpath = Path(pName)
            allFiles = list()
            recurse_me = dat_format.get(DataFormatType.RECURSIVE, True)
            if recurse_me:
                for ext in all_ext:
                    for path in searchpath.rglob("*" + ext):
                        format_type, file_info = parser.parse_file(path.name)
                        if format_type is not None:
                            file_info[DataFormatType.FORMAT_TYPE] = format_type
                            file_info[AcquisiTags.DATA_PATH] = path
                            file_info[AcquisiTags.BASE_PATH] = path.parent
                            file_info[AcquisiTags.DATASET] = None
                            entry = pd.DataFrame.from_dict([file_info])

                            allFiles.append(entry)
            else:
                for ext in all_ext:
                    for path in searchpath.glob("*" + ext):
                        format_type, file_info = parser.parse_file(path.name)
                        if format_type is not None:
                            file_info[DataFormatType.FORMAT_TYPE] = format_type
                            file_info[AcquisiTags.DATA_PATH] = path
                            file_info[AcquisiTags.BASE_PATH] = path.parent
                            file_info[AcquisiTags.DATASET] = None
                            entry = pd.DataFrame.from_dict([file_info])

                            allFiles.append(entry)
                    if control_params is not None:
                        controlsearchpath = searchpath.joinpath(control_folder)
                        for path in controlsearchpath.glob("*" + ext):
                            format_type, file_info = parser.parse_file(path.name)
                            if format_type is not None:
                                file_info[DataFormatType.FORMAT_TYPE] = format_type
                                file_info[AcquisiTags.DATA_PATH] = path
                                file_info[AcquisiTags.BASE_PATH] = path.parent
                                file_info[AcquisiTags.DATASET] = None
                                entry = pd.DataFrame.from_dict([file_info])

                                allFiles.append(entry)
            if allFiles:
                return dat_form, pd.concat(allFiles, ignore_index=True)
            else:
                return None
        else:
            return None

def initialize_and_load_dataset(acquisition, metadata_params, stage=PipeStages.PROCESSED):

    video_info = acquisition.loc[acquisition[DataFormatType.FORMAT_TYPE] == DataFormatType.VIDEO]
    mask_info = acquisition.loc[acquisition[DataFormatType.FORMAT_TYPE] == DataFormatType.MASK]
    metadata_info = acquisition.loc[acquisition[DataFormatType.FORMAT_TYPE] == DataFormatType.METADATA]
    im_info = acquisition.loc[acquisition[DataFormatType.FORMAT_TYPE] == DataFormatType.IMAGE]
    query_info = acquisition.loc[acquisition[DataFormatType.FORMAT_TYPE] == DataFormatType.QUERYLOC]

    # Read in directly-entered metatags.
    meta_fields = {}
    if metadata_params is not None:
        for metatag in MetaTags:
            thetag = metadata_params.get(metatag)
            if thetag is not None \
                    and thetag is not MetaTags.METATAG \
                    and thetag is not MetaTags.TYPE:
                meta_fields[metatag] = metadata_params.get(metatag)

    # Load our externally sourced metadata
    meta_fields, metadata_path = load_metadata(metadata_params, metadata_info)

    # Take metadata gleaned from our filename, as well as our metadata files,
    # and combine them into a single dictionary.
    combined_meta_dict = video_info.squeeze().to_dict() | meta_fields

    # Add paths to things that we may want, depending on the stage we're at.
    if not im_info.empty:
        combined_meta_dict[AcquisiTags.IMAGE_PATH] = im_info.at[im_info.index[0], AcquisiTags.DATA_PATH]

    if not query_info.empty:
        combined_meta_dict[DataTags.QUERYLOC] = query_info[DataTags.QUERYLOC].unique().tolist()
        combined_meta_dict[AcquisiTags.QUERYLOC_PATH] = query_info[AcquisiTags.DATA_PATH].unique().tolist()


    if not mask_info.empty:
        mask_path = mask_info.at[mask_info.index[0], AcquisiTags.DATA_PATH]
    else:
        mask_path = None

    if not video_info.empty:
        print()
        print(Fore.GREEN +"Initializing and loading dataset: " + video_info.at[video_info.index[0], AcquisiTags.DATA_PATH].name)
        dataset = load_dataset(video_info.at[video_info.index[0], AcquisiTags.DATA_PATH],
                               mask_path,
                               metadata_path,
                               combined_meta_dict,
                               stage)
    else:
        warnings.warn("Failed to detect dataset.")
        dataset = None

    return dataset


def load_dataset(video_path, mask_path=None, extra_metadata_path=None, dataset_metadata=None, stage=PipeStages.PROCESSED):

    mask_data = None
    metadata = dataset_metadata
    metadata[AcquisiTags.VIDEO_PATH] = dataset_metadata[AcquisiTags.DATA_PATH]
    metadata[AcquisiTags.MASK_PATH] = mask_path
    metadata[AcquisiTags.META_PATH] = extra_metadata_path

    if video_path.exists():
        resource = load_video(video_path)
        video_data = resource.data
        if MetaTags.FRAMERATE not in metadata:
            metadata[MetaTags.FRAMERATE] = resource.metadict.get(MetaTags.FRAMERATE)
    else:
        warning("Video path does not exist at: "+str(video_path))
        return None

    if mask_path:
        if mask_path.exists():
            mask_res = load_video(mask_path)
            mask_data = mask_res.data / mask_res.data.max()
            mask_data[mask_data <= 0] = 0
            mask_data[mask_data > 0] = 1
            # Mask our video data correspondingly.
            premask_dtype = video_data.dtype
            video_data = (video_data * mask_data).astype(premask_dtype)
            mask_data = mask_data.astype(premask_dtype)
        else:
            warning("Mask path does not exist at: "+str(mask_path))
    else:
        # If we don't have a mask path, then just make our mask from the video data
        mask_data = (video_data != 0).astype(video_data.dtype)

    avg_image_data = None
    if AcquisiTags.IMAGE_PATH in metadata:
        avg_image_data = cv2.imread(metadata.get(AcquisiTags.IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    # For importing the query locations
    queryloc_data = []
    if AcquisiTags.QUERYLOC_PATH in metadata and MetaTags.QUERY_LOCATIONS not in metadata:

        querylocs = metadata.get(AcquisiTags.QUERYLOC_PATH)
        for locpath in querylocs:
            match locpath.suffix:
                case ".csv":
                    queryloc_data.append(pd.read_csv(locpath, header=None, encoding="utf-8-sig").to_numpy())
                case ".txt":
                    queryloc_data.append(pd.read_csv(locpath, sep=None, header=None, encoding="utf-8-sig").to_numpy())
                case "":
                    if locpath.name == "All Pixels":
                        xm, ym = np.meshgrid(np.arange(video_data.shape[1]),np.arange(video_data.shape[0]), indexing="xy")

                        queryloc_data.append( np.column_stack((xm.flatten().astype(np.int16), ym.flatten().astype(np.int16))))

    elif MetaTags.QUERY_LOCATIONS in metadata:
        queryloc_data.append(metadata.get(MetaTags.QUERY_LOCATIONS))

    # For importing the framestamps of the video- these are the temporal, monotonic frame indexes of the video, in case frames were dropped
    # in the pipeline or in the processing stages
    stamps = metadata.get(MetaTags.FRAMESTAMPS)
    if stamps is None:
        stamps = np.arange(video_data.shape[-1])

    # For importing metadata RE: the stimulus delivery
    stimulus_sequence = None
    if AcquisiTags.STIMSEQ_PATH in metadata and MetaTags.STIMULUS_SEQ not in metadata:
        stimulus_sequence = pd.read_csv(metadata.get(AcquisiTags.STIMSEQ_PATH), header=None, encoding="utf-8-sig").to_numpy()
    elif MetaTags.STIMULUS_SEQ in metadata:
        stimulus_sequence = metadata.get(MetaTags.STIMULUS_SEQ)
    elif stage == PipeStages.PIPELINED:
        if Dataset.stimseq_fName is None:
            Dataset.stimseq_fName = filedialog.askopenfilename(title="Stimulus sequence not detected in metadata. Select a stimulus sequence file.", initialdir=metadata.get(AcquisiTags.BASE_PATH, None))
        stimulus_sequence = np.cumsum(pd.read_csv(Dataset.stimseq_fName, header=None,encoding="utf-8-sig").to_numpy())


    return Dataset(video_data, mask_data, avg_image_data, metadata, queryloc_data, stamps, stimulus_sequence, stage)


def preprocess_dataset(dataset, pipeline_params, reference_dataset=None):

    if reference_dataset is None:
        reference_dataset = dataset

    # First do custom preprocessing steps, e.g. things implemented expressly for and by the OCVL
    # If you would like to "bake in" custom pipeline steps, please contact the OCVL using GitHub's Issues
    # or submit a pull request.
    custom_steps = pipeline_params.get(Pipeline.CUSTOM)
    if custom_steps is not None:
        # For "baked in" dewarping- otherwise, data is expected to be dewarped already
        pre_dewarp = custom_steps.get("dewarp")
        match pre_dewarp:
            case "ocvl":
                # Framestamps from the MEAO AOSLO  start at 1, instead of 0- shift to fix.
                dataset.framestamps = dataset.framestamps-1

                if dataset.metadata[AcquisiTags.META_PATH] is not None:
                    dat_metadata = pd.read_csv(dataset.metadata[AcquisiTags.META_PATH], encoding="utf-8-sig",
                                               skipinitialspace=True)

                    ncc = 1 - dat_metadata["NCC"].to_numpy(dtype=float)
                    dataset.reference_frame_idx = min(range(len(ncc)), key=ncc.__getitem__)

                    # Dewarp our data.
                    # First find out how many strips we have.
                    numstrips = 0
                    for col in dat_metadata.columns.tolist():
                        if "XShift" in col:
                            numstrips += 1

                    xshifts = np.zeros([ncc.shape[0], numstrips])
                    yshifts = np.zeros([ncc.shape[0], numstrips])

                    for col in dat_metadata.columns.tolist():
                        shiftrow = col.strip().split("_")[0][5:]
                        npcol = dat_metadata[col].to_numpy()
                        if npcol.dtype == "object":
                            npcol[npcol == " "] = np.nan
                        if col != "XShift" and "XShift" in col:
                            xshifts[:, int(shiftrow)] = npcol
                        if col != "YShift" and "YShift" in col:
                            yshifts[:, int(shiftrow)] = npcol

                    # Determine the residual error in our dewarping, and obtain the maps
                    dataset.video_data, map_mesh_x, map_mesh_y = dewarp_2D_data(dataset.video_data,
                                                                                yshifts, xshifts)

                    # Dewarp our mask too.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(action="ignore", message="invalid value encountered in cast")

                        for f in range(dataset.num_frames):
                            norm_frame = dataset.mask_data[..., f].astype("float32")
                            norm_frame[norm_frame == 0] = np.nan

                            dataset.mask_data[..., f] = cv2.remap(norm_frame,
                                                                  map_mesh_x, map_mesh_y,
                                                                  interpolation=cv2.INTER_NEAREST)

    # Trim the video down to a smaller/different size, if desired.
    trim = pipeline_params.get(Pipeline.TRIM)
    if trim is not None:
        start_idx = trim.get("start_idx")
        end_idx = trim.get("end_idx")
        dataset.video_data = dataset.video_data[..., start_idx:end_idx]
        dataset.mask_data = dataset.mask_data[..., start_idx:end_idx]
        dataset.num_frames = dataset.video_data.shape[-1]
        dataset.framestamps = dataset.framestamps[(dataset.framestamps < end_idx) & (dataset.framestamps >= start_idx)]

    align_dat = reference_dataset.video_data.copy()
    mask_dat = reference_dataset.mask_data.copy()
    # Try and figure out the reference frame, if we haven't already.
    # This is based on the simple method of determining the maximum area subtended by our mask.
    if dataset.reference_frame_idx is None:
        amt_data = np.zeros([dataset.num_frames])
        for f in range(dataset.num_frames):
            amt_data[f] = mask_dat[..., f].flatten().sum()

        dataset.reference_frame_idx = amt_data.argmax()
        del amt_data

    # Flat field the video for alignment, if desired.
    if flat_field is not None and flat_field:
        align_dat = flat_field(align_dat)

    # Gaussian blur the data first before aligning, if requested
    gausblur = pipeline_params.get(Pipeline.GAUSSIAN_BLUR)
    if gausblur is not None and gausblur != 0.0:
        for f in range(align_dat.shape[-1]):
            align_dat[..., f] = gaussian_filter(align_dat[..., f], sigma=gausblur)
        align_dat *= mask_dat

    # Then crop the data, if requested
    mask_roi = pipeline_params.get(Pipeline.MASK_ROI)
    if mask_roi is not None:
        r = mask_roi.get("r")
        c = mask_roi.get("c")
        width = mask_roi.get("width")
        height = mask_roi.get("height")

        # Everything outside the roi specified should be zero
        # This approach is RAM intensive, but easy.
        align_dat = align_dat[r:r + height, c:c + width, :]

        mask_dat = mask_dat[r:r + height, c:c + width, :]


    # Finally, correct for residual torsion if requested
    correct_torsion = pipeline_params.get(Pipeline.CORRECT_TORSION)
    if correct_torsion is not None and correct_torsion:
        align_dat, xforms, inliers, mask_dat = optimizer_stack_align(align_dat, mask_dat,
                                                                     reference_idx=dataset.reference_frame_idx,
                                                                     dropthresh=0, justalign=True, transformtype="rigid")

        # Apply the transforms to the unfiltered, cropped, etc. trimmed dataset
        og_dtype = dataset.video_data.dtype

        for f in range(dataset.num_frames):
            if inliers[f]:
                norm_frame = dataset.video_data[..., f].astype("float32")
                # Make all masked data nan so that when we transform them we don't have weird edge effects
                norm_frame[dataset.mask_data[..., f] == 0] = np.nan

                norm_frame = cv2.warpAffine(norm_frame, xforms[f],
                                            (norm_frame.shape[1], norm_frame.shape[0]),
                                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                            borderValue=np.nan)

                dataset.mask_data[..., f] = np.isfinite(norm_frame).astype(og_dtype)  # Our new mask corresponds to the real data.
                norm_frame[np.isnan(norm_frame)] = 0  # Make anything that was nan into a 0, to be kind to non nan-types
                dataset.video_data[..., f] = norm_frame.astype(og_dtype)

        dataset.avg_image_data, awp = weighted_z_projection(dataset.video_data, dataset.mask_data)

    return dataset


class Dataset:
    # Static var used on such occasions that a user doesn't provide an explicit method determining the stimulus sequence
    # pattern (either via a metadata file, or via explicitly setting it in the json file)
    stimseq_fName = None

    def __init__(self, video_data=None, mask_data=None, avg_image_data=None, metadata=None, query_locations=[],
                 framestamps=None, stimseq=None, stage=PipeStages.PROCESSED):

        # Paths to the data used here.
        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

        # Information about the dataset
        self.stage = stage
        self.framerate = self.metadata.get(MetaTags.FRAMERATE)
        self.num_frames = -1
        self.width = -1
        self.height = -1
        self.framestamps = framestamps
        self.reference_frame_idx = None
        self.stimtrain_frame_stamps = stimseq

        self.video_data = video_data
        self.mask_data = mask_data
        self.avg_image_data = avg_image_data

        if video_data is not None:
            self.num_frames = video_data.shape[-1]
            self.width = video_data.shape[1]
            self.height = video_data.shape[0]

        self.stimtrain_path = self.metadata.get(AcquisiTags.STIMSEQ_PATH)
        self.video_path = self.metadata.get(AcquisiTags.VIDEO_PATH)
        self.mask_path = self.metadata.get(AcquisiTags.MASK_PATH)
        self.base_path = self.metadata.get(AcquisiTags.BASE_PATH)

        self.prefix = self.metadata.get(AcquisiTags.PREFIX)

        if self.video_path:
            # If we don't have supplied definitions of the base path of the dataset or the filename prefix,
            # then guess.
            if not self.base_path:
                self.base_path = self.video_path.parent
            if not self.prefix:
                self.prefix = self.video_path.with_suffix("")

            self.image_path = self.metadata.get(AcquisiTags.IMAGE_PATH, None)
            # If we don't have supplied definitions of the image associated with this dataset,
            # then guess.
            if self.image_path is None:
                imname = None
                if (stage is PipeStages.PROCESSED or stage is PipeStages.PIPELINED) and \
                        self.prefix.with_name(self.prefix.name + ".tif").exists():

                        imname = self.prefix.with_name(self.prefix.name + ".tif")

                if not imname:
                    for filename in self.base_path.glob("*_ALL_ACQ_AVG.tif"):
                        imname = filename

                if not imname and stage is PipeStages.PIPELINED:
                    warnings.warn("Unable to detect viable average image file; generating one from video. Dataset functionality may be limited.")
                    self.image_path = None
                    self.avg_image_data, _ = weighted_z_projection(self.video_data, self.mask_data)
                else:
                    print(Fore.YELLOW + "Automatically detected the average image "+ str(imname.name) +". **Please verify your image format string**: " )
                    self.image_path = imname
                    self.metadata[AcquisiTags.IMAGE_PATH] = self.image_path
                    self.avg_image_data = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

            self.query_coord_paths = self.metadata.get(AcquisiTags.QUERYLOC_PATH)
            # If we don't have query locations associated with this dataset, then try to guess.
            if self.query_coord_paths is None and not query_locations:

                coordname = None
                if (stage is PipeStages.PROCESSED or stage is PipeStages.PIPELINED) and \
                        self.prefix.with_name(self.prefix.name + "_coords.csv").exists():

                        coordname = self.prefix.with_name(self.prefix.name + "_coords.csv")

                    # If we don't have an image specific to this dataset, search for the all acq avg
                if coordname is None:
                    for filename in self.base_path.glob("*_ALL_ACQ_AVG_coords.csv"):
                            coordname = filename

                if coordname is None and stage is PipeStages.PIPELINED:
                    warnings.warn("Unable to detect viable query location file for dataset at: "+ str(self.video_path))
                    self.metadata[AcquisiTags.QUERYLOC_PATH] = []
                    self.query_coord_paths = []
                elif stage is PipeStages.PIPELINED:
                    print(Fore.YELLOW+"Automatically detected the query locations: "+str(coordname.name) + ". **Please verify your queryloc format string**" )
                    # Update our metadata structure, and our internally stored query coord paths.
                    self.metadata[AcquisiTags.QUERYLOC_PATH] = [coordname]
                    self.query_coord_paths = [coordname]
                    match self.query_coord_paths[0].suffix:
                        case ".csv":
                            query_locations = [pd.read_csv(self.query_coord_paths[0], header=None, encoding="utf-8-sig").to_numpy()]
                        case ".txt":
                            query_locations = [pd.read_csv(self.query_coord_paths[0], sep=None, header=None, encoding="utf-8-sig").to_numpy()]

            self.query_loc = []
            for queries in query_locations:
                self.query_loc.append(np.unique(queries, axis=0))

            self.query_status = [np.full(locs.shape[0], "Included", dtype=object) for locs in query_locations]
            self.iORG_signals = [None] * len(query_locations)
            self.summarized_iORGs = [None] * len(query_locations)


    def clear_video_data(self):
        del self.video_data
        del self.mask_data

    def load_data(self, force_reload=False):

        # Go down the line, loading data that doesn't already exist in this dataset.
        if (not self.video_data or force_reload) and os.path.exists(self.video_path):
            resource = load_video(self.video_path)

            self.video_data = resource.data
            self.framerate = resource.metadict[MetaTags.FRAMERATE]
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

        if (not self.query_loc or force_reload) and os.path.exists(self.query_coord_paths):
            self.query_loc = pd.read_csv(self.query_coord_paths, delimiter=',', header=None,
                                         encoding="utf-8-sig").to_numpy()

        if (not self.avg_image_data or force_reload) and os.path.exists(self.image_path) :
            self.avg_image_data = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if (not self.stimtrain_frame_stamps or force_reload) and os.path.exists(self.stimtrain_path):
            self.stimtrain_frame_stamps = np.cumsum(np.squeeze(pd.read_csv(self.stimtrain_path, delimiter=',', header=None,
                                                                           encoding="utf-8-sig").to_numpy()))
        else:
            self.stimtrain_frame_stamps = 0

    def save_data(self, suffix):
        save_video(self.video_path[0:-4]+suffix+".avi", self.video_data, self.framerate)


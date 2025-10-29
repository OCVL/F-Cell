import glob
import json
import os
import pickle
import warnings
from enum import Enum, StrEnum
from logging import warning
from pathlib import Path, PurePath
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
from colorama import Fore
from file_tag_parser.tags.json_format_constants import DataFormat, AcquisiPaths
from scipy.ndimage import gaussian_filter

from ocvl.function.preprocessing.improc import optimizer_stack_align, dewarp_2D_data, flat_field, weighted_z_projection, \
    norm_video
from ocvl.function.utility.json_format_constants import DataTags, MetaTags, AcquisiParams, \
    PreAnalysisPipeline, \
    ControlParams, Analysis, NormParams, DebugParams, DisplayParams, SegmentParams
from ocvl.function.utility.resources import load_video, save_video, save_tiff_stack

stimseq_fName = None

class Stages(Enum):
    RAW = 0,
    PREANALYSIS = 1,
    ANALYSIS = 2,
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
        if ext_metadata.at[ext_metadata.index[0], AcquisiPaths.DATA_PATH].exists():
            metadata_path = ext_metadata.at[ext_metadata.index[0], AcquisiPaths.DATA_PATH]
            metatype = metadata_params.get(MetaTags.TYPE)
            loadfields = metadata_params.get(MetaTags.FIELDS_OF_INTEREST)

            headher = metadata_params.get(MetaTags.HEADERS, 0)
            if metatype == "text_file":
                dat_metadata = pd.read_csv(ext_metadata.at[ext_metadata.index[0], AcquisiPaths.DATA_PATH],
                                           encoding="utf-8-sig", skipinitialspace=True, header=headher)

                for field, column in loadfields.items():
                    met_dat = dat_metadata.get(column, pd.Series())
                    if not met_dat.empty:
                        meta_fields[field] = met_dat.to_numpy()
                    if field == MetaTags.FRAMESTAMPS:
                        meta_fields[field].sort()
            elif metatype == "database":
                pass
            elif metatype == "mat_file":
                pass
            elif metatype == "pickle_file":
                pass

        else:
            metadata_path = None
    else:
        metadata_path = None

    return meta_fields, metadata_path


def obtain_analysis_output_path(current_folder, timestamp, analysis_params, mkdir=True):

    if current_folder is not isinstance(current_folder, Path):
        current_folder = Path(current_folder)

    output_folder = analysis_params.get(Analysis.OUTPUT_FOLDER)
    if output_folder is None:
        output_folder = PurePath("Results")
    else:
        output_folder = PurePath(output_folder)

    if analysis_params.get(Analysis.OUTPUT_SUBFOLDER, True):
        output_subfolder_method = analysis_params.get(Analysis.OUTPUT_SUBFOLDER_METHOD) #Check subfolder naming method
        if output_subfolder_method == 'DateTime': #Only supports saving things to a subfolder with a unique timestamp currently
            output_dt_subfolder = PurePath(timestamp)
        else:
            output_dt_subfolder = PurePath(timestamp)

        result_folder = current_folder.joinpath(output_folder, output_dt_subfolder)
        if mkdir:
            result_folder.mkdir(parents=True, exist_ok=True)
    else:
        result_folder = current_folder.joinpath(output_folder)
        if mkdir:
            result_folder.mkdir(parents=True, exist_ok=True)

    return result_folder

def initialize_and_load_dataset(folder, vidID, prefilter=None, timestamp=None, database=pd.DataFrame(),
                                params=None, stage=Stages.PREANALYSIS):

    if params is None:
        params = dict()

    display_params = params.get(DisplayParams.NAME, dict())
    analysis_params = params.get(Analysis.PARAMS, dict())
    debug_params = display_params.get(DebugParams.NAME, dict())
    metadata_params = params.get(MetaTags.METATAG, dict())
    seg_params = analysis_params.get(SegmentParams.NAME, dict())


    metadata_form = metadata_params.get(DataFormat.METADATA, dict())
    seg_pixelwise = seg_params.get(SegmentParams.PIXELWISE, False)  # Default to NO pixelwise analyses. Otherwise, add one.

    # Construct the filters we'll need for grabbing things related to our dataset.
    if prefilter is None:
        prefilter = True

    if folder is not None:
        folder_filter = database[AcquisiPaths.BASE_PATH] == folder
    else:
        folder_filter = True

    if vidID is not None:
        vidid_filter = database[DataTags.VIDEO_ID] == vidID
    else:
        vidid_filter = True


    refim_filter = database[DataFormat.FORMAT_TYPE] == DataFormat.IMAGE
    mask_filter = database[DataFormat.FORMAT_TYPE] == DataFormat.MASK
    meta_filter = database[DataFormat.FORMAT_TYPE] == DataFormat.METADATA
    qloc_filter = database[DataFormat.FORMAT_TYPE] == DataFormat.QUERYLOC
    vidtype_filter = database[DataFormat.FORMAT_TYPE] == DataFormat.VIDEO

    # Get the reference images and query locations and this video number, only for the mode and folder mask we want.
    slice_of_life = prefilter & folder_filter & (vidid_filter | (refim_filter | qloc_filter))

    acquisition = database.loc[slice_of_life]

    video_info = acquisition.loc[acquisition[DataFormat.FORMAT_TYPE] == DataFormat.VIDEO]
    mask_info = acquisition.loc[acquisition[DataFormat.FORMAT_TYPE] == DataFormat.MASK]
    metadata_info = acquisition.loc[acquisition[DataFormat.FORMAT_TYPE] == DataFormat.METADATA]
    im_info = acquisition.loc[acquisition[DataFormat.FORMAT_TYPE] == DataFormat.IMAGE]
    query_info = acquisition.loc[acquisition[DataFormat.FORMAT_TYPE] == DataFormat.QUERYLOC]

    if video_info.shape[0] > 1:
        warnings.warn(f"WARNING: MULTIPLE VIDEOs WITH ID: {vidID} DETECTED!! Only loading dataset with first ID.")
        database.drop(index=video_info.index[1:], inplace=True)

        if vidID is not None:
            vidid_filter = database[DataTags.VIDEO_ID] == vidID
        else:
            vidid_filter = True
        slice_of_life = prefilter & folder_filter & (vidid_filter | (refim_filter | qloc_filter))

        acquisition = database.loc[slice_of_life]
        video_info = acquisition.loc[acquisition[DataFormat.FORMAT_TYPE] == DataFormat.VIDEO]

    if mask_info.shape[0] > 1:
        warnings.warn(f"WARNING: MULTIPLE MASK VIDEOs WITH ID: {vidID} DETECTED!! Only loading dataset with first ID.")
        database.drop(index=mask_info.index[1:], inplace=True)

        if vidID is not None:
            vidid_filter = database[DataTags.VIDEO_ID] == vidID
        else:
            vidid_filter = True
        slice_of_life = prefilter & folder_filter & (vidid_filter | (refim_filter | qloc_filter))

        acquisition = database.loc[slice_of_life]
        mask_info = acquisition.loc[acquisition[DataFormat.FORMAT_TYPE] == DataFormat.MASK]

    if metadata_info.shape[0] > 1:
        warnings.warn(f"WARNING: MULTIPLE METADATA FILES WITH ID: {vidID} DETECTED!! Only loading dataset with first ID.")
        database.drop(index=metadata_info.index[1:], inplace=True)

        if vidID is not None:
            vidid_filter = database[DataTags.VIDEO_ID] == vidID
        else:
            vidid_filter = True
        slice_of_life = prefilter & folder_filter & (vidid_filter | (refim_filter | qloc_filter))

        acquisition = database.loc[slice_of_life]
        metadata_info = acquisition.loc[acquisition[DataFormat.FORMAT_TYPE] == DataFormat.METADATA]

    result_path = PurePath()
    if stage == Stages.PREANALYSIS:
        pass

    elif stage == Stages.ANALYSIS:
        result_path = obtain_analysis_output_path(folder, timestamp, analysis_params, mkdir=False)

    # Read in directly-entered metatags.
    meta_fields = {}
    if metadata_params is not None:
        for metatag in MetaTags:
            thetag = metadata_params.get(metatag)
            if thetag is not None \
                    and thetag is not MetaTags.METATAG \
                    and thetag is not MetaTags.TYPE:
                meta_fields[metatag] = metadata_params.get(metatag)

    # Load our externally sourced, organic, no-gmo metadata
    meta_fields, metadata_path = load_metadata(metadata_params, metadata_info)

    # Take metadata gleaned from our filename, as well as our metadata files,
    # and combine them into a single dictionary.
    combined_meta_dict = video_info.squeeze().to_dict() | meta_fields

    # Add paths to things that we may want, depending on the stage we're at.
    if not im_info.empty:
        combined_meta_dict[AcquisiPaths.IMAGE_PATH] = im_info.at[im_info.index[0], AcquisiPaths.DATA_PATH]

    if not query_info.empty:
        combined_meta_dict[DataTags.QUERYLOC] = query_info[DataTags.QUERYLOC].unique().tolist()
        combined_meta_dict[AcquisiPaths.QUERYLOC_PATH] = query_info[AcquisiPaths.DATA_PATH].unique().tolist()


    if not mask_info.empty:
        mask_path = mask_info.at[mask_info.index[0], AcquisiPaths.DATA_PATH]
    else:
        mask_path = None

    if not video_info.empty:
        print()
        print(Fore.GREEN +"Initializing and loading dataset: " + video_info.at[video_info.index[0], AcquisiPaths.DATA_PATH].name)
        dataset = load_dataset(video_info.at[video_info.index[0], AcquisiPaths.DATA_PATH],
                               mask_path,
                               metadata_path,
                               combined_meta_dict,
                               stage)
    else:
        warnings.warn("Failed to detect dataset.")
        dataset = None

    new_entries = None
    if stage == Stages.PREANALYSIS and dataset is not None:
        pass

    elif stage == Stages.ANALYSIS and dataset is not None:
        # We could have multiple sub_datasets if the user input multiple stimulus sequences.
        for sub_dataset in dataset:
            postprocess_dataset(sub_dataset, analysis_params, result_path, debug_params)

        new_entries = pd.DataFrame()

        # If we didn't find an average image in our database, but were able to automagically detect or make one,
        # Then add the automagically detected one to our database.
        if database.loc[slice_of_life & refim_filter].empty and dataset[0].avg_image_data is not None:
            base_entry = database[slice_of_life & vidtype_filter].copy()
            base_entry.loc[base_entry.index[0], DataFormat.FORMAT_TYPE] = DataFormat.IMAGE
            base_entry.loc[base_entry.index[0], AcquisiPaths.DATA_PATH] = dataset[0].image_path
            base_entry.loc[base_entry.index[0], AcquisiPaths.DATASET] = None

            # Update the database
            new_entries = pd.concat([new_entries, base_entry], ignore_index=True)

        # Check to see if our dataset's number of query locations matches the ones we thought we found
        # (can happen if the query location format doesn't match, but dataset was able to find a candidate)
        if len(query_info) < len(dataset[0].query_loc):
            # If we have too few, then tack on some extra dataframes so we can track these found query locations, and add them to our database, using the dataset as a basis.
            base_entry = database[slice_of_life & vidtype_filter].copy()
            base_entry.loc[0, DataFormat.FORMAT_TYPE] = DataFormat.QUERYLOC
            base_entry.loc[0, AcquisiPaths.DATASET] = None

            for i in range(len(dataset[0].query_loc) - len(query_info)):
                base_entry.loc[base_entry.index[0], AcquisiPaths.DATA_PATH] = dataset[0].query_coord_paths[i]
                base_entry.loc[base_entry.index[0], DataTags.QUERYLOC] = "Auto_Detected_" + str(i)

                # Update the database
                new_entries = pd.concat([new_entries, base_entry], ignore_index=True)

        # If we can't find any query locations, or if we just want it, default to querying all pixels.
        if (len(dataset[0].query_loc) == 0 or seg_pixelwise) and Path("All Pixels") not in dataset[0].query_coord_paths:
            seg_pixelwise = True  # Set this to true, if we find that query loc for this dataset is 0

            xm, ym = np.meshgrid(np.arange(dataset[0].video_data.shape[1]),
                                 np.arange(dataset[0].video_data.shape[0]))

            xm = np.reshape(xm, (xm.size, 1))
            ym = np.reshape(ym, (ym.size, 1))

            allcoord_data = np.hstack((xm, ym))

            # If one dataset needs this, then all sub datasets do too.
            for sub_dataset in dataset:
                sub_dataset.query_loc.append(allcoord_data)
                sub_dataset.query_status = [np.full(locs.shape[0], "Included", dtype=object) for locs in sub_dataset.query_loc]
                sub_dataset.query_coord_paths.append(Path("All Pixels"))
                sub_dataset.metadata[AcquisiPaths.QUERYLOC_PATH].append(Path("All Pixels"))
                sub_dataset.iORG_signals = [None] * len(sub_dataset.query_loc)
                sub_dataset.summarized_iORGs = [None] * len(sub_dataset.query_loc)

            base_entry = database[slice_of_life & vidtype_filter].copy()
            base_entry.loc[base_entry.index[0], DataFormat.FORMAT_TYPE] = DataFormat.QUERYLOC
            base_entry.loc[base_entry.index[0], AcquisiPaths.DATA_PATH] = dataset[0].query_coord_paths[-1]
            base_entry.loc[base_entry.index[0], DataTags.QUERYLOC] = "All Pixels"
            base_entry.loc[base_entry.index[0], AcquisiPaths.DATASET] = None

            # Update the database
            new_entries = pd.concat([new_entries, base_entry], ignore_index=True)

        if len(dataset) == 1:
            database.loc[slice_of_life & vidtype_filter, AcquisiPaths.DATASET] = dataset[0]
        elif len(dataset) > 1:
            # If we have subdatasets (from multiple stimuli in one video),
            # then we need to make new entries for each of them so that they're processed correctly.

            for i, sub_dataset in enumerate(dataset):
                base_entry = database[slice_of_life & vidtype_filter].copy()
                base_entry.loc[slice_of_life & vidtype_filter, AcquisiPaths.DATASET] = sub_dataset
                base_entry.loc[slice_of_life & vidtype_filter, DataTags.VIDEO_ID] = base_entry.loc[slice_of_life & vidtype_filter, DataTags.VIDEO_ID].values[0] + "_stim_" + str(i)

                # Update the database with the new entry.
                new_entries = pd.concat([new_entries, base_entry], ignore_index=True)

            # Once we've completed this, drop the original row so we don't confuse things in the future.
            database.drop(database[slice_of_life & vidtype_filter].index, inplace=True)


    return dataset, new_entries




def load_dataset(video_path, mask_path=None, extra_metadata_path=None, dataset_metadata=None, stage=Stages.PREANALYSIS):

    mask_data = None
    metadata = dataset_metadata
    metadata[AcquisiPaths.VIDEO_PATH] = dataset_metadata[AcquisiPaths.DATA_PATH]
    metadata[AcquisiPaths.MASK_PATH] = mask_path
    metadata[AcquisiPaths.META_PATH] = extra_metadata_path

    if video_path.exists():
        resource = load_video(video_path, metadata.get(MetaTags.VIDEO, dict()).get(MetaTags.FIELDS_OF_INTEREST, None))

        video_data = resource.data

        if MetaTags.FRAMERATE not in metadata and MetaTags.FRAMERATE not in metadata.get(MetaTags.VIDEO, dict()):
            metadata[MetaTags.FRAMERATE] = resource.metadict.get(MetaTags.FRAMERATE)
        elif MetaTags.FRAMERATE in metadata.get(MetaTags.VIDEO, dict()):
            metadata[MetaTags.FRAMERATE] = metadata.get(MetaTags.VIDEO, None).get(MetaTags.FRAMERATE)
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
    if metadata.get(AcquisiPaths.IMAGE_PATH) and metadata.get(AcquisiPaths.IMAGE_PATH).exists():
        avg_image_data = cv2.imread(metadata.get(AcquisiPaths.IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    # For importing the query locations
    queryloc_data = []
    if AcquisiPaths.QUERYLOC_PATH in metadata and MetaTags.QUERY_LOCATIONS not in metadata:

        querylocs = metadata.get(AcquisiPaths.QUERYLOC_PATH)

        for locpath in querylocs:
            if locpath.exists():
                match locpath.suffix:
                    case ".csv":
                        queryloc_data.append(pd.read_csv(locpath, header=None, encoding="utf-8-sig").to_numpy())
                    case ".txt":
                        queryloc_data.append(pd.read_csv(locpath, sep=None, header=None, encoding="utf-8-sig").to_numpy())
                    case "":
                        if locpath.name == "All Pixels":
                            xm, ym = np.meshgrid(np.arange(video_data.shape[1]),
                                                 np.arange(video_data.shape[0]))

                            xm = np.reshape(xm, (xm.size, 1))
                            ym = np.reshape(ym, (ym.size, 1))

                            allcoord_data = np.hstack((xm, ym))

                            queryloc_data.append(allcoord_data)
            elif locpath.name == "All Pixels":
                xm, ym = np.meshgrid(np.arange(video_data.shape[1]),
                                     np.arange(video_data.shape[0]))

                xm = np.reshape(xm, (xm.size, 1))
                ym = np.reshape(ym, (ym.size, 1))

                allcoord_data = np.hstack((xm, ym))

                queryloc_data.append( allcoord_data)
            else:
                warnings.warn("Query location path does not exist: "+str(locpath))


    elif AcquisiPaths.QUERYLOC_PATH in metadata and MetaTags.QUERY_LOCATIONS in metadata and \
        metadata.get(MetaTags.QUERY_LOCATIONS).get(MetaTags.FIELDS_OF_INTEREST, None) is not None: # @TODO: Remove duplicate code.
        # If we have query locations defined in the metadata alongside fields to load, then that means we have specific fields
        # from a query location file that we need to grab.

        querylocs = metadata.get(AcquisiPaths.QUERYLOC_PATH)
        fieldnames = metadata.get(MetaTags.QUERY_LOCATIONS).get(MetaTags.FIELDS_OF_INTEREST, None)

        for locpath in querylocs:
            if locpath.exists():
                match locpath.suffix:
                    case ".csv":
                        queryloc_data.append(pd.read_csv(locpath, usecols=fieldnames, encoding="utf-8-sig").to_numpy())
                    case ".txt":
                        queryloc_data.append(pd.read_csv(locpath, usecols=fieldnames, encoding="utf-8-sig").to_numpy())
                    case "":
                        if locpath.name == "All Pixels":
                            xm, ym = np.meshgrid(np.arange(video_data.shape[1]),
                                                 np.arange(video_data.shape[0]))

                            xm = np.reshape(xm, (xm.size, 1))
                            ym = np.reshape(ym, (ym.size, 1))

                            allcoord_data = np.hstack((xm, ym))

                            queryloc_data.append(allcoord_data)
            elif locpath.name == "All Pixels":
                xm, ym = np.meshgrid(np.arange(video_data.shape[1]),
                                     np.arange(video_data.shape[0]))

                xm = np.reshape(xm, (xm.size, 1))
                ym = np.reshape(ym, (ym.size, 1))

                allcoord_data = np.hstack((xm, ym))

                queryloc_data.append(allcoord_data)
            else:
                warnings.warn("Query location path does not exist: " + str(locpath))
    elif MetaTags.QUERY_LOCATIONS in metadata:
        queryloc_data.append(metadata.get(MetaTags.QUERY_LOCATIONS))

    # For importing the framestamps of the video- these are the temporal, monotonic frame indexes of the video, in case frames were dropped
    # in the pipeline or in the processing stages
    stamps = metadata.get(MetaTags.FRAMESTAMPS)
    if stamps is None:
        stamps = np.arange(video_data.shape[-1])

    # For importing metadata RE: the stimulus delivery
    stimulus_sequence = None
    if AcquisiParams.STIMSEQ_PATH in metadata and MetaTags.STIMULUS_SEQ not in metadata and metadata.get(AcquisiParams.STIMSEQ_PATH, Path()).exists():
        stimulus_sequence = pd.read_csv(metadata.get(AcquisiParams.STIMSEQ_PATH), header=None, encoding="utf-8-sig").to_numpy()
    elif MetaTags.STIMULUS_SEQ in metadata and metadata.get(AcquisiParams.STIMSEQ_PATH, Path()).exists():
        stimulus_sequence = np.cumsum(np.array(metadata.get(MetaTags.STIMULUS_SEQ), dtype="int"))
    elif stage == Stages.ANALYSIS:
        while Dataset.stimseq_fName is None:
            Dataset.stimseq_fName = filedialog.askopenfilename(title="Stimulus sequence not detected in metadata. Select a stimulus sequence file.",
                                                               initialdir=metadata.get(AcquisiPaths.BASE_PATH, None),
                                                               filetypes=[("Stimulus Sequence files", "*.csv")])
            if Dataset.stimseq_fName is None or not Path(Dataset.stimseq_fName).exists():
                Dataset.stimseq_fName = None

        stimulus_sequence = np.cumsum(pd.read_csv(Dataset.stimseq_fName, header=None,encoding="utf-8-sig").to_numpy())

    dataset = None
    # Stimulus sequences should be arranged in 3s- that is, pre-stimulus, during-stimulus, and post-stimulus.
    # In a case where there is more than that, then cut up the video data into multiple sub-datasets for future processing.
    if stimulus_sequence is None or len(stimulus_sequence) == 3:
        dataset = [Dataset(video_data, mask_data, avg_image_data, metadata, queryloc_data, stamps, stimulus_sequence, stage)]
    elif len(stimulus_sequence) > 3:
        print(Fore.YELLOW + "Detected multiple stimuli in this dataset. Breaking into subdatasets for analysis...")
        dataset = []
        for i in range(0, len(stimulus_sequence)-2, 2):
            sub_seq = stimulus_sequence[i:i+3].copy()
            # Subtract the preceding sequence value from this sub-dataset, if available.
            if i!=0:
                seq_ind = np.arange(stimulus_sequence[i - 1], sub_seq[2])
                sub_seq -= stimulus_sequence[i - 1]
            else:
                seq_ind = np.arange(0, sub_seq[2])

            subset_stamps, _, std_indices = np.intersect1d(seq_ind, stamps, return_indices=True)
            # Subtract the first framestamp from this sub-dataset, its now frame 0.
            subset_stamps -= subset_stamps[0]

            dataset.append(Dataset(video_data[..., std_indices],
                                   mask_data[..., std_indices],
                                   avg_image_data, metadata, queryloc_data, subset_stamps, sub_seq, stage))

    elif len(stimulus_sequence) < 3:
        warnings.warn("Stimulus sequences must be interpretable in sets of 3, in pre-stim, stim, post-stim format. Unable to analyze dataset.")
        return None

    return dataset


def preprocess_dataset(dataset, params, reference_dataset=None):

    if reference_dataset is None:
        reference_dataset = dataset

    # First do custom preprocessing steps, e.g. things implemented expressly for and by the OCVL
    # If you would like to "bake in" custom pipeline steps, please contact the OCVL using GitHub's Issues
    # or submit a pull request.
    custom_steps = params.get(PreAnalysisPipeline.CUSTOM)
    if custom_steps is not None:
        # For "baked in" dewarping- otherwise, data is expected to be dewarped already
        pre_dewarp = custom_steps.get("dewarp")
        match pre_dewarp:
            case "ocvl":
                # Framestamps from the MEAO AOSLO  start at 1, instead of 0- shift to fix.
                dataset.framestamps = dataset.framestamps-1

                if dataset.metadata[AcquisiPaths.META_PATH] is not None:
                    dat_metadata = pd.read_csv(dataset.metadata[AcquisiPaths.META_PATH], encoding="utf-8-sig",
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
                                                                                yshifts, xshifts,
                                                                                fitshifts=True)

                    # Dewarp our mask too.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(action="ignore", message="invalid value encountered in cast")

                        for f in range(dataset.num_frames):
                            norm_frame = dataset.mask_data[..., f].astype("float32")
                            norm_frame[norm_frame == 0] = np.nan

                            dataset.mask_data[..., f] = cv2.remap(norm_frame,
                                                                  map_mesh_x, map_mesh_y,
                                                                  interpolation=cv2.INTER_NEAREST)
            case "demotion":
                if dataset.metadata[AcquisiPaths.META_PATH] is not None:
                    # Temp until a better way of finding dmp files is found.
                    #dumpfile = dataset.metadata[AcquisiTags.META_PATH].stem[0:-len("_acceptable_frames")]+".dmp"
                    with open(dataset.metadata[AcquisiPaths.META_PATH], "rb") as file:
                        pick = pickle.load(file, encoding='latin1')

                        ff_translation_info_rowshift = pick['full_frame_ncc']['row_shifts']
                        ff_translation_info_colshift = pick['full_frame_ncc']['column_shifts']
                        strip_translation_info = pick['sequence_interval_data_list']

                        minmaxpix = np.empty([1, 2])

                        for frame in strip_translation_info:
                            for frame_contents in frame:
                                ref_pixels = frame_contents['slow_axis_pixels_in_current_frame_interpolated']
                                minmaxpix = np.append(minmaxpix, [[ref_pixels[0], ref_pixels[-1]]], axis=0)

                        minmaxpix = minmaxpix[1:, :]
                        topmostrow = minmaxpix[:, 0].max()
                        bottommostrow = minmaxpix[:, 1].min()

                        # print np.array([pick['strip_cropping_ROI_2'][-1]])
                        # The first row is the crop ROI.

                        slow_axis_array = np.zeros([len(strip_translation_info), 1000], dtype=int)
                        unaligned_col_shifts = np.zeros([len(strip_translation_info), 1000])
                        unaligned_row_shifts = np.zeros([len(strip_translation_info), 1000])
                        frame_inds = np.full( [len(strip_translation_info)], -1, dtype=int)

                        for i, frame in enumerate(strip_translation_info):
                            if len(frame) > 0:
                                #print("************************ Frame " + str(frame[0]['frame_index']) + "************************")
                                # print "Adjusting the rows...."
                                frame_inds[i] = frame[0]['frame_index']
                                slow_axis_pixels = np.zeros([1])
                                frame_col_shifts = np.zeros([1])
                                frame_row_shifts = np.zeros([1])

                                for frame_contents in frame:
                                    slow_axis_pixels = np.append(slow_axis_pixels,
                                                                 frame_contents['slow_axis_pixels_in_reference_frame'])

                                    ff_row_shift = ff_translation_info_rowshift[frame_inds[i]]
                                    ff_col_shift = ff_translation_info_colshift[frame_inds[i]]

                                    # First set the relative shifts
                                    row_shift = (np.subtract(frame_contents['slow_axis_pixels_in_reference_frame'],
                                                             frame_contents['slow_axis_pixels_in_current_frame_interpolated']))
                                    col_shift = (frame_contents['fast_axis_pixels_in_reference_frame_interpolated'])

                                    # These will contain all of the motion, not the relative motion between the aligned frames-
                                    # So then subtract the full frame row shift
                                    row_shift = np.add(row_shift, ff_row_shift)
                                    col_shift = np.add(col_shift, ff_col_shift)
                                    frame_col_shifts = np.append(frame_col_shifts, col_shift)
                                    frame_row_shifts = np.append(frame_row_shifts, row_shift)

                                slow_axis_pixels = slow_axis_pixels[1:]
                                frame_col_shifts = frame_col_shifts[1:]
                                frame_row_shifts = frame_row_shifts[1:]

                                slow_axis_array[i, 0:len(slow_axis_pixels)] = slow_axis_pixels.astype(int)
                                unaligned_col_shifts[i, 0:len(frame_col_shifts)] = frame_col_shifts
                                unaligned_row_shifts[i, 0:len(frame_row_shifts)] = frame_row_shifts

                        # Reshuffle the above so its actually in the order that they appear in the video. Wild, I know.
                        resort_args = np.argsort(frame_inds)
                        frame_inds = frame_inds[resort_args]
                        slow_axis_array = slow_axis_array[resort_args, :]
                        unaligned_col_shifts = unaligned_col_shifts[resort_args, :]
                        unaligned_row_shifts = unaligned_row_shifts[resort_args, :]

                        # Then, remove frame inds that weren't filled do to a variety of reasons.
                        slow_axis_array = slow_axis_array[frame_inds >= 0, :]
                        unaligned_col_shifts = unaligned_col_shifts[frame_inds >= 0, :]
                        unaligned_row_shifts = unaligned_row_shifts[frame_inds >= 0, :]
                        frame_inds = frame_inds[frame_inds >= 0]


                        # Update our framestamps with the sorted indexes
                        dataset.framestamps = frame_inds

                        # Find the ROI associated with this particular dataset.
                        roi = np.array([])
                        height = 0
                        for i in range(len(pick['strip_cropping_ROI_2'])):
                            roi = pick['strip_cropping_ROI_2'][i]
                            if np.all( dataset.avg_image_data.shape == np.array([roi[1]-roi[0], roi[3]-roi[2]]) ):
                                height = roi[1]-roi[0]
                                break

                        ref_max_slow_axis = np.nanmax(slow_axis_array[0,:])
                        ref_min_slow_axis = np.nanmin(slow_axis_array[0, :])

                        max_slow_axis = np.amax(slow_axis_array)
                        min_slow_axis = np.amin(slow_axis_array)

                        slow_axis_size = max_slow_axis - min_slow_axis + 1
                        slow_axis_array -= min_slow_axis

                        colshifts = dict()
                        rowshifts = dict()
                        longest_colshifts = 0
                        longest_rowshifts = 0
                        # Each row in these arrays are a frame
                        for frame_ind in range(slow_axis_array.shape[0]):

                            # Find the last element index (will be the max)
                            maxind = np.argmax(slow_axis_array[frame_ind, :])

                            # Append the shift values for that particular location to the shift
                            for i in range(maxind):
                                if slow_axis_array[frame_ind, i] not in colshifts:
                                    colshifts[slow_axis_array[frame_ind, i]] = np.array([])
                                    rowshifts[slow_axis_array[frame_ind, i]] = np.array([])

                                colshifts[slow_axis_array[frame_ind, i]] = np.hstack((colshifts[slow_axis_array[frame_ind, i]],
                                                                                        unaligned_col_shifts[frame_ind, i]))

                                rowshifts[slow_axis_array[frame_ind, i]] = np.hstack((rowshifts[slow_axis_array[frame_ind, i]],
                                                                                        unaligned_row_shifts[frame_ind, i]))


                        all_colshifts = np.full((np.amax(slow_axis_array), slow_axis_array.shape[0]), np.nan)
                        all_rowshifts = np.full((np.amax(slow_axis_array), slow_axis_array.shape[0]), np.nan)

                        for row, shifts in colshifts.items():
                            all_colshifts[row, 0:len(shifts)] = shifts
                        for row, shifts in rowshifts.items():
                            all_rowshifts[row, 0:len(shifts)] = shifts

                        del colshifts, rowshifts

                        all_colshifts = all_colshifts[roi[0]:roi[1], :].T
                        all_rowshifts = all_rowshifts[roi[0]:roi[1], :].T

                        # Determine the residual error in our dewarping, and obtain the maps
                        dataset.video_data, map_mesh_x, map_mesh_y = dewarp_2D_data(dataset.video_data,
                                                                                    -all_rowshifts, -all_colshifts)

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
    trim = params.get(PreAnalysisPipeline.TRIM)
    if trim is not None:
        start_frm = int(trim.get("start_frm",0))
        end_frm = int(trim.get("end_frm",-1))
        if end_frm == -1:
            end_frm = np.amax(dataset.framestamps)+1
        goodinds = np.argwhere((dataset.framestamps <= end_frm) & (dataset.framestamps >= start_frm)).flatten()
        dataset.framestamps = dataset.framestamps[goodinds]
        dataset.video_data = dataset.video_data[..., goodinds]
        dataset.mask_data = dataset.mask_data[..., goodinds]
        dataset.num_frames = dataset.video_data.shape[-1]


    align_dat = reference_dataset.video_data.copy()
    mask_dat = reference_dataset.mask_data.copy()
    # Try and figure out the reference frame, if we haven't already.
    # This is based on the simple method of determining the maximum area subtended by our mask.
    if dataset.reference_frame_idx is None or dataset.reference_frame_idx >= dataset.num_frames:
        amt_data = np.zeros([dataset.num_frames])
        for f in range(dataset.num_frames):
            amt_data[f] = mask_dat[..., f].flatten().sum()

        dataset.reference_frame_idx = amt_data.argmax()
        del amt_data

    # Flat field the video for alignment, if desired.
    if params.get(PreAnalysisPipeline.FLAT_FIELD, False):
        align_dat = flat_field(align_dat, mask_dat)

    # Gaussian blur the data first before aligning, if requested
    gausblur = params.get(PreAnalysisPipeline.GAUSSIAN_BLUR, 0.0)
    if gausblur is not None and gausblur != 0.0:
        for f in range(align_dat.shape[-1]):
            align_dat[..., f] = gaussian_filter(align_dat[..., f], sigma=gausblur)
        align_dat *= mask_dat

    # Then crop the data, if requested
    mask_roi = params.get(PreAnalysisPipeline.MASK_ROI)
    if mask_roi is not None:
        r = mask_roi.get("r")
        c = mask_roi.get("c")
        width = mask_roi.get("width")
        height = mask_roi.get("height")

        # Everything outside the roi should be cropped
        align_dat = align_dat[r:r + height, c:c + width, :]
        mask_dat = mask_dat[r:r + height, c:c + width, :]


    # Finally, correct for residual torsion if requested
    correct_torsion = params.get(PreAnalysisPipeline.CORRECT_TORSION)
    if correct_torsion is not None and correct_torsion:
        align_dat, xforms, inliers, mask_dat = optimizer_stack_align(align_dat, mask_dat,
                                                                     reference_idx=dataset.reference_frame_idx,
                                                                     determine_initial_shifts=False,
                                                                     dropthresh=0, justalign=True,
                                                                     transformtype=params.get(PreAnalysisPipeline.INTRA_STACK_XFORM, "rigid"))

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


def postprocess_dataset(dataset, analysis_params, result_folder, debug_params):

    norm_params = analysis_params.get(NormParams.NAME, dict())
    norm_scope = norm_params.get(NormParams.SCOPE, "frame") # Default: Standardizes the video to each frame's mean and stddev
    norm_method = norm_params.get(NormParams.METHOD, "score")  # Default: Standardizes the video to a unit mean and stddev
    rescale_norm = norm_params.get(NormParams.RESCALED, True)  # Default: Rescales the data back into AU to make results easier to interpret
    res_mean = norm_params.get(NormParams.MEAN, 70.0)  # Default: Rescales to a mean of 70 - these values are based on "ideal" datasets
    res_stddev = norm_params.get(NormParams.STD, 35.0)  # Default: Rescales to a std dev of 35

    # Flat field the video for analysis if desired.
    if analysis_params.get(Analysis.FLAT_FIELD, False):
        dataset.video_data = flat_field(dataset.video_data, dataset.mask_data)

    # Gaussian blur the data first before analysis, if requested
    gausblur = analysis_params.get(Analysis.GAUSSIAN_BLUR, 0.0)
    if gausblur is not None and gausblur != 0.0:
        for f in range(dataset.video_data.shape[-1]):
            dataset.video_data[..., f] = gaussian_filter(dataset.video_data[..., f], sigma=gausblur)
        dataset.video_data *= dataset.mask_data

    # Normalize the video to reduce the influence of framewide intensity changes, if requested.
    if norm_scope == "frame" and norm_method != "none":
        dataset.video_data = norm_video(dataset.video_data, norm_method=norm_method,
                                        rescaled=rescale_norm,
                                        rescale_mean=res_mean, rescale_std=res_stddev)

    if debug_params.get(DebugParams.OUTPUT_NORM_VIDEO, False):
        result_folder.mkdir(parents=True, exist_ok=True)
        save_tiff_stack(result_folder.joinpath(dataset.video_path.stem + "_" + norm_method + "_norm.tif"),
                        dataset.video_data)

    return dataset


class Dataset:
    # Static var used on such occasions that a user doesn't provide an explicit method determining the stimulus sequence
    # pattern (either via a metadata file, or via explicitly setting it in the json file)
    stimseq_fName = None

    def __init__(self, video_data=None, mask_data=None, avg_image_data=None, metadata=None, query_locations=[],
                 framestamps=None, stimseq=None, stage=Stages.PREANALYSIS):

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

        self.stimtrain_path = self.metadata.get(AcquisiParams.STIMSEQ_PATH)
        self.video_path = self.metadata.get(AcquisiPaths.VIDEO_PATH)
        self.mask_path = self.metadata.get(AcquisiPaths.MASK_PATH)
        self.base_path = self.metadata.get(AcquisiPaths.BASE_PATH)

        self.prefix = self.metadata.get(AcquisiParams.PREFIX)

        if self.video_path:
            # If we don't have supplied definitions of the base path of the dataset or the filename prefix,
            # then guess.
            if not self.base_path:
                self.base_path = self.video_path.parent
            if not self.prefix:
                self.prefix = self.video_path.with_suffix("")

            self.image_path = self.metadata.get(AcquisiPaths.IMAGE_PATH, None)
            # If we don't have supplied definitions of the image associated with this dataset,
            # then guess.
            if self.image_path is None:
                imname = None
                if (stage is Stages.PREANALYSIS or stage is Stages.ANALYSIS) and \
                        self.prefix.with_name(self.prefix.name + ".tif").exists():

                        imname = self.prefix.with_name(self.prefix.name + ".tif")

                if not imname:
                    for filename in self.base_path.glob("*_ALL_ACQ_AVG.tif"):
                        imname = filename

                if not imname: # and stage is Stages.ANALYSIS:
                    warnings.warn("Unable to detect viable average image file; generating one from video. Dataset functionality may be limited.")
                    self.image_path = None
                    self.avg_image_data, _ = weighted_z_projection(self.video_data, self.mask_data)
                else:
                    print(Fore.YELLOW + "Automatically detected the average image "+ str(imname.name) +". **Please verify your image format string**: " )
                    self.image_path = imname
                    self.metadata[AcquisiPaths.IMAGE_PATH] = self.image_path
                    self.avg_image_data = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

            self.query_coord_paths = self.metadata.get(AcquisiPaths.QUERYLOC_PATH)
            # If we don't have query locations associated with this dataset, then try to guess.
            if self.query_coord_paths is None and not query_locations:

                coordname = None
                if (stage is Stages.PREANALYSIS or stage is Stages.ANALYSIS) and \
                        self.prefix.with_name(self.prefix.name + "_coords.csv").exists():

                        coordname = self.prefix.with_name(self.prefix.name + "_coords.csv")

                    # If we don't have an image specific to this dataset, search for the all acq avg
                if coordname is None:
                    for filename in self.base_path.glob("*_ALL_ACQ_AVG_coords.csv"):
                            coordname = filename

                if coordname is None and stage is Stages.ANALYSIS:
                    print(Fore.YELLOW+"Unable to detect viable query location file for dataset at: "+ str(self.video_path) +". Dataset is either a control, or will be converted to a pixelwise analysis.")
                    self.metadata[AcquisiPaths.QUERYLOC_PATH] = []
                    self.query_coord_paths = []
                elif stage is Stages.ANALYSIS:
                    print(Fore.YELLOW+"Automatically detected the query locations: "+str(coordname.name) + ". **Please verify your queryloc format string**" )
                    # Update our metadata structure, and our internally stored query coord paths.
                    self.metadata[AcquisiPaths.QUERYLOC_PATH] = [coordname]
                    self.query_coord_paths = [coordname]
                    match self.query_coord_paths[0].suffix:
                        case ".csv":
                            query_locations = [pd.read_csv(self.query_coord_paths[0], header=None, encoding="utf-8-sig").to_numpy()]
                        case ".txt":
                            query_locations = [pd.read_csv(self.query_coord_paths[0], sep=None, header=None, encoding="utf-8-sig").to_numpy()]

            self.query_loc = []
            for queries in query_locations:
                _, idx = np.unique(queries, return_index=True, axis=0)

                self.query_loc.append(queries[np.sort(idx)])

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


#  Copyright (c) 2021. Robert F Cooper
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import json
from enum import StrEnum
from importlib.metadata import metadata
from itertools import repeat
from logging import warning
from multiprocessing import pool
from pathlib import Path, PurePath

import cv2
import numpy as np
import multiprocessing as mp
import os
from os import walk
from os.path import splitext
from tkinter import *
from tkinter import filedialog, simpledialog
from tkinter import ttk

import pandas
from scipy.ndimage import binary_dilation, gaussian_filter
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from ocvl.function.preprocessing.improc import flat_field, weighted_z_projection, simple_image_stack_align, \
    optimizer_stack_align, dewarp_2D_data, simple_dataset_list_align
from ocvl.function.utility.format_parser import FormatParser
from ocvl.function.utility.generic import Dataset, PipeStages, initialize_and_load_dataset, AcquisiTags
from ocvl.function.utility.json_format_constants import FormatTypes, DataTags, MetaTags
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_video
import parse

from ocvl.function.utility.temporal_signal_utils import trim_video


class PipelineParams(StrEnum):
    GAUSSIAN_BLUR = "gaus_blur",
    MASK_ROI = "mask_roi",
    TRIM = "trim",
    MODALITIES = "modalities",
    CORRECT_TORSION = "correct_torsion",
    CUSTOM = "custom"
    OUTPUT_FOLDER = "output_folder"
    GROUP_BY = "group_by"


# Need to try this:
# https://mathematica.stackexchange.com/questions/199928/removing-horizontal-noise-artefacts-from-a-sem-image

def extract_and_parse_metadata(config_json_path, pName):
    with open(json_fName, 'r') as config_json_path:
        dat_form = json.load(config_json_path)

        allFilesColumns = [AcquisiTags.DATASET, AcquisiTags.DATA_PATH, FormatTypes.FORMAT]
        allFilesColumns.extend([d.value for d in DataTags])

        processed_dat_format = dat_form.get("processed")
        if processed_dat_format is not None:

            pipeline_params = processed_dat_format.get("pipeline_params")

            im_form = processed_dat_format.get(FormatTypes.IMAGE)
            vid_form = processed_dat_format.get(FormatTypes.VIDEO)
            mask_form = processed_dat_format.get(FormatTypes.MASK)

            metadata_form = None
            metadata_params = None
            if processed_dat_format.get(MetaTags.METATAG) is not None:
                metadata_params = processed_dat_format.get(MetaTags.METATAG)
                metadata_form = metadata_params.get(FormatTypes.METADATA)

            if vid_form is not None:

                # Grab our extensions, make sure to check them all.
                all_ext = (vid_form[vid_form.rfind(".", -5, -1):],)
                all_ext = all_ext + (mask_form[mask_form.rfind(".", -5, -1):],) if mask_form and mask_form[
                                                                                                 mask_form.rfind(".",
                                                                                                                 -5,
                                                                                                                 -1):] not in all_ext else all_ext
                all_ext = all_ext + (im_form[im_form.rfind(".", -5, -1):],) if im_form and im_form[
                                                                                           im_form.rfind(".", -5,
                                                                                                         -1):] not in all_ext else all_ext
                all_ext = all_ext + (
                metadata_form[metadata_form.rfind(".", -5, -1):],) if metadata_form and metadata_form[
                                                                                        metadata_form.rfind(".", -5,
                                                                                                            -1):] not in all_ext else all_ext

                # Construct the parser we'll use for each of these forms
                parser = FormatParser(vid_form, mask_form, im_form, metadata_form)

                # Parse out the locations and filenames, store them in a hash table by location.
                searchpath = Path(pName)
                allFiles = list()
                for ext in all_ext:
                    for path in searchpath.glob("*" + ext):
                        format_type, file_info = parser.parse_file(path.name)
                        file_info[DataTags.FORMAT_TYPE] = format_type
                        file_info[AcquisiTags.DATA_PATH] = path
                        file_info[AcquisiTags.BASE_PATH] = path.parent
                        file_info[AcquisiTags.DATASET] = None
                        entry = pd.DataFrame.from_dict([file_info])

                        allFiles.append(entry)

                return dat_form, pd.concat(allFiles, ignore_index=True)
            else:
                warning("Unable to detect \"processed\" json value!")
                return None
        else:
            return None


def preprocess_dataset(dataset, pipeline_params):
    # First do custom preprocessing steps, e.g. things implemented expressly for and by the OCVL
    # If you would like to "bake in" custom pipeline steps, please contact the OCVL using GitHub's Issues
    # or submit a pull request.
    custom_steps = pipeline_params.get(PipelineParams.CUSTOM)
    if custom_steps is not None:
        # For "baked in" dewarping- otherwise, data is expected to be dewarped already
        pre_dewarp = custom_steps.get("dewarp")
        match pre_dewarp:
            case "ocvl":
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
                    for f in range(dataset.num_frames):
                        # norm_frame = dataset.video_data[..., f].astype("float32") / dataset.video_data[..., f].max()
                        # norm_frame[norm_frame == 0] = np.nan

                        dataset.mask_data[..., f] = cv2.remap(dataset.mask_data[..., f],
                                                              map_mesh_x, map_mesh_y,
                                                              interpolation=cv2.INTER_NEAREST)

    # Trim the video down to a smaller/different size, if desired.
    trim = pipeline_params.get(PipelineParams.TRIM)
    if trim is not None:
        start_idx = trim.get("start_idx")
        end_idx = trim.get("end_idx")
        dataset.video_data = dataset.video_data[..., start_idx:end_idx]
        dataset.mask_data = dataset.mask_data[..., start_idx:end_idx]
        dataset.num_frames = dataset.video_data.shape[-1]
        dataset.framestamps = dataset.framestamps[(dataset.framestamps < end_idx) & (dataset.framestamps >= start_idx)]

    align_dat = dataset.video_data
    mask_dat = dataset.mask_data
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
    gausblur = pipeline_params.get(PipelineParams.GAUSSIAN_BLUR)
    if gausblur is not None and gausblur != 0.0:
        align_dat = gaussian_filter(align_dat, sigma=gausblur)
        align_dat *= mask_dat

    # Then crop the data, if requested
    mask_roi = pipeline_params.get(PipelineParams.MASK_ROI)
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
    correct_torsion = pipeline_params.get(PipelineParams.CORRECT_TORSION)
    if correct_torsion is not None and correct_torsion:
        align_dat, xforms, inliers, mask_dat = optimizer_stack_align(align_dat, mask_dat,
                                                                     reference_idx=dataset.reference_frame_idx,
                                                                     dropthresh=0, justalign=True)

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

                dataset.mask_data[..., f] = np.isfinite(norm_frame).astype(
                    og_dtype)  # Our new mask corresponds to the real data.
                norm_frame[np.isnan(norm_frame)] = 0  # Make anything that was nan into a 0, to be kind to non nan-types
                dataset.video_data[..., f] = norm_frame.astype(og_dtype)

        dataset.avg_image_data, awp = weighted_z_projection(dataset.video_data, dataset.mask_data)

    return dataset

if __name__ == "__main__":

    root = Tk()
    root.lift()
    w = 1
    h = 1
    x = root.winfo_screenwidth() / 4
    y = root.winfo_screenheight() / 4
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.

    pName = filedialog.askdirectory(title="Select the folder containing all videos of interest.", parent=root)
    #pName = "P:\\RFC_Projects\\F-Cell_Generalization_Test_Data"
    if not pName:
        quit()

    x = root.winfo_screenwidth() / 2 - 128
    y = root.winfo_screenheight() / 2 - 128
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.
    root.update()


   # json_fName = filedialog.askopenfilename(title="Select the parameter json file.", parent=root)
    json_fName = "C:\\Users\\cooperro\\Documents\\F-Cell\\pyORG_Calculation\\config_files\\meao.json"
    if not json_fName:
        quit()

    with mp.Pool(processes=int(np.round(mp.cpu_count()/2 ))) as pool:

        dat_form, allData = extract_and_parse_metadata(json_fName, pName)

        processed_dat_format = dat_form.get("processed")
        pipeline_params = processed_dat_format.get("pipeline_params")
        modes_of_interest = pipeline_params.get(PipelineParams.MODALITIES)

        metadata_params = None
        if processed_dat_format.get(MetaTags.METATAG) is not None:
            metadata_params = processed_dat_format.get(MetaTags.METATAG)
            metadata_form = metadata_params.get(FormatTypes.METADATA)

        acquisition = dict()

        # Group files together based on location, modality, and video number
        # If we've selected modalities of interest, only process those; otherwise, process them all.
        if modes_of_interest is None:
            modes_of_interest = allData[DataTags.MODALITY].unique().tolist()

        for mode in modes_of_interest:
            modevids = allData.loc[allData[DataTags.MODALITY] == mode]

            vidnums = np.unique(modevids[DataTags.VIDEO_ID].to_numpy())
            for num in vidnums:
                # Find the rows associated with this video number, and
                # extract the rows corresponding to this acquisition.
                acquisition = modevids.loc[modevids[DataTags.VIDEO_ID] == num]

                if (acquisition[DataTags.FORMAT_TYPE] == FormatTypes.MASK).sum() <= 1 and \
                        (acquisition[DataTags.FORMAT_TYPE] == FormatTypes.METADATA).sum() <= 1 and \
                        (acquisition[DataTags.FORMAT_TYPE] == FormatTypes.VIDEO).sum() == 1:

                    video_info = acquisition.loc[acquisition[DataTags.FORMAT_TYPE] == FormatTypes.VIDEO]
                    mask_info = acquisition.loc[acquisition[DataTags.FORMAT_TYPE] == FormatTypes.MASK]
                    metadata_info = acquisition.loc[acquisition[DataTags.FORMAT_TYPE] == FormatTypes.METADATA]
                    im_info = acquisition.loc[acquisition[DataTags.FORMAT_TYPE] == FormatTypes.IMAGE]

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
                    if not metadata_info.empty and metadata_params is not None:
                        if metadata_info.at[metadata_info.index[0], AcquisiTags.DATA_PATH].exists():
                            metadata_path = metadata_info.at[metadata_info.index[0], AcquisiTags.DATA_PATH]
                            metatype = metadata_params.get(MetaTags.TYPE)
                            loadfields = metadata_params.get(MetaTags.FIELDS_OF_INTEREST)

                            if metatype == "text_file":
                                dat_metadata = pd.read_csv(metadata_info.at[metadata_info.index[0], AcquisiTags.DATA_PATH], encoding="utf-8-sig", skipinitialspace=True)

                                for field, column in loadfields.items():
                                    meta_fields[field] = dat_metadata[column].to_numpy()
                            elif metatype == "database":
                                pass
                            elif metatype == "mat_file":
                                pass

                        else:
                            metadata_path = None
                    else:
                        metadata_path = None

                    # Take metadata gleaned from our filename, as well as our metadata files,
                    # and combine them into a single dictionary.
                    combined_meta_dict = video_info.squeeze().to_dict() | meta_fields

                    # Add paths to things that we may want, depending on the stage we're at.
                    if not im_info.empty:
                        combined_meta_dict[AcquisiTags.IMAGE_PATH] = im_info.at[im_info.index[0], AcquisiTags.DATA_PATH]

                    if not mask_info.empty:
                        mask_path = mask_info.at[mask_info.index[0],AcquisiTags.DATA_PATH]
                    else:
                        mask_path = None


                    if not video_info.empty:
                        print("Initializing and loading dataset: "+ video_info.at[video_info.index[0],AcquisiTags.DATA_PATH].name)
                        dataset = initialize_and_load_dataset(video_info.at[video_info.index[0],AcquisiTags.DATA_PATH],
                                                              mask_path,
                                                              metadata_path,
                                                              combined_meta_dict)

                        # Run the processing pipeline on this dataset, with params specified by the json.
                        # When done, put it into the database.
                        allData.loc[video_info.index, AcquisiTags.DATASET] = preprocess_dataset(dataset, pipeline_params)

                    else:
                        warning("Unable to find a video path specified for vidnum: "+num)
                else:
                    warning("Detected more than one video or mask associated with vidnum: "+num)


        # Remove all entries without associated datasets.
        allData.drop(allData[allData[AcquisiTags.DATASET].isnull()].index, inplace=True)

        grouping = pipeline_params.get(PipelineParams.GROUP_BY)
        if grouping is not None:
            for row in allData.itertuples():
                print( grouping.format_map(row._asdict()) )
                allData.loc[row.Index, PipelineParams.GROUP_BY] = grouping.format_map(row._asdict())

            groups = allData[PipelineParams.GROUP_BY].unique().tolist()
        else:
            groups =[""] # If we don't have any groups, then just make the list an empty string.

        for group in groups:
            if group != "":
                group_datasets = allData.loc[allData[PipelineParams.GROUP_BY] == group]
            else:
                group_datasets = allData

            for mode in modes_of_interest:
                modevids = group_datasets.loc[group_datasets[DataTags.MODALITY] == mode]

                vidnums = modevids[DataTags.VIDEO_ID].to_numpy()
                datasets = modevids[AcquisiTags.DATASET].to_list()
                avg_images = np.dstack([data.avg_image_data for data in datasets])

                print("Selecting ideal central frame for mode and location: "+mode)

                dist_res = pool.starmap_async(simple_image_stack_align, zip(repeat(avg_images),
                                                                            repeat(None),
                                                                            np.arange(len(datasets)) ))
                shift_info = dist_res.get()

                # Determine the average
                avg_loc_dist = np.zeros(len(shift_info))
                f = 0
                for allshifts in shift_info:
                    allshifts = np.stack(allshifts)
                    allshifts **= 2
                    allshifts = np.sum(allshifts, axis=1)
                    avg_loc_dist[f] = np.mean(np.sqrt(allshifts))  # Find the average distance to this reference.
                    f += 1

                avg_loc_idx = np.argsort(avg_loc_dist)
                dist_ref_idx = avg_loc_idx[0]

                print("Determined most central dataset with video number: " + str(vidnums[dist_ref_idx]) + ".")

                central_dataset = datasets[dist_ref_idx]

                # Align the stack of average images from all datasets
                avg_images, ref_xforms, inliers, avg_masks  = optimizer_stack_align(avg_images,
                                                                                    (avg_images > 0),
                                                                                    dist_ref_idx,
                                                                                    determine_initial_shifts=True,
                                                                                    dropthresh=0.0, transformtype="affine")

                # Z Project each of our image types
                avg_avg_images, avg_avg_mask = weighted_z_projection(avg_images)

                # Save the (now pipelined) datasets. First, we need to figure out if the user has a preferred
                # pipeline filename structure.
                output_folder = pipeline_params.get(PipelineParams.OUTPUT_FOLDER)
                if output_folder is None:
                    output_folder = PurePath("Functional Pipeline").joinpath(group)
                else:
                    output_folder = PurePath(output_folder).joinpath(group)

                # Determine the filename for the superaverage using the central-most dataset.
                pipelined_dat_format = dat_form.get("pipelined")
                if pipelined_dat_format is not None:
                    pipe_im_form = pipelined_dat_format.get(FormatTypes.IMAGE)
                    if pipe_im_form is not None:
                        pipe_im_fname = pipe_im_form.format_map(central_dataset.metadata)

                # Make sure our output folder exists.
                central_dataset.metadata[AcquisiTags.BASE_PATH].joinpath(output_folder).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(central_dataset.metadata[AcquisiTags.BASE_PATH].joinpath(output_folder, pipe_im_fname),
                            avg_avg_images)
                save_video(central_dataset.metadata[AcquisiTags.BASE_PATH].joinpath(output_folder, Path(pipe_im_fname).with_suffix(".avi")),
                           avg_images, 1)

                print("Outputting data...")
                for dataset, xform in zip(datasets, ref_xforms):

                    # Make sure our output folder exists.
                    dataset.metadata[AcquisiTags.BASE_PATH].joinpath(output_folder).mkdir(parents=True, exist_ok=True)

                    (rows, cols) = dataset.video_data.shape[0:2]

                    if pipelined_dat_format is not None:
                        pipe_vid_form = pipelined_dat_format.get(FormatTypes.VIDEO)
                        pipe_mask_form = pipelined_dat_format.get(FormatTypes.MASK)
                        pipe_meta_form = pipelined_dat_format.get(MetaTags.METATAG)

                        if pipe_vid_form is not None:
                            pipe_vid_fname = pipe_vid_form.format_map(dataset.metadata)
                        if pipe_mask_form is not None:
                            pipe_mask_fname = pipe_mask_form.format_map(dataset.metadata)
                        if pipe_meta_form is not None:
                            pipe_meta_form = pipe_meta_form.get(FormatTypes.METADATA)
                            if pipe_meta_form is not None:
                                pipe_meta_fname = pipe_meta_form.format_map(dataset.metadata)


                    og_dtype = dataset.video_data.dtype
                    for i in range(dataset.num_frames):  # Make all of the data in our dataset relative as well.
                        tmp = dataset.video_data[..., i].astype("float32")
                        tmp[np.round(tmp) == 0] = np.nan
                        tmp = cv2.warpAffine(tmp, xform,(cols, rows),
                                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                        tmp[np.isnan(tmp)] = 0
                        dataset.video_data[..., i] = tmp.astype(og_dtype)

                        tmp = dataset.mask_data[..., i].astype("float32")
                        tmp[np.round(tmp) == 0] = np.nan
                        tmp = cv2.warpAffine(tmp, xform,(cols, rows),
                                             flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
                        tmp[np.isnan(tmp)] = 0
                        dataset.mask_data[..., i] = tmp.astype(og_dtype)

                    out_meta = pd.DataFrame(dataset.framestamps, columns=["FrameStamps"])
                    out_meta.to_csv(dataset.metadata[AcquisiTags.BASE_PATH].joinpath(output_folder, pipe_meta_fname), index=False)
                    save_video(dataset.metadata[AcquisiTags.BASE_PATH].joinpath(output_folder, pipe_vid_fname), dataset.video_data,
                               framerate=dataset.framerate)




    print("PK FIRE")


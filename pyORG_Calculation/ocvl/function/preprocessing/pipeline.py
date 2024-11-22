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
import os
from enum import StrEnum
from itertools import repeat
from logging import warning
from pathlib import Path, PurePath

import cv2
import numpy as np
import multiprocessing as mp
from tkinter import *
from tkinter import filedialog, ttk

import pandas as pd


from ocvl.function.preprocessing.improc import weighted_z_projection, simple_image_stack_align, \
    optimizer_stack_align
from ocvl.function.utility.dataset import extract_and_parse_metadata, initialize_and_load_dataset, \
    preprocess_dataset

from ocvl.function.utility.json_format_constants import FormatTypes, DataTags, MetaTags, PipelineParams, AcquisiTags
from ocvl.function.utility.resources import save_video


# Need to try this:
# https://mathematica.stackexchange.com/questions/199928/removing-horizontal-noise-artefacts-from-a-sem-image

if __name__ == "__main__":

    root = Tk()
    root.lift()
    w = 256
    h = 128
    x = root.winfo_screenwidth() / 4
    y = root.winfo_screenheight() / 4
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.

    pName = filedialog.askdirectory(title="Select the folder containing all videos of interest.", parent=root)
    #pName = "P:\\RFC_Projects\\F-Cell_Generalization_Test_Data"
    if not pName:
        quit()


    root.update()

    config_path = Path(os.path.dirname(__file__)).parent.parent.parent.joinpath("config_files")

    config_files = [filepath.name for filepath in config_path.glob("*.json")]

    # combo = ttk.Combobox(root, values=config_files)
    # combo.current(1)
    # combo.pack()
    # root.mainloop()
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

                        # Run the preprocessing pipeline on this dataset, with params specified by the json.
                        # When done, put it into the database.
                        print("Preprocessing dataset...")
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


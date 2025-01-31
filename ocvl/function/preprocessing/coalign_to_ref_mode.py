import datetime
import json
import os
from itertools import repeat
from logging import warning
from pathlib import Path, PurePath

import cv2
import numpy as np
import multiprocessing as mp
from tkinter import *
from tkinter import filedialog, ttk
from scipy.ndimage import gaussian_filter
import pandas as pd


from ocvl.function.preprocessing.improc import weighted_z_projection, simple_image_stack_align, \
    optimizer_stack_align
from ocvl.function.utility.dataset import parse_file_metadata, load_dataset, \
    preprocess_dataset, initialize_and_load_dataset

from ocvl.function.utility.json_format_constants import DataFormatType, DataTags, MetaTags, PipelineParams, AcquisiTags
from ocvl.function.utility.resources import save_video

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

    # We should be 3 levels up from here. Kinda jank, will need to change eventually
    config_path = Path(os.path.dirname(__file__)).parent.parent.parent.joinpath("config_files")

    json_fName = filedialog.askopenfilename(title="Select the configuration json file.", initialdir=config_path, parent=root)
    if not json_fName:
        quit()

    with mp.Pool(processes=int(np.round(mp.cpu_count()/2 ))) as pool:

        dat_form, allData = parse_file_metadata(json_fName, pName, "processed")

        processed_dat_format = dat_form.get("processed")
        pipeline_params = processed_dat_format.get("pipeline_params")
        modes_of_interest = pipeline_params.get(PipelineParams.MODALITIES)

        output_folder = pipeline_params.get(PipelineParams.OUTPUT_FOLDER)
        if output_folder is None:
            output_folder = PurePath("Functional Pipeline")
        else:
            output_folder = PurePath(output_folder)

        metadata_params = None
        if processed_dat_format.get(MetaTags.METATAG) is not None:
            metadata_params = processed_dat_format.get(MetaTags.METATAG)
            metadata_form = metadata_params.get(DataFormatType.METADATA)

        acquisition = dict()

        # Group files together based on location, modality, and video number
        # If we've selected modalities of interest, only process those; otherwise, process them all.
        if modes_of_interest is None:
            modes_of_interest = allData.loc[DataTags.MODALITY].unique().tolist()

        for mode in modes_of_interest:
            modevids = allData.loc[allData[DataTags.MODALITY] == mode]

            vidnums = np.unique(modevids[DataTags.VIDEO_ID].to_numpy())
            for num in vidnums:
                # Find the rows associated with this video number, and
                # extract the rows corresponding to this acquisition.
                acquisition = modevids.loc[modevids[DataTags.VIDEO_ID] == num]

                im_info = acquisition.loc[acquisition[DataFormatType.FORMAT_TYPE] == DataFormatType.IMAGE]
                allData.loc[im_info.index, AcquisiTags.DATASET] = initialize_and_load_dataset(acquisition, metadata_params)



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

            group_folder = output_folder.joinpath(group)

            for m, mode in enumerate(modes_of_interest):
                ref_xforms=[]

                modevids = group_datasets.loc[group_datasets[DataTags.MODALITY] == mode]

                vidnums = modevids[DataTags.VIDEO_ID].to_numpy()
                datasets = modevids[AcquisiTags.DATASET].to_list()
                avg_images = np.dstack([data.avg_image_data for data in datasets])

                if m == 0:
                    print("Selecting ideal central frame for mode and location: " + mode)
                    dist_res = pool.starmap_async(simple_image_stack_align, zip(repeat(avg_images),
                                                                                repeat(None),
                                                                                np.arange(len(datasets))))
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

                    print("Selecting ideal central frame for mode and location: " + mode)

                    dist_res = pool.starmap_async(simple_image_stack_align, zip(repeat(avg_images),
                                                                                repeat(None),
                                                                                np.arange(len(datasets))))
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

                    # Gaussian blur the data first before aligning, if requested
                    gausblur = pipeline_params.get(PipelineParams.GAUSSIAN_BLUR)
                    align_dat = avg_images.copy()
                    if gausblur is not None and gausblur != 0.0:
                        for f in range(avg_images.shape[-1]):
                            align_dat[..., f] = gaussian_filter(avg_images[..., f], sigma=gausblur)

                    # Align the stack of average images from all datasets
                    align_dat, ref_xforms, inliers, avg_masks = optimizer_stack_align(align_dat,
                                                                                      (align_dat > 0),
                                                                                      dist_ref_idx,
                                                                                      determine_initial_shifts=True,
                                                                                      dropthresh=0.0, transformtype="affine")

                # Apply the transforms to the unfiltered, cropped, etc. trimmed datasets, using the reference mode
                for f in range(avg_images.shape[-1]):
                    if inliers[f]:
                        avg_images[..., f] = cv2.warpAffine(avg_images[..., f], ref_xforms[f],
                                                            (avg_images.shape[1], avg_images.shape[0]),
                                                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                                            borderValue=np.nan)

                # Z Project each of our image types
                avg_avg_images, avg_avg_mask = weighted_z_projection(avg_images)

                # Determine the filename for the superaverage using the central-most dataset.
                pipelined_dat_format = dat_form.get("pipelined")
                if pipelined_dat_format is not None:
                    pipe_im_form = pipelined_dat_format.get(DataFormatType.IMAGE)
                    if pipe_im_form is not None:
                        pipe_im_fname = pipe_im_form.format_map(central_dataset.metadata)

                # Make sure our output folder exists.
                central_dataset.metadata[AcquisiTags.BASE_PATH].joinpath(group_folder).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(central_dataset.metadata[AcquisiTags.BASE_PATH].joinpath(group_folder, pipe_im_fname),
                            avg_avg_images)

                print("Outputting data...")
                for dataset, xform in zip(datasets, ref_xforms):
                    # Make sure our output folder exists.
                    dataset.metadata[AcquisiTags.BASE_PATH].joinpath(group_folder).mkdir(parents=True, exist_ok=True)

                    (rows, cols) = dataset.video_data.shape[0:2]

                    if pipelined_dat_format is not None:
                        pipe_vid_form = pipelined_dat_format.get(DataFormatType.VIDEO)
                        pipe_mask_form = pipelined_dat_format.get(DataFormatType.MASK)
                        pipe_meta_form = pipelined_dat_format.get(MetaTags.METATAG)

                        if pipe_vid_form is not None:
                            pipe_vid_fname = pipe_vid_form.format_map(dataset.metadata)
                        if pipe_mask_form is not None:
                            pipe_mask_fname = pipe_mask_form.format_map(dataset.metadata)
                        if pipe_meta_form is not None:
                            pipe_meta_form = pipe_meta_form.get(DataFormatType.METADATA)
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
                    out_meta.to_csv(dataset.metadata[AcquisiTags.BASE_PATH].joinpath(group_folder, pipe_meta_fname), index=False)
                    save_video(dataset.metadata[AcquisiTags.BASE_PATH].joinpath(group_folder, pipe_vid_fname), dataset.video_data,
                               framerate=dataset.framerate)



    print("PK FIRE")
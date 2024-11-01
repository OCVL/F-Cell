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
from pathlib import Path

import cv2
import numpy as np
import multiprocessing as mp
import os
from os import walk
from os.path import splitext
from tkinter import *
from tkinter import filedialog, simpledialog
from tkinter import ttk


from scipy.ndimage import binary_dilation, gaussian_filter
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from ocvl.function.preprocessing.improc import flat_field, weighted_z_projection, simple_image_stack_align, \
    optimizer_stack_align, dewarp_2D_data
from ocvl.function.utility.format_parser import FormatParser
from ocvl.function.utility.generic import Dataset, PipeStages, initialize_and_load_dataset, AcquisiTags
from ocvl.function.utility.json_format_constants import FormatTypes, DataTags, MetaTags
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_video
import parse

class PipelineParams(StrEnum):
    GAUSSIAN_BLUR = "gaus_blur",
    MASK_ROI = "mask_roi",
    MODALITIES = "modalities",
    CORRECT_TORSION = "correct_torsion",
    CUSTOM = "custom"

def initialize_and_load_meao(file, a_mode, ref_mode):
    print(file)
    dataset = MEAODataset(file, analysis_modality=a_mode, ref_modality=ref_mode, stage=PipeStages.PROCESSED)

    dataset.load_processed_data(clip_top=16)

    imp, awp = weighted_z_projection(dataset.video_data, dataset.mask_data)

    # pyplot.imshow(awp, cmap='gray')
    # pyplot.show()

    ref_imp, rwp = weighted_z_projection(dataset.ref_video_data, dataset.ref_mask_data)

    return dataset, imp, awp, ref_imp, rwp


def run_generic_pipeline(pName, tkroot):
    root = tkroot
    pb = ttk.Progressbar(root, orient=HORIZONTAL, length=512)
    pb.grid(column=0, row=0, columnspan=2, padx=3, pady=5)
    pb_label = ttk.Label(root, text="Initializing setup...")
    pb_label.grid(column=0, row=1, columnspan=2)
    pb.start()

    # Resize our root to show our progress bar.
    w = 512
    h = 64
    x = root.winfo_screenwidth() / 2 - 256
    y = root.winfo_screenheight() / 2 - 64
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.update()

    allFiles = dict()

    # Parse out the locations and filenames, store them in a hash table.
    for (dirpath, dirnames, filenames) in walk(pName):
        for fName in filenames:
            if splitext(fName)[1] == ".avi":
                splitfName = fName.split("_")

                loc = splitfName[5]
                print("Found location "+loc)
                if loc not in allFiles:
                    allFiles[loc] = []
                    allFiles[loc].append(os.path.join(pName, fName))
                else:
                    allFiles[loc].append(os.path.join(pName, fName))

        break # Break after the first run so we don't go recursive.


    for loc in allFiles:
        r = 0
        pb["maximum"] = len(allFiles[loc])
        for toload in allFiles[loc]:
            pb["value"] = r
            pb_label["text"] = "Processing " + os.path.basename(os.path.realpath(toload)) + "..."
            pb.update()
            pb_label.update()

            dataset = Dataset(toload, stage=PipeStages.RAW)

            dataset.load_data()
            dataset.video_data = flat_field(dataset.video_data)

            dataset.save_data("_ff")
            r += 1

def run_demotion_pipeline(pName, tkroot):
    root = tkroot
    pb = ttk.Progressbar(root, orient=HORIZONTAL, length=512)
    pb.grid(column=0, row=0, columnspan=2, padx=3, pady=5)
    pb_label = ttk.Label(root, text="Initializing setup...")
    pb_label.grid(column=0, row=1, columnspan=2)
    pb.start()

    # Resize our root to show our progress bar.
    w = 512
    h = 64
    x = root.winfo_screenwidth() / 2 - 256
    y = root.winfo_screenheight() / 2 - 64
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.update()

    allFiles = dict()

    # Parse out the locations and filenames, store them in a hash table.
    for (dirpath, dirnames, filenames) in walk(pName):
        for fName in filenames:
            if splitext(fName)[1] == ".avi":
                splitfName = fName.split("_")

                loc = splitfName[5]
                print("Found location "+loc)
                if loc not in allFiles:
                    allFiles[loc] = []
                    allFiles[loc].append(os.path.join(pName, fName))
                else:
                    allFiles[loc].append(os.path.join(pName, fName))

        break # Break after the first run so we don't go recursive.




# Need to try this:
# https://mathematica.stackexchange.com/questions/199928/removing-horizontal-noise-artefacts-from-a-sem-image
def run_meao_pipeline(pName, tkroot):
    root = tkroot
    a_mode = simpledialog.askstring(title="Input the analysis modality string: ",
                                   prompt="Input the analysis modality string:",
                                   initialvalue="760nm", parent=root)
    if not a_mode:
        a_mode = "760nm"

    ref_mode = simpledialog.askstring(title="Input the *alignment reference* modality string. ",
                                      prompt="Input the *alignment reference* modality string:", initialvalue=a_mode, parent=root)
    if not ref_mode:
        ref_mode = "760nm"

    # For debugging.
    # a_mode = "760nm"
    # ref_mode = "Confocal"

    print("Selected analysis modality name of: " + a_mode + ", and a reference modality of: " + ref_mode)

    allFiles = dict()
    allFiles["Unknown"] = []  # Prep an empty list for all the locations we can't parse.
    totFiles = 0
    # Parse out the locations and filenames, store them in a hash table.
    for (dirpath, dirnames, filenames) in walk(pName):
        for fName in filenames:
            if a_mode in fName and splitext(fName)[1] == ".avi":
                splitfName = fName.split("_")

                if splitfName[3][0] == "(" and splitfName[3][-1] == ")":
                    loc = splitfName[3]
                    if loc not in allFiles:
                        allFiles[loc] = []
                        allFiles[loc].append(os.path.join(pName, fName))
                    else:
                        allFiles[loc].append(os.path.join(pName, fName))
                else:
                    allFiles["Unknown"].append(os.path.join(pName, fName))

                totFiles += 1

        break  # Break after the first run so we don't go recursive.

    if not allFiles:
        pass  # Handle this for non-MEAO data.

    # If Unknown is empty (implying we have a location for everything), then remove it from the dict
    if not allFiles["Unknown"]:
        del allFiles["Unknown"]

    # Filter through the list, ensuring we only have paths pertaining to our analysis mode.
    # (The MEAODataset will take care of the rest)
    for loc in allFiles:
        allFiles[loc] = [file for file in allFiles[loc] if
                         "_mask" not in file and "extract_reg_cropped" in file and a_mode in file]

    pb = ttk.Progressbar(root, orient=HORIZONTAL, length=512)
    pb.grid(column=0, row=0, columnspan=2, padx=3, pady=5)
    pb_label = ttk.Label(root, text="Initializing setup...")
    pb_label.grid(column=0, row=1, columnspan=2)
    pb.start()

    # Resize our root to show our progress bar.
    w = 512
    h = 64
    x = root.winfo_screenwidth() / 2 - 256
    y = root.winfo_screenheight() / 2 - 64
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.update()

    # Create a pool of threads for processing.
    with mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:
        for loc in allFiles:

            first = True
            r = 0
            pb["maximum"] = len(allFiles[loc])
            for toload in allFiles[loc]:
                #tic = time.perf_counter()
                pb["value"] = r
                pb_label["text"] = "Processing " + os.path.basename(os.path.realpath(toload)) + "..."
                pb.update()
                pb_label.update()
                if not first:
                    dataset[r], a_im_proj[..., r], a_weight_proj[..., r], ref_im_proj[..., r], ref_weight_proj[..., r] = initialize_and_load_meao(toload, a_mode, ref_mode)
                else:
                    dat, imp, awp, ref_imp, rwp = initialize_and_load_meao(toload, a_mode, ref_mode)

                    dataset = np.empty((len(allFiles[loc])), dtype=type(dat))
                    a_im_proj = np.empty((imp.shape[0], imp.shape[1], len(allFiles[loc])), dtype=imp.dtype)
                    a_weight_proj = np.empty((awp.shape[0], awp.shape[1], len(allFiles[loc])), dtype=awp.dtype)
                    ref_im_proj = np.empty((ref_imp.shape[0], ref_imp.shape[1], len(allFiles[loc])), dtype=ref_imp.dtype)
                    ref_weight_proj = np.empty((rwp.shape[0], rwp.shape[1], len(allFiles[loc])), dtype=rwp.dtype)

                    dataset[r] = dat
                    a_im_proj[..., r] = imp
                    a_weight_proj[..., r] = awp
                    ref_im_proj[..., r] = ref_imp
                    ref_weight_proj[..., r] = awp
                    first = False

                r += 1
                #toc = time.perf_counter()
                #print(f"Processed in {toc - tic:0.4f} seconds")


            num_vid_proj = ref_im_proj.shape[-1]
            print("Selecting ideal central frame...")
            dist_res = pool.starmap_async(simple_image_stack_align, zip(repeat(a_im_proj),
                                                                        repeat(a_weight_proj > 0),
                                                                        range(len(allFiles[loc]))))
            shift_info = dist_res.get()

            avg_loc_dist = np.zeros(len(shift_info))
            f = 0
            for allshifts in shift_info:
                # allshifts = simple_image_stack_align(vid.data, mask, f)
                print(allshifts)
                allshifts = np.stack(allshifts)
                allshifts **= 2
                allshifts = np.sum(allshifts, axis=1)
                avg_loc_dist[f] = np.mean(np.sqrt(allshifts))  # Find the average distance to this reference.
                f += 1

            avg_loc_idx = np.argsort(avg_loc_dist)
            dist_ref_idx = avg_loc_idx[0]

            print("Determined most central frame as: " + str(dist_ref_idx) + ".")

            # Begin writing our results to disk.
            writepath = os.path.join(pName, "Functional Pipeline", loc)
            Path(writepath).mkdir(parents=True, exist_ok=True)
            # save_video(
            #             "\\\\134.48.93.176\\Raw Study Data\\00-64774\\MEAOSLO1\\20210824\\Processed\\Functional Pipeline\\pre_selected_stk.avi",
            #             ref_im_proj.astype("uint8"), 29.4)
            #
            # save_video(
            #             "\\\\134.48.93.176\\Raw Study Data\\00-64774\\MEAOSLO1\\20210824\\Processed\\Functional Pipeline\\apre_selected_stk.avi",
            #             a_im_proj.astype("uint8"), 29.4)

            if dataset[0].has_ref_video:
                all_im_proj, ref_xforms, inliers = optimizer_stack_align(np.dstack((a_im_proj, ref_im_proj)),
                                                                       np.dstack(((a_weight_proj > 0), (ref_weight_proj > 0))),
                                                                       dist_ref_idx, determine_initial_shifts=True,
                                                                       dropthresh=0.0, transformtype="affine")
                numa = a_im_proj.shape[-1]
                #a_im_proj = all_im_proj[...,0:numa]
                #ref_im_proj = all_im_proj[..., numa:]
            else:
                all_im_proj, ref_xforms, inliers = optimizer_stack_align(a_im_proj,
                                                                       (a_weight_proj > 0),
                                                                       dist_ref_idx, determine_initial_shifts=True,
                                                                       dropthresh=0.0, transformtype="affine")
                #a_im_proj = all_im_proj

            numa = a_im_proj.shape[-1]

            # Use the xforms from each type (reference/analysis) to do the alignment.
            # Inliers will be determined by the reference modality.
            for f in range(len(ref_xforms)):
                if inliers[f]:
                    if f < numa:
                        (rows, cols) = dataset[f].video_data.shape[0:2]

                        tmp = a_im_proj[..., f].astype("float32")
                        tmp[np.round(tmp) == 0] = np.nan
                        a_im_proj[..., f] = cv2.warpAffine(tmp, ref_xforms[f],  (cols, rows),
                                                           flags=cv2.INTER_LINEAR  | cv2.WARP_INVERSE_MAP, borderValue=np.nan).astype("uint8")

                        for i in range(dataset[f].num_frames):  # Make all of the data in our dataset relative as well.
                            tmp = dataset[f].video_data[..., i].astype("float32")
                            tmp[np.round(tmp) == 0] = np.nan
                            dataset[f].video_data[..., i] = cv2.warpAffine(tmp, ref_xforms[f],
                                                                               (cols, rows),
                                                                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                            tmp = dataset[f].mask_data[..., i].astype("float32")
                            tmp[np.round(tmp) == 0] = np.nan
                            dataset[f].mask_data[..., i] = cv2.warpAffine(tmp, ref_xforms[f],
                                                                          (cols, rows),
                                                                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
                    else:
                        f_offset = f-numa

                        (rows, cols) = dataset[f_offset].ref_video_data.shape[0:2]

                        tmp = ref_im_proj[..., f_offset].astype("float32")
                        tmp[np.round(tmp)==0] = np.nan
                        ref_im_proj[..., f_offset] = cv2.warpAffine(tmp, ref_xforms[f], (cols, rows),
                                                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=np.nan).astype("uint8")

                        for i in range(dataset[f_offset].num_frames):  # Make all of the data in our dataset relative as well.
                            tmp = dataset[f_offset].ref_video_data[..., i].astype("float32")
                            tmp[np.round(tmp) == 0] = np.nan
                            dataset[f_offset].ref_video_data[..., i] = cv2.warpAffine(tmp, ref_xforms[f],
                                                                           (cols, rows),
                                                                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP).astype("uint8")
                            tmp = dataset[f_offset].ref_mask_data[..., i].astype("float32")
                            tmp[np.round(tmp) == 0] = np.nan
                            dataset[f_offset].ref_mask_data[..., i] = cv2.warpAffine(tmp, ref_xforms[f],
                                                                          (cols, rows),
                                                                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype("uint8")




            base_ref_frame = os.path.basename(os.path.realpath(dataset[dist_ref_idx].video_path))
            common_prefix = base_ref_frame.split("_")
            analysis_zproj_fname = "_".join(common_prefix[0:6]) + "_" + dataset[dist_ref_idx].analysis_modality + "_" + \
                                   "ALL_ACQ_AVG.tif"
            analysis_vid_fname = "_".join(common_prefix[0:6]) + "_" + dataset[dist_ref_idx].analysis_modality + "_" + \
                                   "ALL_ACQ_STK.avi"
            ref_zproj_fname = "_".join(common_prefix[0:6]) + "_" + dataset[dist_ref_idx].reference_modality + "_" + \
                              "ALL_ACQ_AVG.tif"
            ref_vid_fname = "_".join(common_prefix[0:6]) + "_" + dataset[dist_ref_idx].reference_modality + "_" + \
                            "ALL_ACQ_STK.avi"

            # This would be needed... if our drop threshold was any higher.
            # dataset = dataset[inliers]
            # a_weight_proj = a_weight_proj[..., inliers]
            # a_im_proj = a_im_proj[..., inliers]
            # ref_im_proj = ref_im_proj[..., inliers]

            # Crop to the area that X images overlap. (start with all)
            if dataset[0].has_ref_video:
                mask_area = np.dstack(((a_weight_proj > 0), (ref_weight_proj > 0)))
            else:
                mask_area = a_weight_proj > 0
            mask_area = np.sum(mask_area.astype("uint8"), axis=-1)
            mask_area[mask_area < int(np.amax(mask_area)/2)] = 0
            mask_area[mask_area >= int(np.amax(mask_area)/2)] = 1
            mask_area = binary_dilation(mask_area, structure=np.ones((3, 3))).astype("uint8")

            cropx, cropy, cropw, croph = cv2.boundingRect(mask_area)

            wmult = np.ceil(cropw / 16) # For correct display, the codec needs us to output the images in multiples of 16.
            hmult = np.ceil(croph / 16)

            if (cropx + (wmult*16)) > a_im_proj.shape[1]:
                wmult -= 1

            if (cropy + (hmult*16)) > a_im_proj.shape[1]:
                hmult -= 1

            cropw = int(wmult * 16)
            croph = int(hmult * 16)

            # Crop and output the data.
            for data in dataset:
                data.video_data = data.video_data[cropy:(cropy+croph), cropx:(cropx+cropw), :]
                data.mask_data = data.mask_data[cropy:(cropy + croph), cropx:(cropx + cropw), :]
                data.ref_video_data = data.ref_video_data[cropy:(cropy + croph), cropx:(cropx + cropw), :]
                data.ref_mask_data = data.mask_data[cropy:(cropy + croph), cropx:(cropx + cropw), :]

                # Save the pipelined dataset.
                metadata = pd.DataFrame(data.framestamps, columns=["FrameStamps"])
                metadata.to_csv(os.path.join(writepath, data.ref_filename[:-4] + "_piped.csv"), index=False)
                save_video(os.path.join(writepath, data.ref_filename[:-4] + "_piped.avi"), data.ref_video_data, data.framerate)
                metadata.to_csv(os.path.join(writepath, data.filename[:-4] + "_piped.csv"), index=False)
                save_video(os.path.join(writepath, data.filename[:-4] + "_piped.avi"), data.video_data, data.framerate)


            a_weight_proj = a_weight_proj[cropy:(cropy+croph), cropx:(cropx+cropw), :]
            a_im_proj = a_im_proj[cropy:(cropy+croph), cropx:(cropx+cropw), :]
            ref_im_proj = ref_im_proj[cropy:(cropy+croph), cropx:(cropx+cropw), :]

            # Z Project each of our image types
            ref_zproj, weight_zproj = weighted_z_projection(ref_im_proj)
            analysis_zproj, weight_zproj = weighted_z_projection(a_im_proj)

            # After we z-project everything, dump it to disk.
            cv2.imwrite(os.path.join(writepath, ref_zproj_fname), ref_zproj.astype("uint8"))
            cv2.imwrite(os.path.join(writepath, analysis_zproj_fname), analysis_zproj.astype("uint8"))
            save_video(os.path.join(writepath, ref_vid_fname), ref_im_proj.astype("uint8"), 29.4)
            save_video(os.path.join(writepath, analysis_vid_fname), a_im_proj.astype("uint8"), 29.4)


            del dataset
            del a_im_proj
            del a_weight_proj
            del ref_im_proj
            print("Completed processing of location " + loc)
        pb.stop()


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

   # pName = filedialog.askdirectory(title="Select the folder containing all videos of interest.", parent=root)
    pName = "R:\\00-10397\\FFB_IndivInvest_2020_PRO38673\\MEAOSLO1\\20221117\\Processed"
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

    with open(json_fName, 'r') as json_f:
        dat_form = json.load(json_f)

        allFilesColumns = [AcquisiTags.DATA_PATH, FormatTypes.FORMAT]
        allFilesColumns.extend([d.value for d in DataTags])
        allData = pd.DataFrame(columns=allFilesColumns)
        acquisition = dict()

        processed_dat_format = dat_form.get("processed")
        if processed_dat_format:

            pipeline_params = processed_dat_format.get("pipeline_params")
            modes_of_interest = pipeline_params.get(PipelineParams.MODALITIES)

            im_form = processed_dat_format.get(FormatTypes.IMAGE)
            vid_form = processed_dat_format.get(FormatTypes.VIDEO)
            mask_form = processed_dat_format.get(FormatTypes.MASK)

            metadata_form = None
            metadata_params = None
            if processed_dat_format.get(MetaTags.METATAG):
                metadata_params = processed_dat_format.get(MetaTags.METATAG)
                metadata_form = metadata_params.get(FormatTypes.METADATA)

            if vid_form:

                # Grab our extensions, make sure to check them all.
                all_ext = (vid_form[vid_form.rfind(".", -5, -1):],)
                all_ext = all_ext + (mask_form[mask_form.rfind(".", -5, -1):],) if mask_form and mask_form[mask_form.rfind(".", -5, -1):] not in all_ext else all_ext
                all_ext = all_ext + (im_form[im_form.rfind(".", -5, -1):],) if im_form and im_form[im_form.rfind(".", -5, -1):] not in all_ext else all_ext
                all_ext = all_ext + (metadata_form[metadata_form.rfind(".", -5, -1):],) if metadata_form and metadata_form[metadata_form.rfind(".", -5, -1):] not in all_ext else all_ext

                # Construct the parser we'll use for each of these forms
                parser = FormatParser(vid_form, mask_form, im_form, metadata_form)

                # Parse out the locations and filenames, store them in a hash table by location.
                searchpath = Path(pName)
                allFiles = list()
                for ext in all_ext:
                    for path in searchpath.glob("*"+ext):
                        format_type, file_info = parser.parse_file(path.name)
                        file_info[DataTags.FORMAT_TYPE] = format_type
                        file_info[AcquisiTags.DATA_PATH] = path
                        entry = pd.DataFrame.from_dict([file_info])

                        allFiles.append(entry)

                allData = pd.concat(allFiles, ignore_index=True)

                # Group files together based on Video number
                vidnums = np.unique(allData[DataTags.VIDEO_ID].to_numpy())
                for num in vidnums:
                    # Find the rows associated with this video number, and
                    # extract the rows corresponding to this acquisition.
                    acquisition = allData.loc[allData[DataTags.VIDEO_ID] == num]

                    # If we've selected modalities of interest, only process those; otherwise, process them all.
                    if modes_of_interest:
                        for mode in modes_of_interest:
                            modevids = acquisition.loc[acquisition[DataTags.MODALITY] == mode]
                            if (modevids[DataTags.FORMAT_TYPE] == FormatTypes.MASK).sum() <= 1 and \
                               (modevids[DataTags.FORMAT_TYPE] == FormatTypes.METADATA).sum() <= 1 and \
                               (modevids[DataTags.FORMAT_TYPE] == FormatTypes.VIDEO).sum() == 1:

                                video_info = modevids.loc[modevids[DataTags.FORMAT_TYPE] == FormatTypes.VIDEO]
                                mask_info = modevids.loc[modevids[DataTags.FORMAT_TYPE] == FormatTypes.MASK]
                                metadata_info = modevids.loc[modevids[DataTags.FORMAT_TYPE] == FormatTypes.METADATA]
                                im_info = modevids.loc[modevids[DataTags.FORMAT_TYPE] == FormatTypes.IMAGE]

                                # Load our metadata file
                                if not metadata_info.empty and metadata_params:
                                    if metadata_info.at[metadata_info.index[0], AcquisiTags.DATA_PATH].exists():
                                        metatype = metadata_params.get(MetaTags.TYPE)
                                        loadfields = metadata_params.get(MetaTags.FIELDS_OF_INTEREST)

                                        if metatype == "text_file":
                                            dat_metadata = pd.read_csv(metadata_info.at[metadata_info.index[0], AcquisiTags.DATA_PATH], encoding="utf-8-sig", skipinitialspace=True)

                                            meta_fields = {}
                                            for field, column in loadfields.items():
                                                meta_fields[field] = dat_metadata[column].to_numpy()
                                        elif metatype == "database":
                                            pass
                                        elif metatype == "mat_file":
                                            pass

                                # Take metadata gleaned from our filename, as well as our metadata files,
                                # and combine them into a single dictionary.
                                combined_meta_dict = video_info.squeeze().to_dict() | meta_fields

                                # Add paths to things that we may want, depending on the stage we're at.
                                combined_meta_dict[AcquisiTags.IMAGE_PATH] = im_info.at[im_info.index[0], AcquisiTags.DATA_PATH]

                                dataset = initialize_and_load_dataset(video_info.at[video_info.index[0],AcquisiTags.DATA_PATH],
                                                            mask_info.at[mask_info.index[0],AcquisiTags.DATA_PATH],
                                                            metadata_info.at[metadata_info.index[0], AcquisiTags.DATA_PATH],
                                                            combined_meta_dict)

                                # Run the processing pipeline on this dataset, with params specified by the json.

                                # First do custom preprocessing steps, e.g. things implemented expressly for and by the OCVL
                                # If you would like to "bake in" custom pipeline steps, please contact the OCVL using GitHub's Issues
                                # or submit a pull request.
                                custom_steps = pipeline_params.get(PipelineParams.CUSTOM)
                                if custom_steps:
                                    # For "baked in" dewarping- otherwise, data is expected to be dewarped already
                                    pre_dewarp = custom_steps.get("dewarp")
                                    match pre_dewarp:
                                        case "ocvl":
                                            if dataset.metadata[AcquisiTags.META_PATH] is not None:
                                                dat_metadata = pd.read_csv(dataset.metadata[AcquisiTags.META_PATH], encoding="utf-8-sig", skipinitialspace=True)

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

                                # Gaussian blur the data first before aligning, if requested
                                gausblur = pipeline_params.get(PipelineParams.GAUSSIAN_BLUR)

                                if gausblur is not None and gausblur != 0.0:
                                    align_dat = gaussian_filter(dataset.video_data, sigma=gausblur)
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
                                    tmp = np.zeros_like(align_dat)
                                    tmp[r:r+height, c:c+width, :] = align_dat[r:r+height, c:c+width, :]
                                    align_dat = tmp

                                    tmp = np.zeros_like(mask_dat)
                                    tmp[r:r+height, c:c+width, :] = mask_dat[r:r+height, c:c+width, :]
                                    mask_dat = tmp

                                # Finally, correct for residual torsion if requested
                                correct_torsion = pipeline_params.get(PipelineParams.CORRECT_TORSION)
                                if correct_torsion is not None and correct_torsion:
                                    align_dat, xforms, inliers, mask_dat = optimizer_stack_align(align_dat, dataset.mask_data,
                                                                                                 reference_idx=dataset.reference_frame_idx,
                                                                                                 dropthresh=0)




                            else:
                                warning("Detected more than one video or mask associated with vidnum: "+num)

        else:
            warning("Unable to detect \"processed\" json value!")
    # run_generic_pipeline(pName, tkroot=root)

    print("PK FIRE")
    #run_meao_pipeline(pName, tkroot=root)

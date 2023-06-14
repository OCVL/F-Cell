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
from itertools import repeat
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
from scipy.ndimage import binary_dilation
import pandas as pd

from ocvl.function.preprocessing.improc import flat_field, weighted_z_projection, simple_image_stack_align, \
    optimizer_stack_align
from ocvl.function.utility.generic import GenericDataset, PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_video


def initialize_and_load_meao(file, a_mode, ref_mode):
    print(file)
    dataset = MEAODataset(file, analysis_modality=a_mode, ref_modality=ref_mode, stage=PipeStages.PROCESSED)

    dataset.load_processed_data()

    imp, wp = weighted_z_projection(dataset.video_data, dataset.mask_data)

    ref_imp = weighted_z_projection(dataset.ref_video_data, dataset.mask_data)

    return dataset, imp, wp, ref_imp[0]


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

            dataset = GenericDataset(toload, stage=PipeStages.RAW)

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
                    dataset[r], a_im_proj[..., r], weight_proj[..., r], ref_im_proj[..., r] = initialize_and_load_meao(toload, a_mode, ref_mode)
                else:
                    dat, imp, wp, ref_imp = initialize_and_load_meao(toload, a_mode, ref_mode)

                    dataset = np.empty((len(allFiles[loc])), dtype=type(dat))
                    a_im_proj = np.empty((imp.shape[0], imp.shape[1], len(allFiles[loc])), dtype=imp.dtype)
                    weight_proj = np.empty((wp.shape[0], wp.shape[1], len(allFiles[loc])), dtype=wp.dtype)
                    ref_im_proj = np.empty((ref_imp.shape[0], ref_imp.shape[1], len(allFiles[loc])), dtype=ref_imp.dtype)

                    dataset[r] = dat
                    a_im_proj[..., r] = imp
                    weight_proj[..., r] = wp
                    ref_im_proj[..., r] = ref_imp
                    first = False

                r += 1
                #toc = time.perf_counter()
                #print(f"Processed in {toc - tic:0.4f} seconds")


            num_vid_proj = ref_im_proj.shape[-1]
            print("Selecting ideal central frame...")
            dist_res = pool.starmap_async(simple_image_stack_align, zip(repeat(ref_im_proj.astype("uint8")),
                                                                        repeat(np.ceil(weight_proj).astype("uint8")),
                                                                        range(len(allFiles[loc]))))
            shift_info = dist_res.get()

            avg_loc_dist = np.zeros(len(shift_info))
            f = 0
            for allshifts in shift_info:
                # allshifts = simple_image_stack_align(vid.data, mask, f)
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

            ref_im_proj, ref_xforms, inliers = optimizer_stack_align(ref_im_proj.astype("uint8"),
                                                                (weight_proj > 0).astype("uint8"),
                                                                dist_ref_idx, determine_initial_shifts=True,
                                                                dropthresh=0.0, transformtype="rigid")

            # Use the xforms from each type (reference/analysis) to do the alignment.
            # Inliers will be determined by the reference modality.
            for f in range(len(ref_xforms)):
                if inliers[f]:

                    for i in range(dataset[f].num_frames):  # Make all of the data in our dataset relative as well.
                        (rows, cols) = dataset[f].video_data.shape[0:2]
                        dataset[f].video_data[..., i] = cv2.warpAffine(dataset[f].video_data[..., i], ref_xforms[f],
                                                                       (cols, rows),
                                                                       flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                        dataset[f].ref_video_data[..., i] = cv2.warpAffine(dataset[f].ref_video_data[..., i], ref_xforms[f],
                                                                           (cols, rows),
                                                                           flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                        dataset[f].mask_data[..., i] = cv2.warpAffine(dataset[f].mask_data[..., i], ref_xforms[f],
                                                                      (cols, rows),
                                                                      flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)


                    a_im_proj[..., f] = cv2.warpAffine(a_im_proj[..., f], ref_xforms[f],
                                                     a_im_proj[..., f].shape[::-1],
                                                     flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                    weight_proj[..., f] = cv2.warpAffine(weight_proj[..., f], ref_xforms[f],
                                                         weight_proj[..., f].shape[::-1],
                                                         flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)

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
            dataset = dataset[inliers]
            weight_proj = weight_proj[..., inliers]
            a_im_proj = a_im_proj[..., inliers]
            ref_im_proj = ref_im_proj[..., inliers]

            # Crop to the area that X images overlap. (start with all)
            mask_area = weight_proj > 0
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
                data.ref_video_data = data.ref_video_data[cropy:(cropy + croph), cropx:(cropx + cropw), :]
                data.mask_data = data.mask_data[cropy:(cropy + croph), cropx:(cropx + cropw), :]

                # Save the pipelined dataset.
                metadata = pd.DataFrame(data.framestamps, columns=["FrameStamps"])
                metadata.to_csv(os.path.join(writepath, data.filename[:-4] + "_piped.csv"), index=False)
                save_video(os.path.join(writepath, data.filename[:-4] + "_piped.avi"), data.video_data, data.framerate)

            weight_proj = weight_proj[cropy:(cropy+croph), cropx:(cropx+cropw), :]
            a_im_proj = a_im_proj[cropy:(cropy+croph), cropx:(cropx+cropw), :]
            ref_im_proj = ref_im_proj[cropy:(cropy+croph), cropx:(cropx+cropw), :]

            # Z Project each of our image types
            ref_zproj, weight_zproj = weighted_z_projection(ref_im_proj, weight_proj)
            analysis_zproj, weight_zproj = weighted_z_projection(a_im_proj, weight_proj)

            # After we z-project everything, dump it to disk.



            cv2.imwrite(os.path.join(writepath, ref_zproj_fname), ref_zproj.astype("uint8"))
            cv2.imwrite(os.path.join(writepath, analysis_zproj_fname), analysis_zproj.astype("uint8"))
            save_video(os.path.join(writepath, analysis_vid_fname), a_im_proj, 29.4)
            save_video(os.path.join(writepath, ref_vid_fname), ref_im_proj, 29.4)

            del dataset
            del a_im_proj
            del weight_proj
            del ref_im_proj
            print("Completed processing of location " + loc)
        pb.stop()


if __name__ == "__main__":
    # dataset = MEAODataset("\\\\134.48.93.176\\Raw Study Data\\00-64774\\MEAOSLO1\\20210824\\Processed\\00-64774_20210824_OS_(-1,0)_1x1_622_760nm1_extract_reg_cropped.avi",
    #                       analysis_modality="760nm", ref_modality="Confocal", stage=PipeStages.PROCESSED)
    #
    # dataset.load_unpipelined_data()
    #vid = load_video("\\\\134.48.93.176\\Raw Study Data\\00-64774\\MEAOSLO1\\20210824\\Processed\\Functional Pipeline\\(-1,0)\\stimulus\\00-64774_20210824_OS_(-1,0)_1x1_727_Confocal_ALL_ACQ_STK.avi")
    # vid = load_video(
    #      "E:\\Dropbox (Personal)\\Grant_Proposals\\2022_Feb_R01\\LSO_Prelim_data\\Test\\Subject1_Session20220112_OD_(1.5,0)_1.2x0.8_43028_Confocal1_extract_reg_cropped.avi")
    #
    # mask = np.ones(vid.data.shape, dtype="uint8")
    # mask[vid.data == 0] = 0
    # numfrm = vid.data.shape[-1]
    #
    # optimizer_stack_align(vid.data, mask, 33, determine_initial_shifts=True, transformtype="affine")

    #
    # print("wtfbbq")
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

    if not pName:
        quit()

    x = root.winfo_screenwidth() / 2 - 128
    y = root.winfo_screenheight() / 2 - 128
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.
    root.update()

    # run_generic_pipeline(pName, tkroot=root)

    run_meao_pipeline(pName, tkroot=root)

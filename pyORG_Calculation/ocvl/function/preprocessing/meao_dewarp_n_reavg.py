import os
from os import walk
from os.path import splitext
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import cv2
import numpy as np
import pandas as pd

from ocvl.function.preprocessing.improc import dewarp_2D_data, optimizer_stack_align, weighted_z_projection
from ocvl.function.utility.resources import load_video

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

    if not pName:
        quit()

    x = root.winfo_screenwidth() / 2 - 128
    y = root.winfo_screenheight() / 2 - 128
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.
    root.update()

    allFiles = dict()
    totFiles = 0
    # Parse out the locations and filenames, store them in a hash table.
    for (dirpath, dirnames, filenames) in walk(pName):
        for fName in filenames:
            if splitext(fName)[1] == ".avi":
                splitfName = fName.split("_")
                acqID = splitfName[5]

                if acqID not in allFiles:
                    allFiles[acqID] = []
                    allFiles[acqID].append(os.path.join(pName, fName))
                else:
                    allFiles[acqID].append(os.path.join(pName, fName))

                totFiles += 1
        break  # Break after the first run so we don't go recursive.

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

    r = 0
    pb["maximum"] = totFiles
    for fid in allFiles:
        for video_path in allFiles[fid]:
            if "_mask" in video_path:
                r += 1
                continue

            pb["value"] = r
            pb_label["text"] = "Processing " + os.path.basename(os.path.realpath(video_path)) + "..."
            print("Processing " + os.path.basename(os.path.realpath(video_path)) + "...")
            pb.update()
            pb_label.update()
            mask_path = video_path[0:-4] + "_mask.avi"
            metadata_path = video_path[0:-3] + "csv"

            # Load our video data.
            res = load_video(video_path)
            num_frames = res.data.shape[-1]
            width = res.data.shape[1]
            height = res.data.shape[0]
            video_data = res.data
            # Load our mask data.
            res = load_video(mask_path)

            mask_data = res.data / 255
            mask_data[mask_data < 0] = 0
            video_data = (video_data * mask_data).astype("uint8")

            metadata = pd.read_csv(metadata_path, delimiter=',', encoding="utf-8-sig")
            metadata.columns = metadata.columns.str.strip()

            framestamps = metadata["OriginalFrameNumber"].to_numpy()
            ncc = 1 - metadata["NCC"].to_numpy(dtype=float)
            reference_frame_idx = min(range(len(ncc)), key=ncc.__getitem__)

            # Dewarp our data.
            # First find out how many strips we have.
            numstrips = 0
            for col in metadata.columns.tolist():
                if "XShift" in col:
                    numstrips += 1

            xshifts = np.zeros([ncc.shape[0], numstrips])
            yshifts = np.zeros([ncc.shape[0], numstrips])

            for col in metadata.columns.tolist():
                shiftrow = col.strip().split("_")[0][5:]
                npcol = metadata[col].to_numpy()
                if npcol.dtype == "object":
                    npcol[npcol == " "] = np.nan
                if col != "XShift" and "XShift" in col:
                    xshifts[:, int(shiftrow)] = npcol
                if col != "YShift" and "YShift" in col:
                    yshifts[:, int(shiftrow)] = npcol

            # Determine the residual error in our dewarping, and obtain the maps
            video_data, map_mesh_x, map_mesh_y = dewarp_2D_data(video_data, yshifts, xshifts)

            for f in range(num_frames):
                mask_data[..., f] = cv2.remap(mask_data[..., f].astype("float64"), map_mesh_x,
                                              map_mesh_y, interpolation=cv2.INTER_NEAREST)

            # Clamp our values.
            mask_data[mask_data < 0] = 0
            mask_data[mask_data >= 1] = 1

            if "760nm" in video_path:
                thresh = 0
            else:
                thresh = 0.5

            # Determine and remove residual torsion.
            video_data, xforms, inliers = optimizer_stack_align(video_data, mask_data,
                                                                         reference_idx=reference_frame_idx,
                                                                         dropthresh=thresh)
            (rows, cols) = video_data.shape[0:2]
            for f in range(num_frames):
                if xforms[f] is not None:
                    mask_data[..., f] = cv2.warpAffine(mask_data[..., f], xforms[f],
                                                            (cols, rows),
                                                            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)

            avg_path = video_path[0:-11] + "dewarpavg.tif"
            avg_im, sum_map = weighted_z_projection(video_data, mask_data)

            cv2.imwrite(avg_path, avg_im.astype("uint8"))

            r += 1


import os
from multiprocessing import Pool
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from numpy import random

from scipy.spatial.distance import pdist, squareform

from ocvl.function.analysis.iORG_signal_extraction import extract_signals, normalize_signals, standardize_signals, \
    refine_coord, refine_coord_to_stack, exclude_signals
from ocvl.function.analysis.iORG_profile_analyses import summarize_iORG_signals, wavelet_iORG, extract_texture_profiles, \
    iORG_signal_metrics, pooled_variance
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.dataset import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_video
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles, densify_temporal_matrix, trim_video


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]



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

    stimtrain_fName = filedialog.askopenfilename(title="Select the stimulus train file.", parent=root)

    if not stimtrain_fName:
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
    controlpath = None
    searchpath = Path(pName)
    for path in searchpath.rglob("*.avi"):
        if "piped" in path.name:
            splitfName = path.name.split("_")

            if (path.parent.parent == searchpath or path.parent == searchpath):
                if path.parent not in allFiles:
                    allFiles[path.parent] = []
                    allFiles[path.parent].append(path)

                    if "control" in path.parent.name:
                        # print("DETECTED CONTROL DATA AT: " + str(path.parent))
                        controlpath = path.parent
                else:
                    allFiles[path.parent].append(path)

            totFiles += 1

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



    # [ 0:"bob"  1:"moe" 2:"larry" 3:"curly"]
    # Loops through all locations in allFiles
    for l, loc in enumerate(allFiles):
        #if loc == controlpath:
         #   continue

        first = True

        res_dir = loc.joinpath("Results")  # creates a results folder within loc ex: R:\00-23045\MEAOSLO1\20220325\Functional\Processed\Functional Pipeline\(1,0)\Results
        res_dir.mkdir(exist_ok=True)  # actually makes the directory if it doesn't exist. if it exists it does nothing.

        this_dirname = res_dir.parent.name

        r = 0
        pb["maximum"] = len(allFiles[loc])
        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(allFiles[loc])).reversed())
        max_frmstamp = 0

        # Loops through all the files within this location
        for file in allFiles[loc]:

            # Ignores the All_ACQ_AVG tif while running through the files in this location
            if "ALL_ACQ_AVG" not in file.name:
                # Waitbar stuff
                pb["value"] = r
                pb_label["text"] = "Processing " + file.name + "..."
                pb.update()
                pb_label.update()
                print("Processing " + file.name + "...")

                # Loading in the pipelined data (calls the load_pipelined_data() fxn
                dataset = MEAODataset(file.as_posix(), stimtrain_path=stimtrain_fName,
                                      analysis_modality="760nm", ref_modality="760nm", stage=PipeStages.PIPELINED)
                dataset.load_pipelined_data()

                # Initialize the dict for individual cells.
                if first:
                    og_framestamps = []
                    cell_framestamps = []
                    mean_cell_profiles = []
                    texture_cell_profiles = []
                    full_cell_profiles = []


                    framerate = dataset.framerate
                    stimulus_train = dataset.stimtrain_frame_stamps
                    ref_im = dataset.reference_im

                    width = dataset.video_data.shape[1]
                    height = dataset.video_data.shape[0]
                    y = np.arange(0, dataset.video_data.shape[0])
                    x = np.arange(0, dataset.video_data.shape[1])
                    xv, yv = np.meshgrid(x, y)

                    xv = np.reshape(xv, (xv.size, 1))
                    yv = np.reshape(yv, (yv.size, 1))

                    reference_coord_data = np.hstack((xv, yv))
                    del x, y, xv, yv

                    for c in range(len(reference_coord_data)):
                        og_framestamps.append([])
                        cell_framestamps.append([])
                        mean_cell_profiles.append([])
                        texture_cell_profiles.append([])
                        full_cell_profiles.append([])

                    first = False

                norm_video_data = norm_video(dataset.video_data, norm_method="mean", rescaled=True)

                temp_profiles = extract_signals(norm_video_data, reference_coord_data, seg_radius=0,
                                                seg_mask="disk", summary="mean")

                temp_profiles = standardize_signals(temp_profiles, dataset.framestamps,
                                                    stimulus_stamp=stimulus_train[0], method="mean_sub")

                temp_profiles, good_profiles = exclude_signals(temp_profiles, dataset.framestamps,
                                                               critical_region=np.arange(stimulus_train[0] - int(0.1 * framerate),
                                                                           stimulus_train[1] + int(0.2 * framerate)),
                                                               critical_fraction=0.5)

                stdize_profiles, reconst_framestamps, nummissed = reconstruct_profiles(temp_profiles,
                                                                                       dataset.framestamps,
                                                                                       method="L1",
                                                                                       threshold=0.3)

                # Put the profile of each cell into its own array
                for c in range(len(reference_coord_data)):
                    og_framestamps[c].append(dataset.framestamps)
                    cell_framestamps[c].append(reconst_framestamps)
                    mean_cell_profiles[c].append(stdize_profiles[c, :])
                    # texture_cell_profiles[c].append(homogeneity[c, :])
                    # full_cell_profiles[c].append(full_profiles[:, :, :, c])

                r += 1

                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]

        del dataset, temp_profiles, stdize_profiles

        # Rows: Acquisitions
        # Cols: Framestamps
        # Depth: Coordinates
        all_cell_iORG = np.full((len(allFiles[loc]), max_frmstamp + 1), np.nan)

        full_framestamp_range = np.arange(max_frmstamp+1)

        indiv_mad_iORG = np.full( (len(reference_coord_data), max_frmstamp + 1), np.nan)

        # Make 3D matricies, where:
        # The first dimension (rows) is individual acquisitions, where NaN corresponds to missing data
        # The second dimension (columns) is time
        # The third dimension is each tracked coordinate
        cell_amp = np.full( (len(reference_coord_data), len(allFiles[loc])), np.nan)
        peak_scale = np.full_like(cell_amp, np.nan)
        indiv_fad = np.full_like(cell_amp, np.nan)
        prestim_mean = np.full_like(cell_amp, np.nan)

        prestim_ind = np.flatnonzero(np.logical_and(full_framestamp_range < stimulus_train[0],
                                                    full_framestamp_range >= (stimulus_train[0] - int(1 * framerate))))
        poststim_ind = np.flatnonzero(np.logical_and(full_framestamp_range >= stimulus_train[0],
                                                     full_framestamp_range < (stimulus_train[0] + int(1 * framerate))))

        for c in range(len(reference_coord_data)):
            all_cell_iORG = np.full_like(all_cell_iORG, np.nan)
            for i, profile in enumerate(mean_cell_profiles[c]):
                all_cell_iORG[i, cell_framestamps[c][i]] = profile

                prestim_mean[c, i] = np.nanmean(all_cell_iORG[i, prestim_ind])

            # What about a temporal histogram?
            indiv_fad[c, :], _, _, indiv_mad = iORG_signal_metrics(all_cell_iORG[:, :], full_framestamp_range,
                                                                   filter_type="MS", notch_filter=None, display=False, fwhm_size=11,
                                                                   prestim_window_idx=prestim_ind, poststim_window_idx=poststim_ind - 3)

            prestim_deriv= np.gradient(np.nanmean(indiv_mad[:, 0:prestim_ind[-1]], axis=0))

            prestim_gradient=np.nancumsum(np.repeat(np.nanmean(prestim_deriv),indiv_mad.shape[1]))

            indiv_mad_iORG[c, :] = np.squeeze(np.nanmean(indiv_mad, axis=0))-prestim_gradient
            indiv_fad[indiv_fad == 0] = np.nan


        # Log transform the data... or don't.
        log_indiv_mad_iORG = np.log(indiv_mad_iORG)
        log_indiv_fad = np.log(indiv_fad)



        histbins = np.arange(start=0.9, stop=2.0, step=0.025)

        plt.figure(131)
        plt.hist(np.log(np.nanmean(indiv_fad, axis=-1)), bins=np.arange(3, 6, 0.05), density=True, label=loc.name)
        plt.title("Log Maximum absolute deviation")
        plt.show(block=False)
        plt.legend(loc="best")
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_logmad.svg"))


        video_profiles = np.reshape(log_indiv_mad_iORG, (height, width, max_frmstamp + 1))

        print("Video 5th percentile: " + str(np.nanpercentile(video_profiles[:], 1)))
        print("Video 99th percentile: " + str(np.nanpercentile(video_profiles[:], 99)))

        hist_normie = Normalize(vmin=np.nanpercentile(video_profiles[:], 1), vmax=np.nanpercentile(video_profiles[:], 99))
        hist_mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("inferno"), norm=hist_normie)

        save_video(res_dir.joinpath(this_dirname + "_logmad_pixelpop_iORG.avi").as_posix(),
                   video_profiles, 15.9,
                   scalar_mapper=hist_mapper)
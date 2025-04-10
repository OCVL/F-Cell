import os
from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist, squareform
from skimage.feature import peak_local_max
from ssqueezepy.experimental import scale_to_freq

from ocvl.function.analysis.iORG_signal_extraction import extract_signals, normalize_signals, standardize_signals, \
    refine_coord, refine_coord_to_stack, exclude_signals
from ocvl.function.analysis.iORG_profile_analyses import summarize_iORG_signals, wavelet_iORG
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.dataset import Stages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_video, save_tiff_stack
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles


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

            if path.parent not in allFiles:
                allFiles[path.parent] = []
                allFiles[path.parent].append(path)

                if "control" in path.parent.name:
                    print("DETECTED CONTROL DATA AT: " + str(path.parent))
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

    first = True
    outputcsv = True
    cell_framestamps = []
    cell_profiles = []


    # Before we start, get an estimate of the "noise" from the control signals.
    sig_threshold_im = None
    segmentation_radius = None  # If set to None, then try and autodetect from the data.
   # controlpath = None # TEMP!
    if controlpath is not None:
        print("Processing control data to find noise floor...")

        first = True


        r = 0
        all_vars = []
        for file in allFiles[controlpath]:
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
                                      analysis_modality="760nm", ref_modality="760nm", stage=Stages.ANALYSIS)
                dataset.load_pipelined_data()

                # Initialize the dict for individual cells.
                if first:
                    reference_coord_data = dataset.coord_data
                    framerate = dataset.framerate
                    stimulus_train = dataset.stimtrain_frame_stamps
                    simple_amp = np.empty((len(allFiles), len(reference_coord_data)))
                    simple_amp[:] = np.nan
                    ref_im = dataset.reference_im
                    full_profiles = []

                    reference_coord_data = refine_coord(ref_im, dataset.coord_data)
                    all_scales = np.full((reference_coord_data.shape[0], 3), np.nan)

                    coorddist = pdist(reference_coord_data, "euclidean")
                    coorddist = squareform(coorddist)
                    coorddist[coorddist == 0] = np.amax(coorddist.flatten())
                    mindist = np.amin( coorddist, axis=-1)

                    if not segmentation_radius:
                        segmentation_radius = np.round(np.nanmean(mindist) / 4) if np.round(np.nanmean(mindist) / 4) >= 1 else 1

                        segmentation_radius = int(segmentation_radius)
                        print("Detected segmentation radius: " + str(segmentation_radius))

                    for c in range(len(dataset.coord_data)):
                        cell_framestamps.append([])
                        cell_profiles.append([])

                    first = False

                dataset.coord_data = refine_coord_to_stack(dataset.video_data, ref_im, reference_coord_data)

                norm_video_data = norm_video(dataset.video_data, norm_method="score", rescaled=True,
                                             rescale_mean=70, rescale_std=35)

                temp_profiles = extract_signals(norm_video_data, dataset.coord_data, seg_radius=segmentation_radius,
                                                seg_mask="disk", summary="mean")

                temp_profiles, num_removed = exclude_signals(temp_profiles, dataset.framestamps,
                                                             critical_region=np.arange(stimulus_train[0] - int(0.2 * framerate),
                                                                           stimulus_train[1] + int(0.2 * framerate)),
                                                             critical_fraction=0.4)

                stdize_profiles = standardize_signals(temp_profiles, dataset.framestamps, stimulus_train[0],
                                                      method="mean_sub")

                stdize_profiles, reconst_framestamps, nummissed = reconstruct_profiles(temp_profiles,
                                                                                       dataset.framestamps,
                                                                                       method="L1",
                                                                                       threshold=0.3)

                ctrl_wavelets, scales, coi = wavelet_iORG(stdize_profiles, reconst_framestamps, dataset.framerate)

                cell_var = np.full((len(ctrl_wavelets), scales.shape[0]), np.nan)

                for i in range( len(ctrl_wavelets) ):
                    this_cwt = ctrl_wavelets[i]
                    if np.all(~np.isnan(this_cwt)):
                        cell_var[i, :] = np.abs(this_cwt[:, int((this_cwt.shape[1]-1)/2 ) ])

                all_vars.append(cell_var)

                r += 1

        all_vars = np.vstack(all_vars)
        avg_var = np.nanvar(all_vars, axis=0)
        avg_all = np.nanmean(all_vars, axis=0)


        # For 95% significance.
        sig_threshold = avg_all+np.sqrt(avg_var)*1.96 #3.85 for 95, for 97.5, 6.63 for 99th from Chi squared distribution
        sig_threshold_im = np.repeat(np.asmatrix(sig_threshold).transpose(), len(reconst_framestamps), axis=1)
        sig_threshold_im[coi < 1] = np.amax(sig_threshold)
        plt.figure(12)
        plt.plot(scales, sig_threshold)
        plt.gca().set_xscale("log")
        plt.show(block=False)
        del ctrl_wavelets


    # [ 0:"bob"  1:"moe" 2:"larry" 3:"curly"]


    # Loops through all locations in allFiles
    for l, loc in enumerate(allFiles):
        if loc == controlpath:
            continue

        first = True
        res_dir = loc.joinpath(
            "Results")  # creates a results folder within loc ex: R:\00-23045\MEAOSLO1\20220325\Functional\Processed\Functional Pipeline\(1,0)\Results
        res_dir.mkdir(exist_ok=True)  # actually makes the directory if it doesn't exist. if it exists it does nothing.

        this_dirname = res_dir.parent.name

        r = 0
        pb["maximum"] = len(allFiles[loc])
        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(allFiles[loc])).reversed())
        max_frmstamp = 0

        # Loops through all the files within this location
        for file in allFiles[loc]:

            # Ignores the All_ACQ_AVG tif while running through the files in this location
            if "ALL_ACQ_AVG" not in file.name: # REMOVE MEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
                # Waitbar stuff
                pb["value"] = r
                pb_label["text"] = "Processing " + file.name + "..."
                pb.update()
                pb_label.update()
                print("Processing " + file.name + "...")

                # Loading in the pipelined data (calls the load_pipelined_data() fxn
                dataset = MEAODataset(file.as_posix(), stimtrain_path=stimtrain_fName,
                                      analysis_modality="760nm", ref_modality="760nm", stage=Stages.ANALYSIS)
                dataset.load_pipelined_data()

                # Initialize the dict for individual cells.
                if first:
                    og_framestamps = []
                    cell_framestamps = []
                    mean_cell_profiles = []
                    texture_cell_profiles = []
                    full_cell_profiles = []

                    reference_coord_data = dataset.coord_data
                    framerate = dataset.framerate
                    stimulus_train = dataset.stimtrain_frame_stamps
                    ref_im = dataset.reference_im

                    #reference_coord_data = dataset.query_loc - 1
                    reference_coord_data = refine_coord(ref_im, dataset.coord_data)  # REINSTATE MEEEEEEEEEEEEEEEEEEEEEEE

                    all_scales = np.full((reference_coord_data.shape[0], 4), np.nan)

                    coorddist = pdist(reference_coord_data, "euclidean")
                    coorddist = squareform(coorddist)
                    coorddist[coorddist == 0] = np.amax(coorddist.flatten())
                    mindist = np.amin(coorddist, axis=-1)

                    if not segmentation_radius:
                        segmentation_radius = np.round(np.nanmean(mindist) / 4) if np.round(
                            np.nanmean(mindist) / 4) >= 1 else 1

                        segmentation_radius = int(segmentation_radius)
                        print("Detected segmentation radius: " + str(segmentation_radius))

                    for c in range(len(dataset.coord_data)):
                        og_framestamps.append([])
                        cell_framestamps.append([])
                        mean_cell_profiles.append([])
                        texture_cell_profiles.append([])
                        full_cell_profiles.append([])

                    first = False

                dataset.coord_data = refine_coord_to_stack(dataset.video_data, ref_im, reference_coord_data)

                # full_profiles = extract_profiles(dataset.video_data, dataset.query_loc, seg_radius=segmentation_radius+1,
                #                                  summary="none", sigma=1)

                norm_video_data = norm_video(dataset.video_data, norm_method="score", rescaled=True,
                                             rescale_mean=70, rescale_std=35)

                temp_profiles = extract_signals(norm_video_data, dataset.coord_data, seg_radius=segmentation_radius,
                                                seg_mask="disk", summary="mean")

                temp_profiles = standardize_signals(temp_profiles, dataset.framestamps,
                                                    stimulus_stamp=stimulus_train[0], method="mean_sub")

                temp_profiles, good_profiles = exclude_signals(temp_profiles, dataset.framestamps,
                                                               critical_region=np.arange(
                                                                    stimulus_train[0] - int(0.2 * framerate),
                                                                    stimulus_train[1] + int(0.2 * framerate)),
                                                               critical_fraction=0.5)

                # full_profiles[:, :, :, ~good_profiles] = np.nan

                stdize_profiles, reconst_framestamps, nummissed = reconstruct_profiles(temp_profiles,
                                                                                       dataset.framestamps,
                                                                                       method="L1",
                                                                                       threshold=0.3)

                # Put the profile of each cell into its own array
                for c in range(len(dataset.coord_data)):
                    og_framestamps[c].append(dataset.framestamps)
                    cell_framestamps[c].append(reconst_framestamps)
                    mean_cell_profiles[c].append(stdize_profiles[c, :])
                    # texture_cell_profiles[c].append(homogeneity[c, :])
                    # full_cell_profiles[c].append(full_profiles[:, :, :, c])

                r += 1

                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]

                del dataset, temp_profiles, stdize_profiles
            else:
                continue


        # Rows: Acquisitions
        # Cols: Framestamps
        # Depth: Coordinate
        all_cell_iORG = np.full((len(allFiles[loc]), max_frmstamp + 1, len(reference_coord_data)), np.nan)

        full_framestamp_range = np.arange(max_frmstamp + 1)
        cell_power_iORG = np.full((len(reference_coord_data), max_frmstamp + 1), np.nan)

        cwt_window_start = int(-0.25*framerate)
        cwt_window_end = int(0.5*framerate)
        # Make 3D matricies, where:
        # The first dimension (rows) is individual acquisitions, where NaN corresponds to missing data
        # The second dimension (columns) is time
        # The third dimension is each tracked coordinate
        cell_amp = np.zeros( (len(reference_coord_data), len(allFiles[loc])) )
        cell_amp[:] = np.nan
        peak_scale = np.zeros( (len(reference_coord_data), len(allFiles[loc])) )
        peak_scale[:] = np.nan
        for c in range(len(reference_coord_data)):
            for i, profile in enumerate(mean_cell_profiles[c]):
                # if i<2: # REMOVE MEEEE
                    all_cell_iORG[i, cell_framestamps[c][i], c] = profile

                # save_tiff_stack(res_dir.joinpath(allFiles[loc][i].name[0:-4] + "cell(" + str(reference_coord_data[c][0]) + "," +
                #                                   str(reference_coord_data[c][1]) + ")_vid_" + str(i) + ".tif"), full_profiles[i][:, :, :, c])

            # plt.figure(11)
            # plt.clf()
            # plt.subplot(2, 2, 1)
            # plt.imshow(ref_im)
            # plt.plot(reference_coord_data[c][0], reference_coord_data[c][1], "r*")

            # What about a temporal histogram?
            # What about https://github.com/jyhmiinlin/pynufft based windowed short time fourier transforms?
            if np.any(np.isfinite(all_cell_iORG[:, :, c])):
                allcell_CWT, scales, coi = wavelet_iORG(all_cell_iORG[:, :, c], full_framestamp_range, framerate,
                                                        sig_threshold_im, display=True)

                print(c)
                # plt.figure(42)
                # plt.clf()
                for t, t_cwt in enumerate(allcell_CWT):

                    if np.any(np.isfinite(t_cwt)):

                        cwt_mod = np.abs(t_cwt)
                        cwt_phase = np.arctan(np.imag(t_cwt) / np.real(t_cwt))
                        cwt_phase_unwrapped = np.unwrap(cwt_phase, period=np.pi, axis=1)

                        scalecutoff = np.where(scales > 0.5) # Make sure we're not just marking drift.

                        cwt_window = cwt_mod[scalecutoff[0], stimulus_train[0]+cwt_window_start:stimulus_train[1]+cwt_window_end]
                        peak_idx = peak_local_max(cwt_window, exclude_border=True)

                        if peak_idx.size != 0:

                            peak_dist = np.zeros((len(peak_idx), 1))
                            peak_val = np.zeros((len(peak_idx), 1))
                            # Find the peak closest to the stimulus delivery, and highest.
                            for i, peakloc in enumerate(peak_idx):
                                peak_dist[i] = peakloc[1] + cwt_window_start
                                peak_val[i] = cwt_window[peakloc[0], peakloc[1]]

                            maxvalind = np.argmax(peak_val)
                            mindist = np.amin(peak_dist)
                            peak_scale[c, t] = scales[peak_idx[maxvalind][0]]
                            cell_amp[c, t] = np.amax(peak_val)
                            # if cell_amp[c, t] < 0.6:
                            # print(peak_dist)

                            # plt.suptitle(str(t))
                            # ax1 = plt.subplot(2, 5, t+1)
                            # ax1.imshow(cwt_mod)
                            # ax1.plot(stimulus_train[0]+cwt_window_start+peak_idx[maxvalind][1], peak_idx[maxvalind][0], "b*")
                            # ax2 = plt.subplot(2, 5, t+6)
                            # ax2.plot(all_cell_iORG[t, :, c], "r")
                            # plt.ylim((0, cwt_window.shape[0]))



                            # plt.figure(12, figsize=(cwt_window.shape[1] / 2, cwt_window.shape[0] / 2))
                            # plt.clf()
                            # ax1 = plt.gca()
                            # ax1.imshow(cwt_phase, aspect='auto')
                            # ax1.plot(stimulus_train[0] + cwt_window_start + peak_idx[maxvalind][1], peak_idx[maxvalind][0],
                            #          "r*")
                            # ax2 = ax1.twinx()
                            # ax2.plot(all_cell_iORG[t, :, c], "r")


                            # plt.show(block=False)
                            # plt.draw()
                            # plt.waitforbuttonpress()
                        else:
                            mindist = np.nan
                            peak_scale[c, t] = np.nan
                            cell_amp[c, t] = np.nan

                    #print(cell_amp[t])

                # plt.figure(110)
                # # plt.clf()
                # plt.plot(peak_scale[c, :], cell_amp[c, :], "ko")
                # # print(peak_scale)
                # plt.show(block=False)
                # plt.draw()
                # plt.pause(0.1)
                # plt.waitforbuttonpress()
                # plt.close("all")
                # indiv_resp = pd.DataFrame(all_cell_iORG[:, :, c])
                # indiv_resp.to_csv(res_dir.joinpath(file.name[0:-4] + "cell_" + str(c) + "_cell_profiles.csv"),
                # header=False, index=False)
            mean_cell_profiles[c] = []
            cell_framestamps[c] = []

        print(l)
        all_scales[0:peak_scale.shape[0], l-1] = np.nanmean(peak_scale, axis=-1)



        plt.figure(9)
        plt.hist(np.nanmean(peak_scale, axis=-1), bins=100)
        plt.draw()
        plt.waitforbuttonpress()

    peak_ska = pd.DataFrame(all_scales)
    peak_ska.to_csv(searchpath.joinpath("peak_scales.csv"))

        # for c in range(len(reference_coord_data)):
        #     cell_power_iORG[c, :], numincl = signal_power_iORG(all_cell_iORG[:, :, c], full_framestamp_range,
        #                                                        summary_method="rms", window_size=1)
        #
        #     prestim_amp = np.nanmedian(cell_power_iORG[c, 0:stimulus_train[0]])
        #     poststim_amp = np.nanmedian(cell_power_iORG[c, stimulus_train[1]:(stimulus_train[1] + 10)])
        #
        #     simple_amp[l, c] = poststim_amp - prestim_amp
        #
        # # TODO: Calling the coordclip fxn to return the simple_amp that corresponds to a 100 cone ROI
        # # clippedcoords = coordclip(query_loc, 10, 100, 'i')
        #
        # # plt.figure(0)
        # # plt.clf()
        # # for a in range(all_cell_iORG.shape[0]):
        # #     plt.plot(full_framestamp_range, all_cell_iORG[a, :, c])
        # #     plt.plot(stimulus_train[0], poststim_amp, "rD")
        # # plt.hist(simple_amp)
        # # plt.plot(cell_power_iORG[c, :])
        # #   plt.show(block=False)
        # #   plt.waitforbuttonpress()
        # # plt.savefig(res_dir.joinpath(this_dirname +  + "_allcell_iORG_amp.png"))
        #
        # # find the cells with the min, med, and max amplitude
        # min_amp = np.nanmin(simple_amp[0, :])
        # [min_amp_row, min_amp_col] = np.where(simple_amp == min_amp)
        # # print('min_amp ',min_amp)
        # med_amp = np.nanmedian(simple_amp[0, :])
        # near_med_amp = find_nearest(simple_amp[0, :], med_amp)
        # [med_amp_row, med_amp_col] = np.where(simple_amp == near_med_amp)
        #
        # # print('med_amp ', med_amp)
        # max_amp = np.nanmax(simple_amp[0, :])
        # [max_amp_row, max_amp_col] = np.where(simple_amp == max_amp)
        # # print('max_amp ', max_amp)
        #
        # plt.figure(1)
        # histbins = np.arange(start=-0.2, stop=1.5, step=0.025)
        # plt.hist(simple_amp[l, :], bins=histbins)
        # # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp.png"))
        # # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp.svg"))
        # plt.close(plt.gcf())
        #
        # hist_normie = Normalize(vmin=histbins[0], vmax=histbins[-1])
        # hist_mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("magma"), norm=hist_normie)
        #
        # # simple_amp_norm = (simple_amp-histbins[0])/(histbins[-1] - histbins[0])
        #
        # plt.figure(2)
        # vor = Voronoi(reference_coord_data)
        # voronoi_plot_2d(vor, show_vertices=False, show_points=False)
        # for c, cell in enumerate(vor.regions[1:]):
        #     if not -1 in cell:
        #         poly = [vor.vertices[i] for i in cell]
        #         plt.fill(*zip(*poly), color=hist_mapper.to_rgba(simple_amp[l, c]))
        # ax = plt.gca()
        # ax.set_aspect("equal", adjustable="box")
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_voronoi.png"))
        # # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_voronoi.svg"))
        # plt.close(plt.gcf())
        #
        # # plotting the cells with the min/med/max amplitude
        # plt.figure(300)
        # # plt.plot(np.reshape(full_framestamp_range,(1,176)).astype('float32'),cell_power_iORG[min_amp_col,:])
        # plt.plot(np.reshape(full_framestamp_range, (176, 1)).astype('float32'),
        #          np.transpose(cell_power_iORG[min_amp_col, :]))
        # plt.plot(np.reshape(full_framestamp_range, (176, 1)).astype('float32'),
        #          np.transpose(cell_power_iORG[med_amp_col, :]))
        # plt.plot(np.reshape(full_framestamp_range, (176, 1)).astype('float32'),
        #          np.transpose(cell_power_iORG[max_amp_col, :]))
        # # This also works...
        # # plt.plot(full_framestamp_range.astype('float32'),
        # #         np.ravel(cell_power_iORG[min_amp_col, :]))
        #
        # # should really be the cell_framestamps that correspond to the cells on the x axis
        # # need to fix the bug with the framstamps being empty first though
        # # plt.plot(cell_framestamps[min_amp_col, :],cell_power_iORG[min_amp_col, :])
        # plt.savefig(res_dir.joinpath(this_dirname + "_MinMedMax_amp_cones.png"))
        # plt.show(block=False)
        # plt.close(plt.gcf())
        #
        # # output cell_power_iORG to csv (optional)
        # if outputcsv:
        #     import csv
        #
        #     csv_dir = res_dir.joinpath(this_dirname + "_cell_power_iORG.csv")
        #     print(csv_dir)
        #     f = open(csv_dir, 'w', newline="")
        #     writer = csv.writer(f, delimiter=',')
        #     writer.writerows(cell_power_iORG)
        #     f.close
        #
        # print("Done!")
        # print(stimulus_train)

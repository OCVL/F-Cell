import math
import os
from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL, simpledialog

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import patches as ptch
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist, squareform

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles, \
    refine_coord, refine_coord_to_stack, exclude_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG, wavelet_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.pycoordclip import coordclip
from ocvl.function.utility.resources import save_video, save_tiff_stack
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles
from datetime import datetime, date, time, timezone


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

    a_mode = simpledialog.askstring(title="Input the analysis modality string: ",
                                    prompt="Input the analysis modality string:",
                                    initialvalue="760nm", parent=root)
    if not a_mode:
        a_mode = "760nm"

    ref_mode = simpledialog.askstring(title="Input the *alignment reference* modality string. ",
                                      prompt="Input the *alignment reference* modality string:", initialvalue=a_mode,
                                      parent=root)
    if not ref_mode:
        ref_mode = "760nm"

    # Allow for user to specify the segmentation radius - Stolen from BDB
    segmentation_radius = None
    segmentation_radius = simpledialog.askstring(title="Segmentation Radius",
                                                 prompt="Input the segmentation radius to be used:",
                                                 initialvalue=segmentation_radius,
                                                 parent=root)
    if segmentation_radius == "":
        segmentation_radius = None
    else:
        segmentation_radius = int(segmentation_radius)

    x = root.winfo_screenwidth() / 2 - 128
    y = root.winfo_screenheight() / 2 - 128
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.
    root.update()

    allFiles = dict()

    totFiles = 0
    # Parse out the locations and filenames, store them in a hash table.
    searchpath = Path(pName)
    for path in searchpath.rglob("*.avi"):
        if "piped" in path.name:
            splitfName = path.name.split("_")

            if path.parent not in allFiles:
                allFiles[path.parent] = []
                allFiles[path.parent].append(path)
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

    # [ 0:"bob"  1:"moe" 2:"larry" 3:"curly"]
    # Loops through all locations in allFiles
    for l, loc in enumerate(allFiles):
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
            if "ALL_ACQ_AVG" not in file.name:
                # Waitbar stuff
                pb["value"] = r
                pb_label["text"] = "Processing " + file.name + "..."
                pb.update()
                pb_label.update()
                print("Processing " + file.name + "...")

                # Loading in the pipelined data (calls the load_pipelined_data() fxn
                dataset = MEAODataset(file.as_posix(), stimtrain_path=stimtrain_fName,
                                      analysis_modality=a_mode, ref_modality=ref_mode, stage=PipeStages.PIPELINED)
                dataset.load_pipelined_data()

                # Initialize the dict for individual cells.
                if first:
                    reference_coord_data = dataset.coord_data
                    framerate = dataset.framerate
                    stimulus_train = dataset.stimtrain_frame_stamps
                    ref_im = dataset.reference_im
                    full_profiles = []

                    coorddist = pdist(reference_coord_data, "euclidean")
                    coorddist = squareform(coorddist)
                    coorddist[coorddist == 0] = np.amax(coorddist.flatten())
                    mindist = np.amin(coorddist, axis=-1)

                    reference_coord_data = refine_coord(ref_im, dataset.coord_data)

                    simple_amp = np.empty((len(reference_coord_data), 1))
                    simple_amp[:] = np.nan
                    log_amp = np.empty((len(reference_coord_data), 1))
                    log_amp[:] = np.nan
                    amp_plus1_log = np.empty((len(reference_coord_data), 1))
                    amp_plus1_log[:] = np.nan

                    if not segmentation_radius:
                        segmentation_radius = np.round(np.nanmean(mindist) / 4) if np.round(np.nanmean(mindist) / 4) >= 1 else 1

                        segmentation_radius = int(segmentation_radius)
                        print("Detected segmentation radius: " + str(segmentation_radius))

                    for c in range(len(dataset.coord_data)):
                        cell_framestamps.append([])
                        cell_profiles.append([])

                    first = False

                dataset.coord_data = refine_coord_to_stack(dataset.video_data, ref_im, reference_coord_data)

                full_profiles.append(extract_profiles(dataset.video_data, dataset.coord_data, seg_radius=segmentation_radius, summary="none"))
                temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data, seg_radius=segmentation_radius, summary="median")

                # print(str((stimulus_train[0] - int(0.15 * framerate)) / framerate) + " to " + str(
                #    (stimulus_train[1] + int(0.2 * framerate)) / framerate))
                temp_profiles, num_removed = exclude_profiles(temp_profiles, dataset.framestamps,
                                                 critical_region=np.arange(stimulus_train[0] - int(0.1 * framerate),
                                                                           stimulus_train[1] + int(0.2 * framerate)),
                                                 critical_fraction=0.4)

                norm_temporal_profiles = norm_profiles(temp_profiles, norm_method="mean", video_ref=dataset.video_data)
                stdize_profiles = standardize_profiles(norm_temporal_profiles, dataset.framestamps, stimulus_train[0],
                                                       method="mean_sub")
                #stdize_profiles, dataset.framestamps, nummissed = reconstruct_profiles(stdize_profiles,
                #                                                                       dataset.framestamps)

                # Put the profile of each cell into its own array
                for c in range(len(dataset.coord_data)):
                    cell_framestamps[c].append(
                        dataset.framestamps)  # HERE IS THE PROBLEM, YO - appends extra things to the same cell, despite not being long enough
                    cell_profiles[c].append(stdize_profiles[c, :])
                r += 1

                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]

        #del dataset, temp_profiles, norm_temporal_profiles, stdize_profiles
        del temp_profiles, norm_temporal_profiles, stdize_profiles


        # Rows: Acquisitions
        # Cols: Framestamps
        # Depth: Coordinate
        all_cell_iORG = np.empty((len(allFiles[loc]), max_frmstamp + 1, len(reference_coord_data)))
        all_cell_iORG[:] = np.nan

        full_framestamp_range = np.arange(max_frmstamp + 1)
        cell_power_iORG = np.empty((len(reference_coord_data), max_frmstamp + 1))
        cell_power_iORG[:] = np.nan

        # Make 3D matricies, where:
        # The first dimension (rows) is individual acquisitions, where NaN corresponds to missing data
        # The second dimension (columns) is time
        # The third dimension is each tracked coordinate
        for c in range(len(reference_coord_data)):
            for i, profile in enumerate(cell_profiles[c]):
                all_cell_iORG[i, cell_framestamps[c][i], c] = profile


            cell_profiles[c] = []
            cell_framestamps[c] = []

        for c in range(len(reference_coord_data)):
            cell_power_iORG[c, :], numincl = signal_power_iORG(all_cell_iORG[:, :, c], full_framestamp_range,
                                                               summary_method="rms", window_size=1)

            prestim_amp = np.nanmean(cell_power_iORG[c, 0:stimulus_train[0]])
            poststim_amp = np.nanmax(cell_power_iORG[c, stimulus_train[1]:(stimulus_train[1] + 10)])

            simple_amp[c, 0] = poststim_amp - prestim_amp

        # TODO: Calling the coordclip fxn to return the simple_amp that corresponds to a 100 cone ROI
        # clippedcoords = coordclip(coord_data, 10, 100, 'i')

        # plt.figure(0)
        # plt.clf()
        # for a in range(all_cell_iORG.shape[0]):
        #     plt.plot(full_framestamp_range, all_cell_iORG[a, :, c])
        #     plt.plot(stimulus_train[0], poststim_amp, "rD")
        # plt.hist(simple_amp)
        # plt.plot(cell_power_iORG[c, :])
        #   plt.show(block=False)
        #   plt.waitforbuttonpress()
        # plt.savefig(res_dir.joinpath(this_dirname +  + "_allcell_iORG_amp.png"))

        # find the cells with the min, med, and max amplitude
        min_amp = np.nanmin(simple_amp[0, :])
        [min_amp_row, min_amp_col] = np.where(simple_amp == min_amp)
        # print('min_amp ',min_amp)
        med_amp = np.nanmedian(simple_amp[0, :])
        near_med_amp = find_nearest(simple_amp[0, :], med_amp)
        [med_amp_row, med_amp_col] = np.where(simple_amp == near_med_amp)

        # print('med_amp ', med_amp)
        max_amp = np.nanmax(simple_amp[0, :])
        [max_amp_row, max_amp_col] = np.where(simple_amp == max_amp)
        # print('max_amp ', max_amp)

        dt = datetime.now()
        now_timestamp = dt.strftime("%Y_%m_%d_%H_%M_%S")

        plt.figure(1)
        histbins = np.arange(start=-0.1, stop=0.3, step=0.01) #Humans: -0.2, 1.5, 0.025
        plt.hist(simple_amp[l, :], bins=histbins)
        # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp_" + now_timestamp + ".png"))
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp.svg"))
        plt.close(plt.gcf())



        plt.figure(40) # log hist
        histbins_log = np.arange(start=-3, stop=-0.6, step=0.01)  # Humans: -0.2, 1.5, 0.025
        log_amp = np.log10(simple_amp) # Old: log_amp = np.log10(simple_amp[l,:])
        plt.hist(log_amp, bins=histbins_log)
        # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_log_amp_hist_" + now_timestamp + ".png"))
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp.svg"))
        plt.close(plt.gcf())


        plt.figure(41)  # log hist +1
        histbins_logp1 = np.arange(start=0, stop=0.3, step=0.01)  # Humans: -0.2, 1.5, 0.025
        amp_plus1_log = np.log10(simple_amp + 1)
        print("min ", np.nanmin(amp_plus1_log))
        print("max ", np.nanmax(amp_plus1_log))
        plt.hist(amp_plus1_log, bins=histbins_logp1)
        # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_log_plus1_amp_hist_" + now_timestamp + ".png"))
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp.svg"))
        plt.close(plt.gcf())

        hist_normie = Normalize(vmin=histbins[0], vmax=histbins[-1])
        hist_mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("magma"), norm=hist_normie)

        # simple_amp_norm = (simple_amp-histbins[0])/(histbins[-1] - histbins[0])

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
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_voronoi_" + now_timestamp + ".png"))
        # # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_voronoi.svg"))
        # plt.close(plt.gcf())

        fin_log_amp = np.isfinite(log_amp[:, 0].copy())
        ColorTest = hist_mapper.to_rgba(log_amp[fin_log_amp, 0])


        plt.figure(22)
        plt.imshow(ref_im, cmap='gray', vmin=0, vmax=255)
        #plt.scatter(reference_coord_data[:, 0], reference_coord_data[:, 1], s=(1+(segmentation_radius*2)),
        #            c=ColorTest, alpha=0.5)
        plt.scatter(reference_coord_data[fin_log_amp, 0], reference_coord_data[fin_log_amp, 1], s=(1+(segmentation_radius*2)),
                   c=ColorTest, alpha=0.5)

        #plt.gca().invert_yaxis()
        ax = plt.gca()
        ax.axis('off')
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_indvallcell_iORG_falsecoloroverlay_wImage" + now_timestamp + ".svg"),
                    transparent=True, dpi=72, bbox_inches = "tight", pad_inches = 0)
        plt.close(plt.gcf())

        plt.figure(26)
        plt.imshow(ref_im, cmap='gray', vmin=0, vmax=255)
        plt.scatter(reference_coord_data[fin_log_amp, 0], reference_coord_data[fin_log_amp, 1], s=(1 + (segmentation_radius * 2)),
                    c=ColorTest, alpha=0.5)
        ax = plt.gca()
        plt.show(block=False)
        plt.savefig(
            res_dir.joinpath(this_dirname + "_indvallcell_iORG_falsecoloroverlay_wImagewAxes" + now_timestamp + ".svg"),
            transparent=True, dpi=72, bbox_inches="tight", pad_inches=0)
        plt.close(plt.gcf())

        plt.figure(28)
        plt.imshow(ref_im, cmap='gray', vmin=0, vmax=255)
        plt.scatter(reference_coord_data[fin_log_amp, 0], reference_coord_data[fin_log_amp, 1], s=(1 + (segmentation_radius * 2)),
                    c=ColorTest, alpha=0.5)
        ax = plt.gca()
        plt.colorbar(cax=ax)
        plt.show(block=False)
        plt.savefig(
            res_dir.joinpath(this_dirname + "_indvallcell_iORG_falsecoloroverlay_colorbar" + now_timestamp + ".svg"),
            transparent=True, dpi=72, bbox_inches="tight", pad_inches=0)
        plt.close(plt.gcf())

        plt.figure(24)
        #plt.imshow(ref_im, cmap='gray', vmin=0, vmax=255)
        #ax = plt.gca()
        #plt.clf()
        plt.scatter(reference_coord_data[fin_log_amp, 0], reference_coord_data[fin_log_amp, 1], s=(1 + (segmentation_radius * 2)),
                    c=ColorTest, alpha=0.5)

        # color=hist_mapper.to_rgba(simple_amp[c, 0]
        plt.xlim([0, np.size(ref_im, 0)])
        plt.ylim([0, np.size(ref_im, 1)])
        plt.gca().invert_yaxis()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_indvallcell_iORG_falsecoloroverlay" + now_timestamp + ".svg"),
                    transparent=True, dpi=72, bbox_inches = "tight", pad_inches = 0)
        plt.close(plt.gcf())



        # shifting axes
        norm_cell_power_iORG = cell_power_iORG.copy()
        for c in range(norm_cell_power_iORG.shape[0]):
            norm_cell_power_iORG[c,:] -= norm_cell_power_iORG[c,0]


        # plotting the cells with the min/med/max amplitude
        # plt.figure(300)
        # plt.ylim(-0.5, 1)
        #
        # plt.plot(np.reshape(full_framestamp_range, (stimulus_train[2], 1)).astype('float64'),
        #           np.transpose(norm_cell_power_iORG[266, :]),'b')
        #
        # plt.plot(np.reshape(full_framestamp_range, (stimulus_train[2], 1)).astype('float64'),
        #          np.transpose(norm_cell_power_iORG[235, :]), 'g')
        #
        # plt.plot(np.reshape(full_framestamp_range, (stimulus_train[2], 1)).astype('float64'),
        #          np.transpose(norm_cell_power_iORG[539, :]),'r')
        # stim_rect = ptch.Rectangle((dataset.stimtrain_frame_stamps[0], 0),
        #               (dataset.stimtrain_frame_stamps[1] - dataset.stimtrain_frame_stamps[0]), 1, color = 'gray', alpha = 0.5)
        # plt.gca().add_patch(stim_rect)
        #
        # plt.savefig(res_dir.joinpath(this_dirname + "norm_min_med_max" + now_timestamp + ".svg"))
        # plt.show(block=False)
        # plt.close(plt.gcf())

        # output cell_power_iORG to csv (optional)
        if outputcsv:
            import csv

            csv_dir = res_dir.joinpath(this_dirname + "_cell_power_iORG_" + now_timestamp + ".csv")
            outdata = pd.DataFrame(cell_power_iORG)
            outdata.to_csv(csv_dir, index=False)

            amp_dir = res_dir.joinpath(this_dirname + "_cell_amplitude_" + now_timestamp + ".csv")
            outdata = pd.DataFrame(simple_amp)
            outdata.to_csv(amp_dir, index=False)

            log_amp_dir = res_dir.joinpath(this_dirname + "log10_cell_amplitude_" + now_timestamp + ".csv")
            outdata = pd.DataFrame(log_amp)
            outdata.to_csv(log_amp_dir, index=False)

            log_amp_dir_p1 = res_dir.joinpath(this_dirname + "log10_cell_amplitude_plus1" + now_timestamp + ".csv")
            outdata = pd.DataFrame(amp_plus1_log)
            outdata.to_csv(log_amp_dir_p1, index=False)

        print("Done!")
        print(stimulus_train)

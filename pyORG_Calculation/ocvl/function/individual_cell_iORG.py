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
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG, wavelet_iORG, iORG_signal_metrics
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
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
        segmentation_radius = None

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

                    indiv_iORG_amp = np.empty((len(reference_coord_data), 1))
                    indiv_iORG_amp[:] = np.nan
                    log_indiv_iORG_amp = np.empty((len(reference_coord_data), 1))
                    log_indiv_iORG_amp[:] = np.nan
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

                norm_video_data = norm_video(dataset.video_data, norm_method="score", rescaled=True,
                                             rescale_mean=70, rescale_std=35)

                full_profiles.append(extract_profiles(norm_video_data, dataset.coord_data, seg_radius=segmentation_radius, summary="none"))
                temp_profiles = extract_profiles(norm_video_data, dataset.coord_data, seg_radius=segmentation_radius,
                                                 seg_mask="disk", summary="mean")

                # print(str((stimulus_train[0] - int(0.15 * framerate)) / framerate) + " to " + str(
                #    (stimulus_train[1] + int(0.2 * framerate)) / framerate))
                temp_profiles, num_removed = exclude_profiles(temp_profiles, dataset.framestamps,
                                                 critical_region=np.arange(stimulus_train[0] - int(0.2 * framerate),
                                                                           stimulus_train[1] + int(0.2 * framerate)),
                                                 critical_fraction=0.5)

                stdize_profiles = standardize_profiles(temp_profiles, dataset.framestamps, stimulus_train[0],
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
        del temp_profiles,  stdize_profiles


        # Rows: Acquisitions
        # Cols: Framestamps
        # Depth: Coordinate
        all_cell_iORG = np.empty((len(allFiles[loc]), max_frmstamp + 1, len(reference_coord_data)))
        all_cell_iORG[:] = np.nan

        all_frmstamps = np.arange(max_frmstamp + 1)
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

        # cell_power_iORG[119, :], numincl = signal_power_iORG(all_cell_iORG[:, :, 119], full_framestamp_range,
        #                                                    dataset.stimtrain_frame_stamps, summary_method="rms",
        #                                                      window_size=1, display=True)

        indiv_iORG_amp = np.full((len(reference_coord_data), 1), np.nan)
        indiv_iORG_implicit = np.full((len(reference_coord_data), 1), np.nan)

        for c in range(len(reference_coord_data)):

            cell_power_iORG[c, :], numincl = signal_power_iORG(all_cell_iORG[:, :, c],
                                                               dataset.stimtrain_frame_stamps, summary_method="rms",
                                                               window_size=1, display=False)
            avg_numdata = np.nanmean(numincl)
            prestim_ind = np.logical_and(all_frmstamps < dataset.stimtrain_frame_stamps[0],
                                         all_frmstamps >= (dataset.stimtrain_frame_stamps[0] - int(0.75 * dataset.framerate)))
            poststim_ind = np.logical_and(all_frmstamps >= dataset.stimtrain_frame_stamps[1],
                                          all_frmstamps < (dataset.stimtrain_frame_stamps[1] + int(0.75 * dataset.framerate)))
            poststim = cell_power_iORG[c, poststim_ind]

            if poststim.size == 0 or avg_numdata < (all_cell_iORG.shape[0]/2):
                poststim_amp = np.NaN
                prestim_amp = np.NaN

            else:
                thispower = cell_power_iORG[c, :]

                indiv_iORG_amp[c], indiv_iORG_implicit[c] = iORG_signal_metrics(thispower[None, :], dataset.framestamps,
                                                                        filter_type="none", display=False,
                                                                        prestim_idx=prestim_ind,
                                                                        poststim_idx=poststim_ind)[1:3]
            

        log_indiv_iORG_amp = np.log(indiv_iORG_amp)


        # plt.figure(999)
        # plt.plot(cell_power_iORG.transpose())
        # plt.show(block=False)
        # plt.waitforbuttonpress()


        # TODO: Calling the coordclip fxn to return the simple_amp that corresponds to a 100 cone ROI
        # clippedcoords = coordclip(coord_data, 10, 100, 'i')


        dt = datetime.now()
        now_timestamp = dt.strftime("%Y_%m_%d_%H_%M_%S")

        plt.figure(1)
        histbins = np.arange(start=0, stop=110, step=1) #Humans: -0.2, 1.5, 0.025 Animal: start=-0.1, stop=0.3, step=0.01
        plt.hist(indiv_iORG_amp[:, 0], bins=histbins)
        # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp_" + now_timestamp + ".png"))
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp.svg"))
        plt.close(plt.gcf())

        # plt.figure(18)
        # histbins = np.arange(start=0, stop=100, step=0.1) #Humans: -0.2, 1.5, 0.025 Animal: start=-0.1, stop=0.3, step=0.01
        # plt.hist(simple_amp[:, 0], bins=histbins, density=True, histtype="step", cumulative=True)
        # # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp_cumhist_" + now_timestamp + ".png"))
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp_cumhist_" + now_timestamp + ".svg"))
        # plt.close(plt.gcf())



        plt.figure(40) # log hist
        histbins_log = np.arange(start=0, stop=5.5, step=0.05)  # Humans: -0.2, 1.5, 0.025 Animal: start=-3, stop=-0.6, step=0.01 stop=round(np.nanmax(log_amp))
        plt.hist(log_indiv_iORG_amp, bins=histbins_log)
        # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_log_amp_hist_" + now_timestamp + ".png"))
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_log_amp_hist" + now_timestamp + ".svg"))
        plt.close(plt.gcf())

        plt.figure(48) # log hist
        histbins_log = np.arange(start=0, stop=5.5, step=0.01)  # Humans: -0.2, 1.5, 0.025 Animal: start=-3, stop=-0.6, step=0.01 stop=round(np.nanmax(log_amp))
        plt.hist(log_indiv_iORG_amp, bins=histbins_log, density=True, histtype="step", cumulative=True)
        # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_log_amp_cumhist_" + now_timestamp + ".png"))
        plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_log_amp_cumhist_" + now_timestamp + ".svg"))
        plt.close(plt.gcf())


        # plt.figure(41)  # log hist +1
        # histbins_logp1 = np.arange(start=0, stop=5.5, step=0.01)  # Humans: -0.2, 1.5, 0.025 Animal: start=-3, stop=-0.6, step=0.01
        # amp_plus1_log[:, 0] = np.log10(simple_amp[:, 0]+1)
        # print("min ", np.nanmin(amp_plus1_log))
        # print("max ", np.nanmax(amp_plus1_log))
        # plt.hist(amp_plus1_log, bins=histbins_logp1)
        # # plt.plot(cell_power_iORG[c, :], "k-", alpha=0.05)
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_log_plus1_amp_hist_" + now_timestamp + ".png"))
        # # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_amp.svg"))
        # plt.close(plt.gcf())


        hist_normie = Normalize(vmin=0.25, vmax=5.5)
        hist_mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("magma"), norm=hist_normie)


        # plt.figure(2)
        # vor = Voronoi(reference_coord_data)
        # voronoi_plot_2d(vor, show_vertices=False, show_points=False)
        # for c, cell in enumerate(vor.regions[1:]):
        #     if not -1 in cell:
        #         poly = [vor.vertices[i] for i in cell]
        #         plt.fill(*zip(*poly), color=hist_mapper.to_rgba(log_amp[c, 0]))
        # ax = plt.gca()
        # ax.set_aspect("equal", adjustable="box")
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_allcell_iORG_logvoronoi_" + now_timestamp + ".png"))
        # plt.close(plt.gcf())
        #
        fin_log_amp = np.isfinite(log_indiv_iORG_amp[:, 0].copy())
        ColorTest = hist_mapper.to_rgba(log_indiv_iORG_amp[fin_log_amp, 0])
        #ColorTest = hist_mapper.to_rgba(log_amp[:, 0])


        # plt.figure(22)
        # plt.imshow(ref_im, cmap='gray', vmin=0, vmax=255)
        # #plt.scatter(reference_coord_data[:, 0], reference_coord_data[:, 1], s=(1+(segmentation_radius*2)),
        # #            c=ColorTest, alpha=0.5)
        # plt.scatter(reference_coord_data[fin_log_amp, 0], reference_coord_data[fin_log_amp, 1], s=(1+(segmentation_radius*2)),
        #            c=ColorTest, alpha=0.5)

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

        #Voronoi and false color overlay colorbar
        fig7, ax7 = plt.subplots()
        ax7.set_aspect('equal')
        tst = ax7.scatter(reference_coord_data[fin_log_amp, 0], reference_coord_data[fin_log_amp, 1], s=(1 + (segmentation_radius * 2)),
                    c=ColorTest, alpha=0.5)
        fig7.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap("magma"), norm=hist_normie), ax7)
        plt.savefig(res_dir.joinpath(this_dirname + "_indvallcell_iORG_falsecoloroverlay_colorbar" + now_timestamp + ".svg"),
            transparent=False, dpi=72, bbox_inches="tight", pad_inches=0)
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
                    transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
        plt.close(plt.gcf())



        # shifting axes
        norm_cell_power_iORG = cell_power_iORG.copy()
        for c in range(norm_cell_power_iORG.shape[0]):
            norm_cell_power_iORG[c,:] -= norm_cell_power_iORG[c,0]


        # output cell_power_iORG to csv (optional)
        if outputcsv:

            csv_dir = res_dir.joinpath(this_dirname + "_cell_power_iORG_" + now_timestamp + ".csv")
            outdata = pd.DataFrame(cell_power_iORG)
            outdata.to_csv(csv_dir, index=False)

            amp_dir = res_dir.joinpath(this_dirname + "_cell_amplitude_" + now_timestamp + ".csv")
            outdata = pd.DataFrame(indiv_iORG_amp)
            outdata.to_csv(amp_dir, index=False)

            log_amp_dir = res_dir.joinpath(this_dirname + "log_cell_amplitude_" + now_timestamp + ".csv")
            outdata = pd.DataFrame(log_indiv_iORG_amp)
            outdata.to_csv(log_amp_dir, index=False)

            log_cumhist = np.histogram(log_indiv_iORG_amp, bins=histbins_log, density=True)

            log_cumhist_dir = res_dir.joinpath(this_dirname + "_log_amplitude_cumhist_" + now_timestamp + ".csv")
            outdata = pd.DataFrame(log_cumhist)
            outdata.to_csv(log_cumhist_dir, index=False)

            # log_amp_dir_p1 = res_dir.joinpath(this_dirname + "log10_cell_amplitude_plus1" + now_timestamp + ".csv")
            # outdata = pd.DataFrame(amp_plus1_log)
            # outdata.to_csv(log_amp_dir_p1, index=False)

        print("Done!")
        print(stimulus_train)

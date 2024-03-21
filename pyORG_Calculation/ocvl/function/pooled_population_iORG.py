import os
from multiprocessing import Pool
from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL, simpledialog

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles, \
    refine_coord, refine_coord_to_stack, exclude_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG, iORG_signal_metrics
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_tiff_stack
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles, trim_video
from datetime import datetime, date, time, timezone

def pop_iORG(dataset):
    temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data[perm, :], seg_radius=2,
                                     display=False, sigma=1)

    temp_profiles, num_removed = exclude_profiles(temp_profiles, dataset.framestamps,
                                                  critical_region=np.arange(
                                                      dataset.stimtrain_frame_stamps[0] - int(0.1 * dataset.framerate),
                                                      dataset.stimtrain_frame_stamps[1] + int(0.2 * dataset.framerate)),
                                                  critical_fraction=0.4)

    stdize_profiles = standardize_profiles(temp_profiles, dataset.framestamps,
                                           dataset.stimtrain_frame_stamps[0], method="mean_sub", display=False)

    tmp_iorg, tmp_incl = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="rms")

    _, amplitude, intrinsic_time = iORG_signal_metrics(tmp_iorg, dataset.framestamps,
                                                       filter_type="MS1", display=False,
                                                       prestim_idx=prestim_ind, poststim_idx=poststim_ind)


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

    # Allow for user to specify the segmentation radius
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


    reference_coord_data = None
    maxnum_cells = None

    skipnum = 0

    for loc in allFiles:
        res_dir = loc.joinpath("Results")
        res_dir.mkdir(exist_ok=True)

        this_dirname = res_dir.parent.name

        r = 0
        pb["maximum"] = len(allFiles[loc])
        pop_iORG = []
        pop_iORG_implicit = np.empty((len(allFiles[loc])-skipnum+1))
        pop_iORG_implicit[:] = np.nan
        pop_iORG_recover = np.empty((len(allFiles[loc]) - skipnum + 1))
        pop_iORG_recover[:] = np.nan
        pop_iORG_amp = np.empty((len(allFiles[loc]) - skipnum + 1))
        pop_iORG_amp[:] = np.nan
        pop_iORG_num = []
        framestamps = []
        max_frmstamp = 0
        plt.figure(0)
        plt.clf()
        first = True
        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(allFiles[loc])))

        for file in allFiles[loc]:

            if "ALL_ACQ_AVG" not in file.name and r >= skipnum:
                pb["value"] = r
                pb_label["text"] = "Processing " + file.name + "..."
                pb.update()
                pb_label.update()

                dataset = MEAODataset(file.as_posix(), analysis_modality=a_mode, ref_modality=ref_mode,
                                      stimtrain_path=stimtrain_fName, stage=PipeStages.PIPELINED)
                dataset.load_pipelined_data()

                if first:
                    reference_coord_data = refine_coord(dataset.reference_im, dataset.coord_data)
                    coorddist = pdist(reference_coord_data, "euclidean")
                    coorddist = squareform(coorddist)
                    coorddist[coorddist == 0] = np.amax(coorddist.flatten())
                    mindist = np.amin( coorddist, axis=-1)

                    if not segmentation_radius:
                        segmentation_radius = np.round(np.nanmean(mindist) / 4) if np.round(np.nanmean(mindist) / 4) >= 1 else 1

                        segmentation_radius = int(segmentation_radius)
                        print("Detected segmentation radius: " + str(segmentation_radius))

                    full_profiles = []
                    first = False

                dataset.coord_data = reference_coord_data
                
                dataset.coord_data = refine_coord_to_stack(dataset.video_data, dataset.reference_im, reference_coord_data)

                dataset.video_data = norm_video(dataset.video_data, norm_method="mean", rescaled=True)

                # Clip out data beyond two seconds before and after.
                # dataset.video_data, dataset.framestamps = trim_video(dataset.video_data, dataset.framestamps,
                #                                                      dataset.stimtrain_frame_stamps[0] + int(2 * dataset.framerate))

                if maxnum_cells is not None:
                    numiter=10000

                else:
                    perm = np.arange(len(dataset.coord_data))
                print("Analyzing " + str(len(perm)) + " cells.")

                temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data[perm, :], seg_radius=segmentation_radius,
                                                 display=False, sigma=1)


                temp_profiles, valid_profiles = exclude_profiles(temp_profiles, dataset.framestamps,
                                                                 critical_region=np.arange(
                                                                  dataset.stimtrain_frame_stamps[0] - int(0.1 * dataset.framerate),
                                                                  dataset.stimtrain_frame_stamps[1] + int(0.2 * dataset.framerate)),
                                                                 critical_fraction=0.1)

                if np.sum(~valid_profiles) == len(perm):
                    pop_iORG_amp[r] = np.NaN
                    pop_iORG_implicit[r] = np.NaN
                    pop_iORG_recover[r] = np.NaN
                    print(file.name + " was dropped due to all cells being excluded.")

                stdize_profiles = standardize_profiles(temp_profiles, dataset.framestamps,
                                                       dataset.stimtrain_frame_stamps[0], method="mean_sub")

                # plt.figure(1)
                # plt.clf()
                # for i in range(stdize_profiles.shape[0]):
                #     plt.plot(dataset.framestamps, stdize_profiles[i, :])
                # plt.show(block=False)
                # plt.waitforbuttonpress()

                tmp_iorg, tmp_incl = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="rms",
                                                       window_size=1)

                tmp_iorg = standardize_profiles(tmp_iorg[None, :], dataset.framestamps,
                                                dataset.stimtrain_frame_stamps[0], method="mean_sub")

                tmp_iorg = np.squeeze(tmp_iorg)

                prestim_ind = np.flatnonzero(np.logical_and(dataset.framestamps < dataset.stimtrain_frame_stamps[0],
                                             dataset.framestamps >= (dataset.stimtrain_frame_stamps[0] - int(
                                                 1 * dataset.framerate))))
                poststim_ind = np.flatnonzero(np.logical_and(dataset.framestamps >= dataset.stimtrain_frame_stamps[1],
                                              dataset.framestamps < (dataset.stimtrain_frame_stamps[1] + int(
                                                  1 * dataset.framerate))))
                poststim_loc = dataset.framestamps[poststim_ind]
                prestim_amp = np.nanmedian(tmp_iorg[prestim_ind])
                poststim = tmp_iorg[poststim_ind]

                if poststim.size == 0:
                    poststim_amp = np.NaN
                    prestim_amp = np.NaN
                    pop_iORG_amp[r] = np.NaN
                    pop_iORG_implicit[r] = np.NaN
                    pop_iORG_recover[r] = np.NaN
                else:
                    poststim_amp = np.quantile(poststim, [0.95])
                    max_frmstmp = poststim_loc[np.argmax(poststim)] - dataset.stimtrain_frame_stamps[0]
                    final_val = np.mean(tmp_iorg[-5:])

                    framestamps.append(dataset.framestamps)
                    pop_iORG.append(tmp_iorg)
                    pop_iORG_num.append(tmp_incl)

                    pop_iORG_amp[r], pop_iORG_implicit[r] = iORG_signal_metrics(tmp_iorg[None, :], dataset.framestamps,
                                                                      filter_type="none", display=False,
                                                                      prestim_idx=prestim_ind,
                                                                      poststim_idx=poststim_ind)[1:3]

                    pop_iORG_recover[r] = 1 - ((final_val - prestim_amp) / pop_iORG_amp[r])
                    pop_iORG_implicit[r] = 1000*pop_iORG_implicit[r] / dataset.framerate

                    print("Signal metrics based iORG Amplitude: " + str(pop_iORG_amp[r]) +
                          " Implicit time (ms): " + str(pop_iORG_implicit[r]) +
                          " Recovery fraction: " + str(pop_iORG_recover[r]))

                    plt.figure(0)

                    plt.xlabel("Time (seconds)")
                    plt.ylabel("Response")
                    plt.plot(dataset.framestamps/dataset.framerate, pop_iORG[r - skipnum], color=mapper.to_rgba(r - skipnum, norm=False),
                             label=file.name)
                    plt.show(block=False)
                    #plt.savefig(res_dir.joinpath(file.name[0:-4] + "_pop_iORG.png"))
                    r += 1

                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]

        dt = datetime.now()
        now_timestamp = dt.strftime("%Y_%m_%d_%H_%M_%S")

        plt.vlines(dataset.stimtrain_frame_stamps[0] / dataset.framerate, -1, 10, color="red")
        plt.xlim([0,  6])
        plt.ylim([-5, 60]) #was 60
        #plt.legend()

        plt.savefig( res_dir.joinpath(this_dirname + "_pop_iORG_" + now_timestamp + ".svg"))
        plt.savefig( res_dir.joinpath(this_dirname + "_pop_iORG_" + now_timestamp + ".png"))

        # plt.figure(14)
        # plt.plot(np.nanmean(np.log(pop_iORG_amp), axis=-1),
        #          np.nanstd(np.log(pop_iORG_amp), axis=-1),".")
        # plt.title("logAMP mean vs logAMP std dev")
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_pop_iORG_logamp_vs_stddev.svg"))
        #
        # plt.figure(15)
        # plt.plot(np.nanmean(pop_iORG_amp, axis=-1),
        #          np.nanstd(pop_iORG_amp, axis=-1),".")
        # plt.title("AMP vs std dev")
        # plt.show(block=False)
        # plt.savefig(res_dir.joinpath(this_dirname + "_pop_iORG_amp_vs_stddev.svg"))
        print("Pop mean iORG amplitude: " + str(np.nanmean(pop_iORG_amp, axis=-1)) +
              "Pop stddev iORG amplitude: " + str(np.nanmean(pop_iORG_amp, axis=-1)) )


        # pop_amp_dFrame = pd.DataFrame(np.concatenate((np.array(pop_iORG_amp, ndmin=2).transpose(),
        #                                               np.array(pop_iORG_implicit, ndmin=2).transpose(),
        #                                               np.array(pop_iORG_recover, ndmin=2).transpose()), axis=1),
        #                               columns=["Amplitude", "Implicit time", "Recovery %"])
        # pop_amp_dFrame.to_csv(res_dir.joinpath(this_dirname + "_pop_iORG_stats_" + now_timestamp + ".csv"))

        # Grab all of the
        all_iORG = np.empty((len(pop_iORG), max_frmstamp+1))
        all_iORG[:] = np.nan
        all_incl = np.empty((len(pop_iORG), max_frmstamp + 1))
        all_incl[:] = np.nan
        for i, iorg in enumerate(pop_iORG):
            all_incl[i, framestamps[i]] = pop_iORG_num[i]
            all_iORG[i, framestamps[i]] = iorg

        # Pooled variance calc
        pooled_iORG = np.nansum( all_incl*all_iORG, axis=0 ) / np.nansum(all_incl, axis=0)
        #pooled_stddev_iORG = np.sqrt(pooled_var_iORG)
        all_frmstamps = np.arange(max_frmstamp+1)

        prestim_ind = np.logical_and(all_frmstamps < dataset.stimtrain_frame_stamps[0],
                                     all_frmstamps >= (dataset.stimtrain_frame_stamps[0] - int(1 * dataset.framerate)))
        poststim_ind = np.logical_and(all_frmstamps >= dataset.stimtrain_frame_stamps[1],
                                      all_frmstamps < (dataset.stimtrain_frame_stamps[1] + int(1 * dataset.framerate)))
        poststim_loc = all_frmstamps[poststim_ind]
        prestim_amp = np.nanmedian(pooled_iORG[prestim_ind])
        poststim = pooled_iORG[poststim_ind]

        if poststim.size == 0:
            poststim_amp = np.NaN
            prestim_amp = np.NaN
            pop_iORG_amp[r] = np.NaN
            pop_iORG_implicit[r] = np.NaN
            pop_iORG_recover[r] = np.NaN
        else:
            poststim_amp = np.quantile(poststim, [0.95])
            max_frmstmp = poststim_loc[np.argmax(poststim)] - dataset.stimtrain_frame_stamps[0]

            final_val = np.mean(pooled_iORG[-5:])
            pop_iORG_implicit[r] = 1000 * max_frmstmp / dataset.framerate
            pop_iORG_amp[r] = (poststim_amp - prestim_amp)
            pop_iORG_recover[r] = 1 - ((final_val - prestim_amp) / pop_iORG_amp[r])

        print("Pooled iORG Avg Amplitude: " + str(pop_iORG_amp[r]) + " Implicit time (ms): " + str(pop_iORG_implicit[r]) +
              " Recovery fraction: " + str(pop_iORG_recover[r]))

        pop_amp_dFrame = pd.DataFrame(np.concatenate((np.array(pop_iORG_amp, ndmin=2).transpose(),
                                                      np.array(pop_iORG_implicit, ndmin=2).transpose(),
                                                      np.array(pop_iORG_recover, ndmin=2).transpose()), axis=1),
                                      columns=["Amplitude", "Implicit time", "Recovery %"])
        pop_amp_dFrame.to_csv(res_dir.joinpath(this_dirname + "_pop_iORG_stats_" + now_timestamp + ".csv"))

        # Output the pop iORG for each frame and trial -- wanted feature
        tmp_lst = []
        for i in range(0, len(pop_iORG)):
            tmp = "Trial " + str(i)
            tmp_lst.append(tmp)

        pop_by_frame_dFrame = pd.DataFrame(np.array(pop_iORG).transpose(), columns=tmp_lst)
        pop_by_frame_dFrame.to_csv(res_dir.joinpath(this_dirname + "_pop_iORG_" + now_timestamp + ".csv"))

        plt.figure(1)

        plt.plot(all_frmstamps / dataset.framerate, pooled_iORG)
        plt.vlines(dataset.stimtrain_frame_stamps[0] / dataset.framerate, -1, 10, color="red")
        plt.xlim([0, 6])
        plt.ylim([-5, 60]) #was 1, 60
        plt.xlabel("Time (seconds)")
        plt.ylabel("Response")
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_pooled_pop_iORG_" + now_timestamp + ".png"))
        plt.savefig(res_dir.joinpath(this_dirname + "_pooled_pop_iORG_" + now_timestamp + ".svg"))
        print("Done!")



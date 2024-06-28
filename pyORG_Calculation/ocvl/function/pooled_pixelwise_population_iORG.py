import os
from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import numpy as np
import pandas as pd
import matplotlib
import re
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_video
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles
from datetime import datetime, date, time, timezone

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
    skipnum = 0

    all_pooled_init = 1
    pooled_incl = 0

    for loc in allFiles:
        res_dir = loc.joinpath("Results")
        res_dir.mkdir(exist_ok=True)

        this_dirname = res_dir.parent.name

        r = 0
        pb["maximum"] = len(allFiles[loc])
        pop_iORG = []
        profile_data = []
        pop_iORG_amp = np.empty((len(allFiles[loc])-skipnum+1))
        pop_iORG_amp[:] = np.nan
        pop_iORG_num = []
        framestamps = []
        max_frmstamp = 0
        plt.figure(0)
        plt.clf()

        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(allFiles[loc])).reversed())

        for file in allFiles[loc]:

            if "ALL_ACQ_AVG" not in file.name and r >= skipnum:
                pb["value"] = r
                pb_label["text"] = "Processing " + file.name + "..."
                pb.update()
                pb_label.update()

                print("Processing " + file.name + "...")

                dataset = MEAODataset(file.as_posix(), analysis_modality="760nm", ref_modality="760nm",
                                      stimtrain_path=stimtrain_fName, stage=PipeStages.PIPELINED)
                dataset.load_pipelined_data()

                if first:
                    width = dataset.video_data.shape[1]
                    height = dataset.video_data.shape[0]
                    y = np.arange(0, dataset.video_data.shape[0])
                    x = np.arange(0, dataset.video_data.shape[1])
                    xv, yv =np.meshgrid(x, y)

                    xv = np.reshape(xv, (xv.size, 1))
                    yv = np.reshape(yv, (yv.size, 1))

                    coord_data = np.hstack((xv, yv))
                    del x, y, xv, yv
                    first = False

                temp_profiles = extract_profiles(dataset.video_data, coord_data, seg_radius=0)

                stdize_profiles = standardize_profiles(temp_profiles, dataset.framestamps,
                                                       dataset.stimtrain_frame_stamps[0], method="mean_sub")
                #stdize_profiles, dataset.framestamps, nummissed = reconstruct_profiles(stdize_profiles, dataset.framestamps)
                #plt.savefig(res_dir.joinpath(this_dirname +  "_all_std_profiles.svg"))

                tmp_iorg, tmp_incl = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="rms", window_size=1)

                tmp_iorg = standardize_profiles(tmp_iorg[None, :], dataset.framestamps,
                                                dataset.stimtrain_frame_stamps[0], method="mean_sub")
                tmp_iorg = np.squeeze(tmp_iorg)

                prestim_amp = np.nanmedian(tmp_iorg[0:dataset.stimtrain_frame_stamps[0]])
                poststim = tmp_iorg[dataset.stimtrain_frame_stamps[1]:(dataset.stimtrain_frame_stamps[1] + 15)]
                if poststim.size == 0:
                    poststim_amp = 0
                    prestim_amp = 0
                    pop_iORG_amp[r] = 0
                else:
                    poststim_amp = np.amax(poststim)
                    pop_iORG_amp[r] = (poststim_amp - prestim_amp)

                    profile_data.append(stdize_profiles)
                    framestamps.append(dataset.framestamps)
                    pop_iORG.append(tmp_iorg)
                    pop_iORG_num.append(tmp_incl)

                    print("iORG Simple Amplitude: " + str(pop_iORG_amp[r]) + " (prestim: " + str(prestim_amp) +
                          " poststim: " + str(poststim_amp) + ")")

                    plt.figure(0)
                    plt.plot(dataset.framestamps, pop_iORG[r - skipnum], color=mapper.to_rgba(r - skipnum, norm=False),
                             label=file.name)
                    plt.show(block=False)

                    r += 1

                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]


        dt = datetime.now()
        now_timestamp = dt.strftime("%Y_%m_%d_%H_%M_%S")

        #plt.legend()
        plt.savefig( res_dir.joinpath(this_dirname + "_pixelpop_iORG_" + now_timestamp + ".svg"))
        plt.savefig( res_dir.joinpath(this_dirname + "_pixelpop_iORG_" + now_timestamp + ".png") )




        # Grab all of the
        all_iORG = np.empty((len(pop_iORG), max_frmstamp+1))
        all_iORG[:] = np.nan
        all_incl = np.empty((len(pop_iORG), max_frmstamp + 1))
        all_incl[:] = np.nan
        num_profiles = np.zeros((1, max_frmstamp + 1))
        all_profiles = np.zeros((len(coord_data), max_frmstamp + 1))

        for i, iorg in enumerate(pop_iORG):
            all_incl[i, framestamps[i]] = pop_iORG_num[i]
            all_iORG[i, framestamps[i]] = iorg

            profile_data[i][np.isnan(profile_data[i])] = 0
            all_profiles[:, framestamps[i]] += profile_data[i] * profile_data[i]
            num_profiles[0, framestamps[i]] += 1

        all_profiles /= num_profiles
        all_profiles = np.sqrt(all_profiles)

        prestimRMS = np.nanmedian(all_profiles[:, 0:dataset.stimtrain_frame_stamps[0]])

        all_profiles -= prestimRMS

       # all_profiles = np.log(all_profiles)

        all_profiles[~np.isfinite(all_profiles)] = 0
        video_profiles = np.reshape(all_profiles, (height, width, max_frmstamp+1))

        hist_normie = Normalize(vmin=np.nanpercentile(video_profiles[:], 2.5), vmax=np.nanpercentile(video_profiles[:], 99))
        hist_mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("inferno"), norm=hist_normie)

        save_video(res_dir.joinpath(this_dirname + "_pooled_pixelpop_iORG_" + now_timestamp + ".avi").as_posix(),
                   video_profiles, dataset.framerate,
                   scalar_mapper=hist_mapper)

        print("Video 5th percentile: " + str(np.nanpercentile(video_profiles[:], 2.5)))
        print("Video 99th percentile: " + str(np.nanpercentile(video_profiles[:], 99)))
        # video_profiles -= np.nanpercentile(video_profiles[:], 5)
        video_profiles -= hist_normie.vmin
        video_profiles /= (hist_normie.vmax - hist_normie.vmin)

        video_profiles[video_profiles < 0] = 0
        video_profiles[video_profiles > 1] = 1
        video_profiles *= 255
        save_video(res_dir.joinpath(this_dirname + "_pooled_pixelpop_iORG_gray_" + now_timestamp + ".avi").as_posix(),
                   video_profiles, 29.4)

        # Pooled variance calc

        pooled_iORG = np.nansum( all_incl*all_iORG, axis=0 ) / np.nansum(all_incl, axis=0)
        #pooled_stddev_iORG = np.sqrt(pooled_var_iORG)

        prestim_amp = np.nanmedian(pooled_iORG[0:dataset.stimtrain_frame_stamps[0]])
        poststim = pooled_iORG[dataset.stimtrain_frame_stamps[1]:(dataset.stimtrain_frame_stamps[1] + 15)]
        if poststim.size == 0:
            poststim_amp = 0
        else:
            poststim_amp = np.amax(poststim)

        pop_iORG_amp[r] = (poststim_amp - prestim_amp)

        print("iORG Avg Amplitude: " + str(pop_iORG_amp[r]) + " (prestim: " + str(prestim_amp) +
              " poststim: " + str(poststim_amp) + ")")

        pop_dFrame = pd.DataFrame(np.concatenate((all_iORG,
                                                  np.reshape(pooled_iORG, (1, len(pooled_iORG)))), axis=0))
        pop_dFrame.to_csv(res_dir.joinpath(this_dirname + "_pixelpop_iORG_" + now_timestamp + ".csv"), header=False)

        pop_amp_dFrame = pd.DataFrame(pop_iORG_amp)
        pop_amp_dFrame.to_csv(res_dir.joinpath(this_dirname + "_pixelpop_iORG_amp_" + now_timestamp + ".csv"), header=False)

        plt.figure(1)
        plt.clf()
        plt.plot(pooled_iORG)
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_pooled_pixelpop_iORG_" + now_timestamp + ".png"))
        plt.savefig(res_dir.joinpath(this_dirname + "_pooled_pixelpop_iORG_" + now_timestamp + ".svg"))
        print("Done!")

        if all_pooled_init == 1:
            all_trial_pooled = np.empty([len(allFiles), len(pooled_iORG)])
            all_trial_pooled[:] = 0
            tmp_dir_pooled = []
            all_pooled_init = 2


        if re.search(',', this_dirname):
            all_trial_pooled[pooled_incl] = pooled_iORG
            tmp_dir_pooled.append(this_dirname)

            pooled_incl += 1

    # plt.figure(40)
    # plt.ylim(0, 1)
    #
    # plt.plot(range(len(all_trial_pooled[0,])), all_trial_pooled[0,], range(len(all_trial_pooled[1,])),
    #         all_trial_pooled[1,], range(len(all_trial_pooled[2,])), all_trial_pooled[2,],
    #        range(len(all_trial_pooled[3,])), all_trial_pooled[3,])
    #
    # # plt.plot(range(len(all_trial_pooled[0,])), all_trial_pooled[0,], range(len(all_trial_pooled[1,])),
    # #         all_trial_pooled[1,])
    # stim_rect = matplotlib.patches.Rectangle((dataset.stimtrain_frame_stamps[0], 0),
    #                                          (dataset.stimtrain_frame_stamps[1] - dataset.stimtrain_frame_stamps[0]), 1, color = 'gray', alpha = 0.5)
    # plt.gca().add_patch(stim_rect)
    # plt.savefig(searchpath.joinpath(splitfName[0] + "_all_trials_pooled_pixelpop_iORG_" + now_timestamp + ".png"))
    # plt.savefig(searchpath.joinpath(splitfName[0] + "_all_trials_pooled_pixelpop_iORG_" + now_timestamp + ".svg"))


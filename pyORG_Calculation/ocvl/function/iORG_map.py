import os
from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.dataset import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles

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

    maxnum_cells = None
    skipnum = 0

    for loc in allFiles:
        res_dir = loc.joinpath("Results")
        res_dir.mkdir(exist_ok=True)

        this_dirname = res_dir.parent.name

        r = 0
        pb["maximum"] = len(allFiles[loc])
        pop_iORG = []
        pop_iORG_num=[]
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

                dataset = MEAODataset(file.as_posix(), analysis_modality="760nm", ref_modality="760nm",
                                      stimtrain_path=stimtrain_fName, stage=PipeStages.PIPELINED)
                dataset.load_pipelined_data()

                if maxnum_cells is not None:
                    perm = np.random.permutation(len(dataset.coord_data))
                    perm = perm[0:maxnum_cells]
                else:
                    perm = np.arange(len(dataset.coord_data))
                print("Analyzing " + str(len(perm)) + " cells.")

                temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data[perm, :], seg_radius=2)
                norm_temporal_profiles = norm_profiles(temp_profiles, norm_method="mean")
                stdize_profiles = standardize_profiles(norm_temporal_profiles, dataset.framestamps,
                                                       dataset.stimtrain_frame_stamps[0], method="mean_sub")
                #stdize_profiles, dataset.framestamps, nummissed = reconstruct_profiles(stdize_profiles, dataset.framestamps)
                #plt.savefig(os.path.join(res_dir, file[0:-4] + "_all_std_profiles.svg"))

                framestamps.append(dataset.framestamps)
                tmp_iorg, tmp_incl = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="var", window_size=1)
                pop_iORG.append(tmp_iorg)
                pop_iORG_num.append(tmp_incl)
                plt.figure(0)
                plt.plot(dataset.framestamps, pop_iORG[r-skipnum], color=mapper.to_rgba(r-skipnum, norm=False), label=file.name)
                plt.show(block=False)

                if dataset.framestamps[-1] > max_frmstamp:
                    max_frmstamp = dataset.framestamps[-1]

            r += 1
        #plt.legend()
        plt.savefig( res_dir.joinpath(this_dirname + "_pop_iORG.png") )

        del dataset
        # Grab all of the
        all_iORG = np.empty((len(pop_iORG), max_frmstamp+1))
        all_iORG[:] = np.nan
        all_incl = np.empty((len(pop_iORG), max_frmstamp + 1))
        all_incl[:] = np.nan
        for i, iorg in enumerate(pop_iORG):
            all_incl[i, framestamps[i]] = pop_iORG_num[i]
            all_iORG[i, framestamps[i]] = iorg

        # Pooled variance calc
        pooled_var_iORG = np.nansum( (all_incl-1)*all_iORG, axis=0 ) / np.nansum(all_incl-1, axis=0)
        pooled_stddev_iORG = np.sqrt(pooled_var_iORG)

        pop_dFrame = pd.DataFrame(np.concatenate((all_iORG,
                                                  np.reshape(pooled_stddev_iORG, (1, len(pooled_stddev_iORG)))), axis=0))
        pop_dFrame.to_csv(res_dir.joinpath(this_dirname + "_pop_iORG.csv"), header=False)

        plt.figure(1)
        plt.clf()
        plt.plot(pooled_stddev_iORG)
        plt.show(block=False)
        plt.savefig(res_dir.joinpath(this_dirname + "_pooled_pop_iORG.png"))
        plt.savefig(res_dir.joinpath(this_dirname + "_pooled_pop_iORG.svg"))
        print("Done!")


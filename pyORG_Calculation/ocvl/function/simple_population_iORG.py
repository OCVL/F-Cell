import os
from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL

from matplotlib import pyplot as plt

from ocvl.function.analysis.iORG_signal_extraction import extract_profiles, norm_profiles, standardize_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.dataset import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles

if __name__ == "__main__":
    root = Tk() #If you use this outside of main, shit goes sideways...
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

    allFiles = dict()  # Is a "relational database" or a "hash table" AKA- key : value

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

    for loc in allFiles:
        res_dir = loc.joinpath("Results")
        res_dir.mkdir(exist_ok=True)

        this_dirname = res_dir.parent.name

        r = 0
        pb["maximum"] = len(allFiles[loc]) # Updates the max for the progressbar- sets top value for this location
        for file in allFiles[loc]:
            if "ALL_ACQ_AVG" not in file.name: # If the file isn't our average of all videos tif file, then process it
                # More progressbar shit
                pb["value"] = r
                pb_label["text"] = "Processing " + file.name + "..."
                pb.update()
                pb_label.update()

                dataset = MEAODataset(file.as_posix(), analysis_modality="760nm", ref_modality="760nm",
                                      stimtrain_path=stimtrain_fName, stage=PipeStages.PIPELINED) # To handle loading of datasets (of difference stages)
                dataset.load_data()

                temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data) # Extracts temporal profiles from data
                norm_temporal_profiles = norm_profiles(temp_profiles, norm_method="mean") # Normalizes the temporal profiles
                stdize_profiles = standardize_profiles(norm_temporal_profiles, dataset.framestamps,
                                                       dataset.stimtrain_frame_stamps[0], method="mean_sub") # Standardizes the temporal profiles
                #stdize_profiles, dataset.framestamps, nummissed = reconstruct_profiles(stdize_profiles, dataset.framestamps) # Reconstruct via compressive sensing

                pop_iORG, num_incl = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="rms", window_size=0)

                plt.figure(0)
                plt.clf()
                plt.plot(dataset.framestamps, pop_iORG)
                plt.show(block=False)
                plt.savefig(res_dir.joinpath(this_dirname + "_pop_iORG.png"))
                r += 1

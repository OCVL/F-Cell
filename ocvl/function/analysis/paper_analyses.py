import itertools
import json
import sys
import warnings
from pathlib import Path
from tkinter import filedialog, Tk
import matplotlib as mpl
import mpl_axes_aligner
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore
from joblib._multiprocessing_helpers import mp
from matplotlib import pyplot as plt
import cv2
import os

from ocvl.function.preprocessing.improc import flat_field


# mpl.rcParams['axes.spines.right'] = False
# mpl.rcParams['axes.spines.top'] = False

def parallel_collate_direct(the_path):

    database = pd.DataFrame(columns=["seq_ind", "norm_method", "seg_radius","refine_to_ref","refine_to_vid",
                                     "stdization", "stdization_start", "sum_method",
                                     "prestim","poststim","amplitude"])

    with open(the_path, 'r') as json_path:
        dat_form = json.load(json_path)

        summed_data = None
        for datpath in the_path.parent.glob("*pop_summary*"):
            summed_data = pd.read_csv(datpath, index_col=0, header=1)

        row_ind = 0
        seq_ind = 0
        for index, row in summed_data.iterrows():
            if index != "Pooled" and row.loc["Amplitude"] is not None and not np.isnan(row["Amplitude"]):
                # Grab the values of the things that were permuted.
                database.loc[row_ind, "seq_ind"] = seq_ind
                database.loc[row_ind, "norm_method"] = dat_form["analysis"]["analysis_params"]["normalization"]["method"]
                database.loc[row_ind, "seg_radius"] = dat_form["analysis"]["analysis_params"]["segmentation"]["radius"]
                database.loc[row_ind, "refine_to_ref"] = dat_form["analysis"]["analysis_params"]["segmentation"]["refine_to_ref"]
                database.loc[row_ind, "refine_to_vid"] = dat_form["analysis"]["analysis_params"]["segmentation"]["refine_to_vid"]
                database.loc[row_ind, "stdization"] = dat_form["analysis"]["analysis_params"]["standardization"]["method"]
                database.loc[row_ind, "stdization_start"] = dat_form["analysis"]["analysis_params"]["standardization"]["start"]
                database.loc[row_ind, "sum_method"] = dat_form["analysis"]["analysis_params"]["summary"]["method"]
                database.loc[row_ind, "prestim"] = dat_form["analysis"]["analysis_params"]["summary"]["metrics"]["prestim"][0]
                database.loc[row_ind, "poststim"] = dat_form["analysis"]["analysis_params"]["summary"]["metrics"]["poststim"][1]
                database.loc[row_ind, "amplitude"] = row.loc["Amplitude"]

                row_ind+=1
            seq_ind+=1

    return database

def parallel_collate_var(the_path):

    database = pd.DataFrame(columns=["seq_ind", "norm_method", "seg_radius","refine_to_ref","refine_to_vid",
                                     "stdization", "stdization_start", "sum_method",
                                     "prestim","poststim","variance"])

    with open(the_path, 'r') as json_path:
        dat_form = json.load(json_path)

        summed_data = None
        for datpath in the_path.parent.glob("*pop_summary*"):
            summed_data = pd.read_csv(datpath, index_col=0, header=1)

        row_ind = 0


        database.loc[row_ind, "norm_method"] = dat_form["analysis"]["analysis_params"]["normalization"]["method"]
        database.loc[row_ind, "seg_radius"] = dat_form["analysis"]["analysis_params"]["segmentation"]["radius"]
        database.loc[row_ind, "refine_to_ref"] = dat_form["analysis"]["analysis_params"]["segmentation"]["refine_to_ref"]
        database.loc[row_ind, "refine_to_vid"] = dat_form["analysis"]["analysis_params"]["segmentation"]["refine_to_vid"]
        database.loc[row_ind, "stdization"] = dat_form["analysis"]["analysis_params"]["standardization"]["method"]
        database.loc[row_ind, "stdization_start"] = dat_form["analysis"]["analysis_params"]["standardization"]["start"]
        database.loc[row_ind, "sum_method"] = dat_form["analysis"]["analysis_params"]["summary"]["method"]
        database.loc[row_ind, "prestim"] = dat_form["analysis"]["analysis_params"]["summary"]["metrics"]["prestim"][0]
        database.loc[row_ind, "poststim"] = dat_form["analysis"]["analysis_params"]["summary"]["metrics"]["poststim"][1]
        database.loc[row_ind, "variance"] = summed_data.loc[:, "Amplitude"].var()




    return database


if __name__ == "__main__":
    root = Tk()
    root.lift()
    w = 1
    h = 1
    x = root.winfo_screenwidth() / 4
    y = root.winfo_screenheight() / 4
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.

    inny = 2

    if inny == 1:
        pName = None

        pName = filedialog.askdirectory(title="Select the folder containing RMS data.", initialdir=None, parent=root)

        if not pName:
            sys.exit(1)

        searchpath = Path(pName)

        rmsdata = []
        rms_cov = []

        for path in searchpath.glob("*.csv"):
            dataset = pd.read_csv(path, index_col=0, header=1)
            tmp = dataset.loc[:,"Amplitude"].to_numpy()
            print(path)
            rmsdata.append(tmp[:-1])
            rms_cov.append( np.nanstd(tmp[:-1])/np.nanmean(tmp[:-1]) )

        rmsdata=np.array(list(itertools.zip_longest(*rmsdata, fillvalue=np.nan))).T
        rms_cov = np.array(rms_cov)

        pName = filedialog.askdirectory(title="Select the folder containing STDDEV data.", initialdir=None, parent=root)

        if not pName:
            sys.exit(1)

        searchpath = Path(pName)

        stddata = []
        std_cov = []

        for path in searchpath.glob("*.csv"):
            dataset = pd.read_csv(path, index_col=0, header=1)

            tmp = dataset.loc[:, "Amplitude"].to_numpy()
            print(path)
            stddata.append(tmp[:-1])
            std_cov.append( np.nanstd(tmp[:-1])/np.nanmean(tmp[:-1]) )

        stddata=np.array(list(itertools.zip_longest(*stddata, fillvalue=np.nan))).T
        std_cov = np.array(std_cov)

        normmap = mpl.colors.Normalize(vmin=0, vmax=rmsdata.shape[0], clip=True)
        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("inferno"), norm=normmap)

        plt.figure("Amplitude comparison")
        for r in range(rmsdata.shape[0]):
            plt.scatter(rmsdata[r,:], stddata[r,:], s=35.0 ,color=mapper.to_rgba(r), linewidth=0)

        plt.ylabel("Cooper 2017 (std dev)")
        plt.ylim((0,10))
        plt.xlabel("Gaffney 2024 (rms)")
        plt.xlim((0,50))

        plt.figure("CoV comparison")
        for r in range(rmsdata.shape[0]):
            plt.scatter(rms_cov[r], std_cov[r], color=mapper.to_rgba(r))
        plt.plot([0, 0.5], [0, 0.5], c='k' )
        plt.xlim((0,0.5))
        plt.ylim((0,0.5))


        plt.show()
    elif inny == 2:
        pName = None

        pName = filedialog.askdirectory(title="Select the folder containing data, or cancel to stop adding it.", initialdir=None, parent=root)
        sig_data = dict()
        amp_data = dict()
        time = dict()
        while len(pName) != 0:

            searchpath = Path(pName)

            sig_data[searchpath.name] = []
            amp_data[searchpath.name] = []
            time[searchpath.name] = np.zeros((1,))

            for path in searchpath.rglob("*pop_sum_iORG*"):
                dataset = pd.read_csv(path, index_col=0)

                if len(dataset.columns) > len(time[searchpath.name]):
                    time[searchpath.name] = np.array(dataset.columns, dtype="float32")

                sig_data[searchpath.name].append(dataset.loc["Pooled", :].to_numpy())

            sig_data[searchpath.name] = np.array(list(itertools.zip_longest(*sig_data[searchpath.name], fillvalue=np.nan))).T

            for path in searchpath.rglob("*pop_summary*"):
                dataset = pd.read_csv(path, index_col=0, header=1)
                tmp = dataset.loc[:, "Amplitude"].to_numpy()

                amp_data[searchpath.name].append(np.nanstd(tmp[:-1])/np.nanmean(tmp[:-1]))

            amp_data[searchpath.name] = np.array(amp_data[searchpath.name])

            pName = filedialog.askdirectory(title="Select the folder containing data, or cancel to stop adding it.", initialdir=None, parent=root)


        normmap = mpl.colors.Normalize(vmin=0, vmax=len(sig_data), clip=True)
        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("inferno"), norm=normmap)


        r=0
        plt.figure("Signal versus")
        for key, siggies in sig_data.items():
            maxval = np.nanmax(siggies.flatten())
            for sig in siggies:
                # plt.plot(time[key], sig/maxval, color=mapper.to_rgba(r), label=key)
                plt.plot(time[key], sig, color=mapper.to_rgba(r), label=key)
            r += 1

        r=0
        plt.figure("Mean+-95predict")
        for key, siggies in sig_data.items():
            maxval = np.nanmax(siggies.flatten())

            #normsigs = siggies/maxval
            normsigs = siggies

            datmean = np.nanmean(normsigs, axis=0)
            datstd = np.nanstd(normsigs, axis=0)


            plt.plot(time[key], datmean, linewidth=3, label=key)
            plt.gca().fill_between(time[key], datmean - 1.96 * datstd, datmean + 1.96 * datstd,
                                   alpha=0.2, interpolate=True)
            # plt.show(block=False)
            # plt.waitforbuttonpress()
            r += 1


        plt.figure("CoV versus")
        sns.barplot(data=amp_data,capsize=.1, errorbar="sd")
        sns.swarmplot(data=amp_data,color="0", alpha=.35)




        plt.show()

        print("huh")

    elif inny == 3:

        pName = filedialog.askdirectory(title="Select the folder containing all data of interest.")
        if not pName:
            sys.exit(1)


        ind = 0

        potentials = list(Path(pName).rglob("*.json"))

        with mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:
            datalines = pool.map(parallel_collate_var, potentials)

        database = pd.concat(datalines)
        database.to_csv(Path(pName).joinpath("donedonedone_var.csv"))

    elif inny ==4:
        # Prompt user to select a folder
        Tk().withdraw()  # Hide the root window
        folder_path = filedialog.askdirectory(title='Select the folder containing .avi files')

        # Check if user canceled folder selection
        if not folder_path:
            raise Exception('No folder selected. Exiting script.')

        # Get list of all .avi files in the folder
        all_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]

        # Separate files into "Direct" and "Reflected" lists
        direct_files = [f for f in all_files if 'Direct' in f]
        reflected_files = [f for f in all_files if 'Reflected' in f]

        print(f'Found {len(direct_files)} Direct files and {len(reflected_files)} Reflected files.')

        # Process Direct files
        for i in range(0, len(direct_files)):  # Start from index 1 to match MATLAB's 2-based indexing
            direct_file_path = os.path.join(folder_path, direct_files[i])
            reflected_file_path = os.path.join(folder_path, reflected_files[i])
            print(f'Loading Direct file: {direct_files[i]}')

            # Load video files
            direct_cap = cv2.VideoCapture(direct_file_path)
            reflect_cap = cv2.VideoCapture(reflected_file_path)

            frame_count = int(min(direct_cap.get(cv2.CAP_PROP_FRAME_COUNT),
                                  reflect_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            width = int(direct_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(direct_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Prepare output file path
            phase_file_name = direct_files[i].replace('Direct', 'PhaseMag')
            phase_file_path = os.path.join(folder_path, phase_file_name)

            # Define video writer with MJPG encoding for grayscale AVI
            fourcc = cv2.VideoWriter_fourcc(*'Y800')
            out = cv2.VideoWriter(phase_file_path, fourcc, direct_cap.get(cv2.CAP_PROP_FPS), (width, height),
                                  isColor=False)

            frame_idx = 0
            while direct_cap.isOpened() and reflect_cap.isOpened():
                ret1, direct_frame = direct_cap.read()
                ret2, reflect_frame = reflect_cap.read()

                if not ret1 or not ret2:
                    break

                # Convert to grayscale and float
                direct_gray = cv2.cvtColor(direct_frame, cv2.COLOR_BGR2GRAY).astype(float)
                reflect_gray = cv2.cvtColor(reflect_frame, cv2.COLOR_BGR2GRAY).astype(float)



                phase_frm = (flat_field(direct_gray, sigma=10) - flat_field(reflect_gray, sigma=10))

                # Standardize.
                frame_norm = np.nanmean(phase_frm)
                frame_std = np.nanstd(phase_frm)
                phase_frm = (phase_frm - frame_norm) / frame_std
                phase_frm *= 35

                with warnings.catch_warnings():
                    warnings.filterwarnings(action="ignore", message="invalid value encountered in cast")
                    phase_mag = np.abs(phase_frm).astype(np.uint8)

                frame_idx += 1

                # Write frame to output video
                out.write(phase_mag)

            direct_cap.release()
            reflect_cap.release()
            out.release()

        plt.close()



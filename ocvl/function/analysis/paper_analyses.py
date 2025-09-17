import itertools
import json
import sys
from pathlib import Path
from tkinter import filedialog, Tk
import matplotlib as mpl
import mpl_axes_aligner
import numpy as np
import pandas as pd
from colorama import Fore
from joblib._multiprocessing_helpers import mp
from matplotlib import pyplot as plt

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


    inny = input("Metrics (1), Signals (2), or whole folder summary (purmutations; 3)?")


    if inny == "1":
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
    elif inny == "2":
        pName = None

        pName = filedialog.askdirectory(title="Select the folder containing RMS data.", initialdir=None, parent=root)

        if not pName:
            sys.exit(1)

        searchpath = Path(pName)

        rmsdata = []
        timestamps = np.zeros((1,))

        for path in searchpath.glob("*.csv"):
            dataset = pd.read_csv(path, index_col=0)

            if len(dataset.columns) > len(timestamps):
                timestamps = np.array(dataset.columns, dtype="float32")

            rmsdata.append(dataset.loc["Pooled", :].to_numpy())

        rmsdata = np.array(list(itertools.zip_longest(*rmsdata, fillvalue=np.nan))).T

        axcolor = "tab:blue"
        figgy, ax1 = plt.subplots()
        for i in range(rmsdata.shape[0]):
            ax1.plot(timestamps, rmsdata[i, :], color=axcolor)
        ax1.set_ylabel("Gaffney et. al (2024)", color=axcolor)
        ax1.tick_params(axis="y", labelcolor=axcolor)

        pName = filedialog.askdirectory(title="Select the folder containing STDDEV data.", initialdir=None, parent=root)

        if not pName:
            sys.exit(1)

        searchpath = Path(pName)

        stddata = []
        timestamps = np.zeros((1,))

        for path in searchpath.glob("*.csv"):
            dataset = pd.read_csv(path, index_col=0)

            if len(dataset.columns) > len(timestamps):
                timestamps = np.array(dataset.columns, dtype="float32")

            stddata.append(dataset.loc["Pooled", :].to_numpy())

        stddata = np.array(list(itertools.zip_longest(*stddata, fillvalue=np.nan))).T

        axcolor = "tab:orange"
        ax2 = ax1.twinx()
        for i in range(stddata.shape[0]):
            ax2.plot(timestamps, stddata[i, :], color=axcolor)

        ax2.set_ylabel("Cooper et. al (2017)", color=axcolor)
        ax2.tick_params(axis="y", labelcolor=axcolor)

        mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.1)
        plt.show()

        print("huh")

    if inny == "3":

        pName = filedialog.askdirectory(title="Select the folder containing all data of interest.")
        if not pName:
            sys.exit(1)


        ind = 0

        potentials = list(Path(pName).rglob("*.json"))

        with mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:
            datalines = pool.map(parallel_collate_var, potentials)

        database = pd.concat(datalines)
        database.to_csv(Path(pName).joinpath("donedonedone_var.csv"))


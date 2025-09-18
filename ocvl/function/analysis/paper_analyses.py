import itertools
import json
import sys
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
        for key, siggies in sig_data.items():
            maxval = np.nanmax(siggies.flatten())
            plt.figure("Signal versus")
            for sig in siggies:
                plt.plot(time[key], sig/maxval, color=mapper.to_rgba(r), label=key)
            r += 1


        plt.figure("CoV versus")
        sns.barplot(data=amp_data,capsize=.1, ci="sd")
        sns.swarmplot(data=amp_data,color="0", alpha=.35)




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


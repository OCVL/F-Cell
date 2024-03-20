from pathlib import Path
from tkinter import filedialog, Tk

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":

    metric_header = "Amplitude"

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

    resultdir = Path(pName)

    allFiles = dict()

    totFiles = 0
    # Parse out the locations and filenames, store them in a hash table.
    searchpath = Path(pName)
    for path in searchpath.rglob("*.csv"):
        if "pop_iORG_stats" in path.name:
            splitfName = path.name.split("_")

            firstcoord = str(abs(float(splitfName[0].split(",")[0][1:])))

            if (path.parent.parent == searchpath or path.parent == searchpath):
                if path.parent not in allFiles:
                    allFiles[path.parent] = []
                    allFiles[path.parent].append(path)
                else:
                    allFiles[path.parent].append(path)

            totFiles += 1

    all_data = pd.DataFrame()
    all_pooled_data = pd.DataFrame()
    all_SE = pd.DataFrame()

    # plt.figure(42)

    # NEVER grow a dataframe rowwise
    # https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-and-then-filling-it
    for subid in allFiles:

        subframe = pd.DataFrame()
        pooled_data = np.full((1, len(allFiles[subid])), np.nan)
        loc = []
        f = 0
        for file in allFiles[subid]:

            stat_table = pd.read_csv(file, delimiter=",", header=0,encoding="utf-8-sig", index_col=0)
            stat_table = stat_table.dropna(how="all")  # Drop rows with all nans- no point in taking up space.

            splitfName = file.name.split("_")

            firstcoord = str(abs(float(splitfName[0].split(",")[0][1:])))

            loc.append(firstcoord)
            pooled_data[0, f] = (stat_table[metric_header].iloc[-1])

            allcoi = stat_table[metric_header][0:-1]

            indices = pd.MultiIndex.from_product([[file.parent.name], np.arange(0, len(allcoi))],
                                                 names=["ID", "Acquisition Number"])
            allcoi.index = indices
            allcoi = allcoi.rename(firstcoord)
            allcoi = allcoi.to_frame()

            subframe=pd.concat([subframe, allcoi], axis=1)

            f+=1

        themean = subframe.mean()
        stddev = subframe.std()
        # plt.plot(themean, stddev, "*")

        all_data=pd.concat([all_data, subframe])
        pooled_data = pd.DataFrame(pooled_data, columns=loc, index=[subid.name])
        all_pooled_data = pd.concat([all_pooled_data, pooled_data])

        pooled_data = pd.DataFrame(stddev, columns=[subid.name+"_SE"]).T
        all_SE = pd.concat([all_SE, pooled_data])
        print("wut")

    all_data = all_data.sort_index(axis=1)
    all_pooled_data = all_pooled_data.sort_index(axis=1)
    all_SE = all_SE.sort_index(axis=1)

    bins = np.full(len(all_pooled_data.columns), np.nan)
    mean_data = np.full(len(all_pooled_data.columns), np.nan)
    stddev_data = np.full(len(all_pooled_data.columns), np.nan)
    conf_interval = np.full(len(all_pooled_data.columns), np.nan)


    i=0
    for loc in all_pooled_data.columns:

        bins[i] = float(loc)
        mean_data[i] = all_pooled_data[loc].mean()
        stddev_data[i] = all_pooled_data[loc].std()
        num_data = all_pooled_data[loc].count()

        conf_interval[i] = 2 * stddev_data[i] / np.sqrt(num_data)
        i+=1


    plt.figure(1)
    plt.fill_between(bins, mean_data + conf_interval, color=[0, 0, 0, 0.3])
    plt.fill_between(bins, mean_data - conf_interval, color=[1, 1, 1, 1])
    plt.plot(bins, mean_data)
    plt.title("Population mean and confidence interval")
    plt.xlim([0, 9])
    plt.ylim([0, 50])
    plt.savefig(resultdir.joinpath(resultdir.name + "_pop_iORG_mean_n_conf.svg"))




    for sub_id in all_pooled_data.index:
        xval = all_pooled_data.loc[sub_id].index.to_numpy().astype(float)
        yval = all_pooled_data.loc[sub_id].values
        isfini = np.isfinite(yval)
        xval= xval[isfini]
        yval = yval[isfini]
        se = all_SE.loc[sub_id+"_SE"].values
        se= se[isfini]
        plt.figure(2)
        plt.errorbar(xval, yval, yerr=se, ms=10, marker=".")
        plt.figure(3)
        plt.errorbar(xval, yval-yval[-2], yerr=se, ms=10, marker=".")

    plt.figure(2)
    plt.legend(all_pooled_data.index, loc="lower left")
    plt.xlim([0, 9])
    plt.ylim([0, 50])
    plt.savefig(resultdir.joinpath(resultdir.name + "_indiv_sub_pop_iORG_mean_n_conf.svg"))

    plt.figure(3)
    plt.legend(all_pooled_data.index, loc="lower left")
    plt.xlim([0, 9])
    plt.ylim([-25, 25])
    plt.savefig(resultdir.joinpath(resultdir.name + "_indiv_sub_pop_iORG_mean_n_conf_firstsub.svg"))

    plt.show(block=False)
    print("Concatenate this, bitch")

    all_data.to_csv(resultdir.joinpath(resultdir.name+"_collated_"+metric_header+"_data.csv"))
    all_pooled_data = pd.concat([all_pooled_data, all_SE])
    all_pooled_data.to_csv(resultdir.joinpath(resultdir.name+"_collated_POOLED_"+metric_header+"_data.csv"))
    plt.waitforbuttonpress()
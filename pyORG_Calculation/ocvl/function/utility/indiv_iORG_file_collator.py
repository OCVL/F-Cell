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


    # For looking at the Xth percentile on a per-subject basis compared to our normative baseline,
    # as well as determining where data should be grouped for IRDs
    normie_path = searchpath.joinpath("Normal_collated_Amplitude_meandata.csv")

    if normie_path.is_file():

        perc_cutoff = 0.05
        normie_table = pd.read_csv(normie_path, index_col=0).sort_index()
        normie_dat = normie_table.to_numpy()
        above_cut = np.argmax(normie_dat > perc_cutoff, axis=1, keepdims=True)

        normie_locs = normie_table.index.to_numpy().astype(float)
        normie_vals = normie_table.columns.to_numpy()[above_cut.flatten()].astype(float)

    for path in searchpath.rglob("*.csv"):
        if "log_amplitude_cumhist" in path.name:
            splitfName = path.name.split("_")

            firstcoord = str(abs(float(splitfName[0].split(",")[0][1:])))

            if (path.parent.parent == searchpath or path.parent == searchpath):
                if firstcoord not in allFiles:
                    allFiles[firstcoord] = []
                    allFiles[firstcoord].append(path)
                else:
                    allFiles[firstcoord].append(path)

            totFiles += 1


    all_data = pd.DataFrame()
    all_means = pd.DataFrame()
    subID = dict()
    all_cumhist = dict()
    # NEVER grow a dataframe rowwise, or baby Jesus will cry
    # https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-and-then-filling-it
    for locid in allFiles:

        if normie_path.is_file():
            normie_loc_ind = np.argmin(abs(abs(float(locid)) - normie_locs))
            thisloc = normie_locs[normie_loc_ind]
        else:
            thisloc = float(locid)

        if str(thisloc) not in subID:
            subID[str(thisloc)] = []
            all_cumhist[str(thisloc)] = np.full((len(allFiles[locid]), 550), np.nan)
        else:
            all_cumhist[str(thisloc)] = np.append(all_cumhist[str(thisloc)], np.full((len(allFiles[locid]), 550), np.nan), axis=0)
        print(thisloc)
        # The above may change depending on how many bins we go with.
        f = 0


        plt.figure(str(thisloc*10))

        for locfile in allFiles[locid]:

            stat_table = pd.read_csv(locfile, delimiter=",", header=0, encoding="utf-8-sig")
            bins = stat_table.loc[1, :]

            subID[str(thisloc)].append(locfile.parent.name)

            cumhist = np.nancumsum(stat_table.loc[0, :])
            cumhist /= np.amax(cumhist.flatten()) # Normalize to 1 as its a probability density, not a probability mass, function

            all_cumhist[str(thisloc)][f, :] = cumhist
            f+=1

            plt.plot(bins, cumhist)

        # all_cumhist=np.nancumsum(all_cumhist, axis=1)
        # all_cumhist /= np.amax(all_cumhist.flatten())
        plt.show(block=False)
        plt.legend(subID[str(thisloc)])
        plt.savefig(resultdir.joinpath(resultdir.name + "_" + str(thisloc) + "_all_cumulative_histograms.svg"))

        mean_cumhist = np.mean(all_cumhist[str(thisloc)], axis=0)
        stddev_cumhist = np.std(all_cumhist[str(thisloc)], axis=0)
        conf_interval = 2 * stddev_cumhist / np.sqrt(len(allFiles[locid]))

        plt.figure(str(thisloc*10+100))
        plt.fill_between(bins, mean_cumhist + conf_interval, color=[0, 0, 0, 0.3])
        plt.fill_between(bins, mean_cumhist - conf_interval, color=[1, 1, 1, 1])
        plt.plot(bins, mean_cumhist)
        plt.savefig(resultdir.joinpath(resultdir.name+ "_"+str(thisloc)+"_mean_n_conf_cumulative_histogram.svg"))

        plt.show(block=False)
        #plt.waitforbuttonpress()

        indices = pd.MultiIndex.from_product([[thisloc], subID[str(thisloc)]],
                                             names=["Location", "ID"])
        subframe = pd.DataFrame(all_cumhist[str(thisloc)], index=indices)

        all_data=pd.concat([all_data, subframe])
        all_means=pd.concat([all_means, pd.DataFrame(mean_cumhist[np.newaxis, :], index=[thisloc], columns=bins)])
        print("wut")


    print("Concatenate this, bitch")


    if normie_path.is_file():
        perc_data = pd.DataFrame()

        allSubFiles = dict()
        totSubFiles = 0
        for path in searchpath.rglob("*.csv"):
            if "log_amplitude_cumhist" in path.name:
                splitfName = path.name.split("_")

                if (path.parent.parent == searchpath or path.parent == searchpath):
                    if path.parent not in allSubFiles:
                        allSubFiles[path.parent] = []
                        allSubFiles[path.parent].append(path)
                    else:
                        allSubFiles[path.parent].append(path)

                totSubFiles += 1

        plt.figure(42)
        subID = []
        for subid in allSubFiles:

            fiftperc = []
            loc = []
            subID.append(subid.name)
            print(subid.name)
            f = 0
            for file in allSubFiles[subid]:
                splitfName = file.name.split("_")

                firstcoord = abs(float(splitfName[0].split(",")[0][1:]))

                normie_loc_ind = np.argmin(abs(firstcoord-normie_locs))

                stat_table = pd.read_csv(file, delimiter=",", header=0, encoding="utf-8-sig")
                bins = stat_table.loc[1, :].to_numpy()

                cumhist = np.nancumsum(stat_table.loc[0, :])
                cumhist /= np.amax(cumhist.flatten())

                # Find the percentage of cones in this subject that are within the normative percentage we set above.
                perc_healthy = 100.0*(1-cumhist[np.flatnonzero(bins >= normie_vals[normie_loc_ind])[0]])
                print(normie_locs[normie_loc_ind])
                print(cumhist[np.flatnonzero(bins >= normie_vals[normie_loc_ind])[0]])
                print(perc_healthy)

                fiftperc.append(perc_healthy)
                loc.append(firstcoord)


                f+=1

            indices = pd.MultiIndex.from_product([loc, [subid.name]],
                                                 names=["Location", "ID"])
            subframe = pd.DataFrame(fiftperc, index=indices)

            perc_data = pd.concat([perc_data, subframe])


            plt.plot(loc, fiftperc,".-")
            plt.show(block=False)
            plt.legend(subID)
            plt.savefig(resultdir.joinpath(resultdir.name + "_percent_overnormie.svg"))

        perc_data.to_csv(resultdir.joinpath(resultdir.name + "_percent_overnormie_data.csv"))


all_data.to_csv(resultdir.joinpath(resultdir.name+"_collated_"+metric_header+"_data.csv"))
all_means.to_csv(resultdir.joinpath(resultdir.name + "_collated_" + metric_header + "_meandata.csv"))
plt.waitforbuttonpress()
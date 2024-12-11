import os
import multiprocessing as mp
from pathlib import Path, PurePath
from tkinter import Tk, filedialog, ttk, HORIZONTAL, simpledialog

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles, \
    refine_coord, refine_coord_to_stack, exclude_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG, iORG_signal_metrics
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.dataset import PipeStages, parse_file_metadata, initialize_and_load_dataset
from ocvl.function.utility.json_format_constants import PipelineParams, MetaTags, DataType, DataTags, AcquisiTags, \
    SegmentParams, ExclusionParams, NormParams, STDParams, ORGTags

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

    # We should be 3 levels up from here. Kinda jank, will need to change eventually
    config_path = Path(os.path.dirname(__file__)).parent.parent.joinpath("config_files")

    json_fName = filedialog.askopenfilename(title="Select the configuration json file.", initialdir=config_path, parent=root)
    if not json_fName:
        quit()

    # Grab all the folders/data here.
    dat_form, allData = parse_file_metadata(json_fName, pName, "pipelined")

    stimtrain_fName = filedialog.askopenfilename(title="Select the stimulus train file.", initialdir=pName, parent=root)

    if not stimtrain_fName:
        quit()

    x = root.winfo_screenwidth() / 2 - 128
    y = root.winfo_screenheight() / 2 - 128
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.
    root.update()

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

    #with mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:

    piped_dat_format = dat_form.get("pipelined")
    processed_dat_format = dat_form.get("processed")
    pipeline_params = processed_dat_format.get("pipeline_params")
    analysis_params = piped_dat_format.get("analysis_params")
    modes_of_interest = analysis_params.get(PipelineParams.MODALITIES)

    metadata_params = None
    if piped_dat_format.get(MetaTags.METATAG) is not None:
        metadata_params = piped_dat_format.get(MetaTags.METATAG)
        metadata_form = metadata_params.get(DataType.METADATA)

    # If we've selected modalities of interest, only process those; otherwise, process them all.
    if modes_of_interest is None:
        modes_of_interest = allData[DataTags.MODALITY].unique().tolist()
        print("NO MODALITIES SELECTED! Processing them all....")

    grouping = pipeline_params.get(PipelineParams.GROUP_BY)
    if grouping is not None:
        for row in allData.itertuples():
            #print(grouping.format_map(row._asdict()))
            allData.loc[row.Index, PipelineParams.GROUP_BY] = grouping.format_map(row._asdict())

        groups = allData[PipelineParams.GROUP_BY].unique().tolist()
    else:
        groups = [""]  # If we don't have any groups, then just make the list an empty string.

    reference_coord_data = None
    maxnum_cells = None
    skipnum = 0

    # Snag all of our parameter dictionaries that we'll use here.
    # Default to an empty dictionary so that we can query against it with fewer if statements.
    seg_params = analysis_params.get(SegmentParams.NAME, dict())
    norm_params = analysis_params.get(NormParams.NAME, dict())
    excl_params = analysis_params.get(ExclusionParams.NAME, dict())
    std_params = analysis_params.get(STDParams.NAME, dict())

    # First break things down by group, defined by the user in the config file.
    # We like to use (LocX,LocY), but this is by no means the only way.
    for group in groups:
        if group != "":
            group_datasets = allData.loc[allData[PipelineParams.GROUP_BY] == group]
        else:
            group_datasets = allData

        # While we're going to process by group, respect the folder structure used by the user here, and only group
        # and analyze things from the same folder
        folder_groups = pd.unique(group_datasets[AcquisiTags.BASE_PATH]).tolist()

        # Respect the users' folder structure. If things are in different folders, analyze them separately.
        for folder in folder_groups:

            output_folder = analysis_params.get(PipelineParams.OUTPUT_FOLDER)
            if output_folder is None:
                output_folder = PurePath("Results")
            else:
                output_folder = PurePath(output_folder)

            result_folder = folder.joinpath(output_folder)

            result_folder.mkdir(exist_ok=True)

            data_in_folder = group_datasets.loc[group_datasets[AcquisiTags.BASE_PATH] == folder]

            # Load each modality
            for mode in modes_of_interest:
                mode_data = data_in_folder.loc[data_in_folder[DataTags.MODALITY] == mode]

                data_vidnums = mode_data[DataTags.VIDEO_ID].unique().tolist()

                reference_images = (mode_data[DataType.FORMAT] == DataType.IMAGE)
                query_locations = (mode_data[DataType.FORMAT] == DataType.QUERYLOC)
                numdata = len(mode_data)

                # Make data storage structures for each of our query location lists- one is for results,
                # The other for checking which query points went into our analysis.
                pop_iORG_result_datframes = [pd.DataFrame(index=data_vidnums, columns=[ORGTags.AMPLITUDE, ORGTags.IMPLICT_TIME, ORGTags.RECOVERY_PERCENT]) for i in range(query_locations.sum())]

                query_status = [pd.DataFrame(columns=data_vidnums) for i in range(query_locations.sum())]

                # Load each dataset (delineated by different video numbers)
                for vidnum in data_vidnums:

                    data = mode_data.loc[(mode_data[DataTags.VIDEO_ID] == vidnum) | reference_images | query_locations]

                    r = 0
                    pb["maximum"] = numdata
                    pop_iORG = []
                    pop_iORG_implicit = np.empty((numdata-skipnum+1))
                    pop_iORG_implicit[:] = np.nan
                    pop_iORG_recover = np.empty((numdata - skipnum + 1))
                    pop_iORG_recover[:] = np.nan
                    pop_iORG_amp = np.empty((numdata - skipnum + 1))
                    pop_iORG_amp[:] = np.nan
                    pop_iORG_num = []
                    framestamps = []
                    max_frmstamp = 0
                    plt.figure(0)
                    plt.clf()

                    first = True
                    mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", numdata))

                    pb["value"] = r
                    pb_label["text"] = "Processing " + data[AcquisiTags.DATA_PATH].name + "..."
                    pb.update()
                    pb_label.update()

                    # for later: allData.loc[ind, AcquisiTags.DATASET]
                    # Actually load the dataset, and all its metadata.
                    dataset = initialize_and_load_dataset(data, metadata_params)

                    # Perform analyses on each query location set for each dataset.
                    for i in range(len(dataset.query_loc)):

                        '''
                        *** This section is where we actually do dataset summary and analysis. (population iORG) ***
                        '''

                        query_status[i] = query_status[i].reindex(pd.MultiIndex.from_tuples(list(map(tuple, dataset.query_loc[i]))), fill_value="Included")

                        if seg_params.get(SegmentParams.REFINE_TO_REF, True):
                            reference_coord_data = refine_coord(dataset.avg_image_data, dataset.query_loc[i])
                            coorddist = pdist(reference_coord_data, "euclidean")
                            coorddist = squareform(coorddist)
                            coorddist[coorddist == 0] = np.amax(coorddist.flatten())
                            mindist = np.amin( coorddist, axis=-1)

                        # If not defined, then we default to "auto" which determines it from the spacing of the query points
                        segmentation_radius = seg_params.get(SegmentParams.RADIUS, "auto")
                        if segmentation_radius == "auto":
                            segmentation_radius = np.round(np.nanmean(mindist) / 4) if np.round(np.nanmean(mindist) / 4) >= 1 else 1

                            segmentation_radius = int(segmentation_radius)
                            print("Detected segmentation radius: " + str(segmentation_radius))

                        dataset.query_loc[i] = reference_coord_data

                        if seg_params.get(SegmentParams.REFINE_TO_VID, True):
                            dataset.query_loc[i], valid_signals, excl_reason  = refine_coord_to_stack(dataset.video_data, dataset.avg_image_data,
                                                                                                     reference_coord_data)
                            # Update our audit path.
                            query_status[i].loc[~valid_signals, vidnum] = excl_reason[~valid_signals]


                        # Normalize the video to reduce framewide intensity changes
                        method = norm_params.get(NormParams.NORM_METHOD, "score") # Default: Standardizes the video to a unit mean and stddev
                        rescale = norm_params.get(NormParams.NORM_RESCALE, True) # Default: Rescales the data back into AU to make results easier to interpret
                        res_mean = norm_params.get(NormParams.NORM_MEAN, 70) # Default: Rescales to a mean of 70 - these values are based on "ideal" datasets
                        res_stddev = norm_params.get(NormParams.NORM_STD, 35)  # Default: Rescales to a std dev of 35

                        dataset.video_data = norm_video(dataset.video_data, norm_method=method, rescaled=rescale,
                                                        rescale_mean=res_mean, rescale_std=res_stddev)

                        # Extract the signals
                        seg_shape = seg_params.get(SegmentParams.SHAPE, "disk")
                        seg_summary = seg_params.get(SegmentParams.SUMMARY, "mean")
                        iORG_signals, excl_reason = extract_profiles(dataset.video_data, dataset.query_loc[i], seg_radius=segmentation_radius,
                                                                     seg_mask=seg_shape, summary=seg_summary)

                            # Update our audit path.
                        to_update = np.logical_xor(np.all(np.isfinite(iORG_signals), axis=1), valid_signals)
                        valid_signals = np.all(np.isfinite(iORG_signals), axis=1) & valid_signals
                        query_status[i].loc[to_update, vidnum] = excl_reason[to_update]

                        # Exclude signals that don't pass our criterion
                        type = excl_params.get(ExclusionParams.TYPE)
                        units = excl_params.get(ExclusionParams.UNITS)
                        start = excl_params.get(ExclusionParams.START)
                        stop = excl_params.get(ExclusionParams.END)
                        cutoff_fraction = excl_params.get(ExclusionParams.FRACTION)

                        if units == "time":
                            start_ind = int(start * dataset.framerate)
                            stop_ind = int(stop * dataset.framerate)
                        else: #if units == "frames":
                            start_ind = int(start)
                            stop_ind = int(stop)

                        if type == "relative":
                            start_ind = dataset.stimtrain_frame_stamps[0] + start_ind
                            stop_ind = dataset.stimtrain_frame_stamps[1] + stop_ind
                        else: #if type == "absolute":
                            start_ind = start_ind
                            stop_ind = stop_ind

                        excl_profiles, valid_profiles, excl_reason = exclude_profiles(iORG_signals, dataset.framestamps,
                                                                         critical_region=np.arange(start_ind, stop_ind),
                                                                         critical_fraction=cutoff_fraction)
                        to_update = np.logical_xor(valid_profiles, valid_signals)
                        query_status[i].loc[to_update, vidnum] = excl_reason[to_update]

                        if np.sum(~valid_profiles) == len(dataset.query_loc[i]):
                            pop_iORG_amp[r] = np.NaN
                            pop_iORG_implicit[r] = np.NaN
                            pop_iORG_recover[r] = np.NaN
                            print(file.name + " was dropped due to all cells being excluded.")

                        prestim_ind = np.flatnonzero(np.logical_and(dataset.framestamps < dataset.stimtrain_frame_stamps[0],
                                                     dataset.framestamps >= (dataset.stimtrain_frame_stamps[0] - int(1 * dataset.framerate))))
                        poststim_ind = np.flatnonzero(np.logical_and(dataset.framestamps >= dataset.stimtrain_frame_stamps[1],
                                                      dataset.framestamps < (dataset.stimtrain_frame_stamps[1] + int(1 * dataset.framerate))))

                        stdize_profiles = standardize_profiles(iORG_signals, dataset.framestamps,
                                                               dataset.stimtrain_frame_stamps[0], method="mean_sub", std_indices=prestim_ind)

                        tmp_iorg, tmp_incl = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="rms",
                                                               window_size=1)

                        plt.figure(9)
                        plt.plot(dataset.framestamps, tmp_incl)
                        plt.show(block=False)

                        # This is just to make them all at the same baseline.
                        tmp_iorg = standardize_profiles(tmp_iorg[None, :], dataset.framestamps,
                                                        dataset.stimtrain_frame_stamps[0], method="mean_sub", std_indices=prestim_ind)

                        tmp_iorg = np.squeeze(tmp_iorg)

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
                            pop_iORG_implicit[r] = pop_iORG_implicit[r] / dataset.framerate

                            print("Signal metrics based iORG Amplitude: " + str(pop_iORG_amp[r]) +
                                  " Implicit time (s): " + str(pop_iORG_implicit[r]) +
                                  " Recovery fraction: " + str(pop_iORG_recover[r]))

                            plt.figure(0)
                            # plt.subplot(2,5,r - skipnum+1)

                            plt.xlabel("Time (seconds)")
                            plt.ylabel("Response")
                            plt.plot(dataset.framestamps/dataset.framerate, pop_iORG[r - skipnum], color=mapper.to_rgba(r - skipnum, norm=False),
                                     label=file.name)

                            plt.show(block=False)
                            #plt.xlim([0, 4])
                            # plt.ylim([-5, 40])
                            #plt.savefig(output_folder.joinpath(file.name[0:-4] + "_pop_iORG.png"))
                            r += 1

                        if dataset.framestamps[-1] > max_frmstamp:
                            max_frmstamp = dataset.framestamps[-1]

                dt = datetime.now()
                now_timestamp = dt.strftime("%Y_%m_%d_%H_%M_%S")

                # plt.vlines(dataset.stimtrain_frame_stamps[0] / dataset.framerate, -1, 10, color="red")
                #plt.xlim([0,  4])
                # plt.ylim([-5, 60]) #was 60
                #plt.legend()

                plt.savefig( output_folder.joinpath(this_dirname + "_pop_iORG_" + now_timestamp + ".svg"))
                plt.savefig( output_folder.joinpath(this_dirname + "_pop_iORG_" + now_timestamp + ".png"))

                # plt.figure(14)
                # plt.plot(np.nanmean(np.log(pop_iORG_amp), axis=-1),
                #          np.nanstd(np.log(pop_iORG_amp), axis=-1),".")
                # plt.title("logAMP mean vs logAMP std dev")
                # plt.show(block=False)
                # plt.savefig(output_folder.joinpath(this_dirname + "_pop_iORG_logamp_vs_stddev.svg"))
                #
                # plt.figure(15)
                # plt.plot(np.nanmean(pop_iORG_amp, axis=-1),
                #          np.nanstd(pop_iORG_amp, axis=-1),".")
                # plt.title("AMP vs std dev")
                # plt.show(block=False)
                # plt.savefig(output_folder.joinpath(this_dirname + "_pop_iORG_amp_vs_stddev.svg"))
                print("Pop mean iORG amplitude: " + str(np.nanmean(pop_iORG_amp, axis=-1)) +
                      "Pop stddev iORG amplitude: " + str(np.nanmean(pop_iORG_amp, axis=-1)) )


                # pop_amp_dFrame = pd.DataFrame(np.concatenate((np.array(pop_iORG_amp, ndmin=2).transpose(),
                #                                               np.array(pop_iORG_implicit, ndmin=2).transpose(),
                #                                               np.array(pop_iORG_recover, ndmin=2).transpose()), axis=1),
                #                               columns=["Amplitude", "Implicit time", "Recovery %"])
                # pop_amp_dFrame.to_csv(output_folder.joinpath(this_dirname + "_pop_iORG_stats_" + now_timestamp + ".csv"))

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

                pop_data_dFrame = pd.DataFrame(np.concatenate((np.array(all_iORG, ndmin=2).transpose(),
                                                              np.array(pooled_iORG, ndmin=2).transpose()), axis=1))
                pop_data_dFrame.to_csv(output_folder.joinpath(this_dirname + "_pop_iORG_signals_" + now_timestamp + ".csv"))

                plt.figure(9)
                plt.plot(all_frmstamps, np.nansum(all_incl, axis=0))
                plt.show(block=False)

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
                    _, pop_iORG_amp[r], pop_iORG_implicit[r], _, pop_iORG_recover[r] = iORG_signal_metrics(pooled_iORG[None, :],
                                                                                                        dataset.framestamps,
                                                                                                        filter_type="none", display=False,
                                                                                                        prestim_idx=prestim_ind,
                                                                                                        poststim_idx=poststim_ind)
                    pop_iORG_implicit[r] /= dataset.framerate

                print("Pooled iORG Avg Amplitude: " + str(pop_iORG_amp[r]) + " Implicit time (s): " + str(pop_iORG_implicit[r]) +
                      " Recovery fraction: " + str(pop_iORG_recover[r]))

                pop_amp_dFrame = pd.DataFrame(np.concatenate((np.array(pop_iORG_amp, ndmin=2).transpose(),
                                                              np.array(pop_iORG_implicit, ndmin=2).transpose(),
                                                              np.array(pop_iORG_recover, ndmin=2).transpose()), axis=1),
                                              columns=["Amplitude", "Implicit time", "Recovery %"])
                pop_amp_dFrame = pop_amp_dFrame.dropna(how="all") # Drop rows with all nans- no point in taking up space.
                pop_amp_dFrame.to_csv(output_folder.joinpath(this_dirname + "_pop_iORG_stats_" + now_timestamp + ".csv"))

                plt.figure(10)

                plt.plot(all_frmstamps / dataset.framerate, pooled_iORG)
                plt.vlines(dataset.stimtrain_frame_stamps[0] / dataset.framerate, -1, 10, color="red")
                plt.xlim([0, 6])
                # plt.ylim([-5, 60]) #was 1, 60
                plt.xlabel("Time (seconds)")
                plt.ylabel("Response")
                plt.show(block=False)
                plt.savefig(output_folder.joinpath(this_dirname + "_pooled_pop_iORG_" + now_timestamp + ".png"))
                plt.savefig(output_folder.joinpath(this_dirname + "_pooled_pop_iORG_" + now_timestamp + ".svg"))
                print("Done!")
                plt.waitforbuttonpress()


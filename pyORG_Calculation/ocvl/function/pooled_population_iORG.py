import os
import multiprocessing as mp
import warnings
from itertools import repeat
from pathlib import Path, PurePath
from tkinter import Tk, filedialog, ttk, HORIZONTAL, simpledialog

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform

from ocvl.function.analysis.iORG_signal_extraction import extract_profiles, norm_profiles, standardize_profiles, \
    refine_coord, refine_coord_to_stack, exclude_profiles, extract_n_refine_iorg_signals
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG, iORG_signal_metrics
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.dataset import PipeStages, parse_file_metadata, initialize_and_load_dataset
from ocvl.function.utility.json_format_constants import PipelineParams, MetaTags, DataFormatType, DataTags, AcquisiTags, \
    SegmentParams, ExclusionParams, NormParams, STDParams, ORGTags, SummaryParams, ControlParams, DisplayParams

from datetime import datetime, date, time, timezone

def extract_control_iORG_summaries(params):
    control_vidnum, control_dataset, control_query_status, analysis_params, query_locs, stimtrain_frame_stamps = params

    control_dataset.query_loc = [None] * len(query_locs)
    control_dataset.iORG_signals = [None] * len(query_locs)
    control_dataset.summarized_iORGs = [None] * len(query_locs)

    for q in range(len(query_locs)):
        # Use the control data, but the query locations and stimulus info from the stimulus data.
        (control_dataset.iORG_signals[q],
         control_dataset.summarized_iORGs[q],
         control_query_status[q].loc[:, control_vidnum],
         control_dataset.query_loc[q]) = extract_n_refine_iorg_signals(control_dataset, analysis_params,
                                                                       query_loc=query_locs[q],
                                                                       stimtrain_frame_stamps=stimtrain_frame_stamps)
    return control_vidnum, control_dataset, control_query_status


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

    x = root.winfo_screenwidth() / 2 - 128
    y = root.winfo_screenheight() / 2 - 128
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.
    root.update()

    pb = ttk.Progressbar(root, orient=HORIZONTAL, length=1024)
    pb.grid(column=0, row=0, columnspan=3, padx=3, pady=5)
    pb_label = ttk.Label(root, text="Initializing setup...")
    pb_label.grid(column=0, row=1, columnspan=3)
    pb.start()
    # Resize our root to show our progress bar.
    w = 1024
    h = 64
    x = root.winfo_screenwidth() / 2 - 256
    y = root.winfo_screenheight() / 2 - 64
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.update()



    piped_dat_format = dat_form.get("pipelined")
    processed_dat_format = dat_form.get("processed")
    pipeline_params = processed_dat_format.get("pipeline_params")
    analysis_params = piped_dat_format.get("analysis_params")
    display_params = piped_dat_format.get("display_params")
    modes_of_interest = analysis_params.get(PipelineParams.MODALITIES)

    metadata_params = None
    if piped_dat_format.get(MetaTags.METATAG) is not None:
        metadata_params = piped_dat_format.get(MetaTags.METATAG)
        metadata_form = metadata_params.get(DataFormatType.METADATA)

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

    # Snag all of our parameter dictionaries that we'll use here.
    # Default to an empty dictionary so that we can query against it with fewer if statements.
    control_params = analysis_params.get(ControlParams.NAME, dict())
    control_loc = control_params.get(ControlParams.LOCATION, "implicit")
    control_folder = control_params.get(ControlParams.FOLDER_NAME, "control")

    sum_params = analysis_params.get(SummaryParams.NAME, dict())
    sum_method = sum_params.get(SummaryParams.METHOD, "rms")
    sum_control = sum_params.get(SummaryParams.CONTROL, "none")

    output_folder = analysis_params.get(PipelineParams.OUTPUT_FOLDER)
    if output_folder is None:
        output_folder = PurePath("Results")
    else:
        output_folder = PurePath(output_folder)

    # First break things down by group, defined by the user in the config file.
    # We like to use (LocX,LocY), but this is by no means the only way.
    for group in groups:
        if group != "":
            group_datasets = allData.loc[allData[PipelineParams.GROUP_BY] == group]
        else:
            group_datasets = allData

        group_datasets[AcquisiTags.STIM_PRESENT] = False
        reference_images = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.IMAGE)
        query_locations = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.QUERYLOC)
        only_vids = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.VIDEO)

        # While we're going to process by group, respect the folder structure used by the user here, and only group
        # and analyze things from the same folder
        folder_groups = pd.unique(group_datasets[AcquisiTags.BASE_PATH]).tolist()

        # Respect the users' folder structure. If things are in different folders, analyze them separately.
        for folder in folder_groups:

            if folder.stem == output_folder.stem:
                continue

            result_folder = folder.joinpath(output_folder)
            result_folder.mkdir(exist_ok=True)

            folder_mask = (group_datasets[AcquisiTags.BASE_PATH] == folder)

            data_in_folder = group_datasets.loc[folder_mask]
            iORG_result_datframes = []

            # Load each modality
            for mode in modes_of_interest:
                this_mode = (group_datasets[DataTags.MODALITY] == mode)
                slice_of_life = folder_mask & this_mode

                data_vidnums = group_datasets.loc[slice_of_life & only_vids, DataTags.VIDEO_ID].unique().tolist()
                #data_vidnums = group_datasets.loc[slice_of_life, DataTags.VIDEO_ID].unique().tolist()

                # Make data storage structures for each of our query location lists- one is for results,
                # The other for checking which query points went into our analysis.
                iORG_result_datframes.append([pd.DataFrame(index=data_vidnums, columns=list(ORGTags)) for i in range((slice_of_life & query_locations).sum())])

                query_status = [pd.DataFrame(columns=data_vidnums) for i in range((slice_of_life & query_locations).sum())]

                pb["maximum"] = len(data_vidnums)
                # Load each dataset (delineated by different video numbers), normalize it, standardize it, etc.
                for v, vidnum in enumerate(data_vidnums):

                    this_vid = (group_datasets[DataTags.VIDEO_ID] == vidnum)

                    # Get the reference images and query locations and this video number, only for the mode and folder mask we want.
                    slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))

                    pb["value"] = v
                    mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(data_vidnums)))

                    # for later: allData.loc[ind, AcquisiTags.DATASET]
                    # Actually load the dataset, and all its metadata.
                    dataset = initialize_and_load_dataset(group_datasets.loc[slice_of_life], metadata_params)

                    if dataset is not None:
                        group_datasets.loc[slice_of_life & only_vids, AcquisiTags.DATASET] = dataset

                        if control_loc == "folder" and folder.stem == control_folder:
                            # If we're in the control folder, then we're a control video.
                            group_datasets.loc[slice_of_life & only_vids, AcquisiTags.STIM_PRESENT] = False
                            continue
                        else:
                            group_datasets.loc[slice_of_life & only_vids, AcquisiTags.STIM_PRESENT] = len(dataset.stimtrain_frame_stamps) > 1
                    else:
                        for q in range(len(query_status)):
                            query_status[q].loc[:, vidnum] = "Dataset Failed To Load"
                        continue

                    # Perform analyses on each query location set for each stimulus dataset.
                    for q in range(len(dataset.query_loc)):


                        pb_label["text"] = "Processing query file " + str(q) + " in dataset #" + str(vidnum) + " from the " + str(
                            mode) + " modality in group " + str(group) + " and folder " + folder.stem + "..."
                        pb.update()
                        pb_label.update()
                        print("Processing query file " + str(dataset.metadata.get(AcquisiTags.QUERYLOC_PATH,Path())[q].stem) +
                              " in dataset #" + str(vidnum) + " from the " + str(mode) + " modality in group "
                              + str(group) + " and folder " + folder.stem + "...")

                        '''
                        *** This section is where we do dataset summary. ***
                        '''
                        valid_signals = np.full((dataset.query_loc[q].shape[0]), True)
                        # The below maps each query loc (some coordinates) to a tuple, then forms those tuples into a list.
                        query_status[q] = query_status[q].reindex(pd.MultiIndex.from_tuples(list(map(tuple, dataset.query_loc[q]))), fill_value="Included")

                        (dataset.iORG_signals[q],
                         dataset.summarized_iORGs[q],
                         query_status[q].loc[:,vidnum],
                         dataset.query_loc[q]) = extract_n_refine_iorg_signals(dataset, analysis_params, query_loc=dataset.query_loc[q])


                    # Once we've extracted the iORG signals, remove the video and mask data as its likely to have a large memory footprint.
                    dataset.clear_video_data()

                query_paths = group_datasets.loc[slice_of_life & query_locations, AcquisiTags.DATA_PATH].tolist()
                for q in range(len(query_status)):
                    query_status[q].to_csv(result_folder.joinpath("query_loc_status_"+str(folder.stem) + "_"+ str(mode) +
                                                                  "_"+query_paths[q].stem +".csv"))


        # If desired, make the summarized iORG relative to controls in some way.
        # Control data is expected to be applied to the WHOLE group.
        for folder in folder_groups:

            result_folder = folder.joinpath(output_folder)

            folder_mask = (group_datasets[AcquisiTags.BASE_PATH] == folder)

            data_in_folder = group_datasets.loc[folder_mask]
            iORG_result_datframes = []

            # Load each modality
            for mode in modes_of_interest:
                this_mode = (group_datasets[DataTags.MODALITY] == mode)
                slice_of_life = folder_mask & this_mode

                has_stim = (group_datasets[AcquisiTags.STIM_PRESENT])

                stim_data_vidnums = np.sort(group_datasets.loc[slice_of_life & only_vids & has_stim, DataTags.VIDEO_ID].unique()).tolist()
                control_data_vidnums = np.sort(group_datasets.loc[this_mode & only_vids & ~has_stim, DataTags.VIDEO_ID].unique()).tolist()

                # Make data storage structures for each of our query location lists- one is for results,
                # The other for checking which query points went into our analysis.
                iORG_result_datframes.append([pd.DataFrame(index=stim_data_vidnums, columns=list(ORGTags)) for i in range((slice_of_life & query_locations).sum())])

                control_query_status = [pd.DataFrame(columns=control_data_vidnums) for i in range((slice_of_life & ~has_stim & query_locations).sum())]

                pb["maximum"] = len(stim_data_vidnums)

                with mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:

                    # Load each dataset (delineated by different video numbers), and process it relative to the control data.
                    for v, stim_vidnum in enumerate(stim_data_vidnums):

                        this_vid = (group_datasets[DataTags.VIDEO_ID] == stim_vidnum)

                        slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))

                        # Grab the stim dataset associated with this video number.
                        stim_dataset = group_datasets.loc[slice_of_life & only_vids, AcquisiTags.DATASET].values[0]

                        # Make a temp storage for the signals and summary data we'll need from the control data.
                        [None] * len(control_data_vidnums)

                        # Process all control datasets in accordance with the stimulus datasets' parameters,
                        # e.g. stimulus location/duration, combine them, and do whatever the user wants with them.

                        pb_label["text"] = "Processing query files in control datasets for stimulus video " + str(
                            stim_vidnum) + " from the " + str(mode) + " modality in group " + str(
                            group) + " and folder " + folder.stem + "..."
                        pb.update()
                        pb_label.update()
                        print("Processing query files in control datasets for stim video" + str(
                            stim_vidnum) + " from the " + str(mode) + " modality in group " + str(
                            group) + " and folder " + folder.stem + "...")

                        res = pool.map(extract_control_iORG_summaries, zip(control_data_vidnums, group_datasets.loc[this_mode & only_vids & ~has_stim, AcquisiTags.DATASET].tolist(), repeat(control_query_status),
                                                                               repeat(analysis_params), repeat(stim_dataset.query_loc), repeat(stim_dataset.stimtrain_frame_stamps)))

                        print("...Done.")

                        # Take all of the results, and collate them for summary iORGs.
                        control_vidnums, control_datasets, control_query_statuses = map(list, zip(*res))


                        for control_query in control_query_statuses:
                            for q in range(len(control_query)):
                                filled_dat = control_query[q].dropna(axis=1, how="all")

                                if len(filled_dat.columns) == 1:
                                    control_query_status[q][filled_dat.columns[0]] = filled_dat.iloc[:,0]
                                else:
                                    warnings.warn("More than one column filled during control iORG summary; results may be inaccurate.")

                        query_paths = group_datasets.loc[slice_of_life & query_locations, AcquisiTags.DATA_PATH].tolist()
                        for q in range(len(control_query_status)):
                            control_query_status[q].to_csv(result_folder.joinpath("query_loc_status_" + str(folder.stem) + "_" + str(mode) +
                                                           "_" + query_paths[q].stem + "_controldata.csv"))

                        # After we've processed all the control data with the parameters of the stimulus data, combine it,
                        # do whatever against stimulus data, and analyze it
                        max_frmstamp = stim_dataset.framestamps[-1]

                        for control_data in control_datasets:
                            max_frmstamp = np.maximum(max_frmstamp, np.amax(control_data.framestamps))


                        for q in range(len(stim_dataset.query_loc)):

                            control_iORG_summaries = np.full((len(control_datasets), max_frmstamp + 1), np.nan)
                            control_iORG_N = np.full((len(control_datasets), max_frmstamp + 1), np.nan)

                            for c, control_data in enumerate(control_datasets):
                                control_iORG_N[c, control_data.framestamps] = np.sum(np.isfinite(control_data.iORG_signals[q]))
                                control_iORG_summaries[c, control_data.framestamps] = control_data.summarized_iORGs[q]

                            control_iORG_summary = np.nansum(control_iORG_N * control_iORG_summaries, axis=0) / np.nansum(control_iORG_N, axis=0)
                            control_framestamps = np.flatnonzero(np.isfinite(control_iORG_summary))

                            stim_iORG_summary = np.full((max_frmstamp + 1,), np.nan)
                            stim_iORG_summary[stim_dataset.framestamps] = stim_dataset.summarized_iORGs[q]

                            if sum_control == "subtract":
                                stim_dataset.summarized_iORGs[q] = stim_iORG_summary - control_iORG_summary
                            elif sum_control == "divide":
                                stim_dataset.summarized_iORGs[q] = stim_iORG_summary / control_iORG_summary
                            elif sum_control == "none":
                                stim_dataset.summarized_iORGs[q] = stim_iORG_summary

                            display_params.get(DisplayParams.POP_SUMMARY_OVERLAP)
                            plt.figure(q)
                            plt.plot(stim_dataset.framestamps, stim_dataset.summarized_iORGs[q][stim_dataset.framestamps])
                            plt.plot(control_framestamps, control_iORG_summary[control_framestamps])
                            plt.show(block=False)

                        stim_dataset.framestamps = np.arange(max_frmstamp + 1)

                        #display_params







                            #TODO: Add plotting class for the below
                            # plt.figure(9)
                            # plt.plot(dataset.framestamps, num_signals_per_sample)
                            # plt.show(block=False)

                            #TODO: Assess the difference this makes. This is just to make them all at the same baseline.
                            #summarized_iORG = standardize_profiles(summarized_iORG[None, :], dataset.framestamps, std_indices=prestim_ind, method="mean_sub")
                            #summarized_iORG = np.squeeze(summarized_iORG)

                            # poststim_ind = np.flatnonzero(np.logical_and(dataset.framestamps >= dataset.stimtrain_frame_stamps[1],
                            #                               dataset.framestamps < (dataset.stimtrain_frame_stamps[1] + int(1 * dataset.framerate))))
                            #
                            # poststim_loc = dataset.framestamps[poststim_ind]
                            # prestim_amp = np.nanmedian(summarized_iORG[prestim_ind])
                            # poststim = summarized_iORG[poststim_ind]
                            #
                            # if poststim.size == 0:
                            #
                            #     pop_iORG_amp[r] = np.NaN
                            #     pop_iORG_implicit[r] = np.NaN
                            #     pop_iORG_recover[r] = np.NaN
                            # else:
                            #     poststim_amp = np.quantile(poststim, [0.95])
                            #     max_frmstmp = poststim_loc[np.argmax(poststim)] - dataset.stimtrain_frame_stamps[0]
                            #     final_val = np.mean(summarized_iORG[-5:])
                            #
                            #     framestamps.append(dataset.framestamps)
                            #     pop_iORG.append(summarized_iORG)
                            #     pop_iORG_num.append(num_signals_per_sample)
                            #
                            #     pop_iORG_amp[r], pop_iORG_implicit[r] = iORG_signal_metrics(summarized_iORG[None, :], dataset.framestamps,
                            #                                                                 filter_type="none", display=False,
                            #                                                                 prestim_idx=prestim_ind,
                            #                                                                 poststim_idx=poststim_ind)[1:3]
                            #
                            #     pop_iORG_recover[r] = 1 - ((final_val - prestim_amp) / pop_iORG_amp[r])
                            #     pop_iORG_implicit[r] = pop_iORG_implicit[r] / dataset.framerate
                            #
                            #     print("Signal metrics based iORG Amplitude: " + str(pop_iORG_amp[r]) +
                            #           " Implicit time (s): " + str(pop_iORG_implicit[r]) +
                            #           " Recovery fraction: " + str(pop_iORG_recover[r]))

                                #TODO: Add plotting class for the below
                                # plt.figure(0)
                                # plt.xlabel("Time (seconds)")
                                # plt.ylabel("Response")
                                # plt.plot(dataset.framestamps/dataset.framerate, pop_iORG[r - skipnum], color=mapper.to_rgba(r - skipnum, norm=False),
                                #          label=file.name)
                                # plt.show(block=False)
                                # plt.xlim([0, 4])
                                # plt.ylim([-5, 40])
                                # plt.savefig(output_folder.joinpath(file.name[0:-4] + "_pop_iORG.png"))

                            # if dataset.framestamps[-1] > max_frmstamp:
                            #     max_frmstamp = dataset.framestamps[-1]

                    #TODO: Add plotting class for the below
                    # dt = datetime.now()
                    # now_timestamp = dt.strftime("%Y_%m_%d_%H_%M_%S")
                    # plt.vlines(dataset.stimtrain_frame_stamps[0] / dataset.framerate, -1, 10, color="red")
                    # plt.xlim([0,  4])
                    # plt.ylim([-5, 60]) #was 60
                    # plt.legend()
                    # plt.savefig( output_folder.joinpath(this_dirname + "_pop_iORG_" + now_timestamp + ".svg"))
                    # plt.savefig( output_folder.joinpath(this_dirname + "_pop_iORG_" + now_timestamp + ".png"))
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
                    # print("Pop mean iORG amplitude: " + str(np.nanmean(pop_iORG_amp, axis=-1)) +
                    #       "Pop stddev iORG amplitude: " + str(np.nanmean(pop_iORG_amp, axis=-1)) )


                    # pop_amp_dFrame = pd.DataFrame(np.concatenate((np.array(pop_iORG_amp, ndmin=2).transpose(),
                    #                                               np.array(pop_iORG_implicit, ndmin=2).transpose(),
                    #                                               np.array(pop_iORG_recover, ndmin=2).transpose()), axis=1),
                    #                               columns=["Amplitude", "Implicit time", "Recovery %"])
                    # pop_amp_dFrame.to_csv(output_folder.joinpath(this_dirname + "_pop_iORG_stats_" + now_timestamp + ".csv"))

                    # Grab all of the
                    # all_iORG = np.empty((len(pop_iORG), max_frmstamp+1))
                    # all_iORG[:] = np.nan
                    # all_incl = np.empty((len(pop_iORG), max_frmstamp + 1))
                    # all_incl[:] = np.nan
                    # for q, iorg in enumerate(pop_iORG):
                    #     all_incl[q, framestamps[q]] = pop_iORG_num[q]
                    #     all_iORG[q, framestamps[q]] = iorg
                    #
                #
                # # Pooled variance calc
                # pooled_iORG = np.nansum( all_incl*all_iORG, axis=0 ) / np.nansum(all_incl, axis=0)
                # #pooled_stddev_iORG = np.sqrt(pooled_var_iORG)
                # all_frmstamps = np.arange(max_frmstamp+1)
                #
                # pop_data_dFrame = pd.DataFrame(np.concatenate((np.array(all_iORG, ndmin=2).transpose(),
                #                                               np.array(pooled_iORG, ndmin=2).transpose()), axis=1))
                # pop_data_dFrame.to_csv(output_folder.joinpath(this_dirname + "_pop_iORG_signals_" + now_timestamp + ".csv"))
                #
                # plt.figure(9)
                # plt.plot(all_frmstamps, np.nansum(all_incl, axis=0))
                # plt.show(block=False)
                #
                # prestim_ind = np.logical_and(all_frmstamps < dataset.stimtrain_frame_stamps[0],
                #                              all_frmstamps >= (dataset.stimtrain_frame_stamps[0] - int(1 * dataset.framerate)))
                # poststim_ind = np.logical_and(all_frmstamps >= dataset.stimtrain_frame_stamps[1],
                #                               all_frmstamps < (dataset.stimtrain_frame_stamps[1] + int(1 * dataset.framerate)))
                # poststim_loc = all_frmstamps[poststim_ind]
                # prestim_amp = np.nanmedian(pooled_iORG[prestim_ind])
                # poststim = pooled_iORG[poststim_ind]
                #
                # if poststim.size == 0:
                #     poststim_amp = np.NaN
                #     prestim_amp = np.NaN
                #     pop_iORG_amp[r] = np.NaN
                #     pop_iORG_implicit[r] = np.NaN
                #     pop_iORG_recover[r] = np.NaN
                # else:
                #     _, pop_iORG_amp[r], pop_iORG_implicit[r], _, pop_iORG_recover[r] = iORG_signal_metrics(pooled_iORG[None, :],
                #                                                                                         dataset.framestamps,
                #                                                                                         filter_type="none", display=False,
                #                                                                                         prestim_idx=prestim_ind,
                #                                                                                         poststim_idx=poststim_ind)
                #     pop_iORG_implicit[r] /= dataset.framerate
                #
                # print("Pooled iORG Avg Amplitude: " + str(pop_iORG_amp[r]) + " Implicit time (s): " + str(pop_iORG_implicit[r]) +
                #       " Recovery fraction: " + str(pop_iORG_recover[r]))
                #
                # pop_amp_dFrame = pd.DataFrame(np.concatenate((np.array(pop_iORG_amp, ndmin=2).transpose(),
                #                                               np.array(pop_iORG_implicit, ndmin=2).transpose(),
                #                                               np.array(pop_iORG_recover, ndmin=2).transpose()), axis=1),
                #                               columns=["Amplitude", "Implicit time", "Recovery %"])
                # pop_amp_dFrame = pop_amp_dFrame.dropna(how="all") # Drop rows with all nans- no point in taking up space.
                # pop_amp_dFrame.to_csv(output_folder.joinpath(this_dirname + "_pop_iORG_stats_" + now_timestamp + ".csv"))
                #
                # plt.figure(10)
                #
                # plt.plot(all_frmstamps / dataset.framerate, pooled_iORG)
                # plt.vlines(dataset.stimtrain_frame_stamps[0] / dataset.framerate, -1, 10, color="red")
                # plt.xlim([0, 6])
                # # plt.ylim([-5, 60]) #was 1, 60
                # plt.xlabel("Time (seconds)")
                # plt.ylabel("Response")
                # plt.show(block=False)
                # plt.savefig(output_folder.joinpath(this_dirname + "_pooled_pop_iORG_" + now_timestamp + ".png"))
                # plt.savefig(output_folder.joinpath(this_dirname + "_pooled_pop_iORG_" + now_timestamp + ".svg"))
                # print("Done!")
                # plt.waitforbuttonpress()


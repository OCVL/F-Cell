import os
import multiprocessing as mp
import warnings
from itertools import repeat
from pathlib import Path, PurePath
from tkinter import Tk, filedialog, ttk, HORIZONTAL, simpledialog, messagebox

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform

from ocvl.function.analysis.iORG_signal_extraction import extract_profiles, norm_profiles, standardize_profiles, \
    refine_coord, refine_coord_to_stack, exclude_profiles, extract_n_refine_iorg_signals
from ocvl.function.analysis.iORG_profile_analyses import summarize_iORG_signals, iORG_signal_metrics
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.dataset import PipeStages, parse_file_metadata, initialize_and_load_dataset
from ocvl.function.utility.json_format_constants import PipelineParams, MetaTags, DataFormatType, DataTags, AcquisiTags, \
    SegmentParams, ExclusionParams, NormParams, STDParams, MetricTags, SummaryParams, ControlParams, DisplayParams, \
    MetricTags

from datetime import datetime, date, time, timezone

def extract_control_iORG_summaries(params):
    control_vidnum, control_dataset, control_query_status, analysis_params, query_locs, stimtrain_frame_stamps = params

    control_query_stat = control_query_status.copy()
    control_dataset.query_loc = [None] * len(query_locs)
    control_dataset.iORG_signals = [None] * len(query_locs)
    control_dataset.summarized_iORGs = [None] * len(query_locs)

    for q in range(len(query_locs)):
        # Use the control data, but the query locations and stimulus info from the stimulus data.
        (control_dataset.iORG_signals[q],
         control_dataset.summarized_iORGs[q],
         control_query_stat[q].loc[:, control_vidnum],
         control_dataset.query_loc[q]) = extract_n_refine_iorg_signals(control_dataset, analysis_params,
                                                                       query_loc=query_locs[q],
                                                                       stimtrain_frame_stamps=stimtrain_frame_stamps)
    return control_vidnum, control_dataset, control_query_stat


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

    norm_params = analysis_params.get(NormParams.NAME, dict())
    method = norm_params.get(NormParams.NORM_METHOD,"score")  # Default: Standardizes the video to a unit mean and stddev
    rescale = norm_params.get(NormParams.NORM_RESCALE,True)  # Default: Rescales the data back into AU to make results easier to interpret
    res_mean = norm_params.get(NormParams.NORM_MEAN, 70)  # Default: Rescales to a mean of 70 - these values are based on "ideal" datasets
    res_stddev = norm_params.get(NormParams.NORM_STD, 35)  # Default: Rescales to a std dev of 35

    # Snag all of our parameter dictionaries that we'll use here.
    # Default to an empty dictionary so that we can query against it with fewer if statements.
    control_params = analysis_params.get(ControlParams.NAME, dict())
    control_loc = control_params.get(ControlParams.LOCATION, "implicit")
    control_folder = control_params.get(ControlParams.FOLDER_NAME, "control")

    sum_params = analysis_params.get(SummaryParams.NAME, dict())
    sum_method = sum_params.get(SummaryParams.METHOD, "rms")
    sum_window = sum_params.get(SummaryParams.WINDOW_SIZE, 1)
    sum_control = sum_params.get(SummaryParams.CONTROL, "none")

    metrics = sum_params.get(SummaryParams.METRICS, dict())
    metrics_type = metrics.get(SummaryParams.TYPE, ["aur", "amplitude", "imp_time", "rec_amp"])
    metrics_measured_to = metrics.get(SummaryParams.MEASURED_TO, "stim-relative")
    metrics_units = metrics.get(SummaryParams.UNITS, "time")
    metrics_prestim = np.array(metrics.get(SummaryParams.PRESTIM, [-1, 0]), dtype=int)
    metrics_poststim = np.array(metrics.get(SummaryParams.POSTSTIM, [0, 1]), dtype=int)


    pop_overlap_params = display_params.get(DisplayParams.POP_SUMMARY_OVERLAP, dict())
    pop_seq_params = display_params.get(DisplayParams.POP_SUMMARY_SEQ, dict())
    indiv_summary_params = display_params.get(DisplayParams.INDIV_SUMMARY, dict())
    saveas_ext = display_params.get(DisplayParams.SAVEAS, "png")


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

        # Use a nested dictionary to track the query status of all query locations; these will later be used
        # in conjuction with status tracked at the dataset level.
        all_query_status = dict()

        # Respect the users' folder structure. If things are in different folders, analyze them separately.
        for folder in folder_groups:

            if folder.stem == output_folder.stem:
                continue

            result_folder = folder.joinpath(output_folder)
            result_folder.mkdir(exist_ok=True)

            folder_mask = (group_datasets[AcquisiTags.BASE_PATH] == folder)

            data_in_folder = group_datasets.loc[folder_mask]

            all_query_status[folder] = dict()

            # Load each modality
            for mode in modes_of_interest:
                this_mode = (group_datasets[DataTags.MODALITY] == mode)
                slice_of_life = folder_mask & this_mode

                data_vidnums = group_datasets.loc[slice_of_life & only_vids, DataTags.VIDEO_ID].unique().tolist()

                # Make data storage structures for each of our query location lists for checking which query points went into our analysis.
                query_loc_names = group_datasets.loc[slice_of_life & query_locations, DataTags.QUERYLOC].unique().tolist()

                first = True
                all_query_status[folder][mode] = [pd.DataFrame(columns=data_vidnums) for i in range((slice_of_life & query_locations).sum())]


                pb["maximum"] = len(data_vidnums)
                # Load each dataset (delineated by different video numbers), normalize it, standardize it, etc.
                for v, vidnum in enumerate(data_vidnums):

                    this_vid = (group_datasets[DataTags.VIDEO_ID] == vidnum)

                    # Get the reference images and query locations and this video number, only for the mode and folder mask we want.
                    slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))

                    pb["value"] = v

                    # Actually load the dataset, and all its metadata.
                    dataset = initialize_and_load_dataset(group_datasets.loc[slice_of_life], metadata_params)

                    if dataset is not None:
                        # Normalize the video to reduce the influence of framewide intensity changes
                        dataset.video_data = norm_video(dataset.video_data, norm_method=method, rescaled=rescale,
                                                        rescale_mean=res_mean, rescale_std=res_stddev)

                        group_datasets.loc[slice_of_life & only_vids, AcquisiTags.DATASET] = dataset

                        if control_loc == "folder" and folder.stem == control_folder:
                            # If we're in the control folder, then we're a control video- and we shouldn't extract
                            # any iORGs until later as our stimulus deliveries may vary.
                            group_datasets.loc[slice_of_life & only_vids, AcquisiTags.STIM_PRESENT] = False

                            continue
                        else:
                            group_datasets.loc[slice_of_life & only_vids, AcquisiTags.STIM_PRESENT] = len(dataset.stimtrain_frame_stamps) > 1
                    else:
                        for q in range(len(all_query_status[folder][mode])):
                            all_query_status[folder][mode][q].loc[:, vidnum] = "Dataset Failed To Load"
                        warnings.warn("Video number "+str(vidnum)+ ": Dataset Failed To Load")
                        continue

                    # Perform analyses on each query location set for each stimulus dataset.
                    for q in range(len(dataset.query_loc)):

                        # If this is the first time a video of this mode and this folder is loaded, then initialize the query status dataframe
                        # Such that each row corresponds to the original coordinate locations based on the reference image.
                        if first:
                            # The below maps each query loc (some coordinates) to a tuple, then forms those tuples into a list.
                            all_query_status[folder][mode][q] = all_query_status[folder][mode][q].reindex(pd.MultiIndex.from_tuples(list(map(tuple, dataset.query_loc[q]))), fill_value="Included")
                            first = False

                        pb_label["text"] = "Processing query file " + query_loc_names[q] + " in dataset #" + str(vidnum) + " from the " + str(
                            mode) + " modality in group " + str(group) + " and folder " + folder.stem + "..."
                        pb.update()
                        pb_label.update()
                        print("Processing query file " + str(dataset.metadata.get(AcquisiTags.QUERYLOC_PATH,Path())[q].stem) +
                              " in dataset #" + str(vidnum) + " from the " + str(mode) + " modality in group "
                              + str(group) + " and folder " + folder.stem + "...")

                        '''
                        *** This section is where we do dataset summary. ***
                        '''
                        (dataset.iORG_signals[q],
                         dataset.summarized_iORGs[q],
                         dataset.query_status[q],
                         dataset.query_loc[q]) = extract_n_refine_iorg_signals(dataset, analysis_params, query_loc=dataset.query_loc[q])

                    # Once we've extracted the iORG signals, remove the video and mask data as it's likely to have a large memory footprint.
                    dataset.clear_video_data()

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
                stim_datasets = group_datasets.loc[slice_of_life & only_vids & has_stim, AcquisiTags.DATASET].tolist()
                control_data_vidnums = np.sort(group_datasets.loc[this_mode & only_vids & ~has_stim, DataTags.VIDEO_ID].unique()).tolist()
                control_datasets = group_datasets.loc[this_mode & only_vids & ~has_stim, AcquisiTags.DATASET].tolist()

                if not stim_datasets:
                    continue

                # Make data storage structures for our results
                query_loc_names = group_datasets.loc[slice_of_life & query_locations, DataTags.QUERYLOC].unique().tolist()
                result_cols = pd.MultiIndex.from_product([query_loc_names, list(MetricTags)])
                iORG_result_datframes = pd.DataFrame(index=stim_data_vidnums, columns=result_cols)

                pb["maximum"] = len(stim_data_vidnums)
                display_dict = {} # A dictionary to store our figure labels and associated filenames for easy saving later.


                # Determine if all stimulus data in this folder and mode has the same form and contents;
                # if so, we can just process the control data *one* time, saving a lot of time.
                # ALSO, we can perform an individual iORG analysis by combining a cells' iORGs across acquisitions
                max_frmstamp = -1
                all_locs = None
                all_timestamps = None
                first_run = True
                uniform_datasets = True
                for d, dataset in enumerate(stim_datasets):
                    locs = dataset.query_loc
                    the_timestamps = dataset.stimtrain_frame_stamps
                    max_frmstamp = np.maximum(max_frmstamp, np.amax(dataset.framestamps))

                    if d != 0:
                        if np.all(all_timestamps.shape != the_timestamps.shape) and np.all(all_timestamps != the_timestamps):
                            warnings.warn("Does not qualify for fast control processing: The stimulus timestamps do not match.")
                            uniform_datasets = False
                            break

                        if len(locs) == len(all_locs):
                            for l, the_locs in enumerate(locs):
                                if np.all(the_locs.shape != all_locs[l].shape) and np.all(the_locs != all_locs[l]):
                                    warnings.warn("Does not qualify for fast control processing: The query locations do not match.")
                                    uniform_datasets = False
                                    break
                        else:
                            warnings.warn("Does not qualify for fast control processing: The number of query locations do not match.")
                            uniform_datasets = False
                            break
                    else:
                        all_locs = locs
                        all_timestamps = the_timestamps

                for control_data in control_datasets:
                    max_frmstamp = np.maximum(max_frmstamp, np.amax(control_data.framestamps))

                stim_iORG_summary = [None] * len(all_locs)
                stim_pop_iORG_summary = [None] * len(all_locs)
                stim_iORG_signals = None

                control_iORG_summary = [None] * len(all_locs)
                control_pop_iORG_summary = [None] * len(all_locs)
                control_framestamps = [None] * len(all_locs)

                if uniform_datasets:
                    stim_iORG_signals = [None] * len(all_locs)
                    for q in range(len(all_locs)):
                        stim_iORG_signals[q] = np.full((len(stim_datasets), all_locs[q].shape[0], max_frmstamp + 1), np.nan)

                # Load each dataset (delineated by different video numbers), and process it relative to the control data.
                mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(stim_data_vidnums)))
                for v, stim_vidnum in enumerate(stim_data_vidnums):

                    control_query_status = [pd.DataFrame(columns=control_data_vidnums) for i in range((slice_of_life & ~has_stim & query_locations).sum())]

                    this_vid = (group_datasets[DataTags.VIDEO_ID] == stim_vidnum)

                    slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))

                    # Grab the stim dataset associated with this video number.
                    stim_dataset = group_datasets.loc[slice_of_life & only_vids, AcquisiTags.DATASET].values[0]

                    # Process all control datasets in accordance with the stimulus datasets' parameters,
                    # e.g. stimulus location/duration, combine them, and do whatever the user wants with them.
                    if not uniform_datasets or first_run:
                        with (mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool):

                            first_run = False
                            control_iORG_summary = [None] * len(stim_dataset.query_loc)
                            control_pop_iORG_summary = [None] * len(stim_dataset.query_loc)
                            control_framestamps = [None] * len(stim_dataset.query_loc)
                            control_datasets = group_datasets.loc[this_mode & only_vids & ~has_stim, AcquisiTags.DATASET].tolist()

                            pb_label["text"] = "Processing query files in control datasets for stimulus video " + str(
                                stim_vidnum) + " from the " + str(mode) + " modality in group " + str(
                                group) + " and folder " + folder.stem + "..."
                            pb.update()
                            pb_label.update()
                            print("Processing query files in control datasets for stim video" + str(
                                stim_vidnum) + " from the " + str(mode) + " modality in group " + str(
                                group) + " and folder " + folder.stem + "...")

                            res = pool.map(extract_control_iORG_summaries, zip(control_data_vidnums, control_datasets, repeat(control_query_status.copy()),
                                                                                   repeat(analysis_params), repeat(stim_dataset.query_loc),
                                                                                 repeat(stim_dataset.stimtrain_frame_stamps)))

                            print("...Done.")

                            # Take all of the results, and collate them for summary iORGs.
                            control_vidnums, control_datasets, control_query_status_res = map(list, zip(*res))

                            for control_query in control_query_status_res:
                                for q in range(len(control_query)):
                                    filled_dat = control_query[q].dropna(axis=1, how="all")

                                    if len(filled_dat.columns) == 1:
                                        control_query_status[q][filled_dat.columns[0]] = filled_dat.iloc[:,0]
                                    elif len(filled_dat.columns) > 1:
                                        warnings.warn("More than one column filled during control iORG summary; results may be inaccurate.")
                                    else:
                                        warnings.warn("Column missing from control iORG summary.")

                            # After we've processed all the control data with the parameters of the stimulus data, combine it
                            for q in range(len(control_query_status)):

                                control_pop_iORG_summaries = np.full((len(control_datasets), max_frmstamp + 1), np.nan)
                                control_pop_iORG_N = np.full((len(control_datasets), max_frmstamp + 1), np.nan)
                                control_iORG_summaries = np.full((len(control_datasets), stim_dataset.query_loc[q].shape[0],  max_frmstamp + 1), np.nan)

                                for c, control_data in enumerate(control_datasets):
                                    control_pop_iORG_N[c, control_data.framestamps] = np.sum(np.isfinite(control_data.iORG_signals[q]))
                                    control_iORG_summaries[c, :, control_data.framestamps] = control_data.iORG_signals[q].T
                                    control_pop_iORG_summaries[c, control_data.framestamps] = control_data.summarized_iORGs[q]

                                # Summarize each of the cells' iORGs
                                # *** Removed for now, may add later.
                                # for c in range(control_iORG_summaries.shape[1]):
                                #     tot_sig = np.nansum(np.any(np.isfinite(control_iORG_summaries[:, c, :]), axis=1))
                                #     control_query_status[q].loc[c, "Num Viable iORGs"] = tot_sig
                                #
                                #     if tot_sig > sum_params.get(SummaryParams.INDIV_CUTOFF, 5):
                                #         control_query_status[q].loc[c, "Viable for single-cell summary?"] = True
                                #         control_iORG_summary[q][c, :], _ = summarize_iORG_signals(control_iORG_summaries[:, c, :],
                                #                                                                np.arange(max_frmstamp + 1),
                                #                                                                summary_method=sum_method,
                                #                                                                window_size=sum_window)


                                control_pop_iORG_summary[q] = np.nansum(control_pop_iORG_N * control_pop_iORG_summaries,
                                                                        axis=0) / np.nansum(control_pop_iORG_N, axis=0)
                                control_framestamps[q] = np.flatnonzero(np.isfinite(control_pop_iORG_summary[q]))

                                # First write the control data to a file.
                                control_query_status[q].to_csv(result_folder.joinpath("query_loc_status_" + str(folder.stem) + "_" + str(mode) +
                                                           "_" + query_loc_names[q] + "coords_controldata.csv"))

                    ''' *** Population iORG analyses start here *** '''
                    for q in range(len(stim_dataset.query_loc)):
                        all_query_status[folder][mode][q].loc[:, stim_vidnum] = stim_dataset.query_status[q]

                        stim_pop_summary = np.full((max_frmstamp + 1,), np.nan)
                        stim_pop_summary[stim_dataset.framestamps] = stim_dataset.summarized_iORGs[q]
                        stim_framestamps = np.arange(max_frmstamp + 1)

                        if sum_control == "subtraction":
                            stim_dataset.summarized_iORGs[q] = stim_pop_summary - control_pop_iORG_summary[q]
                        elif sum_control == "division":
                            stim_dataset.summarized_iORGs[q] = stim_pop_summary / control_pop_iORG_summary[q]
                        elif sum_control == "none":
                            stim_dataset.summarized_iORGs[q] = stim_pop_summary

                        ''' This section is for display of the above iORG summaries, as specified by the user. '''
                        # This shows all summaries overlapping.
                        if pop_overlap_params:

                            disp_stim = pop_overlap_params.get(DisplayParams.DISP_STIMULUS, True)
                            disp_cont = pop_overlap_params.get(DisplayParams.DISP_CONTROL, True)
                            disp_rel = pop_overlap_params.get(DisplayParams.DISP_RELATIVE, True)
                            how_many = disp_stim + disp_cont + disp_rel

                            overlap_label = "Query file " + query_loc_names[q] + ": summarized using "+ sum_method+ " of " +mode +" iORGs in "+folder.stem
                            plt.figure( overlap_label )
                            display_dict[mode+"_pop_iORG_"+ sum_method +"_overlapping_"+query_loc_names[q]+"coords_"+folder.stem] = overlap_label
                            ind = 1
                            if how_many > 1 and disp_stim:
                                plt.subplot(1, how_many, ind)
                                ind += 1
                            if disp_stim:
                                plt.title("Stimulus iORG")
                                plt.plot(stim_framestamps / stim_dataset.framerate, stim_pop_summary)
                                plt.xlabel("Time (s)")
                                plt.ylabel(sum_method)
                            if how_many > 1 and disp_cont:
                                plt.subplot(1, how_many, ind)
                                ind += 1
                            if disp_cont:
                                plt.title("Summarized control iORG")
                                plt.plot(control_framestamps[q] / stim_dataset.framerate, control_pop_iORG_summary[q][control_framestamps[q]])
                                plt.xlabel("Time (s)")
                                plt.ylabel(sum_method)
                            if how_many > 1 and disp_rel:
                                plt.subplot(1, how_many, ind)
                                ind += 1
                            if disp_rel:
                                plt.title("Stimulus relative to control iORG via " + sum_control)
                                plt.plot(stim_framestamps/stim_dataset.framerate, stim_dataset.summarized_iORGs[q])
                                plt.xlabel("Time (s)")
                                plt.ylabel(sum_method)

                            plt.show(block=False)

                        # This shows all summaries in temporal sequence.
                        if pop_seq_params:
                            num_in_seq = pop_seq_params.get(DisplayParams.NUM_IN_SEQ, 0)

                            vidnum_seq = np.array(stim_data_vidnums).astype(np.int32)
                            vidnum_seq -= vidnum_seq[0]
                            # Go through all our video numbers until they're in a sequence we can work with.
                            while np.any(vidnum_seq>num_in_seq):
                                vidnum_seq[vidnum_seq>num_in_seq] -= np.amin(vidnum_seq[vidnum_seq>num_in_seq])

                            seq_row = int(np.ceil(num_in_seq/5))

                            if pop_seq_params.get(DisplayParams.DISP_STIMULUS, True):
                                seq_stim_label = "Query file " + query_loc_names[q] + ": Stimulus iORG temporal sequence of " +mode +" iORGs in "+folder.stem
                                plt.figure(seq_stim_label)
                                display_dict[mode + "_pop_iORG_" + sum_method + "_sequential_stim_only_" + query_loc_names[q] + "coords_" + folder.stem] = seq_stim_label

                                plt.subplot(seq_row, 5, (vidnum_seq[v] % num_in_seq) + 1)
                                plt.title("Acquisition "+ str(vidnum_seq[v] % num_in_seq) + " of " + str(num_in_seq))
                                plt.plot(stim_framestamps / stim_dataset.framerate, stim_pop_summary)
                                plt.xlabel("Time (s)")
                                plt.ylabel(sum_method)

                            if pop_seq_params.get(DisplayParams.DISP_RELATIVE, True):
                                seq_rel_label = "Query file " + query_loc_names[q] + "Stimulus relative to control iORG via " + sum_control +" temporal sequence"
                                plt.figure(seq_rel_label)
                                display_dict[mode + "_pop_iORG_" + sum_method + "_sequential_relative_" + query_loc_names[q] + "coords_" + folder.stem] = seq_rel_label

                                plt.subplot(seq_row, 5, (vidnum_seq[v] % num_in_seq) + 1)
                                plt.title("Acquisition "+ str(vidnum_seq[v] % num_in_seq) + " of " + str(num_in_seq))
                                plt.plot(stim_framestamps/stim_dataset.framerate,stim_dataset.summarized_iORGs[q])
                                plt.xlabel("Time (s)")
                                plt.ylabel(sum_method)

                            plt.show(block=False)

                        metrics_prestim = np.array(metrics.get(SummaryParams.PRESTIM, [-1, 0]), dtype=int)
                        metrics_poststim = np.array(metrics.get(SummaryParams.POSTSTIM, [0, 1]), dtype=int)
                        if metrics_units == "time":
                            metrics_prestim = np.round(metrics_prestim * dataset.framerate)
                            metrics_poststim = np.round(metrics_poststim * dataset.framerate)
                        else:  # if units == "frames":
                            metrics_prestim = np.round(metrics_prestim)
                            metrics_poststim = np.round(metrics_poststim)

                        if metrics_measured_to == "stim-relative":
                            metrics_prestim = stim_dataset.stimtrain_frame_stamps[0] + metrics_prestim
                            metrics_poststim = stim_dataset.stimtrain_frame_stamps[1] + metrics_poststim

                        # Make the list of indices that should correspond to pre and post stimulus
                        metrics_prestim = np.arange(start=metrics_prestim[0], stop=metrics_prestim[1], step=1, dtype=int)
                        metrics_poststim = np.arange(start=metrics_poststim[0], stop=metrics_poststim[1], step=1, dtype=int)
                        # Find the indexes of the framestamps corresponding to our pre and post stim frames;
                        prestim = np.flatnonzero(np.isin(stim_dataset.framestamps, metrics_prestim))
                        poststim = np.flatnonzero(np.isin(stim_dataset.framestamps, metrics_poststim))

                        # if we're missing an *end* framestamp in our window, interpolate to find the value there,
                        # and add it temporarily to our signal to make sure things like AUR work correctly.
                        iORG_summary = stim_dataset.summarized_iORGs[q][stim_dataset.framestamps]
                        iORG_frmstmp = stim_dataset.framestamps
                        if not np.any(iORG_frmstmp[poststim] == metrics_poststim[-1]):
                            inter_val = np.interp(metrics_poststim[-1], iORG_frmstmp, iORG_summary)
                            # Find where to insert the interpolant and its framestamp
                            next_highest = np.argmax(iORG_frmstmp > metrics_poststim[-1])
                            iORG_summary = np.insert(iORG_summary, next_highest, inter_val)
                            iORG_frmstmp = np.insert(iORG_frmstmp, next_highest, metrics_poststim[-1])
                            poststim = np.append(poststim, next_highest)

                        amplitude, implicit_time, aur, recovery = iORG_signal_metrics(iORG_summary, iORG_frmstmp, stim_dataset.framerate,
                                                                                      prestim, poststim)

                        for metric in metrics_type:
                            if metric == "aur":
                                iORG_result_datframes.loc[stim_vidnum, (query_loc_names[q], MetricTags.AUR)] = aur
                            elif metric == "amplitude":
                                iORG_result_datframes.loc[stim_vidnum, (query_loc_names[q],  MetricTags.AMPLITUDE)] = amplitude
                            elif metric == "imp_time":
                                iORG_result_datframes.loc[stim_vidnum, (query_loc_names[q],  MetricTags.IMPLICT_TIME)] = implicit_time
                            elif metric == "rec_amp":
                                iORG_result_datframes.loc[stim_vidnum, (query_loc_names[q],  MetricTags.RECOVERY_PERCENT)] = recovery


                # Average all stimulus population iORGs, and prep for individual cone analyses.
                pooled_framerate = [None] * len(all_locs)
                for q in range(len(all_locs)):
                    stim_pop_iORG_summaries = np.full((len(stim_datasets), max_frmstamp + 1), np.nan)
                    stim_pop_iORG_N = np.full((len(stim_datasets), max_frmstamp + 1), np.nan)
                    pooled_framerate[q] = np.full((len(stim_datasets), ), np.nan)
                    iORG_frmstmp = np.arange(max_frmstamp + 1)

                    if uniform_datasets:
                        stim_iORG_summaries = np.full((len(stim_datasets), stim_datasets[0].query_loc[q].shape[0], max_frmstamp + 1),
                                                      np.nan)
                        stim_iORG_summary[q] = np.full((stim_datasets[0].query_loc[q].shape[0], max_frmstamp + 1),
                                                      np.nan)

                    for d, stim_dataset in enumerate(stim_datasets):
                        pooled_framerate[q] = stim_dataset.framerate
                        stim_pop_iORG_N[d, stim_dataset.framestamps] = np.nansum(np.isfinite(stim_dataset.iORG_signals[q]), axis=0)
                        stim_pop_iORG_summaries[d, :] = stim_dataset.summarized_iORGs[q]
                        # If all stimulus datasets are uniform,
                        # we can also summarize individual iORGs by combining a cells' iORGs across acquisitions.
                        if uniform_datasets:
                            stim_iORG_summaries[d, :, stim_dataset.framestamps] = stim_dataset.iORG_signals[q].T

                    pooled_framerate[q] = np.unique(pooled_framerate[q])
                    if len(pooled_framerate[q]) != 1:
                        warnings.warn("The framerate of the iORGs analyzed in "+folder.stem + " is inconsistent! Results may be incorrect.")
                        pooled_framerate[q] = pooled_framerate[q][0]

                    ''' *** Pool the summarized population iORGs *** '''
                    stim_pop_iORG_summary[q] = np.nansum(stim_pop_iORG_N * stim_pop_iORG_summaries,axis=0) / np.nansum(stim_pop_iORG_N, axis=0)

                    metrics_prestim = np.array(metrics.get(SummaryParams.PRESTIM, [-1, 0]), dtype=int)
                    metrics_poststim = np.array(metrics.get(SummaryParams.POSTSTIM, [0, 1]), dtype=int)
                    if metrics_units == "time":
                        metrics_prestim = np.round(metrics_prestim * pooled_framerate[q])
                        metrics_poststim = np.round(metrics_poststim * pooled_framerate[q])
                    else:  # if units == "frames":
                        metrics_prestim = np.round(metrics_prestim)
                        metrics_poststim = np.round(metrics_poststim)

                    if metrics_measured_to == "stim-relative":
                        metrics_prestim = stim_dataset.stimtrain_frame_stamps[0] + metrics_prestim
                        metrics_poststim = stim_dataset.stimtrain_frame_stamps[1] + metrics_poststim

                    # Make the list of indices that should correspond to pre and post stimulus
                    metrics_prestim = np.arange(start=metrics_prestim[0], stop=metrics_prestim[1], step=1,
                                                dtype=int)
                    metrics_poststim = np.arange(start=metrics_poststim[0], stop=metrics_poststim[1], step=1,
                                                 dtype=int)

                    finite_iORG = np.isfinite(stim_pop_iORG_summary[q])
                    iORG_summary = stim_pop_iORG_summary[q][finite_iORG]
                    iORG_frmstmp = iORG_frmstmp[finite_iORG]
                    # Find the indexes of the framestamps corresponding to our pre and post stim frames;
                    prestim = np.flatnonzero(np.isin(iORG_frmstmp, metrics_prestim))
                    poststim = np.flatnonzero(np.isin(iORG_frmstmp, metrics_poststim))

                    # if we're missing an *end* framestamp in our window, interpolate to find the value there,
                    # and add it temporarily to our signal to make sure things like AUR work correctly.
                    if not np.any(iORG_frmstmp[poststim] == metrics_poststim[-1]):
                        inter_val = np.interp(metrics_poststim[-1], iORG_frmstmp, iORG_summary)
                        # Find where to insert the interpolant and its framestamp
                        next_highest = np.argmax(iORG_frmstmp > metrics_poststim[-1])
                        iORG_summary = np.insert(iORG_summary, next_highest, inter_val)
                        iORG_frmstmp = np.insert(iORG_frmstmp, next_highest, metrics_poststim[-1])
                        poststim = np.append(poststim, next_highest)

                    amplitude, implicit_time, aur, recovery = iORG_signal_metrics(iORG_summary, iORG_frmstmp,
                                                                                  pooled_framerate[q],
                                                                                  prestim, poststim)
                    for metric in metrics_type:
                        if metric == "aur":
                            iORG_result_datframes.loc["Pooled", (query_loc_names[q], MetricTags.AUR)] = aur
                        elif metric == "amplitude":
                            iORG_result_datframes.loc[
                                "Pooled", (query_loc_names[q], MetricTags.AMPLITUDE)] = amplitude
                        elif metric == "imp_time":
                            iORG_result_datframes.loc[
                                "Pooled", (query_loc_names[q], MetricTags.IMPLICT_TIME)] = implicit_time
                        elif metric == "rec_amp":
                            iORG_result_datframes.loc[
                                "Pooled", (query_loc_names[q], MetricTags.RECOVERY_PERCENT)] = recovery

                    # Display the pooled population data
                    if pop_overlap_params:
                        overlap_label = "Pooled data summarized with " + sum_method + " of " + mode + " iORGs in " + folder.stem
                        plt.figure(overlap_label)
                        display_dict["pooled_" + mode + "_pop_iORG_" + sum_method + "_overlapping"] = overlap_label
                        plt.title("Pooled summarized iORGs")
                        plt.plot(np.arange(max_frmstamp + 1) / pooled_framerate[q], stim_pop_iORG_summary[q],
                                 label=query_loc_names[q] + " relative to control")
                        plt.xlabel("Time (s)")
                        plt.ylabel(sum_method)
                        plt.legend()
                        plt.show(block=False)

                    # If we have a uniform dataset, summarize each cell's iORG too.
                    ''' *** Individual iORG analyses start here *** '''
                    if uniform_datasets:
                        iORG_frmstmp = np.arange(max_frmstamp + 1)

                        plt.figure("Individual-Cell iORG Summaries")

                        for c in range(stim_iORG_summaries.shape[1]):
                            tot_sig = np.nansum(np.any(np.isfinite(stim_iORG_summaries[:, c, :]), axis=1))
                            idx = all_query_status[folder][mode][q].index[c]
                            all_query_status[folder][mode][q].loc[idx, "Num Viable iORGs"] = tot_sig

                            # If we have more signals than our cutoff, then continue with the summary.
                            if tot_sig > sum_params.get(SummaryParams.INDIV_CUTOFF, 5):
                                all_query_status[folder][mode][q].loc[idx, "Viable for single-cell summary?"] = True
                                stim_iORG_summary[q][c, :], _ = summarize_iORG_signals(stim_iORG_summaries[:, c, :], iORG_frmstmp,
                                                                                       summary_method=sum_method,
                                                                                       window_size=sum_window)

                                if sum_control == "subtraction":
                                    stim_iORG_summary[q][c, :] = stim_iORG_summary[q][c, :] - control_pop_iORG_summary[q]
                                elif sum_control == "division":
                                    stim_iORG_summary[q][c, :] = stim_iORG_summary[q][c, :] / control_pop_iORG_summary[q]
                                elif sum_control == "none":
                                    stim_iORG_summary[q][c, :] = stim_iORG_summary[q][c, :]

                                plt.plot(iORG_frmstmp, stim_iORG_summary[q][c, :])
                            else:
                                all_query_status[folder][mode][q].loc[idx, "Viable for single-cell summary?"] = False

                        plt.show(block=False)



                    all_query_status[folder][mode][q].to_csv(result_folder.joinpath("query_loc_status_" + str(folder.stem) + "_" + str(mode) +
                                               "_" + query_loc_names[q] + "coords.csv"))




                respath = result_folder.joinpath("pop_summary_metrics_" + str(folder.stem) + "_" + str(mode) + ".csv")
                tryagain = True
                while tryagain:
                    try:
                        iORG_result_datframes.to_csv(respath)
                        tryagain = False
                    except PermissionError:
                        tryagain=messagebox.askyesno(title="File: " + str(respath) + " is unable to be written.",
                                                      message="The result file may be open. Close the file, then try to write again?")

                if display_params.get(DisplayParams.PAUSE_PER_FOLDER, False):
                    plt.waitforbuttonpress()

                # Save the figures to the result folder, if requested.
                for fname, figname in display_dict.items():
                    plt.figure(figname)
                    plt.gcf().set_size_inches(10, 4)
                    for ext in saveas_ext:
                        plt.savefig(result_folder.joinpath(fname+"."+ext), dpi=300)
                    plt.close(figname)


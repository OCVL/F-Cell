import gc
import json
import os
import multiprocessing as mp
import warnings
from itertools import repeat
from pathlib import Path, PurePath
from tkinter import Tk, filedialog, ttk, HORIZONTAL, messagebox

import cv2
import numpy as np
import pandas as pd
from colorama import Fore
from matplotlib import pyplot as plt
import matplotlib as mpl
from datetime import datetime

from scipy.ndimage import gaussian_filter

from ocvl.function.analysis.iORG_signal_extraction import extract_n_refine_iorg_signals
from ocvl.function.analysis.iORG_profile_analyses import summarize_iORG_signals, iORG_signal_metrics
from ocvl.function.display.iORG_data_display import display_iORG_pop_summary, display_iORG_pop_summary_seq, \
    display_iORG_summary_histogram, display_iORG_summary_overlay, display_iORGs
from ocvl.function.preprocessing.improc import norm_video, flat_field
from ocvl.function.utility.dataset import parse_file_metadata, initialize_and_load_dataset, Stages, postprocess_dataset, \
    obtain_analysis_output_path
from ocvl.function.utility.json_format_constants import PreAnalysisPipeline, MetaTags, DataFormatType, DataTags, \
    AcquisiTags, \
    NormParams, SummaryParams, ControlParams, DisplayParams, \
    MetricTags, Analysis, SegmentParams, ConfigFields, DebugParams
from ocvl.function.utility.resources import save_tiff_stack, save_video

if __name__ == "__main__":

    mpl.rcParams['lines.linewidth'] = 2.5

    dt = datetime.now()
    start_timestamp = dt.strftime("%Y%m%d_%H%M")

    root = Tk()
    root.lift()
    w = 1
    h = 1
    x = root.winfo_screenwidth() / 4
    y = root.winfo_screenheight() / 4
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.

    pName = None
    json_fName = Path()
    dat_form = dict()
    allData = pd.DataFrame()

    while allData.empty:
        pName = filedialog.askdirectory(title="Select the folder containing all videos of interest.", initialdir=pName, parent=root)
        if not pName:
            quit()

        # We should be 3 levels up from here. Kinda jank, will need to change eventually
        config_path = Path(os.path.dirname(__file__)).parent.parent.joinpath("config_files")

        json_fName = filedialog.askopenfilename(title="Select the configuration json file.", initialdir=config_path, parent=root)
        if not json_fName:
            quit()

        # Grab all the folders/data here.
        dat_form, allData = parse_file_metadata(json_fName, pName, Analysis.NAME)

        if allData.empty:
            tryagain= messagebox.askretrycancel("No data detected.", "No data detected in folder using patterns detected in json. \nSelect new folder (retry) or exit? (cancel)")
            if not tryagain:
                quit()

    x = root.winfo_screenwidth() / 2 - 128
    y = root.winfo_screenheight() / 2 - 128
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.
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

    analysis_dat_format = dat_form.get(Analysis.NAME)
    preanalysis_dat_format = dat_form.get(PreAnalysisPipeline.NAME, dict())
    pipeline_params = preanalysis_dat_format.get(PreAnalysisPipeline.PARAMS, dict())
    analysis_params = analysis_dat_format.get(Analysis.PARAMS)
    display_params = analysis_dat_format.get(DisplayParams.NAME)
    modes_of_interest = analysis_params.get(Analysis.MODALITIES)

    seg_params = analysis_params.get(SegmentParams.NAME, dict())
    seg_pixelwise = seg_params.get(SegmentParams.PIXELWISE, False)  # Default to NO pixelwise analyses. Otherwise, add one.

    # Snag all of our parameter dictionaries that we'll use here.
    # Default to an empty dictionary so that we can query against it with fewer if statements.
    control_params = analysis_dat_format.get(ControlParams.NAME, dict())
    control_loc = control_params.get(ControlParams.LOCATION, "implicit")
    control_folder_name = control_params.get(ControlParams.FOLDER_NAME, "control")

    sum_params = analysis_params.get(SummaryParams.NAME, dict())
    sum_method = sum_params.get(SummaryParams.METHOD, "rms")
    sum_window = sum_params.get(SummaryParams.WINDOW_SIZE, 1)
    sum_control = sum_params.get(SummaryParams.CONTROL, "none")

    metrics = sum_params.get(SummaryParams.METRICS, dict())
    metrics_type = metrics.get(SummaryParams.TYPE, ["amplitude", "imp_time"])
    metrics_measured_to = metrics.get(SummaryParams.MEASURED_TO, "stim-relative")
    metrics_units = metrics.get(SummaryParams.UNITS, "time")
    metrics_prestim = np.array(metrics.get(SummaryParams.PRESTIM, [-1, 0]), dtype=int)
    metrics_poststim = np.array(metrics.get(SummaryParams.POSTSTIM, [0, 1]), dtype=int)

    metrics_tags =[]
    for metric in metrics_type:
        # Need to do this in a pythonic way. Honestly it's fucking hideous.
        if metric == "aur":
            metrics_tags.append(MetricTags.AUR)
        elif metric == "amp":
            metrics_tags.append(MetricTags.AMPLITUDE)
        elif metric == "logamp":
            metrics_tags.append(MetricTags.LOG_AMPLITUDE)
        elif metric == "amp_imp_time":
            metrics_tags.append(MetricTags.AMP_IMPLICIT_TIME)
        elif metric == "halfamp_imp_time":
            metrics_tags.append(MetricTags.HALFAMP_IMPLICIT_TIME)
        elif metric == "rec_amp":
            metrics_tags.append(MetricTags.RECOVERY_PERCENT)

    pop_overlap_params = display_params.get(DisplayParams.POP_SUMMARY_OVERLAP, dict())
    debug_params = display_params.get(DebugParams.NAME, dict())
    pop_seq_params = display_params.get(DisplayParams.POP_SUMMARY_SEQ, dict())
    indiv_overlap_params = display_params.get(DisplayParams.INDIV_SUMMARY_OVERLAP, dict())
    indiv_summary = display_params.get(DisplayParams.INDIV_SUMMARY, dict())
    saveas_ext = display_params.get(DisplayParams.SAVEAS, "png")

    # Debug parameters. All of these default to off, unless explicitly flagged on in the json.

    # Extract all of the unique values that we'll be iterating over for the analyses.
    if DataTags.DATA_ID in allData:
        subject_IDs = allData[DataTags.DATA_ID].unique()

        if np.size(subject_IDs) > 1:
            warnings.warn("MORE THAN 1 SUBJECT ID DETECTED!! Labeling outputs with first ID")
    else:
        warnings.warn("NO SUBJECT ID FIELD DETECTED IN allData! Labeling outputs with dummy subject ID")
        subject_IDs = [''] #Trying empty subject ID


    metadata_params = None
    if analysis_dat_format.get(MetaTags.METATAG) is not None:
        metadata_params = analysis_dat_format.get(MetaTags.METATAG)
        metadata_form = metadata_params.get(DataFormatType.METADATA)

    # If we've selected modalities of interest, only process those; otherwise, process them all.
    if modes_of_interest is None:
        modes_of_interest = allData[DataTags.MODALITY].unique().tolist()
        print("NO MODALITIES SELECTED! Processing all....")

    grouping = pipeline_params.get(PreAnalysisPipeline.GROUP_BY)
    if grouping is not None:
        for row in allData.itertuples():
            #print(grouping.format_map(row._asdict()))
            allData.loc[row.Index, PreAnalysisPipeline.GROUP_BY] = grouping.format_map(row._asdict())

        groups = allData[PreAnalysisPipeline.GROUP_BY].unique().tolist()
    else:
        allData[PreAnalysisPipeline.GROUP_BY] = ""
        groups = [""]  # If we don't have any groups, then just make the list an empty string.

    output_folder = analysis_params.get(Analysis.OUTPUT_FOLDER)
    if output_folder is None:
        output_folder = PurePath("Results")
    else:
        output_folder = PurePath(output_folder)


    subfolder_flag = 0

    if analysis_params.get(Analysis.OUTPUT_SUBFOLDER, True): #Is output subfolder field true (ie does the user want to save to a subfolder?)
        output_subfolder_method = analysis_params.get(Analysis.OUTPUT_SUBFOLDER_METHOD) #Check subfolder naming method
        if output_subfolder_method == 'DateTime': #Only supports saving things to a subfolder with a unique timestamp currently
            output_dt_subfolder = PurePath(start_timestamp)
        else:
            output_dt_subfolder = PurePath(start_timestamp)


    with (mp.Pool(processes=mp.cpu_count() // 2) as the_pool):
        # First break things down by group, defined by the user in the config file.
        # We like to use (LocX,LocY), but this is by no means the only way.
        for group in groups:

            group_filter = allData[PreAnalysisPipeline.GROUP_BY] == group

            allData.loc[group_filter, AcquisiTags.STIM_PRESENT] = True
            allData.loc[group_filter, AcquisiTags.STIM_PRESENT] = allData.loc[group_filter, AcquisiTags.STIM_PRESENT].astype(bool)

            refim_filter = allData[DataFormatType.FORMAT_TYPE] == DataFormatType.IMAGE
            qloc_filter = allData[DataFormatType.FORMAT_TYPE] == DataFormatType.QUERYLOC
            vidtype_filter = allData[DataFormatType.FORMAT_TYPE] == DataFormatType.VIDEO

            # While we're going to process by group, respect the folder structure used by the user here, and only group
            # and analyze things from the same folder
            folder_groups = pd.unique(allData.loc[group_filter, AcquisiTags.BASE_PATH]).tolist()

            # Use a nested dictionary to track the query status of all query locations; these will later be used
            # in conjuction with status tracked at the dataset level.
            all_query_status = dict()
            display_dict = {}  # A dictionary to store our figure labels and associated filenames for easy saving later.

            # Load each modality
            for mode in modes_of_interest:

                all_query_status[mode] = dict()

                mode_filter = (allData[DataTags.MODALITY] == mode)
                

                # Find the control data, if any, in this group and mode, and load it.
                if control_loc == "folder" and any(control_folder_name == fold.name for fold in folder_groups):

                    control_folder_filter = allData[AcquisiTags.BASE_PATH].apply(lambda ra: ra.name == control_folder_name)

                    cntl_slice_of_life = group_filter & control_folder_filter & mode_filter
                    
                    data_vidnums = np.sort(allData.loc[cntl_slice_of_life & vidtype_filter, DataTags.VIDEO_ID].unique()).tolist()

                    pb["maximum"] = len(data_vidnums)+1
                    for v, vidnum in enumerate(data_vidnums):
                        pb["value"] = v
                        pb_label["text"] = "Loading control dataset "+str(vidnum)+"... ("+str(v)+" of "+str(len(data_vidnums))+")"
                        pb.update()
                        pb_label.update()

                        vidid_filter = allData[DataTags.VIDEO_ID] == vidnum

                        allData.loc[cntl_slice_of_life & vidid_filter & vidtype_filter, AcquisiTags.STIM_PRESENT] = False
                        control_path = allData.loc[cntl_slice_of_life & vidid_filter & vidtype_filter, AcquisiTags.BASE_PATH].values[0]

                        # Actually load the control dataset.
                        pre_filter = (allData[PreAnalysisPipeline.GROUP_BY] == group) & \
                                     (allData[DataTags.MODALITY] == mode)
                        dataset, new_entries = initialize_and_load_dataset(control_path, vidnum, pre_filter, start_timestamp, allData,
                                                                           analysis_dat_format, stage=Stages.ANALYSIS)

                        if dataset is not None:
                            pass # Do something with control data? Nah, not yet probably...

                    pb["value"] = len(data_vidnums)
                    pb_label["text"] = "Loading control dataset " + str(vidnum) + "... Done!"
                    pb.update()
                    pb_label.update()

                # Respect the users' folder structure. If things are in different folders, analyze them separately.
                for folder in folder_groups:

                    # Initialize data lists
                    pop_iORG_result_datframe = []
                    indiv_iORG_result_datframe = []

                    if folder.name == output_folder.name:
                        continue
                    if control_loc == "folder" and folder.name == control_folder_name:
                        continue # Don't reload control data.

                    result_path = obtain_analysis_output_path(folder, start_timestamp, analysis_params)

                    folder_filter = allData[AcquisiTags.BASE_PATH] == folder
                    slice_of_life = group_filter & mode_filter & folder_filter


                    data_vidnums = np.sort(allData.loc[slice_of_life & vidtype_filter, DataTags.VIDEO_ID].unique()).tolist()

                    # Make data storage structures for each of our query location lists for checking which query points went into our analysis.
                    if not allData.loc[slice_of_life & qloc_filter].empty:
                        if DataTags.QUERYLOC not in allData.columns:
                            allData.loc[slice_of_life & qloc_filter, DataTags.QUERYLOC] = ""

                        query_loc_names = allData.loc[slice_of_life & qloc_filter, DataTags.QUERYLOC].unique().tolist()
                        for q, query_loc_name in enumerate(query_loc_names):
                            if len(query_loc_name) == 0:
                                query_loc_names[q] = " "
                            else:
                                query_loc_names[q] = query_loc_name.strip('_- ')
                    else:
                        query_loc_names = []


                    first = True
                    # If we detected query locations, then initialize this folder and mode with their metadata.
                    all_query_status[mode][folder] = [pd.DataFrame(columns=data_vidnums) for i in range((slice_of_life & qloc_filter).sum())]
                    auto_selected_values = pd.DataFrame(columns=query_loc_names)

                    pb["maximum"] = len(data_vidnums)+1
                    # Load each dataset (delineated by different video numbers), normalize it, standardize it, etc.
                    for v, vidnum in enumerate(data_vidnums):

                        pb["value"] = v
                        # Actually load the dataset, and all its metadata.
                        pre_filter = (allData[PreAnalysisPipeline.GROUP_BY] == group) & \
                                     (allData[DataTags.MODALITY] == mode)
                        dataset, new_entries = initialize_and_load_dataset(folder, vidnum, pre_filter, start_timestamp, allData,
                                                                                         analysis_dat_format, stage=Stages.ANALYSIS)

                        if dataset is not None:
                            # If we have new entries, and any of them are query locations, add them to our database.
                            # For now, we won't add new items to that database that are not query locations.
                            if len(new_entries) > 0 and \
                                    not new_entries[new_entries[DataFormatType.FORMAT_TYPE] == DataFormatType.QUERYLOC].empty:

                                for ind, newbie in new_entries[new_entries[DataFormatType.FORMAT_TYPE] == DataFormatType.QUERYLOC].iterrows():
                                    query_loc_names.append(newbie[AcquisiTags.DATA_PATH].name)
                                    all_query_status[mode][folder].append(pd.DataFrame(columns=data_vidnums))

                                # Update the database.
                                allData = pd.concat([allData, new_entries], ignore_index=True)

                                # Update the filters.
                                group_filter = allData[PreAnalysisPipeline.GROUP_BY] == group
                                mode_filter = allData[DataTags.MODALITY] == mode
                                folder_filter = allData[AcquisiTags.BASE_PATH] == folder
                                refim_filter = allData[DataFormatType.FORMAT_TYPE] == DataFormatType.IMAGE
                                qloc_filter = allData[DataFormatType.FORMAT_TYPE] == DataFormatType.QUERYLOC
                                vidtype_filter = allData[DataFormatType.FORMAT_TYPE] == DataFormatType.VIDEO


                        else:
                            for q in range(len(all_query_status[mode][folder])):
                                all_query_status[mode][folder][q].loc[:, vidnum] = "Dataset Failed To Load"
                            warnings.warn("Video number " + str(vidnum) + ": Dataset Failed To Load")

                        # Perform analyses on each query location set for each stimulus dataset.
                        for q in range(len(dataset.query_loc)):
                            pb_label["text"] = "Processing query locs \"" + query_loc_names[q] + "\" in dataset #" + str(vidnum) + " from the " + str(
                                                mode) + " modality in group " + str(group) + " and folder " + folder.name + "..."
                            pb.update()
                            pb_label.update()
                            print(Fore.WHITE +"Processing query locs \"" + str(dataset.metadata.get(AcquisiTags.QUERYLOC_PATH,[Path()])[q].name) +
                                  "\" in dataset #" + str(vidnum) + " from the " + str(mode) + " modality in group "
                                  + str(group) + " and folder " + folder.name + "...")

                            '''
                            *** This section is where we do dataset summary. ***
                            '''
                            (dataset.iORG_signals[q],
                             dataset.summarized_iORGs[q],
                             dataset.query_status[q],
                             dataset.query_loc[q],
                             auto_detect_vals) = extract_n_refine_iorg_signals(dataset, analysis_dat_format,#
                                                                                   query_loc=dataset.query_loc[q],
                                                                                   query_loc_name=query_loc_names[q],
                                                                                   thread_pool=the_pool)


                            # If this is the first time a video of this mode and this folder is loaded, then initialize the query status dataframe
                            # Such that each row corresponds to the original coordinate locations based on the reference image.
                            # Also add to our auto-detection tracking dataframe.
                            if first:
                                # The below maps each query loc (some coordinates) to a tuple, then forms those tuples into a list.
                                all_query_status[mode][folder][q] = all_query_status[mode][folder][q].reindex(pd.MultiIndex.from_tuples(list(map(tuple, dataset.query_loc[q]))), fill_value="Included")

                                auto_detect_vals = pd.DataFrame.from_dict(auto_detect_vals, orient="index", columns=[query_loc_names[q]])
                                auto_selected_values = auto_selected_values.combine_first(auto_detect_vals)

                        first = False

                        # Once we've extracted the iORG signals, remove the video and mask data as it's likely to have a large memory footprint.
                        dataset.clear_video_data()

                    pb["value"] = len(data_vidnums)
                    pb_label["text"] = "Processing query locs ... Done!"
                    pb.update()
                    pb_label.update()

                    has_stim = allData.loc[:, AcquisiTags.STIM_PRESENT].astype(bool)
                    slice_of_life = group_filter & folder_filter & mode_filter
                    stim_datasets = allData.loc[slice_of_life & vidtype_filter & has_stim, AcquisiTags.DATASET].tolist()
                    control_datasets = allData.loc[group_filter & mode_filter & vidtype_filter & ~has_stim, AcquisiTags.DATASET].tolist()
                    stim_data_vidnums = np.sort(allData.loc[slice_of_life & vidtype_filter & has_stim, DataTags.VIDEO_ID].unique()).tolist()
                    control_data_vidnums = np.sort(allData.loc[group_filter & mode_filter & vidtype_filter & ~has_stim, DataTags.VIDEO_ID].unique()).tolist()


                    if not stim_datasets:
                        continue

                    result_cols = pd.MultiIndex.from_product([query_loc_names, list(MetricTags)])
                    pop_iORG_result_datframe = pd.DataFrame(index=stim_data_vidnums, columns=result_cols)

                    pb["maximum"] = len(stim_data_vidnums)+1

                    # Determine if all stimulus data in this folder and mode has the same form and contents;
                    # if so, we can just process the control data *one* time, saving a lot of time.
                    # ALSO, we can perform an individual iORG analysis by combining a cells' iORGs across acquisitions
                    max_frmstamp = -1
                    all_locs = None
                    all_timestamps = None
                    first_run = True
                    uniform_datasets = True
                    for d, dataset in enumerate(stim_datasets):
                        locs = dataset.query_loc.copy()
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
                                del the_locs
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
                    control_pop_iORG_summary_pooled = [None] * len(all_locs)
                    control_framestamps = [None] * len(all_locs)

                    if uniform_datasets:
                        stim_iORG_signals = [None] * len(all_locs)


                    # Load each dataset (delineated by different video numbers), and process it relative to the control data.
                    for v, stim_vidnum in enumerate(stim_data_vidnums):


                        vidid_filter = (allData[DataTags.VIDEO_ID] == stim_vidnum)

                        # Get the reference images and query locations and this video number, only for the mode and folder we want.
                        slice_of_life = group_filter & folder_filter & mode_filter & vidid_filter

                        # Grab the stim dataset associated with this video number.
                        stim_dataset = allData.loc[slice_of_life & vidtype_filter & has_stim, AcquisiTags.DATASET].values[0]
                        control_query_status = [pd.DataFrame(columns=control_data_vidnums) for i in range(len(stim_dataset.iORG_signals))]

                        if not uniform_datasets or first_run:

                            # Some parameters require us to treat control datasets differently than stimulus datasets.
                            # For example, when using xor segmentation, we want to use the exact same area as the stimulus query points
                            # without doing xor again. So, we adjust them for control analysis.
                            control_dat_format_params = analysis_dat_format.copy()
                            if control_dat_format_params.get(Analysis.PARAMS).get(SegmentParams.NAME, dict()):
                                if control_dat_format_params.get(Analysis.PARAMS)[SegmentParams.NAME].get(SegmentParams.SHAPE) == "xor":
                                    control_dat_format_params.get(Analysis.PARAMS)[SegmentParams.NAME][SegmentParams.SHAPE] = "disk"
                                    control_dat_format_params.get(Analysis.PARAMS)[SegmentParams.NAME][SegmentParams.RADIUS] = 1

                            # Initialize these, even if we don't have them, so that they can be correctly handled for being
                            # None later.
                            control_iORG_signals = [None] * len(stim_dataset.query_loc)
                            control_iORG_summary = [None] * len(stim_dataset.query_loc)
                            control_pop_iORG_summary = [None] * len(stim_dataset.query_loc)
                            control_pop_iORG_summary_pooled = [None] * len(stim_dataset.query_loc)
                            control_framestamps = [None] * len(stim_dataset.query_loc)
                            control_framestamps_pooled = [None] * len(stim_dataset.query_loc)

                            first_run = False
                            # Only do the below if we have actual control datasets.
                            if control_datasets:
                                pb_label["text"] = "Processing query files in control datasets for stimulus video " + str(
                                    stim_vidnum) + " from the " + str(mode) + " modality in group " + str(
                                    group) + " and folder " + folder.name + "..."
                                pb.update()
                                pb_label.update()
                                print(Fore.GREEN+"Processing query files in control datasets for stim video " + str(
                                    stim_vidnum) + " from the " + str(mode) + " modality in group " + str(
                                    group) + " and folder " + folder.name + "...")

                                # Prep our control datasets for refilling
                                for cd, control_data in enumerate(control_datasets):
                                    control_data.iORG_signals = [None] * len(stim_dataset.query_loc)
                                    control_data.summarized_iORGs = [None] * len(stim_dataset.query_loc)
                                    control_data.query_loc = [None] * len(stim_dataset.query_loc)

                                # After we've processed all the control data with the parameters of the stimulus data, combine it
                                for q in range(len(stim_dataset.query_loc)):
                                    control_pop_iORG_summaries = np.full((len(control_datasets), max_frmstamp + 1), np.nan)
                                    control_pop_iORG_N = np.full((len(control_datasets), max_frmstamp + 1), np.nan)
                                    control_iORG_sigs = np.full((len(control_datasets), stim_dataset.query_loc[q].shape[0], max_frmstamp + 1), np.nan)

                                    for cd, control_data in enumerate(control_datasets):

                                        # Use the control data, but the query locations and stimulus info from the stimulus data.
                                        (control_data.iORG_signals[q],
                                         control_data.summarized_iORGs[q],
                                         control_query_stat,
                                         control_data.query_loc[q],
                                         _) = extract_n_refine_iorg_signals(control_data,
                                                                                                    control_dat_format_params,
                                                                                                    query_loc=stim_dataset.query_loc[q],
                                                                                                    query_loc_name=query_loc_names[q],
                                                                                                    stimtrain_frame_stamps=stim_dataset.stimtrain_frame_stamps,
                                                                                                    thread_pool=the_pool)

                                        control_query_status[q].loc[:, cd] = control_query_stat
                                        control_pop_iORG_N[cd, control_data.framestamps] = np.sum(np.isfinite(control_data.iORG_signals[q]))
                                        control_iORG_sigs[cd, :, control_data.framestamps] = control_data.iORG_signals[q].T
                                        control_pop_iORG_summaries[cd, control_data.framestamps] = control_data.summarized_iORGs[q]

                                        # Wipe these immediately as they're not used again.
                                        control_data.iORG_signals[q] = None
                                        control_data.summarized_iORGs[q] = None

                                        # Summarize each of the cells' iORGs
                                        # *** Removed for now, may add later.
                                        # for c in range(control_iORG_signals.shape[1]):
                                        #     tot_sig = np.nansum(np.any(np.isfinite(control_iORG_signals[:, c, :]), axis=1))
                                        #     control_query_status[q].loc[c, "Num Viable iORGs"] = tot_sig
                                        #
                                        #     if tot_sig > sum_params.get(SummaryParams.INDIV_CUTOFF, 5):
                                        #         control_query_status[q].loc[c, "Viable for single-cell summary?"] = True
                                        #         control_iORG_summary[q][c, :], _ = summarize_iORG_signals(control_iORG_signals[:, c, :],
                                        #                                                                np.arange(max_frmstamp + 1),
                                        #                                                                summary_method=sum_method,
                                        #                                                                window_size=sum_window)
                                    if debug_params.get(DebugParams.PLOT_INDIV_STANDARDIZED_ORGS, False):
                                        control_iORG_signals[q] = control_iORG_sigs
                                    else:
                                        control_iORG_signals[q] = None

                                    control_pop_iORG_summary[q] = control_pop_iORG_summaries
                                    control_framestamps[q] = []
                                    for r in range(control_pop_iORG_summary[q].shape[0]):
                                        control_framestamps[q].append(np.flatnonzero(np.isfinite(control_pop_iORG_summaries[r, :])))

                                    with warnings.catch_warnings():
                                        warnings.filterwarnings(action="ignore", message="invalid value encountered in divide")
                                        control_pop_iORG_summary_pooled[q] = np.nansum(control_pop_iORG_N * control_pop_iORG_summaries,
                                                                                       axis=0) / np.nansum(control_pop_iORG_N, axis=0)
                                    control_framestamps_pooled[q] = np.flatnonzero(np.isfinite(control_pop_iORG_summary_pooled[q]))


                                    # First write the control data to a file.
                                    control_query_status[q].to_csv(result_path.joinpath(str(subject_IDs[0]) +"_"+folder.name +  "_" + str(mode) + "_query_loc_status_" +
                                                                query_loc_names[q] + "coords_controldata_"+ start_timestamp+".csv"))
                                del control_query_stat
                                del control_iORG_sigs
                                del control_query_status
                                del control_pop_iORG_N
                                del control_pop_iORG_summaries
                                gc.collect()

                        ''' *** Population iORG analyses here *** '''
                        for q in range(len(stim_dataset.query_loc)):
                            with warnings.catch_warnings():
                                warnings.filterwarnings(action="ignore", message="indexing past lexsort depth may impact performance.")
                                all_query_status[mode][folder][q].loc[:, stim_vidnum] = stim_dataset.query_status[q]

                            stim_pop_summary = np.full((max_frmstamp + 1,), np.nan)
                            stim_pop_summary[stim_dataset.framestamps] = stim_dataset.summarized_iORGs[q]
                            stim_framestamps = np.arange(max_frmstamp + 1)

                            if sum_control == "subtraction" and control_datasets:
                                stim_dataset.summarized_iORGs[q] = stim_pop_summary - control_pop_iORG_summary_pooled[q]
                            elif sum_control == "division" and control_datasets:
                                stim_dataset.summarized_iORGs[q] = stim_pop_summary / control_pop_iORG_summary_pooled[q]
                            else:
                                stim_dataset.summarized_iORGs[q] = stim_pop_summary

                            ''' This section is for display of the above iORG summaries, as specified by the user. '''
                            # This shows all summaries overlapping.
                            if pop_overlap_params:

                                overlap_label = "Query file " + query_loc_names[q] + ": summarized using "+ sum_method+ " of " +mode +" iORGs in "+folder.name
                                display_dict[str(subject_IDs[0]) +"_"+folder.name + "_" + mode+"_pop_iORG_"+ sum_method +"_overlapping_"+query_loc_names[q]+"coords_"+ start_timestamp] = overlap_label

                                display_iORG_pop_summary(stim_framestamps, stim_pop_summary, stim_dataset.summarized_iORGs[q], stim_vidnum,
                                                         control_framestamps[q], control_pop_iORG_summary[q], control_data_vidnums,
                                                         control_framestamps_pooled[q], control_pop_iORG_summary_pooled[q],
                                                         stim_dataset.stimtrain_frame_stamps,
                                                         stim_dataset.framerate, sum_method, sum_control, overlap_label, pop_overlap_params)
                                plt.show(block=False)

                            # This shows all summaries in temporal sequence.
                            if pop_seq_params:
                                num_in_seq = pop_seq_params.get(DisplayParams.NUM_IN_SEQ, 0)
                                ax_params = pop_seq_params.get(DisplayParams.AXES, dict())
                                xlimits = (ax_params.get(DisplayParams.XMIN, None), ax_params.get(DisplayParams.XMAX, None))
                                ylimits = (ax_params.get(DisplayParams.YMIN, None), ax_params.get(DisplayParams.YMAX, None))

                                vidnum_seq = np.array(stim_data_vidnums).astype(np.int32)
                                vidnum_seq -= vidnum_seq[0]
                                # Go through all our video numbers until they're in a sequence we can work with.
                                while np.any(vidnum_seq>num_in_seq):
                                    vidnum_seq[vidnum_seq>num_in_seq] -= np.amin(vidnum_seq[vidnum_seq>num_in_seq])

                                seq_row = int(np.ceil(num_in_seq/5))

                                if pop_seq_params.get(DisplayParams.DISP_STIMULUS, True):
                                    seq_stim_label = "Query file " + query_loc_names[q] + ": Stimulus iORG temporal sequence of " +mode +" iORGs in "+folder.name
                                    display_dict[str(subject_IDs[0]) +"_"+folder.name + "_" + mode + "_pop_iORG_" + sum_method + "_sequential_stim_only_" + query_loc_names[q] + "coords_" + start_timestamp] = seq_stim_label

                                    display_iORG_pop_summary_seq(stim_framestamps, stim_pop_summary, vidnum_seq[v], stim_dataset.stimtrain_frame_stamps,
                                                                 stim_dataset.framerate, sum_method, seq_stim_label,
                                                                 pop_seq_params)
                                    plt.show(block=False)

                                if pop_seq_params.get(DisplayParams.DISP_RELATIVE, True):
                                    seq_rel_label = "Query file " + query_loc_names[q] + "Stimulus relative to control iORG via " + sum_control +" temporal sequence"
                                    display_dict[str(subject_IDs[0]) +"_"+folder.name +  "_" + mode + "_pop_iORG_" + sum_method + "_sequential_relative_" + query_loc_names[q] + "coords_"+start_timestamp] = seq_rel_label

                                    display_iORG_pop_summary_seq(stim_framestamps, stim_dataset.summarized_iORGs[q], vidnum_seq[v], stim_dataset.stimtrain_frame_stamps,
                                                                 stim_dataset.framerate, sum_method, seq_rel_label,
                                                                 pop_seq_params)
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
                                metrics_poststim = stim_dataset.stimtrain_frame_stamps[0] + metrics_poststim

                            # Make the list of indices that should correspond to pre and post stimulus
                            metrics_prestim = np.arange(start=metrics_prestim[0], stop=metrics_prestim[1], step=1, dtype=int)
                            metrics_poststim = np.arange(start=metrics_poststim[0], stop=metrics_poststim[1], step=1, dtype=int)

                            amplitude, amp_implicit_time, halfamp_implicit_time, aur, recovery = iORG_signal_metrics(stim_dataset.summarized_iORGs[q][stim_dataset.framestamps],
                                                                                                                     stim_dataset.framestamps, stim_dataset.framerate,
                                                                                                                     metrics_prestim, metrics_poststim, the_pool)

                            for metric in metrics_type:
                                if metric == "aur":
                                    pop_iORG_result_datframe.loc[stim_vidnum, (query_loc_names[q], MetricTags.AUR)] = aur[0]
                                elif metric == "amp":
                                    pop_iORG_result_datframe.loc[stim_vidnum, (query_loc_names[q], MetricTags.AMPLITUDE)] = amplitude[0]
                                elif metric == "logamp":
                                    pop_iORG_result_datframe.loc[stim_vidnum, (query_loc_names[q], MetricTags.LOG_AMPLITUDE)] = np.log(amplitude[0])
                                elif metric == "amp_imp_time":
                                    pop_iORG_result_datframe.loc[stim_vidnum, (query_loc_names[q], MetricTags.AMP_IMPLICIT_TIME)] = amp_implicit_time[0]
                                elif metric == "halfamp_imp_time":
                                    pop_iORG_result_datframe.loc[stim_vidnum, (query_loc_names[q], MetricTags.HALFAMP_IMPLICIT_TIME)] = halfamp_implicit_time[0]
                                elif metric == "rec_amp":
                                    pop_iORG_result_datframe.loc[stim_vidnum, (query_loc_names[q], MetricTags.RECOVERY_PERCENT)] = recovery[0]


                    ''' *** Average all stimulus population iORGs, and do individual cone analyses *** '''
                    indiv_iORG_result =[None] * len(all_locs)
                    for q in range(len(all_locs)):

                        indiv_iORG_result[q] = pd.DataFrame(index=all_query_status[mode][folder][q].index, columns=list(MetricTags))
                        #indiv_iORG_result[q].sort_index(inplace=True)
                        stim_pop_iORG_summaries = np.full((len(stim_datasets), max_frmstamp + 1), np.nan)
                        stim_pop_iORG_N = np.full((len(stim_datasets), max_frmstamp + 1), np.nan)
                        stimtrain = [None] * len(stim_datasets)

                        pooled_framerate = np.full((len(stim_datasets),), np.nan)
                        finite_iORG_frmstmp = np.arange(max_frmstamp + 1)


                        if uniform_datasets:
                            stim_iORG_signals[q] = np.full((len(stim_datasets), stim_datasets[0].query_loc[q].shape[0], max_frmstamp + 1), np.nan)
                            stim_iORG_summary[q] = np.full((stim_datasets[0].query_loc[q].shape[0], max_frmstamp + 1), np.nan)

                        for d, stim_dataset in enumerate(stim_datasets):
                            pooled_framerate[d] = stim_dataset.framerate
                            stimtrain[d] = stim_dataset.stimtrain_frame_stamps
                            stim_pop_iORG_N[d, stim_dataset.framestamps] = np.nansum(np.isfinite(stim_dataset.iORG_signals[q]), axis=0)
                            stim_pop_iORG_summaries[d, :] = stim_dataset.summarized_iORGs[q]

                            ''' Clean up old data so that we minimize our memory footprint '''
                            stim_dataset.summarized_iORGs[q] = []
                            # If all stimulus datasets are uniform,
                            # we can also summarize individual iORGs by combining a cells' iORGs across acquisitions.
                            if uniform_datasets:
                                stim_iORG_signals[q][d, :, stim_dataset.framestamps] = stim_dataset.iORG_signals[q].T

                                ''' Clean up old data so that we minimize our memory footprint '''
                                stim_dataset.iORG_signals[q] = []
                        gc.collect()
                        pooled_framerate = np.unique(pooled_framerate)
                        if len(pooled_framerate) != 1:
                            warnings.warn("The framerate of the iORGs analyzed in "+folder.name + " is inconsistent: ("+str(pooled_framerate)+"). Pooled results may be incorrect.")
                            pooled_framerate = pooled_framerate[0]

                        stimtrain = np.unique(pd.DataFrame(stimtrain).values.astype(np.int32), axis=0)
                        if stimtrain.shape[0] != 1:
                            warnings.warn("The stimulus frame train of the iORGs analyzed in " + folder.name + " is inconsistent! Pooled results may be incorrect.")

                        stimtrain = stimtrain[0]

                        # Debug - to look at individual cell raw traces.
                        if debug_params.get(DebugParams.PLOT_INDIV_STANDARDIZED_ORGS, False) or \
                           debug_params.get(DebugParams.OUTPUT_INDIV_STANDARDIZED_ORGS, False):

                            if debug_params.get(DebugParams.PLOT_INDIV_STANDARDIZED_ORGS, "all") == "all":
                                cell_inds = range(stim_iORG_signals[q].shape[1])
                            else:
                                cell_inds = debug_params.get(DebugParams.PLOT_INDIV_STANDARDIZED_ORGS, 0)

                            for c in cell_inds:
                                overlap_label = "Debug: View stdz " + mode + " iORGs in " + folder.name + " signals for cell: "+str(c)+ " at: " + str(stim_datasets[0].query_loc[q][c,:])
                                if len(cell_inds) < 10:
                                    display_dict[query_loc_names[q]+"_stdz_" + mode + "_iORGs_" + folder.name + "_"+str(c)+ "_at_" + str(stim_datasets[0].query_loc[q][c,:])] = overlap_label

                                if np.any(np.isfinite(stim_iORG_signals[q][:, c, :])):
                                    if debug_params.get(DebugParams.OUTPUT_INDIV_STANDARDIZED_ORGS, False):
                                        sigout = pd.DataFrame(stim_iORG_signals[q][:, c, :], index=stim_data_vidnums)
                                        sigout.to_csv(result_path.joinpath(overlap_label+".csv"))
                                        del sigout
                                        gc.collect()

                                    if control_iORG_signals[q]:
                                        display_iORGs(finite_iORG_frmstmp, stim_iORG_signals[q][:, c, :], query_loc_names[q],
                                                      finite_iORG_frmstmp, control_iORG_signals[q][:, c, :], control_data_vidnums,
                                                      stim_datasets[0].avg_image_data, stim_datasets[0].query_loc[q][c,:],
                                                      stim_delivery_frms = stimtrain, framerate = pooled_framerate,
                                                      figure_label = overlap_label, params = debug_params)
                                    else:
                                        display_iORGs(finite_iORG_frmstmp, stim_iORG_signals[q][:, c, :], query_loc_names[q],
                                                      image=stim_datasets[0].avg_image_data, cell_loc=stim_datasets[0].query_loc[q][c,:],
                                                      stim_delivery_frms = stimtrain, framerate = pooled_framerate,
                                                      figure_label = overlap_label, params = debug_params)

                                    if len(cell_inds) > 10:
                                        plt.show(block=False)
                                        plt.waitforbuttonpress()
                                        plt.close()

                        ''' *** Pool the summarized population iORGs *** '''
                        with warnings.catch_warnings():
                            warnings.filterwarnings(action="ignore", message="invalid value encountered in divide")
                            nandata = np.all(np.isnan(stim_pop_iORG_summaries), axis=0)
                            stim_pop_iORG_summary[q] = np.nansum(stim_pop_iORG_N * stim_pop_iORG_summaries,axis=0) / np.nansum(stim_pop_iORG_N, axis=0)
                            stim_pop_iORG_summary[q][nandata] = np.nan

                        metrics_prestim = np.array(metrics.get(SummaryParams.PRESTIM, [-1, 0]), dtype=int)
                        metrics_poststim = np.array(metrics.get(SummaryParams.POSTSTIM, [0, 1]), dtype=int)
                        if metrics_units == "time":
                            metrics_prestim = np.round(metrics_prestim * pooled_framerate)
                            metrics_poststim = np.round(metrics_poststim * pooled_framerate)
                        else:  # if units == "frames":
                            metrics_prestim = np.round(metrics_prestim)
                            metrics_poststim = np.round(metrics_poststim)

                        if metrics_measured_to == "stim-relative":
                            metrics_prestim = stimtrain[0] + metrics_prestim
                            metrics_poststim = stimtrain[0] + metrics_poststim

                        # Make the list of indices that should correspond to pre and post stimulus
                        metrics_prestim = np.arange(start=metrics_prestim[0], stop=metrics_prestim[1], step=1,
                                                    dtype=int)
                        metrics_poststim = np.arange(start=metrics_poststim[0], stop=metrics_poststim[1], step=1,
                                                     dtype=int)


                        amplitude, amp_implicit_time, halfamp_implicit_time, aur, recovery = iORG_signal_metrics(stim_pop_iORG_summary[q], finite_iORG_frmstmp,
                                                                                                                 pooled_framerate,
                                                                                                                 metrics_prestim, metrics_poststim, the_pool)

                        for metric in metrics_type:
                            if metric == "aur":
                                pop_iORG_result_datframe.loc["Pooled", (query_loc_names[q], MetricTags.AUR)] = aur[0]
                            elif metric == "amp":
                                pop_iORG_result_datframe.loc["Pooled", (query_loc_names[q], MetricTags.AMPLITUDE)] = amplitude[0]
                            elif metric == "logamp":
                                pop_iORG_result_datframe.loc["Pooled", (query_loc_names[q], MetricTags.LOG_AMPLITUDE)] = np.log(amplitude[0])
                            elif metric == "amp_imp_time":
                                pop_iORG_result_datframe.loc["Pooled", (query_loc_names[q], MetricTags.AMP_IMPLICIT_TIME)] = amp_implicit_time[0]
                            elif metric == "halfamp_imp_time":
                                pop_iORG_result_datframe.loc["Pooled", (query_loc_names[q], MetricTags.HALFAMP_IMPLICIT_TIME)] = halfamp_implicit_time[0]
                            elif metric == "rec_amp":
                                pop_iORG_result_datframe.loc["Pooled", (query_loc_names[q], MetricTags.RECOVERY_PERCENT)] = recovery[0]

                        ''' *** Display the pooled population data *** '''
                        if pop_overlap_params.get(DisplayParams.DISP_POOLED, False):
                            overlap_label = "Pooled data summarized with " + sum_method + " of " + mode + " iORGs in " + folder.name
                            display_dict[str(subject_IDs[0]) +"_"+folder.name +  "_" + mode + "_pooled_pop_iORG_" + sum_method + "_overlapping_"+ start_timestamp] = overlap_label

                            display_iORG_pop_summary(np.arange(max_frmstamp + 1), stim_pop_iORG_summary[q], stim_vidnum=query_loc_names[q],
                                                     stim_delivery_frms=stimtrain, framerate=pooled_framerate, sum_method=sum_method, sum_control=sum_control,
                                                     figure_label=overlap_label, params=pop_overlap_params)
                            plt.show(block=False)
                            if sum_control !=  "none":
                                plt.title("Pooled "+ sum_method.upper() +" iORGs relative\nto control iORG via " + sum_control)
                            else:
                                plt.title("Pooled " + sum_method.upper() + " iORGs. (No control)")

                        # If we have a uniform dataset, summarize each cell's iORG too.
                        ''' *** Individual iORG analyses start here *** '''
                        if uniform_datasets:
                            all_frmstmp = np.arange(max_frmstamp + 1)

                            if indiv_overlap_params:
                                if indiv_overlap_params.get(DisplayParams.DISP_STIMULUS, False) or \
                                        indiv_overlap_params.get(DisplayParams.DISP_CONTROL, False) or \
                                        indiv_overlap_params.get(DisplayParams.DISP_RELATIVE, False):
                                    overlap_label = "Individual-Cell iORGs summarized with " + sum_method + " of " + mode + " iORGs in " + folder.name
                                    display_dict[str(subject_IDs[0]) +"_"+folder.name +  "_" + mode + "_indiv_iORG_" + sum_method + "_overlapping_"+ start_timestamp] = overlap_label

                            all_tot_sig = np.nansum(np.any(np.isfinite(stim_iORG_signals[q]), axis=2), axis=0)
                            viable_sig = all_tot_sig >= sum_params.get(SummaryParams.INDIV_CUTOFF, 5)

                            with warnings.catch_warnings():
                                warnings.filterwarnings(action="ignore", message="indexing past lexsort depth may impact performance.")

                                all_query_status[mode][folder][q].loc[:, "Num Viable iORGs"] = all_tot_sig
                                all_query_status[mode][folder][q].loc[:, "Viable for single-cell summary?"] = viable_sig

                            # If they're not viable, nan them.
                            stim_iORG_signals[q][:, np.where(~viable_sig), :] = np.nan

                            allcell_iORG_summary, _ = summarize_iORG_signals(stim_iORG_signals[q], all_frmstmp,
                                                                          summary_method=sum_method,
                                                                          window_size=sum_window,
                                                                          pool=the_pool)

                            # Calculate the relativized individual cell iORGs
                            if sum_control == "subtraction" and control_datasets:
                                stim_iORG_summary[q] = allcell_iORG_summary - np.repeat(control_pop_iORG_summary_pooled[q][None, :], allcell_iORG_summary.shape[0], axis=0)
                            elif sum_control == "division" and control_datasets:
                                stim_iORG_summary[q] = allcell_iORG_summary / np.repeat(control_pop_iORG_summary_pooled[q][None, :], allcell_iORG_summary.shape[0], axis=0)
                            else:
                                stim_iORG_summary[q] = allcell_iORG_summary

                            ''' *** Display individual iORG summaries  *** '''
                            if indiv_overlap_params:
                                if indiv_overlap_params.get(DisplayParams.DISP_STIMULUS, False) or \
                                        indiv_overlap_params.get(DisplayParams.DISP_CONTROL, False) or \
                                        indiv_overlap_params.get(DisplayParams.DISP_RELATIVE, False):

                                    for c in range(stim_iORG_signals[q].shape[1]):
                                        display_iORG_pop_summary(all_frmstmp, allcell_iORG_summary[c, :], stim_iORG_summary[q][c, :], None,
                                                             all_frmstmp, control_pop_iORG_summary_pooled[q],
                                                             stim_delivery_frms=stimtrain,framerate=pooled_framerate, sum_method=sum_method, sum_control=sum_control,
                                                             figure_label=overlap_label, params=indiv_overlap_params)

                            # amplitude, amp_implicit_time, halfamp_implicit_time, aur, recovery
                            res = iORG_signal_metrics(stim_iORG_summary[q], all_frmstmp, pooled_framerate, metrics_prestim, metrics_poststim, the_pool)

                            with warnings.catch_warnings():
                                warnings.filterwarnings(action="ignore", message="indexing past lexsort depth may impact performance.")

                                for m, metric in enumerate(res):
                                    match m:
                                        case 0: # amplitude
                                            indiv_iORG_result[q].loc[:, MetricTags.AMPLITUDE] = metric
                                            indiv_iORG_result[q].loc[:, MetricTags.LOG_AMPLITUDE] = np.log(metric)
                                        case 1: # amp_implicit_time
                                            indiv_iORG_result[q].loc[:, MetricTags.AMP_IMPLICIT_TIME] = metric
                                        case 2: # halfamp_implicit_time
                                            indiv_iORG_result[q].loc[:, MetricTags.HALFAMP_IMPLICIT_TIME] = metric
                                        case 3: # aur
                                            indiv_iORG_result[q].loc[:, MetricTags.AUR] = metric
                                        case 4: # recovery
                                            indiv_iORG_result[q].loc[:, MetricTags.RECOVERY_PERCENT] = metric

                            if indiv_overlap_params:
                                plt.show(block=False)
                            indiv_respath = result_path.joinpath(str(subject_IDs[0]) +"_"+folder.name + "_" + mode + "_indiv_summary_metrics_" + query_loc_names[q] + "coords_" +start_timestamp+".csv")

                            if indiv_summary.get(DisplayParams.HISTOGRAM):

                                overlap_label = "Individual-Cell iORGs metric histograms\nfrom " + mode + " iORGs in " + folder.name
                                display_dict[str(subject_IDs[0]) +"_"+folder.name +  "_" + mode + "_indiv_iORG_" + sum_method + "_metric_histograms_"+ start_timestamp] = overlap_label

                                display_iORG_summary_histogram(indiv_iORG_result[q], metrics_tags, False, query_loc_names[q],
                                                               overlap_label, indiv_summary)
                                plt.suptitle(overlap_label)


                            if indiv_summary.get(DisplayParams.CUMULATIVE_HISTOGRAM):
                                overlap_label = "Individual-Cell iORGs metric cumulative histograms\nfrom " + mode + " iORGs in " + folder.name
                                display_dict[str(subject_IDs[0]) +"_"+folder.name + "_" + mode + "_indiv_iORG_" + sum_method + "_metric_cumul_histograms_"+ start_timestamp] = overlap_label

                                display_iORG_summary_histogram(indiv_iORG_result[q], metrics_tags, True, query_loc_names[q],
                                                               overlap_label, indiv_summary)
                                plt.suptitle(overlap_label)

                            if indiv_summary.get(DisplayParams.MAP_OVERLAY):
                                ax_params = indiv_summary.get(DisplayParams.AXES, dict())

                                for metric in metrics_tags:
                                    if indiv_iORG_result[q].loc[:, metric].count() != 0:
                                        label = "Individual iORG "+metric+" from " + mode + "\nusing query locations " + query_loc_names[q] + " in " + folder.name
                                        display_dict[str(subject_IDs[0]) +"_"+folder.name + "_" + mode + "_indiv_iORG_" + sum_method + "_" + metric + "_overlay_" + query_loc_names[q] +"_"+ start_timestamp] = label

                                        refim = allData.loc[group_filter & folder_filter & mode_filter & refim_filter, AcquisiTags.DATA_PATH].values[0]

                                        metric_res = indiv_iORG_result[q].loc[:, metric].values.astype(float)
                                        coords = np.array(indiv_iORG_result[q].loc[:, metric].index.to_list())

                                        display_iORG_summary_overlay(metric_res, coords, cv2.imread(refim, cv2.IMREAD_GRAYSCALE),
                                                                     metric, label, indiv_summary)

                            if indiv_summary.get(DisplayParams.ORG_VIDEO):
                                ax_params = indiv_summary.get(DisplayParams.AXES, dict())


                                starting = ax_params.get(DisplayParams.CMIN, np.nanpercentile(stim_iORG_summary[q].flatten(), 2.5))
                                stopping = ax_params.get(DisplayParams.CMAX, np.nanpercentile(stim_iORG_summary[q].flatten(), 99))


                                normmap = mpl.colors.Normalize(vmin=starting, vmax=stopping, clip=True)
                                mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap(ax_params.get(DisplayParams.CMAP, "viridis")),
                                                               norm=normmap)

                                video_profiles = np.reshape(stim_iORG_summary[q], (stim_datasets[0].avg_image_data.shape[0],
                                                                                         stim_datasets[0].avg_image_data.shape[1],
                                                                                         -1), copy=True)

                                video_profiles[np.isnan(video_profiles)] = starting
                                save_video(result_path.joinpath(str(subject_IDs[0]) + "_" + mode +"_pooled_pixelpop_iORG_" + sum_method + "_" +folder.name +"_"+ start_timestamp + ".avi"),
                                           video_profiles, pooled_framerate.item(),
                                           scalar_mapper=mapper)

                            tryagain = True
                            while tryagain:
                                try:
                                    indiv_iORG_result[q].sort_index(inplace=True)
                                    indiv_iORG_result[q].to_csv(indiv_respath)
                                    tryagain = False
                                except PermissionError:
                                    tryagain = messagebox.askyesno(title="File: " + str(indiv_respath) + " is unable to be written.",
                                        message="The result file may be open. Close the file, then try to write again?")

                            # figure_label = "Debug: Included cells from "+ mode + " in query location: "+ query_loc_names[q] + " in " + folder.name
                            # plt.figure(figure_label)
                            # display_dict["Debug_"+mode + "_inc_cells_"+query_loc_names[q] ] = figure_label
                            # refim = group_datasets.loc[folder_mask & (this_mode & reference_images), AcquisiTags.DATA_PATH].values[0]
                            # plt.title(figure_label)
                            # plt.imshow(cv2.imread(refim, cv2.IMREAD_GRAYSCALE), cmap='gray')
                            # viability = all_query_status[mode][folder][q].loc[:, "Viable for single-cell summary?"]
                            #
                            # viable = []
                            # nonviable = []
                            # for coords, viability in viability.items():
                            #     if viability:
                            #         viable.append(coords)
                            #     else:
                            #         nonviable.append(coords)
                            #
                            # viable = np.array(viable)
                            # nonviable = np.array(nonviable)
                            # if viable.size > 0:
                            #     plt.scatter(viable[:, 0], viable[:, 1], s=7, c="c")
                            # if nonviable.size >0:
                            #     plt.scatter(nonviable[:, 0], nonviable[:, 1], s=7, c="red")
                            # plt.show(block=False)
                            # plt.waitforbuttonpress()

                        all_query_status[mode][folder][q].sort_index(inplace=True)
                        all_query_status[mode][folder][q].to_csv(result_path.joinpath(str(subject_IDs[0]) +"_"+folder.name +  "_" + mode + "_query_loc_status_" + str(folder.name) +
                                                   "_" + query_loc_names[q] + "_coords_" + start_timestamp + ".csv"))

                    auto_selected_values.to_csv(result_path.joinpath(str(subject_IDs[0]) +"_"+folder.name +  "_" + mode + "_autodetected_params_" + str(folder.name) +"_"+ start_timestamp + ".csv"))

                    respath = result_path.joinpath(str(subject_IDs[0]) +"_"+folder.name +  "_" + mode + "_pop_summary_metrics_" + start_timestamp +".csv")
                    tryagain = True
                    while tryagain:
                        try:
                            pop_iORG_result_datframe.to_csv(respath)
                            tryagain = False
                        except PermissionError:
                            tryagain=messagebox.askyesno(title="File: " + str(respath) + " is unable to be saved.",
                                                          message="The result file may be open. Close the file, then try to write again?")

                    if display_params.get(DisplayParams.PAUSE_PER_FOLDER, False):
                        plt.waitforbuttonpress()

                    # Save the figures to the result folder, if requested.
                    plt.show(block=False)
                    for fname, figname in display_dict.items():
                        plt.figure(figname)
                        sublayout = plt.gca().get_gridspec()
                        if sublayout is not None:
                            plt.gcf().set_size_inches(5*sublayout.ncols, 5*sublayout.nrows)
                        else:
                            plt.gcf().set_size_inches(5, 5)

                        plt.draw()
                        for ext in saveas_ext:
                            tryagain = True
                            while tryagain:
                                try:
                                    plt.savefig(result_path.joinpath(fname+"."+ext), dpi=300)
                                    tryagain = False
                                except PermissionError:
                                    tryagain = messagebox.askyesno(
                                        title="Figure " + str(fname+"."+ext) + " is unable to be saved.",
                                        message="The figure file may be open. Close the file, then try to write again?")

                        plt.close(figname)

                    # Save the json used to analyze this data, for auditing.
                    out_json = Path(json_fName).stem + "_ran_at_" + start_timestamp + ".json"
                    out_json = result_path.joinpath(out_json)

                    audit_json_dict = {ConfigFields.VERSION: dat_form.get(ConfigFields.VERSION, "none"),
                                       ConfigFields.DESCRIPTION: dat_form.get(ConfigFields.DESCRIPTION, "none"),
                                       PreAnalysisPipeline.NAME: preanalysis_dat_format,
                                       Analysis.NAME: analysis_dat_format}

                    with open(out_json, 'w') as f:
                        json.dump(audit_json_dict, f, indent=2)

# figure_label = "Debug: Included cells from "+ mode + " in query location: "+ query_loc_names[q] + " in " + folder.name
# plt.figure(figure_label)
# display_dict["Debug_"+mode + "_inc_cells_"+query_loc_names[q] ] = figure_label
#
# plt.title(figure_label)
# plt.imshow(dataset.avg_image_data, cmap='gray')
# viability = all_query_status[mode][folder][q].loc[:, "Viable for single-cell summary?"]
#
# viable = []
# nonviable = []
# for coords, viability in viability.items():
#     if viability:
#         viable.append(coords)
#     else:
#         nonviable.append(coords)
#
# viable = np.array(viable)
# nonviable = np.array(nonviable)
# if viable.size > 0:
#     plt.scatter(viable[:, 0], viable[:, 1], s=7, c="c")
# if nonviable.size >0:
#     plt.scatter(nonviable[:, 0], nonviable[:, 1], s=7, c="red")
# plt.show(block=False)
# plt.waitforbuttonpress()
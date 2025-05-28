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
from ocvl.function.utility.dataset import parse_file_metadata, initialize_and_load_dataset, Stages
from ocvl.function.utility.json_format_constants import PreAnalysisPipeline, MetaTags, DataFormatType, DataTags, \
    AcquisiTags, \
    NormParams, SummaryParams, ControlParams, DisplayParams, \
    MetricTags, Analysis, SegmentParams, ConfigFields, DebugParams
from ocvl.function.utility.resources import save_tiff_stack, save_video

if __name__ == "__main__":

    mpl.rcParams['lines.linewidth'] = 2.5

    dt = datetime.now()
    now_timestamp = dt.strftime("%Y%m%d_%H%M")

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
    preanalysis_dat_format = dat_form.get(PreAnalysisPipeline.NAME)
    pipeline_params = preanalysis_dat_format.get(PreAnalysisPipeline.PARAMS)
    analysis_params = analysis_dat_format.get(Analysis.PARAMS)
    display_params = analysis_dat_format.get(DisplayParams.NAME)
    modes_of_interest = analysis_params.get(PreAnalysisPipeline.MODALITIES)

    if 'IDnum' in allData:
        subject_IDs = allData['IDnum'].unique()

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
        groups = [""]  # If we don't have any groups, then just make the list an empty string.

    norm_params = analysis_params.get(NormParams.NAME, dict())
    norm_method = norm_params.get(NormParams.NORM_METHOD, "score")  # Default: Standardizes the video to a unit mean and stddev
    rescale_norm = norm_params.get(NormParams.NORM_RESCALE, True)  # Default: Rescales the data back into AU to make results easier to interpret
    res_mean = norm_params.get(NormParams.NORM_MEAN, 70)  # Default: Rescales to a mean of 70 - these values are based on "ideal" datasets
    res_stddev = norm_params.get(NormParams.NORM_STD, 35)  # Default: Rescales to a std dev of 35

    seg_params = analysis_params.get(SegmentParams.NAME, dict())
    seg_pixelwise = seg_params.get(SegmentParams.PIXELWISE, False)  # Default to NO pixelwise analyses. Otherwise, add one.

    # Snag all of our parameter dictionaries that we'll use here.
    # Default to an empty dictionary so that we can query against it with fewer if statements.
    control_params = analysis_dat_format.get(ControlParams.NAME, dict())
    control_loc = control_params.get(ControlParams.LOCATION, "implicit")
    control_folder = control_params.get(ControlParams.FOLDER_NAME, "control")

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
    output_norm_vid = debug_params.get(DebugParams.OUTPUT_NORM_VIDEO, False)


    output_folder = analysis_params.get(Analysis.OUTPUT_FOLDER)
    if output_folder is None:
        output_folder = PurePath("Results")
    else:
        output_folder = PurePath(output_folder)


    subfolder_flag = 0

    if analysis_params.get(Analysis.OUTPUT_SUBFOLDER, True): #Is output subfolder field true (ie does the user want to save to a subfolder?)
        output_subfolder_method = analysis_params.get(Analysis.OUTPUT_SUBFOLDER_METHOD) #Check subfolder naming method
        if output_subfolder_method == 'DateTime': #Only supports saving things to a subfolder with a unique timestamp currently
            output_dt_subfolder = PurePath(now_timestamp)
        else:
            output_dt_subfolder = PurePath(now_timestamp)


    with (mp.Pool(processes=mp.cpu_count() // 2) as the_pool):

        # First break things down by group, defined by the user in the config file.
        # We like to use (LocX,LocY), but this is by no means the only way.
        for group in groups:

            if group != "":
                group_datasets = allData.loc[allData[PreAnalysisPipeline.GROUP_BY] == group]
            else:
                group_datasets = allData

            group_datasets.loc[: ,AcquisiTags.STIM_PRESENT] = False
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

                if folder.name == output_folder.name:
                    continue

                folder_mask = (group_datasets[AcquisiTags.BASE_PATH] == folder)

                data_in_folder = group_datasets.loc[folder_mask]

                all_query_status[folder] = dict()

                # Load each modality
                for mode in modes_of_interest:
                    this_mode = (group_datasets[DataTags.MODALITY] == mode)
                    slice_of_life = folder_mask & this_mode

                    data_vidnums = group_datasets.loc[slice_of_life & only_vids, DataTags.VIDEO_ID].unique().tolist()

                    # Make data storage structures for each of our query location lists for checking which query points went into our analysis.
                    if not group_datasets.loc[slice_of_life & query_locations].empty:
                        query_loc_names = group_datasets.loc[slice_of_life & query_locations, DataTags.QUERYLOC].unique().tolist()
                        for q, query_loc_name in enumerate(query_loc_names):
                            if len(query_loc_name) == 0:
                                query_loc_names[q] = " "
                            else:
                                query_loc_names[q] = query_loc_name.strip('_- ')
                    else:
                        query_loc_names = []

                    first = True
                    # If we detected query locations, then initialize this folder and mode with their metadata.
                    all_query_status[folder][mode] = [pd.DataFrame(columns=data_vidnums) for i in range((slice_of_life & query_locations).sum())]

                    pb["maximum"] = len(data_vidnums)
                    # Load each dataset (delineated by different video numbers), normalize it, standardize it, etc.
                    for v, vidnum in enumerate(data_vidnums):

                        this_vid = (group_datasets[DataTags.VIDEO_ID] == vidnum)

                        # Get the reference images and query locations and this video number, only for the mode and folder mask we want.
                        slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))

                        pb["value"] = v

                        # Actually load the dataset, and all its metadata.
                        dataset = initialize_and_load_dataset(group_datasets.loc[slice_of_life], metadata_params, stage=Stages.ANALYSIS)

                        if dataset is not None:
                            # Flat field the video for analysis if desired.
                            if analysis_params.get(PreAnalysisPipeline.FLAT_FIELD, False):
                                dataset.video_data = flat_field(dataset.video_data, dataset.mask_data)

                            # Gaussian blur the data first before analysis, if requested
                            gausblur = analysis_params.get(PreAnalysisPipeline.GAUSSIAN_BLUR, 0.0)
                            if gausblur is not None and gausblur != 0.0:
                                for f in range(dataset.video_data.shape[-1]):
                                    dataset.video_data[..., f] = gaussian_filter(dataset.video_data[..., f], sigma=gausblur)


                            if output_norm_vid:
                                if analysis_params.get(Analysis.OUTPUT_SUBFOLDER, True):
                                    result_folder = folder.joinpath(output_folder, output_dt_subfolder)
                                    result_folder.mkdir(parents=True, exist_ok=True)
                                else:
                                    result_folder = folder.joinpath(output_folder)
                                    result_folder.mkdir(parents=True, exist_ok=True)

                                save_tiff_stack(result_folder.joinpath(dataset.video_path.stem+"_"+norm_method+"_norm.tif"), dataset.video_data)

                            # Normalize the video to reduce the influence of framewide intensity changes
                            dataset.video_data = norm_video(dataset.video_data, norm_method=norm_method,
                                                                rescaled=rescale_norm,
                                                                rescale_mean=res_mean, rescale_std=res_stddev)

                            group_datasets.loc[slice_of_life & only_vids, AcquisiTags.DATASET] = dataset

                            # If we didn't find an average image in our database, but were able to automagically detect or make one,
                            # Then add the automagically detected one to our database.
                            if group_datasets.loc[slice_of_life & reference_images].empty and dataset.avg_image_data is not None:
                                base_entry = group_datasets[slice_of_life & only_vids].copy()
                                base_entry.loc[base_entry.index[0], DataFormatType.FORMAT_TYPE] = DataFormatType.IMAGE
                                base_entry.loc[base_entry.index[0], AcquisiTags.DATA_PATH] = dataset.image_path
                                base_entry.loc[base_entry.index[0], AcquisiTags.DATASET] = None

                                # Update the database, and update all of our logical indices
                                group_datasets = pd.concat([group_datasets, base_entry], ignore_index=True)

                                reference_images = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.IMAGE)
                                query_locations = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.QUERYLOC)
                                only_vids = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.VIDEO)
                                this_mode = (group_datasets[DataTags.MODALITY] == mode)
                                folder_mask = (group_datasets[AcquisiTags.BASE_PATH] == folder)
                                this_vid = (group_datasets[DataTags.VIDEO_ID] == vidnum)
                                slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))


                            # Check to see if our dataset's number of query locations matches the ones we thought we found
                            # (can happen if the query location format doesn't match, but dataset was able to find a candidate)
                            if len(all_query_status[folder][mode]) < len(dataset.query_loc):
                                # If we have too few, then tack on some extra dataframes so we can track these found query locations, and add them to our database, using the dataset as a basis.
                                base_entry = group_datasets[slice_of_life & only_vids].copy()
                                base_entry.loc[0, DataFormatType.FORMAT_TYPE] = DataFormatType.QUERYLOC
                                base_entry.loc[0, AcquisiTags.DATASET] = None

                                for i in range(len(dataset.query_loc) - len(all_query_status[folder][mode])):
                                    base_entry.loc[0,AcquisiTags.DATA_PATH] = dataset.query_coord_paths[i]
                                    base_entry.loc[0,DataTags.QUERYLOC] = "Auto_Detected_" + str(i)

                                    # Update the database, and update all of our logical indices
                                    group_datasets = pd.concat([group_datasets, base_entry], ignore_index=True)

                                    reference_images = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.IMAGE)
                                    query_locations = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.QUERYLOC)
                                    only_vids = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.VIDEO)
                                    this_mode = (group_datasets[DataTags.MODALITY] == mode)
                                    folder_mask = (group_datasets[AcquisiTags.BASE_PATH] == folder)
                                    this_vid = (group_datasets[DataTags.VIDEO_ID] == vidnum)
                                    slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))

                                    all_query_status[folder][mode].append(pd.DataFrame(columns=data_vidnums))
                                    query_loc_names.append(base_entry.loc[0,DataTags.QUERYLOC])

                            if control_loc == "folder" and folder.name == control_folder:
                                # If we're in the control folder, then we're a control video- and we shouldn't extract
                                # any iORGs until later as our stimulus deliveries may vary.
                                group_datasets.loc[slice_of_life & only_vids, AcquisiTags.STIM_PRESENT] = False

                                continue
                            else:
                                group_datasets.loc[slice_of_life & only_vids, AcquisiTags.STIM_PRESENT] = len(dataset.stimtrain_frame_stamps) > 1

                                # If we can't find any query locations, or if we just want it, default to querying all pixels.
                                if (len(dataset.query_loc) == 0 or seg_pixelwise) and Path("All Pixels") not in dataset.query_coord_paths:
                                    seg_pixelwise = True  # Set this to true, if we find that query loc for this dataset is 0

                                    xm, ym = np.meshgrid(np.arange(dataset.video_data.shape[1]),
                                                         np.arange(dataset.video_data.shape[0]))

                                    xm = np.reshape(xm, (xm.size, 1))
                                    ym = np.reshape(ym, (ym.size, 1))

                                    allcoord_data = np.hstack((xm, ym))

                                    dataset.query_loc.append(allcoord_data)
                                    dataset.query_status = [np.full(locs.shape[0], "Included", dtype=object) for locs in dataset.query_loc]
                                    dataset.query_coord_paths.append(Path("All Pixels"))
                                    dataset.metadata[AcquisiTags.QUERYLOC_PATH].append(Path("All Pixels"))
                                    dataset.iORG_signals = [None] * len(dataset.query_loc)
                                    dataset.summarized_iORGs = [None] * len(dataset.query_loc)

                                    all_query_status[folder][mode].append(pd.DataFrame(columns=data_vidnums))
                                    query_loc_names.append(dataset.query_coord_paths[-1].name)

                                    base_entry = group_datasets[slice_of_life & only_vids].copy()
                                    base_entry.loc[0, DataFormatType.FORMAT_TYPE] = DataFormatType.QUERYLOC
                                    base_entry.loc[0, AcquisiTags.DATA_PATH] = dataset.query_coord_paths[-1]
                                    base_entry.loc[0, DataTags.QUERYLOC] = "All Pixels"
                                    base_entry.loc[0, AcquisiTags.DATASET] = None

                                    # Update the database, and update all of our logical indices
                                    group_datasets = pd.concat([group_datasets, base_entry], ignore_index=True)
                                    reference_images = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.IMAGE)
                                    query_locations = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.QUERYLOC)
                                    only_vids = (group_datasets[DataFormatType.FORMAT_TYPE] == DataFormatType.VIDEO)
                                    this_mode = (group_datasets[DataTags.MODALITY] == mode)
                                    folder_mask = (group_datasets[AcquisiTags.BASE_PATH] == folder)
                                    this_vid = (group_datasets[DataTags.VIDEO_ID] == vidnum)
                                    slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))

                        else:
                            for q in range(len(all_query_status[folder][mode])):
                                all_query_status[folder][mode][q].loc[:, vidnum] = "Dataset Failed To Load"
                            warnings.warn("Video number "+str(vidnum)+ ": Dataset Failed To Load")
                            continue

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
                             dataset.query_loc[q]) = extract_n_refine_iorg_signals(dataset, analysis_dat_format,#
                                                                                   query_loc=dataset.query_loc[q],
                                                                                   query_loc_name=query_loc_names[q],
                                                                                   thread_pool=the_pool)

                            # figure_label = "Debug: Included cells from "+ mode + " in query location: "+ query_loc_names[q] + " in " + folder.name
                            # plt.figure(figure_label)
                            # display_dict["Debug_"+mode + "_inc_cells_"+query_loc_names[q] ] = figure_label
                            #
                            # plt.title(figure_label)
                            # plt.imshow(dataset.avg_image_data, cmap='gray')
                            # viability = all_query_status[folder][mode][q].loc[:, "Viable for single-cell summary?"]
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

                            # If this is the first time a video of this mode and this folder is loaded, then initialize the query status dataframe
                            # Such that each row corresponds to the original coordinate locations based on the reference image.
                            if first:
                                # The below maps each query loc (some coordinates) to a tuple, then forms those tuples into a list.
                                all_query_status[folder][mode][q] = all_query_status[folder][mode][q].reindex(pd.MultiIndex.from_tuples(list(map(tuple, dataset.query_loc[q]))), fill_value="Included")
                                #all_query_status[folder][mode][q].sort_index(inplace=True)

                        first = False

                        # Once we've extracted the iORG signals, remove the video and mask data as it's likely to have a large memory footprint.
                        dataset.clear_video_data()

            # If desired, make the summarized iORG relative to controls in some way.
            # Control data is expected to be applied to the WHOLE group.
            for folder in folder_groups:

                folder_mask = (group_datasets[AcquisiTags.BASE_PATH] == folder)

                data_in_folder = group_datasets.loc[folder_mask]
                pop_iORG_result_datframe = []
                indiv_iORG_result_datframe = []

                if analysis_params.get(Analysis.OUTPUT_SUBFOLDER, True):
                    result_folder = folder.joinpath(output_folder, output_dt_subfolder)
                    result_folder.mkdir(parents=True, exist_ok=True)
                else:
                    result_folder = folder.joinpath(output_folder)
                    result_folder.mkdir(parents=True,exist_ok=True)

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

                    # Make data storage structures for our results.
                    if not group_datasets.loc[slice_of_life & query_locations].empty:
                        query_loc_names = group_datasets.loc[slice_of_life & query_locations, DataTags.QUERYLOC].unique().tolist()
                        for q, query_loc_name in enumerate(query_loc_names):
                            if len(query_loc_name) == 0:
                                query_loc_names[q] = " "
                            else:
                                query_loc_names[q] = query_loc_name.strip('_- ')
                    else:
                        query_loc_names = []

                    result_cols = pd.MultiIndex.from_product([query_loc_names, list(MetricTags)])
                    pop_iORG_result_datframe = pd.DataFrame(index=stim_data_vidnums, columns=result_cols)

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
                    mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(stim_data_vidnums)))
                    for v, stim_vidnum in enumerate(stim_data_vidnums):

                        control_query_status = [pd.DataFrame(columns=control_data_vidnums) for i in range((slice_of_life & query_locations).sum())]

                        this_vid = (group_datasets[DataTags.VIDEO_ID] == stim_vidnum)

                        slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))

                        # Grab the stim dataset associated with this video number.
                        stim_dataset = group_datasets.loc[slice_of_life & only_vids, AcquisiTags.DATASET].values[0]

                        # Process all control datasets in accordance with the stimulus datasets' parameters,
                        # e.g. stimulus location/duration, combine them, and do whatever the user wants with them.
                        control_datasets = group_datasets.loc[this_mode & only_vids & ~has_stim, AcquisiTags.DATASET].tolist()

                        if (not uniform_datasets or first_run):

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
                                         control_data.query_loc[q]) = extract_n_refine_iorg_signals(control_data,
                                                                                                    control_dat_format_params,
                                                                                                    query_loc=stim_dataset.query_loc[q],
                                                                                                    query_loc_name=query_loc_names[q],
                                                                                                    stimtrain_frame_stamps=stim_dataset.stimtrain_frame_stamps,
                                                                                                    thread_pool=the_pool)

                                        control_query_status[q].loc[:, cd] = control_query_stat
                                        control_pop_iORG_N[cd, control_data.framestamps] = np.sum(np.isfinite(control_data.iORG_signals[q]))
                                        control_iORG_sigs[cd, :, control_data.framestamps] = control_data.iORG_signals[q].T
                                        control_pop_iORG_summaries[cd, control_data.framestamps] = control_data.summarized_iORGs[q]

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
                                    control_iORG_signals[q] = control_iORG_sigs
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
                                    control_query_status[q].to_csv(result_folder.joinpath(str(subject_IDs[0]) + "_query_loc_status_" + str(folder.name) + "_" + str(mode) +
                                                               "_" + query_loc_names[q] + "coords_controldata.csv"))

                        ''' *** Population iORG analyses here *** '''
                        for q in range(len(stim_dataset.query_loc)):
                            with warnings.catch_warnings():
                                warnings.filterwarnings(action="ignore", message="indexing past lexsort depth may impact performance.")
                                all_query_status[folder][mode][q].loc[:, stim_vidnum] = stim_dataset.query_status[q]

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
                                display_dict[str(subject_IDs[0]) + "_" + mode+"_pop_iORG_"+ sum_method +"_overlapping_"+query_loc_names[q]+"coords_"+folder.name] = overlap_label

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
                                    display_dict[str(subject_IDs[0]) + "_" + mode + "_pop_iORG_" + sum_method + "_sequential_stim_only_" + query_loc_names[q] + "coords_" + folder.name] = seq_stim_label

                                    display_iORG_pop_summary_seq(stim_framestamps, stim_pop_summary, vidnum_seq[v], stim_dataset.stimtrain_frame_stamps,
                                                                 stim_dataset.framerate, sum_method, seq_stim_label,
                                                                 pop_seq_params)
                                    plt.show(block=False)

                                if pop_seq_params.get(DisplayParams.DISP_RELATIVE, True):
                                    seq_rel_label = "Query file " + query_loc_names[q] + "Stimulus relative to control iORG via " + sum_control +" temporal sequence"
                                    display_dict[str(subject_IDs[0]) + "_" + mode + "_pop_iORG_" + sum_method + "_sequential_relative_" + query_loc_names[q] + "coords_" + folder.name] = seq_rel_label

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

                        indiv_iORG_result[q] = pd.DataFrame(index=all_query_status[folder][mode][q].index, columns=list(MetricTags))
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
                            # If all stimulus datasets are uniform,
                            # we can also summarize individual iORGs by combining a cells' iORGs across acquisitions.
                            if uniform_datasets:
                                stim_iORG_signals[q][d, :, stim_dataset.framestamps] = stim_dataset.iORG_signals[q].T

                        pooled_framerate = np.unique(pooled_framerate)
                        if len(pooled_framerate) != 1:
                            warnings.warn("The framerate of the iORGs analyzed in "+folder.name + " is inconsistent: ("+str(pooled_framerate)+"). Pooled results may be incorrect.")
                            pooled_framerate = pooled_framerate[0]

                        stimtrain = np.unique(pd.DataFrame(stimtrain).values.astype(np.int32), axis=0)
                        if stimtrain.shape[0] != 1:
                            warnings.warn("The stimulus frame train of the iORGs analyzed in " + folder.name + " is inconsistent! Pooled results may be incorrect.")

                        stimtrain = stimtrain[0]

                        # Debug - to look at individual cell raw traces.
                        if debug_params.get(DebugParams.PLOT_INDIV_STANDARDIZED_ORGS, False) or debug_params.get(DebugParams.OUTPUT_INDIV_STANDARDIZED_ORGS, False):
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
                                        sigout.to_csv(result_folder.joinpath(overlap_label+".csv"))

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
                            stim_pop_iORG_summary[q] = np.nansum(stim_pop_iORG_N * stim_pop_iORG_summaries,axis=0) / np.nansum(stim_pop_iORG_N, axis=0)

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
                            display_dict[str(subject_IDs[0]) + "_" + "pooled_" + mode + "_pop_iORG_" + sum_method + "_overlapping"] = overlap_label

                            display_iORG_pop_summary(np.arange(max_frmstamp + 1), stim_pop_iORG_summary[q], stim_vidnum=query_loc_names[q],
                                                     stim_delivery_frms=stimtrain, framerate=pooled_framerate, sum_method=sum_method, sum_control=sum_control,
                                                     figure_label=overlap_label, params=pop_overlap_params)
                            plt.show(block=False)

                            plt.title("Pooled "+ sum_method +"iORGs relative to control iORG via " + sum_control)

                        # If we have a uniform dataset, summarize each cell's iORG too.
                        ''' *** Individual iORG analyses start here *** '''
                        if uniform_datasets:
                            all_frmstmp = np.arange(max_frmstamp + 1)

                            if indiv_overlap_params:
                                if indiv_overlap_params.get(DisplayParams.DISP_STIMULUS, False) or \
                                        indiv_overlap_params.get(DisplayParams.DISP_CONTROL, False) or \
                                        indiv_overlap_params.get(DisplayParams.DISP_RELATIVE, False):
                                    overlap_label = "Individual-Cell iORGs summarized with " + sum_method + " of " + mode + " iORGs in " + folder.name
                                    display_dict[str(subject_IDs[0]) + "_" + mode + "_indiv_iORG_" + sum_method + "_overlapping"] = overlap_label

                            all_tot_sig = np.nansum(np.any(np.isfinite(stim_iORG_signals[q]), axis=2), axis=0)
                            viable_sig = all_tot_sig >= sum_params.get(SummaryParams.INDIV_CUTOFF, 5)

                            with warnings.catch_warnings():
                                warnings.filterwarnings(action="ignore", message="indexing past lexsort depth may impact performance.")

                                all_query_status[folder][mode][q].loc[:, "Num Viable iORGs"] = all_tot_sig
                                all_query_status[folder][mode][q].loc[:, "Viable for single-cell summary?"] = viable_sig

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
                            indiv_respath = result_folder.joinpath(str(subject_IDs[0]) + "_indiv_summary_metrics" + str(folder.name) + "_" + str(mode) +
                                                       "_" + query_loc_names[q] + "coords.csv")

                            if indiv_summary.get(DisplayParams.HISTOGRAM):

                                overlap_label = "Individual-Cell iORGs metric histograms from " + mode + " iORGs in " + folder.name
                                display_dict[str(subject_IDs[0]) + "_" + mode + "_indiv_iORG_" + sum_method + "_metric_histograms"] = overlap_label

                                display_iORG_summary_histogram(indiv_iORG_result[q], metrics_tags, False, query_loc_names[q],
                                                               overlap_label, indiv_summary)


                            if indiv_summary.get(DisplayParams.CUMULATIVE_HISTOGRAM):
                                overlap_label = "Individual-Cell iORGs metric cumulative histograms from " + mode + " iORGs in " + folder.name
                                display_dict[str(subject_IDs[0]) + "_" + mode + "_indiv_iORG_" + sum_method + "_metric_cumul_histograms"] = overlap_label

                                display_iORG_summary_histogram(indiv_iORG_result[q], metrics_tags, True, query_loc_names[q],
                                                               overlap_label, indiv_summary)

                            if indiv_summary.get(DisplayParams.MAP_OVERLAY):
                                ax_params = indiv_summary.get(DisplayParams.AXES, dict())

                                for metric in metrics_tags:
                                    if indiv_iORG_result[q].loc[:, metric].count() != 0:
                                        label = "Individual iORG "+metric+" from " + mode + " using query locations: " + query_loc_names[q] + " in " + folder.name
                                        display_dict[str(subject_IDs[0]) + "_" + mode + "_indiv_iORG_" + sum_method + "_" + metric + "_overlay_" + query_loc_names[q]] = label

                                        refim = group_datasets.loc[folder_mask & (this_mode & reference_images), AcquisiTags.DATA_PATH].values[0]

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
                                save_video(result_folder.joinpath("pooled_pixelpop_iORG_" + now_timestamp + ".avi"),
                                           video_profiles, pooled_framerate.item(),
                                           scalar_mapper=mapper)

                            tryagain = True
                            while tryagain:
                                try:
                                    indiv_iORG_result[q].sort_index(inplace=True)
                                    indiv_iORG_result[q].to_csv(indiv_respath)
                                    tryagain = False
                                except PermissionError:
                                    tryagain = messagebox.askyesno(
                                        title="File: " + str(indiv_respath) + " is unable to be written.",
                                        message="The result file may be open. Close the file, then try to write again?")

                            # figure_label = "Debug: Included cells from "+ mode + " in query location: "+ query_loc_names[q] + " in " + folder.name
                            # plt.figure(figure_label)
                            # display_dict["Debug_"+mode + "_inc_cells_"+query_loc_names[q] ] = figure_label
                            # refim = group_datasets.loc[folder_mask & (this_mode & reference_images), AcquisiTags.DATA_PATH].values[0]
                            # plt.title(figure_label)
                            # plt.imshow(cv2.imread(refim, cv2.IMREAD_GRAYSCALE), cmap='gray')
                            # viability = all_query_status[folder][mode][q].loc[:, "Viable for single-cell summary?"]
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

                        all_query_status[folder][mode][q].sort_index(inplace=True)
                        all_query_status[folder][mode][q].to_csv(result_folder.joinpath(str(subject_IDs[0]) + "_query_loc_status_" + str(folder.name) + "_" + str(mode) +
                                                   "_" + query_loc_names[q] + "coords.csv"))


                    respath = result_folder.joinpath(str(subject_IDs[0]) + "_pop_summary_metrics_" + str(folder.name) + "_" + str(mode) + ".csv")
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
                        plt.gcf().set_size_inches(5*sublayout.ncols, 5*sublayout.nrows)
                        plt.draw()
                        for ext in saveas_ext:
                            tryagain = True
                            while tryagain:
                                try:
                                    plt.savefig(result_folder.joinpath(fname+"."+ext), dpi=300)
                                    tryagain = False
                                except PermissionError:
                                    tryagain = messagebox.askyesno(
                                        title="Figure " + str(fname+"."+ext) + " is unable to be saved.",
                                        message="The figure file may be open. Close the file, then try to write again?")

                        plt.close(figname)

                    # Save the json used to analyze this data, for auditing.
                    out_json = Path(json_fName).stem + "_ran_at_" + now_timestamp + ".json"
                    out_json = result_folder.joinpath(out_json)

                    audit_json_dict = {ConfigFields.VERSION: dat_form.get(ConfigFields.VERSION, "none"),
                                       ConfigFields.DESCRIPTION: dat_form.get(ConfigFields.DESCRIPTION, "none"),
                                       PreAnalysisPipeline.NAME: preanalysis_dat_format,
                                       Analysis.NAME: analysis_dat_format}

                    with open(out_json, 'w') as f:
                        json.dump(audit_json_dict, f, indent=2)

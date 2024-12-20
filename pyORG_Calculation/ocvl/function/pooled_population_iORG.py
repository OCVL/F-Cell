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
from ocvl.function.utility.json_format_constants import PipelineParams, MetaTags, DataFormatType, DataTags, AcquisiTags, \
    SegmentParams, ExclusionParams, NormParams, STDParams, ORGTags, SummaryParams

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
    seg_params = analysis_params.get(SegmentParams.NAME, dict())
    norm_params = analysis_params.get(NormParams.NAME, dict())
    excl_params = analysis_params.get(ExclusionParams.NAME, dict())
    std_params = analysis_params.get(STDParams.NAME, dict())
    sum_params = analysis_params.get(SummaryParams.NAME, dict())
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
                for vidnum in data_vidnums:

                    this_vid = (group_datasets[DataTags.VIDEO_ID] == vidnum)

                    slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))


                    pb["value"] = vidnum
                    mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(data_vidnums)))

                    # for later: allData.loc[ind, AcquisiTags.DATASET]
                    # Actually load the dataset, and all its metadata.
                    dataset = initialize_and_load_dataset(group_datasets.loc[slice_of_life], metadata_params)

                    if dataset is not None:
                        group_datasets.loc[slice_of_life & only_vids, AcquisiTags.DATASET] = dataset
                        group_datasets.loc[slice_of_life & only_vids, AcquisiTags.STIM_PRESENT] = len(dataset.stimtrain_frame_stamps) > 1
                    else:
                        for q in range(len(query_status)):
                            query_status[q].loc[:, vidnum] = "Dataset Failed To Load"
                        continue

                    # Perform analyses on each query location set for each dataset.
                    for q in range(len(dataset.query_loc)):

                        pb_label["text"] = "Processing query file " + str(q) + " in dataset #" + str(vidnum) + " from the " + str(
                            mode) + " modality in group " + str(group) + " and folder " + folder.stem + "..."
                        pb.update()
                        pb_label.update()
                        print("Processing query file " + str(dataset.metadata.get(AcquisiTags.QUERYLOC_PATH,Path())[q].stem) +
                              " in dataset #" + str(vidnum) + " from the " + str(mode) + " modality in group "
                              + str(group) + " and folder " + folder.stem + "...")

                        '''
                        *** This section is where we actually do dataset summary and analysis. (population iORG) ***
                        '''

                        query_status[q] = query_status[q].reindex(pd.MultiIndex.from_tuples(list(map(tuple, dataset.query_loc[q]))), fill_value="Included")

                        if seg_params.get(SegmentParams.REFINE_TO_REF, True):
                            reference_coord_data = refine_coord(dataset.avg_image_data, dataset.query_loc[q])
                        else:
                            reference_coord_data = dataset.query_loc[q]

                        coorddist = pdist(reference_coord_data, "euclidean")
                        coorddist = squareform(coorddist)
                        coorddist[coorddist == 0] = np.amax(coorddist.flatten())
                        mindist = np.amin(coorddist, axis=-1)

                        # If not defined, then we default to "auto" which determines it from the spacing of the query points
                        segmentation_radius = seg_params.get(SegmentParams.RADIUS, "auto")
                        if segmentation_radius == "auto":
                            segmentation_radius = np.round(np.nanmean(mindist) / 4) if np.round(np.nanmean(mindist) / 4) >= 1 else 1

                            segmentation_radius = int(segmentation_radius)
                            print("Detected segmentation radius: " + str(segmentation_radius))

                        dataset.query_loc[q] = reference_coord_data

                        if seg_params.get(SegmentParams.REFINE_TO_VID, True):
                            dataset.query_loc[q], valid_signals, excl_reason  = refine_coord_to_stack(dataset.video_data, dataset.avg_image_data,
                                                                                                      reference_coord_data)
                            # Update our audit path.
                            query_status[q].loc[~valid_signals, vidnum] = excl_reason[~valid_signals]
                        else:
                            valid_signals = np.full((dataset.query_loc[q].shape[0]), True)


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
                        iORG_signals, excl_reason = extract_profiles(dataset.video_data, dataset.query_loc[q], seg_radius=segmentation_radius,
                                                                     seg_mask=seg_shape, summary=seg_summary)
                        # Update our audit path.
                        valid = np.all(np.isfinite(iORG_signals), axis=1)
                        to_update = ~(~valid_signals | valid) # Use the inverse of implication to find which ones to update.
                        valid_signals = valid & valid_signals
                        query_status[q].loc[to_update, vidnum] = excl_reason[to_update]

                        # Only do the below if we're in a stimulus trial- otherwise, we can't know what the control data
                        # will be used for, or what critical region/standardization indicies it'll need.
                        if group_datasets.loc[slice_of_life & only_vids, AcquisiTags.STIM_PRESENT].values[0]:

                            # Exclude signals that don't pass our criterion
                            excl_type = excl_params.get(ExclusionParams.TYPE)
                            excl_units = excl_params.get(ExclusionParams.UNITS)
                            excl_start = excl_params.get(ExclusionParams.START)
                            excl_stop = excl_params.get(ExclusionParams.STOP)
                            excl_cutoff_fraction = excl_params.get(ExclusionParams.FRACTION)

                            if excl_units == "time":
                                excl_start_ind = int(excl_start * dataset.framerate)
                                excl_stop_ind = int(excl_stop * dataset.framerate)
                            else: #if units == "frames":
                                excl_start_ind = int(excl_start)
                                excl_stop_ind = int(excl_stop)

                            if excl_type == "stim-relative":
                                excl_start_ind = dataset.stimtrain_frame_stamps[0] + excl_start_ind
                                excl_stop_ind = dataset.stimtrain_frame_stamps[1] + excl_stop_ind
                            else: #if type == "absolute":
                                pass
                                #excl_start_ind = excl_start_ind
                                #excl_stop_ind = excl_stop_ind
                            crit_region = np.arange(excl_start_ind, excl_stop_ind)

                            iORG_signals, valid, excl_reason = exclude_profiles(iORG_signals, dataset.framestamps,
                                                                                 critical_region=crit_region,
                                                                                 critical_fraction=excl_cutoff_fraction)
                            # Update our audit path.
                            to_update = ~(~valid_signals | valid) # Use the inverse of implication to find which ones to update.
                            valid_signals = valid & valid_signals
                            query_status[q].loc[to_update, vidnum] = excl_reason[to_update]

                            # Standardize individual signals
                            std_meth = std_params.get(STDParams.METHOD, "mean_sub")
                            std_type =  std_params.get(STDParams.METHOD, "stim-relative")
                            std_units = std_params.get(STDParams.UNITS, "time")
                            std_start = std_params.get(STDParams.START, -1)
                            std_stop = std_params.get(STDParams.STOP, 0)

                            if excl_units == "time":
                                std_start = int(std_start * dataset.framerate)
                                std_stop = int(std_stop * dataset.framerate)
                            else:  # if units == "frames":
                                std_start = int(std_start)
                                std_stop = int(std_stop)

                            if excl_type == "stim-relative":
                                std_start = dataset.stimtrain_frame_stamps[0] + std_start
                                std_stop = dataset.stimtrain_frame_stamps[1] + std_stop

                            std_ind = np.arange(std_start, std_stop)

                            iORG_signals = standardize_profiles(iORG_signals, dataset.framestamps, std_indices=std_ind, method=std_meth)

                            sum_method = sum_params.get(SummaryParams.METHOD, "rms")
                            sum_window = sum_params.get(SummaryParams.WINDOW_SIZE, 1)
                            summarized_iORG, num_signals_per_sample = signal_power_iORG(iORG_signals, dataset.framestamps, summary_method=sum_method,
                                                                                        window_size=sum_window)

                            dataset.iORG_signals[q] = iORG_signals
                            dataset.summarized_iORGs[q] = summarized_iORG

                        else:
                            dataset.iORG_signals[q] = iORG_signals

                for q in range(len(query_status)):
                    query_status[q].to_csv(result_folder.joinpath("query_loc_status_"+str(folder.stem) + "_"+ str(mode) +"_"+dataset.metadata.get(AcquisiTags.QUERYLOC_PATH,Path())[q].stem +".csv"))


        # If desired, make the summarized iORG relative to controls in some way.
        # Control data is expected to be applied to the WHOLE group.
        # Respect the users' folder structure. If things are in different folders, analyze them separately.
        for folder in folder_groups:

            result_folder = folder.joinpath(output_folder)

            folder_mask = (group_datasets[AcquisiTags.BASE_PATH] == folder)

            data_in_folder = group_datasets.loc[folder_mask]
            iORG_result_datframes = []

            # Load each modality
            for mode in modes_of_interest:
                this_mode = (group_datasets[DataTags.MODALITY] == mode)

                stimless = ~(group_datasets[AcquisiTags.STIM_PRESENT])

                slice_of_life = folder_mask & this_mode

                data_vidnums = group_datasets[DataTags.VIDEO_ID].unique().tolist()

                # Make data storage structures for each of our query location lists- one is for results,
                # The other for checking which query points went into our analysis.
                iORG_result_datframes.append([pd.DataFrame(index=data_vidnums, columns=list(ORGTags)) for i in range(query_locations.sum())])

                # Load each dataset (delineated by different video numbers), normalize it, standardize it, etc.
                for vidnum in data_vidnums:

                    this_vid = (group_datasets[DataTags.VIDEO_ID] == vidnum)

                    slice_of_life = folder_mask & (this_mode & (this_vid | (reference_images | query_locations)))

                    pb["maximum"] = len(data_vidnums)
                    pb["value"] = vidnum
                    mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(data_vidnums)))

                    # for later: allData.loc[ind, AcquisiTags.DATASET]
                    # Actually load the dataset, and all its metadata.
                    group_datasets.loc[slice_of_life & only_vids, AcquisiTags.DATASET]



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
            for m, mode in enumerate(modes_of_interest):
                mode_data = data_in_folder.loc[data_in_folder[DataTags.MODALITY] == mode]

                control = mode_data.loc[(mode_data[DataFormatType.FORMAT_TYPE] == DataFormatType.VIDEO) & ~(mode_data[AcquisiTags.STIM_PRESENT])]

                iORG_result_datframes

                for vidnum in data_vidnums:

                    data = mode_data.loc[mode_data[DataTags.VIDEO_ID] == vidnum]

                    pb["maximum"] = len(data_vidnums)
                    pb["value"] = vidnum
                    mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", len(data_vidnums)))


                    # Perform analyses on each query location set for each dataset.
                    for q in range(len(dataset.query_loc)):


                        #TODO: Add plotting class for the below
                        # plt.figure(9)
                        # plt.plot(dataset.framestamps, num_signals_per_sample)
                        # plt.show(block=False)

                        #TODO: Assess the difference this makes. This is just to make them all at the same baseline.
                        #summarized_iORG = standardize_profiles(summarized_iORG[None, :], dataset.framestamps, std_indices=prestim_ind, method="mean_sub")
                        #summarized_iORG = np.squeeze(summarized_iORG)

                        poststim_ind = np.flatnonzero(np.logical_and(dataset.framestamps >= dataset.stimtrain_frame_stamps[1],
                                                      dataset.framestamps < (dataset.stimtrain_frame_stamps[1] + int(1 * dataset.framerate))))

                        poststim_loc = dataset.framestamps[poststim_ind]
                        prestim_amp = np.nanmedian(summarized_iORG[prestim_ind])
                        poststim = summarized_iORG[poststim_ind]

                        if poststim.size == 0:

                            pop_iORG_amp[r] = np.NaN
                            pop_iORG_implicit[r] = np.NaN
                            pop_iORG_recover[r] = np.NaN
                        else:
                            poststim_amp = np.quantile(poststim, [0.95])
                            max_frmstmp = poststim_loc[np.argmax(poststim)] - dataset.stimtrain_frame_stamps[0]
                            final_val = np.mean(summarized_iORG[-5:])

                            framestamps.append(dataset.framestamps)
                            pop_iORG.append(summarized_iORG)
                            pop_iORG_num.append(num_signals_per_sample)

                            pop_iORG_amp[r], pop_iORG_implicit[r] = iORG_signal_metrics(summarized_iORG[None, :], dataset.framestamps,
                                                                                        filter_type="none", display=False,
                                                                                        prestim_idx=prestim_ind,
                                                                                        poststim_idx=poststim_ind)[1:3]

                            pop_iORG_recover[r] = 1 - ((final_val - prestim_amp) / pop_iORG_amp[r])
                            pop_iORG_implicit[r] = pop_iORG_implicit[r] / dataset.framerate

                            print("Signal metrics based iORG Amplitude: " + str(pop_iORG_amp[r]) +
                                  " Implicit time (s): " + str(pop_iORG_implicit[r]) +
                                  " Recovery fraction: " + str(pop_iORG_recover[r]))

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

                        if dataset.framestamps[-1] > max_frmstamp:
                            max_frmstamp = dataset.framestamps[-1]

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
                all_iORG = np.empty((len(pop_iORG), max_frmstamp+1))
                all_iORG[:] = np.nan
                all_incl = np.empty((len(pop_iORG), max_frmstamp + 1))
                all_incl[:] = np.nan
                for q, iorg in enumerate(pop_iORG):
                    all_incl[q, framestamps[q]] = pop_iORG_num[q]
                    all_iORG[q, framestamps[q]] = iorg


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


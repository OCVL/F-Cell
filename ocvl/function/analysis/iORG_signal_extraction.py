import warnings

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

from skimage.morphology import disk

from ocvl.function.analysis.iORG_profile_analyses import summarize_iORG_signals
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.json_format_constants import SegmentParams, NormParams, ExclusionParams, STDParams, \
    SummaryParams
from scipy.spatial.distance import pdist, squareform

def extract_n_refine_iorg_signals(dataset, analysis_params, query_loc=None, stimtrain_frame_stamps=None):

    if query_loc is None:
        query_loc= dataset.query_loc[0]

    if stimtrain_frame_stamps is None:
        stimtrain_frame_stamps = dataset.stimtrain_frame_stamps

    # Round the query locations

    query_loc = np.round(query_loc)
    og = query_loc.copy()

    # Snag all of our parameter dictionaries that we'll use here.
    # Default to an empty dictionary so that we can query against it with fewer if statements.
    seg_params = analysis_params.get(SegmentParams.NAME, dict())
    seg_shape = seg_params.get(SegmentParams.SHAPE, "disk") # Default: A disk shaped mask over each query coordinate
    seg_summary = seg_params.get(SegmentParams.SUMMARY, "mean") # Default the average of the intensity of the query coordinate

    excl_params = analysis_params.get(ExclusionParams.NAME, dict())
    excl_type = excl_params.get(ExclusionParams.TYPE, "stim-relative") # Default: Relative to the stimulus delivery location
    excl_units = excl_params.get(ExclusionParams.UNITS, "time")
    excl_start = excl_params.get(ExclusionParams.START, -0.2)
    excl_stop = excl_params.get(ExclusionParams.STOP, 0.2)
    excl_cutoff_fraction = excl_params.get(ExclusionParams.FRACTION, 0.5)

    std_params = analysis_params.get(STDParams.NAME, dict())
    std_meth = std_params.get(STDParams.METHOD, "mean_sub") # Default: Subtracts the mean from each iORG signal
    std_type = std_params.get(STDParams.TYPE, "stim-relative") # Default: Relative to the stimulus delivery location
    std_units = std_params.get(STDParams.UNITS, "time")
    std_start = std_params.get(STDParams.START, -1)
    std_stop = std_params.get(STDParams.STOP, 0)

    sum_params = analysis_params.get(SummaryParams.NAME, dict())
    sum_method = sum_params.get(SummaryParams.METHOD, "rms")
    sum_window = sum_params.get(SummaryParams.WINDOW_SIZE, 1)

    query_status = np.full(query_loc.shape[0], "Included", dtype=object)
    valid_signals = np.full((query_loc.shape[0]), True)

    if seg_params.get(SegmentParams.REFINE_TO_REF, True):
        query_loc, valid, excl_reason = refine_coord(dataset.avg_image_data, query_loc.copy())
    else:
        excl_reason = np.full(query_loc.shape[0], "Included", dtype=object)
        valid = np.full((query_loc.shape[0]), True)

    # Update our audit path.
    to_update = ~(~valid_signals | valid)  # Use the inverse of implication to find which ones to update.
    valid_signals = valid & valid_signals
    query_status[to_update] = excl_reason[to_update]

    coorddist = pdist(query_loc, "euclidean")
    coorddist = squareform(coorddist)
    coorddist[coorddist == 0] = np.amax(coorddist.flatten())
    mindist = np.amin(coorddist, axis=-1)

    # If not defined, then we default to "auto" which determines it from the spacing of the query points
    segmentation_radius = seg_params.get(SegmentParams.RADIUS, "auto")
    if segmentation_radius == "auto":
        segmentation_radius = np.round(np.nanmean(mindist) / 4) if np.round(np.nanmean(mindist) / 4) >= 1 else 1

        segmentation_radius = int(segmentation_radius)
        print("Detected segmentation radius: " + str(segmentation_radius))
    else:
        segmentation_radius = int(segmentation_radius)
        print("Chosen segmentation radius: " + str(segmentation_radius))

    if seg_params.get(SegmentParams.REFINE_TO_VID, True):
        query_loc, valid, excl_reason = refine_coord_to_stack(dataset.video_data, dataset.avg_image_data,
                                                              query_loc.copy())
    else:
        excl_reason = np.full(query_loc.shape[0], "Included", dtype=object)
        valid = np.full((query_loc.shape[0]), True)

    # Update our audit path.
    to_update = ~(~valid_signals | valid)  # Use the inverse of implication to find which ones to update.
    valid_signals = valid & valid_signals
    query_status[to_update] = excl_reason[to_update]

    # Extract the signals
    iORG_signals, excl_reason = extract_signals(dataset.video_data, query_loc.copy(),
                                                seg_radius=segmentation_radius,
                                                seg_mask=seg_shape, summary=seg_summary)
    # Update our audit path.
    valid = np.any(np.isfinite(iORG_signals), axis=1)
    to_update = ~(~valid_signals | valid)  # Use the inverse of implication to find which ones to update.
    valid_signals = valid & valid_signals
    query_status[to_update] = excl_reason[to_update]

    # Should only do the below if we're in a stimulus trial- otherwise, we can't know what the control data
    # will be used for, or what critical region/standardization indicies it'll need.

    # Exclude signals that don't pass our criterion
    if excl_units == "time":
        excl_start_ind = int(excl_start * dataset.framerate)
        excl_stop_ind = int(excl_stop * dataset.framerate)
    else:  # if units == "frames":
        excl_start_ind = int(excl_start)
        excl_stop_ind = int(excl_stop)

    if excl_type == "stim-relative":
        excl_start_ind = stimtrain_frame_stamps[0] + excl_start_ind
        excl_stop_ind = stimtrain_frame_stamps[1] + excl_stop_ind
    else:  # if type == "absolute":
        pass
        # excl_start_ind = excl_start_ind
        # excl_stop_ind = excl_stop_ind
    crit_region = np.arange(excl_start_ind, excl_stop_ind)

    iORG_signals, valid, excl_reason = exclude_signals(iORG_signals, dataset.framestamps,
                                                       critical_region=crit_region,
                                                       critical_fraction=excl_cutoff_fraction)
    # Update our audit path.
    to_update = ~(~valid_signals | valid)  # Use the inverse of implication to find which ones to update.
    valid_signals = valid & valid_signals
    query_status[to_update] = excl_reason[to_update]

    # Standardize individual signals
    if std_units == "time":
        std_start_ind = int(std_start * dataset.framerate)
        std_stop_ind = int(std_stop * dataset.framerate)
    else:  # if units == "frames":
        std_start_ind = int(std_start)
        std_stop_ind = int(std_stop)

    if std_type == "stim-relative":
        std_start_ind = stimtrain_frame_stamps[0] + std_start_ind
        std_stop_ind = stimtrain_frame_stamps[1] + std_stop_ind

    std_ind = np.arange(std_start_ind, std_stop_ind)

    iORG_signals, valid, excl_reason = standardize_signals(iORG_signals, dataset.framestamps, std_indices=std_ind, method=std_meth)
    # Update our audit path.
    to_update = ~(~valid_signals | valid)  # Use the inverse of implication to find which ones to update.
    valid_signals = valid & valid_signals
    query_status[to_update] = excl_reason[to_update]

    summarized_iORG, num_signals_per_sample = summarize_iORG_signals(iORG_signals, dataset.framestamps,
                                                                     summary_method=sum_method,
                                                                     window_size=sum_window)

    return iORG_signals, summarized_iORG, query_status, query_loc



def refine_coord(ref_image, coordinates, search_radius=1, numiter=2):

    im_size = ref_image.shape

    query_status = np.full(coordinates.shape[0], "Included", dtype=object)

    # Generate an inclusion list for our coordinates- those that are unanalyzable should be excluded before analysis.
    pluscoord = coordinates + search_radius*2*numiter # Include extra region to avoid edge effects
    includelist = pluscoord[:, 0] < im_size[1]
    includelist &= pluscoord[:, 1] < im_size[0]
    query_status[pluscoord[:, 0] >= im_size[1]] = "Reference refinement area outside image bounds (right side)"
    query_status[pluscoord[:, 1] >= im_size[0]] = "Reference refinement area outside image bounds (bottom side)"
    del pluscoord

    minuscoord = coordinates - search_radius*2*numiter # Include extra region to avoid edge effects
    includelist &= minuscoord[:, 0] >= 0
    includelist &= minuscoord[:, 1] >= 0
    query_status[minuscoord[:, 0] < 0] = "Reference refinement area outside image bounds (left side)"
    query_status[minuscoord[:, 1] < 0] = "Reference refinement area outside image bounds (top side)"
    del minuscoord

    coordinates = np.round(coordinates).astype("int")

    for i in range(coordinates.shape[0]):
        if includelist[i]:
            for iter in range(numiter):
                coord = coordinates[i, :]

                ref_template = ref_image[(coord[1] - search_radius):(coord[1] + search_radius + 1),
                                         (coord[0] - search_radius):(coord[0] + search_radius + 1)]

                minV, maxV, minL, maxL = cv2.minMaxLoc(ref_template)

                maxL = np.array(maxL)-search_radius # Make relative to the center.
                coordinates[i, :] = coord + maxL

                if np.all(maxL == 0):
                    # print("Unchanged. Breaking...")
                    break

    return coordinates, includelist, query_status


def refine_coord_to_stack(image_stack, ref_image, coordinates, search_radius=2, threshold=0.3):

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="invalid value encountered in cast")

        ref_image = ref_image.astype("uint8")
        image_stack = image_stack.astype("uint8")

    im_size = image_stack.shape

    search_region = 2*search_radius # Include extra region for edge effects

    query_status = np.full(coordinates.shape[0], "Included", dtype=object)

    # Generate an inclusion list for our coordinates- those that are unanalyzable should be excluded from refinement.
    pluscoord = coordinates + search_region
    includelist = pluscoord[:, 0] < im_size[1]
    includelist &= pluscoord[:, 1] < im_size[0]
    query_status[pluscoord[:, 0] >= im_size[1]] = "Stack refinement area outside image bounds (right side)"
    query_status[pluscoord[:, 1] >= im_size[0]] = "Stack refinement area outside image bounds (bottom side)"
    del pluscoord

    minuscoord = coordinates - search_region
    includelist &= minuscoord[:, 0] >= 0
    includelist &= minuscoord[:, 1] >= 0
    query_status[minuscoord[:, 0] < 0] = "Stack refinement area outside image bounds (left side)"
    query_status[minuscoord[:, 1] < 0] = "Stack refinement area outside image bounds (top side)"
    del minuscoord

    coordinates = np.round(coordinates).astype("int")

    for i in range(coordinates.shape[0]):
        if includelist[i]:
            coord = coordinates[i, :]

            stack_data = image_stack[(coord[1] - search_region):(coord[1] + search_region + 1),
                                     (coord[0] - search_region):(coord[0] + search_region + 1),
                                      :]
            stack_im = np.nanmean(stack_data, axis=-1).astype("uint8")
            ref_template = ref_image[(coord[1] - search_radius):(coord[1] + search_radius + 1),
                                     (coord[0] - search_radius):(coord[0] + search_radius + 1)]

            match_reg = cv2.matchTemplate(stack_im, ref_template, cv2.TM_CCOEFF_NORMED)
            minV, maxV, minL, maxL = cv2.minMaxLoc(match_reg)
            maxL = np.array(maxL) - search_radius  # Make relative to the center.
            if threshold < maxV: # If the alignment is over our NCC threshold (empirically, 0.3 works well), then do the alignment.
                coordinates[i, :] = coord + maxL

    return coordinates, includelist, query_status


def extract_signals(image_stack, coordinates=None, seg_mask="box", seg_radius=1, summary="mean", sigma=None, display=False):
    """
    This function extracts temporal profiles from a 3D matrix, where the first two dimensions are assumed to
    contain data from a single time point (a single image)

    :param image_stack: a YxXxZ numpy matrix, where there are Y rows, X columns, and Z samples.
    :param coordinates: input as X/Y, these mark locations the locations that will be extracted from all S samples.
    :param seg_mask: the mask shape that will be used to extract temporal profiles. Can be "box" or "disk".
    :param seg_radius: the radius of the mask shape that will be used.
    :param summary: the method used to summarize the area inside the segmentation radius. Default: "mean",
                    Options: "mean", "median"
    :param sigma: Precede extraction with a per-frame Gaussian filter of a supplied sigma. If none, no filtering is applied.

    :return: an NxM numpy matrix with N cells and M temporal samples of some signal.
    """

    if coordinates is None:
        pass # Todo: create coordinates for every position in the image stack.

    #im_stack = image_stack.astype("float32")

    im_stack_mask = image_stack == 0
    im_stack_mask = cv2.morphologyEx(im_stack_mask.astype("uint8"), cv2.MORPH_OPEN, np.ones((3, 3)),
                                     borderType=cv2.BORDER_CONSTANT, borderValue=1)

    im_stack = image_stack.astype("float32")
    im_stack[im_stack_mask.astype("bool")] = np.nan  # Anything that is outside our main image area should be made a nan.

    im_size = im_stack.shape
    if sigma is not None:
        for f in range(im_size[-1]):
            im_stack[..., f] = cv2.GaussianBlur(im_stack[..., f], ksize=(0, 0), sigmaX=sigma)

    query_status = np.full(coordinates.shape[0], "Included", dtype=object)

    pluscoord = coordinates + seg_radius
    includelist = pluscoord[:, 0] < im_size[1]
    includelist &= pluscoord[:, 1] < im_size[0]
    query_status[pluscoord[:, 0] >= im_size[1]] = "Segmentation outside image bounds (right side)"
    query_status[pluscoord[:, 1] >= im_size[0]] = "Segmentation outside image bounds (bottom side)"
    del pluscoord

    minuscoord = coordinates - seg_radius
    includelist &= minuscoord[:, 0] >= 0
    includelist &= minuscoord[:, 1] >= 0
    query_status[minuscoord[:, 0] < 0] = "Segmentation outside image bounds (left side)"
    query_status[minuscoord[:, 1] < 0] = "Segmentation outside image bounds (top side)"
    del minuscoord

    coordinates = np.floor(coordinates).astype("int")

    if summary != "none":
        signal_data = np.full((coordinates.shape[0], im_stack.shape[-1]), np.nan)
    else:
        signal_data = np.full((seg_radius * 2 + 1, seg_radius * 2 + 1,
                                 im_stack.shape[-1], coordinates.shape[0]), np.nan)

    if seg_mask == "box": # Handle more in the future...
        for i in range(coordinates.shape[0]):
            if includelist[i]:
                coord = coordinates[i, :]
                fullcolumn = im_stack[(coord[1] - seg_radius):(coord[1] + seg_radius + 1),
                                      (coord[0] - seg_radius):(coord[0] + seg_radius + 1), :]

                coldims = fullcolumn.shape
                coordcolumn = np.reshape(fullcolumn, (coldims[0]*coldims[1], coldims[2]), order="F")
                #print(coord)
                # No partial columns allowed. If there are nans in the column, wipe it out entirely.
                nani = np.any(np.isnan(coordcolumn), axis=0)
                coordcolumn[:, nani] = np.nan

                if np.all(np.isnan(coordcolumn.flatten())):
                    query_status[i] = "Missing Data at Query Location"
                    continue

                if summary == "mean":
                    signal_data[i, nani] = np.nan
                    signal_data[i, np.invert(nani)] = np.mean(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "median":
                    signal_data[i, nani] = np.nan
                    signal_data[i, np.invert(nani)] = np.nanmedian(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "sum":
                    signal_data[i, nani] = np.nan
                    signal_data[i, np.invert(nani)] = np.nansum(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "none":

                    signal_data[:, :, nani, i] = 0
                    signal_data[:, :, np.invert(nani), i] = fullcolumn[:, :, np.invert(nani)]

    elif seg_mask == "disk":
        for i in range(coordinates.shape[0]):
            if includelist[i]:
                coord = coordinates[i, :]
                fullcolumn = im_stack[(coord[1] - seg_radius):(coord[1] + seg_radius + 1),
                                      (coord[0] - seg_radius):(coord[0] + seg_radius + 1), :]
                mask = disk(seg_radius+1, dtype=fullcolumn.dtype)
                mask = mask[1:-1, 1:-1]
                mask = np.repeat(mask[:, :, None], fullcolumn.shape[-1], axis=2)

                coldims = fullcolumn.shape
                coordcolumn = np.reshape(fullcolumn, (coldims[0]*coldims[1], coldims[2]), order="F")
                mask = np.reshape(mask, (coldims[0] * coldims[1], coldims[2]), order="F")

                maskedout = np.where(mask == 0)
                coordcolumn[maskedout] = 0 # Areas that are masked shouldn't be considered in the partial column test below.
                # No partial columns allowed. If there are nans in the column, mark it to be wiped out entirely.
                nani = np.any(np.isnan(coordcolumn), axis=0)

                # Make our mask 0s into nans
                mask[mask == 0] = np.nan
                coordcolumn = coordcolumn * mask
                coordcolumn[:, nani] = np.nan

                if np.all(np.isnan(coordcolumn.flatten())):
                    query_status[i] = "Missing Data at Query Location"
                    continue

                if summary == "mean":
                    signal_data[i, nani] = np.nan
                    signal_data[i, np.invert(nani)] = np.nanmean(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "median":
                    signal_data[i, nani] = np.nan
                    signal_data[i, np.invert(nani)] = np.nanmedian(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "sum":
                    signal_data[i, nani] = np.nan
                    signal_data[i, np.invert(nani)] = np.nansum(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "none":

                    signal_data[:, :, nani, i] = 0
                    signal_data[:, :, np.invert(nani), i] = fullcolumn[:, :, np.invert(nani)]

    if display:
        plt.figure(1)
        for i in range(signal_data.shape[0]):
            plt.plot(signal_data[i, :]-signal_data[i, 0])
        plt.show()

    return signal_data, query_status

def exclude_signals(temporal_signals, framestamps,
                    critical_region=None, critical_fraction=0.5, require_full_signal=False):
    """
    A bit of code used to remove cells that don't have enough data in the critical region of a signal. This is typically
    surrounding a stimulus.

    :param temporal_signals: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_data.
    :param critical_region: A set of values containing the critical region of a signal- if a cell doesn't have data here,
                            then drop its entire signal from consideration.
    :param critical_fraction: The fraction of real values required to consider the signal valid.
    :param require_full_signal: Require a full profile instead of merely a fraction of the critical region.
    :return: a NxM numpy matrix of pared-down profiles, where profiles that don't fit the criterion are dropped.
    """

    query_status = np.full(temporal_signals.shape[0], "Included", dtype=object)

    if critical_region is not None:

        crit_inds = np.where(np.isin(framestamps, critical_region))[0]
        crit_remove = 0
        good_profiles = np.full((temporal_signals.shape[0],), True)
        for i in range(temporal_signals.shape[0]):
            this_fraction = np.sum(~np.isnan(temporal_signals[i, crit_inds])) / len(critical_region)

            if this_fraction < critical_fraction:
                crit_remove += 1
                temporal_signals[i, :] = np.nan
                good_profiles[i] = False

                query_status[i] = ("Only had data for " + f"{this_fraction * 100:.2f}" + "% of the req'd data in the crit region (frames " +
                                   str(critical_region[0]) + " - " + str(critical_region[-1]) + ").")

    if require_full_signal:
        for i in range(temporal_signals.shape[0]):
            if np.any(~np.isfinite(temporal_signals[i, :])) and good_profiles[i]:

                temporal_signals[i, :] = np.nan
                good_profiles[i] = False
                query_status[i] = "Incomplete profile, and the function req. full profiles."
                crit_remove += 1

    if critical_region is not None or require_full_signal:
        print(str(crit_remove) + "/" + str(temporal_signals.shape[0]) + " cells were cleared due to missing data at stimulus delivery.")

    return temporal_signals, good_profiles, query_status

def normalize_signals(temporal_signals, norm_method="mean", rescaled=False, video_ref=None):
    """
    This function normalizes the columns of the data (a single sample of all cells) using a method supplied by the user.

    :param temporal_signals: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param norm_method: The normalization method chosen by the user. Default is "mean". Options: "mean", "median"
    :param rescaled: Whether or not to keep the data at the original scale (only modulate the numbers in place). Useful
                     if you want the data to stay in the same units. Default: False. Options: True/False
    :param video_ref: A video reference (WxHxM) that can be used for normalization instead of the profile values.

    :return: a NxM numpy matrix of normalized temporal profiles.
    """

    if norm_method == "mean":
        all_norm = np.nanmean(temporal_signals[:])
        # plt.figure()
        # tmp = np.nanmean(temporal_data, axis=0)
        # plt.plot(tmp/np.amax(tmp))
        if video_ref is None:
            framewise_norm = np.nanmean(temporal_signals, axis=0)
        else:
            # Determine each frame's mean.
            framewise_norm = np.empty([video_ref.shape[-1]])
            for f in range(video_ref.shape[-1]):
                frm = video_ref[:, :, f].flatten().astype("float32")
                frm[frm == 0] = np.nan
                framewise_norm[f] = np.nanmean(frm)

            all_norm = np.nanmean(framewise_norm)
            #plt.plot(framewise_norm/np.amax(framewise_norm))
           # plt.show()
    elif norm_method == "median":
        all_norm = np.nanmedian(temporal_signals[:])
        if video_ref is None:
            framewise_norm = np.nanmedian(temporal_signals, axis=0)
        else:
            # Determine each frame's mean.
            framewise_norm = np.empty([video_ref.shape[-1]])
            for f in range(video_ref.shape[-1]):
                frm = video_ref[:, :, f].flatten().astype("float32")
                frm[frm == 0] = np.nan
                framewise_norm[f] = np.nanmedian(frm)
            all_norm = np.nanmean(framewise_norm)

    else:
        all_norm = np.nanmean(temporal_signals[:])
        if video_ref is None:
            framewise_norm = np.nanmean(temporal_signals, axis=0)
        else:
            # Determine each frame's mean.
            framewise_norm = np.empty([video_ref.shape[-1]])
            for f in range(video_ref.shape[-1]):
                frm = video_ref[:, :, f].flatten().astype("float32")
                frm[frm == 0] = np.nan
                framewise_norm[f] = np.nanmean(frm)
            all_norm = np.nanmean(framewise_norm)
        warnings.warn("The \"" + norm_method + "\" normalization type is not recognized. Defaulting to mean.")

    if rescaled: # Provide the option to simply scale the data, instead of keeping it in relative terms
        ratio = framewise_norm / all_norm
        return np.divide(temporal_signals, ratio[None, :])
    else:
        return np.divide(temporal_signals, framewise_norm[None, :])


def standardize_signals(temporal_signals, framestamps, std_indices, method="linear_std", critical_fraction=0.3):
    """
    This function standardizes each temporal profile (here, the rows of the supplied data) according to the provided
    arguments.

    :param temporal_signals: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_data.
    :param std_indices: The range of indices to use when standardizing.
    :param method: The method used to standardize. Default is "linear_std", which subtracts a linear fit to
                    each signal before stimulus_stamp, followed by a standardization based on that pre-stamp linear-fit
                    subtracted data. This was used in Cooper et al 2017/2020.
                    Current options include: "linear_std", "linear_vast", "relative_change", and "mean_sub"
    :param critical_fraction: The fraction of real values required to consider the signal valid.

    :return: a NxM numpy matrix of standardized temporal profiles
    """
    if len(std_indices) == 0:
        warnings.warn("Time before the stimulus framestamp doesn't exist in the provided list! No standardization performed.")
        return temporal_signals

    query_status = np.full(temporal_signals.shape[0], "Included", dtype=object)
    valid_stdization = np.full(temporal_signals.shape[0], True, dtype=bool)

    if method == "linear_std":
        # Standardize using Autoscaling preceded by a linear fit to remove
        # any residual low-frequency changes
        for i in range(temporal_signals.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[std_indices])
            prestim_profile = np.squeeze(temporal_signals[i, std_indices])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            if np.sum(goodind) >= np.floor(len(goodind)*critical_fraction):
                thefit = Polynomial.fit(prestim_frmstmp[goodind], prestim_profile[goodind], deg=1)
                fitvals = thefit(prestim_frmstmp[goodind]) # The values we'll subtract from the profile

                prestim_nofit_mean = np.nanmean(prestim_profile[goodind])
                prestim_mean = np.nanmean(prestim_profile[goodind]-fitvals)
                prestim_std = np.nanstd(prestim_profile[goodind]-fitvals)

                temporal_signals[i, :] = ((temporal_signals[i, :] - prestim_nofit_mean) / prestim_std)
            else:
                query_status[i] = "Incomplete signal for standardization (req'd " +str(critical_fraction)+", had " +str(len(goodind)*critical_fraction)+")"
                valid_stdization[i] = False
                temporal_signals[i, :] = np.nan

    elif method == "linear_vast":
        # Standardize using variable stability, or VAST scaling, preceeded by a linear fit:
        # https://www.sciencedirect.com/science/article/pii/S0003267003000941
        # this scaling is defined as autoscaling divided by the CoV.
        for i in range(temporal_signals.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[std_indices])
            prestim_profile = np.squeeze(temporal_signals[i, std_indices])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            if np.sum(goodind) >= np.floor(len(goodind)*critical_fraction):
                thefit = Polynomial.fit(prestim_frmstmp[goodind], prestim_profile[goodind], deg=1)
                fitvals = thefit(prestim_frmstmp[goodind]) # The values we'll subtract from the profile

                prestim_nofit_mean = np.nanmean(prestim_profile[goodind])
                prestim_mean = np.nanmean(prestim_profile[goodind]-fitvals)
                prestim_std = np.nanstd(prestim_profile[goodind]-fitvals)

                temporal_signals[i, :] = ((temporal_signals[i, :] - prestim_nofit_mean) / prestim_std) / \
                                         (prestim_std / prestim_nofit_mean)
            else:
                query_status[i] = "Incomplete signal for standardization (req'd " +str(critical_fraction)+", had " +str(len(goodind)*critical_fraction)+")"
                valid_stdization[i] = False
                temporal_signals[i, :] = np.nan

    elif method == "relative_change":
        # Make our output a representation of the relative change of the signal
        for i in range(temporal_signals.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[std_indices])
            prestim_profile = np.squeeze(temporal_signals[i, std_indices])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            if np.sum(goodind) >= np.floor(len(goodind)*critical_fraction):
                prestim_mean = np.nanmean(prestim_profile[goodind])
                temporal_signals[i, :] -= prestim_mean
                temporal_signals[i, :] /= prestim_mean
                temporal_signals[i, :] *= 100
            else:
                query_status[i] = "Incomplete signal for standardization (req'd " +str(critical_fraction)+", had " +str(len(goodind)*critical_fraction)+")"
                valid_stdization[i] = False
                temporal_signals[i, :] = np.nan

    elif method == "mean_sub":
        # Make our output just a prestim mean-subtracted signal.
        for i in range(temporal_signals.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[std_indices])
            prestim_profile = np.squeeze(temporal_signals[i, std_indices])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            if np.sum(goodind) >= np.floor(len(goodind)*critical_fraction):
                prestim_mean = np.nanmean(prestim_profile[goodind])
                temporal_signals[i, :] -= prestim_mean
            else:
                query_status[i] = "Incomplete signal for standardization (req'd " +str(critical_fraction)+", had " +str(len(goodind)*critical_fraction)+")"
                valid_stdization[i] = False
                temporal_signals[i, :] = np.nan

    return temporal_signals, valid_stdization, query_status


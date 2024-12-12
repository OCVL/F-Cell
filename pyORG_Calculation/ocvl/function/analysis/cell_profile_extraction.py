import warnings

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

from skimage.morphology import disk
from skimage.morphology.footprints import ellipse, octagon

from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.dataset import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles

def refine_coord(ref_image, coordinates, search_radius=1, numiter=2):

    im_size = ref_image.shape

    # Generate an inclusion list for our coordinates- those that are unanalyzable should be excluded before analysis.
    pluscoord = coordinates + search_radius*2*numiter # Include extra region to avoid edge effects
    includelist = pluscoord[:, 0] < im_size[1]
    includelist &= pluscoord[:, 1] < im_size[0]
    del pluscoord

    minuscoord = coordinates - search_radius*2*numiter # Include extra region to avoid edge effects
    includelist &= minuscoord[:, 0] >= 0
    includelist &= minuscoord[:, 1] >= 0
    del minuscoord

    coordinates = np.round(coordinates[includelist, :]).astype("int")

    for i in range(coordinates.shape[0]):
        if includelist[i]:
            for iter in range(numiter):
                coord = coordinates[i, :]

                ref_template = ref_image[(coord[1] - search_radius):(coord[1] + search_radius + 1),
                                         (coord[0] - search_radius):(coord[0] + search_radius + 1)]

                minV, maxV, minL, maxL = cv2.minMaxLoc(ref_template)

                maxL = np.array(maxL)-search_radius # Make relative to the center.
                # print(coord)
                coordinates[i, :] = coord + maxL
                # print(" to: " + str(coordinates[i, :]))

                # plt.figure(11)
                # plt.clf()
                # plt.imshow(ref_template)
                # plt.plot(maxL[0]+search_radius,maxL[1]+search_radius,"r*")
                # plt.show(block=False)
                # plt.waitforbuttonpress()

                if np.all(maxL == 0):
                    # print("Unchanged. Breaking...")
                    break

            # plt.figure(12)
            # plt.clf()
            # plt.imshow(ref_image)
            # plt.plot(coordinates[i, 0],coordinates[i, 1],"r*")
            # plt.imshow( ref_image[(coordinates[i, 1] - 5):(coordinates[i, 1] + 5 + 1),
            #             (coordinates[i, 0] - 5):(coordinates[i, 0] + 5 + 1)])
            #
            # plt.waitforbuttonpress()



    return coordinates


def refine_coord_to_stack(image_stack, ref_image, coordinates, search_radius=2, threshold=0.3):
    ref_image = ref_image.astype("uint8")
    image_stack = image_stack.astype("uint8")
    #image_stack[image_stack == 0] = np.nan # Anything that is equal to 0 should be excluded from consideration.

    im_size = image_stack.shape

    search_region = 2*search_radius # Include extra region for edge effects

    query_status = np.full(coordinates.shape[0], "Included", dtype=object)

    # Generate an inclusion list for our coordinates- those that are unanalyzable should be excluded from refinement.
    pluscoord = coordinates + search_region
    includelist = pluscoord[:, 0] < im_size[1]
    includelist &= pluscoord[:, 1] < im_size[0]
    query_status[pluscoord[:, 0] < im_size[1]] = "Query loc refinement area outside image bounds (right side)"
    query_status[pluscoord[:, 0] < im_size[0]] = "Query loc refinement area outside image bounds (bottom side)"
    del pluscoord

    minuscoord = coordinates - search_region
    includelist &= minuscoord[:, 0] >= 0
    includelist &= minuscoord[:, 1] >= 0
    query_status[minuscoord[:, 0] >= 0] = "Query loc refinement area outside image bounds (top side)"
    query_status[minuscoord[:, 1] >= 0] = "Query loc refinement area outside image bounds (left side)"
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


def extract_profiles(image_stack, coordinates=None, seg_mask="box", seg_radius=1, summary="mean", sigma=None, display=False):
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
    query_status[pluscoord[:, 0] < im_size[1]] = "Segmentation extended outside image bounds (right side)"
    query_status[pluscoord[:, 1] < im_size[0]] = "Segmentation extended outside image bounds (bottom side)"
    del pluscoord

    minuscoord = coordinates - seg_radius
    includelist &= minuscoord[:, 0] >= 0
    includelist &= minuscoord[:, 1] >= 0
    query_status[minuscoord[:, 0] >= 0] = "Segmentation extended outside image bounds (top side)"
    query_status[minuscoord[:, 1] >= 0] = "Segmentation extended outside image bounds (left side)"
    del minuscoord

    coordinates = np.floor(coordinates).astype("int")

    if summary != "none":
        profile_data = np.full((coordinates.shape[0], im_stack.shape[-1]), np.nan)
    else:
        profile_data = np.full((seg_radius * 2 + 1, seg_radius * 2 + 1,
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

                if summary == "mean":
                    profile_data[i, nani] = np.nan
                    profile_data[i, np.invert(nani)] = np.mean(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "median":
                    profile_data[i, nani] = np.nan
                    profile_data[i, np.invert(nani)] = np.nanmedian(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "sum":
                    profile_data[i, nani] = np.nan
                    profile_data[i, np.invert(nani)] = np.nansum(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "none":

                    profile_data[:, :, nani, i] = 0
                    profile_data[:, :, np.invert(nani), i] = fullcolumn[:, :, np.invert(nani)]

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

                if summary == "mean":
                    profile_data[i, nani] = np.nan
                    profile_data[i, np.invert(nani)] = np.nanmean(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "median":
                    profile_data[i, nani] = np.nan
                    profile_data[i, np.invert(nani)] = np.nanmedian(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "sum":
                    profile_data[i, nani] = np.nan
                    profile_data[i, np.invert(nani)] = np.nansum(coordcolumn[:, np.invert(nani)], axis=0)
                elif summary == "none":

                    profile_data[:, :, nani, i] = 0
                    profile_data[:, :, np.invert(nani), i] = fullcolumn[:, :, np.invert(nani)]

    if display:
        plt.figure(1)
        for i in range(profile_data.shape[0]):
            plt.plot(profile_data[i, :]-profile_data[i, 0])
        plt.show()

    return profile_data, query_status

def exclude_profiles(temporal_profiles, framestamps,
                     critical_region=None, critical_fraction=0.5, require_full_profile=False):
    """
    A bit of code used to remove cells that don't have enough data in the critical region of a signal. This is typically
    surrounding a stimulus.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_data.
    :param critical_region: A set of values containing the critical region of a signal- if a cell doesn't have data here,
                            then drop its entire signal from consideration.
    :param critical_fraction: The percentage of real values required to consider the signal valid.
    :param require_full_profile: Require a full profile instead of merely a fraction of the critical region.
    :return: a NxM numpy matrix of pared-down profiles, where profiles that don't fit the criterion are dropped.
    """

    query_status = np.full(temporal_profiles.shape[0], "Included", dtype=object)

    if critical_region is not None:

        crit_inds = np.where(np.isin(framestamps, critical_region))[0]
        crit_remove = 0
        good_profiles = np.full((temporal_profiles.shape[0], ), True)
        for i in range(temporal_profiles.shape[0]):
            this_fraction = np.sum(~np.isnan(temporal_profiles[i, crit_inds])) / len(critical_region)

            if this_fraction < critical_fraction:
                crit_remove += 1
                temporal_profiles[i, :] = np.nan
                good_profiles[i] = False
                query_status[i] = ("Only had " + str(this_fraction) + " of the required data in the critical region spanning " +
                                   str(critical_region[0]) + " to " + str(critical_region[-1]) + ".")

    if require_full_profile:
        for i in range(temporal_profiles.shape[0]):
            if np.any(~np.isfinite(temporal_profiles[i, :])) and good_profiles[i]:

                temporal_profiles[i, :] = np.nan
                good_profiles[i] = False
                query_status[i] = "Did not have a complete profile (e.g. all framestamps supplied by the function arg"
                crit_remove += 1

    if critical_region is not None or require_full_profile:
        print(str(crit_remove) + "/"+str(temporal_profiles.shape[0])+" cells were cleared due to missing data at stimulus delivery.")

    return temporal_profiles, good_profiles, query_status

def norm_profiles(temporal_profiles, norm_method="mean", rescaled=False, video_ref=None):
    """
    This function normalizes the columns of the data (a single sample of all cells) using a method supplied by the user.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param norm_method: The normalization method chosen by the user. Default is "mean". Options: "mean", "median"
    :param rescaled: Whether or not to keep the data at the original scale (only modulate the numbers in place). Useful
                     if you want the data to stay in the same units. Default: False. Options: True/False
    :param video_ref: A video reference (WxHxM) that can be used for normalization instead of the profile values.

    :return: a NxM numpy matrix of normalized temporal profiles.
    """

    if norm_method == "mean":
        all_norm = np.nanmean(temporal_profiles[:])
        # plt.figure()
        # tmp = np.nanmean(temporal_data, axis=0)
        # plt.plot(tmp/np.amax(tmp))
        if video_ref is None:
            framewise_norm = np.nanmean(temporal_profiles, axis=0)
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
        all_norm = np.nanmedian(temporal_profiles[:])
        if video_ref is None:
            framewise_norm = np.nanmedian(temporal_profiles, axis=0)
        else:
            # Determine each frame's mean.
            framewise_norm = np.empty([video_ref.shape[-1]])
            for f in range(video_ref.shape[-1]):
                frm = video_ref[:, :, f].flatten().astype("float32")
                frm[frm == 0] = np.nan
                framewise_norm[f] = np.nanmedian(frm)
            all_norm = np.nanmean(framewise_norm)

    else:
        all_norm = np.nanmean(temporal_profiles[:])
        if video_ref is None:
            framewise_norm = np.nanmean(temporal_profiles, axis=0)
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
        return np.divide(temporal_profiles, ratio[None, :])
    else:
        return np.divide(temporal_profiles, framewise_norm[None, :])


def standardize_profiles(temporal_profiles, framestamps, stimulus_stamp, method="linear_std", display=False, std_indices=None):
    """
    This function standardizes each temporal profile (here, the rows of the supplied data) according to the provided
    arguments.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_data.
    :param stimulus_stamp: The framestamp at which to limit the standardization. For functional studies, this is often
                            the stimulus framestamp.
    :param method: The method used to standardize. Default is "linear_std", which subtracts a linear fit to
                    each signal before stimulus_stamp, followed by a standardization based on that pre-stamp linear-fit
                    subtracted data. This was used in Cooper et al 2017/2020.
                    Current options include: "linear_std", "linear_vast", "relative_change", and "mean_sub"
    :param std_indices: The range of indices to use when standardizing. Defaults to the full prestimulus range.

    :return: a NxM numpy matrix of standardized temporal profiles.
    """

    if std_indices is None:
        prestimulus_idx = np.where(framestamps <= stimulus_stamp, True, False)
    else:
        prestimulus_idx = std_indices

    if len(prestimulus_idx) == 0:
        warnings.warn("Time before the stimulus framestamp doesn't exist in the provided list! No standardization performed.")
        return temporal_profiles

    if method == "linear_std":
        # Standardize using Autoscaling preceded by a linear fit to remove
        # any residual low-frequency changes
        for i in range(temporal_profiles.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(temporal_profiles[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            if np.sum(goodind) > 5:
                thefit = Polynomial.fit(prestim_frmstmp[goodind], prestim_profile[goodind], deg=1)
                fitvals = thefit(prestim_frmstmp[goodind]) # The values we'll subtract from the profile

                prestim_nofit_mean = np.nanmean(prestim_profile[goodind])
                prestim_mean = np.nanmean(prestim_profile[goodind]-fitvals)
                prestim_std = np.nanstd(prestim_profile[goodind]-fitvals)

                temporal_profiles[i, :] = ((temporal_profiles[i, :] - prestim_nofit_mean) / prestim_std)
            else:
                temporal_profiles[i, :] = np.nan

    elif method == "linear_vast":
        # Standardize using variable stability, or VAST scaling, preceeded by a linear fit:
        # https://www.sciencedirect.com/science/article/pii/S0003267003000941
        # this scaling is defined as autoscaling divided by the CoV.
        for i in range(temporal_profiles.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(temporal_profiles[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            if np.sum(goodind) > 5:
                thefit = Polynomial.fit(prestim_frmstmp[goodind], prestim_profile[goodind], deg=1)
                fitvals = thefit(prestim_frmstmp[goodind]) # The values we'll subtract from the profile

                prestim_nofit_mean = np.nanmean(prestim_profile[goodind])
                prestim_mean = np.nanmean(prestim_profile[goodind]-fitvals)
                prestim_std = np.nanstd(prestim_profile[goodind]-fitvals)

                temporal_profiles[i, :] = ((temporal_profiles[i, :] - prestim_nofit_mean) / prestim_std) / \
                                          (prestim_std / prestim_nofit_mean)
            else:
                temporal_profiles[i, :] = np.nan

    elif method == "relative_change":
        # Make our output a representation of the relative change of the signal
        for i in range(temporal_profiles.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(temporal_profiles[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            prestim_mean = np.nanmean(prestim_profile[goodind])
            temporal_profiles[i, :] -= prestim_mean
            temporal_profiles[i, :] /= prestim_mean
            temporal_profiles[i, :] *= 100

    elif method == "mean_sub":
        # Make our output just a prestim mean-subtracted signal.
        for i in range(temporal_profiles.shape[0]):
            prestim_frmstmp = np.squeeze(framestamps[prestimulus_idx])
            prestim_profile = np.squeeze(temporal_profiles[i, prestimulus_idx])
            goodind = np.isfinite(prestim_profile) # Removes nans, infs, etc.

            if np.sum(goodind) > 5:
                prestim_mean = np.nanmean(prestim_profile[goodind])
                temporal_profiles[i, :] -= prestim_mean
            else:
                temporal_profiles[i, :] = np.nan

    if display:
        plt.figure(1)
        for i in range(temporal_profiles.shape[0]):

            plt.plot(framestamps, temporal_profiles[i, :])
            #plt.waitforbuttonpress()

        plt.show(block=True)

    return temporal_profiles

#
# if __name__ == "__main__":
#     dataset = MEAODataset(
#         "\\\\134.48.93.176\\Raw Study Data\\00-33388\\MEAOSLO1\\20210924\\Functional Processed\\Functional Pipeline\\(-1,-0.2)\\00-33388_20210924_OS_(-1,-0.2)_1x1_939_760nm1_extract_reg_cropped_piped.avi",
#                            analysis_modality="760nm", ref_modality="760nm", stage=PipeStages.PIPELINED)
#
#     dataset.load_pipelined_data()
#
#     iORG_signals = extract_profiles(dataset.video_data, dataset.query_loc)
#     norm_temporal_profiles = norm_profiles(iORG_signals, norm_method="mean")
#     stdize_profiles = standardize_profiles(norm_temporal_profiles, dataset.framestamps, 55, method="mean_sub")
#     stdize_profiles, dataset.framestamps, nummissed = reconstruct_profiles(stdize_profiles, dataset.framestamps)
#
#
#     pop_iORG = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_methods="std", window_size=0)
#     plt.plot(dataset.framestamps, pop_iORG)
#     plt.show()

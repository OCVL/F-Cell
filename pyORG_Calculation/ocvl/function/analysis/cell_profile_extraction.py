import warnings

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles

def refine_coord(ref_image, coordinates, search_radius=2):

    im_size = ref_image.shape

    # Generate an exclusion list for our coordinates- those that are unanalyzable should be excluded before analysis.
    pluscoord = coordinates + search_radius
    excludelist = pluscoord[:, 0] < im_size[1]
    excludelist |= pluscoord[:, 1] < im_size[0]
    del pluscoord

    minuscoord = coordinates - search_radius
    excludelist |= minuscoord[:, 0] > 0
    excludelist |= minuscoord[:, 1] > 0
    del minuscoord

    coordinates = np.round(coordinates[excludelist, :]).astype("int")

    for i in range(coordinates.shape[0]):
        coord = coordinates[i, :]

        ref_template = ref_image[(coord[1] - search_radius):(coord[1] + search_radius + 1),
                                 (coord[0] - search_radius):(coord[0] + search_radius + 1)]

        minV, maxV, minL, maxL = cv2.minMaxLoc(ref_template)

        maxL = np.array(maxL)-search_radius # Make relative to the center.
        coordinates[i, :] = coord + maxL

    return coordinates


def refine_coord_to_stack(image_stack, ref_image, im_range, coordinates, search_radius=3):
    ref_image = ref_image.astype("uint8")
    image_stack = image_stack.astype("uint8")
    #image_stack[image_stack == 0] = np.nan # Anything that is equal to 0 should be excluded from consideration.

    im_size = image_stack.shape

    search_region = 2*search_radius # Include extra region for edge effects

    # Generate an exclusion list for our coordinates- those that are unanalyzable should be excluded before analysis.
    pluscoord = coordinates + search_region
    excludelist = pluscoord[:, 0] < im_size[1]
    excludelist |= pluscoord[:, 1] < im_size[0]
    del pluscoord

    minuscoord = coordinates - search_region
    excludelist |= minuscoord[:, 0] > 0
    excludelist |= minuscoord[:, 1] > 0
    del minuscoord

    coordinates = np.round(coordinates[excludelist, :]).astype("int")

    #search_mask = np.zeros(2*search_region)
    #search_mask[(coord[1] - search_radius):(coord[1] + search_radius + 1),
                #(coord[0] - search_radius):(coord[0] + search_radius + 1)]

    for i in range(coordinates.shape[0]):
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
        # print(coord)
        coordinates[i, :] = coord + maxL
        # print(coordinates[i, :])
        # plt.figure(10)
        # plt.imshow(stack_im)
        # plt.figure(11)
        # plt.imshow(match_reg)
        # plt.show(block=False)
        # print(maxL)
    return coordinates


def extract_profiles(image_stack, coordinates=None, seg_mask="box", seg_radius=1, summary="mean", centroid=None, display=False):
    """
    This function extracts temporal profiles from a 3D matrix, where the first two dimensions are assumed to
    contain data from a single time point (a single image)

    :param image_stack: a YxXxZ numpy matrix, where there are Y rows, X columns, and Z samples.
    :param coordinates: input as X/Y, these mark locations the locations that will be extracted from all S samples.
    :param seg_mask: the mask shape that will be used to extract temporal profiles.
    :param seg_radius: the radius of the mask shape that will be used.
    :param summary: the method used to summarize the area inside the segmentation radius. Default: "mean",
                    Options: "mean", "median"
    :param centroid: precede extraction at each stage with a centroid using a supplied method. Default: None,
                    Options: "voronoi", "simple", "xcorr"

    :return: an NxM numpy matrix with N cells and M temporal samples of some signal.
    """

    if coordinates is None:
        pass # Todo: create coordinates for every position in the image stack.

    image_stack = image_stack.astype("float64")
    image_stack[image_stack == 0] = np.nan # Anything that is equal to 0 should be excluded from consideration.

    im_size = image_stack.shape

    # Generate an exclusion list for our coordinates- those that are unanalyzable should be excluded before analysis.
    pluscoord = coordinates + seg_radius
    excludelist = pluscoord[:, 0] < im_size[1]
    excludelist |= pluscoord[:, 1] < im_size[0]
    del pluscoord

    minuscoord = coordinates - seg_radius
    excludelist |= minuscoord[:, 0] > 0
    excludelist |= minuscoord[:, 1] > 0
    del minuscoord

    coordinates = np.round(coordinates[excludelist, :]).astype("int")

    if summary != "none":
        profile_data = np.empty((coordinates.shape[0], image_stack.shape[-1]))
    else:
        profile_data = np.empty((seg_radius*2 + 1, seg_radius*2 + 1,
                                 image_stack.shape[-1], coordinates.shape[0]))

    if seg_mask == "box": # Handle more in the future...
        for i in range(coordinates.shape[0]):

            coord = coordinates[i, :]
            fullcolumn = image_stack[(coord[1]-seg_radius):(coord[1]+seg_radius+1),
                                      (coord[0]-seg_radius):(coord[0]+seg_radius+1), :]

            coldims = fullcolumn.shape
            coordcolumn = np.reshape(fullcolumn, (coldims[0]*coldims[1], coldims[2]), order="F")
            # No partial columns allowed. If there are nans in the column, wipe it out entirely.
            nani = np.any(np.isnan(coordcolumn), axis=0)
            coordcolumn[:, nani] = np.nan

            if summary == "mean":
                profile_data[i, nani] = np.nan
                profile_data[i, np.invert(nani)] = np.mean(coordcolumn[:, np.invert(nani)], axis=0)
            elif summary == "median":
                profile_data[i, nani] = np.nan
                profile_data[i, np.invert(nani)] = np.nanmedian(coordcolumn[:, np.invert(nani)], axis=0)
            elif summary == "none":
                profile_data[:, :, nani] = 0
                profile_data[:, :, np.invert(nani), i] = fullcolumn[:, :, np.invert(nani)]

    if display:
        plt.figure(1)
        for i in range(profile_data.shape[0]):
            plt.plot(profile_data[i, :]-profile_data[i, 0])

        plt.show(block=False)

    return profile_data

def norm_profiles(temporal_profiles, norm_method="mean", rescaled=False):
    """
    This function normalizes the columns of the data (a single sample of all cells) using a method supplied by the user.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param norm_method: The normalization method chosen by the user. Default is "mean". Options: "mean", "median"
    :param rescaled: Whether or not to keep the data at the original scale (only modulate the numbers in place). Useful
                     if you want the data to stay in the same units. Default: False. Options: True/False

    :return: a NxM numpy matrix of normalized temporal profiles.
    """

    if norm_method == "mean":
        all_norm = np.nanmean(temporal_profiles[:])
        framewise_norm = np.nanmean(temporal_profiles, axis=0)
    elif norm_method == "median":
        all_norm = np.nanmedian(temporal_profiles[:])
        framewise_norm = np.nanmedian(temporal_profiles, axis=0)
    else:
        all_norm = np.nanmean(temporal_profiles[:])
        framewise_norm = np.nanmean(temporal_profiles, axis=0)
        warnings.warn("The \"" + norm_method + "\" normalization type is not recognized. Defaulting to mean.")

    if rescaled: # Provide the option to simply scale the data, instead of keeping it in relative terms
        ratio = framewise_norm / all_norm
        return np.divide(temporal_profiles, ratio[None, :])
    else:
        return np.divide(temporal_profiles, framewise_norm[None, :])


def standardize_profiles(temporal_profiles, framestamps, stimulus_stamp, method="linear_std", display=False):
    """
    This function standardizes each temporal profile (here, the rows of the supplied data) according to the provided
    arguments.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_profiles.
    :param stimulus_stamp: The framestamp at which to limit the standardization. For functional studies, this is often
                            the stimulus framestamp.
    :param method: The method used to standardize. Default is "linear_std", which subtracts a linear fit to
                    each signal before stimulus_stamp, followed by a standardization based on that pre-stamp linear-fit
                    subtracted data. This was used in Cooper et al 2017/2020.
                    Current options include: "linear_std", "linear_vast", "relative_change", and "mean_sub"

    :return: a NxM numpy matrix of standardized temporal profiles.
    """

    prestimulus_idx = np.where(framestamps <= stimulus_stamp, True, False)
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
            plt.clf()
            plt.plot(framestamps, temporal_profiles[i, :])
            plt.waitforbuttonpress()

        plt.show(block=False)



    return temporal_profiles

#
# if __name__ == "__main__":
#     dataset = MEAODataset(
#         "\\\\134.48.93.176\\Raw Study Data\\00-33388\\MEAOSLO1\\20210924\\Functional Processed\\Functional Pipeline\\(-1,-0.2)\\00-33388_20210924_OS_(-1,-0.2)_1x1_939_760nm1_extract_reg_cropped_piped.avi",
#                            analysis_modality="760nm", ref_modality="760nm", stage=PipeStages.PIPELINED)
#
#     dataset.load_pipelined_data()
#
#     temp_profiles = extract_profiles(dataset.video_data, dataset.coord_data)
#     norm_temporal_profiles = norm_profiles(temp_profiles, norm_method="mean")
#     stdize_profiles = standardize_profiles(norm_temporal_profiles, dataset.framestamps, 55, method="mean_sub")
#     stdize_profiles, dataset.framestamps, nummissed = reconstruct_profiles(stdize_profiles, dataset.framestamps)
#
#
#     pop_iORG = signal_power_iORG(stdize_profiles, dataset.framestamps, summary_method="std", window_size=0)
#     plt.plot(dataset.framestamps, pop_iORG)
#     plt.show()

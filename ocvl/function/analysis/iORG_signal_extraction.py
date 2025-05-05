import time
import warnings
from itertools import repeat
from multiprocessing import RawArray, shared_memory, get_context

import cv2

import numpy as np
from colorama import Fore
from joblib._multiprocessing_helpers import mp
from matplotlib import pyplot as plt

from numpy.polynomial import Polynomial
from scipy.spatial import Delaunay

from skimage.morphology import disk

from ocvl.function.analysis.iORG_profile_analyses import summarize_iORG_signals
from ocvl.function.display.iORG_data_display import display_iORGs
from ocvl.function.preprocessing.improc import norm_video
from ocvl.function.utility.json_format_constants import SegmentParams, NormParams, ExclusionParams, STDParams, \
    SummaryParams, PreAnalysisPipeline, DebugParams, DisplayParams, Analysis
from scipy.spatial.distance import pdist, squareform

def extract_n_refine_iorg_signals(dataset, analysis_dat_format, query_loc=None, query_loc_name=None, stimtrain_frame_stamps=None,
                                  thread_pool=None):

    if query_loc is None:
        query_loc= dataset.query_loc[0].copy()
    if query_loc_name is None:
        query_loc_name= ""
    if stimtrain_frame_stamps is None:
        stimtrain_frame_stamps = dataset.stimtrain_frame_stamps
    if thread_pool is None:
        if query_loc.shape[0] <= 2000:
            poolsize = 1
        else:
            chunk_size = 250
            poolsize = query_loc.shape[0] // chunk_size
            poolsize = poolsize if poolsize <= mp.cpu_count() // 2 else mp.cpu_count() // 2
            thread_pool = mp.Pool(processes=poolsize)

    query_status = np.full(query_loc.shape[0], "Included", dtype=object)
    valid_signals = np.full((query_loc.shape[0]), True)

    # Round the query locations
    query_loc = np.round(query_loc.copy())

    analysis_params = analysis_dat_format.get(Analysis.PARAMS)

    # Debug parameters. All of these default to off, unless explicitly flagged on in the json.
    display_params = analysis_dat_format.get(DisplayParams.NAME, dict())
    debug_params = display_params.get(DebugParams.NAME, dict())
    plot_extracted_orgs = debug_params.get(DebugParams.PLOT_POP_EXTRACTED_ORGS, False)
    plot_stdize_orgs = debug_params.get(DebugParams.PLOT_POP_STANDARDIZED_ORGS, False)

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


    # If the refine to ref parameter is set to true, and these query points aren't pixelwise or xor.
    if seg_params.get(SegmentParams.REFINE_TO_REF, True) and query_loc_name != "All Pixels" and seg_shape != "xor":
        query_loc, valid, excl_reason = refine_coord(dataset.avg_image_data, query_loc.copy())
    else:
        excl_reason = np.full(query_loc.shape[0], "Included", dtype=object)
        valid = np.full((query_loc.shape[0]), True)

    # Update our audit path.
    to_update = ~(~valid_signals | valid)  # Use the inverse of implication to find which ones to update.
    valid_signals = valid & valid_signals
    query_status[to_update] = excl_reason[to_update]


    if seg_params.get(SegmentParams.REFINE_TO_VID, True) and query_loc_name != "All Pixels" and seg_shape != "xor":
        query_loc, valid, excl_reason = refine_coord_to_stack(dataset.video_data, dataset.avg_image_data,
                                                              query_loc.copy())
    else:
        excl_reason = np.full(query_loc.shape[0], "Included", dtype=object)
        valid = np.full((query_loc.shape[0]), True)

    # Update our audit path.
    to_update = ~(~valid_signals | valid)  # Use the inverse of implication to find which ones to update.
    valid_signals = valid & valid_signals
    query_status[to_update] = excl_reason[to_update]


    # If not defined, then we default to "auto" which determines it from the spacing of the query points
    # However, if there are too many coordinates (spiking the number
    segmentation_radius = seg_params.get(SegmentParams.RADIUS, "auto")
    if segmentation_radius == "auto" and query_loc_name != "All Pixels":

        tri = Delaunay(query_loc, qhull_options="QJ")

        mindist = np.full((query_loc.shape[0],), np.nan)
        for i in range(query_loc.shape[0]):
            # Found this on stack overflow. Seriously terribly documented function.
            index_pointers, indices = tri.vertex_neighbor_vertices
            result_ids = indices[index_pointers[i]:index_pointers[i + 1]]

            coorddist = pdist(query_loc[result_ids, :], "euclidean")
            coorddist[coorddist == 0] = np.amax(coorddist.flatten())
            mindist[i] = np.amin(coorddist.flatten())

        segmentation_radius = np.round(np.nanmean(mindist) / 4) if np.round(np.nanmean(mindist) / 4) >= 1 else 1

        segmentation_radius = int(segmentation_radius)
        print("Detected segmentation radius: " + str(segmentation_radius))
    elif segmentation_radius != "auto":
        segmentation_radius = int(segmentation_radius)
        print("Chosen segmentation radius: " + str(segmentation_radius))
    else:
        segmentation_radius = 1
        print("Pixelwise segmentation radius: " + str(segmentation_radius))


    # Extract the signals
    iORG_signals, excl_reason, coordinates = extract_signals(dataset.video_data, query_loc.copy(),
                                                seg_radius=segmentation_radius,
                                                seg_mask=seg_shape, summary=seg_summary, pool=thread_pool)
    # If we're doing xor, then replace the query locs and
    # status and with what we determined in the above step.
    if seg_shape == "xor":
        valid_signals = np.any(np.isfinite(iORG_signals), axis=1)
        query_status = excl_reason
        query_loc = coordinates.copy()
        del coordinates
    else:
        # Update our audit path.
        valid = np.any(np.isfinite(iORG_signals), axis=1)
        to_update = ~(~valid_signals | valid)  # Use the inverse of implication to find which ones to update.
        valid_signals = valid & valid_signals
        query_status[to_update] = excl_reason[to_update]

    if plot_extracted_orgs:
        display_iORGs(dataset.framestamps, iORG_signals, query_loc_name,
                      stim_delivery_frms=stimtrain_frame_stamps, framerate=dataset.framerate,
                      figure_label=query_loc_name+" extracted ORG signals", params=debug_params )
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()

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

    iORG_signals, valid, excl_reason = standardize_signals(iORG_signals, dataset.framestamps, std_indices=std_ind,
                                                           method=std_meth, pool=thread_pool)

    # Update our audit path.
    to_update = ~(~valid_signals | valid)  # Use the inverse of implication to find which ones to update.
    valid_signals = valid & valid_signals
    query_status[to_update] = excl_reason[to_update]

    if plot_stdize_orgs:
        display_iORGs(dataset.framestamps, iORG_signals, query_loc_name,
                      stim_delivery_frms=stimtrain_frame_stamps, framerate=dataset.framerate,
                      figure_label="Standardized ORG signals", params=debug_params )
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()


    summarized_iORG, num_signals_per_sample = summarize_iORG_signals(iORG_signals, dataset.framestamps,
                                                                     summary_method=sum_method,
                                                                     window_size=sum_window)

    # If the user has a mask definition, then make sure we invalidate cells outside of it.
    mask_roi = analysis_params.get(PreAnalysisPipeline.MASK_ROI)
    if mask_roi is not None:
        excl_reason = np.full(query_loc.shape[0], "Included", dtype=object)

        r = mask_roi.get("r", -1)
        c = mask_roi.get("c", -1)
        width = mask_roi.get("width", -1)
        height = mask_roi.get("height", -1)
        if r == -1:
            r = 0
        if c == -1:
            c = 0
        if width == -1:
            width = np.amax(query_loc[:, 0])
        if height == -1:
            height = np.amax(query_loc[:, 1])

        # Generate an inclusion list for our coordinates- those that are unanalyzable should be excluded before analysis.
        pluscoord = query_loc.copy()
        pluscoord[:, 0] = pluscoord[:, 0] + c
        pluscoord[:, 1] = pluscoord[:, 1] + r
        valid = pluscoord[:, 0] < width
        valid &= pluscoord[:, 1] < height
        excl_reason[pluscoord[:, 0] >= width] = "Outside of user selected ROI (right side)"
        excl_reason[pluscoord[:, 1] >= height] = "Outside of user selected ROI (bottom side)"
        del pluscoord

        minuscoord = query_loc.copy()
        minuscoord[:, 0] = minuscoord[:, 0] - c
        minuscoord[:, 1] = minuscoord[:, 1] - r
        valid &= minuscoord[:, 0] >= 0
        valid &= minuscoord[:, 1] >= 0
        excl_reason[minuscoord[:, 0] < 0] = "Outside of user selected ROI (left side)"
        excl_reason[minuscoord[:, 1] < 0] = "Outside of user selected ROI (top side)"
        del minuscoord

    else:
        excl_reason = np.full(query_loc.shape[0], "Included", dtype=object)
        valid = np.full((query_loc.shape[0]), True)

    # Update our audit path.
    to_update = ~(~valid_signals | valid)  # Use the inverse of implication to find which ones to update.
    valid_signals = valid & valid_signals
    query_status[to_update] = excl_reason[to_update]


    # Wipe out the signals of the invalid signals.
    iORG_signals[~valid_signals, :] = np.nan
    print(Fore.YELLOW+str(np.sum(~valid_signals)) + "/" + str(valid_signals.shape[0]) + " query locations were removed from consideration.")

    return iORG_signals, summarized_iORG, query_status, query_loc


def refine_coord(ref_image, coordinates, search_radius=1, numiter=2):

    im_size = ref_image.shape

    query_status = np.full(coordinates.shape[0], "Included", dtype=object)

    # Generate an inclusion list for our coordinates- those that are unanalyzable should be excluded before analysis.
    pluscoord = coordinates.copy() + search_radius*2*numiter # Include extra region to avoid edge effects
    includelist = pluscoord[:, 0] < im_size[1]
    includelist &= pluscoord[:, 1] < im_size[0]
    query_status[pluscoord[:, 0] >= im_size[1]] = "Reference refinement area outside image bounds (right side)"
    query_status[pluscoord[:, 1] >= im_size[0]] = "Reference refinement area outside image bounds (bottom side)"
    del pluscoord

    minuscoord = coordinates.copy() - search_radius*2*numiter # Include extra region to avoid edge effects
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
    pluscoord = coordinates.copy() + search_region
    includelist = pluscoord[:, 0] < im_size[1]
    includelist &= pluscoord[:, 1] < im_size[0]
    query_status[pluscoord[:, 0] >= im_size[1]] = "Stack refinement area outside image bounds (right side)"
    query_status[pluscoord[:, 1] >= im_size[0]] = "Stack refinement area outside image bounds (bottom side)"
    del pluscoord

    minuscoord = coordinates.copy() - search_region
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


def extract_signals(image_stack, coordinates=None, seg_mask="box", seg_radius=1, summary="mean", sigma=None, pool=None):
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

    coordinates = np.round(coordinates.copy()).astype("int")

    im_stack_mask = image_stack == 0
    im_stack_mask = cv2.morphologyEx(im_stack_mask.astype("uint8"), cv2.MORPH_OPEN, np.ones((3, 3)),
                                     borderType=cv2.BORDER_CONSTANT, borderValue=1)

    im_stack = image_stack.astype("float32")
    im_stack[im_stack_mask.astype("bool")] = np.nan  # Anything that is outside our main image area should be made a nan.

    if seg_mask == "xor":
        coord_mask = np.zeros(image_stack.shape[0:2], dtype="uint8")
        coord_mask[coordinates[:, 1], coordinates[:, 0]] = 1

        cellradius = disk(seg_radius + 2, dtype=coord_mask.dtype)
        coord_mask = cv2.morphologyEx(coord_mask, cv2.MORPH_DILATE, kernel=cellradius,
                                      borderType=cv2.BORDER_CONSTANT, borderValue=0)

        coordinates = np.fliplr(np.argwhere(coord_mask == 0))

        coord_mask = np.repeat(coord_mask[:, :, None], image_stack.shape[-1], axis=2)

        im_stack[coord_mask.astype("bool")] = np.nan  # Mask out anything around our coordinate points.
        seg_radius = 1 # Make our radius 1 if we're doing xor.


    im_size = im_stack.shape
    if sigma is not None:
        for f in range(im_size[-1]):
            im_stack[..., f] = cv2.GaussianBlur(im_stack[..., f], ksize=(0, 0), sigmaX=sigma)

    query_status = np.full(coordinates.shape[0], "Included", dtype=object)

    pluscoord = coordinates.copy() + seg_radius
    includelist = pluscoord[:, 0] < im_size[1]
    includelist &= pluscoord[:, 1] < im_size[0]
    query_status[pluscoord[:, 0] >= im_size[1]] = "Segmentation outside image bounds (right side)"
    query_status[pluscoord[:, 1] >= im_size[0]] = "Segmentation outside image bounds (bottom side)"
    del pluscoord

    minuscoord = coordinates.copy() - seg_radius
    includelist &= minuscoord[:, 0] >= 0
    includelist &= minuscoord[:, 1] >= 0
    query_status[minuscoord[:, 0] < 0] = "Segmentation outside image bounds (left side)"
    query_status[minuscoord[:, 1] < 0] = "Segmentation outside image bounds (top side)"
    del minuscoord

    coordinates = np.floor(coordinates).astype(np.int32)

    if summary != "none":
        signal_data = np.full((coordinates.shape[0], im_stack.shape[-1]), np.nan)
    else:
        signal_data = np.full((seg_radius * 2 + 1, seg_radius * 2 + 1,
                                 im_stack.shape[-1], coordinates.shape[0]), np.nan)

    # Convert our video and coord lists to shared memory blocks (since we're only reading them)
    shared_vid_block = shared_memory.SharedMemory(name="video", create=True, size=im_stack.nbytes)
    np_video = np.ndarray(im_stack.shape, dtype=np.float32, buffer=shared_vid_block.buf)
    np_video[:] = im_stack[:]

    shared_que_block = shared_memory.SharedMemory(name="query", create=True, size=coordinates.nbytes)
    np_coords = np.ndarray(coordinates.shape, dtype=np.int32, buffer=shared_que_block.buf)
    np_coords[:] = coordinates[:]

    chunk_size = 250
    if pool is None:
        pool = mp.Pool(processes=1)

    goodinds = np.arange(coordinates.shape[0])[includelist]  # Only process the indices that are good.


    if seg_mask == "box" or seg_mask == "xor":
        mapres = pool.imap(_extract_box, zip(goodinds,
                                              repeat(shared_vid_block.name), repeat(im_stack.shape),
                                              repeat(shared_que_block.name), repeat(coordinates.shape),
                                              repeat(seg_radius), repeat(summary)),
                                              chunksize=chunk_size )
    elif seg_mask == "disk":

        mapres = pool.imap(_extract_disk, zip(goodinds,
                                          repeat(shared_vid_block.name), repeat(im_stack.shape),
                                          repeat(shared_que_block.name), repeat(coordinates.shape),
                                          repeat(seg_radius), repeat(summary)),
                           chunksize=chunk_size )

    for i, data, status in mapres:
        signal_data[i, :] = data
        query_status[i] = status

    shared_vid_block.close()
    shared_vid_block.unlink()
    shared_que_block.close()
    shared_que_block.unlink()

    return signal_data, query_status, coordinates

def _extract_box(params):
    i, vid_name, vid_shape, coord_name, coord_shape, seg_radius, summary = params

    signal_data = np.full((vid_shape[-1], ), np.nan)

    shared_vid_block = shared_memory.SharedMemory(name=vid_name)
    video = np.ndarray(vid_shape, dtype=np.float32, buffer=shared_vid_block.buf)

    shared_que_block = shared_memory.SharedMemory(name=coord_name)
    coords = np.ndarray(coord_shape, dtype=np.int32, buffer=shared_que_block.buf)

    coord = coords[i, :]
    fullcolumn = video[(coord[1] - seg_radius):(coord[1] + seg_radius + 1),
                 (coord[0] - seg_radius):(coord[0] + seg_radius + 1), :]

    coldims = fullcolumn.shape
    coordcolumn = np.reshape(fullcolumn.copy(), (coldims[0] * coldims[1], coldims[2]), order="F")
    # print(coord)
    # No partial columns allowed. If there are nans in the column, wipe it out entirely.
    nani = np.any(np.isnan(coordcolumn), axis=0)
    coordcolumn[:, nani] = np.nan

    if np.all(np.isnan(coordcolumn.flatten())):
        query_status = "Missing Data at Query Location"
    else:
        query_status = "Included"

    if summary == "mean":
        signal_data[nani] = np.nan
        signal_data[np.invert(nani)] = np.mean(coordcolumn[:, np.invert(nani)], axis=0)
    elif summary == "median":
        signal_data[nani] = np.nan
        signal_data[np.invert(nani)] = np.nanmedian(coordcolumn[:, np.invert(nani)], axis=0)
    elif summary == "sum":
        signal_data[nani] = np.nan
        signal_data[np.invert(nani)] = np.nansum(coordcolumn[:, np.invert(nani)], axis=0)

    shared_vid_block.close()
    shared_que_block.close()

    return i, signal_data, query_status

def _extract_disk(params):
    i, vid_name, vid_shape, coord_name, coord_shape, seg_radius, summary = params

    signal_data = np.full((vid_shape[-1], ), np.nan)

    shared_vid_block = shared_memory.SharedMemory(name=vid_name)
    video = np.ndarray(vid_shape, dtype=np.float32, buffer=shared_vid_block.buf)

    shared_que_block = shared_memory.SharedMemory(name=coord_name)
    coords = np.ndarray(coord_shape, dtype=np.int32, buffer=shared_que_block.buf)

    coord = coords[i, :]
    fullcolumn = video[(coord[1] - seg_radius):(coord[1] + seg_radius + 1),
                       (coord[0] - seg_radius):(coord[0] + seg_radius + 1), :]
    mask = disk(seg_radius + 1, dtype=fullcolumn.dtype)
    mask = mask[1:-1, 1:-1]
    mask = np.repeat(mask[:, :, None], fullcolumn.shape[-1], axis=2)

    coldims = fullcolumn.shape
    coordcolumn = np.reshape(fullcolumn.copy(), (coldims[0] * coldims[1], coldims[2]), order="F")
    mask = np.reshape(mask, (coldims[0] * coldims[1], coldims[2]), order="F")

    maskedout = np.where(mask == 0)
    coordcolumn[maskedout] = 0  # Areas that are masked shouldn't be considered in the partial column test below.
    # No partial columns allowed. If there are nans in the column, mark it to be wiped out entirely.
    nani = np.any(np.isnan(coordcolumn), axis=0)

    # Make our mask 0s into nans
    mask[mask == 0] = np.nan
    coordcolumn = coordcolumn * mask
    coordcolumn[:, nani] = np.nan

    if np.all(np.isnan(coordcolumn.flatten())):
        query_status = "Missing Data at Query Location"
    else:
        query_status = "Included"

    if summary == "mean":
        signal_data[nani] = np.nan
        signal_data[np.invert(nani)] = np.nanmean(coordcolumn[:, np.invert(nani)], axis=0)
    elif summary == "median":
        signal_data[nani] = np.nan
        signal_data[np.invert(nani)] = np.nanmedian(coordcolumn[:, np.invert(nani)], axis=0)
    elif summary == "sum":
        signal_data[nani] = np.nan
        signal_data[np.invert(nani)] = np.nansum(coordcolumn[:, np.invert(nani)], axis=0)

    shared_vid_block.close()
    shared_que_block.close()

    return i, signal_data, query_status


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

        good_profiles = np.full((temporal_signals.shape[0],), True)
        for i in range(temporal_signals.shape[0]):
            this_fraction = np.sum(~np.isnan(temporal_signals[i, crit_inds])) / len(critical_region)

            if this_fraction < critical_fraction:

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


def standardize_signals(temporal_signals, framestamps, std_indices, method="linear_std", critical_fraction=0.3, pool=None):
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

    req_framenums = int(np.floor(len(std_indices)*critical_fraction))
    prestim_frmstmp, _, std_indices = np.intersect1d(std_indices, framestamps, return_indices=True)

    if len(std_indices) == 0:
        #warnings.warn("Time before the stimulus framestamp doesn't exist in the provided list! No standardization performed.")
        print(Fore.RED+ "Time before the stimulus framestamp doesn't exist in the provided list! No standardization performed for this dataset.")
        query_status = np.full(temporal_signals.shape[0], "No prestimulus data for standardization.", dtype=object)
        valid_stdization = np.full(temporal_signals.shape[0], False, dtype=bool)
        return temporal_signals, valid_stdization, query_status

    stdized_signals = np.full_like(temporal_signals, np.nan)
    query_status = np.full(temporal_signals.shape[0], "Included", dtype=object)
    valid_stdization = np.full(temporal_signals.shape[0], True, dtype=bool)


    shared_block = shared_memory.SharedMemory(name="signals", create=True, size=temporal_signals.nbytes)
    np_temporal = np.ndarray(temporal_signals.shape, dtype=temporal_signals.dtype, buffer=shared_block.buf)
    # Copy data to our shared array.
    np_temporal[:] = temporal_signals[:]


    chunk_size = 250
    if pool is None:
        pool = mp.Pool(processes=1)

    res = None
    if method == "std":
        res = pool.imap(_std, zip(range(temporal_signals.shape[0]), repeat(shared_block.name),
                                      repeat(temporal_signals.shape), repeat(temporal_signals.dtype),
                                      repeat(std_indices), repeat(req_framenums) ),
                                      chunksize=chunk_size)
    elif method == "linear_std":
        # Standardize using Autoscaling preceded by a linear fit to remove
        # any residual low-frequency changes

        res = pool.imap(_linear_std, zip(range(temporal_signals.shape[0]), repeat(shared_block.name),
                                  repeat(temporal_signals.shape), repeat(temporal_signals.dtype),
                                  repeat(std_indices), repeat(req_framenums), repeat(prestim_frmstmp)),
                                  chunksize=chunk_size)

    elif method == "linear_vast":
        # Standardize using variable stability, or VAST scaling, preceeded by a linear fit:
        # https://www.sciencedirect.com/science/article/pii/S0003267003000941
        # this scaling is defined as autoscaling divided by the CoV.
        res = pool.imap(_linear_vast, zip(range(temporal_signals.shape[0]), repeat(shared_block.name),
                                         repeat(temporal_signals.shape), repeat(temporal_signals.dtype),
                                         repeat(std_indices), repeat(req_framenums), repeat(prestim_frmstmp)),
                                         chunksize=chunk_size)

    elif method == "relative_change":
        # Make our output a representation of the relative change of the signal
        res = pool.imap(_relative_change, zip(range(temporal_signals.shape[0]), repeat(shared_block.name),
                                      repeat(temporal_signals.shape), repeat(temporal_signals.dtype),
                                      repeat(std_indices), repeat(req_framenums) ),
                                      chunksize=chunk_size)

    elif method == "mean_sub":

        res = pool.imap(_mean_sub, zip(range(temporal_signals.shape[0]), repeat(shared_block.name),
                                      repeat(temporal_signals.shape), repeat(temporal_signals.dtype),
                                      repeat(std_indices), repeat(req_framenums) ),
                                      chunksize=chunk_size)

    for i, signal, valid, numgoodind in res:
        stdized_signals[i,: ] = signal
        valid_stdization[i] = valid
        if not valid:
            query_status[i] = "Incomplete signal for standardization (req'd " + "{:.2f}".format(critical_fraction) + ", had " + "{:.2f}".format(numgoodind / req_framenums) + ")"

    shared_block.close()
    shared_block.unlink()

    return stdized_signals, valid_stdization, query_status


def _std(params):
    i, mem_name, signal_shape, the_dtype, std_indices, req_framenums = params

    shared_block = shared_memory.SharedMemory(name=mem_name)
    signals = np.ndarray(signal_shape, dtype=the_dtype, buffer=shared_block.buf)

    prestim_profile = np.squeeze(signals[i, std_indices])
    goodind = np.array(np.isfinite(prestim_profile))  # Removes nans, infs, etc.
    numgoodind = np.sum(goodind)

    if numgoodind >= req_framenums:
        prestim_mean = np.nanmean(prestim_profile[goodind])
        prestim_std = np.nanstd(prestim_profile[goodind])

        temporal_signal = ((signals[i, :] - prestim_mean) / prestim_std)
        valid_stdization = True
    else:
        valid_stdization = False
        temporal_signal = np.full_like(signals[i, :], np.nan)

    shared_block.close()
    return i, temporal_signal, valid_stdization, numgoodind

def _linear_std(params):
    i, mem_name, signal_shape, the_dtype, std_indices, req_framenums, prestim_frmstmp = params

    shared_block = shared_memory.SharedMemory(name=mem_name)
    signals = np.ndarray(signal_shape, dtype=the_dtype, buffer=shared_block.buf)

    prestim_profile = np.squeeze(signals[i, std_indices])
    goodind = np.array(np.isfinite(prestim_profile))  # Removes nans, infs, etc.
    numgoodind = np.sum(goodind)

    if numgoodind >= req_framenums:
        thefit = Polynomial.fit(prestim_frmstmp[goodind], prestim_profile[goodind], deg=1)
        fitvals = thefit(prestim_frmstmp[goodind])  # The values we'll subtract from the profile

        prestim_mean = np.nanmean(prestim_profile[goodind])
        prestim_std = np.nanstd(prestim_profile[goodind]-fitvals)

        temporal_signal = ((signals[i, :] - prestim_mean) / prestim_std)
        valid_stdization = True
    else:
        valid_stdization = False
        temporal_signal = np.full_like(signals[i, :], np.nan)

    shared_block.close()
    return i, temporal_signal, valid_stdization, numgoodind

def _linear_vast(params):
    i, mem_name, signal_shape, the_dtype, std_indices, req_framenums, prestim_frmstmp = params

    shared_block = shared_memory.SharedMemory(name=mem_name)
    signals = np.ndarray(signal_shape, dtype=the_dtype, buffer=shared_block.buf)

    prestim_profile = np.squeeze(signals[i, std_indices])
    goodind = np.array(np.isfinite(prestim_profile))  # Removes nans, infs, etc.
    numgoodind = np.sum(goodind)

    if numgoodind >= req_framenums:
        thefit = Polynomial.fit(prestim_frmstmp[goodind], prestim_profile[goodind], deg=1)
        fitvals = thefit(prestim_frmstmp[goodind])  # The values we'll subtract from the profile

        prestim_mean = np.nanmean(prestim_profile[goodind])
        prestim_std = np.nanstd(prestim_profile[goodind]-fitvals)

        temporal_signal = ((signals[i, :] - prestim_mean) / prestim_std) / (prestim_std / prestim_mean)
        valid_stdization = True
    else:
        valid_stdization = False
        temporal_signal = np.full_like(signals[i, :], np.nan)

    shared_block.close()
    return i, temporal_signal, valid_stdization, numgoodind

def _relative_change(params):
    i, mem_name, signal_shape, the_dtype, std_indices, req_framenums = params

    shared_block = shared_memory.SharedMemory(name=mem_name)
    signals = np.ndarray(signal_shape, dtype=the_dtype, buffer=shared_block.buf)

    prestim_profile = np.squeeze(signals[i, std_indices])
    goodind = np.array(np.isfinite(prestim_profile))  # Removes nans, infs, etc.
    numgoodind = np.sum(goodind)

    if numgoodind >= req_framenums:
        prestim_mean = np.nanmean(prestim_profile[goodind])
        temporal_signal = 100* ((signals[i, :] - prestim_mean) / prestim_mean)
        valid_stdization = True
    else:
        valid_stdization = False
        temporal_signal = np.full_like(signals[i, :], np.nan)

    shared_block.close()
    return i, temporal_signal, valid_stdization, numgoodind

def _mean_sub(params):
    i, mem_name, signal_shape, the_dtype, std_indices, req_framenums = params

    shared_block = shared_memory.SharedMemory(name=mem_name)
    signals = np.ndarray(signal_shape, dtype=the_dtype, buffer=shared_block.buf)

    prestim_profile = np.squeeze(signals[i, std_indices])
    goodind = np.array(np.isfinite(prestim_profile))  # Removes nans, infs, etc.
    numgoodind = np.sum(goodind)

    if numgoodind >= req_framenums:
        prestim_mean = np.nanmean(prestim_profile[goodind])
        temporal_signal = signals[i, :] - prestim_mean
        valid_stdization = True
    else:
        valid_stdization = False
        temporal_signal = np.full_like(signals[i, :], np.nan)

    shared_block.close()
    return i, temporal_signal, valid_stdization, numgoodind
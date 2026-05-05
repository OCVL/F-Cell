import logging
import warnings
from itertools import repeat
from multiprocessing import shared_memory
from typing import Tuple, Iterator, Any

import numpy as np
import scipy
from joblib._multiprocessing_helpers import mp
from matplotlib import pyplot as plt
from numpy import ndarray, dtype
from scipy import signal
from scipy.interpolate import Akima1DInterpolator, make_smoothing_spline, interp1d
from scipy.ndimage import center_of_mass, convolve1d
from scipy.signal import savgol_filter, hilbert, envelope

from ocvl.function.utility.json_format_constants import SummaryParams, MetricParams
from scipy.signal import savgol_filter
from scipy.stats import pearsonr


def summarize_iORG_signals(temporal_signals: np.ndarray, framestamps: np.ndarray, summary_method:str ="rms",
                           window_size:int = 1, fraction_thresh:float =0.25, pool:mp.Pool =None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Summarizes the summary on a supplied dataset, using a variety of power based summary methods published in
    Cooper et. al. 2025, Gaffney et. al. 2024, Cooper et. al. 2020, and Cooper et. al. 2017.

    :param temporal_signals: If 2D, an NxM numpy matrix with N cells OR acquisitions from a single cell,
                                and M temporal samples of some signal. If 3D, an NxCxM numpy matrix with N acquisitions
                                from C cells, and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_data.
    :param summary_method: The method used to summarize the population at each sample. Current options include:
                            "rms, "variance", "stddev", "avg", and "envelope". Default: "rms"
    :param window_size: The window size used to summarize the population at each sample. Can be an odd integer from
                        1 (no window) to M/2. Default: 1
    :param fraction_thresh: The fraction of the values inside the sample window that must be finite in order for the power
                            to be calculated- otherwise, the value will be considered np.nan.
    :param pool: A multiprocessing pool object, that can be used for multithreaded operations. Default: None

    :return: a NxM summarized summarized signal.
    """

    chunk_size = 250
    if pool is None:
        pool = mp.Pool(processes=1)

    num_signals = temporal_signals.shape[0]
    num_samples = temporal_signals.shape[-1]

    if window_size != 0:
        if window_size % 2 < 1:
            raise Exception("Window size must be an odd integer.")
        else:
            window_radius = int((window_size - 1) / 2)
    else:
        window_radius = 0

    # If the window radius isn't 0, then densify the matrix, and pad our profiles
    # and densify our matrix (add nans to make sure the signal has a sample for every point).
    temporal_data = None
    shared_block = None
    if window_radius != 0:
        shared_block = shared_memory.SharedMemory(name="signals", create=True, size=temporal_signals.nbytes)

        if len(temporal_signals.shape) == 2:
            temporal_data = np.ndarray((num_signals, 1, framestamps[-1]+1), dtype=temporal_signals.dtype, buffer=shared_block.buf)
            temporal_data[:, 0, framestamps] = temporal_signals

        if len(temporal_signals.shape) == 3:
            temporal_data = np.ndarray((num_signals, temporal_signals.shape[1], framestamps[-1]+1), dtype=temporal_signals.dtype, buffer=shared_block.buf)
            temporal_data[:, :, framestamps] = temporal_signals

        temporal_data = np.pad(temporal_data, ((0, 0), (0, 0), (window_radius, window_radius)), "symmetric")

    else:
        temporal_data = temporal_signals

    if len(temporal_signals.shape) == 2:
        num_incl = np.zeros((1, num_samples), dtype=np.uint32)
        summary = np.full((1, num_samples), np.nan, dtype=np.float32)
    if len(temporal_signals.shape) == 3:
        num_incl = np.zeros((temporal_signals.shape[1], num_samples), dtype=np.uint32)
        summary = np.full((temporal_signals.shape[1], num_samples), np.nan, dtype=np.float32)


    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")

        if summary_method == "variance":

            if window_radius == 0:
                summary = np.nanvar(temporal_data, axis=0)
                num_incl = np.sum(np.isfinite(temporal_data), axis=0)
            elif window_size < (num_samples / 2):

                res = pool.imap(_summary_variance, zip(range(temporal_data.shape[1]), repeat(shared_block.name),
                                                       repeat(temporal_data.shape), repeat(temporal_data.dtype),
                                                       repeat(window_radius), repeat(fraction_thresh)),
                                chunksize=250)
            else:
                raise Exception("Window size must be less than half of the number of samples")
        elif summary_method == "stddev":

            if window_radius == 0:
                summary = np.nanstd(temporal_data, axis=0)
                num_incl = np.sum(np.isfinite(temporal_data), axis=0)
            elif window_size < (num_samples / 2):

                res = pool.imap(_summary_stddev, zip(range(temporal_data.shape[1]), repeat(shared_block.name),
                                                     repeat(temporal_data.shape), repeat(temporal_data.dtype),
                                                     repeat(window_radius), repeat(fraction_thresh)),
                                chunksize=250)
            else:
                raise Exception("Window size must be less than half of the number of samples")
        elif summary_method == "rms":

            if window_radius == 0:
                summary = np.nanmean(np.square(temporal_data), axis=0)  # Average second
                summary = np.sqrt(summary)  # Sqrt last
                num_incl = np.sum(np.isfinite(temporal_data), axis=0)
            elif window_size < (num_samples / 2):

                res = pool.imap(_summary_rms, zip(range(temporal_data.shape[1]), repeat(shared_block.name),
                                              repeat(temporal_data.shape), repeat(temporal_data.dtype),
                                              repeat(window_radius), repeat(fraction_thresh)),
                                              chunksize=250)
            else:
                raise Exception("Window size must be less than half of the number of samples")
        elif summary_method == "mean":

            if window_radius == 0:
                summary = np.nanmean(temporal_data, axis=0)  # Average second
                num_incl = np.sum(np.isfinite(temporal_data), axis=0)
            elif window_size < (num_samples / 2):

                res = pool.imap(_summary_stddev, zip(range(temporal_data.shape[1]), repeat(shared_block.name),
                                                     repeat(temporal_data.shape), repeat(temporal_data.dtype),
                                                     repeat(window_radius), repeat(fraction_thresh)),
                                chunksize=250)
            else:
                raise Exception("Window size must be less than half of the number of samples")
        elif summary_method == "envelope":

            if window_radius == 0:
                for c in range(temporal_data.shape[1]):
                    if np.any(np.isfinite(temporal_data[:, c, :])):
                        padding = int(temporal_data.shape[-1]/2)

                        cell_signals =  temporal_data[:,c,:].copy()
                        cell_signals = iORG_signal_filter(cell_signals, framestamps, filter_type="MS")

                        # for acq_ind in range(cell_signals.shape[0]):
                        #     if np.any(np.isfinite(temporal_data[acq_ind,c, :])):
                        #         plt.figure(f"cell")
                        #         plt.clf()
                        #         plt.plot(temporal_data[acq_ind,c, :])
                        #         plt.plot(cell_signals[acq_ind, :])
                        #         plt.axvline(58, ymin=0, ymax=1, color='k')
                        #         plt.show(block=False)
                        #         plt.waitforbuttonpress()

                        cell_signals = np.pad(cell_signals, ((0, 0), (padding, padding)),
                                               "constant", constant_values=np.nan)

                        for acq_ind in range(cell_signals.shape[0]):
                            finite_window_frms = np.flatnonzero(np.isfinite(cell_signals[acq_ind,:]))

                            if len(finite_window_frms) > fraction_thresh*cell_signals.shape[1]:
                                interper = interp1d(finite_window_frms, cell_signals[acq_ind, finite_window_frms])
                                interpinds = np.arange(start=finite_window_frms[0], stop=finite_window_frms[-1])
                                cell_signals[acq_ind,interpinds] = interper(interpinds)

                                cell_signals[acq_ind, np.isnan(cell_signals[acq_ind, :])] = 0

                                # cell_signals[acq_ind,:] = np.abs(hilbert(cell_signals[acq_ind,:]))
                                # plt.figure(f"allcell")
                                # plt.clf()
                                # plt.plot(cell_signals[acq_ind,:])
                                # plt.plot(np.abs(hilbert(cell_signals[acq_ind,:])))
                                # cell_signals[acq_ind, :] = envelope(cell_signals[acq_ind,:], bp_in=(5, None), residual=None)
                                # plt.plot(cell_signals[acq_ind, :])
                                # plt.show(block=False)
                                # plt.waitforbuttonpress()
                                # print("")
                            else:
                                cell_signals[acq_ind, :] = np.nan

                        # plt.figure(f"allcell")
                        # plt.clf()
                        # plt.plot(cell_signals.transpose())
                        # plt.waitforbuttonpress()

                        cell_signals = cell_signals[:, padding:-padding]

                        if np.any(np.isfinite(cell_signals)):
                            #Envelope RMS
                            # mean prestim subtract

                            prestim_mean = np.nanmean(cell_signals[:, 55:58], axis=1)
                            cell_signals -= prestim_mean[:,np.newaxis]

                            summary[c,:] = np.sqrt(np.nanmean(np.square(cell_signals[:,framestamps]), axis=0))  # Sqrt last
                            num_incl = np.sum(np.isfinite(cell_signals), axis=0)

                            # plt.figure(f"allcell")
                            # plt.clf()
                            #
                            # plt.plot(cell_signals.transpose())
                            # plt.plot(framestamps, summary[c, :], color="k", linewidth=4)
                            # plt.axvline(58, ymin=0, ymax=1, color='k')
                            # #plt.xlim((0,150))
                            # #plt.ylim((-5, 5))
                            # #plt.plot(temporal_data[:,c,:].transpose())
                            # plt.show(block=False)
                            # plt.waitforbuttonpress()


            else:
                print("Awww shit")
                pass
        else:
            raise Exception("Invalid summary_method")

        if window_radius != 0 and window_size < (num_samples / 2):
            for c, summa, num_inc in res:
                summa = summa[window_radius:-window_radius]
                summa = summa[framestamps]
                num_inc = num_inc[window_radius:-window_radius]
                num_inc = num_inc[framestamps]

                summary[c, :] = summa
                num_incl[c] = num_inc

    if window_radius != 0:
        shared_block.close()
        shared_block.unlink()


    return summary, num_incl

def _summary_variance(params: Tuple[int, str, np.ndarray, np.dtype, int, float] ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Summarize a set of signals using variance; designed to be supplied to a multiprocessing pool.

    :param params: a tuple containing: the cell index (int),
                    The name of the SharedMemory buffer storing the temporal signals (str),
                    The shape of the SharedMemory buffer,
                    The datatype of the SharedMemory buffer,
                    The window radius (int),
                    And the fraction of datapoints required for calculating a value at a given position.

    :returns: A tuple containing:
              The cell index that was processed by this thread (int)
              The summarized signal (numpy ndarray)
              How many signals were included at each time lag.
    """

    c, mem_name, signal_shape, the_dtype, window_radius, fraction_thresh = params

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")

        shared_block = shared_memory.SharedMemory(name=mem_name)
        all_signals = np.ndarray(signal_shape, dtype=the_dtype, buffer=shared_block.buf)

        cell_signals = all_signals[:, c, :]
        summary = np.full((cell_signals.shape[1],), np.nan)
        num_incl = np.full((cell_signals.shape[1],), np.nan)

        for i in range(window_radius, cell_signals.shape[1] - window_radius):

            samples = cell_signals[:, (i - window_radius):(i + window_radius + 1)]
            if np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                summary[i] = np.nanvar(samples[:])
                num_incl[i] = np.sum(np.isfinite(samples[:]))

        shared_block.close()

    return c, summary, num_incl

def _summary_stddev(params: Tuple[int, str, np.ndarray, np.dtype, int, float] ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Summarize a set of signals using standard deviation; designed to be supplied to a multiprocessing pool.

    :param params: a tuple containing: the cell index (int),
                    The name of the SharedMemory buffer storing the temporal signals (str),
                    The shape of the SharedMemory buffer,
                    The datatype of the SharedMemory buffer,
                    The window radius (int),
                    And the fraction of datapoints required for calculating a value at a given position.

    :returns: A tuple containing:
              The cell index that was processed by this thread (int)
              The summarized signal (numpy ndarray)
              How many signals were included at each time lag.
    """
    c, mem_name, signal_shape, the_dtype, window_radius, fraction_thresh = params

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")

        shared_block = shared_memory.SharedMemory(name=mem_name)
        all_signals = np.ndarray(signal_shape, dtype=the_dtype, buffer=shared_block.buf)

        cell_signals = all_signals[:, c, :]
        summary = np.full((cell_signals.shape[1],), np.nan)
        num_incl = np.full((cell_signals.shape[1],), np.nan)

        for i in range(window_radius, cell_signals.shape[1] - window_radius):

            samples = cell_signals[:, (i - window_radius):(i + window_radius + 1)]
            if np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                summary[i] = np.nanstd(samples[:])
                num_incl[i] = np.sum(np.isfinite(samples[:]))

        shared_block.close()

    return c, summary, num_incl

def _summary_rms(params: Tuple[int, str, np.ndarray, np.dtype, int, float] ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Summarize a set of signals using RMS; designed to be supplied to a multiprocessing pool.

    :param params: a tuple containing: the cell index (int),
                    The name of the SharedMemory buffer storing the temporal signals (str),
                    The shape of the SharedMemory buffer,
                    The datatype of the SharedMemory buffer,
                    The window radius (int),
                    And the fraction of datapoints required for calculating a value at a given position.

    :returns: A tuple containing:
              The cell index that was processed by this thread (int)
              The summarized signal (numpy ndarray)
              How many signals were included at each time lag.
    """
    c, mem_name, signal_shape, the_dtype, window_radius, fraction_thresh = params

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")

        shared_block = shared_memory.SharedMemory(name=mem_name)
        all_signals = np.ndarray(signal_shape, dtype=the_dtype, buffer=shared_block.buf)

        cell_signals = all_signals[:, c, :]
        summary = np.full((cell_signals.shape[1],), np.nan)
        num_incl = np.full((cell_signals.shape[1],), np.nan)

        cell_signals *= cell_signals  # Square first
        for i in range(window_radius, cell_signals.shape[1] - window_radius):

            samples = cell_signals[:, (i - window_radius):(i + window_radius + 1)]
            if samples[:].size != 0 and np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                summary[i] = np.nanmean(samples[:])  # Average second
                summary[i] = np.sqrt(summary[i])  # Sqrt last
                num_incl[i] = np.sum(np.isfinite(samples[:]))

        shared_block.close()

    return c, summary, num_incl

def _summary_avg(params: Tuple[int, str, np.ndarray, np.dtype, int, float] ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Summarize a set of signals using the average; designed to be supplied to a multiprocessing pool.

    :param params: a tuple containing: the cell index (int),
                    The name of the SharedMemory buffer storing the temporal signals (str),
                    The shape of the SharedMemory buffer,
                    The datatype of the SharedMemory buffer,
                    The window radius (int),
                    And the fraction of datapoints required for calculating a value at a given position.

    :returns: A tuple containing:
              The cell index that was processed by this thread (int)
              The summarized signal (numpy ndarray)
              How many signals were included at each time lag.
    """
    c, mem_name, signal_shape, the_dtype, window_radius, fraction_thresh = params

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")

        shared_block = shared_memory.SharedMemory(name=mem_name)
        all_signals = np.ndarray(signal_shape, dtype=the_dtype, buffer=shared_block.buf)

        cell_signals = all_signals[:, c, :]
        summary = np.full((cell_signals.shape[1],), np.nan)
        num_incl = np.full((cell_signals.shape[1],), np.nan)

        for i in range(window_radius, cell_signals.shape[1] - window_radius):

            samples = cell_signals[:, (i - window_radius):(i + window_radius + 1)]
            if np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                summary[i] = np.nanmean(samples[:])
                num_incl[i] = np.sum(np.isfinite(samples[:]))

        shared_block.close()

    return c, summary, num_incl

def iORG_signal_correlation(stim_iORG_signals, control_iORG_signals) -> Tuple[np.ndarray, np.ndarray]:

    # Pre-allocating
    iORG_corr = np.full(len(stim_iORG_signals), np.nan)
    iORG_corr_p_val = np.full(len(stim_iORG_signals), np.nan)

    for i in range(len(stim_iORG_signals)):
        nan_mask = ~np.isnan(stim_iORG_signals[i]) & ~np.isnan(control_iORG_signals[i])
        iORG_corr[i], iORG_corr_p_val[i] = pearsonr(stim_iORG_signals[i, nan_mask], control_iORG_signals[i, nan_mask])
        del nan_mask # can't preallocate or save for each i since it will change size depending on the number of nan frames per cell...

    return iORG_corr, iORG_corr_p_val


def _determine_pre_n_post_stim_frms(params: dict, stimulus_onset_frmstamp: int, framerate:float = 1.0) -> Tuple[ndarray[Tuple[int], dtype[Any]], ndarray[Tuple[int], dtype[Any]]]:
    """
    Determine the pre- and post- stimulus frames depending on a set of rules defined in a dict supplied by the user.

    :param params: A dictionary defining SummaryParams.UNITS, SummaryParams.MEASURED_TO, SummaryParams.PRESTIM, and SummaryParams.POSTSTIM.
    :param stimulus_onset_frmstamp: The framestamp (e.g the frame number) where the stimulus is delivered for this dataset.
    :param framerate: The framerate of the device. Divides the framestamp if SummaryParams.UNITS is "time".

    :return: A tuple of frames corresponding to all of the prestimulus and poststimulus frames.
    """

    if params is None:
        params = dict()

    metrics_units = params.get(SummaryParams.UNITS, "time")
    metrics_measured_to = params.get(SummaryParams.MEASURED_TO, "stim-relative")
    prestim = np.array(params.get(SummaryParams.PRESTIM, [-1, 0]))
    poststim = np.array(params.get(SummaryParams.POSTSTIM, [0, 1]))

    if metrics_units == "time":
        prestim = np.round(prestim * framerate)
        poststim = np.round(poststim * framerate)
    else:  # if units == "frames":
        prestim = np.round(prestim)
        poststim = np.round(poststim)

    if metrics_measured_to == "stim-relative":
        prestim = stimulus_onset_frmstamp + prestim
        poststim = stimulus_onset_frmstamp + poststim

        # Make the list of indices that should correspond to the pre and post stimulus frames we're analyzing
    if len(prestim) > 1:
        prestim = np.arange(start=prestim[0], stop=prestim[1], step=1, dtype=int)
    else:
        prestim = np.full((1,), prestim[0], dtype=int)

    if len(poststim) > 1:
        poststim = np.arange(start=poststim[0], stop=poststim[1], step=1, dtype=int)
    else:
        poststim = np.full((1,), poststim[0], dtype=int)

    return prestim, poststim


def iORG_signal_metrics(temporal_signals: np.ndarray, framestamps: np.ndarray,
                        framerate:float = 1.0, all_poststim_frms:np.ndarray = None, params:dict = None, pool:mp.pool = None) -> dict:
    """
    Extracts metrics from N signals that are T samples long; metrics include amplitude, implicit time and more.
    Metrics are returned in a dictionary corresponding to the metrics themselves.

    :param temporal_signals: An NxT numpy array storing all the signals to be analyzed.
    :param framestamps: An 1xT numpy int array containing the framestamps of each of the samples in temporal_signals.
    :param framerate: The framerate of the device (float).
    :param all_poststim_frms: The frames within the framestamps that are considered "post stimulus" for the purpose of analysis.
                                Metrics are ONLY extracted from within this window.
    :param params: A dictionary containing the parameters for analysis, such as SMOOTHING for any pre-extraction smoothing,
                   AMPLITUDE_PERCENTILE for the percentile used to isolate the signal amplitude, and TYPE which is a list of
                   metrics that will be extracted from the provided signals.
    :param pool: A multiprocessing pool that can be used to accelerate analyses that require interpolation,
                 such as implicit time.
    :return: A dictionary containing the metrics extracted from the provided signals; each key/value pair corresponds to
             a single metrics.
    """

    if params is None:
        params = dict()

    # Extract our parameters
    smoothing = params.get(SummaryParams.SMOOTHING, None)
    amplitude_percentile = params.get(SummaryParams.AMPLITUDE_PERCENTILE, 0.99)
    metrics_type = params.get(SummaryParams.TYPE, ["amp", "amp_imp_time"])

    desired_prestim_frms, desired_poststim_frms = _determine_pre_n_post_stim_frms(params, all_poststim_frms[0], framerate)

    if temporal_signals.ndim == 1:
        temporal_signals = temporal_signals[None, :]

    finite_data = np.isfinite(temporal_signals)
    logger = logging.getLogger("ORG_Logger")

    result_dict = dict()

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")

        # Find the indexes of the framestamps corresponding to the analyzed pre and post stim frames;
        prestim_window_idx = np.flatnonzero(np.isin(framestamps, desired_prestim_frms))
        poststim_window_idx = np.flatnonzero(np.isin(framestamps, desired_poststim_frms))

        if np.all(~finite_data) or desired_prestim_frms.size == 0 or desired_poststim_frms.size==0 or \
            poststim_window_idx.size == 0 or prestim_window_idx.size == 0 or prestim_window_idx is None or poststim_window_idx is None:
            return result_dict

        chunk_size = 250
        if pool is None:
            pool = mp.Pool(processes=1)

        # This only smooths the signal if smooth is defined.
        if smoothing is not None:
            temporal_signals = iORG_signal_filter(temporal_signals, framestamps, framerate, fwhm_size=21, filter_type=smoothing)

        prestim = temporal_signals[:, prestim_window_idx]
        prestim_frms = framestamps[prestim_window_idx]
        poststim = temporal_signals[:, poststim_window_idx]
        poststim_frms = framestamps[poststim_window_idx]

        prestim_val = np.nanmedian(prestim, axis=1)

        if poststim.size != 1 and poststim.shape[1] != 1:
            poststim_val = np.nanquantile(poststim, [amplitude_percentile], axis=1).flatten()
        else:
            poststim_val = poststim[:, 0]

        # ** Amplitude **
        amplitude = np.abs(poststim_val - prestim_val)

        if MetricParams.AMPLITUDE in metrics_type or MetricParams.HALFAMP_IMPLICIT_TIME in metrics_type:
            result_dict[MetricParams.AMPLITUDE] = amplitude
        if MetricParams.LOG_AMPLITUDE in metrics_type or MetricParams.HALFAMP_IMPLICIT_TIME in metrics_type:
            result_dict[MetricParams.LOG_AMPLITUDE] = np.log(amplitude)

        # ** Recovery percentage **
        final_val = np.nanmean(temporal_signals[:, -5:], axis=1)
        recovery =  ((final_val-prestim_val)-amplitude)/(framestamps[-1]-poststim_frms[0]) #np.abs(((final_val-prestim_val)-amplitude)/amplitude)

        if MetricParams.RECOVERY_PERCENT in metrics_type:
            result_dict[MetricParams.RECOVERY_PERCENT] = recovery

        # ** Area Under the Curve (est. by trapezoidal rule) **
        aur = np.full((temporal_signals.shape[0],), np.nan)

        # ** Area Under the Derivative (est. by trapezoidal rule) **
        aurd = np.full((temporal_signals.shape[0],), np.nan)

        # ** Implicit time **
        amp_implicit_time = np.full_like(amplitude, np.nan)
        halfamp_implicit_time = np.full_like(amplitude, np.nan)

        shared_block = shared_memory.SharedMemory(name="signals", create=True, size=temporal_signals.nbytes)
        np_temporal_signals = np.ndarray(temporal_signals.shape, dtype=temporal_signals.dtype, buffer=shared_block.buf)
        # Copy data to our shared array.
        np_temporal_signals[:] = temporal_signals[:]

        # If we are considering multiple frames, then we can interpolate to determine the precise implicit times.
        # if desired_poststim_frms.size > 1:
        res = pool.imap(_extract_extra_metrics, zip(range(temporal_signals.shape[0]), repeat(shared_block.name),
                                                    repeat(temporal_signals.shape), repeat(temporal_signals.dtype), repeat(framestamps),
                                                    repeat(all_poststim_frms), repeat(desired_poststim_frms), repeat(framerate),
                                                    prestim_val, poststim_val, amplitude),
                        chunksize=chunk_size)

        for i, amp_imp, halfamp_imp, au, aua in res:
            amp_implicit_time[i] = (amp_imp- all_poststim_frms[0]) / framerate
            halfamp_implicit_time[i] = (halfamp_imp - all_poststim_frms[0]) / framerate
            aur[i] = au
            aurd[i] = aua

        if MetricParams.AMP_IMPLICIT_TIME in metrics_type or MetricParams.AMPLITUDE in metrics_type or MetricParams.LOG_AMPLITUDE in metrics_type:
            result_dict[MetricParams.AMP_IMPLICIT_TIME] = amp_implicit_time
        if MetricParams.HALFAMP_IMPLICIT_TIME in metrics_type:
            result_dict[MetricParams.HALFAMP_IMPLICIT_TIME] = halfamp_implicit_time
        if MetricParams.AUR in metrics_type:
            result_dict[MetricParams.AUR] = aur
        if MetricParams.AURD in metrics_type:
            result_dict[MetricParams.AURD] = aurd


        shared_block.close()
        shared_block.unlink()

    return result_dict

def iORG_signal_filter(temporal_signals: ndarray, framestamps:ndarray, framerate:float = 1.0,
                       filter_type:str = None, fwhm_size:int = 14, notch_filter:bool = None, pool:mp.pool = None) -> np.ndarray:
    """
    Filters the iORG signals using a variety of less common, better performing FIR filters, including the Savgol filter,
    the MS and MS1 filters (Schmid et. al.). It can also filter the signals using more common moving mean and spline-based
    filters.

    :param temporal_signals: An NxT numpy array storing all the signals to be analyzed.
    :param framestamps: An 1xT numpy int array containing the framestamps of each of the samples in temporal_signals.
    :param framerate: The framerate of the device (float).
    :param filter_type: A string with the filter name. Currently accepted: "savgol", "MS", "MS1", "movmean", and "spline".
    :param fwhm_size: The full-width at half max target of the filter design. Only used for "savgol", "MS" and "MS1".
    :param notch_filter: Experimental. A notch filter designed to remove breathing artifacts in humans.
    :param pool: A multiprocessing pool that can be used to accelerate analyses that require interpolation,
                 such as implicit time.

    :return: The filtered signals, sharing the same size as the input array (temporal_signals).
    """

    chunk_size = 250
    if pool is None:
        pool = mp.Pool(processes=1)

    finite_data = np.isfinite(temporal_signals)

    # First we filter the data with a notch filter (to possibly remove artifacts from breathing or other things).
    if notch_filter is not None:
        # sos = signal.butter(10, notch_filter, "bandstop", fs=29.5, output='sos')
        # sos = signal.iirdesign([1.45, 2.15], [1.5, 2.1], gpass=1, gstop=60, fs=29.5, output='sos')
        sos = signal.iirdesign([1.35, 2], [1.4, 1.95], gpass=0.1, gstop=60, fs=framerate, output='sos')
        # sos = signal.iirdesign(1.1, 1, gpass=0.1, gstop=60, fs=91, output='sos')
        butter_filtered_profiles = np.full_like(temporal_signals, np.nan)
        for i in range(temporal_signals.shape[0]):
            if np.any(finite_data[i, :]):
                butter_filtered_profiles[i, finite_data[i, :]] = signal.sosfiltfilt(sos, temporal_signals[i, finite_data[i, :]])
    else:
        butter_filtered_profiles = temporal_signals

    # Then we filter the data to remove noise; each of these were tested and worked reasonably well, though MS1 is used
    # currently.
    if filter_type == "savgol":
        filtered_profiles = savgol_filter(butter_filtered_profiles, window_length=fwhm_size, polyorder=4, mode="mirror",
                                          axis=1)
    elif filter_type == "MS":
        # Formulas from Schmid et al- these are MS filters.
        alpha = 4
        n = 6
        m = np.round(fwhm_size * (0.5739 + 0.1850*n + 0.1495*np.log(n)) - 1 ).astype("int") #(filter_size - 1)/2
        x = np.linspace(-m, m, (2*m+1)) / (m + 1)

        window = np.exp(-alpha * (x ** 2)) + np.exp(-alpha * ((x + 2) ** 2)) + np.exp(-alpha * ((x - 2) ** 2)) \
                 - 2 * np.exp(-alpha) - np.exp(-9 * alpha)

        x[int(m)] = 1  # This makes that location invalid- But! No more true_divide error from dividing by zero.
        adj_sinc = np.sin( ((n+4)/2)* np.pi*x ) / (((n+4)/2)*np.pi*x)
        adj_sinc[int(m)] = 0

        if n == 6:
            j = 0
            v = 1
            k = 0.00172 + 0.02437 / (1.64375 - m)**3
            correction = np.zeros_like(adj_sinc)
            for i in range(len(x)):
                correction[i] = np.sum(k * x[i]*np.sin((2*j+v)*np.pi*x[i]))
            adj_sinc += correction

        trunc_sinc = adj_sinc*window
        trunc_sinc /= np.sum(trunc_sinc)

        filtered_profiles = np.full_like(butter_filtered_profiles, np.nan)
        for i in range(temporal_signals.shape[0]):
            filtered_profiles[i, finite_data[i, :]] = convolve1d(butter_filtered_profiles[i, finite_data[i, :]],
                                                                 trunc_sinc, mode="reflect")
            # The below is my attempt at making an nan-ignoring convolution.
            # It works, but doesn't look much different than just dropping the nan values and doing the convolution
            # as usual.
            # padsize = int(len(trunc_sinc) / 2)
            # paddedsig = np.pad(butter_filtered_profiles[i, :], padsize, mode="reflect")
            # for j in range(padsize, temporal_data.shape[1]+padsize):
            #     filtered_profiles[i, j-padsize] = np.nansum(paddedsig[j-padsize:j+padsize+1] * trunc_sinc)


    elif filter_type == "MS1":
        # Formulas from Schmid et al- these are MS1 filters.
        alpha = 4
        n = 4
        m = np.round(fwhm_size * (-0.1516 + 0.2791 * n + 0.2704 *np.log(n)) - 1 ).astype("int")
        x = np.linspace(-m, m, (2*m+1)) / (m + 1)
        window = np.exp(-alpha * (x ** 2)) + np.exp(-alpha * ((x + 2) ** 2)) + np.exp(-alpha * ((x - 2) ** 2)) \
                 - 2 * np.exp(-alpha) - np.exp(-9 * alpha)

        x[int(m)] = 1  # This makes that location invalid- But! No more true_divide error from dividing by zero.
        adj_sinc = np.sin( ((n+2)/2)* np.pi*x ) / (((n+2)/2)*np.pi*x)
        adj_sinc[int(m)] = 0

        if n == 4:
            j = 0
            k = 0.02194 + 0.05028 / (0.76562 - m)**3
            correction = np.zeros_like(adj_sinc)
            for i in range(len(x)):
                correction[i] = np.sum(k * x[i]*np.sin((j+1)*np.pi*x[i]))
            adj_sinc += correction

        trunc_sinc = adj_sinc*window
        trunc_sinc /= np.sum(trunc_sinc)

        filtered_profiles = np.full_like(butter_filtered_profiles, np.nan)
        for i in range(temporal_signals.shape[0]):
            filtered_profiles[i, finite_data[i, :]] = convolve1d(butter_filtered_profiles[i, finite_data[i, :]],
                                                                 trunc_sinc, mode="reflect")
    elif filter_type == "movmean":
        filtered_profiles = scipy.ndimage.convolve1d(butter_filtered_profiles, weights=np.ones((5))/5, axis=1)

    elif filter_type == "spline": # This is found automatically using the Generalized Cross Validation, per scipy.
        filtered_profiles = np.full_like(butter_filtered_profiles, np.nan)

        for c in range(temporal_signals.shape[0]):
            if np.sum(finite_data[c, :])>5:
                filtered_profiles[c,:] = make_smoothing_spline(framestamps[finite_data[c, :]], butter_filtered_profiles[c,finite_data[c, :]])(framestamps)
    else:
        filtered_profiles = butter_filtered_profiles

    return filtered_profiles


def pooled_variance(data:np.ndarray, axis=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the pooled variance of a given dataset- essentially a "weighted variance" approach. Weights are determined
    by counting the number of non-NaN values along the given axis.

    :param data: A ndarray for pooling. NaN values will be excluded.
    :param axis: The axis to perform pooling over.
    :return: A tuple of ndarrays containing  pooled values, less whatever dimension was chosen by the caller,
            and the weights used in the pooling calculation.
    """
    if len(data.shape) == 1:
        return np.zeros_like(data), data

    finers = np.isfinite(data)
    goodrow = np.any(finers, axis=1)

    datamean = np.nanmean(data[goodrow, :], axis=axis)
    datavar = np.nanvar(data[goodrow, :], axis=axis)

    datacount = np.nansum(finers[goodrow, :], axis=axis) - 1 # -1 is Bessels correction

    return np.sum(datavar*datacount) / np.sum(datacount), np.sum(datamean*datacount) / np.sum(datacount)


def _extract_extra_metrics(params: Tuple[int, str, np.ndarray, np.dtype, np.ndarray, np.ndarray,
                                        np.ndarray, float, float, float, float]) -> Tuple[int, float, float, float, float]:
    """
    Extract metrics that may require multithreading to be feasible. Should be run in a multiprocessing pool.

    :param params: a tuple containing: the cell index (int),
                    The name of the SharedMemory buffer storing the temporal signals (str),
                    The shape of the SharedMemory buffer,
                    The datatype of the SharedMemory buffer,
                    The framestamps of the data,
                    The possible poststim frames,
                    The desired poststim frames,
                    The framerate of the data,
                    The prestimulus value of this cell,
                    The poststimulus value of this cell,
                    And the amplitude of this cell.

    :returns: A tuple containing:
              The cell index that was processed by this thread,
              The implicit time,
              Implicit time of half the amplitude,
              Area under the curve,
              And the area of the absolute derivative

    """
    (i, mem_name, signal_shape, the_dtype, framestamps, all_poststim_frms,
     desired_poststim_frms, framerate, prestim_val, poststim_val, amplitude) = params

    shared_block = shared_memory.SharedMemory(name=mem_name)
    temporal_signals = np.ndarray(signal_shape, dtype=the_dtype, buffer=shared_block.buf)

    finite_iORG = np.isfinite(temporal_signals[i, :])
    auc = np.nan
    aurd = np.nan
    amp_implicit_time = np.nan
    halfamp_implicit_time = np.nan

    if np.sum(finite_iORG) > 1:
        valid_auc = False # If we don't have any data past the desired_poststim_frm, we can't analyze auc. Assume no

        finite_window_data = temporal_signals[i, finite_iORG]
        finite_window_frms = framestamps[finite_iORG]

        # if we're missing an *end* framestamp in our window, interpolate to find the value there,
        # and add it temporarily to our signal to make sure things like AUR work correctly.
        if desired_poststim_frms.size > 1 and not np.any(finite_window_frms == desired_poststim_frms[-1]):
            inter_val = np.interp(desired_poststim_frms[-1], finite_window_frms, finite_window_data)
            # Find where to insert the interpolant and its framestamp
            if np.any(finite_window_frms > desired_poststim_frms[-1]):
                next_highest = np.argmax(finite_window_frms > desired_poststim_frms[-1])
                finite_window_data = np.insert(finite_window_data, next_highest, inter_val)
                finite_window_frms = np.insert(finite_window_frms, next_highest, desired_poststim_frms[-1])
                valid_auc = True

        elif np.any(finite_window_frms == desired_poststim_frms[-1]):
            valid_auc = True

        all_poststim_idx = np.flatnonzero(np.isin(finite_window_frms, all_poststim_frms))
        poststim_window_idx = np.flatnonzero(np.isin(finite_window_frms, desired_poststim_frms))

        finite_post_data = finite_window_data[all_poststim_idx]
        finite_post_frms = finite_window_frms[all_poststim_idx]

        if poststim_window_idx.size > 1:
            finite_window_data = finite_window_data[poststim_window_idx]
            finite_window_frms = finite_window_frms[poststim_window_idx]

            if valid_auc:
                auc = np.trapezoid(finite_window_data-finite_window_data[0], x=finite_window_frms / framerate)

                grad_profiles = np.abs(np.gradient(finite_window_data, finite_window_frms / framerate))
                aurd = np.trapezoid(grad_profiles, x=finite_window_frms / framerate)

                # plt.figure(f"auad")
                # plt.clf()
                # plt.subplot(3, 1, 3)
                # plt.plot(finite_window_frms / framerate,finite_window_data)
                # plt.subplot(3,1,2)
                # plt.plot(finite_window_frms / framerate, grad_profiles)
                # plt.subplot(3, 1, 3)
                # plt.plot(finite_window_frms / framerate, cumulative_trapezoid(np.abs(grad_profiles), x=finite_window_frms / framerate, initial=0))
                # plt.show(block=False)
                # plt.waitforbuttonpress()


            amp_val_interp = Akima1DInterpolator(finite_window_frms, finite_window_data - poststim_val, method="makima")

            if amp_val_interp.roots().size != 0:
                amp_implicit_time = amp_val_interp.roots()[0]

        elif poststim_window_idx.size == 1:

            finite_window_data = finite_post_data[all_poststim_idx <= poststim_window_idx[0]]
            finite_window_frms = finite_post_frms[all_poststim_idx <= poststim_window_idx[0]]

            grad_profiles = np.gradient(finite_window_data, finite_window_frms / framerate)

            aurd = np.trapezoid(np.abs(grad_profiles),x=finite_window_frms / framerate)

            amp_implicit_time = desired_poststim_frms[0]


        if finite_post_frms.size > 1:
            halfamp_val_interp = Akima1DInterpolator(finite_post_frms, finite_post_data - ((amplitude / 2) + prestim_val),
                                                     method="makima")
            if halfamp_val_interp.roots().size != 0:
                halfamp_implicit_time = halfamp_val_interp.roots()[0]

    shared_block.close()

    return i, amp_implicit_time, halfamp_implicit_time, auc, aurd
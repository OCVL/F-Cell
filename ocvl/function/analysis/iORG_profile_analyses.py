from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy
from joblib._multiprocessing_helpers import mp
from numpy.fft import fftshift
from scipy import signal
from matplotlib import patches as ptch
from scipy.fft import fft
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import center_of_mass, convolve1d, median_filter
from scipy.signal import savgol_filter, convolve, freqz
from skimage.feature import graycomatrix, graycoprops
from matplotlib import pyplot as plt

from ssqueezepy import wavelets, p2up, cwt
from ssqueezepy.experimental import scale_to_freq

from ocvl.function.utility.resources import save_tiff_stack
from ocvl.function.utility.temporal_signal_utils import densify_temporal_matrix, reconstruct_profiles


def summarize_iORG_signals(temporal_profiles, framestamps, summary_method="var", window_size=1, fraction_thresh=0.25, display=False, stim_idx=None):
    """
    Summarizes the iORG on a supplied dataset, using a variety of power based summary methods published in
    Gaffney et. al. 2024, Cooper et. al. 2020, and Cooper et. al. 2017.

    :param temporal_profiles: A NxM numpy matrix with N cells OR acquisitions from a single cell,
                                and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_data.
    :param summary_method: The method used to summarize the population at each sample. Current options include:
                            "var", "std", and "moving_rms". Default: "var"
    :param window_size: The window size used to summarize the population at each sample. Can be an odd integer from
                        1 (no window) to M/2. Default: 1
    :param fraction_thresh: The fraction of the values inside the sample window that must be finite in order for the power
                            to be calculated- otherwise, the value will be considered np.nan.
    :param display: Display a plot of and pause following calculation of the signal power. (True/False) Default: False
    :param stim_idx: If display is True, this variable stores the stimulus indices and overlays it on the power plots. Default: None

    :return: a 1xM population iORG signal
    """

    temporal_data = temporal_profiles.copy()

    if window_size != 0:
        if window_size % 2 < 1:
            raise Exception("Window size must be an odd integer.")
        else:
            window_radius = int((window_size - 1) / 2)
    else:
        window_radius = 0

    if window_radius != 0:
        # If the window radius isn't 0, then densify the matrix, and pad our profiles
        # Densify our matrix a bit.
        temporal_data = densify_temporal_matrix(temporal_data, framestamps)
        temporal_data = np.pad(temporal_data, ((0, 0), (window_radius, window_radius)), "symmetric")


    num_signals = temporal_data.shape[0]
    num_samples = temporal_data.shape[1]

    num_incl = np.zeros((num_samples))
    iORG = np.empty((num_samples))
    iORG[:] = np.nan

    if summary_method == "var":
        if window_radius == 0:
            iORG = np.nanvar(temporal_data, axis=0)
            num_incl = np.sum(np.isfinite(temporal_data), axis=0)

        elif window_size < (num_samples / 2):

            for i in range(window_radius, num_samples - window_radius):

                samples = temporal_data[:, (i - window_radius):(i + window_radius + 1)]
                if np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                    iORG[i] = np.nanvar(samples[:])
                    num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[window_radius:-window_radius]
            iORG = iORG[framestamps]
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")
    elif summary_method == "std":
        if window_radius == 0:
            iORG = np.nanstd(temporal_data, axis=0)
            num_incl = np.sum(np.isfinite(temporal_data), axis=0)

        elif window_size < (num_samples / 2):

            for i in range(window_radius, num_samples - window_radius):

                samples = temporal_data[:, (i - window_radius):(i + window_radius + 1)]
                if np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                    iORG[i] = np.nanstd(samples[:])
                    num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[window_radius:-window_radius]
            iORG = iORG[framestamps]
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")
    elif summary_method == "rms":
        if window_radius == 0:

            iORG = np.nanmean(temporal_data*temporal_data, axis=0)  # Average second
            iORG = np.sqrt(iORG)  # Sqrt last
            num_incl = np.sum(np.isfinite(temporal_data), axis=0)

        elif window_size < (num_samples / 2):

            temporal_data *= temporal_data  # Square first
            for i in range(window_radius, num_samples - window_radius):

                samples = temporal_data[:, (i - window_radius):(i + window_radius + 1)]
                if samples[:].size != 0 and np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                    iORG[i] = np.nanmean(samples[:])  # Average second
                    iORG[i] = np.sqrt(iORG[i])  # Sqrt last
                # num_incl[i] = np.sum(samples[:] != np.nan)

            iORG = iORG[window_radius:-window_radius]
            iORG = iORG[framestamps]
            # num_incl = num_incl[framestamps]
            num_incl = np.sum(np.isfinite(temporal_data), axis=0)
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")
    elif summary_method == "avg":
        if window_radius == 0:
            iORG = np.nanmean(temporal_data, axis=0)  # Average
            num_incl = np.sum(np.isfinite(temporal_data), axis=0)

        elif window_size < (num_samples / 2):

            for i in range(window_radius, num_samples - window_radius):

                samples = temporal_data[:, (i - window_radius):(i + window_radius + 1)]
                if samples[:].size != 0 and np.sum(np.isfinite(samples[:])) > np.ceil(samples.size * fraction_thresh):
                    iORG[i] = np.nanmean(samples[:])  # Average

            iORG = iORG[window_radius:-window_radius]
            iORG = iORG[framestamps]
            # num_incl = num_incl[framestamps]
            num_incl = np.sum(np.isfinite(temporal_data), axis=0)
            num_incl = num_incl[framestamps]
        else:
            raise Exception("Window size must be less than half of the number of samples")
    else:
        raise Exception("Invalid summary_method")

    if display and np.sum(np.any(np.isfinite(temporal_profiles), axis=1)) >= 1:
        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", temporal_profiles.shape[0]))
        plt.figure(42)
        plt.clf()
        for i in range(temporal_profiles.shape[0]):
            plt.subplot(1, 2, 1)
            plt.title("Raw data")
            plt.plot(framestamps, temporal_profiles[i, :], color=mapper.to_rgba(i, norm=False))
            plt.ylim((-200, 200))
            #plt.xlim((0, framestamps[-1]))
            plt.xlim((0,175))
            #plt.gca().set_box_aspect(1)

        if stim_idx:
            stim_rect = ptch.Rectangle((stim_idx[0], 0),
                                       (stim_idx[1] - stim_idx[0]), 200,
                                       color='gray', alpha=0.5)
        plt.gca().add_patch(stim_rect)
        plt.subplot(1, 2, 2)
        plt.title("Signal power iORG")
        plt.plot(framestamps, iORG, color='black')
        plt.ylim((-200, 200))
        #plt.xlim((0, framestamps[-1]))
        plt.xlim((0,175))
        #plt.gca().set_box_aspect(1)
        if stim_idx:
            stim_rect = ptch.Rectangle((stim_idx[0], 0),
                                      (stim_idx[1] - stim_idx[0]), 200,
                                      color='gray', alpha=0.5)
        plt.gca().add_patch(stim_rect)

        plt.waitforbuttonpress()


    return iORG, num_incl


def wavelet_iORG(temporal_profiles, framestamps, fps, sig_threshold=None, display=False):
    padtype = "reflect"
    temporal_data = temporal_profiles.copy()

    time = framestamps / fps
    the_wavelet = wavelets.Wavelet(("gmw", {"gamma": 3, "beta": 1}))
    # the_wavelet = wavelets.Wavelet(("hhhat", {"mu": 1}))
    # the_wavelet = wavelets.Wavelet(("bump", {"mu": 1, "s": 1.2}))
    # the_wavelet.viz()

    allWx = []
    allScales = np.nan
    coi_im = np.nan

    if display:
        mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis", temporal_data.shape[0]))
        plt.figure(12)



    for r in range(temporal_data.shape[0]):
        if display:
            plt.clf()
            signal = plt.subplot(2, 2, 2)
            waveletd3 = plt.subplot(2, 2, 3)
            waveletthresh = plt.subplot(2, 2, 4)

            signal.plot(time, temporal_data[r, :], color=mapper.to_rgba(r, norm=False), marker="o", markersize=2)

        if np.all(np.isfinite(temporal_data[r, :])):
            Wx, scales = cwt(temporal_data[r, :], wavelet=the_wavelet, t=time, padtype=padtype, scales="log",
                             l1_norm=True, nv=64)
            mod_Wx = np.abs(Wx)
            # Converts our scales to samples, and determines the coi.
            # To convert to Hz:
            # wc [(cycles*radians)/samples] / (2pi [radians]) * fs [samples/second]
            # = fc [cycles/second]

            freq_scales = scale_to_freq(scales, the_wavelet, len(temporal_data[r, :]), fs=fps, padtype=padtype)
            coi_scales = (scales * the_wavelet.std_t) / the_wavelet.wc_ct
            coi_im = np.ones_like(mod_Wx)

            for s, scale_ind in enumerate(coi_scales):
                scale_ind = int(scale_ind + 1)
                coi_im[s, 0:scale_ind] = 0
                coi_im[s, -scale_ind:] = 0

            if sig_threshold is not None:
                overthresh = mod_Wx > sig_threshold
                mod_Wx[mod_Wx <= sig_threshold] = 0
                Wx[mod_Wx <= sig_threshold] = 0

                if display:
                    waveletd3.imshow(mod_Wx)
                    waveletthresh.imshow(np.reshape(overthresh, mod_Wx.shape))  # , extent=(0, framestamps[150], scales[0], scales[-1]))
                    plt.show(block=False)
                    plt.waitforbuttonpress()
            else:
                if display:
                    waveletd3.imshow(mod_Wx)
                    plt.show(block=False)
                    plt.waitforbuttonpress()

            allWx.append(Wx)
            allScales = freq_scales
        else:
            allWx.append(np.nan)

    #        if np.all(~(temporal_data[r, 0:150] == 0)):
    # waveletd3.plot(np.nanvar(np.abs(Wx), axis=1))
    # plt.waitforbuttonpress()

    return allWx, allScales, coi_im


def extract_texture(full_profiles, cellind, numlevels, summary_methods):
    contrast = np.full((1), np.nan)
    dissimilarity = np.full((1), np.nan)
    homogeneity = np.full((1), np.nan)
    energy = np.full((1), np.nan)
    correlation = np.full((1), np.nan)
    glcmmean = np.full((1), np.nan)
    ent = np.full((1), np.nan)
    if "contrast" in summary_methods or "all" in summary_methods:
        contrast = np.full((full_profiles.shape[-2]), np.nan)
    if "dissimilarity" in summary_methods or "all" in summary_methods:
        dissimilarity = np.full((full_profiles.shape[-2]), np.nan)
    if "homogeneity" in summary_methods or "all" in summary_methods:
        homogeneity = np.full((full_profiles.shape[-2]), np.nan)
    if "energy" in summary_methods or "all" in summary_methods:
        energy = np.full((full_profiles.shape[-2]), np.nan)
    if "correlation" in summary_methods or "all" in summary_methods:
        correlation = np.full((full_profiles.shape[-2]), np.nan)
    if "glcmmean" in summary_methods or "all" in summary_methods:
        glcmmean = np.full((full_profiles.shape[-2]), np.nan)
    if "entropy" in summary_methods or "all" in summary_methods:
        ent = np.full((full_profiles.shape[-2]), np.nan)

    avg = np.empty((full_profiles.shape[-2], 1))

    minlvl = 0  # np.nanmin(full_profiles[:, :, :, cellind].flatten())
    maxlvl = 255  # np.nanmax(full_profiles[:, :, :, cellind].flatten())

    thisprofile = np.round(((full_profiles[:, :, :, cellind] - minlvl) / (maxlvl - minlvl)) * (numlevels - 1))
    thisprofile[thisprofile >= numlevels] = numlevels - 1
    thisprofile = thisprofile.astype("uint8")
    # com=[]
    for f in range(full_profiles.shape[-2]):
        avg[f] = np.mean(full_profiles[1:-1, 1:-1, f, cellind].flatten())

        if avg[f] != 0:
            grayco = graycomatrix(thisprofile[:, :, f], distances=[1], angles=[0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                                  levels=numlevels, normed=True, symmetric=True)

            grayco_invar = np.mean(grayco, axis=-1)
            grayco_invar = grayco_invar[..., None]

            if "contrast" in summary_methods or "all" in summary_methods:
                contrast[f] = graycoprops(grayco_invar, prop="contrast")
            if "dissimilarity" in summary_methods or "all" in summary_methods:
                dissimilarity[f] = graycoprops(grayco_invar, prop="dissimilarity")
            if "homogeneity" in summary_methods or "all" in summary_methods:
                homogeneity[f] = graycoprops(grayco_invar, prop="homogeneity")
            if "energy" in summary_methods or "all" in summary_methods:
                energy[f] = graycoprops(grayco_invar, prop="energy")
            if "correlation" in summary_methods or "all" in summary_methods:
                correlation[f] = graycoprops(grayco_invar, prop="correlation")
            if "glcmmean" in summary_methods or "all" in summary_methods:
                I, J = np.ogrid[0:numlevels, 0:numlevels]
                glcmmean[f] = (np.sum(I * np.squeeze(grayco_invar), axis=(0, 1)) + np.sum(J * np.squeeze(grayco_invar),
                                                                                          axis=(0, 1))) / 2
            if "entropy" in summary_methods or "all" in summary_methods:
                loggray = -np.log(grayco_invar, where=grayco_invar > 0)
                loggray[~np.isfinite(loggray)] = 0
                ent[f] = np.sum(grayco_invar * loggray, axis=(0, 1))

    return contrast, dissimilarity, homogeneity, energy, correlation, glcmmean


def extract_texture_profiles(full_profiles, summary_methods=("all"), numlevels=32, framestamps=None, display=False):
    """
    This function extracts textures using GLCM from a set of cell profiles.

    :param full_profiles: A numpy array of shape NxMxFxC, where NxM are the 2D dimensions of the area surrounding the
                          cell coordinate, F is the number of frames in the video, and C is number of cells.
    :param summary_methods: A tuple contraining "contrast", "dissimilarity", "homogeneity", "energy", "correlation",
                           "gclmmean", "entropy", or "all". (Default: "all")
    :param numlevels: The number of graylevels to quantize the input data to. (Default: 32)
    :param framestamps: The framestamps for each frame; used for data display. (Default: None)
    :param display: Enables display of the texture "profile" for each supplied cell. (Default: False)

    :return: A dictionary where each key is the texture name, and each value contains CxF cell texture profiles.
    """

    # Assumes supplied by extract_profiles, which has the below shapes.
    if "contrast" in summary_methods or "all" in summary_methods:
        contrast = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "dissimilarity" in summary_methods or "all" in summary_methods:
        dissimilarity = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "homogeneity" in summary_methods or "all" in summary_methods:
        homogeneity = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "energy" in summary_methods or "all" in summary_methods:
        energy = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "correlation" in summary_methods or "all" in summary_methods:
        correlation = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "glcmmean" in summary_methods or "all" in summary_methods:
        glcmmean = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))
    if "entropy" in summary_methods or "all" in summary_methods:
        entropy = np.empty((full_profiles.shape[-1], full_profiles.shape[-2]))

    retdict = {}

    with Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:

        reconst = pool.starmap_async(extract_texture,
                                     zip(repeat(full_profiles.astype("uint16")), range(full_profiles.shape[-1]),
                                         repeat(numlevels), repeat(summary_methods)))

        res = reconst.get()
        for c, result in enumerate(res):
            if "contrast" in summary_methods or "all" in summary_methods:
                contrast[c, :] = np.array(result[0])
            if "dissimilarity" in summary_methods or "all" in summary_methods:
                dissimilarity[c, :] = np.array(result[1])
            if "homogeneity" in summary_methods or "all" in summary_methods:
                homogeneity[c, :] = np.array(result[2])
            if "energy" in summary_methods or "all" in summary_methods:
                energy[c, :] = np.array(result[3])
            if "correlation" in summary_methods or "all" in summary_methods:
                correlation[c, :] = np.array(result[4])
            if "glcmmean" in summary_methods or "all" in summary_methods:
                glcmmean[c, :] = np.array(result[5])
            if "entropy" in summary_methods or "all" in summary_methods:
                entropy[c, :] = np.array(result[5])

    if "contrast" in summary_methods or "all" in summary_methods:
        retdict["contrast"] = contrast
    if "dissimilarity" in summary_methods or "all" in summary_methods:
        retdict["dissimilarity"] = dissimilarity
    if "homogeneity" in summary_methods or "all" in summary_methods:
        retdict["homogeneity"] = homogeneity
    if "energy" in summary_methods or "all" in summary_methods:
        retdict["energy"] = energy
    if "correlation" in summary_methods or "all" in summary_methods:
        retdict["correlation"] = correlation
    if "glcmmean" in summary_methods or "all" in summary_methods:
        retdict["glcmmean"] = glcmmean
    if "entropy" in summary_methods or "all" in summary_methods:
        retdict["entropy"] = entropy

    if display:
        for c in range(full_profiles.shape[-1]):
            plt.figure(0)
            plt.clf()
            if "glcmmean" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 1)
                plt.title("glcmmean")
                plt.plot(framestamps, glcmmean[c, :], "k")
            if "correlation" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 2)
                plt.title("correlation")
                plt.plot(framestamps, correlation[c, :], "k")
            if "homogeneity" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 3)
                plt.title("homogeneity")
                plt.plot(framestamps, homogeneity[c, :], linestyle="-")
                plt.plot(framestamps, savgol_filter(homogeneity[c, :], 9, 2), "b", linewidth=3)
            if "energy" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 4)
                plt.title("energy")
                plt.plot(framestamps, energy[c, :], linestyle="-")
            if "contrast" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 5)
                plt.title("contrast")
                plt.plot(framestamps, contrast[c, :], linestyle="-")
            if "entropy" in summary_methods or "all" in summary_methods:
                plt.subplot(2, 3, 6)
                plt.title("entropy")
                plt.plot(framestamps, entropy[c, :], linestyle="-")
                plt.show(block=False)
            plt.waitforbuttonpress()

    return retdict

    # Temp: for center of mass calculation.
    #         # com.append(center_of_mass(full_profiles[:, :, f, cellind]) - np.round(full_profiles.shape[0]/2))
    #         # glcmmean[f] = np.sqrt(np.sum(com[f]**2))


def iORG_signal_metrics(temporal_profiles, framestamps, framerate=1,
                        prestim_window_idx=None, poststim_window_idx=None):

    if temporal_profiles.ndim == 1:
        temporal_profiles = temporal_profiles[None, :]

    finite_data = np.isfinite(temporal_profiles)

    if np.all(~finite_data) or len(prestim_window_idx) == 0 or len(poststim_window_idx)==0:
        return np.full((temporal_profiles.shape[0]), np.nan), np.full((temporal_profiles.shape[0]), np.nan), \
               np.full((temporal_profiles.shape[0]), np.nan), np.full(temporal_profiles.shape, np.nan)

    if prestim_window_idx is None:
        prestim_window_idx = np.zeros((1,))

    if poststim_window_idx is None:
        poststim_window_idx = np.arange(1, temporal_profiles.shape[1])

    grad_profiles = np.sqrt( (1/(framerate**2)) + (np.gradient(temporal_profiles, axis=1)**2)) # Don't need to factor in the dx, because it gets removed anyway in the next step.

    pre_abs_diff_profiles = np.abs(grad_profiles[:, prestim_window_idx])
    if np.size(pre_abs_diff_profiles) <=1:
        pre_abs_diff_profiles = np.zeros((1,1))
    cum_pre_abs_diff_profiles = np.nancumsum(pre_abs_diff_profiles, axis=1)

    post_abs_diff_profiles = np.abs(grad_profiles[:, poststim_window_idx])
    cum_post_abs_diff_profiles = np.nancumsum(post_abs_diff_profiles, axis=1)

    cum_pre_abs_diff_profiles[cum_pre_abs_diff_profiles == 0] = np.nan
    cum_post_abs_diff_profiles[cum_post_abs_diff_profiles == 0] = np.nan
    prefad = np.amax(cum_pre_abs_diff_profiles, axis=1)
    postfad = np.amax(cum_post_abs_diff_profiles, axis=1)

    prestim = temporal_profiles[:, prestim_window_idx]
    poststim = temporal_profiles[:, poststim_window_idx]
    poststim_frms = framestamps[poststim_window_idx]

    prestim_val = np.nanmedian(prestim, axis=1)
    poststim_val = np.nanquantile(poststim, [0.99], axis=1).flatten()

    # ** Amplitude **
    amplitude = np.abs(poststim_val - prestim_val)

    # ** Area Under the Response (est. by trapezoidal rule) **
    auc = np.full((temporal_profiles.shape[0], ), np.nan)
    for c in range(temporal_profiles.shape[0]):
        auc[c] = np.trapezoid(poststim[c, np.isfinite(poststim[c,:])], x=poststim_frms[np.isfinite(poststim[c,:])] / framerate)


    final_val = np.nanmean(temporal_profiles[:,-5:], axis=1)

    # ** Recovery percentage **
    recovery = 1 - ((final_val-prestim_val)/amplitude)

    # ** Implicit time **
    implicit_time = np.full_like(amplitude, np.nan)
    for c in range(temporal_profiles.shape[0]):
        whereabove = np.flatnonzero(poststim[c, :] > poststim_val[c])

        if np.any(whereabove) and np.any(np.isfinite(whereabove)):
            implicit_time[c] = (whereabove[0] + poststim_window_idx[0] - prestim_window_idx[-1]) / framerate

    return amplitude, implicit_time, auc, recovery

def iORG_signal_filter(temporal_profiles, framestamps, framerate=1, filter_type=None, fwhm_size=14, notch_filter=None):

    finite_data = np.isfinite(temporal_profiles)

    # First we filter the data with a notch filter (to possibly remove artifacts from breathing or other things.
    if notch_filter is not None:
        # sos = signal.butter(10, notch_filter, "bandstop", fs=29.5, output='sos')
        # sos = signal.iirdesign([1.45, 2.15], [1.5, 2.1], gpass=1, gstop=60, fs=29.5, output='sos')
        sos = signal.iirdesign([1.35, 2], [1.4, 1.95], gpass=0.1, gstop=60, fs=framerate, output='sos')
        # sos = signal.iirdesign(1.1, 1, gpass=0.1, gstop=60, fs=91, output='sos')
        butter_filtered_profiles = np.full_like(temporal_profiles, np.nan)
        for i in range(temporal_profiles.shape[0]):
            if np.any(finite_data[i, :]):
                butter_filtered_profiles[i, finite_data[i, :]] = signal.sosfiltfilt(sos, temporal_profiles[i, finite_data[i, :]])
    else:
        butter_filtered_profiles = temporal_profiles

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
        for i in range(temporal_profiles.shape[0]):
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
        for i in range(temporal_profiles.shape[0]):
            filtered_profiles[i, finite_data[i, :]] = convolve1d(butter_filtered_profiles[i, finite_data[i, :]],
                                                                 trunc_sinc, mode="reflect")
    elif filter_type == "movmean":
        filtered_profiles = scipy.ndimage.convolve1d(butter_filtered_profiles, weights=np.ones((5))/5, axis=1)
    elif filter_type == "none" or filter_type is None:
        filtered_profiles = butter_filtered_profiles

    return filtered_profiles


def pooled_variance(data, axis=1):

    if len(data.shape) == 1:
        return np.zeros_like(data), data

    finers = np.isfinite(data)
    goodrow = np.any(finers, axis=1)

    datamean = np.nanmean(data[goodrow, :], axis=axis)
    datavar = np.nanvar(data[goodrow, :], axis=axis)

    datacount = np.nansum(finers[goodrow, :], axis=axis) - 1 # -1 is Bessels correction

    return np.sum(datavar*datacount) / np.sum(datacount), np.sum(datamean*datacount) / np.sum(datacount)


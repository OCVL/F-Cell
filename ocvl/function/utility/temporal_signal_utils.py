import multiprocessing as mp
from itertools import repeat

import numpy as np
import scipy as sp

from scipy.interpolate import make_lsq_spline
from sklearn import linear_model
from pynufft import NUFFT


def trim_video(video_data, framestamps, new_max_framestamp):

    wheregood = np.where(framestamps <= new_max_framestamp)

    return np.squeeze(video_data[:, :, wheregood]), framestamps[wheregood]

def densify_temporal_matrix(temporal_profiles, framestamps, max_framestamp=None, value=np.nan):
    """
    By default, most algorithms in this package work with sparsely sampled data by default (governed by the framestamps)
    to save what little ram we can. This function takes this sparse representation and makes it "dense" for algorithms
    that rely on temporally coupled samples (like those that use windows)

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_profiles.
    :param max_framestamp: The maximum number of samples we expect in this signal. Defaults to the maximum value in the
                            framestamp array.

    :return: The dense temporal profile matrix, where missing data is filled with "np.nan"s
    """


    if max_framestamp is None:
        max_framestamp = np.max(framestamps)+1

    if len(temporal_profiles.shape) == 1:
        num_signals = 1
        densified_profiles = np.empty(max_framestamp)
        densified_profiles[:] = value
    else:
        num_signals = temporal_profiles.shape[0]
        densified_profiles = np.empty((num_signals, max_framestamp))
        densified_profiles[:] = value

    i=0
    for j in range(max_framestamp):
        if j == framestamps[i]:
            # If j corresponds to the next frame stamp, slot that column in, and increment to the next frame stamp.
            if len(temporal_profiles.shape) == 1:
                densified_profiles[j] = temporal_profiles[i]
            else:
                densified_profiles[:, j] = temporal_profiles[:, i]
            i += 1


    return densified_profiles


def l1_compressed_sensing(temporal_profiles, framestamps, c, threshold=None):
    """
    This function uses

    :param temporal_profiles:
    :param framestamps:
    :param c:
    :return:
    """

    fullrange = np.arange(framestamps[-1] + 1)
    finers = np.isfinite(temporal_profiles[c, :])
    nummissing = int(((framestamps[-1] +1) - len(framestamps)) + np.sum(np.invert(finers)))

    if np.sum(finers) != 0 and (threshold is None or nummissing/framestamps[-1] <= threshold):

        sigmean = np.mean(temporal_profiles[c, finers])
        sigstd = np.std(temporal_profiles[c, finers])

        D = sp.fft.dct(np.eye(framestamps[-1] + 1), norm="ortho", orthogonalize=True)

        A = D[framestamps[finers], :]
        lasso = linear_model.Lasso(alpha=0.001, max_iter=2000)
        lasso.fit(A, (temporal_profiles[c, finers]-sigmean)/sigstd)

        reconstruction = sp.fft.idct(lasso.coef_.reshape((len(fullrange),)), axis=0,
                                     norm="ortho", orthogonalize=True) * sigstd + sigmean

        reconstruction[ reconstruction == 0 ] = np.nan

    else:
        reconstruction = densify_temporal_matrix(temporal_profiles[c, :], framestamps)
        reconstruction[:] = np.nan
        nummissing = np.nan

    return reconstruction, nummissing

def MS1_interp(temporal_profiles, framestamps, c, threshold=None, fwhm_size = 7):

    fullrange = np.arange(framestamps[-1] + 1)
    finers = np.isfinite(temporal_profiles[c, :])
    nummissing = ((framestamps[-1] +1) - len(framestamps)) + np.sum(np.invert(finers))
    reconstruction = np.full((len(fullrange),), np.nan)

    if threshold is None or nummissing/framestamps[-1] <= threshold:

        # Formulas from Schmid et al- these are MS1 filters.
        alpha = 4
        n = 4
        m = np.round(fwhm_size * (-0.1516 + 0.2791 * n + 0.2704 * np.log(n)) - 1).astype("int")

        x = np.linspace(-m, m, (2 * m + 1)) / (m + 1)
        window = np.exp(-alpha * (x ** 2)) + np.exp(-alpha * ((x + 2) ** 2)) + np.exp(-alpha * ((x - 2) ** 2)) \
                 - 2 * np.exp(-alpha) - np.exp(-9 * alpha)

        x[int(m)] = 1 # This makes that location invalid- But! No more true_divide error from dividing by zero.
        adj_sinc = np.sin( ((n + 2) / 2) * np.pi * x) / (((n + 2) / 2) * np.pi * x)
        adj_sinc[int(m)] = 0

        if n == 4:
            j = 0
            k = 0.02194 + 0.05028 / (0.76562 - m) ** 3
            correction = np.zeros_like(adj_sinc)
            for i in range(len(x)):
                correction[i] = np.sum(k * x[i] * np.sin((j + 1) * np.pi * x[i]))
            adj_sinc += correction

        trunc_sinc = adj_sinc * window
        trunc_sinc /= np.sum(trunc_sinc)

        padsize = int(len(trunc_sinc) / 2)

        reconstruction[framestamps] = temporal_profiles[c, :]
        # reconstruction = convolve1d(reconstruction,trunc_sinc, mode="reflect")
        paddedsig = np.pad(reconstruction, padsize, mode="reflect")
        trunc_sinc = np.flip(trunc_sinc)
        for j in range(padsize, len(fullrange)+padsize):
            roi = paddedsig[j - padsize:j + padsize + 1]
            # weightedavg = np.nansum(window*roi)/np.sum(window[np.isfinite(roi)])
            # roi[~np.isfinite(roi)] = weightedavg

            reconstruction[j - padsize] = np.nansum(roi * trunc_sinc)

    else:
        nummissing = np.nan

    return reconstruction, nummissing




def reconstruct_profiles(temporal_profiles, framestamps, method="L1", threshold=0.2, critical_region=None, ms_fwhm=7):
    """
    This function reconstructs the missing profile data using compressed sensing techniques.

    :param temporal_profiles: A NxM numpy matrix with N cells and M temporal samples of some signal.
    :param framestamps: A 1xM numpy matrix containing the associated frame stamps for temporal_profiles.
    :param method: The method used for compressive sensing. Defaults to an L1 norm based approach.
    :param threshold: The threshold at which we will simply drop a signal. As we are not doing true compressive sensing,
                      the default is 0.2 (80% of the signal must be present).

    :return: A tuple containing the reconstructed signals, the full framestamp range of the signals, and the number of
             missing framestamps from each signal.
    """
    fullrange = np.arange(framestamps[-1] + 1)

    reconstruction = np.full((temporal_profiles.shape[0], len(fullrange)), np.nan)
    nummissing = np.full((temporal_profiles.shape[0], 1), np.nan)

    if method == "L1":

        # l1_compressed_sensing( temporal_profiles, framestamps, 0)
        # Create a pool of threads for processing.
        with mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:

            res = pool.starmap(l1_compressed_sensing, zip(repeat(temporal_profiles), repeat(framestamps),
                                                          range(temporal_profiles.shape[0]), repeat(threshold)))

            for c, result in enumerate(res):
                reconstruction[c, :] = np.array(result[0])
                nummissing[c] = float(result[1])
                # if ~np.isnan(nummissing[c]):
                #     plt.figure(66)
                #     plt.clf()
                #     plt.plot(framestamps, temporal_profiles[c, :])
                #     plt.plot(reconstruction[c, :])
                #     plt.waitforbuttonpress()



    elif method == "NUFFT":
        Nufft = NUFFT()

        fsamp = np.arange(0 , framestamps[-1])/ framestamps[-1]
        #om = np.random.randn(temporal_profiles.shape[1]*2, 1)

        for c in range(temporal_profiles.shape[0]):
            finers = np.isfinite(temporal_profiles[c, :])
            good_frms = framestamps[finers, np.newaxis]
            Nufft.plan(good_frms, (int(temporal_profiles[c, finers].shape[-1]),) , (int(good_frms[-1]),), (6,))

            freqy = Nufft.forward(temporal_profiles[c, finers])

            resto = Nufft.solve(freqy, "L1TVOLS", maxiter=30, rho=1)

            # plt.figure(0)
            # plt.subplot(2, 1, 1)
            # plt.plot(framestamps[finers], temporal_profiles[c, finers], "-d")
            # plt.figure(0)
            # plt.subplot(2, 1, 2)
            # plt.plot(fullrange, resto, "-d")
            # plt.waitforbuttonpress()
    elif method == "lsq_spline":

        for c in range(temporal_profiles.shape[0]):

            finers = np.isfinite(temporal_profiles[c, :])

            nummissing = ((framestamps[-1] + 1) - len(framestamps)) + np.sum(np.invert(finers))

            if threshold is None or nummissing / framestamps[-1] <= threshold:

                finite_framestamps = framestamps[finers]
                finite_profiles =  temporal_profiles[c, finers]

                order = 3
                knot_rate = (order+1)
                if critical_region is not None:

                    crit_inds = np.where(np.isin(finite_framestamps, critical_region))[0]

                    preknots = finite_framestamps[range(0, crit_inds[0]-knot_rate, knot_rate)]
                    midknots = finite_framestamps[crit_inds]
                    postknots = finite_framestamps[range(crit_inds[-1]+knot_rate, len(finite_framestamps) , knot_rate)]

                    knots = np.r_[(finite_framestamps[0],) * order,
                                  preknots,
                                  midknots,
                                  postknots,
                                  (finite_framestamps[-1],) * order]
                else:
                    knots = np.r_[(finite_framestamps[0],) * order,
                                  finite_framestamps[range(0, len(finite_framestamps), knot_rate)],
                                  (finite_framestamps[-1],) * order]
                lsqSfit = make_lsq_spline(finite_framestamps, finite_profiles, knots, k=order)

                reconstruction[c, :] = lsqSfit(fullrange)

                # plt.figure(42)
                # plt.clf()
                # plt.plot(framestamps, temporal_profiles[c, :])
                # plt.plot(fullrange, reconstruction[c, :])
                # plt.waitforbuttonpress()

    elif method == "MS1_interp":


        with mp.Pool(processes=int(np.round(mp.cpu_count() / 2))) as pool:

            reconst = pool.starmap_async(MS1_interp, zip(repeat(temporal_profiles), repeat(framestamps),
                                                         range(temporal_profiles.shape[0]), repeat(threshold),
                                                         repeat(ms_fwhm)))
            res = reconst.get()
            for c, result in enumerate(res):
                reconstruction[c, :] = np.array(result[0])
                nummissing[c] = np.array(result[1])

                # if ~np.isnan(nummissing[c]):
                #     plt.figure(66)
                #     plt.clf()
                #     plt.plot(framestamps, temporal_profiles[c, :])
                #     plt.plot(reconstruction[c, :])
                #     plt.waitforbuttonpress()


    print(str(100 * np.nanmean(nummissing) / len(fullrange)) + "% signal reconstructed on average.")

    return reconstruction, fullrange, nummissing
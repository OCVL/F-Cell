from pathlib import Path

import emd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import viridis
from numpy import hanning
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import make_smoothing_spline
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.sparse import diags, eye, spdiags
from scipy.sparse.linalg import spsolve
from vmdpy import VMD

from ocvl.function.analysis.acmd import acmd

# data = pd.read_csv("R:\\00-36828\\FFB_IndivInvest_2020_PRO38673\\MEAOSLO1\\20260218\\Functional - iORG\\Processed\\Functional Pipeline_20260219_10\\(1.5,0.1)\\548nm_0p068\\Results\\20260413_144647\\iORG_Data\\00-36828_548nm_0p068_760nm_indiv_sum_iORG_rms_20260413_144647.csv")
#
# time = data.columns[2:].to_numpy().astype(float)
# temporal_signals = data.iloc[:,2:].to_numpy()
#
#
#
# # for r in range(temporal_signals.shape[0]):
# r = int(input("pick"))
# while not r==-1:
#
#     finite_window_data = temporal_signals[r, :]
#     finite = np.isfinite(finite_window_data)
#     framestamps = np.flatnonzero(finite)
#
#     if np.any(finite) and  np.sum(finite)>5:
#
#         filtered_profiles = make_smoothing_spline(framestamps, finite_window_data[finite] )(framestamps)
#         grad_profiles = np.abs(np.gradient(filtered_profiles, time))
#         aurd = np.trapezoid(grad_profiles, x=time)
#
#         plt.figure(f"auad")
#         plt.clf()
#         plt.subplot(3, 1, 1)
#         plt.plot(time, filtered_profiles, 'k-o')
#         plt.plot(time, finite_window_data)
#         plt.subplot(3, 1, 2)
#         plt.plot(time, grad_profiles)
#         plt.subplot(3, 1, 3)
#         plt.plot(time, cumulative_trapezoid(grad_profiles, x=time, initial=0))
#         plt.show()
#
#
#     r = int(input("pick"))



metricdata = pd.read_csv("R:\\00-36828\\FFB_IndivInvest_2020_PRO38673\\MEAOSLO1\\20260218\\Functional - iORG\\Processed - Rob\\Functional Pipeline_20260414_14\\(1.5,0.1)\\548nm\\Results\\20260417_113014\\00-36828_548nm_760nm_indiv_summary_rms_metrics_ coords_20260417_113014.csv")

sumdata = pd.read_csv("R:\\00-36828\\FFB_IndivInvest_2020_PRO38673\\MEAOSLO1\\20260218\\Functional - iORG\\Processed - Rob\\Functional Pipeline_20260414_14\\(1.5,0.1)\\548nm\\Results\\20260417_113014\\iORG_Data\\00-36828_548nm_760nm_indiv_sum_iORG_rms_20260417_113014.csv")

time = sumdata.columns[2:].to_numpy().astype(float)
sumdata = sumdata.iloc[:,2:].to_numpy()

dat_path = Path("R:\\00-36828\\FFB_IndivInvest_2020_PRO38673\\MEAOSLO1\\20260218\\Functional - iORG\\Processed - Rob\\Functional Pipeline_20260414_14\\(1.5,0.1)\\548nm\\Results\\20260417_113014\\iORG_Data\\")

all_siggies = np.full((20, 2327,176), np.nan)
i=0
for f in dat_path.glob("*_iORGs.csv"):
    data = pd.read_csv( dat_path / f)

    thedats = data.iloc[:,2:].to_numpy()
    all_siggies[i, :, 0:thedats.shape[1]] = thedats
    i+=1

gt = (metricdata.loc[:,"Log Amplitude"] >= 1.5)
lt = metricdata.loc[:,"Log Amplitude"] <=2.5

slice = metricdata.loc[gt & lt]


# --- VMD Parameters ---
alpha = 500      # bandwidth constraint (larger = narrower modes)
tau   = 0.        # noise tolerance (0 for noisy signals)
K     = 3         # number of modes to extract — you must specify this upfront
DC    = 1         # include a DC residual (captures drift) — set to 1
init  = 1         # initialise omegas uniformly
tol   = 1e-7      # convergence tolerance


for c in slice.itertuples():
    ind = c.Index

    # Try and detect signals that do and do not respond from a given trial.
    # squared_sigs = np.square(all_siggies[:, ind, :])
    # prestim_avg_nrg = np.nanmean(squared_sigs[:,29:58], axis=1)
    # poststim_avg_nrg = np.nanmean(squared_sigs[:, 58:87], axis=1)
    #
    # log_nrg_change = np.log10(poststim_avg_nrg)-np.log10(prestim_avg_nrg)
    # low_resp = np.nanquantile(log_nrg_change,0.5)
    #
    # all_summary = np.nanmean(np.square(all_siggies[:, ind, :]), axis=0)
    # all_summary = np.sqrt(all_summary)
    #
    # nolow_summary = np.nanmean(np.square(all_siggies[log_nrg_change>low_resp, ind, :]), axis=0)  # Average second
    # nolow_summary = np.sqrt(nolow_summary)  # Sqrt last
    #
    # # plt.figure()
    # # plt.violinplot(log_nrg_change)
    #
    #plt.figure(f"Cell: {ind} with amplitude: {c.Amplitude}")
    # plt.plot(time, all_siggies[log_nrg_change>low_resp, ind, :].transpose(), "r", linewidth=1)
    # plt.plot(time, all_siggies[log_nrg_change<=low_resp, ind, :].transpose(), "b", linewidth=1)
    # plt.plot(time, all_summary, "k-",linewidth=3)
    # plt.plot(time, nolow_summary, "g-", linewidth=3)
    # plt.ylim((-100, 100))
    # plt.xlim((0, 5))
    # plt.show()

    # Concat all sigs
    # oversample = 4
    #
    # concat_sig = all_siggies[0, ind, :].copy()
    #
    # thissig = concat_sig[np.isfinite(concat_sig)]
    # last_val = thissig[-1]
    #
    # for acq in range(1, 20):
    #     concat_sig = np.append(concat_sig, last_val+all_siggies[acq, ind, :])
    #
    #     thissig = concat_sig[np.isfinite(concat_sig)]
    #     last_val = thissig[-1]
    #
    # full_concat_time = np.arange(0, len(concat_sig)) / (29.4)
    # stim_time = np.arange(1.97, len(concat_sig) / 29.4, step=176 / 29.4)
    #
    # finite_dat = np.isfinite(concat_sig)
    # concat_sig = concat_sig[finite_dat]
    #
    # concat_time = np.arange(0, len(concat_sig))/ 29.4
    #
    #
    # win_size = 28
    # win_fwhm = 10
    # gaus_win = hanning(win_size)
    #
    # SFT= ShortTimeFFT(gaus_win, hop=int(win_size/4), fs=29.4, mfft=win_size*oversample, scale_to='magnitude')
    # t_lo, t_hi = SFT.extent(len(concat_sig))[:2]
    #
    # Sx = SFT.stft(concat_sig)
    #
    # plt.imshow(abs(Sx), origin='lower', aspect='auto', cmap='viridis', extent=SFT.extent(len(concat_sig)))
    #
    # rescaled = concat_sig / np.nanmax(abs(concat_sig))
    #
    # plt.plot(concat_time, (rescaled*5)+10, color='y')
    #
    # IMFs = emd.sift.complete_ensemble_sift(concat_sig, nensembles=100, ensemble_noise=0.2)
    # #emd.plotting.plot_imfs(IMFs, sample_rate=29.4)
    #
    # IF_est, IA_est, s_est = acmd(concat_sig, 29.4, np.full_like(concat_sig, 0), alpha0=1e-7, beta=1e-9)
    # plt.figure("concat freq 0hz")
    # plt.subplot(3, 1, 1)
    # plt.plot(full_concat_time[finite_dat], IA_est * IF_est)
    # plt.plot(full_concat_time[finite_dat], IA_est)
    # plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
    # #plt.ylim((0, 100))
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(full_concat_time[finite_dat], IF_est)
    # plt.plot(stim_time, np.full_like(stim_time, 0), 'g*')
    # plt.ylim((-1, 2))
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(full_concat_time[finite_dat], concat_sig)
    # plt.plot(full_concat_time[finite_dat], s_est)
    # plt.plot(full_concat_time[finite_dat], (IMFs[:, -1]+IMFs[:, -2]))
    # plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
    # #plt.ylim((-150, 150))
    #
    # concat_sig -= s_est
    # #
    # IF_est, IA_est, s_est = acmd(concat_sig, 29.4, np.full_like(concat_sig, 2.2), alpha0=1e-3, beta=1e-4)
    # plt.figure("concat freq 2.5hz")
    # plt.subplot(3, 1, 1)
    # plt.plot(full_concat_time[finite_dat], IA_est * IF_est)
    # plt.plot(full_concat_time[finite_dat], IA_est)
    # plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
    # plt.ylim((0, 100))
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(full_concat_time[finite_dat], IF_est)
    # plt.plot(stim_time, np.full_like(stim_time, 0), 'g*')
    # plt.ylim((-1, 5))
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(full_concat_time[finite_dat], concat_sig)
    # plt.plot(full_concat_time[finite_dat], s_est)
    # plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
    # plt.ylim((-100, 100))
    #
    # plt.show()

    #
    for acq in range(20):
         finite_dat = np.isfinite(all_siggies[acq, ind, :])
         if np.any(finite_dat):
    #         plt.figure(f"Cell #: {ind}, acq: {acq} at ({c[1]}, {c[2]})")
    #         plt.plot(time, all_siggies[acq, ind, :], label=str(acq))
    #
            finite_sig = all_siggies[acq, ind, finite_dat]
            stim_time = 58/29.4
    #
    # #         '''Ensemble EMD'''
    # #         IMFs = emd.sift.complete_ensemble_sift(all_siggies[acq, ind, finite_dat], nensembles=100, ensemble_noise=0.5)
    # #         drift = IMFs[:, -1]
    # #         emd.plotting.plot_imfs(IMFs, sample_rate=29.4)
    #
    #         ''' VMD '''
    #         # if finite_sig.size % 2 != 0:
    #         #     finite_sig = np.append(finite_sig, finite_sig[-1])
    #         #
    #         #     u, u_hat, omega = VMD(finite_sig, alpha, tau, K, DC, init, tol)
    #         #     drift_idx = np.argmin(omega[-1, :])  # omega[-1] = final freq estimates
    #         #     drift = u[drift_idx, :]
    #         #
    #         #     all_siggies[acq, ind, finite_dat] = finite_sig[:-1] - drift[:-1]
    #         # else:
    #         #
    #         #     u, u_hat, omega = VMD(finite_sig, alpha, tau, K, DC, init, tol)
    #         #     drift_idx = np.argmin(omega[-1, :])  # omega[-1] = final freq estimates
    #         #     drift = u[drift_idx, :]
    #         #
    #         #     all_siggies[acq, ind, finite_dat] = finite_sig- drift
    #         #
    #         # plt.figure()
    #         # plt.plot(time[finite_dat], finite_sig)
    #         # plt.plot(time[finite_dat], u[1, :], label=f"Component 1: {acq}")
    #         # plt.plot(time, all_siggies[acq, ind, :], label=f"Detrended: {acq}")
    #         # plt.ylim((-100, 100))
    #         # plt.xlim((0, 5))
    #
    #         '''ACMD'''
    #
            IF_est, IA_est, drift_est = acmd(finite_sig, 29.4, np.full_like(finite_sig, 0.1), alpha0=5e-7, beta=1e-8)
            plt.figure(f"concat freq 0hz with amplitude: {c.Amplitude}")
            plt.subplot(3, 1, 1)
            plt.plot(time[finite_dat], IA_est * IF_est)
            plt.plot(time[finite_dat], IA_est)
            plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
            # plt.ylim((0, 100))

            plt.subplot(3, 1, 2)
            plt.plot(time[finite_dat], IF_est)
            plt.plot(stim_time, np.full_like(stim_time, 0), 'g*')
            plt.ylim((-1, 2))

            plt.subplot(3, 1, 3)
            plt.plot(time[finite_dat], finite_sig)
            plt.plot(time[finite_dat], drift_est)
            plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
            # plt.ylim((-150, 150))

            finite_sig -= drift_est

            IF_est, IA_est, s_est = acmd(finite_sig, 29.4, np.full_like(finite_sig, 3), alpha0=1e-2, beta=1e-4)
            plt.figure(f"concat freq 1hz with amplitude: {c.Amplitude}")
            plt.subplot(3, 1, 1)
            plt.plot(time[finite_dat], IA_est * IF_est)
            plt.plot(time[finite_dat], IA_est)
            plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
            plt.ylim((0, 100))

            plt.subplot(3, 1, 2)
            plt.plot(time[finite_dat], IF_est)
            plt.plot(stim_time, np.full_like(stim_time, 0), 'g*')
            plt.ylim((-1, 5))

            plt.subplot(3, 1, 3)
            plt.plot(time[finite_dat], finite_sig)
            plt.plot(time[finite_dat], s_est)
            plt.plot(time[finite_dat], s_est+drift_est)
            plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
            plt.ylim((-100, 100))

            plt.figure(f"residual: {c.Amplitude}")
            plt.plot(time[finite_dat],finite_sig-s_est)

            # finite_sig -= s_est
            #
            # IF_est, IA_est, s_est = acmd(finite_sig, 29.4, np.full_like(finite_sig, 2), alpha0=1e-4, beta=1e-4)
            # plt.figure("concat freq 2hz")
            # plt.subplot(3, 1, 1)
            # plt.plot(time[finite_dat], IA_est * IF_est)
            # plt.plot(time[finite_dat], IA_est)
            # plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
            # plt.ylim((0, 100))
            #
            # plt.subplot(3, 1, 2)
            # plt.plot(time[finite_dat], IF_est)
            # plt.plot(stim_time, np.full_like(stim_time, 0), 'g*')
            # plt.ylim((-1, 5))
            #
            # plt.subplot(3, 1, 3)
            # plt.plot(time[finite_dat], finite_sig)
            # plt.plot(time[finite_dat], s_est)
            # plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
            # plt.ylim((-100, 100))

            plt.show()

            # IF_est, IA_est, s_est = acmd(finite_sig, 29.4, np.full_like(finite_sig, 5))
            # # plt.figure("amp 3")
            # # plt.plot(time[finite_dat], IA_est)
            # # plt.plot(time[finite_dat], finite_sig / 10)
            # # plt.figure("freq 3")
            # # plt.plot(time[finite_dat], IF_est)
            # # plt.plot(time[finite_dat], finite_sig / 10)
            # plt.figure("sig 3")
            # plt.plot(time[finite_dat], s_est)
            # plt.plot(time[finite_dat], finite_sig)
            # plt.ylim((-100, 100))
            # plt.xlim((0, 5))
            # finite_sig -= s_est
            #
            # IF_est, IA_est, s_est = acmd(finite_sig, 29.4, np.full_like(finite_sig, 10))
            # # plt.figure("amp 3")
            # # plt.plot(time[finite_dat], IA_est)
            # # plt.plot(time[finite_dat], finite_sig / 10)
            # # plt.figure("freq 3")
            # # plt.plot(time[finite_dat], IF_est)
            # # plt.plot(time[finite_dat], finite_sig / 10)
            # plt.figure("sig 4")
            # plt.plot(time[finite_dat], s_est)
            # plt.plot(time[finite_dat], finite_sig)
            # plt.ylim((-100, 100))
            # plt.xlim((0, 5))
    #
    #
    #
    #
    # #
    #
    # #
    # #         # emd.plotting.plot_imfs(IMFs, sample_rate=29.4)
    # #         # eIF_init = np.full(len(finite_sig), 2.0)
    # #
    #         plt.show()
            #plt.waitforbuttonpress()

    # plt.legend()
    # plt.show()



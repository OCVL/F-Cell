from pathlib import Path

import emd
import numpy as np
import pandas as pd
import ssqueezepy
from matplotlib import pyplot as plt
from matplotlib.pyplot import viridis
from numpy import hanning
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import make_smoothing_spline
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.sparse import diags, eye, spdiags
from scipy.sparse.linalg import spsolve
from ssqueezepy import ssq_stft, extract_ridges
from vmdpy import VMD

from ocvl.function.analysis.acmd import acmd

def viz(x, Tf, ridge_idxs, yticks=None, ssq=False, transform='cwt', show_x=True):
    if show_x:
        plt.plot(x, title="x(t)", show=1,
             xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]")

    ylabel = ("Frequency scales [1/Hz]" if (transform == 'cwt' and not ssq) else
              "Frequencies [Hz]")
    title = "abs({}{}) w/ ridge_idxs".format("SSQ_" if ssq else "",
                                             transform.upper())

    ikw = dict(cmap='turbo', yticks=yticks, title=title)
    pkw = dict(linestyle='--', color='k', xlabel="Time [samples]", ylabel=ylabel,
               xlims=(0, Tf.shape[1]))

    plt.figure()
    ssqueezepy.visuals.imshow(Tf, **ikw, show=0)
    ssqueezepy.visuals.plot(ridge_idxs, **pkw, show=1)

if __name__ == "__main__":

    disp_figs = True

    metricdata = pd.read_csv("R:\\00-36828\\FFB_IndivInvest_2020_PRO38673\\MEAOSLO1\\20260218\\Functional - iORG\\Processed - Rob\\Functional Pipeline_20260414_14\\(1.5,0.1)\\548nm\\Results\\20260417_113014\\00-36828_548nm_760nm_indiv_summary_rms_metrics_ coords_20260417_113014.csv")

    sumdata = pd.read_csv("R:\\00-36828\\FFB_IndivInvest_2020_PRO38673\\MEAOSLO1\\20260218\\Functional - iORG\\Processed - Rob\\Functional Pipeline_20260414_14\\(1.5,0.1)\\548nm\\Results\\20260417_113014\\iORG_Data\\00-36828_548nm_760nm_indiv_sum_iORG_rms_20260417_113014.csv")

    time = sumdata.columns[2:].to_numpy().astype(float)
    time_window = time[58:73]
    sumdata = sumdata.iloc[:,2:].to_numpy()

    dat_path = Path("R:\\00-36828\\FFB_IndivInvest_2020_PRO38673\\MEAOSLO1\\20260218\\Functional - iORG\\Processed - Rob\\Functional Pipeline_20260414_14\\(1.5,0.1)\\548nm\\Results\\20260417_113014\\iORG_Data\\")

    all_siggies = np.full((20, 2327,176), np.nan)
    i=0
    for f in dat_path.glob("*_iORGs.csv"):
        data = pd.read_csv( dat_path / f)

        thedats = data.iloc[:,2:].to_numpy()
        all_siggies[i, :, 0:thedats.shape[1]] = thedats
        i+=1

    gt = (metricdata.loc[:,"Log Amplitude"] >= 1.1)
    lt = metricdata.loc[:,"Log Amplitude"] <=1.3

    slice = metricdata.loc[gt & lt]


    plt.hist(metricdata.loc[:,"Log Amplitude"], bins=50)
    plt.show()

    framerate = 29.4

    all_amps = np.full((2327,), np.nan)
    all_phi = np.full_like(all_amps, np.nan)

    for c in slice.itertuples():
        ind = c.Index
        print(ind)
        if ind >= 0:
            cell_amps = np.full((20, 176), np.nan)
            cell_phi = np.full((20, 176), np.nan)
            cell_aurd = np.full((20, ), np.nan)
            decomp_signals = np.full((20, 176), np.nan)
            aurd_signals = np.full((20, 176), np.nan)


            for acq in range(20):
                finite_dat = np.isfinite(all_siggies[acq, ind, :])
                if np.any(finite_dat):
                    finite_sig = all_siggies[acq, ind, finite_dat]
                    stim_time = 58/framerate
                    finite_time = time[finite_dat]
                    '''
                    First, I tried Ensemble EMD, then VMD, and settled on adaptive chirp mode decomposition (ACMD) because
                    it can handle rapidly varying frequencies (which our signals do).
                    '''

                    IF_est, IA_est, drift_est = acmd(finite_sig, framerate, np.full_like(finite_sig, 0.1), alpha0=5e-7, beta=1e-8)
                    if disp_figs:
                        plt.figure(f"concat freq 0hz with amplitude: {c.Amplitude}")
                        # plt.subplot(3, 1, 1)
                        # plt.plot(finite_time, IA_est * IF_est)
                        # plt.plot(finite_time, IA_est)
                        # plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
                        # # plt.ylim((0, 100))
                        #
                        # plt.subplot(3, 1, 2)
                        # plt.plot(finite_time, IF_est)
                        # plt.plot(stim_time, np.full_like(stim_time, 0), 'g*')
                        # plt.ylim((-1, 2))
                        #
                        # plt.subplot(3, 1, 3)
                        plt.plot(finite_time, finite_sig)
                        plt.plot(finite_time, drift_est)
                        # plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
                        # plt.ylim((-150, 150))

                    finite_sig -= drift_est

                    IF_est, IA_est, s_est = acmd(finite_sig, framerate, np.full_like(finite_sig, 3), alpha0=1e-2, beta=0.5e-4)

                    cell_amps[acq, finite_dat] = IA_est

                    decomp_signals[acq, finite_dat] = s_est

                    if disp_figs:
                        plt.figure(f"concat freq 1hz with amplitude: {c.Amplitude:.2f}")
                        plt.subplot(3, 1, 1)
                        # plt.plot(finite_time, IA_est * IF_est)
                        plt.plot(finite_time, IA_est)
                        # plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
                        # plt.ylim((0, 100))
                        #
                        plt.subplot(3, 1, 2)
                        plt.plot(finite_time, IF_est)
                        # plt.plot(stim_time, np.full_like(stim_time, 0), 'g*')
                        # plt.ylim((-1, 5))
                        #
                        plt.subplot(3, 1, 3)
                        plt.plot(finite_time, finite_sig)
                        plt.plot(finite_time, s_est)
                        # plt.plot(finite_time, s_est+drift_est)
                        plt.plot(stim_time, np.full_like(stim_time, 50), 'g*')
                        plt.ylim((-100, 100))




                        win_size = 28
                        win_fwhm = 10
                        win = hanning(win_size)

                        Tsx, Sx, ssq_freqs_s, Sfs, *_ = ssq_stft(finite_sig, window=win, fs=29.4)

                        skw = dict(penalty=0.1, n_ridges=1, transform='stft')
                        stft_ridges = extract_ridges(Sx, Sfs, bw=win_fwhm, **skw)



                        # viz(finite_sig, Sx, stft_ridges, Sfs, ssq=0, transform='stft', show_x=0)

                        SFT= ShortTimeFFT(win, hop=1, fs=29.4, mfft=win_size*4, scale_to='magnitude')
                        t_lo, t_hi = SFT.extent(len(finite_sig))[:2]

                        # Sx = SFT.stft(finite_sig)
                        plt.figure("dft")
                        plt.imshow(abs(Sx), origin='lower', aspect='auto', cmap='viridis', extent=SFT.extent(len(finite_sig)))

                        rescaled = finite_sig / np.nanmax(abs(finite_sig))

                        plt.plot(finite_time, (rescaled*5)+10, color='y')
                        plt.plot(finite_time, Sfs[stft_ridges].flatten(), 'y-', linewidth=3)



                    timeidx = np.flatnonzero(np.isin(finite_time, time_window))
                    if timeidx.size > 5:

                        cum_phase = 2 * np.pi * cumulative_trapezoid(IF_est, x=finite_time, initial=0)
                        cell_phi[acq, finite_dat] = cum_phase

                        grad_profiles = np.abs(np.gradient(s_est[timeidx], finite_time[timeidx]))
                        cell_aurd[acq] = np.trapezoid(grad_profiles, x=finite_time[timeidx])

                        if disp_figs:
                            plt.figure(f"phase_reconstruction test")
                            plt.plot(finite_time, cell_phi[acq, finite_dat], label=f"#{acq}")
                            plt.xlim((1.9, 3))
                            plt.legend()
                            plt.figure(f"aur test amp")
                            plt.plot(grad_profiles)
                            plt.show()

            timeidx = np.flatnonzero(np.isin(time, time_window))

            all_phi[ind] = np.nanmean(cell_phi[:,timeidx[-1]]-cell_phi[:,timeidx[0]])
            all_amps[ind] = np.nanmean(cell_aurd)

            print(f"Ind: {ind}, OG amplitude: {c.Amplitude:.2f}; Final Phase: {all_phi[ind]:.2f}; Final AUR: {all_amps[ind]:.2f} ")

            if disp_figs:
                plt.figure(f"concat freq 1hz with amplitude: {c.Amplitude:.2f}")
                plt.subplot(3, 1, 1)
                plt.plot(time, np.nanmean(cell_amps, axis=0), 'k-', linewidth=3)

                plt.figure(f"phase_reconstruction test")
                plt.plot(time_window, np.nanmean(cell_phi, axis=0), 'k-', linewidth=3)

                plt.show()
    plt.figure("amplitudes")
    plt.hist(all_amps, bins=100)

    plt.figure("phases")
    plt.hist(all_phi, bins=100)
    plt.show()


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


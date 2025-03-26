import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from ocvl.function.utility.json_format_constants import DisplayParams, MetricTags


def display_iORG_pop_summary(stim_framestamps, stim_pop_summary, relative_pop_summary=None, stim_vidnum="",
                             control_framestamps=None, control_pop_iORG_summary=None, control_vidnums=None,
                             control_framestamps_pooled=None, control_pop_iORG_summary_pooled=None,
                             framerate=15.0, sum_method="", sum_control="", figure_label="", params=None):

    if control_vidnums is None:
        control_vidnums = [""]
    if params is None:
        params = dict()

    disp_stim = params.get(DisplayParams.DISP_STIMULUS, True) and np.all(stim_pop_summary)
    disp_cont = params.get(DisplayParams.DISP_CONTROL, True) and np.all(control_pop_iORG_summary)
    disp_rel = params.get(DisplayParams.DISP_RELATIVE, True) and np.all(relative_pop_summary)

    ax_params = params.get(DisplayParams.AXES, dict())
    xlimits = (ax_params.get(DisplayParams.XMIN, None), ax_params.get(DisplayParams.XMAX, None))
    ylimits = (ax_params.get(DisplayParams.YMIN, None), ax_params.get(DisplayParams.YMAX, None))
    how_many = np.sum([disp_stim, disp_cont, disp_rel])

    plt.figure(figure_label)

    ind = 1
    if how_many > 1 and disp_stim:
        plt.subplot(1, how_many, ind)
        ind += 1
    if disp_stim:
        dispinds = np.isfinite(stim_pop_summary)
        plt.title("Stimulus iORG")
        plt.plot(stim_framestamps[dispinds] / framerate, stim_pop_summary[dispinds],label=str(stim_vidnum))
        plt.xlabel("Time (s)")
        plt.ylabel(sum_method)
        if all(xlimits): plt.xlim(xlimits)
        if all(ylimits): plt.ylim(ylimits)
    if how_many > 1 and disp_cont:
        plt.subplot(1, how_many, ind)
        ind += 1
    if disp_cont  and plt.gca().get_title() != "Control iORGs":  # The last bit ensures we don't spam the subplots with control data.
        plt.title("Control iORGs")
        for r in range(control_pop_iORG_summary.shape[0]):
            plt.plot(control_framestamps[r] / framerate, control_pop_iORG_summary[r, control_framestamps[r]], label=str(control_vidnums[r]))

        if np.all(control_pop_iORG_summary_pooled):
            plt.plot(control_framestamps_pooled / framerate, control_pop_iORG_summary_pooled[control_framestamps_pooled], 'k--', linewidth=4)
        plt.xlabel("Time (s)")
        plt.ylabel(sum_method)
        if all(xlimits): plt.xlim(xlimits)
        if all(ylimits): plt.ylim(ylimits)
        if ax_params.get(DisplayParams.LEGEND, False): plt.legend()
    if how_many > 1 and disp_rel:
        plt.subplot(1, how_many, ind)
        ind += 1
    if disp_rel:
        dispinds = np.isfinite(relative_pop_summary)
        plt.title("Stimulus relative to control iORG via " + sum_control)
        plt.plot(stim_framestamps[dispinds] / framerate, relative_pop_summary[dispinds],
                 label=str(stim_vidnum))
        plt.xlabel("Time (s)")
        plt.ylabel(sum_method)
        if all(xlimits): plt.xlim(xlimits)
        if all(ylimits): plt.ylim(ylimits)

    if ax_params.get(DisplayParams.LEGEND, False): plt.legend()



def display_iORG_pop_summary_seq(framestamps, pop_summary, vidnum_seq, framerate=15.0, sum_method="",
                                 figure_label="", params=None):

    if params is None:
        params = dict()
    num_in_seq = params.get(DisplayParams.NUM_IN_SEQ, 0)
    ax_params = params.get(DisplayParams.AXES, dict())
    xlimits = (ax_params.get(DisplayParams.XMIN, None), ax_params.get(DisplayParams.XMAX, None))
    ylimits = (ax_params.get(DisplayParams.YMIN, None), ax_params.get(DisplayParams.YMAX, None))

    seq_row = int(np.ceil(num_in_seq / 5))

    plt.figure(figure_label)
    plt.subplot(seq_row, 5, (vidnum_seq % num_in_seq) + 1)

    plt.title("Acquisition " + str(vidnum_seq % num_in_seq) + " of " + str(num_in_seq))
    plt.plot(framestamps / framerate, pop_summary)
    plt.xlabel("Time (s)")
    plt.ylabel(sum_method)
    if all(xlimits): plt.xlim(xlimits)
    if all(ylimits): plt.ylim(ylimits)

    if ax_params.get(DisplayParams.LEGEND, False): plt.legend()


def display_iORG_summary_histogram(iORG_result=pd.DataFrame(), metrics=None, cumulative=False, data_label="", figure_label="", params=None):

    if metrics is None:
        metrics = list(MetricTags)
    if params is None:
        params = dict()

    ax_params = params.get(DisplayParams.AXES, dict())
    xlimits = (ax_params.get(DisplayParams.XMIN, None), ax_params.get(DisplayParams.XMAX, None))

    plt.figure(figure_label)

    numsub = np.sum(len(metrics))
    subind = 1
    for metric in metrics:
        if iORG_result.loc[:, metric].count() != 0:
            metric_res = iORG_result.loc[:, metric].values.astype(float)

            plt.subplot(numsub, 1, subind)
            subind += 1

            if all(xlimits) and ax_params.get(DisplayParams.XSTEP):
                histbins = np.arange(start=xlimits[0], stop=xlimits[1], step=ax_params.get(DisplayParams.XSTEP))
                if not cumulative:
                    plt.hist(metric_res, bins=histbins, label=data_label)
                else:
                    plt.hist(metric_res, bins=histbins, label=data_label, density=True, histtype="step",
                             cumulative=True)
            else:
                if not cumulative:
                    plt.hist(metric_res, bins=ax_params.get(DisplayParams.NBINS, 50), label=data_label)
                else:
                    plt.hist(metric_res, bins=ax_params.get(DisplayParams.NBINS, 50), label=data_label,
                             density=True, histtype="step", cumulative=True)

            plt.title(metric)
            if all(xlimits): plt.xlim(xlimits)
            if ax_params.get(DisplayParams.LEGEND, False): plt.legend()

            plt.show(block=False)

def display_iORG_summary_overlay(values, coordinates, image, colorbar_label="", figure_label="", params=None):

    if params is None:
        params = dict()

    ax_params = params.get(DisplayParams.AXES, dict())

    plt.figure(figure_label)
    plt.title(figure_label)
    plt.imshow(image, cmap='gray')

    if ax_params.get(DisplayParams.XMIN, None) and ax_params.get(DisplayParams.XMAX, None) and ax_params.get(
            DisplayParams.XSTEP, None):
        histbins = np.arange(start=ax_params.get(DisplayParams.XMIN),
                             stop=ax_params.get(DisplayParams.XMAX),
                             step=ax_params.get(DisplayParams.XSTEP))
    else:
        histbins = np.arange(start=np.nanpercentile(values, 1), stop=np.nanpercentile(values, 99),
                             step=(np.nanpercentile(values, 99) - np.nanpercentile(values, 1)) / 100)

    normmap = mpl.colors.Normalize(vmin=histbins[0], vmax=histbins[-1], clip=True)
    mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap(ax_params.get(DisplayParams.CMAP, "viridis")),
                                   norm=normmap)

    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=10, c=mapper.to_rgba(values))
    plt.colorbar(mapper, ax=plt.gca(), label=colorbar_label)
    plt.show(block=False)
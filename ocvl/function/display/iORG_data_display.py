import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ocvl.function.utility.json_format_constants import DisplayParams, MetricTags


def _update_plot_colors(data_color):

    the_lines = plt.gca().get_lines()
    the_patches = plt.gca().patches
    if data_color is not None:

        # If the color is a colormap, then make each item span that colormap.
        if data_color in plt.colormaps():
            # Span all of our patches over this colormap.
            normmap = mpl.colors.Normalize(vmin=0, vmax=len(the_patches), clip=True)
            mapper = plt.cm.ScalarMappable(cmap=data_color, norm=normmap)
            for l, patch in enumerate(the_patches):
                if patch.get_label(): # Only do this for labelled patches.
                    # Get our patches' alphas so that we can respect them
                    edgealpha = patch.get_edgecolor()[-1]
                    facealpha = patch.get_facecolor()[-1]

                    patch.set_facecolor(mapper.to_rgba(l)[0:3] + (facealpha,))
                    patch.set_edgecolor(mapper.to_rgba(l)[0:3] + (edgealpha,))

            # Span all of our lines over this colormap.
            normmap = mpl.colors.Normalize(vmin=0, vmax=len(the_lines), clip=True)
            mapper = plt.cm.ScalarMappable(cmap=data_color, norm=normmap)
            for l, line in enumerate(the_lines):
                # Update this line's color in line with how many lines are on the plot.
                line.set_color(mapper.to_rgba(l))

                # Also update everything that is associated with this line, making sure to preserve the alpha.
                for child in plt.gca().findobj(lambda obj: obj.get_label() == line.get_label() and obj is not line):

                    if child.get_alpha() is None:
                        alphy = 0.9
                    else:
                        alphy = child.get_alpha()
                    if isinstance(child, Line2D):
                        child.set_color(mapper.to_rgba(l)[0:3] + (alphy,))
                    elif isinstance(child, FillBetweenPolyCollection):
                        child.set_facecolor(mapper.to_rgba(l)[0:3] + (alphy,))


        # If the color is something included in matplotlib, then set all lines equal to that.
        elif data_color in mpl.colors.CSS4_COLORS or data_color in mpl.colors.BASE_COLORS:
            # Update all of our patches.
            for l, patch in enumerate(the_patches):
                if patch.get_label(): # Only do this for labelled patches.
                    # Get our patches' alphas so that we can respect them
                    edgealpha = patch.get_edgecolor()[-1]
                    facealpha = patch.get_facecolor()[-1]

                    patch.set_facecolor(data_color[0:3] + (facealpha,))
                    patch.set_edgecolor(data_color[0:3] + (edgealpha,))

            # Update all of our lines.
            for l, line in enumerate(the_lines):
                line.set_color(data_color)

                # Also update everything that is associated with this line, making sure to preserve the alpha.
                for child in plt.gca().findobj(lambda obj: obj.get_label() == line.get_label() and obj is not line):

                    if child.get_alpha() is None:
                        alphy = 0.9
                    else:
                        alphy = child.get_alpha()
                    if isinstance(child, Line2D):
                        child.set_color(data_color + (alphy,))
                    elif isinstance(child, FillBetweenPolyCollection):
                        child.set_facecolor(data_color + (alphy,))

        # If it's a numerical tuple of an appropriate length, then we can also use those.
        elif isinstance(data_color, tuple) and all(isinstance(item, float) for item in data_color) \
            and all(isinstance(item, int) for item in data_color) and (len(data_color) == 3 or len(data_color) == 4):

            # Update all of our patches.
            for l, patch in enumerate(the_patches):
                if patch.get_label(): # Only do this for labelled patches.
                    # Get our patches' alphas so that we can respect them
                    edgealpha = patch.get_edgecolor()[-1]
                    facealpha = patch.get_facecolor()[-1]

                    patch.set_facecolor(data_color[0:3] + (facealpha,))
                    patch.set_edgecolor(data_color[0:3] + (edgealpha,))

            # Update all of our lines.
            for l, line in enumerate(the_lines):
                line.set_color(data_color)

                # Also update everything that is associated with this line, making sure to preserve the alpha.
                for child in plt.gca().findobj(lambda obj: obj.get_label() == line.get_label() and obj is not line):

                    if child.get_alpha() is None:
                        alphy = 0.9
                    else:
                        alphy = child.get_alpha()
                    if isinstance(child, Line2D):
                        child.set_color(data_color + (alphy,))
                    elif isinstance(child, FillBetweenPolyCollection):
                        child.set_facecolor(data_color + (alphy,))


def display_iORG_pop_summary(stim_framestamps, stim_pop_summary, relative_pop_summary=None, stim_vidnum="",
                             control_framestamps=None, control_pop_iORG_summary=None, control_vidnums=None,
                             control_framestamps_pooled=None, control_pop_iORG_summary_pooled=None,
                             stim_delivery_frms=None,framerate=15.0, sum_method="", sum_control="", figure_label="", params=None,
                             stim_error=None, control_error=None, rel_error=None, data_color=None):

    if control_vidnums is None:
        control_vidnums = [""]
    if params is None:
        params = dict()

    disp_stim = params.get(DisplayParams.DISP_STIMULUS, False) and np.any(stim_pop_summary)
    disp_cont = params.get(DisplayParams.DISP_CONTROL, False) and np.any(control_pop_iORG_summary)
    disp_rel = params.get(DisplayParams.DISP_RELATIVE, False) and np.any(relative_pop_summary)

    ax_params = params.get(DisplayParams.AXES, dict())
    if data_color is None:
        data_color = ax_params.get(DisplayParams.CMAP, "viridis")

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
        plt.title(sum_method.upper()+"-summarized iORGs\n(Stimulus acquisitions)")
        plt.plot(stim_framestamps[dispinds] / framerate, stim_pop_summary[dispinds], label=str(stim_vidnum))
        plt.xlabel("Time (s)")
        plt.ylabel(sum_method.upper())

        # If we have error bounds, plot them too.
        if stim_error is not None and len(stim_pop_summary) == len(stim_error):
            plt.gca().fill_between(stim_framestamps[dispinds]/ framerate, stim_pop_summary[dispinds]-stim_error[dispinds],
                                   stim_pop_summary[dispinds]+stim_error[dispinds], alpha=0.2, label=str(stim_vidnum), interpolate=True)

        _update_plot_colors(data_color)

        if stim_delivery_frms is not None and len(plt.gca().get_lines()) == 1:
            for i in range(1, len(stim_delivery_frms), 2):
                plt.gca().axvspan(float(stim_delivery_frms[i-1]/ framerate),
                                  float(stim_delivery_frms[i]/ framerate), facecolor='g', alpha=0.5)


        if not None in xlimits: plt.xlim(xlimits)
        if not None in ylimits: plt.ylim(ylimits)
        if ax_params.get(DisplayParams.LEGEND, False) and not disp_rel: plt.legend()

    if how_many > 1 and disp_cont:
        plt.subplot(1, how_many, ind)
        ind += 1
    if disp_cont and plt.gca().get_title() != (sum_method.upper()+"-summarized iORGs\n(Control acquisitions)"):  # The last bit ensures we don't spam the subplots with control data.
        plt.title(sum_method.upper()+"-summarized iORGs\n(Control acquisitions)")
        for r in range(control_pop_iORG_summary.shape[0]):
            plt.plot(control_framestamps[r] / framerate, control_pop_iORG_summary[r, control_framestamps[r]], label=str(control_vidnums[r]))

        plt.xlabel("Time (s)")
        plt.ylabel(sum_method.upper())

        _update_plot_colors(data_color)

        if control_pop_iORG_summary_pooled is not None:
            plt.plot(control_framestamps_pooled / framerate, control_pop_iORG_summary_pooled[control_framestamps_pooled], 'k--', linewidth=4)

        if not None in xlimits: plt.xlim(xlimits)
        if not None in ylimits: plt.ylim(ylimits)
        if ax_params.get(DisplayParams.LEGEND, False): plt.legend()

    if how_many > 1 and disp_rel:
        plt.subplot(1, how_many, ind)
        ind += 1
    if disp_rel:
        dispinds = np.isfinite(relative_pop_summary)
        plt.title(sum_method.upper()+"-summarized stimulus iORGs relative\nto control via " + sum_control)
        plt.plot(stim_framestamps[dispinds] / framerate, relative_pop_summary[dispinds],
                 label=str(stim_vidnum))
        plt.xlabel("Time (s)")
        plt.ylabel(sum_method.upper())

        _update_plot_colors(data_color)

        if stim_delivery_frms is not None and len(plt.gca().get_lines()) == 1:
            for i in range(1, len(stim_delivery_frms), 2):
                plt.gca().axvspan(float(stim_delivery_frms[i-1]/ framerate),
                                  float(stim_delivery_frms[i]/ framerate), facecolor='g', alpha=0.5)

        if not None in xlimits: plt.xlim(xlimits)
        if not None in ylimits: plt.ylim(ylimits)

        if ax_params.get(DisplayParams.LEGEND, False): plt.legend(loc="upper left")


def display_iORGs(stim_framestamps=None, stim_iORGs=None, stim_vidnums="",
                  control_framestamps=None, control_iORGs=None, control_vidnums="",
                  image=None, cell_loc=None,
                  stim_delivery_frms=None, framerate=15.0, figure_label="", params=None, data_color=None):

    if params is None:
        params = dict()


    disp_im = image is not None and cell_loc is not None
    disp_stim = params.get(DisplayParams.DISP_STIMULUS, False) and stim_iORGs is not None
    disp_cont = params.get(DisplayParams.DISP_CONTROL, False) and control_iORGs is not None

    if cell_loc is not None and len(cell_loc.shape) == 1:
        cell_loc = cell_loc[None, :]

    if disp_stim and stim_framestamps is None:
        stim_framestamps = np.arange(stim_iORGs.shape[1])
    if not disp_stim and disp_cont and stim_framestamps is None:
        stim_framestamps = np.arange(control_iORGs.shape[1])

    ax_params = params.get(DisplayParams.AXES, dict())

    if data_color is None: data_color = ax_params.get(DisplayParams.CMAP, "viridis")

    show_legend = ax_params.get(DisplayParams.LEGEND, False) and \
                  isinstance(stim_vidnums, list) and isinstance(control_vidnums, list)
    xlimits = (ax_params.get(DisplayParams.XMIN, None), ax_params.get(DisplayParams.XMAX, None))
    ylimits = (ax_params.get(DisplayParams.YMIN, None), ax_params.get(DisplayParams.YMAX, None))
    linethickness = ax_params.get(DisplayParams.LINEWIDTH, 2.5)
    how_many = np.sum([disp_im, disp_stim, disp_cont])

    if disp_stim and not isinstance(stim_vidnums, list):
        stim_vidnums = [stim_vidnums] * stim_iORGs.shape[0]
    if disp_cont and not isinstance(control_vidnums, list):
        control_vidnums = [control_vidnums] * control_iORGs.shape[0]

    plt.figure(figure_label)

    ind = 1
    if how_many > 1 and disp_im:
        plt.subplot(1, how_many, ind)
        ind += 1
    if disp_im:
        plt.title("Cell location")
        plt.imshow(image, cmap='gray')
        plt.plot(cell_loc[:, 0], cell_loc[:, 1], "*", markersize=6)

    if how_many > 1 and disp_stim:
        plt.subplot(1, how_many, ind)
        ind += 1
    if disp_stim:
        plt.title("Stimulus iORG")
        for r in range(stim_iORGs.shape[0]):
            dispinds = np.isfinite(stim_iORGs[r])
            plt.plot(stim_framestamps[dispinds] / framerate, stim_iORGs[r, dispinds], linewidth=linethickness,
                     label=str(stim_vidnums[r]))
        plt.xlabel("Time (s)")
        plt.ylabel("A.U.")

        _update_plot_colors(data_color)

        if stim_delivery_frms is not None and len(plt.gca().get_lines()) == 1:
            for i in range(1, len(stim_delivery_frms), 2):
                plt.gca().axvspan(float(stim_delivery_frms[i-1]/ framerate),
                                  float(stim_delivery_frms[i]/ framerate), facecolor='g', alpha=0.5)

        if not None in xlimits: plt.xlim(xlimits)
        if not None in ylimits: plt.ylim(ylimits)
        if show_legend: plt.legend()

    if how_many > 1 and disp_cont:
        plt.subplot(1, how_many, ind)
        ind += 1
    if disp_cont  and plt.gca().get_title() != "Control iORGs":  # The last bit ensures we don't spam the subplots with control data.
        plt.title("Control iORGs")
        for r in range(control_iORGs.shape[0]):
            dispinds = np.isfinite(stim_iORGs[r])
            plt.plot(control_framestamps[dispinds] / framerate, control_iORGs[r, dispinds], linewidth=linethickness, label=str(control_vidnums[r]))
        plt.xlabel("Time (s)")
        plt.ylabel("A.U.")

        _update_plot_colors(data_color)

        if not None in xlimits: plt.xlim(xlimits)
        if not None in ylimits: plt.ylim(ylimits)
        if show_legend: plt.legend()


def display_iORG_pop_summary_seq(framestamps, pop_summary, vidnum_seq, stim_delivery_frms=None,
                                 framerate=15.0, sum_method="",
                                 figure_label="", params=None, data_color=None):

    if params is None:
        params = dict()
    num_in_seq = params.get(DisplayParams.NUM_IN_SEQ, 0)
    ax_params = params.get(DisplayParams.AXES, dict())
    if data_color is None: data_color = ax_params.get(DisplayParams.CMAP, "viridis")

    xlimits = (ax_params.get(DisplayParams.XMIN, None), ax_params.get(DisplayParams.XMAX, None))
    ylimits = (ax_params.get(DisplayParams.YMIN, None), ax_params.get(DisplayParams.YMAX, None))

    seq_row = int(np.ceil(num_in_seq / 5))

    plt.figure(figure_label)
    plt.subplot(seq_row, 5, (vidnum_seq % num_in_seq) + 1)

    plt.title("Acquisition " + str(vidnum_seq % num_in_seq) + " of " + str(num_in_seq))
    plt.plot(framestamps / framerate, pop_summary, label=str(vidnum_seq))
    plt.xlabel("Time (s)")
    plt.ylabel(sum_method.upper())
    if not None in xlimits: plt.xlim(xlimits)
    if not None in ylimits: plt.ylim(ylimits)

    _update_plot_colors(data_color)

    if stim_delivery_frms is not None and len(plt.gca().get_lines()) == 1:
        for i in range(1, len(stim_delivery_frms), 2):
            plt.gca().axvspan(float(stim_delivery_frms[i - 1] / framerate),
                              float(stim_delivery_frms[i] / framerate), facecolor='g', alpha=0.5)

    if ax_params.get(DisplayParams.LEGEND, False): plt.legend(loc="upper left")


def display_iORG_summary_histogram(iORG_result=pd.DataFrame(), metrics=None, cumulative=False, data_label="", figure_label="", params=None,
                                   data_color = None):

    if metrics is None:
        metrics = list(MetricTags)
    if params is None:
        params = dict()

    ax_params = params.get(DisplayParams.AXES, dict())
    if data_color is None: data_color = ax_params.get(DisplayParams.CMAP, "viridis")
    xlimits = (ax_params.get(DisplayParams.XMIN, None), ax_params.get(DisplayParams.XMAX, None))

    plt.figure(figure_label)

    numsub = np.sum(len(metrics))
    subind = 1
    for metric in metrics:
        if iORG_result.loc[:, metric].count() != 0:
            metric_res = iORG_result.loc[:, metric].values.astype(float)

            plt.subplot(numsub, 1, subind)
            subind += 1

            if not None in xlimits and ax_params.get(DisplayParams.XSTEP):
                histbins = np.arange(start=xlimits[0], stop=xlimits[1], step=ax_params.get(DisplayParams.XSTEP))
                if not cumulative:
                    plt.hist(metric_res, bins=histbins, label=data_label, histtype="step", linewidth=2.5)
                else:
                    plt.hist(metric_res, bins=histbins, label=data_label, density=True, histtype="step",
                             cumulative=True, linewidth=2.5)
            else:
                if not cumulative:
                    plt.hist(metric_res, bins=ax_params.get(DisplayParams.NBINS, 50), histtype="step", linewidth=2.5,
                             label=data_label)
                else:
                    plt.hist(metric_res, bins=ax_params.get(DisplayParams.NBINS, 50), label=data_label,
                             density=True, histtype="step", cumulative=True, linewidth=2.5)

            plt.title(metric)
            if not None in xlimits: plt.xlim(xlimits)

            _update_plot_colors(data_color)

            # the_histos = plt.gca().containers
            # normmap = mpl.colors.Normalize(vmin=0, vmax=len(the_histos), clip=True)
            # mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap(ax_params.get(DisplayParams.CMAP, "viridis")),
            #                                norm=normmap)
            # for l, histy in enumerate(the_histos):
            #     for patch in histy.patches:
            #         patch.set_color(mapper.to_rgba(l))

            if ax_params.get(DisplayParams.LEGEND, False): plt.legend()
            plt.show(block=False)

def display_iORG_summary_overlay(values, coordinates, image, colorbar_label="", figure_label="", params=None):

    if params is None:
        params = dict()

    ax_params = params.get(DisplayParams.AXES, dict())

    fig=plt.figure(figure_label)
    ax = plt.axes()
    plt.title(figure_label)
    plt.imshow(image, cmap='gray')
    ax.set_axis_off()

    if ax_params.get(DisplayParams.XMIN, None) and ax_params.get(DisplayParams.XMAX, None) and ax_params.get(
            DisplayParams.XSTEP, None):
        histbins = np.arange(start=ax_params.get(DisplayParams.XMIN),
                             stop=ax_params.get(DisplayParams.XMAX),
                             step=ax_params.get(DisplayParams.XSTEP))
    else:
        starting = np.nanpercentile(values, 1)
        stopping = np.nanpercentile(values, 99)
        stepping = (stopping - starting) / 100
        if stepping != 0:
            histbins = np.arange(start=starting, stop=stopping, step=stepping)
        else:
            histbins = np.array([0, 1])

    normmap = mpl.colors.Normalize(vmin=histbins[0], vmax=histbins[-1], clip=True)
    mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap(ax_params.get(DisplayParams.CMAP, "viridis")),
                                   norm=normmap)

    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=10, c=mapper.to_rgba(values))

    divider=make_axes_locatable(ax)
    cax=divider.append_axes("right",size="5%", pad=0.1)
    plt.colorbar(mapper, cax=cax, label=colorbar_label)
    plt.show(block=False)


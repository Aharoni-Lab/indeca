import itertools as itt
import warnings

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec


def plot_traces(tr_dict, **kwargs):
    trs = []
    for tr_name, tr_dat in tr_dict.items():
        t = go.Scatter(y=tr_dat, name=tr_name, mode="lines", **kwargs)
        trs.append(t)
    return trs


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def plot_met_ROC(metdf, grad_color: bool = True):
    if "group" not in metdf.columns:
        metdf["group"] = ""
    fig = plt.figure(constrained_layout=True, figsize=(8, 4))
    gs = GridSpec(3, 2, figure=fig)
    ax_err = fig.add_subplot(gs[0, 0])
    ax_scl = fig.add_subplot(gs[1, 0])
    ax_f1 = fig.add_subplot(gs[2, 0])
    ax_roc = fig.add_subplot(gs[:, 1])
    ax_roc.invert_xaxis()
    lw = 2
    ls = ["solid"] if grad_color else ["dotted", "dashed", "dashdot"]
    for (grp, grpdf), cur_ls in zip(metdf.groupby("group"), itt.cycle(ls)):
        try:
            oidx = int(grpdf["opt_idx"].dropna().unique().item())
        except ValueError:
            oidx = None
        th = np.array(grpdf["thres"])
        ax_err.set_yscale("log")
        ax_err.set_xlabel("Threshold")
        ax_err.set_ylabel("Error")
        if grad_color:
            ax_err.plot(th, grpdf["objs"], alpha=0)
            colored_line(x=th, y=grpdf["objs"], c=th, ax=ax_err, linewidths=lw)
        else:
            ax_err.plot(th, grpdf["objs"], ls=cur_ls)
        if oidx is not None:
            ax_err.axvline(th[oidx], ls="dotted", color="gray")
        ax_scl.set_xlabel("Threshold")
        ax_scl.set_ylabel("Scale")
        if grad_color:
            ax_scl.plot(th, grpdf["scals"], alpha=0)
            colored_line(x=th, y=grpdf["scals"], c=th, ax=ax_scl, linewidths=lw)
        else:
            ax_scl.plot(th, grpdf["scals"], ls=cur_ls)
        if oidx is not None:
            ax_scl.axvline(th[oidx], ls="dotted", color="gray")
        ax_f1.set_xlabel("Threshold")
        ax_f1.set_ylabel("f1 Score")
        if grad_color:
            ax_f1.plot(th, grpdf["f1"], alpha=0)
            colored_line(x=th, y=grpdf["f1"], c=th, ax=ax_f1, linewidths=lw)
        else:
            ax_f1.plot(th, grpdf["f1"], ls=cur_ls)
        if oidx is not None:
            ax_f1.axvline(th[oidx], ls="dotted", color="gray")
        ax_roc.set_xlabel("Precision")
        ax_roc.set_ylabel("Recall")
        if grad_color:
            ax_roc.plot(grpdf["prec"], grpdf["recall"], alpha=0)
            colored_line(
                x=grpdf["prec"], y=grpdf["recall"], c=th, ax=ax_roc, linewidths=lw
            )
        else:
            ax_roc.plot(grpdf["prec"], grpdf["recall"], label=grp, ls=cur_ls)
        if oidx is not None:
            ax_roc.plot(
                grpdf["prec"].iloc[oidx],
                grpdf["recall"].iloc[oidx],
                marker="x",
                color="gray",
                markersize=15,
            )
    fig.legend()
    return fig

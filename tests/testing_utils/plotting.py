import itertools as itt
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon
from statannotations.Annotator import Annotator


def plot_traces(tr_dict, **kwargs):
    trs = []
    for tr_name, tr_dat in tr_dict.items():
        t = go.Scatter(
            y=tr_dat, name=tr_name, legendgroup=tr_name, mode="lines", **kwargs
        )
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


def plot_met_ROC_thres(
    metdf,
    grad_color: bool = True,
    fig=None,
    lw=2,
    grid_kws=dict(),
    trim_xlabs=True,
    log_err=True,
    annt_color="gray",
    annt_lw=1,
):
    if "group" not in metdf.columns:
        metdf["group"] = ""
    if fig is None:
        fig = plt.figure(constrained_layout=True, figsize=(8, 4))
    gs = GridSpec(3, 2, figure=fig, **grid_kws)
    ax_err = fig.add_subplot(gs[0, 0])
    ax_scl = fig.add_subplot(gs[1, 0])
    ax_f1 = fig.add_subplot(gs[2, 0])
    ax_roc = fig.add_subplot(gs[:, 1])
    ax_roc = move_yax_right(ax_roc)
    ax_roc.set_aspect("equal", adjustable="datalim")
    ls = ["solid"] if grad_color else ["dotted", "dashed", "dashdot"]
    for (grp, grpdf), cur_ls in zip(metdf.groupby("group"), itt.cycle(ls)):
        try:
            oidx = int(grpdf["opt_idx"].dropna().unique().item())
        except ValueError:
            oidx = None
        th = np.array(grpdf["thres"])
        if log_err:
            ax_err.set_yscale("log")
        if trim_xlabs:
            ax_err.set_xticklabels([])
            ax_err.set_xlabel("")
        else:
            ax_err.set_xlabel("Threshold (A.U)")
        ax_err.set_ylabel("Error\n(A.U.)")
        if grad_color:
            ax_err.plot(th, grpdf["objs"], alpha=0)
            colored_line(x=th, y=grpdf["objs"], c=th, ax=ax_err, linewidths=lw)
        else:
            ax_err.plot(th, grpdf["objs"], ls=cur_ls)
        if oidx is not None:
            ax_err.axvline(th[oidx], ls="dotted", lw=annt_lw, color=annt_color)
        if trim_xlabs:
            ax_scl.set_xticklabels([])
            ax_scl.set_xlabel("")
        else:
            ax_scl.set_xlabel("Threshold (A.U)")
        ax_scl.set_ylabel("Scale\n(A.U.)")
        if grad_color:
            ax_scl.plot(th, grpdf["scals"], alpha=0)
            colored_line(x=th, y=grpdf["scals"], c=th, ax=ax_scl, linewidths=lw)
        else:
            ax_scl.plot(th, grpdf["scals"], ls=cur_ls)
        if oidx is not None:
            ax_scl.axvline(th[oidx], ls="dotted", lw=annt_lw, color=annt_color)
        ax_f1.set_xlabel("Threshold (A.U)")
        ax_f1.set_ylabel("F1\nscore")
        if grad_color:
            ax_f1.plot(th, grpdf["f1"], alpha=0)
            colored_line(x=th, y=grpdf["f1"], c=th, ax=ax_f1, linewidths=lw)
        else:
            ax_f1.plot(th, grpdf["f1"], ls=cur_ls)
        if oidx is not None:
            ax_f1.axvline(th[oidx], ls="dotted", lw=annt_lw, color=annt_color)
        ax_roc.set_xlabel("Precision")
        ax_roc.set_ylabel("Recall")
        ax_roc.plot([1.05], [1.05], alpha=0)  # extend datalim
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
                color=annt_color,
                lw=annt_lw,
                markersize=12,
            )
    if metdf["group"].nunique() > 1:
        fig.legend()
    return fig


def plot_met_ROC_scale(metdf, iterdf, opt_scale, grad_color: bool = True):
    fig = plt.figure(constrained_layout=True, figsize=(8, 4))
    gs = GridSpec(3, 2, figure=fig)
    ax_obj = fig.add_subplot(gs[0, 0])
    ax_f1 = fig.add_subplot(gs[1, 0])
    ax_iter = fig.add_subplot(gs[2, 0])
    ax_roc = fig.add_subplot(gs[:, 1])
    ax_roc.invert_xaxis()
    lw = 2
    ls = ["solid"] if grad_color else ["dotted", "dashed", "dashdot"]
    scls = metdf["scale"]
    ax_obj.set_xlabel("Scale")
    ax_obj.set_ylabel("Error")
    if grad_color:
        ax_obj.plot(scls, metdf["objs"], alpha=0)
        colored_line(x=scls, y=metdf["objs"], c=scls, ax=ax_obj, linewidths=lw)
    else:
        ax_obj.plot(scls, metdf["objs"], ls=ls)
    ax_obj.axvline(opt_scale, ls="dotted", color="gray")
    ax_iter.set_xlabel("Iter")
    ax_iter.set_ylabel("Scale")
    ax_iter.plot(iterdf["iter"], iterdf["scale"])
    ax_iter.axhline(opt_scale, ls="dotted", color="gray")
    ax_f1.set_xlabel("Scale")
    ax_f1.set_ylabel("f1 Score")
    if grad_color:
        ax_f1.plot(scls, metdf["f1"], alpha=0)
        colored_line(x=scls, y=metdf["f1"], c=scls, ax=ax_f1, linewidths=lw)
    else:
        ax_f1.plot(scls, metdf["f1"], ls=ls)
    ax_f1.axvline(opt_scale, ls="dotted", color="gray")
    ax_roc.set_xlabel("Precision")
    ax_roc.set_ylabel("Recall")
    if grad_color:
        ax_roc.plot(metdf["prec"], metdf["recall"], alpha=0)
        colored_line(
            x=metdf["prec"], y=metdf["recall"], c=scls, ax=ax_roc, linewidths=lw
        )
    else:
        ax_roc.plot(metdf["prec"], metdf["recall"], ls=ls)
    fig.legend()
    return fig


def plot_agg_boxswarm(
    dat,
    row,
    col,
    x,
    y,
    hue=None,
    facet_kws=dict(),
    box_kws={"saturation": 0.5},
    swarm_kws={"size": 5, "linewidth": 1.2},
    annt_pairs=None,
    annt_group=None,
):
    if hue is None:
        hue = x
    g = sns.FacetGrid(dat, row=row, col=col, **facet_kws)
    g.map_dataframe(
        sns.boxplot,
        x=x,
        y=y,
        hue=hue,
        showfliers=False,
        **box_kws,
    )
    g.map_dataframe(
        sns.swarmplot,
        x=x,
        y=y,
        hue=hue,
        edgecolor="auto",
        warn_thresh=0.9,
        **swarm_kws,
    )
    if annt_pairs is not None:
        g.map_dataframe(agg_annot, x=x, y=y, pairs=annt_pairs)
    if annt_group is not None:
        g.map_dataframe(agg_annot_group, x=x, y=y, group=annt_group)
    g.tight_layout()
    return g


def agg_annot(data, pairs, x, y, color=None, **kwargs):
    ax = plt.gca()
    annt = Annotator(ax, pairs, data=data, x=x, y=y)
    annt.configure(test="Wilcoxon", text_format="star", loc="outside", line_width=0)
    annt.apply_and_annotate()


def agg_annot_group(data, group, x, y, color=None):
    ax = plt.gca()
    for xlabA, Blabs in group.items():
        datA = data.loc[data[x] == xlabA, y]
        pval_df = []
        for xlabB in Blabs:
            datB = data.loc[data[x] == xlabB, y]
            res = wilcoxon(datA, datB)
            pval_df.append(
                pd.DataFrame(
                    [
                        {
                            "labA": xlabA,
                            "labB": xlabB,
                            "stat": res.statistic,
                            "pval": res.pvalue,
                        }
                    ]
                )
            )
        pval_df = pd.concat(pval_df)
        pval = pval_df["pval"].max()
        y_sh = 0.08
        if pval < 1e-4:
            text = "****"
        elif pval < 1e-3:
            text = "***"
        elif pval < 1e-2:
            text = "**"
        elif pval < 5e-2:
            text = "*"
        else:
            text = "ns"
            y_sh = 0.04
        yloc = datA.max() * 1.15
        ax.plot([xlabA], [yloc], alpha=0)
        ax.text(
            x=xlabA,
            y=yloc - y_sh * datA.max(),
            s=text,
            ha="center",
            va="center",
        )


def plot_pipeline_iter(
    data,
    color,
    dhm0=None,
    dhm1=None,
    aggregate=True,
    swarm_kws={"linewidth": 1},
    box_kws=dict(),
    **kwargs,
):
    ax = plt.gca()
    mthd = data["method"].unique().item()
    met = data["metric"].unique().item()
    use_all = data["use_all"].unique().item()
    if met == "dhm0" and dhm1 is not None:
        ax.axhline(dhm0, color="grey", ls=":")
    elif met == "dhm1" and dhm1 is not None:
        ax.axhline(dhm1, color="grey", ls=":")
    if aggregate and mthd == "cnmf":
        data = data.groupby(["qthres", "test_id"])["value"].median().reset_index()
    elif aggregate and use_all:
        data = data.groupby(["iter", "test_id"])["value"].median().reset_index()
    if mthd == "indeca":
        data = data.astype({"iter": int})
        sns.swarmplot(
            data,
            x="iter",
            y="value",
            ax=ax,
            color=color,
            edgecolor="auto",
            warn_thresh=0.9,
            **swarm_kws,
        )
        sns.lineplot(data, x="iter", y="value", ax=ax, color=color, **kwargs)
        ax.set_xlabel("Iteration")
    elif mthd == "cnmf":
        data = data.astype({"qthres": float})
        if met != "f1":
            data["value"] = data["value"].where(data["qthres"] == 0.5)
        sns.swarmplot(
            data,
            x="qthres",
            y="value",
            ax=ax,
            color=color,
            edgecolor="auto",
            warn_thresh=0.9,
            **swarm_kws,
        )
        sns.boxplot(
            data,
            x="qthres",
            y="value",
            color=color,
            ax=ax,
            fill=False,
            showfliers=False,
            **box_kws,
        )
        ax.set_xlabel("Threshold (quantile)")


def move_yax_right(ax):
    # ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelright=True, labelleft=False)
    ax.yaxis.set_label_position("right")
    ax.yaxis.label.set_rotation(270)
    ax.yaxis.label.set_verticalalignment("bottom")
    ax.spines["right"].set_position(("outward", 0))
    ax.spines["right"].set_visible(True)
    return ax

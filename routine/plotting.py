import copy
import itertools as itt

import colorcet as cc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import cm
from plotly import graph_objects as go
from plotly.colors import convert_colors_to_same_type, unlabel_rgb
from plotly.express.colors import qualitative
from plotly.subplots import make_subplots

from .utilities import enumerated_product, norm


def facet_plotly(
    data: pd.DataFrame,
    facet_row: str = None,
    facet_col: str = None,
    title_dim: str = None,
    margin_titles: bool = True,
    specs: dict = None,
    col_wrap: int = None,
    **kwargs,
):
    data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    row_crd = data[facet_row].unique()
    col_crd = data[facet_col].unique()
    layout_ls = []
    iiter = 0
    for (ir, ic), (r, c) in enumerated_product(row_crd, col_crd):
        dat_sub = data[(data[facet_row] == r) & (data[facet_col] == c)]
        if not len(dat_sub) > 0:
            continue
        if title_dim is not None:
            title = dat_sub[title_dim].unique().item()
        else:
            if facet_row == "DUMMY_FACET_ROW":
                title = "{}={}".format(facet_col, c)
            elif facet_col == "DUMMY_FACET_COL":
                title = "{}={}".format(facet_row, r)
            else:
                title = "{}={}; {}={}".format(facet_row, r, facet_col, c)
        if col_wrap is not None:
            ir = iiter // col_wrap
            ic = iiter % col_wrap
            iiter += 1
        layout_ls.append(
            {"row": ir, "col": ic, "row_label": r, "col_label": c, "title": title}
        )
    layout = pd.DataFrame(layout_ls).set_index(["row_label", "col_label"])
    if col_wrap is not None:
        nrow, ncol = int(layout["row"].max() + 1), int(layout["col"].max() + 1)
    else:
        nrow, ncol = len(row_crd), len(col_crd)
    if specs is not None:
        specs = np.full((nrow, ncol), specs).tolist()
    if margin_titles:
        fig = make_subplots(
            rows=nrow,
            cols=ncol,
            row_titles=row_crd.tolist(),
            column_titles=col_crd.tolist(),
            specs=specs,
            **kwargs,
        )
    else:
        fig = make_subplots(
            rows=nrow,
            cols=ncol,
            subplot_titles=layout["title"].values,
            specs=specs,
            **kwargs,
        )
    return fig, layout


def handle_single_facet(data, facet_row, facet_col):
    data = data.copy()
    if facet_row is None:
        data["DUMMY_FACET_ROW"] = ""
        facet_row = "DUMMY_FACET_ROW"
    if facet_col is None:
        data["DUMMY_FACET_COL"] = ""
        facet_col = "DUMMY_FACET_COL"
    return data, facet_row, facet_col


def scatter_3d(
    data,
    facet_row,
    facet_col,
    legend_dim: str = None,
    title_dim: str = None,
    col_wrap: int = None,
    **kwargs,
):
    data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    fig, layout = facet_plotly(
        data,
        facet_row,
        facet_col,
        title_dim,
        col_wrap=col_wrap,
        specs={"type": "scene"},
    )
    if legend_dim is not None:
        show_legend = {l: True for l in data[legend_dim].unique()}
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        if legend_dim is not None:
            for llab, subdf in facet_df.groupby(legend_dim, observed=True):
                show_leg = show_legend[llab]
                cur_args = transform_arguments(subdf, kwargs)
                trace = go.Scatter3d(
                    name=llab, legendgroup=llab, showlegend=show_leg, **cur_args
                )
                if show_leg:
                    show_legend[llab] = False
                fig.add_trace(trace, row=ir + 1, col=ic + 1)
        else:
            cur_args = transform_arguments(facet_df, kwargs)
            trace = go.Scatter3d(showlegend=False, **cur_args)
            fig.add_trace(trace, row=ir + 1, col=ic + 1)
    return fig


def transform_arguments(data: pd.DataFrame, arguments: dict):
    arg_ret = copy.deepcopy(arguments)
    if arg_ret.get("x"):
        arg_ret["x"] = data[arg_ret["x"]].values
    if arg_ret.get("y"):
        arg_ret["y"] = data[arg_ret["y"]].values
    if arg_ret.get("z"):
        arg_ret["z"] = data[arg_ret["z"]].values
    if arg_ret.get("customdata"):
        arg_ret["customdata"] = data[arg_ret["customdata"]].values
    if arg_ret.get("error_x"):
        x_err = data[arg_ret["error_x"]].values
        if np.nansum(x_err) > 0:
            arg_ret["error_x"] = {"array": x_err}
        else:
            del arg_ret["error_x"]
    if arg_ret.get("error_y"):
        y_err = data[arg_ret["error_y"]].values
        if np.nansum(y_err) > 0:
            arg_ret["error_y"] = {"array": y_err}
        else:
            del arg_ret["error_y"]
    if arg_ret.get("text"):
        try:
            arg_ret["text"] = data[arg_ret["text"]].values
        except KeyError:
            del arg_ret["text"]
    mkopts = arg_ret.get("marker")
    if mkopts:
        if mkopts.get("color"):
            try:
                color = data[mkopts["color"]].values
                if len(np.unique(color)) == 1:
                    mkopts["color"] = np.unique(color).item()
                else:
                    mkopts["color"] = color
            except KeyError:
                pass
        if mkopts.get("size"):
            try:
                size = data[mkopts["size"]].values
                if len(np.unique(size)) == 1:
                    mkopts["size"] = np.unique(size).item()
                else:
                    mkopts["size"] = size
            except KeyError:
                pass
        if mkopts.get("symbol"):
            try:
                symb = data[mkopts["symbol"]].values
                if len(np.unique(symb)) == 1:
                    mkopts["symbol"] = np.unique(symb).item()
                else:
                    mkopts["symbol"] = symb
            except KeyError:
                pass
    return arg_ret


def scatter_agg(
    data,
    x,
    y,
    facet_row,
    facet_col,
    col_wrap: int = None,
    legend_dim: str = None,
    title_dim: str = None,
    show_point_legend=False,
    subplot_args: dict = dict(),
    **kwargs,
):
    data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    fig, layout = facet_plotly(
        data, facet_row, facet_col, title_dim, col_wrap=col_wrap, **subplot_args
    )
    grp_dims = [facet_row, facet_col, x]
    idx_dims = [facet_row, facet_col]
    if legend_dim is not None:
        grp_dims.append(legend_dim)
        idx_dims.append(legend_dim)
        show_legend = {l: True for l in data[legend_dim].unique()}
    data_agg = (
        data.groupby(grp_dims, observed=True)[y]
        .agg(["mean", "sem"])
        .reset_index()
        .merge(data, on=grp_dims)
        .set_index(idx_dims)
        .sort_index()
    )
    kwargs["x"] = x
    kwargs["y"] = y
    kwargs_agg = copy.deepcopy(kwargs)
    kwargs_agg["y"] = "mean"
    kwargs_agg["error_y"] = "sem"
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        trace_ls = []
        if legend_dim is not None:
            for llab, subdf in facet_df.groupby(legend_dim, observed=True):
                show_leg = show_legend[llab]
                cur_args = transform_arguments(subdf, kwargs)
                ndata = subdf.groupby(x)[y].count().max()
                if ndata > 1:
                    strp = go.Box(
                        name=llab,
                        legendgroup=llab,
                        showlegend=show_leg and show_point_legend,
                        boxpoints="all",
                        fillcolor="rgba(255,255,255,0)",
                        hoveron="points",
                        line={"color": "rgba(255,255,255,0)"},
                        pointpos=0,
                        opacity=0.4,
                        **cur_args,
                    )
                    trace_ls.append(strp)
                subdf_agg = data_agg.loc[rlab, clab, llab].reset_index()
                cur_args = transform_arguments(subdf_agg, kwargs_agg)
                ln = go.Scatter(
                    name=llab,
                    legendgroup=llab,
                    showlegend=show_leg,
                    mode="lines",
                    **cur_args,
                )
                trace_ls.append(ln)
                if show_leg:
                    show_legend[llab] = False
        else:
            cur_args = transform_arguments(facet_df, kwargs)
            ndata = facet_df.groupby(x)[y].count().max()
            if ndata > 1:
                strp = go.Box(
                    boxpoints="all",
                    fillcolor="rgba(255,255,255,0)",
                    hoveron="points",
                    line={"color": "rgba(255,255,255,0)"},
                    pointpos=0,
                    opacity=0.4,
                    **cur_args,
                )
                trace_ls.append(strp)
            facet_df_agg = data_agg.loc[rlab, clab].reset_index()
            cur_args = transform_arguments(facet_df_agg, kwargs_agg)
            ln = go.Scatter(mode="lines", **cur_args)
            trace_ls.append(ln)
        for t in trace_ls:
            fig.add_trace(t, row=ir + 1, col=ic + 1)
    return fig


def imshow(
    data,
    facet_row,
    facet_col,
    margin_titles=True,
    title_dim: str = None,
    subplot_args: dict = dict(),
    equal_aspect=True,
    **kwargs,
):
    # data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    if equal_aspect:
        xcol, ycol = kwargs["x"], kwargs["y"]
        row_h = data.groupby(facet_row, sort=False, observed=True)[ycol].nunique()
        col_w = data.groupby(facet_col, sort=False, observed=True)[xcol].nunique()
        subplot_args["row_heights"] = row_h.tolist()
        subplot_args["column_widths"] = col_w.tolist()
    fig, layout = facet_plotly(
        data, facet_row, facet_col, title_dim, margin_titles, **subplot_args
    )
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        cur_args = transform_arguments(facet_df, kwargs)
        trace = go.Heatmap(name="<br>".join([rlab, clab]), **cur_args)
        fig.add_trace(trace, row=ir + 1, col=ic + 1)
    return fig


def map_gofunc(
    data,
    func,
    facet_row,
    facet_col,
    margin_titles=True,
    title_dim: str = None,
    subplot_args: dict = dict(),
    **kwargs,
):
    fig, layout = facet_plotly(
        data, facet_row, facet_col, title_dim, margin_titles, **subplot_args
    )
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        cur_args = transform_arguments(facet_df, kwargs)
        trace = func(name="<br>".join([rlab, clab]), **cur_args)
        fig.add_trace(trace, row=ir + 1, col=ic + 1)
    return fig


def map_colors(a, cc=qualitative.D3, return_colors=False):
    cmap = dict()
    for x, c in zip(np.unique(a), itt.cycle(cc)):
        cmap[x] = convert_colors_to_same_type(c)[0][0]
    if return_colors:
        return a.map(cmap)
    else:
        return cmap


def add_color_opacity(rgb, alpha):
    r, g, b = unlabel_rgb(rgb)
    return "rgba({},{},{},{})".format(r, g, b, alpha)

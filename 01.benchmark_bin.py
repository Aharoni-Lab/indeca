# %% import and definition
import itertools as itt
import os

import cv2
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from tqdm.auto import tqdm

from routine.plotting import map_gofunc
from routine.update_bin import (
    construct_G,
    construct_R,
    estimate_coefs,
    max_thres,
    solve_deconv,
    solve_deconv_bin,
)
from routine.utilities import norm

IN_PATH = {
    "org": "./intermediate/simulated/simulated-ar-samp.nc",
    "upsamp": "./intermediate/simulated/simulated-ar-upsamp.nc",
}
INT_PATH = "./intermediate/benchmark_bin"
FIG_PATH = "./figs/benchmark_bin"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10
PARAM_EST_AR = False
PARAM_SIG_LEV = (0.05, 0.5)

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


def dilate1d(a, kernel):
    return cv2.dilate(a.astype(float), kernel).squeeze()


def compute_dist(trueS, newS, metric, corr_dilation=0):
    if metric == "correlation" and corr_dilation:
        kn = np.ones(2 * corr_dilation + 1)
        trueS = xr.apply_ufunc(
            dilate1d,
            trueS.compute(),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            kwargs={"kernel": kn},
        )
        newS = xr.apply_ufunc(
            dilate1d,
            newS.compute(),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            kwargs={"kernel": kn},
        )
    if metric == "edit":
        dist = np.array(
            [
                Levenshtein.distance(
                    np.array(trueS.sel(unit_id=uid)), np.array(newS.sel(unit_id=uid))
                )
                for uid in trueS.coords["unit_id"]
            ]
        )
    else:
        dist = np.diag(
            cdist(
                trueS.transpose("unit_id", "frame"),
                newS.transpose("unit_id", "frame"),
                metric=metric,
            )
        )
    Sname = newS.name.split("-")
    if "org" in Sname:
        mthd = "original"
        Sname.remove("org")
    elif "upsamp" in Sname:
        mthd = "upsampled"
        Sname.remove("upsamp")
    elif "updn" in Sname:
        mthd = "updn"
        Sname.remove("updn")
    else:
        mthd = "unknown"
    return pd.DataFrame(
        {
            "variable": "-".join(Sname),
            "method": mthd,
            "metric": metric,
            "unit_id": trueS.coords["unit_id"],
            "dist": dist,
        }
    )


def norm_per_cell(S):
    return xr.apply_ufunc(
        norm,
        S.astype(float).compute(),
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
    )


def compute_metrics(S, S_true, mets, nthres: int = None, coarsen=None):
    S, S_true = S.dropna("frame"), S_true.dropna("frame")
    if nthres is not None:
        S_ls = max_thres(S, nthres)
    else:
        S_ls = [S]
    if coarsen is not None:
        S_ls = [s.coarsen(coarsen).sum() for s in S_ls]
        S_ls = [
            s.assign_coords({"frame": np.ceil(s.coords["frame"]).astype(int)})
            for s in S_ls
        ]
    res_ls = [compute_dist(S_true, curS, met) for curS, met in itt.product(S_ls, mets)]
    return pd.concat(res_ls)


# %% temporal update
sps_penal = 1
max_iters = 50
for up_type, up_factor in {"org": 1, "upsamp": PARAM_UPSAMP}.items():
    # get data
    sim_ds = xr.open_dataset(IN_PATH[up_type])
    C_gt = sim_ds["C"].dropna("frame", how="all")
    subset = C_gt.coords["unit_id"]
    np.random.seed(42)
    sig_lev = xr.DataArray(
        np.random.uniform(
            low=PARAM_SIG_LEV[0], high=PARAM_SIG_LEV[1], size=C_gt.sizes["unit_id"]
        ),
        dims=["unit_id"],
        coords={"unit_id": C_gt.coords["unit_id"]},
        name="sig_lev",
    )
    noise = np.random.normal(loc=0, scale=1, size=C_gt.shape)
    Y_solve = (C_gt * sig_lev + noise).sel(unit_id=subset)
    updt_ds = [Y_solve.rename("Y_solve"), sig_lev.sel(unit_id=subset)]
    iter_df = []
    # update
    res = {"C": [], "S": [], "b": [], "C-bin": [], "S-bin": [], "b-bin": [], "scal": []}
    for y in tqdm(
        Y_solve.transpose("unit_id", "frame"), total=Y_solve.sizes["unit_id"]
    ):
        # parameters
        y_norm = np.array(y)
        T = len(y_norm)
        g, tn = estimate_coefs(y_norm, p=2, noise_freq=0.4, use_smooth=False, add_lag=0)
        if PARAM_EST_AR:
            G = construct_G(g, T * up_factor, fromTau=False)
        else:
            G = construct_G(
                (PARAM_TAU_D * up_factor, PARAM_TAU_R * up_factor),
                T * up_factor,
                fromTau=True,
            )
        R = construct_R(T, up_factor)
        # org algo
        c, s, b = solve_deconv(y_norm, G, l1_penal=sps_penal * tn, R=R)
        res["C"].append(c.squeeze())
        res["S"].append(s.squeeze())
        res["b"].append(b)
        # bin algo
        c_bin, s_bin, b_bin, scale, it_df = solve_deconv_bin(y_norm, G, R)
        it_df["unit_id"] = y.coords["unit_id"].item()
        it_df["up_type"] = up_type
        iter_df.append(it_df)
        res["C-bin"].append(c_bin.squeeze())
        res["S-bin"].append(s_bin.squeeze())
        res["b-bin"].append(b_bin)
        res["scal"].append(scale)
    # save variables
    for vname, dat in res.items():
        dat = np.stack(dat)
        if dat.ndim == 1:
            updt_ds.append(
                xr.DataArray(
                    dat,
                    dims="unit_id",
                    coords={"unit_id": Y_solve.coords["unit_id"]},
                    name=vname,
                )
            )
        elif dat.ndim == 2:
            updt_ds.append(
                xr.DataArray(
                    dat,
                    dims=["unit_id", "frame"],
                    coords={
                        "unit_id": Y_solve.coords["unit_id"],
                        "frame": (
                            sim_ds.coords["frame"]
                            if up_type == "upsamp"
                            else Y_solve.coords["frame"]
                        ),
                    },
                    name=vname,
                )
            )
        else:
            raise ValueError
    updt_ds = xr.merge(updt_ds)
    updt_ds.to_netcdf(os.path.join(INT_PATH, "updt_ds-{}.nc".format(up_type)))
    iter_df = pd.concat(iter_df, ignore_index=True)
    iter_df.to_feather(os.path.join(INT_PATH, "iter_df-{}.feat".format(up_type)))


# %% compute metrics
def compute_ROC_percell(S, S_true, nthres=None, ds=None, th_min=0.1, th_max=0.9):
    S, S_true = np.array(S), np.array(S_true)
    pos_idx, neg_idx = np.where(S_true > 0)[0], np.where(S_true == 0)[0]
    true_pos = np.sum(np.array(S_true[pos_idx]))
    true_neg = len(neg_idx)
    if nthres is not None:
        S_ls, thres = max_thres(
            S, nthres, th_min=th_min, th_max=th_max, return_thres=True, ds=ds
        )
    else:
        S_ls = [S]
        thres = [-1]
    pos_dev = np.array([np.abs(s[pos_idx] - S_true[pos_idx]).sum() for s in S_ls])
    neg_dev = np.array([s[neg_idx].sum() for s in S_ls])
    return pd.DataFrame(
        {
            "thres": thres,
            "true_pos": (1 - pos_dev / true_pos).clip(0, 1),
            "false_pos": neg_dev / true_neg,
        }
    )


def compute_ROC(S, S_true, metadata=None, **kwargs):
    met_df = []
    for uid in S.coords["unit_id"]:
        met = compute_ROC_percell(S.sel(unit_id=uid), S_true.sel(unit_id=uid), **kwargs)
        met["unit_id"] = uid.item()
        met_df.append(met)
    met_df = pd.concat(met_df, ignore_index=True)
    if metadata is not None:
        for key, val in metadata.items():
            met_df[key] = val
    return met_df


th_range = (0.01, 0.99)
nthres = 981
met_df = []
for up_type, true_ds in IN_PATH.items():
    updt_ds = xr.open_dataset(os.path.join(INT_PATH, "updt_ds-{}.nc".format(up_type)))
    true_ds = xr.open_dataset(true_ds)
    if up_type == "upsamp":
        metS = compute_ROC(
            updt_ds["S"],
            true_ds["S"].dropna("frame", how="all"),
            nthres=nthres,
            th_min=th_range[0],
            th_max=th_range[1],
            ds=PARAM_UPSAMP,
            metadata={"method": "CNMF", "dataset": "upsamp-down"},
        )
        metS_bin = compute_ROC(
            updt_ds["S-bin"].coarsen({"frame": PARAM_UPSAMP}).sum(),
            true_ds["S"].dropna("frame", how="all"),
            metadata={"method": "minian-bin", "dataset": "upsamp-down"},
        )
        metS_up = compute_ROC(
            updt_ds["S"],
            true_ds["S_true"],
            nthres=nthres,
            th_min=th_range[0],
            th_max=th_range[1],
            metadata={"method": "CNMF", "dataset": "upsamp"},
        )
        metS_up_bin = compute_ROC(
            updt_ds["S-bin"],
            true_ds["S_true"],
            metadata={"method": "minian-bin", "dataset": "upsamp"},
        )
        met_df.extend([metS, metS_bin, metS_up, metS_up_bin])
    else:
        metS = compute_ROC(
            updt_ds["S"],
            true_ds["S"],
            nthres=nthres,
            th_min=th_range[0],
            th_max=th_range[1],
            metadata={"method": "CNMF", "dataset": "org"},
        )
        metS_bin = compute_ROC(
            updt_ds["S-bin"],
            true_ds["S"],
            metadata={"method": "minian-bin", "dataset": "org"},
        )
        met_df.extend([metS, metS_bin])
met_df = pd.concat(met_df, ignore_index=True)
met_df["thres"] = met_df["thres"].round(5)
met_df.to_feather(os.path.join(INT_PATH, "metrics.feat"))


# %% plot metrics
def plot_met(data, color=None):
    ax = plt.gca()
    met_cnmf = data[data["method"] == "CNMF"]
    met_bin = data[data["method"] == "minian-bin"]
    sns.lineplot(
        met_cnmf,
        x="false_pos",
        y="true_pos",
        hue="unit_id",
        estimator=None,
        lw=1,
        alpha=0.9,
        ax=ax,
    )
    sns.scatterplot(
        met_bin,
        x="false_pos",
        y="true_pos",
        hue="unit_id",
        s=50,
        marker="X",
        zorder=2.5,
        ax=ax,
    )
    ax.axline((0, 0), slope=1, lw=1, alpha=0.8, color="black", ls=":")


def plot_met_scatter(data, color=None):
    ax = plt.gca()
    met_cnmf = data[data["method"] == "CNMF"]
    met_bin = data[data["method"] == "minian-bin"]
    sns.scatterplot(met_cnmf, x="false_pos", y="true_pos", hue="thres", s=10, ax=ax)
    sns.scatterplot(
        met_bin, x="false_pos", y="true_pos", s=40, c="black", marker="x", ax=ax
    )
    ax.axline((0, 0), slope=1, lw=1, alpha=0.8, color="black", ls=":")


met_df = pd.read_feather(os.path.join(INT_PATH, "metrics.feat"))
np.random.seed(42)
met_sub = met_df[
    met_df["unit_id"].isin(np.random.choice(met_df["unit_id"].unique(), 10))
].astype({"unit_id": str})
g = sns.FacetGrid(met_sub, col="dataset", sharex=False, sharey=False)
g.map_dataframe(plot_met)
met_sub = met_df[
    np.logical_or(
        met_df["thres"].isin(np.linspace(0.1, 0.9, 9)), met_df["method"] == "minian-bin"
    )
].astype({"thres": str})
g.figure.savefig(os.path.join(FIG_PATH, "ROC.svg"), bbox_inches="tight")
g = sns.FacetGrid(met_sub, col="dataset", sharex=False, sharey=False)
g.map_dataframe(plot_met_scatter)
g.add_legend()
g.figure.savefig(os.path.join(FIG_PATH, "ROC_scatter.svg"), bbox_inches="tight")

# %% plot alpha correlations
sig_scal = updt_ds[["sig_lev", "scal-upsamp"]].to_dataframe()
sig_scal["S_gt"] = S_gt.mean("frame").to_series()
sig_scal["S_org"] = S_updn.mean("frame").to_series()
sig_scal["S_bin"] = S_bin_updn.mean("frame").to_series()
fig = make_subplots(
    rows=1,
    cols=3,
    shared_xaxes="all",
    shared_yaxes=False,
    subplot_titles=["Original", "Binary", "Binary Scale"],
    horizontal_spacing=0.1,
)
fig.add_trace(
    go.Histogram2dContour(x=sig_scal["sig_lev"], y=sig_scal["S_org"], showscale=False),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram2dContour(x=sig_scal["sig_lev"], y=sig_scal["S_bin"], showscale=False),
    row=1,
    col=2,
)
fig.add_trace(
    go.Histogram2dContour(
        x=sig_scal["sig_lev"], y=sig_scal["scal-upsamp"], showscale=False
    ),
    row=1,
    col=3,
)
fig.add_trace(
    go.Scatter(
        x=sig_scal["sig_lev"],
        y=sig_scal["S_org"],
        mode="markers",
        marker=dict(
            symbol="circle",
            opacity=0.6,
            color="white",
            size=6,
            line=dict(width=1),
        ),
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=sig_scal["sig_lev"],
        y=sig_scal["S_bin"],
        mode="markers",
        marker=dict(
            symbol="circle",
            opacity=0.6,
            color="white",
            size=6,
            line=dict(width=1),
        ),
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=sig_scal["sig_lev"],
        y=sig_scal["scal-upsamp"],
        mode="markers",
        marker=dict(
            symbol="circle",
            opacity=0.6,
            color="white",
            size=6,
            line=dict(width=1),
        ),
        showlegend=False,
    ),
    row=1,
    col=3,
)
fig.update_xaxes(title_text="Signal Level")
fig.update_yaxes(title_text="Mean S", row=1, col=1)
fig.update_yaxes(title_text="Mean S-bin", row=1, col=2)
fig.update_yaxes(title_text="Alpha scale", row=1, col=3)
fig.write_html(os.path.join(FIG_PATH, "scaling.html"))


# %% plot examples
nsamp = min(5, len(subset))
fig_dict = {
    "original": [S_gt, C_gt, Y_solve, S_org, S_bin_org] + max_thres(S_org, 9),
    "updn": [S_gt, C_gt, Y_solve, S_updn, S_bin_updn]
    + [
        s.coarsen({"frame": 10}).sum().assign_coords(frame=S_gt.coords["frame"])
        for s in max_thres(S_up.rename("S-updn"), 9)
    ],
}
met_sub = (
    met_res[met_res["variable"] == "S-bin"]
    .sort_values(["method", "metric"])
    .set_index(["method", "metric"])
)
for mthd, plt_trs in fig_dict.items():
    cur_uids = met_sub.loc[mthd, "edit"].sort_values("dist")["unit_id"]
    for met_grp, exp_set in {
        "best": cur_uids[:nsamp],
        "worst": cur_uids[-nsamp:],
        "fair": cur_uids[int(len(cur_uids) / 2) : int(len(cur_uids) / 2) + nsamp],
    }.items():
        plt_dat = pd.concat(
            [
                norm_per_cell(tr.sel(unit_id=np.array(exp_set))).to_dataframe()
                for tr in plt_trs
            ],
            axis="columns",
        ).reset_index()
        plt_dat = plt_dat.melt(id_vars=["frame", "unit_id"]).sort_values(
            ["unit_id", "variable", "frame"]
        )
        fig = px.line(
            plt_dat, facet_row="unit_id", x="frame", y="value", color="variable"
        )
        fig.update_layout(height=nsamp * 150)
        fig.write_html(os.path.join(FIG_PATH, "exp-{}-{}.html".format(mthd, met_grp)))

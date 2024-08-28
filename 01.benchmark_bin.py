# %% import and definition
import itertools as itt
import os

import cv2
import Levenshtein
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import xarray as xr
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from routine.update_bin import (
    construct_G,
    construct_R,
    estimate_coefs,
    max_thres,
    solve_deconv,
    solve_deconv_bin,
)
from routine.utilities import norm

IN_PATH = "./intermediate/simulated/simulated-upsamp.nc"
INT_PATH = "./intermediate/benchmark_bin"
FIG_PATH = "./figs/benchmark_bin"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10
PARAM_EST_AR = False
PARAM_SIG_MEAN = 2
PARAM_SIG_VAR = 2

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# %% temporal update
sim_ds = xr.open_dataset(IN_PATH)
subset = sim_ds["A"].coords["unit_id"][:5]
Y_solve, A, C_gt, S_gt, C_gt_true, S_gt_true = (
    sim_ds["Y"],
    sim_ds["A"],
    sim_ds["C"],
    sim_ds["S"],
    sim_ds["C_true"],
    sim_ds["S_true"],
)
A, C_gt, S_gt = (
    A.sel(unit_id=subset),
    C_gt.sel(unit_id=subset).dropna("frame", how="all"),
    S_gt.sel(unit_id=subset).dropna("frame", how="all"),
)
np.random.seed(42)
sig_lev = xr.DataArray(
    np.random.normal(
        loc=PARAM_SIG_MEAN, scale=PARAM_SIG_VAR, size=C_gt.sizes["unit_id"]
    ).clip(0.1, 10),
    dims=["unit_id"],
    coords={"unit_id": C_gt.coords["unit_id"]},
    name="sig_lev",
)
noise = np.random.normal(loc=0, scale=1, size=C_gt.shape)
Y_solve = C_gt * sig_lev + noise
updt_ds = [Y_solve.rename("Y_solve"), sig_lev]
sps_penal = 10
max_iters = 50
metric_df = []
for up_type, up_factor in {"org": 1, "upsamp": PARAM_UPSAMP}.items():
    res = {"C": [], "S": [], "b": [], "C-bin": [], "S-bin": [], "b-bin": [], "scal": []}
    for y in tqdm(
        Y_solve.transpose("unit_id", "frame"), total=Y_solve.sizes["unit_id"]
    ):
        # parameters
        y_norm = np.array(norm(y))
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
        res["C"].append(c)
        res["S"].append(s)
        res["b"].append(b)
        # bin algo
        c_bin, s_bin, b_bin, scale, met_df = solve_deconv_bin(y_norm, G, R)
        met_df["unit_id"] = y.coords["unit_id"].item()
        met_df["up_type"] = up_type
        metric_df.append(met_df)
        res["C-bin"].append(c_bin)
        res["S-bin"].append(s_bin)
        res["b-bin"].append(b_bin)
        res["scal"].append(scale)
    # save variables
    for vname, dat in res.items():
        if vname.startswith("b") or vname.startswith("scal"):
            updt_ds.append(
                xr.DataArray(
                    np.array(dat),
                    dims="unit_id",
                    coords={"unit_id": A.coords["unit_id"]},
                    name="-".join([vname, up_type]),
                )
            )
        else:
            updt_ds.append(
                xr.DataArray(
                    np.concatenate(dat, axis=1),
                    dims=["frame", "unit_id"],
                    coords={
                        "frame": (
                            C_gt_true.coords["frame"]
                            if up_type == "upsamp"
                            else Y_solve.coords["frame"]
                        ),
                        "unit_id": A.coords["unit_id"],
                    },
                    name="-".join([vname, up_type]),
                )
            )
updt_ds = xr.merge(updt_ds)
updt_ds.to_netcdf(os.path.join(INT_PATH, "updt_ds.nc"))
metric_df = pd.concat(metric_df, ignore_index=True)
metric_df.to_feather(os.path.join(INT_PATH, "metrics.feat"))


# %% plot example and metrics
def dilate1d(a, kernel):
    return cv2.dilate(a.astype(float), kernel).squeeze()


def compute_dist(trueS, newS, metric, corr_dilation=1):
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


updt_ds = xr.open_dataset(os.path.join(INT_PATH, "updt_ds.nc"))
true_ds = xr.open_dataset(IN_PATH).isel(unit_id=updt_ds.coords["unit_id"])
subset = updt_ds.coords["unit_id"]
S_gt, S_gt_true, C_gt = (
    true_ds["S"].dropna("frame", how="all"),
    true_ds["S_true"],
    true_ds["C"].dropna("frame", how="all"),
)
S_org, S_bin_org, S_up, S_bin_up, Y_solve = (
    updt_ds["S-org"].dropna("frame", how="all"),
    updt_ds["S-bin-org"].dropna("frame", how="all"),
    updt_ds["S-upsamp"],
    updt_ds["S-bin-upsamp"],
    updt_ds["Y_solve"].dropna("frame", how="all"),
)
S_updn, S_bin_updn = (
    S_up.coarsen({"frame": 10}).sum().rename("S-updn"),
    S_bin_up.coarsen({"frame": 10}).sum().rename("S-bin-updn"),
)
S_updn = S_updn.assign_coords({"frame": np.ceil(S_updn.coords["frame"]).astype(int)})
S_bin_updn = S_bin_updn.assign_coords(
    {"frame": np.ceil(S_bin_updn.coords["frame"]).astype(int)}
)
met_ds = [
    (S_org, S_gt, {"mets": ["correlation"]}),
    (S_org, S_gt, {"mets": ["correlation", "hamming", "edit"], "nthres": 9}),
    (S_bin_org, S_gt, {"mets": ["correlation", "hamming", "edit"]}),
    (S_up, S_gt_true, {"mets": ["correlation"]}),
    (S_up, S_gt_true, {"mets": ["correlation", "hamming", "edit"], "nthres": 9}),
    (S_bin_up, S_gt_true, {"mets": ["correlation", "hamming", "edit"]}),
    (S_updn, S_gt, {"mets": ["correlation"]}),
    (
        S_up.rename("S-updn"),
        S_gt,
        {
            "mets": ["correlation", "hamming", "edit"],
            "nthres": 9,
            "coarsen": {"frame": 10},
        },
    ),
    (S_bin_updn, S_gt, {"mets": ["correlation", "hamming", "edit"]}),
]
met_res = pd.concat(
    [compute_metrics(m[0], m[1], **m[2]) for m in met_ds], ignore_index=True
)
sns.set_theme(style="darkgrid")
g = sns.FacetGrid(
    met_res,
    row="metric",
    col="method",
    sharey="row",
    sharex="col",
    aspect=2,
    hue="variable",
    margin_titles=True,
)
g.map_dataframe(
    sns.violinplot,
    x="variable",
    y="dist",
    bw_adjust=0.5,
    cut=0.3,
    saturation=0.6,
    # log_scale=True,
)
# g.map_dataframe(sns.swarmplot, x="variable", y="dist", edgecolor="auto", linewidth=1)
g.tick_params(axis="x", rotation=90)
g.figure.savefig(os.path.join(FIG_PATH, "metrics.svg"), dpi=500, bbox_inches="tight")
nsamp = min(10, len(subset))
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

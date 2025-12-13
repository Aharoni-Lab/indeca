import itertools as itt

import cv2
import dask as da
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from dtaidistance import dtw
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

from indeca.core.deconv import max_thres
from indeca.utils import norm


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


def apply_dtw(src_arr, dst_arr, psi=0, window=10, **kwargs):
    src_arr, dst_arr = np.array(src_arr).astype(float), np.array(dst_arr).astype(float)
    best_path = dtw.warping_path_fast(
        src_arr,
        dst_arr,
        psi=psi,
        window=window,
        **kwargs,
    )
    best_path = pd.DataFrame(best_path, columns=["src", "tgt"])
    best_path["src_val"] = np.nan
    for src_idx, src_df in best_path.groupby("src"):
        idx = src_df.index
        best_path.loc[idx, "src_val"] = src_arr[src_idx] / len(idx)
    warp = best_path.groupby("tgt")["src_val"].sum()
    # assert src_arr.sum() == warp.sum()
    return warp


def compute_ROC_percell(
    S, S_true, nthres=None, ds=None, th_min=0.1, th_max=0.9, use_warp=False, meta=dict()
):
    S, S_true = np.nan_to_num(np.array(S)), np.nan_to_num(np.array(S_true))
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
    if use_warp:
        S_ls = [np.array(apply_dtw(s, S_true)) for s in S_ls]
    pos_dev = np.array([np.abs(s[pos_idx] - S_true[pos_idx]).sum() for s in S_ls])
    neg_dev = np.array([s[neg_idx].sum() for s in S_ls])
    corr = np.array([np.corrcoef(s, S_true)[0, 1] for s in S_ls])
    cos = np.array(
        [
            cosine_similarity(s.reshape((1, -1)), S_true.reshape((1, -1))).item()
            for s in S_ls
        ]
    )
    return pd.DataFrame(
        {
            "thres": thres,
            "true_pos": (1 - pos_dev / true_pos).clip(0, 1),
            "false_pos": neg_dev / true_neg,
            "corr": corr,
            "cosine": cos,
            **meta,
        }
    )


def compute_ROC(S, S_true, metadata=None, **kwargs):
    met_df = []
    for uid in S.coords["unit_id"]:
        met = da.delayed(compute_ROC_percell)(
            S.sel(unit_id=uid),
            S_true.sel(unit_id=uid),
            meta={"unit_id": uid.item()},
            **kwargs,
        )
        met_df.append(met)
    met_df = da.compute(met_df)[0]
    met_df = pd.concat(met_df, ignore_index=True)
    met_df["f1"] = met_df["true_pos"] / (
        met_df["true_pos"] + 0.5 * (1 - met_df["true_pos"] + met_df["false_pos"])
    )
    if metadata is not None:
        for key, val in metadata.items():
            met_df[key] = val
    return met_df


def plot_ROC(data, color=None):
    ax = plt.gca()
    met_thres = data[data["method"].isin(["CNMF", "indeca-scal"])]
    met_bin = data[data["method"] == "indeca"]
    sns.lineplot(
        met_thres,
        x="false_pos",
        y="true_pos",
        hue="method",
        estimator=None,
        lw=1,
        alpha=0.9,
        ax=ax,
    )
    sns.scatterplot(
        met_bin,
        x="false_pos",
        y="true_pos",
        s=50,
        marker="X",
        zorder=2.5,
        color="black",
        ax=ax,
    )
    ax.axline((0, 0), slope=1, lw=1, alpha=0.8, color="black", ls=":")


def plot_ROC_scatter(data, color=None):
    ax = plt.gca()
    met_cnmf = data[data["method"] == "CNMF"]
    met_bin = data[data["method"] == "indeca"]
    sns.scatterplot(met_cnmf, x="false_pos", y="true_pos", hue="thres", s=10, ax=ax)
    sns.scatterplot(
        met_bin, x="false_pos", y="true_pos", s=40, c="black", marker="x", ax=ax
    )
    ax.axline((0, 0), slope=1, lw=1, alpha=0.8, color="black", ls=":")


def plot_corr(data, color=None):
    ax = plt.gca()
    met_thres = data[data["method"].isin(["CNMF", "indeca-scal"])]
    met_bin = data[data["method"] == "indeca"]
    sns.lineplot(met_thres, x="thres", y="corr", hue="method", ax=ax)
    ax.axhline(met_bin["corr"].item(), lw=2, alpha=0.8, color="black", ls=":")

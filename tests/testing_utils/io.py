import os

import fsspec
import pandas as pd
from scipy.io import loadmat


def download_realds(target_path, target_dataset, remote_path="Ground_truth/"):
    fs = fsspec.filesystem("github", org="HelmchenLabSoftware", repo="Cascade")
    fs.get(
        remote_path + target_dataset,
        os.path.join(target_path, target_dataset),
        recursive=True,
    )


def load_gt_mat(matfile, varname="CAttached"):
    dat = loadmat(matfile, simplify_cells=True)[varname]
    fluo_df = (
        pd.DataFrame({k: v for k, v in dat.items() if k in ["fluo_time", "fluo_mean"]})
        .sort_values("fluo_time")
        .reset_index()
        .rename(columns={"index": "frame"})
    )
    ap_df = pd.DataFrame({"ap_time": dat["events_AP"] / 1e4}).sort_values("ap_time")
    ap_df = pd.merge_asof(
        ap_df,
        fluo_df[["frame", "fluo_time"]],
        left_on="ap_time",
        right_on="fluo_time",
        direction="nearest",
    )
    fluo_df = fluo_df.merge(
        ap_df.groupby("frame")["ap_time"].count().rename("ap_count"),
        on="frame",
        how="left",
    )
    fluo_df["ap_count"] = fluo_df["ap_count"].fillna(0).astype(int)
    return fluo_df, ap_df


def load_gt_ds(ds_path, return_ap=False):
    fluo = []
    ap = []
    for icell, matfile in enumerate(
        filter(lambda fn: fn.endswith(".mat"), os.listdir(ds_path))
    ):
        fluo_df, ap_df = load_gt_mat(os.path.join(ds_path, matfile))
        fluo_df["unit_id"] = icell
        ap_df["unit_id"] = icell
        fluo.append(fluo_df)
        ap.append(ap_df)
    fluo = pd.concat(fluo, ignore_index=True)
    ap = pd.concat(ap, ignore_index=True)
    Y = fluo.set_index(["unit_id", "frame"])["fluo_mean"].rename("Y").to_xarray()
    if return_ap:
        return Y, ap, fluo
    else:
        S_true = (
            fluo.set_index(["unit_id", "frame"])["ap_count"]
            .rename("S_true")
            .to_xarray()
        )
        return Y, S_true

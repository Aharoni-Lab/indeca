import os

import fsspec
import numpy as np
import pandas as pd
from scipy.io import loadmat


def download_realds(target_path, target_dataset, remote_path="Ground_truth/"):
    fs = fsspec.filesystem("github", org="HelmchenLabSoftware", repo="Cascade")
    fs.get(
        remote_path + target_dataset,
        target_path,
        recursive=True,
    )


def load_gt_mat(matfile):
    rec_ls = load_recordings_from_file(matfile)
    fluo_df_all = []
    ap_df_all = []
    for i, rec in enumerate(rec_ls):
        fluo_df = (
            pd.DataFrame({"fluo_time": rec["t"], "fluo_mean": rec["dff"]})
            .sort_values("fluo_time")
            .reset_index()
            .rename(columns={"index": "frame"})
        )
        ap_df = pd.DataFrame({"ap_time": rec["spikes"]}).sort_values("ap_time")
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
        fluo_df["trial"] = i
        ap_df["trial"] = i
        fluo_df["fps"] = rec["frame_rate"]
        ap_df["fps"] = rec["frame_rate"]
        fluo_df_all.append(fluo_df)
        ap_df_all.append(ap_df)
    fluo_df_all = pd.concat(fluo_df_all, ignore_index=True)
    ap_df_all = pd.concat(ap_df_all, ignore_index=True)
    assert fluo_df_all.notnull().all(axis=None)
    assert ap_df_all.notnull().all(axis=None)
    return fluo_df_all, ap_df_all


def load_recordings_from_file(file_path):
    """Load all recordings from a given file into list of dictionaries

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the recorded ground truth file (*.mat)

    Returns
    --------
    recording_list: list of dictionaries
        List of recordings for given file
        Each recording is a dictionary with 't', 'dff', 'spikes', 'frame_rate' as keys
        'spikes' are the spike times in seconds, aligned to the time 't' and fluorescence 'dff'
    """

    data = loadmat(file_path)["CAttached"][0]
    recording_list = list()
    for i, trial in enumerate(data):
        keys = trial[0][0].dtype.descr
        keys_unfolded = list(sum(keys, ()))
        traces_index = int(keys_unfolded.index("fluo_mean") / 2)
        fluo_time_index = int(keys_unfolded.index("fluo_time") / 2)
        events_index = int(keys_unfolded.index("events_AP") / 2)
        # spikes
        events = trial[0][0][events_index]
        events = events[
            ~np.isnan(events)
        ]  # exclude NaN entries for the Theis et al. data sets
        ephys_sampling_rate = 1e4
        event_time = events / ephys_sampling_rate
        # fluorescence
        fluo_times = np.squeeze(trial[0][0][fluo_time_index])
        fluo_times = fluo_times[~np.isnan(fluo_times)]
        frame_rate = 1 / np.mean(np.diff(fluo_times))
        traces_mean = np.squeeze(trial[0][0][traces_index])
        traces_mean = traces_mean[: fluo_times.shape[0]]
        traces_mean = traces_mean[~np.isnan(fluo_times)]
        recording_list.append(
            dict(
                dff=traces_mean, t=fluo_times, spikes=event_time, frame_rate=frame_rate
            )
        )
    return recording_list


def load_gt_ds(ds_path):
    fluo = []
    ap = []
    for icell, matfile in enumerate(
        filter(lambda fn: fn.endswith(".mat"), os.listdir(ds_path))
    ):
        fluo_df, ap_df = load_gt_mat(os.path.join(ds_path, matfile))
        fluo_df["unit_id"] = str(icell) + "-" + fluo_df["trial"].astype(str)
        ap_df["unit_id"] = str(icell) + "-" + ap_df["trial"].astype(str)
        fluo.append(fluo_df)
        ap.append(ap_df)
    fluo = pd.concat(fluo, ignore_index=True)
    ap = pd.concat(ap, ignore_index=True)
    Y = fluo.set_index(["unit_id", "frame"])["fluo_mean"].rename("Y").to_xarray()
    S_true = (
        fluo.set_index(["unit_id", "frame"])["ap_count"].rename("S_true").to_xarray()
    )
    return Y, S_true, ap, fluo

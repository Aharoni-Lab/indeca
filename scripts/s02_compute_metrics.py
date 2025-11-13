# %% imports and definitions
import ast
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from indeca.utils import norm
from tests.testing_utils.io import load_gt_ds, subset_gt_ds
from tests.testing_utils.metrics import compute_metrics
from tests.testing_utils.plotting import plot_traces

IN_DPATH = "./tests/data/"
IN_MLSPIKE_RES = "./tests/output/data/mlspike/"
IN_OASIS_RES = "./tests/output/data/oasis"
IN_INDECA_RES = "./tests/output/data/func/test_demo_pipeline_realds/"
OUT_EXT_PATH = "./tests/output/data/external/"
OUT_IND_PATH = "./tests/output/data/agg/metrics"
FIG_PATH = "./tests/output/figs/func/mlspike_comparison"

os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(OUT_EXT_PATH, exist_ok=True)
os.makedirs(OUT_IND_PATH, exist_ok=True)

# %% compute metrics for oasis
res_df = []
ncfiles = [f for f in os.listdir(IN_OASIS_RES) if f.endswith(".nc")]
for ncf in tqdm(ncfiles, desc="dataset"):
    dsname = os.path.splitext(ncf)[0]
    Y, S_true, ap_df, fluo_df = load_gt_ds(os.path.join(IN_DPATH, dsname))
    Y, S_true, ap_df, fluo_df = subset_gt_ds(Y, S_true, ap_df, fluo_df, dsname)
    oasis_ds = xr.open_dataset(os.path.join(IN_OASIS_RES, ncf))
    S = oasis_ds["S"].assign_coords(unit_id=Y.coords["unit_id"])
    for uid in tqdm(np.array(Y.coords["unit_id"]), desc="cell", leave=False):
        s_true = S_true.sel(unit_id=uid)
        sb = S.sel(unit_id=uid)
        cur_ap = ap_df.loc[uid]
        cur_fluo = fluo_df.loc[uid]
        met_dict = compute_metrics(
            s_slv=sb,
            s_ref=s_true,
            ap_df=cur_ap,
            fluo_df=cur_fluo,
            tdist_thres=5,
            compute_f1=False,
        )
        meta_dict = {
            "dsname": dsname,
            "method": "oasis",
            "use_all": False,
            "unit_id": uid,
        }
        res_df.append(meta_dict | met_dict)
met_path = os.path.join(OUT_EXT_PATH, "oasis")
os.makedirs(met_path, exist_ok=True)
res_df = pd.DataFrame(res_df)
res_df.to_feather(os.path.join(met_path, "metrics.feat"))

# %% compute metrics for mlspike
res_df = []
ncfiles = [f for f in os.listdir(IN_MLSPIKE_RES) if f.endswith(".nc")]
for ncf in tqdm(ncfiles, desc="dataset"):
    dsname = os.path.splitext(ncf)[0]
    Y, S_true, ap_df, fluo_df = load_gt_ds(os.path.join(IN_DPATH, dsname))
    Y, S_true, ap_df, fluo_df = subset_gt_ds(Y, S_true, ap_df, fluo_df, dsname)
    mlspk_ds = xr.open_dataset(os.path.join(IN_MLSPIKE_RES, ncf))
    S = mlspk_ds["S"].assign_coords(unit_id=Y.coords["unit_id"])
    for uid in tqdm(np.array(Y.coords["unit_id"]), desc="cell", leave=False):
        s_true = S_true.sel(unit_id=uid)
        sb = S.sel(unit_id=uid)
        cur_ap = ap_df.loc[uid]
        cur_fluo = fluo_df.loc[uid]
        met_dict = compute_metrics(
            s_slv=sb,
            s_ref=s_true,
            ap_df=cur_ap,
            fluo_df=cur_fluo,
            tdist_thres=5,
        )
        meta_dict = {
            "dsname": dsname,
            "method": "mlspike",
            "use_all": False,
            "unit_id": uid,
        }
        res_df.append(meta_dict | met_dict)
met_path = os.path.join(OUT_EXT_PATH, "mlspike")
os.makedirs(met_path, exist_ok=True)
res_df = pd.DataFrame(res_df)
res_df.to_feather(os.path.join(met_path, "metrics.feat"))

# %% compute metrics for indeca
res_df = []
ncfiles = [f for f in os.listdir(IN_INDECA_RES) if f.endswith(".nc")]
for ncf in tqdm(ncfiles, desc="dataset"):
    ds_indeca = xr.open_dataset(os.path.join(IN_INDECA_RES, ncf))
    pat = re.compile(r"^(?P<dsname>.+)-(?P<use_all>True|False)\.nc$")
    ma_dict = pat.match(ncf).groupdict()
    dsname, use_all = ma_dict["dsname"], ma_dict["use_all"]
    S = ds_indeca["S"]
    Y, S_true, ap_df, fluo_df = load_gt_ds(os.path.join(IN_DPATH, dsname))
    Y, S_true, ap_df, fluo_df = subset_gt_ds(Y, S_true, ap_df, fluo_df, dsname)
    for uid in tqdm(np.array(Y.coords["unit_id"]), desc="cell", leave=False):
        s_true = S_true.sel(unit_id=uid)
        sb = S.sel(unit_id=uid)
        cur_ap = ap_df.loc[uid]
        cur_fluo = fluo_df.loc[uid]
        met_dict = compute_metrics(
            s_slv=sb, s_ref=s_true, ap_df=cur_ap, fluo_df=cur_fluo, tdist_thres=5
        )
        meta_dict = {
            "dsname": dsname,
            "method": "indeca",
            "use_all": ast.literal_eval(use_all),
            "unit_id": uid,
            "ncell": "None",
            "tau_init": "None",
            "iter": "None",
            "test_id": "None",
            "upsamp": "None",
        }
        res_df.append(meta_dict | met_dict)
res_df = pd.DataFrame(res_df)
res_df.to_feather(os.path.join(OUT_IND_PATH, "metrics.feat"))

# %% load data and plot traces
res_df = []
ncfiles = [f for f in os.listdir(IN_MLSPIKE_RES) if f.endswith(".nc")]
for ncf in tqdm(ncfiles, desc="dataset"):
    dsname = os.path.splitext(ncf)[0]
    try:
        ds_indeca_all = xr.open_dataset(
            os.path.join(IN_INDECA_RES, "{}-True.nc".format(dsname))
        )
        ds_indeca_ind = xr.open_dataset(
            os.path.join(IN_INDECA_RES, "{}-False.nc".format(dsname))
        )
        S_ind_all = ds_indeca_all["S"]
        S_ind_ind = ds_indeca_ind["S"]
    except FileNotFoundError:
        continue
    Y, S_true, ap_df, fluo_df = load_gt_ds(os.path.join(IN_DPATH, dsname))
    Y, S_true, ap_df, fluo_df = subset_gt_ds(Y, S_true, ap_df, fluo_df, dsname)
    mlspk_ds = xr.open_dataset(os.path.join(IN_MLSPIKE_RES, ncf))
    oasis_ds = xr.open_dataset(os.path.join(IN_OASIS_RES, ncf))
    S_mlspk = mlspk_ds["S"].assign_coords(unit_id=Y.coords["unit_id"])
    S_oasis = oasis_ds["S"].assign_coords(unit_id=Y.coords["unit_id"])
    met_indeca = pd.read_feather(os.path.join(OUT_IND_PATH, "metrics.feat")).set_index(
        ["dsname", "unit_id", "use_all"]
    )["f1"]
    met_mlspike = pd.read_feather(
        os.path.join(OUT_EXT_PATH, "mlspike", "metrics.feat")
    ).set_index(["dsname", "unit_id"])["f1"]
    titles = []
    for uid in np.array(Y.coords["unit_id"]):
        titles.append(
            "uid: {}, indeca_all: {:.3f}, indeca_ind: {:.3f}, mlspike: {:.3f}".format(
                uid,
                met_indeca.loc[dsname, uid, True],
                met_indeca.loc[dsname, uid, False],
                met_mlspike.loc[dsname, uid],
            )
        )
    # plotting
    ncell = Y.shape[0]
    fig = make_subplots(rows=ncell, subplot_titles=titles)
    for iu, uid in enumerate(np.array(Y.coords["unit_id"])):
        fig.add_traces(
            plot_traces(
                {
                    "y": norm(Y.sel(unit_id=uid)) * 10,
                    "s_true": S_true.sel(unit_id=uid),
                    "mlspk": S_mlspk.sel(unit_id=uid),
                    "s_all": S_ind_all.sel(unit_id=uid),
                    "s_ind": S_ind_ind.sel(unit_id=uid),
                    "s_oasis": S_oasis.sel(unit_id=uid),
                }
            ),
            rows=iu + 1,
            cols=1,
        )
    fig.update_layout(height=350 * ncell, width=1400)
    fig.write_html(os.path.join(FIG_PATH, "{}.html".format(dsname)))

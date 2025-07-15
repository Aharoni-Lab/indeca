# %% imports and definitions
import os

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from tests.testing_utils.io import load_gt_ds, subset_gt_ds
from tests.testing_utils.metrics import assignment_distance, nzidx_int

IN_DPATH = "./tests/data/"
IN_MLSPIKE_RES = "./tests/output/data/mlspike/"
OUT_PATH = "./tests/output/data/external/mlspike"

os.makedirs(OUT_PATH, exist_ok=True)

# %% load data and compute metrics
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
        sb_idx = nzidx_int(np.array(sb).astype(int))
        t_sb = np.interp(sb_idx, cur_fluo["frame"], cur_fluo["fluo_time"])
        t_ap = cur_ap["ap_time"]
        mdist, f1, prec, rec = assignment_distance(
            t_ref=np.atleast_1d(t_ap),
            t_slv=np.atleast_1d(t_sb),
            tdist_thres=1,
        )
        res_df.append(
            pd.DataFrame(
                [
                    {
                        "dsname": dsname,
                        "method": "mlspike",
                        "use_all": False,
                        "unit_id": uid,
                        "mdist": mdist,
                        "f1": f1,
                        "prec": prec,
                        "rec": rec,
                    }
                ]
            )
        )
res_df = pd.concat(res_df, ignore_index=True)
res_df.to_feather(os.path.join(OUT_PATH, "metrics.feat"))

# %% imports and definitions
import os

import numpy as np
import xarray as xr
from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
from tqdm.auto import tqdm

from tests.testing_utils.io import load_gt_ds, subset_gt_ds

IN_DPATH = "./tests/data/"
OUT_PATH = "./tests/output/data/oasis"
DSNAMES = [
    "DS01-OGB1-m-V1",
    "DS02-OGB1-2-m-V1",
    # "DS03-Cal520-m-S1",
    "DS04-OGB1-zf-pDp",
    "DS06-GCaMP6f-zf-aDp",
    # "DS07-GCaMP6f-zf-dD",
    "DS08-GCaMP6f-zf-OB",
    "DS09-GCaMP6f-m-V1",
    "DS10-GCaMP6f-m-V1-neuropil-corrected",
    "DS11-GCaMP6f-m-V1-neuropil-corrected",
    "DS12-GCaMP6s-m-V1-neuropil-corrected",
    "DS13-GCaMP6s-m-V1-neuropil-corrected",
    "DS14-GCaMP6s-m-V1",
    "DS15-GCaMP6s-m-V1",
    "DS16-GCaMP6s-m-V1",
    "DS17-GCaMP5k-m-V1",
    "DS18-R-CaMP-m-CA3",
    "DS19-R-CaMP-m-S1",
    "DS20-jRCaMP1a-m-V1",
    "DS21-jGECO1a-m-V1",
    "DS22-OGB1-m-SST-V1",
    "DS23-OGB1-m-PV-V1",
    "DS24-GCaMP6f-m-PV-V1",
    "DS25-GCaMP6f-m-SST-V1",
    "DS26-GCaMP6f-m-VIP-V1",
    "DS27-GCaMP6f-m-PV-vivo-V1",
    "DS28-XCaMPgf-m-V1",
    "DS29-GCaMP7f-m-V1",
    "DS30-GCaMP8f-m-V1",
    "DS31-GCaMP8m-m-V1",
    "DS32-GCaMP8s-m-V1",
    "DS33-Interneurons2023-m-V1",
    "DS40-GCaMP6s-spinal-cord-excitatory",
    "DS41-GCaMP6s-spinal-cord-inhibitory",
    "X-DS09-GCaMP6f-m-V1",
]
QTHRES = [0.01, 0.05, 0.1, 0.2, 0.5]

os.makedirs(OUT_PATH, exist_ok=True)

# %% test caiman
res_df = []
for dsname in tqdm(DSNAMES, desc="dataset"):
    Y, S_true, ap_df, fluo_df = load_gt_ds(os.path.join(IN_DPATH, dsname))
    Y, S_true, ap_df, fluo_df = subset_gt_ds(Y, S_true, ap_df, fluo_df, dsname)
    S_ls = []
    for uid in tqdm(np.array(Y.coords["unit_id"]), desc="cell", leave=False):
        assert len(ap_df) > 0
        y = np.array(Y.sel(unit_id=uid))
        s_true = S_true.sel(unit_id=uid)
        c, bl, c1, g, sn, cur_s, lam = constrained_foopsi(y, p=2)
        S_ls.append(
            xr.DataArray(
                cur_s, dims=["frame"], coords={"frame": Y.coords["frame"]}
            ).assign_coords({"unit_id": uid})
        )
    S_cm = xr.concat(S_ls, dim="unit_id")
    ds = xr.merge([Y.rename("Y"), S_true.rename("S_true"), S_cm.rename("S")])
    ds.to_netcdf(os.path.join(OUT_PATH, "{}.nc".format(dsname)))

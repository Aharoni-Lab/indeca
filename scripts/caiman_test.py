# %% imports and definitions
import os

import numpy as np
import pandas as pd
from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm

from tests.testing_utils.io import load_gt_ds, subset_gt_ds
from tests.testing_utils.metrics import assignment_distance, dtw_corr, nzidx_int

IN_DPATH = "./tests/data/"
OUT_PATH = "./tests/output/data/external/caiman"
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
    for uid in tqdm(np.array(Y.coords["unit_id"]), desc="cell", leave=False):
        assert len(ap_df) > 0
        y = np.array(Y.sel(unit_id=uid))
        s_true = S_true.sel(unit_id=uid)
        c, bl, c1, g, sn, cur_s, lam = constrained_foopsi(y, p=2)
        corr_raw = np.corrcoef(s_true, cur_s)[0, 1]
        corr_gs = np.corrcoef(
            gaussian_filter1d(s_true, 1), gaussian_filter1d(cur_s, 1)
        )[0, 1]
        corr_dtw = dtw_corr(s_true, cur_s)
        for qthres in QTHRES:
            sb = np.around(cur_s / (qthres * cur_s.max())).astype(int)
            # tau_d, tau_r = tau_cnmf[iu, :]
            # try:
            #     (dhm0, dhm1), _ = find_dhm(
            #         True, np.array([tau_d, tau_r]), np.array([1, -1])
            #     )
            # except AssertionError:
            #     dhm0, dhm1 = 0, 0
            cur_ap = ap_df.loc[uid]
            cur_fluo = fluo_df.loc[uid]
            sb_idx = nzidx_int(sb)
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
                            "method": "oasis",
                            "use_all": False,
                            "unit_id": uid,
                            "qthres": qthres,
                            "mdist": mdist,
                            "f1": f1,
                            "prec": prec,
                            "rec": rec,
                            # "dhm0": dhm0,
                            # "dhm1": dhm1,
                            "corr_raw": corr_raw,
                            "corr_gs": corr_gs,
                            "corr_dtw": corr_dtw,
                        }
                    ]
                )
            )
res_df = pd.concat(res_df, ignore_index=True)
res_df.to_feather(os.path.join(OUT_PATH, "metrics.feat"))

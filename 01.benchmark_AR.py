# %% import and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr

from routine.update_AR import (
    convolve_g,
    convolve_h,
    fit_sumexp,
    solve_fit_h,
    solve_g,
    solve_h,
)
from routine.update_bin import construct_G, estimate_coefs
from routine.utilities import scal_like

IN_PATH = "./intermediate/simulated/simulated.nc"
INT_PATH = "./intermediate/benchmark_ar"
FIG_PATH = "./figs/benchmark_ar"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10
PARAM_EST_AR = True

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# %% AR update reverse solve approach
sim_ds = xr.open_dataset(IN_PATH)
Y, A, C_gt, S_gt, C_gt_true, S_gt_true = (
    sim_ds["Y"],
    sim_ds["A"],
    sim_ds["C"].dropna("frame", how="all"),
    sim_ds["S"].dropna("frame", how="all"),
    sim_ds["C_true"],
    sim_ds["S_true"],
)
c, s = np.array(C_gt.isel(unit_id=0)).reshape((-1, 1)), np.array(
    S_gt.isel(unit_id=0)
).reshape((-1, 1))
noise_lev = [0, 1, 2, 5, 10]
methods = [
    # "est-smth0",
    "est-smth20",
    "est-smth100",
    "est-naive",
    "solve-l1",
    # "solve-l2",
    "free-l1",
]
res_df = []
for ns in noise_lev:
    np.random.seed(0)
    y = c + ns * (np.random.random(c.shape) - 0.5)
    res_df.append(
        pd.DataFrame(
            {
                "method": "y",
                "noise": ns,
                "frame": np.arange(len(y)),
                "value": y.squeeze(),
            }
        )
    )
    res_df.append(
        pd.DataFrame(
            {
                "method": "c",
                "noise": ns,
                "frame": np.arange(len(y)),
                "value": c.squeeze(),
            }
        )
    )
    res_df.append(
        pd.DataFrame(
            {
                "method": "s",
                "noise": ns,
                "frame": np.arange(len(y)),
                "value": scal_like(s.squeeze(), c),
            }
        )
    )
    for mthd in methods:
        m, param = mthd.split("-")
        if m == "est":
            if param.startswith("smth"):
                g, tn = estimate_coefs(
                    y, p=2, noise_freq=0.1, use_smooth=True, add_lag=int(param[4:])
                )
            else:
                g, tn = estimate_coefs(
                    y, p=2, noise_freq=0.5, use_smooth=False, add_lag=0
                )
            c_est = scal_like(convolve_g(s, g), c)
        elif m == "solve":
            g = solve_g(y, s, norm=param)
            G = construct_G(g, len(y))
            y_est = (G @ y).squeeze()
            res_df.append(
                pd.DataFrame(
                    {
                        "method": mthd + "-y",
                        "noise": ns,
                        "frame": np.arange(len(y)),
                        "value": y_est,
                    }
                )
            )
            c_est = scal_like(convolve_g(s, g), c)
        elif m == "free":
            h_nopen = solve_h(y, s)
            _, h_nopen_fit = fit_sumexp(h_nopen, 2)
            lams, h, h_fit, mets, h_df = solve_fit_h(y, s)
            c_est = convolve_h(s, h)
            g = None
            res_df.append(
                pd.DataFrame(
                    {
                        "method": mthd + "-no_penal",
                        "noise": ns,
                        "frame": np.arange(len(y)),
                        "value": convolve_h(s, h_nopen),
                    }
                )
            )
            res_df.append(
                pd.DataFrame(
                    {
                        "method": mthd + "-no_penal-fit",
                        "noise": ns,
                        "frame": np.arange(len(y)),
                        "value": convolve_h(s, h_nopen_fit),
                    }
                )
            )
            res_df.append(
                pd.DataFrame(
                    {
                        "method": mthd + "-fit",
                        "noise": ns,
                        "frame": np.arange(len(y)),
                        "value": convolve_h(s, h_fit),
                    }
                )
            )
        print("method: {}, g: {}".format(mthd, g))
        res_df.append(
            pd.DataFrame(
                {
                    "method": mthd,
                    "noise": ns,
                    "frame": np.arange(len(y)),
                    "value": c_est,
                }
            )
        )
res_df = pd.concat(res_df, ignore_index=True)
fig = px.line(
    res_df, facet_row="noise", x="frame", y="value", color="method", line_group="noise"
)
fig.write_html(os.path.join(FIG_PATH, "AR_update.html"))

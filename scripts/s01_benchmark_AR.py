# %% import and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from tqdm.auto import tqdm

from minian_bin.simulation import AR2exp, eval_exp, find_dhm, tau2AR
from minian_bin.AR_kernel import (
    convolve_g,
    convolve_h,
    fit_sumexp,
    solve_fit_h,
    solve_g,
    solve_h,
)
from minian_bin.update_bin import construct_G, estimate_coefs
from minian_bin.utils import scal_like

IN_PATH = "./intermediate/simulated/simulated.nc"
INT_PATH = "./intermediate/benchmark_ar"
FIG_PATH = "./figs/benchmark_ar"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10
PARAM_EST_AR = True

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


# %% benchmark individual update approaches
def updt_est(y, s, **kwargs):
    g, _ = estimate_coefs(y, **kwargs)
    return convolve_g(s, g)


def updt_solve(y, s, **kwargs):
    g = solve_g(y, s, **kwargs)
    return convolve_g(s, g)


def updt_free(y, s, use_fit=True, **kwargs):
    lams, ps, h, h_fit, mets, h_df = solve_fit_h(y, s, **kwargs)
    if use_fit:
        return convolve_h(s, h_fit)
    else:
        return convolve_h(s, h)


sim_ds = xr.open_dataset(IN_PATH)
Y, A, C_gt, S_gt = (
    sim_ds["Y"],
    sim_ds["A"],
    sim_ds["C"].dropna("frame", how="all"),
    sim_ds["S"].dropna("frame", how="all"),
)
c, s = np.array(C_gt.isel(unit_id=0)).reshape((-1, 1)), np.array(
    S_gt.isel(unit_id=0)
).reshape((-1, 1))
noise_lev = [0, 0.5, 1, 2, 3]
methods = {
    "est-smth0.1": lambda y, s: updt_est(
        y, s, p=2, noise_freq=0.1, use_smooth=True, add_lag=20
    ),
    "est-smth0.05": lambda y, s: updt_est(
        y, s, p=2, noise_freq=0.05, use_smooth=True, add_lag=20
    ),
    "est-naive": lambda y, s: updt_est(
        y, s, p=2, noise_freq=0.5, use_smooth=False, add_lag=20
    ),
    "solve": updt_solve,
    "free": lambda y, s: updt_free(y, s, use_fit=False, fit_method="numerical"),
    "free-fit": lambda y, s: updt_free(y, s, use_fit=True, fit_method="numerical"),
}
res_df = []
for ns in noise_lev:
    np.random.seed(42)
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
    for mthd, updt_func in methods.items():
        y_est = scal_like(updt_func(y, s), c)
        res_df.append(
            pd.DataFrame(
                {
                    "method": mthd,
                    "noise": ns,
                    "frame": np.arange(len(y)),
                    "value": y_est,
                }
            )
        )
res_df = pd.concat(res_df, ignore_index=True)
fig = px.line(
    res_df, facet_row="noise", x="frame", y="value", color="method", line_group="noise"
)
fig.write_html(os.path.join(FIG_PATH, "exp_traces.html"))


# %% benchmark update across multiple cells
def free_kn(y, s, **kwargs):
    res = solve_fit_h(y, s, **kwargs)
    taus = -1 / res[0]
    if (np.imag(taus) == 0).all():
        return True, taus, res[1]
    else:
        return AR2exp(*tau2AR(*taus))


sim_ds = xr.open_dataset(IN_PATH)
subset = slice(0, 100)
C_gt, S_gt = (
    sim_ds["C"].dropna("frame", how="all").isel(unit_id=subset),
    sim_ds["S"].dropna("frame", how="all").isel(unit_id=subset),
)
noise_lev = [0, 0.5, 1, 2, 3]
methods = {
    "free-full": lambda y, s: free_kn(y, s, fit_method="solve"),
    "free-full-num": lambda y, s: free_kn(y, s, fit_method="numerical"),
    "free-ind": lambda y, s: free_kn(y, s, fit_method="solve"),
    "free-ind-num": lambda y, s: free_kn(y, s, fit_method="numerical"),
    "est-smth0.1": lambda y, s: AR2exp(
        *estimate_coefs(y, p=2, noise_freq=0.1, use_smooth=True, add_lag=20)[0]
    ),
    "est-smth0.05": lambda y, s: AR2exp(
        *estimate_coefs(y, p=2, noise_freq=0.05, use_smooth=True, add_lag=20)[0]
    ),
    "est-naive": lambda y, s: AR2exp(
        *estimate_coefs(y, p=2, noise_freq=0.5, use_smooth=False, add_lag=20)[0]
    ),
    "solve": lambda y, s: AR2exp(*solve_g(y, s)),
}
res_df = []
for ns in noise_lev:
    np.random.seed(42)
    c_all, s_all = np.array(C_gt.transpose("unit_id", "frame")), np.array(
        S_gt.transpose("unit_id", "frame")
    )
    y_all = c_all + ns * (np.random.random(c_all.shape) - 0.5)
    for mthd, update_func in methods.items():
        if "-full" in mthd:
            y_in, s_in = y_all.reshape(
                (1, y_all.shape[0], y_all.shape[1])
            ), s_all.reshape((1, s_all.shape[0], s_all.shape[1]))
        else:
            y_in, s_in = y_all, s_all
        for yy, ss in tqdm(zip(y_in, s_in), total=y_in.shape[0]):
            is_biexp, tconst, coefs = update_func(yy, ss)
            try:
                (r0, r1), t_hat = find_dhm(is_biexp, tconst, coefs)
            except AssertionError:
                r0, r1 = np.nan, np.nan
            res_df.append(
                pd.DataFrame(
                    [
                        {
                            "method": mthd,
                            "noise": ns,
                            "is_biexp": is_biexp,
                            "tconst0": tconst[0],
                            "tconst1": tconst[1],
                            "coef0": coefs[0],
                            "coef1": coefs[1],
                            "r0": r0,
                            "r1": r1,
                            "t_hat": t_hat,
                        }
                    ]
                )
            )
res_df = pd.concat(res_df, ignore_index=True)
res_df.to_feather(os.path.join(INT_PATH, "coefs.feat"))

# %% plot result
res_df = pd.read_feather(os.path.join(INT_PATH, "coefs.feat"))
plt_df = res_df[
    res_df["method"].isin(
        [
            "free-full-num",
            "free-ind-num",
            "est-smth0.1",
            "est-smth0.05",
            "est-naive",
        ]
    )
].copy()
plt_df["method"] = plt_df["method"].replace(
    {"free-full-num": "free-full", "free-ind-num": "free-ind"}
)
(r0_true, r1_true), _ = find_dhm(*AR2exp(*tau2AR(6, 1)))
fig = px.scatter(
    plt_df,
    x="r0",
    y="r1",
    color="method",
    facet_col="noise",
    symbol="is_biexp",
    symbol_map={True: "circle", False: "circle-open"},
)
fig.add_vline(r0_true, line_dash="dot", line_color="grey", line_width=1, opacity=0.65)
fig.add_hline(r1_true, line_dash="dot", line_color="grey", line_width=1, opacity=0.65)
fig.update_traces(marker={"size": 5, "opacity": 0.8})
fig.write_html(os.path.join(FIG_PATH, "coefs.html"))


# %% debugging all algorithm
def free_kn(y, s, **kwargs):
    res = solve_fit_h(y, s, **kwargs)
    taus = -1 / res[0]
    if (np.imag(taus) == 0).all():
        return True, taus, res[1]
    else:
        return AR2exp(*tau2AR(*taus))


sim_ds = xr.open_dataset(IN_PATH)
subset = slice(0, 100)
C_gt, S_gt = (
    sim_ds["C"].dropna("frame", how="all").isel(unit_id=subset),
    sim_ds["S"].dropna("frame", how="all").isel(unit_id=subset),
)
plt_df = (
    sim_ds.isel(unit_id=0)[["C", "S"]]
    .to_dataframe()
    .reset_index()
    .melt(id_vars=["frame", "unit_id"])
    .dropna()
)
fig = px.line(plt_df, x="frame", y="value", color="variable")
fig.write_html(os.path.join(FIG_PATH, "dbg-raw_traces.html"))
c_all, s_all = np.array(C_gt.transpose("unit_id", "frame")), np.array(
    S_gt.transpose("unit_id", "frame")
)
res = []
mets = []
for ns in [0, 0.5, 1, 2, 3]:
    print("noise level: {}".format(ns))
    np.random.seed(42)
    y_all = c_all + ns * (np.random.random(c_all.shape) - 0.5)
    lams, ps, h, h_fit, metric_df, h_df = solve_fit_h(
        y_all, s_all, max_iters=10, verbose=True, fit_method="numerical"
    )
    taus = -1 / lams
    if (np.imag(taus) == 0).all():
        is_biexp, tconst, coefs = True, taus, ps
    else:
        is_biexp, tconst, coefs = AR2exp(*tau2AR(*taus))
    (r0, r1), t_hat = find_dhm(is_biexp, tconst, coefs)
    h_df_tall = h_df.melt(id_vars=["iter", "smth_penal", "frame"])
    h_df_tall["value"] = np.real(h_df_tall["value"])
    h_df_tall["noise"] = ns
    mets.append(
        pd.DataFrame(
            [
                {
                    "noise": ns,
                    "is_biexp": is_biexp,
                    "tconst0": tconst[0],
                    "tconst1": tconst[1],
                    "coef0": coefs[0],
                    "coef1": coefs[1],
                    "r0": r0,
                    "r1": r1,
                    "t_hat": t_hat,
                }
            ]
        )
    )
    res.append(h_df_tall)
res = pd.concat(res, ignore_index=True)
mets = pd.concat(mets, ignore_index=True)
fig = px.line(
    res,
    x="frame",
    y="value",
    color="smth_penal",
    facet_col="variable",
    facet_row="noise",
)
fig.write_html(os.path.join(FIG_PATH, "dbg-free-all.html"))

# %% debugging exp fit
t = np.linspace(0, 50, 1000)
y = eval_exp(t, is_biexp, tconst, coefs)
(r0, r1), t_hat = find_dhm(is_biexp, tconst, coefs)
fig = px.line(x=t, y=y)
fig.add_vline(r0, line_color="grey", line_dash="dot")
fig.add_vline(r1, line_color="grey", line_dash="dot")
fig.show()

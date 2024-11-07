# %% 0. Handle imports and definitions
import os

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sps
import xarray as xr
from plotly.subplots import make_subplots
from scipy.linalg import convolution_matrix
from scipy.optimize import direct
from tqdm.auto import tqdm

from minian_bin.benchmark_utils import compute_ROC
from minian_bin.simulation import exp_pulse, tau2AR
from minian_bin.update_bin import max_thres, prob_deconv, solve_deconv, solve_deconv_l0
from minian_bin.update_pipeline import pipeline_bin, pipeline_cnmf
from minian_bin.utilities import scal_lstsq

IN_PATH = {
    "org": "./intermediate/simulated/simulated-exp-samp.nc",
    "upsamp": "./intermediate/simulated/simulated-exp-upsamp.nc",
}
INT_PATH = "./intermediate/better_l0"
FIG_PATH = "./figs/better_l0"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10
PARAM_EST_AR = False
PARAM_SIG_LEV = (1, 5)

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# %% better l0
# get data
sim_ds = xr.open_dataset(IN_PATH["org"])
C_gt = sim_ds["C"].dropna("frame", how="all")
subset = C_gt.coords["unit_id"]
np.random.seed(42)
sig_lev = xr.DataArray(
    np.sort(
        np.random.uniform(
            low=PARAM_SIG_LEV[0],
            high=PARAM_SIG_LEV[1],
            size=C_gt.sizes["unit_id"],
        )
    ),
    dims=["unit_id"],
    coords={"unit_id": C_gt.coords["unit_id"]},
    name="sig_lev",
)
use_l0 = True
noise = np.random.normal(loc=0, scale=1, size=C_gt.shape)
Y_solve = (C_gt * sig_lev + noise).sel(unit_id=subset).transpose("unit_id", "frame")
kn, _, _ = exp_pulse(PARAM_TAU_D, PARAM_TAU_R, nsamp=60)
ar_coef = np.array(tau2AR(PARAM_TAU_D, PARAM_TAU_R))
K = sps.csc_matrix(
    convolution_matrix(kn, Y_solve.sizes["frame"])[: Y_solve.sizes["frame"], :]
)
RK = K.todense()
prob = prob_deconv(
    Y_solve.sizes["frame"], coef_len=2, ar_mode=True, amp_constraint=True
)
C_ls, S_ls = [], []
metrics = []
for uid in np.arange(5, 100, 5):
    y = Y_solve.sel(unit_id=uid)
    sig = sig_lev.sel(unit_id=uid)
    for penal in tqdm(np.linspace(0, 20, 50)):
        if use_l0:
            c, s, b, err, met_df = solve_deconv_l0(
                np.array(y),
                prob,
                ar_coef,
                l0_penal=penal,
                scale=np.array(sig),
                return_obj=True,
            )
        else:
            c, s, b, err = solve_deconv(
                np.array(y),
                prob,
                ar_coef,
                l1_penal=penal,
                scale=np.array(sig),
                return_obj=True,
                warm_start=penal > 0,
            )
            met_df = pd.DataFrame([{"err": err}])
        th_svals = max_thres(np.abs(s), 1000, th_min=0, th_max=1)
        th_cvals = [RK @ ss for ss in th_svals]
        th_scals = [scal_lstsq(cc, y) for cc in th_cvals]
        th_objs = [
            np.linalg.norm(y - scl * np.array(cc).squeeze() - b, ord=1)
            for scl, cc in zip(th_scals, th_cvals)
        ]
        opt_idx = np.argmin(th_objs)
        opt_s = th_svals[opt_idx]
        opt_obj = th_objs[opt_idx]
        opt_scal = th_scals[opt_idx]
        met_df["penal"] = penal
        met_df["err"] = err
        met_df["err_opt"] = opt_obj
        met_df["scale"] = opt_scal
        met_df["unit_id"] = uid
        C_ls.append(c)
        S_ls.append(s)
        metrics.append(met_df)
metrics = pd.concat(metrics, ignore_index=True)
# metrics.to_feather(os.path.join(INT_PATH, "metrics.feat"))

# %% plotting
fig = px.line(metrics, x="penal", y="err", color="unit_id")
fig.write_html(os.path.join(FIG_PATH, "err.html"))
fig = px.line(metrics, x="penal", y="err_opt", color="unit_id")
fig.write_html(os.path.join(FIG_PATH, "err_opt.html"))


# %% try direct l0
def solve_deconv_l0_err(x, y, prob, kn, scal, RK, return_err_only=True):
    c, s, b, err, met_df = solve_deconv_l0(
        np.array(y),
        prob,
        kn,
        l0_penal=x,
        scale=scal,
        return_obj=True,
    )
    th_svals = max_thres(np.abs(s), 1000, th_min=0, th_max=1)
    th_cvals = [RK @ ss for ss in th_svals]
    th_scals = [scal_lstsq(cc, y) for cc in th_cvals]
    th_objs = [
        np.linalg.norm(y - scl * np.array(cc).squeeze() - b, ord=1)
        for scl, cc in zip(th_scals, th_cvals)
    ]
    opt_idx = np.argmin(th_objs)
    opt_s = th_svals[opt_idx]
    opt_obj = th_objs[opt_idx]
    opt_scal = th_scals[opt_idx]
    if return_err_only:
        return opt_obj
    else:
        return opt_s, opt_obj, opt_scal


sim_ds = xr.open_dataset(IN_PATH["org"])
C_gt = sim_ds["C"].dropna("frame", how="all")
subset = C_gt.coords["unit_id"]
max_iters = 100
np.random.seed(42)
sig_lev = xr.DataArray(
    np.sort(
        np.random.uniform(
            low=PARAM_SIG_LEV[0],
            high=PARAM_SIG_LEV[1],
            size=C_gt.sizes["unit_id"],
        )
    ),
    dims=["unit_id"],
    coords={"unit_id": C_gt.coords["unit_id"]},
    name="sig_lev",
)
noise = np.random.normal(loc=0, scale=1, size=C_gt.shape)
Y_solve = (C_gt * sig_lev + noise).sel(unit_id=subset).transpose("unit_id", "frame")
kn, _, _ = exp_pulse(PARAM_TAU_D, PARAM_TAU_R, nsamp=60)
K = sps.csc_matrix(
    convolution_matrix(kn, Y_solve.sizes["frame"])[: Y_solve.sizes["frame"], :]
)
RK = K.todense()
prob = prob_deconv(
    Y_solve.sizes["frame"], ar_mode=True, coef_len=2, amp_constraint=True
)
C_ls, S_ls = [], []
metrics = []
for uid in tqdm(np.arange(5, 100, 20)):
    y = Y_solve.sel(unit_id=uid)
    sig = np.array(sig_lev.sel(unit_id=uid))
    t_d, t_r, scal = tau2AR(PARAM_TAU_D, PARAM_TAU_R, return_scl=True)
    coef = np.array([t_d, t_r])
    scal = scal * sig
    ub = np.linalg.norm(y, 1)
    for i_iter in range(max_iters):
        c, s, b, err, met_df = solve_deconv_l0(
            np.array(y),
            prob,
            coef,
            l0_penal=ub,
            scale=scal,
            return_obj=True,
        )
        if (s > 0).sum() > 0:
            ub = ub * 2
            print("uid: {}, ub: {}".format(uid, ub))
            break
        else:
            ub = ub / 2
    else:
        print("max ub iterations reached")
    res = direct(
        solve_deconv_l0_err,
        args=(y, prob, coef, scal, RK),
        bounds=[(0, ub)],
        maxfun=20,
    )
    l0_opt = res.x
    opt_s, opt_obj, opt_scal = solve_deconv_l0_err(
        l0_opt, y, prob, coef, scal, RK, return_err_only=False
    )
    metrics.append(
        pd.DataFrame(
            [
                {
                    "unit_id": uid,
                    "ub": ub,
                    "l0_opt": res.x,
                    "success": res.success,
                    "nfev": res.nfev,
                    "err": opt_obj,
                    "scal": opt_scal,
                }
            ]
        )
    )
metrics = pd.concat(metrics, ignore_index=True)
metrics.to_feather(os.path.join(INT_PATH, "metrics.feat"))

# %% 0. Handle imports and definitions
import functools as fct
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
from scipy.special import huber
from tqdm.auto import tqdm

from benchmarks.benchmark_utils import compute_ROC
from minian_bin.deconv import DeconvBin
from minian_bin.simulation import AR2tau, exp_pulse, tau2AR
from minian_bin.update_bin import (
    max_thres,
    prob_deconv,
    prob_deconv_osqp,
    solve_deconv,
    solve_deconv_l0,
)
from minian_bin.pipeline import pipeline_bin, pipeline_cnmf
from minian_bin.utils import scal_lstsq

IN_PATH = {
    "org": "./intermediate/simulated/simulated-samp.nc",
    "upsamp": "./intermediate/simulated/simulated-upsamp.nc",
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
use_l0 = False
penal_max = {"l1": 0.02, "l2": 0.11, "huber": 0.08}
noise = np.random.normal(loc=0, scale=1, size=C_gt.shape)
Y_solve = (C_gt * sig_lev + noise).sel(unit_id=subset).transpose("unit_id", "frame")
kn, _, _ = exp_pulse(PARAM_TAU_D, PARAM_TAU_R, nsamp=60)
ar_coef = np.array(tau2AR(PARAM_TAU_D, PARAM_TAU_R))
K = sps.csc_matrix(
    convolution_matrix(kn, Y_solve.sizes["frame"])[: Y_solve.sizes["frame"], :]
)
RK = K.todense()
for nm in ["huber", "l1", "l2"]:
    prob = prob_deconv(
        Y_solve.sizes["frame"],
        coef_len=2,
        ar_mode=True,
        amp_constraint=True,
        norm=nm,
    )
    C_ls, S_ls = [], []
    metrics = []
    for uid in np.arange(5, 100, 15):
        y = np.array(Y_solve.sel(unit_id=uid))
        sig = sig_lev.sel(unit_id=uid)
        if nm == "l1":
            pmax = np.sum(np.abs(y))
        elif nm == "l2":
            pmax = np.sum(y**2)
        elif nm == "huber":
            pmax = np.sum(huber(1, y)) * 2
        for penal in tqdm(np.linspace(0, penal_max[nm], 100)):
            if use_l0:
                c, s, b, err, met_df = solve_deconv_l0(
                    y,
                    prob,
                    ar_coef,
                    l0_penal=penal * pmax,
                    scale=np.array(sig),
                    return_obj=True,
                )
            else:
                c, s, b, err = solve_deconv(
                    y,
                    prob,
                    ar_coef,
                    l1_penal=penal * pmax,
                    scale=np.array(sig),
                    return_obj=True,
                    warm_start=penal > 0,
                    solver=cp.CLARABEL,
                )
                met_df = pd.DataFrame([{"err": err}])
            th_svals = max_thres(np.abs(s), 1000, th_min=0, th_max=1)
            th_cvals = [RK @ ss for ss in th_svals]
            th_scals = [scal_lstsq(cc, y) for cc in th_cvals]
            if nm == "l1":
                th_objs = [
                    np.abs(y - scl * np.array(cc).squeeze() - b).sum()
                    for scl, cc in zip(th_scals, th_cvals)
                ]
            elif nm == "l2":
                th_objs = [
                    np.sum((y - scl * np.array(cc).squeeze() - b) ** 2)
                    for scl, cc in zip(th_scals, th_cvals)
                ]
            elif nm == "huber":
                th_objs = [
                    np.sum(huber(1, y - scl * np.array(cc).squeeze() - b)) * 2
                    for scl, cc in zip(th_scals, th_cvals)
                ]
            opt_idx = np.argmin(th_objs)
            opt_s = th_svals[opt_idx]
            opt_obj = th_objs[opt_idx]
            opt_scal = th_scals[opt_idx]
            met_df["penal"] = penal
            met_df["err"] = err
            met_df["rel_err"] = err / pmax
            met_df["err_opt"] = opt_obj
            met_df["rel_err_opt"] = opt_obj / pmax
            met_df["scale"] = opt_scal
            met_df["unit_id"] = uid
            C_ls.append(c)
            S_ls.append(s)
            metrics.append(met_df)
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_feather(os.path.join(INT_PATH, "metrics-{}.feat".format(nm)))
    fig = px.line(
        metrics, x="penal", y="rel_err", color="unit_id", template="plotly_dark"
    )
    fig.write_html(os.path.join(FIG_PATH, "{}-err.html".format(nm)))
    fig = px.line(
        metrics, x="penal", y="rel_err_opt", color="unit_id", template="plotly_dark"
    )
    fig.write_html(os.path.join(FIG_PATH, "{}-err_opt.html".format(nm)))
    fig = px.line(
        metrics, x="penal", y="scale", color="unit_id", template="plotly_dark"
    )
    fig.write_html(os.path.join(FIG_PATH, "{}-scale.html".format(nm)))


# %% try direct l0
def solve_deconv_l0_err(
    x, y, prob, prob_osqp, kn, scal, RK, return_err_only=True, backend="cvxpy"
):
    c, s, b, err, met_df = solve_deconv_l0(
        np.array(y),
        prob,
        prob_osqp,
        kn,
        l0_penal=x,
        scale=scal,
        return_obj=True,
        backend=backend,
    )
    th_svals = max_thres(np.abs(s), 1000, th_min=0, th_max=1)
    th_cvals = [RK @ ss for ss in th_svals]
    th_scals = [scal_lstsq(cc, y) for cc in th_cvals]
    th_objs = [
        np.linalg.norm(y - scl * np.array(cc).squeeze() - b, ord=2)
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


backend = "emosqp"
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
np.random.seed(42)
noise = np.random.normal(loc=0, scale=1, size=C_gt.shape)
Y_solve = (C_gt * sig_lev + noise).sel(unit_id=subset).transpose("unit_id", "frame")
kn, _, _ = exp_pulse(PARAM_TAU_D, PARAM_TAU_R, nsamp=60)
K = sps.csc_matrix(
    convolution_matrix(kn, Y_solve.sizes["frame"])[: Y_solve.sizes["frame"], :]
)
RK = K.todense()
upsamp = 1
prob = prob_deconv(
    Y_solve.sizes["frame"] * upsamp,
    ar_mode=False,
    coef_len=60,
    amp_constraint=True,
    norm="l2",
)
C_ls, S_ls = [], []
metrics = []
for uid in tqdm(np.arange(5, 100, 20)):
    y = np.tile(Y_solve.sel(unit_id=uid), upsamp)
    sig = np.array(sig_lev.sel(unit_id=uid))
    # t_d, t_r, scal = tau2AR(PARAM_TAU_D, PARAM_TAU_R, return_scl=True)
    # coef = np.array([t_d, t_r])
    # scal = scal * sig
    scal = sig
    prob_osqp = prob_deconv_osqp(y, kn, scal)
    ub = np.linalg.norm(y, 1)
    for i_iter in range(max_iters):
        c, s, b, err, met_df_osqp = solve_deconv_l0(
            np.array(y),
            prob,
            prob_osqp,
            kn,
            l0_penal=ub,
            scale=scal,
            return_obj=True,
            backend=backend,
        )
        # c, sA, b, err, met_df_A = solve_deconv_l0(
        #     np.array(y),
        #     prob,
        #     kn,
        #     l0_penal=ub,
        #     scale=scal,
        #     return_obj=True,
        #     backend="cvxpy",
        # )
        # c, sB, b, err, met_df_B = solve_deconv_l0(
        #     np.array(y),
        #     prob,
        #     kn,
        #     l0_penal=ub,
        #     scale=scal,
        #     return_obj=True,
        #     backend="osqp",
        # )
        # if (sA > 0).sum() != (sB > 0).sum():
        #     raise ValueError
        # else:
        #     s = sA
        if (s > 0).sum() > 0:
            ub = ub * 2
            print("uid: {}, ub: {}".format(uid, ub))
            break
        else:
            ub = ub / 2
    else:
        print("max ub iterations reached")
    res = direct(
        fct.partial(solve_deconv_l0_err, backend=backend),
        args=(y, prob, prob_osqp, kn, scal, RK),
        bounds=[(0, ub)],
        maxfun=1000,
        vol_tol=1e-4,
    )
    l0_opt = res.x[0]
    opt_s, opt_obj, opt_scal = solve_deconv_l0_err(
        l0_opt, y, prob, prob_osqp, kn, scal, RK, return_err_only=False, backend=backend
    )
    metrics.append(
        pd.DataFrame(
            [
                {
                    "unit_id": uid,
                    "ub": ub,
                    "l0_opt": l0_opt,
                    "success": res.success,
                    "nfev": res.nfev,
                    "err": opt_obj,
                    "scal": opt_scal,
                }
            ]
        )
    )
metrics = pd.concat(metrics, ignore_index=True)
metrics.to_feather(os.path.join(INT_PATH, "metrics_{}.feat".format(backend)))

# %% debugging individual iteration
met_osqp = pd.read_feather(os.path.join(INT_PATH, "metrics_osqp.feat"))
met_cvx = pd.read_feather(os.path.join(INT_PATH, "metrics_cvxpy.feat"))
met_cuosqp = pd.read_feather(os.path.join(INT_PATH, "metrics_cuosqp.feat"))
met_emosqp = pd.read_feather(os.path.join(INT_PATH, "metrics_emosqp.feat"))
sim_ds = xr.open_dataset(IN_PATH["org"])
C_gt = sim_ds["C"].dropna("frame", how="all")
uid = 5
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
np.random.seed(42)
noise = np.random.normal(loc=0, scale=1, size=C_gt.shape)
Y_solve = (C_gt * sig_lev + noise).transpose("unit_id", "frame")
kn, _, _ = exp_pulse(PARAM_TAU_D, PARAM_TAU_R, nsamp=60)
K = sps.csc_matrix(
    convolution_matrix(kn, Y_solve.sizes["frame"])[: Y_solve.sizes["frame"], :]
)
RK = K.todense()
y = Y_solve.sel(unit_id=uid)
scal = np.array(sig_lev.sel(unit_id=uid))
l0_penal = np.array(met_osqp["l0_opt"])[0]
prob = prob_deconv(
    Y_solve.sizes["frame"],
    ar_mode=False,
    coef_len=60,
    amp_constraint=True,
    norm="l2",
)
c, s_osqp, b, err, met_df_osqp = solve_deconv_l0(
    np.array(y),
    prob,
    kn,
    l0_penal=l0_penal,
    scale=scal,
    return_obj=True,
    backend="osqp",
)
c, s_cuosqp, b, err, met_df_cuosqp = solve_deconv_l0(
    np.array(y),
    prob,
    kn,
    l0_penal=l0_penal,
    scale=scal,
    return_obj=True,
    backend="cuosqp",
)

# %% try new deconv pipeline
it_df = pd.read_feather(
    "./intermediate/benchmark_pipeline_5best_est/iter_df-org.feat"
).set_index(["iter", "unit_id"])

sim_ds = xr.open_dataset(IN_PATH["org"])
C_gt = sim_ds["C"].dropna("frame", how="all")
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
# np.random.seed(42)
noise = np.random.normal(loc=0, scale=1, size=C_gt.shape)
Y_solve = (C_gt * sig_lev + noise).transpose("unit_id", "frame")
theta = tau2AR(PARAM_TAU_D, PARAM_TAU_R)
# _, _, p = AR2tau(theta[0], theta[1], solve_amp=True)
# kn, _, _ = exp_pulse(
#     PARAM_TAU_D,
#     PARAM_TAU_R,
#     p_d=p,
#     p_r=-p,
#     nsamp=Y_solve.sizes["frame"],
#     trunc_thres=1e-10,
# )
# trunc_idx = np.where(kn > 0)[0].max()
# kn = kn[:trunc_idx]
metrics = []
# for uid in tqdm(np.arange(5, 100, 20)):
for uid in tqdm(np.arange(95, 100)):
    y = np.array(Y_solve.sel(unit_id=uid))
    # dcv = DeconvBin(y=y, tau=(PARAM_TAU_D, PARAM_TAU_R), norm="l2", backend="osqp")
    theta = it_df.loc[0, uid][["g0", "g1"]].astype(float).values
    dcv = DeconvBin(y=y, theta=theta, norm="l2", backend="cvxpy")
    dcv_osqp = DeconvBin(y=y, theta=theta, norm="l2", backend="osqp")
    # dcv = DeconvBin(y=y, coef=kn, norm="l2", backend="cvxpy")
    cur_s, cur_c, cur_scal, cur_obj, cur_penal = dcv.solve_scale(reset_scale=True)
    cur_s_osqp, cur_c_osqp, cur_scal_osqp, cur_obj_osqp, cur_penal_osqp = (
        dcv_osqp.solve_scale(reset_scale=True)
    )
    s1 = dcv.solve()
    s1_osqp = dcv_osqp.solve()
    tau = it_df.loc[1, uid][["tau_d", "tau_r"]].astype(float).values
    dcv.update(tau=tau)
    dcv_osqp.update(tau=tau)
    s2 = dcv.solve()
    dcv_osqp.prob.update_settings(verbose=True)
    s2_osqp = dcv_osqp.solve()
    dcv_osqp.prob.update_settings(verbose=False)
    cur_s, cur_c, cur_scal, cur_obj, cur_penal = dcv.solve_scale()
    cur_s_osqp, cur_c_osqp, cur_scal_osqp, cur_obj_osqp, cur_penal_osqp = (
        dcv_osqp.solve_scale()
    )
    break
    cur_met = pd.DataFrame(
        [
            {
                "uid": uid,
                "nnz": (cur_s > 0).sum(),
                "scale": cur_scal,
                "scale_true": sig_lev.sel(unit_id=uid).item(),
                "obj": cur_obj,
                "penal": cur_penal,
            }
        ]
    )
    metrics.append(cur_met)
# metrics = pd.concat(metrics, ignore_index=True)
# %%
fig = px.line(cur_s)
fig.add_scatter(y=cur_s_osqp)
# fig.add_scatter(y=cur_s_osqp)
fig.show()

# %%
s2_ref = s2
s_ref = cur_s

# %%
metrics.to_feather("./met_ar_cvxpy.feat")
# print("free kernel, cvxpy:")
# print(pd.read_feather("./met_free_cvxpy.feat"))
# print("free kernel, osqp:")
# print(pd.read_feather("./met_free_osqp.feat"))
print("ar mode, cvxpy:")
print(pd.read_feather("./met_ar_cvxpy.feat"))
print("ar mode, osqp:")
print(pd.read_feather("./met_ar_osqp.feat"))

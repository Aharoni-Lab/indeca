# %% This lays out a potential workflow for the general use of minian-bin.
# Workflow is:
# 0. Handle imports and definitions
# 1. Generate or inport dataset at normal FPS and upsampled for calcium imaging
# 2. estimate initial guess at convolution kernel
# 3. Solve for non-binarized 's'
# 4. Binarize 's' using a single threshold across all cells and determine per cell scaling factor
# 5. Update free kernel based on binarized spikes
# 6. Optionally fit free kernel to bi-exponential and generate new kernel from this
# 7. Iterate back to step 4 and repeat until some metric is reached

# %% 0. Handle imports and definitions
import os

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr

from minian_bin.benchmark_utils import compute_ROC, norm_per_cell
from minian_bin.update_pipeline import pipeline_bin, pipeline_cnmf

IN_PATH = {
    "org": "./intermediate/simulated/simulated-exp-samp.nc",
    "upsamp": "./intermediate/simulated/simulated-exp-upsamp.nc",
}
INT_PATH = "./intermediate/benchmark_pipeline_5best_init"
FIG_PATH = "./figs/benchmark_pipeline_5best_init"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10
PARAM_EST_AR = False
PARAM_SIG_LEV = (1, 5)

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


# %% 1. Generate or import dataset at normal FPS for calcium imaging
sps_penal = 1
max_iters = 1
# for up_type, up_factor in {"org": 1, "upsamp": PARAM_UPSAMP}.items():
for up_type, up_factor in {"org": 1}.items():
    # get data
    sim_ds = xr.open_dataset(IN_PATH[up_type])
    C_gt = sim_ds["C"].dropna("frame", how="all")
    # C_gt = norm_per_cell(C_gt)
    subset = C_gt.coords["unit_id"][-5:]
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
    updt_ds = [Y_solve.rename("Y_solve"), sig_lev.sel(unit_id=subset)]
    iter_df = []
    # update
    (C_cnmf, S_cnmf) = pipeline_cnmf(
        np.array(Y_solve),
        up_factor,
        ar_mode=False,
        sps_penal=1,
        est_noise_freq=0.06,
        est_use_smooth=True,
        est_add_lag=50,
    )
    C_bin, S_bin, iter_df, C_bin_iter, S_bin_iter, h_iter, h_fit_iter = pipeline_bin(
        np.array(Y_solve),
        up_factor,
        max_iters=max_iters,
        tau_init=np.array([PARAM_TAU_D * up_factor, PARAM_TAU_R * up_factor]),
        return_iter=True,
        ar_use_all=True,
        ar_mode=False,
        est_noise_freq=0.06,
        est_use_smooth=True,
        est_add_lag=50,
    )
    res = {
        "C": C_bin,
        "S": S_bin,
        "C_iter": C_bin_iter,
        "S_iter": S_bin_iter,
        "C_cnmf": C_cnmf,
        "S_cnmf": S_cnmf,
    }
    # save variables
    for vname, dat in res.items():
        dat = np.stack(dat)
        if dat.ndim == 1:
            updt_ds.append(
                xr.DataArray(
                    dat,
                    dims="unit_id",
                    coords={"unit_id": Y_solve.coords["unit_id"]},
                    name=vname,
                )
            )
        elif dat.ndim == 2:
            updt_ds.append(
                xr.DataArray(
                    dat,
                    dims=["unit_id", "frame"],
                    coords={
                        "unit_id": Y_solve.coords["unit_id"],
                        "frame": (
                            sim_ds.coords["frame"]
                            if up_type == "upsamp"
                            else Y_solve.coords["frame"]
                        ),
                    },
                    name=vname,
                )
            )
        elif dat.ndim == 3:
            updt_ds.append(
                xr.DataArray(
                    dat,
                    dims=["iter", "unit_id", "frame"],
                    coords={
                        "iter": np.arange(dat.shape[0]),
                        "unit_id": Y_solve.coords["unit_id"],
                        "frame": (
                            sim_ds.coords["frame"]
                            if up_type == "upsamp"
                            else Y_solve.coords["frame"]
                        ),
                    },
                    name=vname,
                )
            )
        else:
            raise ValueError
    updt_ds = xr.merge(updt_ds)
    updt_ds.to_netcdf(os.path.join(INT_PATH, "updt_ds-{}.nc".format(up_type)))
    iter_df.to_feather(os.path.join(INT_PATH, "iter_df-{}.feat".format(up_type)))

# %% plot iteration performance
for up_type, in_path in IN_PATH.items():
    try:
        updt_ds = xr.open_dataset(
            os.path.join(INT_PATH, "updt_ds-{}.nc".format(up_type))
        )
        iter_df = pd.read_feather(
            os.path.join(INT_PATH, "iter_df-{}.feat".format(up_type))
        )
    except FileNotFoundError:
        continue
    sim_ds = xr.open_dataset(in_path)
    S_iter, S_true = updt_ds["S_iter"], sim_ds["S"]
    met_df = []
    for i in np.array(S_iter.coords["iter"]):
        met = compute_ROC(S_iter.sel(iter=i), S_true, metadata={"iter": i})
        met_df.append(met)
    met_df = pd.concat(met_df, ignore_index=True)
    fig_f1 = px.line(met_df, x="iter", y="f1", color="unit_id")
    fig_f1.write_html(os.path.join(FIG_PATH, "f1-{}.html".format(up_type)))
    iter_df["tau_d_diff"] = iter_df["tau_d"] - 6
    iter_df["tau_r_diff"] = iter_df["tau_r"] - 1
    itdf = iter_df.melt(
        id_vars=["iter", "cell"],
        var_name="coef",
        value_vars=["tau_d_diff", "tau_r_diff"],
        value_name="diff",
    )
    fig_coef = px.line(itdf, x="iter", y="diff", color="cell", line_dash="coef")
    fig_coef.write_html(os.path.join(FIG_PATH, "coef-{}.html".format(up_type)))

# %%
import matplotlib.pyplot as plt
import seaborn as sns

iter_df = pd.read_feather(os.path.join(INT_PATH, "iter_df-org.feat"))
tau_path = iter_df.groupby("iter").median().reset_index()
fig, axs = plt.subplots(2, figsize=(4, 6))
sns.lineplot(iter_df, x="iter", y="err", ax=axs[0])
axs[0].axhline(y=np.linalg.norm(noise, axis=1).mean(), c="black", lw=2, ls=":")
axs[0].set_title("error")
axs[1].plot(tau_path["tau_d"], tau_path["tau_r"], c="grey")
sns.scatterplot(
    tau_path,
    x="tau_d",
    y="tau_r",
    hue="iter",
    ax=axs[1],
    zorder=2.5,
)
axs[1].set_title("taus")
fig.tight_layout()

# %% test conv before thresholding
import plotly.graph_objects as go
import scipy.sparse as sps
from plotly.subplots import make_subplots
from scipy.linalg import convolution_matrix

from minian_bin.simulation import ar_pulse, exp_pulse, tau2AR
from minian_bin.update_bin import max_thres, scal_lstsq, solve_deconv

uid = 0
iiter = 0
ar_coef = tau2AR(6, 1)
sig = iter_df.set_index(["cell", "iter"]).loc[uid, iiter]["scale"]
sig_gt = sig_lev.sel(unit_id=subset[uid]).item()
y = Y_solve.isel(unit_id=uid)
s = updt_ds["S_iter"].isel(iter=iiter, unit_id=uid)
c = updt_ds["C_iter"].isel(iter=iiter, unit_id=uid) * sig
c_gt = sim_ds["C"].sel(unit_id=subset[uid]) * sig_gt
s_gt = sim_ds["S"].sel(unit_id=subset[uid])
# test mixin
kn = exp_pulse(6, 1, 60, p_d=1, p_r=-1)[0]
K = sps.csc_matrix(convolution_matrix(kn, len(y))[: len(y), :])
nthres = 1000
c_val, s_val, _ = solve_deconv(
    np.array(y), kn=kn, ar_mode=False, scale=sig_gt, amp_constraint=True
)
c_val, s_val = c_val.squeeze(), s_val.squeeze()
csizes = np.array([1, 3, 5, 7, 9])
res = []
svals = np.empty((len(csizes), len(s_val)))
for ics, csize in enumerate(csizes):
    hw_csize = int(csize / 2)
    s_conv = np.convolve(s_val, np.ones(csize) / csize)[
        hw_csize : len(s_val) + hw_csize
    ]
    th_svals, thres = max_thres(
        np.abs(s_conv), nthres, th_min=1e-4, th_max=1 - 1e-4, return_thres=True
    )
    th_cvals = [K @ ss for ss in th_svals]
    th_scals = [scal_lstsq(cc, y) for cc in th_cvals]
    th_objs = [
        np.linalg.norm(y - scl * np.array(cc).squeeze())
        for scl, cc in zip(th_scals, th_cvals)
    ]
    opt_idx = np.argmin(th_objs)
    opt_s = th_svals[opt_idx].astype(float)
    opt_c = th_cvals[opt_idx].astype(float)
    opt_obj = th_objs[opt_idx]
    opt_scal = th_scals[opt_idx]
    svals[ics, :] = opt_s
    res.append(
        pd.DataFrame({"csize": csize, "thres": thres, "scal": th_scals, "err": th_objs})
    )
res = pd.concat(res, ignore_index=True)
# c_mixin, s_mixin, _ = solve_deconv(
#     np.array(y), kn=kn, ar_mode=False, scale=sig_gt, mixin=True, solver="ECOS_BB"
# )
# res_pvt = (
#     res[["csize", "scal", "err"]]
#     .drop_duplicates()
#     .pivot(index="csize", columns="scal", values="err")
# )
# fig = px.imshow(res_pvt, x=res_pvt.columns, y=res_pvt.index)
err_gt = np.linalg.norm(y - c_gt)
fig = px.line(res, x="scal", y="err", facet_row="csize")
fig.add_hline(y=err_gt, line_color="grey", line_dash="dash", line_width=2)
fig.write_html("dbg-obj.html")
fig = make_subplots(2, 1, shared_xaxes=False, shared_yaxes=False)
fig.add_trace(go.Scatter(y=y, mode="lines", name="y"), row=1, col=1)
fig.add_trace(go.Scatter(y=s, mode="lines", name="s"), row=1, col=1)
fig.add_trace(go.Scatter(y=c, mode="lines", name="c"), row=1, col=1)
fig.add_trace(go.Scatter(y=s_val, mode="lines", name="s_val"), row=1, col=1)
fig.add_trace(go.Scatter(y=s_conv, mode="lines", name="s_conv"), row=1, col=1)
fig.add_trace(go.Scatter(y=opt_s, mode="lines", name="opt_s"), row=1, col=1)
fig.add_trace(go.Scatter(y=opt_c, mode="lines", name="opt_c"), row=1, col=1)
# fig.add_trace(
#     go.Scatter(y=s_mixin.squeeze(), mode="lines", name="s_mixin"), row=1, col=1
# )
# fig.add_trace(
#     go.Scatter(y=c_mixin.squeeze() * sig_gt, mode="lines", name="c_mixin"), row=1, col=1
# )
fig.add_trace(go.Scatter(y=s_gt, mode="lines", name="s_gt"), row=1, col=1)
fig.add_trace(go.Scatter(y=c_gt, mode="lines", name="c_gt"), row=1, col=1)
fig.add_trace(go.Scatter(y=h_iter[iiter], mode="lines", name="h_free"), row=2, col=1)
fig.add_trace(go.Scatter(y=h_fit_iter[iiter], mode="lines", name="h_fit"), row=2, col=1)
# fig.add_trace(
#     go.Scatter(
#         y=ar_pulse(ar_coef[0], ar_coef[1], len(h))[0], mode="lines", name="h_gt_ar"
#     )
# )
fig.add_trace(
    go.Scatter(y=exp_pulse(6, 1, len(h_iter[iiter]))[0], mode="lines", name="h_gt_exp"),
    row=2,
    col=1,
)
fig.write_html("./dbg-iter.html")

# %% test l0 penal
import plotly.graph_objects as go
import scipy.sparse as sps
from plotly.subplots import make_subplots
from scipy.linalg import convolution_matrix

from minian_bin.simulation import ar_pulse, exp_pulse, tau2AR
from minian_bin.update_AR import solve_fit_h
from minian_bin.update_bin import max_thres, scal_lstsq, solve_deconv, solve_deconv_l0

uid = 0
iiter = 0
sig = iter_df.set_index(["cell", "iter"]).loc[uid, iiter]["scale"]
sig_gt = sig_lev.sel(unit_id=subset[uid]).item()
y = Y_solve.isel(unit_id=uid)
s = updt_ds["S_iter"].isel(iter=iiter, unit_id=uid)
c = updt_ds["C_iter"].isel(iter=iiter, unit_id=uid) * sig
c_gt = sim_ds["C"].sel(unit_id=subset[uid]) * sig_gt
s_gt = sim_ds["S"].sel(unit_id=subset[uid])
kn = exp_pulse(6, 1, 60, p_d=1, p_r=-1)[0]
K = sps.csc_matrix(convolution_matrix(kn, len(y))[: len(y), :])
nthres = 1000
c_val, s_val, _ = solve_deconv(
    np.array(y), kn=kn, ar_mode=False, scale=sig_gt, amp_constraint=True
)
c_val, s_val = c_val.squeeze(), s_val.squeeze()
l0_penal = 1
max_iters = 10
res = []
svals = []
metric_df = None
i = 0
while i < max_iters:
    c_penal, s_penal, _, met_df = solve_deconv_l0(
        np.array(y),
        kn=kn,
        ar_mode=False,
        scale=sig_gt,
        amp_constraint=True,
        l0_penal=l0_penal,
        verbose=False,
    )
    c_penal, s_penal = c_penal.squeeze(), s_penal.squeeze()
    th_svals, thres = max_thres(
        np.abs(s_penal), nthres, th_min=1e-4, th_max=1 - 1e-4, return_thres=True
    )
    th_cvals = [K @ ss for ss in th_svals]
    th_scals = [scal_lstsq(cc, y) for cc in th_cvals]
    th_objs = [
        np.linalg.norm(y - scl * np.array(cc).squeeze())
        for scl, cc in zip(th_scals, th_cvals)
    ]
    opt_idx = np.argmin(th_objs)
    opt_s = th_svals[opt_idx].astype(float).squeeze()
    opt_c = th_cvals[opt_idx].astype(float).squeeze()
    opt_obj = th_objs[opt_idx]
    opt_scal = th_scals[opt_idx]
    nnz_l0 = (s_penal > 0).sum()
    nnz_opt = (opt_s > 0).sum()
    print(
        "l0 penal: {:.4f}, nnz_l0: {}, scal: {:.3f}, nnz_opt: {}".format(
            l0_penal, nnz_l0, opt_scal, nnz_opt
        )
    )
    svals.append(opt_s)
    res.append(
        pd.DataFrame(
            {
                "l0_penal": l0_penal,
                "thres": thres,
                "scal": th_scals,
                "err": th_objs,
                "iter": i,
            }
        )
    )
    metric_df = pd.concat(
        [
            metric_df,
            pd.DataFrame(
                [
                    {
                        "l0_penal": l0_penal,
                        "scale": opt_scal,
                        "obj": opt_obj,
                        "iter": i,
                        "same_nnz": nnz_l0 == nnz_opt,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    l0_ub = metric_df[metric_df["same_nnz"]]["l0_penal"].min()
    l0_lb = metric_df[~metric_df["same_nnz"]]["l0_penal"].max()
    if np.isnan(l0_ub):
        l0_penal = l0_lb * 2
    elif np.isnan(l0_lb):
        l0_penal = l0_ub / 2
    else:
        l0_penal = (l0_ub + l0_lb) / 2
    i += 1
res = pd.concat(res, ignore_index=True)
lams_l0, ps_l0, h_l0, h_fit_l0, _, _ = solve_fit_h(
    np.array(y) / sig_gt, opt_s, N=2, s_len=60, ar_mode=False
)
lams, ps, h, h_fit, _, _ = solve_fit_h(
    np.array(y) / sig_gt, s, N=2, s_len=60, ar_mode=False
)
# res_pvt = (
#     res[["csize", "scal", "err"]]
#     .drop_duplicates()
#     .pivot(index="csize", columns="scal", values="err")
# )
# fig = px.imshow(res_pvt, x=res_pvt.columns, y=res_pvt.index)
err_gt = np.linalg.norm(y - c_gt)
fig = px.line(res, x="scal", y="err", facet_row="l0_penal", facet_row_spacing=1e-2)
fig.add_hline(y=err_gt, line_color="grey", line_dash="dash", line_width=2)
fig.update_layout(height=res["l0_penal"].nunique() * 150)
fig.write_html("dbg-obj.html")
fig = make_subplots(2, 1, shared_xaxes=False, shared_yaxes=False)
fig.add_trace(go.Scatter(y=y, mode="lines", name="y"), row=1, col=1)
fig.add_trace(go.Scatter(y=s, mode="lines", name="s"), row=1, col=1)
fig.add_trace(go.Scatter(y=c, mode="lines", name="c"), row=1, col=1)
fig.add_trace(go.Scatter(y=s_val, mode="lines", name="s_val"), row=1, col=1)
fig.add_trace(go.Scatter(y=s_penal, mode="lines", name="s_penal"), row=1, col=1)
fig.add_trace(go.Scatter(y=opt_s, mode="lines", name="opt_s"), row=1, col=1)
fig.add_trace(go.Scatter(y=opt_c, mode="lines", name="opt_c"), row=1, col=1)
# fig.add_trace(
#     go.Scatter(y=s_mixin.squeeze(), mode="lines", name="s_mixin"), row=1, col=1
# )
# fig.add_trace(
#     go.Scatter(y=c_mixin.squeeze() * sig_gt, mode="lines", name="c_mixin"), row=1, col=1
# )
fig.add_trace(go.Scatter(y=s_gt, mode="lines", name="s_gt"), row=1, col=1)
fig.add_trace(go.Scatter(y=c_gt, mode="lines", name="c_gt"), row=1, col=1)
fig.add_trace(go.Scatter(y=h, mode="lines", name="h_free"), row=2, col=1)
fig.add_trace(go.Scatter(y=h_fit, mode="lines", name="h_fit"), row=2, col=1)
fig.add_trace(go.Scatter(y=h_l0, mode="lines", name="h_free_l0"), row=2, col=1)
fig.add_trace(go.Scatter(y=h_fit_l0, mode="lines", name="h_fit_l0"), row=2, col=1)
# fig.add_trace(
#     go.Scatter(
#         y=ar_pulse(ar_coef[0], ar_coef[1], len(h))[0], mode="lines", name="h_gt_ar"
#     )
# )
fig.add_trace(
    go.Scatter(y=exp_pulse(6, 1, len(h_iter[iiter]))[0], mode="lines", name="h_gt_exp"),
    row=2,
    col=1,
)
fig.write_html("./dbg-iter.html")


# %% 3.3 Estimate spiking from kernel and upsampled C
# Function to solve for spike estimates given calcium trace and kernel
def solve_s(y, h, norm="l1", sparsity_penalty=0):
    y, h = y.squeeze(), h.squeeze()

    num_samples = len(y)

    # Baseline fluorescence for each cell
    b = cp.Variable()

    # Spike train for each cell and time point
    s = cp.Variable(num_samples)

    # Convolution term: applying cp.conv() for each cell separately

    conv_term = cp.conv(h, s)[
        :num_samples
    ]  # Convolve spike train with kernel for each cell

    # Norm choice (l1 or l2)
    norm_ord = {"l1": 1, "l2": 2}[norm]

    # Objective function: minimize reconstruction error + sparsity penalty
    obj = cp.Minimize(
        cp.norm(y - conv_term - b, norm_ord)  # Properly broadcast baseline
        #  + sparsity_penalty * cp.norm(s, 1)  # L1 sparsity penalty
    )

    # Constraints: non-negative spikes and baseline
    cons = [s >= 0, b >= 0]

    # Define and solve the problem
    prob = cp.Problem(obj, cons)
    prob.solve()  # Using SCS solver to handle large-scale problems
    return s.value

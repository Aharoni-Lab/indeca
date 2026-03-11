"""Binary pursuit deconvolution pipeline.

Supports parametric (bi-exponential) and free-form kernel estimation,
Dask parallelism, chunked cell processing, and real-time dashboard.

Usage::

    from pipeline import run_pipeline, run_pipeline_chunked, PipelineConfig

    # Parametric (default)
    result = run_pipeline(Y, PipelineConfig(tau_init=(8.0, 0.5)))

    # Free-form kernel estimation (mirrors Rust kernel_est.rs + biexp_fit.rs)
    result = run_pipeline(Y, PipelineConfig(ar_free_kernel=True, ar_smooth_penalty=0.001))

    # Dask parallel
    from dask.distributed import Client, LocalCluster
    result = run_pipeline(Y, config, da_client=Client(LocalCluster(n_workers=8)))

    # Chunked
    result = run_pipeline_chunked(Y, config, chunk_size=5)
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import medfilt, find_peaks

from indeca.core.simulation import tau2AR, AR2tau, solve_p, exp_pulse, ar_pulse
from indeca.core.AR_kernel import AR_upsamp_real, estimate_coefs, updateAR
from indeca.core.deconv import DeconvBin, construct_R
from indeca.core.deconv.deconv import InputParams


# ═══════════════════════════════════════════════════════════════════════
# Configuration & Result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Top-level configuration for the binary pursuit pipeline."""

    # Core
    up_factor: int = 1
    p: int = 2

    # Preprocessing
    med_wnd: Optional[Union[int, Literal["auto"]]] = None
    dff: bool = True

    # Initialization
    tau_init: Optional[Tuple[float, float]] = None
    est_noise_freq: Optional[float] = None
    est_use_smooth: bool = False
    est_add_lag: int = 20
    est_nevt: Optional[int] = 10

    # Deconvolution
    nthres: int = 1000
    norm: Literal["l1", "l2", "huber"] = "l2"
    penal: Optional[Literal["l0", "l1"]] = None
    backend: Literal["osqp", "cvxpy", "cuosqp"] = "osqp"
    use_base: bool = True
    reset_scale: bool = True
    err_weighting: Optional[Literal["fft", "corr", "adaptive"]] = None
    masking_radius: Optional[int] = None
    pks_polish: bool = True
    ncons_thres: Optional[Union[int, Literal["auto"]]] = None
    min_rel_scl: Optional[Union[float, Literal["auto"]]] = None
    atol: float = 1e-3

    # AR Update
    ar_use_all: bool = True
    ar_kn_len: int = 100
    ar_norm: Literal["l1", "l2"] = "l2"
    ar_prop_best: Optional[float] = None
    ar_free_kernel: bool = False
    ar_smooth_penalty: float = 0.0

    # Convergence
    max_iters: int = 50
    err_atol: float = 1e-4
    err_rtol: float = 5e-2
    use_rel_err: bool = True
    n_best: Optional[int] = 3

    # Callbacks
    on_iteration: Optional[Callable] = None


@dataclass
class PipelineResult:
    """Result container."""
    opt_C: np.ndarray
    opt_S: np.ndarray
    metric_df: pd.DataFrame
    C_ls: List[np.ndarray] = field(default_factory=list)
    S_ls: List[np.ndarray] = field(default_factory=list)
    h_ls: List[np.ndarray] = field(default_factory=list)
    h_fit_ls: List[np.ndarray] = field(default_factory=list)
    scal_ls: List[np.ndarray] = field(default_factory=list)
    converged: bool = False
    convergence_reason: str = ""
    n_iters: int = 0


# ═══════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════

def _compute_dff(s, window_size=100, q=0.2):
    ser = pd.Series(s).astype(float)
    f0 = ser.rolling(window=window_size, min_periods=1).quantile(q)
    return (ser - f0).to_numpy()


def preprocess_traces(Y, med_wnd, dff, ar_kn_len):
    if med_wnd is not None:
        actual_wnd = ar_kn_len if med_wnd == "auto" else int(med_wnd)
        for iy in range(Y.shape[0]):
            Y[iy, :] = Y[iy, :] - medfilt(Y[iy, :], actual_wnd * 2 + 1)
    if dff:
        for iy in range(Y.shape[0]):
            Y[iy, :] = _compute_dff(Y[iy, :], window_size=ar_kn_len * 5, q=0.2)
    return Y


# ═══════════════════════════════════════════════════════════════════════
# AR Initialization
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ARParams:
    theta: np.ndarray
    tau: np.ndarray
    ps: np.ndarray


def initialize_ar_params(Y, tau_init, p, up_factor, ar_kn_len,
                         est_noise_freq, est_use_smooth, est_add_lag):
    ncell = Y.shape[0]
    if tau_init is not None:
        theta_single = tau2AR(tau_init[0], tau_init[1])
        _, _, pp = AR2tau(theta_single[0], theta_single[1], solve_amp=True)
        ps_vec = np.array([pp, -pp])
        theta = np.tile(theta_single, (ncell, 1))
        tau = np.tile(tau_init, (ncell, 1))
        ps = np.tile(ps_vec, (ncell, 1))
    else:
        theta = np.empty((ncell, p))
        tau = np.empty((ncell, p))
        ps = np.empty((ncell, p))
        for icell, y in enumerate(Y):
            cur_theta, _ = estimate_coefs(
                y, p=p, noise_freq=est_noise_freq,
                use_smooth=est_use_smooth, add_lag=est_add_lag,
            )
            cur_theta, cur_tau, cur_p = AR_upsamp_real(
                cur_theta, upsamp=up_factor, fit_nsamp=ar_kn_len
            )
            tau[icell, :] = cur_tau
            theta[icell, :] = cur_theta
            ps[icell, :] = cur_p
    return ARParams(theta=theta, tau=tau, ps=ps)


# ═══════════════════════════════════════════════════════════════════════
# Deconvolver Initialization (local path only)
# ═══════════════════════════════════════════════════════════════════════

def initialize_deconvolvers(Y, ar_params, cfg, dashboard=None):
    """Create DeconvBin instances. Dashboard attached AFTER construction."""
    theta, tau, ps = ar_params.theta, ar_params.tau, ar_params.ps
    dcv = []
    for i, y in enumerate(Y):
        d = DeconvBin(InputParams(
            y=y, theta=theta[i], tau=tau[i], ps=ps[i],
            coef_len=cfg.ar_kn_len, upsamp=cfg.up_factor,
            nthres=cfg.nthres, norm=cfg.norm, penal=cfg.penal,
            use_base=cfg.use_base, err_weighting=cfg.err_weighting,
            masking_radius=cfg.masking_radius, pks_polish=cfg.pks_polish,
            ncons_thres=cfg.ncons_thres, min_rel_scl=cfg.min_rel_scl,
            atol=cfg.atol, backend=cfg.backend,
            dashboard=None, dashboard_uid=i,
        ))
        if dashboard is not None:
            d.dashboard = dashboard
            d.dashboard_uid = i
        dcv.append(d)
    return dcv


# ═══════════════════════════════════════════════════════════════════════
# Dask Worker Function (standalone — no DeconvBin serialization)
# ═══════════════════════════════════════════════════════════════════════

def _solve_cell(y, theta, tau, ps, cfg_dict, i_iter, reset_scale):
    """Create a DeconvBin and solve on a Dask worker."""
    d = DeconvBin(InputParams(
        y=y, theta=theta, tau=tau, ps=ps,
        coef_len=cfg_dict["ar_kn_len"], upsamp=cfg_dict["up_factor"],
        nthres=cfg_dict["nthres"], norm=cfg_dict["norm"],
        penal=cfg_dict["penal"], use_base=cfg_dict["use_base"],
        err_weighting=cfg_dict["err_weighting"],
        masking_radius=cfg_dict["masking_radius"],
        pks_polish=cfg_dict["pks_polish"],
        ncons_thres=cfg_dict["ncons_thres"],
        min_rel_scl=cfg_dict["min_rel_scl"],
        atol=cfg_dict["atol"], backend=cfg_dict["backend"],
        dashboard=None, dashboard_uid=0,
    ))
    return d.solve_scale(reset_scale=i_iter <= 1 or reset_scale)


def _config_to_dict(cfg):
    """Extract fields Dask workers need as a plain picklable dict."""
    return {
        "ar_kn_len": cfg.ar_kn_len, "up_factor": cfg.up_factor,
        "nthres": cfg.nthres, "norm": cfg.norm, "penal": cfg.penal,
        "use_base": cfg.use_base, "err_weighting": cfg.err_weighting,
        "masking_radius": cfg.masking_radius, "pks_polish": cfg.pks_polish,
        "ncons_thres": cfg.ncons_thres, "min_rel_scl": cfg.min_rel_scl,
        "atol": cfg.atol, "backend": cfg.backend,
    }


# ═══════════════════════════════════════════════════════════════════════
# Deconvolution Step
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DeconvStepResult:
    S: np.ndarray
    C: np.ndarray
    scale: np.ndarray
    err: np.ndarray
    err_rel: np.ndarray
    nnz: np.ndarray
    penal: np.ndarray


def run_deconv_step(Y, deconvolvers, i_iter, reset_scale,
                    da_client=None, config=None, ar_params=None):
    """Dask: _solve_cell per worker. Local: pre-built deconvolvers."""
    res = []
    if da_client is not None and config is not None and ar_params is not None:
        cfg_dict = _config_to_dict(config)
        futures = []
        for icell in range(Y.shape[0]):
            f = da_client.submit(
                _solve_cell, Y[icell], ar_params.theta[icell],
                ar_params.tau[icell], ar_params.ps[icell],
                cfg_dict, i_iter, reset_scale,
            )
            futures.append(f)
        res = da_client.gather(futures)
    else:
        for icell in range(Y.shape[0]):
            r = deconvolvers[icell].solve_scale(reset_scale=i_iter <= 1 or reset_scale)
            res.append(r)

    return DeconvStepResult(
        S=np.stack([r[0].squeeze() for r in res], axis=0, dtype=float),
        C=np.stack([r[1].squeeze() for r in res], axis=0),
        scale=np.array([r[2] for r in res]),
        err=np.array([r[3] for r in res]),
        err_rel=np.array([r[4] for r in res]),
        nnz=np.array([r[5] for r in res]),
        penal=np.array([r[6] for r in res]),
    )


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def _find_dhm_safe(tau, scale):
    from indeca.core.simulation import find_dhm
    try:
        return np.array(find_dhm(True, (tau[0], tau[1]), (scale, -scale))[0], dtype=float)
    except Exception:
        return np.array([np.nan, np.nan])


def make_cur_metric(i_iter, ncell, theta, tau, scale, deconv_result,
                    deconvolvers, use_rel_err):
    dhm = np.stack(
        [_find_dhm_safe((t0, t1), s) for t0, t1, s in zip(tau.T[0], tau.T[1], scale)],
        axis=0,
    )
    return pd.DataFrame({
        "iter": i_iter, "cell": np.arange(ncell),
        "g0": theta.T[0], "g1": theta.T[1],
        "tau_d": tau.T[0], "tau_r": tau.T[1],
        "dhm0": dhm.T[0], "dhm1": dhm.T[1],
        "err": deconv_result.err, "err_rel": deconv_result.err_rel,
        "scale": scale, "penal": deconv_result.penal,
        "nnz": deconv_result.nnz,
        "obj": deconv_result.err_rel if use_rel_err else deconv_result.err,
        "wgt_len": [d.wgt_len for d in deconvolvers],
    })


def update_dashboard_metrics(dashboard, cur_metric, i_iter, max_iters):
    if dashboard is None:
        return
    try:
        for uid in range(len(cur_metric)):
            dashboard.update(
                uid=int(cur_metric.iloc[uid]["cell"]),
                tau_d=float(cur_metric.iloc[uid]["tau_d"]),
                tau_r=float(cur_metric.iloc[uid]["tau_r"]),
                err=float(cur_metric.iloc[uid]["obj"]),
                scale=float(cur_metric.iloc[uid]["scale"]),
            )
        dashboard.set_iter(min(i_iter + 1, max_iters - 1))
    except Exception as e:
        print(f"  [dashboard] metrics update failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# AR Update — Parametric (bi-exponential via updateAR)
# ═══════════════════════════════════════════════════════════════════════

def select_best_spikes(S_ls, scal_ls, err_rel, metric_df, n_best, i_iter, tau_init):
    S = S_ls[-1]
    scale = scal_ls[-1]
    metric_df = metric_df.set_index(["iter", "cell"])
    if n_best is not None and i_iter >= n_best:
        ncell = S.shape[0]
        S_best = np.empty_like(S)
        scal_best = np.empty_like(scale)
        err_wt = np.empty_like(err_rel)
        metric_best = metric_df if tau_init is not None else metric_df.loc[1:, :]
        for icell, cell_met in metric_best.groupby("cell", sort=True):
            cell_met = cell_met.reset_index().sort_values("obj", ascending=True)
            cur_idx = np.array(cell_met["iter"][:n_best])
            metric_df.loc[(i_iter, icell), "best_idx"] = ",".join(cur_idx.astype(str))
            S_best[icell, :] = np.sum(
                np.stack([S_ls[i][icell, :] for i in cur_idx], axis=0), axis=0
            ) > (n_best / 2)
            scal_best[icell] = np.mean([scal_ls[i][icell] for i in cur_idx])
            err_wt[icell] = -np.mean([metric_df.loc[(i, icell), "err_rel"] for i in cur_idx])
    else:
        S_best, scal_best, err_wt = S, scale, -err_rel
    metric_df = metric_df.reset_index()
    return S_best, scal_best, err_wt, metric_df


def make_S_ar(S_best, est_nevt, T, up_factor, ar_kn_len):
    if est_nevt is not None:
        S_ar, R = [], construct_R(T, up_factor)
        for s in S_best:
            Rs = R @ s
            s_pks, pk_prop = find_peaks(Rs, height=1, distance=ar_kn_len * up_factor)
            pk_ht = pk_prop["peak_heights"]
            top_idx = s_pks[np.argsort(pk_ht)[-est_nevt:]]
            mask = np.zeros_like(Rs, dtype=bool)
            mask[top_idx] = True
            s_ma = np.zeros_like(s)
            s_ma[::up_factor] = Rs * mask
            S_ar.append(s_ma)
        S_ar = np.stack(S_ar, axis=0)
    else:
        S_ar = S_best
    return S_ar


def update_ar_parameters(Y, S_ar, scal_best, err_wt, ar_use_all,
                         ar_kn_len, ar_norm, ar_prop_best, up_factor, p, ncell,
                         dashboard=None):
    """Parametric AR update (bi-exponential). Falls back on overflow."""
    if ar_use_all:
        if ar_prop_best is not None:
            ar_nbest = max(int(np.round(ar_prop_best * ncell)), 1)
            ar_best_idx = np.argsort(err_wt)[-ar_nbest:]
        else:
            ar_best_idx = slice(None)

        Y_sub, S_sub, scal_sub = Y[ar_best_idx], S_ar[ar_best_idx], scal_best[ar_best_idx]
        max_h_len = min(ar_kn_len * up_factor, Y_sub.shape[1] - 1)

        try:
            cur_tau, ps, ar_scal, h, h_fit = updateAR(
                Y_sub, S_sub, scal_sub,
                N=p, h_len=max_h_len, norm=ar_norm, up_factor=up_factor,
            )
        except (ValueError, MemoryError) as e:
            warnings.warn(f"Multi-unit AR update failed ({e}), falling back to best cell")
            best_cell = np.argmax(
                err_wt[ar_best_idx] if isinstance(ar_best_idx, np.ndarray) else err_wt
            )
            cur_tau, ps, ar_scal, h, h_fit = updateAR(
                Y[best_cell:best_cell + 1], S_ar[best_cell:best_cell + 1],
                scal_best[best_cell:best_cell + 1],
                N=p, h_len=max_h_len, norm=ar_norm, up_factor=up_factor,
            )

        if dashboard is not None:
            try:
                dashboard.update(h=np.array(h[:ar_kn_len * up_factor], dtype=float),
                                 h_fit=np.array(h_fit[:ar_kn_len * up_factor], dtype=float))
            except Exception as e:
                print(f"  [dashboard] kernel update failed: {e}")
        tau = np.tile(cur_tau, (ncell, 1))
    else:
        tau = np.empty((ncell, p))
        ps = h = h_fit = None
        for icell, (y, s) in enumerate(zip(Y, S_ar)):
            max_h_len = min(ar_kn_len, len(y) - 1)
            try:
                cur_tau, cur_ps, ar_scal, cur_h, cur_h_fit = updateAR(
                    y, s, scal_best[icell],
                    N=p, h_len=max_h_len, norm=ar_norm, up_factor=up_factor,
                )
            except (ValueError, MemoryError) as e:
                warnings.warn(f"AR update failed for cell {icell}: {e}")
                continue
            if dashboard is not None:
                try:
                    dashboard.update(uid=icell, h=cur_h, h_fit=cur_h_fit)
                except Exception:
                    pass
            tau[icell, :] = cur_tau
            ps, h, h_fit = cur_ps, cur_h, cur_h_fit
    return tau, ps, h, h_fit


# ═══════════════════════════════════════════════════════════════════════
# AR Update — Free-form kernel (mirrors Rust kernel_est.rs + biexp_fit.rs)
# ═══════════════════════════════════════════════════════════════════════

def update_ar_parameters_free_kernel(
    Y, S_ar, scal_best, err_wt, ar_use_all,
    ar_kn_len, ar_norm, ar_prop_best, up_factor, p, ncell,
    dashboard=None, smooth_penalty=0.0,
):
    """Free-form kernel estimation, then bi-exponential fit to extract tau.

    Step 1: solve_h() estimates kernel shape directly from data via CVXPY
            (Python equivalent of Rust kernel_est.rs FISTA estimator).
    Step 2: fit_sumexp_gd() fits h(t) = beta*(exp(-t/tau_d) - exp(-t/tau_r))
            to the free-form estimate (equivalent of Rust biexp_fit.rs).

    Falls back to parametric updateAR on failure.
    """
    from indeca.core.AR_kernel import solve_h, fit_sumexp_gd

    def _estimate_one(y, s, scal, h_len):
        """Free-form estimate + biexp fit for one cell or group."""
        h_free = solve_h(
            y, s, scal, h_len=h_len, norm=ar_norm,
            up_factor=up_factor, smth_penalty=smooth_penalty,
        )

        # Find positive portion for fitting
        pos_idx = np.where(h_free > 0)[0]
        start = max(pos_idx[0], 1) if len(pos_idx) > 0 else 1
        h_to_fit = h_free[start - 1:]

        try:
            lams, ps_fit, scal_fit, h_fit_portion = fit_sumexp_gd(
                h_to_fit, fit_amp="scale"
            )
        except RuntimeError:
            lams, ps_fit, scal_fit, h_fit_portion = fit_sumexp_gd(
                h_to_fit, fit_amp=False
            )

        cur_tau = -1 / lams * up_factor
        h_fit = np.zeros_like(h_free)
        h_fit[:len(h_fit_portion)] = h_fit_portion

        return cur_tau, ps_fit, h_free, h_fit

    if ar_use_all:
        if ar_prop_best is not None:
            ar_nbest = max(int(np.round(ar_prop_best * ncell)), 1)
            ar_best_idx = np.argsort(err_wt)[-ar_nbest:]
        else:
            ar_best_idx = slice(None)

        Y_sub, S_sub, scal_sub = Y[ar_best_idx], S_ar[ar_best_idx], scal_best[ar_best_idx]
        max_h_len = min(ar_kn_len * up_factor, Y_sub.shape[1] - 1)

        try:
            cur_tau, ps, h, h_fit = _estimate_one(Y_sub, S_sub, scal_sub, max_h_len)
        except (ValueError, MemoryError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Free-kernel AR update failed ({e}), falling back to parametric")
            return update_ar_parameters(
                Y, S_ar, scal_best, err_wt, ar_use_all,
                ar_kn_len, ar_norm, ar_prop_best, up_factor, p, ncell,
                dashboard=dashboard,
            )

        if dashboard is not None:
            try:
                dashboard.update(
                    h=np.array(h[:ar_kn_len * up_factor], dtype=float),
                    h_fit=np.array(h_fit[:ar_kn_len * up_factor], dtype=float),
                )
            except Exception as e:
                print(f"  [dashboard] free-kernel update failed: {e}")

        tau = np.tile(cur_tau, (ncell, 1))

    else:
        tau = np.empty((ncell, p))
        ps = h = h_fit = None

        for icell, (y, s) in enumerate(zip(Y, S_ar)):
            max_h_len = min(ar_kn_len, len(y) - 1)
            try:
                cur_tau, cur_ps, cur_h, cur_h_fit = _estimate_one(
                    y, s, scal_best[icell], max_h_len,
                )
            except (ValueError, MemoryError, np.linalg.LinAlgError) as e:
                warnings.warn(f"Free-kernel failed for cell {icell}: {e}")
                continue

            if dashboard is not None:
                try:
                    dashboard.update(uid=icell, h=cur_h, h_fit=cur_h_fit)
                except Exception:
                    pass

            tau[icell, :] = cur_tau
            ps, h, h_fit = cur_ps, cur_h, cur_h_fit

    return tau, ps, h, h_fit


# ═══════════════════════════════════════════════════════════════════════
# Propagate AR Update
# ═══════════════════════════════════════════════════════════════════════

def propagate_ar_update(deconvolvers, tau, scal_best, ar_use_all, ar_params=None):
    """Update local deconvolvers AND ar_params in-place."""
    if ar_params is not None:
        ar_params.tau = tau
        ar_params.theta = np.array([tau2AR(t[0], t[1]) for t in tau])
        ncell = len(scal_best)
        for i in range(ncell):
            t = tau[0] if ar_use_all else tau[i]
            p = solve_p(t[0], t[1])
            ar_params.ps[i] = np.array([p, -p])

    if ar_use_all:
        cur_tau = tau[0]
        for idx, d in enumerate(deconvolvers):
            d.update(tau=cur_tau, scale=scal_best[idx])
    else:
        for idx, d in enumerate(deconvolvers):
            d.update(tau=tau[idx], scale=scal_best[idx])


# ═══════════════════════════════════════════════════════════════════════
# Convergence
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ConvergenceResult:
    converged: bool
    reason: str


def check_convergence(metric_df, cur_metric, S, S_ls, i_iter, err_atol, err_rtol):
    metric_prev = metric_df[metric_df["iter"] < i_iter].dropna(subset=["obj", "scale"])
    metric_last = metric_df[metric_df["iter"] == i_iter - 1].dropna(subset=["obj", "scale"])
    if len(metric_prev) == 0:
        return ConvergenceResult(False, "")
    err_cur = cur_metric.set_index("cell")["obj"]
    err_last = metric_last.set_index("cell")["obj"]
    err_best = metric_prev.groupby("cell")["obj"].min()
    ncell = S.shape[0]
    if (np.abs(err_cur - err_last) < err_atol).all():
        return ConvergenceResult(True, "Converged: absolute error tolerance")
    if (np.abs(err_cur - err_last) < err_rtol * err_best).all():
        return ConvergenceResult(True, "Converged: relative error tolerance")
    T_up = S.shape[1]
    S_best = np.empty((ncell, T_up))
    for uid, udf in metric_prev.groupby("cell"):
        best_iter = udf.set_index("iter")["obj"].idxmin()
        S_best[uid, :] = S_ls[best_iter][uid, :]
    if np.abs(S - S_best).sum() < 1:
        return ConvergenceResult(True, "Converged: spike pattern stabilized")
    err_all = metric_prev.pivot(columns="iter", index="cell", values="obj")
    diff_all = np.abs(err_cur.values.reshape((-1, 1)) - err_all.values)
    if (diff_all.min(axis=1) < err_atol).all():
        return ConvergenceResult(True, "Trapped: local optimal err")
    if len(S_ls) > 1:
        diff_all = np.array([np.abs(S - prev_s).sum() for prev_s in S_ls[:-1]])
        if (diff_all < 1).sum() > 1:
            return ConvergenceResult(True, "Trapped: local optimal s")
    return ConvergenceResult(False, "")


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(
    Y: np.ndarray,
    config: PipelineConfig,
    dashboard: Any = None,
    da_client: Any = None,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full binary pursuit deconvolution pipeline."""
    Y = Y.copy().astype(float)
    ncell, T = Y.shape
    up_factor = config.up_factor
    p = config.p

    if verbose:
        mode_str = "free-kernel" if config.ar_free_kernel else "parametric"
        print(f"Pipeline: {ncell} cells, {T} timepoints, up_factor={up_factor}, "
              f"ar_mode={mode_str}{', dask=True' if da_client else ''}")

    # 1. Preprocess
    Y = preprocess_traces(Y, config.med_wnd, config.dff, config.ar_kn_len)

    # 2. Initialize AR
    ar_params = initialize_ar_params(
        Y, config.tau_init, p, up_factor, config.ar_kn_len,
        config.est_noise_freq, config.est_use_smooth, config.est_add_lag,
    )
    theta, tau = ar_params.theta, ar_params.tau

    # 3. Initialize local deconvolvers
    dcv = initialize_deconvolvers(Y, ar_params, config, dashboard=dashboard)

    # 4. State
    C_ls, S_ls, scal_ls, h_ls, h_fit_ls = [], [], [], [], []
    metric_df = pd.DataFrame(columns=[
        "iter", "cell", "g0", "g1", "tau_d", "tau_r",
        "err", "err_rel", "nnz", "scale", "best_idx", "obj", "wgt_len",
    ])
    h = h_fit = None
    convergence_reason = ""
    converged = False
    final_iter = 0

    # 5. Main loop
    for i_iter in range(config.max_iters):
        final_iter = i_iter
        if verbose:
            print(f"  Iter {i_iter}/{config.max_iters}", end="", flush=True)

        # 5.1 Deconvolution
        deconv_result = run_deconv_step(
            Y, dcv, i_iter, config.reset_scale,
            da_client=da_client, config=config, ar_params=ar_params,
        )
        scale = deconv_result.scale
        if verbose:
            print(f"  err={deconv_result.err.mean():.4f}  scale={scale.mean():.4f}",
                  flush=True)

        # 5.2 Metrics
        cur_metric = make_cur_metric(
            i_iter, ncell, theta, tau, scale, deconv_result, dcv, config.use_rel_err,
        )
        metric_df = pd.concat([metric_df, cur_metric], ignore_index=True)
        update_dashboard_metrics(dashboard, cur_metric, i_iter, config.max_iters)

        # 5.3 Save
        C_ls.append(deconv_result.C)
        S_ls.append(deconv_result.S)
        scal_ls.append(scale)
        if i_iter == 0:
            h_ls.append(np.full(T * up_factor, np.nan))
            h_fit_ls.append(np.full(T * up_factor, np.nan))
        else:
            h_ls.append(h)
            h_fit_ls.append(h_fit)

        # 5.4 Best spikes
        S_best, scal_best, err_wt, metric_df = select_best_spikes(
            S_ls, scal_ls, deconv_result.err_rel, metric_df,
            config.n_best, i_iter, config.tau_init,
        )

        # 5.5 AR spike train
        S_ar = make_S_ar(S_best, config.est_nevt, T, up_factor, config.ar_kn_len)

        # 5.6 AR update — free-form or parametric
        if config.ar_free_kernel:
            tau, ps, h, h_fit = update_ar_parameters_free_kernel(
                Y, S_ar, scal_best, err_wt,
                config.ar_use_all, config.ar_kn_len, config.ar_norm, config.ar_prop_best,
                up_factor, p, ncell, dashboard=dashboard,
                smooth_penalty=config.ar_smooth_penalty,
            )
        else:
            tau, ps, h, h_fit = update_ar_parameters(
                Y, S_ar, scal_best, err_wt,
                config.ar_use_all, config.ar_kn_len, config.ar_norm, config.ar_prop_best,
                up_factor, p, ncell, dashboard=dashboard,
            )
        theta = np.array([tau2AR(t[0], t[1]) for t in tau])

        # 5.7 Propagate
        propagate_ar_update(dcv, tau, scal_best, config.ar_use_all, ar_params=ar_params)

        # 5.8 Convergence
        conv_result = check_convergence(
            metric_df, cur_metric, deconv_result.S, S_ls,
            i_iter, config.err_atol, config.err_rtol,
        )

        if config.on_iteration is not None:
            config.on_iteration(i_iter, cur_metric, {
                "S": deconv_result.S, "C": deconv_result.C,
                "scale": scale, "tau": tau, "h": h,
            })

        if conv_result.converged:
            convergence_reason = conv_result.reason
            converged = True
            if verbose:
                print(f"  → {conv_result.reason}")
            break
    else:
        convergence_reason = "Max iterations reached"
        if verbose:
            print(f"  → {convergence_reason}")

    # 6. Finalize
    opt_C = np.empty((ncell, T * up_factor))
    opt_S = np.empty((ncell, T * up_factor))
    mobj = metric_df.groupby("iter")["obj"].median()
    opt_idx_all = mobj.idxmin()
    for icell in range(ncell):
        if config.ar_use_all:
            opt_idx = opt_idx_all
        else:
            cell_df = metric_df[metric_df["cell"] == icell]
            opt_idx = metric_df.loc[cell_df["obj"].idxmin(), "iter"]
        opt_C[icell, :] = C_ls[opt_idx][icell, :]
        opt_S[icell, :] = S_ls[opt_idx][icell, :]

    if verbose:
        print(f"Pipeline complete: {opt_S.sum():.0f} total spikes detected")

    return PipelineResult(
        opt_C=opt_C, opt_S=opt_S, metric_df=metric_df,
        C_ls=C_ls, S_ls=S_ls, h_ls=h_ls, h_fit_ls=h_fit_ls, scal_ls=scal_ls,
        converged=converged, convergence_reason=convergence_reason,
        n_iters=final_iter + 1,
    )


# ═══════════════════════════════════════════════════════════════════════
# Chunked Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline_chunked(
    Y: np.ndarray, config: PipelineConfig, chunk_size: int = 5,
    dashboard: Any = None, da_client: Any = None, verbose: bool = True,
) -> PipelineResult:
    """Run pipeline in chunks of cells, then concatenate."""
    ncell, T = Y.shape
    chunks = [(start, min(start + chunk_size, ncell)) for start in range(0, ncell, chunk_size)]

    if verbose:
        print(f"Chunked pipeline: {ncell} cells in {len(chunks)} chunks of up to {chunk_size}")

    all_C, all_S, all_metrics = [], [], []
    total_iters = 0

    for i_chunk, (c_start, c_end) in enumerate(chunks):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Chunk {i_chunk+1}/{len(chunks)}: cells {c_start}–{c_end-1}")
            print(f"{'='*50}")

        result = run_pipeline(
            Y[c_start:c_end], config,
            dashboard=dashboard, da_client=da_client, verbose=verbose,
        )

        chunk_metrics = result.metric_df.copy()
        chunk_metrics["cell"] = chunk_metrics["cell"] + c_start
        chunk_metrics["chunk"] = i_chunk

        all_C.append(result.opt_C)
        all_S.append(result.opt_S)
        all_metrics.append(chunk_metrics)
        total_iters = max(total_iters, result.n_iters)

        if verbose:
            print(f"Chunk {i_chunk+1} done: {result.opt_S.sum():.0f} spikes, {result.n_iters} iters")

    opt_C = np.concatenate(all_C, axis=0)
    opt_S = np.concatenate(all_S, axis=0)
    metric_df = pd.concat(all_metrics, ignore_index=True)

    if verbose:
        print(f"\nAll chunks complete: {opt_S.sum():.0f} total spikes")

    return PipelineResult(
        opt_C=opt_C, opt_S=opt_S, metric_df=metric_df,
        converged=True, convergence_reason=f"Completed {len(chunks)} chunks",
        n_iters=total_iters,
    )

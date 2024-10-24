import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sps
import xarray as xr
from line_profiler import profile
from scipy.linalg import convolution_matrix

from .cnmf import filt_fft, get_ar_coef, noise_fft
from .simulation import tau2AR
from .utilities import scal_lstsq


def construct_G(fac: np.ndarray, T: int, fromTau=False):
    fac = np.array(fac)
    assert fac.shape == (2,)
    if fromTau:
        fac = np.array(tau2AR(*fac))
    return sps.dia_matrix(
        (
            np.tile(np.concatenate(([1], -fac)), (T, 1)).T,
            -np.arange(len(fac) + 1),
        ),
        shape=(T, T),
    ).tocsc()


def construct_R(T: int, up_factor: int):
    rs_vec = np.zeros(T * up_factor)
    rs_vec[:up_factor] = 1
    return sps.coo_matrix(
        np.stack([np.roll(rs_vec, up_factor * i) for i in range(T)], axis=0)
    )


def estimate_coefs(
    y: np.ndarray, p: int, noise_freq: tuple, use_smooth: bool, add_lag: int
):
    tn = noise_fft(y, noise_range=(noise_freq, 1))
    if use_smooth:
        y_ar = filt_fft(y.squeeze(), noise_freq, "low")
        tn_ar = noise_fft(y_ar, noise_range=(noise_freq, 1))
    else:
        y_ar, tn_ar = y, tn
    g = get_ar_coef(y_ar, np.nan_to_num(tn_ar), p=p, add_lag=add_lag)
    return g, tn


def prob_deconv(
    y_len: int,
    coef_len: int = 60,
    ar_mode: bool = True,
    use_base: bool = False,
    R: np.ndarray = None,
    norm: str = "l1",
    amp_constraint: bool = False,
    mixin: bool = False,
):
    if R is None:
        T = y_len
        R = np.eye(T)
    else:
        T = R.shape[1]
    y = cp.Parameter((y_len, 1), name="y")
    c = cp.Variable((T, 1), nonneg=True, name="c")
    s = cp.Variable((T, 1), nonneg=True, name="s", boolean=mixin)
    R = cp.Constant(R, name="R")
    scale = cp.Parameter(value=1, name="scale", nonneg=True)
    l1_penal = cp.Parameter(value=0, name="l1_penal", nonneg=True)
    w_l0 = cp.Parameter(
        shape=T, value=np.ones(T), nonneg=True, name="w_l0"
    )  # product of l0_penal * w!
    coef = cp.Parameter(shape=coef_len, name="coef")
    if use_base:
        b = cp.Variable(nonneg=True, name="b")
    else:
        b = cp.Constant(value=0, name="b")
    err_term = cp.norm(y - scale * R @ c - b, p={"l1": 1, "l2": 2}[norm])
    obj = cp.Minimize(err_term + w_l0.T @ cp.abs(s) + l1_penal * cp.norm(s, 1))
    if ar_mode:
        G = sum(
            [
                cp.diag(cp.promote(-coef[i], (T - i - 1,)), -i - 1)
                for i in range(coef_len)
            ]
        ) + np.eye(T)
        cons = [s == G @ c]
    else:
        H = sum([cp.diag(cp.promote(coef[i], (T - i,)), -i) for i in range(coef_len)])
        cons = [c == H @ s, s[-1] == 0]
    if amp_constraint:
        cons.append(s <= 1)
    prob = cp.Problem(obj, cons)
    prob.data_dict = {
        "y": y,
        "c": c,
        "s": s,
        "b": b,
        "R": R,
        "coef": coef,
        "scale": scale,
        "l1_penal": l1_penal,
        "w_l0": w_l0,
        "err_term": err_term,
    }
    return prob


@profile
def solve_deconv(
    y: np.ndarray,
    prob: cp.Problem,
    coef: np.ndarray,
    l1_penal: float = 0,
    scale: float = 1,
    return_obj: bool = False,
    solver=None,
):
    c, s, b = prob.data_dict["c"], prob.data_dict["s"], prob.data_dict["b"]
    prob.data_dict["y"].value = y.reshape((-1, 1))
    prob.data_dict["scale"].value = scale
    prob.data_dict["l1_penal"].value = l1_penal
    prob.data_dict["coef"].value = coef
    prob.solve(solver=solver)
    if return_obj:
        return c.value, s.value, b.value, prob.data_dict["err_term"].value
    else:
        return c.value, s.value, b.value


@profile
def solve_deconv_l0(
    y: np.ndarray,
    prob: cp.Problem,
    coef: np.ndarray,
    l0_penal: float = 0,
    scale: float = 1,
    return_obj: bool = False,
    max_iters=50,
    delta=1e-6,
    rtol=1e-4,
    verbose=False,
):
    c, s, b = prob.data_dict["c"], prob.data_dict["s"], prob.data_dict["b"]
    T = c.shape[0]
    prob.data_dict["y"].value = y.reshape((-1, 1))
    prob.data_dict["w_l0"].value = l0_penal * np.ones(T)
    prob.data_dict["scale"].value = scale
    prob.data_dict["coef"].value = coef
    i = 0
    metric_df = None
    s_last = None
    while i < max_iters:
        try:
            obj_best = metric_df["obj"][1:].min()
        except TypeError:
            obj_best = np.inf
        try:
            prob.solve(warm_start=bool(i))
        except cp.SolverError:
            prob.solve(
                solver=cp.OSQP,
                max_iter=int(1e6),
                eps_abs=1e-4,
                eps_rel=1e-4,
                verbose=False,
                warm_start=bool(i),
            )
        s_new = np.where(s.value > delta, s.value, 0)
        if verbose:
            print(
                "l0_penal: {:.3f}, iter: {}, nnz: {}".format(
                    l0_penal, i, (s_new > 0).sum()
                )
            )
        obj_gap = prob.value - obj_best
        metric_df = pd.concat(
            [
                metric_df,
                pd.DataFrame(
                    [
                        {
                            "obj": prob.value,
                            "iter": i,
                            "nnz": (s_new > 0).sum(),
                            "obj_gap": obj_gap,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        if np.abs(obj_gap) < rtol * obj_best:
            break
        elif s_last is not None and ((s_new > 0) == (s_last > 0)).all():
            break
        else:
            prob.data_dict["w_l0"].value = (
                l0_penal * np.ones(T) / (delta * np.ones(T) + s_new.squeeze())
            )
            s_last = s_new
            i += 1
    else:
        warnings.warn(
            "l0 heuristic did not converge in {} iterations".format(max_iters)
        )
    if return_obj:
        return (c.value, s_new, b.value, prob.data_dict["err_term"].value, metric_df)
    else:
        return c.value, s_new, b.value, metric_df


@profile
def solve_deconv_bin(
    y: np.ndarray,
    prob: cp.Problem,
    prob_cons: cp.Problem,
    coef: np.ndarray,
    ar_mode: bool = True,
    R: np.ndarray = None,
    nthres=1000,
    tol: float = 1e-6,
    max_iters: int = 50,
    use_l0=True,
    norm="l1",
):
    # parameters
    if ar_mode:
        G = construct_G(coef, R.shape[1])
        K = sps.linalg.inv(G)
    else:
        K = sps.csc_matrix(convolution_matrix(coef, R.shape[1])[: R.shape[1], :])
    RK = (R @ K).todense()
    _, s_init, _ = solve_deconv(y, prob, coef=coef)
    scale = np.ptp(s_init)
    if use_l0:
        l0_penal = 1
    else:
        l0_penal = np.nan
    # iterations
    metric_df = None
    niter = 0
    while niter < max_iters:
        if use_l0:
            _, s_bin, b_bin, lb, _ = solve_deconv_l0(
                y, prob_cons, coef=coef, scale=scale, return_obj=True, l0_penal=l0_penal
            )
        else:
            _, s_bin, b_bin, lb = solve_deconv(
                y, prob_cons, coef=coef, scale=scale, return_obj=True
            )
        th_svals = max_thres(np.abs(s_bin), nthres, th_min=0, th_max=1)
        th_cvals = [RK @ ss for ss in th_svals]
        th_scals = [scal_lstsq(cc, y) for cc in th_cvals]
        th_objs = [
            np.linalg.norm(
                y - scl * np.array(cc).squeeze() - b_bin, ord={"l1": 1, "l2": 2}[norm]
            )
            for scl, cc in zip(th_scals, th_cvals)
        ]
        opt_idx = np.argmin(th_objs)
        opt_s = th_svals[opt_idx]
        opt_obj = th_objs[opt_idx]
        opt_scal = th_scals[opt_idx]
        try:
            opt_obj_idx = metric_df["obj"].idxmin()
            opt_obj_last = metric_df.loc[opt_obj_idx, "obj"].item()
            opt_scal_last = metric_df.loc[opt_obj_idx, "scale"].item()
            scale_dup = metric_df["scale"].duplicated().any()
        except TypeError:
            opt_obj_last = np.inf
            opt_scal_last = opt_scal
            scale_dup = False
        if scale_dup:
            scale_new = opt_scal_last
        else:
            scale_new = (opt_scal + opt_scal_last) / 2
        nnz_bin = (s_bin > 0).sum()
        nnz_th = (opt_s > 0).sum()
        metric_df = pd.concat(
            [
                metric_df,
                pd.DataFrame(
                    [
                        {
                            "scale": scale,
                            "obj": opt_obj,
                            "lb": lb,
                            "iter": niter,
                            "nnz_bin": nnz_bin,
                            "nnz_th": nnz_th,
                            "same_nnz": nnz_bin == nnz_th,
                            "l0_penal": l0_penal,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        if np.abs(scale_new - scale) <= tol:
            metric_df["converged"] = True
            break
        elif abs(opt_obj_last - opt_obj) <= tol:
            metric_df["converged"] = True
            break
        else:
            if use_l0:
                l0_ub = metric_df[metric_df["same_nnz"]]["l0_penal"].min()
                l0_lb = metric_df[~metric_df["same_nnz"]]["l0_penal"].max()
                if np.isnan(l0_ub):
                    l0_penal = l0_lb * 2
                elif np.isnan(l0_lb):
                    l0_penal = l0_ub / 2
                else:
                    l0_penal = (l0_ub + l0_lb) / 2
            scale = scale_new
            niter += 1
    else:
        metric_df["converged"] = False
        warnings.warn("max scale iteration reached")
    return K @ opt_s, opt_s, b_bin, opt_scal, s_bin, metric_df


def max_thres(
    a: xr.DataArray,
    nthres: int,
    th_min=0.1,
    th_max=0.9,
    ds=None,
    return_thres=False,
    th_amplitude=False,
):
    amax = a.max()
    thres = np.linspace(th_min, th_max, nthres)
    if th_amplitude:
        S_ls = [np.floor_divide(a, amax * th) for th in thres]
    else:
        S_ls = [(a > amax * th) for th in thres]
    if ds is not None:
        S_ls = [sum_downsample(s, ds) for s in S_ls]
    if return_thres:
        return S_ls, thres
    else:
        return S_ls


def sum_downsample(a, factor):
    return np.convolve(a, np.ones(factor), mode="full")[factor - 1 :: factor]


def solve_deconv_mixin(
    y: np.ndarray,
    G: np.ndarray = None,
    kn: np.ndarray = None,
    ar_mode: bool = True,
    R: np.ndarray = None,
    nthres=1000,
):
    # parameters
    if ar_mode:
        assert G is not None, "deconv matrix `G` must be provided in ar mode"
        K = sps.linalg.inv(G)
    else:
        assert kn is not None, "convolution kernel `kn` must be provided in non-ar mode"
        K = sps.csc_matrix(convolution_matrix(kn, R.shape[1])[: R.shape[1], :])
    RK = (R @ K).todense()
    _, s_init, _, lb = solve_deconv(
        y, G=G, kn=kn, R=R, ar_mode=ar_mode, return_obj=True
    )
    th_svals = max_thres(np.abs(s_init), nthres, th_min=0, th_max=1)
    th_cvals = [RK @ ss for ss in th_svals]
    th_scals = [scal_lstsq(cc, y) for cc in th_cvals]
    svals = np.empty((nthres, R.shape[1]))
    cvals = np.empty((nthres, R.shape[1]))
    bvals = np.empty(nthres)
    objvals = np.empty(nthres)
    for i, scl in enumerate(th_scals):
        cur_c, cur_s, cur_b, cur_obj = solve_deconv(
            y,
            G=G,
            kn=kn,
            R=R,
            ar_mode=ar_mode,
            scale=scl,
            return_obj=True,
            mixin=True,
            solver="ECOS_BB",
        )
        cvals[i, :] = cur_c.squeeze()
        svals[i, :] = cur_s.squeeze()
        bvals[i] = cur_b
        objvals[i] = cur_obj
    opt_idx = np.argmin(objvals)
    opt_obj = objvals[opt_idx]
    opt_c = cvals[opt_idx, :]
    opt_s = svals[opt_idx, :]
    opt_scal = th_scals[opt_idx]
    opt_b = bvals[opt_idx]
    metric_df = pd.DataFrame(
        [
            {
                "scale": opt_scal,
                "obj": opt_obj,
                "lb": lb,
                "iter": 0,
                "converged": True,
            }
        ]
    )
    return opt_c, opt_s, opt_b, opt_scal, s_init, metric_df

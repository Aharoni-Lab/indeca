import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sps
import xarray as xr
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


def solve_deconv(
    y: np.ndarray,
    G: np.ndarray = None,
    kn: np.ndarray = None,
    ar_mode: bool = True,
    use_base: bool = False,
    norm: str = "l1",
    l1_penal: float = 0,
    diff_penal: float = 0,
    scale: float = 1,
    R: np.ndarray = None,
    return_obj: bool = False,
    amp_constraint=False,
    mixin=False,
    solver=None,
):
    if ar_mode:
        assert G is not None, "deconv matrix `G` must be provided in ar mode"
    else:
        assert kn is not None, "convolution kernel `kn` must be provided in non-ar mode"
    y = y.reshape((-1, 1))
    if R is None:
        T = y.shape[0]
        R = np.eye(T)
    else:
        T = R.shape[1]
    c = cp.Variable((T, 1))
    if mixin:
        s = cp.Variable((T, 1), boolean=True)
    else:
        s = cp.Variable((T, 1))
    if use_base:
        b = cp.Variable()
    else:
        b = cp.Constant(0)
    p = {"l1": 1, "l2": 2}[norm]
    cons = [c >= 0, s >= 0, b >= 0]
    obj = cp.Minimize(
        cp.norm(y - scale * R @ c - b, p=p)
        + l1_penal * cp.norm(s, 1)
        + diff_penal * cp.norm(cp.diff(s), 1)
    )
    if ar_mode:
        cons.append(s == G @ c)
    else:
        cons.extend([c[:, 0] == cp.convolve(kn, s[:, 0])[:T], s[-1] == 0])
    if amp_constraint:
        cons.append(s <= 1)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver)
    if return_obj:
        return c.value, s.value, b.value, prob.value
    else:
        return c.value, s.value, b.value


def solve_deconv_bin(
    y: np.ndarray,
    G: np.ndarray = None,
    kn: np.ndarray = None,
    ar_mode: bool = True,
    R: np.ndarray = None,
    nthres=1000,
    tol: float = 1e-6,
    max_iters: int = 50,
):
    # parameters
    if ar_mode:
        assert G is not None, "deconv matrix `G` must be provided in ar mode"
        K = sps.linalg.inv(G)
    else:
        assert kn is not None, "convolution kernel `kn` must be provided in non-ar mode"
        K = sps.csc_matrix(convolution_matrix(kn, R.shape[1])[: R.shape[1], :])
    RK = (R @ K).todense()
    _, s_init, _ = solve_deconv(y, G=G, kn=kn, R=R, ar_mode=ar_mode)
    scale = np.ptp(s_init)
    # interations
    metric_df = None
    niter = 0
    while niter < max_iters:
        _, s_bin, b_bin, lb = solve_deconv(
            y,
            G=G,
            kn=kn,
            scale=scale,
            R=R,
            return_obj=True,
            amp_constraint=True,
            ar_mode=ar_mode,
        )
        th_svals = max_thres(np.abs(s_bin), nthres, th_min=0, th_max=1)
        th_cvals = [RK @ ss for ss in th_svals]
        th_scals = [scal_lstsq(cc, y) for cc in th_cvals]
        th_objs = [
            np.linalg.norm(y - scl * np.array(cc).squeeze() - b_bin)
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
        metric_df = pd.concat(
            [
                metric_df,
                pd.DataFrame(
                    [{"scale": scale, "obj": opt_obj, "lb": lb, "iter": niter}]
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

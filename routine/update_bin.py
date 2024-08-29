import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sps
import xarray as xr

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
    G: np.ndarray,
    l1_penal: float = 0,
    scale: float = 1,
    R: np.ndarray = None,
    return_obj: bool = False,
):
    y = y.reshape((-1, 1))
    if R is None:
        T = y.shape[0]
        R = np.eye(T)
    else:
        T = R.shape[1]
    c = cp.Variable((T, 1))
    s = cp.Variable((T, 1))
    b = cp.Variable()
    obj = cp.Minimize(cp.norm(y - scale * R @ c - b) + l1_penal * cp.norm(s))
    cons = [s == G @ c, c >= 0, s >= 0, b >= 0]
    prob = cp.Problem(obj, cons)
    prob.solve()
    if return_obj:
        return c.value, s.value, b.value, prob.value
    else:
        return c.value, s.value, b.value


def solve_deconv_bin(
    y: np.ndarray,
    G: np.ndarray,
    R: np.ndarray = None,
    nthres=1000,
    tol: float = 1e-6,
    max_iters: int = 50,
):
    # parameters
    Gi = sps.linalg.inv(G)
    RGi = (R @ Gi).todense()
    _, s_init, _ = solve_deconv(y, G, R=R)
    scale = np.ptp(s_init)
    # interations
    metric_df = None
    niter = 0
    while niter < max_iters:
        _, s_bin, b_bin, lb = solve_deconv(y, G, scale=scale, R=R, return_obj=True)
        th_svals = max_thres(s_bin, nthres, rename=False)
        th_cvals = [RGi @ ss for ss in th_svals]
        th_scals = [scal_lstsq(cc, y) for cc in th_cvals]
        th_objs = [
            np.linalg.norm(y - scl * np.array(cc).squeeze() - b_bin)
            for scl, cc in zip(th_scals, th_cvals)
        ]
        opt_idx = np.argmin(th_objs)
        opt_s = th_svals[opt_idx]
        opt_obj = th_objs[opt_idx]
        try:
            opt_obj_idx = metric_df["obj"].idxmin()
            opt_obj_last = metric_df.loc[opt_obj_idx, "obj"].item()
            opt_scal_last = metric_df.loc[opt_obj_idx, "scale"].item()
            scale_dup = metric_df["scale"].duplicated().any()
        except TypeError:
            opt_obj_last = np.inf
            opt_scal_last = th_scals[opt_idx]
            scale_dup = False
        if scale_dup:
            scale_new = opt_scal_last
        else:
            scale_new = (th_scals[opt_idx] + opt_scal_last) / 2
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
    return Gi @ opt_s, opt_s, b_bin, scale, metric_df


def max_thres(a: xr.DataArray, nthres: int, rename=True):
    amax = a.max()
    if rename:
        return [
            (a > amax * th).rename(a.name + "-th_{:.1f}".format(th))
            for th in np.linspace(0.1, 0.9, nthres)
        ]
    else:
        return [(a > amax * th) for th in np.linspace(0.1, 0.9, nthres)]

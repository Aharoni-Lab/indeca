# %% import and definitions
import os
import warnings

import dask.array as darr
import numba as nb
import numpy as np
import sparse
import xarray as xr
from numpy import random
from scipy.ndimage import gaussian_filter1d
from scipy.stats import multivariate_normal

from .minian_functions import save_minian, shift_perframe, write_video


def gauss_cell(
    height: int,
    width: int,
    sz_mean: float,
    sz_sigma: float,
    sz_min: float,
    cent=None,
    norm=True,
):
    # generate centroid
    if cent is None:
        cent = np.atleast_2d([random.randint(height), random.randint(width)])
    # generate size
    sz_h = np.clip(
        random.normal(loc=sz_mean, scale=sz_sigma, size=cent.shape[0]), sz_min, None
    )
    sz_w = np.clip(
        random.normal(loc=sz_mean, scale=sz_sigma, size=cent.shape[0]), sz_min, None
    )
    # generate grid
    grid = np.moveaxis(np.mgrid[:height, :width], 0, -1)
    A = np.zeros((cent.shape[0], height, width))
    for idx, (c, hs, ws) in enumerate(zip(cent, sz_h, sz_w)):
        pdf = multivariate_normal.pdf(grid, mean=c, cov=np.array([[hs, 0], [0, ws]]))
        if norm:
            pmin, pmax = pdf.min(), pdf.max()
            pdf = (pdf - pmin) / (pmax - pmin)
        A[idx] = pdf
    return A


@nb.jit(nopython=True, nogil=True, cache=True)
def apply_arcoef(s: np.ndarray, g: np.ndarray):
    c = np.zeros_like(s)
    for idx in range(len(g), len(s)):
        c[idx] = s[idx] + c[idx - len(g) : idx] @ g
    return c


def ar_trace(frame: int, P: np.ndarray, g: np.ndarray):
    S = markov_fire(frame, P).astype(float)
    C = apply_arcoef(S, g)
    return C, S


def exp_trace(frame: int, P: np.ndarray, tau_d: float, tau_r: float, trunc_thres=1e-6):
    S = markov_fire(frame, P).astype(float)
    t = np.arange(1, frame + 1)
    v = np.exp(-t / tau_d) - np.exp(-t / tau_r)
    v = v[v > trunc_thres]
    C = np.convolve(v, S, mode="full")[:frame]
    return C, S


def markov_fire(frame: int, P: np.ndarray):
    assert P.shape == (2, 2)
    assert (P.sum(axis=1) == 1).all()
    S = np.zeros(frame, dtype=int)
    for i in range(1, len(S)):
        S[i] = np.random.choice([0, 1], p=P[S[i - 1], :])
    return S


def random_walk(
    n_stp,
    stp_var: float = 1,
    constrain_factor: float = 0,
    ndim=1,
    norm=False,
    integer=True,
    nn=False,
    smooth_var=None,
):
    if constrain_factor > 0:
        walk = np.zeros(shape=(n_stp, ndim))
        for i in range(n_stp):
            try:
                last = walk[i - 1]
            except IndexError:
                last = 0
            walk[i] = last + random.normal(
                loc=-constrain_factor * last, scale=stp_var, size=ndim
            )
        if integer:
            walk = np.around(walk).astype(int)
    else:
        stps = random.normal(loc=0, scale=stp_var, size=(n_stp, ndim))
        if integer:
            stps = np.around(stps).astype(int)
        walk = np.cumsum(stps, axis=0)
    if smooth_var is not None:
        for iw in range(ndim):
            walk[:, iw] = gaussian_filter1d(walk[:, iw], smooth_var)
    if norm:
        walk = (walk - walk.min(axis=0)) / (walk.max(axis=0) - walk.min(axis=0))
    elif nn:
        walk = np.clip(walk, 0, None)
    return walk


def simulate_data(
    ncell: int,
    dims: dict,
    sig_scale: float,
    sz_mean: float,
    sz_sigma: float,
    sz_min: float,
    tmp_P: np.ndarray,
    tmp_tau_d: float,
    tmp_tau_r: float,
    post_offset: float,
    post_gain: float,
    bg_nsrc: int,
    bg_tmp_var: float,
    bg_cons_fac: float,
    bg_smth_var: float,
    mo_stp_var: float,
    mo_cons_fac: float = 1,
    cent=None,
    zero_thres=1e-8,
    chk_size=1000,
    upsample: int = 1,
):
    ff, hh, ww = (
        dims["frame"],
        dims["height"],
        dims["width"],
    )
    shifts = xr.DataArray(
        darr.from_array(
            random_walk(ff, ndim=2, stp_var=mo_stp_var, constrain_factor=mo_cons_fac),
            chunks=(chk_size, -1),
        ),
        dims=["frame", "shift_dim"],
        coords={"frame": np.arange(ff), "shift_dim": ["height", "width"]},
        name="shifts",
    )
    pad = np.absolute(shifts).max().values.item()
    if pad > 20:
        warnings.warn("maximum shift is {}, clipping".format(pad))
        shifts = shifts.clip(-20, 20)
    if cent is None:
        cent = np.stack(
            [
                np.random.randint(pad * 2, hh, size=ncell),
                np.random.randint(pad * 2, ww, size=ncell),
            ],
            axis=1,
        )
    A = gauss_cell(
        2 * pad + hh,
        2 * pad + ww,
        sz_mean=sz_mean,
        sz_sigma=sz_sigma,
        sz_min=sz_min,
        cent=cent,
    )
    A = darr.from_array(
        sparse.COO.from_numpy(np.where(A > zero_thres, A, 0)), chunks=-1
    )
    traces = [
        exp_trace(ff * upsample, tmp_P, tmp_tau_d * upsample, tmp_tau_r * upsample)
        for _ in range(len(cent))
    ]
    if upsample > 1:
        C_true = darr.from_array(
            np.stack([t[0] for t in traces]).T, chunks=(chk_size, -1)
        )
        S_true = darr.from_array(
            np.stack([t[1] for t in traces]).T, chunks=(chk_size, -1)
        )
        C = darr.from_array(
            np.stack(
                [
                    np.convolve(t[0], np.ones(upsample), "valid")[::upsample]
                    for t in traces
                ]
            ).T,
            chunks=(chk_size, -1),
        )
        S = darr.from_array(
            np.stack(
                [
                    np.convolve(t[1], np.ones(upsample), "valid")[::upsample]
                    for t in traces
                ]
            ).T,
            chunks=(chk_size, -1),
        )
    else:
        C = darr.from_array(np.stack([t[0] for t in traces]).T, chunks=(chk_size, -1))
        S = darr.from_array(np.stack([t[1] for t in traces]).T, chunks=(chk_size, -1))
    cent_bg = np.stack(
        [
            np.random.randint(pad, pad + hh, size=bg_nsrc),
            np.random.randint(pad, pad + ww, size=bg_nsrc),
        ],
        axis=1,
    )
    A_bg = gauss_cell(
        2 * pad + hh,
        2 * pad + ww,
        sz_mean=sz_mean * 60,
        sz_sigma=sz_sigma * 10,
        sz_min=sz_min,
        cent=cent_bg,
    )
    A_bg = darr.from_array(
        sparse.COO.from_numpy(np.where(A_bg > zero_thres, A_bg, 0)), chunks=-1
    )
    C_bg = darr.from_array(
        random_walk(
            ff,
            ndim=bg_nsrc,
            stp_var=bg_tmp_var,
            norm=False,
            integer=False,
            nn=True,
            constrain_factor=bg_cons_fac,
            smooth_var=bg_smth_var,
        ),
        chunks=(chk_size, -1),
    )
    Y = darr.blockwise(
        computeY,
        "fhw",
        A,
        "uhw",
        C,
        "fu",
        A_bg,
        "bhw",
        C_bg,
        "fb",
        shifts.data,
        "fs",
        dtype=np.uint8,
        sig_scale=sig_scale,
        noise_scale=0.1,
        post_offset=post_offset,
        post_gain=post_gain,
    )
    if pad > 0:
        Y = Y[:, pad:-pad, pad:-pad]
        A = A[:, pad:-pad, pad:-pad]
    uids, hs, ws = np.arange(ncell), np.arange(hh), np.arange(ww)
    if upsample > 1:
        fs_true = np.arange(ff * upsample)
        fs = fs_true[int(upsample / 2) : min(-int(upsample / 2) + 1, -1) : upsample]
    else:
        fs = np.arange(ff)
    Y = xr.DataArray(
        Y,
        dims=["frame", "height", "width"],
        coords={"frame": fs, "height": hs, "width": ws},
        name="Y",
    )
    A = xr.DataArray(
        A.compute().todense(),
        dims=["unit_id", "height", "width"],
        coords={"unit_id": uids, "height": hs, "width": ws},
        name="A",
    )
    C = xr.DataArray(
        C, dims=["frame", "unit_id"], coords={"unit_id": uids, "frame": fs}, name="C"
    )
    S = xr.DataArray(
        S, dims=["frame", "unit_id"], coords={"unit_id": uids, "frame": fs}, name="S"
    )
    if upsample > 1:
        C_true = xr.DataArray(
            C_true,
            dims=["frame", "unit_id"],
            coords={"unit_id": uids, "frame": fs_true},
            name="C_true",
        )
        S_true = xr.DataArray(
            S_true,
            dims=["frame", "unit_id"],
            coords={"unit_id": uids, "frame": fs_true},
            name="S_true",
        )
        return Y, A, C, S, C_true, S_true, shifts
    else:
        return Y, A, C, S, shifts


def generate_data(dpath, save_Y=False, use_minian=False, **kwargs):
    dat_vars = simulate_data(**kwargs)
    Y = dat_vars[0]
    if not save_Y:
        dat_vars = dat_vars[1:]
    if use_minian:
        for dat in dat_vars:
            save_minian(dat, dpath=os.path.join(dpath, "simulated"), overwrite=True)
        write_video(
            Y,
            vpath=dpath,
            vname="simulated",
            vext="avi",
            options={"r": "60", "pix_fmt": "gray", "vcodec": "ffv1"},
            chunked=True,
        )
        return tuple(dat_vars)
    else:
        ds = xr.merge(dat_vars)
        ds.to_netcdf(dpath)
        return ds


def computeY(A, C, A_bg, C_bg, shifts, sig_scale, noise_scale, post_offset, post_gain):
    A, C, A_bg, C_bg, shifts = A[0], C[0], A_bg[0], C_bg[0], shifts[0]
    Y = sparse.tensordot(C, A, axes=1)
    Y *= sig_scale
    Y_bg = sparse.tensordot(C_bg, A_bg, axes=1)
    Y += Y_bg
    del Y_bg
    for i, sh in enumerate(shifts):
        Y[i, :, :] = shift_perframe(Y[i, :, :], sh, fill=0)
    noise = np.random.normal(scale=noise_scale, size=Y.shape)
    Y += noise
    del noise
    Y += post_offset
    Y *= post_gain
    np.clip(Y, 0, 255, out=Y)
    return Y.astype(np.uint8)


def tau2AR(tau_d, tau_r):
    z1, z2 = np.exp(-1 / tau_d), np.exp(-1 / tau_r)
    return z1 + z2, -z1 * z2


def AR2tau(theta1, theta2):
    rts = np.roots([1, -theta1, -theta2])
    z1, z2 = rts[0], rts[1]
    tau_d, tau_r = -1 / np.log(z1), -1 / np.log(z2)
    return tau_d, tau_r

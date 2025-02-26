# %% import and definitions
import warnings

import dask.array as darr
import numpy as np
import pandas as pd
import sparse
import xarray as xr
from numpy import random
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import root_scalar
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm


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


# @nb.jit(nopython=True, nogil=True, cache=True)
def apply_arcoef(s: np.ndarray, g: np.ndarray, shifted: bool = False):
    c = np.zeros(len(s), dtype=float)
    for i in range(len(s)):
        if shifted:
            sidx = i - 1
        else:
            sidx = i
        if i > 1:
            c[i] = s[sidx] + g[0] * c[i - 1] + g[1] * c[i - 2]
        elif i > 0:
            c[i] = s[sidx] + g[0] * c[i - 1]
        else:
            if sidx >= 0:
                c[i] = s[sidx]
            else:
                c[i] = 0
    return c


def apply_exp(
    s: np.ndarray,
    tau_d: float,
    tau_r: float,
    p_d: float = 1,
    p_r: float = -1,
    kn_len: int = None,
    trunc_thres: float = None,
):
    if kn_len is None:
        kn_len = len(s)
    t = np.arange(kn_len).astype(float)
    if tau_d > tau_r and tau_r > 0:
        kn = np.abs(p_d * np.exp(-t / tau_d) + p_r * np.exp(-t / tau_r))
    elif tau_d > 0:
        kn = np.abs(p_d * np.exp(-t / tau_d))
        kn[0] = 0
        warnings.warn(
            "Ignoring rise time, tau_d: {:.2f}, tau_r: {:.2f}".format(tau_d, tau_r)
        )
    else:
        raise ValueError("Invalid tau_d: {:.2f}, tau_r: {:.2f}".format(tau_d, tau_r))
    if trunc_thres is not None:
        trunc_idx = np.where(kn > trunc_thres)[0].max()
        kn = kn[:trunc_idx]
    return np.convolve(kn, s, mode="full")[: len(s)]


def ar_trace(
    frame: int,
    P: np.ndarray,
    g: np.ndarray = None,
    tau_d: float = None,
    tau_r: float = None,
    shifted: bool = False,
):
    if g is None:
        g = np.array(tau2AR(tau_d, tau_r))
    S = markov_fire(frame, P).astype(float)
    C = apply_arcoef(S, g, shifted=shifted)
    return C, S


def exp_trace(frame: int, P: np.ndarray, tau_d: float, tau_r: float, trunc_thres=1e-6):
    # uses a 2 state markov model to generate more 'bursty' spike trains
    S = markov_fire(frame, P).astype(float)
    t = np.arange(0, frame)
    # Creates bi-exponential convolution kernel
    v = np.exp(-t / tau_d) - np.exp(-t / tau_r)
    # Trims the length of the kernel once it reaches a small value
    v = v[: np.where(v > trunc_thres)[0].max()]
    # Convolves spiking with kernel to generate upscaled calcium
    C = np.convolve(v, S, mode="full")[:frame]
    return C, S


def markov_fire(frame: int, P: np.ndarray):
    # makes sure markov probabilities are correct shape
    assert P.shape == (2, 2)
    # make sure probabilities sum to 1
    assert (P.sum(axis=1) == 1).all()
    while True:
        # allocate array for spiking and generate
        S = np.zeros(frame, dtype=int)
        for i in range(1, len(S)):
            S[i] = np.random.choice([0, 1], p=P[S[i - 1], :])
        # make sure at least one firing exists
        if S.sum() > 0:
            break
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


def simulate_traces(
    num_cells: int,
    length_in_sec: float,
    tmp_P: np.ndarray,
    tmp_tau_d: float,
    tmp_tau_r: float,
    approx_fps: float = 30,
    spike_sampling_rate=500,
    noise: float = 0.01,
):
    # TODO: make this compatible with exp_trace and incorporate this with rest
    # of the simulation pipeline
    upsample_factor = np.round(spike_sampling_rate / approx_fps).astype(int)
    fps = spike_sampling_rate / upsample_factor
    num_samples = np.round(length_in_sec * fps).astype(int)
    tmp_tau_d = tmp_tau_d * fps
    tmp_tau_r = tmp_tau_r * fps

    traces = []
    for i in tqdm(range(num_cells), desc="Simulating cells", unit="cell"):
        C_upsampled, S_upsampled, C, S = exp_trace(
            num_samples, tmp_P, tmp_tau_d, tmp_tau_r, upsample_factor=upsample_factor
        )
        traces.append({"C_true": C_upsampled, "S_true": S_upsampled, "C": C, "S": S})
    # Add Gaussian noise to C
    for trace in traces:
        noise_array = np.random.normal(0, noise, size=trace["C"].shape)
        trace["C_noisy"] = trace["C"] + noise_array

    # Create DataFrame with all data
    df = pd.DataFrame(traces)
    df["fps"] = fps
    df["upsample_factor"] = upsample_factor
    df["spike_sampling_rate"] = spike_sampling_rate
    return df


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
        ar_trace(
            ff * upsample,
            tmp_P,
            tau_d=tmp_tau_d * upsample,
            tau_r=tmp_tau_r * upsample,
        )
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


def generate_data(dpath, save_Y=False, **kwargs):
    dat_vars = simulate_data(**kwargs)
    if not save_Y:
        dat_vars = dat_vars[1:]
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
        Y[i, :, :] = shift_frame(Y[i, :, :], sh, fill=0)
    noise = np.random.normal(scale=noise_scale, size=Y.shape)
    Y += noise
    del noise
    Y += post_offset
    Y *= post_gain
    np.clip(Y, 0, 255, out=Y)
    return Y.astype(np.uint8)


def tau2AR(tau_d, tau_r, p=1, return_scl=False):
    z1, z2 = np.exp(-1 / tau_d), np.exp(-1 / tau_r)
    if return_scl:
        scl = p * (z1 - z2)
        return np.real(z1 + z2), np.real(-z1 * z2), scl
    else:
        return np.real(z1 + z2), np.real(-z1 * z2)


def AR2tau(theta1, theta2, solve_amp: bool = False):
    rts = np.roots([1, -theta1, -theta2])
    z1, z2 = rts
    if np.imag(z1) == 0 and np.isclose(z1, 0) and z1 < 0:
        z1 = np.abs(z1)
    if np.imag(z2) == 0 and np.isclose(z2, 0) and z2 < 0:
        z2 = np.abs(z2)
    tau_d, tau_r = np.nan_to_num([-1 / np.log(z1), -1 / np.log(z2)])
    if solve_amp:
        p = 1 / (np.exp(-1 / tau_d) - np.exp(-1 / tau_r))
        return tau_d, tau_r, p
    else:
        return tau_d, tau_r


def AR2exp(theta1, theta2):
    tau_d, tau_r = AR2tau(theta1, theta2)
    if np.imag(tau_d) == 0 and np.imag(tau_r) == 0:  # real exponentials
        L = np.array([[1, 1], [np.exp(-1 / tau_d), np.exp(-1 / tau_r)]])
        coef = np.linalg.inv(L) @ np.array([1, theta1])
        return True, np.array([tau_d, tau_r]), coef
    else:  # complex exponentials: convert to real solution (exp + trig)
        a, b = (
            -np.real(tau_d) / np.absolute(tau_d) ** 2,
            np.imag(tau_d) / np.absolute(tau_d) ** 2,
        )
        coef = np.array([1, (theta1 * np.exp(-a) - np.cos(b)) / np.sin(b)])
        return False, np.array([a, b]), coef


def generate_pulse(nsamp):
    t = np.arange(nsamp).astype(float)
    pulse = np.zeros_like(t)
    pulse[0] = 1
    return pulse, t


def ar_pulse(theta1, theta2, nsamp, shifted=False):
    pulse, t = generate_pulse(nsamp)
    ar = apply_arcoef(pulse, np.array([theta1, theta2]), shifted=shifted)
    return ar, t, pulse


def exp_pulse(
    tau_d, tau_r, nsamp, p_d=1, p_r=-1, kn_len: int = None, trunc_thres: float = None
):
    pulse, t = generate_pulse(nsamp)
    exp = apply_exp(pulse, tau_d, tau_r, p_d, p_r, kn_len, trunc_thres)
    return exp, t, pulse


def eval_exp(t, is_biexp, tconst, coefs):
    if is_biexp:
        tau_d, tau_r = tconst
        c1, c2 = coefs
        if tau_r > 0:
            return c1 * np.exp(-t / tau_d) + c2 * np.exp(-t / tau_r)
        else:
            return c1 * np.exp(-t / tau_d) + c2
    else:
        a, b = tconst
        c1, c2 = coefs
        return np.exp(a * t) * (c1 * np.cos(b * t) + c2 * np.sin(b * t))


def find_dhm(is_biexp, tconst, coefs, verbose=False):
    if is_biexp:
        tau_d, tau_r = tconst
        c1, c2 = coefs
        if tau_r == 0:
            return (0, -tau_d * np.log(0.5)), 0
        if c1 > 0 and c2 < 0:
            t_hat = (
                (tau_d * tau_r) / (tau_d - tau_r) * np.log(-(c2 * tau_d) / (c1 * tau_r))
            )
            fmax = eval_exp(t_hat, is_biexp, tconst, coefs)
            t_end = -tau_d * np.log(
                fmax * 0.49 / c1
            )  # make the target < 0.5 to account for numerical errors
        else:
            t_hat = 0
            fmax = eval_exp(t_hat, is_biexp, tconst, coefs)
            if c1 > c2:  # use the dominant postive term to determine bracket end
                t_end = -tau_d * np.log((fmax * 0.49 - max(0, c2)) / c1)
            else:
                t_end = -tau_r * np.log((fmax * 0.49 - max(0, c1)) / c2)
    else:
        a, b = tconst
        c1, c2 = coefs
        t_hat = (1 / b) * np.arctan2(c2 * b + c1 * a, c1 * b - c2 * a)
        t_end = (1 / b) * np.arctan2(-c1, c2)
        if t_end <= 0:
            t_end = (1 / b) * (np.arctan2(-c1, c2) + 2 * np.pi)
    f0 = eval_exp(0, is_biexp, tconst, coefs)
    fmax = eval_exp(t_hat, is_biexp, tconst, coefs)
    if verbose:
        print("t_hat: {}, t_end: {}".format(t_hat, t_end))
        print(
            "f0: {}, fmax: {}, fend: {}".format(
                f0, fmax, eval_exp(t_end, is_biexp, tconst, coefs)
            )
        )
    rt0 = root_scalar(
        lambda t: eval_exp(t, is_biexp, tconst, coefs) - (fmax + f0) / 2,
        bracket=[0, t_hat],
    )
    rt1 = root_scalar(
        lambda t: eval_exp(t, is_biexp, tconst, coefs) - fmax / 2,
        bracket=[t_hat, t_end],
    )
    assert rt0.converged and rt1.converged
    return (rt0.root, rt1.root), t_hat


def shift_frame(fm, sh, fill=np.nan):
    if np.isnan(fm).all():
        return fm
    sh = np.around(sh).astype(int)
    fm = np.roll(fm, sh, axis=np.arange(fm.ndim))
    index = [slice(None) for _ in range(fm.ndim)]
    for ish, s in enumerate(sh):
        index = [slice(None) for _ in range(fm.ndim)]
        if s > 0:
            index[ish] = slice(None, s)
            fm[tuple(index)] = fill
        elif s == 0:
            continue
        elif s < 0:
            index[ish] = slice(s, None)
            fm[tuple(index)] = fill
    return fm

"""Utility functions for deconv module."""

import numpy as np
import scipy.sparse as sps
from numba import njit
from scipy.signal import ShortTimeFFT
from indeca.core.simulation import tau2AR


def get_stft_spec(x: np.ndarray, stft: ShortTimeFFT) -> np.ndarray:
    """Compute STFT spectrogram."""
    spec = np.abs(stft.stft(x)) ** 2
    t = stft.t(len(x))
    t_mask = np.logical_and(t >= 0, t < len(x))
    return spec[:, t_mask]


def construct_R(T: int, up_factor: int):
    """Construct the resampling matrix R."""
    if up_factor > 1:
        return sps.csc_matrix(
            (
                np.ones(T * up_factor),
                (np.repeat(np.arange(T), up_factor), np.arange(T * up_factor)),
            ),
            shape=(T, T * up_factor),
        )
    else:
        return sps.eye(T, format="csc")


def sum_downsample(a, factor):
    """Sum downsample array a by factor."""
    return np.convolve(a, np.ones(factor), mode="full")[factor - 1 :: factor]


def construct_G(fac: np.ndarray, T: int, fromTau=False):
    """Construct the generator matrix G."""
    # I think we should be able to remove fromTau argument since we don't use it anywhere.
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


def max_thres(
    a: np.ndarray,
    nthres: int,
    th_min=0.1,
    th_max=0.9,
    ds=None,
    return_thres=False,
    th_amplitude=False,
    delta=1e-6,
    reverse_thres=False,
    nz_only: bool = False,
):
    """Threshold array a with nthres levels."""
    # Accept any array-like; normalized to numpy.
    a = np.asarray(a)
    amax = a.max()
    if reverse_thres:
        thres = np.linspace(th_max, th_min, nthres)
    else:
        thres = np.linspace(th_min, th_max, nthres)
    if th_amplitude:
        S_ls = [np.floor_divide(a, (amax * th).clip(delta, None)) for th in thres]
    else:
        S_ls = [(a > (amax * th).clip(delta, None)) for th in thres]
    if ds is not None:
        S_ls = [sum_downsample(s, ds) for s in S_ls]
    if nz_only:
        Snz = [ss.sum() > 0 for ss in S_ls]
        S_ls = [ss for ss, nz in zip(S_ls, Snz) if nz]
        thres = [th for th, nz in zip(thres, Snz) if nz]
    if return_thres:
        return S_ls, thres
    else:
        return S_ls


@njit(nopython=True, nogil=True, cache=True)
def bin_convolve(
    coef: np.ndarray, s: np.ndarray, nzidx_s: np.ndarray = None, s_len: int = None
):
    """Binary convolution implemented in numba."""
    coef_len = len(coef)
    if s_len is None:
        s_len = len(s)
    out = np.zeros(s_len)
    nzidx = np.where(s)[0]
    if nzidx_s is not None:
        nzidx = nzidx_s[nzidx].astype(
            np.int64
        )  # astype to fix numpa issues on GPU on Windows
    for i0 in nzidx:
        i1 = min(i0 + coef_len, s_len)
        clen = i1 - i0
        out[i0:i1] += coef[:clen]
    return out


@njit(nopython=True, nogil=True, cache=True)
def max_consecutive(arr):
    """Find maximum consecutive ones."""
    max_count = 0
    current_count = 0
    for value in arr:
        if value:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count

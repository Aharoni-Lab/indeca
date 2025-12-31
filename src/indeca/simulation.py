"""
Simulation utilities for generating synthetic calcium imaging data.

This module provides functions for simulating calcium imaging data including:
- Spike train generation using Markov chain models
- Calcium dynamics using autoregressive (AR) or bi-exponential kernels
- Spatial footprint generation for simulated neurons
- Full video simulation with motion artifacts and background signals

The simulations are used for algorithm validation and benchmarking, allowing
comparison of inferred spikes against known ground truth.
"""

import warnings
from typing import Optional, Tuple, Union

import dask.array as darr
import numpy as np
import pandas as pd
import sparse
import xarray as xr
from numpy import random
from numpy.random import Generator
from numpy.typing import NDArray
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
    cent: Optional[NDArray] = None,
    norm: bool = True,
) -> NDArray:
    """
    Generate 2D Gaussian spatial footprints for simulated neurons.

    Creates spatial footprints by placing 2D Gaussian distributions at specified
    or random locations. Each neuron's footprint is independently sized based on
    the provided size distribution parameters.

    Parameters
    ----------
    height : int
        Height of the spatial field in pixels.
    width : int
        Width of the spatial field in pixels.
    sz_mean : float
        Mean size (variance) of the Gaussian footprints.
    sz_sigma : float
        Standard deviation of the size distribution.
    sz_min : float
        Minimum allowed size (variance) for footprints.
    cent : NDArray, optional
        Centroids of shape (n_cells, 2) specifying [row, col] positions.
        If None, centroids are randomly generated.
    norm : bool, default=True
        If True, normalize each footprint to [0, 1] range.

    Returns
    -------
    NDArray
        Spatial footprints of shape (n_cells, height, width).
    """
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


def apply_arcoef(s: NDArray, g: NDArray, shifted: bool = False) -> NDArray:
    """
    Apply AR(2) coefficients to a spike train to generate calcium dynamics.

    Implements the autoregressive relationship:
        c[t] = s[t] + g[0] * c[t-1] + g[1] * c[t-2]

    This models calcium indicator dynamics where calcium concentration at each
    time point depends on the current spike and previous calcium values.

    Parameters
    ----------
    s : NDArray
        Spike train of shape (n_timepoints,). Can be binary (0/1) or continuous.
    g : NDArray
        AR(2) coefficients of shape (2,), where g[0] = γ₁ and g[1] = γ₂.
        These determine the decay characteristics of the calcium response.
    shifted : bool, default=False
        If True, use spike from previous time point (s[t-1]) instead of s[t].
        This models a delay between spike and calcium response.

    Returns
    -------
    NDArray
        Calcium trace of shape (n_timepoints,).

    See Also
    --------
    tau2AR : Convert time constants to AR coefficients.
    apply_exp : Apply bi-exponential kernel via convolution.
    """
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
    s: NDArray,
    tau_d: float,
    tau_r: float,
    p_d: float = 1,
    p_r: float = -1,
    kn_len: Optional[int] = None,
    trunc_thres: Optional[float] = None,
) -> NDArray:
    """
    Apply bi-exponential kernel to a spike train via convolution.

    Convolves the spike train with a bi-exponential kernel of the form:
        h(t) = p_d * exp(-t/τ_d) + p_r * exp(-t/τ_r)

    This models calcium indicator dynamics with distinct rise and decay phases.

    Parameters
    ----------
    s : NDArray
        Spike train of shape (n_timepoints,).
    tau_d : float
        Decay time constant in frames. Must be positive and > tau_r.
    tau_r : float
        Rise time constant in frames. Must be positive and < tau_d.
    p_d : float, default=1
        Amplitude coefficient for decay component.
    p_r : float, default=-1
        Amplitude coefficient for rise component. Typically negative to create
        the characteristic rising phase.
    kn_len : int, optional
        Length of the kernel. If None, uses len(s).
    trunc_thres : float, optional
        Truncate kernel when amplitude falls below this threshold.
        Improves computational efficiency for long traces.

    Returns
    -------
    NDArray
        Calcium trace of shape (n_timepoints,).

    Raises
    ------
    ValueError
        If tau_d is not positive.

    See Also
    --------
    apply_arcoef : Apply AR coefficients directly.
    tau2AR : Convert time constants to AR coefficients.
    """
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
        trunc_idx = np.where(kn >= trunc_thres)[0].max() + 1
        kn = kn[:trunc_idx]
    return np.convolve(kn, s, mode="full")[: len(s)]


def ar_trace(
    frame: int,
    P: NDArray,
    g: Optional[NDArray] = None,
    tau_d: Optional[float] = None,
    tau_r: Optional[float] = None,
    shifted: bool = False,
    rng: Optional[Generator] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Generate a calcium trace with Markovian spike train using AR dynamics.

    Generates a spike train using a 2-state Markov chain and applies AR(2)
    dynamics to produce a calcium trace.

    Parameters
    ----------
    frame : int
        Number of time frames to simulate.
    P : NDArray
        Markov transition matrix of shape (2, 2). P[i, j] is the probability
        of transitioning from state i to state j.
    g : NDArray, optional
        AR(2) coefficients of shape (2,). If None, computed from tau_d and tau_r.
    tau_d : float, optional
        Decay time constant. Required if g is None.
    tau_r : float, optional
        Rise time constant. Required if g is None.
    shifted : bool, default=False
        If True, apply one-frame delay between spike and calcium response.
    rng : Generator, optional
        NumPy random generator for reproducibility.

    Returns
    -------
    C : NDArray
        Calcium trace of shape (frame,).
    S : NDArray
        Binary spike train of shape (frame,).

    See Also
    --------
    exp_trace : Generate trace using bi-exponential convolution.
    markov_fire : Generate Markovian spike train.
    """
    if g is None:
        g = np.array(tau2AR(tau_d, tau_r))
    S = markov_fire(frame, P, rng=rng).astype(float)
    C = apply_arcoef(S, g, shifted=shifted)
    return C, S


def exp_trace(
    frame: int, P: NDArray, tau_d: float, tau_r: float, trunc_thres: float = 1e-6
) -> Tuple[NDArray, NDArray]:
    """
    Generate a calcium trace with Markovian spike train using bi-exponential kernel.

    Uses a 2-state Markov model to generate bursty spike trains, then convolves
    with a bi-exponential kernel to produce realistic calcium dynamics.

    Parameters
    ----------
    frame : int
        Number of time frames to simulate.
    P : NDArray
        Markov transition matrix of shape (2, 2).
    tau_d : float
        Decay time constant in frames.
    tau_r : float
        Rise time constant in frames.
    trunc_thres : float, default=1e-6
        Truncate kernel when amplitude falls below this threshold.

    Returns
    -------
    C : NDArray
        Calcium trace of shape (frame,).
    S : NDArray
        Binary spike train of shape (frame,).

    See Also
    --------
    ar_trace : Generate trace using AR dynamics.
    """
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


def markov_fire(
    frame: int, P: NDArray, rng: Optional[Generator] = None
) -> NDArray:
    """
    Generate a binary spike train using a 2-state Markov chain.

    Simulates neural firing as a two-state process where the transition
    probabilities determine burst characteristics. Ensures at least one
    spike is generated.

    Parameters
    ----------
    frame : int
        Number of time frames to simulate.
    P : NDArray
        Markov transition matrix of shape (2, 2). P[0, 1] controls the
        probability of starting a spike from quiescence, and P[1, 1]
        controls burst continuation probability.

    rng : Generator, optional
        NumPy random generator for reproducibility. If None, creates a
        new default generator.

    Returns
    -------
    NDArray
        Binary spike train of shape (frame,) with dtype int.

    Raises
    ------
    AssertionError
        If P is not shape (2, 2) or rows don't sum to 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    # makes sure markov probabilities are correct shape
    assert P.shape == (2, 2)
    # make sure probabilities sum to 1
    assert (P.sum(axis=1) == 1).all()
    while True:
        # allocate array for spiking and generate
        S = np.zeros(frame, dtype=int)
        for i in range(1, len(S)):
            S[i] = rng.choice([0, 1], p=P[S[i - 1], :])
        # make sure at least one firing exists
        if S.sum() > 0:
            break
    return S


def random_walk(
    n_stp: int,
    stp_var: float = 1,
    constrain_factor: float = 0,
    ndim: int = 1,
    norm: bool = False,
    integer: bool = True,
    nn: bool = False,
    smooth_var: Optional[float] = None,
) -> NDArray:
    """
    Generate a random walk with optional constraints and smoothing.

    Used for simulating motion artifacts and background signal fluctuations.

    Parameters
    ----------
    n_stp : int
        Number of time steps.
    stp_var : float, default=1
        Variance of step sizes (standard deviation of Gaussian steps).
    constrain_factor : float, default=0
        Mean-reversion strength. If > 0, steps are biased toward origin.
        Higher values produce more constrained walks.
    ndim : int, default=1
        Number of dimensions for the walk.
    norm : bool, default=False
        If True, normalize output to [0, 1] range per dimension.
    integer : bool, default=True
        If True, round walk values to integers.
    nn : bool, default=False
        If True, clip negative values to zero (non-negative).
    smooth_var : float, optional
        If provided, apply Gaussian smoothing with this sigma.

    Returns
    -------
    NDArray
        Random walk of shape (n_stp, ndim).
    """
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
    tmp_P: NDArray,
    tmp_tau_d: float,
    tmp_tau_r: float,
    approx_fps: float = 30,
    spike_sampling_rate: int = 500,
    noise: float = 0.01,
) -> pd.DataFrame:
    """
    Simulate calcium traces for multiple cells with configurable parameters.

    Parameters
    ----------
    num_cells : int
        Number of cells to simulate.
    length_in_sec : float
        Duration of simulation in seconds.
    tmp_P : NDArray
        Markov transition matrix of shape (2, 2) for spike generation.
    tmp_tau_d : float
        Decay time constant in seconds.
    tmp_tau_r : float
        Rise time constant in seconds.
    approx_fps : float, default=30
        Approximate frames per second for the output.
    spike_sampling_rate : int, default=500
        Internal sampling rate for spike generation in Hz.
    noise : float, default=0.01
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: C_true, S_true, C, S, C_noisy, fps,
        upsample_factor, spike_sampling_rate.

    Notes
    -----
    This function is marked for future integration with exp_trace and the
    rest of the simulation pipeline.
    """
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
    tmp_P: NDArray,
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
    cent: Optional[NDArray] = None,
    zero_thres: float = 1e-8,
    chk_size: int = 1000,
    upsample: int = 1,
) -> Union[
    Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray],
    Tuple[
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
    ],
]:
    """
    Simulate complete calcium imaging data including video, spatial footprints, and signals.

    Generates synthetic calcium imaging data with realistic characteristics including
    spatial footprints, temporal dynamics, background fluctuations, motion artifacts,
    and noise.

    Parameters
    ----------
    ncell : int
        Number of cells to simulate.
    dims : dict
        Dictionary with keys 'frame', 'height', 'width' specifying video dimensions.
    sig_scale : float
        Signal amplitude scaling factor.
    sz_mean : float
        Mean size of cell spatial footprints.
    sz_sigma : float
        Standard deviation of cell sizes.
    sz_min : float
        Minimum cell size.
    tmp_P : NDArray
        Markov transition matrix of shape (2, 2) for spike generation.
    tmp_tau_d : float
        Decay time constant in frames.
    tmp_tau_r : float
        Rise time constant in frames.
    post_offset : float
        Baseline offset added to video.
    post_gain : float
        Gain factor applied to video (for converting to uint8).
    bg_nsrc : int
        Number of background signal sources.
    bg_tmp_var : float
        Temporal variance of background signals.
    bg_cons_fac : float
        Constraint factor for background temporal dynamics.
    bg_smth_var : float
        Smoothing variance for background signals.
    mo_stp_var : float
        Step variance for motion simulation.
    mo_cons_fac : float, default=1
        Constraint factor for motion.
    cent : NDArray, optional
        Predefined cell centroids of shape (ncell, 2).
    zero_thres : float, default=1e-8
        Threshold below which spatial footprint values are set to zero.
    chk_size : int, default=1000
        Chunk size for Dask arrays.
    upsample : int, default=1
        Temporal upsampling factor for higher resolution spike timing.

    Returns
    -------
    tuple
        If upsample == 1: (Y, A, C, S, shifts)
        If upsample > 1: (Y, A, C, S, C_true, S_true, shifts)

        Where:
        - Y: Video data (frame, height, width)
        - A: Spatial footprints (unit_id, height, width)
        - C: Calcium traces (frame, unit_id)
        - S: Spike trains (frame, unit_id)
        - C_true, S_true: High-resolution versions when upsampled
        - shifts: Motion shifts (frame, shift_dim)
    """
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


def generate_data(dpath: str, save_Y: bool = False, **kwargs) -> xr.Dataset:
    """
    Generate and save simulated calcium imaging data to a NetCDF file.

    Wrapper around simulate_data that saves the results to disk.

    Parameters
    ----------
    dpath : str
        Path to save the NetCDF file.
    save_Y : bool, default=False
        If True, include the video data Y in the saved dataset.
        Video data can be large, so it's excluded by default.
    **kwargs
        Additional arguments passed to simulate_data.

    Returns
    -------
    xr.Dataset
        The merged dataset containing all simulation outputs.
    """
    dat_vars = simulate_data(**kwargs)
    if not save_Y:
        dat_vars = dat_vars[1:]
    ds = xr.merge(dat_vars)
    ds.to_netcdf(dpath)
    return ds


def computeY(
    A: NDArray,
    C: NDArray,
    A_bg: NDArray,
    C_bg: NDArray,
    shifts: NDArray,
    sig_scale: float,
    noise_scale: float,
    post_offset: float,
    post_gain: float,
) -> NDArray:
    """
    Compute fluorescence video from spatial and temporal components.

    Combines cell signals, background signals, motion shifts, and noise to
    generate a realistic calcium imaging video. Used as a Dask blockwise function.

    Parameters
    ----------
    A : NDArray
        Cell spatial footprints of shape (n_cells, height, width).
    C : NDArray
        Cell temporal signals of shape (n_frames, n_cells).
    A_bg : NDArray
        Background spatial footprints of shape (n_bg, height, width).
    C_bg : NDArray
        Background temporal signals of shape (n_frames, n_bg).
    shifts : NDArray
        Motion shifts of shape (n_frames, 2) for [height, width] shifts.
    sig_scale : float
        Signal amplitude scaling factor.
    noise_scale : float
        Standard deviation of additive Gaussian noise.
    post_offset : float
        Baseline offset added after scaling.
    post_gain : float
        Gain factor for converting to uint8 range.

    Returns
    -------
    NDArray
        Video data of shape (n_frames, height, width) with dtype uint8.
    """
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


def tau2AR(
    tau_d: float, tau_r: float, p: float = 1, return_scl: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, float]]:
    """
    Convert bi-exponential time constants to AR(2) coefficients.

    Transforms decay and rise time constants (τ_d, τ_r) into autoregressive
    coefficients (θ₁, θ₂) that produce equivalent dynamics.

    The relationship is:
        z₁ = exp(-1/τ_d), z₂ = exp(-1/τ_r)
        θ₁ = z₁ + z₂, θ₂ = -z₁ * z₂

    Parameters
    ----------
    tau_d : float
        Decay time constant in frames.
    tau_r : float
        Rise time constant in frames.
    p : float, default=1
        Amplitude scaling factor for the bi-exponential.
    return_scl : bool, default=False
        If True, also return the scaling factor.

    Returns
    -------
    theta0 : float
        First AR coefficient (γ₁).
    theta1 : float
        Second AR coefficient (γ₂).
    scl : float, optional
        Scaling factor, returned if return_scl=True.

    See Also
    --------
    AR2tau : Inverse conversion from AR coefficients to time constants.
    """
    z1, z2 = np.exp(-1 / tau_d), np.exp(-1 / tau_r)
    theta0, theta1 = np.real(z1 + z2), np.real(-z1 * z2)
    if theta1 == 0:
        warnings.warn(
            "Zero AR coefficient detect. Adding a small eps to keep sparsity pattern"
        )
        theta1 = np.finfo(float).eps
    if return_scl:
        scl = p * (z1 - z2)
        return theta0, theta1, scl
    else:
        return theta0, theta1


def AR2tau(
    theta1: float, theta2: float, solve_amp: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, float]]:
    """
    Convert AR(2) coefficients to bi-exponential time constants.

    Inverse of tau2AR. Finds the roots of the characteristic polynomial
    and converts to time constants.

    Parameters
    ----------
    theta1 : float
        First AR coefficient (γ₁).
    theta2 : float
        Second AR coefficient (γ₂).
    solve_amp : bool, default=False
        If True, also compute and return the amplitude scaling factor.

    Returns
    -------
    tau_d : float
        Decay time constant in frames. May be complex if AR process is oscillatory.
    tau_r : float
        Rise time constant in frames. May be complex if AR process is oscillatory.
    p : float, optional
        Amplitude scaling factor, returned if solve_amp=True.

    Notes
    -----
    If the AR coefficients correspond to an oscillatory (underdamped) system,
    the returned time constants will be complex numbers.

    See Also
    --------
    tau2AR : Forward conversion from time constants to AR coefficients.
    AR2exp : Full conversion including amplitude coefficients.
    """
    rts = np.roots([1, -theta1, -theta2])
    z1, z2 = rts
    if np.imag(z1) == 0 and np.isclose(z1, 0) and z1 < 0:
        z1 = np.abs(z1)
    if np.imag(z2) == 0 and np.isclose(z2, 0) and z2 < 0:
        z2 = np.abs(z2)
    tau_d, tau_r = np.nan_to_num([-1 / np.log(z1), -1 / np.log(z2)])
    if solve_amp:
        p = solve_p(tau_d, tau_r)
        return tau_d, tau_r, p
    else:
        return tau_d, tau_r


def solve_p(tau_d: float, tau_r: float) -> float:
    """
    Compute amplitude scaling factor for bi-exponential kernel.

    Calculates the scaling factor p such that the bi-exponential kernel
    h(t) = p * (exp(-t/τ_d) - exp(-t/τ_r)) integrates properly with the
    AR representation.

    Parameters
    ----------
    tau_d : float
        Decay time constant in frames.
    tau_r : float
        Rise time constant in frames.

    Returns
    -------
    float
        Amplitude scaling factor.

    Raises
    ------
    AssertionError
        If the result is NaN or infinite.
    """
    p = 1 / (np.exp(-1 / tau_d) - np.exp(-1 / tau_r))
    assert not (np.isnan(p) or np.isinf(p))
    return p


def AR2exp(
    theta1: float, theta2: float
) -> Tuple[bool, NDArray, NDArray]:
    """
    Convert AR(2) coefficients to exponential representation with coefficients.

    Determines whether the AR process corresponds to real bi-exponential
    dynamics or complex (oscillatory) dynamics, and returns the appropriate
    parameters.

    Parameters
    ----------
    theta1 : float
        First AR coefficient (γ₁).
    theta2 : float
        Second AR coefficient (γ₂).

    Returns
    -------
    is_biexp : bool
        True if the dynamics are real bi-exponential, False if oscillatory.
    tconst : NDArray
        Time constants of shape (2,). If is_biexp=True, contains [τ_d, τ_r].
        If is_biexp=False, contains [a, b] for exp(at) * (cos(bt) + sin(bt)).
    coef : NDArray
        Amplitude coefficients of shape (2,) for the exponential terms.

    See Also
    --------
    eval_exp : Evaluate the exponential representation at given times.
    """
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


def generate_pulse(nsamp: int) -> Tuple[NDArray, NDArray]:
    """
    Generate a unit impulse (delta function) for kernel analysis.

    Parameters
    ----------
    nsamp : int
        Number of samples.

    Returns
    -------
    pulse : NDArray
        Impulse signal of shape (nsamp,) with pulse[0]=1 and zeros elsewhere.
    t : NDArray
        Time indices of shape (nsamp,).
    """
    t = np.arange(nsamp).astype(float)
    pulse = np.zeros_like(t)
    pulse[0] = 1
    return pulse, t


def ar_pulse(
    theta1: float, theta2: float, nsamp: int, shifted: bool = False
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compute the impulse response of an AR(2) process.

    Parameters
    ----------
    theta1 : float
        First AR coefficient (γ₁).
    theta2 : float
        Second AR coefficient (γ₂).
    nsamp : int
        Number of samples for the response.
    shifted : bool, default=False
        If True, apply one-sample delay.

    Returns
    -------
    ar : NDArray
        Impulse response of shape (nsamp,).
    t : NDArray
        Time indices of shape (nsamp,).
    pulse : NDArray
        Input impulse of shape (nsamp,).

    See Also
    --------
    exp_pulse : Impulse response using bi-exponential convolution.
    """
    pulse, t = generate_pulse(nsamp)
    ar = apply_arcoef(pulse, np.array([theta1, theta2]), shifted=shifted)
    return ar, t, pulse


def exp_pulse(
    tau_d: float,
    tau_r: float,
    nsamp: int,
    p_d: float = 1,
    p_r: float = -1,
    kn_len: Optional[int] = None,
    trunc_thres: Optional[float] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compute the impulse response using bi-exponential convolution.

    Parameters
    ----------
    tau_d : float
        Decay time constant in frames.
    tau_r : float
        Rise time constant in frames.
    nsamp : int
        Number of samples for the response.
    p_d : float, default=1
        Decay amplitude coefficient.
    p_r : float, default=-1
        Rise amplitude coefficient.
    kn_len : int, optional
        Kernel length for convolution.
    trunc_thres : float, optional
        Threshold for kernel truncation.

    Returns
    -------
    exp : NDArray
        Impulse response of shape (nsamp,).
    t : NDArray
        Time indices of shape (nsamp,).
    pulse : NDArray
        Input impulse of shape (nsamp,).

    See Also
    --------
    ar_pulse : Impulse response using AR coefficients.
    """
    pulse, t = generate_pulse(nsamp)
    exp = apply_exp(pulse, tau_d, tau_r, p_d, p_r, kn_len, trunc_thres)
    return exp, t, pulse


def eval_exp(
    t: NDArray, is_biexp: bool, tconst: NDArray, coefs: NDArray
) -> NDArray:
    """
    Evaluate exponential response at given time points.

    Computes the value of an exponential kernel (either bi-exponential or
    oscillatory) at specified times.

    Parameters
    ----------
    t : NDArray
        Time points at which to evaluate.
    is_biexp : bool
        If True, use bi-exponential form. If False, use oscillatory form.
    tconst : NDArray
        Time constants of shape (2,). For bi-exponential: [τ_d, τ_r].
        For oscillatory: [a, b] where response is exp(at) * (c1*cos(bt) + c2*sin(bt)).
    coefs : NDArray
        Amplitude coefficients of shape (2,).

    Returns
    -------
    NDArray
        Evaluated response values at each time point.

    See Also
    --------
    AR2exp : Convert AR coefficients to exponential parameters.
    """
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


def find_dhm(
    is_biexp: bool, tconst: NDArray, coefs: NDArray, verbose: bool = False
) -> Tuple[Tuple[float, float], float]:
    """
    Find Distance to Half Maximum (DHM) metrics for a calcium kernel.

    Computes temporal metrics that characterize kernel dynamics:
    - DHM_r: Time to rise from baseline to half-maximum
    - DHM_d: Time to decay from peak to half-maximum

    These metrics are robust to oscillatory tails and provide interpretable
    measures of calcium indicator dynamics.

    Parameters
    ----------
    is_biexp : bool
        If True, kernel is bi-exponential. If False, it's oscillatory.
    tconst : NDArray
        Time constants of shape (2,). See eval_exp for interpretation.
    coefs : NDArray
        Amplitude coefficients of shape (2,).
    verbose : bool, default=False
        If True, print intermediate values for debugging.

    Returns
    -------
    dhm : Tuple[float, float]
        (DHM_r, DHM_d) - rise and decay half-max times.
    t_peak : float
        Time of peak amplitude.

    Raises
    ------
    AssertionError
        If root finding does not converge.

    Notes
    -----
    DHM metrics are computed based on the first threshold-crossing in each
    direction, making them robust to oscillatory behavior in the kernel tail.
    """
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


def shift_frame(
    fm: NDArray, sh: NDArray, fill: float = np.nan
) -> NDArray:
    """
    Shift a frame by integer offsets and fill edges.

    Applies circular shift (roll) to a frame and fills the vacated edges
    with a specified value. Used for simulating motion artifacts.

    Parameters
    ----------
    fm : NDArray
        Frame to shift, can be 2D or 3D.
    sh : NDArray
        Shift values for each dimension, shape (ndim,).
    fill : float, default=np.nan
        Value to fill vacated edges.

    Returns
    -------
    NDArray
        Shifted frame with same shape as input.
    """
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

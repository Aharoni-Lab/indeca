"""
AR kernel estimation and manipulation utilities.

This module provides functions for estimating and refining autoregressive (AR)
kernel parameters from calcium imaging data. It implements the kernel update
step of the InDeCa algorithm, which uses inferred spike trains to iteratively
estimate interpretable bi-exponential calcium dynamics.

Key functionality:
- Solve for AR coefficients from data
- Fit sum-of-exponentials models to impulse responses
- Estimate noise levels from power spectral density
- Convert between AR coefficients and time constants
"""

import warnings
from typing import Literal, Optional, Tuple, Union

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sps
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import lstsq, toeplitz
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acovf

from indeca.deconv import construct_G, construct_R
from indeca.simulation import AR2tau, ar_pulse, solve_p, tau2AR


def convolve_g(s: NDArray, g: NDArray) -> NDArray:
    """
    Convolve a signal with the inverse AR filter (i.e., the kernel H = G⁻¹).

    Constructs the AR matrix G from coefficients g, computes its inverse G⁻¹,
    and applies it to the input signal. Since H = G⁻¹, this effectively
    convolves with the impulse response kernel derived from the AR coefficients.

    For spike input s, this produces calcium c = G⁻¹s = Hs.

    Parameters
    ----------
    s : NDArray
        Input signal of shape (T,).
    g : NDArray
        AR coefficients of shape (p,), typically (2,) for AR(2).

    Returns
    -------
    NDArray
        Output signal G⁻¹s of shape (T,).

    See Also
    --------
    convolve_h : Convolve with a general kernel h directly.
    construct_G : Build the AR relationship matrix G.

    Notes
    -----
    The AR relationship is Gc = s. This function computes c = G⁻¹s,
    which is equivalent to convolving s with the kernel h where Hs = c.
    """
    G = construct_G(g, len(s))
    Gi = sps.linalg.inv(G)
    return np.array(Gi @ s.reshape((-1, 1))).squeeze()


def convolve_h(s: NDArray, h: NDArray) -> NDArray:
    """
    Convolve a signal with a general kernel using matrix multiplication.

    Builds the full convolution matrix H from kernel h and applies it to
    the input signal s. The result is equivalent to np.convolve(h, s)
    but uses explicit matrix construction.

    Parameters
    ----------
    s : NDArray
        Input signal of shape (T,).
    h : NDArray
        Convolution kernel (impulse response) of shape (T,).

    Returns
    -------
    NDArray
        Convolved output Hs of shape (T,).

    See Also
    --------
    convolve_g : Convolve with the inverse AR filter G⁻¹.
    """
    T = len(s)
    H0 = h.reshape((-1, 1))
    H1n = [
        np.vstack([np.zeros(i).reshape((-1, 1)), h[:-i].reshape((-1, 1))])
        for i in range(1, T)
    ]
    H = np.hstack([H0] + H1n)
    return np.real(np.array(H @ s.reshape((-1, 1))).squeeze())


def solve_g(
    y: NDArray, s: NDArray, norm: str = "l1", masking: bool = False
) -> Tuple[float, float]:
    """
    Solve for AR(2) coefficients that best relate signals y and s.

    Finds AR coefficients (θ₁, θ₂) that minimize the reconstruction
    error ||G @ y - s|| subject to stability constraints, where G is
    the AR matrix parameterized by θ₁ and θ₂.

    Parameters
    ----------
    y : NDArray
        First input signal of shape (T,).
    s : NDArray
        Second input signal of shape (T,).
    norm : str, default="l1"
        Error norm to minimize: "l1" or "l2".
    masking : bool, default=False
        If True, only consider time points where s > 0.

    Returns
    -------
    theta_1 : float
        First AR coefficient (θ₁), constrained to be non-negative.
    theta_2 : float
        Second AR coefficient (θ₂), constrained to be non-positive.

    Notes
    -----
    The constraints θ₁ ≥ 0 and θ₂ ≤ 0 ensure the AR process has
    appropriate decay characteristics (real, positive time constants).
    """
    T = len(s)
    theta_1, theta_2 = cp.Variable(), cp.Variable()
    G = (
        np.eye(T)
        + np.diag(-np.ones(T - 1), -1) * theta_1
        + np.diag(-np.ones(T - 2), -2) * theta_2
    )
    if masking:
        idx = np.where(s)[0]
        M = np.zeros((len(idx), T))
        for i, j in enumerate(idx):
            M[i, j] = 1
    else:
        M = np.eye(T)
    if norm == "l2":
        obj = cp.Minimize(cp.norm(M @ (G @ y - s)))
    elif norm == "l1":
        obj = cp.Minimize(cp.norm(M @ (G @ y - s), 1))
    cons = [theta_1 >= 0, theta_2 <= 0]
    prob = cp.Problem(obj, cons)
    prob.solve()
    return theta_1.value, theta_2.value


def fit_sumexp(
    y: NDArray, N: int, x: Optional[NDArray] = None, use_l1: bool = False
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Fit a sum of N exponentials to data using the Prony-like method.

    Uses cumulative integration and eigenvalue decomposition to extract
    exponential rates and amplitudes from the input signal.

    Parameters
    ----------
    y : NDArray
        Input signal of shape (T,).
    N : int
        Number of exponential terms to fit.
    x : NDArray, optional
        Time points of shape (T,). If None, uses np.arange(T).
    use_l1 : bool, default=False
        If True, use L1 norm for fitting (more robust to outliers).

    Returns
    -------
    lams : NDArray
        Exponential rates (λ) of shape (N,). Sorted in descending order.
        The exponential form is exp(λ * t), so λ < 0 for decay.
    ps : NDArray
        Amplitude coefficients of shape (N,).
    y_fit : NDArray
        Fitted signal of shape (T,).

    References
    ----------
    .. [1] http://arxiv.org/abs/physics/0305019
    .. [2] https://github.juangburgos.com/FitSumExponentials/lab/index.html
    """
    # ref: http://arxiv.org/abs/physics/0305019
    # ref: https://github.juangburgos.com/FitSumExponentials/lab/index.html
    T = len(y)
    if x is None:
        x = np.arange(T)
    Y_int = np.zeros((T, N))
    Y_int[:, 0] = cumulative_trapezoid(y, x, initial=0)
    for i in range(1, N):
        Y_int[:, i] = cumulative_trapezoid(Y_int[:, i - 1], x, initial=0)
    X_pow = np.zeros((T, N))
    for i, pow in enumerate(range(N)[::-1]):
        X_pow[:, i] = x**pow
    Y = np.concatenate([Y_int, X_pow], axis=1)
    if use_l1:
        A = lst_l1(Y, y)
    else:
        A = np.linalg.inv(Y.T @ Y) @ Y.T @ y
    A_bar = np.vstack(
        [A[:N], np.hstack([np.eye(N - 1), np.zeros(N - 1).reshape(-1, 1)])]
    )
    lams = np.sort(np.linalg.eigvals(A_bar))[::-1]
    X_exp = np.hstack([np.exp(lam * x).reshape((-1, 1)) for lam in lams])
    if use_l1:
        ps = lst_l1(X_exp, y)
    else:
        ps = np.linalg.inv(X_exp.T @ X_exp) @ X_exp.T @ y
    y_fit = X_exp @ ps
    return lams, ps, y_fit


def fit_sumexp_split(y: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Fit sum of exponentials by splitting at the peak.

    Separately fits single exponentials to the rising and decaying
    portions of the signal. Useful for signals with distinct rise
    and decay phases.

    Parameters
    ----------
    y : NDArray
        Input signal of shape (T,).

    Returns
    -------
    lams : NDArray
        Exponential rates [λ_decay, λ_rise] of shape (2,).
    ps : NDArray
        Amplitude coefficients [p_decay, p_rise] of shape (2,).
    y_fit : NDArray
        Fitted signal of shape (T,).
    """
    T = len(y)
    x = np.arange(T)
    idx_split = np.argmax(y)
    lam_r, p_r, y_fit_r = fit_sumexp(y[:idx_split], 1, x=x[:idx_split])
    lam_d, p_d, y_fit_d = fit_sumexp(y[idx_split:], 1, x=x[idx_split:])
    return (
        np.array([lam_d, lam_r]),
        np.array([p_d, p_r]),
        np.concatenate([y_fit_r, y_fit_d]),
    )


def fit_sumexp_gd(
    y: NDArray,
    x: Optional[NDArray] = None,
    y_weight: Optional[NDArray] = None,
    fit_amp: Union[bool, str] = True,
    interp_factor: int = 100,
) -> Tuple[NDArray, NDArray, float, NDArray]:
    """
    Fit bi-exponential using gradient descent (curve_fit).

    Uses nonlinear least squares to fit a bi-exponential kernel to the
    input signal. Initial guesses are estimated from the signal shape.

    Parameters
    ----------
    y : NDArray
        Input signal of shape (T,).
    x : NDArray, optional
        Time points of shape (T,). If None, uses np.arange(T).
    y_weight : NDArray, optional
        Per-point weights for fitting (inverse variance).
    fit_amp : bool or str, default=True
        How to handle amplitude:
        - True: Fit time constants, compute amplitude from AR normalization
        - False: Fit time constants only, use p = [1, -1]
        - "scale": Fit time constants and an overall scaling factor
    interp_factor : int, default=100
        Interpolation factor for initial guess estimation.

    Returns
    -------
    lams : NDArray
        Exponential rates [-1/τ_d, -1/τ_r] of shape (2,).
    p : NDArray
        Amplitude coefficients of shape (2,).
    scal : float
        Overall scaling factor (1.0 if fit_amp != "scale").
    y_fit : NDArray
        Fitted signal of shape (T,).

    Warnings
    --------
    Issues a warning if τ_d ≤ τ_r (decay faster than rise), and swaps them.
    """
    T = len(y)
    if x is None:
        x = np.arange(T)
    x_interp = np.linspace(x[0], x[-1], interp_factor * len(x))
    y_interp = np.interp(x_interp, x, y)
    idx_max = np.argmax(y)
    idx_max_interp = np.argmax(y_interp)
    fmax = y[idx_max]
    f0 = y[0]
    if idx_max_interp > 0:
        tau_r_init = (
            np.argmin(
                np.abs(y_interp[:idx_max_interp] - f0 - (1 - 1 / np.e) * (fmax - f0))
            )
            / interp_factor
        )
    else:
        tau_r_init = 0
    tau_d_init = (
        np.argmin(np.abs(y_interp[idx_max_interp:] - (1 / np.e) * fmax))
        + idx_max_interp
    ) / interp_factor
    if fit_amp == "scale":
        res = curve_fit(
            lambda x, d, r, scal: scal
            * (np.exp(-x / d) - np.exp(-x / r))
            / (np.exp(-1 / d) - np.exp(-1 / r)),
            x,
            y,
            p0=(tau_d_init, tau_r_init, 1),
            bounds=(0, np.inf),
            sigma=y_weight,
            absolute_sigma=True,
            max_nfev=5000,
            # loss="huber",
            # f_scale=1e-2,
            # tr_solver="exact",
        )
        tau_d, tau_r, scal = res[0]
        p = np.array([1, -1]) / (np.exp(-1 / tau_d) - np.exp(-1 / tau_r))
    elif fit_amp is True:
        res = curve_fit(
            lambda x, d, r: (np.exp(-x / d) - np.exp(-x / r))
            / (np.exp(-1 / d) - np.exp(-1 / r)),
            x,
            y,
            p0=(tau_d_init, tau_r_init),
            bounds=(0, np.inf),
        )
        tau_d, tau_r = res[0]
        p = np.array([1, -1]) / (np.exp(-1 / tau_d) - np.exp(-1 / tau_r))
        scal = 1
    else:
        res = curve_fit(
            lambda x, d, r: np.exp(-x / d) - np.exp(-x / r),
            x,
            y,
            p0=(tau_d_init, tau_r_init),
            bounds=(0, np.inf),
        )
        tau_d, tau_r = res[0]
        p = np.array([1, -1])
        scal = 1
    if tau_d <= tau_r:
        warnings.warn(
            "decaying time smaller than rising time: "
            f"tau_d: {tau_d}, tau_r: {tau_r}\n"
            "reversing coefficients"
        )
        tau_d, tau_r = tau_r, tau_d
        p = p[::-1]
    return (
        -1 / np.array([tau_d, tau_r]),
        p,
        scal,
        scal * (p[0] * np.exp(-x / tau_d) + p[1] * np.exp(-x / tau_r)),
    )


def fit_sumexp_iter(
    y: NDArray, max_iters: int = 50, atol: float = 1e-3, **kwargs
) -> Tuple[NDArray, float, float, NDArray, pd.DataFrame]:
    """
    Iteratively fit bi-exponential with amplitude refinement.

    Alternates between fitting time constants and updating the amplitude
    normalization until convergence.

    Parameters
    ----------
    y : NDArray
        Input signal of shape (T,).
    max_iters : int, default=50
        Maximum number of iterations.
    atol : float, default=1e-3
        Absolute tolerance for amplitude convergence.
    **kwargs
        Additional arguments passed to fit_sumexp_gd.

    Returns
    -------
    lams : NDArray
        Final exponential rates of shape (2,).
    p : float
        Final amplitude normalization factor.
    scal : float
        Overall scaling factor.
    y_fit : NDArray
        Fitted signal of shape (T,).
    coef_df : pd.DataFrame
        Iteration history with columns: i_iter, p, tau_d, tau_r.

    Warnings
    --------
    Issues a warning if max_iters is reached without convergence.
    """
    _, _, scal, y_fit = fit_sumexp_gd(y, fit_amp="scale")
    y_norm = y / scal
    p = 1
    coef_df = []
    for i_iter in range(max_iters):
        lams, _, _, y_fit = fit_sumexp_gd(y_norm / p, fit_amp=False, **kwargs)
        taus = -1 / lams
        p_new = 1 / (np.exp(lams[0]) - np.exp(lams[1]))
        coef_df.append(
            pd.DataFrame(
                [
                    {
                        "i_iter": i_iter,
                        "p": p,
                        "tau_d": taus[0],
                        "tau_r": taus[1],
                    }
                ]
            )
        )
        if np.abs(p_new - p) < atol:
            break
        else:
            p = p_new
    else:
        warnings.warn("max scale iteration reached for sumexp fitting")
    coef_df = pd.concat(coef_df, ignore_index=True)
    return lams, p, scal, y_fit, coef_df


def lst_l1(A: NDArray, b: NDArray) -> NDArray:
    """
    Solve least squares with L1 norm using convex optimization.

    Minimizes ||b - A @ x||_1 using CVXPY.

    Parameters
    ----------
    A : NDArray
        Design matrix of shape (m, n).
    b : NDArray
        Target vector of shape (m,).

    Returns
    -------
    NDArray
        Solution vector of shape (n,).

    Raises
    ------
    AssertionError
        If the optimization does not reach optimal status.
    """
    x = cp.Variable(A.shape[1])
    obj = cp.Minimize(cp.norm(b - A @ x, 1))
    prob = cp.Problem(obj)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    return x.value


def solve_h(
    y: NDArray,
    s: NDArray,
    scal: NDArray,
    err_wt: Optional[NDArray] = None,
    h_len: int = 60,
    norm: str = "l2",
    smth_penalty: float = 0,
    ignore_len: int = 0,
    up_factor: int = 1,
) -> NDArray:
    """
    Solve for the convolution kernel h given observed data and spikes.

    Estimates an unconstrained kernel h that minimizes reconstruction error:
        ||y - scale * R @ (h * s) - b||

    where R is the downsampling matrix and b is a baseline offset.

    Parameters
    ----------
    y : NDArray
        Observed signal. Shape (T,) for single unit or (n_cells, T)
        for multiple units with shared kernel.
    s : NDArray
        Input signal (e.g., spike trains). Shape matches y.
    scal : NDArray
        Amplitude scaling factors. Shape (1,) or (n_cells, 1).
    err_wt : NDArray, optional
        Per-timepoint error weights of shape matching y.
    h_len : int, default=60
        Length of the kernel to estimate.
    norm : str, default="l2"
        Error norm: "l1" or "l2".
    smth_penalty : float, default=0
        L1 penalty on kernel differences (smoothness regularization).
    ignore_len : int, default=0
        Number of initial kernel samples to exclude from smoothness penalty.
    up_factor : int, default=1
        Temporal upsampling factor.

    Returns
    -------
    NDArray
        Estimated kernel h, zero-padded to length T.

    Notes
    -----
    Uses CLARABEL solver for convex optimization. The baseline offset b
    is constrained to be non-negative.
    """
    y, s = y.squeeze(), s.squeeze()
    assert y.ndim == s.ndim
    multi_unit = y.ndim > 1
    if multi_unit:
        ncell, T = s.shape
        y_len = y.shape[1]
    else:
        T = len(s)
        y_len = len(y)
    R = construct_R(y_len, up_factor)
    if h_len is None:
        h_len = T
    else:
        h_len = min(h_len, T)
    if multi_unit:
        b = cp.Variable((ncell, 1))
    else:
        b = cp.Variable()
    h = cp.Variable(h_len)
    h = cp.hstack([h, 0])
    if multi_unit:
        conv_term = cp.vstack([R @ cp.convolve(ss, h)[:T] for ss in s])
    else:
        conv_term = R @ cp.convolve(s, h)[:T]
    diff_term = y - cp.multiply(scal.reshape((-1, 1)), conv_term) - b
    if err_wt is not None:
        err_wt = np.sqrt(err_wt) if norm == "l2" else err_wt
        diff_term = cp.multiply(err_wt.reshape((-1, 1)), diff_term)
    if norm == "l1":
        err_term = cp.norm(diff_term, 1)
    elif norm == "l2":
        err_term = cp.sum_squares(diff_term)
    obj = cp.Minimize(err_term + smth_penalty * cp.norm(cp.diff(h[ignore_len:]), 1))
    cons = [b >= 0]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    return np.concatenate([h.value, np.zeros(T - h_len - 1)])


def solve_fit_h(
    y: NDArray,
    s: NDArray,
    scal: NDArray,
    N: int = 2,
    s_len: int = 60,
    norm: str = "l1",
    tol: float = 1e-3,
    max_iters: int = 30,
    verbose: bool = False,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, pd.DataFrame, pd.DataFrame]:
    """
    Iteratively solve for kernel with smoothing to ensure real exponentials.

    Uses binary search on smoothing penalty to find the minimum regularization
    that produces a kernel with real (not complex) exponential rates.

    Parameters
    ----------
    y : NDArray
        Observed signal.
    s : NDArray
        Input signal (e.g., spike trains).
    scal : NDArray
        Amplitude scaling factors.
    N : int, default=2
        Number of exponential terms to fit.
    s_len : int, default=60
        Kernel length.
    norm : str, default="l1"
        Error norm for solve_h.
    tol : float, default=1e-3
        Tolerance for smoothing penalty binary search.
    max_iters : int, default=30
        Maximum number of iterations.
    verbose : bool, default=False
        If True, print iteration progress.

    Returns
    -------
    lams : NDArray
        Exponential rates of shape (N,).
    ps : NDArray
        Amplitude coefficients of shape (N,).
    h : NDArray
        Estimated kernel.
    h_fit : NDArray
        Fitted exponential kernel.
    metric_df : pd.DataFrame
        Iteration metrics with columns: iter, smth_penal, isreal.
    h_df : pd.DataFrame
        Kernel history with columns: iter, smth_penal, h, h_fit, frame.
    """
    metric_df = None
    h_df = None
    smth_penal = 0
    niter = 0
    while niter < max_iters:
        h = solve_h(y, s, scal, s_len, norm, smth_penal)
        lams, ps, h_fit = fit_sumexp(h, N)
        met = {
            "iter": niter,
            "smth_penal": smth_penal,
            "isreal": (np.imag(lams) == 0).all(),
        }
        if verbose:
            print(met)
        metric_df = pd.concat([metric_df, pd.DataFrame([met])])
        h_df = pd.concat(
            [
                h_df,
                pd.DataFrame(
                    {
                        "iter": niter,
                        "smth_penal": smth_penal,
                        "h": h,
                        "h_fit": h_fit,
                        "frame": np.arange(len(h)),
                    }
                ),
            ]
        )
        smth_ub = metric_df.loc[metric_df["isreal"], "smth_penal"].min()
        smth_lb = metric_df.loc[~metric_df["isreal"], "smth_penal"].max()
        if smth_ub == 0:
            break
        elif np.isnan(smth_ub):
            smth_penal = max(metric_df["smth_penal"].max(), 1) * 2
        elif np.isnan(smth_lb):
            smth_penal = smth_ub / 2
        else:
            assert smth_ub >= smth_lb
            if met["isreal"] and smth_ub - smth_lb < tol:
                break
            else:
                smth_penal = (smth_ub + smth_lb) / 2
        niter += 1
    else:
        warnings.warn("max smth iteration reached")
    return lams, ps, h, h_fit, metric_df, h_df


def solve_fit_h_num(
    y: NDArray,
    s: NDArray,
    scal: NDArray,
    err_wt: Optional[NDArray] = None,
    N: int = 2,
    h_len: int = 60,
    norm: str = "l2",
    up_factor: int = 1,
) -> Tuple[NDArray, NDArray, float, NDArray, NDArray]:
    """
    Solve for kernel and fit bi-exponential numerically.

    Combines solve_h and fit_sumexp_gd to estimate a kernel and fit
    a bi-exponential model to it.

    Parameters
    ----------
    y : NDArray
        Observed signal.
    s : NDArray
        Input signal (e.g., spike trains).
    scal : NDArray
        Amplitude scaling factors.
    err_wt : NDArray, optional
        Per-timepoint error weights.
    N : int, default=2
        Number of exponential terms.
    h_len : int, default=60
        Kernel length.
    norm : str, default="l2"
        Error norm.
    up_factor : int, default=1
        Temporal upsampling factor.

    Returns
    -------
    lams : NDArray
        Exponential rates of shape (N,).
    p : NDArray
        Amplitude coefficients of shape (N,).
    scal : float
        Fitted scaling factor.
    h : NDArray
        Estimated kernel.
    h_fit_pad : NDArray
        Fitted kernel, zero-padded to match h length.
    """
    if y.ndim == 1:
        ylen = len(y)
    else:
        ylen = y.shape[1]
    if h_len >= ylen:
        warnings.warn("Coefficient length longer than data")
        h_len = ylen - 1
    h = solve_h(y, s, scal, err_wt=err_wt, h_len=h_len, norm=norm, up_factor=up_factor)
    try:
        pos_idx = max(np.where(h > 0)[0][0], 1)  # ignore any preceding negative terms
    except IndexError:
        pos_idx = 1
    try:
        lams, p, scal, h_fit = fit_sumexp_gd(h[pos_idx - 1 :], fit_amp="scale")
    except RuntimeError:
        lams, p, scal, h_fit = fit_sumexp_gd(h[pos_idx - 1 :], fit_amp=False)
    h_fit_pad = np.zeros_like(h)
    h_fit_pad[: len(h_fit)] = h_fit
    return lams, p, scal, h, h_fit_pad


def updateAR(
    y: NDArray,
    s: NDArray,
    scal: NDArray,
    err_wt: Optional[NDArray] = None,
    N: int = 2,
    h_len: int = 60,
    norm: str = "l2",
    up_factor: int = 1,
    pre_agg: bool = True,
) -> Tuple[NDArray, NDArray, float, NDArray, NDArray]:
    """
    Update AR parameters from data and inferred spikes.

    Main kernel update function for InDeCa. Estimates time constants
    from the relationship between observed fluorescence and spike trains.

    Parameters
    ----------
    y : NDArray
        Observed fluorescence of shape (n_cells, T) or (T,).
    s : NDArray
        Spike trains matching y shape.
    scal : NDArray
        Amplitude scaling factors.
    err_wt : NDArray, optional
        Per-timepoint error weights.
    N : int, default=2
        Number of exponential terms (typically 2 for rise and decay).
    h_len : int, default=60
        Kernel length in frames.
    norm : str, default="l2"
        Error norm for kernel estimation.
    up_factor : int, default=1
        Temporal upsampling factor.
    pre_agg : bool, default=True
        If True, aggregate spikes before fitting (more efficient).

    Returns
    -------
    taus : NDArray
        Time constants [τ_d, τ_r] of shape (2,), in original time units.
    ps : NDArray
        Amplitude coefficients of shape (2,).
    ar_scal : float
        Scaling factor from fit.
    h : NDArray
        Estimated kernel (zero-padded).
    h_fit : NDArray
        Fitted bi-exponential kernel (zero-padded).

    Notes
    -----
    This implements the kernel update step from the InDeCa algorithm,
    using the inferred spikes to estimate a denoised calcium dynamics kernel.
    """
    if not pre_agg:
        lams, ps, ar_scal, h, h_fit = solve_fit_h_num(
            y, s, scal, err_wt=err_wt, N=N, h_len=h_len, norm=norm, up_factor=up_factor
        )
        return -1 / lams, ps, ar_scal, h, h_fit
    else:
        multi_unit = y.ndim > 1
        if multi_unit:
            T = s.shape[1]
            y_len = y.shape[1]
        else:
            T = len(s)
            y_len = len(y)
        R = construct_R(y_len, up_factor)
        h_len = int(h_len / up_factor)
        lams, ps, ar_scal, h, h_fit = solve_fit_h_num(
            y, s @ R.T, scal, err_wt=err_wt, N=N, h_len=h_len, norm=norm, up_factor=1
        )
        return (
            -1 / lams * up_factor,
            ps,
            ar_scal,
            np.concatenate([h, np.zeros(T - h_len - 1)]),
            np.concatenate([h_fit, np.zeros(T - h_len - 1)]),
        )


def solve_g_cons(
    y: NDArray,
    s: NDArray,
    lam_tol: float = 1e-6,
    lam_start: float = 1,
    max_iter: int = 30,
) -> Tuple[float, float]:
    """
    Fit AR coefficients with constraint for real exponentials.

    Uses iterative penalty adjustment to find AR coefficients that
    correspond to real (not complex) time constants.

    Parameters
    ----------
    y : NDArray
        First input signal of shape (T,).
    s : NDArray
        Second input signal of shape (T,).
    lam_tol : float, default=1e-6
        Tolerance for penalty convergence.
    lam_start : float, default=1
        Initial penalty value.
    max_iter : int, default=30
        Maximum number of iterations.

    Returns
    -------
    th1 : float
        First AR coefficient.
    th2 : float
        Second AR coefficient.

    Notes
    -----
    The characteristic equation θ₁² + 4θ₂ < 0 indicates complex roots
    (oscillatory response). This function iteratively adjusts the penalty
    to find coefficients on the boundary of the real/complex region.
    """
    T = len(s)
    i_iter = 0
    lam = lam_start
    lam_last = lam_start
    ch_last = -np.inf
    while i_iter < max_iter:
        theta_1, theta_2 = cp.Variable(), cp.Variable()
        G = (
            np.eye(T)
            + np.diag(-np.ones(T - 1), -1) * theta_1
            + np.diag(-np.ones(T - 2), -2) * theta_2
        )
        obj = cp.Minimize(cp.norm(G @ y - s) + lam * (-theta_2 - theta_1))
        cons = [theta_1 >= 0, theta_2 <= 0]
        prob = cp.Problem(obj, cons)
        prob.solve()
        th1, th2 = theta_1.value, theta_2.value
        ch_root = th1**2 + 4 * th2
        if ch_root > 0:
            lam_new = lam / 2
        else:
            if ch_last > 0:
                lam_new = lam + (lam_last - lam) / 2
            else:
                lam_new = lam * 2
        if (lam - lam_new) >= 0 and (lam - lam_new) <= lam_tol:
            break
        else:
            i_iter += 1
            lam_last = lam
            lam = lam_new
            ch_last = ch_root
            print(
                "th1: {}, th2: {}, ch: {}, lam: {}".format(th1, th2, ch_root, lam_last)
            )
    else:
        warnings.warn("max lam iteration reached")
    return th1, th2


def estimate_coefs(
    y: NDArray,
    p: int,
    noise_freq: Optional[Tuple[float, float]],
    use_smooth: bool,
    add_lag: int,
) -> Tuple[NDArray, float]:
    """
    Estimate AR coefficients from noisy data.

    Uses Yule-Walker equations with optional smoothing and noise estimation
    to fit AR coefficients to the input signal.

    Parameters
    ----------
    y : NDArray
        Input fluorescence signal of shape (T,).
    p : int
        Order of the AR process.
    noise_freq : tuple of float, optional
        Frequency range (low, high) as fraction of Nyquist for noise estimation.
        If None, assumes zero noise.
    use_smooth : bool
        If True, low-pass filter the signal before AR estimation.
    add_lag : int
        Additional lags to include in the Yule-Walker estimation.

    Returns
    -------
    g : NDArray
        AR coefficients of shape (p,).
    tn : float
        Estimated noise level.

    See Also
    --------
    get_ar_coef : Core AR coefficient estimation.
    filt_fft : FFT-based filtering.
    noise_fft : Noise estimation from PSD.
    """
    if noise_freq is None:
        tn = 0
    else:
        tn = noise_fft(y, noise_range=(noise_freq, 1))
    if use_smooth:
        y_ar = filt_fft(y.squeeze(), noise_freq, "low")
        tn_ar = noise_fft(y_ar, noise_range=(noise_freq, 1))
    else:
        y_ar, tn_ar = y, tn
    g = get_ar_coef(y_ar, np.nan_to_num(tn_ar), p=p, add_lag=add_lag)
    return g, tn


def filt_fft(x: np.ndarray, freq: float, btype: str) -> np.ndarray:
    """
    Filter 1d timeseries by zero-ing bands in the fft signal.

    Parameters
    ----------
    x : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency.
    btype : str
        Either `"low"` or `"high"` specify low or high pass filtering.

    Returns
    -------
    x_filt : np.ndarray
        Filtered timeseries.
    """
    _T = len(x)
    if btype == "low":
        zero_range = slice(int(freq * _T), None)
    elif btype == "high":
        zero_range = slice(None, int(freq * _T))
    xfft = np.fft.rfft(x)
    xfft[zero_range] = 0
    return np.fft.irfft(xfft, len(x))


def noise_fft(
    px: np.ndarray, noise_range=(0.25, 0.5), noise_method="logmexp", threads=1
) -> float:
    """
    Estimates noise of the input by aggregating power spectral density within
    `noise_range`.

    The PSD is estimated using FFT.

    Parameters
    ----------
    px : np.ndarray
        Input data.
    noise_range : tuple, optional
        Range of noise frequency to be aggregated as a fraction of sampling
        frequency. By default `(0.25, 0.5)`.
    noise_method : str, optional
        Method of aggreagtion for noise. Should be one of `"mean"` `"median"`
        `"logmexp"` or `"sum"`. By default "logmexp".
    threads : int, optional
        Number of threads to use for pyfftw. By default `1`.

    Returns
    -------
    noise : float
        The estimated noise level of input.

    See Also
    -------
    get_noise_fft
    """
    _T = len(px)
    nr = np.around(np.array(noise_range) * _T).astype(int)
    px = 1 / _T * np.abs(np.fft.rfft(px)[nr[0] : nr[1]]) ** 2
    if noise_method == "mean":
        return np.sqrt(px.mean())
    elif noise_method == "median":
        return np.sqrt(px.median())
    elif noise_method == "logmexp":
        eps = np.finfo(px.dtype).eps
        return np.sqrt(np.exp(np.log(px + eps).mean()))
    elif noise_method == "sum":
        return np.sqrt(px.sum())


def get_ar_coef(
    y: np.ndarray, sn: float, p: int, add_lag: int, pad: int = None
) -> np.ndarray:
    """
    Estimate Autoregressive coefficients of order `p` given a timeseries `y`.

    Parameters
    ----------
    y : np.ndarray
        Input timeseries.
    sn : float
        Estimated noise level of the input `y`.
    p : int
        Order of the autoregressive process.
    add_lag : int
        Additional number of timesteps of covariance to use for the estimation.
    pad : int, optional
        Length of the output. If not `None` then the resulting coefficients will
        be zero-padded to this length. By default `None`.

    Returns
    -------
    g : np.ndarray
        The estimated AR coefficients.
    """
    if add_lag == "p":
        max_lag = p * 2
    else:
        max_lag = p + add_lag
    cov = acovf(y, fft=True)
    C_mat = toeplitz(cov[:max_lag], cov[:p]) - sn**2 * np.eye(max_lag, p)
    g = lstsq(C_mat, cov[1 : max_lag + 1])[0]
    if pad:
        res = np.zeros(pad)
        res[: len(g)] = g
        return res
    else:
        return g


def AR_upsamp_real(
    theta: Tuple[float, float], upsamp: int = 1, fit_nsamp: int = 1000
) -> Tuple[Tuple[float, float], NDArray, NDArray]:
    """
    Compute upsampled AR parameters ensuring real exponentials.

    Converts AR coefficients to time constants, scales for upsampling,
    and converts back. Ensures the result corresponds to real (not complex)
    bi-exponential dynamics.

    Parameters
    ----------
    theta : tuple of float
        AR coefficients (θ₁, θ₂) at original sampling rate.
    upsamp : int, default=1
        Upsampling factor.
    fit_nsamp : int, default=1000
        Number of samples to use for impulse response fitting.

    Returns
    -------
    theta_up : tuple of float
        Upsampled AR coefficients (θ₁', θ₂').
    tau_up : NDArray
        Upsampled time constants [τ_d, τ_r] of shape (2,).
    p_up : NDArray
        Amplitude coefficients [p, -p] of shape (2,).

    Raises
    ------
    AssertionError
        If τ_d ≤ τ_r or if amplitude is invalid (NaN, inf, or ≤ 0).
    """
    tr = ar_pulse(*theta, nsamp=fit_nsamp, shifted=True)[0]
    lams, cur_p, scl, tr_fit = fit_sumexp_gd(tr, fit_amp=True)
    tau = -1 / lams
    tau_up = tau * upsamp
    theta_up = tau2AR(*tau_up)
    td, tr = tau_up
    p = solve_p(td, tr)
    assert td > tr
    assert p > 0 and not (np.isinf(p) or np.isnan(p))
    return theta_up, np.array([td, tr]), np.array([p, -p])

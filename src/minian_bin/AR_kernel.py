import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import lstsq, toeplitz
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acovf

from minian_bin.deconv import construct_G, construct_R
from minian_bin.simulation import AR2tau, ar_pulse, tau2AR


def convolve_g(s, g):
    G = construct_G(g, len(s))
    Gi = sps.linalg.inv(G)
    return np.array(Gi @ s.reshape((-1, 1))).squeeze()


def convolve_h(s, h):
    T = len(s)
    H0 = h.reshape((-1, 1))
    H1n = [
        np.vstack([np.zeros(i).reshape((-1, 1)), h[:-i].reshape((-1, 1))])
        for i in range(1, T)
    ]
    H = np.hstack([H0] + H1n)
    return np.real(np.array(H @ s.reshape((-1, 1))).squeeze())


def solve_g(y, s, norm="l1", masking=False):
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


def fit_sumexp(y, N, x=None, use_l1=False):
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


def fit_sumexp_split(y):
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


def fit_sumexp_gd(y, x=None, y_weight=None, fit_amp=True, interp_factor=100):
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
    return (
        -1 / np.array([tau_d, tau_r]),
        p,
        scal,
        scal * (p[0] * np.exp(-x / tau_d) + p[1] * np.exp(-x / tau_r)),
    )


def fit_sumexp_iter(y, max_iters=50, atol=1e-3, **kwargs):
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


def lst_l1(A, b):
    x = cp.Variable(A.shape[1])
    obj = cp.Minimize(cp.norm(b - A @ x, 1))
    prob = cp.Problem(obj)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    return x.value


def solve_h(y, s, scal, h_len=60, norm="l2", smth_penalty=0, ignore_len=0, up_factor=1):
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
    if norm == "l1":
        err_term = cp.norm(y - cp.multiply(scal.reshape((-1, 1)), conv_term) - b, 1)
    elif norm == "l2":
        err_term = cp.sum_squares(y - cp.multiply(scal.reshape((-1, 1)), conv_term) - b)
    obj = cp.Minimize(err_term + smth_penalty * cp.norm(cp.diff(h[ignore_len:]), 1))
    cons = [b >= 0]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    return np.concatenate([h.value, np.zeros(T - h_len - 1)])


def solve_fit_h(
    y,
    s,
    scal,
    N=2,
    s_len=60,
    norm="l1",
    tol=1e-3,
    max_iters: int = 30,
    verbose=False,
):
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


def solve_fit_h_num(y, s, scal, N=2, s_len=60, norm="l2", up_factor=1):
    h = solve_h(y, s, scal, s_len, norm, up_factor=up_factor)
    try:
        pos_idx = max(np.where(h > 0)[0][0], 1)  # ignore any preceding negative terms
    except IndexError:
        pos_idx = 1
    lams, p, scal, h_fit = fit_sumexp_gd(h[pos_idx - 1 :], fit_amp="scale")
    h_fit_pad = np.zeros_like(h)
    h_fit_pad[: len(h_fit)] = h_fit
    return lams, p, scal, h, h_fit_pad


def solve_g_cons(y, s, lam_tol=1e-6, lam_start=1, max_iter=30):
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
    y: np.ndarray, p: int, noise_freq: tuple, use_smooth: bool, add_lag: int
):
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


def AR_upsamp_real(theta, upsamp: int = 1, fit_nsamp: int = 1000):
    tau_d, tau_r, p = AR2tau(*theta, solve_amp=True)
    tau = np.array([tau_d, tau_r])
    if (np.imag(tau) != 0).any() or p == np.inf:
        tr = ar_pulse(*theta, nsamp=fit_nsamp, shifted=True)[0]
        lams, cur_p, scl, tr_fit = fit_sumexp_gd(tr, fit_amp=True)
        tau = -1 / lams
    tau_up = tau * upsamp
    theta_up = tau2AR(*tau_up)
    td, tr, p = AR2tau(*theta_up, solve_amp=True)
    return theta_up, np.array([td, tr]), np.array([p, -p])

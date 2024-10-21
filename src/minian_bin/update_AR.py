import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit

from .update_bin import construct_G
from .utilities import scal_lstsq


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
    X_exp = np.hstack([np.exp(l * x).reshape((-1, 1)) for l in lams])
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


def fit_sumexp_gd(y, x=None, fit_amp=True, interp_factor=100, ar_mode: bool = True):
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
    if fit_amp:
        if ar_mode:
            fit_func = lambda x, p, d, r: p * np.exp(-x / d) - (p - 1) * np.exp(-x / r)
            p0 = 2
        else:
            fit_func = lambda x, p, d, r: p * np.exp(-x / d) - p * np.exp(-x / r)
            p0 = 1
        res = curve_fit(
            fit_func,
            x,
            y,
            p0=(p0, tau_d_init, tau_r_init),
            bounds=(0, np.inf),
            max_nfev=1e6,
        )
        p, tau_d, tau_r = res[0]
        if ar_mode:
            p = np.array([p, 1 - p])
        else:
            p = np.array([p, -p])
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
    if tau_d <= tau_r:
        warnings.warn(
            "decaying time smaller than rising time: tau_d: {}, tau_r: {}\nreversing coefficients".format(
                tau_d, tau_r
            )
        )
        tau_d, tau_r = tau_r, tau_d
    return (
        -1 / np.array([tau_d, tau_r]),
        p,
        p[0] * np.exp(-x / tau_d) + p[1] * np.exp(-x / tau_r),
    )


def fit_sumexp_iter(y, max_iters=50, err_atol=1e-3, err_rtol=1e-3, **kwargs):
    err_org = np.linalg.norm(y)
    err_tol = max(err_atol, err_rtol * err_org)
    err = err_org
    p = y.max()
    coef_df = []
    for i_iter in range(max_iters):
        lams, ps, y_fit = fit_sumexp_gd(y / p, fit_amp=False, **kwargs)
        taus = -1 / lams
        err_last = err
        err = np.linalg.norm(y - y_fit * p)
        coef_df.append(
            pd.DataFrame(
                [
                    {
                        "i_iter": i_iter,
                        "p": p,
                        "tau_d": taus[0],
                        "tau_r": taus[1],
                        "err": err,
                    }
                ]
            )
        )
        if np.abs(err - err_last) < err_tol:
            break
        else:
            p = scal_lstsq(y_fit, y)
    else:
        warnings.warn("max scale iteration reached for sumexp fitting")
    coef_df = pd.concat(coef_df, ignore_index=True)
    return lams, ps * p, y_fit, coef_df


def lst_l1(A, b):
    x = cp.Variable(A.shape[1])
    obj = cp.Minimize(cp.norm(b - A @ x, 1))
    prob = cp.Problem(obj)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    return x.value


def solve_h(y, s, scal, s_len=60, norm="l1", smth_penalty=0, ignore_len=0):
    y, s = y.squeeze(), s.squeeze()
    assert y.ndim == s.ndim
    multi_unit = y.ndim > 1
    if multi_unit:
        ncell, T = s.shape
    else:
        T = len(s)
    if s_len is None:
        s_len = T
    else:
        s_len = min(s_len, T)
    if multi_unit:
        b = cp.Variable((ncell, 1))
    else:
        b = cp.Variable()
    h = cp.Variable(s_len)
    h = cp.hstack([h, 0])
    if multi_unit:
        conv_term = cp.vstack([cp.convolve(ss, h)[:T] for ss in s])
    else:
        conv_term = cp.convolve(s, h)[:T]
    norm_ord = {"l1": 1, "l2": 2}[norm]
    obj = cp.Minimize(
        cp.norm(y - cp.multiply(scal.reshape((-1, 1)), conv_term) - b, norm_ord)
        + smth_penalty * cp.norm(cp.diff(h[ignore_len:]), 1)
    )
    cons = [b >= 0]
    prob = cp.Problem(obj, cons)
    prob.solve()
    return np.concatenate([h.value, np.zeros(T - s_len - 1)])


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


def solve_fit_h_num(
    y,
    s,
    scal,
    N=2,
    s_len=60,
    norm="l1",
    tol=1e-3,
    max_iters: int = 30,
    ar_mode: bool = True,
):
    metric_df = None
    h_df = None
    i_iter = 0
    while i_iter < max_iters:
        h = solve_h(y, s, scal, s_len, norm)
        lams, ps, h_fit, _ = fit_sumexp_iter(h, ar_mode=ar_mode)
        taus = -1 / lams
        met = pd.DataFrame(
            {
                "iter": i_iter,
                "tau_d": taus[0],
                "tau_r": taus[1],
                "p0": ps[0],
                "p1": ps[1],
                "scal": scal,
            }
        )
        metric_df = pd.concat([metric_df, met], ignore_index=True)
        h_df = pd.concat(
            [
                h_df,
                pd.DataFrame(
                    {
                        "iter": i_iter,
                        "h": h,
                        "h_max": h.max(),
                        "h_fit": h_fit,
                        "frame": np.arange(len(h)),
                    }
                ),
            ]
        )
        if np.abs(ps[0] - 1) < tol:
            break
        else:
            scal = scal * ps[0]
            i_iter += 1
    else:
        warnings.warn("max h fitting iteration reached")
    return lams, ps, h, h_fit, metric_df, h_df


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

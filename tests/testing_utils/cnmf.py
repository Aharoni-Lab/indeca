import cvxpy as cp
import numpy as np
import scipy.sparse as sps

from minian_bin.AR_kernel import estimate_coefs
from minian_bin.simulation import AR2tau, exp_pulse, tau2AR


def pipeline_cnmf(
    Y,
    up_factor=1,
    p=2,
    ar_mode=True,
    ar_kn_len=60,
    tau_init=None,
    est_noise_freq=0.4,
    est_use_smooth=True,
    est_add_lag=20,
    sps_penal=1,
):
    # 0. housekeeping
    ncell, T = Y.shape
    R = construct_R(T, up_factor)
    # 1. estimate parameters
    g = np.empty((ncell, p))
    tau = np.empty((ncell, p))
    ps = np.empty((ncell, p))
    tn = np.empty(ncell)
    for icell, y in enumerate(Y):
        cur_g, cur_tn = estimate_coefs(
            y,
            p=p,
            noise_freq=est_noise_freq,
            use_smooth=est_use_smooth,
            add_lag=est_add_lag,
        )
        g[icell, :] = cur_g
        tau[icell, :] = AR2tau(*cur_g)
        ps[icell, :] = np.array([1, -1])
        tn[icell] = cur_tn
    if tau_init is not None:
        g = np.tile(tau2AR(tau_init[0], tau_init[1]), (ncell, 1))
        tau = np.tile(tau_init, (ncell, 1))
        ps = np.tile([1, -1], (ncell, 1))
    C_cnmf, S_cnmf = np.empty((ncell, T * up_factor)), np.empty((ncell, T * up_factor))
    # 2 cnmf algorithm
    prob = prob_deconv(
        T, coef_len=p if ar_mode else ar_kn_len, ar_mode=ar_mode, use_base=True, R=R
    )
    for icell, y in enumerate(Y):
        if ar_mode:
            cur_coef = g[icell]
        else:
            cur_coef = exp_pulse(
                tau[icell, 0],
                tau[icell, 1],
                ar_kn_len,
                p_d=ps[icell, 0],
                p_r=ps[icell, 1],
            )[0]
        c, s, _ = solve_deconv(y, prob, coef=cur_coef, l1_penal=sps_penal * tn[icell])
        C_cnmf[icell, :] = c.squeeze()
        S_cnmf[icell, :] = s.squeeze()
    return C_cnmf, S_cnmf, tau


def construct_R(T: int, up_factor: int):
    rs_vec = np.zeros(T * up_factor)
    rs_vec[:up_factor] = 1
    return sps.coo_matrix(
        np.stack([np.roll(rs_vec, up_factor * i) for i in range(T)], axis=0)
    )


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
        R = sps.eye(T)
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
    if norm == "l1":
        err_term = cp.sum(cp.abs(y - scale * R @ c - b))
    elif norm == "l2":
        err_term = cp.sum_squares(y - scale * R @ c - b)
    elif norm == "huber":
        err_term = cp.sum(cp.huber(y - scale * R @ c - b))
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
        cons = [c == H @ s]
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


def solve_deconv(
    y: np.ndarray,
    prob: cp.Problem,
    coef: np.ndarray,
    l1_penal: float = 0,
    scale: float = 1,
    return_obj: bool = False,
    solver=None,
    warm_start=False,
):
    c, s, b = prob.data_dict["c"], prob.data_dict["s"], prob.data_dict["b"]
    prob.data_dict["y"].value = y.reshape((-1, 1))
    prob.data_dict["scale"].value = scale
    prob.data_dict["l1_penal"].value = l1_penal
    prob.data_dict["coef"].value = coef
    prob.solve(solver=solver, warm_start=warm_start)
    if return_obj:
        return c.value, s.value, b.value, prob.data_dict["err_term"].value
    else:
        return c.value, s.value, b.value

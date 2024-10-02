import warnings

import numpy as np
import pandas as pd
from line_profiler import profile
from tqdm.auto import tqdm, trange

from .simulation import AR2tau, ar_pulse, exp_pulse, tau2AR
from .update_AR import construct_G, fit_sumexp_gd, solve_fit_h
from .update_bin import (
    construct_R,
    estimate_coefs,
    prob_deconv,
    solve_deconv,
    solve_deconv_bin,
    sum_downsample,
)


@profile
def pipeline_bin(
    Y,
    up_factor=1,
    p=2,
    ar_mode=True,
    tau_init=None,
    return_iter=False,
    max_iters=50,
    err_tol=1e-3,
    est_noise_freq=0.4,
    est_use_smooth=True,
    est_add_lag=20,
    deconv_nthres=1000,
    deconv_norm="l1",
    deconv_scal_tol=1e-5,
    deconv_max_iters=50,
    deconv_use_l0=True,
    ar_use_all=True,
    ar_kn_len=60,
    ar_norm="l1",
):
    # 0. housekeeping
    ncell, T = Y.shape
    R = construct_R(T, up_factor)
    # 1. estimate initial guess at convolution kernel
    if tau_init is not None:
        g = np.tile(tau2AR(tau_init[0], tau_init[1]), (ncell, 1))
        tau = np.tile(tau_init, (ncell, 1))
        ps = np.tile([1, -1], (ncell, 1))
    else:
        if up_factor > 1:
            raise NotImplementedError(
                "Estimation of AR coefficient with upsampling is not implemented"
            )
        g = np.empty((ncell, p))
        tau = np.empty((ncell, p))
        ps = np.empty((ncell, p))
        for icell, y in enumerate(Y):
            cur_g, _ = estimate_coefs(
                y,
                p=p,
                noise_freq=est_noise_freq,
                use_smooth=est_use_smooth,
                add_lag=est_add_lag,
            )
            g[icell, :] = cur_g
            cur_tau = AR2tau(*cur_g)
            if not ar_mode and (np.imag(cur_tau) != 0).any():
                tr = ar_pulse(*cur_g, nsamp=ar_kn_len)[0]
                tr[0] = 0
                lams, cur_p, tr_fit = fit_sumexp_gd(tr, ar_mode=ar_mode)
                tau[icell, :] = -1 / lams
                ps[icell, :] = cur_p
            else:
                tau[icell, :] = cur_tau
                ps[icell, :] = np.array([1, -1])
    # 2. iteration loop
    C_ls = []
    S_ls = []
    scal_ls = []
    h_ls = []
    h_fit_ls = []
    metric_df = pd.DataFrame(
        {
            "iter": -1,
            "cell": np.arange(ncell),
            "g0": g.T[0],
            "g1": g.T[1],
            "tau_d": tau.T[0],
            "tau_r": tau.T[1],
            "p0": ps.T[0],
            "p1": ps.T[1],
            "err": np.nan,
            "scale": np.nan,
        }
    )
    prob = prob_deconv(
        T, coef_len=p if ar_mode else ar_kn_len, ar_mode=ar_mode, R=R, norm=deconv_norm
    )
    prob_cons = prob_deconv(
        T,
        coef_len=p if ar_mode else ar_kn_len,
        ar_mode=ar_mode,
        R=R,
        norm=deconv_norm,
        amp_constraint=True,
    )
    for i_iter in trange(max_iters, desc="iteration"):
        # 2.1 deconvolution
        C, S, scale, err = (
            np.empty((ncell, T * up_factor)),
            np.empty((ncell, T * up_factor)),
            np.empty(ncell),
            np.empty(ncell),
        )
        for icell, y in tqdm(
            enumerate(Y), total=Y.shape[0], desc="deconv", leave=False
        ):
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
            c_bin, s_bin, _, scl, _, _ = solve_deconv_bin(
                y,
                prob,
                prob_cons,
                coef=cur_coef,
                R=R,
                nthres=deconv_nthres,
                tol=deconv_scal_tol,
                max_iters=deconv_max_iters,
                ar_mode=ar_mode,
                use_l0=deconv_use_l0,
                norm=deconv_norm,
            )
            C[icell, :] = c_bin.squeeze()
            S[icell, :] = s_bin.squeeze()
            scale[icell] = scl
            err[icell] = np.linalg.norm(y - c_bin.squeeze())
        # 2.2 update AR
        best_idx = (
            metric_df.set_index("iter").groupby("cell", sort=True)["err"].idxmin()
        )
        if len(best_idx.dropna()) == ncell:
            S_best = np.stack(
                [S_ls[best_i][uid, :] for uid, best_i in best_idx.items()], axis=0
            )
            scal_best = np.stack(
                [scal_ls[best_i][uid] for uid, best_i in best_idx.items()], axis=0
            )
        else:
            S_best = S
            scal_best = scale
        if up_factor > 1:
            S_ar = np.stack([sum_downsample(s, up_factor) for s in S_best], axis=0)
        else:
            S_ar = S_best
        if ar_use_all:
            lams, ps, h, h_fit, _, _ = solve_fit_h(
                Y, S_ar, scal_best, N=p, s_len=ar_kn_len, norm=ar_norm, ar_mode=ar_mode
            )
            tau = np.tile(-1 / lams, (ncell, 1))
            g = np.tile(tau2AR(*(-1 / lams)), (ncell, 1))
            ps = np.tile(ps, (ncell, 1))
        else:
            g = np.empty((ncell, p))
            tau = np.empty((ncell, p))
            ps = np.empty((ncell, p))
            for icell, (y, s) in enumerate(zip(Y, S_ar)):
                lams, cur_ps, _, _, _, _ = solve_fit_h(
                    y, s, scal_best, N=p, s_len=ar_kn_len, norm=ar_norm, ar_mode=ar_mode
                )
                tau[icell, :] = -1 / lams
                g[icell, :] = tau2AR(*(-1 / lams))
                ps[icell, :] = cur_ps
        # 2.3 save iteration results
        cur_metric = pd.DataFrame(
            {
                "iter": i_iter,
                "cell": np.arange(ncell),
                "g0": g.T[0],
                "g1": g.T[1],
                "tau_d": tau.T[0],
                "tau_r": tau.T[1],
                "p0": ps.T[0],
                "p1": ps.T[1],
                "err": err,
                "scale": scale,
                "best_idx": best_idx,
            }
        )
        metric_df = pd.concat([metric_df, cur_metric], ignore_index=True)
        C_ls.append(C)
        S_ls.append(S)
        scal_ls.append(scale)
        h_ls.append(h)
        h_fit_ls.append(h_fit)
        # 2.4 check convergence
        metric_last = metric_df[metric_df["iter"] < i_iter].dropna()
        if len(metric_last) > 0:
            err_cur = cur_metric.set_index("cell")["err"]
            err_best = metric_last.groupby("cell")["err"].min()
            # converged by err
            if (np.abs(err_cur - err_best) < err_tol).all():
                break
            # converged by s
            S_best = np.empty((ncell, T * up_factor))
            for uid, udf in metric_last.groupby("cell"):
                best_iter = udf.set_index("iter")["err"].idxmin()
                S_best[uid, :] = S_ls[best_iter][uid, :]
            if np.abs(S - S_best).sum() < 1:
                break
            # trapped
            err_all = metric_last.pivot(columns="iter", index="cell", values="err")
            diff_all = np.abs(err_cur.values.reshape((-1, 1)) - err_all.values)
            if (diff_all.min(axis=1) < err_tol).all():
                warnings.warn("Solution trapped in local optimal err")
                break
            # trapped by s
            diff_all = np.array([np.abs(S - prev_s).sum() for prev_s in S_ls[:-1]])
            if (diff_all < 1).sum() > 1:
                warnings.warn("Solution trapped in local optimal s")
                break
    else:
        warnings.warn("Max interation reached")
    opt_C, opt_S = np.empty((ncell, T * up_factor)), np.empty((ncell, T * up_factor))
    for icell in range(ncell):
        opt_idx = metric_df.loc[
            metric_df[metric_df["cell"] == icell]["err"].idxmin(), "iter"
        ]
        opt_C[icell, :] = C_ls[opt_idx][icell, :]
        opt_S[icell, :] = S_ls[opt_idx][icell, :]
    if return_iter:
        return opt_C, opt_S, metric_df, C_ls, S_ls, h_ls, h_fit_ls
    else:
        return opt_C, opt_S, metric_df


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
    return C_cnmf, S_cnmf

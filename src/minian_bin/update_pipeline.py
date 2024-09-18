import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange

from .simulation import AR2tau, tau2AR
from .update_AR import construct_G, solve_fit_h
from .update_bin import construct_R, estimate_coefs, solve_deconv_bin, sum_downsample


def pipeline_bin(
    Y,
    up_factor=1,
    p=2,
    tau_init=None,
    return_iter=False,
    max_iters=50,
    err_tol=1e-3,
    est_noise_freq=0.4,
    est_use_smooth=True,
    est_add_lag=20,
    deconv_nthres=1000,
    deconv_scal_tol=1e-5,
    deconv_max_iters=50,
    ar_use_all=True,
    ar_kn_len=60,
    ar_norm="l1",
):
    # 0. housekeeping
    ncell, T = Y.shape
    R = construct_R(T, up_factor)
    # 1. estimate initial guess at convolution kernel
    if tau_init is not None:
        g = np.array(tau2AR(tau_init[0], tau_init[1]))
        tau = tau_init
    else:
        if up_factor > 1:
            raise NotImplementedError(
                "Estimation of AR coefficient with upsampling is not implemented"
            )
        g = np.empty((ncell, p))
        tau = np.empty((ncell, p))
        for icell, y in enumerate(Y):
            cur_g, _ = estimate_coefs(
                y,
                p=p,
                noise_freq=est_noise_freq,
                use_smooth=est_use_smooth,
                add_lag=est_add_lag,
            )
            g[icell, :] = cur_g
            tau[icell, :] = AR2tau(*cur_g)
    # 2. iteration loop
    C_ls = []
    S_ls = []
    metric_df = pd.DataFrame(
        {
            "iter": -1,
            "cell": np.arange(ncell),
            "g0": g.T[0],
            "g1": g.T[1],
            "tau_d": tau.T[0],
            "tau_r": tau.T[1],
            "err": np.nan,
            "scale": np.nan,
        }
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
            if g.ndim == 2:
                G = construct_G(g[icell], T * up_factor)
            elif g.ndim == 1:
                G = construct_G(g, T * up_factor)
            else:
                raise ValueError("g has wrong number of dimensions: {}".format(g.shape))
            c_bin, s_bin, _, scl, _, _ = solve_deconv_bin(
                y,
                G,
                R,
                nthres=deconv_nthres,
                tol=deconv_scal_tol,
                max_iters=deconv_max_iters,
            )
            C[icell, :] = c_bin.squeeze()
            S[icell, :] = s_bin.squeeze()
            scale[icell] = scl
            err[icell] = np.linalg.norm(y - c_bin.squeeze())
        # 2.2 update AR
        if up_factor > 1:
            S_ar = np.stack(
                [sum_downsample(s, up_factor) for s in S], axis=0
            ) * scale.reshape((-1, 1))
        else:
            S_ar = S * scale.reshape((-1, 1))
        if ar_use_all:
            lams, ps, _, _, _, _ = solve_fit_h(
                Y, S_ar, N=p, s_len=ar_kn_len, norm=ar_norm
            )
            tau = -1 / lams
            g = np.array(tau2AR(*tau))
        else:
            g = np.empty((ncell, p))
            tau = np.empty((ncell, p))
            for icell, (y, s) in enumerate(zip(Y, S_ar)):
                lams, ps, _, _, _, _ = solve_fit_h(
                    y, s, N=p, s_len=ar_kn_len, norm=ar_norm
                )
                tau[icell, :] = -1 / lams
                g[icell, :] = tau2AR(*(-1 / lams))
        # 2.3 save iteration results
        cur_metric = pd.DataFrame(
            {
                "iter": i_iter,
                "cell": np.arange(ncell),
                "g0": g.T[0],
                "g1": g.T[1],
                "tau_d": tau.T[0],
                "tau_r": tau.T[1],
                "err": err,
                "scale": scale,
            }
        )
        metric_df = pd.concat([metric_df, cur_metric], ignore_index=True)
        C_ls.append(C)
        S_ls.append(S)
        # 2.4 check convergence
        metric_last = metric_df[metric_df["iter"] < i_iter].dropna()
        if len(metric_last) > 0:
            err_cur = cur_metric.set_index("cell")["err"]
            err_best = metric_last.groupby("cell")["err"].min()
            # converged
            if (np.abs(err_cur - err_best) < err_tol).all():
                break
            # trapped
            err_all = metric_last.pivot(columns="iter", index="cell", values="err")
            if (
                np.nanmin(
                    np.abs(err_cur.values.reshape((-1, 1)) - err_all.values), axis=1
                )
                < err_tol
            ).all():
                warnings.warn("Solution trapped in local optimal")
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
        return opt_C, opt_S, metric_df, C_ls, S_ls
    else:
        return opt_C, opt_S, metric_df


def pipeline_cnmf():
    pass

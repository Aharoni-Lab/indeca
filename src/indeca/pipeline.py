import warnings

import numpy as np
import pandas as pd
from line_profiler import profile
from tqdm.auto import tqdm, trange

from .AR_kernel import AR_upsamp_real, estimate_coefs, updateAR
from .dashboard import Dashboard
from .deconv import DeconvBin
from .logging_config import get_module_logger
from .simulation import AR2tau, find_dhm, tau2AR

# Initialize logger for this module
logger = get_module_logger("pipeline")
logger.info("Pipeline module initialized")  # Test message on import


@profile
def pipeline_bin(
    Y,
    up_factor=1,
    p=2,
    tau_init=None,
    return_iter=False,
    max_iters=50,
    n_best=3,
    use_rel_err=True,
    err_atol=1e-4,
    err_rtol=5e-2,
    est_noise_freq=0.4,
    est_use_smooth=True,
    est_add_lag=20,
    deconv_nthres=1000,
    deconv_norm="l2",
    deconv_atol=1e-3,
    deconv_penal=None,
    deconv_backend="osqp",
    deconv_err_weighting=None,
    deconv_use_base=True,
    deconv_reset_scl=True,
    deconv_masking_radius=None,
    deconv_pks_polish=None,
    deconv_ncons_thres=None,
    deconv_min_rel_scl=None,
    ar_use_all=True,
    ar_kn_len=100,
    ar_norm="l2",
    da_client=None,
    spawn_dashboard=True,
):
    """Binary pursuit pipeline for spike inference.

    Parameters
    ----------
    Y : array-like
        Input fluorescence trace
    ...

    Returns
    -------
    dict
        Dictionary containing results of the pipeline
    """
    logger.info("Starting binary pursuit pipeline")
    # 0. housekeeping
    ncell, T = Y.shape
    logger.debug(
        "Pipeline parameters: "
        f"up_factor={up_factor}, p={p}, max_iters={max_iters}, "
        f"n_best={n_best}, deconv_backend={deconv_backend}, "
        f"ar_use_all={ar_use_all}, ar_kn_len={ar_kn_len}"
        f"{ncell} cells with {T} timepoints"
    )
    if spawn_dashboard:
        if da_client is not None:
            logger.debug("Using Dask client for distributed computation")
            dashboard = da_client.submit(
                Dashboard, Y=Y, kn_len=ar_kn_len, actor=True
            ).result()
        else:
            logger.debug("Running in single-machine mode")
            dashboard = Dashboard(Y=Y, kn_len=ar_kn_len)
    else:
        dashboard = None
    # 1. estimate initial guess at convolution kernel
    if tau_init is not None:
        logger.debug(f"Using provided tau_init: {tau_init}")
        theta = tau2AR(tau_init[0], tau_init[1])
        _, _, pp = AR2tau(theta[0], theta[1], solve_amp=True)
        ps = np.array([pp, -pp])
        theta = np.tile(tau2AR(tau_init[0], tau_init[1]), (ncell, 1))
        tau = np.tile(tau_init, (ncell, 1))
        ps = np.tile(ps, (ncell, 1))
    else:
        logger.debug("Computing initial tau values")
        theta = np.empty((ncell, p))
        tau = np.empty((ncell, p))
        ps = np.empty((ncell, p))
        for icell, y in enumerate(Y):
            cur_theta, _ = estimate_coefs(
                y,
                p=p,
                noise_freq=est_noise_freq,
                use_smooth=est_use_smooth,
                add_lag=est_add_lag,
            )
            cur_theta, cur_tau, cur_p = AR_upsamp_real(
                cur_theta, upsamp=up_factor, fit_nsamp=ar_kn_len
            )
            tau[icell, :] = cur_tau
            theta[icell, :] = cur_theta
            ps[icell, :] = cur_p
    scale = np.empty(ncell)
    # 2. iteration loop
    C_ls = []
    S_ls = []
    scal_ls = []
    h_ls = []
    h_fit_ls = []
    metric_df = pd.DataFrame(
        columns=[
            "iter",
            "cell",
            "g0",
            "g1",
            "tau_d",
            "tau_r",
            "err",
            "err_rel",
            "nnz",
            "scale",
            "best_idx",
            "obj",
            "wgt_len",
        ]
    )
    if da_client is not None:
        dcv = [
            da_client.submit(
                lambda yy, th, tau, ps: DeconvBin(
                    y=yy,
                    theta=th,
                    tau=tau,
                    ps=ps,
                    coef_len=ar_kn_len,
                    upsamp=up_factor,
                    nthres=deconv_nthres,
                    norm=deconv_norm,
                    penal=deconv_penal,
                    use_base=deconv_use_base,
                    err_weighting=deconv_err_weighting,
                    masking_radius=deconv_masking_radius,
                    pks_polish=deconv_pks_polish,
                    ncons_thres=deconv_ncons_thres,
                    min_rel_scl=deconv_min_rel_scl,
                    atol=deconv_atol,
                    backend=deconv_backend,
                    dashboard=dashboard,
                    dashboard_uid=i,
                ),
                y,
                theta[i],
                tau[i],
                ps[i],
            )
            for i, y in enumerate(Y)
        ]
    else:
        dcv = [
            DeconvBin(
                y=y,
                theta=theta[i],
                tau=tau[i],
                ps=ps[i],
                coef_len=ar_kn_len,
                upsamp=up_factor,
                nthres=deconv_nthres,
                norm=deconv_norm,
                penal=deconv_penal,
                use_base=deconv_use_base,
                err_weighting=deconv_err_weighting,
                masking_radius=deconv_masking_radius,
                pks_polish=deconv_pks_polish,
                ncons_thres=deconv_ncons_thres,
                min_rel_scl=deconv_min_rel_scl,
                atol=deconv_atol,
                backend=deconv_backend,
                dashboard=dashboard,
                dashboard_uid=i,
            )
            for i, y in enumerate(Y)
        ]
    for i_iter in trange(max_iters, desc="iteration"):
        logger.info(f"Starting iteration {i_iter}/{max_iters}")
        # 2.1 deconvolution
        res = []
        for icell, y in tqdm(
            enumerate(Y), total=Y.shape[0], desc="deconv", leave=False
        ):
            if da_client is not None:
                r = da_client.submit(
                    lambda d: d.solve_scale(
                        reset_scale=i_iter <= 1 or deconv_reset_scl
                    ),
                    dcv[icell],
                )
            else:
                r = dcv[icell].solve_scale(reset_scale=i_iter <= 1 or deconv_reset_scl)
            res.append(r)
        if da_client is not None:
            res = da_client.gather(res)
        S = np.stack([r[0].squeeze() for r in res], axis=0, dtype=float)
        C = np.stack([r[1].squeeze() for r in res], axis=0)
        scale = np.array([r[2] for r in res])
        err = np.array([r[3] for r in res])
        err_rel = np.array([r[4] for r in res])
        nnz = np.array([r[5] for r in res])
        penal = np.array([r[6] for r in res])
        logger.debug(
            f"Iteration {i_iter} stats - Mean error: {err.mean():.4f}, Mean scale: {scale.mean():.4f}"
        )
        # 2.2 save iteration results
        dhm = np.stack(
            [
                find_dhm(True, (t0, t1), (s, -s))[0]
                for t0, t1, s in zip(tau.T[0], tau.T[1], scale)
            ]
        )
        cur_metric = pd.DataFrame(
            {
                "iter": i_iter,
                "cell": np.arange(ncell),
                "g0": theta.T[0],
                "g1": theta.T[1],
                "tau_d": tau.T[0],
                "tau_r": tau.T[1],
                "dhm0": dhm.T[0],
                "dhm1": dhm.T[1],
                "err": err,
                "err_rel": err_rel,
                "scale": scale,
                "penal": penal,
                "nnz": nnz,
                "obj": err_rel if use_rel_err else err,
                "wgt_len": [d.wgt_len for d in dcv],
            }
        )
        if dashboard is not None:
            dashboard.update(
                tau_d=cur_metric["tau_d"].squeeze(),
                tau_r=cur_metric["tau_r"].squeeze(),
                err=cur_metric["obj"].squeeze(),
                scale=cur_metric["scale"].squeeze(),
            )
            dashboard.set_iter(min(i_iter + 1, max_iters - 1))
        metric_df = pd.concat([metric_df, cur_metric], ignore_index=True)
        C_ls.append(C)
        S_ls.append(S)
        scal_ls.append(scale)
        try:
            h_ls.append(h)
            h_fit_ls.append(h_fit)
        except UnboundLocalError:
            h_ls.append(np.full(T * up_factor, np.nan))
            h_fit_ls.append(np.full(T * up_factor, np.nan))
        # 2.3 update AR
        metric_df = metric_df.set_index(["iter", "cell"])
        if n_best is not None and i_iter >= n_best:
            S_best = np.empty_like(S)
            scal_best = np.empty_like(scale)
            if tau_init is not None:
                metric_best = metric_df
            else:
                metric_best = metric_df.loc[1:, :]
            for icell, cell_met in metric_best.groupby("cell", sort=True):
                cell_met = cell_met.reset_index().sort_values("obj", ascending=True)
                cur_idx = np.array(cell_met["iter"][:n_best])
                metric_df.loc[(i_iter, icell), "best_idx"] = ",".join(
                    cur_idx.astype(str)
                )
                S_best[icell, :] = np.sum(
                    np.stack([S_ls[i][icell, :] for i in cur_idx], axis=0), axis=0
                ) > (n_best / 2)
                scal_best[icell] = np.mean([scal_ls[i][icell] for i in cur_idx])
        else:
            S_best = S
            scal_best = scale
        metric_df = metric_df.reset_index()
        S_ar = S_best
        if ar_use_all:
            cur_tau, ps, ar_scal, h, h_fit = updateAR(
                Y,
                S_ar,
                scal_best,
                N=p,
                h_len=ar_kn_len * up_factor,
                norm=ar_norm,
                up_factor=up_factor,
            )
            if dashboard is not None:
                dashboard.update(
                    h=h[: ar_kn_len * up_factor], h_fit=h_fit[: ar_kn_len * up_factor]
                )
            tau = np.tile(cur_tau, (ncell, 1))
            for idx, d in enumerate(dcv):
                if da_client is not None:
                    da_client.submit(
                        lambda dd: dd.update(tau=cur_tau, scale=scal_best[idx]), d
                    )
                else:
                    d.update(tau=cur_tau, scale=scal_best[idx])
            logger.debug(
                f"Updating AR parameters for all cells: tau:{tau}, ar_scal: {ar_scal}"
            )
        else:
            theta = np.empty((ncell, p))
            tau = np.empty((ncell, p))
            for icell, (y, s) in enumerate(zip(Y, S_ar)):
                cur_tau, ps, ar_scal, h, h_fit = updateAR(
                    y,
                    s,
                    scal_best[icell],
                    N=p,
                    h_len=ar_kn_len,
                    norm=ar_norm,
                    up_factor=up_factor,
                )
                if dashboard is not None:
                    dashboard.update(uid=icell, h=h, h_fit=h_fit)
                tau[icell, :] = cur_tau
                if da_client is not None:
                    da_client.submit(
                        lambda dd: dd.update(tau=cur_tau, scale=scal_best[icell]),
                        dcv[icell],
                    )
                else:
                    dcv[icell].update(tau=cur_tau, scale=scal_best[icell])
                logger.debug(
                    f"Updating AR parameters for cell {icell}: tau:{tau}, ar_scal: {ar_scal}"
                )
        # 2.4 check convergence
        metric_prev = metric_df[metric_df["iter"] < i_iter].dropna(
            subset=["obj", "scale"]
        )
        metric_last = metric_df[metric_df["iter"] == i_iter - 1].dropna(
            subset=["obj", "scale"]
        )
        if len(metric_prev) > 0:
            err_cur = cur_metric.set_index("cell")["obj"]
            err_last = metric_last.set_index("cell")["obj"]
            err_best = metric_prev.groupby("cell")["obj"].min()
            # converged by err
            if (np.abs(err_cur - err_last) < err_atol).all():
                logger.info("Converged: absolute error tolerance reached")
                break
            # converged by relative err
            if (np.abs(err_cur - err_last) < err_rtol * err_best).all():
                logger.info("Converged: relative error tolerance reached")
                break
            # converged by s
            S_best = np.empty((ncell, T * up_factor))
            for uid, udf in metric_prev.groupby("cell"):
                best_iter = udf.set_index("iter")["obj"].idxmin()
                S_best[uid, :] = S_ls[best_iter][uid, :]
            if np.abs(S - S_best).sum() < 1:
                logger.info("Converged: spike pattern stabilized")
                break
            # trapped
            err_all = metric_prev.pivot(columns="iter", index="cell", values="obj")
            diff_all = np.abs(err_cur.values.reshape((-1, 1)) - err_all.values)
            if (diff_all.min(axis=1) < err_atol).all():
                logger.warning("Solution trapped in local optimal err")
                break
            # trapped by s
            diff_all = np.array([np.abs(S - prev_s).sum() for prev_s in S_ls[:-1]])
            if (diff_all < 1).sum() > 1:
                logger.warning("Solution trapped in local optimal s")
                break
    else:
        logger.warning("Max iteration reached without convergence")
    # Compute final results
    opt_C, opt_S = np.empty((ncell, T * up_factor)), np.empty((ncell, T * up_factor))
    mobj = metric_df.groupby("iter")["obj"].median()
    opt_idx_all = mobj.idxmin()
    for icell in range(ncell):
        if ar_use_all:
            opt_idx = opt_idx_all
        else:
            opt_idx = metric_df.loc[
                metric_df[metric_df["cell"] == icell]["obj"].idxmin(), "iter"
            ]
        opt_C[icell, :] = C_ls[opt_idx][icell, :]
        opt_S[icell, :] = S_ls[opt_idx][icell, :]
    C_ls.append(opt_C)
    S_ls.append(opt_S)
    if dashboard is not None:
        dashboard.stop()
    logger.info("Pipeline completed successfully")
    if return_iter:
        return opt_C, opt_S, metric_df, C_ls, S_ls, h_ls, h_fit_ls
    else:
        return opt_C, opt_S, metric_df

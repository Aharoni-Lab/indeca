import warnings

import numpy as np
import pandas as pd
from line_profiler import profile
from tqdm.auto import tqdm, trange

from .AR_kernel import estimate_coefs, fit_sumexp_gd, solve_fit_h_num
from .dashboard import Dashboard
from .deconv import DeconvBin
from .logging_config import get_module_logger
from .simulation import AR2tau, ar_pulse, tau2AR

# Initialize logger for this module
logger = get_module_logger("pipeline")
logger.info("Pipeline module initialized")  # Test message on import

# Add a more visible test message
logger.warning("TESTING PIPELINE LOGGER - THIS SHOULD APPEAR IN LOG FILE")


@profile
def pipeline_bin(
    Y,
    up_factor=1,
    p=2,
    tau_init=None,
    return_iter=False,
    max_iters=50,
    n_best=3,
    err_atol=1e-1,
    err_rtol=5e-2,
    est_noise_freq=0.4,
    est_use_smooth=True,
    est_add_lag=20,
    deconv_nthres=1000,
    deconv_norm="l1",
    deconv_atol=1e-3,
    deconv_penal="l1",
    deconv_backend="cvxpy",
    ar_use_all=True,
    ar_kn_len=100,
    ar_norm="l1",
    da_client=None,
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
    logger.debug(
        "Pipeline parameters: "
        f"up_factor={up_factor}, p={p}, max_iters={max_iters}, "
        f"n_best={n_best}, deconv_backend={deconv_backend}, "
        f"ar_use_all={ar_use_all}, ar_kn_len={ar_kn_len}"
    )

    # 0. housekeeping
    ncell, T = Y.shape
    logger.info(f"Processing {ncell} cells with {T} timepoints")

    if da_client is not None:
        logger.debug("Using Dask client for distributed computation")
        dashboard = da_client.submit(
            Dashboard, Y=Y, kn_len=ar_kn_len, actor=True
        ).result()
    else:
        logger.debug("Running in single-machine mode")
        dashboard = Dashboard(Y=Y, kn_len=ar_kn_len)

    # 1. estimate initial guess at convolution kernel
    logger.info("Estimating initial convolution kernel")
    if tau_init is not None:
        logger.debug(f"Using provided tau_init: {tau_init}")
        theta = np.tile(tau2AR(tau_init[0], tau_init[1]), (ncell, 1))
        tau = np.tile(tau_init, (ncell, 1))
    else:
        logger.debug("Computing initial tau values")
        theta = np.empty((ncell, p))
        tau = np.empty((ncell, p))
        for icell, y in enumerate(Y):
            if icell % 10 == 0:  # Log progress every 10 cells
                logger.debug(f"Processing initial estimate for cell {icell}/{ncell}")
            cur_theta, _ = estimate_coefs(
                y,
                p=p,
                noise_freq=est_noise_freq,
                use_smooth=est_use_smooth,
                add_lag=est_add_lag,
            )
            tau_d, tau_r, cur_p = AR2tau(*cur_theta, solve_amp=True)
            cur_tau = np.array([tau_d, tau_r])
            if (np.imag(cur_tau) != 0).any():
                # fit and convert tau to real value
                logger.warning(
                    f"Complex tau values detected for cell {icell}, converting to real values"
                )
                tr = ar_pulse(*cur_theta, nsamp=ar_kn_len, shifted=True)[0]
                lams, cur_p, scl, tr_fit = fit_sumexp_gd(tr, fit_amp="scale")
                cur_tau = (-1 / lams) * up_factor
            cur_theta = tau2AR(cur_tau[0], cur_tau[1], cur_p)
            tau[icell, :] = cur_tau
            theta[icell, :] = cur_theta

    scale = np.empty(ncell)
    # 2. iteration loop
    logger.info("Starting iteration loop")
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
            "scale",
            "best_idx",
        ]
    )

    if da_client is not None:
        logger.debug("Setting up distributed deconvolution")
        dcv = [
            da_client.submit(
                lambda yy, tt: DeconvBin(
                    y=yy,
                    theta=tt,
                    coef_len=ar_kn_len,
                    upsamp=up_factor,
                    nthres=deconv_nthres,
                    norm=deconv_norm,
                    penal=deconv_penal,
                    atol=deconv_atol,
                    backend=deconv_backend,
                    dashboard=dashboard,
                    dashboard_uid=i,
                ),
                y,
                theta[i],
            )
            for i, y in enumerate(Y)
        ]
    else:
        logger.debug("Setting up local deconvolution")
        dcv = [
            DeconvBin(
                y=y,
                theta=theta[i],
                coef_len=ar_kn_len,
                upsamp=up_factor,
                nthres=deconv_nthres,
                norm=deconv_norm,
                penal=deconv_penal,
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
        logger.debug("Performing deconvolution")
        res = []
        for icell, y in tqdm(
            enumerate(Y), total=Y.shape[0], desc="deconv", leave=False
        ):
            if da_client is not None:
                r = da_client.submit(
                    lambda d: d.solve_scale(reset_scale=i_iter <= 1), dcv[icell]
                )
            else:
                r = dcv[icell].solve_scale(reset_scale=i_iter <= 1)
            res.append(r)

        if da_client is not None:
            res = da_client.gather(res)

        S = np.stack([r[0].squeeze() for r in res], axis=0, dtype=float)
        C = np.stack([r[1].squeeze() for r in res], axis=0)
        scale = np.array([r[2] for r in res])
        err = np.array([r[3] for r in res])
        penal = np.array([r[4] for r in res])

        # Log some statistics about the results
        logger.debug(
            f"Iteration {i_iter} stats - Mean error: {err.mean():.4f}, Mean scale: {scale.mean():.4f}"
        )

        # 2.2 save iteration results
        cur_metric = pd.DataFrame(
            {
                "iter": i_iter,
                "cell": np.arange(ncell),
                "g0": theta.T[0],
                "g1": theta.T[1],
                "tau_d": tau.T[0],
                "tau_r": tau.T[1],
                "err": err,
                "scale": scale,
                "penal": penal,
            }
        )

        dashboard.update(
            tau_d=cur_metric["tau_d"].squeeze(),
            tau_r=cur_metric["tau_r"].squeeze(),
            err=cur_metric["err"].squeeze(),
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
        if n_best is not None and i_iter > n_best:
            logger.debug(f"Selecting best {n_best} iterations for update")
            S_best = np.empty_like(S)
            scal_best = np.empty_like(scale)
            for icell, cell_met in metric_df.loc[1:, :].groupby("cell", sort=True):
                cell_met = cell_met.reset_index().sort_values("err", ascending=True)
                cur_idx = np.array(cell_met["iter"][:n_best])
                metric_df.loc[(i_iter, icell), "best_idx"] = ",".join(
                    cur_idx.astype(str)
                )
                S_best[icell, :] = np.sum(
                    np.stack([S_ls[i][icell, :] for i in cur_idx], axis=0), axis=0
                ) > (n_best / 2)
                scal_best[icell] = np.median([scal_ls[i][icell] for i in cur_idx])
        else:
            S_best = S
            scal_best = scale

        metric_df = metric_df.reset_index()
        S_ar = S_best

        if ar_use_all:
            logger.debug("Updating AR parameters using all cells")
            lams, ps, ar_scal, h, h_fit = solve_fit_h_num(
                Y,
                S_ar,
                scal_best,
                N=p,
                s_len=ar_kn_len * up_factor,
                norm=ar_norm,
                up_factor=up_factor,
            )
            dashboard.update(
                h=h[: ar_kn_len * up_factor], h_fit=h_fit[: ar_kn_len * up_factor]
            )
            cur_tau = -1 / lams
            tau = np.tile(cur_tau, (ncell, 1))
            for d in dcv:
                if da_client is not None:
                    da_client.submit(
                        lambda dd: dd.update(tau=cur_tau, scale_mul=ar_scal), d
                    )
                else:
                    d.update(tau=cur_tau, scale_mul=ar_scal)
        else:
            logger.debug("Updating AR parameters per cell")
            theta = np.empty((ncell, p))
            tau = np.empty((ncell, p))
            for icell, (y, s) in enumerate(zip(Y, S_ar)):
                lams, ps, ar_scal, h, h_fit = solve_fit_h_num(
                    y, s, scal_best, N=p, s_len=ar_kn_len, norm=ar_norm
                )
                dashboard.update(uid=icell, h=h, h_fit=h_fit)
                cur_tau = -1 / lams
                tau[icell, :] = cur_tau
                if da_client is not None:
                    da_client.submit(
                        lambda dd: dd.update(tau=cur_tau, scale_mul=ar_scal), dcv[icell]
                    )
                else:
                    dcv[icell].update(tau=cur_tau, scale_mul=ar_scal)

        # 2.4 check convergence
        metric_prev = metric_df[metric_df["iter"] < i_iter].dropna(
            subset=["err", "scale"]
        )
        metric_last = metric_df[metric_df["iter"] == i_iter - 1].dropna(
            subset=["err", "scale"]
        )

        if len(metric_prev) > 0:
            err_cur = cur_metric.set_index("cell")["err"]
            err_last = metric_last.set_index("cell")["err"]
            err_best = metric_prev.groupby("cell")["err"].min()

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
                best_iter = udf.set_index("iter")["err"].idxmin()
                S_best[uid, :] = S_ls[best_iter][uid, :]

            if np.abs(S - S_best).sum() < 1:
                logger.info("Converged: spike pattern stabilized")
                break

            # trapped
            err_all = metric_prev.pivot(columns="iter", index="cell", values="err")
            diff_all = np.abs(err_cur.values.reshape((-1, 1)) - err_all.values)
            if (diff_all.min(axis=1) < err_atol).all():
                msg = "Solution trapped in local optimal err"
                warnings.warn(msg)  # API-level warning for users
                logger.warning(msg)  # Operational warning for logs
                break

            # trapped by s
            diff_all = np.array([np.abs(S - prev_s).sum() for prev_s in S_ls[:-1]])
            if (diff_all < 1).sum() > 1:
                msg = "Solution trapped in local optimal s"
                warnings.warn(msg)  # API-level warning for users
                logger.warning(msg)  # Operational warning for logs
                break
    else:
        # Max iteration reached
        msg = "Max iteration reached without convergence"
        warnings.warn(msg)  # API-level warning for users
        logger.warning(msg)  # Operational warning for logs

    # Compute final results
    logger.info("Computing final results")
    opt_C, opt_S = np.empty((ncell, T * up_factor)), np.empty((ncell, T * up_factor))
    for icell in range(ncell):
        opt_idx = metric_df.loc[
            metric_df[metric_df["cell"] == icell]["err"].idxmin(), "iter"
        ]
        opt_C[icell, :] = C_ls[opt_idx][icell, :]
        opt_S[icell, :] = S_ls[opt_idx][icell, :]

    dashboard.stop()
    logger.info("Pipeline completed successfully")

    if return_iter:
        return opt_C, opt_S, metric_df, C_ls, S_ls, h_ls, h_fit_ls
    else:
        return opt_C, opt_S, metric_df

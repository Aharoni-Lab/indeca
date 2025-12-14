"""Binary pursuit deconvolution pipeline.

This module contains the main pipeline_bin function that orchestrates
the entire deconvolution process in a readable, top-down manner.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from line_profiler import profile
from tqdm.auto import trange

from indeca.core.simulation import tau2AR
from indeca.dashboard.dashboard import Dashboard
from indeca.utils.logging_config import get_module_logger

from .ar_update import (
    make_S_ar,
    propagate_ar_update,
    select_best_spikes,
    update_ar_parameters,
)
from .config import DeconvPipelineConfig
from .convergence import check_convergence
from .init import initialize_ar_params, initialize_deconvolvers
from .iteration import run_deconv_step
from .metrics import append_metrics, make_cur_metric, update_dashboard
from .preprocess import preprocess_traces
from .types import IterationState

logger = get_module_logger("pipeline")


@profile
def pipeline_bin(
    Y: np.ndarray,
    *,
    config: DeconvPipelineConfig,
    da_client: Any = None,
    spawn_dashboard: bool = True,
    return_iter: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray, pd.DataFrame, list, list, list, list],
]:
    """Binary pursuit pipeline for spike inference.

    This is the main entry point for the deconvolution pipeline.
    It orchestrates preprocessing, initialization, iterative deconvolution,
    AR updates, and convergence checking.

    Parameters
    ----------
    Y : np.ndarray
        Input fluorescence traces, shape (ncell, T)
    config : DeconvPipelineConfig
        Pipeline configuration
    da_client : Client or None
        Dask client for distributed execution. None for local execution.
    spawn_dashboard : bool
        Whether to spawn a real-time dashboard
    return_iter : bool
        Whether to return per-iteration results

    Returns
    -------
    opt_C : np.ndarray
        Optimal calcium traces, shape (ncell, T * up_factor)
    opt_S : np.ndarray
        Optimal spike trains, shape (ncell, T * up_factor)
    metric_df : pd.DataFrame
        Per-iteration metrics
    C_ls : list (only if return_iter=True)
        Calcium traces per iteration
    S_ls : list (only if return_iter=True)
        Spike trains per iteration
    h_ls : list (only if return_iter=True)
        Impulse responses per iteration
    h_fit_ls : list (only if return_iter=True)
        Fitted impulse responses per iteration
    """
    logger.info("Starting binary pursuit pipeline")

    # Unpack config
    up_factor = config.up_factor
    p = config.p
    preprocess_cfg = config.preprocess
    init_cfg = config.init
    deconv_cfg = config.deconv
    ar_cfg = config.ar_update
    conv_cfg = config.convergence

    # 0. Housekeeping
    ncell, T = Y.shape
    logger.debug(
        f"Pipeline parameters: "
        f"up_factor={up_factor}, p={p}, max_iters={conv_cfg.max_iters}, "
        f"n_best={conv_cfg.n_best}, backend={deconv_cfg.backend}, "
        f"ar_use_all={ar_cfg.use_all}, ar_kn_len={ar_cfg.kn_len}, "
        f"{ncell} cells with {T} timepoints"
    )

    # 1. Preprocessing
    Y = preprocess_traces(
        Y,
        med_wnd=preprocess_cfg.med_wnd,
        dff=preprocess_cfg.dff,
        ar_kn_len=ar_cfg.kn_len,
    )

    # 2. Dashboard setup
    if spawn_dashboard:
        if da_client is not None:
            logger.debug("Using Dask client for distributed computation")
            dashboard = da_client.submit(
                Dashboard, Y=Y, kn_len=ar_cfg.kn_len, actor=True
            ).result()
        else:
            logger.debug("Running in single-machine mode")
            dashboard = Dashboard(Y=Y, kn_len=ar_cfg.kn_len)
    else:
        dashboard = None

    # 3. Initialize AR parameters
    ar_params = initialize_ar_params(
        Y,
        tau_init=init_cfg.tau_init,
        p=p,
        up_factor=up_factor,
        ar_kn_len=ar_cfg.kn_len,
        est_noise_freq=init_cfg.est_noise_freq,
        est_use_smooth=init_cfg.est_use_smooth,
        est_add_lag=init_cfg.est_add_lag,
    )
    theta = ar_params.theta
    tau = ar_params.tau

    # 4. Initialize deconvolvers
    dcv = initialize_deconvolvers(
        Y,
        ar_params,
        ar_kn_len=ar_cfg.kn_len,
        up_factor=up_factor,
        nthres=deconv_cfg.nthres,
        norm=deconv_cfg.norm,
        penal=deconv_cfg.penal,
        use_base=deconv_cfg.use_base,
        err_weighting=deconv_cfg.err_weighting,
        masking_radius=deconv_cfg.masking_radius,
        pks_polish=deconv_cfg.pks_polish,
        ncons_thres=deconv_cfg.ncons_thres,
        min_rel_scl=deconv_cfg.min_rel_scl,
        atol=deconv_cfg.atol,
        backend=deconv_cfg.backend,
        dashboard=dashboard,
        da_client=da_client,
    )

    # 5. Initialize iteration state
    state = IterationState.empty(T, up_factor)
    scale = np.empty(ncell)

    # 6. Main iteration loop
    for i_iter in trange(conv_cfg.max_iters, desc="iteration"):
        logger.info(f"Starting iteration {i_iter}/{conv_cfg.max_iters}")

        # 6.1 Deconvolution step
        deconv_result = run_deconv_step(
            Y,
            dcv,
            i_iter=i_iter,
            reset_scale=deconv_cfg.reset_scale,
            da_client=da_client,
        )
        scale = deconv_result.scale

        logger.debug(
            f"Iteration {i_iter} stats - "
            f"Mean error: {deconv_result.err.mean():.4f}, "
            f"Mean scale: {scale.mean():.4f}"
        )

        # 6.2 Update metrics
        cur_metric = make_cur_metric(
            i_iter=i_iter,
            ncell=ncell,
            theta=theta,
            tau=tau,
            scale=scale,
            deconv_result=deconv_result,
            deconvolvers=dcv,
            use_rel_err=conv_cfg.use_rel_err,
        )
        update_dashboard(dashboard, cur_metric, i_iter, conv_cfg.max_iters)
        state.metric_df = append_metrics(state.metric_df, cur_metric)

        # 6.3 Save iteration results
        state.C_ls.append(deconv_result.C)
        state.S_ls.append(deconv_result.S)
        state.scal_ls.append(scale)

        # Handle h_ls / h_fit_ls (not available on first iteration)
        if i_iter == 0:
            state.h_ls.append(np.full(T * up_factor, np.nan))
            state.h_fit_ls.append(np.full(T * up_factor, np.nan))
        else:
            state.h_ls.append(h)
            state.h_fit_ls.append(h_fit)

        # 6.4 Select best spikes for AR update
        S_best, scal_best, err_wt, state.metric_df = select_best_spikes(
            state.S_ls,
            state.scal_ls,
            deconv_result.err_rel,
            state.metric_df,
            n_best=conv_cfg.n_best,
            i_iter=i_iter,
            tau_init=init_cfg.tau_init,
        )

        # 6.5 Create spike train for AR estimation
        S_ar = make_S_ar(
            S_best,
            est_nevt=init_cfg.est_nevt,
            T=T,
            up_factor=up_factor,
            ar_kn_len=ar_cfg.kn_len,
        )

        # 6.6 Update AR parameters
        tau, ps, h, h_fit = update_ar_parameters(
            Y,
            S_ar,
            scal_best,
            err_wt,
            ar_use_all=ar_cfg.use_all,
            ar_kn_len=ar_cfg.kn_len,
            ar_norm=ar_cfg.norm,
            ar_prop_best=ar_cfg.prop_best,
            up_factor=up_factor,
            p=p,
            ncell=ncell,
            dashboard=dashboard,
        )

        # Update theta to match the new tau values
        # (required for correct metric reporting in make_cur_metric)
        theta = np.array([tau2AR(t[0], t[1]) for t in tau])

        if ar_cfg.use_all:
            logger.debug(
                f"Updating AR parameters for all cells: tau={tau[0]}"
            )
        else:
            logger.debug(f"Updated AR parameters per-cell")

        # 6.7 Propagate AR update to deconvolvers
        propagate_ar_update(
            dcv,
            tau,
            scal_best,
            ar_use_all=ar_cfg.use_all,
            da_client=da_client,
        )

        # 6.8 Check convergence
        conv_result = check_convergence(
            state.metric_df,
            cur_metric,
            deconv_result.S,
            state.S_ls,
            i_iter=i_iter,
            err_atol=conv_cfg.err_atol,
            err_rtol=conv_cfg.err_rtol,
        )

        if conv_result.converged:
            if "trapped" in conv_result.reason.lower():
                logger.warning(conv_result.reason)
            else:
                logger.info(conv_result.reason)
            break
    else:
        logger.warning("Max iteration reached without convergence")

    # 7. Compute final results
    opt_C, opt_S = _finalize_results(
        state, ncell, T, up_factor, ar_cfg.use_all
    )

    # 8. Cleanup
    if dashboard is not None:
        dashboard.stop()

    logger.info("Pipeline completed successfully")

    if return_iter:
        return (
            opt_C,
            opt_S,
            state.metric_df,
            state.C_ls,
            state.S_ls,
            state.h_ls,
            state.h_fit_ls,
        )
    else:
        return opt_C, opt_S, state.metric_df


def _finalize_results(
    state: IterationState,
    ncell: int,
    T: int,
    up_factor: int,
    ar_use_all: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute final optimal results from iteration history.

    Parameters
    ----------
    state : IterationState
        Accumulated iteration state
    ncell : int
        Number of cells
    T : int
        Original trace length
    up_factor : int
        Upsampling factor
    ar_use_all : bool
        Whether using shared AR

    Returns
    -------
    opt_C : np.ndarray
        Optimal calcium traces
    opt_S : np.ndarray
        Optimal spike trains
    """
    metric_df = state.metric_df
    C_ls = state.C_ls
    S_ls = state.S_ls

    opt_C = np.empty((ncell, T * up_factor))
    opt_S = np.empty((ncell, T * up_factor))

    # mobj = metric_df.groupby("iter")["obj"].median()
    # opt_idx_all = mobj.idxmin()
    # NOTE: Original pipeline always selected the last iteration (-1),
    # regardless of metric-based selection. We preserve that behavior here.
    # (The metric-based selection logic was present but unused in the original.)
    opt_idx = -1

    for icell in range(ncell):
        opt_C[icell, :] = C_ls[opt_idx][icell, :]
        opt_S[icell, :] = S_ls[opt_idx][icell, :]

    # Append optimal to lists (matching original behavior)
    C_ls.append(opt_C)
    S_ls.append(opt_S)

    return opt_C, opt_S


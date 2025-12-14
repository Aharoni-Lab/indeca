"""Metrics construction and update functions.

Handles building the per-iteration metrics DataFrame.
"""

from typing import Any, List

import numpy as np
import pandas as pd

from indeca.core.simulation import find_dhm

from .types import DeconvStepResult


def make_cur_metric(
    i_iter: int,
    ncell: int,
    theta: np.ndarray,
    tau: np.ndarray,
    scale: np.ndarray,
    deconv_result: DeconvStepResult,
    deconvolvers: List[Any],
    use_rel_err: bool,
) -> pd.DataFrame:
    """Construct the metrics DataFrame for the current iteration.

    Parameters
    ----------
    i_iter : int
        Current iteration index
    ncell : int
        Number of cells
    theta : np.ndarray
        AR coefficients, shape (ncell, p)
    tau : np.ndarray
        Time constants, shape (ncell, 2)
    scale : np.ndarray
        Scale factors, shape (ncell,)
    deconv_result : DeconvStepResult
        Results from deconvolution step
    deconvolvers : list
        List of DeconvBin instances
    use_rel_err : bool
        Whether to use relative error for objective

    Returns
    -------
    pd.DataFrame
        Metrics for the current iteration
    """
    # Compute half-max durations
    dhm = np.stack(
        [
            np.array(find_dhm(True, (t0, t1), (s, -s))[0], dtype=float)
            for t0, t1, s in zip(tau.T[0], tau.T[1], scale)
        ],
        axis=0,
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
            "err": deconv_result.err,
            "err_rel": deconv_result.err_rel,
            "scale": scale,
            "penal": deconv_result.penal,
            "nnz": deconv_result.nnz,
            "obj": deconv_result.err_rel if use_rel_err else deconv_result.err,
            "wgt_len": [d.wgt_len for d in deconvolvers],
        }
    )

    return cur_metric


def append_metrics(
    metric_df: pd.DataFrame,
    cur_metric: pd.DataFrame,
) -> pd.DataFrame:
    """Append current iteration metrics to the accumulated DataFrame.

    Parameters
    ----------
    metric_df : pd.DataFrame
        Accumulated metrics from previous iterations
    cur_metric : pd.DataFrame
        Metrics from current iteration

    Returns
    -------
    pd.DataFrame
        Updated metrics DataFrame
    """
    return pd.concat([metric_df, cur_metric], ignore_index=True)


def update_dashboard(
    dashboard: Any,
    cur_metric: pd.DataFrame,
    i_iter: int,
    max_iters: int,
) -> None:
    """Update the dashboard with current iteration metrics.

    Parameters
    ----------
    dashboard : Dashboard or None
        Dashboard instance
    cur_metric : pd.DataFrame
        Current iteration metrics
    i_iter : int
        Current iteration index
    max_iters : int
        Maximum number of iterations
    """
    if dashboard is not None:
        dashboard.update(
            tau_d=cur_metric["tau_d"].squeeze(),
            tau_r=cur_metric["tau_r"].squeeze(),
            err=cur_metric["obj"].squeeze(),
            scale=cur_metric["scale"].squeeze(),
        )
        dashboard.set_iter(min(i_iter + 1, max_iters - 1))

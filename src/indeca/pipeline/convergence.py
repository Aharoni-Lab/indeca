"""Convergence checking functions.

Handles all convergence and trapping detection logic.
"""

from typing import List

import numpy as np
import pandas as pd

from .types import ConvergenceResult


def check_convergence(
    metric_df: pd.DataFrame,
    cur_metric: pd.DataFrame,
    S: np.ndarray,
    S_ls: List[np.ndarray],
    *,
    i_iter: int,
    err_atol: float,
    err_rtol: float,
) -> ConvergenceResult:
    """Check if the pipeline has converged or is trapped.

    Checks multiple convergence criteria:
    1. Absolute error tolerance
    2. Relative error tolerance
    3. Spike pattern stabilization
    4. Trapped in local optimum (error)
    5. Trapped in local optimum (spike pattern)

    Parameters
    ----------
    metric_df : pd.DataFrame
        Accumulated metrics from previous iterations
    cur_metric : pd.DataFrame
        Metrics from current iteration
    S : np.ndarray
        Current spike trains, shape (ncell, T * up_factor)
    S_ls : list of np.ndarray
        Spike trains from all iterations
    i_iter : int
        Current iteration index
    err_atol : float
        Absolute error tolerance
    err_rtol : float
        Relative error tolerance

    Returns
    -------
    ConvergenceResult
        Result indicating if converged and why
    """
    # Need at least one previous iteration
    metric_prev = metric_df[metric_df["iter"] < i_iter].dropna(
        subset=["obj", "scale"]
    )
    metric_last = metric_df[metric_df["iter"] == i_iter - 1].dropna(
        subset=["obj", "scale"]
    )

    if len(metric_prev) == 0:
        return ConvergenceResult(converged=False, reason="")

    err_cur = cur_metric.set_index("cell")["obj"]
    err_last = metric_last.set_index("cell")["obj"]
    err_best = metric_prev.groupby("cell")["obj"].min()
    ncell = S.shape[0]

    # Check 1: Converged by absolute error
    if (np.abs(err_cur - err_last) < err_atol).all():
        return ConvergenceResult(
            converged=True, reason="Converged: absolute error tolerance reached"
        )

    # Check 2: Converged by relative error
    if (np.abs(err_cur - err_last) < err_rtol * err_best).all():
        return ConvergenceResult(
            converged=True, reason="Converged: relative error tolerance reached"
        )

    # Check 3: Converged by spike pattern stabilization
    T_up = S.shape[1]
    S_best = np.empty((ncell, T_up))
    for uid, udf in metric_prev.groupby("cell"):
        best_iter = udf.set_index("iter")["obj"].idxmin()
        S_best[uid, :] = S_ls[best_iter][uid, :]

    if np.abs(S - S_best).sum() < 1:
        return ConvergenceResult(
            converged=True, reason="Converged: spike pattern stabilized"
        )

    # Check 4: Trapped by error (current error very close to some past error)
    err_all = metric_prev.pivot(columns="iter", index="cell", values="obj")
    diff_all = np.abs(err_cur.values.reshape((-1, 1)) - err_all.values)
    if (diff_all.min(axis=1) < err_atol).all():
        return ConvergenceResult(
            converged=True, reason="Solution trapped in local optimal err"
        )

    # Check 5: Trapped by spike pattern (current pattern matches > 1 past pattern)
    if len(S_ls) > 1:
        diff_all = np.array([np.abs(S - prev_s).sum() for prev_s in S_ls[:-1]])
        if (diff_all < 1).sum() > 1:
            return ConvergenceResult(
                converged=True, reason="Solution trapped in local optimal s"
            )

    return ConvergenceResult(converged=False, reason="")


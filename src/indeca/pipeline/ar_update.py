"""AR parameter update functions.

Handles spike selection, AR estimation, and parameter propagation.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from indeca.core.AR_kernel import updateAR
from indeca.core.deconv import construct_R

from .types import ARUpdateResult


def select_best_spikes(
    S_ls: List[np.ndarray],
    scal_ls: List[np.ndarray],
    err_rel: np.ndarray,
    metric_df: pd.DataFrame,
    *,
    n_best: Optional[int],
    i_iter: int,
    tau_init: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Select best spikes based on n_best iterations.

    Parameters
    ----------
    S_ls : list of np.ndarray
        Spike trains from all iterations
    scal_ls : list of np.ndarray
        Scale factors from all iterations
    err_rel : np.ndarray
        Relative errors from current iteration
    metric_df : pd.DataFrame
        Accumulated metrics
    n_best : int or None
        Number of best iterations to use
    i_iter : int
        Current iteration index
    tau_init : tuple or None
        Initial tau values (affects metric selection)

    Returns
    -------
    S_best : np.ndarray
        Best spike trains, shape (ncell, T * up_factor)
    scal_best : np.ndarray
        Best scale factors, shape (ncell,)
    err_wt : np.ndarray
        Error weights (negative err_rel), shape (ncell,)
    metric_df : pd.DataFrame
        Updated metric DataFrame with best_idx column
    """
    S = S_ls[-1]  # Current iteration spikes
    scale = scal_ls[-1]  # Current iteration scales

    metric_df = metric_df.set_index(["iter", "cell"])

    if n_best is not None and i_iter >= n_best:
        ncell = S.shape[0]
        S_best = np.empty_like(S)
        scal_best = np.empty_like(scale)
        err_wt = np.empty_like(err_rel)

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
            err_wt[icell] = -np.mean(
                [metric_df.loc[(i, icell), "err_rel"] for i in cur_idx]
            )
    else:
        S_best = S
        scal_best = scale
        err_wt = -err_rel

    metric_df = metric_df.reset_index()
    return S_best, scal_best, err_wt, metric_df


def make_S_ar(
    S_best: np.ndarray,
    *,
    est_nevt: Optional[int],
    T: int,
    up_factor: int,
    ar_kn_len: int,
) -> np.ndarray:
    """Create spike train for AR estimation with optional peak masking.

    Parameters
    ----------
    S_best : np.ndarray
        Best spike trains, shape (ncell, T * up_factor)
    est_nevt : int or None
        Number of top events to use. None uses all spikes.
    T : int
        Original trace length
    up_factor : int
        Upsampling factor
    ar_kn_len : int
        AR kernel length

    Returns
    -------
    np.ndarray
        Spike train for AR estimation, shape (ncell, T * up_factor)
    """
    if est_nevt is not None:
        S_ar = []
        R = construct_R(T, up_factor)

        for s in S_best:
            Rs = R @ s
            s_pks, pk_prop = find_peaks(
                Rs, height=1, distance=ar_kn_len * up_factor
            )
            pk_ht = pk_prop["peak_heights"]
            top_idx = s_pks[np.argsort(pk_ht)[-est_nevt:]]
            mask = np.zeros_like(Rs, dtype=bool)
            mask[top_idx] = True
            Rs_ma = Rs * mask
            s_ma = np.zeros_like(s)
            s_ma[::up_factor] = Rs_ma
            S_ar.append(s_ma)

        S_ar = np.stack(S_ar, axis=0)
    else:
        S_ar = S_best

    return S_ar


def update_ar_parameters(
    Y: np.ndarray,
    S_ar: np.ndarray,
    scal_best: np.ndarray,
    err_wt: np.ndarray,
    *,
    ar_use_all: bool,
    ar_kn_len: int,
    ar_norm: str,
    ar_prop_best: Optional[float],
    up_factor: int,
    p: int,
    ncell: int,
    dashboard: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update AR parameters based on current spike estimates.

    Parameters
    ----------
    Y : np.ndarray
        Input traces, shape (ncell, T)
    S_ar : np.ndarray
        Spike trains for AR estimation, shape (ncell, T * up_factor)
    scal_best : np.ndarray
        Best scale factors, shape (ncell,)
    err_wt : np.ndarray
        Error weights, shape (ncell,)
    ar_use_all : bool
        Whether to use all cells for shared AR update
    ar_kn_len : int
        AR kernel length
    ar_norm : str
        Norm for AR fitting
    ar_prop_best : float or None
        Proportion of best cells to use
    up_factor : int
        Upsampling factor
    p : int
        AR model order
    ncell : int
        Number of cells
    dashboard : Dashboard or None
        Dashboard instance

    Returns
    -------
    tau : np.ndarray
        Updated time constants, shape (ncell, 2)
    ps : np.ndarray
        Updated peak coefficients
    h : np.ndarray
        Impulse response
    h_fit : np.ndarray
        Fitted impulse response
    """
    if ar_use_all:
        # Shared AR update across cells
        if ar_prop_best is not None:
            ar_nbest = max(int(np.round(ar_prop_best * ncell)), 1)
            ar_best_idx = np.argsort(err_wt)[-ar_nbest:]
        else:
            ar_best_idx = slice(None)

        cur_tau, ps, ar_scal, h, h_fit = updateAR(
            Y[ar_best_idx],
            S_ar[ar_best_idx],
            scal_best[ar_best_idx],
            N=p,
            h_len=ar_kn_len * up_factor,
            norm=ar_norm,
            up_factor=up_factor,
        )

        if dashboard is not None:
            dashboard.update(
                h=h[: ar_kn_len * up_factor],
                h_fit=h_fit[: ar_kn_len * up_factor],
            )

        tau = np.tile(cur_tau, (ncell, 1))
    else:
        # Per-cell AR update
        tau = np.empty((ncell, p))
        
        # NOTE: Original pipeline only retained the last cell's ps/h/h_fit
        # when ar_use_all=False. We preserve this behavior explicitly.
        ps = None
        h = None
        h_fit = None

        for icell, (y, s) in enumerate(zip(Y, S_ar)):
            cur_tau, cur_ps, ar_scal, cur_h, cur_h_fit = updateAR(
                y,
                s,
                scal_best[icell],
                N=p,
                h_len=ar_kn_len,
                norm=ar_norm,
                up_factor=up_factor,
            )

            if dashboard is not None:
                dashboard.update(uid=icell, h=cur_h, h_fit=cur_h_fit)

            tau[icell, :] = cur_tau

            # Overwrite on each iteration; only last cell's values are kept
            ps = cur_ps
            h = cur_h
            h_fit = cur_h_fit

    return tau, ps, h, h_fit


def propagate_ar_update(
    deconvolvers: List[Any],
    tau: np.ndarray,
    scal_best: np.ndarray,
    *,
    ar_use_all: bool,
    da_client: Any,
) -> None:
    """Propagate AR parameter updates to deconvolvers.

    Parameters
    ----------
    deconvolvers : list
        List of DeconvBin instances
    tau : np.ndarray
        Updated time constants, shape (ncell, 2)
    scal_best : np.ndarray
        Best scale factors, shape (ncell,)
    ar_use_all : bool
        Whether using shared AR (affects which tau to use)
    da_client : Client or None
        Dask client for distributed execution
    """
    if ar_use_all:
        # All cells share the same tau (use tau[0])
        cur_tau = tau[0]
        for idx, d in enumerate(deconvolvers):
            if da_client is not None:
                da_client.submit(
                    lambda dd: dd.update(tau=cur_tau, scale=scal_best[idx]), d
                )
            else:
                d.update(tau=cur_tau, scale=scal_best[idx])
    else:
        # Per-cell tau
        for idx, d in enumerate(deconvolvers):
            if da_client is not None:
                da_client.submit(
                    lambda dd: dd.update(tau=tau[idx], scale=scal_best[idx]),
                    deconvolvers[idx],
                )
            else:
                d.update(tau=tau[idx], scale=scal_best[idx])


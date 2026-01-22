"""Initialization functions for the binary pursuit pipeline.

Handles AR parameter estimation and DeconvBin instance creation.
"""

from typing import List, Optional, Tuple, Any

import numpy as np

from indeca.core.AR_kernel import AR_upsamp_real, estimate_coefs
from indeca.core.deconv import DeconvBin
from indeca.core.simulation import AR2tau, tau2AR

from .types import ARParams


def initialize_ar_params(
    Y: np.ndarray,
    *,
    tau_init: Optional[Tuple[float, float]],
    p: int,
    up_factor: int,
    ar_kn_len: int,
    est_noise_freq: Optional[float],
    est_use_smooth: bool,
    est_add_lag: int,
) -> ARParams:
    """Initialize AR model parameters.

    If tau_init is provided, uses those values for all cells.
    Otherwise, estimates AR parameters from the data for each cell.

    Parameters
    ----------
    Y : np.ndarray
        Input traces, shape (ncell, T)
    tau_init : tuple or None
        Initial (tau_d, tau_r) values. If None, estimate from data.
    p : int
        AR model order (typically 2)
    up_factor : int
        Upsampling factor
    ar_kn_len : int
        AR kernel length for fitting
    est_noise_freq : float or None
        Noise frequency for estimation
    est_use_smooth : bool
        Whether to use smoothing during estimation
    est_add_lag : int
        Additional lag for estimation

    Returns
    -------
    ARParams
        Initialized AR parameters (theta, tau, ps)
    """
    ncell = Y.shape[0]

    if tau_init is not None:
        # Use provided tau values for all cells
        theta = tau2AR(tau_init[0], tau_init[1])
        _, _, pp = AR2tau(theta[0], theta[1], solve_amp=True)
        ps = np.array([pp, -pp])

        theta = np.tile(tau2AR(tau_init[0], tau_init[1]), (ncell, 1))
        tau = np.tile(tau_init, (ncell, 1))
        ps = np.tile(ps, (ncell, 1))
    else:
        # Estimate AR parameters from data
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

    return ARParams(theta=theta, tau=tau, ps=ps)


def initialize_deconvolvers(
    Y: np.ndarray,
    ar_params: ARParams,
    *,
    ar_kn_len: int,
    up_factor: int,
    nthres: int,
    norm: str,
    penal: Optional[str],
    use_base: bool,
    err_weighting: Optional[str],
    masking_radius: Optional[int],
    pks_polish: bool,
    ncons_thres: Optional[int],
    min_rel_scl: Optional[float],
    atol: float,
    backend: str,
    dashboard: Any,
    da_client: Any,
) -> List[Any]:
    """Create DeconvBin instances for all cells.

    Parameters
    ----------
    Y : np.ndarray
        Input traces, shape (ncell, T)
    ar_params : ARParams
        Initialized AR parameters
    ar_kn_len : int
        AR kernel length
    up_factor : int
        Upsampling factor
    nthres : int
        Number of thresholds
    norm : str
        Norm type ("l1", "l2", "huber")
    penal : str or None
        Penalty type
    use_base : bool
        Whether to use baseline
    err_weighting : str or None
        Error weighting method
    masking_radius : int or None
        Masking radius
    pks_polish : bool
        Whether to polish peaks
    ncons_thres : int or None
        Consecutive spikes threshold
    min_rel_scl : float or None
        Minimum relative scale
    atol : float
        Absolute tolerance
    backend : str
        Solver backend
    dashboard : Dashboard or None
        Dashboard instance
    da_client : Client or None
        Dask client for distributed execution

    Returns
    -------
    list
        List of DeconvBin instances (or futures if using Dask)
    """
    theta = ar_params.theta
    tau = ar_params.tau
    ps = ar_params.ps

    if da_client is not None:
        # Distributed execution
        dcv = [
            da_client.submit(
                lambda yy, th, tau_i, ps_i: DeconvBin(
                    y=yy,
                    theta=th,
                    tau=tau_i,
                    ps=ps_i,
                    coef_len=ar_kn_len,
                    upsamp=up_factor,
                    nthres=nthres,
                    norm=norm,
                    penal=penal,
                    use_base=use_base,
                    err_weighting=err_weighting,
                    masking_radius=masking_radius,
                    pks_polish=pks_polish,
                    ncons_thres=ncons_thres,
                    min_rel_scl=min_rel_scl,
                    atol=atol,
                    backend=backend,
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
        # Local execution
        dcv = [
            DeconvBin(
                y=y,
                theta=theta[i],
                tau=tau[i],
                ps=ps[i],
                coef_len=ar_kn_len,
                upsamp=up_factor,
                nthres=nthres,
                norm=norm,
                penal=penal,
                use_base=use_base,
                err_weighting=err_weighting,
                masking_radius=masking_radius,
                pks_polish=pks_polish,
                ncons_thres=ncons_thres,
                min_rel_scl=min_rel_scl,
                atol=atol,
                backend=backend,
                dashboard=dashboard,
                dashboard_uid=i,
            )
            for i, y in enumerate(Y)
        ]

    return dcv

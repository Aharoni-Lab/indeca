"""Legacy interface for the binary pursuit pipeline.

This module provides backward compatibility with the old flat-kwargs API.
New code should use the config-based API from binary_pursuit.py.

.. deprecated::
    Use `pipeline_bin` from `indeca.pipeline.binary_pursuit` with
    `DeconvPipelineConfig` instead.
"""

import warnings
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from line_profiler import profile

from indeca.utils.logging_config import get_module_logger

from .binary_pursuit import pipeline_bin as _pipeline_bin_new
from .config import DeconvPipelineConfig

logger = get_module_logger("pipeline")
logger.info("Pipeline module initialized")


@profile
def pipeline_bin(
    Y: np.ndarray,
    up_factor: int = 1,
    p: int = 2,
    tau_init: Optional[Tuple[float, float]] = None,
    return_iter: bool = False,
    max_iters: int = 50,
    n_best: Optional[int] = 3,
    use_rel_err: bool = True,
    err_atol: float = 1e-4,
    err_rtol: float = 5e-2,
    est_noise_freq: Optional[float] = None,
    est_use_smooth: bool = False,
    est_add_lag: int = 20,
    est_nevt: Optional[int] = 10,
    med_wnd: Optional[Union[int, Literal["auto"]]] = None,
    dff: bool = True,
    deconv_nthres: int = 1000,
    deconv_norm: Literal["l1", "l2", "huber"] = "l2",
    deconv_atol: float = 1e-3,
    deconv_penal: Optional[Literal["l0", "l1"]] = None,
    deconv_backend: Literal["osqp", "cvxpy", "cuosqp"] = "osqp",
    deconv_err_weighting: Optional[Literal["fft", "corr", "adaptive"]] = None,
    deconv_use_base: bool = True,
    deconv_reset_scl: bool = True,
    deconv_masking_radius: Optional[int] = None,
    deconv_pks_polish: bool = True,
    deconv_ncons_thres: Optional[Union[int, Literal["auto"]]] = None,
    deconv_min_rel_scl: Optional[Union[float, Literal["auto"]]] = None,
    ar_use_all: bool = True,
    ar_kn_len: int = 100,
    ar_norm: Literal["l1", "l2"] = "l2",
    ar_prop_best: Optional[float] = None,
    da_client=None,
    spawn_dashboard: bool = True,
) -> Union[
    Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray, pd.DataFrame, list, list, list, list],
]:
    """Binary pursuit pipeline for spike inference (legacy interface).

    .. deprecated::
        This function signature is deprecated. Use the config-based API:

        >>> from indeca.pipeline import pipeline_bin, DeconvPipelineConfig
        >>> config = DeconvPipelineConfig(up_factor=2, ...)
        >>> opt_C, opt_S, metrics = pipeline_bin(Y, config=config)

    Parameters
    ----------
    Y : array-like
        Input fluorescence trace, shape (ncell, T)
    up_factor : int
        Upsampling factor for spike times
    p : int
        AR model order
    tau_init : tuple or None
        Initial (tau_d, tau_r) values. If None, estimate from data.
    return_iter : bool
        Whether to return per-iteration results
    max_iters : int
        Maximum number of iterations
    n_best : int or None
        Number of best iterations to average for spike selection
    use_rel_err : bool
        Whether to use relative error for objective
    err_atol : float
        Absolute error tolerance for convergence
    err_rtol : float
        Relative error tolerance for convergence
    est_noise_freq : float or None
        Frequency for noise estimation
    est_use_smooth : bool
        Whether to use smoothing during AR estimation
    est_add_lag : int
        Additional lag samples for AR estimation
    est_nevt : int or None
        Number of top spike events for AR update
    med_wnd : int, "auto", or None
        Window size for median filtering
    dff : bool
        Whether to compute dF/F normalization
    deconv_nthres : int
        Number of thresholds for thresholding step
    deconv_norm : str
        Norm for data fidelity
    deconv_atol : float
        Absolute tolerance for solver
    deconv_penal : str or None
        Penalty type for sparsity
    deconv_backend : str
        Solver backend
    deconv_err_weighting : str or None
        Error weighting method
    deconv_use_base : bool
        Whether to include a baseline term
    deconv_reset_scl : bool
        Whether to reset scale at each iteration
    deconv_masking_radius : int or None
        Radius for masking around spikes
    deconv_pks_polish : bool
        Whether to polish peaks after solving
    deconv_ncons_thres : int, "auto", or None
        Max consecutive spikes threshold
    deconv_min_rel_scl : float, "auto", or None
        Minimum relative scale
    ar_use_all : bool
        Whether to use all cells for AR update (shared tau)
    ar_kn_len : int
        Kernel length for AR fitting
    ar_norm : str
        Norm for AR fitting
    ar_prop_best : float or None
        Proportion of best cells to use for AR update
    da_client : Client or None
        Dask client for distributed execution
    spawn_dashboard : bool
        Whether to spawn a real-time dashboard

    Returns
    -------
    opt_C : np.ndarray
        Optimal calcium traces
    opt_S : np.ndarray
        Optimal spike trains
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
    # Emit deprecation warning
    warnings.warn(
        "The flat-kwargs signature of pipeline_bin() is deprecated. "
        "Use the config-based API instead:\n"
        "  from indeca.pipeline import pipeline_bin, DeconvPipelineConfig\n"
        "  config = DeconvPipelineConfig.from_legacy_kwargs(...)\n"
        "  result = pipeline_bin(Y, config=config)",
        DeprecationWarning,
        stacklevel=2,
    )

    # Build config from legacy kwargs
    config = DeconvPipelineConfig.from_legacy_kwargs(
        up_factor=up_factor,
        p=p,
        tau_init=tau_init,
        max_iters=max_iters,
        n_best=n_best,
        use_rel_err=use_rel_err,
        err_atol=err_atol,
        err_rtol=err_rtol,
        est_noise_freq=est_noise_freq,
        est_use_smooth=est_use_smooth,
        est_add_lag=est_add_lag,
        est_nevt=est_nevt,
        med_wnd=med_wnd,
        dff=dff,
        deconv_nthres=deconv_nthres,
        deconv_norm=deconv_norm,
        deconv_atol=deconv_atol,
        deconv_penal=deconv_penal,
        deconv_backend=deconv_backend,
        deconv_err_weighting=deconv_err_weighting,
        deconv_use_base=deconv_use_base,
        deconv_reset_scl=deconv_reset_scl,
        deconv_masking_radius=deconv_masking_radius,
        deconv_pks_polish=deconv_pks_polish,
        deconv_ncons_thres=deconv_ncons_thres,
        deconv_min_rel_scl=deconv_min_rel_scl,
        ar_use_all=ar_use_all,
        ar_kn_len=ar_kn_len,
        ar_norm=ar_norm,
        ar_prop_best=ar_prop_best,
    )

    # Delegate to new implementation
    return _pipeline_bin_new(
        Y,
        config=config,
        da_client=da_client,
        spawn_dashboard=spawn_dashboard,
        return_iter=return_iter,
    )


# Keep legacy name available for explicit imports
pipeline_bin_legacy = pipeline_bin

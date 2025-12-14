"""Shared type definitions for the binary pursuit pipeline.

These types define the data structures passed between pipeline steps,
making the data flow explicit and typed.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class ARParams:
    """AR model parameters for all cells.

    Attributes:
        theta: AR coefficients, shape (ncell, p)
        tau: Time constants (tau_d, tau_r), shape (ncell, 2)
        ps: Peak coefficients, shape (ncell, p)
    """

    theta: np.ndarray
    tau: np.ndarray
    ps: np.ndarray


@dataclass
class DeconvStepResult:
    """Result of a single deconvolution step.

    Attributes:
        S: Spike train, shape (ncell, T * up_factor)
        C: Calcium trace, shape (ncell, T * up_factor)
        scale: Scale factors, shape (ncell,)
        err: Absolute errors, shape (ncell,)
        err_rel: Relative errors, shape (ncell,)
        nnz: Non-zero counts, shape (ncell,)
        penal: Penalty values, shape (ncell,)
    """

    S: np.ndarray
    C: np.ndarray
    scale: np.ndarray
    err: np.ndarray
    err_rel: np.ndarray
    nnz: np.ndarray
    penal: np.ndarray


@dataclass
class ARUpdateResult:
    """Result of AR parameter update step.

    Attributes:
        tau: Updated time constants, shape (ncell, 2) or (1, 2) if use_all
        ps: Updated peak coefficients
        ar_scal: AR scale factor
        h: Estimated impulse response
        h_fit: Fitted impulse response
    """

    tau: np.ndarray
    ps: np.ndarray
    ar_scal: float
    h: np.ndarray
    h_fit: np.ndarray


@dataclass
class ConvergenceResult:
    """Result of convergence check.

    Attributes:
        converged: Whether convergence criteria are met
        reason: Human-readable reason for convergence/non-convergence
    """

    converged: bool
    reason: str


@dataclass
class IterationState:
    """State accumulated across iterations.

    Attributes:
        C_ls: List of calcium traces per iteration
        S_ls: List of spike trains per iteration
        scal_ls: List of scale factors per iteration
        h_ls: List of impulse responses per iteration
        h_fit_ls: List of fitted impulse responses per iteration
        metric_df: DataFrame with per-iteration metrics
    """

    C_ls: List[np.ndarray]
    S_ls: List[np.ndarray]
    scal_ls: List[np.ndarray]
    h_ls: List[np.ndarray]
    h_fit_ls: List[np.ndarray]
    metric_df: pd.DataFrame

    @classmethod
    def empty(cls, T: int, up_factor: int) -> "IterationState":
        """Create an empty iteration state."""
        return cls(
            C_ls=[],
            S_ls=[],
            scal_ls=[],
            h_ls=[],
            h_fit_ls=[],
            metric_df=pd.DataFrame(
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
            ),
        )


@dataclass
class PipelineResult:
    """Final result of the pipeline.

    Attributes:
        opt_C: Optimal calcium traces, shape (ncell, T * up_factor)
        opt_S: Optimal spike trains, shape (ncell, T * up_factor)
        metric_df: DataFrame with all iteration metrics
        C_ls: List of calcium traces per iteration (if return_iter=True)
        S_ls: List of spike trains per iteration (if return_iter=True)
        h_ls: List of impulse responses per iteration (if return_iter=True)
        h_fit_ls: List of fitted impulse responses per iteration (if return_iter=True)
    """

    opt_C: np.ndarray
    opt_S: np.ndarray
    metric_df: pd.DataFrame
    C_ls: Optional[List[np.ndarray]] = None
    S_ls: Optional[List[np.ndarray]] = None
    h_ls: Optional[List[np.ndarray]] = None
    h_fit_ls: Optional[List[np.ndarray]] = None

"""Per-iteration deconvolution step functions.

Handles running solve_scale on all cells and collecting results.
"""

from typing import Any, List

import numpy as np
from tqdm.auto import tqdm

from .types import DeconvStepResult


def run_deconv_step(
    Y: np.ndarray,
    deconvolvers: List[Any],
    *,
    i_iter: int,
    reset_scale: bool,
    da_client: Any,
) -> DeconvStepResult:
    """Run one deconvolution iteration across all cells.

    Parameters
    ----------
    Y : np.ndarray
        Input traces, shape (ncell, T)
    deconvolvers : list
        List of DeconvBin instances (or futures)
    i_iter : int
        Current iteration index
    reset_scale : bool
        Whether to reset scale this iteration
    da_client : Client or None
        Dask client for distributed execution

    Returns
    -------
    DeconvStepResult
        Results from this deconvolution step
    """
    res = []

    for icell, _ in tqdm(enumerate(Y), total=Y.shape[0], desc="deconv", leave=False):
        if da_client is not None:
            r = da_client.submit(
                lambda d: d.solve_scale(reset_scale=i_iter <= 1 or reset_scale),
                deconvolvers[icell],
            )
        else:
            r = deconvolvers[icell].solve_scale(reset_scale=i_iter <= 1 or reset_scale)
        res.append(r)

    if da_client is not None:
        res = da_client.gather(res)

    # Unpack results
    S = np.stack([r[0].squeeze() for r in res], axis=0, dtype=float)
    C = np.stack([r[1].squeeze() for r in res], axis=0)
    scale = np.array([r[2] for r in res])
    err = np.array([r[3] for r in res])
    err_rel = np.array([r[4] for r in res])
    nnz = np.array([r[5] for r in res])
    penal = np.array([r[6] for r in res])

    return DeconvStepResult(
        S=S,
        C=C,
        scale=scale,
        err=err,
        err_rel=err_rel,
        nnz=nnz,
        penal=penal,
    )

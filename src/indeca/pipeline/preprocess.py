"""Preprocessing functions for the binary pursuit pipeline.

These are pure functions that transform input traces before deconvolution.
"""

from typing import Optional, Union, Literal

import numpy as np
from scipy.signal import medfilt

from indeca.utils.utils import compute_dff


def preprocess_traces(
    Y: np.ndarray,
    *,
    med_wnd: Optional[Union[int, Literal["auto"]]] = None,
    dff: bool = True,
    ar_kn_len: int = 100,
) -> np.ndarray:
    """Preprocess fluorescence traces.

    This function applies median filtering and/or dF/F normalization
    to the input traces. The input array is modified in place for
    efficiency (matching the original pipeline behavior).

    Parameters
    ----------
    Y : np.ndarray
        Input fluorescence traces, shape (ncell, T)
    med_wnd : int, "auto", or None
        Window size for median filtering. If "auto", uses ar_kn_len.
        If None, skips median filtering.
    dff : bool
        Whether to apply dF/F normalization.
    ar_kn_len : int
        AR kernel length, used for window sizing.

    Returns
    -------
    np.ndarray
        Preprocessed traces, shape (ncell, T).
        Note: This may be the same array as Y (modified in place).
    """
    # Median filtering
    if med_wnd is not None:
        actual_wnd = ar_kn_len if med_wnd == "auto" else med_wnd
        for iy, y in enumerate(Y):
            Y[iy, :] = y - medfilt(y, actual_wnd * 2 + 1)

    # dF/F normalization
    if dff:
        for iy, y in enumerate(Y):
            Y[iy, :] = compute_dff(y, window_size=ar_kn_len * 5, q=0.2)

    return Y


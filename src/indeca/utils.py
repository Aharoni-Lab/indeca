"""
Utility functions for signal processing and array manipulation.

This module provides helper functions for normalization, scaling, least squares
fitting, and fluorescence signal preprocessing used throughout the InDeCa package.
"""

import itertools as itt
from typing import Generator, Iterable, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def norm(a: ArrayLike) -> NDArray:
    """
    Normalize an array to the range [0, 1] using min-max normalization.

    Handles the case where all values are equal by returning zeros.

    Parameters
    ----------
    a : ArrayLike
        Input array to normalize. Can contain NaN values.

    Returns
    -------
    NDArray
        Normalized array with values in [0, 1]. If all input values are
        equal, returns an array of zeros with the same shape.

    Examples
    --------
    >>> norm(np.array([1, 2, 3, 4, 5]))
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    amin, amax = np.nanmin(a), np.nanmax(a)
    diff = amax - amin
    if diff > 0:
        return (a - amin) / diff
    else:
        return a - amin


def scal_lstsq(a: NDArray, b: NDArray, fit_intercept: bool = False) -> NDArray:
    """
    Solve a least squares scaling problem to find coefficients.

    Finds coefficients that minimize ||a @ coef - b||_2.

    Parameters
    ----------
    a : NDArray
        Design matrix of shape (n_samples,) or (n_samples, n_features).
        If 1D, will be reshaped to (n_samples, 1).
    b : NDArray
        Target vector of shape (n_samples,) or (n_samples, 1).
    fit_intercept : bool, default=False
        If True, adds a column of ones to ``a`` to fit an intercept term.

    Returns
    -------
    NDArray
        Solution coefficients. Shape is (n_features,) or (n_features + 1,)
        if ``fit_intercept=True``, where the last element is the intercept.

    Examples
    --------
    >>> a = np.array([1, 2, 3, 4])
    >>> b = np.array([2, 4, 6, 8])
    >>> scal_lstsq(a, b)
    array([2.])
    """
    if a.ndim == 1:
        a = a.reshape((-1, 1))
    if fit_intercept:
        a = np.concatenate([a, np.ones_like(a)], axis=1)
    return np.linalg.lstsq(a, b.squeeze(), rcond=None)[0]


def scal_like(src: NDArray, tgt: NDArray, zero_center: bool = True) -> NDArray:
    """
    Scale the source array to match the range of the target array.

    Parameters
    ----------
    src : NDArray
        Source array to be scaled.
    tgt : NDArray
        Target array whose range is used for scaling.
    zero_center : bool, default=True
        If True, scales only by the range ratio (preserves zero).
        If False, also shifts to match the target's minimum.

    Returns
    -------
    NDArray
        Scaled array with the same shape as ``src``.

    Examples
    --------
    >>> src = np.array([0, 1, 2])
    >>> tgt = np.array([0, 10, 20])
    >>> scal_like(src, tgt, zero_center=True)
    array([ 0., 10., 20.])
    """
    smin, smax = np.nanmin(src), np.nanmax(src)
    tmin, tmax = np.nanmin(tgt), np.nanmax(tgt)
    if zero_center:
        return src / (smax - smin) * (tmax - tmin)
    else:
        return (src - smin) / (smax - smin) * (tmax - tmin) + tmin


def enumerated_product(
    *args: Iterable,
) -> Generator[Tuple[Tuple[int, ...], Tuple], None, None]:
    """
    Generate the Cartesian product of iterables with their indices.

    Yields tuples containing both the indices and values from the
    Cartesian product of the input iterables.

    Parameters
    ----------
    *args : Iterable
        Variable number of iterables to compute the product over.

    Yields
    ------
    Tuple[Tuple[int, ...], Tuple]
        A tuple of (indices, values) where indices is a tuple of integer
        indices into each input iterable, and values is a tuple of the
        corresponding elements.

    Examples
    --------
    >>> list(enumerated_product(['a', 'b'], [1, 2]))
    [((0, 0), ('a', 1)), ((0, 1), ('a', 2)), ((1, 0), ('b', 1)), ((1, 1), ('b', 2))]
    """
    yield from zip(itt.product(*(range(len(x)) for x in args)), itt.product(*args))


def compute_dff(s: ArrayLike, window_size: int = 100, q: float = 0.10) -> NDArray:
    """
    Compute ΔF/F₀ (change in fluorescence) for a calcium signal.

    Estimates baseline fluorescence F₀ using a rolling quantile and computes
    the difference from the signal. This is a common preprocessing step for
    calcium imaging data to remove baseline drift.

    Parameters
    ----------
    s : ArrayLike
        Raw fluorescence signal, shape (n_timepoints,).
    window_size : int, default=100
        Size of the rolling window for baseline estimation in frames.
    q : float, default=0.10
        Quantile to use for baseline estimation (0 to 1). Lower values
        are more robust to transient calcium events.

    Returns
    -------
    NDArray
        Baseline-subtracted fluorescence signal (F - F₀), same shape as input.

    Notes
    -----
    This implementation returns F - F₀ rather than (F - F₀) / F₀ to avoid
    division by small baseline values which can amplify noise.
    """
    s = pd.Series(s).astype(float)
    f0 = s.rolling(window=window_size, min_periods=1).quantile(q)
    # dff = (s - f0) / f0
    dff = s - f0
    return dff.to_numpy()

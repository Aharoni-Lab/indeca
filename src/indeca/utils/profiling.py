"""Yappi profiling utilities for pipeline-level performance analysis.

This module provides a minimal context manager for yappi profiling,
outputting pstat-format files compatible with snakeviz visualization.

Usage
-----
>>> from indeca.utils.profiling import yappi_profile
>>> from indeca.pipeline import pipeline_bin, DeconvPipelineConfig
>>>
>>> with yappi_profile("pipeline.prof"):
...     pipeline_bin(Y, config=config, spawn_dashboard=False)
>>>
>>> # View results: snakeviz pipeline.prof
"""

from contextlib import contextmanager
from typing import Generator

import yappi


@contextmanager
def yappi_profile(outfile: str, clock: str = "wall") -> Generator[None, None, None]:
    """Context manager for yappi profiling.

    Wraps code execution with yappi profiling and saves results
    in pstat format for visualization with snakeviz or other tools.

    Parameters
    ----------
    outfile : str
        Output filename for profile stats (e.g., "pipeline.prof").
        The file will be saved in pstat format.
    clock : str, optional
        Clock type for timing measurements:
        - "wall": Wall-clock time (real elapsed time, includes I/O and waiting)
        - "cpu": CPU time (actual computation time, excludes I/O and waiting)
        Default is "wall".

    Yields
    ------
    None
        The context manager yields nothing; profiling is automatic.

    Examples
    --------
    Basic usage with wall-clock timing:

    >>> with yappi_profile("pipeline.prof"):
    ...     result = expensive_function()

    Using CPU time instead:

    >>> with yappi_profile("cpu_profile.prof", clock="cpu"):
    ...     result = expensive_function()

    Notes
    -----
    View the resulting profile with:
        snakeviz pipeline.prof

    The pstat format is compatible with Python's standard pstats module
    and various visualization tools like snakeviz, gprof2dot, and pyprof2calltree.
    """
    yappi.set_clock_type(clock)
    yappi.clear_stats()
    yappi.start()
    try:
        yield
    finally:
        yappi.stop()
        yappi.get_func_stats().save(outfile, type="pstat")


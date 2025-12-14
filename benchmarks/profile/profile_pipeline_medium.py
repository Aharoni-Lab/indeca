#!/usr/bin/env python
"""Medium deterministic benchmark for pipeline profiling.

Configuration: 50 cells x 5,000 frames
Purpose: Realistic workload testing and performance regression detection

Usage:
    # Quick runtime check
    python benchmarks/profile/profile_pipeline_medium.py

    # With yappi profiling (wall-clock time)
    python benchmarks/profile/profile_pipeline_medium.py --profile

    # With yappi profiling (CPU time)
    python benchmarks/profile/profile_pipeline_medium.py --profile --clock cpu

    # View results
    snakeviz benchmarks/profile/output/profile_pipeline_medium.prof
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indeca.core.simulation import ar_trace
from indeca.pipeline import DeconvPipelineConfig, pipeline_bin_new

# Benchmark parameters
NCELL = 50
T = 5000
SEED = 42
TAU_D = 6.0
TAU_R = 1.0
SIGNAL_LEVEL = (1.0, 5.0)
NOISE_STD = 1.0
MAX_ITERS = 15

# Markov transition matrix for spike generation (P[from_state, to_state])
MARKOV_P = np.array([[0.95, 0.05], [0.8, 0.2]])

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"


def make_test_data(ncell: int, T: int, seed: int) -> np.ndarray:
    """Generate deterministic synthetic calcium imaging data.

    Parameters
    ----------
    ncell : int
        Number of cells
    T : int
        Number of time frames
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Y : np.ndarray
        Noisy calcium traces, shape (ncell, T)
    """
    rng = np.random.default_rng(seed)

    # Generate signal levels for each cell
    sig_levels = np.sort(rng.uniform(SIGNAL_LEVEL[0], SIGNAL_LEVEL[1], size=ncell))

    # Generate traces
    Y = np.zeros((ncell, T))
    for i in range(ncell):
        C, S = ar_trace(T, MARKOV_P, tau_d=TAU_D, tau_r=TAU_R, rng=rng)
        noise = rng.normal(0, NOISE_STD, size=T)
        Y[i] = C * sig_levels[i] + noise

    return Y


def get_config() -> DeconvPipelineConfig:
    """Get fixed pipeline configuration for benchmarking."""
    return DeconvPipelineConfig.from_legacy_kwargs(
        up_factor=1,
        max_iters=MAX_ITERS,
        ar_use_all=True,
        est_noise_freq=0.06,
        est_use_smooth=True,
        est_add_lag=50,
        deconv_norm="l2",
        deconv_backend="osqp",
    )


def run_benchmark(profile: bool = False, clock: str = "wall") -> float:
    """Run the benchmark.

    Parameters
    ----------
    profile : bool
        Whether to enable yappi profiling
    clock : str
        Clock type for yappi: "wall" or "cpu"

    Returns
    -------
    elapsed : float
        Elapsed time in seconds
    """
    # Generate data
    print(f"Generating test data: {NCELL} cells x {T} frames (seed={SEED})")
    Y = make_test_data(NCELL, T, SEED)
    config = get_config()

    print(f"Running pipeline (max_iters={MAX_ITERS})...")

    if profile:
        from indeca.utils.profiling import yappi_profile

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        prof_path = OUTPUT_DIR / "profile_pipeline_medium.prof"
        print(f"Profiling enabled (clock={clock})")
        print(f"Profile output: {prof_path}")

        t0 = time.perf_counter()
        with yappi_profile(str(prof_path), clock=clock):
            C, S, metrics = pipeline_bin_new(
                Y, config=config, spawn_dashboard=False, da_client=None
            )
        elapsed = time.perf_counter() - t0

        print(f"\nView profile with: snakeviz {prof_path}")
    else:
        t0 = time.perf_counter()
        C, S, metrics = pipeline_bin_new(
            Y, config=config, spawn_dashboard=False, da_client=None
        )
        elapsed = time.perf_counter() - t0

    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Medium benchmark for pipeline profiling"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable yappi profiling",
    )
    parser.add_argument(
        "--clock",
        choices=["wall", "cpu"],
        default="wall",
        help="Clock type for profiling (default: wall)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Pipeline Benchmark: MEDIUM")
    print(f"  Cells: {NCELL}")
    print(f"  Frames: {T}")
    print(f"  Max iterations: {MAX_ITERS}")
    print("=" * 60)

    elapsed = run_benchmark(profile=args.profile, clock=args.clock)

    print("=" * 60)
    print(f"Total runtime: {elapsed:.3f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()


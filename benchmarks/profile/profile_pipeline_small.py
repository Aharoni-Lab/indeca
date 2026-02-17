#!/usr/bin/env python
"""Small deterministic benchmark for pipeline profiling.

Configuration: 10 cells x 1,000 frames
Purpose: Quick runtime checks and fast profiling iterations

Usage:
    # Quick runtime check
    python benchmarks/profile/profile_pipeline_small.py

    # With yappi profiling (wall-clock time)
    python benchmarks/profile/profile_pipeline_small.py --profile

    # With yappi profiling (CPU time)
    python benchmarks/profile/profile_pipeline_small.py --profile --clock cpu

    # View results
    snakeviz benchmarks/profile/output/profile_pipeline_small.prof
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indeca.core.simulation import ar_trace, tau2AR
from indeca.pipeline import DeconvPipelineConfig, pipeline_bin_new

# Benchmark parameters
NCELL = 10
T = 1000
SEED = 42

# Set global random seed for full reproducibility
# This ensures all random operations (including those using np.random.* directly)
# are deterministic across runs
np.random.seed(SEED)
TAU_D = 6.0
TAU_R = 1.0
SIGNAL_LEVEL = (1.0, 5.0)
NOISE_STD = 1.0
MAX_ITERS = 10

# Markov transition matrix for spike generation (P[from_state, to_state])
# State 0 = no spike, State 1 = spike
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


def plot_pipeline_results(
    Y: np.ndarray, C: np.ndarray, S: np.ndarray, output_dir: Path
) -> None:
    """Plot and save pipeline results visualization.
    
    For each cell, overlays:
    - Y: Input fluorescence trace (test data)
    - C: Deconvolved calcium trace
    - S: Inferred spike train
    
    Parameters
    ----------
    Y : np.ndarray
        Input fluorescence traces, shape (ncell, T)
    C : np.ndarray
        Deconvolved calcium traces, shape (ncell, T * up_factor)
    S : np.ndarray
        Inferred spike trains, shape (ncell, T * up_factor)
    output_dir : Path
        Directory to save the plot
    """
    ncell, T = Y.shape
    T_up = C.shape[1]
    
    # Create figure with subplots (one row per cell)
    fig, axes = plt.subplots(ncell, 1, figsize=(14, 2.5 * ncell), sharex=True)
    if ncell == 1:
        axes = [axes]
    
    time_axis = np.arange(T)
    time_axis_up = np.arange(T_up)
    
    for i in range(ncell):
        ax = axes[i]
        
        # Normalize traces for better visualization (optional scaling)
        y_norm = Y[i]
        c_norm = C[i]
        s_norm = S[i]
        
        # Plot input fluorescence (Y) - use original time axis
        ax.plot(
            time_axis,
            y_norm,
            linewidth=1.2,
            alpha=0.7,
            color="blue",
            label="Y (input)",
            zorder=1,
        )
        
        # Plot deconvolved calcium (C) - use upsampled time axis
        ax.plot(
            time_axis_up,
            c_norm,
            linewidth=1.0,
            alpha=0.8,
            color="green",
            label="C (calcium)",
            zorder=2,
        )
        
        # Plot inferred spikes (S) - use upsampled time axis
        # Scale spikes for visibility (multiply by max of Y or C for relative scaling)
        scale_factor = max(y_norm.max(), c_norm.max()) * 0.3
        spike_times = np.where(S[i] > 0.1)[0]  # Threshold for visualization
        if len(spike_times) > 0:
            spike_heights = S[i][spike_times] * scale_factor
            ax.vlines(
                spike_times,
                0,
                spike_heights,
                colors="red",
                linewidths=2.0,
                alpha=0.9,
                label="S (spikes)",
                zorder=3,
            )
        # Also plot spike trace as line for continuity
        ax.plot(
            time_axis_up,
            S[i] * scale_factor,
            linewidth=0.8,
            alpha=0.5,
            color="red",
            linestyle="--",
            zorder=2,
        )
        
        ax.set_ylabel(f"Cell {i+1}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"Cell {i+1}: Input (Y), Calcium (C), and Spikes (S)", fontsize=11)
    
    axes[-1].set_xlabel("Time (frames)", fontsize=10)
    fig.suptitle(
        f"Pipeline Results: {ncell} cells × {T} frames (upsampled to {T_up})",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "pipeline_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved pipeline results plot: {plot_path}")


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
        # Provide initial tau values matching the data generation to avoid AR fitting issues
        tau_init=(TAU_D, TAU_R),
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
    # Reset random seed for reproducibility
    # This ensures deterministic behavior across runs
    np.random.seed(SEED)
    
    # Generate data
    print(f"Generating test data: {NCELL} cells x {T} frames (seed={SEED})")
    Y = make_test_data(NCELL, T, SEED)
    
    config = get_config()

    print(f"Running pipeline (max_iters={MAX_ITERS})...")

    if profile:
        from indeca.utils.profiling import yappi_profile

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        prof_path = OUTPUT_DIR / "profile_pipeline_small.prof"
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

    # Plot and save results
    print("Plotting results...")
    plot_pipeline_results(Y, C, S, OUTPUT_DIR)

    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Small benchmark for pipeline profiling"
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
    print("Pipeline Benchmark: SMALL")
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


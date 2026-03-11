#!/usr/bin/env python3
"""Test runner for the binary pursuit deconvolution pipeline.

Usage::

    python -m indeca.pipeline.test_pipeline
    python -m indeca.pipeline.test_pipeline --config configs/benchmark.json
    python -m indeca.pipeline.test_pipeline --ar_free_kernel --ar_smooth_penalty 0.001
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# Synthetic Data
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    ncell=3, T=3000, tau_d=8.0, tau_r=0.5,
    noise_std=0.05, spike_rate=0.02, amplitude=1.0,
    baseline=0.5, seed=42,
):
    from indeca.core.simulation import tau2AR, apply_arcoef
    rng = np.random.default_rng(seed)
    g = np.array(tau2AR(tau_d, tau_r))
    S_true = np.zeros((ncell, T))
    C_true = np.zeros((ncell, T))
    Y = np.zeros((ncell, T))
    for i in range(ncell):
        spikes = rng.random(T) < spike_rate
        spikes[:5] = False
        S_true[i, :] = spikes.astype(float) * amplitude
        C_true[i, :] = apply_arcoef(S_true[i, :], g, shifted=True)
        Y[i, :] = baseline + C_true[i, :] + rng.normal(0, noise_std, T)
    return Y, C_true, S_true, {"tau_d": tau_d, "tau_r": tau_r}


# ═══════════════════════════════════════════════════════════════════════
# Real Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_data_from_file(path, data_key="Y", scale=100.0):
    path = str(path)
    S_true = None
    if path.endswith(".npy"):
        Y = np.load(path)
        if Y.ndim == 1: Y = Y.reshape(1, -1)
    elif path.endswith(".npz"):
        data = np.load(path)
        Y = data.get(data_key)
        if Y is None:
            for k in ["Y", "traces", "F", "fluorescence", "data"]:
                if k in data: Y = data[k]; break
            else: raise KeyError(f"Key '{data_key}' not found. Available: {list(data.keys())}")
        for k in ["S_true", "S", "spikes"]:
            if k in data: S_true = data[k]; break
        if Y.ndim == 1: Y = Y.reshape(1, -1)
    elif path.endswith(".nc"):
        import xarray as xr
        ds = xr.open_dataset(path)
        Y = ds[data_key] if data_key in ds else None
        if Y is None: raise KeyError(f"Key '{data_key}' not found. Available: {list(ds.data_vars)}")
        Y = Y.dropna("frame") if "frame" in Y.dims else Y
        for k in ["S_true", "S", "spikes"]:
            if k in ds: S_true = np.array(ds[k]); break
        Y = np.array(Y)
        if Y.ndim == 1: Y = Y.reshape(1, -1)
    else:
        raise ValueError(f"Unsupported: {path}")
    return Y.astype(float) * scale, S_true, {"path": path, "shape": Y.shape}


def load_benchmark_dataset(dsname, local_path="./data/realds/", project_root=None):
    from pathlib import Path as P
    if project_root is None:
        candidate = P(__file__).resolve().parent
        for _ in range(10):
            if (candidate / "tests").is_dir(): break
            candidate = candidate.parent
        else: raise FileNotFoundError("Cannot find tests/ directory")
        project_root = candidate
    pr = str(P(project_root).resolve())
    if pr not in sys.path: sys.path.insert(0, pr)
    from tests.testing_utils.io import download_realds, load_gt_ds
    ds_path = os.path.join(local_path, dsname)
    if not os.path.exists(ds_path) or not os.listdir(ds_path):
        print(f"  Downloading: {dsname}"); download_realds(local_path, dsname)
    Y, S_true, _, _ = load_gt_ds(ds_path)
    Y, S_true = Y.dropna("frame"), S_true.dropna("frame")
    act = S_true.max("frame") > 0
    Y, S_true = Y.sel(unit_id=act), S_true.sel(unit_id=act)
    return np.array(Y) * 100, np.array(S_true), {"dataset": dsname, "n_active": int(act.sum().item())}


# ═══════════════════════════════════════════════════════════════════════
# Spike Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_spike_metrics(S_true, S_inferred, tolerance=3):
    ncell, T = S_true.shape
    T_up = S_inferred.shape[1]
    up = max(T_up // T, 1)
    metrics = []
    for i in range(ncell):
        true_idx = set(np.where(S_true[i] > 0)[0])
        inf_idx = set((np.where(S_inferred[i] > 0)[0] / up).astype(int))
        tp, matched = 0, set()
        for it in sorted(inf_idx):
            for tt in true_idx - matched:
                if abs(it - tt) <= tolerance: tp += 1; matched.add(tt); break
        fp, fn = len(inf_idx) - tp, len(true_idx) - tp
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-10)
        metrics.append({"cell": i, "true": len(true_idx), "inf": len(inf_idx),
                         "tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1})
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Static Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_results(Y, S_true, result, save_dir=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ncell, T = Y.shape
    up = max(result.opt_S.shape[1] // T, 1)
    fig, axes = plt.subplots(ncell + 1, 1, figsize=(18, 3 * (ncell + 1)))
    if ncell == 0: return
    if not hasattr(axes, '__len__'): axes = [axes]
    for i in range(ncell):
        ax = axes[i]
        ax.plot(np.arange(T), Y[i], color="gray", alpha=0.4, lw=0.5, label="Raw")
        idx = np.where(result.opt_S[i] > 0)[0]
        ax.scatter(idx / up, np.full(len(idx), Y[i].max() * 1.05),
                   marker="|", color="red", s=20, alpha=0.7, label="Inferred")
        if S_true is not None:
            tidx = np.where(S_true[i] > 0)[0]
            ax.scatter(tidx, np.full(len(tidx), Y[i].max() * 1.1),
                       marker="|", color="blue", s=20, alpha=0.7, label="True")
        ax.set_ylabel(f"Cell {i}"); ax.set_xlim(0, T)
        if i == 0: ax.legend(fontsize=7, loc="upper right")
    ax = axes[-1]
    for c in result.metric_df["cell"].unique():
        d = result.metric_df[result.metric_df["cell"] == c]
        ax.plot(d["iter"], d["obj"], marker="o", ms=3, label=f"Cell {c}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Objective")
    ax.set_title(f"Convergence ({result.convergence_reason})"); ax.legend(fontsize=7)
    plt.tight_layout()
    out = Path(save_dir) / "pipeline_results.png" if save_dir else Path("pipeline_results.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"  Saved: {out}"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test the deconvolution pipeline")

    # Config file
    parser.add_argument("--config", type=str, default=None, help="JSON config file")

    # Data source
    dg = parser.add_mutually_exclusive_group()
    dg.add_argument("--data", type=str, default=None)
    dg.add_argument("--dataset", type=str, default=None)

    # Data options
    parser.add_argument("--data_key", type=str, default="Y")
    parser.add_argument("--data_scale", type=float, default=100.0)
    parser.add_argument("--data_path", type=str, default="./data/realds/")

    # Synthetic
    parser.add_argument("--ncell", type=int, default=3)
    parser.add_argument("--T", type=int, default=3000)
    parser.add_argument("--tau_d", type=float, default=8.0)
    parser.add_argument("--tau_r", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    # Pipeline
    parser.add_argument("--max_iters", type=int, default=5)
    parser.add_argument("--up_factor", type=int, default=1)
    parser.add_argument("--backend", type=str, default="osqp")
    parser.add_argument("--ar_kn_len", type=int, default=None)
    parser.add_argument("--est_noise_freq", type=float, default=None)
    parser.add_argument("--est_use_smooth", action="store_true")
    parser.add_argument("--est_add_lag", type=int, default=20)
    parser.add_argument("--min_rel_scl", type=str, default=None)


    # Deconvolution
    parser.add_argument("--penal", type=str, default=None)
    parser.add_argument("--err_weighting", type=str, default=None)
    parser.add_argument("--ncons_thres", type=str, default=None)

    # AR update mode
    parser.add_argument("--ar_free_kernel", action="store_true",
                        help="Free-form kernel estimation (mirrors Rust kernel_est.rs)")
    parser.add_argument("--ar_smooth_penalty", type=float, default=0.0,
                        help="TV smoothness penalty for free kernel (Rust smooth_lambda)")

    # Chunking & Dask
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--dask", action="store_true")
    parser.add_argument("--dask_workers", type=int, default=8)

    # Dashboard
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--dashboard_port", type=int, default=54321)

    # Output
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save", type=str, default=None)

    args = parser.parse_args()

    # ── Load config file ────────────────────────────────────────────
    if args.config is not None:
        with open(args.config) as f:
            file_args = json.load(f)
        for key, value in file_args.items():
            if not any(f"--{key}" in a for a in sys.argv[1:]):
                setattr(args, key, value)

    print("=" * 60)
    print("Binary Pursuit Pipeline — Test Runner")
    print("=" * 60)

    # ── Load data ───────────────────────────────────────────────────
    S_true = None
    if args.data is not None:
        print(f"\n1. Loading: {args.data}")
        Y, S_true, _ = load_data_from_file(args.data, args.data_key, args.data_scale)
        tau_init = (args.tau_d, args.tau_r)
    elif args.dataset is not None:
        print(f"\n1. Loading dataset: {args.dataset}")
        Y, S_true, meta = load_benchmark_dataset(args.dataset, args.data_path)
        print(f"   Shape: {Y.shape}, active: {meta['n_active']}")
        tau_init = None
    else:
        print("\n1. Generating synthetic data...")
        Y, _, S_true, _ = generate_synthetic_data(
            args.ncell, args.T, args.tau_d, args.tau_r, args.noise, seed=args.seed)
        print(f"   Shape: {Y.shape}, spikes: {int(S_true.sum())}")
        tau_init = (args.tau_d, args.tau_r)

    ncell, T = Y.shape

    # ── Configure pipeline ──────────────────────────────────────────
    print("\n2. Configuring pipeline...")
    from .pipeline import PipelineConfig, run_pipeline, run_pipeline_chunked

    ar_kn_len = args.ar_kn_len or min(100, T // 10)
    ncons_thres = args.ncons_thres
    if ncons_thres is not None and ncons_thres != "auto":
        ncons_thres = int(ncons_thres)

    config = PipelineConfig(
        up_factor=args.up_factor, tau_init=tau_init,
        max_iters=args.max_iters, backend=args.backend, ar_kn_len=ar_kn_len,
        dff=True, use_base=True, pks_polish=True,
        n_best=min(3, args.max_iters),
        est_noise_freq=args.est_noise_freq,
        est_use_smooth=args.est_use_smooth,
        est_add_lag=args.est_add_lag,
        penal=args.penal, err_weighting=args.err_weighting,
        ncons_thres=ncons_thres,
        ar_free_kernel=args.ar_free_kernel,
        ar_smooth_penalty=args.ar_smooth_penalty,
    )
    mode = "free-kernel" if config.ar_free_kernel else "parametric"
    print(f"   backend={config.backend}, ar_kn_len={ar_kn_len}, "
          f"ar_mode={mode}, tau_init={tau_init}")

    # ── Dask ────────────────────────────────────────────────────────
    da_client = None
    if args.dask:
        print("\n   Starting Dask cluster...")
        import dask; from dask.distributed import Client, LocalCluster
        dask.config.set({"distributed.scheduler.work-stealing": False,
                         "distributed.scheduler.worker-ttl": None})
        cluster = LocalCluster(n_workers=args.dask_workers, threads_per_worker=1, processes=True)
        da_client = Client(cluster)
        print(f"   Dask: {da_client.dashboard_link} ({args.dask_workers} workers)")

    # ── Dashboard ───────────────────────────────────────────────────
    dashboard = None
    if args.dashboard:
        try:
            from indeca.dashboard import Dashboard
            dashboard = Dashboard(Y=Y, kn_len=ar_kn_len, max_iters=args.max_iters,
                                  port=args.dashboard_port)
            print(f"   Dashboard: http://localhost:{args.dashboard_port}")
        except ImportError as e:
            print(f"   Dashboard unavailable: {e}")

    # ── Run ─────────────────────────────────────────────────────────
    print("\n3. Running pipeline...")
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.chunk_size is not None:
            result = run_pipeline_chunked(Y, config, args.chunk_size,
                                          dashboard=dashboard, da_client=da_client, verbose=True)
        else:
            result = run_pipeline(Y, config, dashboard=dashboard,
                                  da_client=da_client, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"\n   Elapsed: {elapsed:.1f}s | Converged: {result.converged} "
          f"({result.convergence_reason}) | Iters: {result.n_iters} | "
          f"Spikes: {result.opt_S.sum():.0f}")

    # ── Evaluate ────────────────────────────────────────────────────
    if S_true is not None:
        print("\n4. Metrics:")
        for m in compute_spike_metrics(S_true, result.opt_S):
            print(f"   Cell {m['cell']}: F1={m['F1']:.2f} P={m['P']:.2f} R={m['R']:.2f}")

    # ── Save ────────────────────────────────────────────────────────
    if args.save:
        sd = Path(args.save); sd.mkdir(parents=True, exist_ok=True)
        d = {"opt_C": result.opt_C, "opt_S": result.opt_S, "Y": Y}
        if S_true is not None: d["S_true"] = S_true
        np.savez(sd / "results.npz", **d)
        result.metric_df.to_csv(sd / "metrics.csv", index=False)
        print(f"\n   Saved to {sd}")

    # ── Plot ────────────────────────────────────────────────────────
    if not args.no_plot:
        try: plot_results(Y, S_true, result, save_dir=args.save)
        except ImportError: print("   matplotlib missing")

    # ── Cleanup ─────────────────────────────────────────────────────
    if da_client: da_client.close()
    if dashboard:
        try: dashboard.stop()
        except: pass

    print("\n" + "=" * 60 + "\nDone.\n" + "=" * 60)
    return result, compute_spike_metrics(S_true, result.opt_S) if S_true is not None else None


if __name__ == "__main__":
    result, metrics = main()

import os

import dask as da
import numpy as np
import pandas as pd
import logging
import logging.handlers
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from dask.distributed import Client, LocalCluster
from plotly.subplots import make_subplots
from routine.io import download_realds, load_gt_ds
from routine.utils import compute_ROC
from pathlib import Path

from minian_bin.deconv import construct_R
from minian_bin.pipeline import pipeline_bin
from minian_bin import set_package_log_level

LOCAL_DS_PATH = "./data/realds/"
DS_LS = ["X-DS09-GCaMP6f-m-V1"]
INT_PATH = "./intermediate/benchmark_realds"
FIG_PATH = "./figs/benchmark_realds"
PARAM_MAX_ITERS = 20
PARAM_UP_FAC = 1
PARAM_KN_LEN = 400

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# Configure logging for the benchmark script
def setup_benchmark_logging(level=logging.INFO):
    """Set up logging for the benchmark script."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure the benchmark logger
    logger = logging.getLogger("benchmark")

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "benchmark.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set formatter for handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Set levels for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Set up logging
logger = setup_benchmark_logging(logging.INFO)
# Set minian_bin package logging level
set_package_log_level(logging.DEBUG)  # Set to DEBUG to see all logging messages

logger.info("Starting benchmark script")
logger.debug(f"Parameters: MAX_ITERS={PARAM_MAX_ITERS}, UP_FAC={PARAM_UP_FAC}, KN_LEN={PARAM_KN_LEN}")

da.config.set(
    {
        "distributed.scheduler.work-stealing": False,
        "distributed.scheduler.worker-ttl": None,
    }
)  # avoid pickling error

for dsname in DS_LS:
    if not os.path.exists(os.path.join(LOCAL_DS_PATH, dsname)) or not os.listdir(
        os.path.join(LOCAL_DS_PATH, dsname)
    ):
        logger.info(f"Downloading dataset: {dsname}")
        download_realds(LOCAL_DS_PATH, dsname)
        logger.info(f"Download completed for {dsname}")

if __name__ == "__main__":
    logger.info("Initializing Dask cluster")
    cluster = LocalCluster(
        n_workers=16,
        threads_per_worker=1,
        processes=True,
        dashboard_address="0.0.0.0:12345",
    )
    client = Client(cluster)
    logger.info(f"Dask cluster initialized")

    # Set parameters for subsetting data
    frame_subset = slice(0, 6000)  # Take first 6000 frames
    n_traces = 20  # Number of traces/units to keep
    logger.debug(f"Data subsetting parameters: frames={frame_subset}, n_traces={n_traces}")
    
    for dsname in DS_LS:
        logger.info(f"Processing dataset: {dsname}")
        Y, S_true = load_gt_ds(os.path.join(LOCAL_DS_PATH, dsname))
        logger.debug(f"Loaded dataset shape - Y: {Y.shape}, S_true: {S_true.shape}")
        
        # First subset by frames
        Y, S_true = Y.dropna("frame").sel(frame=frame_subset), S_true.dropna("frame").sel(frame=frame_subset)
        logger.debug(f"After frame subsetting - Y: {Y.shape}, S_true: {S_true.shape}")
        
        # Get active units
        act_uid = S_true.max("frame") > 0
        Y, S_true = Y.sel(unit_id=act_uid), S_true.sel(unit_id=act_uid)
        logger.info(f"Number of active units: {act_uid.sum().item()}")
        
        # Take first N traces
        if n_traces is not None:
            Y = Y.isel(unit_id=slice(0, n_traces))
            S_true = S_true.isel(unit_id=slice(0, n_traces))
            logger.debug(f"Selected first {n_traces} traces")
        
        Y = Y * 100
        updt_ds = [Y.rename("Y"), S_true.rename("S_true")]
        R = construct_R(Y.sizes["frame"], PARAM_UP_FAC)
        
        logger.info("Starting pipeline processing")
        C_bin, S_bin, iter_df, C_bin_iter, S_bin_iter, h_iter, h_fit_iter = (
            pipeline_bin(
                np.array(Y),
                PARAM_UP_FAC,
                max_iters=PARAM_MAX_ITERS,
                return_iter=True,
                ar_use_all=True,
                ar_kn_len=PARAM_KN_LEN,
                est_noise_freq=0.05,
                est_use_smooth=True,
                est_add_lag=50,
                deconv_norm="l2",
                deconv_backend="osqp",
                da_client=client,
            )
        )
        logger.info("Pipeline processing completed")
        
        logger.debug("Computing final results")
        res = {
            "C": (R @ C_bin.T).T,
            "S": (R @ S_bin.T).T,
            "C_iter": [(R @ c.T).T for c in C_bin_iter],
            "S_iter": [(R @ s.T).T for s in S_bin_iter],
            "h_iter": [R @ h for h in h_iter],
            "h_fit_iter": [R @ h for h in h_fit_iter],
        }
        
        dims = {
            "C": ("unit_id", "frame"),
            "S": ("unit_id", "frame"),
            "C_iter": ("iter", "unit_id", "frame"),
            "S_iter": ("iter", "unit_id", "frame"),
            "h_iter": ("iter", "frame"),
            "h_fit_iter": ("iter", "frame"),
        }
        crds = {
            "unit_id": Y.coords["unit_id"],
            "frame": Y.coords["frame"],
            "iter": np.arange(iter_df["iter"].max() + 1),
        }
        iter_df["unit_id"] = iter_df["cell"].map(
            {i: u.item() for i, u in enumerate(Y.coords["unit_id"])}
        )
        
        # save variables
        logger.info("Saving results")
        for vname, dat in res.items():
            dat = np.stack(dat)
            cur_dims = dims[vname]
            updt_ds.append(
                xr.DataArray(
                    dat,
                    dims=cur_dims,
                    coords={d: crds[d] for d in cur_dims},
                    name=vname,
                )
            )
        updt_ds = xr.merge(updt_ds)
        
        output_path = os.path.join(INT_PATH, f"updt_ds-{dsname}.nc")
        updt_ds.to_netcdf(output_path)
        logger.info(f"Saved dataset to {output_path}")
        
        iter_path = os.path.join(INT_PATH, f"iter_df-{dsname}.feat")
        iter_df.to_feather(iter_path)
        logger.info(f"Saved iteration data to {iter_path}")

    # plotting
    logger.info("Starting plotting phase")
    for dsname in DS_LS:
        logger.info(f"Generating plots for dataset: {dsname}")
        try:
            updt_ds = xr.open_dataset(
                os.path.join(INT_PATH, f"updt_ds-{dsname}.nc")
            )
            iter_df = pd.read_feather(
                os.path.join(INT_PATH, f"iter_df-{dsname}.feat")
            )
            logger.debug(f"Loaded results for plotting - Dataset: {dsname}")
        except FileNotFoundError:
            logger.warning(f"Results not found for dataset: {dsname}")
            continue
            
        Y, S_iter, S_true, C_iter, h_iter, h_fit_iter = (
            updt_ds["Y"],
            updt_ds["S_iter"],
            updt_ds["S_true"],
            updt_ds["C_iter"],
            updt_ds["h_iter"],
            updt_ds["h_fit_iter"],
        )
        
        logger.debug("Computing ROC metrics")
        met_df = []
        for i_iter in np.array(S_iter.coords["iter"]):
            met = compute_ROC(
                S_iter.sel(iter=i_iter), S_true, metadata={"iter": i_iter}
            )
            met_df.append(met)
        met_df = pd.concat(met_df, ignore_index=True)
        
        logger.debug("Generating F1 score plot")
        fig_f1 = px.line(met_df, x="iter", y="f1", color="unit_id")
        f1_path = os.path.join(FIG_PATH, f"f1-{dsname}.html")
        fig_f1.write_html(f1_path)
        logger.info(f"Saved F1 score plot to {f1_path}")
        
        logger.debug("Generating coefficient plot")
        itdf = iter_df.melt(
            id_vars=["iter", "cell"],
            var_name="coef",
            value_vars=["tau_d", "tau_r", "scale", "err"],
            value_name="value",
        )
        fig_coef = px.line(
            itdf,
            x="iter",
            y="value",
            color="coef",
            line_dash="coef",
            line_group="cell",
            markers=True,
        )
        coef_path = os.path.join(FIG_PATH, f"coef-{dsname}.html")
        fig_coef.write_html(coef_path)
        logger.info(f"Saved coefficient plot to {coef_path}")
        
        logger.debug("Generating trace plots for each iteration")
        for i_iter in np.array(S_iter.coords["iter"]):
            uids = np.array(updt_ds.coords["unit_id"])
            fig = make_subplots(
                updt_ds.sizes["unit_id"],
                2,
                shared_xaxes=True,
                shared_yaxes=True,
                row_titles=["unit_id: {}".format(u) for u in uids],
                horizontal_spacing=1e-2,
                column_width=[0.8, 0.2],
            )
            for iu, uid in enumerate(uids):
                irow = iu + 1
                scl = iter_df.set_index(["iter", "unit_id"]).loc[i_iter, uid]["scale"]
                fig.add_trace(
                    go.Scatter(
                        y=Y.sel(unit_id=uid),
                        mode="lines",
                        name="y",
                        legendgroup="y",
                    ),
                    row=irow,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        y=S_true.sel(unit_id=uid),
                        mode="lines",
                        name="s_gt",
                        legendgroup="s_gt",
                    ),
                    row=irow,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        y=S_iter.sel(unit_id=uid, iter=i_iter),
                        mode="lines",
                        name="s",
                        legendgroup="s",
                    ),
                    row=irow,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        y=C_iter.sel(unit_id=uid, iter=i_iter) * scl,
                        mode="lines",
                        name="c",
                        legendgroup="c",
                    ),
                    row=irow,
                    col=1,
                )
                if "unit_id" in h_iter.dims:
                    cur_h = h_iter.sel(unit_id=uid, iter=i_iter)[:PARAM_KN_LEN]
                    cur_h_fit = h_fit_iter.sel(unit_id=uid, iter=i_iter)[:PARAM_KN_LEN]
                else:
                    cur_h = h_iter.sel(iter=i_iter)[:PARAM_KN_LEN]
                    cur_h_fit = h_fit_iter.sel(iter=i_iter)[:PARAM_KN_LEN]
                fig.add_trace(
                    go.Scatter(
                        y=cur_h * scl, mode="lines", name="h_free", legendgroup="h_free"
                    ),
                    row=irow,
                    col=2,
                )
                fig.add_trace(
                    go.Scatter(
                        y=cur_h_fit * scl,
                        mode="lines",
                        name="h_fit",
                        legendgroup="h_fit",
                    ),
                    row=irow,
                    col=2,
                )
            fig.update_layout(height=200 * updt_ds.sizes["unit_id"])
            trace_path = os.path.join(FIG_PATH, f"trs-{dsname}-iter{i_iter}.html")
            fig.write_html(trace_path)
            logger.debug(f"Saved trace plot for iteration {i_iter} to {trace_path}")
            
    logger.info("Benchmark script completed successfully")
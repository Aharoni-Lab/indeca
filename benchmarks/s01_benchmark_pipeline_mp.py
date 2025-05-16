import os

import dask as da
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from tests.testing_utils.cnmf import pipeline_cnmf

from indeca.pipeline import pipeline_bin

IN_PATH = {
    "org": "./intermediate/simulated/simulated-samp.nc",
    "upsamp": "./intermediate/simulated/simulated-upsamp.nc",
}
INT_PATH = "./intermediate/benchmark_pipeline_100cell_est"
FIG_PATH = "./figs/benchmark_pipeline_100cell_est"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10
PARAM_EST_AR = False
PARAM_SIG_LEV = (1, 5)

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

da.config.set({"distributed.scheduler.work-stealing": False})  # avoid pickling error

if __name__ == "__main__":
    cluster = LocalCluster(
        n_workers=30,
        threads_per_worker=1,
        processes=True,
        dashboard_address="0.0.0.0:12345",
    )
    client = Client(cluster)
    sps_penal = 1
    max_iters = 30
    # for up_type, up_factor in {"org": 1, "upsamp": PARAM_UPSAMP}.items():
    for up_type, up_factor in {"org": 1}.items():
        # get data
        sim_ds = xr.open_dataset(IN_PATH[up_type])
        C_gt = sim_ds["C"].dropna("frame", how="all")
        # C_gt = norm_per_cell(C_gt)
        subset = C_gt.coords["unit_id"]
        np.random.seed(42)
        sig_lev = xr.DataArray(
            np.sort(
                np.random.uniform(
                    low=PARAM_SIG_LEV[0],
                    high=PARAM_SIG_LEV[1],
                    size=C_gt.sizes["unit_id"],
                )
            ),
            dims=["unit_id"],
            coords={"unit_id": C_gt.coords["unit_id"]},
            name="sig_lev",
        )
        noise = np.random.normal(loc=0, scale=1, size=C_gt.shape)
        Y_solve = (
            (C_gt * sig_lev + noise).sel(unit_id=subset).transpose("unit_id", "frame")
        )
        updt_ds = [Y_solve.rename("Y_solve"), sig_lev.sel(unit_id=subset)]
        iter_df = []
        # update
        (C_cnmf, S_cnmf) = pipeline_cnmf(
            np.array(Y_solve),
            up_factor,
            ar_mode=False,
            sps_penal=1,
            est_noise_freq=0.06,
            est_use_smooth=True,
            est_add_lag=50,
        )
        (
            C_bin,
            S_bin,
            iter_df,
            C_bin_iter,
            S_bin_iter,
            h_iter,
            h_fit_iter,
        ) = pipeline_bin(
            np.array(Y_solve),
            up_factor,
            max_iters=max_iters,
            # tau_init=np.array([PARAM_TAU_D * up_factor, PARAM_TAU_R * up_factor]),
            return_iter=True,
            ar_use_all=True,
            est_noise_freq=0.06,
            est_use_smooth=True,
            est_add_lag=50,
            deconv_norm="l2",
            deconv_backend="osqp",
            da_client=client,
        )
        res = {
            "C": C_bin,
            "S": S_bin,
            "C_iter": C_bin_iter,
            "S_iter": S_bin_iter,
            "C_cnmf": C_cnmf,
            "S_cnmf": S_cnmf,
            "h_iter": h_iter,
            "h_fit_iter": h_fit_iter,
        }
        dims = {
            "C": ("unit_id", "frame"),
            "S": ("unit_id", "frame"),
            "C_iter": ("iter", "unit_id", "frame"),
            "S_iter": ("iter", "unit_id", "frame"),
            "C_cnmf": ("unit_id", "frame"),
            "S_cnmf": ("unit_id", "frame"),
            "h_iter": ("iter", "frame"),
            "h_fit_iter": ("iter", "frame"),
        }
        crds = {
            "unit_id": Y_solve.coords["unit_id"],
            "frame": (
                sim_ds.coords["frame"]
                if up_type == "upsamp"
                else Y_solve.coords["frame"]
            ),
            "iter": np.arange(iter_df["iter"].max() + 1),
        }
        # save variables
        iter_df["unit_id"] = iter_df["cell"].map(
            {i: u.item() for i, u in enumerate(Y_solve.coords["unit_id"])}
        )
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
        updt_ds.to_netcdf(os.path.join(INT_PATH, "updt_ds-{}.nc".format(up_type)))
        iter_df.to_feather(os.path.join(INT_PATH, "iter_df-{}.feat".format(up_type)))

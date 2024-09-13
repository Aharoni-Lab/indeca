# %% This lays out a potential workflow for the general use of minian-bin.
# Workflow is:
# 0. Handle imports and definitions
# 1. Generate or inport dataset at normal FPS and upsampled for calcium imaging
# 2. estimate initial guess at convolution kernel
# 3. Solve for non-binarized 's'
# 4. Binarize 's' using a single threshold across all cells and determine per cell scaling factor
# 5. Update free kernel based on binarized spikes
# 6. Optionally fit free kernel to bi-exponential and generate new kernel from this
# 7. Iterate back to step 4 and repeat until some metric is reached

# %% 0. Handle imports and definitions
import os

import cvxpy as cp
import numpy as np
import xarray as xr

from minian_bin.update_pipeline import pipeline_bin

IN_PATH = {
    "org": "./intermediate/simulated/simulated-ar-samp.nc",
    "upsamp": "./intermediate/simulated/simulated-ar-upsamp.nc",
}
INT_PATH = "./intermediate/benchmark_pipeline"
FIG_PATH = "./figs/benchmark_pipeline"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10
PARAM_EST_AR = False
PARAM_SIG_LEV = (1, 10)

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


# %% 1. Generate or import dataset at normal FPS for calcium imaging
sps_penal = 1
max_iters = 50
for up_type, up_factor in {"org": 1, "upsamp": PARAM_UPSAMP}.items():
    # get data
    sim_ds = xr.open_dataset(IN_PATH[up_type])
    C_gt = sim_ds["C"].dropna("frame", how="all")
    # C_gt = norm_per_cell(C_gt)
    subset = C_gt.coords["unit_id"][-5:]
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
    # Y_solve = (C_gt * sig_lev + noise).sel(unit_id=subset)
    Y_solve = C_gt.sel(unit_id=subset).transpose("unit_id", "frame")
    updt_ds = [Y_solve.rename("Y_solve"), sig_lev.sel(unit_id=subset)]
    iter_df = []
    # update
    C_bin_iter, S_bin_iter, iter_df = pipeline_bin(
        np.array(Y_solve),
        up_factor,
        max_iters=10,
        tau_init=np.array([PARAM_TAU_D * up_factor, PARAM_TAU_R * up_factor]),
        save_iter=True,
        ar_use_all=False,
    )
    res = {"C": C_bin_iter, "S": S_bin_iter}
    # save variables
    for vname, dat in res.items():
        dat = np.stack(dat)
        if dat.ndim == 1:
            updt_ds.append(
                xr.DataArray(
                    dat,
                    dims="unit_id",
                    coords={"unit_id": Y_solve.coords["unit_id"]},
                    name=vname,
                )
            )
        elif dat.ndim == 2:
            updt_ds.append(
                xr.DataArray(
                    dat,
                    dims=["unit_id", "frame"],
                    coords={
                        "unit_id": Y_solve.coords["unit_id"],
                        "frame": (
                            sim_ds.coords["frame"]
                            if up_type == "upsamp"
                            else Y_solve.coords["frame"]
                        ),
                    },
                    name=vname,
                )
            )
        elif dat.ndim == 3:
            updt_ds.append(
                xr.DataArray(
                    dat,
                    dims=["iter", "unit_id", "frame"],
                    coords={
                        "iter": np.arange(dat.shape[0]),
                        "unit_id": Y_solve.coords["unit_id"],
                        "frame": (
                            sim_ds.coords["frame"]
                            if up_type == "upsamp"
                            else Y_solve.coords["frame"]
                        ),
                    },
                    name=vname,
                )
            )
        else:
            raise ValueError
    updt_ds = xr.merge(updt_ds)
    updt_ds.to_netcdf(os.path.join(INT_PATH, "updt_ds-{}.nc".format(up_type)))
    iter_df.to_feather(os.path.join(INT_PATH, "iter_df-{}.feat".format(up_type)))


# %% 3.3 Estimate spiking from kernel and upsampled C
# Function to solve for spike estimates given calcium trace and kernel
def solve_s(y, h, norm="l1", sparsity_penalty=0):
    y, h = y.squeeze(), h.squeeze()

    num_samples = len(y)

    # Baseline fluorescence for each cell
    b = cp.Variable()

    # Spike train for each cell and time point
    s = cp.Variable(num_samples)

    # Convolution term: applying cp.conv() for each cell separately

    conv_term = cp.conv(h, s)[
        :num_samples
    ]  # Convolve spike train with kernel for each cell

    # Norm choice (l1 or l2)
    norm_ord = {"l1": 1, "l2": 2}[norm]

    # Objective function: minimize reconstruction error + sparsity penalty
    obj = cp.Minimize(
        cp.norm(y - conv_term - b, norm_ord)  # Properly broadcast baseline
        #  + sparsity_penalty * cp.norm(s, 1)  # L1 sparsity penalty
    )

    # Constraints: non-negative spikes and baseline
    cons = [s >= 0, b >= 0]

    # Define and solve the problem
    prob = cp.Problem(obj, cons)
    prob.solve()  # Using SCS solver to handle large-scale problems
    return s.value

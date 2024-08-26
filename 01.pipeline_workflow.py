# %% This lays out a potential workflow for the general use of minian-bin.
# Workflow is:
# 0. Handle imports and definitions
# 1. Generate or inport dataset at normal FPS for calcium imaging
# 2. Upscale data to ~1KHz sampling
# 3. estimate initial guess at convolution kernel
# 4. Solve for non-binarized 's'
# 5. Binarize 's' using a single threshold across all cells and determine per cell scaling factor
# 6. Update free kernel based on binarized spikes
# 7. Optionally fit free kernel to bi-exponential and generate new kernel from this
# 8. Iterate back to step 4 and repeat until some metric is reached

# %% 0. Handle imports and definitions
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from tqdm.auto import tqdm
from routine.simulation import generate_data

OUT_PATH = "./intermediate/simulated/simulated.nc"

# %% 1. Generate or inport dataset at normal FPS for calcium imaging

PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

np.random.seed(42)
ds = generate_data(
    dpath=OUT_PATH,
    ncell=100,
    upsample=PARAM_UPSAMP,
    dims={"height": 256, "width": 256, "frame": 2000},
    sig_scale=1,
    sz_mean=3,
    sz_sigma=0.6,
    sz_min=0.1,
    tmp_P=np.array([[0.998, 0.002], [0.75, 0.25]]),
    tmp_tau_d=PARAM_TAU_D,
    tmp_tau_r=PARAM_TAU_R,
    bg_nsrc=0,
    bg_tmp_var=0,
    bg_cons_fac=0,
    bg_smth_var=0,
    mo_stp_var=0,
    mo_cons_fac=0,
    post_offset=1,
    post_gain=50,
    save_Y=True,
)
# %% 2. Upscale data to ~1KHz sampling
# %% 3. estimate initial guess at convolution kernel
# %% 4. Solve for non-binarized 's'
# %% 5. Binarize 's' using a single threshold across all cells and determine per cell scaling factor
# %% 6. Update free kernel based on binarized spikes
# %% 7. Optionally fit free kernel to bi-exponential and generate new kernel from this
# %% 8. Iterate back to step 4 and repeat until some metric is reached
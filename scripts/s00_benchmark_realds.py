# %% import and definitions
import os

from routine.io import download_realds, load_gt_ds

LOCAL_DS_PATH = "./data/realds/"

# %% download dataset
if not os.path.exists(LOCAL_DS_PATH):
    download_realds(LOCAL_DS_PATH)

# %% load data
C_true, S_true = load_gt_ds(os.path.join(LOCAL_DS_PATH, "DS01-OGB1-m-V1"))

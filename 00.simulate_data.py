# %% import and definition
import itertools as itt
import os

import numpy as np

from routine.simulation import generate_data

OUT_PATH = "./intermediate/simulated/"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# %% generate data
for (upsamp_lab, upsamp), (useAR_lab, useAR) in itt.product(
    {"samp": 1, "upsamp": PARAM_UPSAMP}.items(),
    {"exp": False, "ar": True}.items(),
):
    np.random.seed(42)
    ds = generate_data(
        dpath=os.path.join(
            OUT_PATH, "simulated-{}-{}.nc".format(useAR_lab, upsamp_lab)
        ),
        ncell=100,
        upsample=upsamp,
        useAR=useAR,
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

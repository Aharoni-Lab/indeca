import os

import pandas as pd

from indeca.deconv import construct_R
from indeca.simulation import ar_pulse, tau2AR
from indeca.utils import scal_lstsq


def load_agg_result(res_path):
    try:
        res_files = list(filter(lambda fn: fn.endswith(".feat"), os.listdir(res_path)))
    except FileNotFoundError:
        return
    return pd.concat(
        [pd.read_feather(res_path / f) for f in res_files], ignore_index=True
    )


def get_upsamp_scale(taus, upsamp_ref, upsamp_slv, nsamp=100):
    ar_ref = tau2AR(taus[0] * upsamp_ref, taus[1] * upsamp_ref)
    ar_slv = tau2AR(taus[0] * upsamp_slv, taus[1] * upsamp_slv)
    pulse_ref, _, _ = ar_pulse(
        ar_ref[0], ar_ref[1], nsamp=nsamp * upsamp_ref, shifted=True
    )
    pulse_slv, _, _ = ar_pulse(
        ar_slv[0], ar_slv[1], nsamp=nsamp * upsamp_slv, shifted=True
    )
    R_ref = construct_R(nsamp, upsamp_ref)
    R_slv = construct_R(nsamp, upsamp_slv)
    return scal_lstsq(R_slv @ pulse_slv, R_ref @ pulse_ref, fit_intercept=False).item()

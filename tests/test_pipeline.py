import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from minian_bin.deconv import construct_R
from minian_bin.pipeline import pipeline_bin

from .testing_utils.cnmf import pipeline_cnmf
from .testing_utils.io import download_realds, load_gt_ds
from .testing_utils.metrics import assignment_distance, df_assign_metadata


@pytest.fixture()
def temp_data_dir():
    return "./intermediate/data"


@pytest.fixture(params=[slice(0, 10)] + [i for i in range(10)])
def param_subset_cell(request):
    return request.param


@pytest.fixture(params=[1, 2])
def param_upsamp(request):
    return request.param


@pytest.fixture(params=[60])
def param_ar_kn_len(request):
    return request.param


@pytest.fixture(params=[0.05])
def param_noise_freq(request):
    return request.param


@pytest.fixture(params=[50])
def param_add_lag(request):
    return request.param


@pytest.fixture(params=[10])
def param_max_iters(request):
    return request.param


@pytest.fixture(params=["X-DS09-GCaMP6f-m-V1"])
def fixt_realds(temp_data_dir, param_subset_cell, request):
    dsname = request.param
    if not os.path.exists(os.path.join(temp_data_dir, dsname)) or not os.listdir(
        os.path.join(temp_data_dir, dsname)
    ):
        download_realds(temp_data_dir, dsname)
    Y, ap_df, fluo_df = load_gt_ds(os.path.join(temp_data_dir, dsname), return_ap=True)
    Y = Y.sel(unit_id=param_subset_cell).dropna("frame")
    fmin, fmax = Y.coords["frame"].min().item(), Y.coords["frame"].max().item()
    ap_df = ap_df.loc[param_subset_cell]
    fluo_df = fluo_df.loc[param_subset_cell]
    ap_df = ap_df[ap_df["frame"].between(fmin, fmax)].copy()
    fluo_df = fluo_df[fluo_df["frame"].between(fmin, fmax)].copy()
    return (Y, ap_df, fluo_df)


@pytest.mark.slow
class TestDemoPipeline:
    def test_demo_pipeline(
        self,
        fixt_realds,
        param_upsamp,
        param_max_iters,
        param_ar_kn_len,
        param_noise_freq,
        param_add_lag,
        results_bag,
    ):
        # act
        Y, ap_df, fluo_df = fixt_realds
        (
            C_bin,
            S_bin,
            iter_df,
            C_bin_iter,
            S_bin_iter,
            h_iter,
            h_fit_iter,
        ) = pipeline_bin(
            np.atleast_2d(Y),
            param_upsamp,
            max_iters=param_max_iters,
            return_iter=True,
            ar_use_all=True,
            ar_kn_len=param_ar_kn_len,
            est_noise_freq=param_noise_freq,
            est_use_smooth=True,
            est_add_lag=param_add_lag,
            deconv_norm="l2",
            deconv_backend="osqp",
            spawn_dashboard=False,
        )
        # save results
        res_df = []
        for i_iter, sbin in enumerate(S_bin_iter):
            for iu, uid in enumerate(np.atleast_1d(Y.coords["unit_id"])):
                sb = sbin[iu, :]
                cur_ap = ap_df.loc[uid]
                cur_fluo = fluo_df.loc[uid]
                sb_idx = np.where(sb)[0] / param_upsamp
                t_sb = np.interp(sb_idx, cur_fluo["frame"], cur_fluo["fluo_time"])
                t_ap = cur_ap["ap_time"]
                mdist, f1, prec, rec = assignment_distance(
                    t_ref=np.array(t_ap), t_slv=np.array(t_sb), tdist_thres=1
                )
                res_df.append(
                    pd.DataFrame(
                        [
                            {
                                "method": "minian-bin",
                                "use_all": "unit_id" in Y.dims,
                                "unit_id": uid,
                                "iter": i_iter,
                                "mdist": mdist,
                                "f1": f1,
                                "prec": prec,
                                "rec": rec,
                            }
                        ]
                    )
                )
        res_df = pd.concat(res_df, ignore_index=True)
        results_bag.data = res_df

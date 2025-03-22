import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from minian_bin.deconv import construct_R
from minian_bin.pipeline import pipeline_bin
from minian_bin.simulation import find_dhm

from .testing_utils.cnmf import pipeline_cnmf
from .testing_utils.io import download_realds, load_gt_ds
from .testing_utils.metrics import assignment_distance, df_assign_metadata


@pytest.fixture()
def temp_data_dir():
    return "./intermediate/data"


@pytest.fixture(params=[slice(0, 10)] + [i for i in range(10)])
def param_subset_cell(request):
    return request.param


@pytest.fixture(params=[(0, 10000)])
def param_subset_fm(request):
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
def fixt_realds(temp_data_dir, param_subset_cell, param_subset_fm, request):
    dsname = request.param
    if not os.path.exists(os.path.join(temp_data_dir, dsname)) or not os.listdir(
        os.path.join(temp_data_dir, dsname)
    ):
        download_realds(temp_data_dir, dsname)
    Y, ap_df, fluo_df = load_gt_ds(os.path.join(temp_data_dir, dsname), return_ap=True)
    Y = Y.isel(frame=slice(*param_subset_fm)).dropna("frame")
    ap_df = ap_df[ap_df["frame"].between(*param_subset_fm)]
    fluo_df = fluo_df[fluo_df["frame"].between(*param_subset_fm)]
    ap_ct = ap_df.groupby("unit_id")["ap_time"].count().reset_index()
    act_uids = np.array(ap_ct.loc[ap_ct["ap_time"] > 1, "unit_id"])
    Y = Y.sel(unit_id=act_uids)
    try:
        Y = Y.isel(unit_id=param_subset_cell)
    except IndexError:
        raise IndexError(
            "Cannot select {} active cells with frame subset {}".format(
                param_subset_cell, param_subset_fm
            )
        )
    Y = Y * 100
    uids = np.array(Y.coords["unit_id"])
    ap_df = ap_df.set_index("unit_id").loc[uids]
    fluo_df = fluo_df.set_index("unit_id").loc[uids]
    return Y, ap_df, fluo_df


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
        C_cnmf, S_cnmf, tau_cnmf = pipeline_cnmf(
            np.atleast_2d(Y), up_factor=1, est_noise_freq=0.06, sps_penal=0
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
        iter_df = iter_df.set_index(["iter", "cell"])
        res_df = []
        for i_iter, sbin in enumerate(S_bin_iter):
            for iu, uid in enumerate(np.atleast_1d(Y.coords["unit_id"])):
                sb = sbin[iu, :]
                tau_d, tau_r = iter_df.loc[(i_iter, iu), ["tau_d", "tau_r"]]
                try:
                    (dhm0, dhm1), _ = find_dhm(
                        True, np.array([tau_d, tau_r]), np.array([1, -1])
                    )
                except AssertionError:
                    dhm0, dhm1 = 0, 0
                if len(ap_df) > 0:
                    cur_ap = ap_df.loc[uid]
                    cur_fluo = fluo_df.loc[uid]
                    sb_idx = np.where(sb)[0] / param_upsamp
                    t_sb = np.interp(sb_idx, cur_fluo["frame"], cur_fluo["fluo_time"])
                    t_ap = cur_ap["ap_time"]
                    mdist, f1, prec, rec = assignment_distance(
                        t_ref=np.atleast_1d(t_ap),
                        t_slv=np.atleast_1d(t_sb),
                        tdist_thres=1,
                    )
                else:
                    mdist, f1, prec, rec = np.nan, 0, 0, 0
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
                                "dhm0": dhm0,
                                "dhm1": dhm1,
                            }
                        ]
                    )
                )
        for iu, uid in enumerate(np.atleast_1d(Y.coords["unit_id"])):
            for qthres in [0.25, 0.5, 0.75]:
                sb = S_cnmf[iu, :] > qthres
                tau_d, tau_r = tau_cnmf[iu, :]
                try:
                    (dhm0, dhm1), _ = find_dhm(
                        True, np.array([tau_d, tau_r]), np.array([1, -1])
                    )
                except AssertionError:
                    dhm0, dhm1 = 0, 0
                if len(ap_df) > 0:
                    cur_ap = ap_df.loc[uid]
                    cur_fluo = fluo_df.loc[uid]
                    sb_idx = np.where(sb)[0] / param_upsamp
                    t_sb = np.interp(sb_idx, cur_fluo["frame"], cur_fluo["fluo_time"])
                    t_ap = cur_ap["ap_time"]
                    mdist, f1, prec, rec = assignment_distance(
                        t_ref=np.atleast_1d(t_ap),
                        t_slv=np.atleast_1d(t_sb),
                        tdist_thres=1,
                    )
                else:
                    mdist, f1, prec, rec = np.nan, 0, 0, 0
                res_df.append(
                    pd.DataFrame(
                        [
                            {
                                "method": "cnmf",
                                "use_all": "unit_id" in Y.dims,
                                "unit_id": uid,
                                "qthres": qthres,
                                "mdist": mdist,
                                "f1": f1,
                                "prec": prec,
                                "rec": rec,
                                "dhm0": dhm0,
                                "dhm1": dhm1,
                            }
                        ]
                    )
                )
        res_df = pd.concat(res_df, ignore_index=True)
        results_bag.data = res_df

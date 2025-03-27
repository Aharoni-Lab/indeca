import itertools as itt
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from plotly.subplots import make_subplots

from minian_bin.pipeline import pipeline_bin
from minian_bin.simulation import find_dhm

from .conftest import fixt_y
from .testing_utils.cnmf import pipeline_cnmf
from .testing_utils.metrics import assignment_distance
from .testing_utils.plotting import plot_traces


class TestPipeline:

    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize("upsamp", [1, pytest.param(2, marks=pytest.mark.slow)])
    @pytest.mark.parametrize("max_iter", [5])
    @pytest.mark.parametrize("ncell", [1, pytest.param(10, marks=pytest.mark.slow)])
    @pytest.mark.parametrize("ar_kn_len", [60])
    @pytest.mark.parametrize(
        "ns_lev",
        [0] + [pytest.param(n, marks=pytest.mark.slow) for n in [0.1, 0.2, 0.5]],
    )
    @pytest.mark.parametrize("est_noise_freq", [None])
    @pytest.mark.parametrize("est_add_lag", [10])
    def test_pipeline(
        self,
        taus,
        rand_seed,
        upsamp,
        max_iter,
        ncell,
        ar_kn_len,
        ns_lev,
        est_noise_freq,
        est_add_lag,
        results_bag,
        test_fig_path_html,
    ):
        # act
        Y, C, C_org, S, S_org, scales = fixt_y(
            taus=taus,
            rand_seed=rand_seed,
            upsamp=upsamp,
            ncell=ncell,
            ns_lev=ns_lev,
            squeeze=False,
        )
        C_cnmf, S_cnmf, tau_cnmf = pipeline_cnmf(
            Y,
            up_factor=1,
            est_noise_freq=est_noise_freq,
            est_add_lag=est_add_lag,
            est_use_smooth=False,
            sps_penal=0,
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
            Y,
            upsamp,
            max_iters=max_iter,
            return_iter=True,
            ar_use_all=True,
            ar_kn_len=ar_kn_len,
            est_noise_freq=est_noise_freq,
            est_use_smooth=False,
            est_add_lag=est_add_lag,
            spawn_dashboard=False,
        )
        # save results
        iter_df = iter_df.set_index(["iter", "cell"])
        (dhm0, dhm1), _ = find_dhm(
            True, np.array([taus[0], taus[1]]), np.array([1, -1])
        )
        res_df = [
            pd.DataFrame(
                [{"method": "gt", "use_all": True, "dhm0": dhm0, "dhm1": dhm1}]
            )
        ]
        for uid in range(Y.shape[0]):
            for i_iter, sbin in enumerate(S_bin_iter):
                sb = sbin[uid, :]
                tau_d, tau_r = iter_df.loc[(i_iter, uid), ["tau_d", "tau_r"]]
                try:
                    (dhm0, dhm1), _ = find_dhm(
                        True, np.array([tau_d, tau_r]), np.array([1, -1])
                    )
                except AssertionError:
                    dhm0, dhm1 = 0, 0
                mdist, f1, prec, rec = assignment_distance(
                    s_ref=S_org[uid, :], s_slv=sb, tdist_thres=3
                )
                res_df.append(
                    pd.DataFrame(
                        [
                            {
                                "method": "minian-bin",
                                "use_all": Y.shape[0] > 1,
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
        for uid in range(Y.shape[0]):
            for qthres in [0.25, 0.5, 0.75]:
                sb = S_cnmf[uid, :] > qthres
                tau_d, tau_r = tau_cnmf[uid, :]
                try:
                    (dhm0, dhm1), _ = find_dhm(
                        True, np.array([tau_d, tau_r]), np.array([1, -1])
                    )
                except AssertionError:
                    dhm0, dhm1 = 0, 0
                mdist, f1, prec, rec = assignment_distance(
                    s_ref=S_org[uid, :], s_slv=sb, tdist_thres=3
                )
                res_df.append(
                    pd.DataFrame(
                        [
                            {
                                "method": "cnmf",
                                "use_all": False,
                                "iter": "final",
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
        # plotting
        niter = len(S_bin_iter)
        ncell = Y.shape[0]
        fig = make_subplots(rows=niter, cols=ncell)
        for uid, i_iter in itt.product(range(ncell), range(niter)):
            sb = S_bin_iter[i_iter][uid, :]
            cb = C_bin_iter[i_iter][uid, :]
            tau_d, tau_r = iter_df.loc[(i_iter, uid), ["tau_d", "tau_r"]]
            fig.add_traces(
                plot_traces(
                    {
                        "y": Y[uid, :],
                        "c_true": C_org[uid, :],
                        "s_true": S_org[uid, :],
                        "c_bin": cb,
                        "s_bin": sb,
                    }
                ),
                rows=i_iter + 1,
                cols=uid + 1,
            )
        fig.update_layout(height=350 * niter, width=1200 * ncell)
        fig.write_html(test_fig_path_html)
        # assertion
        if ns_lev == 0 and upsamp == 1:
            f1_last = res_df.set_index(["method", "iter"]).loc[
                ("minian-bin", niter - 1), "f1"
            ]
            # TODO: make this pass with all f1 == 1
            assert f1_last.median() == 1


@pytest.mark.slow
class TestDemoPipeline:
    @pytest.mark.parametrize("upsamp", [1])
    @pytest.mark.parametrize("max_iter", [5])
    @pytest.mark.parametrize("ar_kn_len", [100])
    @pytest.mark.parametrize("est_noise_freq", [None])
    @pytest.mark.parametrize("est_add_lag", [10])
    @pytest.mark.parametrize("fixt_realds", [{"subset_cell": (0, 5)}], indirect=True)
    def test_demo_pipeline(
        self,
        fixt_realds,
        upsamp,
        max_iter,
        ar_kn_len,
        est_noise_freq,
        est_add_lag,
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
            up_factor=upsamp,
            max_iters=max_iter,
            return_iter=True,
            ar_use_all=True,
            ar_kn_len=ar_kn_len,
            est_noise_freq=est_noise_freq,
            est_use_smooth=False,
            est_add_lag=est_add_lag,
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
                    sb_idx = np.where(sb)[0] / upsamp
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
                    sb_idx = np.where(sb)[0] / upsamp
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
                                "use_all": False,
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

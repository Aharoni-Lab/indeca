import itertools as itt
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

from indeca.deconv import DeconvBin, construct_R
from indeca.pipeline import pipeline_bin
from indeca.simulation import find_dhm

from .conftest import fixt_realds, fixt_y
from .testing_utils.cnmf import pipeline_cnmf
from .testing_utils.metrics import assignment_distance, dtw_corr, nzidx_int
from .testing_utils.plotting import plot_traces


class TestPipeline:
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize("upsamp", [1, pytest.param(2, marks=pytest.mark.slow)])
    @pytest.mark.parametrize("max_iter", [10])
    @pytest.mark.parametrize("ncell", [1, pytest.param(10, marks=pytest.mark.slow)])
    @pytest.mark.parametrize("ar_kn_len", [60])
    @pytest.mark.parametrize(
        "ns_lev",
        [0] + [pytest.param(n, marks=pytest.mark.slow) for n in [0.1, 0.2, 0.5]],
    )
    @pytest.mark.parametrize("est_noise_freq", [None])
    @pytest.mark.parametrize("est_add_lag", [10])
    @pytest.mark.parametrize("err_weighting", [None, "adaptive"])
    @pytest.mark.parametrize("ar_use_all", [True, False])
    def test_pipeline(
        self,
        taus,
        rand_seed,
        upsamp,
        max_iter,
        ncell,
        ar_kn_len,
        ns_lev,
        err_weighting,
        ar_use_all,
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
            deconv_err_weighting=err_weighting,
            deconv_pks_polish=True,
            ar_use_all=ar_use_all,
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
        res_df = [pd.DataFrame([{"method": "gt", "dhm0": dhm0, "dhm1": dhm1}])]
        for uid in range(Y.shape[0]):
            for i_iter, sbin in enumerate(S_bin_iter):
                sb = sbin[uid, :]
                try:
                    tau_d, tau_r = iter_df.loc[(i_iter, uid), ["tau_d", "tau_r"]]
                except KeyError:
                    tau_d, tau_r = np.nan, np.nan
                try:
                    (dhm0, dhm1), _ = find_dhm(
                        True, np.array([tau_d, tau_r]) / upsamp, np.array([1, -1])
                    )
                except AssertionError:
                    dhm0, dhm1 = 0, 0
                mdist, f1, prec, rec = assignment_distance(
                    s_ref=S_org[uid, :-1], s_slv=sb[:-1], tdist_thres=3
                )
                res_df.append(
                    pd.DataFrame(
                        [
                            {
                                "method": "indeca",
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
                    s_ref=S_org[uid, :-1], s_slv=sb[:-1], tdist_thres=3
                )
                res_df.append(
                    pd.DataFrame(
                        [
                            {
                                "method": "cnmf",
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
        if ns_lev == 0:
            f1_last = res_df.set_index(["method", "iter"]).loc[
                ("indeca", niter - 1), "f1"
            ]
            assert f1_last.min() == 1


@pytest.mark.slow
class TestDemoPipeline:
    @pytest.mark.parametrize("upsamp", [None])
    @pytest.mark.parametrize("max_iter", [10])
    @pytest.mark.parametrize("ar_kn_len", [200])
    @pytest.mark.parametrize("est_noise_freq", [None])
    @pytest.mark.parametrize("est_add_lag", [200])
    @pytest.mark.parametrize(
        "dsname",
        [
            "DS01-OGB1-m-V1",
            "DS02-OGB1-2-m-V1",
            # "DS03-Cal520-m-S1",
            "DS04-OGB1-zf-pDp",
            "DS06-GCaMP6f-zf-aDp",
            # "DS07-GCaMP6f-zf-dD",
            "DS08-GCaMP6f-zf-OB",
            "DS09-GCaMP6f-m-V1",
            "DS10-GCaMP6f-m-V1-neuropil-corrected",
            "DS11-GCaMP6f-m-V1-neuropil-corrected",
            "DS12-GCaMP6s-m-V1-neuropil-corrected",
            "DS13-GCaMP6s-m-V1-neuropil-corrected",
            "DS14-GCaMP6s-m-V1",
            "DS15-GCaMP6s-m-V1",
            "DS16-GCaMP6s-m-V1",
            "DS17-GCaMP5k-m-V1",
            "DS18-R-CaMP-m-CA3",
            "DS19-R-CaMP-m-S1",
            "DS20-jRCaMP1a-m-V1",
            "DS21-jGECO1a-m-V1",
            "DS22-OGB1-m-SST-V1",
            "DS23-OGB1-m-PV-V1",
            "DS24-GCaMP6f-m-PV-V1",
            "DS25-GCaMP6f-m-SST-V1",
            "DS26-GCaMP6f-m-VIP-V1",
            "DS27-GCaMP6f-m-PV-vivo-V1",
            "DS28-XCaMPgf-m-V1",
            "DS29-GCaMP7f-m-V1",
            "DS30-GCaMP8f-m-V1",
            "DS31-GCaMP8m-m-V1",
            "DS32-GCaMP8s-m-V1",
            "DS33-Interneurons2023-m-V1",
            "DS40-GCaMP6s-spinal-cord-excitatory",
            "DS41-GCaMP6s-spinal-cord-inhibitory",
            "X-DS09-GCaMP6f-m-V1",
        ],
    )
    @pytest.mark.parametrize("ncell", [1, None])
    @pytest.mark.parametrize("nfm", [None])
    @pytest.mark.parametrize("penalty", [None])
    @pytest.mark.parametrize("tau_init", [None])
    @pytest.mark.parametrize("ar_use_all", [True, False])
    @pytest.mark.line_profile.with_args(DeconvBin.solve_thres)
    def test_demo_pipeline_realds(
        self,
        upsamp,
        max_iter,
        ar_kn_len,
        est_noise_freq,
        est_add_lag,
        dsname,
        ncell,
        nfm,
        penalty,
        tau_init,
        ar_use_all,
        results_bag,
        test_fig_path_html,
        func_data_dir,
    ):
        # act
        Y, S_true, ap_df, fluo_df = fixt_realds(dsname, ncell, nfm)
        if upsamp is None:
            upsamp = max(int(S_true.max().item()), 1)
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
            tau_init=tau_init,
            return_iter=True,
            deconv_use_base=True,
            deconv_penal=penalty,
            deconv_err_weighting="adaptive",
            deconv_masking_radius=None,
            deconv_pks_polish=False,
            deconv_ncons_thres=upsamp * 2,
            deconv_min_rel_scl=None,
            ar_use_all=ar_use_all,
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
                s_true = S_true[iu, :]
                R = construct_R(len(s_true), upsamp)
                Rsb = R @ sb
                try:
                    tau_d, tau_r = iter_df.loc[(i_iter, iu), ["tau_d", "tau_r"]]
                except KeyError:
                    tau_d, tau_r = np.nan, np.nan
                try:
                    (dhm0, dhm1), _ = find_dhm(
                        True, np.array([tau_d, tau_r]), np.array([1, -1])
                    )
                except AssertionError:
                    dhm0, dhm1 = 0, 0
                if len(ap_df) > 0:
                    cur_ap = ap_df.loc[uid]
                    cur_fluo = fluo_df.loc[uid]
                    sb = np.around(Rsb / Rsb.max() * np.array(s_true).max())
                    sb_idx = nzidx_int(np.array(sb).astype(int))
                    t_sb = np.interp(sb_idx, cur_fluo["frame"], cur_fluo["fluo_time"])
                    t_ap = cur_ap["ap_time"]
                    mdist, f1, prec, rec = assignment_distance(
                        t_ref=np.atleast_1d(t_ap),
                        t_slv=np.atleast_1d(t_sb),
                        tdist_thres=fluo_df["fluo_time"].diff().median() * 2.5,
                    )
                    corr_raw = np.corrcoef(s_true, Rsb)[0, 1]
                    corr_gs = np.corrcoef(
                        gaussian_filter1d(s_true, 1), gaussian_filter1d(Rsb, 1)
                    )[0, 1]
                    # corr_dtw = dtw_corr(s_true, Rsb)
                else:
                    mdist, f1, prec, rec, corr_raw, corr_gs = np.nan, 0, 0, 0, 0, 0
                res_df.append(
                    pd.DataFrame(
                        [
                            {
                                "method": "indeca",
                                "use_all": ar_use_all,
                                "unit_id": uid,
                                "iter": i_iter,
                                "mdist": mdist,
                                "f1": f1,
                                "prec": prec,
                                "rec": rec,
                                "dhm0": dhm0,
                                "dhm1": dhm1,
                                "corr_raw": corr_raw,
                                "corr_gs": corr_gs,
                                # "corr_dtw": corr_dtw,
                            }
                        ]
                    )
                )
        res_df = pd.concat(res_df, ignore_index=True)
        results_bag.data = res_df
        # save raw traces
        R = construct_R(len(s_true), upsamp)
        if ncell is None:
            S = xr.DataArray(
                R @ (S_bin_iter[-1]).T,
                dims=["frame", "unit_id"],
                coords={"unit_id": Y.coords["unit_id"], "frame": Y.coords["frame"]},
                name="S",
            )
            C = xr.DataArray(
                R @ (C_bin_iter[-1]).T,
                dims=["frame", "unit_id"],
                coords={"unit_id": Y.coords["unit_id"], "frame": Y.coords["frame"]},
                name="C",
            )
            ds = xr.merge([S, C])
            ds.to_netcdf(
                os.path.join(func_data_dir, "{}-{}.nc".format(dsname, ar_use_all))
            )
        # plotting
        niter = len(S_bin_iter)
        ncell = Y.shape[0]
        fig = make_subplots(rows=niter, cols=ncell)
        for uid, i_iter in itt.product(range(ncell), range(niter)):
            sb = S_bin_iter[i_iter][uid, :]
            cb = C_bin_iter[i_iter][uid, :]
            Rsb = R @ sb
            Rcb = R @ cb
            try:
                scal = iter_df.loc[(i_iter, uid), "scale"]
            except KeyError:
                opt_iter = iter_df.xs(uid, level=1)["obj"].idxmin()
                scal = iter_df.loc[(opt_iter, uid), "scale"]
            fig.add_traces(
                plot_traces(
                    {
                        "y": Y[uid, :] / scal,
                        "s_true": S_true[uid, :],
                        "c_bin": cb,
                        "s_bin": sb,
                        "R@c_bin": Rcb,
                        "R@s_bin": Rsb,
                    }
                ),
                rows=i_iter + 1,
                cols=uid + 1,
            )
        fig.update_layout(height=350 * niter, width=1200 * ncell)
        fig.write_html(test_fig_path_html)

    @pytest.mark.parametrize("est_noise_freq", [None])
    @pytest.mark.parametrize("est_add_lag", [200])
    @pytest.mark.parametrize(
        "dsname",
        [
            "DS01-OGB1-m-V1",
            "DS02-OGB1-2-m-V1",
            # "DS03-Cal520-m-S1",
            "DS04-OGB1-zf-pDp",
            "DS06-GCaMP6f-zf-aDp",
            # "DS07-GCaMP6f-zf-dD",
            "DS08-GCaMP6f-zf-OB",
            "DS09-GCaMP6f-m-V1",
            "DS10-GCaMP6f-m-V1-neuropil-corrected",
            "DS11-GCaMP6f-m-V1-neuropil-corrected",
            "DS12-GCaMP6s-m-V1-neuropil-corrected",
            "DS13-GCaMP6s-m-V1-neuropil-corrected",
            "DS14-GCaMP6s-m-V1",
            "DS15-GCaMP6s-m-V1",
            "DS16-GCaMP6s-m-V1",
            "DS17-GCaMP5k-m-V1",
            "DS18-R-CaMP-m-CA3",
            "DS19-R-CaMP-m-S1",
            "DS20-jRCaMP1a-m-V1",
            "DS21-jGECO1a-m-V1",
            "DS22-OGB1-m-SST-V1",
            "DS23-OGB1-m-PV-V1",
            "DS24-GCaMP6f-m-PV-V1",
            "DS25-GCaMP6f-m-SST-V1",
            "DS26-GCaMP6f-m-VIP-V1",
            "DS27-GCaMP6f-m-PV-vivo-V1",
            "DS28-XCaMPgf-m-V1",
            "DS29-GCaMP7f-m-V1",
            "DS30-GCaMP8f-m-V1",
            "DS31-GCaMP8m-m-V1",
            "DS32-GCaMP8s-m-V1",
            "DS33-Interneurons2023-m-V1",
            "DS40-GCaMP6s-spinal-cord-excitatory",
            "DS41-GCaMP6s-spinal-cord-inhibitory",
            "X-DS09-GCaMP6f-m-V1",
        ],
    )
    @pytest.mark.parametrize("ncell", [1, None])
    @pytest.mark.parametrize("nfm", [None])
    @pytest.mark.line_profile.with_args(pipeline_cnmf)
    def test_demo_pipeline_realds_cnmf(
        self,
        est_noise_freq,
        est_add_lag,
        dsname,
        ncell,
        nfm,
        results_bag,
    ):
        # act
        Y, S_true, ap_df, fluo_df = fixt_realds(dsname, ncell, nfm)
        C_cnmf, S_cnmf, tau_cnmf = pipeline_cnmf(
            np.atleast_2d(Y),
            est_noise_freq=est_noise_freq,
            est_use_smooth=False,
            est_add_lag=est_add_lag,
            sps_penal=0,
        )
        # save results
        res_df = []
        for iu, uid in enumerate(np.atleast_1d(Y.coords["unit_id"])):
            s_true = S_true[iu, :]
            cur_s = S_cnmf[iu, :]
            tau_d, tau_r = tau_cnmf[iu, :]
            try:
                (dhm0, dhm1), _ = find_dhm(
                    True, np.array([tau_d, tau_r]), np.array([1, -1])
                )
            except AssertionError:
                dhm0, dhm1 = 0, 0
            corr_dtw = dtw_corr(s_true, cur_s)
            for qthres in [0.01, 0.05, 0.1, 0.2, 0.5]:
                sb = np.around(cur_s / (qthres * cur_s.max())).astype(int)
                if len(ap_df) > 0:
                    cur_ap = ap_df.loc[uid]
                    cur_fluo = fluo_df.loc[uid]
                    sb_idx = nzidx_int(sb)
                    t_sb = np.interp(sb_idx, cur_fluo["frame"], cur_fluo["fluo_time"])
                    t_ap = cur_ap["ap_time"]
                    mdist, f1, prec, rec = assignment_distance(
                        t_ref=np.atleast_1d(t_ap),
                        t_slv=np.atleast_1d(t_sb),
                        tdist_thres=1,
                    )
                    corr_raw = np.corrcoef(s_true, cur_s)[0, 1]
                    corr_gs = np.corrcoef(
                        gaussian_filter1d(s_true, 1), gaussian_filter1d(cur_s, 1)
                    )[0, 1]
                else:
                    mdist, f1, prec, rec, corr_raw, corr_gs = (
                        np.nan,
                        0,
                        0,
                        0,
                        corr_raw,
                        corr_gs,
                    )
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
                                "corr_raw": corr_raw,
                                "corr_gs": corr_gs,
                                "corr_dtw": corr_dtw,
                            }
                        ]
                    )
                )
        res_df = pd.concat(res_df, ignore_index=True)
        results_bag.data = res_df

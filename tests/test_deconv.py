import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from indeca.AR_kernel import AR2tau, tau2AR
from indeca.deconv import DeconvBin

from .conftest import fixt_deconv, fixt_realds, fixt_y
from .testing_utils.metrics import (
    assignment_distance,
    compute_f1_metrics,
    df_assign_metadata,
)
from .testing_utils.plotting import plot_met_ROC_scale, plot_met_ROC_thres, plot_traces


class TestDeconvBin:
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize(
        "backend,upsamp", [("osqp", 1), ("osqp", 2), ("osqp", 5), ("cvxpy", 1)]
    )
    def test_solve(self, taus, rand_seed, backend, upsamp, eq_atol, test_fig_path_html):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus,
            backend=backend,
            rand_seed=rand_seed,
            upsamp=upsamp,
            deconv_kws={"Hlim": None},
        )
        R = deconv.R.value if backend == "cvxpy" else deconv.R
        s_solve, b_solve = deconv.solve(amp_constraint=False, pks_polish=True)
        c_solve = deconv.H @ s_solve
        c_solve_R = R @ c_solve
        # plotting
        fig = go.Figure()
        fig.add_traces(
            plot_traces(
                {
                    "c": c,
                    "s": s,
                    "c_org": c_org,
                    "s_org": s_org,
                    "s_solve": s_solve,
                    "c_solve": c_solve,
                    "c_solve_R": c_solve_R,
                }
            )
        )
        fig.write_html(test_fig_path_html)
        # assertion
        assert np.isclose(b_solve, 0, atol=eq_atol)
        assert np.isclose(s_org, s_solve, atol=eq_atol).all()

    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize("upsamp", [1])
    def test_masking(self, taus, rand_seed, upsamp, eq_atol, test_fig_path_html):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus, rand_seed=rand_seed, upsamp=upsamp, deconv_kws={"Hlim": None}
        )
        s_nomsk, b_nomsk = deconv._solve(amp_constraint=False)
        c_nomsk = deconv.H @ s_nomsk
        deconv._update_mask()
        s_msk, b_msk = deconv._solve(amp_constraint=False)
        c_msk = deconv.H @ s_msk
        s_msk = deconv._pad_s(s_msk)
        c_msk = deconv._pad_c(c_msk)
        # plotting
        fig = go.Figure()
        fig.add_traces(
            plot_traces(
                {
                    "c": c,
                    "s": s,
                    "c_org": c_org,
                    "s_org": s_org,
                    "s_nomsk": s_nomsk,
                    "c_nomsk": c_nomsk,
                    "s_msk": s_msk,
                    "c_msk": c_msk,
                }
            )
        )
        fig.write_html(test_fig_path_html)
        # assertion
        assert np.isclose(b_nomsk, 0, atol=eq_atol)
        assert np.isclose(b_msk, 0, atol=eq_atol)
        assert set(np.where(s)[0]).issubset(set(deconv.nzidx_s))
        assert np.isclose(s_org, s_nomsk, atol=eq_atol).all()
        assert np.isclose(s_org, s_msk, atol=eq_atol).all()

    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize(
        "rand_seed",
        list(range(3))
        + [pytest.param(i, marks=pytest.mark.slow) for i in range(3, 15)],
    )
    @pytest.mark.parametrize(
        "upsamp",
        [1, 2, 3] + [pytest.param(i, marks=pytest.mark.slow) for i in range(4, 11)],
    )
    @pytest.mark.parametrize(
        "upsamp_y",
        [1, 2, 3] + [pytest.param(i, marks=pytest.mark.slow) for i in range(4, 11)],
    )
    @pytest.mark.parametrize("ns_lev", [0, 0.2, 0.5])
    def test_solve_thres(
        self, taus, rand_seed, upsamp, upsamp_y, ns_lev, test_fig_path_html, results_bag
    ):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus,
            rand_seed=rand_seed,
            upsamp=upsamp,
            upsamp_y=upsamp_y,
            ns_lev=ns_lev,
        )
        s_bin, c_bin, scl, err, intm = deconv.solve_thres(
            scaling=False, return_intm=True, pks_polish=True
        )
        s_direct = intm[0]
        s_bin = s_bin.astype(float)
        mdist, f1, precs, recall = assignment_distance(
            s_ref=s_org, s_slv=s_bin, tdist_thres=5, include_range=(0, len(s_org) - 5)
        )
        # plotting
        fig = go.Figure()
        fig.add_traces(
            plot_traces(
                {
                    "y": y,
                    "c": c,
                    "s": s,
                    "s_solve": deconv.R @ s_bin,
                    "c_solve": deconv.R @ c_bin * deconv.scale,
                    "c_org": c_org,
                    "s_org": s_org,
                    "c_bin": c_bin,
                    "s_bin": s_bin,
                    "s_direct": s_direct,
                }
            )
        )
        fig.write_html(test_fig_path_html)
        # save results
        dat = pd.DataFrame(
            [
                {
                    "tau_d": taus[0],
                    "tau_r": taus[1],
                    "mdist": mdist,
                    "f1": f1,
                    "precs": precs,
                    "recall": recall,
                }
            ]
        )
        results_bag.data = dat
        # assert
        if upsamp == upsamp_y == 1 and ns_lev <= 0.2:
            assert f1 == 1
            assert mdist == 0
        else:
            assert f1 >= 0.6
            assert mdist <= max(upsamp, upsamp_y)

    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize("penal_scaling", [True, False])
    def test_solve_penal(
        self, taus, rand_seed, penal_scaling, test_fig_path_html, results_bag
    ):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus, rand_seed=rand_seed
        )
        s_free, _ = deconv.solve(amp_constraint=False)
        scl_init = np.ptp(s_free)
        deconv.update(scale=scl_init)
        opt_s, opt_c, scl_slv, obj, pn_slv, intm = deconv.solve_penal(
            scaling=penal_scaling, return_intm=True
        )
        s_slv_ma = intm[0]
        s_bin, c_bin, s_slv = np.zeros(deconv.T), np.zeros(deconv.T), np.zeros(deconv.T)
        s_bin[deconv.nzidx_s] = opt_s
        c_bin[deconv.nzidx_c] = opt_c
        s_slv[deconv.nzidx_s] = s_slv_ma
        deconv._reset_cache()
        deconv._reset_mask()
        s_bin = s_bin.astype(float)
        # plotting
        fig = go.Figure()
        fig.add_traces(
            plot_traces(
                {
                    "y": y,
                    "c": c,
                    "s": s,
                    "s_solve": deconv.R @ s_bin,
                    "c_solve": deconv.R @ c_bin,
                    "c_org": c_org,
                    "s_org": s_org,
                    "c_bin": c_bin,
                    "s_bin": s_bin,
                    "s_direct": s_slv,
                }
            )
        )
        fig.write_html(test_fig_path_html)
        # assert
        assert (s_bin == s).all()

    @pytest.mark.parametrize(
        "y_len", [1000, pytest.param(10000, marks=pytest.mark.slow)]
    )
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("upsamp", [1])
    @pytest.mark.parametrize("ns_lev", [0, 0.2, 0.5])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize("penalty", [None])
    @pytest.mark.parametrize("err_weighting", [None, "adaptive"])
    @pytest.mark.parametrize("obj_crit", [None])
    def test_solve_scale(
        self,
        y_len,
        taus,
        upsamp,
        ns_lev,
        rand_seed,
        penalty,
        err_weighting,
        obj_crit,
        results_bag,
        test_fig_path_svg,
        test_fig_path_html,
    ):
        # act
        y, c_true, c_org, s_true, s_org, scale = fixt_y(
            y_len=y_len,
            taus=taus,
            upsamp=upsamp,
            rand_seed=rand_seed,
            ns_lev=ns_lev,
            y_scaling=True,
        )
        taus_up = np.array(taus) * upsamp
        _, _, p = AR2tau(*tau2AR(*taus_up), solve_amp=True)
        deconv = DeconvBin(
            y=y,
            tau=taus,
            ps=(p, -p),
            penal=penalty,
            err_weighting=err_weighting,
        )
        (
            opt_s,
            opt_c,
            cur_scl,
            cur_obj,
            err_rel,
            nnz,
            cur_penal,
            iterdf,
        ) = deconv.solve_scale(return_met=True, obj_crit=obj_crit)
        results_bag.data = iterdf
        # plotting
        fig = go.Figure()
        fig.add_traces(
            plot_traces(
                {
                    "y": y.squeeze(),
                    "s_true": s_true.squeeze(),
                    "c_true": c_true.squeeze(),
                    "opt_s": opt_s.squeeze(),
                    "opt_c": opt_c.squeeze(),
                }
            )
        )
        fig.write_html(test_fig_path_html)
        # assertion
        mdist, f1, prec, rec = assignment_distance(
            s_ref=s_true, s_slv=opt_s, tdist_thres=3
        )
        if ns_lev == 0:
            assert np.isclose(np.array(iterdf["scale"])[-1], scale, atol=1e-2)
        if ns_lev < 0.5:
            assert f1 >= 0.99


@pytest.mark.slow
class TestDemoDeconv:
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(15))
    @pytest.mark.parametrize("upsamp", [1, 2, 5])
    @pytest.mark.parametrize("ns_lev", [0, 0.2, 0.5])
    @pytest.mark.parametrize("thres_scaling", [True])
    def test_demo_solve_thres(
        self,
        taus,
        rand_seed,
        upsamp,
        ns_lev,
        thres_scaling,
        test_fig_path_svg,
        results_bag,
    ):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus, rand_seed=rand_seed, upsamp=upsamp, ns_lev=ns_lev
        )
        s_bin, c_bin, scl, err, intm = deconv.solve_thres(
            scaling=thres_scaling, return_intm=True
        )
        s_slv, thres, svals, cvals, yfvals, scals, objs, opt_idx = intm
        # save results
        metdf = compute_f1_metrics(
            s_org,
            svals,
            {"objs": objs, "scals": scals, "thres": thres, "opt_idx": opt_idx},
            tdist_thres=3,
        )
        results_bag.data = metdf
        # plotting
        fig = plot_met_ROC_thres(metdf)
        fig.savefig(test_fig_path_svg)

    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(15))
    @pytest.mark.parametrize("upsamp", [1, 2])
    @pytest.mark.parametrize("ns_lev", [0, 0.2, 0.5])
    @pytest.mark.parametrize("y_scaling", [False])
    def test_demo_solve_penal(
        self, taus, rand_seed, upsamp, ns_lev, y_scaling, test_fig_path_svg, results_bag
    ):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus,
            rand_seed=rand_seed,
            upsamp=upsamp,
            ns_lev=ns_lev,
            y_scaling=y_scaling,
        )
        _, _, _, _, intm_free = deconv.solve_thres(
            scaling=False, amp_constraint=False, return_intm=True
        )
        _, _, _, _, intm_nopn = deconv.solve_thres(scaling=True, return_intm=True)
        _, _, _, _, opt_penal, intm_pn = deconv.solve_penal(
            scaling=True, return_intm=True
        )
        # save results
        intms = {"CNMF": intm_free, "No Penalty": intm_nopn, "Penalty": intm_pn}
        metdf = []
        for grp, cur_intm in intms.items():
            if grp == "Penalty":
                cur_svals = []
                oidx = intm_pn[7]
                for sv in intm_pn[2]:
                    s_pad = np.zeros(deconv.T)
                    s_pad[deconv.nzidx_s] = sv
                    cur_svals.append(s_pad)
            else:
                cur_svals = cur_intm[2]
            cur_met = compute_f1_metrics(
                s_org,
                cur_svals,
                {
                    "group": grp,
                    "thres": cur_intm[1],
                    "scals": cur_intm[5],
                    "objs": cur_intm[6],
                    "penal": opt_penal if grp == "Penalty" else 0,
                    "opt_idx": cur_intm[7],
                },
                tdist_thres=2,
            )
            metdf.append(cur_met)
        metdf = pd.concat(metdf, ignore_index=True)
        metdf = df_assign_metadata(
            metdf,
            {"tau_d": taus[0], "tau_r": taus[1]},
        )
        results_bag.data = metdf
        # plotting
        fig = plot_met_ROC_thres(metdf, grad_color=False)
        fig.savefig(test_fig_path_svg)
        # assertion
        if ns_lev == 0 and upsamp == 1:
            assert (cur_svals[oidx][:-1] == s[:-1]).all()

    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("upsamp", [1])
    @pytest.mark.parametrize("ns_lev", [0, 0.2, 0.5])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize("penalty", [None])
    @pytest.mark.parametrize("err_weighting", [None, "adaptive"])
    @pytest.mark.parametrize("obj_crit", [None])
    def test_demo_solve_scale(
        self,
        taus,
        upsamp,
        ns_lev,
        rand_seed,
        penalty,
        err_weighting,
        obj_crit,
        test_fig_path_svg,
        test_fig_path_html,
    ):
        # act
        y, c_true, c_org, s_true, s_org, scale = fixt_y(
            taus=taus, upsamp=upsamp, rand_seed=rand_seed, ns_lev=ns_lev
        )
        taus_up = np.array(taus) * upsamp
        _, _, p = AR2tau(*tau2AR(*taus_up), solve_amp=True)
        deconv = DeconvBin(
            y=y,
            tau=taus,
            ps=(p, -p),
            penal=penalty,
            err_weighting=err_weighting,
        )
        (
            opt_s,
            opt_c,
            cur_scl,
            cur_obj,
            err_rel,
            nnz,
            cur_penal,
            iterdf,
        ) = deconv.solve_scale(return_met=True, obj_crit=obj_crit)
        deconv.update(update_weighting=True)
        err_wt = deconv.err_wt.squeeze()
        deconv.update(update_weighting=True, clear_weighting=True, scale=1)
        deconv._reset_mask()
        deconv._reset_cache()
        s_free, b_free = deconv.solve(amp_constraint=False)
        scl_ub = np.ptp(s_free)
        res_df = []
        for scl in np.linspace(0, scl_ub, 100)[1:]:
            deconv.update(scale=scl)
            sbin, cbin, _, _ = deconv.solve_thres(scaling=False, obj_crit=obj_crit)
            deconv.err_wt = err_wt
            obj = deconv._compute_err(s=sbin, obj_crit=obj_crit)
            deconv.err_wt = np.ones_like(err_wt)
            mdist, f1, prec, rec = assignment_distance(
                s_ref=s_true, s_slv=sbin, tdist_thres=3
            )
            res_df.append(
                pd.DataFrame(
                    [
                        {
                            "scale": scl,
                            "objs": obj,
                            "mdist": mdist,
                            "f1": f1,
                            "prec": prec,
                            "recall": rec,
                        }
                    ]
                )
            )
        res_df = pd.concat(res_df, ignore_index=True)
        # plotting
        fig = plot_met_ROC_scale(res_df, iterdf, cur_scl)
        fig.savefig(test_fig_path_svg)
        fig = go.Figure()
        fig.add_traces(
            plot_traces(
                {
                    "y": y.squeeze(),
                    "s_true": s_true.squeeze(),
                    "c_true": c_true.squeeze(),
                    "opt_s": opt_s.squeeze(),
                    "opt_c": opt_c.squeeze(),
                    "err_wt": err_wt,
                }
            )
        )
        fig.write_html(test_fig_path_html)

    @pytest.mark.parametrize("upsamp", [1])
    @pytest.mark.parametrize("ar_kn_len", [100])
    @pytest.mark.parametrize("est_noise_freq", [None])
    @pytest.mark.parametrize("est_add_lag", [10])
    @pytest.mark.parametrize("dsname", ["X-DS09-GCaMP6f-m-V1"])
    @pytest.mark.parametrize("taus", [(21.18, 7.23)])
    @pytest.mark.parametrize("ncell", [1])
    @pytest.mark.parametrize("nfm", [None])
    @pytest.mark.parametrize("penalty", [None])
    @pytest.mark.parametrize("err_weighting", [None, "adaptive"])
    @pytest.mark.parametrize("obj_crit", [None])
    def test_demo_solve_scale_realds(
        self,
        upsamp,
        ar_kn_len,
        est_noise_freq,
        est_add_lag,
        dsname,
        taus,
        ncell,
        nfm,
        penalty,
        err_weighting,
        obj_crit,
        test_fig_path_svg,
        test_fig_path_html,
    ):
        # act
        Y, S_true, ap_df, fluo_df = fixt_realds(dsname, ncell=ncell, nfm=nfm)
        theta = tau2AR(taus[0], taus[1])
        _, _, p = AR2tau(theta[0], theta[1], solve_amp=True)
        deconv = DeconvBin(
            y=Y.squeeze(),
            tau=taus,
            ps=(p, -p),
            penal=penalty,
            err_weighting=err_weighting,
            use_base=True,
        )
        opt_s, opt_c, cur_scl, cur_obj, cur_penal, iterdf = deconv.solve_scale(
            return_met=True, obj_crit=obj_crit
        )
        deconv.update(update_weighting=True)
        err_wt = deconv.err_wt.squeeze()
        deconv.update(update_weighting=True, clear_weighting=True, scale=1)
        deconv._reset_mask()
        deconv._reset_cache()
        s_free, b_free = deconv.solve(amp_constraint=False)
        scl_ub = np.ptp(s_free)
        res_df = []
        for scl in np.linspace(0, scl_ub, 100)[1:]:
            deconv.update(scale=scl)
            sbin, cbin, _, _ = deconv.solve_thres(scaling=False, obj_crit=obj_crit)
            deconv.err_wt = err_wt
            obj = deconv._compute_err(s=sbin, obj_crit=obj_crit)
            deconv.err_wt = np.ones_like(err_wt)
            sb_idx = np.where(sbin)[0] / upsamp
            t_sb = np.interp(sb_idx, fluo_df["frame"], fluo_df["fluo_time"])
            t_ap = ap_df["ap_time"]
            mdist, f1, prec, rec = assignment_distance(
                t_ref=np.atleast_1d(t_ap), t_slv=np.atleast_1d(t_sb), tdist_thres=1
            )
            res_df.append(
                pd.DataFrame(
                    [
                        {
                            "scale": scl,
                            "objs": obj,
                            "mdist": mdist,
                            "f1": f1,
                            "prec": prec,
                            "recall": rec,
                        }
                    ]
                )
            )
        res_df = pd.concat(res_df, ignore_index=True)
        # plotting
        fig = plot_met_ROC_scale(res_df, iterdf, cur_scl)
        fig.savefig(test_fig_path_svg)
        fig = go.Figure()
        fig.add_traces(
            plot_traces(
                {
                    "y": Y.squeeze(),
                    "s_true": S_true.squeeze(),
                    "opt_s": opt_s.squeeze(),
                    "opt_c": opt_c.squeeze(),
                    "err_wt": err_wt,
                }
            )
        )
        fig.write_html(test_fig_path_html)

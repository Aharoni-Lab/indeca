import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from .conftest import fixt_deconv
from .testing_utils.metrics import (
    assignment_distance,
    compute_metrics,
    df_assign_metadata,
)
from .testing_utils.plotting import plot_met_ROC, plot_traces


class TestDeconvBin:
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize(
        "backend,upsamp", [("osqp", 1), ("osqp", 2), ("osqp", 5), ("cvxpy", 1)]
    )
    def test_solve(self, taus, rand_seed, backend, upsamp, eq_atol, test_fig_path_html):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus, backend=backend, rand_seed=rand_seed, upsamp=upsamp
        )
        R = deconv.R.value if backend == "cvxpy" else deconv.R
        s_solve, b_solve = deconv.solve(amp_constraint=False)
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
    @pytest.mark.parametrize("upsamp", [1, 2, 5])
    @pytest.mark.parametrize("upsamp_y", [1, 2, 5])
    @pytest.mark.parametrize(
        "ns_lev",
        [
            0,
            pytest.param(0.2, marks=pytest.mark.xfail),
            pytest.param(0.5, marks=pytest.mark.xfail),
        ],
    )
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
            scaling=False, return_intm=True
        )
        s_direct = intm[0]
        s_bin = s_bin.astype(float)
        mdist, f1, precs, recall = assignment_distance(
            s_ref=s_org, s_slv=s_bin, tdist_thres=5
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
        if upsamp == upsamp_y:  # upsample factor matches ground truth
            assert mdist <= 1
            assert recall >= 0.8
            assert precs >= 0.8
        elif upsamp < upsamp_y:  # upsample factor smaller than ground truth
            assert mdist <= upsamp_y / upsamp
            assert recall >= 0.95
        else:  # upsample factor larger than ground truth
            assert mdist <= 1
            assert precs >= 0.95

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
    ):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus, rand_seed=rand_seed, upsamp=upsamp, ns_lev=ns_lev
        )
        s_bin, c_bin, scl, err, intm = deconv.solve_thres(
            scaling=thres_scaling, return_intm=True
        )
        s_slv, thres, svals, cvals, yfvals, scals, objs, opt_idx = intm
        # plotting
        metdf = compute_metrics(
            s_org,
            svals,
            {"objs": objs, "scals": scals, "thres": thres, "opt_idx": opt_idx},
            tdist_thres=3,
        )
        fig = plot_met_ROC(metdf)
        fig.savefig(test_fig_path_svg)

    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(15))
    @pytest.mark.parametrize("upsamp", [1, 2])
    @pytest.mark.parametrize("ns_lev", [0, 0.2, 0.5])
    @pytest.mark.parametrize("y_scaling", [True])
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
        s_free, _ = deconv.solve(amp_constraint=False)
        scl_init = np.ptp(s_free)
        deconv.update(scale=scl_init)
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
            cur_met = compute_metrics(
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
        fig = plot_met_ROC(metdf, grad_color=False)
        fig.savefig(test_fig_path_svg)
        # assertion
        if ns_lev == 0 and upsamp == 1:
            assert (cur_svals[oidx] == s).all()

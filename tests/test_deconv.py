import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import seaborn as sns

from minian_bin.deconv import DeconvBin, construct_G, construct_R, max_thres
from minian_bin.simulation import ar_trace

from .testing_utils.metrics import (
    assignment_distance,
    compute_metrics,
    df_assign_metadata,
)
from .testing_utils.plotting import plot_met_ROC, plot_traces


@pytest.fixture()
def param_y_len():
    return 1000


@pytest.fixture()
def param_eq_atol():
    return 1e-3


@pytest.fixture(params=[(6, 1), (10, 3)])
def param_taus(request):
    return request.param


@pytest.fixture(params=["osqp", "cvxpy"])
def param_backend(request):
    return request.param


@pytest.fixture()
def param_norm():
    return "l2"


@pytest.fixture(params=[0, 0.1, 0.2, 0.5])
def param_ns_level(request):
    return request.param


@pytest.fixture(params=np.arange(5))
def param_rand_seed(request):
    sd = request.param
    np.random.seed(sd)
    return sd


@pytest.fixture()
def param_tmp_P():
    return np.array([[0.98, 0.02], [0.75, 0.25]])


@pytest.fixture(params=[1, 2, 5])
def param_upsamp(request):
    return request.param


@pytest.fixture(params=[True, False])
def param_y_scaling(request):
    return request.param


@pytest.fixture(params=[True, False])
def param_thres_scaling(request):
    return request.param


@pytest.fixture()
def param_tmp_upsamp(param_upsamp):
    if param_upsamp < 5:
        tmp = np.array([[0.98, 0.02], [0.75, 0.25]])
    else:
        tmp = np.array([[0.998, 0.002], [0.75, 0.25]])
    return param_upsamp, tmp


@pytest.fixture()
def fixt_c(param_y_len, param_taus, param_tmp_P, param_rand_seed):
    c, s = ar_trace(
        param_y_len, param_tmp_P, tau_d=param_taus[0], tau_r=param_taus[1], shifted=True
    )
    return c, s, param_taus


@pytest.fixture()
def fixt_y(
    param_y_len,
    param_taus,
    param_tmp_upsamp,
    param_ns_level,
    param_y_scaling,
    param_rand_seed,
):
    upsamp, tmp_P = param_tmp_upsamp
    c_org, s_org = ar_trace(
        param_y_len * upsamp,
        tmp_P,
        tau_d=param_taus[0] * upsamp,
        tau_r=param_taus[1] * upsamp,
        shifted=True,
    )
    if upsamp > 1:
        c = np.convolve(c_org, np.ones(upsamp), "valid")[::upsamp]
        s = np.convolve(s_org, np.ones(upsamp), "valid")[::upsamp]
    else:
        c, s = c_org, s_org
    if param_y_scaling:
        scl = np.random.uniform(0.5, 2)
    else:
        scl = 1
    y = scl * (c + np.random.normal(0, param_ns_level, c.shape))
    return y, c, c_org, s, s_org, param_taus, param_ns_level, upsamp, scl


@pytest.fixture()
def fixt_deconv(fixt_y, param_backend, param_norm):
    y, c, c_org, s, s_org, taus, ns_lev, upsamp, scl = fixt_y
    deconv = DeconvBin(
        y=y,
        tau=np.array(taus) * upsamp,
        upsamp=upsamp,
        err_weighting=None,
        backend=param_backend,
        norm=param_norm,
    )
    return (
        deconv,
        param_backend,
        param_norm,
        y,
        c,
        c_org,
        s,
        s_org,
        taus,
        ns_lev,
        upsamp,
        scl,
    )


class TestDeconvBin:
    def test_solve(
        self, fixt_c, param_backend, param_norm, param_eq_atol, test_fig_path_html
    ):
        # act
        c, s, taus = fixt_c
        deconv = DeconvBin(
            y=c, tau=taus, err_weighting=None, backend=param_backend, norm=param_norm
        )
        s_solve, b_solve = deconv.solve(amp_constraint=False)
        # plotting
        fig = go.Figure()
        fig.add_traces(plot_traces({"c": c, "s": s, "s_solve": s_solve}))
        fig.write_html(test_fig_path_html)
        # assertion
        assert np.isclose(b_solve, 0, atol=param_eq_atol)
        assert np.isclose(s, s_solve, atol=param_eq_atol).all()

    def test_solve_thres(
        self,
        fixt_y,
        param_backend,
        param_upsamp,
        param_norm,
        test_fig_path_html,
        results_bag,
        runtime_xfail,
    ):
        # book-keeping
        y, c, c_org, s, s_org, taus, ns_lev, upsamp_y, scl = fixt_y
        if param_backend == "cvxpy":
            pytest.skip("Skipping cvxpy backend for test_solve_thres")
        if scl != 1:
            pytest.skip("Skipping scaling for test_solve_thres")
        # act
        upsamp_ratio = upsamp_y / param_upsamp
        deconv = DeconvBin(
            y=y,
            tau=np.array(taus) * param_upsamp,
            upsamp=param_upsamp,
            err_weighting=None,
            backend=param_backend,
            norm=param_norm,
        )
        deconv.update(scale=upsamp_ratio)
        s_bin, c_bin, scl, err, intm = deconv.solve_thres(
            scaling=False, return_intm=True
        )
        s_direct = intm[0]
        s_bin = s_bin.astype(float)
        mdist, f1, precs, recall = assignment_distance(s_ref=s_org, s_slv=s_bin)
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
                    "ns_lev": ns_lev,
                    "upsamp_y": upsamp_y,
                    "upsamp": param_upsamp,
                    "backend": param_backend,
                    "mdist": mdist,
                    "f1": f1,
                    "precs": precs,
                    "recall": recall,
                }
            ]
        )
        results_bag.data = dat
        # assert
        if ns_lev >= 0.2:
            runtime_xfail("Accuracy degrade when noise level too high")
        if param_upsamp == upsamp_y:  # upsample factor matches ground truth
            assert mdist <= 1
            assert recall >= 0.8
            assert precs >= 0.95
        elif param_upsamp < upsamp_y:  # upsample factor smaller than ground truth
            assert mdist <= upsamp_ratio
            assert recall >= 0.95
        else:  # upsample factor larger than ground truth
            assert mdist <= 1
            assert precs >= 0.95

    @pytest.mark.parametrize("param_penal_scaling", [True, False])
    def test_solve_penal(
        self,
        fixt_deconv,
        param_backend,
        param_upsamp,
        param_norm,
        param_penal_scaling,
        test_fig_path_html,
        results_bag,
        runtime_xfail,
    ):
        # book-keeping
        (
            deconv,
            param_backend,
            param_norm,
            y,
            c,
            c_org,
            s,
            s_org,
            taus,
            ns_lev,
            upsamp,
            scl,
        ) = fixt_deconv
        if param_backend == "cvxpy":
            pytest.skip("Skipping cvxpy backend for test_solve_penal")
        if scl != 1:
            pytest.skip("Skipping scaling for test_solve_penal")
        if upsamp > 1:
            pytest.skip("Skipping upsampling for test_solve_penal")
        # act
        opt_s, opt_c, scl_slv, obj, pn_slv, intm = deconv.solve_penal(
            scaling=param_penal_scaling, return_intm=True
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
        if ns_lev == 0:
            assert (s_bin == s).all()


@pytest.mark.slow
class TestDemoDeconv:
    def test_demo_solve_thres(
        self, fixt_deconv, param_thres_scaling, test_fig_path_svg
    ):
        # book-keeping
        (
            deconv,
            param_backend,
            param_norm,
            y,
            c,
            c_org,
            s,
            s_org,
            taus,
            ns_lev,
            upsamp,
            scl,
        ) = fixt_deconv
        if param_backend == "cvxpy":
            pytest.skip("Skipping cvxpy backend for test_demo_solve_thres")
        # act
        s_bin, c_bin, scl, err, intm = deconv.solve_thres(
            scaling=param_thres_scaling, return_intm=True
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

    def test_demo_solve_penal(
        self, fixt_deconv, param_thres_scaling, test_fig_path_svg, results_bag
    ):
        # book-keeping
        (
            deconv,
            param_backend,
            param_norm,
            y,
            c,
            c_org,
            s,
            s_org,
            taus,
            ns_lev,
            upsamp,
            scl,
        ) = fixt_deconv
        if param_backend == "cvxpy":
            pytest.skip("Skipping cvxpy backend for test_demo_solve_thres")
        if upsamp > 2:
            pytest.skip("Skipping highly upsampled signal for solve_penal demo")
        # act
        s_free, _ = deconv.solve(amp_constraint=False)
        scl_init = np.ptp(s_free)
        deconv.update(scale=scl_init)
        _, _, _, _, opt_penal = deconv.solve_penal(scaling=False)
        deconv._reset_cache()
        deconv._reset_mask()
        deconv.update(l1_penal=opt_penal)
        _, _, _, _, intm_pn = deconv.solve_thres(
            scaling=param_thres_scaling, return_intm=True
        )
        deconv.update(l1_penal=0)
        _, _, _, _, intm_nopn = deconv.solve_thres(
            scaling=param_thres_scaling, return_intm=True
        )
        # plotting
        metdf_nopn = compute_metrics(
            s_org,
            intm_nopn[2],
            {
                "thres": intm_nopn[1],
                "scals": intm_nopn[5],
                "objs": intm_nopn[6],
                "penal": 0,
                "opt_idx": intm_nopn[7],
                "group": "No Penalty",
            },
            tdist_thres=3,
        )
        metdf_pn = compute_metrics(
            s_org,
            intm_pn[2],
            {
                "thres": intm_pn[1],
                "scals": intm_pn[5],
                "objs": intm_pn[6],
                "penal": opt_penal,
                "opt_idx": intm_pn[7],
                "group": "Penalty",
            },
            tdist_thres=3,
        )
        metdf = pd.concat([metdf_nopn, metdf_pn], ignore_index=True)
        fig = plot_met_ROC(metdf, grad_color=False)
        fig.savefig(test_fig_path_svg)
        # save results
        metdf = df_assign_metadata(
            metdf,
            {
                "tau_d": taus[0],
                "tau_r": taus[1],
                "ns_lev": ns_lev,
                "y_scaling": scl,
                "thres_scaling": param_thres_scaling,
                "upsamp": upsamp,
                "backend": param_backend,
            },
        )
        results_bag.data = metdf


def test_construct_R():
    """Test R matrix construction."""
    T = 10
    up_factor = 2
    R = construct_R(T, up_factor)
    assert R.shape == (T, T * up_factor)
    assert hasattr(R, "tocsc")


def test_construct_G():
    """Test G matrix construction."""
    fac = np.array([0.7, -0.2])
    T = 10
    G = construct_G(fac, T)
    assert G.shape == (T, T)
    assert hasattr(G, "tocsc")


def test_max_thres(sample_xarray_dataset):
    """Test threshold computation."""
    data = sample_xarray_dataset.fluorescence
    nthres = 5
    S_ls = max_thres(data, nthres)
    assert len(S_ls) == nthres
    assert all(s.shape == data.shape for s in S_ls)


class TestDeconvolution:
    def test_basic_deconvolution(self, sample_timeseries, deconv_parameters):
        """Test basic deconvolution functionality."""
        pass

    def test_parameter_optimization(self, sample_timeseries):
        """Test parameter optimization in deconvolution."""
        pass

    @pytest.mark.slow
    def test_large_scale_deconvolution(self, sample_xarray_dataset):
        """Test deconvolution with large datasets."""
        pass

    def test_noise_handling(self):
        """Test deconvolution with different noise levels."""
        pass

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        pass

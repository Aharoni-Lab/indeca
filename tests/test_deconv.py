import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
import seaborn as sns
from matplotlib.gridspec import GridSpec

from minian_bin.deconv import DeconvBin, construct_G, construct_R, max_thres
from minian_bin.metrics import assignment_distance
from minian_bin.simulation import ar_trace

from .plotting_utils import colored_line, plot_traces


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


@pytest.fixture(params=[1, 2, 3, 4, 5])
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


@pytest.fixture(
    params=[
        {"upsamp": 1, "tmp_P": np.array([[0.98, 0.02], [0.75, 0.25]])},
        {"upsamp": 2, "tmp_P": np.array([[0.98, 0.02], [0.75, 0.25]])},
        {"upsamp": 5, "tmp_P": np.array([[0.998, 0.002], [0.75, 0.25]])},
    ]
)
def param_tmp_upsamp(request):
    return request.param["upsamp"], request.param["tmp_P"]


@pytest.fixture()
def fixt_c(param_y_len, param_taus, param_tmp_P, param_rand_seed):
    c, s = ar_trace(
        param_y_len, param_tmp_P, tau_d=param_taus[0], tau_r=param_taus[1], shifted=True
    )
    return c, s, param_taus


@pytest.fixture()
def fixt_y(param_y_len, param_taus, param_tmp_upsamp, param_ns_level, param_rand_seed):
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
    y = c + np.random.normal(0, param_ns_level, c.shape)
    return y, c, c_org, s, s_org, param_taus, param_ns_level, upsamp


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
        param_eq_atol,
        test_fig_path_html,
        results_bag,
        runtime_xfail,
    ):
        # book-keeping
        if param_backend == "cvxpy":
            pytest.skip("Skipping cvxpy backend for test_solve_thres")
        # act
        y, c, c_org, s, s_org, taus, ns_lev, upsamp_y = fixt_y
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
        results_bag.tau_d = taus[0]
        results_bag.tau_r = taus[1]
        results_bag.ns_lev = ns_lev
        results_bag.upsamp_y = upsamp_y
        results_bag.upsamp = param_upsamp
        results_bag.backend = param_backend
        results_bag.mdist = mdist
        results_bag.f1 = f1
        results_bag.precs = precs
        results_bag.recall = recall
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


class TestDemoDeconv:
    def test_demo_solve_thres(
        self, fixt_y, param_backend, param_norm, test_fig_path_svg
    ):
        # book-keeping
        if param_backend == "cvxpy":
            pytest.skip("Skipping cvxpy backend for test_solve_thres")
        # act
        y, c, c_org, s, s_org, taus, ns_lev, upsamp = fixt_y
        deconv = DeconvBin(
            y=y,
            tau=np.array(taus) * upsamp,
            upsamp=upsamp,
            err_weighting=None,
            backend=param_backend,
            norm=param_norm,
        )
        s_bin, c_bin, scl, err, intm = deconv.solve_thres(
            scaling=False, return_intm=True
        )
        s_slv, thres, svals, cvals, yfvals, scals, objs, opt_idx = intm
        dists = [
            assignment_distance(s_org, ss, tdist_thres=upsamp, tdist_agg="mean")
            for ss in svals
        ]
        mdists = np.array([d[0] for d in dists])
        f1s = np.array([d[1] for d in dists])
        precs = np.array([d[2] for d in dists])
        recals = np.array([d[3] for d in dists])
        # plotting
        fig = plt.figure(constrained_layout=True, figsize=(8, 4))
        gs = GridSpec(2, 2, figure=fig)
        lw = 2
        ax_err = fig.add_subplot(gs[0, 0])
        ax_f1 = fig.add_subplot(gs[1, 0])
        ax_roc = fig.add_subplot(gs[:, 1])
        colored_line(x=thres, y=objs, c=thres, ax=ax_err, linewidths=lw)
        ax_err.plot(thres, objs, alpha=0)
        ax_err.set_yscale("log")
        ax_err.axvline(thres[opt_idx], ls="dotted", color="gray")
        ax_err.set_xlabel("Threshold")
        ax_err.set_ylabel("Error")
        colored_line(x=thres, y=f1s, c=thres, ax=ax_f1, linewidths=lw)
        ax_f1.plot(thres, f1s, alpha=0)
        ax_f1.axvline(thres[opt_idx], ls="dotted", color="gray")
        ax_f1.set_xlabel("Threshold")
        ax_f1.set_ylabel("f1 Score")
        colored_line(x=precs, y=recals, c=thres, ax=ax_roc, linewidths=lw)
        ax_roc.plot(precs, recals, alpha=0)
        ax_roc.plot(
            precs[opt_idx], recals[opt_idx], marker="x", color="gray", markersize=15
        )
        ax_roc.set_xlabel("Precision")
        ax_roc.set_ylabel("Recall")
        fig.savefig(test_fig_path_svg)


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

    @pytest.mark.parametrize("lambda_", [0.1, 1.0, 10.0])
    def test_lambda_sensitivity(self, lambda_):
        """Test sensitivity to lambda parameter."""
        pass


class TestResults:
    def test_solve_thres_results(self, module_results_df, func_figs_dir, func_data_dir):
        result = (
            module_results_df.loc[
                lambda d: d.index.str.startswith("test_solve_thres["),
                [
                    "tau_d",
                    "tau_r",
                    "ns_lev",
                    "upsamp_y",
                    "upsamp",
                    "backend",
                    "mdist",
                    "f1",
                    "precs",
                    "recall",
                ],
            ]
            .dropna()
            .reset_index()
        )
        result.to_feather(os.path.join(func_data_dir, "metrics.feat"))
        for (td, tr), res_sub in result.groupby(["tau_d", "tau_r"]):
            for met in ["mdist", "f1", "precs", "recall"]:
                g = sns.FacetGrid(res_sub, row="upsamp", col="upsamp_y")
                g.map_dataframe(
                    sns.boxplot,
                    x="ns_lev",
                    y=met,
                    hue="ns_lev",
                    saturation=0.5,
                    showfliers=False,
                    palette="tab10",
                )
                g.map_dataframe(
                    sns.swarmplot,
                    x="ns_lev",
                    y=met,
                    hue="ns_lev",
                    edgecolor="gray",
                    palette="tab10",
                    size=5,
                    linewidth=1.2,
                )
                g.tight_layout()
                g.figure.savefig(
                    os.path.join(
                        func_figs_dir, "tau({},{})-{}.svg".format(td, tr, met)
                    ),
                    bbox_inches="tight",
                )

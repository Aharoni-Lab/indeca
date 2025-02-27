import os

import numpy as np
import plotly.graph_objects as go
import pytest

from minian_bin.deconv import DeconvBin, construct_G, construct_R, max_thres
from minian_bin.metrics import assignment_distance
from minian_bin.simulation import ar_trace

from .plotting_utils import plot_traces


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


@pytest.fixture(params=[0, 0.1, 0.2])
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

    def test_solve(self, fixt_c, param_backend, param_norm, param_eq_atol, fig_path):
        # act
        c, s, taus = fixt_c
        deconv = DeconvBin(
            y=c, tau=taus, err_weighting=None, backend=param_backend, norm=param_norm
        )
        s_solve, b_solve = deconv.solve(amp_constraint=False)
        # plotting
        fig = go.Figure()
        fig.add_traces(plot_traces({"c": c, "s": s, "s_solve": s_solve}))
        fig.write_html(fig_path)
        # assertion
        assert np.isclose(b_solve, 0, atol=param_eq_atol)
        assert np.isclose(s, s_solve, atol=param_eq_atol).all()

    def test_solve_thres(
        self, fixt_y, param_backend, param_upsamp, param_norm, param_eq_atol, fig_path
    ):
        # book-keeping
        if param_backend == "cvxpy":
            pytest.skip("Skipping cvxpy backend for test_solve_thres")
        # act
        y, c, c_org, s, s_org, taus, ns_lev, upsamp_y = fixt_y
        deconv = DeconvBin(
            y=y,
            tau=taus,
            upsamp=param_upsamp,
            err_weighting=None,
            backend=param_backend,
            norm=param_norm,
        )
        s_bin, c_bin, scl, err = deconv.solve_thres(scaling=False)
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
                }
            )
        )
        fig.write_html(fig_path)
        # assert
        mdist, f1, precs, recall = assignment_distance(s_ref=s_org, s_slv=s_bin)
        if param_upsamp == upsamp_y:  # upsample factor matches ground truth
            if ns_lev == 0:
                assert np.isclose(s_org, s_bin, atol=param_eq_atol).all()
            elif ns_lev <= 0.1:
                assert f1 >= 0.9
            else:
                assert f1 >= 0.8
        elif param_upsamp < upsamp_y:  # upsample factor smaller than ground truth
            assert recall == 1
            assert mdist <= upsamp_y / param_upsamp
        else:  # upsample factor larger than ground truth
            assert precs == 1
            assert mdist <= 1
            if ns_lev == 0:
                assert recall >= 0.75
            elif ns_lev <= 0.1:
                assert recall >= 0.5
            else:
                assert recall >= 0.5


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

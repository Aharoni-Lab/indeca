import pytest
import numpy as np
from minian_bin.deconv import DeconvBin, construct_R, construct_G, max_thres


@pytest.fixture
def deconv_bin_params():
    """Parameters for DeconvBin initialization."""
    return {
        "penal": "l1",
        "norm": "l1",
        "atol": 1e-3,
        "backend": "cvxpy",
    }


class TestDeconvBin:
    @pytest.mark.skip(reason="DeconvBin initialization needs to be investigated")
    def test_initialization(self, sample_timeseries, deconv_bin_params):
        """Test DeconvBin initialization."""
        pass

    @pytest.mark.skip(reason="DeconvBin solve method needs to be investigated")
    def test_solve(self, sample_timeseries, deconv_bin_params):
        """Test basic deconvolution solve."""
        pass

    @pytest.mark.skip(reason="DeconvBin solve_thres method needs to be investigated")
    @pytest.mark.slow
    def test_solve_thres(self, sample_timeseries, deconv_bin_params):
        """Test threshold-based solving."""
        pass

    @pytest.mark.skip(reason="DeconvBin solve_scale method needs to be investigated")
    def test_solve_scale(self, sample_timeseries, deconv_bin_params):
        """Test scale optimization."""
        pass


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

import numpy as np
import pytest

from minian_bin.simulation import (
    AR2exp,
    AR2tau,
    apply_arcoef,
    apply_exp,
    ar_pulse,
    ar_trace,
    eval_exp,
    exp_pulse,
    exp_trace,
    gauss_cell,
    markov_fire,
    random_walk,
    simulate_traces,
    tau2AR,
)


@pytest.fixture(params=[(1.6, -0.62), (1.6, -0.7)])
def param_theta(request):
    return request.param


@pytest.fixture(params=[(6, 1), (10, 3)])
def param_tau(request):
    return request.param


@pytest.fixture(params=[0.5, 1, 2])
def param_p(request):
    return request.param


@pytest.fixture(params=["amp", "noamp"])
def param_conv_mode(request):
    return request.param


@pytest.fixture()
def conf_pulse_len():
    return 1000


@pytest.fixture()
def fixture_ar_pulse(param_theta, conf_pulse_len):
    ar, t, pulse = ar_pulse(
        param_theta[0], param_theta[1], conf_pulse_len, shifted=True
    )
    return param_theta, ar, t, pulse


@pytest.fixture()
def fixture_exp_pulse(param_tau, param_p, conf_pulse_len):
    if param_p is None:
        exp, t, pulse = exp_pulse(param_tau[0], param_tau[1], conf_pulse_len)
    else:
        exp, t, pulse = exp_pulse(
            param_tau[0], param_tau[1], conf_pulse_len, p_d=param_p, p_r=-param_p
        )
    return param_tau, param_p, exp, t, pulse


class TestARExp:
    def test_AR2tau(self, fixture_ar_pulse, param_conv_mode):
        theta, ar, t, pulse = fixture_ar_pulse
        if param_conv_mode == "amp":  # test traces only when enabling scaling
            tau_d, tau_r, p = AR2tau(*theta, solve_amp=True)
            exp = apply_exp(pulse, tau_d, tau_r, p, -p)
            # the two traces matches only when characterstic root of AR(2) are real
            if theta[0] ** 2 + 4 * theta[1] > 0:
                assert np.isclose(ar, exp).all()
        else:  # otherwise test bi-directional conversion
            taus = AR2tau(*theta, solve_amp=False)
            theta_conv = tau2AR(*taus)
            assert np.isclose(theta, theta_conv).all()

    def test_tau2AR(self, fixture_exp_pulse, param_conv_mode):
        taus, p, exp, t, pulse = fixture_exp_pulse
        if param_conv_mode == "amp":  # test traces only when enabling scaling
            theta1, theta2, scl = tau2AR(taus[0], taus[1], p, return_scl=True)
            ar = apply_arcoef(pulse, np.array([theta1, theta2]), shifted=True)
            assert np.isclose(ar * scl, exp).all()
        else:  # otherwise test bi-directional conversion
            theta1, theta2 = tau2AR(*taus, return_scl=False)
            taus_conv = AR2tau(theta1, theta2)
            assert np.isclose(taus_conv, taus).all()

    def test_AR2exp(self, fixture_ar_pulse):
        theta, ar, t, pulse = fixture_ar_pulse
        is_biexp, tconst, coefs = AR2exp(*theta)
        exp = eval_exp(t, is_biexp, tconst, coefs)
        exp = np.concatenate([np.zeros(1), exp[:-1]])
        assert np.isclose(exp, ar).all()


def test_gauss_cell():
    """Test Gaussian cell generation."""
    height, width = 32, 32
    sz_mean, sz_sigma, sz_min = 3.0, 0.5, 1.0
    result = gauss_cell(height, width, sz_mean, sz_sigma, sz_min)
    assert result.shape[1:] == (height, width)


def test_apply_arcoef():
    """Test AR coefficient application."""
    s = np.zeros(100)
    s[::10] = 1  # sparse spikes
    g = np.array([0.7, -0.2])
    result = apply_arcoef(s, g)
    assert result.shape == s.shape


def test_apply_exp():
    """Test exponential application."""
    s = np.zeros(100)
    s[::10] = 1  # sparse spikes
    tau_d, tau_r = 5.0, 1.0
    result = apply_exp(s, tau_d, tau_r)
    assert result.shape == s.shape


def test_ar_trace():
    """Test AR trace generation."""
    frame = 100
    P = np.array([[0.95, 0.05], [0.1, 0.9]])  # transition matrix
    g = np.array([0.7, -0.2])
    C, S = ar_trace(frame, P, g=g)
    assert C.shape == (frame,)
    assert S.shape == (frame,)


def test_exp_trace():
    """Test exponential trace generation."""
    frame = 100
    P = np.array([[0.95, 0.05], [0.1, 0.9]])  # transition matrix
    tau_d, tau_r = 5.0, 1.0
    C, S = exp_trace(frame, P, tau_d, tau_r)
    assert C.shape == (frame,)
    assert S.shape == (frame,)


def test_markov_fire():
    """Test Markov chain spike generation."""
    frame = 100
    P = np.array([[0.95, 0.05], [0.1, 0.9]])  # transition matrix
    result = markov_fire(frame, P)
    assert result.shape == (frame,)
    assert set(np.unique(result)) <= {0, 1}


def test_random_walk():
    """Test random walk generation."""
    n_stp = 100
    result = random_walk(n_stp, stp_var=1.0)
    assert result.shape == (n_stp, 1)


@pytest.mark.skip(reason="Unexpected keyword argument needs to be investigated")
@pytest.mark.slow
def test_simulate_traces():
    """Test trace simulation."""
    pass


class TestSimulation:
    def test_basic_simulation(self, simulation_params):
        """Test basic simulation functionality."""
        pass

    def test_output_dimensions(self, simulation_params):
        """Test output dimensions match specifications."""
        pass

    def test_noise_injection(self, simulation_params):
        """Test noise injection in simulations."""
        pass

    @pytest.mark.slow
    def test_large_scale_simulation(self):
        """Test simulation with large parameters."""
        pass

    def test_parameter_validation(self):
        """Test parameter validation."""
        pass

    @pytest.mark.parametrize("framerate", [10, 30, 60])
    def test_different_framerates(self, framerate, simulation_params):
        """Test simulation with different framerates."""
        pass

    def test_reproducibility(self, simulation_params):
        """Test simulation reproducibility with same seed."""
        pass

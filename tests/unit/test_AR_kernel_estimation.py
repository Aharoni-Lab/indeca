import numpy as np
import pytest

from indeca.core.AR_kernel import estimate_coefs
from indeca.core.simulation import AR2tau, exp_pulse, tau2AR

from tests.conftest import fixt_y

pytestmark = pytest.mark.unit

@pytest.mark.xfail(reason="yule walker estimation struggle to get accurate")
@pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
@pytest.mark.parametrize("rand_seed", np.arange(3))
def test_estimate_coef(taus, rand_seed):
    # act
    y, c, c_org, s, s_org, scale = fixt_y(taus=taus, rand_seed=rand_seed)
    theta, _ = estimate_coefs(y, p=2, noise_freq=None, use_smooth=False, add_lag=0)
    # assertion
    t0_true, t1_true, p_true = AR2tau(*tau2AR(*taus), solve_amp=True)
    t0_est, t1_est, p_est = AR2tau(*theta, solve_amp=True)
    ps_true, _, _ = exp_pulse(t0_true, t1_true, nsamp=100, p_d=p_true, p_r=-p_true)
    ps_est, _, _ = exp_pulse(t0_est, t1_est, nsamp=100, p_d=p_est, p_r=-p_est)
    assert np.isclose(ps_true, ps_est).all()
    assert np.isclose(theta, tau2AR(*taus)).all()

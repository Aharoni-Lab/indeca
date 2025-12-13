import numpy as np

from indeca.core.deconv import DeconvBin
from indeca.core.simulation import tau2AR


def test_G_matrix_matches_shifted_ar_difference_equation():
    """
    The legacy implementation constructs a (T x T) "G" operator such that:
      - s = G @ c
      - s[-1] == 0  (last row is zeros)
      - for t >= 2: s[t] = c[t+1] - theta0*c[t] - theta1*c[t-1]
      - for t == 1: s[1] = c[2] - theta0*c[1]
      - for t == 0: s[0] = c[1]

    This test guards against accidental dimensional/shift regressions in `G_org`.
    """
    T = 5
    theta = np.array(tau2AR(10.0, 3.0))

    deconv = DeconvBin(
        y_len=T,
        theta=theta,
        coef_len=3,
        backend="osqp",
        free_kernel=False,
        use_base=False,
        norm="l2",
        penal=None,
    )

    # Use the full, unmasked operator for determinism.
    deconv._reset_mask()
    G = deconv.solver.G_org
    assert G.shape == (T, T)

    # Build a c vector with the same boundary condition as the solver (c[0] == 0).
    rng = np.random.default_rng(0)
    c = rng.normal(size=T)
    c[0] = 0.0

    # `scipy.sparse_matrix @ np.ndarray` returns a dense np.ndarray (no `.todense()`).
    s = np.asarray(G @ c.reshape(-1, 1)).squeeze()

    # Last element must be exactly 0 due to bottom row of zeros.
    assert np.isclose(s[-1], 0.0)

    # Check the shifted AR-difference mapping on the first T-1 entries.
    th0, th1 = float(theta[0]), float(theta[1])
    expected = np.zeros(T)
    expected[0] = c[1]
    expected[1] = c[2] - th0 * c[1]
    expected[2] = c[3] - th0 * c[2] - th1 * c[1]
    expected[3] = c[4] - th0 * c[3] - th1 * c[2]
    expected[4] = 0.0

    assert np.allclose(s, expected)



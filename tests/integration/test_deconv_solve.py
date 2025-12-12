import numpy as np
import plotly.graph_objects as go
import pytest

from indeca.core.deconv import DeconvBin
from tests.conftest import fixt_deconv
from tests.testing_utils.plotting import plot_traces

pytestmark = pytest.mark.integration

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

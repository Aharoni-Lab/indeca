import numpy as np
import plotly.graph_objects as go
import pytest

from indeca.core.simulation import AR2tau, tau2AR
from indeca.core.deconv import DeconvBin
from tests.conftest import fixt_y
from tests.testing_utils.metrics import assignment_distance
from tests.testing_utils.plotting import plot_traces

pytestmark = pytest.mark.robustness

class TestDeconvBin:
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

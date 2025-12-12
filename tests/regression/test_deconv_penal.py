import numpy as np
import plotly.graph_objects as go
import pytest

from indeca.core.deconv import DeconvBin
from tests.conftest import fixt_deconv
from tests.testing_utils.plotting import plot_traces

pytestmark = pytest.mark.regression

class TestDeconvBin:
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

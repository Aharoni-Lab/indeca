import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from indeca.core.deconv import DeconvBin
from tests.conftest import fixt_deconv
from tests.testing_utils.metrics import assignment_distance
from tests.testing_utils.plotting import plot_traces

pytestmark = pytest.mark.regression


class TestDeconvBin:
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize(
        "rand_seed",
        list(range(3))
        + [pytest.param(i, marks=pytest.mark.slow) for i in range(3, 15)],
    )
    @pytest.mark.parametrize(
        "upsamp",
        [1, 2, 3] + [pytest.param(i, marks=pytest.mark.slow) for i in range(4, 11)],
    )
    @pytest.mark.parametrize(
        "upsamp_y",
        [1, 2, 3] + [pytest.param(i, marks=pytest.mark.slow) for i in range(4, 11)],
    )
    @pytest.mark.parametrize("ns_lev", [0, 0.2, 0.5])
    def test_solve_thres(
        self, taus, rand_seed, upsamp, upsamp_y, ns_lev, test_fig_path_html, results_bag
    ):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus,
            rand_seed=rand_seed,
            upsamp=upsamp,
            upsamp_y=upsamp_y,
            ns_lev=ns_lev,
        )
        s_bin, c_bin, scl, err, intm = deconv.solve_thres(
            scaling=False, return_intm=True, pks_polish=True
        )
        s_direct = intm[0]
        s_bin = s_bin.astype(float)
        ttol = max(upsamp_y / upsamp, upsamp, upsamp_y)
        mdist, f1, precs, recall = assignment_distance(
            s_ref=s_org,
            s_slv=s_bin,
            tdist_thres=ttol,
            include_range=(0, len(s_org) - max(int(ttol), 5)),
        )
        # plotting
        fig = go.Figure()
        fig.add_traces(
            plot_traces(
                {
                    "y": y,
                    "c": c,
                    "s": s,
                    "s_solve": deconv.R @ s_bin,
                    "c_solve": deconv.R @ c_bin * deconv.scale,
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
                    "mdist": mdist,
                    "f1": f1,
                    "precs": precs,
                    "recall": recall,
                }
            ]
        )
        results_bag.data = dat
        # assert
        if upsamp == upsamp_y == 1 and ns_lev <= 0.2:
            assert f1 == 1
            assert mdist == 0
        else:
            assert f1 >= 0.6
            assert mdist <= max(upsamp, upsamp_y)

import numpy as np
import pytest

from indeca.core.deconv import DeconvBin
from tests.conftest import fixt_deconv
from tests.testing_utils.metrics import compute_f1_metrics
from tests.testing_utils.plotting import plot_met_ROC_thres

pytestmark = pytest.mark.demo

@pytest.mark.slow
class TestDemoDeconv:
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(15))
    @pytest.mark.parametrize("upsamp", [1, 2, 5])
    @pytest.mark.parametrize("ns_lev", [0, 0.2, 0.5])
    @pytest.mark.parametrize("thres_scaling", [True])
    def test_demo_solve_thres(
        self,
        taus,
        rand_seed,
        upsamp,
        ns_lev,
        thres_scaling,
        test_fig_path_svg,
        results_bag,
    ):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus, rand_seed=rand_seed, upsamp=upsamp, ns_lev=ns_lev
        )
        s_bin, c_bin, scl, err, intm = deconv.solve_thres(
            scaling=thres_scaling, return_intm=True
        )
        s_slv, thres, svals, cvals, yfvals, scals, objs, opt_idx = intm
        # save results
        metdf = compute_f1_metrics(
            s_org,
            svals,
            {"objs": objs, "scals": scals, "thres": thres, "opt_idx": opt_idx},
            tdist_thres=3,
        )
        results_bag.data = metdf
        # plotting
        fig = plot_met_ROC_thres(metdf)
        fig.savefig(test_fig_path_svg)

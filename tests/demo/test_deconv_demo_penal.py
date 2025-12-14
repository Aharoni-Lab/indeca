import numpy as np
import pandas as pd
import pytest

from indeca.core.deconv import DeconvBin
from tests.conftest import fixt_deconv
from tests.testing_utils.metrics import compute_f1_metrics, df_assign_metadata
from tests.testing_utils.plotting import plot_met_ROC_thres

pytestmark = pytest.mark.demo


@pytest.mark.slow
class TestDemoDeconv:
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(15))
    @pytest.mark.parametrize("upsamp", [1, 2])
    @pytest.mark.parametrize("ns_lev", [0, 0.2, 0.5])
    @pytest.mark.parametrize("y_scaling", [False])
    def test_demo_solve_penal(
        self, taus, rand_seed, upsamp, ns_lev, y_scaling, test_fig_path_svg, results_bag
    ):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus,
            rand_seed=rand_seed,
            upsamp=upsamp,
            ns_lev=ns_lev,
            y_scaling=y_scaling,
        )
        _, _, _, _, intm_free = deconv.solve_thres(
            scaling=False, amp_constraint=False, return_intm=True
        )
        _, _, _, _, intm_nopn = deconv.solve_thres(scaling=True, return_intm=True)
        _, _, _, _, opt_penal, intm_pn = deconv.solve_penal(
            scaling=True, return_intm=True
        )
        # save results
        intms = {"CNMF": intm_free, "No Penalty": intm_nopn, "Penalty": intm_pn}
        metdf = []
        for grp, cur_intm in intms.items():
            if grp == "Penalty":
                cur_svals = []
                oidx = intm_pn[7]
                for sv in intm_pn[2]:
                    s_pad = np.zeros(deconv.T)
                    s_pad[deconv.nzidx_s] = sv
                    cur_svals.append(s_pad)
            else:
                cur_svals = cur_intm[2]
            cur_met = compute_f1_metrics(
                s_org,
                cur_svals,
                {
                    "group": grp,
                    "thres": cur_intm[1],
                    "scals": cur_intm[5],
                    "objs": cur_intm[6],
                    "penal": opt_penal if grp == "Penalty" else 0,
                    "opt_idx": cur_intm[7],
                },
                tdist_thres=2,
            )
            metdf.append(cur_met)
        metdf = pd.concat(metdf, ignore_index=True)
        metdf = df_assign_metadata(
            metdf,
            {"tau_d": taus[0], "tau_r": taus[1]},
        )
        results_bag.data = metdf
        # plotting
        fig = plot_met_ROC_thres(metdf, grad_color=False)
        fig.savefig(test_fig_path_svg)
        # assertion
        if ns_lev == 0 and upsamp == 1:
            assert (cur_svals[oidx][:-1] == s[:-1]).all()

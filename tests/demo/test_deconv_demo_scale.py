import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from indeca.core.simulation import AR2tau, tau2AR
from indeca.core.deconv import DeconvBin
from tests.conftest import fixt_y
from tests.testing_utils.metrics import assignment_distance
from tests.testing_utils.plotting import plot_met_ROC_scale, plot_traces

pytestmark = pytest.mark.demo


@pytest.mark.slow
class TestDemoDeconv:
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("upsamp", [1])
    @pytest.mark.parametrize("ns_lev", [0, 0.2, 0.5])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize("penalty", [None])
    @pytest.mark.parametrize("err_weighting", [None, "adaptive"])
    @pytest.mark.parametrize("obj_crit", [None])
    def test_demo_solve_scale(
        self,
        taus,
        upsamp,
        ns_lev,
        rand_seed,
        penalty,
        err_weighting,
        obj_crit,
        test_fig_path_svg,
        test_fig_path_html,
    ):
        # act
        y, c_true, c_org, s_true, s_org, scale = fixt_y(
            taus=taus, upsamp=upsamp, rand_seed=rand_seed, ns_lev=ns_lev
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
        deconv.update(update_weighting=True)
        err_wt = deconv.err_wt.squeeze()
        deconv.update(update_weighting=True, clear_weighting=True, scale=1)
        deconv._reset_mask()
        deconv._reset_cache()
        s_free, b_free = deconv.solve(amp_constraint=False)
        scl_ub = np.ptp(s_free)
        res_df = []
        for scl in np.linspace(0, scl_ub, 100)[1:]:
            deconv.update(scale=scl)
            sbin, cbin, _, _ = deconv.solve_thres(scaling=False, obj_crit=obj_crit)
            deconv.err_wt = err_wt
            obj = deconv._compute_err(s=sbin, obj_crit=obj_crit)
            deconv.err_wt = np.ones_like(err_wt)
            mdist, f1, prec, rec = assignment_distance(
                s_ref=s_true, s_slv=sbin, tdist_thres=3
            )
            res_df.append(
                pd.DataFrame(
                    [
                        {
                            "scale": scl,
                            "objs": obj,
                            "mdist": mdist,
                            "f1": f1,
                            "prec": prec,
                            "recall": rec,
                        }
                    ]
                )
            )
        res_df = pd.concat(res_df, ignore_index=True)
        # plotting
        fig = plot_met_ROC_scale(res_df, iterdf, cur_scl)
        fig.savefig(test_fig_path_svg)
        fig = go.Figure()
        fig.add_traces(
            plot_traces(
                {
                    "y": y.squeeze(),
                    "s_true": s_true.squeeze(),
                    "c_true": c_true.squeeze(),
                    "opt_s": opt_s.squeeze(),
                    "opt_c": opt_c.squeeze(),
                    "err_wt": err_wt,
                }
            )
        )
        fig.write_html(test_fig_path_html)

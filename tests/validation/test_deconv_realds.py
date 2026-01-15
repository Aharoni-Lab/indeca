import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from indeca.core.simulation import AR2tau, tau2AR
from indeca.core.deconv import DeconvBin
from tests.conftest import fixt_realds
from tests.testing_utils.metrics import assignment_distance
from tests.testing_utils.plotting import plot_met_ROC_scale, plot_traces

pytestmark = pytest.mark.validation


@pytest.mark.slow
class TestDemoDeconv:
    @pytest.mark.parametrize("upsamp", [1])
    @pytest.mark.parametrize("ar_kn_len", [100])
    @pytest.mark.parametrize("est_noise_freq", [None])
    @pytest.mark.parametrize("est_add_lag", [10])
    @pytest.mark.parametrize("dsname", ["X-DS09-GCaMP6f-m-V1"])
    @pytest.mark.parametrize("taus", [(21.18, 7.23)])
    @pytest.mark.parametrize("ncell", [1])
    @pytest.mark.parametrize("nfm", [None])
    @pytest.mark.parametrize("penalty", [None])
    @pytest.mark.parametrize("err_weighting", [None, "adaptive"])
    @pytest.mark.parametrize("obj_crit", [None])
    def test_demo_solve_scale_realds(
        self,
        upsamp,
        ar_kn_len,
        est_noise_freq,
        est_add_lag,
        dsname,
        taus,
        ncell,
        nfm,
        penalty,
        err_weighting,
        obj_crit,
        test_fig_path_svg,
        test_fig_path_html,
    ):
        # act
        Y, S_true, ap_df, fluo_df = fixt_realds(dsname, ncell=ncell, nfm=nfm)
        theta = tau2AR(taus[0], taus[1])
        _, _, p = AR2tau(theta[0], theta[1], solve_amp=True)
        deconv = DeconvBin(
            y=Y.squeeze(),
            tau=taus,
            ps=(p, -p),
            penal=penalty,
            err_weighting=err_weighting,
            use_base=True,
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
            sb_idx = np.where(sbin)[0] / upsamp
            t_sb = np.interp(sb_idx, fluo_df["frame"], fluo_df["fluo_time"])
            t_ap = ap_df["ap_time"]
            mdist, f1, prec, rec = assignment_distance(
                t_ref=np.atleast_1d(t_ap), t_slv=np.atleast_1d(t_sb), tdist_thres=1
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
                    "y": Y.squeeze(),
                    "s_true": S_true.squeeze(),
                    "opt_s": opt_s.squeeze(),
                    "opt_c": opt_c.squeeze(),
                    "err_wt": err_wt,
                }
            )
        )
        fig.write_html(test_fig_path_html)

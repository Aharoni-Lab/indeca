import numpy as np
import pandas as pd
import pytest

from indeca.AR_kernel import estimate_coefs, solve_fit_h_num
from indeca.simulation import AR2exp, AR2tau, exp_pulse, find_dhm, tau2AR

from .conftest import fixt_y


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


class TestDemoSolveFit:
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize(
        "ns_lev",
        [0] + [pytest.param(n, marks=pytest.mark.slow) for n in [0.1, 0.2, 0.5]],
    )
    @pytest.mark.parametrize("ncell", [30])
    @pytest.mark.parametrize("upsamp", [1, 2, 5])
    def test_demo_solve_fit_h_num(
        self, taus, rand_seed, ns_lev, ncell, upsamp, results_bag
    ):
        # book-keeping
        res_df = []
        Y, C, C_org, S, S_org, scales = fixt_y(
            taus=taus, rand_seed=rand_seed, ns_lev=ns_lev, ncell=ncell, upsamp=upsamp
        )
        dhm_true, _ = find_dhm(True, taus, np.array([1, -1]))
        res_df.append(
            pd.DataFrame(
                [
                    {
                        "method": "truth",
                        "unit": "all",
                        "isreal": True,
                        "tau_d": taus[0],
                        "tau_r": taus[1],
                        "dhm0": dhm_true[0],
                        "dhm1": dhm_true[1],
                        "p0": 1,
                        "p1": -1,
                    }
                ]
            )
        )
        # act
        # cnmf method
        for icell, (y, s) in enumerate(zip(Y, S_org)):
            for mthd, smth in {"cnmf_raw": False, "cnmf_smth": True}.items():
                theta, _ = estimate_coefs(
                    y, p=2, noise_freq=0.1, use_smooth=smth, add_lag=20
                )
                is_biexp, cur_taus, cur_ps = AR2exp(*theta)
                cur_dhm, _ = find_dhm(is_biexp, cur_taus, cur_ps)
                res_df.append(
                    pd.DataFrame(
                        [
                            {
                                "method": mthd,
                                "unit": str(icell),
                                "isreal": is_biexp,
                                "tau_d": cur_taus[0],
                                "tau_r": cur_taus[1],
                                "dhm0": cur_dhm[0],
                                "dhm1": cur_dhm[1],
                                "p0": cur_ps[0],
                                "p1": cur_ps[1],
                            }
                        ]
                    )
                )
            lams, ps, _, _, _ = solve_fit_h_num(
                y, s, np.ones(1), up_factor=upsamp, s_len=60 * upsamp
            )
            tau_fit = -1 / lams / upsamp
            dhm_fit, _ = find_dhm(True, tau_fit, ps)
            assert dhm_fit[0] > 0 and dhm_fit[1] > 0
            res_df.append(
                pd.DataFrame(
                    [
                        {
                            "method": "solve_fit",
                            "unit": str(icell),
                            "isreal": True,
                            "tau_d": tau_fit[0],
                            "tau_r": tau_fit[1],
                            "dhm0": dhm_fit[0],
                            "dhm1": dhm_fit[1],
                            "p0": ps[0],
                            "p1": ps[1],
                        }
                    ]
                )
            )
        # minian-bin method
        lams, ps, ar_scal, h, h_fit = solve_fit_h_num(
            Y, S_org, np.ones(Y.shape[0]), up_factor=upsamp, s_len=60 * upsamp
        )
        tau_fit = -1 / lams / upsamp
        dhm_fit, _ = find_dhm(True, tau_fit, ps)
        res_df.append(
            pd.DataFrame(
                [
                    {
                        "method": "solve_fit",
                        "unit": "all",
                        "isreal": True,
                        "tau_d": tau_fit[0],
                        "tau_r": tau_fit[1],
                        "dhm0": dhm_fit[0],
                        "dhm1": dhm_fit[1],
                        "p0": ps[0],
                        "p1": ps[1],
                    }
                ]
            )
        )
        # save results
        res_df = pd.concat(res_df, ignore_index=True)
        results_bag.data = res_df
        # assertion
        if ns_lev == 0:
            assert np.isclose(dhm_fit, dhm_true, atol=0.1).all()

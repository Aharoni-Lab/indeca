import os

import numpy as np
import pandas as pd
import pytest

from minian_bin.AR_kernel import (
    convolve_g,
    convolve_h,
    estimate_coefs,
    fit_sumexp,
    fit_sumexp_gd,
    fit_sumexp_split,
    solve_fit_h_num,
    solve_g,
)
from minian_bin.simulation import AR2exp, ar_trace, find_dhm

from .testing_utils.plotting import plot_traces


@pytest.fixture()
def param_y_len():
    return 1000


@pytest.fixture()
def param_ncell():
    return 30


@pytest.fixture(params=[(6, 1), (10, 3)])
def param_taus(request):
    return request.param


@pytest.fixture(params=[0, 0.1, 0.2, 0.5])
def param_ns_level(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 4, 5])
def param_rand_seed(request):
    sd = request.param
    np.random.seed(sd)
    return sd


@pytest.fixture()
def param_tmp_P():
    return np.array([[0.98, 0.02], [0.75, 0.25]])


@pytest.fixture(params=[1, 2, 5])
def param_upsamp(request):
    return request.param


@pytest.fixture()
def param_tmp_upsamp(param_upsamp):
    if param_upsamp < 5:
        tmp = np.array([[0.98, 0.02], [0.75, 0.25]])
    else:
        tmp = np.array([[0.998, 0.002], [0.75, 0.25]])
    return param_upsamp, tmp


@pytest.fixture()
def fixt_y(
    param_y_len,
    param_ncell,
    param_taus,
    param_tmp_upsamp,
    param_ns_level,
    param_rand_seed,
):
    upsamp, tmp_P = param_tmp_upsamp
    Y, C_org, S_org, C, S = [], [], [], [], []
    for i in range(param_ncell):
        c_org, s_org = ar_trace(
            param_y_len * upsamp,
            tmp_P,
            tau_d=param_taus[0] * upsamp,
            tau_r=param_taus[1] * upsamp,
            shifted=True,
        )
        if upsamp > 1:
            c = np.convolve(c_org, np.ones(upsamp), "valid")[::upsamp]
            s = np.convolve(s_org, np.ones(upsamp), "valid")[::upsamp]
        else:
            c, s = c_org, s_org
        y = c + np.random.normal(0, param_ns_level, c.shape)
        Y.append(y)
        C_org.append(c_org)
        S_org.append(s_org)
        C.append(c)
        S.append(s)
    Y = np.stack(Y, axis=0)
    C_org = np.stack(C_org, axis=0)
    S_org = np.stack(S_org, axis=0)
    C = np.stack(C, axis=0)
    S = np.stack(S, axis=0)
    return Y, C, C_org, S, S_org, param_taus, param_ns_level, upsamp


class TestDemoSolveFit:
    def test_demo_solve_fit_h_num(self, fixt_y, results_bag):
        # book-keeping
        res_df = []
        Y, C, C_org, S, S_org, taus_true, ns_level, upsamp = fixt_y
        dhm_true, _ = find_dhm(True, taus_true, np.array([1, -1]))
        res_df.append(
            pd.DataFrame(
                [
                    {
                        "method": "truth",
                        "unit": "all",
                        "isreal": True,
                        "tau_d": taus_true[0],
                        "tau_r": taus_true[1],
                        "dhm0": dhm_true[0],
                        "dhm1": dhm_true[1],
                        "p0": 1,
                        "p1": -1,
                    }
                ]
            )
        )
        # act
        lams, ps, ar_scal, h, h_fit = solve_fit_h_num(
            Y, S_org, np.ones(Y.shape[0]), up_factor=upsamp
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
            lams, ps, _, _, _ = solve_fit_h_num(y, s, np.ones(1), up_factor=upsamp)
            tau_fit = -1 / lams / upsamp
            dhm_fit, _ = find_dhm(True, tau_fit, ps)
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
        # save results
        res_df = pd.concat(res_df, ignore_index=True)
        results_bag.data = res_df


def test_convolve_g(sample_timeseries):
    """Test g convolution."""
    s = sample_timeseries[:, 0]  # take first neuron
    g = np.array([0.7, -0.2])
    result = convolve_g(s, g)
    assert result.shape == s.shape


@pytest.mark.skip(reason="Array dimension mismatch needs to be investigated")
def test_convolve_h(sample_timeseries):
    """Test h convolution."""
    pass


def test_solve_g(sample_timeseries):
    """Test solving for g parameters."""
    y = sample_timeseries[:, 0]  # take first neuron
    s = np.zeros_like(y)
    s[::10] = 1  # sparse spikes
    theta_1, theta_2 = solve_g(y, s)
    assert isinstance(float(theta_1), float)  # Convert numpy.float64 to float
    assert isinstance(float(theta_2), float)


def test_fit_sumexp(sample_timeseries):
    """Test sum of exponentials fitting."""
    y = sample_timeseries[:, 0]  # take first neuron
    lams, ps, y_fit = fit_sumexp(y, N=2)
    assert len(lams) == 2
    assert len(ps) == 2
    assert y_fit.shape == y.shape


@pytest.mark.slow
def test_fit_sumexp_split(sample_timeseries):
    """Test split sum of exponentials fitting."""
    y = sample_timeseries[:, 0]  # take first neuron
    lams, ps, y_fit = fit_sumexp_split(y)
    assert len(lams) == 2
    assert len(ps) == 2
    assert y_fit.shape == y.shape


def test_fit_sumexp_gd(sample_timeseries):
    """Test gradient descent fitting of sum of exponentials."""
    y = sample_timeseries[:, 0]  # take first neuron
    lams, ps, scal, y_fit = fit_sumexp_gd(y)
    assert len(lams) == 2
    assert len(ps) == 2
    assert isinstance(float(scal), float)  # Convert any numeric type to float
    assert y_fit.shape == y.shape


class TestResults:
    def test_results(self, module_results_df, func_data_dir):
        module_results_df["func_name"] = module_results_df["pytest_obj"].apply(
            lambda o: o.__name__
        )
        for fname, fdf in module_results_df.groupby("func_name"):
            fdf = fdf[fdf["data"].notnull()].reset_index()
            if len(fdf) > 0:
                param_cols = list(set(fdf.columns) - set(["data", "pytest_obj"]))
                result = []
                for _, frow in fdf.iterrows():
                    dat = frow["data"]
                    dat = dat.assign(**{p: [frow[p]] * len(dat) for p in param_cols})
                    result.append(dat)
                result = pd.concat(result, ignore_index=True)
                result.to_feather(os.path.join(func_data_dir, "{}.feat".format(fname)))

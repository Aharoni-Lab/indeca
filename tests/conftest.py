import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest_harvest import get_session_results_df, get_xdist_worker_id, is_main_process

from minian_bin.deconv import DeconvBin
from minian_bin.simulation import AR2tau, ar_trace, tau2AR

from .testing_utils.io import download_realds, load_gt_ds
from .testing_utils.misc import get_upsamp_scale

AGG_RES_DIR = "tests/output/data/agg_results"
TEST_DATA_DIR = "tests/data"


@pytest.fixture
def temp_data_dir():
    """Fixture to provide a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def output_figs_dir(request):
    test_path = os.path.dirname(request.path)
    fig_dir = os.path.abspath(os.path.join(test_path, "output", "figs"))
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


@pytest.fixture
def output_data_dir(request):
    test_path = os.path.dirname(request.path)
    fig_dir = os.path.abspath(os.path.join(test_path, "output", "data"))
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


@pytest.fixture
def func_figs_dir(request, output_figs_dir):
    test_func = request.function.__name__
    fig_dir = os.path.join(output_figs_dir, test_func)
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


@pytest.fixture
def func_data_dir(request, output_data_dir):
    test_func = request.function.__name__
    dat_dir = os.path.join(output_data_dir, "debug_results", test_func)
    os.makedirs(dat_dir, exist_ok=True)
    return dat_dir


@pytest.fixture()
def test_fig_path_html(request, func_figs_dir):
    test_id = request.node.callspec.id
    return os.path.join(func_figs_dir, "{}.html".format(test_id))


@pytest.fixture()
def test_fig_path_svg(request, func_figs_dir):
    test_id = request.node.callspec.id
    return os.path.join(func_figs_dir, "{}.svg".format(test_id))


@pytest.fixture()
def eq_atol():
    return 1e-1


def fixt_y(
    taus,
    rand_seed=0,
    y_len=1000,
    P_fire=None,
    upsamp=1,
    ns_lev=0,
    y_scaling=False,
    ncell=1,
    squeeze=True,
):
    if P_fire is None:
        if upsamp < 5:
            P_fire = np.array([[0.98, 0.02], [0.75, 0.25]])
        else:
            P_fire = np.array([[0.998, 0.002], [0.75, 0.25]])
    rng = np.random.default_rng(rand_seed)
    Y, C_org, S_org, C, S, scales = [], [], [], [], [], []
    for i in range(ncell):
        c_org, s_org = ar_trace(
            y_len * upsamp,
            P_fire,
            tau_d=taus[0] * upsamp,
            tau_r=taus[1] * upsamp,
            shifted=True,
            rng=rng,
        )
        if upsamp > 1:
            c = np.convolve(c_org, np.ones(upsamp), "valid")[::upsamp]
            s = np.convolve(s_org, np.ones(upsamp), "valid")[::upsamp]
        else:
            c, s = c_org, s_org
        if y_scaling:
            scl = rng.uniform(0.5, 2)
        else:
            scl = 1
        y = scl * (c + rng.normal(0, ns_lev, c.shape) * upsamp)
        Y.append(y)
        C_org.append(c_org)
        S_org.append(s_org)
        C.append(c)
        S.append(s)
        scales.append(scl)
    Y = np.stack(Y, axis=0)
    C_org = np.stack(C_org, axis=0)
    S_org = np.stack(S_org, axis=0)
    C = np.stack(C, axis=0)
    S = np.stack(S, axis=0)
    scales = np.stack(scales, axis=0)
    if squeeze:
        Y, C, C_org, S, S_org, scales = (
            Y.squeeze(),
            C.squeeze(),
            C_org.squeeze(),
            S.squeeze(),
            S_org.squeeze(),
            scales.squeeze(),
        )
    return Y, C, C_org, S, S_org, scales


def fixt_deconv(taus, norm="l2", upsamp=1, upsamp_y=None, backend="osqp", **kwargs):
    if upsamp_y is None:
        upsamp_y = upsamp
    y, c, c_org, s, s_org, scale = fixt_y(taus=taus, upsamp=upsamp_y, **kwargs)
    assert y.ndim == 1, "fixt_deconv only support single cell mode"
    taus_up = np.array(taus) * upsamp
    _, _, p = AR2tau(*tau2AR(*taus_up), solve_amp=True)
    deconv = DeconvBin(
        y=y,
        tau=taus_up,
        ps=np.array([p, -p]),
        upsamp=upsamp,
        err_weighting=None,
        backend=backend,
        norm=norm,
    )
    if upsamp_y != upsamp:
        scl = get_upsamp_scale(taus, upsamp_y, upsamp)
        deconv.update(scale=scl)
    return deconv, y, c, c_org, s, s_org, scale


# @pytest.hookimpl(tryfirst=True)
# def pytest_configure(config):
#     if not hasattr(config, "workerinput"):
#         shutil.rmtree(AGG_RES_DIR, ignore_errors=True)


def fixt_realds(dsname, ncell=None, nfm=None):
    if not os.path.exists(os.path.join(TEST_DATA_DIR, dsname)) or not os.listdir(
        os.path.join(TEST_DATA_DIR, dsname)
    ):
        download_realds(TEST_DATA_DIR, dsname)
    Y, S_true, ap_df, fluo_df = load_gt_ds(os.path.join(TEST_DATA_DIR, dsname))
    if fluo_df["fps"].nunique() > 1:
        warnings.warn("More than one fps found in dataset {}".format(dsname))
        fps_ncell = fluo_df.groupby(["fps"])["unit_id"].nunique()
        fps_keep = fps_ncell.index[fps_ncell.argmax()]
        ap_df = ap_df[ap_df["fps"] == fps_keep].copy()
        fluo_df = fluo_df[fluo_df["fps"] == fps_keep].copy()
        uids = fluo_df["unit_id"].unique()
        Y = Y.sel(unit_id=uids).dropna("frame", how="all").fillna(0)
        S_true = S_true.sel(unit_id=uids, frame=Y.coords["frame"])
    if nfm is not None:
        ap_df = ap_df[ap_df["frame"].between(0, nfm)]
        fluo_df = fluo_df[fluo_df["frame"].between(0, nfm)]
        Y = Y.isel(frame=slice(0, nfm))
        S_true = S_true.isel(frame=slice(0, nfm))
    ap_ct = ap_df.groupby("unit_id")["ap_time"].count().reset_index()
    act_uids = np.array(ap_ct.loc[ap_ct["ap_time"] > 1, "unit_id"])
    if ncell is not None and ncell > len(act_uids):
        warnings.warn(
            "Cannot select {} active cells with {} frames in dataset {}".format(
                ncell, nfm, dsname
            )
        )
    else:
        act_uids = act_uids[:ncell]
    Y = Y.sel(unit_id=act_uids)
    S_true = S_true.sel(unit_id=act_uids)
    ap_df = ap_df.set_index("unit_id").loc[act_uids]
    fluo_df = fluo_df.set_index("unit_id").loc[act_uids]
    Y = Y * 100
    return Y, S_true, ap_df, fluo_df


def pytest_sessionfinish(session):
    """Gather all results and save them to a csv.
    Works both on worker and master nodes, and also with xdist disabled"""
    session_results_df = get_session_results_df(session)
    if not len(session_results_df) > 0:
        return
    session_results_df["func_name"] = session_results_df["pytest_obj"].apply(
        lambda o: o.__name__
    )
    for fname, fdf in session_results_df.groupby("func_name"):
        try:
            fdf = fdf[fdf["data"].notnull()].reset_index()
        except KeyError:
            continue
        if len(fdf) > 0:
            param_cols = list(set(fdf.columns) - set(["data", "pytest_obj"]))
            result = []
            for _, frow in fdf.iterrows():
                dat = frow["data"]
                dat = dat.assign(**{p: [frow[p]] * len(dat) for p in param_cols})
                result.append(dat)
            result = pd.concat(result, ignore_index=True).dropna(
                axis="columns", how="all"
            )
            cvtcols = result.select_dtypes(include="object").columns
            result[cvtcols] = result[cvtcols].fillna("").astype(str)
            suffix = "all" if is_main_process(session) else get_xdist_worker_id(session)
            dat_dir = os.path.join(AGG_RES_DIR, fname)
            os.makedirs(dat_dir, exist_ok=True)
            result.drop_duplicates().to_feather(
                os.path.join(dat_dir, "{}.feat".format(suffix))
            )

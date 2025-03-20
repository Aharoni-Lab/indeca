import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest_harvest import get_session_results_df, get_xdist_worker_id, is_main_process

AGG_RES_DIR = "tests/output/data/agg_results"


@pytest.fixture
def test_data_dir():
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


@pytest.fixture
def sample_timeseries():
    """Fixture to provide a sample time series data."""
    np.random.seed(42)
    return np.random.normal(0, 1, (100, 10))  # 100 timepoints, 10 neurons


@pytest.fixture
def sample_movie():
    """Fixture to provide a sample movie data."""
    np.random.seed(42)
    return np.random.normal(0, 1, (100, 32, 32))  # 100 frames, 32x32 pixels


@pytest.fixture
def sample_xarray_dataset():
    """Create a sample xarray dataset with typical dimensions."""
    np.random.seed(42)
    times = pd.date_range("2024-01-01", periods=100, freq="s")
    data = np.random.normal(0, 1, (100, 32, 32))
    return xr.Dataset(
        {
            "fluorescence": (["time", "height", "width"], data),
        },
        coords={
            "time": times,
            "height": range(32),
            "width": range(32),
        },
    )


@pytest.fixture
def ar_parameters():
    """Sample AR model parameters."""
    return {"order": 2, "coefficients": np.array([0.7, -0.2]), "noise_std": 0.1}


@pytest.fixture
def deconv_parameters():
    """Sample deconvolution parameters."""
    return {
        "lambda_": 1.0,
        "optimize_g": True,
        "g_constraints": {"lb": 0.5, "ub": 0.98},
    }


@pytest.fixture
def pipeline_config():
    """Sample pipeline configuration."""
    return {
        "motion_correction": {"max_shift": 10},
        "spatial_filter": {"sigma": 1.0},
        "temporal_filter": {"window": 5},
    }


@pytest.fixture
def simulation_params():
    """Parameters for simulating calcium imaging data."""
    return {
        "n_neurons": 10,
        "dimensions": (32, 32),
        "duration": 100,
        "framerate": 30,
        "noise_level": 0.1,
    }


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if not hasattr(config, "workerinput"):
        shutil.rmtree(AGG_RES_DIR, ignore_errors=True)


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
            result = pd.concat(result, ignore_index=True)
            suffix = "all" if is_main_process(session) else get_xdist_worker_id(session)
            dat_dir = os.path.join(AGG_RES_DIR, fname)
            os.makedirs(dat_dir, exist_ok=True)
            result.to_feather(os.path.join(dat_dir, "{}.feat".format(suffix)))

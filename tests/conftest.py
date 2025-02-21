import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import xarray as xr
import pandas as pd


@pytest.fixture
def test_data_dir():
    """Fixture to provide a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


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

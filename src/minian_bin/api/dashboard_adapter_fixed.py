"""Fixed adapter for integrating the FastAPI dashboard with Dask."""

import json
import os
import uuid
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
from minian_bin.logging_config import get_module_logger

# Initialize logger
logger = get_module_logger("api.dashboard_adapter_fixed")

# Create data directory for dashboard data
DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "data"))
DATA_DIR.mkdir(exist_ok=True)


def get_data_path(session_id):
    """Get the path to the data file for a session."""
    return DATA_DIR / f"{session_id}.json"


def load_dashboard_data(session_id):
    """Load dashboard data from file."""
    data_path = get_data_path(session_id)
    if data_path.exists():
        try:
            with open(data_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading dashboard data: {e}")
            return None
    return None


def save_dashboard_data(session_id, data):
    """Save dashboard data to file."""
    data_path = get_data_path(session_id)
    try:
        # Create a temporary file and then rename to avoid partial writes
        tmp_path = data_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        tmp_path.replace(data_path)
        return True
    except (IOError, OSError) as e:
        logger.error(f"Error saving dashboard data: {e}")
        return False


def get_or_create_session_data(session_id, Y):
    """Get or create session data."""
    data = load_dashboard_data(session_id)
    if data is None:
        # Initialize new session data
        data = {
            "Y": Y.tolist() if hasattr(Y, "tolist") else Y,
            "traces": {},
            "iterations": {},
            "kernels": {},
        }
        save_dashboard_data(session_id, data)
    return data


def get_all_sessions():
    """Get all available session IDs."""
    return [f.stem for f in DATA_DIR.glob("*.json")]


class DashboardAdapterFixed:
    """A version of DashboardAdapter that works with Dask by using file-based storage."""

    def __init__(
        self,
        Y: np.ndarray = None,
        ncell: int = None,
        T: int = None,
        max_iters: int = 20,
        kn_len: int = 60,
        port: int = 54321,
        session_id: str = None,
        client=None,  # Ignored but kept for API compatibility
    ):
        """Initialize the dashboard adapter.

        Args:
            Y: Input fluorescence traces (ncell x T)
            ncell: Number of cells (if Y not provided)
            T: Number of time points (if Y not provided)
            max_iters: Maximum number of iterations
            kn_len: Kernel length
            port: Port for the FastAPI server (not used directly)
            session_id: Session ID for WebSocket connections
            client: Dask client (ignored to avoid pickle issues)
        """
        self.title = "Dashboard"
        if Y is None:
            assert ncell is not None and T is not None
            Y = np.ones((ncell, T))
        else:
            ncell, T = Y.shape

        self.Y = Y
        self.ncell = ncell
        self.T = T
        self.kn_len = kn_len
        self.max_iters = max_iters
        self.it_update = 0
        self.it_view = 0
        self.session_id = session_id or str(uuid.uuid4())

        # Initialize data storage for this session in main process
        session_data = get_or_create_session_data(self.session_id, Y)

        logger.info(f"Dashboard adapter initialized with session_id {self.session_id}")

    def set_iter(self, it: int):
        """Set the current iteration.

        Args:
            it: Iteration number
        """
        self.it_update = it

    def update(self, uid: int = None, **kwargs):
        """Update dashboard data.

        Args:
            uid: Optional cell/unit ID. If None, update for all cells.
            **kwargs: Keyword arguments with updated values
        """
        # Load current data
        session_data = load_dashboard_data(self.session_id)
        if session_data is None:
            # Re-initialize if data is missing
            session_data = get_or_create_session_data(self.session_id, self.Y)

        if uid is None:
            uids = range(self.ncell)
        else:
            uids = [uid]

        # Track if anything was updated
        updated = False

        for u in uids:
            str_u = str(u)
            # Store trace data
            if "c" in kwargs or "s" in kwargs:
                if str_u not in session_data["traces"]:
                    session_data["traces"][str_u] = {
                        "y": self.Y[u].tolist()
                        if hasattr(self.Y[u], "tolist")
                        else self.Y[u]
                    }
                    updated = True

                if "c" in kwargs:
                    try:
                        c_data = kwargs["c"]
                        if hasattr(c_data, "shape") and len(c_data.shape) > 1:
                            c_data = c_data[u]
                        session_data["traces"][str_u]["c"] = (
                            c_data.tolist() if hasattr(c_data, "tolist") else c_data
                        )
                        updated = True
                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Error storing c data for cell {u}: {e}")

                if "s" in kwargs:
                    try:
                        s_data = kwargs["s"]
                        if hasattr(s_data, "shape") and len(s_data.shape) > 1:
                            s_data = s_data[u]
                        session_data["traces"][str_u]["s"] = (
                            s_data.tolist() if hasattr(s_data, "tolist") else s_data
                        )
                        updated = True
                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Error storing s data for cell {u}: {e}")

            # Store kernel data
            if "h" in kwargs or "h_fit" in kwargs:
                if str_u not in session_data["kernels"]:
                    session_data["kernels"][str_u] = {}
                    updated = True

                if "h" in kwargs:
                    try:
                        h_data = kwargs["h"]
                        if hasattr(h_data, "shape") and len(h_data.shape) > 1:
                            h_data = h_data[u]
                        session_data["kernels"][str_u]["h"] = (
                            h_data.tolist() if hasattr(h_data, "tolist") else h_data
                        )
                        updated = True
                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Error storing h data for cell {u}: {e}")

                if "h_fit" in kwargs:
                    try:
                        h_fit_data = kwargs["h_fit"]
                        if hasattr(h_fit_data, "shape") and len(h_fit_data.shape) > 1:
                            h_fit_data = h_fit_data[u]
                        session_data["kernels"][str_u]["h_fit"] = (
                            h_fit_data.tolist()
                            if hasattr(h_fit_data, "tolist")
                            else h_fit_data
                        )
                        updated = True
                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Error storing h_fit data for cell {u}: {e}")

            # Store iteration data
            if any(k in kwargs for k in ["tau_d", "tau_r", "err", "scale"]):
                if str_u not in session_data["iterations"]:
                    session_data["iterations"][str_u] = {}
                    updated = True

                session_data["iterations"][str_u]["iter"] = self.it_update
                updated = True

                for k in ["tau_d", "tau_r", "err", "scale"]:
                    if k in kwargs:
                        try:
                            val = kwargs[k]
                            if hasattr(val, "item"):
                                val = val.item()
                            elif hasattr(val, "__getitem__") and len(val) > u:
                                val = val[u]
                                if hasattr(val, "item"):
                                    val = val.item()

                            session_data["iterations"][str_u][k] = val
                            updated = True
                        except (IndexError, ValueError, AttributeError) as e:
                            logger.warning(f"Error storing {k} data for cell {u}: {e}")

        # Save changes if anything was updated
        if updated:
            save_dashboard_data(self.session_id, session_data)

    def stop(self):
        """Clean up resources."""
        logger.info(f"Dashboard adapter stopped for session {self.session_id}")


# Function to get dashboard data for API route
def get_dashboard_data():
    """Get all dashboard data for API endpoints."""
    sessions = get_all_sessions()
    return {session: load_dashboard_data(session) for session in sessions}


def get_session_data(session_id):
    """Get dashboard data for a specific session."""
    return load_dashboard_data(session_id)

"""Adapter for integrating the FastAPI dashboard with the existing codebase."""

import asyncio
import logging
import uuid
from asyncio import Future
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np
from dask.distributed import Client

from minian_bin.api.models.dashboard import DataType, DashboardUpdateMessage
from minian_bin.api.websockets.dashboard import connection_manager
from minian_bin.logging_config import get_module_logger

# Initialize logger
logger = get_module_logger("api.dashboard_adapter")


class DashboardAdapter:
    """Adapter for the FastAPI dashboard that mimics the existing Dashboard interface."""

    def __init__(
        self,
        Y: np.ndarray = None,
        ncell: int = None,
        T: int = None,
        max_iters: int = 20,
        kn_len: int = 60,
        port: int = 54321,
        session_id: str = None,
        client: Client = None,
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
            client: Dask client for distributed processing
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
        self.client = client

        # Iteration variable storage (similar to original dashboard)
        self.it_vars = {
            "c": np.full((max_iters, ncell, T), np.nan),
            "s": np.full((max_iters, ncell, T), np.nan),
            "h": np.full((max_iters, ncell, kn_len), np.nan),
            "h_fit": np.full((max_iters, ncell, kn_len), np.nan),
            "scale": np.full((max_iters, ncell), np.nan),
            "tau_d": np.full((max_iters, ncell), np.nan),
            "tau_r": np.full((max_iters, ncell), np.nan),
            "err": np.full((max_iters, ncell), np.nan),
            "penal_err": np.array(
                [
                    [{"penal": [], "scale": [], "err": []} for _ in range(ncell)]
                    for _ in range(max_iters)
                ]
            ),
        }

        # Event loop for async broadcasts
        self.loop = asyncio.get_event_loop()
        logger.info(f"Dashboard adapter initialized with session_id {self.session_id}")

        # Broadcast initial data
        self._broadcast_initial_data()

    def _broadcast_initial_data(self):
        """Broadcast initial data to connected clients."""
        # Broadcast initial Y data
        for u in range(self.ncell):
            self._broadcast_trace_data(u)

        logger.info("Initial data broadcasted")

    def _broadcast_trace_data(self, uid: int):
        """Broadcast trace data for a specific cell.

        Args:
            uid: Cell/Unit ID
        """
        data = {
            "y": self.Y[uid].tolist(),
            "c": self.it_vars["c"][self.it_update, uid].tolist()
            if not np.isnan(self.it_vars["c"][self.it_update, uid]).all()
            else None,
            "s": self.it_vars["s"][self.it_update, uid].tolist()
            if not np.isnan(self.it_vars["s"][self.it_update, uid]).all()
            else None,
        }

        message = {"type": DataType.TRACE, "uid": uid, "data": data}

        self._send_message(message)

    def _broadcast_iteration_data(self, uid: Optional[int] = None):
        """Broadcast iteration data.

        Args:
            uid: Optional cell/unit ID. If None, broadcast for all cells.
        """
        uids = [uid] if uid is not None else range(self.ncell)

        for u in uids:
            data = {
                "iter": self.it_update,
                "scale": float(self.it_vars["scale"][self.it_update, u])
                if not np.isnan(self.it_vars["scale"][self.it_update, u])
                else None,
                "err": float(self.it_vars["err"][self.it_update, u])
                if not np.isnan(self.it_vars["err"][self.it_update, u])
                else None,
                "tau_d": float(self.it_vars["tau_d"][self.it_update, u])
                if not np.isnan(self.it_vars["tau_d"][self.it_update, u])
                else None,
                "tau_r": float(self.it_vars["tau_r"][self.it_update, u])
                if not np.isnan(self.it_vars["tau_r"][self.it_update, u])
                else None,
            }

            message = {"type": DataType.ITERATION, "uid": u, "data": data}

            self._send_message(message)

    def _broadcast_kernel_data(self, uid: int):
        """Broadcast kernel data for a specific cell.

        Args:
            uid: Cell/Unit ID
        """
        data = {
            "h": self.it_vars["h"][self.it_update, uid].tolist()
            if not np.isnan(self.it_vars["h"][self.it_update, uid]).all()
            else None,
            "h_fit": self.it_vars["h_fit"][self.it_update, uid].tolist()
            if not np.isnan(self.it_vars["h_fit"][self.it_update, uid]).all()
            else None,
        }

        message = {"type": DataType.KERNEL, "uid": uid, "data": data}

        self._send_message(message)

    def _send_message(self, message: dict):
        """Send a message to connected WebSocket clients.

        Args:
            message: The message to send
        """
        if self.client is not None:
            # In distributed environment, schedule the broadcast on the client
            self.client.submit(self._schedule_async_broadcast, message)
        else:
            # In local environment, schedule on the event loop
            self._schedule_async_broadcast(message)

    def _schedule_async_broadcast(self, message: dict):
        """Schedule an async broadcast on the event loop.

        Args:
            message: The message to broadcast
        """
        try:
            if asyncio.get_event_loop().is_running():
                asyncio.create_task(self._async_broadcast(message))
            else:
                # Create a new loop for the current thread if needed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._async_broadcast(message))
        except Exception as e:
            logger.error(f"Error scheduling broadcast: {e}")

    async def _async_broadcast(self, message: dict):
        """Broadcast a message to connected WebSocket clients.

        Args:
            message: The message to broadcast
        """
        if self.session_id:
            await connection_manager.broadcast_to_session(message, self.session_id)
        else:
            await connection_manager.broadcast(message)

    def set_iter(self, it: int):
        """Set the current iteration.

        Args:
            it: Iteration number
        """
        if self.it_update == self.it_view:
            self.it_view = it

        self.it_update = it

        # Broadcast iteration data
        self._broadcast_iteration_data()

    def update(self, uid: int = None, **kwargs):
        """Update dashboard data.

        Args:
            uid: Optional cell/unit ID. If None, update for all cells.
            **kwargs: Keyword arguments with updated values
        """
        if uid is None:
            uids = range(self.ncell)
        else:
            uids = [uid]

        for u in uids:
            trace_updated = False
            kernel_updated = False

            for vname, dat in kwargs.items():
                if vname in ["c", "s"]:
                    self.it_vars[vname][self.it_update, u, :] = dat
                    trace_updated = True
                elif vname in ["h", "h_fit"]:
                    self.it_vars[vname][self.it_update, u, :] = dat
                    kernel_updated = True
                elif vname in ["tau_d", "tau_r", "err", "scale"]:
                    try:
                        d = dat.item()
                    except (ValueError, AttributeError):
                        if hasattr(dat, "__getitem__") and len(dat) > u:
                            d = dat[u]
                        else:
                            d = dat

                    self.it_vars[vname][self.it_update, u] = d

                    # Update trace if scale changed
                    if vname == "scale":
                        trace_updated = True
                elif vname in ["penal_err"]:
                    for v in ["penal", "scale", "err"]:
                        self.it_vars[vname][self.it_update, u][v].append(dat[v])

            # Broadcast updates
            if trace_updated:
                self._broadcast_trace_data(u)

            if kernel_updated:
                self._broadcast_kernel_data(u)

        # Always broadcast iteration data
        self._broadcast_iteration_data(uid)

    def stop(self):
        """Stop the dashboard."""
        logger.info("Dashboard adapter stopped")

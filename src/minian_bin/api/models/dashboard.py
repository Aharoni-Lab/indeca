"""Pydantic models for dashboard data."""

from enum import Enum
from typing import Dict, List, Optional, Union, Any

import numpy as np
from pydantic import BaseModel, Field


class DataType(str, Enum):
    """Types of data that can be sent to the dashboard."""

    TRACE = "trace"
    ITERATION = "iteration"
    KERNEL = "kernel"
    ERROR = "error"
    SCALE = "scale"
    TAU = "tau"


class TraceData(BaseModel):
    """Model for trace data."""

    uid: int = Field(..., description="Cell/Unit ID")
    y: List[float] = Field(..., description="Original fluorescence trace")
    c: Optional[List[float]] = Field(None, description="Calcium trace")
    s: Optional[List[float]] = Field(None, description="Spike trace")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
        }


class IterationData(BaseModel):
    """Model for iteration data."""

    iter: int = Field(..., description="Iteration number")
    uid: int = Field(..., description="Cell/Unit ID")
    scale: Optional[float] = Field(None, description="Scale value")
    err: Optional[float] = Field(None, description="Error value")
    tau_d: Optional[float] = Field(None, description="Tau_d value")
    tau_r: Optional[float] = Field(None, description="Tau_r value")


class KernelData(BaseModel):
    """Model for kernel data."""

    uid: int = Field(..., description="Cell/Unit ID")
    h: List[float] = Field(..., description="Kernel values")
    h_fit: Optional[List[float]] = Field(None, description="Fitted kernel values")


class DashboardUpdateMessage(BaseModel):
    """Model for messages sent to the dashboard."""

    type: DataType = Field(..., description="Type of data being sent")
    uid: Optional[int] = Field(None, description="Cell/Unit ID if applicable")
    data: Dict[str, Any] = Field(..., description="Data payload")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
        }


class DashboardConnection(BaseModel):
    """Model for dashboard connection information."""

    client_id: str = Field(..., description="Unique client ID")
    session_id: Optional[str] = Field(None, description="Session ID")

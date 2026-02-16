"""Configuration for deconv module."""

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class DeconvConfig(BaseModel):
    """Configuration for DeconvBin."""

    model_config = {"frozen": True, "extra": "ignore"}

    coef_len: int = Field(
        100, description="Length of the coefficient kernel (e.g. calcium response)."
    )
    scale: float = Field(1.0, description="Global scaling factor.")
    penal: str | None = Field("l1", description="Penalty type ('l1', 'l0', etc.).")
    use_base: bool = Field(False, description="Whether to include a baseline term.")
    upsamp: int = Field(1, description="Upsampling factor.")
    norm: Literal["l1", "l2", "huber"] = Field(
        "l2", description="Norm for data fidelity ('l2', 'l1', 'huber')."
    )
    mixin: bool = Field(
        False, description="Whether to use mixed-integer programming (boolean spikes)."
    )
    backend: Literal["osqp", "cvxpy", "cuosqp"] = Field(
        "osqp",
        description="Solver backend ('osqp', 'cvxpy', 'cuosqp'). Note: emosqp requires codegen and is not supported.",
    )
    free_kernel: bool = Field(
        False,
        description="If True, use convolution constraint instead of AR constraint. Only supported with OSQP backends.",
    )
    nthres: int = Field(1000, description="Number of thresholds for thresholding step.")
    err_weighting: Optional[str] = Field(
        None, description="Error weighting method ('fft', 'corr', 'adaptive', or None)."
    )
    wt_trunc_thres: float = Field(
        1e-2, description="Threshold for truncating error weights."
    )
    masking_radius: Optional[int] = Field(
        None, description="Radius for masking around spikes."
    )
    pks_polish: bool = Field(True, description="Whether to polish peaks after solving.")
    th_min: float = Field(0.0, description="Minimum threshold.")
    th_max: float = Field(1.0, description="Maximum threshold.")
    density_thres: Optional[float] = Field(
        None, description="Max spike density threshold."
    )
    ncons_thres: Union[int, Literal["auto"], None] = Field(
        None, description="Max consecutive spikes threshold. If 'auto', upsamp + 1."
    )
    min_rel_scl: Union[float, Literal["auto"], None] = Field(
        "auto", description="Minimum relative scale. Use None to disable."
    )

    max_iter_l0: int = 30
    max_iter_penal: int = 500
    max_iter_scal: int = 50
    delta_l0: float = 1e-4
    delta_penal: float = 1e-4
    atol: float = 1e-3
    rtol: float = 1e-3
    Hlim: Optional[int] = 1e5

    @model_validator(mode="before")
    @classmethod
    def resolve_auto_fields(cls, data):
        # Resolve "auto" values before constructing the (frozen) model to avoid
        # returning a new instance from an "after" validator (pydantic warns).
        if not isinstance(data, dict):
            return data
        upsamp = data.get("upsamp", 1)
        if data.get("min_rel_scl") == "auto":
            data["min_rel_scl"] = 0.5 / upsamp
        if data.get("ncons_thres") == "auto":
            data["ncons_thres"] = upsamp + 1
        return data

    @model_validator(mode="after")
    def validate_penal(self):
        allowed = {None, "l0", "l1"}
        if self.penal not in allowed:
            raise ValueError(f"Unsupported penal type: {self.penal}")
        return self

    @model_validator(mode="after")
    def validate_compat(self):
        if self.free_kernel and self.backend == "cvxpy":
            raise ValueError("free_kernel=True is not supported with backend='cvxpy'")
        return self

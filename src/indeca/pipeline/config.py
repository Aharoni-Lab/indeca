"""Pydantic v2 configuration models for the binary pursuit pipeline.

These configs make the pipeline self-documenting, enable validation,
and allow easy CLI / config-file usage in the future.
"""

from typing import Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


class PreprocessConfig(BaseModel):
    """Configuration for preprocessing traces."""

    model_config = {"frozen": True}

    med_wnd: Optional[Union[int, Literal["auto"]]] = Field(
        None,
        description="Window size for median filtering. Use 'auto' to set to ar_kn_len, or None to skip.",
    )
    dff: bool = Field(
        True,
        description="Whether to compute dF/F normalization.",
    )


class InitConfig(BaseModel):
    """Configuration for AR parameter initialization."""

    model_config = {"frozen": True}

    tau_init: Optional[Tuple[float, float]] = Field(
        None,
        description="Initial tau values (tau_d, tau_r). If None, estimate from data.",
    )
    est_noise_freq: Optional[float] = Field(
        None,
        description="Frequency for noise estimation. None uses default.",
    )
    est_use_smooth: bool = Field(
        False,
        description="Whether to use smoothing during AR estimation.",
    )
    est_add_lag: int = Field(
        20,
        description="Additional lag samples for AR estimation.",
    )
    est_nevt: Optional[int] = Field(
        10,
        description="Number of top spike events for AR update. None uses all spikes.",
    )


class DeconvStageConfig(BaseModel):
    """Configuration for the deconvolution stage."""

    model_config = {"frozen": True}

    nthres: int = Field(
        1000,
        description="Number of thresholds for thresholding step.",
    )
    norm: Literal["l1", "l2", "huber"] = Field(
        "l2",
        description="Norm for data fidelity.",
    )
    penal: Optional[Literal["l0", "l1"]] = Field(
        None,
        description="Penalty type for sparsity.",
    )
    backend: Literal["osqp", "cvxpy", "cuosqp"] = Field(
        "osqp",
        description="Solver backend.",
    )
    err_weighting: Optional[Literal["fft", "corr", "adaptive"]] = Field(
        None,
        description="Error weighting method.",
    )
    use_base: bool = Field(
        True,
        description="Whether to include a baseline term.",
    )
    reset_scale: bool = Field(
        True,
        description="Whether to reset scale at each iteration.",
    )
    masking_radius: Optional[int] = Field(
        None,
        description="Radius for masking around spikes.",
    )
    pks_polish: bool = Field(
        True,
        description="Whether to polish peaks after solving.",
    )
    ncons_thres: Optional[Union[int, Literal["auto"]]] = Field(
        None,
        description="Max consecutive spikes threshold. 'auto' = upsamp + 1.",
    )
    min_rel_scl: Optional[Union[float, Literal["auto"]]] = Field(
        None,
        description="Minimum relative scale. 'auto' = 0.5 / upsamp.",
    )
    atol: float = Field(
        1e-3,
        description="Absolute tolerance for solver.",
    )


class ARUpdateConfig(BaseModel):
    """Configuration for AR parameter updates."""

    model_config = {"frozen": True}

    use_all: bool = Field(
        True,
        description="Whether to use all cells for AR update (shared tau).",
    )
    kn_len: int = Field(
        100,
        description="Kernel length for AR fitting.",
    )
    norm: Literal["l1", "l2"] = Field(
        "l2",
        description="Norm for AR fitting.",
    )
    prop_best: Optional[float] = Field(
        None,
        description="Proportion of best cells to use for AR update. None uses all.",
    )


class ConvergenceConfig(BaseModel):
    """Configuration for convergence criteria."""

    model_config = {"frozen": True}

    max_iters: int = Field(
        50,
        description="Maximum number of iterations.",
    )
    err_atol: float = Field(
        1e-4,
        description="Absolute error tolerance for convergence.",
    )
    err_rtol: float = Field(
        5e-2,
        description="Relative error tolerance for convergence.",
    )
    use_rel_err: bool = Field(
        True,
        description="Whether to use relative error for objective.",
    )
    n_best: Optional[int] = Field(
        3,
        description="Number of best iterations to average for spike selection.",
    )


class DeconvPipelineConfig(BaseModel):
    """Main configuration for the binary pursuit deconvolution pipeline.

    This is the top-level config that composes all sub-configs.
    """

    model_config = {"frozen": True}

    # Core parameters
    up_factor: int = Field(
        1,
        description="Upsampling factor for spike times.",
    )
    p: int = Field(
        2,
        description="Order of AR model (typically 2 for calcium imaging).",
    )

    # Sub-configs
    preprocess: PreprocessConfig = Field(
        default_factory=PreprocessConfig,
        description="Preprocessing configuration.",
    )
    init: InitConfig = Field(
        default_factory=InitConfig,
        description="Initialization configuration.",
    )
    deconv: DeconvStageConfig = Field(
        default_factory=DeconvStageConfig,
        description="Deconvolution stage configuration.",
    )
    ar_update: ARUpdateConfig = Field(
        default_factory=ARUpdateConfig,
        description="AR update configuration.",
    )
    convergence: ConvergenceConfig = Field(
        default_factory=ConvergenceConfig,
        description="Convergence configuration.",
    )

    @classmethod
    def from_legacy_kwargs(
        cls,
        *,
        up_factor: int = 1,
        p: int = 2,
        tau_init: Optional[Tuple[float, float]] = None,
        max_iters: int = 50,
        n_best: Optional[int] = 3,
        use_rel_err: bool = True,
        err_atol: float = 1e-4,
        err_rtol: float = 5e-2,
        est_noise_freq: Optional[float] = None,
        est_use_smooth: bool = False,
        est_add_lag: int = 20,
        est_nevt: Optional[int] = 10,
        med_wnd: Optional[Union[int, Literal["auto"]]] = None,
        dff: bool = True,
        deconv_nthres: int = 1000,
        deconv_norm: Literal["l1", "l2", "huber"] = "l2",
        deconv_atol: float = 1e-3,
        deconv_penal: Optional[Literal["l0", "l1"]] = None,
        deconv_backend: Literal["osqp", "cvxpy", "cuosqp"] = "osqp",
        deconv_err_weighting: Optional[Literal["fft", "corr", "adaptive"]] = None,
        deconv_use_base: bool = True,
        deconv_reset_scl: bool = True,
        deconv_masking_radius: Optional[int] = None,
        deconv_pks_polish: bool = True,
        deconv_ncons_thres: Optional[Union[int, Literal["auto"]]] = None,
        deconv_min_rel_scl: Optional[Union[float, Literal["auto"]]] = None,
        ar_use_all: bool = True,
        ar_kn_len: int = 100,
        ar_norm: Literal["l1", "l2"] = "l2",
        ar_prop_best: Optional[float] = None,
    ) -> "DeconvPipelineConfig":
        """Create a config from legacy keyword arguments.

        This factory method enables backward compatibility with the old
        flat-kwargs API.
        """
        return cls(
            up_factor=up_factor,
            p=p,
            preprocess=PreprocessConfig(
                med_wnd=med_wnd,
                dff=dff,
            ),
            init=InitConfig(
                tau_init=tau_init,
                est_noise_freq=est_noise_freq,
                est_use_smooth=est_use_smooth,
                est_add_lag=est_add_lag,
                est_nevt=est_nevt,
            ),
            deconv=DeconvStageConfig(
                nthres=deconv_nthres,
                norm=deconv_norm,
                penal=deconv_penal,
                backend=deconv_backend,
                err_weighting=deconv_err_weighting,
                use_base=deconv_use_base,
                reset_scale=deconv_reset_scl,
                masking_radius=deconv_masking_radius,
                pks_polish=deconv_pks_polish,
                ncons_thres=deconv_ncons_thres,
                min_rel_scl=deconv_min_rel_scl,
                atol=deconv_atol,
            ),
            ar_update=ARUpdateConfig(
                use_all=ar_use_all,
                kn_len=ar_kn_len,
                norm=ar_norm,
                prop_best=ar_prop_best,
            ),
            convergence=ConvergenceConfig(
                max_iters=max_iters,
                err_atol=err_atol,
                err_rtol=err_rtol,
                use_rel_err=use_rel_err,
                n_best=n_best,
            ),
        )

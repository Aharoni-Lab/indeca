"""Binary pursuit deconvolution pipeline.

This package provides the binary pursuit pipeline for spike inference
from calcium imaging traces.

Usage (recommended, config-based API)::

    from indeca.pipeline import pipeline_bin, DeconvPipelineConfig

    config = DeconvPipelineConfig(
        up_factor=2,
        convergence=ConvergenceConfig(max_iters=20),
    )
    opt_C, opt_S, metrics = pipeline_bin(Y, config=config)

Usage (legacy, deprecated)::

    from indeca.pipeline import pipeline_bin

    opt_C, opt_S, metrics = pipeline_bin(Y, up_factor=2, max_iters=20)

"""

# New config-based API (recommended)
from .binary_pursuit import pipeline_bin as pipeline_bin_new
from .config import (
    ARUpdateConfig,
    ConvergenceConfig,
    DeconvPipelineConfig,
    DeconvStageConfig,
    InitConfig,
    PreprocessConfig,
)

# Legacy API (deprecated, for backward compatibility)
from .pipeline import pipeline_bin, pipeline_bin_legacy

# Type definitions
from .types import (
    ARParams,
    ARUpdateResult,
    ConvergenceResult,
    DeconvStepResult,
    IterationState,
    PipelineResult,
)

__all__ = [
    # Main entry point (legacy for backward compat, emits deprecation warning)
    "pipeline_bin",
    # New config-based entry point
    "pipeline_bin_new",
    # Legacy explicit name
    "pipeline_bin_legacy",
    # Configuration classes
    "DeconvPipelineConfig",
    "PreprocessConfig",
    "InitConfig",
    "DeconvStageConfig",
    "ARUpdateConfig",
    "ConvergenceConfig",
    # Type definitions
    "ARParams",
    "DeconvStepResult",
    "ARUpdateResult",
    "ConvergenceResult",
    "IterationState",
    "PipelineResult",
]

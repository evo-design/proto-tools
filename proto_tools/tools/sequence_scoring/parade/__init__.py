"""PARADE cell-type-specific UTR activity and mRNA stability prediction."""

from proto_tools.tools.sequence_scoring.parade.parade_activity import (
    ParadeActivityConfig,
    ParadeActivityInput,
    ParadeActivityOutput,
    ParadeActivityResult,
    run_parade_activity,
)
from proto_tools.tools.sequence_scoring.parade.parade_gradient import (
    ParadeActivityTerm,
    ParadeGradientConfig,
    ParadeGradientInput,
    ParadeGradientOutput,
    ParadeGradientResult,
    ParadeTermGradient,
    run_parade_gradient,
)
from proto_tools.tools.sequence_scoring.parade.parade_stability import (
    ParadeStabilityConfig,
    ParadeStabilityInput,
    ParadeStabilityMetrics,
    ParadeStabilityOutput,
    ParadeStabilityResult,
    run_parade_stability,
)
from proto_tools.tools.sequence_scoring.parade.parade_stability_gradient import (
    ParadeStabilityGradientConfig,
    ParadeStabilityGradientInput,
    ParadeStabilityGradientMetrics,
    ParadeStabilityGradientOutput,
    ParadeStabilityLogRatioTerm,
    run_parade_stability_gradient,
)
from proto_tools.tools.sequence_scoring.parade.shared_data_models import (
    PARADE_CELL_LINE_NAMES,
    PARADE_CELL_TYPES,
    ParadeActivityMetrics,
    ParadeCellType,
    ParadeConstructType,
    resolve_cell_type,
)

__all__ = [
    "ParadeStabilityGradientConfig",
    "ParadeStabilityGradientInput",
    "ParadeStabilityLogRatioTerm",
    "ParadeStabilityGradientMetrics",
    "ParadeStabilityGradientOutput",
    "run_parade_stability_gradient",
    # Shared constants and types
    "PARADE_CELL_LINE_NAMES",
    "PARADE_CELL_TYPES",
    "ParadeCellType",
    "ParadeConstructType",
    "resolve_cell_type",
    # Activity
    "ParadeActivityInput",
    "ParadeActivityConfig",
    "ParadeActivityMetrics",
    "ParadeActivityOutput",
    "ParadeActivityResult",
    # Gradient
    "ParadeGradientInput",
    "ParadeGradientConfig",
    "ParadeGradientResult",
    "ParadeActivityTerm",
    "ParadeGradientOutput",
    "ParadeTermGradient",
    # Stability
    "ParadeStabilityInput",
    "ParadeStabilityConfig",
    "ParadeStabilityMetrics",
    "ParadeStabilityOutput",
    "ParadeStabilityResult",
    # Run functions
    "run_parade_activity",
    "run_parade_gradient",
    "run_parade_stability",
]

"""PARADE cell-type-specific UTR activity and mRNA stability prediction."""

from proto_tools.tools.sequence_scoring.parade.parade_activity import (
    ParadeActivityConfig,
    ParadeActivityInput,
    ParadeActivityOutput,
    ParadeActivityResult,
    run_parade_activity,
)
from proto_tools.tools.sequence_scoring.parade.parade_gradient import (
    ParadeGradientConfig,
    ParadeGradientInput,
    ParadeGradientLossTerm,
    ParadeGradientOutput,
    ParadeGradientSampleMetrics,
    ParadeObjectiveDirection,
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
from proto_tools.tools.sequence_scoring.parade.shared_data_models import (
    PARADE_CELL_TYPES,
    PARADE_CHECKPOINTS,
    PARADE_COMMIT,
    ParadeActivityMetrics,
    ParadeCellType,
    ParadeConstructType,
)

__all__ = [
    # Shared constants and types
    "PARADE_CELL_TYPES",
    "PARADE_CHECKPOINTS",
    "PARADE_COMMIT",
    "ParadeCellType",
    "ParadeConstructType",
    # Activity
    "ParadeActivityInput",
    "ParadeActivityConfig",
    "ParadeActivityMetrics",
    "ParadeActivityOutput",
    "ParadeActivityResult",
    # Gradient
    "ParadeGradientInput",
    "ParadeGradientConfig",
    "ParadeGradientLossTerm",
    "ParadeGradientOutput",
    "ParadeGradientSampleMetrics",
    "ParadeObjectiveDirection",
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

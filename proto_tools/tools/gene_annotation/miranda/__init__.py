"""miRanda microRNA target-site prediction."""

from proto_tools.tools.gene_annotation.miranda.miranda_scan import (
    MirandaConfig,
    MirandaInput,
    MirandaOutput,
    MirandaSequenceResult,
    MirandaTargetSite,
    run_miranda_scan,
)

__all__ = [
    "MirandaConfig",
    "MirandaInput",
    "MirandaOutput",
    "MirandaSequenceResult",
    "MirandaTargetSite",
    "run_miranda_scan",
]

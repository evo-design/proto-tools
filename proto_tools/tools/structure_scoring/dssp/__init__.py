"""DSSP secondary-structure assignment."""

from proto_tools.tools.structure_scoring.dssp.dssp import (
    DSSPSecondaryStructureConfig,
    DSSPSecondaryStructureInput,
    DSSPSecondaryStructureMetrics,
    DSSPSecondaryStructureOutput,
    DSSPStructureInput,
    run_dssp_secondary_structure,
)

__all__ = [
    "DSSPSecondaryStructureConfig",
    "DSSPSecondaryStructureInput",
    "DSSPSecondaryStructureMetrics",
    "DSSPSecondaryStructureOutput",
    "DSSPStructureInput",
    "run_dssp_secondary_structure",
]

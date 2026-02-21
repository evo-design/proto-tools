"""BioEmu conformational ensemble sampling tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator

from bio_programming_tools.entities.structures import (
    BFactorType,
    Structure,
    StructureEnsemble,
)
from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    StructurePredictionConfig,
    StructurePredictionInput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import (
    ConfigField,
    return_invalid_protein_chars,
)
from bio_programming_tools.utils.tool_instance import ToolInstance
from bio_programming_tools.utils.tool_io import BaseToolOutput

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
# Input:
class BioEmuInput(StructurePredictionInput):
    """Input object for BioEmu conformational ensemble sampling.

    Attributes:
        complexes (List[StructurePredictionComplex]): Protein complexes to sample.
            BioEmu supports monomer-only inputs, so each complex must contain one
            protein chain.
    """

    SUPPORTED_ENTITY_TYPES = {"protein"}
    ALLOWS_CHAIN_MODIFICATIONS = False

    @field_validator("complexes", mode="after")
    @classmethod
    def validate_complexes(
        cls, complexes: List[StructurePredictionComplex]
    ) -> List[StructurePredictionComplex]:
        """Validate BioEmu input constraints for each complex."""
        for comp_idx, comp in enumerate(complexes):
            if comp.num_chains() != 1:
                raise ValueError(
                    f"Complex {comp_idx} has {comp.num_chains()} chains. "
                    "BioEmu only supports single-chain proteins (monomers)."
                )

            chain_seq = comp.chains[0].sequence
            invalid_chars = return_invalid_protein_chars(chain_seq)
            if invalid_chars:
                raise ValueError(
                    f"Invalid protein characters in complex {comp_idx}: "
                    f"{', '.join(sorted(invalid_chars))}. "
                    "BioEmu requires standard amino acid sequences."
                )

            if len(chain_seq) > 500:
                logger.warning(
                    f"Complex {comp_idx} has {len(chain_seq)} residues. "
                    "BioEmu performance may degrade for sequences >500 residues."
                )

        return complexes


# Output:
class BioEmuOutput(BaseToolOutput):
    """Output object for BioEmu conformational ensemble sampling.

    Attributes:
        ensembles (List[StructureEnsemble]): Generated ensembles, one per
            input complex.
    """

    ensembles: List[StructureEnsemble] = Field(
        description="Generated protein conformational ensembles"
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["pdb", "json"]

    @property
    def output_format_default(self) -> str:
        return "pdb"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path)

        if file_format == "pdb":
            path.mkdir(parents=True, exist_ok=True)
            for ensemble_idx, ensemble in enumerate(self.ensembles):
                ensemble_dir = path / f"ensemble_{ensemble_idx}"
                ensemble_dir.mkdir(exist_ok=True)
                for frame_idx, structure in enumerate(ensemble.structures):
                    (ensemble_dir / f"conformation_{frame_idx}.pdb").write_text(
                        structure.structure_pdb
                    )
            return

        if file_format == "json":
            import json

            json_path = path.with_suffix(".json")
            payload = []
            for ensemble_idx, ensemble in enumerate(self.ensembles):
                payload.append(
                    {
                        "ensemble_id": ensemble_idx,
                        "num_conformations": len(ensemble.structures),
                        "conformations": [
                            {
                                "conformation_id": frame_idx,
                                "pdb_string": structure.structure_pdb,
                            }
                            for frame_idx, structure in enumerate(ensemble.structures)
                        ],
                    }
                )
            with open(json_path, "w") as handle:
                json.dump(payload, handle, indent=2)
            return

        raise ValueError(f"Unsupported format: {file_format}")


# Config:
class BioEmuConfig(StructurePredictionConfig):
    """Configuration object for BioEmu conformational ensemble sampling.

    Attributes:
        num_samples (int): Number of conformations to sample per input sequence.
        model_name (Literal["bioemu-v1.0", "bioemu-v1.1"]): BioEmu model variant.
        filter_samples (bool): Whether to filter lower-quality generated samples.
        batch_size (int): Batch size control for BioEmu internal sampling.
        output_dir (Optional[str]): Optional directory for raw BioEmu outputs.
        device (str): Inference device (inherited).
        verbose (bool): Verbose logging toggle (inherited).
    """

    num_samples: int = ConfigField(
        title="Number of Samples",
        default=500,
        ge=1,
        description="Number of conformations to sample per sequence",
    )
    model_name: Literal["bioemu-v1.0", "bioemu-v1.1"] = ConfigField(
        title="Model Name",
        default="bioemu-v1.1",
        description="BioEmu model variant to use",
        advanced=True,
        reload_on_change=True,
    )
    filter_samples: bool = ConfigField(
        title="Filter Samples",
        default=True,
        description="Whether to filter generated samples using BioEmu quality checks",
        advanced=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=10,
        ge=1,
        description="Batch size control for BioEmu internal sampling",
        advanced=True,
    )
    output_dir: Optional[str] = ConfigField(
        title="Output Directory",
        default=None,
        description="Optional directory for raw BioEmu output files",
        hidden=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="bioemu-sample",
    label="BioEmu Conformational Ensemble Sampling",
    category="structure_dynamics",
    input=BioEmuInput,
    config=BioEmuConfig,
    output=BioEmuOutput,
    description="Protein conformational ensemble sampling using BioEmu",
    uses_gpu=True,
)
def run_bioemu(inputs: BioEmuInput, config: BioEmuConfig, instance=None) -> BioEmuOutput:
    """Generate protein conformational ensembles using BioEmu."""
    logger.debug("Using local venv for BioEmu conformational sampling")

    output = ToolInstance.dispatch(
        "bioemu",
        {
            "sequences": [complex_.chains[0].sequence for complex_ in inputs.complexes],
            "num_samples": config.num_samples,
            "model_name": config.model_name,
            "filter_samples": config.filter_samples,
            "batch_size": config.batch_size,
            "device": config.device,
            "output_dir": config.output_dir,
            "verbose": config.verbose,
        },
        instance=instance,
        verbose=config.verbose,
        reload_on=type(config).reload_fields(),
    )
    raw_results = output["results"]

    ensembles: List[StructureEnsemble] = []
    total_structures = 0
    for comp_idx, comp in enumerate(inputs.complexes):
        sequence = comp.chains[0].sequence
        result = raw_results[comp_idx]

        structures = _pdb_frames_to_structures(
            pdb_frames=result["pdb_frames"], comp_idx=comp_idx
        )
        ensemble = StructureEnsemble(structures=structures, sequence=sequence)
        ensembles.append(ensemble)
        total_structures += len(structures)

    return BioEmuOutput(
        ensembles=ensembles,
        metadata={
            "num_complexes": len(inputs.complexes),
            "total_structures": total_structures,
            "model_name": config.model_name,
        },
    )


# ============================================================================
# Helpers
# ============================================================================
def _pdb_frames_to_structures(
    pdb_frames: List[str],
    comp_idx: int,
) -> List[Structure]:
    """Convert PDB frame strings to Structure objects."""
    structures = []
    for frame_idx, pdb_content in enumerate(pdb_frames):
        structures.append(
            Structure(
                structure_filepath_or_content=pdb_content,
                b_factor_type=BFactorType.UNSPECIFIED,
                metrics={
                    "ensemble_idx": comp_idx,
                    "frame_idx": frame_idx,
                },
                source="bioemu-ensemble",
            )
        )
    return structures

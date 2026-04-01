"""proto_tools/tools/structure_dynamics/bioemu/bioemu_sample.py.

BioEmu conformational ensemble sampling tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator

from proto_tools.entities.structures import (
    BFactorType,
    Structure,
    StructureEnsemble,
)
from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    StructurePredictionConfig,
    StructurePredictionInput,
    _preprocess_structure_prediction_msas,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseToolOutput,
    ConfigField,
    ToolInstance,
    return_invalid_protein_chars,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
# Input:
class BioEmuInput(StructurePredictionInput):
    """Input object for BioEmu conformational ensemble sampling.

    Attributes:
        complexes (list[StructurePredictionComplex]): Protein complexes to sample.
            BioEmu supports monomer-only inputs, so each complex must contain one
            protein chain.
        msas (dict[str, MSA] | None): Pre-computed MSAs keyed by protein sequence.
            Populated by preprocess() or supplied directly. Default: None.
    """

    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein"}
    ALLOWS_CHAIN_MODIFICATIONS: ClassVar[bool] = False

    @field_validator("complexes", mode="after")
    @classmethod
    def validate_complexes(cls, complexes: list[StructurePredictionComplex]) -> list[StructurePredictionComplex]:
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
        ensembles (list[StructureEnsemble]): Generated ensembles, one per
            input complex.
    """

    ensembles: list[StructureEnsemble] = Field(description="Generated protein conformational ensembles")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["pdb", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "pdb"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path)

        if file_format == "pdb":
            path.mkdir(parents=True, exist_ok=True)
            for ensemble_idx, ensemble in enumerate(self.ensembles):
                ensemble_dir = path / f"ensemble_{ensemble_idx}"
                ensemble_dir.mkdir(exist_ok=True)
                for frame_idx, structure in enumerate(ensemble.structures):
                    (ensemble_dir / f"conformation_{frame_idx}.pdb").write_text(structure.structure_pdb)
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

    BioEmu always requires MSA-derived Evoformer embeddings. Unlike structure
    prediction tools (which have an optional ``use_msa`` toggle), this config
    always runs ColabFold search during preprocessing to generate MSAs. If MSAs
    are pre-supplied on the input, the search is skipped.

    Attributes:
        num_samples (int): Number of conformations to sample per input sequence.
        model_name (Literal['bioemu-v1.0', 'bioemu-v1.1']): BioEmu model variant.
        filter_samples (bool): Whether to filter lower-quality generated samples.
        batch_size (int): Batch size control for BioEmu internal sampling.
        output_dir (str | None): Optional directory for raw BioEmu outputs.
        colabfold_search_config (ColabfoldSearchConfig | None): Configuration for
            ColabFold MSA search. Default: Uses ColabfoldSearchConfig defaults.
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
    output_dir: str | None = ConfigField(
        title="Output Directory",
        default=None,
        description="Optional directory for raw BioEmu output files",
        hidden=True,
    )
    colabfold_search_config: ColabfoldSearchConfig | None = ConfigField(
        title="ColabFold Search Config",
        default=None,
        description="Configuration for ColabFold MSA search. If None, uses default settings.",
        hidden=True,
    )

    def preprocess(self, inputs: StructurePredictionInput) -> StructurePredictionInput:  # type: ignore[override]
        """Generate MSAs via ColabFold search (always — BioEmu requires them).

        Skips the search if MSAs are already pre-supplied on ``inputs.msas``.

        Args:
            inputs (StructurePredictionInput): Structure prediction input containing
                complexes and optional pre-computed MSAs.

        Returns:
            StructurePredictionInput: Updated inputs with ``msas`` field populated.
        """
        if self.colabfold_search_config is None:
            self.colabfold_search_config = ColabfoldSearchConfig()
        self.colabfold_search_config.verbose = self.verbose
        return _preprocess_structure_prediction_msas(inputs, self.colabfold_search_config, self.verbose)


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return BioEmuInput(complexes=["MKTL"])  # type: ignore[list-item]


@tool(
    key="bioemu-sample",
    label="BioEmu Conformational Ensemble Sampling",
    category="structure_dynamics",
    input_class=BioEmuInput,
    config_class=BioEmuConfig,
    output_class=BioEmuOutput,
    description="Protein conformational ensemble sampling using BioEmu",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="complexes",
    iterable_output_field="ensembles",
)
def run_bioemu(inputs: BioEmuInput, config: BioEmuConfig | None = None, instance: Any = None) -> BioEmuOutput:
    """Generate protein conformational ensembles using BioEmu."""
    logger.debug("Using local venv for BioEmu conformational sampling")

    # Extract pre-computed MSA A3M strings (populated by preprocess or user)
    msa_a3m_contents = {}
    if inputs.msas:
        for complex_ in inputs.complexes:
            seq = complex_.chains[0].sequence
            msa = inputs.msas.get(seq)
            if msa is not None:
                msa_a3m_contents[seq] = msa.to_a3m_string()
                if config.verbose:  # type: ignore[union-attr]
                    logger.info(f"Loaded MSA for sequence (length {len(seq)}): {len(msa)} homologs")

    output = ToolInstance.dispatch(
        "bioemu",
        {
            "sequences": [complex_.chains[0].sequence for complex_ in inputs.complexes],
            "msa_a3m_contents": msa_a3m_contents,
            "num_samples": config.num_samples,  # type: ignore[union-attr]
            "model_name": config.model_name,  # type: ignore[union-attr]
            "filter_samples": config.filter_samples,  # type: ignore[union-attr]
            "batch_size": config.batch_size,  # type: ignore[union-attr]
            "device": config.device,  # type: ignore[union-attr]
            "output_dir": config.output_dir,  # type: ignore[union-attr]
            "verbose": config.verbose,  # type: ignore[union-attr]
        },
        instance=instance,
        config=config,
    )
    raw_results = output["results"]

    ensembles: list[StructureEnsemble] = []
    total_structures = 0
    for comp_idx, comp in enumerate(inputs.complexes):
        sequence = comp.chains[0].sequence
        result = raw_results[comp_idx]

        structures = _pdb_frames_to_structures(pdb_frames=result["pdb_frames"], comp_idx=comp_idx)
        ensemble = StructureEnsemble(structures=structures, sequence=sequence)
        ensembles.append(ensemble)
        total_structures += len(structures)

    return BioEmuOutput(
        ensembles=ensembles,
        metadata={
            "num_complexes": len(inputs.complexes),
            "total_structures": total_structures,
            "model_name": config.model_name,  # type: ignore[union-attr]
        },
    )


# ============================================================================
# Helpers
# ============================================================================
def _pdb_frames_to_structures(
    pdb_frames: list[str],
    comp_idx: int,
) -> list[Structure]:
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

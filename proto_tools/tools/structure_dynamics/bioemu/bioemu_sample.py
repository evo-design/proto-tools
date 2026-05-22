"""proto_tools/tools/structure_dynamics/bioemu/bioemu_sample.py.

BioEmu conformational ensemble sampling tool.
"""

import json
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
from proto_tools.utils.tool_io import Metrics

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

            chain_seq = comp.chain_sequences[0]  # SUPPORTED_ENTITY_TYPES rejects ligands upstream
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

    ensembles: list[StructureEnsemble] = Field(
        title="Ensembles",
        description="Generated protein conformational ensembles",
    )

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
        model_name (Literal['bioemu-v1.0', 'bioemu-v1.1', 'bioemu-v1.2']):
            Checkpoint variant (v1.1 = Science paper; v1.2 = extended MD + folding-FE).
        filter_samples (bool): Drop unphysical samples (steric clashes, chain breaks).
        batch_size (int): Upstream's ``batch_size_100``; effective batch is
            ``batch_size * (100 / L) ** 2``.
        denoiser_type (Literal['dpm', 'heun']): Sampler algorithm — ``dpm``
            is 50 deterministic steps; ``heun`` is stochastic.
        denoiser_config (str | None): Path to a custom denoiser/steering YAML;
            overrides ``denoiser_type`` when set.
        msa_host_url (str | None): Override the ColabFold MMseqs2 MSA server URL.
        cache_embeds_dir (str | None): Directory to cache MSA embeddings across runs.
        cache_so3_dir (str | None): Directory to cache SO3 precomputations across runs.
        output_dir (str | None): Optional directory for raw BioEmu outputs.
        colabfold_search_config (ColabfoldSearchConfig | None): ColabFold MSA
            search config. Defaults are used when ``None``.
        include_pae_matrix (bool): Inherited but unused (no PAE in conformational sampling).
        device (str): Inference device (inherited).
        verbose (bool): Verbose logging toggle (inherited).
    """

    num_samples: int = ConfigField(
        title="Number of Samples",
        default=500,
        ge=1,
        description="Number of conformations to sample per sequence",
    )
    model_name: Literal["bioemu-v1.0", "bioemu-v1.1", "bioemu-v1.2"] = ConfigField(
        title="Model Name",
        default="bioemu-v1.1",
        description="BioEmu checkpoint (v1.1 = Science paper; v1.2 = extended MD + folding-FE training)",
        reload_on_change=True,
    )
    filter_samples: bool = ConfigField(
        title="Filter Samples",
        default=True,
        description="Drop unphysical samples (steric clashes, chain discontinuities)",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=10,
        ge=1,
        description="Batch size at L=100; effective batch scales as batch_size * (100/L)^2",
    )
    denoiser_type: Literal["dpm", "heun"] = ConfigField(
        title="Denoiser Type",
        default="dpm",
        description="Diffusion sampler algorithm (dpm = 50 deterministic steps; heun = stochastic)",
    )
    denoiser_config: str | None = ConfigField(
        title="Denoiser Config Path",
        default=None,
        description="Path to a custom denoiser/steering YAML; overrides denoiser_type when set",
        examples=["physical_steering.yaml"],
    )
    msa_host_url: str | None = ConfigField(
        title="MSA Host URL",
        default=None,
        description="Override the ColabFold MMseqs2 MSA server URL",
    )
    cache_embeds_dir: str | None = ConfigField(
        title="MSA Embeds Cache Dir",
        default=None,
        description="Directory to cache MSA embeddings across runs",
        include_in_key=False,
    )
    cache_so3_dir: str | None = ConfigField(
        title="SO3 Cache Dir",
        default=None,
        description="Directory to cache SO3 precomputations across runs",
        include_in_key=False,
    )
    output_dir: str | None = ConfigField(
        title="Output Directory",
        default=None,
        description="Optional directory for raw BioEmu output files",
        include_in_key=False,
    )
    colabfold_search_config: ColabfoldSearchConfig | None = ConfigField(
        title="ColabFold Search Config",
        default=None,
        description="Configuration for ColabFold MSA search; None uses defaults",
    )
    include_pae_matrix: bool = ConfigField(
        title="Include PAE Matrix",
        default=False,
        description="Unused by BioEmu (inherited from StructurePredictionConfig)",
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
    from proto_tools.tools.sequence_alignment.msas import MSA

    a3m_path = Path(__file__).parent / "examples" / "example.a3m"
    fixture_msa = MSA.from_file(a3m_path)
    sequence = "MKTL"
    return BioEmuInput(complexes=[sequence], msas={sequence: fixture_msa})  # type: ignore[list-item]


@tool(
    key="bioemu-sample",
    label="BioEmu Conformational Ensemble Sampling",
    category="structure_dynamics",
    input_class=BioEmuInput,
    config_class=BioEmuConfig,
    output_class=BioEmuOutput,
    description="Protein conformational ensemble sampling using BioEmu",
    uses_gpu=True,
    stochastic=True,
    example_input=example_input,
    iterable_input_field="complexes",
    iterable_output_field="ensembles",
)
def run_bioemu(inputs: BioEmuInput, config: BioEmuConfig, instance: Any = None) -> BioEmuOutput:
    """Generate protein conformational ensembles using BioEmu."""
    logger.debug("Using local venv for BioEmu conformational sampling")

    # Extract pre-computed MSA A3M strings (populated by preprocess or user)
    msa_a3m_contents = {}
    if inputs.msas:
        for complex_ in inputs.complexes:
            seq = complex_.chain_sequences[0]
            msa = inputs.msas.get(seq)
            if msa is not None:
                msa_a3m_contents[seq] = msa.to_a3m_string()
                if config.verbose:
                    logger.info(f"Loaded MSA for sequence (length {len(seq)}): {len(msa)} homologs")

    output = ToolInstance.dispatch(
        "bioemu",
        {
            "operation": "sample",
            "sequences": [complex_.chain_sequences[0] for complex_ in inputs.complexes],
            "msa_a3m_contents": msa_a3m_contents,
            "num_samples": config.num_samples,
            "model_name": config.model_name,
            "filter_samples": config.filter_samples,
            "batch_size": config.batch_size,
            "denoiser_type": config.denoiser_type,
            "denoiser_config": config.denoiser_config,
            "msa_host_url": config.msa_host_url,
            "cache_embeds_dir": config.cache_embeds_dir,
            "cache_so3_dir": config.cache_so3_dir,
            "device": config.device,
            "output_dir": config.output_dir,
            "seed": config.seed,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )
    raw_results = output["results"]

    ensembles: list[StructureEnsemble] = []
    total_structures = 0
    for comp_idx, comp in enumerate(inputs.complexes):
        sequence = comp.chain_sequences[0]
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
            "model_name": config.model_name,
            "denoiser_type": config.denoiser_type,
            "denoiser_config": config.denoiser_config,
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
                structure=pdb_content,
                b_factor_type=BFactorType.UNSPECIFIED,
                metrics=Metrics(
                    ensemble_idx=comp_idx,
                    frame_idx=frame_idx,
                ),
                source="bioemu-ensemble",
            )
        )
    return structures

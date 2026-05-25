"""proto_tools/tools/structure_prediction/alphafold3/alphafold3.py.

Protein structure prediction using AlphaFold3.

This module provides standardized interfaces for protein structure prediction
using AlphaFold3 from Google DeepMind.
"""

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from typing import Any, ClassVar

from proto_tools.utils.progress import progress_bar

logger = logging.getLogger(__name__)

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools.structure_prediction.shared_data_models import (
    Complex,
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
    chain_label,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance
from proto_tools.utils.tool_io import Metrics, MetricSpec

# Type alias for AlphaFold3 JSON format
AlphaFold3JSON = dict[str, Any]


# ============================================================================
# Data Models
# ============================================================================
# Input:
class AlphaFold3Input(StructurePredictionInput):
    """Input object for AlphaFold3 structure prediction.

    This class defines the input parameters for predicting 3D structures of proteins,
    nucleic acids, and ligands using AlphaFold3.

    Inherits from ``StructurePredictionInput``.

    Attributes:
        complexes (list[Complex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain one or more sequences of proteins, DNA, RNA, or ligands.
        msas (dict[str, MSA] | None): Pre-computed MSAs keyed by protein sequence.
            Populated by preprocess() or supplied directly. Default: None.
    """

    # AlphaFold3 supports all standard entity types except glycan
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class AlphaFold3Metrics(Metrics):
    """Per-structure metrics emitted by AlphaFold3 prediction.

    Metrics documented in ``metric_spec``:
        avg_plddt (float): Average predicted LDDT score (0-100). Always present.
        avg_pae (float): Average predicted aligned error. Always present.
        pae (list[list[float]]): Full per-residue PAE matrix in Å. Present when include_pae_matrix=True.
        ptm (float): Predicted TM-score (0-1). Depends on model output.
        iptm (float): Interface predicted TM-score (0-1). Depends on model output.
        ranking_score (float): AlphaFold3 ranking score. Depends on model output.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "avg_plddt": {"availability": "always", "type": "float", "min": 0.0, "max": 100.0},
        "avg_pae": {"availability": "always", "type": "float", "min": 0.0, "max": None},
        "pae": {
            "availability": "when include_pae_matrix=True",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": None,
        },
        "ptm": {"availability": "depends on model output", "type": "float", "min": 0.0, "max": 1.0},
        "iptm": {"availability": "depends on model output", "type": "float", "min": 0.0, "max": 1.0},
        "ranking_score": {"availability": "depends on model output", "type": "float", "min": None, "max": None},
    }
    primary_metric: str | None = "avg_plddt"


# Output:
class AlphaFold3Output(StructurePredictionOutput):
    """AlphaFold3 prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures, each carrying an
            :class:`AlphaFold3Metrics` instance on ``.metrics``.
    """


# Config:
class AlphaFold3Config(MSAStructurePredictionConfig):
    """Configuration object for AlphaFold3 structure prediction.

    This class defines configuration parameters for running AlphaFold3, a state-of-the-art
    multi-modal structure predictor.

    Inherits from ``MSAStructurePredictionConfig``.

    Attributes:
        name (str): Name of the folding job. Default: ``"af3_job"``.

        seeds (list[int]): Seeds to use for AlphaFold3 when the common
            ``BaseConfig.seed`` field is unset. Default: ``[0]``. Note:
            AlphaFold3 will do five diffusion samples per seed, so this often
            can be set to a single seed. More seeds are required for complex
            docking tasks, such as antibody-antigen docking.

        output_dir (str | None): Path prefix for the AlphaFold3 output directory.
            Appends ``_af3_results`` to the provided string. If ``None`` (default),
            uses a temporary directory that is automatically cleaned up after inference.
            If specified, creates a persistent directory at the given path that will
            NOT be automatically deleted. Default: ``None``.

        model_dir (str | None): Local path to the directory containing AlphaFold3
            model parameters (a single ``.bin`` or ``.bin.zst`` file per DeepMind's
            release layout). If ``None`` (default), weights are resolved from
            ``PROTO_ALPHAFOLD3_WEIGHTS_DIR``, then ``PROTO_MODEL_CACHE``, then
            ``PROTO_HOME/proto_model_cache/alphafold3/`` (see ``notes/storage.md``).
            Default: ``None``.

        sif_path (str | None): Optional path to a pre-built AlphaFold3 Apptainer
            image (``.sif``). When set, the tool runs ``apptainer run`` against
            this image (which dispatches via the sif's ``%runscript``) instead of
            the in-env Python install. When ``None`` (default), inference.py looks
            for ``$VENV_PATH/alphafold3.sif`` (provisioned by setup.sh) and falls
            back to the env-based install if absent.
            Default: ``None``.

        num_recycles (int): Recycling iterations.
        num_diffusion_samples (int): Diffusion samples per seed;
            total candidates = len(seeds) * num_diffusion_samples.

        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using ColabFold search. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``True``.

        colabfold_search_config (ColabfoldSearchConfig | None): Configuration for
            ColabFold MSA search. Only used when ``use_msa=True``. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``None``.

        include_pae_matrix (bool): Inherited. Default: ``False``.

        device: Device to run the model on (``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        verbose: Whether to print status messages during execution. Inherited
            from ``StructurePredictionConfig``. Default: ``False``.

    """

    name: str = ConfigField(
        title="AlphaFold3 Job Name",
        default="af3_job",
        description="Name of the AlphaFold3 folding job",
    )

    seeds: list[int] = ConfigField(
        title="AlphaFold3 Seeds",
        default=[0],
        description="Seeds to use for AlphaFold3 when the common seed field is unset.",
    )

    output_dir: str | None = ConfigField(
        title="Output Directory Prefix",
        default=None,
        description="Prefix for the AlphaFold3 output directory. If None, uses temp directory with auto-cleanup.",
    )

    model_dir: str | None = ConfigField(
        title="AlphaFold3 Weights Directory",
        default=None,
        description="Directory with AlphaFold3 weights. If unset, resolves from env vars.",
    )

    sif_path: str | None = ConfigField(
        title="AlphaFold3 Apptainer Image",
        default=None,
        description="Pre-built AlphaFold3 .sif image. If unset, prefers provisioned sif then env.",
    )

    num_recycles: int = ConfigField(
        title="Recycling Iterations",
        default=10,
        ge=1,
        description="Recycling iterations through the model. Higher = more accurate but slower.",
    )
    num_diffusion_samples: int = ConfigField(
        title="Diffusion Samples per Seed",
        default=5,
        ge=1,
        description="Diffusion samples per seed; best by ranking score is kept. Total = len(seeds) x samples.",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return AlphaFold3Input(complexes=["MKTL"])  # type: ignore[list-item]


@contextlib.contextmanager
def _config_overrides_env(model_dir: str | None) -> Iterator[None]:
    """Propagate ``config.model_dir`` to ``PROTO_ALPHAFOLD3_WEIGHTS_DIR`` for the dispatch.

    Setup.sh's fail-fast weights precheck and the env_vars.txt passthrough only
    see env vars, not the config. When a caller supplies ``model_dir`` via the
    config it must take precedence (the env var is just a fallback for callers
    who don't set it). We temporarily mirror it onto the env var so setup.sh
    validates the right directory, then restore the original value on exit.

    Args:
        model_dir (str | None): Config-supplied weights directory. When falsy
            (``None`` or empty string), the env var is left untouched and
            resolution falls back to ``PROTO_ALPHAFOLD3_WEIGHTS_DIR`` →
            ``PROTO_MODEL_CACHE`` → ``PROTO_HOME`` defaults.
    """
    if not model_dir:
        yield
        return
    key = "PROTO_ALPHAFOLD3_WEIGHTS_DIR"
    sentinel = object()
    original: Any = os.environ.get(key, sentinel)
    os.environ[key] = model_dir
    try:
        yield
    finally:
        if original is sentinel:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


@tool(
    key="alphafold3-prediction",
    label="AlphaFold3 Structure Prediction",
    category="structure_prediction",
    input_class=AlphaFold3Input,
    config_class=AlphaFold3Config,
    output_class=AlphaFold3Output,
    metrics_class=AlphaFold3Metrics,
    description="Protein structure prediction using AlphaFold3",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="complexes",
    iterable_output_field="structures",
    cacheable=True,
    stochastic=True,
)
def run_alphafold3(
    inputs: AlphaFold3Input,
    config: AlphaFold3Config,
    instance: Any = None,
) -> AlphaFold3Output:
    """Predict protein 3D structures using AlphaFold3."""
    output_structures: list[Structure] = []

    # Seeded: single seed; unseeded: fresh random seeds, count taken from config.seeds for AF3 seed-averaging.
    if config.seed is not None:
        base_seeds = [config.seed]
    else:
        n_seeds = max(len(config.seeds), 1)
        base_seeds = [config.get_random_int() for _ in range(n_seeds)]

    with _config_overrides_env(config.model_dir):
        for dispatch_idx, comp in enumerate(
            progress_bar(
                inputs.complexes,
                desc="Folding structures (AlphaFold3)",
                unit="complex",
                total=len(inputs.complexes),
            )
        ):
            # Shift seeds per complex so duplicate inputs get non-overlapping seed slices.
            step = len(base_seeds)
            model_seeds = [s + dispatch_idx * step for s in base_seeds]

            input_json = _create_input_json_from_complex(
                comp,
                f"{config.name}_{dispatch_idx}",
                model_seeds,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                # Determine output directory
                if config.output_dir is None:
                    # Create inside temp directory for auto-cleanup
                    output_dir = os.path.join(temp_dir, f"{config.name}_{dispatch_idx}_af3_results")
                else:
                    # Create at specified path (persists after execution)
                    output_dir = f"{config.output_dir}_af3_results"

                # Create input directory for MSAs
                input_dir = os.path.join(output_dir, "af3_inputs")
                os.makedirs(input_dir, exist_ok=True)

                # Write pre-computed MSAs to A3M files
                if inputs.msas:
                    input_json = _assign_msas_to_input_json(input_json, inputs.msas, input_dir, config.verbose)  # type: ignore[arg-type]

                # Write input JSON to file for worker protocol
                input_json_path = os.path.join(input_dir, f"{config.name}_{dispatch_idx}.json")
                with open(input_json_path, "w") as f:
                    json.dump(input_json, f, indent=2)

                # Prepare dispatch input. The inference.py side picks the sif path
                # when either config.sif_path is set or setup.sh provisioned one at
                # $VENV_PATH/alphafold3.sif; otherwise it uses the env-based install.
                input_data = {
                    "input_json_path": input_json_path,
                    "output_dir": output_dir,
                    "device": config.device,
                    "model_dir": config.model_dir,
                    "sif_path": config.sif_path,
                    "verbose": config.verbose,
                    "include_pae_matrix": config.include_pae_matrix,
                    "num_recycles": config.num_recycles,
                    "num_diffusion_samples": config.num_diffusion_samples,
                }

                # Dispatch to worker (goes through DeviceManager)
                output_data = ToolInstance.dispatch(
                    "alphafold3",
                    input_data,
                    instance=instance,
                    config=config,
                )

                # Extract results from dict
                pdb_path = output_data["structure_pdb"]
                metrics = AlphaFold3Metrics(**output_data["metrics"])

                output_structures.append(
                    Structure.from_file(
                        pdb_path,
                        b_factor_type=BFactorType.PLDDT,
                        metrics=metrics,
                        source="alphafold3-prediction",
                    )
                )

    return AlphaFold3Output(
        structures=output_structures,
        metadata={
            "num_complexes": len(output_structures),
            "total_chains": sum(s.num_chains for s in output_structures),
        },
    )


# ============================================================================
# Helper Functions
# ============================================================================
def _assign_msas_to_input_json(
    input_json_data: AlphaFold3JSON,
    msas: dict[str, object],
    input_dir: str,
    verbose: bool = False,
) -> AlphaFold3JSON:
    """Write pre-computed MSAs to A3M files and assign paths to input JSON.

    Args:
        input_json_data (AlphaFold3JSON): AlphaFold3 input JSON dictionary to update with MSA paths.
        msas (dict[str, object]): Pre-computed MSAs keyed by sequence string.
        input_dir (str): Directory for MSA output files.
        verbose (bool): Whether to print progress messages.

    Returns:
        AlphaFold3JSON: Updated input_json_data with MSA paths populated.
    """
    msa_dir = os.path.join(input_dir, "msas")
    os.makedirs(msa_dir, exist_ok=True)

    for seq_idx, seq_entry in enumerate(input_json_data["sequences"]):
        if "protein" not in seq_entry:
            continue
        sequence = seq_entry["protein"]["sequence"]
        msa = msas.get(sequence)
        if msa is None:
            continue

        chain_id = seq_entry["protein"]["id"]
        if isinstance(chain_id, list):
            chain_id = chain_id[0]

        a3m_path = os.path.join(msa_dir, f"chain_{chain_id}_{seq_idx}.a3m")
        msa.to_a3m_file(a3m_path, query_index=0)  # type: ignore[attr-defined]

        rel_path = os.path.relpath(a3m_path, input_dir)
        seq_entry["protein"]["unpairedMsaPath"] = rel_path

        if verbose:
            logger.info(f"Assigned MSA to chain {chain_id} ({len(msa)} sequences)")  # type: ignore[arg-type]

    return input_json_data


def _create_input_json_from_complex(
    sp_complex: Complex,
    name: str,
    seed: int | list[int],
) -> AlphaFold3JSON:
    """Create input JSON data for AlphaFold3 inference from a list of components.

    The "alphafold3" JSON dialect is documented here:
    https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md

    Also converts SMILES strings to CCD code using data from here:
    https://files.wwpdb.org/pub/pdb/data/monomers/Components-smiles-stereo-oe.smi

    Args:
        sp_complex (Complex): Complex to predict.
        name (str): Name identifier for this prediction job.
        seed (int | list[int]): Random seed(s) for structure prediction.

    Returns:
        AlphaFold3JSON: Dictionary formatted for AlphaFold3 input JSON.
    """
    if isinstance(seed, int):
        seed = [seed]

    input_json_data = {
        "name": name,
        "modelSeeds": seed,
        "dialect": "alphafold3",
        "version": 2,
        "sequences": [],
    }

    for idx, chain in enumerate(sp_complex.chains):
        mol_type = chain.entity_type  # Currently, we use the same conventions as AlphaFold3.

        if isinstance(chain, Fragment):
            assert chain.smiles is not None  # noqa: S101 -- Fragment validator guarantees non-None
            ccd_code = chain.ccd_code
            if ccd_code is None:
                raise ValueError(
                    f"Unable to map SMILES to CCD code: {chain.smiles}. "
                    "AlphaFold3 requires CCD codes for ligands. "
                    "Provide a known CCD code or a SMILES string that matches the CCD database."
                )
            sequence_entry = {
                "ligand": {
                    "id": chain_label(idx),
                    "ccdCodes": [ccd_code],
                }
            }
            input_json_data["sequences"].append(sequence_entry)  # type: ignore[attr-defined]
            continue

        sequence = chain.sequence

        if mol_type == "dna" or mol_type == "rna":
            # Ignore MSA fields for DNA and RNA.
            sequence_entry = {
                mol_type: {
                    "id": chain_label(idx),
                    "sequence": sequence,
                }
            }

        else:
            # templates=[] is required when running with --norun_data_pipeline;
            # AF3's featurisation validates that every protein chain either has
            # templates (even an empty list explicitly set) or the data pipeline
            # has run to populate them.
            sequence_entry = {
                mol_type: {  # type: ignore[dict-item]
                    "id": chain_label(idx),
                    "sequence": sequence,
                    "pairedMsa": "",
                    "unpairedMsa": "",
                    "templates": [],
                }
            }

        if chain.modifications:
            alphafold3_modifications = []
            for mod in chain.modifications:
                if mol_type == "protein":
                    alphafold3_modifications.append({"ptmType": mod.modification_code, "ptmPosition": mod.position})
                elif mol_type in ("dna", "rna"):
                    alphafold3_modifications.append(
                        {
                            "modificationType": mod.modification_code,
                            "basePosition": mod.position,
                        }
                    )
            sequence_entry[mol_type]["modifications"] = alphafold3_modifications  # type: ignore[assignment, index]

        input_json_data["sequences"].append(sequence_entry)  # type: ignore[attr-defined]

    return input_json_data

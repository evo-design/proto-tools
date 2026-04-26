"""proto_tools/tools/structure_prediction/protenix/protenix.py.

Protein structure prediction using Protenix.

Example:
    >>> from proto_tools.tools.structure_prediction.protenix import run_protenix, ProtenixConfig
    >>> config = ProtenixConfig()
    >>> result = run_protenix(ProtenixInput(complexes=["MVLSPADKTNVKAAW"]), config)
    >>> print(f"Confidence: {result.structures[0].metrics['confidence_score']:.2f}")
"""

import json
import os
import tempfile
from logging import getLogger
from typing import Any, ClassVar, Literal

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.tools.structure_prediction.shared_data_models import (
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = getLogger(__name__)

ProtenixModelName = Literal[
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    "protenix_base_default_v0.5.0",
    "protenix_base_constraint_v0.5.0",
    "protenix_mini_esm_v0.5.0",
    "protenix_mini_ism_v0.5.0",
    "protenix_mini_default_v0.5.0",
    "protenix_tiny_default_v0.5.0",
]


# ============================================================================
# Data Models
# ============================================================================
# Input:
class ProtenixInput(StructurePredictionInput):
    """Input object for Protenix structure prediction.

    This class defines the input parameters for predicting 3D structures of proteins,
    DNA, RNA, and ligands using Protenix, an open-source reimplementation of AlphaFold3.

    Inherits from ``StructurePredictionInput``.

    Attributes:
        complexes (list[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain multiple chains of proteins, DNA, RNA, and/or ligands.
        msas (dict[str, MSA] | None): Pre-computed MSAs keyed by protein sequence.
            Populated by preprocess() or supplied directly. Default: None.

    Note:
        Protenix supports entity types: ``"protein"``, ``"dna"``, ``"rna"``, and ``"ligand"``.
        Entity types are automatically inferred if not explicitly provided. Invalid
        entity types will raise a validation error.

        Protenix supports chain modifications (PTMs, nucleotide modifications) using
        CCD codes, similar to AlphaFold3.
    """

    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True

    def to_json(self, complex_idx: int, name: str) -> list[dict[str, Any]]:
        """Convert a complex to Protenix JSON input format.

        Args:
            complex_idx (int): Index of the complex to convert
            name (str): Name identifier for the prediction job

        Returns:
            list[dict[str, Any]]: List containing a single dict in Protenix JSON format
        """
        comp = self.complexes[complex_idx]
        sequences = []

        for chain in comp.chains:
            entry: dict[str, Any]
            if isinstance(chain, Fragment):
                # Prefer CCD code: Protenix uses internal CCD parameterization,
                # avoiding RDKit↔Protenix SMILES canonicalization mismatches.
                ligand_str = f"CCD_{chain.ccd_code}" if chain.ccd_code else chain.smiles
                entry = {"ligand": {"ligand": ligand_str, "count": 1}}
                sequences.append(entry)
                continue

            e_type = chain.entity_type
            seq = chain.sequence

            if e_type == "protein":
                entry = {"proteinChain": {"sequence": seq, "count": 1}}
                if chain.modifications:
                    entry["proteinChain"]["modifications"] = [
                        {
                            "ptmType": f"CCD_{mod.modification_code}",
                            "ptmPosition": mod.position,
                        }
                        for mod in chain.modifications
                    ]

            elif e_type == "dna":
                entry = {"dnaSequence": {"sequence": seq, "count": 1}}
                if chain.modifications:
                    entry["dnaSequence"]["modifications"] = [
                        {
                            "modificationType": f"CCD_{mod.modification_code}",
                            "basePosition": mod.position,
                        }
                        for mod in chain.modifications
                    ]

            elif e_type == "rna":
                entry = {"rnaSequence": {"sequence": seq, "count": 1}}
                if chain.modifications:
                    entry["rnaSequence"]["modifications"] = [
                        {
                            "modificationType": f"CCD_{mod.modification_code}",
                            "basePosition": mod.position,
                        }
                        for mod in chain.modifications
                    ]

            sequences.append(entry)

        return [{"name": name, "sequences": sequences}]


class ProtenixMetrics(Metrics):
    """Per-structure metrics emitted by Protenix prediction.

    Metrics documented in ``metric_spec``:
        confidence_score (float): Protenix ranking score. Always present.
        ptm (float): Predicted TM-score (0-1). Always present.
        iptm (float): Interface predicted TM-score (0-1). Always present.
        avg_plddt (float): Average predicted LDDT score (0-1). Always present.
        gpde (float): Global predicted distance error. Always present.
        chain_ptm (list[float]): Per-chain pTM scores. Depends on model output.
        chain_plddt (list[float]): Per-chain pLDDT scores. Depends on model output.
        chain_pair_iptm (list[list[float]]): Pairwise chain ipTM. Depends on model output.
        has_clash (bool): Whether clashes were detected. Depends on model output.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "confidence_score": {"availability": "always", "type": "float", "min": None, "max": None},
        "ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
        "iptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
        "avg_plddt": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
        "gpde": {"availability": "always", "type": "float", "min": 0.0, "max": None},
        "chain_ptm": {"availability": "depends on model output", "type": "list[float]", "min": 0.0, "max": 1.0},
        "chain_plddt": {"availability": "depends on model output", "type": "list[float]", "min": 0.0, "max": 1.0},
        "chain_pair_iptm": {
            "availability": "depends on model output",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": 1.0,
        },
        "has_clash": {"availability": "depends on model output", "type": "bool", "min": None, "max": None},
    }
    primary_metric: str | None = "confidence_score"


# Output:
class ProtenixOutput(StructurePredictionOutput):
    """Protenix prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures, each carrying a
            :class:`ProtenixMetrics` instance on ``.metrics``.
    """


# Config:
class ProtenixConfig(MSAStructurePredictionConfig):
    """Configuration object for Protenix structure prediction.

    This class defines configuration parameters for running Protenix, an open-source
    reimplementation of AlphaFold3 by ByteDance Research. Protenix is a multi-modal
    structure prediction model supporting proteins, DNA, RNA, and ligands.

    Inherits from ``MSAStructurePredictionConfig``.

    Attributes:
        model_name (ProtenixModelName): Protenix model variant to use. Available models:

            **Base models** (full-parameter, highest accuracy, 10 recycle iterations,
            200 diffusion steps):

            - ``"protenix_base_default_v1.0.0"``: Recommended default. Best overall
              accuracy with inference-time scaling.
            - ``"protenix_base_20250630_v1.0.0"``: Same architecture as v1.0.0 but
              with a more recent training data cutoff (June 2025).
            - ``"protenix_base_default_v0.5.0"``: Earlier base model version.
            - ``"protenix_base_constraint_v0.5.0"``: Supports contact and pocket
              constraints via additional embedders for incorporating experimental
              structural priors.

            **Mini models** (lightweight, fewer parameters, 4 recycle iterations,
            5 diffusion steps):

            - ``"protenix_mini_default_v0.5.0"``: Compact model for faster predictions.
            - ``"protenix_mini_esm_v0.5.0"``: Uses ESM2 protein language model
              embeddings. Good when MSAs are unavailable.
            - ``"protenix_mini_ism_v0.5.0"``: Uses ISM (inverse structure model)
              embeddings. Alternative to ESM for MSA-free prediction.

            **Tiny model** (ultra-lightweight, fewest parameters, 4 recycle iterations,
            5 diffusion steps):

            - ``"protenix_tiny_default_v0.5.0"``: Smallest and fastest variant for
              high-throughput screening or resource-constrained environments.

            Default: ``"protenix_base_default_v1.0.0"``.

        seeds (list[int]): Random seeds for structure sampling. Each seed produces
            ``num_diffusion_samples`` independent structure samples. Multiple seeds
            increase diversity of the sampled conformations. A single seed is
            sufficient for most use cases; more seeds may help for challenging
            docking tasks such as antibody-antigen complexes.
            Default: ``[0]``.

        num_diffusion_samples (int): Number of independent structure samples to generate
            per seed. Only the best sample (by ranking score) is returned. Higher values
            explore more of the conformational space but increase computation time.
            Must be at least 1. Default: 5.

        num_diffusion_steps (int): Number of denoising steps in the diffusion process.
            Higher values produce more refined structures but are slower. Typical range:
            100-500. Must be at least 1. Default: 200.

        num_pairformer_cycles (int): Number of Pairformer recycling iterations through
            the model. Higher values produce more refined structures but increase
            computation time. Typical range: 3-20. Must be at least 0. Default: 10.

        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using ColabFold search. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``True``.

        colabfold_search_config (ColabfoldSearchConfig | None): Configuration for
            ColabFold MSA search. Only used when ``use_msa=True``. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``None``.

        device: Device to run the model on (e.g., ``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        include_pae_matrix (bool): Inherited; unused by Protenix. Default: ``False``.

        verbose: Whether to print status messages during execution including
            MSA generation, model loading, and prediction progress. Inherited from
            ``StructurePredictionConfig``. Default: ``False``.

        timeout (int): Maximum execution time in seconds. Base models need
            ~10-15 minutes on slower GPUs. Default: 1200.

    """

    model_name: ProtenixModelName = ConfigField(
        title="Model Name",
        default="protenix_base_default_v1.0.0",
        description="Protenix model variant to use for structure prediction",
        reload_on_change=True,
    )

    seeds: list[int] = ConfigField(
        title="Seeds",
        default=[0],
        description=(
            "Random seeds for structure sampling. Each seed produces num_diffusion_samples independent samples."
        ),
        advanced=True,
    )

    num_diffusion_samples: int = ConfigField(
        title="Number of Diffusion Samples",
        default=5,
        ge=1,
        description="Number of independent structure samples per seed (only the best is returned)",
        advanced=True,
    )

    num_diffusion_steps: int = ConfigField(
        title="Number of Diffusion Steps",
        default=200,
        ge=1,
        description="Number of denoising steps in the diffusion process (higher=more refined but slower)",
        advanced=True,
    )

    num_pairformer_cycles: int = ConfigField(
        title="Number of Pairformer Cycles",
        default=10,
        ge=0,
        description="Number of Pairformer recycling iterations (higher=more refined but slower)",
        advanced=True,
    )

    timeout: int = ConfigField(
        title="Timeout",
        default=1200,
        ge=1,
        description="Maximum execution time in seconds (base models need ~10-15 min on slower GPUs)",
        hidden=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProtenixInput(complexes=["MKTL"])  # type: ignore[list-item]


@tool(
    key="protenix-prediction",
    label="Protenix Structure Prediction",
    category="structure_prediction",
    input_class=ProtenixInput,
    config_class=ProtenixConfig,
    output_class=ProtenixOutput,
    description="Multi-modal structure prediction using Protenix (open-source AlphaFold3)",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="complexes",
    iterable_output_field="structures",
    cacheable=True,
)
def run_protenix(
    inputs: ProtenixInput,
    config: ProtenixConfig,
    instance: Any = None,
) -> ProtenixOutput:
    """Predict 3D structures using Protenix.

    Uses Protenix, an open-source reimplementation of AlphaFold3 by ByteDance
    Research, to predict 3D structures of proteins, DNA, RNA, ligands, and their
    complexes. Supports local GPU execution via isolated Python environments.

    All input complexes are batched into a single Protenix CLI call for efficiency,
    avoiding repeated model loading.

    Args:
        inputs (ProtenixInput): Validated input containing one or more complexes to
            predict structures for.
        config (ProtenixConfig): Validated Protenix configuration specifying model
            variant, MSA settings, diffusion parameters, and execution options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ProtenixOutput: Structured output containing:
            - ``structures``: List of ``Structure`` instances, one per input complex
            - Each structure includes coordinates and confidence metrics:

                confidence_score: Ranking score used to select the best sample.
                    Weighted combination of confidence metrics. Range: 0.0-1.0.

                ptm: Predicted Template Modeling score measuring overall structural
                    accuracy. Range: 0.0-1.0.

                iptm: Interface PTM score measuring confidence in inter-chain
                    interfaces. Range: 0.0-1.0.

                avg_plddt: Average per-residue confidence (pLDDT), normalized to
                    0.0-1.0. Higher values indicate more confident predictions.

                gpde: Global Predicted Distance Error in Angstroms. Lower values
                    indicate more confident relative positioning.

                chain_ptm: Per-chain PTM scores.

                chain_plddt: Per-chain pLDDT scores.

                chain_pair_iptm: Pairwise interface PTM scores
                    between all chain pairs.

                has_clash: Whether the predicted structure has steric clashes.

    See Also:
        - Protenix GitHub: https://github.com/bytedance/Protenix
        - Protenix paper: https://www.biorxiv.org/content/10.1101/2025.01.08.631857

    Example:
        >>> inputs = ProtenixInput(complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]])
        >>> config = ProtenixConfig(
        ...     model_name="protenix_base_default_v1.0.0",
        ...     num_diffusion_samples=5,
        ...     verbose=True,
        ... )
        >>> result = run_protenix(inputs, config)
        >>> print(f"Confidence: {result.structures[0].metrics['confidence_score']:.2f}")
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "protenix_output")
        os.makedirs(output_dir)

        # Build batched input JSON for all complexes
        batch_json = []
        for i in range(len(inputs.complexes)):
            batch_json.extend(inputs.to_json(i, name=f"protenix_job_{i}"))

        # Write pre-computed MSAs and inject paths into the batch JSON
        if inputs.msas:
            _write_msas_to_batch_json(batch_json, inputs, config, temp_dir)

        # Write the batch input JSON
        input_json_path = os.path.join(temp_dir, "protenix_input.json")
        with open(input_json_path, "w") as f:
            json.dump(batch_json, f)

        # Prepare input data for inference script.
        # When BaseConfig.seed is set explicitly (e.g. by the seed-reproducibility
        # tests) we override the protenix-specific ``seeds`` list with a single
        # entry derived from ``config.seed`` so the protenix CLI honours the
        # caller's seed instead of the field default ``[0]``.
        seeds_str = str(config.seed) if config.seed is not None else ",".join(str(s) for s in config.seeds)
        input_data = {
            "operation": "predict",
            "input_json_path": input_json_path,
            "output_dir": output_dir,
            "model_name": config.model_name,
            "seeds": seeds_str,
            "num_diffusion_samples": config.num_diffusion_samples,
            "num_diffusion_steps": config.num_diffusion_steps,
            "num_pairformer_cycles": config.num_pairformer_cycles,
            "use_msa": config.use_msa,
        }

        # Call the inference script (single batched call)
        logger.info(f"Running Protenix prediction for {len(inputs.complexes)} complex(es)...")

        input_data["device"] = config.device
        input_data["verbose"] = config.verbose
        output_data = ToolInstance.dispatch(
            "protenix",
            input_data,
            instance=instance,
            config=config,
        )

    # Parse results for each complex
    results = []
    for _i, job_result in enumerate(output_data["results"]):
        raw_metrics = job_result.get("metrics", {})

        def _maybe_float(key: str, scale: float = 1.0, raw: dict[str, Any] = raw_metrics) -> float | None:
            """Return ``float(raw[key]) * scale`` if the key is present, else ``None``.

            ``None`` is stripped by ``Metrics._exclude_none_values`` so absent metrics
            are truly absent in the output (vs. a silent ``0.0`` fallback that would
            be indistinguishable from a legitimate 0.0 value).
            """
            return float(raw[key]) * scale if key in raw else None

        metrics = ProtenixMetrics(
            confidence_score=_maybe_float("ranking_score"),
            ptm=_maybe_float("ptm"),
            iptm=_maybe_float("iptm"),
            avg_plddt=_maybe_float("plddt", scale=1 / 100.0),
            gpde=_maybe_float("gpde"),
            chain_ptm=raw_metrics.get("chain_ptm"),
            chain_plddt=raw_metrics.get("chain_plddt"),
            chain_pair_iptm=raw_metrics.get("chain_pair_iptm"),
            has_clash=raw_metrics.get("has_clash"),
        )

        results.append(
            Structure(
                structure=job_result["structure_cif_output"],
                b_factor_type=BFactorType.PLDDT,
                metrics=metrics,
                source="protenix-prediction",
            )
        )

    return ProtenixOutput(structures=results)


# ============================================================================
# Helper Functions
# ============================================================================
def _write_msas_to_batch_json(
    batch_json: list[dict[str, Any]],
    inputs: ProtenixInput,
    config: ProtenixConfig,
    temp_dir: str,
) -> None:
    """Write pre-computed MSAs to A3M files and inject paths into batch JSON.

    Args:
        batch_json (list[dict[str, Any]]): List of Protenix job dicts (modified in place).
        inputs (ProtenixInput): ProtenixInput with the original complexes and pre-computed MSAs.
        config (ProtenixConfig): ProtenixConfig with verbose setting.
        temp_dir (str): Directory for writing A3M files.
    """
    msa_dir = os.path.join(temp_dir, "msas")
    os.makedirs(msa_dir, exist_ok=True)

    for job_idx, job in enumerate(batch_json):
        sp_complex = inputs.complexes[job_idx]
        protein_seqs, protein_chain_ids = sp_complex.extract_protein_chains()

        if not protein_seqs:
            continue

        msa_paths: dict[str, str] = {}
        for seq, chain_id in zip(protein_seqs, protein_chain_ids, strict=False):
            if seq in inputs.msas:  # type: ignore[operator]
                a3m_path = os.path.join(msa_dir, f"job_{job_idx}_{chain_id}.a3m")
                inputs.msas[seq].to_a3m_file(a3m_path, query_index=0)  # type: ignore[index]
                msa_paths[chain_id] = a3m_path

                if config.verbose:
                    logger.info(
                        f"Assigned MSA to chain {chain_id} in complex {job_idx} ({len(inputs.msas[seq])} sequences)"  # type: ignore[index]
                    )

        # Inject MSA paths into the job's proteinChain entries
        for seq_entry in job["sequences"]:
            if "proteinChain" in seq_entry:
                chain_seq = seq_entry["proteinChain"]["sequence"]
                for chain_id, protein_seq in zip(protein_chain_ids, protein_seqs, strict=False):
                    if protein_seq == chain_seq and chain_id in msa_paths:
                        seq_entry["proteinChain"]["unpairedMsaPath"] = msa_paths[chain_id]
                        break

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

from pydantic import model_validator

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.tools.structure_prediction.shared_data_models import (
    Complex,
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
    chain_label,
    count_structure_tokens,
    normalize_output_chain_ids,
    unwrap_complex_msas,
    write_paired_a3m_with_uniprot_headers,
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
    "protenix-v2",
]

# Only protenix-v2 hard-rejects oversized inputs upstream (runner/inference.py:
# n_token > 2560 raises); other models adjust precision for large inputs instead.
_PROTENIX_V2_MODEL = "protenix-v2"
_PROTENIX_V2_MAX_TOKENS = 2560


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
        complexes (list[Complex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain multiple chains of proteins, DNA, RNA, and/or ligands.
        msas (list[ComplexMSAs] | None): Pre-computed MSAs, one
            entry per complex. Each entry is a ``ComplexMSAs`` (per-chain MSAs keyed by
            chain index); ``paired=True`` marks rows taxonomy-aligned across chains. Populated by preprocess() or supplied directly.
            Default: None.

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
        avg_pae (float): Mean of the per-token PAE matrix in Å. Always present.
        pae (list[list[float]]): Full per-token PAE matrix in Å. Present when include_pae_matrix=True.
        chain_ptm (list[float]): Per-chain pTM scores. Depends on model output.
        chain_plddt (list[float]): Per-chain pLDDT scores. Depends on model output.
        chain_pair_iptm (list[list[float]]): Pairwise chain ipTM. Depends on model output.
        has_clash (bool): Whether clashes were detected. Depends on model output.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "confidence_score": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "better_values_are": "higher",
        },
        "ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "iptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "avg_plddt": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "gpde": {"availability": "always", "type": "float", "min": 0.0, "max": None, "better_values_are": "lower"},
        "avg_pae": {"availability": "always", "type": "float", "min": 0.0, "max": None, "better_values_are": "lower"},
        "pae": {
            "availability": "when include_pae_matrix=True",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": None,
            "better_values_are": "lower",
        },
        "chain_ptm": {
            "availability": "depends on model output",
            "type": "list[float]",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "chain_plddt": {
            "availability": "depends on model output",
            "type": "list[float]",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "chain_pair_iptm": {
            "availability": "depends on model output",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "has_clash": {
            "availability": "depends on model output",
            "type": "bool",
            "min": None,
            "max": None,
            "better_values_are": "lower",
        },
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

            **Scaled-up model** (enhanced capacity, ~464M params, 10 recycle iterations,
            200 diffusion steps; **weights are gated by ByteDance, not auto-downloaded**):

            - ``"protenix-v2"``: Wider representation (``c_z=256``) with scaled-up
              modules. v2 weights are not currently distributed publicly; if you have
              a copy, place it under the configured weights directory. Inputs are
              capped at 2,560 tokens for this model (a hard upstream limit); the other
              models have no token cap.

            Default: ``"protenix_base_default_v1.0.0"``.

        seeds (list[int]): Random seeds for structure sampling. Each seed produces
            ``num_diffusion_samples`` independent structure samples. Multiple seeds
            increase diversity of the sampled conformations. A single seed is
            sufficient for most use cases; more seeds may help for challenging
            docking tasks such as antibody-antigen complexes.
            Default: ``[0]``.

        num_diffusion_samples (int): Independent structure samples per seed; only the
            best by ranking score is returned. Higher = more thorough but slower.
            Default 5 (matches upstream).

        num_diffusion_steps (int | None): Denoising steps in the diffusion process.
            ``None`` uses the upstream schedule: 200 for base/constraint, 5 for mini/tiny.
            Default ``None``.

        num_pairformer_cycles (int | None): Pairformer refinement passes through the model.
            ``None`` uses the upstream schedule: 10 for base/constraint, 4 for mini/tiny.
            Default ``None``.

        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using MMseqs2 homology search. Supplied MSAs are always
            used and override ``use_msa=False``. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``True``.

        pair_heterocomplex_msas (bool): Whether heterocomplex protein chains
            should use taxonomy-paired MSA generation. Inherited from
            ``MSAStructurePredictionConfig``. Default: ``True``.

        msa_search_config (Mmseqs2HomologySearchConfig | None): Configuration for
            MMseqs2 homology search (MSA generation). Only used when ``use_msa=True``.
            Inherited from ``MSAStructurePredictionConfig``. Default: ``None``.

        device: Device to run the model on (e.g., ``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        include_pae_matrix (bool): Attach ``pae`` (``avg_pae`` always emitted). Default: ``False``.

        verbose: Whether to print status messages during execution including
            MSA generation, model loading, and prediction progress. Inherited from
            ``StructurePredictionConfig``. Default: ``False``.

        timeout (int | None): Maximum execution time in seconds. Base models need
            ~10-15 minutes on slower GPUs. ``None`` waits indefinitely. Default: 1200.

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
        description="Random seeds for sampling; each seed produces num_diffusion_samples independent samples.",
    )

    num_diffusion_samples: int = ConfigField(
        title="Number of Diffusion Samples",
        default=5,
        ge=1,
        description="Structure samples per seed; best by ranking score is kept. Higher = more thorough but slower.",
    )

    num_diffusion_steps: int | None = ConfigField(
        title="Number of Diffusion Steps",
        default=None,
        ge=1,
        description="Denoising steps. Default depends on model_name (base/constraint=200, mini/tiny=5).",
    )

    num_pairformer_cycles: int | None = ConfigField(
        title="Number of Pairformer Cycles",
        default=None,
        ge=0,
        description="Pairformer refinement passes. Default depends on model_name (base/constraint=10, mini/tiny=4).",
    )

    timeout: int | None = ConfigField(
        title="Timeout",
        default=1200,
        ge=1,
        description="Maximum execution time in seconds (base models typically need 10-15 min on slower GPUs).",
        include_in_key=False,
    )

    @model_validator(mode="after")
    def resolve_schedule_defaults(self) -> "ProtenixConfig":
        """Fill cycles/steps from model_name when unset; preserve explicit values."""
        is_lightweight = "mini" in self.model_name or "tiny" in self.model_name
        cycles, steps = (4, 5) if is_lightweight else (10, 200)
        if self.num_pairformer_cycles is None:
            self.num_pairformer_cycles = cycles
        if self.num_diffusion_steps is None:
            self.num_diffusion_steps = steps
        return self


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
    metrics_class=ProtenixMetrics,
    description="Multi-modal structure prediction using Protenix (open-source AlphaFold3)",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["complexes", "msas"],
    iterable_output_field="structures",
    cacheable=True,
    stochastic=True,
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
    _enforce_token_limit(inputs.complexes, config.model_name)
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
            "include_pae_matrix": config.include_pae_matrix,
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
    for comp, job_result in zip(inputs.complexes, output_data["results"], strict=True):
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
            avg_pae=_maybe_float("avg_pae"),
            pae=raw_metrics.get("pae"),
            chain_ptm=raw_metrics.get("chain_ptm"),
            chain_plddt=raw_metrics.get("chain_plddt"),
            chain_pair_iptm=raw_metrics.get("chain_pair_iptm"),
            has_clash=raw_metrics.get("has_clash"),
        )

        structure = Structure(
            structure=job_result["structure_cif_output"],
            b_factor_type=BFactorType.PLDDT,
            metrics=metrics,
            source="protenix-prediction",
        )
        results.append(normalize_output_chain_ids(structure, comp.chains))

    return ProtenixOutput(structures=results)


# ============================================================================
# Helper Functions
# ============================================================================
def _enforce_token_limit(complexes: list[Complex], model_name: str) -> None:
    """Reject inputs over protenix-v2's hard 2,560-token cap; other models are uncapped."""
    if model_name != _PROTENIX_V2_MODEL:
        return
    for comp_idx, comp in enumerate(complexes):
        n_tokens = count_structure_tokens(comp.chains)
        if n_tokens > _PROTENIX_V2_MAX_TOKENS:
            raise ValueError(
                f"Complex {comp_idx} has {n_tokens} tokens, exceeding the protenix-v2 "
                f"model's {_PROTENIX_V2_MAX_TOKENS}-token limit."
            )


def _write_msas_to_batch_json(
    batch_json: list[dict[str, Any]],
    inputs: ProtenixInput,
    config: ProtenixConfig,
    temp_dir: str,
) -> None:
    """Write per-complex MSAs to A3M files and inject paths into batch JSON.

    For complexes with multiple protein chains the per-chain MSAs are also
    written as the ``pairedMsaPath`` (rows are taxonomy-aligned by position
    when sourced from a paired query).

    Args:
        batch_json (list[dict[str, Any]]): List of Protenix job dicts (modified in place).
        inputs (ProtenixInput): ProtenixInput with the original complexes and pre-computed MSAs.
        config (ProtenixConfig): ProtenixConfig with verbose setting.
        temp_dir (str): Directory for writing A3M files.
    """
    if inputs.msas is None:
        return

    msa_dir = os.path.join(temp_dir, "msas")
    os.makedirs(msa_dir, exist_ok=True)

    for job_idx, job in enumerate(batch_json):
        sp_complex = inputs.complexes[job_idx]
        per_chain_msas, unpaired_per_chain, is_paired = unwrap_complex_msas(inputs.msas[job_idx])

        protein_chain_indices = [
            i for i, ch in enumerate(sp_complex.chains) if hasattr(ch, "entity_type") and ch.entity_type == "protein"
        ]
        if not protein_chain_indices:
            continue
        n_protein_chains = len(protein_chain_indices)

        # Map chain_idx -> a3m path written for that chain.
        chain_idx_to_path: dict[int, str] = {}
        chain_idx_to_paired_path: dict[int, str] = {}
        for chain_idx in protein_chain_indices:
            msa = per_chain_msas.get(chain_idx)
            if msa is None:
                continue
            chain_id = (
                sp_complex.chains[chain_idx].id
                if getattr(sp_complex.chains[chain_idx], "id", None)
                else chain_label(chain_idx)
            )
            # unpairedMsaPath gets each chain's deep unpaired MSA when available, else the
            # primary MSA; Protenix block-diagonalizes the unpaired rows itself (AF3-style).
            unpaired_msa = (unpaired_per_chain or {}).get(chain_idx)
            if unpaired_msa is None:
                unpaired_msa = msa
            a3m_path = os.path.join(msa_dir, f"job_{job_idx}_chain_{chain_id}_{chain_idx}.a3m")
            unpaired_msa.to_a3m_file(a3m_path, query_index=0)
            chain_idx_to_path[chain_idx] = a3m_path
            if is_paired:
                paired_a3m_path = os.path.join(msa_dir, f"job_{job_idx}_chain_{chain_id}_{chain_idx}.paired.a3m")
                write_paired_a3m_with_uniprot_headers(msa, paired_a3m_path)
                chain_idx_to_paired_path[chain_idx] = paired_a3m_path

            if config.verbose:
                logger.info(
                    f"Assigned MSA to chain {chain_id} in complex {job_idx} "
                    f"(unpaired {len(unpaired_msa)} / paired {len(msa)} sequences)"
                )

        # Inject MSA paths into the job's proteinChain entries in positional order.
        proto_entry_iter = (s for s in job["sequences"] if "proteinChain" in s)
        for proto_pos, seq_entry in enumerate(proto_entry_iter):
            if proto_pos >= n_protein_chains:
                break
            chain_idx = protein_chain_indices[proto_pos]
            path = chain_idx_to_path.get(chain_idx)
            if path is None:
                continue
            seq_entry["proteinChain"]["unpairedMsaPath"] = path
            if is_paired:
                seq_entry["proteinChain"]["pairedMsaPath"] = chain_idx_to_paired_path[chain_idx]

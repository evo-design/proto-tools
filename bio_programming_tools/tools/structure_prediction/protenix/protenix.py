"""
protenix.py
Protein structure prediction using Protenix.

Example:
    >>> from bio_programming_tools.tools.structure_prediction.protenix import run_protenix, ProtenixConfig
    >>> config = ProtenixConfig()
    >>> result = run_protenix(ProtenixInput(complexes=["MVLSPADKTNVKAAW"]), config)
    >>> print(f"Confidence: {result.structures[0].metrics['confidence_score']:.2f}")

"""

from __future__ import annotations

import json
import os
import string
import tempfile
from logging import getLogger
from typing import List, Literal, Optional

from pydantic import model_validator

from bio_programming_tools.entities.structures import BFactorType, Structure
from bio_programming_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
    ColabfoldSearchInput,
    run_colabfold_search,
)
from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import ConfigField
from bio_programming_tools.utils.tool_cache import tool_cache_iterable

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
        complexes (List[StructurePredictionComplex]): List of complexes to predict
            structures for. Inherited from ``StructurePredictionInput``. Each complex
            can contain multiple chains of proteins, DNA, RNA, and/or ligands.

    Note:
        Protenix supports entity types: ``"protein"``, ``"dna"``, ``"rna"``, and ``"ligand"``.
        Entity types are automatically inferred if not explicitly provided. Invalid
        entity types will raise a validation error.

        Protenix supports chain modifications (PTMs, nucleotide modifications) using
        CCD codes, similar to AlphaFold3.
    """

    SUPPORTED_ENTITY_TYPES = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True

    def to_json(self, complex_idx: int, name: str) -> list[dict]:
        """Convert a complex to Protenix JSON input format.

        Args:
            complex_idx: Index of the complex to convert
            name: Name identifier for the prediction job

        Returns:
            List containing a single dict in Protenix JSON format
        """
        comp = self.complexes[complex_idx]
        sequences = []

        for chain in comp.chains:
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

            elif e_type == "ligand":
                entry = {"ligand": {"ligand": seq, "count": 1}}

            sequences.append(entry)

        return [{"name": name, "sequences": sequences}]


# Output:
ProtenixOutput = StructurePredictionOutput

# Config:
class ProtenixConfig(StructurePredictionConfig):
    """Configuration object for Protenix structure prediction.

    This class defines configuration parameters for running Protenix, an open-source
    reimplementation of AlphaFold3 by ByteDance Research. Protenix is a multi-modal
    structure prediction model supporting proteins, DNA, RNA, and ligands.

    Inherits from ``StructurePredictionConfig``.

    Attributes:
        model_name (str): Protenix model variant to use. Available models:

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

        seeds (List[int]): Random seeds for structure sampling. Each seed produces
            ``num_diffusion_samples`` independent structure samples. Multiple seeds
            increase diversity of the sampled conformations. A single seed is
            sufficient for most use cases; more seeds may help for challenging
            docking tasks such as antibody-antigen complexes.
            Default: ``[0]``.

        use_msa (bool): Whether to generate and use Multiple Sequence Alignments (MSAs)
            for protein chains using ColabFold search. If ``False``, runs in single-sequence
            mode without MSAs. Default: ``True``.

        colabfold_search_config (ColabfoldSearchConfig): Configuration for ColabFold
            MSA search. Controls search mode (local/remote), database paths, and other
            MSA generation parameters. Only used when ``use_msa=True``.
            Default: Uses ColabfoldSearchConfig defaults.

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

        device (str): Device to run the model on (e.g., ``"cuda"``, ``"cpu"``). Inherited
            from ``StructurePredictionConfig``. Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during execution including
            MSA generation, model loading, and prediction progress. Inherited from
            ``StructurePredictionConfig``. Default: ``False``.

    """

    model_name: ProtenixModelName = ConfigField(
        title="Model Name",
        default="protenix_base_default_v1.0.0",
        description="Protenix model variant to use for structure prediction",
        reload_on_change=True,
    )

    seeds: List[int] = ConfigField(
        title="Seeds",
        default=[0],
        description=(
            "Random seeds for structure sampling. Each seed produces "
            "num_diffusion_samples independent samples."
        ),
        advanced=True,
    )

    use_msa: bool = ConfigField(
        title="Use MSA",
        default=True,
        description=(
            "Whether to generate and use MSAs for protein chains "
            "using ColabFold search"
        ),
    )

    colabfold_search_config: Optional[ColabfoldSearchConfig] = ConfigField(
        title="ColabFold Search Config",
        default=None,
        description=(
            "Nested configuration for ColabFold MSA search. "
            "If None, uses default settings."
        ),
        hidden=True,
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

    @model_validator(mode="after")
    def sync_nested_config(self):
        """Sync verbose flag with nested colabfold_search_config."""
        if self.colabfold_search_config is None:
            self.colabfold_search_config = ColabfoldSearchConfig()
        self.colabfold_search_config.verbose = self.verbose
        return self


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return ProtenixInput(complexes=["MKTL"])


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
)
@tool_cache_iterable(
    input_iterable_field="complexes",
    output_iterable_field="structures",
    tool_name="protenix-prediction",
)
def run_protenix(
    inputs: ProtenixInput, config: ProtenixConfig | None = None,
    instance=None,
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

    Returns:
        ProtenixOutput: Structured output containing:
            - ``structures``: List of ``Structure`` instances, one per input complex
            - Each structure includes coordinates and confidence metrics:

                confidence_score (float): Ranking score used to select the best sample.
                    Weighted combination of confidence metrics. Range: 0.0-1.0.

                ptm (float): Predicted Template Modeling score measuring overall structural
                    accuracy. Range: 0.0-1.0.

                iptm (float): Interface PTM score measuring confidence in inter-chain
                    interfaces. Range: 0.0-1.0.

                avg_plddt (float): Average per-residue confidence (pLDDT), normalized to
                    0.0-1.0. Higher values indicate more confident predictions.

                gpde (float): Global Predicted Distance Error in Angstroms. Lower values
                    indicate more confident relative positioning.

                chain_ptm (List[float]): Per-chain PTM scores.

                chain_plddt (List[float]): Per-chain pLDDT scores.

                chain_pair_iptm (List[List[float]]): Pairwise interface PTM scores
                    between all chain pairs.

                has_clash (bool): Whether the predicted structure has steric clashes.

    See Also:
        - Protenix GitHub: https://github.com/bytedance/Protenix
        - Protenix paper: https://www.biorxiv.org/content/10.1101/2025.01.08.631857

    Example:
        >>> inputs = ProtenixInput(
        ...     complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]]
        ... )
        >>> config = ProtenixConfig(
        ...     model_name="protenix_base_default_v1.0.0",
        ...     num_diffusion_samples=5,
        ...     verbose=True,
        ... )
        >>> result = run_protenix(inputs, config)
        >>> print(f"Confidence: {result.structures[0].metrics['confidence_score']:.2f}")
    """
    from bio_programming_tools.utils.tool_instance import ToolInstance

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "protenix_output")
        os.makedirs(output_dir)

        # Build batched input JSON for all complexes
        batch_json = []
        for i in range(len(inputs.complexes)):
            batch_json.extend(inputs.to_json(i, name=f"protenix_job_{i}"))

        # Generate MSAs and inject paths into the batch JSON if needed
        if config.use_msa:
            _inject_msa_paths(batch_json, inputs, config, temp_dir)

        # Write the batch input JSON
        input_json_path = os.path.join(temp_dir, "protenix_input.json")
        with open(input_json_path, "w") as f:
            json.dump(batch_json, f)

        # Prepare input data for inference script
        seeds_str = ",".join(str(s) for s in config.seeds)
        input_data = {
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
            verbose=config.verbose,
            timeout=config.timeout,
            reload_on=type(config).reload_fields(),
        )

    # Parse results for each complex
    results = []
    for i, job_result in enumerate(output_data):
        raw_metrics = job_result.get("metrics", {})

        metrics = {
            "confidence_score": float(raw_metrics.get("ranking_score", 0.0)),
            "ptm": float(raw_metrics.get("ptm", 0.0)),
            "iptm": float(raw_metrics.get("iptm", 0.0)),
            "avg_plddt": float(raw_metrics.get("plddt", 0.0)) / 100.0,
            "gpde": float(raw_metrics.get("gpde", 0.0)),
            "chain_ptm": raw_metrics.get("chain_ptm"),
            "chain_plddt": raw_metrics.get("chain_plddt"),
            "chain_pair_iptm": raw_metrics.get("chain_pair_iptm"),
            "has_clash": raw_metrics.get("has_clash"),
        }

        results.append(
            Structure(
                structure_filepath_or_content=job_result["structure_cif_output"],
                b_factor_type=BFactorType.PLDDT,
                metrics=metrics,
                source="protenix-prediction",
            )
        )

    return ProtenixOutput(structures=results)


# ============================================================================
# Helper Functions
# ============================================================================
def _extract_protein_sequences_and_chain_ids(
    sp_complex,
) -> tuple[list[str], list[str]]:
    """Extract protein sequences and their chain IDs from a complex.

    Args:
        sp_complex: StructurePredictionComplex instance

    Returns:
        Tuple of (protein_seqs, protein_chain_ids)
    """
    all_chain_ids = list(string.ascii_uppercase)
    protein_seqs = []
    protein_chain_ids = []
    for i, chain in enumerate(sp_complex.chains):
        if chain.entity_type == "protein":
            protein_seqs.append(chain.sequence)
            protein_chain_ids.append(all_chain_ids[i])
    return protein_seqs, protein_chain_ids


def _inject_msa_paths(
    batch_json: list[dict],
    inputs: ProtenixInput,
    config: ProtenixConfig,
    temp_dir: str,
) -> None:
    """Generate MSAs via ColabFold and inject file paths into the batch JSON.

    For each complex, extracts protein chains, runs ColabFold search, writes
    A3M files, and sets ``unpairedMsaPath`` on the corresponding proteinChain
    entries in the batch JSON.

    Args:
        batch_json: List of Protenix job dicts (modified in place)
        inputs: ProtenixInput with the original complexes
        config: ProtenixConfig with MSA settings
        temp_dir: Directory for writing A3M files
    """
    msa_dir = os.path.join(temp_dir, "msas")
    os.makedirs(msa_dir, exist_ok=True)

    for job_idx, job in enumerate(batch_json):
        sp_complex = inputs.complexes[job_idx]
        protein_seqs, protein_chain_ids = _extract_protein_sequences_and_chain_ids(
            sp_complex
        )

        if not protein_seqs:
            continue

        if config.verbose:
            logger.info(
                f"Generating MSAs for {len(protein_seqs)} protein chain(s) "
                f"in complex {job_idx} using ColabFold search..."
            )

        # Run ColabFold search for all protein chains in this complex
        queries = list(zip(protein_seqs, protein_chain_ids))
        colabfold_input = ColabfoldSearchInput(queries=queries)

        try:
            colabfold_output = run_colabfold_search(
                colabfold_input, config.colabfold_search_config
            )
        except Exception as e:
            raise RuntimeError(f"ColabFold MSA search failed: {e}") from e

        # Write A3M files and build a mapping of chain_id -> a3m_path
        msa_paths = {}
        for result in colabfold_output.results:
            if result.msa is not None:
                a3m_path = os.path.join(
                    msa_dir, f"job_{job_idx}_{result.sequence_id}.a3m"
                )
                result.msa.to_a3m_file(a3m_path, query_index=0)
                msa_paths[result.sequence_id] = a3m_path

                if config.verbose:
                    logger.info(
                        f"Generated MSA for chain {result.sequence_id}: "
                        f"{result.num_homologs_found} homologs found"
                    )
            else:
                if config.verbose:
                    logger.warning(
                        f"No homologs found for chain {result.sequence_id}"
                    )

        # Inject MSA paths into the job's proteinChain entries
        for seq_entry in job["sequences"]:
            if "proteinChain" in seq_entry:
                # Find the chain ID for this protein entry by matching sequence
                chain_seq = seq_entry["proteinChain"]["sequence"]
                for chain_id, protein_seq in zip(protein_chain_ids, protein_seqs):
                    if protein_seq == chain_seq and chain_id in msa_paths:
                        seq_entry["proteinChain"]["unpairedMsaPath"] = msa_paths[
                            chain_id
                        ]
                        break

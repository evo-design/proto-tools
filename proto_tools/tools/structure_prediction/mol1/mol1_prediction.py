"""All-atom biomolecular structure prediction with Mol-1."""

import logging
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Literal

from proto_tools.entities.complex import Complex
from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.entities.structures.msa_conditioning import MSASubsampling
from proto_tools.tools.structure_prediction.mol1.helpers import complex_to_mol1_json
from proto_tools.tools.structure_prediction.msa_plan import (
    SentMSA,
    af3_style_pairing,
    protein_chain_msas,
    short_chain_reason,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    ComplexMSAs,
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
    normalize_output_chain_ids,
    write_paired_a3m_with_uniprot_headers,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)

# MSA rows Mol-1 keeps (data.msa max_size / sample_cutoff, test); its MSA module samples rows
# from them at random, as Protenix's does.
MOL1_MSA_MAX_ROWS = 16384


class Mol1Input(StructurePredictionInput):
    """Complexes and optional protein alignments for Mol-1 prediction.

    Attributes:
        complexes (list[Complex]): Protein, DNA, RNA and ligand assemblies to fold.
        msas (list[ComplexMSAs] | None): Optional alignments, one bundle per complex.
    """

    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class Mol1Config(MSAStructurePredictionConfig):
    """Sampling and execution settings for Mol-1.

    Attributes:
        model_checkpoint (str): Bundled ``mol-1-base`` or a custom checkpoint path.
        num_samples (int): Diffusion samples per complex; only the best is returned.
        num_steps (int): Diffusion denoising steps per sample.
        num_cycles (int): Recycling passes through the model trunk.
        dtype (Literal["bf16", "fp32", "fp16"]): Floating-point precision used during inference.
        chunk_size (int | None): Inference chunk size; smaller values reduce memory.
        sample_diffusion_chunk_size (int | None): Samples denoised together.
        include_pae_matrix (bool): Attach token-level PAE in angstroms.
        use_msa (bool): Generate protein MSAs; supplied MSAs are always used.
        msa_search_config (Mmseqs2HomologySearchConfig | None): Optional MMseqs2 search settings.
        pair_heterocomplex_msas (bool): Generate taxonomy-paired MSAs for protein heterocomplexes.
    """

    model_checkpoint: str = ConfigField(
        default="mol-1-base",
        title="Model Checkpoint",
        description="Bundled 'mol-1-base' or a path to a custom Mol-1 .pt checkpoint.",
        reload_on_change=True,
    )
    num_samples: int = ConfigField(
        default=5,
        ge=1,
        title="Number of Samples",
        description="Diffusion samples per complex; the highest-ranked structure is returned.",
    )
    num_steps: int = ConfigField(
        default=200,
        ge=1,
        title="Denoising Steps",
        description="Diffusion denoising steps per sample; more steps increase runtime.",
    )
    num_cycles: int = ConfigField(
        default=20,
        ge=1,
        title="Recycling Cycles",
        description="Recycling passes through the trunk; more cycles increase runtime.",
    )
    dtype: Literal["bf16", "fp32", "fp16"] = ConfigField(
        default="bf16",
        title="Precision",
        description="Inference precision; mixed precision is used on CUDA.",
    )
    chunk_size: int | None = ConfigField(
        default=1024,
        ge=1,
        title="Chunk Size",
        description="Inference chunk size; smaller chunks reduce peak memory. None disables chunking.",
    )
    sample_diffusion_chunk_size: int | None = ConfigField(
        default=1,
        ge=1,
        title="Diffusion Sample Chunk Size",
        description="Samples denoised together; larger chunks use more memory. None processes all together.",
    )
    include_pae_matrix: bool = ConfigField(
        default=False,
        title="Include PAE Matrix",
        description="Attach the full token-pair PAE matrix in angstroms.",
    )

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """Custom checkpoint files must be available on the execution host."""
        if self.model_checkpoint != "mol-1-base":
            return f"Custom model_checkpoint paths are local files unavailable on device={device!r}. Use 'mol-1-base' remotely or run locally."
        return None


class Mol1Metrics(Metrics):
    """Confidence metrics for the selected Mol-1 structure.

    Metrics documented in ``metric_spec``:
        ranking_score (float): Upstream sample ranking score, including clash penalties.
        avg_plddt (float): Mean atom pLDDT on the 0-100 scale.
        ptm (float): Predicted TM-score on the 0-1 scale.
        iptm (float): Interface pTM on the 0-1 scale; zero for single chains.
        gpde (float): Global predicted distance error in angstroms.
        has_clash (bool): Whether a severe inter-polymer clash was detected.
        chain_plddt (list[float]): Per-chain pLDDT on the 0-1 scale.
        chain_pair_iptm (list[list[float]]): Pairwise interface pTM, with zero diagonal.
        pae (list[list[float]]): Token-level PAE when include_pae_matrix=True.
        avg_pae (float): Mean token-level PAE when include_pae_matrix=True.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "ranking_score": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "better_values_are": "higher",
        },
        "avg_plddt": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "better_values_are": "higher",
        },
        "ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "iptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "gpde": {"availability": "always", "type": "float", "min": 0.0, "max": None, "better_values_are": "lower"},
        "has_clash": {"availability": "always", "type": "bool", "min": None, "max": None, "better_values_are": "lower"},
        "chain_plddt": {
            "availability": "always",
            "type": "list[float]",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "chain_pair_iptm": {
            "availability": "always",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "pae": {
            "availability": "when include_pae_matrix=True",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": None,
            "better_values_are": "lower",
        },
        "avg_pae": {
            "availability": "when include_pae_matrix=True",
            "type": "float",
            "min": 0.0,
            "max": None,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = "ranking_score"


class Mol1Output(StructurePredictionOutput):
    """Best-ranked Mol-1 prediction for each input complex.

    Attributes:
        structures (list[Structure]): Predicted structures carrying Mol1Metrics.
    """


def example_input() -> Mol1Input:
    """Return a short protein for smoke tests."""
    return Mol1Input(complexes=["NLYIQWLKDGGPSSGRPPPS"])  # type: ignore[list-item]


@tool(
    key="mol1-prediction",
    label="Mol-1 Structure Prediction",
    category="structure_prediction",
    input_class=Mol1Input,
    config_class=Mol1Config,
    output_class=Mol1Output,
    metrics_class=Mol1Metrics,
    description="All-atom structure prediction for protein, nucleic-acid and ligand complexes using Mol-1",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["complexes", "msas"],
    iterable_output_field="structures",
    max_chunk_size=1,
    cacheable=True,
    stochastic=True,
)
def run_mol1_prediction(inputs: Mol1Input, config: Mol1Config, instance: Any = None) -> Mol1Output:
    """Predict one all-atom structure per complex with Mol-1.

    Samples ``num_samples`` conformations and returns the highest ranking score.
    Supplied MSAs take precedence over automatic MMseqs2 search. Templates and
    RNA MSA search are disabled; RNA chains use their single sequences.

    Args:
        inputs (Mol1Input): Complexes and optional protein MSAs.
        config (Mol1Config): Sampling, precision and execution settings.
        instance (Any): Optional persistent tool worker.

    Returns:
        Mol1Output: One structure per input, with confidence metrics.

    Example:
        >>> inputs = Mol1Input(complexes=["NLYIQWLKDGGPSSGRPPPS"])
        >>> result = run_mol1_prediction(inputs, Mol1Config(use_msa=False, seed=101))
        >>> result.structures[0].metrics["avg_plddt"]
    """
    logger.debug("Using local venv for Mol-1 prediction")
    structures = []
    for index, comp in enumerate(inputs.complexes):
        with tempfile.TemporaryDirectory() as scratch:
            job = complex_to_mol1_json(comp.chains, "mol1_job")
            complex_msas = inputs.msas[index] if inputs.msas is not None else None
            msa_sent = _write_msas(job, comp, complex_msas, Path(scratch))
            output = ToolInstance.dispatch(
                "mol1",
                {
                    "operation": "predict",
                    "job": job,
                    "seed": config.seed,
                    "item_index": index,
                    "device": config.device,
                    "verbose": config.verbose,
                    **config.model_dump(
                        include={
                            "model_checkpoint",
                            "num_samples",
                            "num_steps",
                            "num_cycles",
                            "dtype",
                            "chunk_size",
                            "sample_diffusion_chunk_size",
                            "include_pae_matrix",
                        }
                    ),
                    "use_msa": inputs.msas is not None or config.use_msa,
                },
                instance=instance,
                config=config,
            )
        structure = Structure(
            structure=output["structure_cif_output"],
            b_factor_type=BFactorType.PLDDT,
            metrics=Mol1Metrics(**output["metrics"]),
            source="mol1-prediction",
        )
        structures.append(
            normalize_output_chain_ids(
                structure,
                comp.chains,
                complex_msas=complex_msas,
                report_msa=True,
                msa_sent=msa_sent,
                msa_subsampling=MSASubsampling(mode="stochastic", max_rows=MOL1_MSA_MAX_ROWS),
            )
        )
    return Mol1Output(structures=structures)


def _write_msas(
    job: dict[str, Any], comp: Complex, complex_msas: ComplexMSAs | None, scratch: Path
) -> dict[int, SentMSA]:
    """Write each protein chain's alignments into the native job; return the rows sent per chain.

    Mol-1 reads MSAs the way Protenix does. ``unpairedMsaPath`` gets the chain's deep unpaired
    MSA when the bundle has one, else its primary MSA. A paired bundle also writes
    ``pairedMsaPath``, which Mol-1 merges into the unpaired rows (``msa_pair_as_unpair``) and pairs
    across chains only for two or more distinct protein sequences. Chains of four or fewer residues
    get no MSA. The worker calls ``infer_predict`` directly, which runs no MSA search, so a chain
    left without a path is predicted from its query alone.

    Args:
        job (dict[str, Any]): The native job from :func:`complex_to_mol1_json`, edited in place;
            its ``sequences`` follow ``comp.chains`` one to one.
        comp (Complex): The complex being predicted.
        complex_msas (ComplexMSAs | None): The complex's MSA bundle, if any.
        scratch (Path): Directory for the written A3M files.

    Returns:
        dict[int, SentMSA]: Rows sent per protein chain index, for the MSA conditioning report.
    """
    sent: dict[int, SentMSA] = {}
    if complex_msas is None:
        return sent
    chains = protein_chain_msas(comp.chains, complex_msas)
    is_paired = complex_msas.paired
    pairs = af3_style_pairing(
        (chain.chain.sequence for chain in chains), sum(chain.primary is not None for chain in chains), is_paired
    )
    for chain in chains:
        entry = job["sequences"][chain.index]["proteinChain"]
        short = short_chain_reason(chain.chain, "Mol-1")
        unpaired_msa = chain.unpaired or chain.primary
        if unpaired_msa is None:
            continue
        msa_path = scratch / f"chain_{chain.index}.a3m"
        unpaired_msa.to_a3m_file(str(msa_path), query_index=0)
        entry["unpairedMsaPath"] = str(msa_path)
        if chain.primary is None:
            # Unpaired-only chain: depth for this chain, never a pairing partner.
            sent[chain.index] = SentMSA(rows=len(unpaired_msa), ignored_reason=short)
            continue
        if is_paired:
            paired_path = scratch / f"chain_{chain.index}.paired.a3m"
            write_paired_a3m_with_uniprot_headers(chain.primary, str(paired_path))
            entry["pairedMsaPath"] = str(paired_path)
        sent[chain.index] = SentMSA(
            rows=len(chain.primary) if is_paired else len(unpaired_msa),
            # Without a deep unpaired MSA the unpaired file repeats the paired rows: no extra depth.
            unpaired_rows=len(unpaired_msa) - 1 if is_paired and chain.unpaired is not None else 0,
            paired=pairs,
            ignored_reason=short,
        )
    return sent

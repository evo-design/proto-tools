"""IPSAE interface quality scoring for protein complex predictions."""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proto_tools.entities.structures import Structure
from proto_tools.entities.structures.structure import BFactorType
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField, InputField, ToolInstance
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class ChainPairScores(BaseModel):
    """Per-chain-pair scoring result from IPSAE.

    Attributes:
        chain1 (str): First chain ID.
        chain2 (str): Second chain ID.
        pair_type (str): Pair type: ``"asym"`` (directional) or ``"max"`` (symmetric maximum).
        ipsae (float): ipSAE score (primary metric, adaptive d0).
        ipsae_d0chn (float): ipSAE with d0 based on chain lengths.
        ipsae_d0dom (float): ipSAE with d0 based on domain size.
        iptm_af (float): AlphaFold-reported interface TM. ``-1.0`` if unavailable.
        iptm_d0chn (float): ipTM computed from PAE with chain-length d0.
        pdockq (float): pDockQ score (Bryant et al. 2022).
        pdockq2 (float): pDockQ2 score (Zhu et al. 2023).
        lis (float): Local Interaction Score (Kim et al. 2024).
    """

    model_config = ConfigDict(extra="forbid")

    chain1: str = Field(description="First chain ID")
    chain2: str = Field(description="Second chain ID")
    pair_type: str = Field(description="'asym' (directional) or 'max' (symmetric maximum)")
    ipsae: float = Field(description="ipSAE score (adaptive d0)")
    ipsae_d0chn: float = Field(description="ipSAE with chain-length d0")
    ipsae_d0dom: float = Field(description="ipSAE with domain-size d0")
    iptm_af: float = Field(description="AlphaFold-reported ipTM (-1.0 if unavailable)")
    iptm_d0chn: float = Field(description="ipTM from PAE with chain-length d0")
    pdockq: float = Field(description="pDockQ (Bryant 2022)")
    pdockq2: float = Field(description="pDockQ2 (Zhu 2023)")
    lis: float = Field(description="Local Interaction Score (Kim 2024)")


class IPSAEMetrics(Metrics):
    """IPSAE interface-quality metrics for a cofolded complex.

    Metrics documented in ``metric_spec``:
        ipsae (float): ipSAE score for the binder-target interface (max of both directions).
        pdockq2 (float): pDockQ2 for the binder-target interface.
        lis (float): LIS for the binder-target interface.
        pdockq (float): pDockQ for the binder-target interface.
        iptm_d0chn (float): ipTM from PAE for the binder-target interface.

    Attributes:
        chain_pair_results (list[ChainPairScores]): Full per-chain-pair breakdown.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "ipsae": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
        },
        "pdockq2": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.5,
        },
        "lis": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
        },
        "pdockq": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
        },
        "iptm_d0chn": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
        },
    }
    primary_metric: str | None = "ipsae"

    chain_pair_results: list[ChainPairScores] = Field(
        default_factory=list,
        description="Full per-chain-pair breakdown from IPSAE",
    )


class IPSAEScoringInput(BaseToolInput):
    """Input for IPSAE interface scoring.

    Attributes:
        structure (Structure): Cofolded complex with per-residue pLDDT in the
            B-factor column and the PAE matrix attached at
            ``structure.metrics['pae_matrix']`` as a square ``list[list[float]]``.
        binder_chain (str): Single-character chain ID of the binder.
        target_chains (list[str]): Target chain ID(s).
    """

    structure: Structure = InputField(description="Cofolded complex with pLDDT B-factors and PAE matrix in metrics")
    binder_chain: str = InputField(description="Single-character chain ID of the binder")
    target_chains: list[str] = InputField(description="Target chain ID(s)")

    @field_validator("target_chains", mode="before")
    @classmethod
    def _normalize_target_chains(cls, value: Any) -> list[str]:
        """Accept comma-separated strings or explicit lists."""
        raw_chain_ids = [value] if isinstance(value, str) else value
        if not isinstance(raw_chain_ids, (list, tuple)) or not all(isinstance(c, str) for c in raw_chain_ids):
            raise ValueError("target_chains must be a string or list of strings")
        target_chains = [chain.strip() for raw in raw_chain_ids for chain in raw.split(",") if chain.strip()]
        if not target_chains:
            raise ValueError("target_chains must contain at least one chain ID")
        return target_chains

    @model_validator(mode="after")
    def _validate(self) -> "IPSAEScoringInput":
        """Validate chain IDs and required confidence signals."""
        if self.binder_chain in self.target_chains:
            raise ValueError(f"binder_chain {self.binder_chain!r} must not appear in target_chains")
        available = set(self.structure.get_chain_ids())
        missing = {self.binder_chain, *self.target_chains} - available
        if missing:
            raise ValueError(f"Chain ID(s) {sorted(missing)} not found in structure. Available: {sorted(available)}")
        pae = self.structure.metrics.get("pae_matrix")
        if pae is None:
            raise ValueError("structure.metrics['pae_matrix'] is missing; attach the PAE matrix before scoring")
        pae_arr = np.asarray(pae)
        if pae_arr.ndim != 2 or pae_arr.shape[0] != pae_arr.shape[1]:
            raise ValueError(f"pae_matrix must be a square 2D matrix, got shape {pae_arr.shape}")
        return self


class IPSAEScoringConfig(BaseConfig):
    """Configuration for IPSAE scoring.

    Attributes:
        pae_cutoff (float): PAE cutoff in Å for interface residue detection.
        distance_cutoff (float): CA-CA distance cutoff in Å for contact detection.
    """

    pae_cutoff: float = ConfigField(
        default=10.0,
        gt=0.0,
        title="PAE Cutoff (Å)",
        description="PAE threshold for interface residue detection.",
    )
    distance_cutoff: float = ConfigField(
        default=10.0,
        gt=0.0,
        title="Distance Cutoff (Å)",
        description="CA-CA distance cutoff for contact detection.",
    )


class IPSAEScoringOutput(BaseToolOutput):
    """Output from IPSAE scoring.

    Attributes:
        metrics (IPSAEMetrics): Scalar metrics plus per-chain-pair breakdown.
    """

    metrics: IPSAEMetrics = Field(description="IPSAE metrics for the input complex")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        path = Path(export_path).with_suffix(f".{file_format}")
        path.write_text(json.dumps(self.metrics.model_dump(), indent=2))


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    fixture = Path(__file__).parent.parent / "pdockq2" / "example_input_fixture.pdb"
    pae_json = Path(__file__).parent.parent / "pdockq2" / "example_input_fixture_pae.json"
    pae = json.loads(pae_json.read_text())
    structure = Structure.from_file(fixture, b_factor_type=BFactorType.PLDDT, metrics={"pae_matrix": pae})
    return IPSAEScoringInput(structure=structure, binder_chain="A", target_chains=["B"])


@tool(
    key="ipsae-scoring",
    label="IPSAE Interface Scoring",
    category="structure_scoring",
    input_class=IPSAEScoringInput,
    config_class=IPSAEScoringConfig,
    output_class=IPSAEScoringOutput,
    description="Score a cofolded protein complex with IPSAE (Dunbrack 2025), computing ipSAE, pDockQ2, LIS, pDockQ, and ipTM",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_ipsae_scoring(
    inputs: IPSAEScoringInput,
    config: IPSAEScoringConfig,
    instance: ToolInstance | None = None,
) -> IPSAEScoringOutput:
    """Compute IPSAE interface metrics for a cofolded protein complex.

    Dispatches to the vendored DunbrackLab/IPSAE script via ToolInstance,
    writing temporary PDB + PAE JSON files and parsing the output.

    Args:
        inputs (IPSAEScoringInput): Cofolded complex plus binder and target chain IDs.
        config (IPSAEScoringConfig): Scoring configuration (PAE and distance cutoffs).
        instance (ToolInstance | None): Tool instance for standalone dispatch.

    Returns:
        IPSAEScoringOutput: ipSAE, pDockQ2, LIS, pDockQ, and ipTM metrics.
    """
    structure = inputs.structure
    binder_chain = inputs.binder_chain
    target_chains = inputs.target_chains

    pae_matrix = structure.metrics.get("pae_matrix")
    if pae_matrix is None:
        raise ValueError("structure.metrics['pae_matrix'] is missing")

    pae_list = pae_matrix if isinstance(pae_matrix, list) else [list(row) for row in pae_matrix]

    plddt = structure.per_residue_plddt
    plddt_list = [float(v) * 100.0 for v in plddt] if plddt is not None else None

    logger.debug("Running IPSAE scoring (pae_cutoff=%s, dist_cutoff=%s)", config.pae_cutoff, config.distance_cutoff)

    result = ToolInstance.dispatch(
        "ipsae",
        {
            "operation": "score",
            "pdb_content": structure.structure_pdb,
            "pae_matrix": pae_list,
            "plddt": plddt_list,
            "pae_cutoff": config.pae_cutoff,
            "dist_cutoff": config.distance_cutoff,
        },
        instance=instance,
        config=config,
    )

    chain_pair_results_raw = result["chain_pair_results"]
    chain_pair_results = _parse_chain_pair_results(chain_pair_results_raw)
    binder_target_max = _extract_binder_target_scores(chain_pair_results, binder_chain, target_chains)

    metrics = IPSAEMetrics(
        ipsae=binder_target_max.get("ipsae", 0.0),
        pdockq2=binder_target_max.get("pdockq2", 0.0),
        lis=binder_target_max.get("lis", 0.0),
        pdockq=binder_target_max.get("pdockq", 0.0),
        iptm_d0chn=binder_target_max.get("iptm_d0chn", 0.0),
        chain_pair_results=chain_pair_results,
    )

    return IPSAEScoringOutput(
        metadata={
            "binder_chain": binder_chain,
            "target_chains": target_chains,
            "pae_cutoff": config.pae_cutoff,
            "distance_cutoff": config.distance_cutoff,
        },
        metrics=metrics,
    )


# ============================================================================
# Helpers
# ============================================================================
def _parse_chain_pair_results(raw_results: list[dict[str, Any]]) -> list[ChainPairScores]:
    """Convert raw IPSAE output dicts to typed ChainPairScores models."""
    return [
        ChainPairScores(
            chain1=str(row.get("Chn1", "")),
            chain2=str(row.get("Chn2", "")),
            pair_type=str(row.get("Type", "asym")),
            ipsae=float(row.get("ipSAE", 0.0)),
            ipsae_d0chn=float(row.get("ipSAE_d0chn", 0.0)),
            ipsae_d0dom=float(row.get("ipSAE_d0dom", 0.0)),
            iptm_af=float(row.get("ipTM_af", -1.0)),
            iptm_d0chn=float(row.get("ipTM_d0chn", 0.0)),
            pdockq=float(row.get("pDockQ", 0.0)),
            pdockq2=float(row.get("pDockQ2", 0.0)),
            lis=float(row.get("LIS", 0.0)),
        )
        for row in raw_results
    ]


def _extract_binder_target_scores(
    chain_pair_results: list[ChainPairScores],
    binder_chain: str,
    target_chains: list[str],
) -> dict[str, float]:
    """Extract the max-type scores for the binder-target interface."""
    target_set = set(target_chains)
    for result in chain_pair_results:
        if result.pair_type != "max":
            continue
        if (result.chain1 == binder_chain and result.chain2 in target_set) or (
            result.chain2 == binder_chain and result.chain1 in target_set
        ):
            return {
                "ipsae": result.ipsae,
                "pdockq2": result.pdockq2,
                "lis": result.lis,
                "pdockq": result.pdockq,
                "iptm_d0chn": result.iptm_d0chn,
            }

    logger.warning("No max-type chain pair found for binder=%r target=%s", binder_chain, sorted(target_set))
    return {}

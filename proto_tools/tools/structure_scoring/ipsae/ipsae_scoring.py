"""IPSAE interface quality scoring for protein complex predictions."""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from proto_tools.entities.structures import ChainSelection, SingleChainSelection, Structure
from proto_tools.entities.structures.structure import BFactorType
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
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

    chain1: str = Field(title="Chain 1", description="First chain ID")
    chain2: str = Field(title="Chain 2", description="Second chain ID")
    pair_type: str = Field(title="Pair Type", description="'asym' (directional) or 'max' (symmetric maximum)")
    ipsae: float = Field(
        title="ipSAE", description="ipSAE interface score with adaptive d0; 0-1, higher indicates a better interface"
    )
    ipsae_d0chn: float = Field(
        title="ipSAE d0chn", description="ipSAE with chain-length d0; 0-1, higher indicates a better interface"
    )
    ipsae_d0dom: float = Field(
        title="ipSAE d0dom", description="ipSAE with domain-size d0; 0-1, higher indicates a better interface"
    )
    iptm_af: float = Field(
        title="ipTM (AlphaFold)",
        description="AlphaFold-reported interface pTM; 0-1, higher is better. Returns -1.0 if unavailable.",
    )
    iptm_d0chn: float = Field(
        title="ipTM d0chn",
        description="Interface pTM recomputed from PAE with chain-length d0; 0-1, higher is better",
    )
    pdockq: float = Field(
        title="pDockQ", description="pDockQ interface score (Bryant 2022); 0-1, higher indicates a better interface"
    )
    pdockq2: float = Field(
        title="pDockQ2", description="pDockQ2 interface score (Zhu 2023); 0-1, higher indicates a better interface"
    )
    lis: float = Field(
        title="LIS", description="Local Interaction Score (Kim 2024); higher values indicate more interface contact"
    )


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
            "better_values_are": "higher",
        },
        "pdockq2": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "lis": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "pdockq": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "iptm_d0chn": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
    }
    primary_metric: str | None = Field(
        default="ipsae",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )

    chain_pair_results: list[ChainPairScores] = Field(
        default_factory=list,
        title="Chain-Pair Scores",
        description="Full per-chain-pair breakdown from IPSAE",
    )


class IPSAEScoringInput(BaseToolInput):
    """Input for IPSAE interface scoring.

    Attributes:
        structure (Structure): Cofolded complex with per-residue pLDDT in the
            B-factor column and the PAE matrix attached at
            ``structure.metrics['pae']`` as a square ``list[list[float]]``.
        binder_chain (SingleChainSelection): Single-character chain ID of the binder.
        target_chains (ChainSelection): Target chain ID(s).
    """

    structure: Structure = InputField(
        title="Input Structure",
        description="Cofolded complex with per-residue pLDDT in B-factors and a PAE matrix attached to its metrics",
    )
    binder_chain: SingleChainSelection = InputField(
        title="Binder Chain",
        description="Single-character chain ID of the binder",
    )
    target_chains: ChainSelection = InputField(
        title="Target Chains",
        description="Target chain ID(s)",
    )

    @model_validator(mode="after")
    def _validate(self) -> "IPSAEScoringInput":
        """Validate chain IDs and required confidence signals."""
        binder = self.binder_chain.chain
        targets = list(self.target_chains.chains)
        if binder in targets:
            raise ValueError(f"binder_chain {binder!r} must not appear in target_chains")
        available = set(self.structure.get_chain_ids())
        missing = {binder, *targets} - available
        if missing:
            raise ValueError(f"Chain ID(s) {sorted(missing)} not found in structure. Available: {sorted(available)}")
        pae = self.structure.metrics.get("pae")
        if pae is None:
            raise ValueError("structure.metrics['pae'] is missing; attach the PAE matrix before scoring")
        pae_arr = np.asarray(pae)
        if pae_arr.ndim != 2 or pae_arr.shape[0] != pae_arr.shape[1]:
            raise ValueError(f"pae must be a square 2D matrix, got shape {pae_arr.shape}")
        if self.structure.per_residue_plddt is None:
            raise ValueError(
                "structure.per_residue_plddt is None; IPSAE requires per-residue pLDDT in the B-factor column. "
                "Set Structure.b_factor_type to PLDDT (raw 0 to 100) or NORMALIZED_PLDDT (0 to 1) before scoring "
                f"(current b_factor_type: {self.structure.b_factor_type.value})."
            )
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

    metrics: IPSAEMetrics = Field(
        title="ipSAE Metrics",
        description="IPSAE metrics for the input complex",
    )

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
    """Minimal valid input for testing and examples.

    Bundled fixture is insulin chains A (21 residues) and B (30 residues)
    extracted from PDB 4INS. The PAE matrix is synthetic: 0 on the diagonal,
    3.0 within each chain, 6.0 across the A/B interface (under the default
    10 Å cutoff so the interface yields non-zero metrics).
    """
    fixture = Path(__file__).parent / "example_input_fixture.pdb"
    n_a, n_b = 21, 30
    n = n_a + n_b
    pae = [[0.0 if i == j else (3.0 if (i < n_a) == (j < n_a) else 6.0) for j in range(n)] for i in range(n)]
    structure = Structure.from_file(fixture, b_factor_type=BFactorType.PLDDT, metrics={"pae": pae})
    return IPSAEScoringInput(
        structure=structure,
        binder_chain=SingleChainSelection(chain="A"),
        target_chains=ChainSelection(chains=["B"]),
    )


@tool(
    key="ipsae-scoring",
    label="IPSAE Interface Scoring",
    category="structure_scoring",
    input_class=IPSAEScoringInput,
    config_class=IPSAEScoringConfig,
    output_class=IPSAEScoringOutput,
    metrics_class=IPSAEMetrics,
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
    binder_chain = inputs.binder_chain.chain
    target_chains = list(inputs.target_chains.chains)

    pae = structure.metrics.get("pae")
    if pae is None:
        raise ValueError("structure.metrics['pae'] is missing")

    pae_list = pae if isinstance(pae, list) else [list(row) for row in pae]

    plddt = structure.per_residue_plddt
    plddt_list = [float(v) * 100.0 for v in plddt] if plddt is not None else None

    logger.debug("Running IPSAE scoring (pae_cutoff=%s, dist_cutoff=%s)", config.pae_cutoff, config.distance_cutoff)

    result = ToolInstance.dispatch(
        "ipsae",
        {
            "operation": "score",
            "pdb_content": structure.structure_pdb,
            "pae": pae_list,
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
    if not binder_target_max:
        available = sorted({(r.chain1, r.chain2) for r in chain_pair_results if r.pair_type == "max"})
        raise ValueError(
            f"ipsae: no interface scores for binder_chain={binder_chain!r} vs "
            f"target_chains={target_chains!r}; available max-type chain pairs: {available}. "
            f"Check that the chain labels match the structure."
        )

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
    """Extract max-type scores for the binder-target interface.

    With multiple target chains, returns the highest-ipSAE pair (not an arbitrary first match).
    """
    target_set = set(target_chains)
    matches = [
        result
        for result in chain_pair_results
        if result.pair_type == "max"
        and (
            (result.chain1 == binder_chain and result.chain2 in target_set)
            or (result.chain2 == binder_chain and result.chain1 in target_set)
        )
    ]
    if not matches:
        logger.warning("No max-type chain pair found for binder=%r target=%s", binder_chain, sorted(target_set))
        return {}

    best = max(matches, key=lambda result: result.ipsae)
    return {
        "ipsae": best.ipsae,
        "pdockq2": best.pdockq2,
        "lis": best.lis,
        "pdockq": best.pdockq,
        "iptm_d0chn": best.iptm_d0chn,
    }

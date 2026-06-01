"""pDockQ2 interface quality scoring for protein complex predictions."""

import json
import logging
import math
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
class InterfacePDockQ2(BaseModel):
    """Per-chain pDockQ2 result.

    Each row summarizes one chain's interface with its cross-chain neighbors:
    the chain IDs it contacts, its interface pLDDT, its normalized interface
    PAE, and the resulting pmiDockQ value.

    Attributes:
        chain_id (str): Chain whose interface these metrics describe.
        neighbor_chains (str): Sorted, concatenated IDs of chains this one
            contacts (e.g. ``"BC"`` if ``chain_id`` contacts B and C).
        if_plddt (float): Mean pLDDT (0-100 scale) over interface residues of
            ``chain_id``.
        norm_pae (float): Mean of ``1 / (1 + (PAE / 10)^2)`` over contact
            residue pairs for this chain.
        pmidockq (float): Per-chain pDockQ2 score in ``[0, 1]``.
    """

    model_config = ConfigDict(extra="forbid")

    chain_id: str = Field(title="Chain ID", description="Chain whose interface is summarized")
    neighbor_chains: str = Field(
        title="Neighbor Chains", description="Concatenated chain IDs that contact this chain at the interface"
    )
    if_plddt: float = Field(title="Interface pLDDT", description="Mean interface pLDDT on the 0-100 scale")
    norm_pae: float = Field(
        title="Normalized PAE",
        description="Normalized PAE confidence averaged over interface contact pairs; 0-1, higher is more confident",
    )
    pmidockq: float = Field(
        title="pmiDockQ",
        description="Per-chain pDockQ2 contribution; 0-1, higher indicates a better interface",
    )


class PDockQ2Metrics(Metrics):
    """pDockQ2 interface-quality metrics for a cofolded complex.

    Metrics documented in ``metric_spec``:
        pdockq2 (float): Overall pDockQ2 score in ``[0, 1]``. Mean of
            ``pmidockq`` over non-binder target chains that contact the binder.
            Always present.
        avg_interface_plddt (float): Mean ``if_plddt`` (0-100 scale) averaged
            across the same target chains used for ``pdockq2``. Always present.
        avg_interface_pae (float): Mean ``norm_pae`` (0-1 scale) averaged
            across the same target chains used for ``pdockq2``. Always present.
        num_interface_contacts (int): Count of residue pairs (binder residue
            in contact with any target residue) counted across target chains
            used for ``pdockq2``. Always present.

    Attributes:
        interfaces (list[InterfacePDockQ2]): Per-target-chain breakdown kept
            as a Pydantic field so it stays out of metric iteration.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "pdockq2": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "avg_interface_plddt": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "better_values_are": "higher",
        },
        "avg_interface_pae": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "num_interface_contacts": {
            "availability": "always",
            "type": "int",
            "min": 0,
            "max": None,
            "better_values_are": "context-dependent",
        },
    }
    primary_metric: str | None = Field(
        default="pdockq2",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )

    interfaces: list[InterfacePDockQ2] = Field(
        default_factory=list,
        title="Per-Chain Interfaces",
        description="Per-target-chain interface breakdown that produces the overall pDockQ2 score",
    )


class PDockQ2Input(BaseToolInput):
    """Input for pDockQ2 interface scoring.

    Attributes:
        structure (Structure): Cofolded complex with per-residue pLDDT in the
            B-factor column (``b_factor_type`` must be ``PLDDT`` or
            ``NORMALIZED_PLDDT``) and the PAE matrix attached at
            ``structure.metrics['pae']`` as a square ``list[list[float]]``
            whose dimension matches the structure's total residue count.
        binder_chain (SingleChainSelection): Single-character chain ID of the
            binder (e.g. VHH).
        target_chains (ChainSelection): Target chain IDs (single character each).
    """

    structure: Structure = InputField(
        title="Input Structure",
        description="Cofolded complex with pLDDT in B-factors and the PAE matrix at metrics['pae']",
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
    def _validate(self) -> "PDockQ2Input":
        """Validate chain IDs exist and required confidence signals are attached."""
        binder = self.binder_chain.chain
        targets = list(self.target_chains.chains)
        for cid in (binder, *targets):
            if len(cid) != 1:
                raise ValueError(f"chain IDs must be a single character, got {cid!r}")
        if binder in targets:
            raise ValueError(f"binder_chain {binder!r} must not appear in target_chains")

        available = set(self.structure.get_chain_ids())
        missing = {binder, *targets} - available
        if missing:
            raise ValueError(f"Chain ID(s) {sorted(missing)} not found in structure. Available: {sorted(available)}")

        plddt = self.structure.per_residue_plddt
        if plddt is None:
            raise ValueError("structure.per_residue_plddt is None; set b_factor_type to PLDDT or NORMALIZED_PLDDT")

        pae = self.structure.metrics.get("pae")
        if pae is None:
            raise ValueError("structure.metrics['pae'] is missing; attach the PAE matrix before scoring")
        pae_arr = np.asarray(pae, dtype=float)
        if pae_arr.ndim != 2 or pae_arr.shape[0] != pae_arr.shape[1]:
            raise ValueError(f"structure.metrics['pae'] must be a square matrix, got shape {pae_arr.shape}")
        if pae_arr.shape[0] != len(plddt):
            raise ValueError(f"PAE shape {pae_arr.shape} does not match residue count {len(plddt)}")
        return self


class PDockQ2Config(BaseConfig):
    """Configuration for pDockQ2 scoring.

    Attributes:
        distance_cutoff (float): CA-CA distance cutoff in Å for interface
            residue detection. Defaults to 10.0, matching germinal's
            ``pDockQ.pDockQ2`` wrapper default.
    """

    distance_cutoff: float = ConfigField(
        default=10.0,
        gt=0.0,
        title="Distance Cutoff (Å)",
        description="CA-CA distance cutoff (Å) for interface residue detection.",
    )


class PDockQ2Output(BaseToolOutput):
    """Output from pDockQ2 scoring.

    Attributes:
        metrics (PDockQ2Metrics): Scalar pDockQ2 metrics plus per-chain
            interface breakdown.
    """

    metrics: PDockQ2Metrics = Field(
        title="pDockQ2 Metrics",
        description="pDockQ2 metrics for the input complex",
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
    """Minimal valid input for testing and examples."""
    fixture = Path(__file__).parent / "example_input_fixture.pdb"
    pae = json.loads((Path(__file__).parent / "example_input_fixture_pae.json").read_text())
    structure = Structure.from_file(fixture, b_factor_type=BFactorType.PLDDT, metrics={"pae": pae})
    return PDockQ2Input(
        structure=structure,
        binder_chain=SingleChainSelection(chain="A"),
        target_chains=ChainSelection(chains=["B"]),
    )


@tool(
    key="pdockq2",
    label="pDockQ2 Interface Quality",
    category="structure_scoring",
    input_class=PDockQ2Input,
    config_class=PDockQ2Config,
    output_class=PDockQ2Output,
    metrics_class=PDockQ2Metrics,
    description="Score a cofolded protein complex with pDockQ2 (Zhu 2023), using pLDDT + PAE to summarize interface quality",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_pdockq2(
    inputs: PDockQ2Input,
    config: PDockQ2Config,
    instance: ToolInstance | None = None,  # noqa: ARG001 — required by tool interface
) -> PDockQ2Output:
    """Compute pDockQ2 (Zhu 2023) for a cofolded protein complex.

    Returns the mean per-chain ``pmidockq`` over chains in ``target_chains``
    that contact ``binder_chain``, plus per-chain debug rows.

    Args:
        inputs (PDockQ2Input): Cofolded complex plus binder and target chain IDs.
        config (PDockQ2Config): Scoring configuration (distance cutoff).
        instance (ToolInstance | None): Unused — pDockQ2 runs fully in-process.

    Returns:
        PDockQ2Output: Scalar pDockQ2 plus per-target-chain breakdown.
    """
    structure = inputs.structure
    binder_chain = inputs.binder_chain.chain
    target_chains = set(inputs.target_chains.chains)
    cutoff = config.distance_cutoff

    per_residue_plddt = structure.per_residue_plddt
    if per_residue_plddt is None:
        raise ValueError("structure.per_residue_plddt is None; cannot score pDockQ2")
    plddt_100 = np.asarray(per_residue_plddt, dtype=float) * 100.0
    pae = np.asarray(structure.metrics["pae"], dtype=float)

    chain_ids, ca_coords_list = _collect_ca_per_residue(structure)
    ca_coords = np.stack(ca_coords_list, axis=0)
    chain_ids_arr = np.asarray(chain_ids)
    n_res = ca_coords.shape[0]

    if n_res != plddt_100.shape[0] or n_res != pae.shape[0]:
        raise ValueError(f"Residue-count mismatch: CA={n_res}, pLDDT={plddt_100.shape[0]}, PAE={pae.shape[0]}.")

    ordered_chains = list(dict.fromkeys(chain_ids))

    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    same_chain = chain_ids_arr[:, None] == chain_ids_arr[None, :]
    in_contact = (dists <= cutoff) & ~same_chain

    per_chain_results: list[InterfacePDockQ2] = []
    for this_chain in ordered_chains:
        rows_mask = chain_ids_arr == this_chain
        contact_block = in_contact[rows_mask]
        if not contact_block.any():
            per_chain_results.append(
                InterfacePDockQ2(chain_id=this_chain, neighbor_chains="", if_plddt=0.0, norm_pae=0.0, pmidockq=0.0)
            )
            continue

        contact_counts = contact_block.sum(axis=1)
        if_plddt = float(np.average(plddt_100[rows_mask], weights=contact_counts))

        pae_values = pae[rows_mask][contact_block]
        norm_pae = float(np.mean(1.0 / (1.0 + (pae_values / 10.0) ** 2)))

        neighbor_chains = sorted(set(chain_ids_arr[contact_block.any(axis=0)].tolist()))
        neighbor_chains_str = "".join(neighbor_chains)

        pmidockq = _pmidockq_sigmoid(if_plddt * norm_pae)
        per_chain_results.append(
            InterfacePDockQ2(
                chain_id=this_chain,
                neighbor_chains=neighbor_chains_str,
                if_plddt=if_plddt,
                norm_pae=norm_pae,
                pmidockq=pmidockq,
            )
        )

    scoring_interfaces = [
        r for r in per_chain_results if r.chain_id in target_chains and binder_chain in set(r.neighbor_chains)
    ]

    if scoring_interfaces:
        pdockq2 = float(np.mean([r.pmidockq for r in scoring_interfaces]))
        avg_if_plddt = float(np.mean([r.if_plddt for r in scoring_interfaces]))
        avg_if_pae = float(np.mean([r.norm_pae for r in scoring_interfaces]))
        binder_rows = chain_ids_arr == binder_chain
        target_cols = np.isin(chain_ids_arr, [r.chain_id for r in scoring_interfaces])
        num_contacts = int(in_contact[np.ix_(binder_rows, target_cols)].sum())
    else:
        logger.warning(
            "No target chains in %s contact binder_chain=%r; reporting pdockq2=0.0",
            sorted(target_chains),
            binder_chain,
        )
        pdockq2 = 0.0
        avg_if_plddt = 0.0
        avg_if_pae = 0.0
        num_contacts = 0

    metrics = PDockQ2Metrics(
        pdockq2=pdockq2,
        avg_interface_plddt=avg_if_plddt,
        avg_interface_pae=avg_if_pae,
        num_interface_contacts=num_contacts,
        interfaces=per_chain_results,
    )

    return PDockQ2Output(
        metadata={
            "binder_chain": binder_chain,
            "target_chains": list(inputs.target_chains.chains),
            "distance_cutoff": cutoff,
        },
        metrics=metrics,
    )


# ============================================================================
# Helpers
# ============================================================================
def _pmidockq_sigmoid(x: float) -> float:
    """Zhu-2023 pmiDockQ sigmoid (L, x0, k, b fit parameters are fixed, not tunable)."""
    return 1.31034849 / (1.0 + math.exp(-0.0747157696 * (x - 84.7326239))) + 0.00501886443


def _collect_ca_per_residue(structure: Structure) -> tuple[list[str], list[np.ndarray]]:
    """Extract per-residue chain ID and CA coordinate (NaN if missing)."""
    chain_ids: list[str] = []
    ca_coords: list[np.ndarray] = []
    nan3 = np.array([np.nan, np.nan, np.nan])
    for model in structure.gemmi_struct:
        for chain in model:
            for residue in chain:
                chain_ids.append(chain.name)
                ca = next((atom for atom in residue if atom.name == "CA"), None)
                ca_coords.append(np.array([ca.pos.x, ca.pos.y, ca.pos.z]) if ca else nan3)
    return chain_ids, ca_coords

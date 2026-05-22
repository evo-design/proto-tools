"""Germinal de novo epitope-targeted antibody design (VHH or scFv)."""

from __future__ import annotations

import csv as _csv
import json as _json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from proto_tools.entities.structures.structure import Structure
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
# Metrics
# ============================================================================
class GerminalDesignMetrics(Metrics):
    """Per-design quality metrics from the Germinal pipeline.

    Metric names match Germinal's ``TRAJECTORY_METRICS_TO_SAVE`` (in
    ``germinal/utils/io.py``) plus the filter metrics declared in
    ``configs/filter/{initial,final}/{vhh,scfv}.yaml``. Populated from
    Germinal's per-stage ``designs.csv`` and the master ``all_trajectories.csv``.
    Some metrics are only present after the design reaches the corresponding
    pipeline stage (e.g. ``external_*`` values appear only after structure
    validation succeeded).

    Attributes:
        metric_spec (ClassVar[dict[str, MetricSpec]]): Per-metric type/range/availability metadata.
        primary_metric (str | None): ``"i_ptm"`` — interface pTM is the single
            best confidence summary for binder-design quality.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        # Trajectory metrics (TRAJECTORY_METRICS_TO_SAVE in germinal/utils/io.py)
        "plddt": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
        "ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
        "i_ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
        "i_pae": {"availability": "always", "type": "float", "min": 0.0, "unit": "Å"},
        "pae": {"availability": "always", "type": "float", "min": 0.0, "unit": "Å"},
        "loss": {"availability": "always", "type": "float"},
        "lm_ll": {"availability": "always", "type": "float"},
        "helix": {"availability": "always", "type": "float"},
        "beta_strand": {"availability": "always", "type": "float"},
        # Filter metrics (configs/filter/{initial,final}/{vhh,scfv}.yaml)
        "clashes": {"availability": "after filtering", "type": "int", "min": 0},
        "sc_rmsd": {"availability": "after filtering", "type": "float", "min": 0.0, "unit": "Å"},
        "binder_near_hotspot": {"availability": "after filtering", "type": "bool"},
        "cdr3_hotspot_contacts": {"availability": "after filtering", "type": "int", "min": 0},
        "percent_interface_cdr": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "interface_shape_comp": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "interface_hbonds": {"availability": "after filtering", "type": "int", "min": 0},
        "surface_hydrophobicity": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "interface_hydrophobicity": {"availability": "after filtering", "type": "float", "min": 0.0},
        "pdockq2": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        # External structure-validation metrics (Chai-1 / AF3 / Protenix)
        "external_plddt": {"availability": "after validation", "type": "float", "min": 0.0, "max": 1.0},
        "external_iptm": {"availability": "after validation", "type": "float", "min": 0.0, "max": 1.0},
        "external_ptm": {"availability": "after validation", "type": "float", "min": 0.0, "max": 1.0},
        "external_pae": {"availability": "after validation", "type": "float", "min": 0.0, "unit": "Å"},
        "external_i_pae": {"availability": "after validation", "type": "float", "min": 0.0, "unit": "Å"},
        "external_i_plddt": {"availability": "after validation", "type": "float", "min": 0.0, "max": 1.0},
        "external_plddt_binder": {"availability": "after validation", "type": "float", "min": 0.0, "max": 1.0},
        "external_chain_ptm": {"availability": "after validation", "type": "float", "min": 0.0, "max": 1.0},
        "external_binder_pae": {"availability": "after validation", "type": "float", "min": 0.0, "unit": "Å"},
        "external_aggregate_score": {"availability": "after validation", "type": "float"},
        # Modern interface-quality metrics
        "ipsae": {"availability": "after validation", "type": "float", "min": 0.0, "max": 1.0},
        "ipsae_pdockq2": {"availability": "after validation", "type": "float", "min": 0.0, "max": 1.0},
        "lis_lis": {"availability": "after validation", "type": "float"},
        "lis_lia": {"availability": "after validation", "type": "float"},
        # PyRosetta interface scoring
        "binder_score": {"availability": "after filtering", "type": "float", "unit": "REU"},
        "interface_packstat": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "interface_dG": {"availability": "after filtering", "type": "float", "unit": "REU"},
        "interface_dSASA": {"availability": "after filtering", "type": "float", "min": 0.0, "unit": "Å²"},
        "interface_dG_SASA_ratio": {"availability": "after filtering", "type": "float", "unit": "REU/Å²"},
        "interface_fraction": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "interface_nres": {"availability": "after filtering", "type": "int", "min": 0},
        "interface_hbond_percentage": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "interface_delta_unsat_hbonds": {"availability": "after filtering", "type": "int", "min": 0},
        "interface_delta_unsat_hbonds_percentage": {
            "availability": "after filtering",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
        },
        "clashes_unrelaxed": {"availability": "after filtering", "type": "int", "min": 0},
        # Developability + composition
        "hydrophobic_patches_binder": {"availability": "after filtering", "type": "int", "min": 0},
        "hydrophobic_patches_struct": {"availability": "after filtering", "type": "int", "min": 0},
        "sap_score": {"availability": "after filtering", "type": "float"},
        "cdr_sap": {"availability": "after filtering", "type": "float"},
        "cdr_hotspot_contacts": {"availability": "after filtering", "type": "int", "min": 0},
        "percent_interface_cdr3": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "alpha_interface": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "beta_interface": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "loops_interface": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "alpha_all": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "beta_all": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "loops_all": {"availability": "after filtering", "type": "float", "min": 0.0, "max": 1.0},
        "n_framework_mutations": {"availability": "after filtering", "type": "int", "min": 0},
    }
    primary_metric: str | None = Field(
        default="i_ptm",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


# ============================================================================
# Input
# ============================================================================
_HOTSPOT_PATTERN = re.compile(r"^[A-Za-z]\d+$")
# Hydra dotpath: lowercase identifier segments joined by '.', optional leading '+' (append).
_OVERRIDE_KEY_PATTERN = re.compile(r"\+?[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*")


class GerminalInput(BaseToolInput):
    """Input for Germinal antibody design.

    Attributes:
        target_pdb (Structure): Target structure. Accepts a file path, raw
            PDB/CIF content string, ``Structure`` object, or a dict in the
            shape produced by ``Structure.model_dump(mode='json')``. Must
            include a chain matching ``target_chain``.
        target_chain (str): Chain ID(s) of the target. Single letter (e.g.
            ``"A"``) or comma-separated for multi-chain targets (e.g. ``"A,B"``).
        binder_chain (str): Chain ID assigned to the designed binder. Default
            ``"B"`` (matches Germinal's source convention in
            ``configs/target/pdl1.yaml``).
        hotspots (list[str]): Hotspot residues on the target in
            ``"<chain_letter><resnum>"`` format (e.g. ``["A37", "A39", "A41"]``).
            These are the residues the designed binder is forced to contact.
        target_name (str | None): Short identifier for this target. Used as the
            Hydra ``target=<name>`` selector and as a prefix in output filenames.
            If ``None``, the inference layer derives one from a hash of the PDB
            content.
        hotspot_residue (str | None): Optional single residue (e.g. ``"W40"``)
            used as the Chai-1 contact-restraint anchor. Mirrors Germinal's
            ``hotspot_residue`` field in ``configs/target/*.yaml``.
    """

    target_pdb: Structure = InputField(
        title="Target PDB",
        description="Target structure",
    )
    target_chain: str = InputField(
        default="A",
        title="Target Chain",
        description="Target chain ID(s); comma-separated for multi-chain targets",
    )
    binder_chain: str = InputField(
        default="B",
        title="Binder Chain",
        description="Chain ID assigned to the designed binder; must differ from target_chain.",
    )
    hotspots: list[str] = InputField(
        default_factory=list,
        title="Hotspots",
        description="Target hotspot residues, e.g. ['A37','A39'] (chain letter + 1-indexed resnum)",
    )
    target_name: str | None = InputField(
        default=None,
        title="Target Name",
        description="Short identifier; defaults to a hash of the PDB content",
    )
    hotspot_residue: str | None = InputField(
        default=None,
        title="Hotspot Residue",
        description="Single-residue contact-restraint anchor for Chai-1 (e.g. 'W40'). Ignored by AF3 / Protenix.",
    )

    @field_validator("hotspots", mode="after")
    @classmethod
    def _validate_hotspot_format(cls, v: list[str]) -> list[str]:
        """Each hotspot must match '<chain_letter><resnum>'."""
        bad = [h for h in v if not _HOTSPOT_PATTERN.match(h)]
        if bad:
            raise ValueError(f"Hotspots must be '<chain><resnum>' (e.g. 'A37'). Bad: {bad}")
        return v

    @field_validator("target_name", mode="after")
    @classmethod
    def _validate_target_name(cls, v: str | None) -> str | None:
        """target_name flows into file paths and a Hydra selector; restrict to safe chars."""
        if v is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", v):
            raise ValueError(f"target_name must be alphanumeric/_/- (got {v!r})")
        return v

    @model_validator(mode="after")
    def _validate_chains(self) -> GerminalInput:
        """Binder chain must differ from target; every hotspot chain must exist on target."""
        target_chains = {c.strip() for c in self.target_chain.split(",")}
        if self.binder_chain in target_chains:
            raise ValueError(
                f"binder_chain ({self.binder_chain!r}) must differ from target_chain {sorted(target_chains)}"
            )
        unknown = {h[0] for h in self.hotspots} - target_chains
        if unknown:
            raise ValueError(f"Hotspot chains {sorted(unknown)} not in target_chain {sorted(target_chains)}")
        return self


# ============================================================================
# Config
# ============================================================================
class GerminalConfig(BaseConfig):
    """Configuration for Germinal antibody design.

    Type-dependent knobs (loss weights, iteration counts, AbLM scales) are not
    exposed here — the ``design_type`` preset wins. Use ``germinal_overrides``
    for arbitrary Hydra ``<key>=<value>`` overrides on the run config.

    Attributes:
        design_type (Literal["vhh", "scfv"]): Run preset selector.
        max_trajectories (int): Hard cap on total trajectories before stopping.
        max_hallucinated_trajectories (int): Cap on trajectories that complete
            the hallucination stage (before MPNN refinement).
        max_passing_designs (int): Stop early once this many designs pass all
            final filters.
        structure_model (Literal["chai", "af3", "protenix"]): Cofolding backend
            for structure validation. Default ``"chai"`` (auto-installed);
            ``"af3"`` and ``"protenix"`` require user-provisioned weights/env
            (see README → Backend Configuration).
        plddt_threshold (float | None): Override final ``external_plddt``
            (VHH: ``> 0.87``, scFv: ``> 0.85``). Distinct from upstream's
            in-loop ``plddt_threshold`` save filter — use ``germinal_overrides``.
        iptm_threshold (float | None): Override final ``external_iptm``
            (preset: ``> 0.74``).
        ipae_threshold (float | None): Override final ``external_pae`` in Å
            (VHH: ``< 7.5``, scFv: ``< 8``).
        ptm_threshold (float | None): Override final ``external_ptm``
            (preset: ``> 0.84``).
        pdockq2_threshold (float | None): Override final ``pdockq2``
            (preset: ``> 0.23``).
        germinal_overrides (dict[str, Any]): Arbitrary Hydra overrides for
            ``run_germinal.py`` (e.g. ``{"logits_steps": 100, "weights_iptm": 1.0}``).
            Applied verbatim as ``<key>=<value>`` CLI args.
        filter_overrides (dict[str, dict[str, dict[str, Any]]]): Override
            filter YAML values. Schema:
            ``{"initial" | "final": {<filter_name>: {"value": <v>, "operator": <op>}}}``.
            Merged on top of the design_type preset.
        device (str): Device for the Germinal subprocess. Forced to ``"cuda"``;
            CPU is not supported by the upstream pipeline.
        output_dir (str | None): Optional persistent output directory. If
            unset, a temp dir is used and discarded after the call.
    """

    design_type: Literal["vhh", "scfv"] = ConfigField(
        title="Design Type",
        default="vhh",
        description="Run preset: 'vhh' (single-domain nanobody) or 'scfv' (scFv).",
        reload_on_change=True,
    )

    max_trajectories: int = ConfigField(
        title="Max Trajectories",
        default=10000,
        ge=1,
        description="Hard cap on total trajectories before stopping (success or failure).",
    )
    max_hallucinated_trajectories: int = ConfigField(
        title="Max Hallucinated Trajectories",
        default=1000,
        ge=1,
        description="Cap on trajectories that complete the hallucination stage (before MPNN refinement).",
    )
    max_passing_designs: int = ConfigField(
        title="Max Passing Designs",
        default=100,
        ge=1,
        description="Stop early once this many designs pass all final filters.",
    )

    structure_model: Literal["chai", "af3", "protenix"] = ConfigField(
        title="Cofolding Backend",
        default="chai",
        description="Cofolding backend: 'chai' (auto-installed) or 'af3' / 'protenix' (user-provisioned).",
        reload_on_change=True,
    )

    plddt_threshold: float | None = ConfigField(
        title="pLDDT Threshold",
        default=None,
        ge=0.0,
        le=1.0,
        description="Override final external_plddt (VHH preset: > 0.87, scFv preset: > 0.85).",
    )
    iptm_threshold: float | None = ConfigField(
        title="ipTM Threshold",
        default=None,
        ge=0.0,
        le=1.0,
        description="Override final external_iptm (preset: > 0.74).",
    )
    ipae_threshold: float | None = ConfigField(
        title="iPAE Threshold (Å)",
        default=None,
        gt=0.0,
        description="Override final external_pae (VHH preset: < 7.5, scFv preset: < 8).",
    )
    ptm_threshold: float | None = ConfigField(
        title="pTM Threshold",
        default=None,
        ge=0.0,
        le=1.0,
        description="Override final external_ptm (preset: > 0.84).",
    )
    pdockq2_threshold: float | None = ConfigField(
        title="pDockQ2 Threshold",
        default=None,
        ge=0.0,
        le=1.0,
        description="Override final pdockq2 (preset: > 0.23).",
    )

    germinal_overrides: dict[str, Any] = ConfigField(
        title="Germinal Hydra Overrides",
        default_factory=dict,
        description="Free-form Hydra overrides (key must be a valid dotpath). Applied as <key>=<value>.",
    )
    filter_overrides: dict[str, dict[str, dict[str, Any]]] = ConfigField(
        title="Filter Overrides",
        default_factory=dict,
        description="Override filter YAMLs: {'initial' | 'final': {key: {'value': v, 'operator': op}}}.",
    )

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device for the Germinal subprocess (forced 'cuda'; CPU not supported).",
        include_in_key=False,
    )
    output_dir: str | None = ConfigField(
        title="Output Directory",
        default=None,
        description="Optional persistent output directory; if unset, a temp dir is used.",
        include_in_key=False,
    )
    # Germinal campaigns run hours-to-days; truncating mid-run wastes compute (default: no cap).
    timeout: int | None = ConfigField(
        title="Timeout",
        default=None,
        ge=1,
        description="Maximum execution time in seconds. None (default) waits indefinitely.",
        include_in_key=False,
    )
    # verbose, seed inherited from BaseConfig.

    @field_validator("germinal_overrides", mode="after")
    @classmethod
    def _validate_override_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Reject Hydra-structural directives that would redirect the run dir or output."""
        for key in v:
            if not _OVERRIDE_KEY_PATTERN.fullmatch(key):
                raise ValueError(f"germinal_overrides key {key!r} is not a valid Hydra dotpath")
            if key.startswith(("hydra.", "hydra/")):
                raise ValueError(f"germinal_overrides key {key!r}: 'hydra.*' overrides are not permitted")
        return v

    @classmethod
    def minimal(cls, **kwargs: Any) -> GerminalConfig:
        """Smoke-test config: 2 trajectories, 1 passing design.

        Args:
            **kwargs (Any): Field values; take precedence over the smoke defaults.

        Returns:
            GerminalConfig: Smoke-test config.
        """
        kwargs.setdefault("max_trajectories", 2)
        kwargs.setdefault("max_hallucinated_trajectories", 1)
        kwargs.setdefault("max_passing_designs", 1)
        return super().minimal(**kwargs)  # type: ignore[return-value]


# ============================================================================
# Output
# ============================================================================
class GerminalDesign(BaseModel):
    """A single Germinal-produced antibody design.

    Attributes:
        sequence_heavy (str): Heavy chain (or VHH) amino-acid sequence.
        sequence_light (str | None): Light chain sequence (scFv only).
        structure (Structure): Predicted binder + target complex.
        metrics (GerminalDesignMetrics): Per-design quality metrics.
        stage_passed (Literal["accepted", "redesign_candidate", "trajectory"]):
            Highest pipeline stage this design reached.
        design_id (str): Germinal's internal design identifier
            (``"<target>_<type>_s<seed>"`` for trajectory-only designs,
            ``"<target>_<type>_s<seed>_abmpnn_<j>"`` after AbMPNN redesign).
        trajectory_index (int): Trajectory seed (parsed from ``_s<seed>``).
            Germinal uses the seed as its unique trajectory identifier.
        mpnn_index (int): AbMPNN sample index (1-based; 0 for trajectory-only
            designs that never reached the redesign stage).
    """

    model_config = ConfigDict(extra="forbid")

    sequence_heavy: str = Field(
        title="Heavy Chain Sequence",
        description="Heavy chain or VHH amino acid sequence",
    )
    sequence_light: str | None = Field(
        default=None,
        title="Light Chain Sequence",
        description="Light chain sequence (scFv only)",
    )
    structure: Structure = Field(
        title="Predicted Complex",
        description="Predicted binder + target complex",
    )
    metrics: GerminalDesignMetrics = Field(
        title="Design Quality Metrics",
        description="Per-design quality metrics from the Germinal pipeline.",
    )
    stage_passed: Literal["accepted", "redesign_candidate", "trajectory"] = Field(
        title="Stage Passed",
        description="Highest pipeline stage this design reached",
    )
    design_id: str = Field(
        title="Design ID",
        description="Germinal's internal design identifier",
    )
    trajectory_index: int = Field(
        ge=0,
        title="Trajectory Seed",
        description="Trajectory seed; extracted from the '_s<seed>' suffix in design_id.",
    )
    mpnn_index: int = Field(
        ge=0,
        title="AbMPNN Index",
        description="AbMPNN sample index (1-based; 0 for trajectory-only designs without redesign).",
    )


class GerminalOutput(BaseToolOutput):
    """Output of a Germinal antibody-design campaign.

    Attributes:
        designs (list[GerminalDesign]): All produced designs across the
            ``accepted``, ``redesign_candidate``, and ``trajectory`` stages.
        pipeline_stats (dict[str, int]): Per-stage counts from Germinal's
            ``failure_counts.csv`` (trajectories attempted, designs accepted,
            and per-filter failure counts).
        num_accepted (int): Computed; number of designs in the ``accepted`` stage.
        num_designs (int): Computed; total number of returned designs.
    """

    designs: list[GerminalDesign] = Field(
        default_factory=list,
        title="Designs",
        description="All produced designs",
    )
    pipeline_stats: dict[str, int] = Field(
        default_factory=dict,
        title="Pipeline Stats",
        description="Per-stage counts from Germinal's failure_counts.csv",
    )

    @computed_field(  # type: ignore[prop-decorator]
        title="Accepted Count",
        description="Number of designs that passed all final filters.",
    )
    @property
    def num_accepted(self) -> int:
        """Number of designs that passed all final filters."""
        return sum(1 for d in self.designs if d.stage_passed == "accepted")

    @computed_field(  # type: ignore[prop-decorator]
        title="Total Designs",
        description="Total number of designs returned across all pipeline stages.",
    )
    @property
    def num_designs(self) -> int:
        """Total number of designs returned across all pipeline stages."""
        return len(self.designs)

    def __len__(self) -> int:
        """Total number of designs."""
        return len(self.designs)

    def __getitem__(self, index: int) -> GerminalDesign:
        """Index into designs."""
        return self.designs[index]

    def __iter__(self) -> Iterator[GerminalDesign]:  # type: ignore[override]
        """Iterate over designs."""
        return iter(self.designs)

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["pdb", "csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Default export format (PDB structures, one per design)."""
        return "pdb"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export designs to disk.

        - ``pdb``: write each design's structure to ``<export_path>/<design_id>.pdb``
        - ``csv``: write a single CSV with one row per design (sequences + flattened metrics)
        - ``json``: write a single JSON file with all designs (excluding structures)
        """
        path = Path(export_path)
        if file_format == "pdb":
            path.mkdir(parents=True, exist_ok=True)
            for d in self.designs:
                d.structure.write_pdb(path / f"{d.design_id}.pdb")
            return
        if file_format in ("csv", "json"):
            out_path = path.with_suffix(f".{file_format}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if file_format == "csv":
                metric_keys = sorted({k for d in self.designs for k in d.metrics})
                with open(out_path, "w", newline="") as f:
                    writer = _csv.writer(f)
                    writer.writerow(["design_id", "stage_passed", "sequence_heavy", "sequence_light", *metric_keys])
                    for d in self.designs:
                        writer.writerow(
                            [
                                d.design_id,
                                d.stage_passed,
                                d.sequence_heavy,
                                d.sequence_light or "",
                                *[d.metrics.get(k, "") for k in metric_keys],
                            ]
                        )
            else:  # json
                payload = {
                    "designs": [d.model_dump(exclude={"structure"}) for d in self.designs],
                    "pipeline_stats": self.pipeline_stats,
                }
                out_path.write_text(_json.dumps(payload, indent=2, default=str))
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> GerminalInput:
    """Minimal valid input: PD-L1 target with three published epitope hotspots."""
    repo_root = Path(__file__).resolve().parents[4]  # germinal -> binder_design -> tools -> proto_tools -> repo
    return GerminalInput(
        target_pdb=Structure.from_file(repo_root / "tests" / "dummy_data" / "pdl1.pdb"),
        target_chain="A",
        binder_chain="B",
        hotspots=["A56", "A66", "A115"],
        target_name="pdl1_example",
    )


@tool(
    key="germinal-design",
    label="Germinal Antibody Design",
    category="binder_design",
    input_class=GerminalInput,
    config_class=GerminalConfig,
    output_class=GerminalOutput,
    metrics_class=GerminalDesignMetrics,
    description="De novo epitope-targeted antibody design (VHH or scFv) using the Germinal pipeline",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    stochastic=True,
)
def run_germinal_design(
    inputs: GerminalInput,
    config: GerminalConfig,
    instance: Any = None,
) -> GerminalOutput:
    """Run a Germinal antibody-design campaign end-to-end.

    Spawns the upstream ``run_germinal.py`` (pinned commit) inside the tool's
    standalone env, parses the resulting ``runs/<exp>/{accepted,redesign_candidates,trajectories}/``
    output trees, and returns a typed :class:`GerminalOutput`. Each call is one
    end-to-end campaign against a single target — Germinal's pipeline is
    stateful within a run so we do not fan out across targets.

    Args:
        inputs (GerminalInput): Target PDB + chain layout + hotspot residues.
        config (GerminalConfig): Pipeline knobs. Defaults match Germinal source;
            see attribute docstrings for source-vs-default mapping.
        instance (Any): Optional :class:`ToolInstance` for subprocess execution.

    Returns:
        GerminalOutput: All produced designs (accepted + redesign-candidate +
            trajectory stages) plus per-stage failure counts.

    Example:
        >>> from proto_tools.tools.binder_design import (
        ...     run_germinal_design,
        ...     GerminalInput,
        ...     GerminalConfig,
        ... )
        >>> inputs = GerminalInput(
        ...     target_pdb="pdbs/pdl1.pdb",
        ...     target_chain="A",
        ...     binder_chain="B",
        ...     hotspots=["A56", "A66", "A115"],
        ...     target_name="pdl1",
        ... )
        >>> config = GerminalConfig(design_type="vhh", max_trajectories=2, max_passing_designs=1)
        >>> result = run_germinal_design(inputs, config)
        >>> result.num_accepted
        0
    """
    logger.debug(
        "Dispatching Germinal subprocess for target %s (design_type=%s, structure_model=%s)",
        inputs.target_name or "<unnamed>",
        config.design_type,
        config.structure_model,
    )

    # Materialize the target Structure to a tempfile for the standalone to read.
    with inputs.target_pdb.temp_file() as target_pdb_path:
        input_data: dict[str, Any] = {
            # Target
            "target_pdb": str(target_pdb_path),
            "target_chain": inputs.target_chain,
            "binder_chain": inputs.binder_chain,
            "hotspots": inputs.hotspots,
            "target_name": inputs.target_name,
            "hotspot_residue": inputs.hotspot_residue,
            # Mode + stopping criteria + backend
            "design_type": config.design_type,
            "max_trajectories": config.max_trajectories,
            "max_hallucinated_trajectories": config.max_hallucinated_trajectories,
            "max_passing_designs": config.max_passing_designs,
            "structure_model": config.structure_model,
            # Optional filter thresholds (None means preset wins)
            "plddt_threshold": config.plddt_threshold,
            "iptm_threshold": config.iptm_threshold,
            "ipae_threshold": config.ipae_threshold,
            "ptm_threshold": config.ptm_threshold,
            "pdockq2_threshold": config.pdockq2_threshold,
            # Escape hatches
            "germinal_overrides": config.germinal_overrides,
            "filter_overrides": config.filter_overrides,
            # Execution
            "device": config.device,
            "output_dir": config.output_dir,
            "seed": config.seed,
            "verbose": config.verbose,
        }
        output_data = ToolInstance.dispatch("germinal", input_data, instance=instance, config=config)

    designs = [
        GerminalDesign(
            sequence_heavy=d["sequence_heavy"],
            sequence_light=d.get("sequence_light"),
            structure=Structure(structure=d["structure_content"], source="germinal-design"),
            metrics=GerminalDesignMetrics(**d.get("metrics", {})),
            stage_passed=d["stage_passed"],
            design_id=d["design_id"],
            trajectory_index=int(d["trajectory_index"]),
            mpnn_index=int(d["mpnn_index"]),
        )
        for d in output_data.get("designs", [])
    ]

    return GerminalOutput(
        designs=designs,
        pipeline_stats=output_data.get("pipeline_stats", {}),
    )

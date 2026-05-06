"""BindCraft binder-design pipeline (AF2 hallucination + ProteinMPNN + PyRosetta filtering)."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from proto_tools.entities.structures import BFactorType, Structure
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

_BINDCRAFT_FIXTURE_PDB = (
    Path(__file__).resolve().parents[2] / "structure_prediction" / "alphafold2" / "example_input_fixture.pdb"
)

# Upstream parity (martinpacesa/BindCraft @ 7cd4ace, default_4stage_multimer.json):
# 51 user-facing ConfigFields below + 10 internal toggles hardcoded in dispatch.
_USER_FACING_UPSTREAM_KEYS: frozenset[str] = frozenset(
    {
        # Algorithm selection
        "design_algorithm",
        "use_multimer_design",
        "omit_AAs",
        "force_reject_AA",
        # Iteration counts
        "soft_iterations",
        "temporary_iterations",
        "hard_iterations",
        "greedy_iterations",
        "greedy_percentage",
        # Loss weights
        "weights_plddt",
        "weights_pae_intra",
        "weights_pae_inter",
        "weights_con_intra",
        "weights_con_inter",
        "weights_helicity",
        "weights_iptm",
        "weights_rg",
        "weights_termini_loss",
        # Loss toggles
        "random_helicity",
        "use_i_ptm_loss",
        "use_rg_loss",
        "use_termini_distance_loss",
        # Contact geometry
        "intra_contact_distance",
        "inter_contact_distance",
        "intra_contact_number",
        "inter_contact_number",
        # Template masking / prediction modifiers
        "rm_template_seq_design",
        "rm_template_seq_predict",
        "rm_template_sc_design",
        "rm_template_sc_predict",
        "predict_initial_guess",
        "predict_bigbang",
        # MPNN refinement
        "enable_mpnn",
        "mpnn_fix_interface",
        "num_seqs",
        "max_mpnn_sequences",
        "sampling_temp",
        "backbone_noise",
        "model_path",
        "mpnn_weights",
        # AF2 validation / beta-protein optimisation
        "num_recycles_design",
        "num_recycles_validation",
        "optimise_beta",
        "optimise_beta_extra_soft",
        "optimise_beta_extra_temp",
        "optimise_beta_recycles_design",
        "optimise_beta_recycles_valid",
        # Stopping / monitoring
        "max_trajectories",
        "enable_rejection_check",
        "acceptance_rate",
        "start_monitoring",
    }
)

# Upstream knobs not surfaced to users: file-output toggles whose artifacts
# proto-tools never returns, plus the standard sample_models flag.
_HARDCODED_INTERNAL_SETTINGS: dict[str, Any] = {
    "sample_models": True,
    "save_design_animations": False,
    "save_design_trajectory_plots": False,
    "save_trajectory_pickle": False,
    "save_mpnn_fasta": False,
    "zip_animations": False,
    "zip_plots": False,
    "remove_unrelaxed_trajectory": True,
    "remove_unrelaxed_complex": True,
    "remove_binder_monomer": True,
}


# ============================================================================
# Metrics
# ============================================================================


class BindCraftMetrics(Metrics):
    """Metrics for one accepted BindCraft binder design.

    Mirrors the ``Average_*`` columns in upstream's ``final_design_stats.csv``
    (the columns the filter check evaluates against), with snake_case names.

    Attributes:
        metric_spec (ClassVar[dict[str, MetricSpec]]): Per-metric metadata
            (type, range, unit). Used by the validation helper in
            ``tests/tool_infra_tests/_metric_helpers.py``.
        primary_metric (str | None): Headline metric name; defaults to ``"avg_iptm"``.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        # AlphaFold2 confidence
        "avg_plddt": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        "avg_ptm": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        "avg_iptm": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        "avg_pae": {"type": "float", "min": 0.0, "unit": "Å"},
        "avg_ipae": {"type": "float", "min": 0.0, "unit": "Å"},
        "avg_iplddt": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        "avg_ss_plddt": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        "avg_binder_plddt": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        "avg_binder_ptm": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        "avg_binder_pae": {"type": "float", "min": 0.0, "unit": "Å"},
        # Rosetta interface energies (REU = Rosetta Energy Units)
        "binder_energy_score": {"type": "float", "unit": "REU"},
        "dG": {"type": "float", "unit": "REU"},
        "dSASA": {"type": "float", "min": 0.0, "unit": "Å^2"},
        "dG_per_dSASA": {"type": "float", "unit": "REU/Å^2"},
        "interface_sasa_pct": {"type": "float", "min": 0.0, "max": 100.0, "unit": "percent"},
        "interface_hydrophobicity": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        "surface_hydrophobicity": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        "shape_complementarity": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        "packstat": {"type": "float", "min": 0.0, "max": 1.0, "unit": "fraction"},
        # H-bonds
        "n_interface_hbonds": {"type": "float", "min": 0.0, "unit": "count"},
        "interface_hbonds_pct": {"type": "float", "min": 0.0, "max": 100.0, "unit": "percent"},
        "n_interface_unsat_hbonds": {"type": "float", "min": 0.0, "unit": "count"},
        "interface_unsat_hbonds_pct": {"type": "float", "min": 0.0, "max": 100.0, "unit": "percent"},
        # Counts
        "n_interface_residues": {"type": "float", "min": 0.0, "unit": "count"},
        # Secondary structure (percentages, 0-100)
        "binder_helix_pct": {"type": "float", "min": 0.0, "max": 100.0, "unit": "percent"},
        "binder_betasheet_pct": {"type": "float", "min": 0.0, "max": 100.0, "unit": "percent"},
        "binder_loop_pct": {"type": "float", "min": 0.0, "max": 100.0, "unit": "percent"},
        "interface_helix_pct": {"type": "float", "min": 0.0, "max": 100.0, "unit": "percent"},
        "interface_betasheet_pct": {"type": "float", "min": 0.0, "max": 100.0, "unit": "percent"},
        "interface_loop_pct": {"type": "float", "min": 0.0, "max": 100.0, "unit": "percent"},
        # RMSDs
        "hotspot_rmsd": {"type": "float", "min": 0.0, "unit": "Å"},
        "target_rmsd": {"type": "float", "min": 0.0, "unit": "Å"},
        "binder_rmsd": {"type": "float", "min": 0.0, "unit": "Å"},
        # Clashes
        "unrelaxed_clashes": {"type": "float", "min": 0.0, "unit": "count"},
        "relaxed_clashes": {"type": "float", "min": 0.0, "unit": "count"},
    }
    primary_metric: str | None = "avg_iptm"


# ============================================================================
# Single accepted design
# ============================================================================


class BindCraftDesign(BaseModel):
    """One accepted binder design from a BindCraft pipeline run.

    Attributes:
        design_name (str): Unique design identifier emitted by upstream
            (e.g. ``"binder_l60_s12345_mpnn3"``).
        binder_sequence (str): Designed binder amino-acid sequence (1-letter codes).
        structure (Structure): Relaxed target+binder complex; B-factors are pLDDT
            on the 0-100 PDB scale (``b_factor_type=PLDDT``).
        metrics (BindCraftMetrics): Per-design averaged metrics that the
            filter check evaluates against.
        seed (int): Random seed of the trajectory that produced this design.
        interface_aas (dict[str, int]): Amino-acid composition at the binder-target interface.
        interface_residues (list[int]): 1-indexed binder residue positions at the interface.
    """

    model_config = ConfigDict(extra="forbid")

    design_name: str = Field(description="Unique design identifier emitted by BindCraft.")
    binder_sequence: str = Field(description="Designed binder amino-acid sequence (1-letter codes).")
    structure: Structure = Field(description="Relaxed target+binder complex with per-residue pLDDT in B-factors.")
    metrics: BindCraftMetrics = Field(description="Per-design averaged metrics evaluated by the filter check.")
    seed: int = Field(ge=0, description="Random seed of the trajectory that produced this design.")
    interface_aas: dict[str, int] = Field(
        default_factory=dict,
        description="Amino-acid composition at the binder-target interface.",
    )
    interface_residues: list[int] = Field(
        default_factory=list,
        description="1-indexed binder residue positions at the interface.",
    )


# ============================================================================
# Input
# ============================================================================


class BindCraftInput(BaseToolInput):
    """Target specification for one BindCraft binder-design run.

    Attributes:
        target_pdb (str): Target structure (file path or PDB-format string).
        target_chain (str): Chain ID(s) of the frozen target (comma-separated for
            multi-chain). Maps to BindCraft's ``chains``.
        target_hotspot_residues (str | None): Comma-separated 1-indexed residue
            positions on the target that the binder must contact. Supports ranges
            (e.g. ``"1-10,56,78"``). ``None`` or empty = unrestricted.
        binder_lengths (tuple[int, int]): ``(min, max)`` binder length range.
            Maps to BindCraft's ``lengths``.
        binder_name (str): Project identifier — used as a prefix in output filenames.
        number_of_final_designs (int): Target accepted-design count. The pipeline
            stops after reaching this count or after ``max_trajectories`` attempts
            (whichever comes first).
    """

    target_pdb: str = InputField(
        description="Target structure (file path or PDB-format string).",
    )
    target_chain: str = InputField(
        default="A",
        description="Chain ID(s) of the frozen target (comma-separated for multi-chain).",
    )
    target_hotspot_residues: str | None = InputField(
        default=None,
        description="Comma-separated 1-indexed residue positions on the target (e.g. '1-10,56,78').",
    )
    binder_lengths: tuple[int, int] = InputField(
        default=(65, 150),
        description="(min, max) binder length range. Upstream default: (65, 150).",
    )
    binder_name: str = InputField(
        default="binder",
        description="Prefix used in each accepted design's name (e.g. 'binder' → 'binder_l60_s12345_mpnn3').",
    )
    number_of_final_designs: int = InputField(
        default=100,
        ge=1,
        description="Target accepted-design count. Set to 1 for one-off sampling. Upstream default: 100.",
    )

    @model_validator(mode="after")
    def _validate_lengths(self) -> Self:
        """Ensure ``binder_lengths`` is ``(min, max)`` with ``min <= max`` and both positive."""
        lo, hi = self.binder_lengths
        if lo <= 0 or hi <= 0:
            raise ValueError(f"binder_lengths must be positive integers, got ({lo}, {hi})")
        if lo > hi:
            raise ValueError(f"binder_lengths must be (min, max) with min <= max, got ({lo}, {hi})")
        return self


# ============================================================================
# Config
# ============================================================================


class BindCraftConfig(BaseConfig):
    """User-facing BindCraft settings.

    Defaults match upstream's pinned ``default_4stage_multimer.json``;
    ``filter_overrides`` merges per-metric thresholds on top of the default-filters
    snapshot at dispatch.

    Attributes:
        design_algorithm (Literal["2stage", "3stage", "4stage", "greedy", "mcmc"]):
            Hallucination algorithm. Drives which iteration-count fields below
            are actually consumed (see each field's ``depends_on``). Upstream
            default: ``"4stage"``.
        use_multimer_design (bool): Use AF2 multimer parameters during hallucination.
            Every upstream preset uses multimer.
        omit_AAs (str): Amino acids to ban during design (no separator). Upstream default: ``"C"``.
        force_reject_AA (bool): Reject any design containing ``omit_AAs``.
        soft_iterations (int): Soft-stage iterations. Used by 2stage/3stage/4stage.
        temporary_iterations (int): Temporary-stage iterations. Used by 3stage/4stage.
        hard_iterations (int): Hard-stage iterations. Used by 3stage/4stage.
        greedy_iterations (int): Greedy/MCMC iterations. Used by 2stage/4stage/greedy/mcmc.
        greedy_percentage (float): Greedy/MCMC mutation rate as % of binder length.
        weights_plddt (float): pLDDT loss weight.
        weights_pae_intra (float): Intra-chain PAE loss weight.
        weights_pae_inter (float): Inter-chain (interface) PAE loss weight.
        weights_con_intra (float): Intra-chain contact loss weight.
        weights_con_inter (float): Inter-chain (interface) contact loss weight.
        weights_helicity (float): Helicity bias weight (negative discourages helices).
        weights_iptm (float): Interface pTM loss weight (only used when ``use_i_ptm_loss=True``).
        weights_rg (float): Radius-of-gyration loss weight (only used when ``use_rg_loss=True``).
        weights_termini_loss (float): N-/C-termini distance loss weight (only used when ``use_termini_distance_loss=True``).
        random_helicity (bool): Randomize the sign of ``weights_helicity`` per trajectory.
        use_i_ptm_loss (bool): Enable interface pTM loss.
        use_rg_loss (bool): Enable radius-of-gyration loss.
        use_termini_distance_loss (bool): Enable termini-distance loss.
        intra_contact_distance (float): Intra-chain contact distance cutoff (Å).
        inter_contact_distance (float): Inter-chain contact distance cutoff (Å).
        intra_contact_number (int): Number of intra-chain contacts per residue.
        inter_contact_number (int): Number of inter-chain contacts per residue.
        rm_template_seq_design (bool): Mask target template sequence during hallucination.
        rm_template_seq_predict (bool): Mask target template sequence during validation.
        rm_template_sc_design (bool): Mask target template side chains during hallucination.
        rm_template_sc_predict (bool): Mask target template side chains during validation.
        predict_initial_guess (bool): Use the trajectory structure as AF2's initial guess.
        predict_bigbang (bool): Use AF2's "Big Bang" recycle initialisation.
        enable_mpnn (bool): Run ProteinMPNN sequence refinement after each accepted trajectory.
            When False, the eight ``mpnn_*`` / ``num_seqs`` / ``max_mpnn_sequences`` /
            ``sampling_temp`` / ``backbone_noise`` / ``model_path`` knobs are inert.
        mpnn_fix_interface (bool): Fix interface residues during MPNN redesign.
        num_seqs (int): Number of MPNN sequences to sample per trajectory.
        max_mpnn_sequences (int): Max MPNN sequences to validate per trajectory.
        sampling_temp (float): MPNN sampling temperature (lower = more deterministic).
        backbone_noise (float): MPNN backbone noise.
        model_path (Literal["v_48_002", "v_48_010", "v_48_020", "v_48_030"]): MPNN model checkpoint name.
        mpnn_weights (Literal["original", "soluble"]): MPNN weight set.
        num_recycles_design (int): AF2 recycles during hallucination.
        num_recycles_validation (int): AF2 recycles during validation.
        optimise_beta (bool): 4stage-only — increase recycles + iterations
            mid-trajectory when the soft-stage output is beta-heavy.
        optimise_beta_extra_soft (int): Extra soft iterations for beta-heavy designs.
        optimise_beta_extra_temp (int): Extra temporary iterations for beta-heavy designs.
        optimise_beta_recycles_design (int): Recycles during hallucination for beta-heavy designs.
        optimise_beta_recycles_valid (int): Recycles during validation for beta-heavy designs.
        max_trajectories (int | bool): Max hallucination trajectories before stopping.
            ``False`` (upstream default) = unlimited; positive int = cap.
        enable_rejection_check (bool): Enable rolling acceptance-rate monitoring
            (stops the run if it stalls).
        acceptance_rate (float): Minimum design acceptance rate to keep running.
        start_monitoring (int): Trajectory count before acceptance-rate monitoring starts.
        filter_overrides (dict[str, Any]): Per-metric threshold overrides merged on top
            of the upstream default filters at dispatch time. Keys are upstream metric
            names (e.g. ``"Average_pLDDT"``); values are upstream filter dicts
            (e.g. ``{"threshold": 0.85, "higher": True}``).
    """

    design_algorithm: Literal["2stage", "3stage", "4stage", "greedy", "mcmc"] = ConfigField(
        title="Design Algorithm",
        default="4stage",
        description="Hallucination algorithm. 4stage (default) commits logits via soft→temporary→hard→greedy.",
    )
    use_multimer_design: bool = ConfigField(
        title="Use Multimer Design",
        default=True,
        description="Use AF2 multimer params during hallucination. Every upstream preset uses multimer.",
        advanced=True,
        reload_on_change=True,
    )
    omit_AAs: str = ConfigField(
        title="Omit Amino Acids",
        default="C",
        description="Comma-separated amino acids to ban during design (e.g. 'C' or 'C,W').",
        advanced=True,
    )
    force_reject_AA: bool = ConfigField(
        title="Force Reject AA",
        default=False,
        description="Drop any MPNN sequence containing residues from omit_AAs (hard reject, not a soft penalty).",
        advanced=True,
    )

    soft_iterations: int = ConfigField(
        title="Soft Iterations",
        default=75,
        ge=0,
        description="Soft-stage hallucination iterations. Used by 2stage / 3stage / 4stage.",
        advanced=True,
        depends_on={"design_algorithm": ["2stage", "3stage", "4stage"]},
    )
    temporary_iterations: int = ConfigField(
        title="Temporary Iterations",
        default=45,
        ge=0,
        description="Temporary-stage hallucination iterations. Used by 3stage / 4stage.",
        advanced=True,
        depends_on={"design_algorithm": ["3stage", "4stage"]},
    )
    hard_iterations: int = ConfigField(
        title="Hard Iterations",
        default=5,
        ge=0,
        description="Hard-stage hallucination iterations. Used by 3stage / 4stage.",
        advanced=True,
        depends_on={"design_algorithm": ["3stage", "4stage"]},
    )
    greedy_iterations: int = ConfigField(
        title="Greedy Iterations",
        default=15,
        ge=0,
        description="Greedy-stage iteration count. Used by 2stage / 4stage / greedy / mcmc (3stage doesn't use it).",
        advanced=True,
        depends_on={"design_algorithm": ["2stage", "4stage", "greedy", "mcmc"]},
    )
    greedy_percentage: float = ConfigField(
        title="Greedy Percentage",
        default=1.0,
        gt=0.0,
        le=100.0,
        description="Mutation rate for greedy/MCMC tries (% of binder length). Used by 2stage / 4stage / greedy / mcmc.",
        advanced=True,
        depends_on={"design_algorithm": ["2stage", "4stage", "greedy", "mcmc"]},
    )

    weights_plddt: float = ConfigField(
        title="pLDDT Loss Weight",
        default=0.1,
        description="Loss weight on AF2 pLDDT (higher = push for more confident-folding designs).",
        advanced=True,
    )
    weights_pae_intra: float = ConfigField(
        title="Intra-Chain PAE Weight",
        default=0.4,
        description="Loss weight on within-binder PAE (higher = push for cleaner internal geometry).",
        advanced=True,
    )
    weights_pae_inter: float = ConfigField(
        title="Inter-Chain PAE Weight",
        default=0.1,
        description="Loss weight on binder↔target interface PAE (higher = push for confident pairing).",
        advanced=True,
    )
    weights_con_intra: float = ConfigField(
        title="Intra-Chain Contact Weight",
        default=1.0,
        description="Loss weight on within-binder C-alpha contacts (encourages a compact fold).",
        advanced=True,
    )
    weights_con_inter: float = ConfigField(
        title="Inter-Chain Contact Weight",
        default=1.0,
        description="Loss weight on binder↔target interface contacts (encourages docking).",
        advanced=True,
    )
    weights_helicity: float = ConfigField(
        title="Helicity Weight",
        default=-0.3,
        description="Helicity bias weight (negative discourages helices, positive encourages them).",
        advanced=True,
    )
    weights_iptm: float = ConfigField(
        title="Interface pTM Weight",
        default=0.05,
        description="Loss weight on AF2 interface pTM (higher = push for higher iPTM).",
        advanced=True,
        depends_on={"use_i_ptm_loss": [True]},
    )
    weights_rg: float = ConfigField(
        title="Radius-of-Gyration Weight",
        default=0.3,
        description="Loss weight on binder radius of gyration (higher = compress toward a tighter binder).",
        advanced=True,
        depends_on={"use_rg_loss": [True]},
    )
    weights_termini_loss: float = ConfigField(
        title="Termini Distance Weight",
        default=0.1,
        description="Loss weight pulling binder N- and C-termini together (for cyclisable backbones).",
        advanced=True,
        depends_on={"use_termini_distance_loss": [True]},
    )

    random_helicity: bool = ConfigField(
        title="Random Helicity",
        default=False,
        description="Randomise the sign of weights_helicity per trajectory.",
        advanced=True,
    )
    use_i_ptm_loss: bool = ConfigField(
        title="Use Interface pTM Loss",
        default=True,
        description="Enable interface pTM loss term (weights_iptm). Pushes for higher iPTM during hallucination.",
        advanced=True,
    )
    use_rg_loss: bool = ConfigField(
        title="Use Radius-of-Gyration Loss",
        default=True,
        description="Enable radius-of-gyration loss.",
        advanced=True,
    )
    use_termini_distance_loss: bool = ConfigField(
        title="Use Termini Distance Loss",
        default=False,
        description="Enable N-/C-termini distance loss.",
        advanced=True,
    )

    intra_contact_distance: float = ConfigField(
        title="Intra-Chain Contact Distance",
        default=14.0,
        gt=0.0,
        description="Intra-chain contact distance cutoff (Å).",
        advanced=True,
    )
    inter_contact_distance: float = ConfigField(
        title="Inter-Chain Contact Distance",
        default=20.0,
        gt=0.0,
        description="Inter-chain (interface) contact distance cutoff (Å).",
        advanced=True,
    )
    intra_contact_number: int = ConfigField(
        title="Intra-Chain Contact Number",
        default=2,
        ge=1,
        description="Number of intra-chain contacts per residue.",
        advanced=True,
    )
    inter_contact_number: int = ConfigField(
        title="Inter-Chain Contact Number",
        default=2,
        ge=1,
        description="Number of inter-chain contacts per residue.",
        advanced=True,
    )

    rm_template_seq_design: bool = ConfigField(
        title="Mask Template Seq (Design)",
        default=False,
        description="Hide target sequence from AF2 template during hallucination (geometry-only target).",
        advanced=True,
    )
    rm_template_seq_predict: bool = ConfigField(
        title="Mask Template Seq (Predict)",
        default=False,
        description="Hide target sequence from AF2 template during validation (geometry-only target).",
        advanced=True,
    )
    rm_template_sc_design: bool = ConfigField(
        title="Mask Template SC (Design)",
        default=False,
        description="Hide target side-chain coordinates from AF2 template during hallucination.",
        advanced=True,
    )
    rm_template_sc_predict: bool = ConfigField(
        title="Mask Template SC (Predict)",
        default=False,
        description="Hide target side-chain coordinates from AF2 template during validation.",
        advanced=True,
    )
    predict_initial_guess: bool = ConfigField(
        title="Predict Initial Guess",
        default=False,
        description="Seed AF2 validation with the hallucinated complex coords (helps hard targets).",
        advanced=True,
    )
    predict_bigbang: bool = ConfigField(
        title="Predict Big Bang",
        default=False,
        description="Seed AF2 validation with the hallucinated atom positions (alternative to initial_guess).",
        advanced=True,
    )

    enable_mpnn: bool = ConfigField(
        title="Enable MPNN",
        default=True,
        description="Refine each hallucinated binder with ProteinMPNN before AF2 re-validation.",
        advanced=True,
    )
    mpnn_fix_interface: bool = ConfigField(
        title="MPNN Fix Interface",
        default=True,
        description="Fix interface residues during MPNN redesign.",
        advanced=True,
        depends_on={"enable_mpnn": [True]},
    )
    num_seqs: int = ConfigField(
        title="MPNN Samples per Traj",
        default=20,
        ge=1,
        description="MPNN sequences sampled per accepted trajectory before filtering.",
        advanced=True,
        depends_on={"enable_mpnn": [True]},
    )
    max_mpnn_sequences: int = ConfigField(
        title="MPNN Validated per Traj",
        default=2,
        ge=1,
        description="Top-scoring MPNN sequences (out of num_seqs) carried forward to AF2 validation.",
        advanced=True,
        depends_on={"enable_mpnn": [True]},
    )
    sampling_temp: float = ConfigField(
        title="MPNN Sampling Temperature",
        default=0.1,
        gt=0.0,
        description="MPNN sampling temperature (lower = more deterministic).",
        advanced=True,
        depends_on={"enable_mpnn": [True]},
    )
    backbone_noise: float = ConfigField(
        title="MPNN Backbone Noise",
        default=0.0,
        ge=0.0,
        description="Std-dev of Gaussian noise added to backbone coords before MPNN sampling (0 = none).",
        advanced=True,
        depends_on={"enable_mpnn": [True]},
    )
    model_path: Literal["v_48_002", "v_48_010", "v_48_020", "v_48_030"] = ConfigField(
        title="MPNN Checkpoint",
        default="v_48_020",
        description="MPNN checkpoint name. Higher trailing digits = trained with more backbone noise.",
        advanced=True,
        depends_on={"enable_mpnn": [True]},
    )
    mpnn_weights: Literal["original", "soluble"] = ConfigField(
        title="MPNN Weights",
        default="soluble",
        description="MPNN weight set. 'soluble' biases toward soluble residues; 'original' is the unbiased release.",
        advanced=True,
        depends_on={"enable_mpnn": [True]},
    )

    num_recycles_design: int = ConfigField(
        title="Recycles (Design)",
        default=1,
        ge=0,
        description="AlphaFold2 recycles during hallucination.",
        advanced=True,
    )
    num_recycles_validation: int = ConfigField(
        title="Recycles (Validation)",
        default=3,
        ge=0,
        description="AlphaFold2 recycles during validation.",
        advanced=True,
    )
    optimise_beta: bool = ConfigField(
        title="Optimise Beta Designs",
        default=True,
        description="When a trajectory looks β-heavy (>15% sheet), bump iterations + recycles to help it converge.",
        advanced=True,
    )
    optimise_beta_extra_soft: int = ConfigField(
        title="Beta: Extra Soft Iterations",
        default=0,
        ge=0,
        description="Extra soft iterations for beta-heavy designs (4stage only — added after soft stage).",
        advanced=True,
        depends_on={"optimise_beta": [True]},
    )
    optimise_beta_extra_temp: int = ConfigField(
        title="Beta: Extra Temp Iterations",
        default=0,
        ge=0,
        description="Extra temporary iterations for beta-heavy designs (4stage only).",
        advanced=True,
        depends_on={"optimise_beta": [True]},
    )
    optimise_beta_recycles_design: int = ConfigField(
        title="Beta: Recycles (Design)",
        default=3,
        ge=0,
        description="AF2 recycles during hallucination for beta-heavy designs.",
        advanced=True,
        depends_on={"optimise_beta": [True]},
    )
    optimise_beta_recycles_valid: int = ConfigField(
        title="Beta: Recycles (Validation)",
        default=3,
        ge=0,
        description="AF2 recycles during validation for beta-heavy designs.",
        advanced=True,
        depends_on={"optimise_beta": [True]},
    )

    max_trajectories: int | bool = ConfigField(
        title="Max Trajectories",
        default=False,
        description="Max hallucination trajectories before stopping. False = unlimited; positive int = cap.",
        include_in_key=False,
    )
    enable_rejection_check: bool = ConfigField(
        title="Enable Rejection Check",
        default=True,
        description="Enable rolling acceptance-rate monitoring (stops the run if it stalls).",
        advanced=True,
        include_in_key=False,
    )
    acceptance_rate: float = ConfigField(
        title="Acceptance Rate",
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Minimum design acceptance rate to keep running.",
        advanced=True,
        include_in_key=False,
        depends_on={"enable_rejection_check": [True]},
    )
    start_monitoring: int = ConfigField(
        title="Monitor Start (Trajectories)",
        default=600,
        ge=0,
        description="Trajectory count before acceptance-rate monitoring starts.",
        advanced=True,
        include_in_key=False,
        depends_on={"enable_rejection_check": [True]},
    )

    filter_overrides: dict[str, Any] = ConfigField(
        title="Filter Overrides",
        default_factory=dict,
        description="Per-metric threshold overrides merged on top of upstream default_filters.json.",
        advanced=True,
    )

    @model_validator(mode="after")
    def _validate_max_trajectories(self) -> Self:
        """``max_trajectories`` must be ``False`` (unlimited) or a positive int (``True`` coerces to 1)."""
        mt = self.max_trajectories
        if mt is False:
            return self
        if mt < 1:
            raise ValueError(f"max_trajectories must be False or a positive int, got {mt!r}.")
        return self

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the tool on.",
        hidden=True,
        include_in_key=False,
    )

    @property
    def devices_per_instance(self) -> int:
        """BindCraft uses one GPU per design run (sequential trajectories)."""
        return 1


# ============================================================================
# Output
# ============================================================================


class BindCraftOutput(BaseToolOutput):
    """Output from a BindCraft binder-design pipeline run.

    Attributes:
        designs (list[BindCraftDesign]): Accepted binder designs (length is at most
            ``BindCraftInput.number_of_final_designs``).
        n_trajectories_run (int): Total trajectories attempted before stopping
            (success or hitting ``max_trajectories``).
        n_designs_accepted (int): Designs that passed all filters (equals
            ``len(designs)``).
    """

    designs: list[BindCraftDesign] = Field(
        default_factory=list,
        description="Accepted binder designs.",
    )
    n_trajectories_run: int = Field(
        default=0,
        ge=0,
        description="Total trajectories attempted before stopping.",
    )
    n_designs_accepted: int = Field(
        default=0,
        ge=0,
        description="Designs that passed all filters (equals len(designs)).",
    )

    def __len__(self) -> int:
        """Number of accepted designs."""
        return len(self.designs)

    def __getitem__(self, index: int) -> BindCraftDesign:
        """Get an accepted design by index."""
        return self.designs[index]

    def __iter__(self) -> Iterator[BindCraftDesign]:  # type: ignore[override]
        """Iterate over accepted designs."""
        return iter(self.designs)

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["pdb", "json"]

    @property
    def output_format_default(self) -> str:
        """Default export format — directory of PDBs + a stats CSV."""
        return "pdb"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Write designs to disk.

        - ``"pdb"``: directory containing one PDB per accepted design plus a
          ``stats.json`` summary of metrics.
        - ``"json"``: single JSON file with the full Pydantic dump (PDB strings inline).

        Args:
            export_path (str | Path): Output path (directory for ``pdb``, file for ``json``).
            file_format (str): One of ``"pdb"`` or ``"json"``.
        """
        if file_format not in self.output_format_options:
            raise ValueError(f"Unsupported format: {file_format}")
        path = Path(export_path)
        if file_format == "pdb":
            path.mkdir(parents=True, exist_ok=True)
            for design in self.designs:
                design.structure.write_pdb(path / f"{design.design_name}.pdb")
            stats_rows = [
                {
                    "design_name": d.design_name,
                    "binder_sequence": d.binder_sequence,
                    "seed": d.seed,
                    **dict(d.metrics.items()),
                }
                for d in self.designs
            ]
            (path / "stats.json").write_text(json.dumps(stats_rows, indent=2))
        else:
            path.write_text(self.model_dump_json(indent=2))


# ============================================================================
# Tool registration
# ============================================================================


def example_input() -> BindCraftInput:
    """Minimal valid input — small target with a single hotspot for one-off sampling."""
    return BindCraftInput(
        target_pdb=str(_BINDCRAFT_FIXTURE_PDB),
        target_chain="A",
        target_hotspot_residues="56",
        binder_lengths=(60, 70),
        binder_name="example",
        number_of_final_designs=1,
    )


@tool(
    key="bindcraft-design",
    label="BindCraft Binder Design",
    category="structure_design",
    input_class=BindCraftInput,
    config_class=BindCraftConfig,
    output_class=BindCraftOutput,
    description=(
        "End-to-end binder design pipeline: AlphaFold2 hallucination + ProteinMPNN refinement "
        "+ AlphaFold2 validation + PyRosetta filtering. Returns accepted binders against a target."
    ),
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
)
def run_bindcraft_design(
    inputs: BindCraftInput,
    config: BindCraftConfig,
    instance: Any = None,
) -> BindCraftOutput:
    """Run the BindCraft binder-design pipeline against a target.

    Args:
        inputs (BindCraftInput): Target + binder spec.
        config (BindCraftConfig): Advanced settings + ``filter_overrides``.
        instance (Any): Optional ``ToolInstance`` for persistent execution.

    Returns:
        BindCraftOutput: Accepted binder designs and run-level counters.
    """
    logger.debug("Dispatching BindCraft design pipeline (target_chain=%s)", inputs.target_chain)

    payload: dict[str, Any] = {
        "operation": "design",
        "target_pdb": inputs.target_pdb,
        "target_chain": inputs.target_chain,
        "target_hotspot_residues": inputs.target_hotspot_residues,
        "binder_lengths": list(inputs.binder_lengths),
        "binder_name": inputs.binder_name,
        "number_of_final_designs": inputs.number_of_final_designs,
        "advanced_settings": {
            **{key: getattr(config, key) for key in _USER_FACING_UPSTREAM_KEYS},
            **_HARDCODED_INTERNAL_SETTINGS,
        },
        "filter_overrides": config.filter_overrides,
        "seed": config.seed,
        "device": config.device,
        "verbose": config.verbose,
    }

    result = ToolInstance.dispatch("bindcraft", payload, instance=instance, config=config)

    designs: list[BindCraftDesign] = []
    for raw in result["designs"]:
        metrics = BindCraftMetrics(**raw["metrics"])
        designs.append(
            BindCraftDesign(
                design_name=raw["design_name"],
                binder_sequence=raw["binder_sequence"],
                structure=Structure(
                    structure=raw["pdb"],
                    b_factor_type=BFactorType.PLDDT,
                    metrics=metrics,
                    source="bindcraft-design",
                ),
                metrics=metrics,
                seed=raw["seed"],
                interface_aas=raw.get("interface_aas", {}),
                interface_residues=raw.get("interface_residues", []),
            )
        )

    return BindCraftOutput(
        designs=designs,
        n_trajectories_run=result["n_trajectories_run"],
        n_designs_accepted=result["n_designs_accepted"],
    )

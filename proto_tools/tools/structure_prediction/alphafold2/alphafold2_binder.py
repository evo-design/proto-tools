"""AlphaFold2/ColabDesign binder-design tool.

Binder design against a frozen target template. Backends: ``"base"`` (upstream
ColabDesign) or ``"germinal"`` (fork with alpha=2.0 logit scaling, persistent
bias, framework contact penalties, and extension losses: rg, i_ptm, helix,
beta_strand, NC). ``compute_gradient=True`` (default) runs forward+backward
and returns the gradient w.r.t. logits; ``False`` runs forward only with
``gradient=None``. Loss, metrics, and Structure are identical in both modes.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import AlphaFold2Metrics
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    ConfigField,
    GradientInput,
    GradientOutput,
    InputField,
    ToolInstance,
    one_hot_protein_logits,
)

logger = logging.getLogger(__name__)

_VALID_LOSS_KEYS = frozenset(
    {
        "plddt",
        "i_plddt",
        "pae",
        "i_pae",
        "con",
        "i_con",
        "exp_res",
        "rmsd",
        "dgram_cce",
        "fape",
        "rg",
        "i_ptm",
        "NC",
        "helix",
        "beta_strand",
    }
)

_BINDER_FIXTURE_PDB = Path(__file__).resolve().parents[4] / "tests" / "dummy_data" / "pdl1.pdb"


class AlphaFold2BinderInput(GradientInput):
    """Input for AlphaFold2 binder-design (forward scoring or gradient).

    Extends GradientInput with target-template structural data required
    for binder redesign against a frozen target.

    Attributes:
        logits (list[list[float]]): Inherited — relaxed sequence logits (L x 20).
        temperature (float): Inherited — softmax temperature.
        target_pdb (str): Target+binder template PDB (file path or PDB-format string).
        target_chain (str): Chain ID(s) of the frozen target in the PDB.
        target_hotspot (str | None): Comma-separated hotspot residue indices on the target.
        binder_chain (str): Binder chain ID in the template PDB.
        design_positions (list[int] | None): Zero-based binder residue indices
            for loss focus (e.g. CDR loops). Germinal backend only.
    """

    target_pdb: str = InputField(
        description="Target+binder template PDB (file path or PDB-format string).",
    )
    target_chain: str = InputField(
        default="A",
        description="Chain ID(s) of the frozen target in the PDB.",
    )
    target_hotspot: str | None = InputField(
        default=None,
        description="Comma-separated hotspot residue indices on the target (e.g. '10,25,42').",
    )
    binder_chain: str = InputField(
        default="H",
        description="Binder chain ID for template-based binder redesign.",
    )
    design_positions: list[int] | None = InputField(
        default=None,
        description="Zero-based binder residue indices for loss focus (e.g. CDR loops).",
    )


class AlphaFold2BinderConfig(BaseConfig):
    """Configuration for AlphaFold2/ColabDesign binder design (forward or backward).

    Binder protocol only — designs a binder against a frozen target structure.
    Set ``backend="germinal"`` to enable Germinal-specific features (alpha=2.0
    logit scaling, persistent bias, framework contact penalties, extension
    loss callbacks, and CDR position-aware losses).

    Attributes:
        bias_redesign (float | None): Persistent softmax bias toward wildtype at
            non-design positions. Germinal backend only.
        omit_aas (str | None): Amino acids to ban (e.g. ``"C,W"``).
        num_recycles (int): AF2 recycling iterations.
        recycle_mode (Literal["last", "sample", "average", "first"]): Which recycle's
            output is used for loss/gradient. ``"last"`` matches Germinal's VHH default;
            ``"average"`` averages across recycles; ``"sample"`` picks one uniformly;
            ``"first"`` uses only recycle 0.
        model_num (int): AF2 parameter set (1-5).
        sample_models (bool): Randomly sample model sets each forward pass.
        soft (float): ColabDesign softmax blending (0=raw logits, 1=full softmax).
            Passed per-step by the gradient optimizer.
        hard (float): ColabDesign hard-sequence blending (0=relaxed, 1=straight-through argmax).
        backend (Literal["base", "germinal"]): ``"base"`` (upstream ColabDesign) or ``"germinal"``
            (Germinal fork with alpha, bias, framework contacts, extension losses).
        starting_binder_seq (str | None): Optional one-letter AA string used to seed
            the binder before gradient updates. Germinal backend only; length must
            equal ``len(logits)``.
        loss_weights (dict[str, float]): Binder-objective weights. Base keys:
            plddt, i_plddt, pae, i_pae, con, i_con, exp_res, rmsd, dgram_cce,
            fape. Germinal extension keys: rg, i_ptm, NC, helix, beta_strand.
        intra_contact_num (int): Intra-chain contacts per residue. Germinal only.
        intra_contact_cutoff (float): Intra-chain distance cutoff (Å). Germinal only.
        inter_contact_num (int): Interface contacts per residue. Germinal only.
        inter_contact_cutoff (float): Interface distance cutoff (Å). Germinal only.
        compute_gradient (bool): Run backward pass and return gradient; ``False``
            for forward-only scoring (returns ``gradient=None``).
    """

    bias_redesign: float | None = ConfigField(
        title="Bias Redesign",
        default=None,
        gt=0.0,
        description="Soft bias strength for non-design positions toward the wildtype template.",
        advanced=True,
    )
    omit_aas: str | None = ConfigField(
        title="Omit Amino Acids",
        default=None,
        description="Comma-separated amino acids to ban during optimization, e.g. 'C,W'.",
        advanced=True,
    )
    num_recycles: int = ConfigField(
        title="Number of Recycles",
        default=3,
        ge=0,
        le=48,
        description="Number of recycling iterations (higher=more refined but slower).",
        advanced=True,
    )
    # No reload_on_change: inference.py:_get_model caches one AF2 model per
    # recycle_mode in self._models, so mode switches reuse the persistent
    # worker instead of triggering a subprocess restart + JAX re-warmup.
    recycle_mode: Literal["last", "sample", "average", "first"] = ConfigField(
        title="Recycle Mode",
        default="last",
        description="Which recycle's output is used for loss/gradient ('last' matches Germinal VHH).",
        advanced=True,
    )
    model_num: int = ConfigField(
        title="Model Number",
        default=1,
        ge=1,
        le=5,
        description="Which AlphaFold2 model parameter set to use (1-5).",
        advanced=True,
        reload_on_change=True,
    )
    sample_models: bool = ConfigField(
        title="Sample Models",
        default=False,
        description="Randomly sample from available AF2 model parameter sets each forward pass.",
        advanced=True,
        include_in_key=False,
    )
    soft: float = ConfigField(
        title="Soft Mixing",
        default=1.0,
        ge=0.0,
        le=1.0,
        description="ColabDesign soft mixing coefficient (0=hard logits, 1=full softmax blend).",
        advanced=True,
    )
    hard: float = ConfigField(
        title="Hard Mixing",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="ColabDesign hard mixing coefficient (0=relaxed, 1=straight-through argmax).",
        advanced=True,
    )
    backend: Literal["base", "germinal"] = ConfigField(
        title="Backend",
        default="base",
        description="ColabDesign backend: 'base' (upstream) or 'germinal' (with alpha, bias, framework contacts).",
        advanced=True,
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only scoring.",
        advanced=True,
    )
    starting_binder_seq: str | None = ConfigField(
        title="Starting Binder Sequence",
        default=None,
        description="Warm-start binder AA sequence (Germinal backend only; length must equal len(logits)).",
        advanced=True,
    )
    loss_weights: dict[str, float] = ConfigField(
        title="Loss Weights",
        default_factory=dict,
        description="Binder-objective weights passed to ColabDesign's set_weights().",
        advanced=True,
    )
    intra_contact_num: int = ConfigField(
        title="Intra Contact Number",
        default=2,
        ge=1,
        description="Number of intra-molecular contacts per residue for the contact loss.",
        advanced=True,
    )
    intra_contact_cutoff: float = ConfigField(
        title="Intra Contact Cutoff",
        default=14.0,
        gt=0.0,
        description="Distance cutoff in angstroms for intra-molecular contacts.",
        advanced=True,
    )
    inter_contact_num: int = ConfigField(
        title="Inter Contact Number",
        default=10,
        ge=1,
        description="Number of inter-molecular (interface) contacts per residue.",
        advanced=True,
    )
    inter_contact_cutoff: float = ConfigField(
        title="Inter Contact Cutoff",
        default=20.0,
        gt=0.0,
        description="Distance cutoff in angstroms for inter-molecular contacts.",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on.",
        hidden=True,
        include_in_key=False,
    )

    @model_validator(mode="after")
    def validate_design_surface(self) -> Self:
        """Validate binder redesign settings."""
        if any(weight < 0 for weight in self.loss_weights.values()):
            raise ValueError("loss_weights must be non-negative.")
        unknown_keys = set(self.loss_weights) - _VALID_LOSS_KEYS
        if unknown_keys:
            raise ValueError(f"Unknown loss_weights keys: {unknown_keys}. Valid keys: {sorted(_VALID_LOSS_KEYS)}")
        if self.starting_binder_seq is not None and self.backend != "germinal":
            raise ValueError("starting_binder_seq requires backend='germinal'.")
        return self


class AlphaFold2BinderOutput(GradientOutput):
    """Binder-design output: loss, metrics, Structure, and optionally the gradient.

    Attributes:
        gradient (list[list[float]] | None): Gradient matrix matching the input logits shape,
            or ``None`` when ``compute_gradient=False``.
        loss (float): Scalar objective value.
        metrics (dict[str, Any]): Scalar auxiliary metrics (avg_plddt, ptm, iptm, avg_pae,
            plus per-loss values for every weighted ColabDesign loss term).
        vocab (list[str]): Amino-acid column ordering.
        structure (Structure): Predicted target+binder complex from the forward pass.
            B-factors are at the raw 0-100 PDB scale; ``b_factor_type=PLDDT`` means
            ``Structure.per_residue_plddt`` normalizes them to ``[0, 1]``.
    """

    # Narrow to Optional for compute_gradient=False mode; only GradientOutput subclass that does this.
    gradient: list[list[float]] | None = Field(  # type: ignore[assignment]
        default=None,
        description="Gradient w.r.t. input logits. None when compute_gradient=False.",
    )
    structure: Structure = Field(description="Predicted target+binder complex with per-residue pLDDT in B-factors.")

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Write the gradient bundle as JSON alongside the structure as a PDB sidecar."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        # Suffix-additive so dotted names (e.g. "step_v1.5") aren't truncated.
        base = Path(export_path)
        pdb_path = base.parent / f"{base.name}.pdb"
        json_path = base.parent / f"{base.name}.json"
        self.structure.write_pdb(pdb_path)
        payload = self.model_dump(include={"gradient", "loss", "metrics", "vocab"}) | {"structure_pdb": pdb_path.name}
        json_path.write_text(json.dumps(payload, indent=2))


def example_input() -> AlphaFold2BinderInput:
    """Minimal valid input — short VHH-like binder with biased logits and PD-L1 template."""
    return AlphaFold2BinderInput(
        logits=one_hot_protein_logits("EVQLVESG", sharpness=2.0),
        target_pdb=str(_BINDER_FIXTURE_PDB),
        binder_chain="B",
    )


@tool(
    key="alphafold2-binder",
    label="AlphaFold2 Binder",
    category="structure_prediction",
    input_class=AlphaFold2BinderInput,
    config_class=AlphaFold2BinderConfig,
    output_class=AlphaFold2BinderOutput,
    description="AF2 binder design against a fixed target: loss + predicted Structure, optionally the gradient w.r.t. logits (base or Germinal backend)",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
)
def run_alphafold2_binder(
    inputs: AlphaFold2BinderInput,
    config: AlphaFold2BinderConfig,
    instance: Any = None,
) -> AlphaFold2BinderOutput:
    """Run one AlphaFold2/ColabDesign binder-design step.

    ``compute_gradient=False`` runs forward only (gradient=None); loss, metrics,
    and Structure are identical to gradient mode.
    """
    logger.debug(
        "Running AlphaFold2 binder design: model=%d, compute_gradient=%s",
        config.model_num,
        config.compute_gradient,
    )
    result = ToolInstance.dispatch(
        "alphafold2",
        {
            "operation": "compute_gradient",
            "logits": inputs.logits,
            "temperature": inputs.temperature,
            "soft": config.soft,
            "hard": config.hard,
            "target_pdb": inputs.target_pdb,
            "target_chain": inputs.target_chain,
            "target_hotspot": inputs.target_hotspot,
            "binder_chain": inputs.binder_chain,
            "design_positions": inputs.design_positions,
            "bias_redesign": config.bias_redesign,
            "omit_aas": config.omit_aas,
            "num_recycles": config.num_recycles,
            "recycle_mode": config.recycle_mode,
            "model_num": config.model_num,
            "sample_models": config.sample_models,
            "starting_binder_seq": config.starting_binder_seq,
            "loss_weights": config.loss_weights,
            "intra_contact_num": config.intra_contact_num,
            "intra_contact_cutoff": config.intra_contact_cutoff,
            "inter_contact_num": config.inter_contact_num,
            "inter_contact_cutoff": config.inter_contact_cutoff,
            "seed": config.seed,
            "backend": config.backend,
            "compute_gradient": config.compute_gradient,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )

    metrics = result["metrics"]
    return AlphaFold2BinderOutput(
        gradient=result["gradient"],
        loss=result["loss"],
        metrics=metrics,
        vocab=result["vocab"],
        structure=Structure(
            structure=result["pdb"],
            b_factor_type=BFactorType.PLDDT,
            metrics=AlphaFold2Metrics(
                avg_plddt=metrics["avg_plddt"],
                ptm=metrics["ptm"],
                iptm=metrics.get("iptm"),
                avg_pae=metrics["avg_pae"],
            ),
            source="alphafold2-binder",
        ),
    )

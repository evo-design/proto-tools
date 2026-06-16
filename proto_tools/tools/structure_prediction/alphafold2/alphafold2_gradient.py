"""Differentiable AlphaFold2/ColabDesign binder-scoring tool.

Scores a binder against a frozen target template. Backends: ``"base"`` (upstream
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
    AminoAcid,
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

_BINDER_FIXTURE_PDB = Path(__file__).resolve().parent / "example_input_fixture.pdb"


class AlphaFold2GradientInput(GradientInput):
    """Input for AlphaFold2 binder-design (forward scoring or gradient).

    Extends GradientInput with target-template structural data required
    for binder redesign against a frozen target.

    Attributes:
        logits (list[list[float]]): Inherited — relaxed sequence logits (L x 20).
        temperature (float): Inherited — softmax temperature.
        target_pdb (Structure): Target+binder template PDB. Accepts a file path,
            raw PDB/CIF content string, ``Structure`` object, or a dict in the
            shape produced by ``Structure.model_dump(mode='json')``.
        target_chain (str): Chain ID(s) of the frozen target in the PDB.
        target_hotspot (str | None): Comma-separated hotspot residue indices on the target.
        binder_chain (str | None): Binder template chain to redesign; None (default) designs de novo.
        design_positions (list[int] | None): Zero-based binder residue indices
            for loss focus (e.g. CDR loops). Germinal backend only.
    """

    target_pdb: Structure = InputField(
        title="Target PDB",
        description="Target+binder template PDB.",
    )
    target_chain: str = InputField(
        default="A",
        title="Target Chain",
        description="Chain ID(s) of the frozen target in the PDB.",
    )
    target_hotspot: str | None = InputField(
        default=None,
        title="Target Hotspot",
        description="Comma-separated hotspot residue indices on the target (e.g. '10,25,42').",
    )
    binder_chain: str | None = InputField(
        default=None,
        title="Binder Chain",
        description="Binder template chain to redesign; None (default) designs the binder de novo.",
    )
    design_positions: list[int] | None = InputField(
        default=None,
        title="Design Positions",
        description="Zero-based binder residue indices for loss focus (e.g. CDR loops).",
    )

    @model_validator(mode="after")
    def validate_chains_present(self) -> Self:
        """Validate target/binder chains exist in target_pdb."""
        # Absent chain -> empty PDB filter -> ColabDesign "Found 0 models".
        present = set(self.target_pdb.get_chain_ids())
        for part in (c.strip() for c in self.target_chain.split(",")):
            if part and part not in present:
                raise ValueError(
                    f"target_chain={part!r} is not a polymer chain in target_pdb (available: {sorted(present)})."
                )
        if self.binder_chain is not None and self.binder_chain not in present:
            raise ValueError(
                f"binder_chain={self.binder_chain!r} is not a polymer chain "
                f"in target_pdb (available: {sorted(present)}). "
                "For de novo binder design set binder_chain=None; for "
                "template redesign supply a PDB containing that chain."
            )
        return self


class AlphaFold2GradientConfig(BaseConfig):
    """Configuration for AlphaFold2/ColabDesign binder design (forward or backward).

    Binder protocol only — designs a binder against a frozen target structure.
    Set ``backend="germinal"`` to enable Germinal-specific features (alpha=2.0
    logit scaling, persistent bias, framework contact penalties, extension
    loss callbacks, and CDR position-aware losses).

    Attributes:
        include_pae_matrix (bool): Attach full per-residue PAE matrix. Default: ``False``.
        bias_redesign (float | None): Persistent softmax bias toward wildtype at
            non-design positions. Germinal backend only.
        omit_aas (list[AminoAcid] | None): Amino acids to ban (e.g. ``["C", "W"]``).
        num_recycles (int): AF2 recycling iterations.
        recycle_mode (Literal["last", "sample", "average", "first"]): Which recycle's
            output is used for loss/gradient. ``"last"`` matches Germinal's VHH default;
            ``"average"`` averages across recycles; ``"sample"`` picks one uniformly;
            ``"first"`` uses only recycle 0.
        model_num (int): AF2 parameter set (1-5).
        sample_models (bool): Randomly sample model sets each forward pass.
        use_multimer (bool): Use AlphaFold multimer parameters for binder protocol.
        rm_target_seq (bool): Mask target template sequence in ``prep_inputs``.
        rm_target_sc (bool): Mask target template side chains in ``prep_inputs``.
        rm_template_ic (bool): Mask inter-chain template contacts in ``prep_inputs``.
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
        intra_contact_num (int): Intra-chain contacts per residue.
        intra_contact_cutoff (float): Intra-chain distance cutoff (Å).
        inter_contact_num (int): Interface contacts per residue.
        inter_contact_cutoff (float): Interface distance cutoff (Å).
        framework_contact_offset (float): Framework contact penalty offset in the
            Germinal inter-chain contact loss. Germinal backend only.
        compute_gradient (bool): Run backward pass and return gradient; ``False``
            for forward-only scoring (returns ``gradient=None``).
    """

    include_pae_matrix: bool = ConfigField(
        title="Include PAE Matrix",
        default=False,
        description="Attach the full per-residue PAE matrix.",
    )
    bias_redesign: float | None = ConfigField(
        title="Bias Redesign",
        default=None,
        gt=0.0,
        description="Soft bias strength for non-design positions toward the wildtype template.",
    )
    omit_aas: list[AminoAcid] | None = ConfigField(
        title="Omit Amino Acids",
        default=None,
        description="Amino acids to ban during optimization (e.g. ['C', 'W']).",
        examples=[["C"], ["C", "W"]],
    )
    num_recycles: int = ConfigField(
        title="Number of Recycles",
        default=3,
        ge=0,
        description="Recycling iterations through the model. Higher = more accurate but slower.",
    )
    # No reload_on_change: inference.py:_get_model caches one AF2 model per
    # recycle_mode in self._models, so mode switches reuse the persistent
    # worker instead of triggering a subprocess restart + JAX re-warmup.
    recycle_mode: Literal["last", "sample", "average", "first"] = ConfigField(
        title="Recycle Mode",
        default="last",
        description="Which recycle's output is used for loss/gradient ('last' matches Germinal VHH).",
    )
    model_num: int = ConfigField(
        title="Model Number",
        default=1,
        ge=1,
        le=5,
        description="Which AlphaFold2 model parameter set to use (1-5).",
        reload_on_change=True,
    )
    sample_models: bool = ConfigField(
        title="Sample Models",
        default=False,
        description="Randomly sample from available AF2 model parameter sets each forward pass.",
        include_in_key=False,
    )
    use_multimer: bool = ConfigField(
        title="Use Multimer",
        default=True,
        description="Use AlphaFold multimer parameters for binder protocol.",
        reload_on_change=True,
    )
    rm_target_seq: bool = ConfigField(
        title="Mask Target Sequence",
        default=True,
        description="Mask target template sequence in ColabDesign prep_inputs.",
    )
    rm_target_sc: bool = ConfigField(
        title="Mask Target Side Chains",
        default=False,
        description="Mask target template side chains in ColabDesign prep_inputs.",
    )
    rm_template_ic: bool = ConfigField(
        title="Mask Inter-chain Contacts",
        default=True,
        description="Mask inter-chain template contacts in ColabDesign prep_inputs.",
    )
    soft: float = ConfigField(
        title="Soft Mixing",
        default=1.0,
        ge=0.0,
        le=1.0,
        description="ColabDesign soft mixing coefficient (0=hard logits, 1=full softmax blend).",
    )
    hard: float = ConfigField(
        title="Hard Mixing",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="ColabDesign hard mixing coefficient (0=relaxed, 1=straight-through argmax).",
    )
    backend: Literal["base", "germinal"] = ConfigField(
        title="Backend",
        default="base",
        description="ColabDesign backend: 'base' (upstream) or 'germinal' (with alpha, bias, framework contacts).",
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only scoring.",
    )
    starting_binder_seq: str | None = ConfigField(
        title="Starting Binder Sequence",
        default=None,
        description="Warm-start binder AA sequence (Germinal backend only; length must equal len(logits)).",
    )
    loss_weights: dict[str, float] = ConfigField(
        title="Loss Weights",
        default_factory=dict,
        description="Binder-objective weights passed to ColabDesign's set_weights().",
    )
    # Contact loss defaults match ColabDesign (mk_af_model.__init__). Override per-pipeline.
    intra_contact_num: int = ConfigField(
        title="Intra Contact Number",
        default=2,  # ColabDesign default
        ge=1,
        description="Number of intra-molecular contacts per residue for the contact loss.",
    )
    intra_contact_cutoff: float = ConfigField(
        title="Intra Contact Cutoff",
        default=14.0,  # ColabDesign default
        gt=0.0,
        description="Distance cutoff in angstroms for intra-molecular contacts.",
    )
    inter_contact_num: int = ConfigField(
        title="Inter Contact Number",
        default=1,  # ColabDesign default (Germinal uses 10, BindCraft uses 2)
        ge=1,
        description="Number of inter-molecular (interface) contacts per residue.",
    )
    inter_contact_cutoff: float = ConfigField(
        title="Inter Contact Cutoff",
        default=21.6875,  # ColabDesign default (Germinal/BindCraft use 20.0)
        gt=0.0,
        description="Distance cutoff in angstroms for inter-molecular contacts.",
    )
    framework_contact_offset: float = ConfigField(
        title="Framework Contact Offset",
        default=1.0,
        gt=0.0,
        description="Penalty offset for framework contacts in the Germinal inter-chain contact loss.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (GPU-required tool overrides BaseConfig 'cpu' default)",
        include_in_key=False,
    )

    @model_validator(mode="after")
    def validate_design_surface(self) -> Self:
        """Validate binder redesign settings."""
        unknown_keys = set(self.loss_weights) - _VALID_LOSS_KEYS
        if unknown_keys:
            raise ValueError(f"Unknown loss_weights keys: {unknown_keys}. Valid keys: {sorted(_VALID_LOSS_KEYS)}")
        if self.starting_binder_seq is not None and self.backend != "germinal":
            raise ValueError("starting_binder_seq requires backend='germinal'.")
        return self


class AlphaFold2GradientOutput(GradientOutput):
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

    gradient: list[list[float]] | None = Field(
        default=None,
        title="Gradient",
        description="Gradient w.r.t. input logits. None when compute_gradient=False.",
    )
    structure: Structure = Field(
        title="Predicted Structure",
        description="Predicted target+binder complex with per-residue pLDDT in B-factors.",
    )

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


def example_input() -> AlphaFold2GradientInput:
    """Minimal valid input — short VHH-like binder with biased logits and PD-L1 template."""
    return AlphaFold2GradientInput(
        logits=one_hot_protein_logits("EVQLVESG", sharpness=2.0),
        target_pdb=Structure.from_file(_BINDER_FIXTURE_PDB),
        binder_chain="B",
    )


@tool(
    key="alphafold2-gradient",
    label="AlphaFold2 Gradient",
    category="structure_prediction",
    input_class=AlphaFold2GradientInput,
    config_class=AlphaFold2GradientConfig,
    output_class=AlphaFold2GradientOutput,
    metrics_class=AlphaFold2Metrics,
    description=(
        "Differentiable AlphaFold2 scoring of a binder against a fixed target. "
        "Returns loss, Structure, and optionally the gradient w.r.t. input logits."
    ),
    uses_gpu=True,
    pin_visible_devices=True,
    example_input=example_input,
    cacheable=False,
    stochastic=True,
)
def run_alphafold2_gradient(
    inputs: AlphaFold2GradientInput,
    config: AlphaFold2GradientConfig,
    instance: Any = None,
) -> AlphaFold2GradientOutput:
    """Run one AlphaFold2/ColabDesign binder-design step.

    ``compute_gradient=False`` runs forward only (gradient=None); loss, metrics,
    and Structure are identical to gradient mode.
    """
    logger.debug(
        "Running AlphaFold2 binder design: model=%d, compute_gradient=%s",
        config.model_num,
        config.compute_gradient,
    )
    # Materialize the target Structure to a tempfile on the local filesystem so
    # the standalone (ColabDesign) can read it as a path. Auto-cleans on exit.
    with inputs.target_pdb.temp_file() as target_pdb_path:
        result = ToolInstance.dispatch(
            "alphafold2",
            {
                "operation": "compute_gradient",
                "logits": inputs.logits,
                "temperature": inputs.temperature,
                "soft": config.soft,
                "hard": config.hard,
                "target_pdb": str(target_pdb_path),
                "target_chain": inputs.target_chain,
                "target_hotspot": inputs.target_hotspot,
                "binder_chain": inputs.binder_chain,
                "design_positions": inputs.design_positions,
                "bias_redesign": config.bias_redesign,
                # ColabDesign's prep_inputs(rm_aa=...) expects a comma-separated string.
                "omit_aas": ",".join(config.omit_aas) if config.omit_aas else None,
                "num_recycles": config.num_recycles,
                "recycle_mode": config.recycle_mode,
                "model_num": config.model_num,
                "sample_models": config.sample_models,
                "use_multimer": config.use_multimer,
                "rm_target_seq": config.rm_target_seq,
                "rm_target_sc": config.rm_target_sc,
                "rm_template_ic": config.rm_template_ic,
                "starting_binder_seq": config.starting_binder_seq,
                "loss_weights": config.loss_weights,
                "intra_contact_num": config.intra_contact_num,
                "intra_contact_cutoff": config.intra_contact_cutoff,
                "inter_contact_num": config.inter_contact_num,
                "inter_contact_cutoff": config.inter_contact_cutoff,
                "framework_contact_offset": config.framework_contact_offset,
                "seed": config.seed,
                "include_pae_matrix": config.include_pae_matrix,
                "backend": config.backend,
                "compute_gradient": config.compute_gradient,
                "device": config.device,
                "verbose": config.verbose,
            },
            instance=instance,
            config=config,
        )

    metrics = result["metrics"]
    return AlphaFold2GradientOutput(
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
                pae=metrics.get("pae"),
            ),
            source="alphafold2-gradient",
        ),
    )

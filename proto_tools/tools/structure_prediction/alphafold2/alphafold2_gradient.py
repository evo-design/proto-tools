"""AlphaFold2/ColabDesign binder-design gradient tool.

Computes structure-prediction gradients for binder redesign against a frozen
target. Supports two ColabDesign backends: ``"base"`` (upstream) for generic
use, and ``"germinal"`` for Germinal-faithful antibody optimization with
alpha=2.0 logit scaling, persistent bias, framework contact penalties, and
extension loss callbacks (rg, i_ptm, helix, beta_strand, NC).
"""

import logging
from typing import Any, Literal

from pydantic import model_validator
from typing_extensions import Self

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    PROTEIN_AMINO_ACIDS,
    BaseConfig,
    ConfigField,
    GradientInput,
    GradientOutput,
    ToolInstance,
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

AlphaFold2GradientInput = GradientInput


class AlphaFold2GradientConfig(BaseConfig):
    """Configuration for AlphaFold2/ColabDesign binder-design gradients.

    Binder protocol only — designs a binder against a frozen target structure.
    Set ``backend="germinal"`` to enable Germinal-specific features (alpha=2.0
    logit scaling, persistent bias, framework contact penalties, extension
    loss callbacks, and CDR position-aware losses).

    Attributes:
        target_pdb (str | None): Target+binder template PDB (file path or PDB string).
        target_chain (str): Frozen target chain ID(s).
        target_hotspot (str | None): Comma-separated hotspot residues on the target.
        binder_chain (str): Binder chain ID in the template PDB.
        design_positions (list[int] | None): Binder residue indices for loss focus
            (e.g. CDR loops). Germinal backend only.
        bias_redesign (float | None): Persistent softmax bias toward wildtype at
            non-design positions. Germinal backend only.
        omit_aas (str | None): Amino acids to ban (e.g. ``"C,W"``).
        num_recycles (int): AF2 recycling iterations.
        model_num (int): AF2 parameter set (1-5).
        sample_models (bool): Randomly sample model sets each forward pass.
        soft (float): ColabDesign softmax blending (0=raw logits, 1=full softmax).
            Passed per-step by the gradient optimizer.
        backend (Literal["base", "germinal"]): ``"base"`` (upstream ColabDesign) or ``"germinal"``
            (Germinal fork with alpha, bias, framework contacts, extension losses).
        loss_weights (dict[str, float]): Binder-objective weights. Base keys:
            plddt, i_plddt, pae, i_pae, con, i_con, exp_res, rmsd, dgram_cce,
            fape. Germinal extension keys: rg, i_ptm, NC, helix, beta_strand.
        intra_contact_num (int): Intra-chain contacts per residue. Germinal only.
        intra_contact_cutoff (float): Intra-chain distance cutoff (Å). Germinal only.
        inter_contact_num (int): Interface contacts per residue. Germinal only.
        inter_contact_cutoff (float): Interface distance cutoff (Å). Germinal only.
    """

    target_pdb: str | None = ConfigField(
        title="Target PDB",
        default=None,
        description="Target+binder template PDB (file path or PDB-format string).",
    )
    target_chain: str = ConfigField(
        title="Target Chain",
        default="A",
        description="Chain ID(s) of the frozen target in the PDB.",
    )
    target_hotspot: str | None = ConfigField(
        title="Target Hotspot",
        default=None,
        description="Comma-separated hotspot residue indices on the target (e.g. '10,25,42').",
    )
    binder_chain: str = ConfigField(
        title="Binder Chain",
        default="H",
        description="Binder chain ID for template-based binder redesign.",
    )
    design_positions: list[int] | None = ConfigField(
        title="Design Positions",
        default=None,
        description="Zero-based binder residue indices for loss focus (e.g. CDR loops).",
        advanced=True,
    )
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
    backend: Literal["base", "germinal"] = ConfigField(
        title="Backend",
        default="base",
        description="ColabDesign backend: 'base' (upstream) or 'germinal' (with alpha, bias, IgLM).",
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
        return self


AlphaFold2GradientOutput = GradientOutput


def example_input() -> AlphaFold2GradientInput:
    """Minimal valid input — short VHH-like binder with biased logits."""
    aa_index = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}
    n_aas = len(PROTEIN_AMINO_ACIDS)
    logits = []
    for residue in "EVQLVESG":
        row = [0.0] * n_aas
        row[aa_index[residue]] = 2.0
        logits.append(row)
    return AlphaFold2GradientInput(logits=logits)


@tool(
    key="alphafold2-gradient",
    label="AlphaFold2 Gradient",
    category="structure_prediction",
    input_class=AlphaFold2GradientInput,
    config_class=AlphaFold2GradientConfig,
    output_class=AlphaFold2GradientOutput,
    description="Compute AF2 binder-design gradients w.r.t. relaxed logits (base or Germinal backend)",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
)
def run_alphafold2_gradient(
    inputs: AlphaFold2GradientInput,
    config: AlphaFold2GradientConfig,
    instance: Any = None,
) -> AlphaFold2GradientOutput:
    """Compute one AlphaFold2/ColabDesign binder-design gradient step."""
    if config.target_pdb is None:
        raise ValueError("target_pdb is required for binder gradient computation.")
    logger.debug("Using local for AlphaFold2 binder gradient: model=%d", config.model_num)
    result = ToolInstance.dispatch(
        "alphafold2",
        {
            "operation": "compute_gradient",
            "logits": inputs.logits,
            "temperature": inputs.temperature,
            "soft": config.soft,
            "target_pdb": config.target_pdb,
            "target_chain": config.target_chain,
            "target_hotspot": config.target_hotspot,
            "binder_chain": config.binder_chain,
            "design_positions": config.design_positions,
            "bias_redesign": config.bias_redesign,
            "omit_aas": config.omit_aas,
            "num_recycles": config.num_recycles,
            "model_num": config.model_num,
            "sample_models": config.sample_models,
            "loss_weights": config.loss_weights,
            "intra_contact_num": config.intra_contact_num,
            "intra_contact_cutoff": config.intra_contact_cutoff,
            "inter_contact_num": config.inter_contact_num,
            "inter_contact_cutoff": config.inter_contact_cutoff,
            "seed": config.seed,
            "backend": config.backend,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )

    return AlphaFold2GradientOutput(
        gradient=result["gradient"],
        loss=result["loss"],
        metrics=result["metrics"],
        vocab=result["vocab"],
    )

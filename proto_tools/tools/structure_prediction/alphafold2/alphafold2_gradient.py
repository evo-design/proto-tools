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
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import AlphaFold2Metrics
from proto_tools.tools.structure_prediction.shared_data_models import StructureGradientResult
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    AminoAcid,
    ConfigField,
    GradientInput,
    GradientOutput,
    InputField,
    ProteinSequenceLogits,
    RelaxationConfig,
    SequenceLogits,
    ToolInstance,
    coerce_to_item_list,
    one_hot_protein_logits,
)
from proto_tools.utils.sequence import normalize_excluded_amino_acids

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


class AlphaFold2GradientCandidate(BaseModel):
    """One candidate: a relaxed binder state with the target it is designed against.

    The target template and its selections belong to the candidate, so a batch may design against
    different targets and each item's chains validate against its own PDB.

    Attributes:
        logits (ProteinSequenceLogits): Relaxed binder state (L x 20) in proto amino-acid order.
        target_pdb (Structure): Target+binder template PDB. Accepts a file path,
            raw PDB/CIF content string, ``Structure`` object, or a dict in the
            shape produced by ``Structure.model_dump(mode='json')``.
        target_chain (str): Chain ID(s) of the frozen target in the PDB.
        target_hotspot (str | None): Comma-separated hotspot residue indices on the target.
        binder_chain (str | None): Binder template chain to redesign; None (default) designs de novo.
        design_positions (list[int] | None): Zero-based binder residue indices
            for loss focus (e.g. CDR loops). Germinal backend only.
    """

    model_config = ConfigDict(extra="forbid")

    logits: ProteinSequenceLogits = Field(
        title="Logits",
        description="Relaxed (L, 20) binder state in proto amino-acid order.",
    )
    target_pdb: Structure = Field(
        title="Target PDB",
        description="Target+binder template PDB.",
    )
    target_chain: str = Field(
        default="A",
        title="Target Chain",
        description="Chain ID(s) of the frozen target in the PDB.",
    )
    target_hotspot: str | None = Field(
        default=None,
        title="Target Hotspot",
        description="Comma-separated hotspot residue indices on the target (e.g. '10,25,42').",
    )
    binder_chain: str | None = Field(
        default=None,
        title="Binder Chain",
        description="Binder template chain to redesign; None (default) designs the binder de novo.",
    )
    design_positions: list[int] | None = Field(
        default=None,
        title="Design Positions",
        description="Zero-based binder residue indices for loss focus (e.g. CDR loops).",
    )

    def iter_sequence_logits(self) -> Iterator[tuple[str, SequenceLogits]]:
        """Expose this item's relaxed state to the shared validation hooks.

        Yields:
            tuple[str, SequenceLogits]: The field name and this item's matrix.
        """
        yield "logits", self.logits

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


class AlphaFold2GradientInput(GradientInput):
    """One or many relaxed binders, each designed against its own target.

    Attributes:
        inputs (list[AlphaFold2GradientCandidate]): One binder-and-target bundle per candidate. A
            single bundle may be passed on its own.
        temperature (float | None): Softmax temperature shared by every item. The default ``1.0``
            reads each state as unnormalized and applies ``softmax(logits / T)``, returning
            logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """

    inputs: list[AlphaFold2GradientCandidate] = InputField(
        title="Relaxed Candidates",
        description="One relaxed binder with its target template per candidate; a single bundle is accepted alone",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one candidate as well as a list of them."""
        return coerce_to_item_list(value)

    @classmethod
    def build_candidate(
        cls,
        probabilities: Sequence[Sequence[float]],
        *,
        temperature: float = 1.0,
        **fields: Any,
    ) -> AlphaFold2GradientCandidate:
        """Build one candidate pairing the converted binder matrix with its target template.

        The template belongs to the candidate, so it and its selections arrive here either shared
        across the call or named in this candidate's own mapping.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, 20)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's remaining fields; must include ``target_pdb``.

        Returns:
            AlphaFold2GradientCandidate: Candidate whose logits reproduce ``probabilities``.

        Raises:
            ValueError: If ``target_pdb`` is missing.
        """
        if "target_pdb" not in fields:
            raise ValueError("alphafold2 gradient from_probabilities requires the target template")
        matrix = ProteinSequenceLogits.from_probabilities(probabilities, temperature=temperature)
        return AlphaFold2GradientCandidate(logits=matrix, **fields)


class AlphaFold2GradientConfig(RelaxationConfig):
    """Configuration for AlphaFold2/ColabDesign binder design (forward or backward).

    Binder protocol only — designs a binder against a frozen target structure.
    Set ``backend="germinal"`` to enable Germinal-specific features (alpha=2.0
    logit scaling, persistent bias, framework contact penalties, extension
    loss callbacks, and CDR position-aware losses).

    Attributes:
        include_pae_matrix (bool): Attach full per-residue PAE matrix. Default: ``False``.
        bias_redesign (float | None): Persistent softmax bias toward wildtype at
            non-design positions. Germinal backend only.
        excluded_amino_acids (list[AminoAcid] | None): Amino acids to ban (e.g. ``["C", "W"]``).
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
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities (1).
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
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
    excluded_amino_acids: list[AminoAcid] | None = ConfigField(
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

    @field_validator("excluded_amino_acids", mode="before")
    @classmethod
    def validate_excluded_amino_acids(cls, value: Any) -> Any:
        """Normalize and validate excluded amino-acid codes."""
        return normalize_excluded_amino_acids(value)

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
    """Binder-design output: one result per item, each with its objective, metrics and structure.

    Attributes:
        results (list[StructureGradientResult]): One per input item, each carrying that item's
            gradient, objective, auxiliary metrics (avg_plddt, ptm, iptm, avg_pae, plus per-loss
            values for every weighted ColabDesign term), and predicted target+binder complex.
            B-factors are at the raw 0-100 PDB scale; ``b_factor_type=PLDDT`` means
            ``Structure.per_residue_plddt`` normalizes them to ``[0, 1]``.
        vocab (list[str]): Amino-acid column ordering.
    """

    results: list[StructureGradientResult] = Field(  # type: ignore[assignment]  # narrows the element type
        title="Results",
        description="One result per input item, in input order, each with its own predicted complex",
    )

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Write one PDB sidecar per result alongside the gradient JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        # Suffix-additive so dotted names (e.g. "step_v1.5") aren't truncated.
        base = Path(export_path)
        json_path = base.parent / f"{base.name}.json"
        sidecars: list[str] = []
        for index, result in enumerate(self.results):
            pdb_path = base.parent / f"{base.name}_{index:04d}.pdb"
            result.structure.write_pdb(pdb_path)
            sidecars.append(pdb_path.name)
        payload = self.model_dump(include={"results", "vocab"}) | {"structure_pdbs": sidecars}
        json_path.write_text(json.dumps(payload, indent=2))


def _reject_unused_softmax_knobs(inputs: AlphaFold2GradientInput, config: AlphaFold2GradientConfig) -> None:
    """Refuse config that ColabDesign would silently ignore in probability mode.

    ``temperature=None`` consumes the supplied probabilities verbatim, which ColabDesign
    expresses as ``soft=0``. Its softmax branch is then unused, so every knob feeding only that
    branch — the ``hard`` straight-through blend, and the ``bias`` behind
    ``excluded_amino_acids`` and ``bias_redesign`` — would have no effect. Refuse them rather
    than drop them silently.

    Args:
        inputs (AlphaFold2GradientInput): Binder input, read for ``temperature``.
        config (AlphaFold2GradientConfig): Config whose relaxation knobs are checked.

    Raises:
        ValueError: If probability mode is combined with a knob that only affects the softmax.
    """
    if inputs.temperature is not None:
        return
    ignored = []
    if config.soft != 1.0:
        ignored.append(f"soft={config.soft}")
    if config.hard != 0.0:
        ignored.append(f"hard={config.hard}")
    if config.excluded_amino_acids:
        ignored.append(f"excluded_amino_acids={config.excluded_amino_acids}")
    if config.bias_redesign is not None:
        ignored.append(f"bias_redesign={config.bias_redesign}")
    if ignored:
        raise ValueError(
            "alphafold2-gradient: temperature=None consumes the supplied probabilities directly, "
            f"leaving ColabDesign's softmax branch unused, so {', '.join(ignored)} would have no "
            "effect. Drop them, or set a positive temperature to optimize in logit space."
        )


def example_input() -> AlphaFold2GradientInput:
    """Minimal valid input — short VHH-like binder with biased logits and PD-L1 template."""
    return AlphaFold2GradientInput(
        inputs=[
            AlphaFold2GradientCandidate(
                logits=ProteinSequenceLogits(one_hot_protein_logits("EVQLVESG", sharpness=2.0)),
                target_pdb=Structure.from_file(_BINDER_FIXTURE_PDB),
                binder_chain="B",
            )
        ]
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
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=2,
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
    _reject_unused_softmax_knobs(inputs, config)
    logger.debug(
        "Running AlphaFold2 binder design: model=%d, compute_gradient=%s",
        config.model_num,
        config.compute_gradient,
    )
    # ExitStack keeps every candidate's target tempfile alive for the dispatch; ColabDesign reads paths.
    with ExitStack() as stack:
        items = [
            {
                "logits": item.logits.root,
                "target_pdb": str(stack.enter_context(item.target_pdb.temp_file())),
                "target_chain": item.target_chain,
                "target_hotspot": item.target_hotspot,
                "binder_chain": item.binder_chain,
                "design_positions": item.design_positions,
            }
            for item in inputs.inputs
        ]
        result = ToolInstance.dispatch(
            "alphafold2",
            {
                "operation": "compute_gradient",
                "items": items,
                "temperature": inputs.temperature,
                "soft": config.soft,
                "hard": config.hard,
                "bias_redesign": config.bias_redesign,
                # ColabDesign's prep_inputs(rm_aa=...) expects a comma-separated string.
                "excluded_amino_acids": (
                    ",".join(config.excluded_amino_acids) if config.excluded_amino_acids else None
                ),
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

    results = []
    for item_result in result["results"]:
        metrics = item_result["metrics"]
        results.append(
            StructureGradientResult(
                gradient=item_result["gradient"],
                loss=item_result["loss"],
                metrics=metrics,
                structure=Structure(
                    structure=item_result["pdb"],
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
        )
    return AlphaFold2GradientOutput(results=results, vocab=result["vocab"])

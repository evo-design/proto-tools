"""proto_tools/tools/structure_prediction/alphafold2/gradient_helpers.py.

Differentiable quantities and shared machinery for the three AlphaFold2 gradient tools.

Each quantity is one class, reused by whichever protocols actually compute it, so a term means
the same thing wherever it appears. A tool exposes only the quantities measured to be live under
its protocol: ``rmsd`` is a real structural error under ``fixbb`` but an inert floor elsewhere,
so it belongs to that tool's catalog alone.

Two quantities are reported as confidences rather than as ColabDesign's losses. ColabDesign
returns ``1 - pLDDT`` and ``1 - P(resolved)``; the worker reports the confidence and negates the
gradient with it, so ``gradient == d(term_value)/d(input)`` holds either way.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import Field, PositiveInt, field_validator

from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import AlphaFold2Metrics
from proto_tools.utils import (
    PROTEIN_AMINO_ACIDS,
    ConfigField,
    DifferentiableTerm,
    GradientConfig,
    GradientMetadata,
    GradientOutput,
    ToolInstance,
    resolve_terms,
)
from proto_tools.utils.tool_io import Directionality

logger = logging.getLogger(__name__)


class AlphaFold2Term(DifferentiableTerm):
    """One AlphaFold2 quantity a design protocol can differentiate.

    Attributes:
        id (str): Fixed identifier, pinned by each concrete quantity.
    """


class AlphaFold2PlddtTerm(AlphaFold2Term):
    """Mean predicted local distance difference test over the designed residues.

    Attributes:
        id (Literal["plddt"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "higher"
    DESCRIPTION: ClassVar[str] = "Mean pLDDT confidence over the designed residues"

    id: Literal["plddt"] = Field(
        default="plddt", title="Term ID", description="Identifies this quantity as mean pLDDT."
    )


class AlphaFold2ExpResTerm(AlphaFold2Term):
    """Mean predicted probability that the designed residues are experimentally resolved.

    Attributes:
        id (Literal["exp_res"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "higher"
    DESCRIPTION: ClassVar[str] = "Mean predicted probability of being experimentally resolved"

    id: Literal["exp_res"] = Field(
        default="exp_res", title="Term ID", description="Identifies this quantity as resolved confidence."
    )


class AlphaFold2PaeTerm(AlphaFold2Term):
    """Normalized predicted aligned error within the designed chain.

    Attributes:
        id (Literal["pae"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"
    DESCRIPTION: ClassVar[str] = "Normalized intra-chain predicted aligned error"

    id: Literal["pae"] = Field(
        default="pae", title="Term ID", description="Identifies this quantity as intra-chain PAE."
    )


class AlphaFold2InterfacePaeTerm(AlphaFold2Term):
    """Normalized predicted aligned error across the binder-target interface.

    Attributes:
        id (Literal["i_pae"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"
    DESCRIPTION: ClassVar[str] = "Normalized interface predicted aligned error"

    id: Literal["i_pae"] = Field(
        default="i_pae", title="Term ID", description="Identifies this quantity as interface PAE."
    )


class AlphaFold2ContactTerm(AlphaFold2Term):
    """ColabDesign's intra-chain contact objective over the designed residues.

    Attributes:
        id (Literal["con"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"
    DESCRIPTION: ClassVar[str] = "Intra-chain contact objective (ColabDesign con loss)"

    id: Literal["con"] = Field(
        default="con", title="Term ID", description="Identifies this quantity as the intra-chain contact loss."
    )


class AlphaFold2InterfaceContactTerm(AlphaFold2Term):
    """ColabDesign's interface contact objective between binder and target.

    Attributes:
        id (Literal["i_con"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"
    DESCRIPTION: ClassVar[str] = "Interface contact objective (ColabDesign i_con loss)"

    id: Literal["i_con"] = Field(
        default="i_con", title="Term ID", description="Identifies this quantity as the interface contact loss."
    )


class AlphaFold2HelixTerm(AlphaFold2Term):
    """Helical-contact objective: the mean binary contact loss over i,i+3 residue pairs.

    Lower values mean more helical contacts. Whether that is wanted depends on the design, so
    this quantity carries no advisory direction and composing it requires an explicit goal.

    Attributes:
        id (Literal["helix"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "context-dependent"
    DESCRIPTION: ClassVar[str] = "Helical contact objective (ColabDesign helix loss)"

    id: Literal["helix"] = Field(
        default="helix", title="Term ID", description="Identifies this quantity as the helix loss."
    )


class AlphaFold2DgramCCETerm(AlphaFold2Term):
    """Distogram cross-entropy between the prediction and the reference structure.

    Attributes:
        id (Literal["dgram_cce"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"
    DESCRIPTION: ClassVar[str] = "Distogram cross-entropy against the reference structure"

    id: Literal["dgram_cce"] = Field(
        default="dgram_cce", title="Term ID", description="Identifies this quantity as distogram cross-entropy."
    )


class AlphaFold2FapeTerm(AlphaFold2Term):
    """Frame-aligned point error between the prediction and the reference structure.

    Attributes:
        id (Literal["fape"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"
    DESCRIPTION: ClassVar[str] = "Frame-aligned point error against the reference structure"

    id: Literal["fape"] = Field(
        default="fape", title="Term ID", description="Identifies this quantity as frame-aligned point error."
    )


class AlphaFold2RmsdTerm(AlphaFold2Term):
    """Structural RMSD between the prediction and the reference structure, in angstroms.

    Attributes:
        id (Literal["rmsd"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"
    DESCRIPTION: ClassVar[str] = "Structural RMSD against the reference backbone"
    UNITS: ClassVar[str | None] = "angstroms"

    id: Literal["rmsd"] = Field(
        default="rmsd", title="Term ID", description="Identifies this quantity as RMSD to the reference."
    )


class AlphaFold2RgTerm(AlphaFold2Term):
    """Radius of gyration of the designed chain's CA atoms beyond the compact-fold expectation.

    The value is ``elu(rg - 2.38 * N**0.365)`` for ``N`` designed residues, as in BindCraft and
    Germinal.

    Attributes:
        id (Literal["rg"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"
    DESCRIPTION: ClassVar[str] = "Radius of gyration in excess of the expected compact value"

    id: Literal["rg"] = Field(default="rg", title="Term ID", description="Identifies this quantity as excess Rg.")


class AlphaFold2TerminiDistanceTerm(AlphaFold2Term):
    """Distance between the designed chain's terminal CA atoms beyond 7 angstroms.

    The value is ``relu(elu(d - 7))``, which is zero once the termini are within 7 angstroms.

    Attributes:
        id (Literal["NC"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"
    DESCRIPTION: ClassVar[str] = "N-to-C terminal CA distance in excess of 7 angstroms"
    UNITS: ClassVar[str | None] = "angstroms"

    id: Literal["NC"] = Field(
        default="NC", title="Term ID", description="Identifies this quantity as termini distance."
    )


class AlphaFold2BetaStrandTerm(AlphaFold2Term):
    """Strand-geometry objective over i,i+3 pairs of the designed residues.

    Each pair scores ``-log P(9.75 <= d <= 11.5)`` from the distogram; the value is the mean of
    the worst third. Lower values mean more strand-like spacing, so like ``helix`` it carries no
    advisory direction.

    Attributes:
        id (Literal["beta_strand"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "context-dependent"
    DESCRIPTION: ClassVar[str] = "Strand-spacing objective over i,i+3 designed pairs"

    id: Literal["beta_strand"] = Field(
        default="beta_strand", title="Term ID", description="Identifies this quantity as the strand loss."
    )


class AlphaFold2InterfacePtmTerm(AlphaFold2Term):
    """Interface predicted TM-score between binder and target.

    Attributes:
        id (Literal["i_ptm"]): Fixed identifier for this kind of quantity.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "higher"
    DESCRIPTION: ClassVar[str] = "Interface predicted TM-score"

    id: Literal["i_ptm"] = Field(default="i_ptm", title="Term ID", description="Identifies this quantity as ipTM.")


class AlphaFold2FrameworkContactTerm(AlphaFold2Term):
    """Contact loss between the target and the binder's framework residues.

    The same contact objective as ``i_con``, restricted to the framework, so higher values mean
    the framework sits farther from the target. Germinal's framework penalty composes it with
    ``i_con`` as ``i_con**2 / max(fw_i_con - offset, 1e-8)``.

    Attributes:
        id (Literal["fw_i_con"]): Fixed identifier for this kind of quantity.
        framework_positions (list[PositiveInt]): 1-indexed binder positions that form the framework.
    """

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "higher"
    DESCRIPTION: ClassVar[str] = "Contact loss between the target and the binder framework"

    id: Literal["fw_i_con"] = Field(
        default="fw_i_con", title="Term ID", description="Identifies this quantity as framework contact loss."
    )
    framework_positions: list[PositiveInt] = Field(
        min_length=1, title="Framework Positions", description="1-indexed binder positions forming the framework."
    )


def build_terms(
    payload_terms: list[dict[str, Any]],
    requested: Sequence[AlphaFold2Term],
    matrix: list[list[float]],
    *,
    temperature: float | None,
    soft: float,
    hard: float,
) -> list[dict[str, Any]]:
    """Turn one candidate's worker entries into the arguments a narrowed TermGradient needs.

    Args:
        payload_terms (list[dict[str, Any]]): Worker entries, each with id, value and gradient.
        requested (Sequence[AlphaFold2Term]): The config's terms, returned with their selectors.
        matrix (list[list[float]]): The relaxed state this candidate was differentiated at.
        temperature (float | None): The input's temperature.
        soft (float): Relaxation blend in force.
        hard (float): Straight-through coefficient in force.

    Returns:
        list[dict[str, Any]]: One keyword mapping per term, in worker order.
    """
    # Every term describes the same relaxed state, so the basis is computed once per candidate.
    basis = GradientMetadata.from_input(
        matrix,
        vocab=list(PROTEIN_AMINO_ACIDS),
        temperature=temperature,
        soft=soft,
        hard=hard,
    )
    by_id = {term.id: term for term in requested}
    built = []
    for entry in payload_terms:
        term = by_id.get(entry["id"])
        if term is None:
            raise ValueError(f"alphafold2 reported term {entry['id']!r}, which was not requested ({sorted(by_id)})")
        if entry["value"] is None:
            raise ValueError(
                f"alphafold2 term {entry['id']!r} has nothing to measure on this input and reported no value"
            )
        built.append(
            {
                "term": term,
                "term_value": float(entry["value"]),
                "gradient": entry["gradient"],
                "metadata": basis,
            }
        )
    return built


def build_structure(item: Mapping[str, Any], *, source: str) -> Structure:
    """Wrap one worker result's predicted structure and its confidence metrics.

    Args:
        item (Mapping[str, Any]): One worker result, carrying ``pdb`` and ``metrics``.
        source (str): Registry key of the tool that produced the structure.

    Returns:
        Structure: The prediction, with pLDDT in the B-factor column.
    """
    metrics = item["metrics"]
    return Structure(
        structure=item["pdb"],
        b_factor_type=BFactorType.PLDDT,
        metrics=AlphaFold2Metrics(
            avg_plddt=metrics["avg_plddt"],
            ptm=metrics.get("ptm"),
            iptm=metrics.get("iptm"),
            avg_pae=metrics.get("avg_pae"),
            pae=metrics.get("pae"),
        ),
        source=source,
    )


class AlphaFold2StructureGradientOutput(GradientOutput):
    """Gradient output whose results each carry a predicted structure.

    Every AlphaFold2 design protocol folds the candidate it differentiates, so each result holds
    a structure. Writing those to sidecars keeps the JSON readable and gives each result a file
    a structure viewer can open.
    """

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Write one PDB sidecar per result alongside the gradient JSON.

        Args:
            export_path (str | Path): Base path; the suffix is added rather than replaced.
            file_format (str): Only ``"json"`` is supported.

        Raises:
            ValueError: If ``file_format`` is anything but ``"json"``.
        """
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        # Suffix-additive so dotted names (e.g. "step_v1.5") are not truncated.
        base = Path(export_path)
        sidecars: list[str] = []
        for index, result in enumerate(self.results):
            pdb_path = base.parent / f"{base.name}_{index:04d}.pdb"
            result.structure.write_pdb(pdb_path)  # type: ignore[attr-defined]
            sidecars.append(pdb_path.name)
        payload = self.model_dump(include={"results", "vocab"}) | {"structure_pdbs": sidecars}
        (base.parent / f"{base.name}.json").write_text(json.dumps(payload, indent=2))


class AlphaFold2DesignGradientConfig(GradientConfig):
    """Settings shared by the three AlphaFold2 design-gradient protocols.

    Attributes:
        compute_gradient (bool): Run the backward passes; ``False`` returns values only.
        num_recycles (int): Structure-module recycling iterations.
        recycle_mode (Literal["last", "sample", "average", "first"]): Which recycle carries the
            gradient.
        model_num (int): Which AlphaFold2 parameter set to run.
        sample_models (bool): Draw the parameter set at random instead of using ``model_num``.
        include_pae_matrix (bool): Attach the full per-residue PAE matrix to the metrics.
        device (str): Device to run the model on.
    """

    if TYPE_CHECKING:
        # Declared by each leaf config, narrowed to that protocol's own terms.
        gradient_terms: Sequence[AlphaFold2Term]

    compute_gradient: bool = ConfigField(
        title="Compute Gradient", default=True, description="Run backward passes; False returns values only"
    )
    num_recycles: int = ConfigField(title="Num Recycles", default=3, ge=0, description="Recycling iterations")
    recycle_mode: Literal["last", "sample", "average", "first"] = ConfigField(
        title="Recycle Mode", default="last", description="Which recycle carries the gradient"
    )
    model_num: int = ConfigField(title="Model Number", default=1, ge=1, le=5, description="AlphaFold2 parameter set")
    sample_models: bool = ConfigField(
        title="Sample Models", default=False, description="Draw the parameter set at random"
    )
    include_pae_matrix: bool = ConfigField(
        title="Include PAE Matrix", default=False, description="Attach the full per-residue PAE matrix"
    )
    device: str = ConfigField(
        title="Device", default="cuda", description="Device to run the model on", include_in_key=False
    )

    # Each tool narrows ``gradient_terms`` to its own protocol; only the validation is shared.
    @field_validator("gradient_terms", mode="before", check_fields=False)
    @classmethod
    def resolve_gradient_terms(cls, value: Any) -> Any:
        """Accept catalog names and term classes alongside instances."""
        if not isinstance(value, list):
            return value
        catalog = cls.supported_terms()
        return [
            resolve_terms([entry], catalog)[0]  # type: ignore[arg-type]
            if isinstance(entry, str | type)
            else entry
            for entry in value
        ]

    @field_validator("gradient_terms", mode="after", check_fields=False)
    @classmethod
    def reject_repeated_terms(cls, value: list[Any]) -> list[Any]:
        """Each quantity yields one term, so asking for it twice is a mistake rather than a sum."""
        asked = [term.id for term in value]
        repeated = sorted({name for name in asked if asked.count(name) > 1})
        if repeated:
            raise ValueError(f"gradient_terms names {repeated} more than once; each quantity yields one term")
        return value


def _reject_unused_softmax_knobs(
    config: AlphaFold2DesignGradientConfig, temperature: float | None, tool_key: str
) -> None:
    """Refuse config the model would silently ignore in probability mode.

    ``temperature=None`` consumes the supplied probabilities verbatim, which ColabDesign expresses
    as ``soft=0``. Its softmax branch is then unused, so ``soft`` and ``hard`` would have no effect
    on the result and no honest value to report in the gradient's metadata.

    Args:
        config (AlphaFold2DesignGradientConfig): Config whose relaxation knobs are checked.
        temperature (float | None): The input's temperature.
        tool_key (str): Registry key, named in the error.

    Raises:
        ValueError: If probability mode is combined with a knob that only affects the softmax.
    """
    if temperature is not None:
        return
    fields = type(config).model_fields
    ignored = [
        f"{name}={getattr(config, name)}" for name in ("soft", "hard") if getattr(config, name) != fields[name].default
    ]
    if ignored:
        raise ValueError(
            f"{tool_key}: temperature=None consumes the supplied probabilities directly, leaving the "
            f"softmax branch unused, so {', '.join(ignored)} would have no effect. Drop them, or set "
            "a positive temperature to optimize in logit space."
        )


def dispatch_design_gradient(
    *,
    protocol: Literal["binder", "fixbb", "hallucination"],
    items: list[dict[str, Any]],
    config: AlphaFold2DesignGradientConfig,
    temperature: float | None,
    tool_key: str,
    extra_payload: Mapping[str, Any] | None = None,
    instance: Any = None,
) -> list[dict[str, Any]]:
    """Run one protocol's design gradients in the AlphaFold2 worker.

    Args:
        protocol (Literal["binder", "fixbb", "hallucination"]): ColabDesign protocol to build.
        items (list[dict[str, Any]]): One entry per candidate, with its logits and prep arguments.
        config (AlphaFold2DesignGradientConfig): The calling tool's config.
        temperature (float | None): The input's temperature.
        tool_key (str): Registry key of the calling tool, named in errors.
        extra_payload (Mapping[str, Any] | None): Protocol-specific payload keys, merged last.
        instance (Any): Optional persistent tool instance.

    Returns:
        list[dict[str, Any]]: One worker result per candidate, in input order.
    """
    _reject_unused_softmax_knobs(config, temperature, tool_key)
    requested = [term.id for term in config.gradient_terms]
    logger.debug("%s: %d candidate(s), terms=%s", tool_key, len(items), requested)

    payload = ToolInstance.dispatch(
        "alphafold2",
        {
            "operation": "compute_design_gradient",
            "protocol": protocol,
            "items": items,
            "gradient_terms": requested,
            "term_options": {term.id: term.model_dump(exclude={"id"}) for term in config.gradient_terms},
            "temperature": temperature,
            "soft": config.soft,
            "hard": config.hard,
            "num_recycles": config.num_recycles,
            "recycle_mode": config.recycle_mode,
            "model_num": config.model_num,
            "sample_models": config.sample_models,
            "seed": config.seed,
            "compute_gradient": config.compute_gradient,
            "device": config.device,
            "verbose": config.verbose,
            "include_pae_matrix": config.include_pae_matrix,
            **(extra_payload or {}),
        },
        instance=instance,
        config=config,
    )
    return cast(list[dict[str, Any]], payload["results"])


__all__ = [
    "AlphaFold2BetaStrandTerm",
    "AlphaFold2ContactTerm",
    "AlphaFold2DesignGradientConfig",
    "AlphaFold2DgramCCETerm",
    "AlphaFold2ExpResTerm",
    "AlphaFold2FapeTerm",
    "AlphaFold2FrameworkContactTerm",
    "AlphaFold2HelixTerm",
    "AlphaFold2InterfaceContactTerm",
    "AlphaFold2InterfacePaeTerm",
    "AlphaFold2InterfacePtmTerm",
    "AlphaFold2PaeTerm",
    "AlphaFold2PlddtTerm",
    "AlphaFold2RgTerm",
    "AlphaFold2RmsdTerm",
    "AlphaFold2Term",
    "AlphaFold2TerminiDistanceTerm",
    "build_structure",
    "build_terms",
    "dispatch_design_gradient",
]

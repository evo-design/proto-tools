"""proto_tools/tools/inverse_folding/caliby/caliby_sample.py.

Caliby Potts-model sequence design, optionally conditioned on a conformer ensemble.
"""

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from proto_tools.entities.complex import Chain
from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures import ChainSelection, Structure
from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedComplex,
    DesignSet,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.sequence import AminoAcid, normalize_excluded_amino_acids
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)

# Checkpoints auto-downloaded by standalone/setup.sh, split by the model's `task`.
DESIGN_CHECKPOINTS: tuple[str, ...] = (
    "caliby",
    "soluble_caliby",
    "soluble_caliby_v1",
    "caliby_distill",
    "soluble_caliby_distill",
)
PACKER_CHECKPOINTS: tuple[str, ...] = ("caliby_packer_000", "caliby_packer_010", "caliby_packer_030")


def validate_checkpoint(value: str, allowed: tuple[str, ...], operation: str) -> str:
    """Accept a bundled checkpoint name from ``allowed`` or any explicit ``.ckpt`` path.

    Args:
        value (str): The ``model_checkpoint`` field value.
        allowed (tuple[str, ...]): Bundled names valid for this operation.
        operation (str): Operation name used in the error message.

    Returns:
        str: The validated value, unchanged.

    Raises:
        ValueError: If ``value`` is a bundled name that is wrong for this operation.
    """
    if value.endswith(".ckpt") or value in allowed:
        return value
    known = set(DESIGN_CHECKPOINTS) | set(PACKER_CHECKPOINTS)
    if value in known:
        raise ValueError(
            f"'{value}' is not a Caliby {operation} checkpoint. Choose one of {list(allowed)}, "
            "or pass a path to a .ckpt file."
        )
    raise ValueError(f"Unknown Caliby checkpoint '{value}'. Choose one of {list(allowed)}, or pass a .ckpt path.")


def validate_conformer_alignment(primary: Structure, conformers: list[Structure]) -> None:
    """Require every conformer to share the primary conformer's chain IDs and residue numbering.

    Caliby aggregates Potts parameters position-wise across the ensemble, and rejects a
    mismatch by comparing residue numbers elementwise rather than residue counts. Checking
    the same thing here reports it against the offending input before the model loads.
    Renumbered ensembles are the common case: a sampler emitting conformers numbered from 1,
    against a primary carrying its deposited numbering, matches on counts but not on numbers.

    Args:
        primary (Structure): The primary conformer.
        conformers (list[Structure]): The other ensemble members.

    Raises:
        ValueError: If any conformer's chain IDs or per-chain residue numbers differ.
    """
    if not conformers:
        return

    primary_chains = primary.get_chain_ids()
    primary_numbering = {cid: primary.get_chain_positions(cid) for cid in primary_chains}
    for index, conformer in enumerate(conformers):
        chains = conformer.get_chain_ids()
        if chains != primary_chains:
            raise ValueError(
                f"conformers[{index}] has chains {chains} but the primary structure has "
                f"{primary_chains}; every conformer must match the primary conformer."
            )
        numbering = {cid: conformer.get_chain_positions(cid) for cid in chains}
        differing = [cid for cid in chains if numbering[cid] != primary_numbering[cid]]
        if differing:
            chain_id = differing[0]
            got, want = numbering[chain_id], primary_numbering[chain_id]
            detail = f"{len(got)} residues numbered {got[0]}..{got[-1]} vs {len(want)} numbered {want[0]}..{want[-1]}"
            raise ValueError(
                f"conformers[{index}] does not align with the primary structure on chain(s) "
                f"{differing}: chain {chain_id} has {detail}. Caliby aggregates Potts parameters "
                "position-wise, so every conformer must carry the primary conformer's residue "
                "numbering; renumber the ensemble to match the primary structure."
            )


class CalibyStructureInput(InverseFoldingStructureInput):
    """A backbone plus an optional conformer ensemble to condition on.

    Extends :class:`InverseFoldingStructureInput` with ``conformers``. The inherited
    ``structure`` is Caliby's **primary conformer**: residue identities are read from it
    and broadcast across the ensemble, and it is the reference frame against which the
    other members' residue indices and chain IDs must align. An empty ``conformers``
    runs single-structure design; a non-empty one runs ensemble-conditioned design,
    which aggregates Potts parameters position-wise over ``[structure, *conformers]``.

    Attributes:
        structure (Structure): Primary conformer. Accepts a file path, raw PDB/CIF
            content string, ``Structure`` object, or a dict in the shape produced by
            ``Structure.model_dump(mode='json')``. Inherited from
            :class:`InverseFoldingStructureInput`.
        conformers (list[Structure]): Additional conformers of the same molecule. Each
            must have the same chain IDs and per-chain residue counts as ``structure``.
            Empty means single-structure design.
        chains_to_redesign (ChainSelection | None): Chains to redesign. ``None`` means
            redesign every chain. Inherited from :class:`InverseFoldingStructureInput`.
        fixed_positions (ResidueSelection | None): Per-chain positions whose residue
            identity is held fixed (1-indexed). Inherited from
            :class:`InverseFoldingStructureInput`.
        symmetric_chains (list[ChainSelection] | None): Chain groups designed to a single
            shared sequence, one group per set of tied chains. Caliby folds each group's
            Potts parameters into one variable before sampling, so the tied chains are
            identical by construction and the reported energy describes that assembly.
            Without it the chains of a homooligomer are designed independently and come
            out different. Chains in a group must share residue numbering.

    Examples:
        >>> # Single structure
        >>> inp = CalibyStructureInput(structure="/path/to/protein.pdb")

        >>> # Ensemble-conditioned, e.g. from a bioemu-sample StructureEnsemble
        >>> inp = CalibyStructureInput(
        ...     structure="/path/to/protein.pdb",
        ...     conformers=ensemble.structures,
        ...     chains_to_redesign=["A"],
        ... )

        >>> # Homotrimer: all three chains share one designed sequence
        >>> inp = CalibyStructureInput(
        ...     structure="/path/to/trimer.cif",
        ...     symmetric_chains=["A", "B", "C"],
        ... )
    """

    conformers: list[Structure] = Field(
        default_factory=list,
        title="Conformers",
        description="Additional conformers of the same molecule; empty = single-structure design.",
    )

    symmetric_chains: list[ChainSelection] | None = Field(
        default=None,
        title="Symmetric Chains",
        description="Chain groups designed to one shared sequence, e.g. [['A', 'B', 'C']] for a homotrimer.",
    )

    @field_validator("symmetric_chains", mode="before")
    @classmethod
    def _wrap_single_symmetry_group(cls, value: Any) -> Any:
        """Accept one bare group, so ``["A", "B", "C"]`` means a single tie."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return [value]
        return value

    @model_validator(mode="after")
    def _validate_conformer_alignment(self) -> "CalibyStructureInput":
        """Require every conformer to align with the primary conformer."""
        validate_conformer_alignment(self.structure, self.conformers)
        return self

    @model_validator(mode="after")
    def _validate_symmetric_chains(self) -> "CalibyStructureInput":
        """Require tied groups to name at least two distinct chains that exist and align."""
        if self.symmetric_chains is None:
            return self

        available = self.structure.get_chain_ids()
        seen: set[str] = set()
        for group in self.symmetric_chains:
            chains = list(group.chains)
            if len(chains) < 2:
                raise ValueError(f"symmetric_chains group {chains} must tie at least two chains")
            if len(set(chains)) != len(chains):
                raise ValueError(f"symmetric_chains group {chains} repeats a chain")
            missing = [cid for cid in chains if cid not in available]
            if missing:
                raise ValueError(f"symmetric_chains names chain(s) {missing} not in structure (have {available})")
            overlap = seen.intersection(chains)
            if overlap:
                raise ValueError(f"chain(s) {sorted(overlap)} appear in more than one symmetric_chains group")
            seen.update(chains)

            # Caliby ties positions by residue number, so the group must share numbering.
            reference = self.structure.get_chain_positions(chains[0])
            for cid in chains[1:]:
                if self.structure.get_chain_positions(cid) != reference:
                    raise ValueError(
                        f"chains {chains[0]!r} and {cid!r} are tied together but carry different residue "
                        "numbering; Caliby ties positions by residue number, so renumber them to match"
                    )
        return self

    @property
    def ensemble_structures(self) -> list[Structure]:
        """The full ensemble in Caliby order: the primary conformer first, then ``conformers``."""
        return [self.structure, *self.conformers]


class CalibySampleInput(BaseToolInput):
    """Input for Caliby sequence design.

    Attributes:
        inputs (list[CalibyStructureInput]): Per-structure inputs, each a primary
            conformer plus optional ensemble members and design constraints.
    """

    inputs: list[CalibyStructureInput] = InputField(
        title="Structure Inputs",
        description="Per-structure inputs, each a primary conformer plus optional conformers.",
    )


class CalibySampleConfig(BaseConfig):
    """Configuration for Caliby sequence design.

    Attributes:
        model_checkpoint (str): Bundled design checkpoint name or a path to a ``.ckpt``
            file. ``'caliby'`` is the default all-chain model; ``'soluble_caliby'``
            excludes transmembrane proteins; the ``_distill`` variants skip ensemble
            generation and suit fast screening but are not recommended for multistate
            design.
        num_sequences_per_structure (int): Sequences to design per input structure.
        batch_size (int): Conformers passed through the model per forward pass when
            conditioning on an ensemble. Reduce on GPU out-of-memory errors. Has no
            effect without conformers, since designs are sampled sequentially from one
            set of Potts parameters rather than batched.
        temperature (float): Final Potts sampling temperature. Caliby anneals from 1.0
            down to this value; raise it for more diverse sequences.
        potts_sweeps (int): MCMC sweeps per designed sequence. More sweeps converge
            closer to the Potts optimum at proportional cost.
        excluded_amino_acids (list[AminoAcid] | None): One-letter codes never placed at
            any designed position.
        use_primary_res_type (bool): Take residue identities from the primary conformer
            rather than each conformer's own. Ensemble-conditioned design only.
        device (str): Device to run the model on.
    """

    model_checkpoint: str = ConfigField(
        title="Model Checkpoint",
        default="caliby",
        description="Bundled design checkpoint name, or a path to a custom .ckpt file.",
        examples=["caliby", "soluble_caliby", "caliby_distill"],
        reload_on_change=True,
    )
    num_sequences_per_structure: int = ConfigField(
        title="Sequences Per Structure",
        default=1,
        ge=1,
        description="Number of sequences to design per input structure.",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=4,
        ge=1,
        description="Conformers per forward pass; ensemble-conditioned design only.",
    )
    temperature: float = ConfigField(
        title="Sampling Temperature",
        default=0.01,
        gt=0.0,
        description="Final Potts sampling temperature, annealed from 1.0; higher = more diverse.",
        examples=[0.01, 0.1, 0.2],
    )
    potts_sweeps: int = ConfigField(
        title="Potts Sweeps",
        default=500,
        ge=1,
        description="MCMC sweeps per designed sequence; more converges closer to the optimum.",
        examples=[500, 1000],
    )
    excluded_amino_acids: list[AminoAcid] | None = ConfigField(
        title="Excluded Amino Acids",
        default=None,
        description="One-letter codes never placed at any designed position.",
        examples=[["C"], ["C", "M"]],
    )
    use_primary_res_type: bool = ConfigField(
        title="Use Primary Res Type",
        default=True,
        description="Take residue identities from the primary conformer (ensemble design only).",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on, e.g. 'cuda', 'cuda:0' or 'cpu'.",
        include_in_key=False,
        examples=["cuda", "cpu"],
    )

    @field_validator("excluded_amino_acids", mode="before")
    @classmethod
    def validate_excluded_amino_acids(cls, value: Any) -> Any:
        """Deduplicate the exclusion list and fold an empty list to None."""
        return normalize_excluded_amino_acids(value)

    @field_validator("model_checkpoint")
    @classmethod
    def validate_model_checkpoint(cls, value: str) -> str:
        """Reject a sidechain-packing checkpoint; design needs a ``task='seq_des'`` model."""
        return validate_checkpoint(value, DESIGN_CHECKPOINTS, "design")

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """A custom ``.ckpt`` path is not present on a hosted worker."""
        if self.model_checkpoint.endswith(".ckpt"):
            return (
                f"model_checkpoint points to a local .ckpt file not available on device='{device}'. "
                "Use a bundled checkpoint name, or run locally with device='cuda'."
            )
        return None


class CalibyDesignMetrics(Metrics):
    """Per-design Potts energy and native-sequence recovery for Caliby.

    Metrics documented in ``metric_spec``:
        potts_energy (float): Potts energy of the designed sequence, summed over
            positions. Probability is proportional to ``exp(-U/T)``, so lower is more
            favorable. Scales with sequence length. Always present.
        avg_potts_energy (float): ``potts_energy`` divided by sequence length, for
            comparison across structures of different sizes. Computed on the same basis
            as ``caliby-score`` so designed and native energies are comparable. Always
            present.
        sequence_recovery (float): Fraction of residues in the redesigned chains matching
            the input structure's native sequence. Positions pinned by ``fixed_positions``
            count toward the numerator, so a heavily constrained design reports a high
            floor. Always present.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "potts_energy": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "better_values_are": "lower",
        },
        "avg_potts_energy": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "better_values_are": "lower",
        },
        "sequence_recovery": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
    }
    primary_metric: str | None = Field(
        default="potts_energy",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class CalibyDesign(DesignedComplex):
    """One Caliby-designed complex.

    Attributes:
        chains (list[Chain | Fragment]): All entities of the design, in input structure
            chain order. Inherited from :class:`DesignedComplex`.
        structure (Structure): The input backbone with the designed residue identities
            threaded onto it. Sidechains are **not** generated during design; run
            ``caliby-pack`` on this structure to place them.
        metrics (CalibyDesignMetrics): Potts energy and native-sequence recovery.
    """

    structure: Structure = Field(
        title="Designed Structure",
        description="Input backbone with designed residue identities; sidechains not packed.",
    )
    metrics: CalibyDesignMetrics = Field(
        default_factory=CalibyDesignMetrics,
        title="Metrics",
        description="Per-design Potts energy and native-sequence recovery.",
    )


class CalibyDesignSet(DesignSet):
    """All Caliby complexes produced for a single input structure.

    Attributes:
        complexes (list[CalibyDesign]): The complexes designed for one input structure.
    """

    complexes: list[CalibyDesign] = Field(  # type: ignore[assignment]
        title="Complexes",
        description="Caliby complexes designed for one input structure.",
    )


class CalibySampleOutput(InverseFoldingOutput):
    """Output of the Caliby sequence design tool.

    Attributes:
        design_sets (list[CalibyDesignSet]): One ``CalibyDesignSet`` per input
            structure, in input order.
    """

    design_sets: list[CalibyDesignSet] = Field(  # type: ignore[assignment]
        title="Design Sets",
        description="One CalibyDesignSet per input structure, in input order.",
    )


@contextlib.contextmanager
def ensemble_paths(structures: list[Structure]) -> Iterator[tuple[Path, list[Path]]]:
    """Materialize an ensemble to temp PDB files, primary conformer first.

    PDB rather than CIF because Caliby reads chain IDs from ``label_asym_id``, which
    ``Structure.write_cif`` leaves unset, so a CIF arrives with every chain annotated
    ``'.'`` and no chain- or residue-level constraint can resolve.

    Args:
        structures (list[Structure]): The ensemble, as ``ensemble_structures`` returns it.

    Yields:
        tuple[Path, list[Path]]: The primary conformer's path, then the other members'
            paths in ensemble order (empty without an ensemble).
    """
    with contextlib.ExitStack() as stack:
        paths = [stack.enter_context(structure.temp_file("pdb")) for structure in structures]
        yield paths[0], paths[1:]


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CalibySampleInput(
        inputs=[
            CalibyStructureInput(
                structure=str(Path(__file__).parents[1] / "example_input_fixture.pdb"),  # type: ignore[arg-type]
            )
        ]
    )


@tool(
    key="caliby-sample",
    label="Caliby Sampling",
    category="inverse_folding",
    input_class=CalibySampleInput,
    config_class=CalibySampleConfig,
    output_class=CalibySampleOutput,
    description="Design protein sequences with Caliby, optionally conditioned on a conformer ensemble",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["inputs"],
    iterable_output_field="design_sets",
    max_chunk_size=32,
    cacheable=True,
    stochastic=True,
    metrics_class=CalibyDesignMetrics,
)
def run_caliby_sample(
    inputs: CalibySampleInput,
    config: CalibySampleConfig,
    instance: Any = None,
) -> CalibySampleOutput:
    """Design protein sequences with Caliby's Potts model.

    Caliby computes Potts parameters from backbone geometry and samples sequences by
    MCMC under the resulting energy. When an input supplies ``conformers``, the Potts
    parameters are aggregated position-wise across the ensemble, which produces
    sequences that are more diverse and more likely to refold to the target.

    Designs are backbone-only: the residue identities are threaded onto the input
    coordinates without sidechains. Pass ``design.structure`` to ``caliby-pack`` to
    place them.

    Args:
        inputs (CalibySampleInput): Structure inputs with optional conformers,
            ``chains_to_redesign`` and ``fixed_positions`` selections.
        config (CalibySampleConfig): Sampling configuration.
        instance (Any): Optional ToolInstance for persistent execution.

    Returns:
        CalibySampleOutput: One ``CalibyDesignSet`` per input structure, in input order.

    Examples:
        >>> inputs = CalibySampleInput(inputs=[CalibyStructureInput(structure="protein.pdb")])
        >>> output = run_caliby_sample(inputs, CalibySampleConfig(num_sequences_per_structure=4))
        >>> output[0][0].metrics["potts_energy"]
    """
    design_sets: list[CalibyDesignSet] = []

    base_seed = config.seed if config.seed is not None else config.get_random_int()
    # Advances across every dispatch (inputs x chunks) so duplicate items get distinct seeds.
    dispatch_idx = 0

    logger.debug("Using local venv for caliby sampling")

    # The flag only reaches ensemble_sample(), so a non-default value without conformers is inert.
    if not config.use_primary_res_type and not any(inp.conformers for inp in inputs.inputs):
        logger.warning(
            "use_primary_res_type=False has no effect: it applies only to ensemble-conditioned "
            "design, and no input supplies conformers."
        )

    for inp in progress_bar(
        inputs.inputs,
        desc="Caliby sampling",
        unit="structure",
        disable=not config.verbose,
    ):
        chain_ids = inp.structure.get_chain_ids()
        # Observed residues per chain. get_chain_sequences() pads each chain break with '-',
        # so its length overcounts any structure with unmodelled regions.
        native_residue_counts = {cid: len(inp.structure.get_chain_positions(cid)) for cid in chain_ids}
        redesign_set = set(inp.chain_ids_to_redesign)
        all_chain_seqs: list[list[str]] = []
        all_energies: list[float] = []
        all_recoveries: list[float] = []
        all_cifs: list[str] = []

        # One dispatch per structure: Caliby computes Potts parameters once and then samples
        # every sequence from them, so splitting the design count costs a re-encode and buys nothing.
        with ensemble_paths(inp.ensemble_structures) as (primary_path, conformer_paths):
            input_dict = {
                "operation": "sample",
                "primary_path": str(primary_path),
                "conformer_paths": [str(p) for p in conformer_paths],
                "chain_ids": chain_ids,
                "chains_to_redesign": inp.chain_ids_to_redesign,
                "fixed_positions": (
                    inp.fixed_positions.to_residue_numbers(inp.structure) if inp.fixed_positions is not None else None
                ),
                "symmetric_chains": (
                    [list(group.chains) for group in inp.symmetric_chains] if inp.symmetric_chains else None
                ),
                "num_sequences": config.num_sequences_per_structure,
                "batch_size": config.batch_size,
                "temperature": config.temperature,
                "potts_sweeps": config.potts_sweeps,
                "excluded_amino_acids": config.excluded_amino_acids,
                "use_primary_res_type": config.use_primary_res_type,
                "seed": base_seed + dispatch_idx,
                "model_checkpoint": config.model_checkpoint,
                "device": config.device,
                "verbose": config.verbose,
            }
            result = ToolInstance.dispatch("caliby", input_dict, instance=instance, config=config)
            all_chain_seqs.extend(result["chain_sequences"])
            all_energies.extend(result["potts_energy"])
            all_recoveries.extend(result["sequence_recovery"])
            all_cifs.extend(result["cif_strings"])
            dispatch_idx += 1

        complexes: list[CalibyDesign] = []
        for chain_seqs, energy, recovery, cif in zip(
            all_chain_seqs, all_energies, all_recoveries, all_cifs, strict=True
        ):
            if len(chain_seqs) != len(chain_ids):
                raise ValueError(
                    f"caliby-sample: model returned {len(chain_seqs)} chains but the input "
                    f"structure has {len(chain_ids)} chains ({chain_ids})"
                )
            for cid, seq in zip(chain_ids, chain_seqs, strict=True):
                expected = native_residue_counts[cid]
                if len(seq) != expected:
                    raise ValueError(
                        f"caliby-sample: designed chain {cid} has {len(seq)} residues but the input "
                        f"structure's chain {cid} has {expected}. Caliby's preprocessing dropped or "
                        "masked residues in this chain, so the design cannot be mapped back onto the "
                        "input numbering; check the chain for unresolved or non-standard residues."
                    )
            chains: list[Chain | Fragment] = [
                Chain(id=cid, sequence=seq) for cid, seq in zip(chain_ids, chain_seqs, strict=True)
            ]
            designed = [cid in redesign_set for cid in chain_ids]
            # The Potts energy covers every position in the graph, so normalize by the full length
            # to stay comparable with caliby-score on the same structure.
            total_length = sum(len(seq) for seq in chain_seqs)
            complexes.append(
                CalibyDesign(
                    chains=chains,
                    designed=designed,
                    structure=Structure(structure=cif, structure_format="cif", source="caliby-sample"),
                    metrics=CalibyDesignMetrics(
                        potts_energy=energy,
                        avg_potts_energy=energy / total_length if total_length else 0.0,
                        sequence_recovery=recovery,
                    ),
                )
            )

        design_sets.append(CalibyDesignSet(complexes=complexes))

    return CalibySampleOutput(design_sets=design_sets)

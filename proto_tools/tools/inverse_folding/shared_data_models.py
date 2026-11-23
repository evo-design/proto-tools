"""proto_tools/tools/inverse_folding/shared_data_models.py.

Shared base schemas (configs and io) for inverse folding tools.
"""

import contextlib
import csv
import json
import logging
from abc import ABC
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator
from typing_extensions import Self

from proto_tools.entities.complex import Chain, Complex, chain_label
from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures import (
    ChainSelection,
    ResidueSelection,
    Structure,
    StructureInputBase,
)
from proto_tools.tools.inverse_folding.structure_staging import StagedStructure, staged_structure
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)


class SequenceStructurePair(BaseModel):
    """Represents a sequence-structure pair for MPNN-based scoring.

    Used by both ProteinMPNN and LigandMPNN scoring tools to pair a protein
    sequence with its corresponding structure for computing compatibility metrics.

    Attributes:
        sequence (str): Protein sequence to score against the structure.
        structure (Structure): Protein structure to score the sequence against.
        excluded_positions (ResidueSelection | None): Per-chain 1-indexed positions left out
            of the aggregate scoring metrics. The residues still condition the model; they
            just do not contribute to the reported likelihood. Accepts ``{"A": [1, 2]}``.
            Sampling tools use ``fixed_positions`` instead, which holds an identity rather
            than omitting a position from a metric.
    """

    model_config = ConfigDict(extra="forbid")

    sequence: str = Field(title="Sequence", description="Sequence to score against the structure")
    structure: Structure = Field(title="Input Structure", description="Structure to score the sequence against")
    excluded_positions: ResidueSelection | None = Field(
        default=None,
        title="Excluded Positions",
        description="Per-chain 1-indexed positions left out of the aggregate scoring metrics.",
    )

    @model_validator(mode="after")
    def _validate_excluded_positions(self) -> "SequenceStructurePair":
        """Reject excluded positions that aren't real residues in ``structure``."""
        if self.excluded_positions is not None:
            self.excluded_positions.validate_against(self.structure, label="excluded_positions")
        return self


class InverseFoldingStructureInput(StructureInputBase):
    """Bundles a structure with per-structure design constraints.

    Pairs a protein structure with optional chain and residue selections that
    govern how the inverse-folding model treats each position. Structure
    resolution and selection-against-structure validation are inherited from
    :class:`StructureInputBase`.

    Attributes:
        structure (Structure): Protein structure. Accepts a file path, raw
            PDB/CIF content string, ``Structure`` object, or a dict in the shape
            produced by ``Structure.model_dump(mode='json')``.
        chains_to_redesign (ChainSelection | None): Chains to redesign. ``None`` means
            redesign every protein chain in the structure. Accepts shorthand ``"A"`` or
            ``["A", "B"]`` at construction.
        fixed_positions (ResidueSelection | None): Per-chain positions whose residue
            identity is held fixed during design (1-indexed). Accepts shorthand
            ``{"A": [1, 2, 3]}`` at construction.

    Examples:
        >>> # Simple structure from file (auto-loaded)
        >>> inp = InverseFoldingStructureInput(structure="/path/to/protein.pdb")
        >>> inp.structure_pdb  # access PDB string directly

        >>> # With chain selection and fixed positions (positions are 1-indexed)
        >>> inp = InverseFoldingStructureInput(
        ...     structure="/path/to/protein.pdb",
        ...     chains_to_redesign=["A", "B"],
        ...     fixed_positions={"A": [1, 2, 3], "B": [10, 11]},
        ... )
    """

    chains_to_redesign: ChainSelection | None = Field(
        default=None,
        title="Chains to Redesign",
        description="Chains to redesign. None = redesign every protein chain in the structure.",
    )
    fixed_positions: ResidueSelection | None = Field(
        default=None,
        title="Fixed Positions",
        description="Per-chain positions whose residue identity is held fixed (1-indexed).",
    )

    @property
    def chain_ids_to_redesign(self) -> list[str]:
        """Resolved list of chain IDs to design (defaults to every protein chain in ``structure``).

        Returns:
            list[str]: Chain IDs the model should redesign. When ``chains_to_redesign`` is
                ``None``, returns every protein chain ID in the input structure.
        """
        if self.chains_to_redesign is not None:
            return list(self.chains_to_redesign.chains)
        return [cid for cid, kind in self.structure.get_chain_types().items() if kind == "protein"]

    @property
    def structure_pdb(self) -> str:
        """PDB string of the structure (delegates to ``Structure.structure_pdb``)."""
        return self.structure.structure_pdb

    @contextlib.contextmanager
    def staged(self, *, atom_record_residues: frozenset[str] = frozenset()) -> Iterator[StagedStructure]:
        """Stage this input's structure for the model, numbered 1..N per chain.

        Translate this input's own selections with the yielded object, since staging renumbers
        residues and may shorten chain IDs:
        ``staged.chains(self.chain_ids_to_redesign)`` and ``staged.positions(self.fixed_positions)``.

        Args:
            atom_record_residues (frozenset[str]): Residue names to write as ``ATOM`` rather
                than ``HETATM``, for a model that reads the record type rather than a table of
                known modified residues.

        Yields:
            StagedStructure: The staged file and the maps needed to read model output back.
        """
        with staged_structure(self.structure, atom_record_residues=atom_record_residues) as staged:
            yield staged


class InverseFoldingInput(BaseToolInput):
    """Tool input for inverse folding sampling operations.

    Wraps a list of :class:`InverseFoldingStructureInput` for use with the
    ``ToolRegistry``.

    Attributes:
        inputs (list[InverseFoldingStructureInput]): Per-structure inputs, each
            containing a structure plus optional ``chains_to_redesign`` and ``fixed_positions``
            selections.

    Examples:
        >>> sampling_input = InverseFoldingInput(
        ...     inputs=[
        ...         InverseFoldingStructureInput(structure="/path/to/protein1.pdb"),
        ...         InverseFoldingStructureInput(
        ...             structure="/path/to/protein2.pdb",
        ...             chains_to_redesign=["A"],
        ...             fixed_positions={"A": [1, 2, 3]},
        ...         ),
        ...     ]
        ... )
    """

    inputs: list[InverseFoldingStructureInput] = InputField(
        title="Structure Inputs",
        description="Per-structure inputs, each containing a structure and optional selections.",
    )


def residue_label(structure: Any, chain_id: str, position: int) -> str:
    """Render a position as ``64 (CRO)`` for a diagnostic, falling back to the bare number."""
    try:
        residue = structure.get_chain_residue_atoms(chain_id)[chain_id][position - 1]
    except (KeyError, IndexError, ValueError):
        return str(position)
    return f"{position} ({residue.name})"


def summarize_positions(structure: Any, chain_id: str, positions: Sequence[int], limit: int = 4) -> str:
    """Render up to ``limit`` labelled positions, then a count, for a diagnostic."""
    labelled = [residue_label(structure, chain_id, position) for position in positions[:limit]]
    remainder = len(positions) - len(labelled)
    return ", ".join(labelled) + (f", and {remainder} more" if remainder > 0 else "")


def validate_residue_mapping(readable: dict[str, list[int]], structure: Structure) -> None:
    """Reject a malformed residue mapping before anything relies on it.

    Args:
        readable (dict[str, list[int]]): Positions the model processes, per chain ID.
        structure (Structure): The structure the mapping describes.

    Raises:
        ValueError: If a mapping is unordered, repeats a position, or falls outside its chain.
    """
    lengths = {chain_id: len(residues) for chain_id, residues in structure.get_chain_residue_atoms().items()}
    for chain_id, positions in readable.items():
        if chain_id not in lengths:
            raise ValueError(f"residue mapping names chain {chain_id}, which is not a polymer chain")
        if list(positions) != sorted(set(positions)):
            raise ValueError(f"residue mapping for chain {chain_id} must be ascending and unique")
        if positions and not (positions[0] >= 1 and positions[-1] <= lengths[chain_id]):
            raise ValueError(f"residue mapping for chain {chain_id} falls outside its {lengths[chain_id]} residues")


def readable_positions_where(structure: Structure, is_read: Callable[[Any], bool]) -> dict[str, list[int]]:
    """Positions a model reads, 1-indexed per chain, from a per-residue gate.

    Every polymer chain is reported, with an empty list where the model reads none of it, so
    a caller can tell a chain the model skips entirely from one the structure does not have.

    Args:
        structure (Structure): The structure that will be staged for the model.
        is_read (Callable[[Any], bool]): Whether the model reads one ``ResidueAtoms``.

    Returns:
        dict[str, list[int]]: Ascending readable positions per polymer chain ID.
    """
    return {
        chain_id: [position for position, residue in enumerate(residues, start=1) if is_read(residue)]
        for chain_id, residues in structure.get_chain_residue_atoms().items()
    }


def unreadable_positions(
    readable: dict[str, list[int]], structure: Structure, chain_ids: Sequence[str] | None = None
) -> dict[str, list[int]]:
    """Polymer positions a model's rule does not read, per chain, ascending.

    The complement of a ``readable_positions`` mapping. These are the positions whose input
    residue is preserved rather than designed, so passing them as ``fixed_positions`` to
    :meth:`DesignedComplex.from_design` is what lets a modification annotation survive there.

    Args:
        readable (dict[str, list[int]]): Positions the model processes, per chain ID.
        structure (Structure): The structure the mapping describes.
        chain_ids (Sequence[str] | None): Report only these chains, or every polymer chain.

    Returns:
        dict[str, list[int]]: Skipped positions per chain; chains with none are omitted.
    """
    wanted = None if chain_ids is None else set(chain_ids)
    skipped: dict[str, list[int]] = {}
    for chain_id, residues in structure.get_chain_residue_atoms().items():
        if wanted is not None and chain_id not in wanted:
            continue
        processed = set(readable.get(chain_id, ()))
        missing = [position for position in range(1, len(residues) + 1) if position not in processed]
        if missing:
            skipped[chain_id] = missing
    return skipped


def preserved_positions(
    readable: dict[str, list[int]], structure: Structure, selection: Any = None
) -> dict[str, list[int]]:
    """Return positions preserved by the parser or fixed by the caller.

    Build ``fixed_positions`` for :meth:`DesignedComplex.from_design`. Include the caller's
    fixed positions as well as parser exclusions so reconstruction checks fixed residues
    and preserves their modification annotations.

    Args:
        readable (dict[str, list[int]]): Positions the model processes, per chain ID.
        structure (Structure): The structure the mapping describes.
        selection (Any): The caller's ``ResidueSelection``, or ``None``.

    Returns:
        dict[str, list[int]]: Preserved positions per chain, ascending, omitting chains with
            none.
    """
    preserved = unreadable_positions(readable, structure)
    requested = (selection.chains if selection is not None else {}) or {}
    for chain_id, positions in requested.items():
        combined = set(preserved.get(chain_id, ())) | set(positions)
        if combined:
            preserved[chain_id] = sorted(combined)
    return preserved


def flat_readable_positions(readable: dict[str, list[int]], structure: Structure) -> list[int]:
    """The positions a model reads, numbered across the whole structure, in chain order.

    A model that skips residues numbers what it reads from 1 with no gaps, so its *n*-th
    position is this list's *n*-th entry. Entry ``i`` is 1-indexed over every polymer residue
    of the structure, counting chains in the order ``get_chain_residue_atoms`` reports them,
    which is the order the parser sees them in the staged file.

    Args:
        readable (dict[str, list[int]]): Positions the model reads, per chain ID, each
            1-indexed within its own chain.
        structure (Structure): The structure the mapping describes.

    Returns:
        list[int]: Structure-wide 1-indexed positions, ascending.
    """
    flat: list[int] = []
    offset = 0
    for chain_id, residues in structure.get_chain_residue_atoms().items():
        flat.extend(offset + position for position in readable.get(chain_id, ()))
        offset += len(residues)
    return flat


def split_fixed_positions(
    selection: Any, readable: dict[str, list[int]], *, renumber: bool
) -> tuple[Any, dict[str, list[int]]]:
    """Restrict a fixed-position selection to the positions the model can address.

    A model cannot fix a residue it never reads, so those positions are reported back to the
    caller's tool rather than sent on; the residue keeps its input identity through
    ``readable_positions`` at reconstruction. Whether the kept positions need renumbering
    depends on how the model addresses a residue: by its ordinal among what it read, or by
    the residue number the staged file gives it.

    Args:
        selection (Any): The caller's ``ResidueSelection``, or ``None``.
        readable (dict[str, list[int]]): Positions the model reads, per chain ID.
        renumber (bool): Whether to renumber kept positions into the model's own numbering.

    Returns:
        tuple[Any, dict[str, list[int]]]: The selection the model can resolve, or ``None``
            when nothing is left, and the dropped positions per chain in the caller's
            numbering, omitting chains that lost none.
    """
    if selection is None:
        return None, {}
    kept_chains: dict[str, list[int]] = {}
    dropped_chains: dict[str, list[int]] = {}
    for chain_id, positions in selection.chains.items():
        chain_readable = readable.get(chain_id, [])
        kept = (
            to_model_numbering(chain_readable, positions) if renumber else sorted(set(positions) & set(chain_readable))
        )
        dropped = sorted(set(positions) - set(chain_readable))
        if kept:
            kept_chains[chain_id] = kept
        if dropped:
            dropped_chains[chain_id] = dropped
    resolved = type(selection)(chains=kept_chains) if kept_chains else None
    return resolved, dropped_chains


def to_model_numbering(readable: Sequence[int], positions: Iterable[int]) -> list[int]:
    """Renumber positions from the structure's numbering into the model's own.

    The inverse of indexing into ``readable``. A model that skips residues numbers what it
    reads from 1 with no gaps, so a caller's position has to be renumbered before it can
    name the same residue to the model. Positions the model does not read have no
    counterpart and are dropped: it cannot act on a residue it never saw.

    Args:
        readable (Sequence[int]): Positions the model reads, ascending, in the caller's
            numbering. Either one chain's or a whole structure's, matching ``positions``.
        positions (Iterable[int]): Positions in the caller's numbering.

    Returns:
        list[int]: The same residues in the model's 1-indexed numbering, ascending.
    """
    rank = {position: index for index, position in enumerate(readable, start=1)}
    return sorted(rank[position] for position in positions if position in rank)


class InverseFoldingConfig(BaseConfig):
    """Base configuration for inverse folding models.

    Contains model hyperparameters for sequence generation. Structure-specific
    selections (``chains_to_redesign``, ``fixed_positions``) are specified on each
    :class:`InverseFoldingStructureInput` in :class:`InverseFoldingInput`.

    Attributes:
        num_sequences_per_structure (int): Total number of sequences to generate
            for each input structure. Sequences are generated in batches of
            ``batch_size``. Defaults to 1.

        batch_size (int | None): Number of sequences to process simultaneously
            on GPU. Larger batches improve throughput but use more GPU memory;
            reduce if encountering out-of-memory errors. Defaults to
            ``num_sequences_per_structure``.

        temperature (float): Sampling temperature; lower = greedier, higher = more diverse.

        device (str): Device to run the model on. Options include 'cuda' (NVIDIA GPU), 'cpu' (CPU execution), or specific GPU devices like 'cuda:0'. Defaults to 'cuda'.
    """

    num_sequences_per_structure: int = ConfigField(
        title="Sequences Per Structure",
        default=1,
        ge=1,
        description="Total number of sequences to generate per input structure.",
    )
    batch_size: int | None = ConfigField(
        title="Batch Size",
        default=None,
        ge=1,
        description="Number of sequences to process simultaneously on GPU. Defaults to num_sequences_per_structure.",
    )

    def readable_positions(self, inp: Any) -> dict[str, list[int]] | None:
        """Positions this model processes, 1-indexed, per chain.

        Structure parsers disagree about modified residues: ProteinMPNN renames the residues in
        ColabDesign's MODRES table and drops anything without a backbone N, ESM-IF1 keeps only
        names biotite maps to one letter, LigandMPNN's rule depends on ``model_type``. A
        toolkit overrides this with its own model's rule.

        ``None`` means no rule is declared, which keeps a toolkit's existing strict behaviour.

        Args:
            inp (Any): One structure input, exposing ``structure``.

        Returns:
            dict[str, list[int]] | None: Ascending positions per chain ID, or ``None``.
        """
        del inp
        return None

    @model_validator(mode="after")
    def resolve_batch_size(self) -> Any:
        """Default batch_size to num_sequences_per_structure if not set."""
        if self.batch_size is None:
            self.batch_size = self.num_sequences_per_structure
        return self

    temperature: float = ConfigField(
        title="Sampling Temperature",
        default=0.1,
        ge=0.0,
        description="Sampling temperature; lower = greedier, higher = more diverse",
        examples=[0.1, 0.5, 1.0],
    )

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on. Options include 'cuda' (NVIDIA GPU), 'cpu' (CPU execution)",
        include_in_key=False,
        examples=["cuda", "cpu"],
    )


def _surviving_modifications(entry: Any, readable: list[int] | None, fixed: list[int] | None) -> list[Any]:
    """Modification annotations that still describe the designed chain.

    Keep annotations at positions excluded by the parser or fixed by the caller. Remove
    them at freely redesigned positions even if the model returns the same letter. For
    example, an unfixed MSE returned as ``M`` is a designed methionine.

    Args:
        entry (Any): The input chain.
        readable (list[int] | None): Positions the model processed, or ``None`` if undeclared.
        fixed (list[int] | None): Positions kept unchanged, excluded ones included.

    Returns:
        list[Any]: The annotations to carry into the output chain.
    """
    annotations = getattr(entry, "modifications", None) or []
    if not annotations or readable is None:
        # No declared mapping means every position was designed, as before this feature.
        return []
    kept = set(fixed or ())
    return [modification for modification in annotations if modification.position in kept]


def _check_fixed_positions_honoured(
    entry: Any, sequence: str, readable: list[int], fixed: list[int], tool: str
) -> None:
    """Reject a design that changed a residue the caller fixed.

    Args:
        entry (Any): The input chain.
        sequence (str): The reconstructed full-length sequence.
        readable (list[int]): Positions the model processed.
        fixed (list[int]): Positions kept unchanged, excluded ones included.
        tool (str): Tool key named in the error.

    Raises:
        ValueError: If a fixed, processed position came back different.
    """
    changed = [
        position
        for position in sorted(set(fixed) & set(readable))
        if sequence[position - 1] != entry.sequence[position - 1]
    ]
    if changed:
        detail = ", ".join(f"{p} ({entry.sequence[p - 1]}->{sequence[p - 1]})" for p in changed)
        raise ValueError(
            f"{tool}: Chain {entry.id} position {detail} was fixed but the model returned a "
            "different residue. Refusing to return a design that violates the requested constraints."
        )


def _splice_design(native: str, designed: str, positions: list[int]) -> str:
    """Place a model's residues at the positions it read, keeping the input's elsewhere.

    Args:
        native (str): The chain's input sequence, one character per polymer residue.
        designed (str): The model's sequence, one character per entry of ``positions``.
        positions (list[int]): 1-indexed positions of ``native`` the model read, ascending.

    Returns:
        str: ``native`` with the designed residues substituted at ``positions``.

    Raises:
        ValueError: If a position falls outside ``native``.
    """
    spliced = list(native)
    for position, residue in zip(positions, designed, strict=True):
        if not 1 <= position <= len(native):
            raise ValueError(f"readable position {position} falls outside a chain of {len(native)} residues")
        spliced[position - 1] = residue
    return "".join(spliced)


class DesignedComplex(Complex):
    """One complete designed complex produced from a single input structure.

    Subclass of the shared :class:`Complex`. Reproduces the input structure's
    polymer chains in input order, exactly as ``Complex.from_structure`` reads
    them, with the redesigned chains' sequences substituted; build one with
    :meth:`from_design`. Ligands are included only for models that condition on
    them. Per-chain redesign status is tracked as a parallel boolean list
    (``designed``), aligned with ``chains`` by position. Ligand ``Fragment``
    entries are always context (``designed[i] is False``); the IF models don't
    redesign ligands.

    ``Complex`` accepts a ``DesignedComplex`` directly
    (real ``isinstance`` via the shared base), so a design feeds a structure
    predictor with no manual reassembly.

    Each inverse folding model defines a :class:`Metrics` subclass declaring its
    per-design ``metric_spec`` (perplexity, sequence recovery, log-likelihood,
    ...) and assigns it to ``metrics``; subclasses narrow the ``metrics`` type.

    Attributes:
        chains (list[Chain | Fragment]): All entities of the design, in input
            structure chain order. Inherited from :class:`Complex`.
        designed (list[bool]): Per-chain redesign status, aligned by position
            with ``chains`` (``len(designed) == len(chains)``). ``True`` for
            chains the model designed; ``False`` for fixed context chains and
            for all ligand entries.
        metrics (SerializeAsAny[Metrics]): Per-design metrics. Each tool assigns
            its own ``Metrics`` subclass; values round-trip via ``Metrics``
            ``extra="allow"``.
    """

    designed: list[bool] = Field(
        title="Designed",
        description="Per-chain redesign status, aligned by position with chains.",
    )
    metrics: SerializeAsAny[Metrics] = Field(
        default_factory=Metrics,
        title="Metrics",
        description="Per-design metrics (model-specific Metrics subclass).",
    )

    @model_validator(mode="after")
    def _validate_designed_length(self) -> "DesignedComplex":
        """Enforce that ``designed`` is aligned 1:1 with ``chains``."""
        if len(self.designed) != len(self.chains):
            raise ValueError(
                f"designed has {len(self.designed)} entries but chains has {len(self.chains)}; "
                "the two lists must be the same length (one bool per chain entry)."
            )
        return self

    @classmethod
    def from_design(
        cls,
        input_structure: Structure,
        sequences: dict[str, str],
        designed: set[str],
        *,
        tool: str,
        include_ligands: bool,
        readable_positions: dict[str, list[int]] | None = None,
        fixed_positions: dict[str, list[int]] | None = None,
        **fields: Any,
    ) -> Self:
        """Build a design from its input structure and the model's redesigned sequences.

        Every chain starts as ``Complex.from_structure`` reads it from ``input_structure``, so
        context chains keep their exact residues, type, and modifications whichever model
        produced the design. Each chain in ``sequences`` takes the model's sequence as a
        protein chain without modifications.

        Args:
            input_structure (Structure): The structure the model designed against.
            sequences (dict[str, str]): Model-emitted sequence per chain ID to substitute.
            designed (set[str]): Chain IDs flagged as designed in ``designed``.
            tool (str): Tool key, named in errors.
            include_ligands (bool): Whether the input's ligands are kept as ``Fragment``
                entries; true only for models that condition on ligands.
            readable_positions (dict[str, list[int]] | None): Per chain, the 1-indexed
                positions this model reads, from ``Config.readable_positions``. A model that
                skips a modified residue returns one residue per readable position rather
                than one per polymer residue; the design is then placed on those positions
                and the skipped residues keep their input identity. ``None`` requires one
                residue per polymer residue.
            fixed_positions (dict[str, list[int]] | None): Positions that will not be
                redesigned, whether the caller fixed them or the parser skipped them, as
                :func:`preserved_positions` builds them. A modification annotation survives at
                one of these when the output residue still matches its parent amino acid.
            fields (Any): Remaining fields of the subclass, such as ``metrics``.

        Returns:
            Self: The design, with one ``designed`` flag per chain entry.

        Raises:
            ValueError: If a designed chain has no sequence, a sequence names a chain that
                is not a polymer chain of ``input_structure``, or its length differs from that
                chain's observed residues.
        """
        # Start with the input chains so context and untouched residues retain their metadata.
        native = Complex.from_structure(input_structure, include_ligands=include_ligands).chains
        # Reject sequences for non-polymer chains and designed chains with no sequence
        polymer_ids = {entry.id for entry in native if isinstance(entry, Chain)}
        unknown = sorted(set(sequences) - polymer_ids)
        if unknown:
            raise ValueError(f"{tool}: designed chain(s) {unknown} are not polymer chains of the input structure")
        missing = sorted(designed - set(sequences))
        if missing:
            raise ValueError(f"{tool}: the model returned no sequence for designed chain(s) {missing}")
        # Substitute each redesigned chain's sequence and pass every other entry through
        chains: list[Chain | Fragment] = []
        for entry in native:
            if isinstance(entry, Chain) and entry.id in sequences:
                sequence = sequences[entry.id]
                # Look up the positions this model reads, which must cover every redesigned chain
                readable = (readable_positions or {}).get(entry.id)
                if readable_positions is not None and readable is None:
                    raise ValueError(
                        f"{tool}: Chain {entry.id} was redesigned but is missing from the residue "
                        "mapping, so the returned sequence cannot be aligned to the structure."
                    )
                if readable is not None:
                    # Any other length would mean guessing which positions the design covers.
                    if len(sequence) != len(readable):
                        raise ValueError(
                            f"{tool}: the model returned {len(sequence)} residues for chain "
                            f"{entry.id}, but this toolkit's readable_positions rule says it reads "
                            f"{len(readable)} of that chain's {len(entry.sequence)} residues. The "
                            "design cannot be placed back on the chain unless those agree. The rule "
                            "most likely no longer matches the installed version of the model."
                        )
                    # Restore residues skipped by the parser around the model's designed positions.
                    sequence = _splice_design(entry.sequence, sequence, readable)
                # Without a readable_positions rule, require one residue per input residue
                elif len(sequence) != len(entry.sequence):
                    raise ValueError(
                        f"{tool}: the model returned {len(sequence)} residues for chain {entry.id}, "
                        f"but that chain has {len(entry.sequence)}. Models skip residues they cannot "
                        "read, such as a chromophore or a residue missing backbone atoms, and which "
                        "ones differs by model. This toolkit has not declared a readable_positions "
                        "rule, so the design cannot be placed back on the chain."
                    )
                # Check fixed residues were kept and keep only modifications that still apply
                chain_fixed = (fixed_positions or {}).get(entry.id) or []
                if readable is not None:
                    _check_fixed_positions_honoured(entry, sequence, readable, chain_fixed, tool)
                preserved = _surviving_modifications(entry, readable, chain_fixed)
                chains.append(Chain(id=entry.id, sequence=sequence, entity_type="protein", modifications=preserved))
            else:
                chains.append(entry)
        # Flag the chains that were designed, aligned with the chain list
        flags = [isinstance(entry, Chain) and entry.id in designed for entry in chains]
        return cls(chains=chains, designed=flags, **fields)

    @property
    def designed_chains(self) -> list[Chain | Fragment]:
        """Only the chains the model actually redesigned."""
        return [c for c, r in zip(self.chains, self.designed, strict=True) if r]

    def design_metrics(self) -> dict[str, Any]:
        """Per-design metric values for this complex."""
        return dict(self.metrics.items())


class DesignSet(BaseModel, ABC):
    """All complexes produced for a single input structure.

    A model generates ``num_sequences_per_structure`` complexes per input; each is a
    :class:`DesignedComplex`. Concrete subclasses narrow ``complexes`` to their
    model-specific :class:`DesignedComplex` subclass for ``model_validate``.

    Attributes:
        complexes (list[SerializeAsAny[DesignedComplex]]): The complexes generated for one input,
            each a complete multi-chain complex.
    """

    complexes: list[SerializeAsAny[DesignedComplex]] = Field(
        title="Complexes",
        description="Designs generated for one input structure, each a complete complex.",
    )

    def __len__(self) -> int:
        """Get the number of complexes."""
        return len(self.complexes)

    def __getitem__(self, index: int) -> DesignedComplex:
        """Get a design by index."""
        return self.complexes[index]

    def __iter__(self) -> Iterator[DesignedComplex]:  # type: ignore[override]
        """Iterate over the complexes."""
        return iter(self.complexes)

    def __repr__(self) -> str:
        """Get a string representation of the design set."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the design set."""
        return f"DesignSet(with {len(self.complexes)} complexes)"

    def get_design_metrics(self, index: int) -> dict[str, Any]:
        """Get the per-design scalar metrics for the design at ``index``."""
        return self.complexes[index].design_metrics()


class InverseFoldingOutput(BaseToolOutput):
    """Output object for inverse folding models.

    Contains one :class:`DesignSet` per input structure, in input order.

    Attributes:
        design_sets (list[SerializeAsAny[DesignSet]]): One ``DesignSet`` per input
            structure. Entry ``i`` holds all complexes for input structure ``i``.
    """

    # Concrete tools narrow the type for validation. SerializeAsAny controls serialization.
    design_sets: list[SerializeAsAny[DesignSet]] = Field(
        title="Design Sets",
        description="One DesignSet per input structure, in input order.",
    )

    def __len__(self) -> int:
        """Get the number of input structures designed for."""
        return len(self.design_sets)

    def __getitem__(self, index: int) -> DesignSet:
        """Get the DesignSet for an input structure by index."""
        return self.design_sets[index]

    def __iter__(self) -> Iterator[DesignSet]:  # type: ignore[override]
        """Iterate over the per-input DesignSets."""
        return iter(self.design_sets)

    def __repr__(self) -> str:
        """Get a string representation of the output."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the output."""
        return f"InverseFoldingOutput(with {len(self.design_sets)} input structures)"

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["fasta", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path)

        if file_format == "fasta":
            path.mkdir(parents=True, exist_ok=True)
            for i, design_set in enumerate(self.design_sets):
                out_file = path / f"input_{i}.fasta"
                with open(out_file, "w") as f:
                    for j, design in enumerate(design_set.complexes):
                        # Write polymer sequences and ligand SMILES with `_chain_` or `_ligand_` headers.
                        for k, entry in enumerate(design.chains):
                            entry_id = entry.id if entry.id is not None else chain_label(k)
                            if isinstance(entry, Chain):
                                f.write(f">input_{i}_design_{j}_chain_{entry_id}\n{entry.sequence}\n")
                            else:
                                ccd_suffix = f"_{entry.ccd_code}" if entry.ccd_code else ""
                                f.write(f">input_{i}_design_{j}_ligand_{entry_id}{ccd_suffix}\n{entry.smiles}\n")

        elif file_format == "json":
            path.mkdir(parents=True, exist_ok=True)

            # Handle potential numpy types in metrics
            def default(obj: Any) -> Any:
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                return str(obj)

            for i, design_set in enumerate(self.design_sets):
                out_file = path / f"input_{i}.json"
                with open(out_file, "w") as f:
                    json.dump(design_set.model_dump(), f, indent=2, default=default)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Scoring Data Models
# ============================================================================
class InverseFoldingScoringMetrics(Metrics):
    """Per-sequence scoring metrics for inverse-folding scorers.

    Shared across ESM-IF1, ProteinMPNN, and LigandMPNN scoring — all three emit
    the same scalar set against a given sequence-structure pair.

    Metrics documented in ``metric_spec``:
        log_likelihood (float): Sum of per-position log-likelihoods given the
            target structure. Always present.
        avg_log_likelihood (float): Mean per-position log-likelihood.
            Always present.
        perplexity (float): exp(-avg_log_likelihood). Always present. Range ``[1, ∞)``.

    Attributes:
        logits (list[list[float]] | None): Per-position logits array
            ``(seq_len, vocab_size)``. ``None`` unless the tool returns logits.
        vocab (list[str] | None): Token ordering for ``logits``.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "log_likelihood": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": 0.0,
            "better_values_are": "higher",
        },
        "avg_log_likelihood": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": 0.0,
            "better_values_are": "higher",
        },
        "perplexity": {
            "availability": "always",
            "type": "float",
            "min": 1.0,
            "max": None,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = Field(
        default="perplexity",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )

    logits: list[list[float]] | None = Field(
        default=None,
        title="Logits",
        description="Per-position logits array as nested list (seq_len, vocab_size)",
    )
    vocab: list[str] | None = Field(
        default=None,
        title="Vocabulary",
        description="Token ordering for logits: logits[:, j] corresponds to vocab[j]",
    )


class InverseFoldingScoringOutput(BaseToolOutput):
    """Standardized output for inverse folding scoring tools.

    Contains scoring results for sequence-structure pairs evaluated by
    ESM-IF1, ProteinMPNN, or LigandMPNN scoring.

    Attributes:
        scores (list[InverseFoldingScoringMetrics]): List of scoring outputs,
            one per input sequence-structure pair. Each entry is a ``Metrics``
            subclass with scalar metrics (accessed via ``score.perplexity`` or
            ``score["perplexity"]``) plus declared ``logits`` / ``vocab`` fields.
    """

    scores: list[InverseFoldingScoringMetrics] = Field(
        title="Scores",
        description="List of scoring outputs, one per input sequence-structure pair",
    )

    @property
    def vocab(self) -> list[str] | None:
        """Token ordering for logits; derived from first score."""
        return self.scores[0].vocab if self.scores else None

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, index: int) -> InverseFoldingScoringMetrics:
        return self.scores[index]

    def __iter__(self) -> Iterator[InverseFoldingScoringMetrics]:  # type: ignore[override]
        return iter(self.scores)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":

            def default(obj: Any) -> Any:
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                return str(obj)

            data = []
            for s in self.scores:
                score_data: dict[str, Any] = dict(s.items())
                if s.logits is not None:
                    score_data["logits"] = s.logits
                if s.vocab is not None:
                    score_data["vocab"] = s.vocab
                data.append(score_data)

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=default)

        elif file_format == "csv":
            if self.scores:
                fieldnames = list(self.scores[0].keys())
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for s in self.scores:
                        writer.writerow(dict(s.items()))
        else:
            raise ValueError(f"Unsupported format: {file_format}")

"""proto_tools/tools/inverse_folding/shared_data_models.py.

Shared base schemas (configs and io) for inverse folding tools.
"""

import contextlib
import csv
import json
from abc import ABC
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, SerializeAsAny, model_validator
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


class SequenceStructurePair(BaseModel):
    """Represents a sequence-structure pair for MPNN-based scoring.

    Used by both ProteinMPNN and LigandMPNN scoring tools to pair a protein
    sequence with its corresponding structure for computing compatibility metrics.

    Attributes:
        sequence (str): Protein sequence to score against the structure.
        structure (Structure): Protein structure to score the sequence against.
        fixed_positions (ResidueSelection | None): Per-chain 1-indexed positions
            excluded from the aggregate scoring metrics. Accepts ``{"A": [1, 2]}``.
    """

    sequence: str = Field(title="Sequence", description="Sequence to score against the structure")
    structure: Structure = Field(title="Input Structure", description="Structure to score the sequence against")
    fixed_positions: ResidueSelection | None = Field(
        default=None,
        title="Fixed Positions",
        description="Per-chain 1-indexed positions excluded from scoring metrics.",
    )

    @model_validator(mode="after")
    def _validate_fixed_positions(self) -> "SequenceStructurePair":
        """Reject fixed positions that aren't real residues in ``structure``."""
        if self.fixed_positions is not None:
            self.fixed_positions.validate_against(self.structure, label="fixed_positions")
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
    def staged(self) -> Iterator[StagedStructure]:
        """Stage this input's structure for the model, numbered 1..N per chain.

        Translate this input's own selections with the yielded object, since staging renumbers
        residues and may shorten chain IDs:
        ``staged.chains(self.chain_ids_to_redesign)`` and ``staged.positions(self.fixed_positions)``.

        Yields:
            StagedStructure: The staged file and the maps needed to read model output back.
        """
        with staged_structure(self.structure) as staged:
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


class ReadableResiduesMixin:
    """Config mixin: declare which residues this model reads, and check selections against it.

    Structure-reading models disagree about modified residues. ProteinMPNN rewrites the 53
    residues in ColabDesign's ``MODRES`` table from ``HETATM`` to ``ATOM`` and then drops
    anything lacking a backbone ``N``; ESM-IF1 ignores the record type and keeps only names
    biotite can map to one letter; LigandMPNN uses two different rules depending on
    ``model_choice``. A single repository-wide rule is therefore wrong for everyone, and a
    rule derived from the staged file cannot see the config at all.

    A toolkit overrides :meth:`readable_positions` with its own model's rule. Returning
    ``None`` -- the default -- means "no rule declared", and nothing is checked, which is
    how a toolkit that has not been migrated keeps its current behaviour.

    The check runs in ``preprocess``, so a selection the model cannot honour fails before
    anything is staged or dispatched, rather than as an assertion from inside the model.
    """

    def readable_positions(self, inp: Any) -> dict[str, list[int]] | None:
        """Positions this model will actually read, 1-indexed, per chain.

        Args:
            inp (Any): One structure input, exposing ``structure``.

        Returns:
            dict[str, list[int]] | None: Ascending positions per chain ID, or ``None`` when
                the toolkit declares no rule.
        """
        del inp  # the base declares no rule; overrides read the structure off it
        return None

    def preprocess(self, inputs: BaseToolInput) -> "BaseToolInput | tuple[BaseToolInput, Any]":
        """Reject position selections the model cannot read, then defer to the base hook.

        Args:
            inputs (BaseToolInput): Tool inputs to prepare.

        Returns:
            BaseToolInput | tuple[BaseToolInput, Any]: Whatever the base hook returns.
        """
        for inp in getattr(inputs, "inputs", None) or []:
            readable = self.readable_positions(inp)
            if readable is not None:
                _reject_unreadable_selections(inp, readable)
        # Cooperative call into BaseConfig.preprocess; untyped here because the mixin
        # deliberately does not inherit it, so Pydantic builds configs from BaseConfig alone.
        return super().preprocess(inputs)  # type: ignore[no-any-return,misc]


def _reject_unreadable_selections(inp: Any, readable: dict[str, list[int]]) -> None:
    """Raise if any per-position selection on ``inp`` names a position the model cannot read.

    Walks the input's own fields rather than taking a list of names, so a toolkit that adds
    another position selection is covered without touching this code. Both a plain
    ``{chain: [positions]}`` selection and one keyed by position, such as Caliby's
    ``pinned_residues``, are read the same way, since iterating the inner mapping yields
    its positions.

    Args:
        inp (Any): One structure input.
        readable (dict[str, list[int]]): Positions the model can read, per chain ID.

    Raises:
        ValueError: If a selected position is not one the model reads.
    """
    for name in type(inp).model_fields:
        chains = getattr(getattr(inp, name, None), "chains", None)
        if not isinstance(chains, dict):
            continue  # ChainSelection holds a list of chain IDs, not per-chain positions
        for chain_id, positions in chains.items():
            unreadable = sorted(p for p in positions if p not in set(readable.get(chain_id, ())))
            if unreadable:
                raise ValueError(
                    f"{name} {unreadable} on chain {chain_id} name residues this model does not "
                    "read, so it cannot honour them. This is usually a modified residue, such as "
                    "a chromophore, or a residue missing the backbone atoms the model requires."
                )


class InverseFoldingConfig(ReadableResiduesMixin, BaseConfig):
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
            fields (Any): Remaining fields of the subclass, such as ``metrics``.

        Returns:
            Self: The design, with one ``designed`` flag per chain entry.

        Raises:
            ValueError: If a designed chain has no sequence, a sequence names a chain that
                is not a polymer chain of ``input_structure``, or its length differs from that
                chain's observed residues.
        """
        native = Complex.from_structure(input_structure, include_ligands=include_ligands).chains
        polymer_ids = {entry.id for entry in native if isinstance(entry, Chain)}
        unknown = sorted(set(sequences) - polymer_ids)
        if unknown:
            raise ValueError(f"{tool}: designed chain(s) {unknown} are not polymer chains of the input structure")
        missing = sorted(designed - set(sequences))
        if missing:
            raise ValueError(f"{tool}: the model returned no sequence for designed chain(s) {missing}")
        chains: list[Chain | Fragment] = []
        for entry in native:
            if isinstance(entry, Chain) and entry.id in sequences:
                sequence = sequences[entry.id]
                readable = (readable_positions or {}).get(entry.id)
                if len(sequence) != len(entry.sequence) and readable is not None and len(sequence) == len(readable):
                    # The model skipped residues it cannot represent -- a chromophore, a
                    # residue with no backbone N. Place the design on the positions it did
                    # read: dropping or shifting the others would misalign everything after
                    # the first skip, which looks like a plausible sequence rather than a bug.
                    sequence = _splice_design(entry.sequence, sequence, readable)
                if len(sequence) != len(entry.sequence):
                    raise ValueError(
                        f"{tool}: designed chain {entry.id} has {len(sequence)} residues but the input "
                        f"structure's chain {entry.id} has {len(entry.sequence)}. The model read fewer "
                        "residues than the chain holds; which residues a model skips differs by model, and "
                        "commonly includes non-standard residues such as selenomethionine and residues "
                        "missing backbone atoms."
                        + (
                            f" This model was expected to read {len(readable)} of them; if that count "
                            "is wrong, the toolkit's readable_positions rule needs correcting."
                            if readable is not None
                            else ""
                        )
                    )
                chains.append(Chain(id=entry.id, sequence=sequence, entity_type="protein"))
            else:
                chains.append(entry)
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

    # SerializeAsAny covers model_dump only; concrete tools narrow this for model_validate.
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
                        # Biopolymers as sequences, ligand Fragments as SMILES; header convention `_chain_` vs `_ligand_` lets consumers filter by type.
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

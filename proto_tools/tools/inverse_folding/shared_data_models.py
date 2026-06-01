"""proto_tools/tools/inverse_folding/shared_data_models.py.

Shared base schemas (configs and io) for inverse folding tools.
"""

import csv
import json
from abc import ABC
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, SerializeAsAny, model_validator

from proto_tools.entities.complex import Chain, Complex, chain_label
from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures import (
    ChainSelection,
    ResidueSelection,
    Structure,
    StructureInputBase,
)
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
            redesign every chain in the structure. Accepts shorthand ``"A"`` or
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
        description="Chains to redesign. None = redesign every chain in the structure.",
    )
    fixed_positions: ResidueSelection | None = Field(
        default=None,
        title="Fixed Positions",
        description="Per-chain positions whose residue identity is held fixed (1-indexed).",
    )

    @property
    def chain_ids_to_redesign(self) -> list[str]:
        """Resolved list of chain IDs to design (defaults to every chain in ``structure``).

        Returns:
            list[str]: Chain IDs the model should redesign. When ``chains_to_redesign`` is
                ``None``, returns every chain ID in the input structure.
        """
        if self.chains_to_redesign is not None:
            return list(self.chains_to_redesign.chains)
        return self.structure.get_chain_ids()

    @property
    def structure_pdb(self) -> str:
        """PDB string of the structure (delegates to ``Structure.structure_pdb``)."""
        return self.structure.structure_pdb


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


class DesignedComplex(Complex):
    """One complete designed complex produced from a single input structure.

    Subclass of the shared :class:`Complex`. Faithfully reproduces the input
    structure's entities (protein/DNA/RNA chains + ligands), in input order,
    with the redesigned chains' sequences substituted. Per-chain redesign status
    is tracked as a parallel boolean list (``designed``), aligned with
    ``chains`` by position. Ligand ``Fragment`` entries are always context
    (``designed[i] is False``); the IF models don't redesign ligands.

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

    Shared across ProteinMPNN, LigandMPNN, and FAMPnn scoring — all three emit
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
    ProteinMPNN, LigandMPNN, or FAMPnn scoring.

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

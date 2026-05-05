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

from proto_tools.entities.structures import Structure
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
    """

    sequence: str = Field(description="Sequence to score against the structure")
    structure: Structure = Field(description="Structure to score the sequence against")


class InverseFoldingStructureInput(BaseModel):
    """Bundles a structure with its per-structure design constraints.

    This model groups a protein structure with its associated chain IDs and fixed
    positions. The structure is automatically loaded and validated on construction.

    Attributes:
        structure (Structure): The protein structure. Accepts file path, PDB content string,
            Structure object, or a dict in the shape produced by ``Structure.model_dump(mode='json')``
            (for HTTP/JSON round-trips). Automatically converted to Structure.
        chain_ids (list[str] | None): Optional list of chain IDs to design. If None, all chains in the
            structure are designed.
        fixed_positions (dict[str, list[int]] | None): Optional dictionary mapping chain IDs to residue positions
            to keep fixed during design. Positions are 1-indexed.

    Examples:
        >>> # Simple structure from file (auto-loaded)
        >>> inp = InverseFoldingStructureInput(structure="/path/to/protein.pdb")
        >>> inp.structure_pdb  # Access PDB string directly

        >>> # With chain selection and fixed positions
        >>> inp = InverseFoldingStructureInput(
        ...     structure="/path/to/protein.pdb",
        ...     chain_ids=["A", "B"],
        ...     fixed_positions={"A": [1, 2, 3], "B": [10, 11]},
        ... )
    """

    structure: Structure = Field(description="Protein structure (auto-loaded from file path or PDB string).")
    chain_ids: list[str] | None = Field(
        default=None,
        description="Chain IDs to design. If None, all chains in the structure are designed.",
    )
    fixed_positions: dict[str, list[int]] | None = Field(
        default=None,
        description="Dictionary mapping chain IDs to residue positions to keep fixed (1-indexed).",
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_and_validate(cls, data: Any) -> Any:
        """Load structure, default chain_ids, and validate constraints."""
        if isinstance(data, dict):
            structure = data.get("structure")
            chain_ids = data.get("chain_ids")
            fixed_positions = data.get("fixed_positions")
        else:
            # Handle object-like input
            structure = getattr(data, "structure", None)
            chain_ids = getattr(data, "chain_ids", None)
            fixed_positions = getattr(data, "fixed_positions", None)

        # 1. Load structure if it's a string or Path

        if isinstance(structure, Structure):
            resolved_structure = structure
        elif isinstance(structure, (str, Path)):
            from proto_tools.entities.structures.utils import SUPPORTED_EXTENSIONS

            if str(structure).lower().endswith(SUPPORTED_EXTENSIONS):
                resolved_structure = Structure.from_file(structure)
            else:
                resolved_structure = Structure(structure=str(structure))
        elif isinstance(structure, dict):
            resolved_structure = Structure(**structure)
        else:
            raise ValueError(f"Unsupported structure type: {type(structure)}")

        available_chains = set(resolved_structure.get_chain_ids())

        # 2. Default chain_ids to all chains if None
        if chain_ids is None:
            resolved_chain_ids = resolved_structure.get_chain_ids()
        else:
            resolved_chain_ids = chain_ids
            # Validate chain_ids exist in structure
            requested_chains = set(resolved_chain_ids)
            if not requested_chains.issubset(available_chains):
                missing = requested_chains - available_chains
                raise ValueError(f"Chain IDs {missing} not found in structure. Available chains: {available_chains}")

        # 3. Validate fixed_positions if provided
        if fixed_positions is not None:
            for chain_id, positions in fixed_positions.items():
                if chain_id not in available_chains:
                    raise ValueError(
                        f"Fixed positions chain '{chain_id}' not in structure. Available chains: {available_chains}"
                    )
                chain_positions = resolved_structure.get_chain_positions(chain_id)
                invalid = set(positions) - set(chain_positions)
                if invalid:
                    raise ValueError(
                        f"Invalid fixed positions {invalid} for chain '{chain_id}'. Valid positions: {chain_positions}"
                    )

        result = {
            "structure": resolved_structure,
            "chain_ids": resolved_chain_ids,
            "fixed_positions": fixed_positions,
        }
        # Pass through extra keys for subclasses (e.g., fixed_sidechain_positions)
        if isinstance(data, dict):
            for k, v in data.items():
                if k not in result:
                    result[k] = v
        return result

    @property
    def structure_pdb(self) -> str:
        """Get the PDB string of the structure."""
        return self.structure.structure_pdb


class InverseFoldingInput(BaseToolInput):
    """Tool input for inverse folding sampling operations.

    Wraps a list of InverseFoldingStructureInput objects for use with the ToolRegistry.

    Attributes:
        inputs (list[InverseFoldingStructureInput]): List of InverseFoldingStructureInput objects, each
            containing a structure and optional chain_ids/fixed_positions constraints.

    Examples:
        >>> sampling_input = InverseFoldingInput(
        ...     inputs=[
        ...         InverseFoldingStructureInput(structure="/path/to/protein1.pdb"),
        ...         InverseFoldingStructureInput(structure="/path/to/protein2.pdb"),
        ...     ]
        ... )
    """

    inputs: list[InverseFoldingStructureInput] = InputField(
        description="List of inverse folding inputs, each containing a structure and constraints."
    )


class InverseFoldingConfig(BaseConfig):
    """Base configuration for inverse folding models.

    Contains model hyperparameters for sequence generation. Structure-specific
    constraints (chain_ids, fixed_positions) are specified in InverseFoldingInput.

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
        advanced=True,
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
        hidden=True,
        include_in_key=False,
        examples=["cuda", "cpu"],
    )


class DesignedSequences(BaseModel, ABC):
    """Represents the output of an inverse folding model produced from a single input structure.

    Each inverse folding model should define a concrete subclass with
    model-specific confidence metrics and metadata. Contains the designed
    amino acid sequence along with per-position and sequence-level quality metrics.

    NOTE: Because inverse folding models can generate multiple sequences for each
    input structure, fields in this class should be lists of length `num_sequences_per_structure`.

    Attributes:
        sequences (list[str]): Designed amino acid sequences in single-letter code.

    Properties:
        metrics: All confidence and quality metrics as a
            dictionary. Excludes the sequence itself.

    Note:
        Subclasses should add model-specific metrics like:
        - Overall sequence confidence/perplexity
        - Log probabilities
        - Temperature used for sampling
        - Model-specific scores (e.g., ProteinMPNN score)
    """

    sequences: list[str] = Field(description="Designed amino acid sequences from the inverse folding model")

    def __len__(self) -> int:
        """Get the number of designed sequences."""
        return len(self.sequences)

    def __getitem__(self, index: int) -> str:
        """Get a designed sequence by index."""
        return self.sequences[index]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over the designed sequences."""
        return iter(self.sequences)

    def __repr__(self) -> str:
        """Get a string representation of the designed sequences."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the designed sequences."""
        return f"DesignedSequences(with {len(self.sequences)} sequences)"

    def get_sequence_metrics(self, index: int) -> dict[str, float]:
        """Get the metrics for a designed sequence by index."""
        # Get all fields that are not sequences
        fields = self.model_dump()
        return {k: v[index] for k, v in fields.items() if k != "sequences" and isinstance(v, list)}


class InverseFoldingOutput(BaseToolOutput):
    """Output object for inverse folding models.

    Contains a list of DesignedSequences objects, one for each input structure.

    Attributes:
        designed_sequences (list[SerializeAsAny[DesignedSequences]]): List of DesignedSequences objects, one for each
            input structure. The order matches the input structures order.
    """

    # SerializeAsAny so subclass-only fields (e.g. ProteinMPNNSequences.perplexity) survive model_dump().
    designed_sequences: list[SerializeAsAny[DesignedSequences]] = Field(
        description="List of sequences predicted for the input structures"
    )

    def __len__(self) -> int:
        """Get the number of designed sequences."""
        return len(self.designed_sequences)

    def __getitem__(self, index: int) -> DesignedSequences:
        """Get a designed sequence by index."""
        return self.designed_sequences[index]

    def __iter__(self) -> Iterator[DesignedSequences]:  # type: ignore[override]
        """Iterate over the designed sequences."""
        return iter(self.designed_sequences)

    def __repr__(self) -> str:
        """Get a string representation of the designed sequences."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the output."""
        return f"InverseFoldingOutput(with {len(self.designed_sequences)} designed structures)"

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
            for i, designs in enumerate(self.designed_sequences):
                out_file = path / f"design_{i}.fasta"
                with open(out_file, "w") as f:
                    f.writelines(f">design_{i}_seq_{j}\n{seq}\n" for j, seq in enumerate(designs.sequences))

        elif file_format == "json":
            path.mkdir(parents=True, exist_ok=True)

            # Handle potential numpy types in metrics
            def default(obj: Any) -> Any:
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                return str(obj)

            for i, designs in enumerate(self.designed_sequences):
                out_file = path / f"design_{i}.json"
                with open(out_file, "w") as f:
                    json.dump(designs.model_dump(), f, indent=2, default=default)
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
        "log_likelihood": {"availability": "always", "type": "float", "min": None, "max": 0.0},
        "avg_log_likelihood": {"availability": "always", "type": "float", "min": None, "max": 0.0},
        "perplexity": {"availability": "always", "type": "float", "min": 1.0, "max": None},
    }
    primary_metric: str | None = "perplexity"

    logits: list[list[float]] | None = Field(
        default=None,
        description="Per-position logits array as nested list (seq_len, vocab_size)",
    )
    vocab: list[str] | None = Field(
        default=None,
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

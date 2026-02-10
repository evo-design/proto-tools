"""
inverse_folding/shared_data_models.py

Shared base schemas (configs and io) for inverse folding tools.
"""

from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bio_programming.bio_tools.tools.utils import BaseConfig, ConfigField
from bio_programming.bio_tools.entities.structures import Structure
from bio_programming.bio_tools.tools.infra.tool_io import BaseToolInput, BaseToolOutput


class SequenceStructurePair(BaseModel):
    """Represents a sequence-structure pair for MPNN-based scoring.

    Used by both ProteinMPNN and LigandMPNN scoring tools to pair a protein
    sequence with its corresponding structure for computing compatibility metrics.

    Attributes:
        sequence: Protein sequence to score against the structure.
        structure: Protein structure to score the sequence against.
    """

    sequence: str = Field(description="Sequence to score against the structure")
    structure: Structure = Field(description="Structure to score the sequence against")


class InverseFoldingStructureInput(BaseModel):
    """Bundles a structure with its per-structure design constraints.

    This model groups a protein structure with its associated chain IDs and fixed
    positions. The structure is automatically loaded and validated on construction.

    Attributes:
        structure: The protein structure. Accepts file path, PDB content string, or
            Structure object. Automatically converted to Structure.
        chain_ids: Optional list of chain IDs to design. If None, all chains in the
            structure are designed.
        fixed_positions: Optional dictionary mapping chain IDs to residue positions
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

    structure: Structure = Field(
        description="Protein structure (auto-loaded from file path or PDB string)."
    )
    chain_ids: Optional[List[str]] = Field(
        default=None,
        description="Chain IDs to design. If None, all chains in the structure are designed.",
    )
    fixed_positions: Optional[Dict[str, List[int]]] = Field(
        default=None,
        description="Dictionary mapping chain IDs to residue positions to keep fixed (1-indexed).",
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_and_validate(cls, data):
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
        from pathlib import Path

        if isinstance(structure, Structure):
            resolved_structure = structure
        elif isinstance(structure, (str, Path)):
            resolved_structure = Structure(
                structure_filepath_or_content=structure
            )
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
                raise ValueError(
                    f"Chain IDs {missing} not found in structure. "
                    f"Available chains: {available_chains}"
                )

        # 3. Validate fixed_positions if provided
        if fixed_positions is not None:
            for chain_id, positions in fixed_positions.items():
                if chain_id not in available_chains:
                    raise ValueError(
                        f"Fixed positions chain '{chain_id}' not in structure. "
                        f"Available chains: {available_chains}"
                    )
                chain_positions = resolved_structure.get_chain_positions(chain_id)
                invalid = set(positions) - set(chain_positions)
                if invalid:
                    raise ValueError(
                        f"Invalid fixed positions {invalid} for chain '{chain_id}'. "
                        f"Valid positions: {chain_positions}"
                    )

        return {
            "structure": resolved_structure,
            "chain_ids": resolved_chain_ids,
            "fixed_positions": fixed_positions,
        }

    @property
    def structure_pdb(self) -> str:
        """Get the PDB string of the structure."""
        return self.structure.structure_pdb


class InverseFoldingInput(BaseToolInput):
    """Tool input for inverse folding sampling operations.

    Wraps a list of InverseFoldingInput objects for use with the ToolRegistry.

    Attributes:
        inputs: List of InverseFoldingInput objects, each containing a structure
            and optional chain_ids/fixed_positions constraints.

    Examples:
        >>> sampling_input = InverseFoldingSamplingInput(
        ...     inputs=[
        ...         InverseFoldingInput(structure="/path/to/protein1.pdb"),
        ...         InverseFoldingInput(structure="/path/to/protein2.pdb"),
        ...     ]
        ... )
    """

    inputs: List[InverseFoldingStructureInput] = Field(
        description="List of inverse folding inputs, each containing a structure and constraints."
    )


class InverseFoldingConfig(BaseConfig):
    """Base configuration for inverse folding models.

    Contains model hyperparameters for sequence generation. Structure-specific
    constraints (chain_ids, fixed_positions) are specified in InverseFoldingInput.

    Attributes:
        batch_size (int): Number of sequences to generate for each input structure. Defaults to 1.

        temperature (float): Controls randomness in sampling from logits. Defaults to 0.1.

        excluded_amino_acids (Optional[List[str]]): List of amino acids that are not allowed in the sequence.
            If None, no amino acids will be disallowed. C is commonly specified. Defaults to None.

        seed (int): Random seed to use for sampling. Defaults to 42.

        device (str): Device to run the model on. Options include 'cuda' (NVIDIA GPU), 'cpu' (CPU execution), or specific GPU devices like 'cuda:0'. Defaults to 'cuda'.

        verbose (bool): Whether to print status messages during execution. Defaults to False.
    """

    batch_size: int = ConfigField(
        title="Number of Sequences to Generate",
        default=1,
        ge=1,
        description="Number of sequences to generate for each input structure.",
        examples=[1, 10],
    )

    temperature: float = ConfigField(
        title="Sampling Temperature",
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Controls randomness in sampling from logits",
        examples=[0.1, 0.5, 1.0],
    )

    excluded_amino_acids: Optional[List[str]] = ConfigField(
        title="Unallowed Amino Acids",
        default=None,
        description="List of amino acids that are not allowed in the sequence. If None, no amino acids will be disallowed",
        examples=["C"],
    )

    seed: int = ConfigField(
        title="Random Seed",
        default=42,
        description="Random seed to use for sampling",
        examples=[42, 123, 456],
        hidden=True,
    )

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on. Options include 'cuda' (NVIDIA GPU), 'cpu' (CPU execution)",
        hidden=True,
        examples=["cuda", "cpu"],
    )

    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages during execution",
        hidden=True,
    )


class DesignedSequences(BaseModel, ABC):
    """Represents the output of an inverse folding model produced from a single input structure.

    Each inverse folding model should define a concrete subclass with
    model-specific confidence metrics and metadata. Contains the designed
    amino acid sequence along with per-position and sequence-level quality metrics.

    NOTE: Because inverse folding models can generate multiple sequences for each
    input structure, fields in this class should be lists of length batch_size.

    Attributes:
        sequences (List[str]): Designed amino acid sequences in single-letter code.

    Properties:
        metrics (Dict[str, float]): All confidence and quality metrics as a
            dictionary. Excludes the sequence itself.

    Note:
        Subclasses should add model-specific metrics like:
        - Overall sequence confidence/perplexity
        - Log probabilities
        - Temperature used for sampling
        - Model-specific scores (e.g., ProteinMPNN score)
    """

    sequences: List[str] = Field(
        description="Designed amino acid sequences from the inverse folding model"
    )

    def __len__(self) -> int:
        """Get the number of designed sequences."""
        return len(self.sequences)

    def __getitem__(self, index: int) -> str:
        """Get a designed sequence by index."""
        return self.sequences[index]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the designed sequences."""
        return iter(self.sequences)

    def __repr__(self) -> str:
        """Get a string representation of the designed sequences."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the designed sequences."""
        return f"DesignedSequences(with {len(self.sequences)} sequences)"

    def get_sequence_metrics(self, index: int) -> Dict[str, float]:
        """Get the metrics for a designed sequence by index."""

        # Get all fields that are not sequences
        fields = self.model_dump()
        return_dict = {
            k: v[index]
            for k, v in fields.items()
            if k != "sequences" and isinstance(v, list)
        }
        return return_dict


class InverseFoldingOutput(BaseToolOutput):
    """Output object for inverse folding models.

    Contains a list of DesignedSequences objects, one for each input structure.

    Attributes:
        designed_sequences (List[DesignedSequences]): List of DesignedSequences objects, one for each input structure.
            The order matches the input structures order.
    """

    designed_sequences: List[DesignedSequences] = Field(
        description="List of sequences predicted for the input structures"
    )

    def __len__(self) -> int:
        """Get the number of designed sequences."""
        return len(self.designed_sequences)

    def __getitem__(self, index: int) -> DesignedSequences:
        """Get a designed sequence by index."""
        return self.designed_sequences[index]

    def __iter__(self) -> Iterator[DesignedSequences]:
        """Iterate over the designed sequences."""
        return iter(self.designed_sequences)

    def __repr__(self) -> str:
        """Get a string representation of the designed sequences."""
        return self.__str__()

    def __str__(self) -> str:
        """Get a string representation of the output."""
        return f"InverseFoldingOutput(with {len(self.designed_sequences)} designed structures)"

    @property
    def output_format_options(self) -> List[str]:
        return ["fasta", "json"]

    @property
    def output_format_default(self) -> str:
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path)

        if file_format == "fasta":
            path.mkdir(parents=True, exist_ok=True)
            for i, designs in enumerate(self.designed_sequences):
                out_file = path / f"design_{i}.fasta"
                with open(out_file, "w") as f:
                    for j, seq in enumerate(designs.sequences):
                        f.write(f">design_{i}_seq_{j}\n{seq}\n")

        elif file_format == "json":
            path.mkdir(parents=True, exist_ok=True)

            # Handle potential numpy types in metrics
            def default(obj):
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
class SequenceScores(BaseModel):
    """Individual sequence score with flexible metrics dict.

    Represents scoring metrics for a single sequence. Metrics can be accessed
    via dict-style (score.metrics["perplexity"]) or attribute-style (score.perplexity).

    Attributes:
        metrics: Dictionary of scalar scoring metrics.
        logits: Optional per-position logits array.
        vocab: Optional token ordering for logits; logits[:, j] corresponds to vocab[j].
    """

    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of scalar scoring metrics",
    )
    logits: Optional[List[List[float]]] = Field(
        default=None,
        description="Per-position logits array as nested list (seq_len, vocab_size)",
    )
    vocab: Optional[List[str]] = Field(
        default=None,
        description="Token ordering for logits: logits[:, j] corresponds to vocab[j]",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to metrics."""
        metrics = object.__getattribute__(self, "metrics")
        if name in metrics:
            return metrics[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to the output."""
        self.metrics[name] = value

    def __iter__(self) -> Iterator[float]:
        return iter(self.metrics.values())


class InverseFoldingScoringOutput(BaseToolOutput):
    """Standardized output for inverse folding scoring tools.

    Contains scoring results for sequence-structure pairs evaluated by
    ProteinMPNN or LigandMPNN scoring.

    Attributes:
        scores (List[SequenceScores]): List of scoring outputs, one per input
            sequence-structure pair. Each entry contains metrics (log_likelihood,
            avg_log_likelihood, perplexity) and optional per-position logits.
    """

    scores: List[SequenceScores] = Field(
        description="List of scoring outputs, one per input sequence-structure pair"
    )

    @property
    def vocab(self) -> Optional[List[str]]:
        """Token ordering for logits; derived from first score."""
        return self.scores[0].vocab if self.scores else None

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, index: int) -> SequenceScores:
        return self.scores[index]

    def __iter__(self) -> Iterator[SequenceScores]:
        return iter(self.scores)

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":
            import json

            def default(obj):
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                return str(obj)

            data = []
            for s in self.scores:
                score_data = dict(s.metrics)
                if s.logits is not None:
                    score_data["logits"] = s.logits
                if s.vocab is not None:
                    score_data["vocab"] = s.vocab
                data.append(score_data)

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=default)

        elif file_format == "csv":
            import csv

            if self.scores:
                fieldnames = list(self.scores[0].metrics.keys())
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for s in self.scores:
                        writer.writerow(s.metrics)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

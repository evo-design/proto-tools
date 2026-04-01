"""proto_tools/tools/sequence_alignment/msas.py.

Contains class for representing multiple sequence alignments (MSAs).
"""

from __future__ import annotations

import os
import shutil
import warnings
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from pyfaidx import Fasta

SUPPORTED_MSA_FILE_EXTENSIONS = [".a3m", ".fasta"]

MAX_SEQS_IN_MEMORY = 10_000


class MSA:
    """Multiple Sequence Alignment (MSA) representation.

    A data structure for storing and analyzing aligned biological sequences.
    Supports both protein and nucleotide alignments. Can be initialized either
    with aligned sequences directly or with a file path. When initialized from
    a file, sequences are streamed from disk on-demand (not cached) to minimize
    memory usage for large MSAs.
    """

    def __init__(
        self,
        aligned_sequences_or_filepath: list[str] | str,
        sequence_ids: list[str] | None = None,
    ) -> None:
        """Initialize the MSA object.

        WARNING: Validation of the MSA is only perfomed automatically when the MSA is in-memory.

        Args:
            aligned_sequences_or_filepath (list[str] | str): List of aligned sequences
                with '-' characters indicating gaps, or a path to an A3M/FASTA file.
            sequence_ids (list[str] | None): Optional list of sequence identifiers (only used when providing sequences directly).
        """
        self._in_memory = False
        self._temp_fasta_path: Path | None = None

        # ============================
        # In-memory initialization
        # ============================
        if isinstance(aligned_sequences_or_filepath, list):
            self._in_memory = True
            self._aligned_sequences = aligned_sequences_or_filepath

            if not self._aligned_sequences:
                raise ValueError("MSA must contain at least two sequences")

            self._sequence_ids = sequence_ids or [f"seq_{i}" for i in range(len(self._aligned_sequences))]

            self._alignment_length = len(self._aligned_sequences[0])
            self._num_sequences = len(self._aligned_sequences)

            if self._num_sequences < 2:
                raise ValueError("MSA must contain at least two sequences")

        # ============================
        # File-backed initialization
        # ============================
        elif isinstance(aligned_sequences_or_filepath, str):
            path = Path(aligned_sequences_or_filepath)
            if not path.exists():
                raise FileNotFoundError(path)

            if path.suffix == ".a3m":
                fasta_path = path.with_suffix(".fasta")
                convert_a3m_to_fasta(
                    a3m_path=path,  # type: ignore[arg-type]
                    fasta_path=fasta_path,  # type: ignore[arg-type]
                )
                self._temp_fasta_path = fasta_path
            else:
                fasta_path = path

            self._fasta = Fasta(
                fasta_path,
                as_raw=True,
                read_long_names=True,
            )

            self._sequence_ids = list(self._fasta.keys())
            self._num_sequences = len(self._sequence_ids)

            if self._num_sequences < 2:
                raise ValueError("MSA must contain at least two sequences")

            # Alignment length from first sequence
            first_id = self._sequence_ids[0]
            self._alignment_length = len(self._fasta[first_id])

            # Convert to in-memory representation if small enough
            if self._num_sequences < MAX_SEQS_IN_MEMORY:
                self._in_memory = True
                self._aligned_sequences = [self._fasta[seq_id][:] for seq_id in self._sequence_ids]

                self.rm_temp_files()
        else:
            raise ValueError(f"Invalid input type: {type(aligned_sequences_or_filepath)}")

        # Validate the MSA
        if self.alignment_length == 0:
            raise ValueError("MSA must contain at least one column")
        if self._in_memory:
            for seq in self._aligned_sequences:
                if not isinstance(seq, str):
                    raise ValueError(f"Sequence {seq} is not a string")
                if len(seq) != self._alignment_length:
                    raise ValueError(f"Sequence {seq} is not the same length as the MSA ({self._alignment_length})")

        # Cache for original sequences
        self._original_sequences = None

    def __iter__(self) -> Iterator[str]:
        if self._in_memory:
            yield from self._aligned_sequences
        else:
            for seq_id in self._sequence_ids:
                yield self._fasta[seq_id][:]

    def iter_with_ids(self) -> Iterator[tuple[str, str]]:
        """Iterate over aligned sequences yielding (identifier, sequence) pairs."""
        if self._in_memory:
            yield from zip(self._sequence_ids, self._aligned_sequences, strict=False)
        else:
            for seq_id in self._sequence_ids:
                yield seq_id, self._fasta[seq_id][:]

    def rm_temp_files(self) -> None:
        """Deletes the temporary FASTA file and its index file if they exist."""
        if self._temp_fasta_path and self._temp_fasta_path.exists():
            os.remove(self._temp_fasta_path)
            fai = self._temp_fasta_path.with_suffix(self._temp_fasta_path.suffix + ".fai")
            if fai.exists():
                os.remove(fai)

    def __del__(self) -> None:
        self.rm_temp_files()

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> str:
        if self._in_memory:
            return self._aligned_sequences[idx]
        return self._fasta[self._sequence_ids[idx]][:]  # type: ignore[no-any-return]

    def __enter__(self) -> Any:
        """Allows the MSA to be used as a context manager.

        Example:
        with MSA(msa_filepath_or_aligned_sequences="path/to/msa.fasta") as msa:
            # Use the MSA
            pass


        # The temporary FASTA file and its index file will be deleted automatically
        after the context manager exits.
        """
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Deletes the temporary FASTA file and its index file if they exist.

        when the MSA is exited as a context manager.
        """
        self.rm_temp_files()

    def __str__(self) -> str:
        return f"MSA(num_sequences={self.num_sequences}, alignment_length={self.alignment_length})"

    def __repr__(self) -> str:
        return self.__str__()

    # ============================
    # Properties
    # ============================

    @property
    def aligned_sequences(self) -> list[str]:
        """Get a list containing the aligned sequences. If the MSA is not.

        in-memory, this will convert the MSA to an in-memory representation.

        WARNING: It is highly recommended to use __iter__ or access via indexing
        instead for large MSAs.
        """
        if self._in_memory:
            return self._aligned_sequences
        warnings.warn("Converting to in-memory representation from file. This may be slow.", stacklevel=2)
        self._aligned_sequences = [self._fasta[seq_id][:] for seq_id in self._sequence_ids]
        self._in_memory = True
        self.rm_temp_files()
        return self._aligned_sequences

    @property
    def original_sequences(self) -> list[str]:
        """Get the list of original sequences (without gap characters)."""
        if self._original_sequences is None and self.num_sequences < MAX_SEQS_IN_MEMORY:  # type: ignore[redundant-expr]
            self._original_sequences = [  # type: ignore[assignment]
                seq.replace("-", "") for seq in self._aligned_sequences
            ]
            return self._original_sequences  # type: ignore[return-value]
        if self._original_sequences is not None:
            return self._original_sequences  # type: ignore[unreachable]
        return [seq.replace("-", "") for seq in self.aligned_sequences]

    @property
    def sequence_ids(self) -> list[str]:
        """Get the list of sequence IDs."""
        return self._sequence_ids

    @property
    def alignment_length(self) -> int:
        """Number of columns in the alignment."""
        return self._alignment_length

    @property
    def num_sequences(self) -> int:
        """Number of sequences in the alignment."""
        return self._num_sequences

    # ============================
    # Gap Statistics
    # ============================

    @property
    def total_gaps(self) -> int:
        """Returns the total number of gaps in the MSA."""
        if self._in_memory:
            return sum(seq.count("-") for seq in self._aligned_sequences)
        return sum(self._fasta[seq_id][:].count("-") for seq_id in self._sequence_ids)

    @property
    def average_gap_fraction(self) -> float:
        """Returns the average fraction of gaps in the MSA."""
        if self.alignment_length == 0:
            return 0.0

        if self._in_memory:
            return sum(seq.count("-") / self.alignment_length for seq in self._aligned_sequences) / self.num_sequences
        return (  # type: ignore[no-any-return]
            sum(self._fasta[seq_id][:].count("-") / self.alignment_length for seq_id in self._sequence_ids)
            / self.num_sequences
        )

    # ============================
    # Column Statistics
    # ============================

    def get_column(self, position: int) -> list[str]:
        """Returns the characters at the given position in the MSA."""
        if position < 0 or position >= self._alignment_length:
            raise IndexError(f"Position {position} out of range [0, {self._alignment_length})")

        if self._in_memory:
            return [seq[position] for seq in self._aligned_sequences]
        return [self._fasta[seq_id][position] for seq_id in self._sequence_ids]

    def get_conservation(self, position: int, exclude_gaps: bool = True) -> float:
        """Fraction of sequences with the most common character at position."""
        column = self.get_column(position)
        chars = [c for c in column if c != "-"] if exclude_gaps else column
        if not chars:
            return 0.0
        counts = Counter(chars)
        return counts.most_common(1)[0][1] / len(chars)

    def get_position_frequencies(self, position: int, include_gaps: bool = False) -> dict[str, float]:
        """Returns the character frequencies at the given position in the MSA."""
        column = self.get_column(position)
        counts = Counter(column) if include_gaps else Counter(c for c in column if c != "-")
        total = sum(counts.values())
        return {char: count / total for char, count in counts.items()} if total else {}

    # ============================
    # I/O
    # ============================
    def to_fasta_string(self) -> str:
        """Returns the MSA as a FASTA string."""
        lines = []
        for seq_id, seq in self.iter_with_ids():
            lines.append(f">{seq_id}")
            lines.append(seq)
        return "\n".join(lines)

    def to_fasta_file(self, fasta_path: str) -> None:
        """Writes the MSA to a FASTA file at the given path."""
        if not self._in_memory and hasattr(self, "_fasta"):
            # If already file-backed, copy the existing FASTA
            src_path = self._temp_fasta_path or self._fasta.filename
            shutil.copy(src_path, fasta_path)
        else:
            # In-memory: write sequences one by one
            with open(fasta_path, "w") as f:
                f.writelines(f">{seq_id}\n{seq}\n" for seq_id, seq in self.iter_with_ids())

    def to_a3m_string(self, query_index: int = 0) -> str:
        """Returns the MSA as an A3M format string.

        In A3M format, gaps in the query sequence are removed from all sequences,
        and positions that are gaps in the query are represented as lowercase letters
        in the other sequences (insertions relative to the query).

        WARNING: This will load the entire MSA into memory. For large MSAs, use
        to_a3m_file() instead.

        Args:
            query_index (int): Index of the sequence to use as the query (default: 0).

        Returns:
            str: A3M formatted string.

        Raises:
            IndexError: If query_index is out of range.
        """
        if query_index < 0 or query_index >= self.num_sequences:
            raise IndexError(f"Query index {query_index} out of range [0, {self.num_sequences})")

        query_seq = self[query_index]

        # Identify positions where query has gaps
        query_gap_positions = {i for i, char in enumerate(query_seq) if char == "-"}

        lines = []
        for _idx, (seq_id, seq) in enumerate(self.iter_with_ids()):
            lines.append(f">{seq_id}")

            a3m_seq = []
            for i, char in enumerate(seq):
                if i in query_gap_positions:
                    # Position is a gap in query - make lowercase (insertion)
                    if char != "-":
                        a3m_seq.append(char.lower())
                    # Skip gaps at insertion positions
                else:
                    # Position is not a gap in query - keep uppercase
                    a3m_seq.append(char)

            lines.append("".join(a3m_seq))

        return "\n".join(lines)

    def to_a3m_file(self, a3m_path: str, query_index: int = 0) -> None:
        """Writes the MSA to an A3M file at the given path.

        In A3M format, gaps in the query sequence are removed from all sequences,
        and positions that are gaps in the query are represented as lowercase letters
        in the other sequences (insertions relative to the query).

        Args:
            a3m_path (str): Path where the A3M file will be written.
            query_index (int): Index of the sequence to use as the query (default: 0).

        Raises:
            IndexError: If query_index is out of range.
        """
        if query_index < 0 or query_index >= self.num_sequences:
            raise IndexError(f"Query index {query_index} out of range [0, {self.num_sequences})")

        query_seq = self[query_index]

        # Identify positions where query has gaps
        query_gap_positions = {i for i, char in enumerate(query_seq) if char == "-"}

        # Write directly to file to avoid loading entire MSA into memory
        with open(a3m_path, "w") as f:
            for seq_id, seq in self.iter_with_ids():
                f.write(f">{seq_id}\n")

                a3m_seq = []
                for i, char in enumerate(seq):
                    if i in query_gap_positions:
                        # Position is a gap in query - make lowercase (insertion)
                        if char != "-":
                            a3m_seq.append(char.lower())
                        # Skip gaps at insertion positions
                    else:
                        # Position is not a gap in query - keep uppercase
                        a3m_seq.append(char)

                f.write("".join(a3m_seq) + "\n")

    # ============================
    # Pydantic/JSON Schema Export
    # ============================
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        """Accept either:.

        - an existing MSA
        - a list[str] of aligned sequences
        - a string filepath
        """

        def validate(v: Any) -> Any:
            if isinstance(v, cls):
                return v
            if isinstance(v, list):
                return cls(v)
            if isinstance(v, str):
                return cls(v)
            raise TypeError("MSA must be an MSA instance, a list of aligned sequences, or a file path")

        def serialize(instance: Any) -> Any:
            """Serialize MSA instance to list of aligned sequences."""
            # For small MSAs, serialize as list of sequences
            # For large file-backed MSAs, this will convert to in-memory
            if instance._in_memory:
                return instance._aligned_sequences
            # For large file-backed MSAs, we need to serialize as list
            # This may be memory intensive but is necessary for serialization
            return list(instance)

        # Create a schema that validates from Python objects
        from_python_schema = core_schema.no_info_plain_validator_function(validate)

        return core_schema.json_or_python_schema(
            json_schema=from_python_schema,
            python_schema=core_schema.union_schema(
                [
                    # Allow instances of MSA to pass through without modification
                    core_schema.is_instance_schema(cls),
                    # Otherwise, use our validation function
                    from_python_schema,
                ]
            ),
            # Define how to serialize the object
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize,
                when_used="json",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: Any,
    ) -> JsonSchemaValue:
        """JSON schema representation of an MSA."""
        return {
            "title": "MSA",
            "description": (
                "Multiple sequence alignment. "
                "Represented as either a list of aligned sequences "
                "or a file path to an MSA (.a3m or .fasta)."
            ),
            "oneOf": [
                {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Aligned sequences (all same length)",
                },
                {
                    "type": "string",
                    "description": "Path to an MSA file (.a3m or .fasta)",
                },
            ],
        }


# ============================================================================
# Conversion Functions
# ============================================================================
def convert_a3m_to_fasta(  # type: ignore[return]
    a3m_path: str,
    fasta_path: str,
) -> Path:
    """Convert an A3M file to a rectangular FASTA MSA with '-' for gaps.

    Args:
        a3m_path (str): Path to input A3M.
        fasta_path (str): Path to output FASTA.

    Raises:
        FileNotFoundError: If the A3M file does not exist.
    """
    with open(a3m_path) as a3m_f, open(fasta_path, "w") as fasta_f:
        for raw_line in a3m_f:
            line = raw_line.replace("\x00", "").rstrip()
            if not line:
                continue
            if line.startswith(">"):
                fasta_f.write(line.strip() + "\n")
            elif line.startswith("#"):
                continue
            else:
                fasta_f.write("".join(c for c in line if not c.islower()) + "\n")

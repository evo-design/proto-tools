"""Multiple Sequence Alignment (MSA) representation as a Pydantic BaseModel."""

from collections import Counter
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

SUPPORTED_MSA_FILE_EXTENSIONS = [".a3m", ".fasta", ".fa"]


class MSA(BaseModel):
    """Multiple Sequence Alignment (MSA) representation.

    A Pydantic model for storing and analyzing aligned biological sequences.
    Supports both protein and nucleotide alignments. Sequences are always held
    in memory. Use ``MSA.from_file()`` to load from FASTA or A3M files.

    Attributes:
        aligned_sequences (list[str]): Aligned sequences with ``-`` characters
            indicating gaps. All sequences must have the same length.
        sequence_ids (list[str]): Identifiers for each sequence. Auto-generated
            as ``seq_0``, ``seq_1``, ... if not provided.
    """

    model_config = ConfigDict(extra="forbid")

    aligned_sequences: list[str] = Field(description="Aligned sequences, all same length, with '-' for gaps")
    sequence_ids: list[str] = Field(description="Sequence identifiers")

    _original_sequences_cache: list[str] | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _auto_generate_ids(cls, data: Any) -> Any:
        """Auto-generate sequence_ids when not provided."""
        if isinstance(data, list):
            return {"aligned_sequences": data, "sequence_ids": [f"seq_{i}" for i in range(len(data))]}
        if isinstance(data, dict) and "aligned_sequences" in data and "sequence_ids" not in data:
            data["sequence_ids"] = [f"seq_{i}" for i in range(len(data["aligned_sequences"]))]
        return data

    @model_validator(mode="after")
    def _validate_alignment(self) -> "MSA":
        """Validate alignment constraints after field assignment."""
        if len(self.aligned_sequences) < 2:
            msg = "MSA must contain at least two sequences"
            raise ValueError(msg)
        if len(self.sequence_ids) != len(self.aligned_sequences):
            msg = (
                f"Number of sequence_ids ({len(self.sequence_ids)}) "
                f"must match number of sequences ({len(self.aligned_sequences)})"
            )
            raise ValueError(msg)
        first_len = len(self.aligned_sequences[0])
        if first_len == 0:
            msg = "MSA must contain at least one column"
            raise ValueError(msg)
        for i, seq in enumerate(self.aligned_sequences):
            if len(seq) != first_len:
                msg = f"Sequence {i} has length {len(seq)}, expected {first_len}"
                raise ValueError(msg)
        return self

    # ============================================================================
    # Factory
    # ============================================================================

    @classmethod
    def from_file(cls, path: str | Path) -> "MSA":
        """Load an MSA from a FASTA or A3M file.

        Args:
            path (str | Path): Path to a ``.fasta`` or ``.a3m`` file.

        Returns:
            MSA: The loaded alignment.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        path = Path(path)
        if path.suffix == ".a3m":
            ids, seqs = _parse_a3m_file(path)
            return cls(aligned_sequences=seqs, sequence_ids=ids)
        if path.suffix in (".fasta", ".fa"):
            return cls.from_fasta_string(path.read_text())
        msg = f"Unsupported MSA file format: {path.suffix}. Supported: {SUPPORTED_MSA_FILE_EXTENSIONS}"
        raise ValueError(msg)

    @classmethod
    def from_fasta_string(cls, text: str) -> "MSA":
        r"""Parse an in-memory FASTA string into an MSA.

        Args:
            text (str): FASTA-formatted text (e.g., ``">a\nMK\n>b\nMA\n"``).

        Returns:
            MSA: The parsed alignment.
        """
        from Bio import SeqIO

        ids: list[str] = []
        seqs: list[str] = []
        for record in SeqIO.parse(StringIO(text), "fasta"):  # type: ignore[no-untyped-call]
            ids.append(record.description)
            seqs.append(str(record.seq))
        return cls(aligned_sequences=seqs, sequence_ids=ids)

    # ============================================================================
    # Sequence access
    # ============================================================================

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over aligned sequences."""
        yield from self.aligned_sequences

    def __getitem__(self, idx: int) -> str:
        """Get a sequence by index."""
        return self.aligned_sequences[idx]

    def __len__(self) -> int:
        """Number of sequences in the alignment."""
        return len(self.aligned_sequences)

    def iter_with_ids(self) -> Iterator[tuple[str, str]]:
        """Iterate over aligned sequences yielding (identifier, sequence) pairs."""
        yield from zip(self.sequence_ids, self.aligned_sequences, strict=False)

    def __str__(self) -> str:
        return f"MSA(num_sequences={self.num_sequences}, alignment_length={self.alignment_length})"

    def __repr__(self) -> str:
        return self.__str__()

    # ============================================================================
    # Properties
    # ============================================================================

    @property
    def alignment_length(self) -> int:
        """Number of columns in the alignment."""
        return len(self.aligned_sequences[0]) if self.aligned_sequences else 0

    @property
    def num_sequences(self) -> int:
        """Number of sequences in the alignment."""
        return len(self.aligned_sequences)

    @property
    def original_sequences(self) -> list[str]:
        """Sequences with gap characters removed."""
        if self._original_sequences_cache is None:
            self._original_sequences_cache = [seq.replace("-", "") for seq in self.aligned_sequences]
        return self._original_sequences_cache

    # ============================================================================
    # Gap Statistics
    # ============================================================================

    @property
    def total_gaps(self) -> int:
        """Total number of gap characters across all sequences."""
        return sum(seq.count("-") for seq in self.aligned_sequences)

    @property
    def average_gap_fraction(self) -> float:
        """Average fraction of gap characters per sequence."""
        if self.alignment_length == 0:
            return 0.0
        return sum(seq.count("-") / self.alignment_length for seq in self.aligned_sequences) / self.num_sequences

    # ============================================================================
    # Column Statistics
    # ============================================================================

    def get_column(self, position: int) -> list[str]:
        """Get the characters at a given alignment position.

        Args:
            position (int): Column index (0-based).

        Returns:
            list[str]: Characters at that position, one per sequence.

        Raises:
            IndexError: If position is out of range.
        """
        if position < 0 or position >= self.alignment_length:
            raise IndexError(f"Position {position} out of range [0, {self.alignment_length})")
        return [seq[position] for seq in self.aligned_sequences]

    def get_conservation(self, position: int, exclude_gaps: bool = True) -> float:
        """Fraction of sequences with the most common character at position.

        Args:
            position (int): Column index (0-based).
            exclude_gaps (bool): Whether to exclude gap characters from the count.

        Returns:
            float: Conservation score between 0.0 and 1.0.
        """
        column = self.get_column(position)
        chars = [c for c in column if c != "-"] if exclude_gaps else column
        if not chars:
            return 0.0
        counts = Counter(chars)
        return counts.most_common(1)[0][1] / len(chars)

    def get_position_frequencies(self, position: int, include_gaps: bool = False) -> dict[str, float]:
        """Character frequencies at a given alignment position.

        Args:
            position (int): Column index (0-based).
            include_gaps (bool): Whether to include gap characters.

        Returns:
            dict[str, float]: Mapping from character to frequency (0.0-1.0).
        """
        column = self.get_column(position)
        counts = Counter(column) if include_gaps else Counter(c for c in column if c != "-")
        total = sum(counts.values())
        return {char: count / total for char, count in counts.items()} if total else {}

    # ============================================================================
    # I/O
    # ============================================================================

    def to_fasta_string(self) -> str:
        """Return the MSA as a FASTA-formatted string (trailing newline included)."""
        return "".join(f">{seq_id}\n{seq}\n" for seq_id, seq in self.iter_with_ids())

    def to_fasta_file(self, fasta_path: str) -> None:
        """Write the MSA to a FASTA file.

        Args:
            fasta_path (str): Destination file path.
        """
        Path(fasta_path).write_text(self.to_fasta_string())

    def _a3m_lines(self, query_index: int = 0) -> Iterator[tuple[str, str]]:
        """Yield ``(seq_id, a3m_sequence)`` pairs for A3M encoding.

        Positions that are gaps in the query are removed from the query and
        represented as lowercase letters in other sequences.

        Args:
            query_index (int): Index of the query sequence (default: 0).

        Yields:
            tuple[str, str]: (sequence_id, a3m_encoded_sequence).

        Raises:
            IndexError: If query_index is out of range.
        """
        if query_index < 0 or query_index >= self.num_sequences:
            raise IndexError(f"Query index {query_index} out of range [0, {self.num_sequences})")

        query_gap_positions = {i for i, char in enumerate(self[query_index]) if char == "-"}

        for seq_id, seq in self.iter_with_ids():
            a3m_seq = []
            for i, char in enumerate(seq):
                if i in query_gap_positions:
                    if char != "-":
                        a3m_seq.append(char.lower())
                else:
                    a3m_seq.append(char)
            yield seq_id, "".join(a3m_seq)

    def to_a3m_string(self, query_index: int = 0) -> str:
        """Return the MSA as an A3M-formatted string.

        Args:
            query_index (int): Index of the query sequence (default: 0).

        Returns:
            str: A3M formatted string.

        Raises:
            IndexError: If query_index is out of range.
        """
        lines = []
        for seq_id, a3m_seq in self._a3m_lines(query_index):
            lines.append(f">{seq_id}")
            lines.append(a3m_seq)
        return "\n".join(lines)

    def to_a3m_file(self, a3m_path: str, query_index: int = 0) -> None:
        """Write the MSA to an A3M file.

        Args:
            a3m_path (str): Destination file path.
            query_index (int): Index of the query sequence (default: 0).

        Raises:
            IndexError: If query_index is out of range.
        """
        with open(a3m_path, "w") as f:
            f.writelines(f">{seq_id}\n{a3m_seq}\n" for seq_id, a3m_seq in self._a3m_lines(query_index))


# ============================================================================
# File Parsing
# ============================================================================


def _parse_a3m_file(path: Path) -> tuple[list[str], list[str]]:
    """Parse an A3M file into sequence IDs and rectangular aligned sequences.

    Lowercase characters (insertions relative to query) are removed to produce
    a rectangular alignment with ``-`` for gaps.

    Args:
        path (Path): Path to an A3M file.

    Returns:
        tuple[list[str], list[str]]: (sequence_ids, aligned_sequences).
    """
    ids: list[str] = []
    seqs: list[str] = []
    current_id: str | None = None
    current_seq: list[str] = []

    with open(path) as f:
        for raw_line in f:
            line = raw_line.replace("\x00", "").rstrip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if current_id is not None:
                    ids.append(current_id)
                    seqs.append("".join(current_seq))
                current_id = line[1:].strip()
                current_seq = []
            else:
                current_seq.append("".join(c for c in line if not c.islower()))

    if current_id is not None:
        ids.append(current_id)
        seqs.append("".join(current_seq))

    return ids, seqs


def convert_a3m_to_fasta(
    a3m_path: str,
    fasta_path: str,
) -> None:
    """Convert an A3M file to a rectangular FASTA MSA with ``-`` for gaps.

    Args:
        a3m_path (str): Path to input A3M.
        fasta_path (str): Path to output FASTA.
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

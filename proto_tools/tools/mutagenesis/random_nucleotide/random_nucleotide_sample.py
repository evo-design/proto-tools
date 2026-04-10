"""proto_tools/tools/mutagenesis/random_nucleotide/random_nucleotide_sample.py.

Random nucleotide sampling with IUPAC degenerate base support.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from proto_tools.tools.masked_models.masking import (
    MASK_TOKEN,
    MaskingStrategy,
    apply_masking_strategy,
)
from proto_tools.tools.mutagenesis.codons import sample_nucleotide
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)

logger = logging.getLogger(__name__)

SubstitutionScheme = Literal[
    "N",
    "R",
    "Y",
    "S",
    "W",
    "K",
    "M",
    "B",
    "D",
    "H",
    "V",
]

# ============================================================================
# Data Models
# ============================================================================


class RandomNucleotideSampleInput(BaseToolInput):
    """Input for random nucleotide sampling.

    Attributes:
        sequences (list[str]): DNA or RNA sequences, possibly containing ``_`` at
            positions to mutate. Accepts a single string or a list.
    """

    sequences: list[str] = InputField(
        description="DNA/RNA sequence(s) to mutate. May contain '_' at positions to sample.",
        examples=["ACGTACGT", ["ACGT_CGT", "AUGC"]],
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[str]:
        """Normalize sequences to a list."""
        if isinstance(value, str):
            return [value]
        for seq in value:
            if seq is None:
                raise ValueError("Sequence cannot be None")
        return value  # type: ignore[no-any-return]


class RandomNucleotideSampleOutput(BaseToolOutput):
    """Output from random nucleotide sampling.

    Attributes:
        sequences (list[str]): Nucleotide sequences with masked positions filled
            by random bases drawn from the configured substitution scheme.
    """

    sequences: list[str] = Field(
        description="Nucleotide sequences with masked positions randomly filled",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["fasta", "txt", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "fasta":
            with open(path, "w") as f:
                f.writelines(f">seq_{i}\n{seq}\n" for i, seq in enumerate(self.sequences))
        elif file_format == "txt":
            with open(path, "w") as f:
                f.writelines(f"{seq}\n" for seq in self.sequences)
        elif file_format == "json":
            with open(path, "w") as f:
                json.dump(self.sequences, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


class RandomNucleotideSampleConfig(BaseConfig):
    """Configuration for random nucleotide sampling.

    Attributes:
        masking_strategy (MaskingStrategy): Controls which positions to mask for sampling.
            Default: random 30%.
        substitution_scheme (SubstitutionScheme): IUPAC ambiguity code defining the nucleotide
            pool for substitutions. ``"N"`` = any base (ACGT);
            ``"R"`` = purines (AG); ``"Y"`` = pyrimidines (CT); etc.
        sequence_type (Literal['auto', 'dna', 'rna']): How to interpret input sequences. ``"auto"``
            detects DNA vs RNA by presence of U; ``"dna"`` or ``"rna"``
            forces the type.
    """

    masking_strategy: MaskingStrategy = ConfigField(
        title="Masking Strategy",
        default_factory=MaskingStrategy,
        description="Controls which positions to mask for sampling. Default: random 30%.",
    )
    substitution_scheme: SubstitutionScheme = ConfigField(
        title="Substitution Scheme",
        default="N",
        description="IUPAC code defining the nucleotide substitution pool.",
    )
    sequence_type: Literal["auto", "dna", "rna"] = ConfigField(
        title="Sequence Type",
        default="auto",
        description="Sequence type: auto-detect, force DNA, or force RNA.",
        hidden=True,
    )

    def preprocess(self, inputs: Any) -> Any:
        """Apply masking strategy unless sequences are already pre-masked."""
        return apply_masking_strategy(self, inputs)


# ============================================================================
# Helper
# ============================================================================


def _detect_sequence_type(sequences: list[str]) -> str:
    """Detect whether sequences are DNA or RNA.

    Args:
        sequences (list[str]): Nucleotide sequences to classify.

    Returns ``"rna"`` if any sequence contains U/u (ignoring mask tokens),
    otherwise ``"dna"``.
    """
    for seq in sequences:
        clean = seq.replace(MASK_TOKEN, "")
        if "U" in clean.upper():
            return "rna"
    return "dna"


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return RandomNucleotideSampleInput(sequences=["ACGTACGT"])


@tool(
    key="random-nucleotide-sample",
    label="Random Nucleotide Sampling",
    category="mutagenesis",
    input_class=RandomNucleotideSampleInput,
    config_class=RandomNucleotideSampleConfig,
    output_class=RandomNucleotideSampleOutput,
    description="Sample nucleotide sequences by filling masked positions with random bases from an IUPAC substitution scheme",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="sequences",
    cacheable=False,
)
def run_random_nucleotide_sample(
    inputs: RandomNucleotideSampleInput,
    config: RandomNucleotideSampleConfig,
    instance: Any = None,  # noqa: ARG001 — required by tool interface
) -> RandomNucleotideSampleOutput:
    """Fill masked positions with random nucleotides from an IUPAC scheme.

    The ``preprocess`` hook on :class:`RandomNucleotideSampleConfig` applies
    the masking strategy before this function runs, so
    ``inputs.sequences`` already contain ``_`` at positions to sample.

    For RNA sequences, sampled T bases are converted to U.

    Args:
        inputs (RandomNucleotideSampleInput): Nucleotide sequences with ``_`` at designable positions.
        config (RandomNucleotideSampleConfig): Sampling configuration.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        RandomNucleotideSampleOutput: RandomNucleotideSampleOutput with sampled sequences.
    """
    rng = random.Random(config.resolved_seed)  # noqa: S311 -- not cryptographic
    scheme = config.substitution_scheme

    # Resolve sequence type
    seq_type = config.sequence_type
    if seq_type == "auto":
        seq_type = _detect_sequence_type(inputs.sequences)  # type: ignore[assignment]  # returns str, not Literal

    is_rna = seq_type == "rna"

    sampled = []
    for seq in inputs.sequences:
        chars = list(seq)
        for i, ch in enumerate(chars):
            if ch == MASK_TOKEN:
                base = sample_nucleotide(scheme, rng=rng)
                if is_rna:
                    base = base.replace("T", "U")
                chars[i] = base
        sampled.append("".join(chars))

    return RandomNucleotideSampleOutput(
        metadata={
            "substitution_scheme": scheme,
            "sequence_type": seq_type,
            "num_sequences": len(inputs.sequences),
        },
        sequences=sampled,
    )

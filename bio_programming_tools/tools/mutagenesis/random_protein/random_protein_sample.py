"""bio_programming_tools/tools/mutagenesis/random_protein/random_protein_sample.py

Random protein sampling with codon scheme-biased amino acid selection."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field

from bio_programming_tools.tools.masked_models.masking import (
    MASK_TOKEN,
    MaskingStrategy,
    apply_masking_strategy,
)
from bio_programming_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
)
from bio_programming_tools.tools.mutagenesis.codons import sample_amino_acid
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, BaseToolOutput, ConfigField

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

# Input: reuse MaskedModelInput (protein sequences, optional _ masks)
RandomProteinSampleInput = MaskedModelInput

CodonScheme = Literal["UNIFORM", "NNN", "NNK", "NNS", "NDT", "DBK", "NRT"]


class RandomProteinSampleOutput(BaseToolOutput):
    """Output from random protein sampling.

    Attributes:
        sequences (list[str]): Sampled protein sequences with masked positions filled
            by random amino acids drawn from the configured codon scheme.
    """

    sequences: List[str] = Field(
        description="Protein sequences with masked positions randomly filled",
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["fasta", "txt", "json"]

    @property
    def output_format_default(self) -> str:
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "fasta":
            with open(path, "w") as f:
                for i, seq in enumerate(self.sequences):
                    f.write(f">seq_{i}\n{seq}\n")
        elif file_format == "txt":
            with open(path, "w") as f:
                for seq in self.sequences:
                    f.write(f"{seq}\n")
        elif file_format == "json":
            import json

            with open(path, "w") as f:
                json.dump(self.sequences, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


class RandomProteinSampleConfig(BaseConfig):
    """Configuration for random protein sampling.

    Attributes:
        masking_strategy (MaskingStrategy): Controls which positions to mask for sampling.
            Default: random 30%.
        codon_scheme (CodonScheme): Codon scheme controlling amino acid sampling
            probabilities. ``"UNIFORM"`` gives equal weight to all 20
            amino acids; other schemes (NNK, NNS, NDT, etc.) weight
            amino acids by the number of codons encoding them.
        seed (int | None): Random seed for reproducibility. Default: ``None``.
    """

    masking_strategy: MaskingStrategy = ConfigField(
        title="Masking Strategy",
        default_factory=MaskingStrategy,
        description="Controls which positions to mask for sampling. Default: random 30%.",
    )
    codon_scheme: CodonScheme = ConfigField(
        title="Codon Scheme",
        default="UNIFORM",
        description="Codon scheme for amino acid sampling probabilities.",
    )
    seed: Optional[int] = ConfigField(
        title="Random Seed",
        default=None,
        description="Random seed for reproducible sampling.",
        advanced=True,
        include_in_key=False,
    )

    def preprocess(self, inputs):
        """Apply masking strategy unless sequences are already pre-masked."""
        return apply_masking_strategy(self, inputs)


# ============================================================================
# Tool Implementation
# ============================================================================

def example_input():
    """Minimal valid input for testing and examples."""
    return RandomProteinSampleInput(sequences=["MKTL"])


@tool(
    key="random-protein-sample",
    label="Random Protein Sampling",
    category="mutagenesis",
    input_class=RandomProteinSampleInput,
    config_class=RandomProteinSampleConfig,
    output_class=RandomProteinSampleOutput,
    description="Sample protein sequences by filling masked positions with random amino acids drawn from a codon scheme",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="sequences",
    cacheable=False,
)
def run_random_protein_sample(
    inputs: RandomProteinSampleInput,
    config: RandomProteinSampleConfig | None = None,
    instance=None,
) -> RandomProteinSampleOutput:
    """Fill masked positions with random amino acids from a codon scheme.

    The ``preprocess`` hook on :class:`RandomProteinSampleConfig` applies
    the masking strategy before this function runs, so
    ``inputs.sequences`` already contain ``_`` at positions to sample.

    Args:
        inputs (RandomProteinSampleInput): Protein sequences with ``_`` at designable positions.
        config (RandomProteinSampleConfig | None): Sampling configuration.

    Returns:
        RandomProteinSampleOutput: RandomProteinSampleOutput with sampled sequences.
    """
    rng = random.Random(config.seed) if config.seed is not None else None
    scheme = config.codon_scheme

    sampled = []
    for seq in inputs.sequences:
        chars = list(seq)
        for i, ch in enumerate(chars):
            if ch == MASK_TOKEN:
                chars[i] = sample_amino_acid(scheme, rng=rng)
        sampled.append("".join(chars))

    return RandomProteinSampleOutput(
        metadata={
            "codon_scheme": scheme,
            "num_sequences": len(inputs.sequences),
        },
        sequences=sampled,
    )

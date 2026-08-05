"""Shared constants and base models for the CodonFM (Encodon) toolkit.

CodonFM / Encodon is a codon-level masked language model (NVIDIA-BioNeMo/CodonFM).
The tools score coding sequences (CDS) at codon resolution: mutation effect
(reference-vs-alternate codon log-likelihood ratio), sequence fitness (mean
log-likelihood), CLS embeddings, and a differentiable codon objective.
"""

import hashlib
import math
import urllib.parse
from itertools import product
from typing import Any, Literal

from pydantic import field_validator

from proto_tools.utils import BaseConfig, BaseToolInput, ConfigField, InputField
from proto_tools.utils.sequence import return_invalid_nucleotide_chars

# The four publicly released Encodon checkpoints (public HuggingFace repos under the NVIDIA Open
# Model License, which governs use but does not gate download). Each repo ships
# ``<name>.safetensors`` + ``config.json`` (both required, in the same directory, by ``EncodonInference``).
CodonFMCheckpoint = Literal[
    "encodon_80m",
    "encodon_600m",
    "encodon_1b",
    "encodon_1b_cdwt",
]

CODONFM_CHECKPOINTS: dict[str, dict[str, str]] = {
    "encodon_80m": {"repo": "nvidia/NV-CodonFM-Encodon-80M-v1", "safetensors": "NV-CodonFM-Encodon-80M-v1.safetensors"},
    "encodon_600m": {
        "repo": "nvidia/NV-CodonFM-Encodon-600M-v1",
        "safetensors": "NV-CodonFM-Encodon-600M-v1.safetensors",
    },
    "encodon_1b": {"repo": "nvidia/NV-CodonFM-Encodon-1B-v1", "safetensors": "NV-CodonFM-Encodon-1B-v1.safetensors"},
    "encodon_1b_cdwt": {
        "repo": "nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1",
        "safetensors": "NV-CodonFM-Encodon-Cdwt-1B-v1.safetensors",
    },
}

# Encodon's ``max_position_embeddings`` is 2048 tokens; two are the CLS/SEP specials, so a CDS is
# capped at (2048 - 2) * 3 = 6138 nt (2046 codons). Longer inputs are out of the training regime.
CODONFM_MAX_TOKENS = 2048
CODONFM_MAX_NT = (CODONFM_MAX_TOKENS - 2) * 3

# The 64 DNA codons in lexicographic order (AAA, AAC, AAG, AAT, ...), including the three
# standard stop codons. This matches the
# tokenizer's ``get_all_codons("dna")`` ordering, and is the column order for the relaxed
# ``(L, 64)`` codon logits consumed by the CodonFM gradient tool. The five special tokens
# (CLS/SEP/UNK/PAD/MASK) are excluded — only concrete DNA codons are design variables.
CODONFM_CODON_VOCAB: list[str] = ["".join(codon) for codon in product("ACGT", repeat=3)]
CODONFM_NUM_CODONS = len(CODONFM_CODON_VOCAB)


def one_hot_codon_logits(sequence: str, *, sharpness: float = 20.0) -> list[list[float]]:
    """Build sharp one-hot ``(L, 64)`` codon logits from a coding sequence (for examples/tests).

    Each codon position becomes a row over :data:`CODONFM_CODON_VOCAB` with ``sharpness`` on the
    observed codon and ``0.0`` elsewhere, so a softmax concentrates on that codon.

    Args:
        sequence (str): A codon-aligned coding sequence (DNA/RNA; ``U`` maps to ``T``).
        sharpness (float): Logit placed on the observed codon.

    Returns:
        list[list[float]]: Row-per-codon logits in :data:`CODONFM_CODON_VOCAB` column order.

    Raises:
        ValueError: If a codon is not one of the 64 DNA codons or ``sharpness`` is non-finite.
    """
    if not math.isfinite(sharpness):
        raise ValueError("sharpness must be finite")
    seq = normalize_codon_sequence(sequence)
    index = {codon: i for i, codon in enumerate(CODONFM_CODON_VOCAB)}
    rows: list[list[float]] = []
    for start in range(0, len(seq), 3):
        codon = seq[start : start + 3]
        if codon not in index:
            raise ValueError(f"one_hot_codon_logits: {codon!r} is not one of the 64 DNA codons")
        row = [0.0] * CODONFM_NUM_CODONS
        row[index[codon]] = sharpness
        rows.append(row)
    return rows


def normalize_codon_sequence(sequence: str) -> str:
    """Uppercase, strip whitespace, map RNA ``U`` to ``T``, and validate a codon-aligned CDS.

    Encodon reads coding sequences in the DNA alphabet at 3-nt codon resolution, so the length
    must be a multiple of 3. ``U`` is accepted (mapped to ``T``) so callers can pass RNA.
    Ambiguous bases are rejected because the upstream tokenizer treats them character-by-character,
    which would shift codon coordinates.

    Args:
        sequence (str): A single coding sequence.

    Returns:
        str: The normalized DNA-alphabet CDS.

    Raises:
        ValueError: If empty, not a multiple of 3 in length, over the length cap, or containing
            characters other than A, C, G, T, or U.
    """
    if not sequence or not sequence.strip():
        raise ValueError("Sequence cannot be empty")
    seq = "".join(sequence.upper().split()).replace("U", "T")
    invalid = return_invalid_nucleotide_chars(seq)
    if invalid:
        raise ValueError(f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid))}")
    if len(seq) % 3 != 0:
        raise ValueError(f"Coding sequence length must be a multiple of 3 (codon-aligned); got length {len(seq)}")
    if len(seq) > CODONFM_MAX_NT:
        raise ValueError(
            f"CodonFM supports CDS up to {CODONFM_MAX_NT} nt ({CODONFM_MAX_TOKENS - 2} codons); got {len(seq)}"
        )
    return seq


class CodonSequenceInput(BaseToolInput):
    """Coding sequences to score with a CodonFM (Encodon) tool.

    Attributes:
        sequences (list[str]): Coding sequence(s) at codon resolution. A single string is
            normalized to a one-item list; ``U`` maps to ``T`` and each length must be a multiple
            of 3 (codon-aligned) and at most 6138 nt.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="Coding sequence(s) to score; RNA (U) maps to DNA (T); length must be a multiple of 3.",
        min_length=1,
        examples=["ATGGTGAGCAAGGGCGAGGAG", ["ATGGTG", "ATGGCCACC"]],
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[Any]:
        """Normalize a single coding sequence to a one-item list."""
        if value is None:
            raise ValueError("sequences cannot be None")
        if isinstance(value, str):
            return [value]
        if not value:
            raise ValueError("sequences cannot be empty")
        return value  # type: ignore[no-any-return]

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate and normalize coding sequences to the DNA alphabet."""
        return [normalize_codon_sequence(sequence) for sequence in sequences]

    def __len__(self) -> int:
        """Return the number of input sequences."""
        return len(self.sequences)


class CodonFMConfig(BaseConfig):
    """Shared runtime + checkpoint configuration for CodonFM (Encodon) tools.

    Attributes:
        model_checkpoint (CodonFMCheckpoint): Which Encodon checkpoint to run. Downloaded on
            demand from its public HuggingFace repo (no token required).
        device (str): Device used for inference.
        batch_size (int): Number of sequences to run per GPU forward pass.
    """

    model_checkpoint: CodonFMCheckpoint = ConfigField(
        title="Model Checkpoint",
        default="encodon_80m",
        description="Encodon checkpoint: encodon_80m | encodon_600m | encodon_1b | encodon_1b_cdwt.",
        reload_on_change=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="CUDA device to run CodonFM inference on.",
        include_in_key=False,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to score simultaneously on GPU.",
        include_in_key=False,
    )


def resolve_checkpoint_source(model_checkpoint: str) -> tuple[str, str, str, str]:
    """Resolve the HuggingFace download URLs + local cache names for an Encodon checkpoint.

    ``EncodonInference`` requires the ``.safetensors`` file and ``config.json`` in the same
    directory, so both are provisioned into one per-checkpoint cache subdir.

    Args:
        model_checkpoint (str): One of the ``CodonFMCheckpoint`` aliases.

    Returns:
        tuple[str, str, str, str]: ``(safetensors_url, config_url, safetensors_filename, cache_subdir)``.
    """
    entry = CODONFM_CHECKPOINTS[model_checkpoint]
    repo, filename = entry["repo"], entry["safetensors"]
    base = f"https://huggingface.co/{urllib.parse.quote(repo)}/resolve/main"
    subdir = f"codonfm-{model_checkpoint}-{hashlib.sha256(repo.encode()).hexdigest()[:8]}"
    return f"{base}/{urllib.parse.quote(filename)}", f"{base}/config.json", filename, subdir

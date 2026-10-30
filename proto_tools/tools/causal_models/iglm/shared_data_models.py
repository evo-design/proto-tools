"""Conditioning tokens, checkpoints, and generation schemas shared by the IgLM toolkit."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)

IGLM_MODEL_CHECKPOINTS = Literal["IgLM", "IgLM-S"]
IgLMChain = Literal["heavy", "light"]
IgLMSpecies = Literal["human", "mouse", "rat", "rabbit", "rhesus", "camel"]

# IgLM conditions on bracketed vocabulary tokens; callers name them in plain words.
IGLM_CHAIN_TOKENS: dict[str, str] = {"heavy": "[HEAVY]", "light": "[LIGHT]"}
IGLM_SPECIES_TOKENS: dict[str, str] = {
    "human": "[HUMAN]",
    "mouse": "[MOUSE]",
    "rat": "[RAT]",
    "rabbit": "[RABBIT]",
    "rhesus": "[RHESUS]",
    "camel": "[CAMEL]",
}

# Positional-embedding cap of both released checkpoints (GPT-2 n_positions=512).
IGLM_MAX_CONTEXT = 512
# Generation cap used by upstream's iglm.IgLM._generate.
IGLM_DEFAULT_MAX_LENGTH = 150


# ============================================================================
# Inputs
# ============================================================================
class IgLMSpan(BaseToolInput):
    """A residue span of an antibody sequence, 1-indexed and inclusive at both ends.

    ``IgLMSpan(start=99, end=106)`` covers residues 99 through 106 — eight residues,
    equivalently ``sequence[98:106]`` in Python slice terms, so the span length is
    ``end - start + 1`` and ``start=1`` is the first residue. This is the proto-tools
    convention for biological coordinates; IgLM upstream numbers the same region
    ``(98, 106)``, 0-indexed with an exclusive end, and the conversion happens once
    when the tool builds its worker payload.

    Attributes:
        start (int): First residue of the span (1-indexed, inclusive).
        end (int): Last residue of the span (1-indexed, inclusive).
    """

    start: int = InputField(
        ge=1,
        title="Start",
        description="First residue of the span (1-indexed, inclusive)",
    )
    end: int = InputField(
        ge=1,
        title="End",
        description="Last residue of the span (1-indexed, inclusive)",
    )

    @model_validator(mode="after")
    def validate_span_order(self) -> "IgLMSpan":
        """Reject spans whose end precedes their start."""
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        return self


class IgLMConditioningInput(BaseToolInput):
    """Per-item chain and species conditioning shared by every IgLM tool.

    Both fields are broadcastable: supply one value to condition every item the
    same way, or one value per item to mix chains and species in a single call.
    Values are case-insensitive and normalized to lower case, so ``"HUMAN"``,
    ``"Human"`` and ``"human"`` are the same conditioning.

    Attributes:
        chains (list[IgLMChain]): Chain type conditioning each item.
        species (list[IgLMSpecies]): Species-of-origin conditioning each item.
    """

    chains: list[IgLMChain] = InputField(
        default=["heavy"],
        broadcastable=True,
        title="Chains",
        description="Chain type per item; one value conditions every item the same way",
    )
    species: list[IgLMSpecies] = InputField(
        default=["human"],
        broadcastable=True,
        title="Species",
        description="Species-of-origin per item; one value conditions every item the same way",
    )

    @field_validator("chains", "species", mode="before")
    @classmethod
    def normalize_conditioning(cls, v: Any) -> Any:
        """Accept a bare value and any casing, normalizing to a lower-case list.

        The conditioning names read as ordinary words rather than identifiers, so
        ``"Heavy"`` and ``"HUMAN"`` are the same request as their lower-case forms;
        only genuinely unknown values should reach the caller as an error.
        """
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            return [item.strip().lower() if isinstance(item, str) else item for item in v]
        return v

    def chain_tokens(self) -> list[str]:
        """Return the model's chain token for each item, in input order."""
        return [IGLM_CHAIN_TOKENS[chain] for chain in self.chains]

    def species_tokens(self) -> list[str]:
        """Return the model's species token for each item, in input order."""
        return [IGLM_SPECIES_TOKENS[species] for species in self.species]


# ============================================================================
# Generation config
# ============================================================================
class IgLMGenerationConfig(BaseConfig):
    """Base configuration for IgLM sequence generation (full-length and infilling).

    Built on ``BaseConfig`` rather than the category's ``CausalModelSampleConfig``
    because IgLM generates a bundle of distinct sequences per input instead of one
    continuation per prompt: ``batch_size`` counts candidate draws rather than prompts,
    and ``prepend_prompt`` has no meaning for infilling, which always returns the whole
    re-designed chain. ``IgLMSampleConfig`` adds ``prepend_prompt`` for the one tool
    that can honour it.

    IgLM resamples until it holds ``num_sequences_per_input`` *distinct*
    sequences per item, discarding malformed generations the same way upstream
    does, so raising the count on a sharply-peaked distribution (low
    ``temperature``, low ``top_p``) costs progressively more attempts.

    Attributes:
        model_checkpoint (IGLM_MODEL_CHECKPOINTS): Weights variant. ``IgLM`` is the
            13M-parameter model from the paper; ``IgLM-S`` is a 1.4M-parameter model.
        num_sequences_per_input (int): Distinct sequences to generate for each input item.
            Defaults to 1 rather than upstream's 1000 so a default call stays cheap.
        temperature (float): Softmax temperature; lower is more conservative.
        top_p (float): Nucleus sampling cutoff over per-position token probabilities.
        max_length (int): Token budget for one generation, counting the two conditioning
            tokens and the prompt. Upstream uses 150; both checkpoints accept up to 512.
            A candidate that runs out of budget before completing IgLM's infilling
            format is discarded and charged against the attempt budget, so too small a
            value surfaces as "yielded only N of M requested unique sequences" rather
            than as short output. Raise it for long chains or long spans.
        batch_size (int): Candidate sequences drawn per forward pass.
        device (str): Device to run the model on.
    """

    model_checkpoint: IGLM_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="IgLM",
        description="IgLM weights variant; 'IgLM' is 13M parameters, 'IgLM-S' is 1.4M",
        reload_on_change=True,
    )
    num_sequences_per_input: int = ConfigField(
        title="Sequences Per Input",
        default=1,
        ge=1,
        description="Distinct sequences to generate for each input item",
    )
    temperature: float = ConfigField(
        title="Temperature",
        default=1.0,
        gt=0.0,
        description="Softmax temperature for sampling; lower is more deterministic",
    )
    top_p: float = ConfigField(
        title="Top P",
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling cutoff over per-position token probabilities",
    )
    max_length: int = ConfigField(
        title="Max Length",
        default=IGLM_DEFAULT_MAX_LENGTH,
        ge=1,
        le=IGLM_MAX_CONTEXT,
        description="Token budget per generation, including conditioning tokens and prompt",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=8,
        ge=1,
        description="Candidate sequences per GPU forward pass; raise for throughput, lower if OOM",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


# ============================================================================
# Generation output
# ============================================================================
class IgLMSamples(BaseModel):
    """The sequences IgLM generated for one input item.

    Attributes:
        sequences (list[str]): Distinct generated sequences, in generation order.
    """

    sequences: list[str] = Field(
        title="Sequences",
        description="Distinct sequences generated for this input, in generation order",
    )

    def __len__(self) -> int:
        """Get the number of generated sequences."""
        return len(self.sequences)

    def __getitem__(self, index: int) -> str:
        """Get a generated sequence by index."""
        return self.sequences[index]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over the generated sequences."""
        return iter(self.sequences)

    def fasta_headers(self, input_index: int) -> list[str]:
        """Return one FASTA header per sequence. Subclasses add what they produce per design."""
        return [f"input_{input_index}_seq_{j}" for j in range(len(self.sequences))]


class IgLMGenerationOutput(BaseToolOutput):
    """Output from IgLM sequence generation, with one bundle per input item.

    Attributes:
        results (list[IgLMSamples]): One ``IgLMSamples`` per input item, in input order.
    """

    results: list[IgLMSamples] = Field(
        title="Results",
        description="Generated sequences bundled per input item, in input order",
    )

    @property
    def sequences(self) -> list[str]:
        """Every generated sequence, flattened across input items in input order.

        Note this is ``num_sequences_per_input`` entries per item, unlike the category's
        ``CausalModelSampleOutput.sequences``, which holds exactly one per prompt. Use
        ``results[i].sequences`` to keep an item's bundle together.
        """
        return [sequence for result in self.results for sequence in result.sequences]

    def __len__(self) -> int:
        """Get the number of input items generated for."""
        return len(self.results)

    def __getitem__(self, index: int) -> IgLMSamples:
        """Get the sequence bundle for an input item by index."""
        return self.results[index]

    def __iter__(self) -> Iterator[IgLMSamples]:  # type: ignore[override]
        """Iterate over the per-input sequence bundles."""
        return iter(self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["fasta", "txt", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path)

        if file_format == "fasta":
            with open(path.with_suffix(".fasta"), "w") as f:
                f.writelines(
                    f">{header}\n{sequence}\n"
                    for i, result in enumerate(self.results)
                    for header, sequence in zip(result.fasta_headers(i), result.sequences, strict=True)
                )
        elif file_format == "txt":
            with open(path.with_suffix(".txt"), "w") as f:
                f.writelines(f"{sequence}\n" for sequence in self.sequences)
        elif file_format == "json":
            with open(path.with_suffix(".json"), "w") as f:
                json.dump([result.model_dump() for result in self.results], f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

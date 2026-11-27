"""Few-shot DNA design with pretrained Evo 2."""

import logging
import random
from itertools import product
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from proto_tools.tools.causal_models.shared_data_models import CausalModelSample, CausalModelSampleOutput
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, ConfigField, InputField, ToolInstance

logger = logging.getLogger(__name__)

ICL_SEPARATOR = "N"
ICL_MODEL_CHECKPOINT = "evo2_7b"
PREFILL_BASES_PER_BATCH = 4096
# NCBI standard genetic code (table 1), in T/C/A/G order at each codon position.
_STANDARD_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_CODON_TO_AA = dict(zip(("".join(bases) for bases in product("TCAG", repeat=3)), _STANDARD_AAS, strict=True))
_AA_TO_CODONS = {
    aa: tuple(codon for codon, residue in _CODON_TO_AA.items() if residue == aa)
    for aa in sorted(set(_STANDARD_AAS) - {"*"})
}


class Evo2ICLInput(BaseToolInput):
    """Pool of DNA or protein examples for in-context design.

    Attributes:
        examples (list[str]): Sequences in FASTA record order, without headers.
            Whitespace is removed and sequences are uppercased.
        sequence_type (Literal["dna", "protein"]): DNA accepts A/C/G/T; protein accepts the 20 standard amino acids.
    """

    sequence_type: Literal["dna", "protein"] = InputField(
        title="Sequence type",
        default="dna",
        description="DNA examples or proteins reverse-translated with the standard genetic code",
    )
    examples: list[str] = InputField(
        title="Example sequences",
        description="One FASTA record per example; select examples per prompt with n_shots",
        min_length=1,
    )

    @field_validator("examples", mode="before")
    @classmethod
    def normalize_single_example(cls, value: Any) -> Any:
        """Allow a single sequence string at the Python call site."""
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def normalize_examples(self) -> "Evo2ICLInput":
        """Validate the selected alphabet while preserving order and multiplicity."""
        normalized = []
        alphabet = set(_AA_TO_CODONS) if self.sequence_type == "protein" else set("ACGT")
        for i, raw_sequence in enumerate(self.examples, start=1):
            sequence = "".join(raw_sequence.split()).upper()
            if not sequence or set(sequence) - alphabet:
                requirement = (
                    "the 20 standard amino acids (no stop or ambiguous residues)"
                    if self.sequence_type == "protein"
                    else "A, C, G, and T; N is reserved for separators"
                )
                raise ValueError(f"Example {i} must contain only {requirement}")
            normalized.append(sequence)
        object.__setattr__(self, "examples", normalized)
        return self

    def dna_examples(self, seed: int | None) -> list[str]:
        """Reverse-translate the pool once, sampling synonymous codons uniformly.

        Args:
            seed (int | None): The codon-selection seed; None draws fresh codons.

        Returns:
            list[str]: DNA examples in original order, reused across prompt draws.
        """
        if self.sequence_type == "dna":
            return list(self.examples)
        rng = random.Random(seed)  # noqa: S311 — reproducible biological sampling, not cryptography
        return ["".join(rng.choice(_AA_TO_CODONS[aa]) for aa in sequence) for sequence in self.examples]


class Evo2ICLConfig(BaseConfig):
    """Sampling controls for one set of ICL-conditioned designs.

    Attributes:
        num_sequences (int): Designs per composed prompt, without a library-level count cap.
        n_shots (int): Examples selected without replacement per prompt, capped by pool size.
        n_prompts (int): Number of prompt draws, each producing num_sequences designs.
        fixed_examples (bool): Use the first n_shots records in order instead of random selection.
        example_seed (int): Seed for example selection, independent of generation.
        spacer (str): DNA delimiter preceding each example and ending each prompt.
        spacer_both_sides (bool): Also append the delimiter after every example.
        min_example_len (int): Minimum length required for every example in the pool.
        force_prompt_threshold (int | None): Compatibility option; vLLM manages chunked prefill.
        codon_seed (int): Seed for uniform reverse translation, independent of generation.
        seed (int | None): Generation seed; None draws fresh designs.
        sequence_length (int): New bases (DNA mode) or amino acids (protein mode) per design;
            the example sequences and separators are excluded from the output.
        temperature (float): Sampling randomness. Lower values favor likely
            nucleotides; higher values explore more alternatives.
        top_k (int): Number of highest-probability tokens considered at each step.
        top_p (float): Cumulative probability cutoff for nucleus sampling.
        batch_size (int): Compatibility option; vLLM schedules designs automatically.
        device (str): Device to run the model on.
        timeout (int | None): Maximum execution time in seconds.
    """

    num_sequences: int = ConfigField(
        title="Designs per prompt",
        default=20,
        ge=1,
        description="Number of designs generated for each composed prompt",
    )
    n_shots: int = ConfigField(
        title="Examples per prompt",
        default=16,
        ge=1,
        description="Examples drawn without replacement; uses all available records if fewer are supplied",
    )
    n_prompts: int = ConfigField(
        title="Number of prompts",
        default=1,
        ge=1,
        description="Independent example draws; each prompt generates num_sequences designs",
    )
    fixed_examples: bool = ConfigField(
        title="Use first examples in order",
        default=False,
        description="Use the first n_shots FASTA records for every prompt instead of sampling examples",
    )
    example_seed: int = ConfigField(
        title="Example-selection seed",
        default=42,
        ge=0,
        lt=2**32,
        description="Reproduces the example draws; independent of the generation seed",
    )
    spacer: str = ConfigField(
        title="Example separator",
        default=ICL_SEPARATOR,
        description="A/C/G/T/N delimiter before each example and at the end; empty joins examples directly",
    )
    spacer_both_sides: bool = ConfigField(
        title="Wrap both sides of each example",
        default=False,
        description="Also append a separator after each example, doubling internal and trailing separators",
    )
    min_example_len: int = ConfigField(
        title="Minimum example length",
        default=1,
        ge=1,
        description="Reject any shorter record, measured in bases for DNA or amino acids for protein",
    )
    force_prompt_threshold: int | None = ConfigField(
        title="Prompt prefill (bases)",
        default=None,
        ge=1,
        description="Compatibility option; vLLM manages chunked prefill automatically",
    )
    codon_seed: int = ConfigField(
        title="Codon-selection seed",
        default=42,
        ge=0,
        lt=2**32,
        description="Seed for uniform protein reverse translation; independent of generation; ignored for DNA",
    )
    seed: int | None = ConfigField(
        title="Generation seed",
        default=None,
        ge=0,
        lt=2**32,
        description="Seed for model sampling; unset draws fresh designs",
    )
    sequence_length: int = ConfigField(
        title="Design length",
        default=200,
        ge=1,
        le=4096,
        description="New bases for DNA input or amino acids for protein input; always generate the full length",
    )
    temperature: float = ConfigField(
        title="Temperature",
        default=1.0,
        gt=0.0,
        description="Sampling randomness; lower favors likely bases, higher increases diversity",
    )
    top_k: int = ConfigField(
        title="Top K",
        default=4,
        ge=1,
        le=512,
        description="Consider this many likely tokens at each step; the draft uses 4",
    )
    top_p: float = ConfigField(
        title="Top P",
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Probability mass retained at each step; the draft uses 0.9",
    )
    batch_size: int = ConfigField(
        title="Batch size",
        default=1,
        ge=1,
        le=8,
        description="Compatibility option; vLLM schedules designs automatically",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=14400,
        ge=1,
        description="Maximum execution time in seconds",
        include_in_key=False,
    )

    @field_validator("spacer")
    @classmethod
    def normalize_spacer(cls, value: str) -> str:
        """Accept DNA delimiters and normalize their case and whitespace."""
        value = "".join(value.split()).upper()
        if set(value) - set("ACGTN"):
            raise ValueError("The example separator must contain only A, C, G, T, and N")
        return value

    @model_validator(mode="after")
    def validate_prefill_threshold(self) -> "Evo2ICLConfig":
        """Require a power-of-two explicit parallel prefill length."""
        threshold = self.force_prompt_threshold
        if threshold is not None and threshold & (threshold - 1):
            raise ValueError("Prompt prefill must be a power of two")
        return self

    @property
    def total_sequences(self) -> int:
        """Return the total designs requested across all prompts."""
        return self.n_prompts * self.num_sequences

    @property
    def prefill_threshold(self) -> int:
        """Return explicit parallel prefill or a batch-aware power of two."""
        if self.force_prompt_threshold is not None:
            return self.force_prompt_threshold
        return 1 << ((PREFILL_BASES_PER_BATCH // self.batch_size).bit_length() - 1)

    def compose_prompts(self, inputs: Evo2ICLInput) -> tuple[list[str], list[list[int]]]:
        """Compose prompts from the example pool without loading the model.

        Args:
            inputs (Evo2ICLInput): Example pool in FASTA record order.

        Returns:
            tuple[list[str], list[list[int]]]: Prompts and their selected 1-indexed example positions.
        """
        import numpy as np

        for i, sequence in enumerate(inputs.examples, start=1):
            if len(sequence) < self.min_example_len:
                raise ValueError(f"Example {i} is shorter than min_example_len={self.min_example_len}")
        examples = inputs.dna_examples(self.codon_seed)
        rng = np.random.RandomState(self.example_seed)
        shots = min(self.n_shots, len(inputs.examples))
        prompts = []
        selections = []
        for _ in range(self.n_prompts):
            indices = (
                list(range(shots))
                if self.fixed_examples
                else [int(i) for i in rng.choice(len(inputs.examples), size=shots, replace=False)]
            )
            selected = [examples[i] for i in indices]
            prompts.append(
                "".join(self.spacer + seq + (self.spacer if self.spacer_both_sides else "") for seq in selected)
                + self.spacer
            )
            selections.append([i + 1 for i in indices])
        return prompts, selections


class Evo2ICLOutput(CausalModelSampleOutput):
    """Generated designs in the input alphabet, with FASTA export.

    Attributes:
        results (list[CausalModelSample]): Requested-length DNA or protein designs.
            Protein translation starts at the first generated base; stops remain *
            and ambiguous codons become X, with warnings. No early stopping is used.
        sequence_type (Literal["dna", "protein"]): Alphabet of the returned designs.
        dna_sequences (list[str]): Corresponding generated DNA for protein results, in matching order.
    """

    sequence_type: Literal["dna", "protein"] = Field(
        default="dna", title="Sequence type", description="Alphabet of the returned designs"
    )
    dna_sequences: list[str] = Field(
        default_factory=list,
        title="Generated DNA",
        description="Underlying DNA for protein designs, in matching order; empty for DNA-mode output",
    )


def example_input() -> Evo2ICLInput:
    """Return small illustrative DNA examples, not a curated cell-type prompt."""
    return Evo2ICLInput(examples=["ACGTACGTACGT", "TGCATGCATGCA"])


@tool(
    key="evo2-icl",
    label="Evo 2 In-context Design",
    category="causal_models",
    input_class=Evo2ICLInput,
    config_class=Evo2ICLConfig,
    output_class=Evo2ICLOutput,
    description="Generate DNA or protein designs from FASTA examples with pretrained Evo 2",
    uses_gpu=True,
    gpu_only=True,
    pin_visible_devices=True,
    stochastic=True,
    example_input=example_input,
)
def run_evo2_icl(inputs: Evo2ICLInput, config: Evo2ICLConfig, instance: Any = None) -> Evo2ICLOutput:
    """Generate DNA designs from configurable draws of in-context examples.

    Args:
        inputs (Evo2ICLInput): Example pool in FASTA record order.
        config (Evo2ICLConfig): Design count, length, and sampling controls.
        instance (Any): Optional persistent Evo 2 worker.

    Returns:
        Evo2ICLOutput: New DNA designs, sampling provenance, and any yield warning.

    Examples:
        >>> from proto_tools.tools import Evo2ICLInput, Evo2ICLConfig, run_evo2_icl
        >>> inputs = Evo2ICLInput(examples=["ACGTACGT", "TGCATGCA"])
        >>> result = run_evo2_icl(inputs, Evo2ICLConfig(num_sequences=20, sequence_length=200))
        >>> result.export("designs", file_format="fasta")
    """
    logger.debug("Using the persistent Evo 2 worker for in-context DNA design")
    prompts, selections = config.compose_prompts(inputs)
    request_prompts = [prompt for prompt in prompts for _ in range(config.num_sequences)]
    protein = inputs.sequence_type == "protein"
    generated_bases = config.sequence_length * (3 if protein else 1)
    result = ToolInstance.dispatch(
        "evo2",
        {
            "operation": "sample",
            "prompts": request_prompts,
            "model_checkpoint": ICL_MODEL_CHECKPOINT,
            "local_path": None,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "temperature": config.temperature,
            "max_new_tokens": generated_bases,
            "cached_generation": True,
            "force_prompt_threshold": config.prefill_threshold,
            "max_seqlen": None,
            "skip_special_tokens": False,
            "stop_at_eos": False,
            "old_kv_cache": None,
            "return_kv_cache": False,
            "return_logits": False,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )
    sequences = result["sequences"]
    if len(sequences) != config.total_sequences:
        raise ValueError(f"Evo 2 returned {len(sequences)} generations for {config.total_sequences} requested designs")
    valid_indices = [
        i
        for i, sequence in enumerate(sequences)
        if len(sequence) == generated_bases and not set(sequence) - set("ACGTN")
    ]
    dna_sequences = [sequences[i] for i in valid_indices]
    translated = (
        [
            "".join(_CODON_TO_AA.get(sequence[i : i + 3], "X") for i in range(0, len(sequence), 3))
            for sequence in dna_sequences
        ]
        if protein
        else dna_sequences
    )
    designs = [CausalModelSample(sequence=sequence) for sequence in translated]
    if not designs:
        raise ValueError("No usable A/C/G/T/N designs were generated. Try different examples or sampling settings.")
    omitted = len(sequences) - len(designs)
    warnings = []
    if omitted:
        warnings.append(
            f"Returned {len(designs)} of {config.total_sequences} requested designs; "
            f"{omitted} generations were empty, invalid, or did not meet the length requirement."
        )
    if protein:
        stop_count = sum("*" in sequence for sequence in translated)
        ambiguous_count = sum("X" in sequence for sequence in translated)
        if stop_count:
            warnings.append(
                f"{stop_count} protein designs contain stop codons (*); full-length translations are retained."
            )
        if ambiguous_count:
            warnings.append(f"{ambiguous_count} protein designs contain ambiguous codons, shown as X.")
    return Evo2ICLOutput(
        results=designs,
        sequence_type=inputs.sequence_type,
        dna_sequences=dna_sequences if protein else [],
        warnings=warnings,
        metadata={
            "model_checkpoint": ICL_MODEL_CHECKPOINT,
            "num_examples": len(inputs.examples),
            "prompt_length": max(map(len, prompts)),
            "prompt_lengths": list(map(len, prompts)),
            "prompt_example_indices": selections,
            "design_prompt_indices": [i // config.num_sequences + 1 for i in valid_indices],
            "n_shots": config.n_shots,
            "examples_per_prompt": len(selections[0]),
            "n_prompts": config.n_prompts,
            "num_sequences_per_prompt": config.num_sequences,
            "fixed_examples": config.fixed_examples,
            "example_seed": config.example_seed,
            "min_example_len": config.min_example_len,
            "spacer_both_sides": config.spacer_both_sides,
            "separator": config.spacer,
            "num_requested": config.total_sequences,
            "num_generated": len(designs),
            "num_omitted": omitted,
            "sequence_length": config.sequence_length,
            "sequence_type": inputs.sequence_type,
            "generated_bases_per_design": generated_bases,
            "genetic_code": 1 if protein else None,
            "reverse_translation": "uniform" if protein else None,
            "codon_seed": config.codon_seed if protein else None,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "batch_size": config.batch_size,
            "prefill_threshold": config.prefill_threshold,
            "seed": config.seed,
        },
    )

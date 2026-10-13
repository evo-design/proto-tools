"""Few-shot DNA design with pretrained Evo 2."""

import logging
from typing import Any

from pydantic import field_validator, model_validator

from proto_tools.tools.causal_models.shared_data_models import CausalModelSample, CausalModelSampleOutput
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, ConfigField, InputField, ToolInstance

logger = logging.getLogger(__name__)

ICL_SEPARATOR = "N"
ICL_MODEL_CHECKPOINT = "evo2_7b"
MAX_PROMPT_BASES = 32_768
MAX_GENERATED_BASES = 1_000_000
PREFILL_BASES_PER_BATCH = 4096


class Evo2ICLInput(BaseToolInput):
    """Example DNA sequences that jointly specify one design task.

    Attributes:
        examples (list[str]): Example sequences, in FASTA record order. Headers
            are not included in the model prompt. Whitespace is removed and
            sequences are uppercased; examples must contain only A, C, G, and T.
    """

    examples: list[str] = InputField(
        title="Example DNA sequences",
        description="One FASTA record per example; all records condition the same design task",
        min_length=1,
    )

    @field_validator("examples", mode="before")
    @classmethod
    def normalize_single_example(cls, value: Any) -> Any:
        """Allow a single DNA string at the Python call site."""
        return [value] if isinstance(value, str) else value

    @field_validator("examples")
    @classmethod
    def normalize_examples(cls, value: list[str]) -> list[str]:
        """Normalize sequence text while preserving example order and multiplicity."""
        normalized = []
        for i, raw_sequence in enumerate(value, start=1):
            sequence = "".join(raw_sequence.split()).upper()
            if not sequence or set(sequence) - set("ACGT"):
                raise ValueError(f"Example {i} must contain only A, C, G, and T; N is reserved for separators")
            normalized.append(sequence)
        prompt_length = sum(map(len, normalized)) + len(normalized) + 1
        if prompt_length > MAX_PROMPT_BASES:
            raise ValueError(f"The assembled prompt must fit within {MAX_PROMPT_BASES:,} bases")
        return normalized

    @property
    def prompt(self) -> str:
        """Return example sequences flanked and separated by one N."""
        return ICL_SEPARATOR + ICL_SEPARATOR.join(self.examples) + ICL_SEPARATOR


class Evo2ICLConfig(BaseConfig):
    """Sampling controls for one set of ICL-conditioned designs.

    Attributes:
        num_sequences (int): Requested number of designs, up to 1,000 per run.
        sequence_length (int): Number of newly generated nucleotides per design;
            the example sequences and separators are excluded from the output.
        temperature (float): Sampling randomness. Lower values favor likely
            nucleotides; higher values explore more alternatives.
        top_k (int): Number of highest-probability tokens considered at each step.
        top_p (float): Cumulative probability cutoff for nucleus sampling.
        batch_size (int): Number of designs decoded together on the GPU.
        device (str): Device to run the model on.
        timeout (int | None): Maximum execution time in seconds.
    """

    num_sequences: int = ConfigField(
        title="Number of designs",
        default=10,
        ge=1,
        le=1000,
        description="Designs to generate together from the example set (up to 1,000)",
    )
    sequence_length: int = ConfigField(
        title="Design length (bp)",
        default=200,
        ge=1,
        le=4096,
        description="New nucleotides per design; 200 bp is the draft's enhancer-design setting",
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
        description="Designs decoded together; lower this if GPU memory runs out",
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

    @model_validator(mode="after")
    def validate_generation_budget(self) -> "Evo2ICLConfig":
        """Bound hosted generation work independently of prompt length."""
        if self.num_sequences * self.sequence_length > MAX_GENERATED_BASES:
            raise ValueError("One run supports at most 1,000,000 generated bases; reduce design count or length")
        return self

    @property
    def prefill_threshold(self) -> int:
        """Bound parallel prompt prefill with a power-of-two prefix per design."""
        return 1 << ((PREFILL_BASES_PER_BATCH // self.batch_size).bit_length() - 1)

    def validate_workload(self, inputs: Evo2ICLInput) -> None:
        """Bound generated tokens plus prompt tokens that need sequential processing."""
        forced_bases = max(0, len(inputs.prompt) - self.prefill_threshold)
        if self.num_sequences * (self.sequence_length + forced_bases) > MAX_GENERATED_BASES:
            raise ValueError("This example set is too large for this many designs; reduce examples or design count")


class Evo2ICLOutput(CausalModelSampleOutput):
    """Generated DNA designs, with FASTA export inherited from the sampling output.

    Attributes:
        results (list[CausalModelSample]): Newly generated designs containing
            only A, C, G, and T, at the requested length. Invalid or prematurely
            terminated generations are omitted and reported in warnings.
    """


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
    description="Generate DNA designs from FASTA examples with pretrained Evo 2, without retraining",
    uses_gpu=True,
    stochastic=True,
    example_input=example_input,
)
def run_evo2_icl(inputs: Evo2ICLInput, config: Evo2ICLConfig, instance: Any = None) -> Evo2ICLOutput:
    """Generate DNA designs conditioned on a shared set of in-context examples.

    Args:
        inputs (Evo2ICLInput): Ordered DNA examples defining one prompt.
        config (Evo2ICLConfig): Design count, length, and sampling controls.
        instance (Any): Optional persistent Evo 2 worker.

    Returns:
        Evo2ICLOutput: New DNA designs, sampling provenance, and any yield warning.

    Examples:
        >>> from proto_tools.tools import Evo2ICLInput, Evo2ICLConfig, run_evo2_icl
        >>> inputs = Evo2ICLInput(examples=["ACGTACGT", "TGCATGCA"])
        >>> result = run_evo2_icl(inputs, Evo2ICLConfig(num_sequences=10, sequence_length=200))
        >>> result.export("designs", file_format="fasta")
    """
    logger.debug("Using the persistent Evo 2 worker for in-context DNA design")
    config.validate_workload(inputs)
    result = ToolInstance.dispatch(
        "evo2",
        {
            "operation": "sample",
            "prompts": [inputs.prompt] * config.num_sequences,
            "model_checkpoint": ICL_MODEL_CHECKPOINT,
            "local_path": None,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "temperature": config.temperature,
            "max_new_tokens": config.sequence_length,
            "cached_generation": True,
            "force_prompt_threshold": config.prefill_threshold,
            "max_seqlen": None,
            "skip_special_tokens": False,
            "stop_at_eos": True,
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
    if len(sequences) != config.num_sequences:
        raise ValueError(f"Evo 2 returned {len(sequences)} generations for {config.num_sequences} requested designs")
    designs = [
        CausalModelSample(sequence=sequence)
        for sequence in sequences
        if len(sequence) == config.sequence_length and not set(sequence) - set("ACGT")
    ]
    if not designs:
        raise ValueError("No full-length A/C/G/T designs were generated. Try different examples or sampling settings.")
    omitted = len(sequences) - len(designs)
    warnings = []
    if omitted:
        warnings.append(
            f"Returned {len(designs)} of {config.num_sequences} requested designs; "
            f"{omitted} generations were short or contained non-ACGT tokens."
        )
    return Evo2ICLOutput(
        results=designs,
        warnings=warnings,
        metadata={
            "model_checkpoint": ICL_MODEL_CHECKPOINT,
            "num_examples": len(inputs.examples),
            "prompt_length": len(inputs.prompt),
            "separator": ICL_SEPARATOR,
            "num_requested": config.num_sequences,
            "num_generated": len(designs),
            "num_omitted": omitted,
            "sequence_length": config.sequence_length,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "batch_size": config.batch_size,
            "prefill_threshold": config.prefill_threshold,
            "seed": config.seed,
        },
    )

"""Pure-CPU mock of a stochastic LM sampler with internal batching.

Implements an LM-shaped two-step inference: a deterministic "forward
pass" maps each prompt to a logits vector over a token vocabulary, then
per-item weighted draws from a single ``random.Random(seed)`` stream
pick SEQ_LEN tokens. Used to e2e-test the framework's stochastic-iterable
cache/dedup routing without GPU or model load.

``items_processed`` is exposed on the output so tests can directly
observe whether the framework dedup'd before the tool ran.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField, InputField

VOCAB = "ACDEFGHIKLMNPQRSTVWY"
SEQ_LEN = 4  # tokens per completion; 20^4 outputs makes collisions negligible


def _logits_from_prompt(prompt: str) -> list[int]:
    """Deterministic 'forward pass': prompt → logits (positive weights) over VOCAB.

    A real LM's transformer computes per-token logits from the prompt
    via embedding + attention. We hash the prompt to get a deterministic
    weight vector that varies per prompt but stays constant across
    runs. Same prompt always produces the same distribution; different
    prompts produce different distributions.

    Args:
        prompt (str): Input prompt to "embed".

    Returns:
        list[int]: One positive weight per token in VOCAB. Used as
            ``weights=`` to ``random.choices`` for weighted sampling.
    """
    h = hashlib.sha256(prompt.encode()).digest()
    return [int(h[i]) + 1 for i in range(len(VOCAB))]  # +1 to avoid zero weights


class MockIterableStochasticInput(BaseToolInput):
    """Input for the stochastic iterable mock.

    Attributes:
        prompts (list[str]): Prompts to generate from. Plays the role of
            a batched LM input.
    """

    prompts: list[str] = InputField(title="Prompts", description="Prompts to generate completions for")


class MockIterableStochasticConfig(BaseConfig):
    """Configuration for the stochastic iterable mock.

    Attributes:
        batch_size (int): Internal batch size for the tool's sampling
            loop. Mimics an LM's ``batch_size`` setting — the framework
            doesn't see or care about this; it's an implementation
            detail of the tool's inference path.
    """

    batch_size: int = ConfigField(
        title="Batch Size",
        default=2,
        ge=1,
        description="Tool-internal batch size for the sampling loop",
    )


class MockIterableStochasticOutput(BaseToolOutput):
    """Output from the stochastic iterable mock.

    Attributes:
        completions (list[str]): Per-prompt completions, in input order.
            Each completion is SEQ_LEN characters drawn from VOCAB
            weighted by the prompt's logits.
        items_processed (int): Number of items the tool's function
            actually received in this call (before framework
            stitch/expand). Tests assert this matches ``len(prompts)``
            even with duplicates — that's the no-dedup invariant for
            stochastic iterables.
    """

    completions: list[str] = Field(
        default_factory=list,
        title="Completions",
        description="Per-prompt sampled tokens",
    )
    items_processed: int = Field(
        default=0,
        title="Items Processed",
        description="Number of items the tool function received from the framework",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with open(path, "w") as f:
                json.dump(
                    {
                        "completions": self.completions,
                        "items_processed": self.items_processed,
                    },
                    f,
                    indent=2,
                )
        else:
            raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> MockIterableStochasticInput:
    """Minimal valid input for testing and examples."""
    return MockIterableStochasticInput(prompts=["AA", "AA", "BB"])


@tool(
    key="mock-iterable-stochastic",
    label="Mock Iterable Stochastic LLM",
    category="testing",
    input_class=MockIterableStochasticInput,
    config_class=MockIterableStochasticConfig,
    output_class=MockIterableStochasticOutput,
    description="Pure-CPU mock of a stochastic LM sampler with internal batching — for e2e testing of stochastic-iterable cache/dedup routing.",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["prompts"],
    iterable_output_field="completions",
    cacheable=True,
    stochastic=True,
)
def run_mock_iterable_stochastic(
    inputs: MockIterableStochasticInput,
    config: MockIterableStochasticConfig,
    instance: Any = None,  # noqa: ARG001 — required by tool interface
) -> MockIterableStochasticOutput:
    """Sample SEQ_LEN tokens per prompt by weighted draws from prompt-derived logits.

    The single ``random.Random`` instance is the analogue of torch's
    global RNG state: each weighted draw consumes it and advances state,
    so two identical prompts in the same internal batch see the same
    logits but pick different tokens — the mechanism stochastic-iterable
    routing relies on.

    Args:
        inputs (MockIterableStochasticInput): Prompts to sample for.
        config (MockIterableStochasticConfig): Sampling configuration.
        instance (Any): Optional ToolInstance (unused — this tool runs
            in-process).

    Returns:
        MockIterableStochasticOutput: Completions plus ``items_processed``
            (count of items the framework handed to the tool function).
    """
    seed = config.seed if config.seed is not None else random.randint(0, 2**31 - 1)  # noqa: S311
    rng = random.Random(seed)  # noqa: S311 -- not cryptographic

    prompts = list(inputs.prompts)
    vocab_chars = list(VOCAB)
    completions: list[str] = []

    for batch_start in range(0, len(prompts), config.batch_size):
        batch = prompts[batch_start : batch_start + config.batch_size]
        batch_logits = [_logits_from_prompt(prompt) for prompt in batch]
        for logits in batch_logits:
            tokens = rng.choices(vocab_chars, weights=logits, k=SEQ_LEN)
            completions.append("".join(tokens))

    return MockIterableStochasticOutput(
        completions=completions,
        items_processed=len(prompts),
    )

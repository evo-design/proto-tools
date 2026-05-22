"""Pure-CPU serial-loop counterpart to ``mock-iterable-stochastic``.

Same generation contract — prompt-derived logits, one weighted draw
per item, single ``random.Random(seed)`` stream — but no internal
batching. Items are processed one at a time, mirroring tools whose
inference path is a pure ``for prompt in prompts:`` loop.

Outputs are bit-identical to the batched variant for the same seed
because both share the same per-item draw contract.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, InputField

VOCAB = "ACDEFGHIKLMNPQRSTVWY"
SEQ_LEN = 4  # matches the batched mock so outputs are bit-identical for the same seed


def _logits_from_prompt(prompt: str) -> list[int]:
    """Deterministic 'forward pass': prompt → logits over VOCAB.

    Same hash-based mapping as in ``mock_iterable_stochastic`` so the
    two mocks produce bit-identical outputs for the same seed.

    Args:
        prompt (str): Input prompt.

    Returns:
        list[int]: One positive weight per VOCAB token.
    """
    h = hashlib.sha256(prompt.encode()).digest()
    return [int(h[i]) + 1 for i in range(len(VOCAB))]


class MockIterableStochasticSerialInput(BaseToolInput):
    """Input for the serial stochastic iterable mock.

    Attributes:
        prompts (list[str]): Prompts to generate from, processed one at
            a time (no internal batching).
    """

    prompts: list[str] = InputField(title="Prompts", description="Prompts to generate completions for")


class MockIterableStochasticSerialConfig(BaseConfig):
    """Configuration for the serial stochastic iterable mock.

    No ``batch_size`` field — by design this tool processes items
    one at a time, mirroring tools whose inference path is a pure
    ``for prompt in prompts:`` loop.
    """


class MockIterableStochasticSerialOutput(BaseToolOutput):
    """Output from the serial stochastic iterable mock.

    Attributes:
        completions (list[str]): Per-prompt sampled tokens, in input
            order. Each completion is one character drawn from VOCAB
            weighted by the prompt's logits.
        items_processed (int): Number of items the tool function
            actually received from the framework. With stochastic-iterable
            routing this equals ``len(prompts)`` even for duplicate
            inputs.
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


def example_input() -> MockIterableStochasticSerialInput:
    """Minimal valid input for testing and examples."""
    return MockIterableStochasticSerialInput(prompts=["AA", "AA", "BB"])


@tool(
    key="mock-iterable-stochastic-serial",
    label="Mock Iterable Stochastic LM (serial)",
    category="testing",
    input_class=MockIterableStochasticSerialInput,
    config_class=MockIterableStochasticSerialConfig,
    output_class=MockIterableStochasticSerialOutput,
    description="Pure-CPU mock of a serial stochastic LM (no internal batching) — for e2e testing of stochastic-iterable routing on tools that loop one item at a time.",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_field="prompts",
    iterable_output_field="completions",
    cacheable=True,
    stochastic=True,
)
def run_mock_iterable_stochastic_serial(
    inputs: MockIterableStochasticSerialInput,
    config: MockIterableStochasticSerialConfig,
    instance: Any = None,  # noqa: ARG001 — required by tool interface
) -> MockIterableStochasticSerialOutput:
    """Sample one token per prompt in a pure serial loop.

    The single ``random.Random`` instance is the analogue of torch's
    global RNG state: each weighted draw consumes it and advances state,
    so two identical prompts processed back-to-back see the same logits
    but pick different tokens — the same mechanism as the batched
    variant, just without internal batching.

    Args:
        inputs (MockIterableStochasticSerialInput): Prompts to sample for.
        config (MockIterableStochasticSerialConfig): Sampling configuration.
        instance (Any): Optional ToolInstance (unused — this tool runs
            in-process).

    Returns:
        MockIterableStochasticSerialOutput: Completions plus ``items_processed``
            (count of items the framework handed to the tool function).
    """
    seed = config.seed if config.seed is not None else random.randint(0, 2**31 - 1)  # noqa: S311
    rng = random.Random(seed)  # noqa: S311 -- not cryptographic

    prompts = list(inputs.prompts)
    vocab_chars = list(VOCAB)

    completions = [
        "".join(rng.choices(vocab_chars, weights=_logits_from_prompt(prompt), k=SEQ_LEN)) for prompt in prompts
    ]

    return MockIterableStochasticSerialOutput(
        completions=completions,
        items_processed=len(prompts),
    )

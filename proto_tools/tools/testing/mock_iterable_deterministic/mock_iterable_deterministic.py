"""Pure-CPU mock of a deterministic per-item scorer with internal batching.

Counterpart to ``mock-iterable-stochastic`` — same prompt-to-logits
"forward pass" but with greedy argmax decoding instead of weighted
sampling. No RNG. Used to e2e-test the framework's per-item
dedup/cache path for deterministic iterable tools.

``items_processed`` is exposed on the output so tests can directly
observe that dedup-collapse happened.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField, InputField

VOCAB = "ACDEFGHIKLMNPQRSTVWY"


def _logits_from_prompt(prompt: str) -> list[int]:
    """Deterministic 'forward pass': prompt → logits over VOCAB.

    Same hash-based mapping as in ``mock_iterable_stochastic``; this
    mock takes the argmax (greedy decoding) instead of sampling.

    Args:
        prompt (str): Input prompt.

    Returns:
        list[int]: One positive weight per VOCAB token.
    """
    h = hashlib.sha256(prompt.encode()).digest()
    return [int(h[i]) + 1 for i in range(len(VOCAB))]


class MockIterableDeterministicInput(BaseToolInput):
    """Input for the deterministic iterable mock.

    Attributes:
        prompts (list[str]): Prompts to score. Plays the role of a
            batched scoring input.
    """

    prompts: list[str] = InputField(title="Prompts", description="Prompts to score")


class MockIterableDeterministicConfig(BaseConfig):
    """Configuration for the deterministic iterable mock.

    Attributes:
        batch_size (int): Internal batch size for the scoring loop.
            Mimics a real scorer's batch_size knob.
    """

    batch_size: int = ConfigField(
        title="Batch Size",
        default=2,
        ge=1,
        description="Tool-internal batch size for the scoring loop",
    )


class MockIterableDeterministicOutput(BaseToolOutput):
    """Output from the deterministic iterable mock.

    Attributes:
        scores (list[str]): Per-prompt scores, in input order. Each
            score is the argmax-token of the prompt's logits — fully
            determined by the prompt.
        items_processed (int): Number of items the tool function
            actually received in this call (before framework
            stitch/expand). For duplicate inputs to a deterministic
            iterable tool, this is the *deduped* count — strictly less
            than ``len(scores)`` once the framework expands the
            stitched output.
    """

    scores: list[str] = Field(
        default_factory=list,
        title="Scores",
        description="Per-prompt deterministic argmax tokens",
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
                        "scores": self.scores,
                        "items_processed": self.items_processed,
                    },
                    f,
                    indent=2,
                )
        else:
            raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> MockIterableDeterministicInput:
    """Minimal valid input for testing and examples."""
    return MockIterableDeterministicInput(prompts=["AA", "BBB", "CCCC"])


@tool(
    key="mock-iterable-deterministic",
    label="Mock Iterable Deterministic Scorer",
    category="testing",
    input_class=MockIterableDeterministicInput,
    config_class=MockIterableDeterministicConfig,
    output_class=MockIterableDeterministicOutput,
    description="Pure-CPU mock of a deterministic per-item scorer with internal batching — for e2e testing of deterministic-iterable dedup behavior.",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_field="prompts",
    iterable_output_field="scores",
    cacheable=True,
)
def run_mock_iterable_deterministic(
    inputs: MockIterableDeterministicInput,
    config: MockIterableDeterministicConfig,
    instance: Any = None,  # noqa: ARG001 — required by tool interface
) -> MockIterableDeterministicOutput:
    """Score each prompt by greedy decoding (argmax of logits) in internal batches.

    Args:
        inputs (MockIterableDeterministicInput): Prompts to score.
        config (MockIterableDeterministicConfig): Scoring configuration.
        instance (Any): Optional ToolInstance (unused).

    Returns:
        MockIterableDeterministicOutput: Per-prompt argmax tokens plus
            ``items_processed`` (count of items the framework handed to
            the tool function).
    """
    prompts = list(inputs.prompts)
    scores: list[str] = []

    for batch_start in range(0, len(prompts), config.batch_size):
        batch = prompts[batch_start : batch_start + config.batch_size]
        # "Forward pass" — logits per prompt
        batch_logits = [_logits_from_prompt(prompt) for prompt in batch]
        # Greedy decoding — argmax token per prompt (no RNG)
        batch_scores = [VOCAB[max(range(len(logits)), key=logits.__getitem__)] for logits in batch_logits]
        scores.extend(batch_scores)

    return MockIterableDeterministicOutput(
        scores=scores,
        items_processed=len(prompts),
    )

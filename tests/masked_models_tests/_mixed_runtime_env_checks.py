"""In-env numerical checks for the shared mixed protein/DNA MLM runtime.

``standalone_helpers.mixed_mlm`` drives torch models, so these checks run **inside** the
glm2 standalone env via :func:`proto_tools.utils.run_in_env`, invoked by
``test_mixed_runtime_checks_run_in_env`` in ``test_mixed_models.py``. A tiny
context-dependent toy MLM stands in for the real checkpoints so the algorithms are tested
without model downloads.

``pytest`` is not installed in the standalone env, so these are plain-``assert`` checks.
``sys.argv[1]`` is the path to ``proto_tools/utils/compressed_array.py``, loaded by file so
the parent package is not imported. ``main()`` runs every check, prints a per-check status
line, and exits non-zero if any fail (which ``run_in_env`` surfaces as ``RuntimeError``).
The leading ``_`` keeps pytest from collecting this file in the base process.
"""

from __future__ import annotations

import contextlib
import importlib.util
import re
import sys
import traceback
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import torch
from standalone_helpers.mixed_mlm import MixedMLMAdapter, MixedMLMRuntime
from standalone_helpers.mixed_sequence import (
    MIXED_VOCAB,
    UPSTREAM_STRAND_TOKENS,
    one_hot_mixed_logits,
    tokenize_mixed_sequence,
)

_LOCUS = "+MA+ac-X"
_TOKENS = ["+", "M", "A", "+", "a", "c", "-", "X"]


def _load_decompress_result() -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location("_compressed_array", sys.argv[1])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decompress_result


_decompress_result = _load_decompress_result()


def _close(actual: float, expected: float, *, rel: float = 1e-6, abs_: float = 1e-12) -> None:
    """Assert ``actual`` matches ``expected`` with ``pytest.approx`` semantics."""
    assert abs(actual - expected) <= max(rel * abs(expected), abs_), f"{actual} != {expected} (rel={rel}, abs={abs_})"


@contextlib.contextmanager
def _expect(exc: type[BaseException], match: str) -> Iterator[None]:
    """Assert that the block raises ``exc`` with a message matching the ``match`` regex."""
    try:
        yield
    except exc as error:
        assert re.search(match, str(error)), f"expected {match!r} in error, got {error!r}"
        return
    raise AssertionError(f"expected {exc.__name__} matching {match!r}")


def _make_runtime() -> MixedMLMRuntime:
    """Build a runtime around a seeded toy MLM with the upstream glm2 vocabulary layout."""
    vocabulary = (
        list(MIXED_VOCAB) + list("XBUZO") + ["<+>", "<->", "<mask>", "<cls>", "<pad>", "<eos>", "<unk>", "<sep>"]
    )
    vocab = {token: index for index, token in enumerate(vocabulary)}

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(len(vocab), 8)
            self.projection = torch.nn.Linear(8, len(vocab))
            self.config = SimpleNamespace(depth=2)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
            embedded = self.embedding(input_ids)
            hidden = torch.tanh(embedded + embedded.mean(dim=1, keepdim=True))
            return SimpleNamespace(logits=self.projection(hidden), hidden_states=(embedded, hidden))

    class TinyAdapter(MixedMLMAdapter):
        toolkit = "glm2"
        checkpoints: ClassVar[dict[str, str]] = {"tattabio/gLM2_150M": "tiny"}
        default_checkpoint = "tattabio/gLM2_150M"

        @property
        def embedding_layer(self):
            return self.model.embedding

        def _load_checkpoint(self, checkpoint):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                model = TinyModel()
            tokenizer = SimpleNamespace(get_vocab=lambda: vocab, mask_token_id=vocab["<mask>"])
            return model, tokenizer

    return MixedMLMRuntime(TinyAdapter())


def _dispatch(runtime: MixedMLMRuntime, operation: str, **payload: Any) -> Any:
    return _decompress_result(runtime.dispatch({"operation": operation, "device": "cpu", **payload}), to_list=True)


def _assert_gradient_respects_alphabets(gradient: np.ndarray) -> None:
    """Markers carry no gradient; protein rows touch only protein columns, DNA rows only DNA."""
    assert np.isfinite(gradient).all()
    assert np.any(gradient != 0.0)
    np.testing.assert_array_equal(gradient[[0, 3, 6, 7]], 0.0)
    np.testing.assert_array_equal(gradient[[1, 2], 20:], 0.0)
    np.testing.assert_array_equal(gradient[[4, 5], :20], 0.0)


def check_score_matches_full_vocab_reference_and_gradient() -> None:
    runtime = _make_runtime()
    # An embedding call first catches invalid reuse of inference tensors in backward.
    _dispatch(runtime, "embeddings", sequences=[_LOCUS], return_logits=True)
    result = _dispatch(runtime, "score", sequences=[_LOCUS], batch_size=2, return_logits=True)["scores"][0]
    vocab = runtime.adapter.tokenizer.get_vocab()
    ids = torch.tensor([[vocab[UPSTREAM_STRAND_TOKENS.get(token, token)] for token in _TOKENS]])
    log_probs = []
    for position in [1, 2, 4, 5]:
        masked = ids.clone()
        masked[0, position] = vocab["<mask>"]
        with torch.no_grad():
            logits = runtime.adapter.model(input_ids=masked).logits[0, position]
            log_probs.append(torch.log_softmax(logits, dim=-1)[ids[0, position]].item())
    _close(result["avg_log_likelihood"], float(np.mean(log_probs)))
    gradient_result = _dispatch(
        runtime,
        "gradient",
        sequence=_LOCUS,
        logits=one_hot_mixed_logits(_LOCUS, sharpness=1.0),
        temperature=None,
        batch_size=2,
    )
    _close(gradient_result["loss"], -float(np.mean(log_probs)))
    _assert_gradient_respects_alphabets(np.asarray(gradient_result["gradient"]))
    assert all(parameter.grad is None for parameter in runtime.adapter.model.parameters())
    assert not runtime.adapter.embedding_layer._forward_hooks


def check_relaxed_gradient_matches_finite_difference() -> None:
    for batch_size in [1, 2, 8]:
        runtime = _make_runtime()
        logits = np.asarray(one_hot_mixed_logits("<+>M<+>a", sharpness=2.0))
        payload = {"sequence": "<+>M<+>a", "logits": logits.tolist(), "temperature": 0.9, "batch_size": batch_size}
        result = _dispatch(runtime, "gradient", **payload)
        losses = []
        epsilon = 0.01
        for delta in [epsilon, -epsilon]:
            perturbed = logits.copy()
            perturbed[3, 21] += delta
            losses.append(
                _dispatch(runtime, "gradient", **{**payload, "logits": perturbed.tolist(), "compute_gradient": False})[
                    "loss"
                ]
            )
        numerical = (losses[0] - losses[1]) / (2 * epsilon)
        _close(result["gradient"][3][21], numerical, rel=0.01, abs_=3e-5)


def check_ste_uses_hard_context_and_soft_backward() -> None:
    temperature = 0.9
    for batch_size in [1, 2, 8]:
        runtime = _make_runtime()
        state = torch.tensor(one_hot_mixed_logits(_LOCUS, sharpness=0.7))
        result = _dispatch(
            runtime,
            "gradient",
            sequence=_LOCUS,
            logits=state.tolist(),
            temperature=temperature,
            batch_size=batch_size,
            use_ste=True,
        )
        score = _dispatch(runtime, "score", sequences=[_LOCUS], batch_size=2)["scores"][0]
        _close(result["loss"], -score["avg_log_likelihood"])

        # Differentiate the toy model's hard context directly, without embedding hooks.
        model = runtime.adapter.model
        vocab = runtime.adapter.tokenizer.get_vocab()
        ids = torch.tensor([vocab[UPSTREAM_STRAND_TOKENS.get(token, token)] for token in _TOKENS])
        hard_embeddings = model.embedding(ids).detach().requires_grad_(True)
        mask_embedding = model.embedding.weight[vocab["<mask>"]].detach()
        positions = [1, 2, 4, 5]
        losses = []
        for position in positions:
            context = torch.cat(
                [hard_embeddings[:position], mask_embedding.unsqueeze(0), hard_embeddings[position + 1 :]]
            )
            prediction = model.projection(torch.tanh(mask_embedding + context.mean(dim=0)))
            losses.append(torch.logsumexp(prediction, dim=0) - prediction[ids[position]])
        reference_loss = torch.stack(losses).mean()
        _close(result["loss"], reference_loss.item())
        (embedding_gradient,) = torch.autograd.grad(reference_loss, hard_embeddings)

        # STE preserves the softmax Jacobian, evaluated against the hard-context derivative.
        expected = torch.zeros_like(state)
        for position in positions:
            columns = slice(0, 20) if _TOKENS[position].isupper() else slice(20, 24)
            probabilities = torch.softmax(state[position, columns] / temperature, dim=0)
            canonical_ids = [vocab[token] for token in MIXED_VOCAB[columns]]
            probability_gradient = model.embedding.weight[canonical_ids] @ embedding_gradient[position]
            expected[position, columns] = (
                probabilities * (probability_gradient - (probabilities * probability_gradient).sum()) / temperature
            )
        gradient = np.asarray(result["gradient"])
        np.testing.assert_allclose(gradient, expected.detach().numpy(), rtol=1e-5, atol=1e-7)
        _assert_gradient_respects_alphabets(gradient)
        assert all(parameter.grad is None for parameter in model.parameters())
        assert not runtime.adapter.embedding_layer._forward_hooks


def check_gradient_removes_embedding_hook_after_failure() -> None:
    runtime = _make_runtime()
    runtime.adapter.load(runtime.adapter.default_checkpoint, "cpu")

    def fail_forward(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic forward failure")

    runtime.adapter.forward = fail_forward
    with _expect(RuntimeError, r"synthetic forward failure"):
        _dispatch(runtime, "gradient", sequence="+A+a", logits=one_hot_mixed_logits("+A+a"))
    assert not runtime.adapter.embedding_layer._forward_hooks


def check_batches_exact_lengths_and_restores_order() -> None:
    runtime = _make_runtime()
    observed = []
    forward = runtime.adapter.forward

    def track(ids: Any, **kwargs: Any) -> Any:
        observed.append(tuple(ids.shape))
        return forward(ids, **kwargs)

    runtime.adapter.forward = track
    sequences = ["<+>M<+>a", "+a", "<->A<+>c", "<->AG"]
    batched = _dispatch(runtime, "embeddings", sequences=sequences, batch_size=4, return_logits=True)["results"]
    assert observed == [(2, 4), (1, 2), (1, 3)], observed
    singles = _dispatch(runtime, "embeddings", sequences=sequences, batch_size=1, return_logits=True)["results"]
    assert all("tokens" not in row for row in batched)
    # Logit rows follow the canonical one-character-per-token sequence.
    assert [len(row["logits"]) for row in batched] == [len(tokenize_mixed_sequence(sequence)) for sequence in sequences]
    for batch, single in zip(batched, singles, strict=True):
        np.testing.assert_allclose(batch["mean_embedding"], single["mean_embedding"], atol=1e-6)
        np.testing.assert_allclose(batch["logits"], single["logits"], atol=1e-6)


def check_sample_preserves_alphabets_and_fixed_context() -> None:
    sequence = "<+>____<+>____<->X"
    metadata = {**dict.fromkeys(range(2, 6), "protein"), **dict.fromkeys(range(7, 11), "dna")}
    for method in ["single_pass", "iterative_refinement"]:
        runtime = _make_runtime()
        payload = {
            "sequences": [sequence] * 3,
            "mask_modalities": [metadata] * 3,
            "seed": 7,
            "sampling_method": method,
            "num_steps": 3,
            "return_logits": True,
        }
        output = _dispatch(runtime, "sample", **payload)["results"]
        repeated = _dispatch(runtime, "sample", **payload)["results"]
        assert [row["sequence"] for row in output] == [row["sequence"] for row in repeated], method
        assert len({row["sequence"] for row in output}) > 1, method
        for row in output:
            assert "tokens" not in row
            assert tokenize_mixed_sequence(row["sequence"]) == list(row["sequence"])
            assert [row["sequence"][i] for i in [0, 5, 10, 11]] == ["+", "+", "-", "X"]
            assert all(token in MIXED_VOCAB[:20] for token in row["sequence"][1:5])
            assert all(token in MIXED_VOCAB[20:] for token in row["sequence"][6:10])
            assert np.asarray(row["logits"]).shape == (12, 24)


def check_rejects_invalid_requests_before_loading() -> None:
    cases = [
        ({"operation": "unknown"}, "unknown operation"),
        ({"operation": "interactions", "sequences": ["+A+a"]}, "Only Minerva"),
        ({"operation": "score", "sequences": ["+A+a"], "batch_size": 0}, "positive integer"),
        ({"operation": "score", "sequences": ["<+>X"]}, "at least one canonical"),
        ({"operation": "embeddings", "sequences": ["+" + "A" * 4096]}, "at most 4096"),
        ({"operation": "sample", "sequences": ["<+>_"], "mask_modalities": [{}]}, "exactly the masked"),
    ]

    def unexpected_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Invalid requests must be rejected before model loading")

    for payload, error in cases:
        runtime = _make_runtime()
        runtime.adapter.load = unexpected_load
        with _expect(ValueError, error):
            runtime.dispatch(payload)


def check_sampling_is_independent_of_mask_mapping_order() -> None:
    positions = [(position, "protein" if position < 6 else "dna") for position in [2, 3, 4, 5, 7, 8, 9, 10]]
    for method in ["single_pass", "iterative_refinement"]:
        runtime = _make_runtime()
        payload = {"sequences": ["<+>____<+>____<->X"] * 3, "seed": 31, "sampling_method": method, "num_steps": 3}
        forward = _dispatch(runtime, "sample", **payload, mask_modalities=[dict(positions)] * 3)
        reverse = _dispatch(runtime, "sample", **payload, mask_modalities=[dict(reversed(positions))] * 3)
        assert [row["sequence"] for row in forward["results"]] == [row["sequence"] for row in reverse["results"]], (
            method
        )


CHECKS = [
    check_score_matches_full_vocab_reference_and_gradient,
    check_relaxed_gradient_matches_finite_difference,
    check_ste_uses_hard_context_and_soft_backward,
    check_gradient_removes_embedding_hook_after_failure,
    check_batches_exact_lengths_and_restores_order,
    check_sample_preserves_alphabets_and_fixed_context,
    check_rejects_invalid_requests_before_loading,
    check_sampling_is_independent_of_mask_mapping_order,
]


def _run_check(check: Any) -> str | None:
    """Run one check; print PASS/FAIL and return the check's name on failure, else None."""
    try:
        check()
    except Exception as error:
        print(f"FAIL {check.__name__}: {type(error).__name__}: {error}")
        traceback.print_exc()
        return str(check.__name__)
    print(f"PASS {check.__name__}")
    return None


def main() -> int:
    """Run every check, report per-check status, and return a non-zero code on any failure."""
    failures = [name for name in (_run_check(check) for check in CHECKS) if name is not None]
    if failures:
        print(f"FAILED {len(failures)}/{len(CHECKS)}: {failures}")
        return 1
    print(f"ALL CHECKS PASSED ({len(CHECKS)}/{len(CHECKS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

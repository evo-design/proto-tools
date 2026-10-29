"""Regression tests for the Evo 2 Flash standalone adapter."""

import importlib.util
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_INFERENCE = Path(__file__).resolve().parents[2] / "proto_tools/tools/causal_models/evo2/standalone/inference.py"


@pytest.fixture
def flash(monkeypatch):
    """Load the actual adapter with only its external GPU dependencies stubbed."""
    helpers = SimpleNamespace(
        get_logger=logging.getLogger,
        log_likelihood_metrics=lambda mean, count: {
            "avg_log_likelihood": mean,
            "log_likelihood": mean * count,
            "perplexity": math.exp(-mean),
        },
        serialize_output=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "standalone_helpers", helpers)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "vllm.inputs", SimpleNamespace(TokensPrompt=lambda **kwargs: kwargs))
    spec = importlib.util.spec_from_file_location("evo2_flash_adapter_test", _INFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = SimpleNamespace(
        tokenize=lambda text: list(text.encode()),
        detokenize=lambda ids: "".join(chr(max(32, i)) for i in ids),
    )
    model = module.Evo2Model()
    model._loaded = True
    model.device = "cuda"
    model.tokenizer = tokenizer
    engine = SimpleNamespace(
        tokenizer=tokenizer,
        max_model_len=1000,
        sampling_kwargs=lambda temperature, top_k, top_p: {"temperature": temperature, "top_k": top_k, "top_p": top_p},
        llm=SimpleNamespace(generate=Mock(), sleep=Mock(), wake_up=Mock()),
        score_sequences=Mock(),
    )
    model.model = engine
    module._model = model
    return module, model, engine


def _outputs(*tokens):
    """Make ordered vLLM response objects with explicit sampled token ids."""
    return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=t)]) for t in tokens]


def test_flash_batches_ragged_prompts_and_assigns_independent_request_seeds(flash):
    """Ragged prompts stay together, with stable request ordering and independent seeds."""
    _, model, engine = flash
    engine.llm.generate.return_value = _outputs([65, 78], [84, 71])
    result = model.sample(["AC", "ACGT"], max_new_tokens=2, stop_at_eos=False, batch_size=1, seed=17)
    requests, params = engine.llm.generate.call_args.args
    assert [r["prompt_token_ids"] for r in requests] == [list(b"AC"), list(b"ACGT")]
    assert [p.seed for p in params] == [17, 18]
    assert all(p.ignore_eos and p.min_tokens == 2 for p in params)
    assert result == {"sequences": ["AN", "TG"], "logits": None, "kv_caches": None}
    engine.llm.generate.assert_called_once()


def test_flash_missing_seed_draws_fresh_request_streams(flash, monkeypatch):
    """Unseeded dispatches must not restart the engine's fixed default RNG stream."""
    module, model, engine = flash
    monkeypatch.setattr(module.secrets, "randbits", Mock(side_effect=[101, 202]))
    engine.llm.generate.return_value = _outputs([65])
    for _ in range(2):
        model.sample(["AC"], max_new_tokens=1)
    assert [c.args[1][0].seed for c in engine.llm.generate.call_args_list] == [101, 202]


def test_flash_sampling_eos_and_special_token_contract(flash):
    """Ordinary sampling still honors EOS, while special-token filtering is explicit."""
    _, model, engine = flash
    engine.llm.generate.return_value = _outputs([65, 0])
    result = model.sample(["AC"], max_new_tokens=10, stop_at_eos=True, skip_special_tokens=True)
    params = engine.llm.generate.call_args.args[1][0]
    assert not params.ignore_eos and params.stop_token_ids == [0] and params.min_tokens == 0
    assert result["sequences"] == ["A"]


@pytest.mark.parametrize("limit", ["max_seqlen", "engine"])
def test_flash_rejects_context_overflow_before_generation(flash, limit):
    """Oversized requests fail clearly before entering the GPU scheduler."""
    _, model, engine = flash
    kwargs = {"max_seqlen": 5} if limit == "max_seqlen" else {}
    if limit == "engine":
        engine.max_model_len = 5
    with pytest.raises(ValueError, match="needs 6 positions"):
        model.sample(["ACGT"], max_new_tokens=2, **kwargs)
    engine.llm.generate.assert_not_called()


def test_flash_detects_short_fixed_length_generation(flash):
    """A truncated response cannot silently satisfy a fixed-length ICL request."""
    _, model, engine = flash
    engine.llm.generate.return_value = _outputs([65])
    with pytest.raises(RuntimeError, match="requested fixed generation length"):
        model.sample(["AC"], max_new_tokens=2, stop_at_eos=False)


def test_flash_score_uses_true_token_counts_without_materializing_logits(flash):
    """Ragged scoring excludes padding and preserves the BOS convention."""
    _, model, engine = flash
    engine.score_sequences.return_value = [-0.5, -1.0]
    result = model.score(["AC", "ACGT"], prepend_bos=False)
    assert [m["log_likelihood"] for m in result["metrics"]] == [-0.5, -3.0]
    assert result["logits"] is None
    assert result["vocab"][65] == "A"
    result = model.score(["AC", "ACGT"], prepend_bos=True)
    assert [m["log_likelihood"] for m in result["metrics"]] == [-1.0, -4.0]
    engine.score_sequences.assert_called_with(["AC", "ACGT"], prepend_bos=True)


def test_flash_continuation_handles_fork_and_release_without_losing_token_identity(flash):
    """Beam-search forks recover exact tokens even when detokenization replaced a control byte."""
    module, _, engine = flash
    engine.llm.generate.return_value = _outputs([65, 0])
    first = module.dispatch({"operation": "sample", "prompts": ["AC"], "max_new_tokens": 2, "return_kv_cache": True})
    handle = first["kv_caches"][0]
    assert handle["type"] == "evo2_kv_cache"
    engine.llm.generate.return_value = _outputs([71], [84])
    second = module.dispatch(
        {
            "operation": "sample",
            "prompts": ["ACA ", "ACA G"],
            "max_new_tokens": 1,
            "old_kv_cache": handle,
            "return_kv_cache": True,
        }
    )
    requests = engine.llm.generate.call_args.args[0]
    assert [r["prompt_token_ids"] for r in requests] == [[65, 67, 65, 0], [65, 67, 65, 0, 71]]
    assert len({x["cache_id"] for x in second["kv_caches"]}) == 2
    module.dispatch({"operation": "release_kv_caches", "kv_caches": [handle]})
    with pytest.raises(ValueError, match="no longer available"):
        module.dispatch({"operation": "sample", "prompts": ["ACA "], "old_kv_cache": handle})
    assert all(module._resolve_cache_handle(h) for h in second["kv_caches"])


def test_flash_continuation_rejects_mismatched_prefix(flash):
    """A handle must not silently change the biological context of a different prompt."""
    _, model, _ = flash
    with pytest.raises(ValueError, match="full cached sequence prefix"):
        model.sample(["TT"], old_kv_cache={"tokens": [65, 67], "text": "AC"})


def test_flash_cache_handles_require_persistent_worker(flash, monkeypatch):
    """A one-shot process cannot return cache handles that immediately become stale."""
    module, _, _ = flash
    monkeypatch.setattr(module, "__name__", "__main__")
    with pytest.raises(RuntimeError, match="persistent ToolInstance worker"):
        module.dispatch({"operation": "sample", "return_kv_cache": True})


@pytest.mark.parametrize("device", ["cpu", "cuda:3"])
def test_flash_device_changes_require_a_new_worker(flash, device):
    """The engine cannot migrate CUDA state in process; Proto's pinned workers restart instead."""
    module, model, _ = flash
    module.to_device("cuda")
    assert model.device == "cuda"
    with pytest.raises(ValueError, match="new worker to change devices"):
        module.to_device(device)

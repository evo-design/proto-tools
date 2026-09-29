"""Tests for mixed protein/DNA sequence contracts and transport."""

import json
import math
from typing import ClassVar

import numpy as np
import pytest
from pydantic import ValidationError
from standalone_helpers.compression import compress_array
from standalone_helpers.mixed_sequence import tokenize_mixed_sequence

import proto_tools
from proto_tools.tools.masked_models.execution import dispatch_masked_model
from proto_tools.tools.masked_models.glm2 import GLM2EmbeddingsConfig
from proto_tools.tools.masked_models.minerva import MinervaEmbeddingsConfig, MinervaInteractionsConfig
from proto_tools.tools.masked_models.mixed_data_models import (
    MIXED_VOCAB,
    MixedEmbeddingsOutput,
    MixedGradientOutput,
    MixedSampleOutput,
    MixedScoringOutput,
    MixedSequenceGradientInput,
    MixedSequenceInput,
    MixedSequenceSampleInput,
    one_hot_mixed_logits,
)
from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.transforms.masking import MaskingStrategy
from proto_tools.utils import ToolInstance
from proto_tools.utils.interaction_models import SequenceInteractionMap, SequenceInteractionsOutput

_LOCUS = "<+>MA<->acX"
_TOKENS = ["<+>", "M", "A", "<->", "a", "c", "X"]


# ── Atomic tokens and modality ──────────────────────────────────────────────


def test_mixed_sequence_token_axis_preserves_case_and_markers():
    inputs = MixedSequenceInput(sequences=_LOCUS)
    assert inputs.sequences == [_LOCUS]
    assert tokenize_mixed_sequence(_LOCUS) == _TOKENS
    assert len(inputs) == 7
    assert tokenize_mixed_sequence("ACGTacgtXBUZO") == list("ACGTacgtXBUZO")


@pytest.mark.parametrize(
    "sequence", ["", "<+><->", "acng", "acu", "AC*", "A C", " A", "A\n", "<mask>", "<sep>", "<+A", "A_"]
)
def test_mixed_parser_rejects_invalid_or_empty_sequences(sequence):
    with pytest.raises(ValueError, match=r"Invalid mixed-sequence character|at least one biological"):
        tokenize_mixed_sequence(sequence)


def test_mixed_sample_mask_metadata_uses_token_coordinates():
    inputs = MixedSequenceSampleInput(sequences="<+>A_<->a_", mask_modalities=[{3: "protein", 6: "dna"}])
    assert len(inputs) == 6
    restored = MixedSequenceSampleInput.model_validate_json(inputs.model_dump_json())
    assert restored.mask_modalities == [{3: "protein", 6: "dna"}]
    assert restored.sequences == inputs.sequences


@pytest.mark.parametrize("modalities", [None, [{}], [{2: "protein"}], [{3: "protein", 4: "dna"}], [{3: "protein"}, {}]])
def test_mixed_sample_rejects_missing_extra_or_misaligned_mask_metadata(modalities):
    with pytest.raises(ValidationError, match=r"exactly the masked token positions|one mapping per sequence"):
        MixedSequenceSampleInput(sequences="<+>A_", mask_modalities=modalities)


def test_mixed_sequence_one_hot_alignment_and_probability_mode():
    rows = one_hot_mixed_logits(_LOCUS, sharpness=1.0)
    assert len(rows) == len(_TOKENS)
    assert list("ACDEFGHIKLMNPQRSTVWYacgt") == MIXED_VOCAB
    for token, row in zip(_TOKENS, rows, strict=True):
        assert len(row) == 24
        if token in MIXED_VOCAB:
            assert row[MIXED_VOCAB.index(token)] == 1.0
            assert sum(row) == 1.0
        else:
            assert not any(row)
    assert MixedSequenceGradientInput(sequence=_LOCUS, logits=rows, temperature=None).logits == rows


@pytest.mark.parametrize("sharpness", [math.nan, math.inf, -math.inf])
def test_mixed_one_hot_rejects_nonfinite_sharpness(sharpness):
    with pytest.raises(ValueError, match=r"sharpness must be finite"):
        one_hot_mixed_logits("Aa", sharpness=sharpness)


def test_mixed_gradient_rejects_alignment_fixed_context_and_nonfinite_values():
    rows = one_hot_mixed_logits(_LOCUS)
    with pytest.raises(ValidationError, match=r"one row per model token"):
        MixedSequenceGradientInput(sequence=_LOCUS, logits=rows[:-1])
    rows[0][0] = 1.0
    with pytest.raises(ValidationError, match=r"Fixed context row 1"):
        MixedSequenceGradientInput(sequence=_LOCUS, logits=rows)
    for row in [[0.0] * 23, [math.nan] * 24, [math.inf] * 24]:
        with pytest.raises(ValidationError, match=r"24 finite numbers"):
            MixedSequenceGradientInput(sequence="A", logits=[row])
    with pytest.raises(ValidationError, match=r"at least one canonical"):
        MixedSequenceGradientInput(sequence="<+>X", logits=[[0.0] * 24] * 2)


@pytest.mark.parametrize("row", [[0.0] * 24, [-1.0] + [0.0] * 23, [0.0] * 20 + [1.0, 0.0, 0.0, 0.0]])
def test_mixed_gradient_probability_mode_rejects_wrong_domains(row):
    with pytest.raises(ValidationError, match=r"probability distribution within its modality"):
        MixedSequenceGradientInput(sequence="A", logits=[row], temperature=None)


# ── Checkpoint-specific limits ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("config_class", "checkpoint", "limit", "depth"),
    [
        (GLM2EmbeddingsConfig, "tattabio/gLM2_150M", 4096, 30),
        (GLM2EmbeddingsConfig, "tattabio/gLM2_650M", 4096, 33),
        (MinervaEmbeddingsConfig, "gbrixi/minerva-mlm", 4096, 33),
        (MinervaEmbeddingsConfig, "gbrixi/minerva-mlm-8k", 8192, 33),
    ],
)
def test_mixed_context_caps_count_atomic_markers(config_class, checkpoint, limit, depth):
    config = config_class(model_checkpoint=checkpoint, repr_layer=depth)
    valid = MixedSequenceInput(sequences="<+>" + "A" * (limit - 1))
    assert config.preprocess(valid) is valid
    with pytest.raises(ValueError, match=f"{limit + 1} tokens; limit is {limit}"):
        config.preprocess(MixedSequenceInput(sequences="<+>" + "A" * limit))
    with pytest.raises(ValidationError, match=r"transformer depth"):
        config_class(model_checkpoint=checkpoint, repr_layer=depth + 1)


@pytest.mark.parametrize("toolkit", ["glm2", "minerva"])
@pytest.mark.parametrize("method", ["entropy", "max-logit"])
def test_automatic_masking_uses_own_toolkit_and_preserves_modalities(monkeypatch, toolkit, method):
    embedding = ToolRegistry.get(f"{toolkit}-embedding")
    sampling = ToolRegistry.get(f"{toolkit}-sample")
    config = sampling.config_model(
        device="cpu", seed=7, masking_strategy=MaskingStrategy(method=method, num_mutations=2)
    )
    calls = []

    def embed(inputs, embedding_config):
        calls.append((inputs, embedding_config))
        return MixedEmbeddingsOutput(
            results=[
                {"tokens": _TOKENS, "mean_embedding": [0.1], "attention_mask": [1] * 7, "logits": [[0.0] * 24] * 7}
            ]
        )

    monkeypatch.setattr(embedding, "function", embed)
    result = config.preprocess(MixedSequenceSampleInput(sequences=_LOCUS))
    assert len(calls) == 1
    inputs, embedding_config = calls[0]
    assert inputs.sequences == [_LOCUS]
    assert embedding_config.model_checkpoint == config.model_checkpoint
    assert embedding_config.tool_key == f"{toolkit}-embedding"
    assert embedding_config.return_logits
    assert embedding_config.device == "cpu"
    tokens = tokenize_mixed_sequence(result.sequences[0], allow_masks=True)
    assert [tokens[i] for i in (0, 3, 6)] == ["<+>", "<->", "X"]
    assert tokens.count("_") == 2
    expected = {i + 1: ("dna" if _TOKENS[i] in "acgt" else "protein") for i, token in enumerate(tokens) if token == "_"}
    assert result.mask_modalities == [expected]


# ── Labeled results, public exports, and compressed transport ────────────────


@pytest.mark.parametrize(
    ("tokens", "values", "error"),
    [
        (["A", "a"], [[0.1]], "square"),
        (["A"], [[0.1, 0.2]], "square"),
        (["A"], [[-0.01]], "finite values"),
        (["A"], [[1.01]], "finite values"),
        (["A"], [[math.nan]], "finite values"),
        (["A"], [[math.inf]], "finite values"),
    ],
)
def test_interaction_map_rejects_invalid_axes_or_probabilities(tokens, values, error):
    with pytest.raises(ValidationError, match=error):
        SequenceInteractionMap(tokens=tokens, values=values)


def test_mixed_output_json_roundtrips_preserve_token_metadata():
    logits = [[float(i)] * 24 for i in range(len(_TOKENS))]
    outputs = [
        MixedEmbeddingsOutput(
            results=[{"tokens": _TOKENS, "mean_embedding": [0.1, 0.2], "attention_mask": [1] * 7, "logits": logits}]
        ),
        MixedSampleOutput(results=[{"sequence": _LOCUS, "tokens": _TOKENS, "logits": logits}]),
        MixedScoringOutput(
            scores=[
                {
                    "tokens": _TOKENS,
                    "scored_positions": [2, 3, 5, 6],
                    "log_likelihood": -4.0,
                    "avg_log_likelihood": -1.0,
                    "perplexity": math.e,
                    "logits": logits,
                    "vocab": MIXED_VOCAB,
                }
            ]
        ),
        MixedGradientOutput(tokens=_TOKENS, gradient=logits, loss=1.0, vocab=MIXED_VOCAB),
        SequenceInteractionsOutput(results=[{"maps": {"protein": {"tokens": _TOKENS, "values": np.eye(7).tolist()}}}]),
    ]
    for output in outputs:
        restored = type(output).model_validate_json(output.model_dump_json())
        assert restored.model_dump() == output.model_dump()


def test_mixed_interactions_export_npz_and_json_preserve_axes(tmp_path):
    output = SequenceInteractionsOutput(
        results=[{"maps": {"base_pairing": {"tokens": ["<+>", "a"], "values": [[0.1, 0.2], [0.2, 0.8]]}}}]
    )
    output._export_output(tmp_path / "contacts.v1", "npz")
    with np.load(tmp_path / "contacts.v1.npz", allow_pickle=False) as archive:
        assert archive["0_base_pairing_tokens"].tolist() == ["<+>", "a"]
        np.testing.assert_allclose(archive["0_base_pairing"], [[0.1, 0.2], [0.2, 0.8]])
        assert archive["0_base_pairing"].dtype == np.float32
    output._export_output(tmp_path / "contacts.v1", "json")
    restored = SequenceInteractionsOutput(results=json.loads((tmp_path / "contacts.v1.json").read_text()))
    assert restored.results == output.results


def test_mixed_dispatch_decompresses_nested_matrices(monkeypatch):
    captured = {}
    matrix = np.array([[0.1, 0.2], [0.2, 0.8]], dtype=np.float32)

    def dispatch(toolkit, payload, *, instance=None, config=None):
        captured.update(toolkit=toolkit, payload=payload)
        return {"results": [{"maps": {"protein": {"tokens": ["<+>", "A"], "values": compress_array(matrix)}}}]}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(dispatch))
    config = MinervaInteractionsConfig(device="cpu", heads=["protein"])
    raw = dispatch_masked_model("minerva", "interactions", MixedSequenceInput(sequences="<+>A"), config, None)
    output = SequenceInteractionsOutput(**raw)
    assert captured["toolkit"] == "minerva"
    assert captured["payload"]["operation"] == "interactions"
    assert captured["payload"]["sequences"] == ["<+>A"]
    assert output.metadata["model_checkpoint"] == "gbrixi/minerva-mlm"
    np.testing.assert_array_equal(output.results[0].maps["protein"].values, matrix)


def test_mixed_public_exports_and_registry_keys():
    for name in [
        "MixedSequenceInput",
        "MixedSequenceGradientInput",
        "MixedSequenceSampleInput",
        "SequenceInteractionMap",
        "SequenceInteractions",
        "MixedSequenceEmbedding",
        "MixedSequenceSample",
        "MIXED_VOCAB",
        "one_hot_mixed_logits",
    ]:
        assert getattr(proto_tools, name) is not None
    keys = {spec.key for spec in ToolRegistry.list_all()}
    for toolkit, prefix in [("glm2", "GLM2"), ("minerva", "Minerva")]:
        for operation in ["embedding", "score", "gradient", "sample"]:
            assert f"{toolkit}-{operation}" in keys
        assert getattr(proto_tools, f"{prefix}EmbeddingsConfig") is not None
    assert "minerva-interactions" in keys


def test_mixed_gradient_export_preserves_tokens_and_values(tmp_path):
    output = MixedGradientOutput(tokens=_TOKENS, gradient=[[0.0] * 24] * len(_TOKENS), loss=1.0, vocab=MIXED_VOCAB)
    output._export_output(tmp_path / "gradient.v1", "json")
    restored = MixedGradientOutput.model_validate_json((tmp_path / "gradient.v1.json").read_text())
    for field in ("tokens", "gradient", "loss", "metrics", "vocab"):
        assert getattr(restored, field) == getattr(output, field)


# ── Numerical runtime checks with a tiny differentiable model ────────────────


@pytest.fixture
def tiny_mixed_runtime():
    """Use a context-dependent toy MLM to test algorithms without model downloads."""
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from standalone_helpers.mixed_mlm import MixedMLMAdapter, MixedMLMRuntime

    vocabulary = (
        list(MIXED_VOCAB) + list("XBUZO") + ["<+>", "<->", "<mask>", "<cls>", "<pad>", "<eos>", "<unk>", "<sep>"]
    )
    vocab = {token: index for index, token in enumerate(vocabulary)}

    class TinyModel(torch.nn.Module):
        def __init__(self):
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


def _tiny_dispatch(runtime, operation, **payload):
    from proto_tools.utils.compressed_array import decompress_result

    return decompress_result(runtime.dispatch({"operation": operation, "device": "cpu", **payload}), to_list=True)


def test_mixed_runtime_score_matches_full_vocab_reference_and_gradient(tiny_mixed_runtime):
    import torch

    runtime = tiny_mixed_runtime
    # An embedding call first catches invalid reuse of inference tensors in backward.
    _tiny_dispatch(runtime, "embeddings", sequences=[_LOCUS], return_logits=True)
    result = _tiny_dispatch(runtime, "score", sequences=[_LOCUS], batch_size=2, return_logits=True)["scores"][0]
    vocab = runtime.adapter.tokenizer.get_vocab()
    ids = torch.tensor([[vocab[token] for token in _TOKENS]])
    log_probs = []
    for position in [1, 2, 4, 5]:
        masked = ids.clone()
        masked[0, position] = vocab["<mask>"]
        with torch.no_grad():
            logits = runtime.adapter.model(input_ids=masked).logits[0, position]
            log_probs.append(torch.log_softmax(logits, dim=-1)[ids[0, position]].item())
    assert result["avg_log_likelihood"] == pytest.approx(np.mean(log_probs), rel=1e-6)
    gradient_result = _tiny_dispatch(
        runtime,
        "gradient",
        sequence=_LOCUS,
        logits=one_hot_mixed_logits(_LOCUS, sharpness=1.0),
        temperature=None,
        batch_size=2,
    )
    assert gradient_result["loss"] == pytest.approx(-np.mean(log_probs), rel=1e-6)
    gradient = np.asarray(gradient_result["gradient"])
    assert np.isfinite(gradient).all()
    assert np.any(gradient != 0.0)
    np.testing.assert_array_equal(gradient[[0, 3, 6]], 0.0)
    np.testing.assert_array_equal(gradient[[1, 2], 20:], 0.0)
    np.testing.assert_array_equal(gradient[[4, 5], :20], 0.0)
    assert all(parameter.grad is None for parameter in runtime.adapter.model.parameters())
    assert not runtime.adapter.embedding_layer._forward_hooks


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_mixed_runtime_relaxed_gradient_matches_finite_difference(tiny_mixed_runtime, batch_size):
    runtime = tiny_mixed_runtime
    logits = np.asarray(one_hot_mixed_logits("<+>Ma", sharpness=2.0))
    payload = {"sequence": "<+>Ma", "logits": logits.tolist(), "temperature": 0.9, "batch_size": batch_size}
    result = _tiny_dispatch(runtime, "gradient", **payload)
    losses = []
    epsilon = 0.01
    for delta in [epsilon, -epsilon]:
        perturbed = logits.copy()
        perturbed[2, 21] += delta
        losses.append(
            _tiny_dispatch(runtime, "gradient", **{**payload, "logits": perturbed.tolist(), "compute_gradient": False})[
                "loss"
            ]
        )
    numerical = (losses[0] - losses[1]) / (2 * epsilon)
    assert result["gradient"][2][21] == pytest.approx(numerical, rel=0.01, abs=3e-5)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_mixed_runtime_ste_uses_hard_context_and_soft_backward(tiny_mixed_runtime, batch_size):
    import torch

    runtime = tiny_mixed_runtime
    temperature = 0.9
    state = torch.tensor(one_hot_mixed_logits(_LOCUS, sharpness=0.7))
    result = _tiny_dispatch(
        runtime,
        "gradient",
        sequence=_LOCUS,
        logits=state.tolist(),
        temperature=temperature,
        batch_size=batch_size,
        use_ste=True,
    )
    score = _tiny_dispatch(runtime, "score", sequences=[_LOCUS], batch_size=2)["scores"][0]
    assert result["loss"] == pytest.approx(-score["avg_log_likelihood"], rel=1e-6)

    # Differentiate the toy model's hard context directly, without embedding hooks.
    model = runtime.adapter.model
    vocab = runtime.adapter.tokenizer.get_vocab()
    ids = torch.tensor([vocab[token] for token in _TOKENS])
    hard_embeddings = model.embedding(ids).detach().requires_grad_(True)
    mask_embedding = model.embedding.weight[vocab["<mask>"]].detach()
    positions = [1, 2, 4, 5]
    losses = []
    for position in positions:
        context = torch.cat([hard_embeddings[:position], mask_embedding.unsqueeze(0), hard_embeddings[position + 1 :]])
        prediction = model.projection(torch.tanh(mask_embedding + context.mean(dim=0)))
        losses.append(torch.logsumexp(prediction, dim=0) - prediction[ids[position]])
    reference_loss = torch.stack(losses).mean()
    assert result["loss"] == pytest.approx(reference_loss.item(), rel=1e-6)
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
    np.testing.assert_allclose(gradient, expected.numpy(), rtol=1e-5, atol=1e-7)
    assert np.isfinite(gradient).all()
    assert np.any(gradient != 0.0)
    np.testing.assert_array_equal(gradient[[0, 3, 6]], 0.0)
    np.testing.assert_array_equal(gradient[[1, 2], 20:], 0.0)
    np.testing.assert_array_equal(gradient[[4, 5], :20], 0.0)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not runtime.adapter.embedding_layer._forward_hooks


def test_mixed_runtime_gradient_removes_embedding_hook_after_failure(tiny_mixed_runtime, monkeypatch):
    runtime = tiny_mixed_runtime
    runtime.adapter.load(runtime.adapter.default_checkpoint, "cpu")

    def fail_forward(*args, **kwargs):
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(runtime.adapter, "forward", fail_forward)
    with pytest.raises(RuntimeError, match=r"synthetic forward failure"):
        _tiny_dispatch(runtime, "gradient", sequence="Aa", logits=one_hot_mixed_logits("Aa"))
    assert not runtime.adapter.embedding_layer._forward_hooks


def test_mixed_runtime_batches_exact_lengths_and_restores_order(tiny_mixed_runtime, monkeypatch):
    runtime = tiny_mixed_runtime
    observed = []
    forward = runtime.adapter.forward

    def track(ids, **kwargs):
        observed.append(tuple(ids.shape))
        return forward(ids, **kwargs)

    monkeypatch.setattr(runtime.adapter, "forward", track)
    sequences = ["<+>Ma", "a", "<->Ac", "aG"]
    batched = _tiny_dispatch(runtime, "embeddings", sequences=sequences, batch_size=4, return_logits=True)["results"]
    assert observed == [(2, 3), (1, 1), (1, 2)]
    singles = _tiny_dispatch(runtime, "embeddings", sequences=sequences, batch_size=1, return_logits=True)["results"]
    assert [row["tokens"] for row in batched] == [tokenize_mixed_sequence(sequence) for sequence in sequences]
    for batch, single in zip(batched, singles, strict=True):
        np.testing.assert_allclose(batch["mean_embedding"], single["mean_embedding"], atol=1e-6)
        np.testing.assert_allclose(batch["logits"], single["logits"], atol=1e-6)


@pytest.mark.parametrize("method", ["single_pass", "iterative_refinement"])
def test_mixed_runtime_sample_preserves_alphabets_and_fixed_context(tiny_mixed_runtime, method):
    sequence = "<+>____<->____X"
    metadata = {**dict.fromkeys(range(2, 6), "protein"), **dict.fromkeys(range(7, 11), "dna")}
    payload = {
        "sequences": [sequence] * 3,
        "mask_modalities": [metadata] * 3,
        "seed": 7,
        "sampling_method": method,
        "num_steps": 3,
        "return_logits": True,
    }
    output = _tiny_dispatch(tiny_mixed_runtime, "sample", **payload)["results"]
    repeated = _tiny_dispatch(tiny_mixed_runtime, "sample", **payload)["results"]
    assert [row["sequence"] for row in output] == [row["sequence"] for row in repeated]
    assert len({row["sequence"] for row in output}) > 1
    for row in output:
        assert [row["tokens"][i] for i in [0, 5, 10]] == ["<+>", "<->", "X"]
        assert all(token in MIXED_VOCAB[:20] for token in row["tokens"][1:5])
        assert all(token in MIXED_VOCAB[20:] for token in row["tokens"][6:10])
        assert np.asarray(row["logits"]).shape == (11, 24)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"operation": "unknown"}, "unknown operation"),
        ({"operation": "interactions", "sequences": ["Aa"]}, "Only Minerva"),
        ({"operation": "score", "sequences": ["Aa"], "batch_size": 0}, "positive integer"),
        ({"operation": "score", "sequences": ["<+>X"]}, "at least one canonical"),
        ({"operation": "embeddings", "sequences": ["A" * 4097]}, "at most 4096"),
        ({"operation": "sample", "sequences": ["<+>_"], "mask_modalities": [{}]}, "exactly the masked"),
    ],
)
def test_mixed_runtime_rejects_invalid_requests_before_loading(tiny_mixed_runtime, monkeypatch, payload, error):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid requests must be rejected before model loading")

    monkeypatch.setattr(tiny_mixed_runtime.adapter, "load", unexpected_load)
    with pytest.raises(ValueError, match=error):
        tiny_mixed_runtime.dispatch(payload)


@pytest.mark.parametrize("method", ["single_pass", "iterative_refinement"])
def test_mixed_runtime_sampling_is_independent_of_mask_mapping_order(tiny_mixed_runtime, method):
    positions = [(position, "protein" if position < 6 else "dna") for position in [2, 3, 4, 5, 7, 8, 9, 10]]
    payload = {"sequences": ["<+>____<->____X"] * 3, "seed": 31, "sampling_method": method, "num_steps": 3}
    forward = _tiny_dispatch(tiny_mixed_runtime, "sample", **payload, mask_modalities=[dict(positions)] * 3)
    reverse = _tiny_dispatch(tiny_mixed_runtime, "sample", **payload, mask_modalities=[dict(reversed(positions))] * 3)
    assert [row["sequence"] for row in forward["results"]] == [row["sequence"] for row in reverse["results"]]

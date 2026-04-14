"""Tests for AbLang antibody language model tools."""

import math
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools.masked_models.ablang import (
    AbLangEmbeddingsConfig,
    AbLangEmbeddingsInput,
    AbLangGerminalGradientConfig,
    AbLangGerminalGradientInput,
    AbLangSampleConfig,
    AbLangSampleInput,
    AbLangScoringConfig,
    AbLangScoringInput,
    run_ablang_embeddings,
    run_ablang_germinal_gradient,
    run_ablang_sample,
    run_ablang_score,
)
from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_VALID_AAS = set(PROTEIN_AMINO_ACIDS)
_CANONICAL_VOCAB = list(PROTEIN_AMINO_ACIDS)

# AbLang's amino acid ordering, from https://github.com/TobiasHeOl/AbLang2/blob/main/ablang2/models/ablang2/vocab.py
GERMINAL_ABLANG_ORDER = list("ARNDCQEGHILKMFPSTWYV")
_ABLANG_VOCAB_SIZE = len(GERMINAL_ABLANG_ORDER) + 1  # +1 for "|" separator
_PIPE_TOKEN_ID = len(GERMINAL_ABLANG_ORDER)


def _import_ablang_inference():
    """Lazy import of the standalone inference module (requires standalone_helpers stub)."""
    sys.modules.setdefault("standalone_helpers", types.SimpleNamespace(serialize_output=lambda value: value))
    from proto_tools.tools.masked_models.ablang.standalone import inference as ablang_inference

    return ablang_inference


_persistent_tool = make_persistent_fixture("ablang")

# Full antibody sequences (heavy + light with constant regions)
HEAVY_FULL = "AVKLVQAGGGVVQPGRSLRLSCIASGFTFSNYGMHWVRQAPGKGLEWVAVIWYNGSRTYYGDSVKGRFTISRDNSKRTLYMQMNSLRTEDTAVYYCARDPDILTAFSFDYWGQGVLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC"
LIGHT_FULL = "SYELTQPPSVSVSPGQTARITCSANALPNQYAYWYQQKPGRAPVMVIYKDTQRPSGIPQRFSSSTSGTTVTLTISGVQAEDEADYYCQAWDNSASIFGGGTKLTVLGQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPIKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS"
PAIRED_SEQ = f"{HEAVY_FULL}|{LIGHT_FULL}"

# Variable domain sequences for ablang1 models (max_position_embeddings=160)
VH_SEQ = "AVKLVQAGGGVVQPGRSLRLSCIASGFTFSNYGMHWVRQAPGKGLEWVAVIWYNGSRTYYGDSVKGRFTISRDNSKRTLYMQMNSLRTEDTAVYYCARDPDILTAFSFDYWGQGVLVTVSS"
VL_SEQ = "SYELTQPPSVSVSPGQTARITCSANALPNQYAYWYQQKPGRAPVMVIYKDTQRPSGIPQRFSSSTSGTTVTLTISGVQAEDEADYYCQAWDNSASIFGGGTKLTVLG"


# ── Input validation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("input_cls", "sequence"),
    [
        (AbLangEmbeddingsInput, "EVQLVESGGGLVQPGG"),
        (AbLangScoringInput, "EVQLVESGGGLVQPGG"),
        (AbLangSampleInput, "EVQL_ESGGGLVQPGG"),
    ],
)
def test_ablang_input_normalizes_single_string(input_cls, sequence):
    inp = input_cls(sequences=sequence)
    assert isinstance(inp.sequences, list)
    assert inp.sequences == [sequence]


@pytest.mark.parametrize("input_cls", [AbLangEmbeddingsInput, AbLangScoringInput, AbLangSampleInput])
def test_ablang_empty_input_raises(input_cls):
    with pytest.raises(ValueError, match="sequences must not be empty"):
        input_cls(sequences=[])


def test_ablang_germinal_gradient_input_requires_20_aa_columns():
    logits = [[0.0] * 20, [1.0] * 20]
    inp = AbLangGerminalGradientInput(logits=logits, temperature=0.5)
    assert inp.logits == logits

    with pytest.raises(ValidationError, match="20 columns"):
        AbLangGerminalGradientInput(logits=[[0.0] * 19], temperature=1.0)


def test_ablang_germinal_gradient_config_validates_single_chain_variable_fragment_layout():
    config = AbLangGerminalGradientConfig(
        use_single_chain_variable_fragment=True,
        heavy_chain_length=4,
        light_chain_length=3,
    )
    assert config.heavy_chain_length == 4
    assert config.light_chain_length == 3

    with pytest.raises(
        ValidationError,
        match="heavy_chain_length and light_chain_length are required when use_single_chain_variable_fragment=True",
    ):
        AbLangGerminalGradientConfig(use_single_chain_variable_fragment=True)

    with pytest.raises(
        ValidationError,
        match="only supported when use_single_chain_variable_fragment=True",
    ):
        AbLangGerminalGradientConfig(use_single_chain_variable_fragment=False, heavy_chain_length=4)


def test_ablang_germinal_gradient_dispatch_contract(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(tool_name, payload, *, instance=None, config=None):
        captured["tool_name"] = tool_name
        captured["payload"] = payload
        captured["instance"] = instance
        captured["config"] = config
        return {
            "gradient": [[0.1] * 20, [0.2] * 20],
            "loss": 0.75,
            "metrics": {"log_likelihood": -0.75, "model_choice": "ablang2-paired"},
            "vocab": list(PROTEIN_AMINO_ACIDS),
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.ablang.ablang_germinal_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    inputs = AbLangGerminalGradientInput(logits=[[0.0] * 20, [1.0] * 20], temperature=0.8)
    config = AbLangGerminalGradientConfig(
        use_single_chain_variable_fragment=True,
        heavy_chain_first=False,
        heavy_chain_length=1,
        light_chain_length=1,
        seed=17,
        device="cpu",
    )
    result = run_ablang_germinal_gradient(inputs=inputs, config=config)

    validate_output(result)
    assert captured["tool_name"] == "ablang"
    assert captured["payload"] == {
        "operation": "compute_germinal_gradient",
        "logits": inputs.logits,
        "temperature": 0.8,
        "use_single_chain_variable_fragment": True,
        "heavy_chain_first": False,
        "heavy_chain_length": 1,
        "light_chain_length": 1,
        "seed": 17,
        "device": "cpu",
        "verbose": False,
    }
    assert result.gradient == [[0.1] * 20, [0.2] * 20]
    assert result.loss == 0.75
    assert result.metrics["log_likelihood"] == -0.75
    assert result.vocab == _CANONICAL_VOCAB


# ── Germinal gradient unit tests (CPU, fake model) ──────────────────────────


class _FakeHookHandle:
    def __init__(self, layer: "_FakeEmbedLayer") -> None:
        self.layer = layer

    def remove(self) -> None:
        self.layer.hook = None


class _FakeEmbedLayer:
    def __init__(self, weight: Any) -> None:
        self.weight = weight
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return _FakeHookHandle(self)

    def __call__(self, token_ids: Any) -> Any:
        output = self.weight[token_ids]
        if self.hook is not None:
            return self.hook(self, (token_ids,), output)
        return output


class _FakeAbLangPaired:
    def __init__(self, embed_layer: _FakeEmbedLayer, projection: Any) -> None:
        self.embed_layer = embed_layer
        self.projection = projection

    def get_aa_embeddings(self) -> _FakeEmbedLayer:
        return self.embed_layer

    def __call__(self, token_ids: Any) -> Any:
        import torch

        embeddings = self.embed_layer(token_ids)
        return torch.einsum("bld,df->blf", embeddings, self.projection)


def _build_fake_paired_model() -> tuple:
    import torch

    ablang_inference = _import_ablang_inference()

    weight = torch.arange(_ABLANG_VOCAB_SIZE * 3, dtype=torch.float32).reshape(_ABLANG_VOCAB_SIZE, 3) / 10.0
    projection = torch.arange(3 * _ABLANG_VOCAB_SIZE, dtype=torch.float32).reshape(3, _ABLANG_VOCAB_SIZE) / 7.0
    embed_layer = _FakeEmbedLayer(weight)

    model = ablang_inference.AbLangModel(model_choice="ablang2-paired")
    model._loaded = True
    model.device = "cpu"
    model.model = SimpleNamespace(AbLang=_FakeAbLangPaired(embed_layer, projection))
    model._ablang_vocab = {aa: idx for idx, aa in enumerate(GERMINAL_ABLANG_ORDER)} | {"|": _PIPE_TOKEN_ID}
    return model, weight, projection


def _manual_expected_gradient(
    logits_list: list[list[float]],
    *,
    temperature: float,
    heavy_chain_first: bool,
    heavy_chain_length: int,
    light_chain_length: int,
    weight: Any,
    projection: Any,
) -> tuple:
    """Compute the expected gradient by manually reproducing the math."""
    import torch
    import torch.nn.functional as F

    logits = torch.tensor(logits_list, dtype=torch.float32, requires_grad=True)
    linker_length = logits.shape[0] - heavy_chain_length - light_chain_length

    if heavy_chain_first:
        active_logits = torch.cat([logits[:heavy_chain_length], logits[-light_chain_length:]], dim=0)
        insert_position = heavy_chain_length
    else:
        active_logits = torch.cat([logits[:light_chain_length], logits[-heavy_chain_length:]], dim=0)
        insert_position = light_chain_length

    probs = F.softmax(active_logits / temperature, dim=-1)
    mapping = torch.zeros((len(_CANONICAL_VOCAB), _ABLANG_VOCAB_SIZE), dtype=torch.float32)
    for idx, aa in enumerate(_CANONICAL_VOCAB):
        mapping[idx, GERMINAL_ABLANG_ORDER.index(aa)] = 1.0
    mapped_probs = probs @ mapping

    token_ids = mapped_probs.argmax(dim=-1)
    hard = F.one_hot(token_ids, num_classes=_ABLANG_VOCAB_SIZE).float()
    one_hot = hard + (mapped_probs - mapped_probs.detach())

    residue_embeddings = one_hot @ weight
    residue_embeddings = torch.cat(
        [
            residue_embeddings[:insert_position],
            weight[_PIPE_TOKEN_ID].unsqueeze(0),
            residue_embeddings[insert_position:],
        ],
        dim=0,
    )
    token_ids = torch.cat(
        [
            token_ids[:insert_position],
            torch.tensor([_PIPE_TOKEN_ID], dtype=torch.long),
            token_ids[insert_position:],
        ],
        dim=0,
    )

    logits_out = torch.einsum("bld,df->blf", residue_embeddings.unsqueeze(0), projection)
    shift_logits = logits_out[:, :-1, :]
    shift_labels = token_ids.unsqueeze(0)[:, 1:]
    position_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(shift_labels.shape)
    loss = position_losses[:, 1:-1].mean()
    (active_gradient,) = torch.autograd.grad(loss, active_logits)

    zeros = torch.zeros((linker_length, logits.shape[1]), dtype=active_gradient.dtype)
    if heavy_chain_first:
        full_gradient = torch.cat(
            [active_gradient[:heavy_chain_length], zeros, active_gradient[-light_chain_length:]],
            dim=0,
        )
    else:
        full_gradient = torch.cat(
            [active_gradient[:light_chain_length], zeros, active_gradient[-heavy_chain_length:]],
            dim=0,
        )

    return full_gradient, loss.item()


def _vendored_germinal_get_ablm_grad(
    logits_list: list[list[float]],
    *,
    temperature: float,
    vh_first: bool,
    vh_len: int,
    vl_len: int,
    weight: Any,
    projection: Any,
) -> tuple:
    """Faithful vendored copy of Germinal's CustomAbLang.get_grad() + get_ablm_grad().

    Source: https://github.com/SantiagoMille/germinal/blob/main/colabdesign/colabdesign/ablang/model.py

    This reproduces Germinal's exact logic including two known bugs in the
    vh_first=False path (separator position and gradient reassembly order).

    Note: Germinal's original code uses ARNDCQEGHILKMFPSTWYV as input column
    order (identity mapping to AbLang vocab). Here we accept canonical-order
    logits to match our implementation's convention, so the mapping matrix
    is non-trivial.
    """
    import torch
    import torch.nn.functional as F

    ablang_vocab = {aa: idx for idx, aa in enumerate(GERMINAL_ABLANG_ORDER)} | {"|": _PIPE_TOKEN_ID}
    mapping_matrix = torch.zeros(len(_CANONICAL_VOCAB), len(ablang_vocab), dtype=torch.float32)
    for idx, aa in enumerate(_CANONICAL_VOCAB):
        mapping_matrix[idx, ablang_vocab[aa]] = 1.0

    current_logits = torch.tensor(logits_list, dtype=torch.float32, requires_grad=True)

    # Germinal always reorders to [heavy, light].
    x = current_logits
    if vh_first:
        x_h, x_l = x[:vh_len], x[-vl_len:]
    else:
        x_l, x_h = x[:vl_len], x[-vh_len:]
    x = torch.cat([x_h, x_l], dim=0)

    probs = F.softmax(x / temperature, dim=-1)
    mapped_probs = probs @ mapping_matrix
    vocab_size = mapped_probs.size(-1)
    idx = mapped_probs.argmax(dim=-1)
    hard = F.one_hot(idx, num_classes=vocab_size).float()
    oh = hard + (mapped_probs - mapped_probs.detach())

    embed_layer_weight = weight
    residue_embeddings = oh @ embed_layer_weight

    # Germinal bug: uses vl_len instead of vh_len when vh_first=False.
    residue_token_ids = idx.detach()
    insert_pos = vh_len if vh_first else vl_len
    separator_embed = embed_layer_weight[ablang_vocab["|"]]
    residue_embeddings = torch.cat(
        (residue_embeddings[:insert_pos], separator_embed.unsqueeze(0), residue_embeddings[insert_pos:]),
        dim=0,
    )
    residue_token_ids = torch.cat(
        (
            residue_token_ids[:insert_pos],
            torch.tensor([ablang_vocab["|"]], dtype=torch.long),
            residue_token_ids[insert_pos:],
        ),
        dim=0,
    )

    token_ids = residue_token_ids.unsqueeze(0)
    input_embeddings = residue_embeddings.unsqueeze(0)
    logits_out = torch.einsum("bld,df->blf", input_embeddings, projection)
    shift_logits = logits_out[:, :-1, :]
    shift_labels = token_ids[:, 1:]
    loss_flat = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    )
    position_losses = loss_flat.reshape(shift_labels.shape)
    position_losses = position_losses[:, 1:-1]
    loss = position_losses.mean()
    ll = -loss.item()
    (grad,) = torch.autograd.grad(loss, x)

    # Germinal bug: always outputs [grad_h, zeros, grad_l] regardless of vh_first.
    grad_h = grad[:vh_len, :]
    grad_l = grad[-vl_len:, :]
    linker_length = current_logits.shape[0] - vh_len - vl_len
    final_grad = torch.cat([grad_h, torch.zeros((linker_length, 20), dtype=grad.dtype), grad_l], dim=0)

    return final_grad.detach(), ll


_UNIT_LOGITS = [
    [0.2 + i / 50.0 for i in range(20)],
    [0.5 - i / 60.0 for i in range(20)],
    [0.1 + i / 70.0 for i in range(20)],
    [0.3 - i / 80.0 for i in range(20)],
    [0.4 + i / 90.0 for i in range(20)],
    [0.6 - i / 100.0 for i in range(20)],
    [0.7 + i / 110.0 for i in range(20)],
]


def test_one_hot_from_logits_maps_canonical_columns_into_germinal_vocab_order() -> None:
    """Map canonical proto-language logits into Germinal's AbLang residue order."""
    torch = pytest.importorskip("torch")

    model, _, _ = _build_fake_paired_model()
    logits = torch.full((3, len(_CANONICAL_VOCAB)), -10.0, dtype=torch.float32)
    for row, residue in enumerate(("C", "W", "A")):
        logits[row, _CANONICAL_VOCAB.index(residue)] = 10.0

    _, token_ids = model._one_hot_from_logits(logits, temperature=0.01)

    assert token_ids.tolist() == [GERMINAL_ABLANG_ORDER.index(residue) for residue in ("C", "W", "A")]


@pytest.mark.parametrize("heavy_chain_first", [True, False])
def test_compute_germinal_gradient_matches_objective_math(heavy_chain_first: bool) -> None:
    """Validate the exact shifted-cross-entropy math and scFv gradient padding."""
    torch = pytest.importorskip("torch")
    model, weight, projection = _build_fake_paired_model()
    heavy_chain_length = 2
    light_chain_length = 3
    temperature = 0.7

    result = model.compute_germinal_gradient(
        logits_list=_UNIT_LOGITS,
        temperature=temperature,
        use_single_chain_variable_fragment=True,
        heavy_chain_first=heavy_chain_first,
        heavy_chain_length=heavy_chain_length,
        light_chain_length=light_chain_length,
        seed=None,
        device="cpu",
    )

    expected_gradient, expected_loss = _manual_expected_gradient(
        _UNIT_LOGITS,
        temperature=temperature,
        heavy_chain_first=heavy_chain_first,
        heavy_chain_length=heavy_chain_length,
        light_chain_length=light_chain_length,
        weight=weight,
        projection=projection,
    )

    result_gradient = torch.tensor(result["gradient"], dtype=torch.float32)
    torch.testing.assert_close(result_gradient, expected_gradient, rtol=1e-5, atol=1e-6)
    assert result["loss"] == pytest.approx(expected_loss, rel=1e-6)
    assert result["metrics"]["log_likelihood"] == pytest.approx(-expected_loss, rel=1e-6)
    assert result["metrics"]["effective_sequence_length"] == heavy_chain_length + light_chain_length
    assert result["metrics"]["linker_length"] == len(_UNIT_LOGITS) - heavy_chain_length - light_chain_length
    assert result["vocab"] == _CANONICAL_VOCAB


def test_germinal_parity_scfv_vh_first_true() -> None:
    """Exact numerical parity with vendored Germinal code when vh_first=True.

    For vh_first=True both codepaths are equivalent, so outputs must match exactly.
    """
    torch = pytest.importorskip("torch")
    model, weight, projection = _build_fake_paired_model()
    vh_len = 3
    vl_len = 2
    temperature = 0.7

    our_result = model.compute_germinal_gradient(
        logits_list=_UNIT_LOGITS,
        temperature=temperature,
        use_single_chain_variable_fragment=True,
        heavy_chain_first=True,
        heavy_chain_length=vh_len,
        light_chain_length=vl_len,
        seed=None,
        device="cpu",
    )

    germinal_grad, germinal_ll = _vendored_germinal_get_ablm_grad(
        _UNIT_LOGITS,
        temperature=temperature,
        vh_first=True,
        vh_len=vh_len,
        vl_len=vl_len,
        weight=weight,
        projection=projection,
    )

    our_grad = torch.tensor(our_result["gradient"], dtype=torch.float32)
    torch.testing.assert_close(our_grad, germinal_grad, rtol=1e-5, atol=1e-6)
    assert our_result["loss"] == pytest.approx(-germinal_ll, rel=1e-6)


def test_germinal_parity_scfv_vh_first_false_diverges() -> None:
    """Our implementation intentionally diverges from Germinal when vh_first=False.

    Germinal has two bugs in the vh_first=False path (when vh_len != vl_len):
    1. Separator inserted at vl_len instead of vh_len after reordering to heavy-first
    2. Gradient always reassembled as [grad_h, zeros, grad_l] regardless of input layout

    We fix both: output gradient matches the caller's input layout [VL, zeros, VH].
    """
    torch = pytest.importorskip("torch")
    model, weight, projection = _build_fake_paired_model()
    vh_len = 3
    vl_len = 2  # != vh_len, so the bugs manifest
    temperature = 0.7

    our_result = model.compute_germinal_gradient(
        logits_list=_UNIT_LOGITS,
        temperature=temperature,
        use_single_chain_variable_fragment=True,
        heavy_chain_first=False,
        heavy_chain_length=vh_len,
        light_chain_length=vl_len,
        seed=None,
        device="cpu",
    )

    germinal_grad, _ = _vendored_germinal_get_ablm_grad(
        _UNIT_LOGITS,
        temperature=temperature,
        vh_first=False,
        vh_len=vh_len,
        vl_len=vl_len,
        weight=weight,
        projection=projection,
    )

    our_grad = torch.tensor(our_result["gradient"], dtype=torch.float32)

    # Outputs differ because of Germinal's two bugs in the vh_first=False path.
    assert not torch.allclose(our_grad, germinal_grad, rtol=1e-5, atol=1e-6)

    # Germinal always outputs [grad_h, zeros, grad_l] — wrong for VL-first input.
    # Our output is [grad_vl, zeros, grad_vh] — matching the input layout.
    linker_len = len(_UNIT_LOGITS) - vh_len - vl_len
    assert all(v == 0.0 for row in our_result["gradient"][vl_len : vl_len + linker_len] for v in row)


def test_compute_germinal_gradient_rejects_overlong_single_chain_variable_fragment_lengths() -> None:
    """Reject paired chain splits that do not fit inside the full relaxed sequence."""
    pytest.importorskip("torch")
    model, _, _ = _build_fake_paired_model()

    with pytest.raises(ValueError, match="cannot exceed the full relaxed sequence length"):
        model.compute_germinal_gradient(
            logits_list=[[0.0] * 20] * 4,
            temperature=1.0,
            use_single_chain_variable_fragment=True,
            heavy_chain_length=3,
            light_chain_length=2,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("use_single_chain_variable_fragment", "expected_model_choice"),
    [(False, "ablang1-heavy"), (True, "ablang2-paired")],
)
def test_dispatch_routes_germinal_gradient_to_expected_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    use_single_chain_variable_fragment: bool,
    expected_model_choice: str,
) -> None:
    """Dispatch should instantiate the same checkpoint family Germinal uses."""
    created_model_choices: list[str] = []

    class _FakeDispatchedModel:
        def __init__(self, model_choice: str) -> None:
            created_model_choices.append(model_choice)
            self.model_choice = model_choice
            self._loaded = False

        def compute_germinal_gradient(self, **_kwargs):
            return {
                "gradient": [[0.0] * 20],
                "loss": 0.0,
                "metrics": {"model_choice": self.model_choice},
                "vocab": list(PROTEIN_AMINO_ACIDS),
            }

    ablang_inference = _import_ablang_inference()
    monkeypatch.setattr(ablang_inference, "AbLangModel", _FakeDispatchedModel)
    monkeypatch.setattr(ablang_inference, "_model", None)

    result = ablang_inference.dispatch(
        {
            "operation": "compute_germinal_gradient",
            "logits": [[0.0] * 20],
            "temperature": 1.0,
            "use_single_chain_variable_fragment": use_single_chain_variable_fragment,
            "heavy_chain_first": True,
            "heavy_chain_length": 1 if use_single_chain_variable_fragment else None,
            "light_chain_length": 1 if use_single_chain_variable_fragment else None,
            "seed": None,
            "device": "cpu",
            "verbose": False,
        }
    )

    assert created_model_choices == [expected_model_choice]
    assert result["metrics"]["model_choice"] == expected_model_choice


# ── Embedding tests ──────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_embeddings_heavy():
    """Test ablang1-heavy embeddings with multiple variable-length sequences."""
    seqs = [VH_SEQ, VH_SEQ[:50]]
    result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=seqs),
        AbLangEmbeddingsConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    validate_output(result)

    assert result.tool_id == "ablang-embedding"
    assert len(result.results) == 2

    for i, seq in enumerate(seqs):
        emb = result.results[i]
        assert len(emb.mean_embedding) == 768
        assert all(math.isfinite(v) for v in emb.mean_embedding)
        assert all(v in (0, 1) for v in emb.attention_mask)
        assert sum(emb.attention_mask) >= len(seq)

    assert result.results[0].mean_embedding != result.results[1].mean_embedding


@pytest.mark.uses_gpu
def test_ablang_embeddings_light():
    """Test ablang1-light embeddings produce correct 768-dim vectors."""
    result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=[VL_SEQ]),
        AbLangEmbeddingsConfig(model_choice="ablang1-light"),
    )
    validate_output(result)

    emb = result.results[0]
    assert len(emb.mean_embedding) == 768
    assert all(math.isfinite(v) for v in emb.mean_embedding)
    assert all(v in (0, 1) for v in emb.attention_mask)
    assert sum(emb.attention_mask) >= len(VL_SEQ)


@pytest.mark.uses_gpu
def test_ablang_embeddings_paired():
    """Test ablang2-paired embeddings produce correct 480-dim vectors."""
    result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=[PAIRED_SEQ]),
        AbLangEmbeddingsConfig(model_choice="ablang2-paired"),
    )
    validate_output(result)

    emb = result.results[0]
    assert len(emb.mean_embedding) == 480
    assert all(math.isfinite(v) for v in emb.mean_embedding)
    assert all(v in (0, 1) for v in emb.attention_mask)
    assert sum(emb.attention_mask) >= len(HEAVY_FULL) + len(LIGHT_FULL)


@pytest.mark.uses_gpu
def test_ablang_embeddings_auto_routing():
    """Test that auto model routing selects correct model and embedding dim."""
    single_result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=[VH_SEQ]),
        AbLangEmbeddingsConfig(model_choice="auto"),
    )
    assert single_result.success
    assert single_result.metadata["model_choice"] == "ablang1-heavy"
    assert len(single_result.results[0].mean_embedding) == 768

    paired_result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=[PAIRED_SEQ]),
        AbLangEmbeddingsConfig(model_choice="auto"),
    )
    assert paired_result.success
    assert paired_result.metadata["model_choice"] == "ablang2-paired"
    assert len(paired_result.results[0].mean_embedding) == 480


# ── Scoring tests ────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_score_heavy():
    """Test ablang1-heavy PLL: natural VH scores higher than poly-A."""
    result = run_ablang_score(
        AbLangScoringInput(sequences=[VH_SEQ, "A" * 20]),
        AbLangScoringConfig(model_choice="ablang1-heavy", scoring_mode="pseudo_log_likelihood"),
    )
    validate_output(result)

    pll_natural = result.scores[0]["pseudo_log_likelihood"]
    pll_poly_a = result.scores[1]["pseudo_log_likelihood"]

    assert result.tool_id == "ablang-score"
    assert math.isfinite(pll_natural) and pll_natural < 0
    assert math.isfinite(pll_poly_a) and pll_poly_a < 0
    assert pll_natural > pll_poly_a


@pytest.mark.uses_gpu
def test_ablang_score_light():
    """Test ablang1-light PLL: natural VL scores higher than poly-A."""
    result = run_ablang_score(
        AbLangScoringInput(sequences=[VL_SEQ, "A" * 20]),
        AbLangScoringConfig(model_choice="ablang1-light", scoring_mode="pseudo_log_likelihood"),
    )
    validate_output(result)

    pll_natural = result.scores[0]["pseudo_log_likelihood"]
    pll_poly_a = result.scores[1]["pseudo_log_likelihood"]

    assert math.isfinite(pll_natural) and pll_natural < 0
    assert math.isfinite(pll_poly_a) and pll_poly_a < 0
    assert pll_natural > pll_poly_a


@pytest.mark.uses_gpu
def test_ablang_score_paired():
    """Test ablang2-paired PLL produces finite negative scores."""
    result = run_ablang_score(
        AbLangScoringInput(sequences=[PAIRED_SEQ]),
        AbLangScoringConfig(model_choice="ablang2-paired", scoring_mode="pseudo_log_likelihood"),
    )
    validate_output(result)

    pll = result.scores[0]["pseudo_log_likelihood"]
    assert math.isfinite(pll) and pll < 0


@pytest.mark.uses_gpu
def test_ablang_score_confidence_mode():
    """Test confidence scoring returns finite scores for all three models."""
    for model, seq in [("ablang1-heavy", VH_SEQ), ("ablang1-light", VL_SEQ), ("ablang2-paired", PAIRED_SEQ)]:
        result = run_ablang_score(
            AbLangScoringInput(sequences=[seq]),
            AbLangScoringConfig(model_choice=model, scoring_mode="confidence"),
        )
        validate_output(result)
        assert math.isfinite(result.scores[0]["confidence"])


# ── Sampling tests ───────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_sample_heavy():
    """Test ablang1-heavy restoration: valid AAs, length preserved, unmasked positions unchanged."""
    mask_pos = 4
    masked_seq = VH_SEQ[:mask_pos] + "_" + VH_SEQ[mask_pos + 1 :]
    result = run_ablang_sample(
        AbLangSampleInput(sequences=[masked_seq]),
        AbLangSampleConfig(model_choice="ablang1-heavy"),
    )
    validate_output(result)

    restored = result.sequences[0]
    assert result.tool_id == "ablang-sample"
    assert len(restored) == len(VH_SEQ)
    assert "_" not in restored
    assert set(restored) <= _VALID_AAS
    assert restored[:mask_pos] == VH_SEQ[:mask_pos]
    assert restored[mask_pos + 1 :] == VH_SEQ[mask_pos + 1 :]


@pytest.mark.uses_gpu
def test_ablang_sample_light():
    """Test ablang1-light restoration: valid AAs, length preserved, unmasked unchanged."""
    mask_pos = 4
    masked_seq = VL_SEQ[:mask_pos] + "_" + VL_SEQ[mask_pos + 1 :]
    result = run_ablang_sample(
        AbLangSampleInput(sequences=[masked_seq]),
        AbLangSampleConfig(model_choice="ablang1-light"),
    )
    validate_output(result)

    restored = result.sequences[0]
    assert len(restored) == len(VL_SEQ)
    assert "_" not in restored
    assert set(restored) <= _VALID_AAS
    assert restored[:mask_pos] == VL_SEQ[:mask_pos]
    assert restored[mask_pos + 1 :] == VL_SEQ[mask_pos + 1 :]


@pytest.mark.uses_gpu
def test_ablang_sample_paired():
    """Test ablang2-paired restoration: valid AAs, length preserved, unmasked unchanged."""
    mask_pos = 4
    masked_heavy = HEAVY_FULL[:mask_pos] + "_" + HEAVY_FULL[mask_pos + 1 :]
    masked_paired = masked_heavy + "|" + LIGHT_FULL
    result = run_ablang_sample(
        AbLangSampleInput(sequences=[masked_paired]),
        AbLangSampleConfig(model_choice="ablang2-paired"),
    )
    validate_output(result)

    restored = result.sequences[0]
    residues_only = restored.replace("|", "")
    assert "_" not in restored
    assert len(restored) == len(masked_paired)
    assert set(residues_only) <= _VALID_AAS
    assert restored[:mask_pos] == HEAVY_FULL[:mask_pos]
    assert restored[mask_pos + 1 : len(HEAVY_FULL)] == HEAVY_FULL[mask_pos + 1 :]


# ── Germinal gradient tests ─────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_germinal_gradient_vhh():
    """Test Germinal VHH gradient: shape, finiteness, and metric consistency."""
    seq_len = 10
    result = run_ablang_germinal_gradient(
        AbLangGerminalGradientInput(logits=[[0.0] * 20] * seq_len, temperature=0.6),
        AbLangGerminalGradientConfig(use_single_chain_variable_fragment=False),
    )
    validate_output(result)

    assert result.tool_id == "ablang-germinal-gradient"
    assert len(result.gradient) == seq_len
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert any(v != 0.0 for row in result.gradient for v in row)
    assert math.isfinite(result.loss) and result.loss > 0
    assert result.vocab == _CANONICAL_VOCAB
    assert result.metrics["sequence_length"] == seq_len
    assert result.metrics["effective_sequence_length"] == seq_len
    assert result.metrics["linker_length"] == 0
    assert result.metrics["model_choice"] == "ablang1-heavy"
    assert result.metrics["objective"] == "germinal_shifted_cross_entropy"
    assert result.metrics["log_likelihood"] == pytest.approx(-result.loss, rel=1e-6)


@pytest.mark.parametrize(
    ("heavy_chain_first", "expected_linker_start"),
    [(True, 4), (False, 5)],
)
@pytest.mark.uses_gpu
def test_ablang_germinal_gradient_single_chain_variable_fragment_zeroes_linker(
    heavy_chain_first: bool,
    expected_linker_start: int,
):
    """Test Germinal paired-chain gradients zero-pad the linker for both chain orders."""
    heavy_chain_length = 4
    linker_len = 3
    light_chain_length = 5
    seq_len = heavy_chain_length + linker_len + light_chain_length

    result = run_ablang_germinal_gradient(
        AbLangGerminalGradientInput(logits=[[0.0] * 20] * seq_len, temperature=0.6),
        AbLangGerminalGradientConfig(
            use_single_chain_variable_fragment=True,
            heavy_chain_length=heavy_chain_length,
            light_chain_length=light_chain_length,
            heavy_chain_first=heavy_chain_first,
        ),
    )
    validate_output(result)

    linker_rows = result.gradient[expected_linker_start : expected_linker_start + linker_len]
    prefix_length = heavy_chain_length if heavy_chain_first else light_chain_length
    suffix_length = light_chain_length if heavy_chain_first else heavy_chain_length
    non_linker_rows = result.gradient[:prefix_length] + result.gradient[-suffix_length:]

    assert len(result.gradient) == seq_len
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert all(v == 0.0 for row in linker_rows for v in row)
    assert any(v != 0.0 for row in non_linker_rows for v in row)
    assert result.vocab == _CANONICAL_VOCAB
    assert result.metrics["sequence_length"] == seq_len
    assert result.metrics["effective_sequence_length"] == heavy_chain_length + light_chain_length
    assert result.metrics["linker_length"] == linker_len
    assert result.metrics["use_single_chain_variable_fragment"] is True
    assert result.metrics["model_choice"] == "ablang2-paired"
    assert result.metrics["objective"] == "germinal_shifted_cross_entropy"


@pytest.mark.uses_gpu
def test_ablang_germinal_gradient_descent_reduces_loss():
    """Taking a step in the negative gradient direction should reduce the loss."""
    seq_len = 10
    initial_logits = [[0.0] * 20] * seq_len

    result_0 = run_ablang_germinal_gradient(
        AbLangGerminalGradientInput(logits=initial_logits, temperature=0.7),
        AbLangGerminalGradientConfig(use_single_chain_variable_fragment=False, seed=42),
    )

    lr = 10.0
    stepped_logits = [[initial_logits[i][j] - lr * result_0.gradient[i][j] for j in range(20)] for i in range(seq_len)]

    result_1 = run_ablang_germinal_gradient(
        AbLangGerminalGradientInput(logits=stepped_logits, temperature=0.7),
        AbLangGerminalGradientConfig(use_single_chain_variable_fragment=False, seed=42),
    )

    assert result_1.loss < result_0.loss


@pytest.mark.uses_gpu
def test_ablang_germinal_gradient_vh_first_false_layout_preserved():
    """vh_first=False returns gradients in [VL, linker_zeros, VH] layout order.

    Verifies that the gradient layout matches the input layout for both orderings,
    with non-zero chain gradients and zero-padded linker positions.
    """
    heavy_chain_length = 5
    light_chain_length = 4
    linker_len = 2
    seq_len = heavy_chain_length + linker_len + light_chain_length

    result_vh_first = run_ablang_germinal_gradient(
        AbLangGerminalGradientInput(logits=[[0.0] * 20] * seq_len, temperature=0.6),
        AbLangGerminalGradientConfig(
            use_single_chain_variable_fragment=True,
            heavy_chain_length=heavy_chain_length,
            light_chain_length=light_chain_length,
            heavy_chain_first=True,
        ),
    )
    result_vl_first = run_ablang_germinal_gradient(
        AbLangGerminalGradientInput(logits=[[0.0] * 20] * seq_len, temperature=0.6),
        AbLangGerminalGradientConfig(
            use_single_chain_variable_fragment=True,
            heavy_chain_length=heavy_chain_length,
            light_chain_length=light_chain_length,
            heavy_chain_first=False,
        ),
    )

    # Linker positions are zero in both layouts
    vh_first_linker = result_vh_first.gradient[heavy_chain_length : heavy_chain_length + linker_len]
    vl_first_linker = result_vl_first.gradient[light_chain_length : light_chain_length + linker_len]
    assert all(v == 0.0 for row in vh_first_linker for v in row)
    assert all(v == 0.0 for row in vl_first_linker for v in row)

    # Non-linker positions have non-zero gradients in both layouts
    vh_first_chains = result_vh_first.gradient[:heavy_chain_length] + result_vh_first.gradient[-light_chain_length:]
    vl_first_chains = result_vl_first.gradient[:light_chain_length] + result_vl_first.gradient[-heavy_chain_length:]
    assert any(v != 0.0 for row in vh_first_chains for v in row)
    assert any(v != 0.0 for row in vl_first_chains for v in row)


# ── Batched tests ────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_batched_operations():
    """Test that batch_size > 1 works for embeddings, scoring, and sampling."""
    seqs = [VH_SEQ, VH_SEQ[:50], VH_SEQ[:80]]

    emb_result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=seqs),
        AbLangEmbeddingsConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    assert emb_result.success
    assert len(emb_result.results) == 3
    assert all(len(r.mean_embedding) == 768 for r in emb_result.results)

    score_result = run_ablang_score(
        AbLangScoringInput(sequences=seqs),
        AbLangScoringConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    assert score_result.success
    assert len(score_result.scores) == 3
    assert all("pseudo_log_likelihood" in s for s in score_result.scores)

    masked_seqs = [s[:4] + "_" + s[5:] for s in seqs]
    sample_result = run_ablang_sample(
        AbLangSampleInput(sequences=masked_seqs),
        AbLangSampleConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    assert sample_result.success
    assert len(sample_result.sequences) == 3
    assert all("_" not in s for s in sample_result.sequences)

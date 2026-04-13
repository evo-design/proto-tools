"""Focused unit tests for Germinal's AbLang gradient backend."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS

sys.modules.setdefault("standalone_helpers", types.SimpleNamespace(serialize_output=lambda value: value))

from proto_tools.tools.masked_models.ablang.standalone import inference as ablang_inference  # noqa: E402

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

CANONICAL_AA_ORDER = list(PROTEIN_AMINO_ACIDS)
GERMINAL_ABLANG_ORDER = list("ARNDCQEGHILKMFPSTWYV")
VOCAB_SIZE = len(GERMINAL_ABLANG_ORDER) + 1
PIPE_TOKEN_ID = len(GERMINAL_ABLANG_ORDER)


class _FakeHookHandle:
    def __init__(self, layer: _FakeEmbedLayer) -> None:
        self.layer = layer

    def remove(self) -> None:
        self.layer.hook = None


class _FakeEmbedLayer:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return _FakeHookHandle(self)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = self.weight[token_ids]
        if self.hook is not None:
            return self.hook(self, (token_ids,), output)
        return output


class _FakeAbLangPaired:
    def __init__(self, embed_layer: _FakeEmbedLayer, projection: torch.Tensor) -> None:
        self.embed_layer = embed_layer
        self.projection = projection

    def get_aa_embeddings(self) -> _FakeEmbedLayer:
        return self.embed_layer

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed_layer(token_ids)
        return torch.einsum("bld,df->blf", embeddings, self.projection)


def _build_fake_paired_model() -> tuple[ablang_inference.AbLangModel, torch.Tensor, torch.Tensor]:
    weight = torch.arange(VOCAB_SIZE * 3, dtype=torch.float32).reshape(VOCAB_SIZE, 3) / 10.0
    projection = torch.arange(3 * VOCAB_SIZE, dtype=torch.float32).reshape(3, VOCAB_SIZE) / 7.0
    embed_layer = _FakeEmbedLayer(weight)

    model = ablang_inference.AbLangModel(model_choice="ablang2-paired")
    model._loaded = True
    model.device = "cpu"
    model.model = SimpleNamespace(AbLang=_FakeAbLangPaired(embed_layer, projection))
    model._ablang_vocab = {aa: idx for idx, aa in enumerate(GERMINAL_ABLANG_ORDER)} | {"|": PIPE_TOKEN_ID}
    return model, weight, projection


def _manual_expected_gradient(
    logits_list: list[list[float]],
    *,
    temperature: float,
    heavy_chain_first: bool,
    heavy_chain_length: int,
    light_chain_length: int,
    weight: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    logits = torch.tensor(logits_list, dtype=torch.float32, requires_grad=True)
    linker_length = logits.shape[0] - heavy_chain_length - light_chain_length

    if heavy_chain_first:
        active_logits = torch.cat([logits[:heavy_chain_length], logits[-light_chain_length:]], dim=0)
        insert_position = heavy_chain_length
    else:
        active_logits = torch.cat([logits[:light_chain_length], logits[-heavy_chain_length:]], dim=0)
        insert_position = light_chain_length

    probs = F.softmax(active_logits / temperature, dim=-1)
    mapping = torch.zeros((len(CANONICAL_AA_ORDER), VOCAB_SIZE), dtype=torch.float32)
    for idx, aa in enumerate(CANONICAL_AA_ORDER):
        mapping[idx, GERMINAL_ABLANG_ORDER.index(aa)] = 1.0
    mapped_probs = probs @ mapping

    token_ids = mapped_probs.argmax(dim=-1)
    hard = F.one_hot(token_ids, num_classes=VOCAB_SIZE).float()
    one_hot = hard + (mapped_probs - mapped_probs.detach())

    residue_embeddings = one_hot @ weight
    residue_embeddings = torch.cat(
        [
            residue_embeddings[:insert_position],
            weight[PIPE_TOKEN_ID].unsqueeze(0),
            residue_embeddings[insert_position:],
        ],
        dim=0,
    )
    token_ids = torch.cat(
        [
            token_ids[:insert_position],
            torch.tensor([PIPE_TOKEN_ID], dtype=torch.long),
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


def test_one_hot_from_logits_maps_canonical_columns_into_germinal_vocab_order() -> None:
    """Map canonical proto-language logits into Germinal's AbLang residue order."""
    model, _, _ = _build_fake_paired_model()
    logits = torch.full((3, len(CANONICAL_AA_ORDER)), -10.0, dtype=torch.float32)
    for row, residue in enumerate(("C", "W", "A")):
        logits[row, CANONICAL_AA_ORDER.index(residue)] = 10.0

    _, token_ids = model._one_hot_from_logits(logits, temperature=0.01)

    assert token_ids.tolist() == [GERMINAL_ABLANG_ORDER.index(residue) for residue in ("C", "W", "A")]


@pytest.mark.parametrize("heavy_chain_first", [True, False])
def test_compute_germinal_gradient_matches_objective_math(heavy_chain_first: bool) -> None:
    """Validate the exact shifted-cross-entropy math and single-chain variable fragment gradient padding."""
    model, weight, projection = _build_fake_paired_model()
    logits_list = [
        [0.2 + i / 50.0 for i in range(20)],
        [0.5 - i / 60.0 for i in range(20)],
        [0.1 + i / 70.0 for i in range(20)],
        [0.3 - i / 80.0 for i in range(20)],
        [0.4 + i / 90.0 for i in range(20)],
        [0.6 - i / 100.0 for i in range(20)],
        [0.7 + i / 110.0 for i in range(20)],
    ]
    heavy_chain_length = 2
    light_chain_length = 3
    temperature = 0.7

    result = model.compute_germinal_gradient(
        logits_list=logits_list,
        temperature=temperature,
        use_single_chain_variable_fragment=True,
        heavy_chain_first=heavy_chain_first,
        heavy_chain_length=heavy_chain_length,
        light_chain_length=light_chain_length,
        seed=None,
        device="cpu",
    )

    expected_gradient, expected_loss = _manual_expected_gradient(
        logits_list,
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
    assert result["metrics"]["linker_length"] == len(logits_list) - heavy_chain_length - light_chain_length
    assert result["metrics"]["use_single_chain_variable_fragment"] is True
    assert result["metrics"]["model_choice"] == "ablang2-paired"
    assert result["metrics"]["objective"] == "germinal_shifted_cross_entropy"
    assert result["vocab"] == CANONICAL_AA_ORDER

    linker_start = heavy_chain_length if heavy_chain_first else light_chain_length
    linker_stop = linker_start + (len(logits_list) - heavy_chain_length - light_chain_length)
    assert torch.count_nonzero(result_gradient[linker_start:linker_stop]) == 0
    assert torch.count_nonzero(torch.cat([result_gradient[:linker_start], result_gradient[linker_stop:]], dim=0)) > 0


def test_compute_germinal_gradient_rejects_overlong_single_chain_variable_fragment_lengths() -> None:
    """Reject paired chain splits that do not fit inside the full relaxed sequence."""
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

    monkeypatch.setattr(ablang_inference, "AbLangModel", _FakeDispatchedModel)
    monkeypatch.setattr(ablang_inference, "_model", None)

    result = ablang_inference.dispatch(
        {
            "operation": "compute_germinal_gradient",
            "logits": [[0.0] * 20],
            "temperature": 1.0,
            "use_single_chain_variable_fragment": use_single_chain_variable_fragment,
        }
    )

    assert created_model_choices == [expected_model_choice]
    assert result["metrics"]["model_choice"] == expected_model_choice

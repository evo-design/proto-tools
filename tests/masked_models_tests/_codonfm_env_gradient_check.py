"""In-env check that the CodonFM relaxed gradient path pads to /8 and returns only input gradients.

The standalone ``inference`` module's gradient path needs torch, so this runs **inside** the
codonfm standalone env via :func:`proto_tools.utils.run_in_env`, invoked by
``test_codonfm_worker_gradient_is_padded_finite_and_does_not_touch_parameter_grads`` in
``test_codonfm.py``. A tiny fake Encodon stands in for the real checkpoint.

``pytest`` is not installed in the standalone env, so this is a plain-``assert`` script; a
failure exits non-zero, which ``run_in_env`` surfaces as ``RuntimeError``. The leading ``_``
keeps pytest from collecting this file in the base process.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import inference  # standalone module; on PYTHONPATH inside the codonfm env
import torch

CODON_VOCAB = ["".join(codon) for codon in itertools.product("ACGT", repeat=3)]


class FakeEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(69, 8, padding_idx=67)
        self.post_ln = torch.nn.LayerNorm(8)
        self.dropout = torch.nn.Dropout(0.0)

    def forward(self, input_ids):
        return self.dropout(self.post_ln(self.word_embeddings(input_ids)))


class FakeModel(torch.nn.Module):
    def __init__(self, embedding: FakeEmbedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.head = torch.nn.Linear(8, 69)

    def forward(self, batch):
        input_ids = next(value for key, value in batch.items() if "input" in str(key).lower())
        assert input_ids.shape[-1] % 8 == 0, f"context {input_ids.shape[-1]} is not a multiple of 8"
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def main() -> None:
    embedding = FakeEmbedding()
    fake_model = FakeModel(embedding)
    fake_model.requires_grad_(False)
    tokenizer = SimpleNamespace(
        codons=CODON_VOCAB,
        encoder={
            **{codon: idx for idx, codon in enumerate(CODON_VOCAB)},
            "<CLS>": 64,
            "<SEP>": 65,
            "<MASK>": 66,
            "<PAD>": 67,
            "<UNK>": 68,
        },
        cls_token_id=64,
        sep_token_id=65,
        mask_token_id=66,
        pad_token_id=67,
    )
    model = inference.CodonFMModel()
    model.inference = SimpleNamespace(model=fake_model)
    model.tokenizer = tokenizer
    model.device = "cpu"
    model._embedding_layer = lambda: embedding

    result = model.compute_gradient([[0.0] * 64 for _ in range(3)], temperature=1.0, batch_size=2, device="cpu")
    assert len(result["gradient"]) == 3
    assert all(math.isfinite(value) for row in result["gradient"] for value in row)
    assert all(parameter.grad is None for parameter in fake_model.parameters())
    print("CODONFM_GRADIENT_OK")


if __name__ == "__main__":
    main()

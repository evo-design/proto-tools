"""Tests for masked model shared data model schemas."""

import json

import pytest

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
    MaskedModelOutput,
    MaskedModelScoringInput,
    MaskedModelScoringOutput,
    SequenceEmbedding,
    SequenceScores,
)

# ── Input normalization ─────────────────────────────────────────────────────


def test_input_string_normalized_to_list():
    assert MaskedModelInput(sequences="MVLSP").sequences == ["MVLSP"]


def test_input_rejects_none_in_sequences():
    with pytest.raises(ValueError, match="cannot be None"):
        MaskedModelInput(sequences=["A", None])


def test_scoring_input_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        MaskedModelScoringInput(sequences=[])


# ── SequenceScores attribute access ─────────────────────────────────────────


def test_sequence_scores_attribute_access():
    scores = SequenceScores(metrics={"perplexity": 1.5})
    assert scores.perplexity == 1.5
    with pytest.raises(AttributeError, match="no attribute"):
        _ = scores.nonexistent


# ── Export ──────────────────────────────────────────────────────────────────


def _make_embedding_output() -> MaskedModelOutput:
    return MaskedModelOutput(
        results=[
            SequenceEmbedding(mean_embedding=[0.1, 0.2, 0.3], attention_mask=[1, 1, 1]),
            SequenceEmbedding(mean_embedding=[0.4, 0.5, 0.6], attention_mask=[1, 1, 0]),
        ],
    )


@pytest.mark.parametrize("fmt", ["csv", "json", "npy"])
def test_embedding_export(fmt, tmp_path):
    _make_embedding_output().export("embeddings", export_path=tmp_path, file_format=fmt)
    assert (tmp_path / f"embeddings.{fmt}").stat().st_size > 0


def test_embedding_export_empty_warns():
    with pytest.warns(UserWarning, match="No embeddings"):
        MaskedModelOutput(results=[])._export_output("/dev/null", "csv")


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_scoring_export(fmt, tmp_path):
    output = MaskedModelScoringOutput(
        scores=[SequenceScores(metrics={"perplexity": 1.5, "log_likelihood": -3.2})],
    )
    output.export("scores", export_path=tmp_path, file_format=fmt)
    exported = tmp_path / f"scores.{fmt}"
    assert exported.stat().st_size > 0
    if fmt == "json":
        assert json.loads(exported.read_text())[0]["perplexity"] == 1.5

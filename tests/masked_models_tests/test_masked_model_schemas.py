"""Tests for masked model shared data model schemas."""

import json
import re

import pytest

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelEmbeddingsOutput,
    MaskedModelInput,
    MaskedModelSampleInput,
    MaskedModelScoringMetrics,
    MaskedModelScoringOutput,
    Projection2D,
    SequenceEmbedding,
)

# ── Input normalization ─────────────────────────────────────────────────────


def test_input_string_normalized_to_list():
    assert MaskedModelInput(sequences="MVLSP").sequences == ["MVLSP"]


def test_input_rejects_none_in_sequences():
    with pytest.raises(ValueError, match="cannot be None"):
        MaskedModelInput(sequences=["A", None])


# ── Sequence alphabet ───────────────────────────────────────────────────────


@pytest.mark.parametrize("model", [MaskedModelInput, MaskedModelSampleInput])
def test_input_accepts_its_own_alphabet(model):
    alphabet = model.SEQUENCE_ALPHABET
    assert model(sequences=[alphabet]).sequences == [alphabet]


def test_sample_input_accepts_the_mask_token():
    assert MaskedModelSampleInput(sequences="MKT_V").sequences == ["MKT_V"]


def test_base_input_rejects_the_mask_token():
    with pytest.raises(ValueError, match=re.escape("does not do. Pass a fully specified sequence")):
        MaskedModelInput(sequences=["MKT_V"])


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        pytest.param("MKT123!!", "'1', '2', '3', '!' starting at position 4", id="digits_and_punctuation"),
        pytest.param("MKTJV", "'J' starting at position 4", id="unrepresentable_residue"),
        pytest.param("MKTAV ", "' ' starting at position 6", id="trailing_space"),
        pytest.param("MKT\tV", "'\\t' starting at position 4", id="tab"),
        pytest.param("MKTav", "'a', 'v' starting at position 4", id="lowercase"),
        pytest.param("MKT-V", "'-' starting at position 4", id="alignment_gap"),
        pytest.param("MKTV*", "'*' starting at position 5", id="translation_stop"),
    ],
)
@pytest.mark.parametrize("model", [MaskedModelInput, MaskedModelSampleInput])
def test_input_rejects_characters_outside_alphabet(model, sequence, expected):
    with pytest.raises(ValueError, match=re.escape(f"invalid character(s) {expected}")):
        model(sequences=[sequence])


def test_literal_mask_token_error_points_at_the_mask_character():
    with pytest.raises(ValueError, match=re.escape("write a masked position as '_', not '<mask>'")):
        MaskedModelInput(sequences=["MKT<mask>V"])


def test_alphabet_error_names_the_offending_sequence():
    with pytest.raises(ValueError, match=r"sequences\[1\]: invalid character\(s\) 'J'"):
        MaskedModelInput(sequences=["MKTV", "MKTJV"])


def test_reported_position_locates_the_first_listed_character():
    """Characters are listed by first occurrence, so the position describes the one named first."""
    with pytest.raises(ValueError, match=re.escape("'z', 'a' starting at position 2")):
        MaskedModelInput(sequences=["MzKTa"])


def test_input_rejects_empty_sequence():
    with pytest.raises(ValueError, match=r"sequences\[0\]: cannot be empty"):
        MaskedModelInput(sequences=[""])


# ── MaskedModelScoringMetrics attribute access ──────────────────────────────


def test_scoring_metrics_attribute_access():
    scores = MaskedModelScoringMetrics(perplexity=1.5)
    assert scores.perplexity == 1.5
    assert scores["perplexity"] == 1.5
    with pytest.raises(AttributeError):
        _ = scores.nonexistent


# ── Projection round-trip ───────────────────────────────────────────────────


def test_sequence_embedding_round_trips_projection():
    e = SequenceEmbedding(
        mean_embedding=[0.1, 0.2],
        attention_mask=[1, 1],
        projection=Projection2D(x=1.5, y=-2.0),
    )
    assert e.model_dump()["projection"] == {"x": 1.5, "y": -2.0}
    restored = SequenceEmbedding.model_validate_json(e.model_dump_json())
    assert restored.projection == Projection2D(x=1.5, y=-2.0)


# ── Export ──────────────────────────────────────────────────────────────────


def _make_embedding_output() -> MaskedModelEmbeddingsOutput:
    return MaskedModelEmbeddingsOutput(
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
        MaskedModelEmbeddingsOutput(results=[])._export_output("/dev/null", "csv")


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_scoring_export(fmt, tmp_path):
    output = MaskedModelScoringOutput(
        scores=[MaskedModelScoringMetrics(perplexity=1.5, log_likelihood=-3.2)],
    )
    output.export("scores", export_path=tmp_path, file_format=fmt)
    exported = tmp_path / f"scores.{fmt}"
    assert exported.stat().st_size > 0
    if fmt == "json":
        assert json.loads(exported.read_text())[0]["perplexity"] == 1.5

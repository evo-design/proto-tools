"""Tests for sequence interaction maps and exports."""

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from proto_tools.entities import SequenceInteractionMap, SequenceInteractions
from proto_tools.utils.interaction_models import SequenceInteractionsOutput

_VALUES = [[0.1, 0.8, 0.0], [0.3, 0.2, 1.0], [0.9, 0.4, 0.5]]


@pytest.mark.parametrize(
    "tokens",
    [["A", "W", "H"], ["A", "U", "N"], ["1", "12", "300"], ["\N{GREEK SMALL LETTER ALPHA}", "residue:10", "[MASK]"]],
)
def test_sequence_interactions_accept_model_independent_labels_and_channels(tokens):
    payload = {"maps": {"custom-channel.v1": {"tokens": tokens, "values": _VALUES}}}
    interactions = SequenceInteractions.model_validate(payload)
    restored = SequenceInteractions.model_validate_json(interactions.model_dump_json())
    assert restored.model_dump() == payload
    # Preserve asymmetric values and nonzero diagonals supplied by the model.
    assert restored.maps["custom-channel.v1"].values[0][1] != restored.maps["custom-channel.v1"].values[1][0]
    assert restored.maps["custom-channel.v1"].values[0][0] == 0.1


@pytest.mark.parametrize("values", [[], [[0.0]], [[0.0, 1.0]], [[0.0], [1.0]], [[0.0, 1.0], [1.0]]])
def test_sequence_interaction_map_rejects_misaligned_or_nonsquare_values(values):
    with pytest.raises(ValidationError, match=r"square and match its token axis"):
        SequenceInteractionMap(tokens=["A", "U"], values=values)


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_sequence_interaction_map_rejects_invalid_probabilities(value):
    with pytest.raises(ValidationError, match=r"finite values in \[0, 1\]"):
        SequenceInteractionMap(tokens=["position:1"], values=[[value]])


@pytest.fixture
def interaction_output():
    return SequenceInteractionsOutput(
        results=[
            SequenceInteractions(
                maps={
                    "rna_base_pairs": SequenceInteractionMap(tokens=["A", "U", "N"], values=_VALUES),
                    "custom-channel.v1": SequenceInteractionMap(tokens=["1", "12", "300"], values=_VALUES),
                }
            ),
            SequenceInteractions(
                maps={"binding": SequenceInteractionMap(tokens=["\N{GREEK SMALL LETTER ALPHA}"], values=[[1.0]])}
            ),
        ]
    )


def test_sequence_interactions_output_and_json_export_round_trip(interaction_output, tmp_path):
    restored = SequenceInteractionsOutput.model_validate_json(interaction_output.model_dump_json())
    assert restored.model_dump() == interaction_output.model_dump()
    interaction_output._export_output(tmp_path / "interactions.v1", "json")
    payload = json.loads((tmp_path / "interactions.v1.json").read_text())
    assert payload == [result.model_dump() for result in interaction_output.results]
    exported = SequenceInteractionsOutput(results=payload)
    assert exported.results == interaction_output.results


def test_sequence_interactions_npz_export_round_trip_without_pickle(interaction_output, tmp_path):
    interaction_output._export_output(tmp_path / "interactions.v1", "npz")
    with np.load(tmp_path / "interactions.v1.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "0_rna_base_pairs",
            "0_rna_base_pairs_tokens",
            "0_custom-channel.v1",
            "0_custom-channel.v1_tokens",
            "1_binding",
            "1_binding_tokens",
        }
        for index, result in enumerate(interaction_output.results):
            for channel, interaction_map in result.maps.items():
                values = archive[f"{index}_{channel}"]
                tokens = archive[f"{index}_{channel}_tokens"]
                assert values.dtype == np.float32
                assert tokens.dtype.kind == "U"
                restored = SequenceInteractionMap(tokens=tokens.tolist(), values=values.tolist())
                assert restored.tokens == interaction_map.tokens
                np.testing.assert_allclose(restored.values, interaction_map.values, rtol=1e-6)


@pytest.mark.parametrize("channels", [["foo", "foo_tokens"], ["foo_tokens", "foo"]])
def test_sequence_interactions_npz_export_rejects_colliding_channel_names(channels, tmp_path):
    interaction_map = SequenceInteractionMap(tokens=["A"], values=[[0.5]])
    output = SequenceInteractionsOutput(results=[SequenceInteractions(maps=dict.fromkeys(channels, interaction_map))])
    with pytest.raises(ValueError, match=r"duplicate NPZ key: 0_foo_tokens"):
        output._export_output(tmp_path / "interactions", "npz")
    assert not (tmp_path / "interactions.npz").exists()

"""Tests for sequence interaction maps and exports."""

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from proto_tools.utils.interaction_models import (
    SequenceInteractionMap,
    SequenceInteractions,
    SequenceInteractionsOutput,
)

_VALUES = [[0.1, 0.8, 0.0], [0.3, 0.2, 1.0], [0.9, 0.4, 0.5]]


@pytest.mark.parametrize(
    "axis_labels",
    [["A", "W", "H"], ["A", "U", "N"], ["1", "12", "300"], ["\N{GREEK SMALL LETTER ALPHA}", "residue:10", "[MASK]"]],
)
def test_sequence_interactions_accept_model_independent_labels_and_channels(axis_labels):
    payload = {"maps": {"custom-channel.v1": {"axis_labels": axis_labels, "values": _VALUES}}}
    interactions = SequenceInteractions.model_validate(payload)
    restored = SequenceInteractions.model_validate_json(interactions.model_dump_json())
    assert restored.model_dump() == payload
    # Preserve asymmetric values and nonzero diagonals supplied by the model.
    assert restored.maps["custom-channel.v1"].values[0][1] != restored.maps["custom-channel.v1"].values[1][0]
    assert restored.maps["custom-channel.v1"].values[0][0] == 0.1


@pytest.mark.parametrize("values", [[], [[0.0]], [[0.0, 1.0]], [[0.0], [1.0]], [[0.0, 1.0], [1.0]]])
def test_sequence_interaction_map_rejects_misaligned_or_nonsquare_values(values):
    with pytest.raises(ValidationError, match=r"square and match its axis labels"):
        SequenceInteractionMap(axis_labels=["A", "U"], values=values)


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_sequence_interaction_map_rejects_invalid_probabilities(value):
    with pytest.raises(ValidationError, match=r"finite values in \[0, 1\]"):
        SequenceInteractionMap(axis_labels=["position:1"], values=[[value]])


@pytest.fixture
def interaction_output():
    return SequenceInteractionsOutput(
        results=[
            SequenceInteractions(
                maps={
                    "rna_base_pairs": SequenceInteractionMap(axis_labels=["A", "U", "N"], values=_VALUES),
                    "custom-channel.v1": SequenceInteractionMap(axis_labels=["1", "12", "300"], values=_VALUES),
                }
            ),
            SequenceInteractions(
                maps={"binding": SequenceInteractionMap(axis_labels=["\N{GREEK SMALL LETTER ALPHA}"], values=[[1.0]])}
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
            "0_rna_base_pairs_axis_labels",
            "0_custom-channel.v1",
            "0_custom-channel.v1_axis_labels",
            "1_binding",
            "1_binding_axis_labels",
        }
        for index, result in enumerate(interaction_output.results):
            for channel, interaction_map in result.maps.items():
                values = archive[f"{index}_{channel}"]
                axis_labels = archive[f"{index}_{channel}_axis_labels"]
                assert values.dtype == np.float32
                assert axis_labels.dtype.kind == "U"
                restored = SequenceInteractionMap(axis_labels=axis_labels.tolist(), values=values.tolist())
                assert restored.axis_labels == interaction_map.axis_labels
                np.testing.assert_allclose(restored.values, interaction_map.values, rtol=1e-6)


@pytest.mark.parametrize("channels", [["foo", "foo_axis_labels"], ["foo_axis_labels", "foo"]])
def test_sequence_interactions_npz_export_rejects_colliding_channel_names(channels, tmp_path):
    interaction_map = SequenceInteractionMap(axis_labels=["A"], values=[[0.5]])
    output = SequenceInteractionsOutput(results=[SequenceInteractions(maps=dict.fromkeys(channels, interaction_map))])
    with pytest.raises(ValueError, match=r"duplicate NPZ key: 0_foo_axis_labels"):
        output._export_output(tmp_path / "interactions", "npz")
    assert not (tmp_path / "interactions.npz").exists()


# ── Plotting ────────────────────────────────────────────────────────────────

# Forward gene "MK", intergenic "acgt", reverse gene "W"; positions 5-8 are DNA.
_LOCUS_LABELS = list("+MK+acgt-W")
_RED = np.array([0xDA, 0x38, 0x32]) / 255


@pytest.fixture
def plt():
    import matplotlib.pyplot as pyplot

    pyplot.switch_backend("Agg")
    yield pyplot
    pyplot.close("all")


def _locus_output(**channels):
    length = len(_LOCUS_LABELS)
    maps = {
        name: {"axis_labels": _LOCUS_LABELS, "values": np.full((length, length), value).tolist()}
        for name, value in channels.items()
    }
    return SequenceInteractionsOutput(results=[{"maps": maps}])


def test_plot_overlay_masks_by_modality_and_lets_base_pairing_win(plt):
    output = _locus_output(protein=0.9, repeat=0.9, base_pairing=0.6)
    _, ax = plt.subplots()
    figure = output.plot(ax=ax)
    rgb = ax.images[0].get_array()
    assert figure is ax.figure
    assert rgb.shape == (10, 10, 3)
    # Base pairing at 0.6 beats a stronger repeat signal between nucleotides.
    np.testing.assert_allclose(rgb[4, 6], 1 - (1 - _RED) * 0.6, atol=1e-6)
    # Protein contacts appear between amino acids but never between a residue and a base.
    assert not np.allclose(rgb[1, 2], 1.0)
    np.testing.assert_allclose(rgb[1, 5], 1.0)
    assert [patch.get_label() for patch in ax.get_legend().get_patches()] == [
        "Base pairing",
        "Repeats",
        "Protein contacts",
    ]


def test_plot_without_modality_mask_keeps_cross_modality_signal(plt):
    _, ax = plt.subplots()
    _locus_output(protein=0.9).plot(ax=ax, mask_by_modality=False, show_track=False)
    assert not np.allclose(ax.images[0].get_array()[1, 5], 1.0)


def test_plot_single_channel_shows_values_with_colorbar(plt):
    _, ax = plt.subplots()
    figure = _locus_output(base_pairing=0.6, protein=0.9).plot(channel="base_pairing", ax=ax)
    values = ax.images[0].get_array()
    assert values.shape == (10, 10)
    assert values[4, 6] == pytest.approx(0.6)
    # The modality mask zeroes base pairing involving amino acids.
    assert values[1, 2] == 0.0
    assert ax.get_title() == "Base pairing"
    assert any(axes.get_label() == "<colorbar>" for axes in figure.axes)


def test_plot_window_crops_to_one_indexed_inclusive_positions(plt):
    _, ax = plt.subplots()
    _locus_output(repeat=0.9).plot(window=(5, 8), ax=ax, show_track=False)
    assert ax.images[0].get_array().shape == (4, 4, 3)
    assert ax.get_xticklabels()[0].get_text() == "5"


def test_plot_single_channel_accepts_custom_channel_names(plt):
    output = SequenceInteractionsOutput(
        results=[{"maps": {"custom-channel.v1": {"axis_labels": ["1", "2", "3"], "values": _VALUES}}}]
    )
    _, ax = plt.subplots()
    output.plot(channel="custom-channel.v1", ax=ax, show_track=False)
    np.testing.assert_allclose(ax.images[0].get_array(), _VALUES)
    with pytest.raises(ValueError, match="No known channel to overlay"):
        output.plot()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"index": 1}, "index 1 is out of range"),
        ({"channel": "contacts"}, "Unknown channel 'contacts'"),
        ({"window": (0, 4)}, "window must satisfy"),
        ({"window": (6, 5)}, "window must satisfy"),
        ({"window": (1, 11)}, "window must satisfy"),
    ],
)
def test_plot_rejects_invalid_index_channel_or_window(plt, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _locus_output(protein=0.5).plot(**kwargs)

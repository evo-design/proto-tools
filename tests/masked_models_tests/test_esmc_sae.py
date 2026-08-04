"""tests/masked_models_tests/test_esmc_sae.py.

Tests for ESM C sparse autoencoder feature extraction.
"""

import json

import pytest
from pydantic import ValidationError

from proto_tools.tools.masked_models.esmc import (
    DESCRIBED_SAE_REPO,
    ESMCSAEFeaturesConfig,
    ESMCSAEFeaturesInput,
    describe_sae_features,
    resolve_sae_repo,
    run_esmc_sae_features,
)
from tests.conftest import benchmark_twice, make_persistent_fixture, random_protein_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("esmc")


# ── Repo resolution ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "expected_repo"),
    [
        # k=64 at a published codebook size serves any layer set.
        # A request for exactly the sweep layer gets the SAE trained for that layer.
        ({}, "biohub/ESMC-300M-sae-layer23-k64-codebook16384"),
        ({"layers": [0, 11, 23]}, "biohub/ESMC-300M-sae-k64-codebook16384"),
        ({"model_checkpoint": "esmc_600m"}, "biohub/ESMC-600M-sae-layer27-k64-codebook16384"),
        # 6B is the only backbone with an all-layer 131072 codebook.
        (
            {"model_checkpoint": "esmc_6b", "codebook_size": 131072},
            "biohub/ESMC-6B-sae-layer60-k64-codebook131072",
        ),
        # MLP-output SAEs are published only at 131072.
        # MLP-output SAEs have no single-layer family, so they always use the all-layer repo.
        (
            {"sae_target": "mlp_outputs", "codebook_size": 131072},
            "biohub/ESMC-300M-sae-mlp-k64-codebook131072",
        ),
        # Multiple layers can only come from the all-layer family.
        ({"layers": [11, 23]}, "biohub/ESMC-300M-sae-k64-codebook16384"),
        # Non-default k falls through to the single-layer sweep.
        ({"k": 256, "codebook_size": 65536}, "biohub/ESMC-300M-sae-layer23-k256-codebook65536"),
        # 300M has no all-layer 131072, but the sweep layer publishes one.
        ({"codebook_size": 131072}, "biohub/ESMC-300M-sae-layer23-k64-codebook131072"),
    ],
)
def test_config_resolves_published_sae_repo(kwargs, expected_repo):
    """Each accepted config maps to the repo that actually publishes that SAE."""
    assert ESMCSAEFeaturesConfig(**kwargs).sae_repo == expected_repo


@pytest.mark.parametrize(
    "kwargs",
    [
        # Sweeping k requires the sweep layer, not an arbitrary one.
        {"k": 256, "codebook_size": 65536, "layers": [11]},
        # No MLP-output SAE at 16384, and no MLP sweep at all.
        {"sae_target": "mlp_outputs"},
        {"sae_target": "mlp_outputs", "k": 256, "codebook_size": 65536},
        # Layer beyond the backbone's depth, and duplicates.
        {"layers": [99]},
        {"layers": [11, 11]},
    ],
)
def test_config_rejects_unpublished_combinations(kwargs):
    """Combinations Biohub never published are rejected at construction."""
    with pytest.raises(ValidationError):
        ESMCSAEFeaturesConfig(**kwargs)


def test_resolve_sae_repo_error_names_alternatives():
    """The rejection message tells the caller which combinations do exist."""
    with pytest.raises(ValueError, match=r"k=64.*codebook_size in \[16384\].*layers=\[23\]"):
        resolve_sae_repo("esmc_300m", "hidden_states", [11], 256, 65536)


def test_layers_default_to_backbone_sweep_layer():
    """``layers=None`` resolves to the ~75%-depth layer for each backbone."""
    assert ESMCSAEFeaturesConfig(model_checkpoint="esmc_300m").resolved_layers == [23]
    assert ESMCSAEFeaturesConfig(model_checkpoint="esmc_600m").resolved_layers == [27]
    assert ESMCSAEFeaturesConfig(model_checkpoint="esmc_6b").resolved_layers == [60]


def test_layers_are_sorted():
    """Requested layers are normalized to ascending order."""
    assert ESMCSAEFeaturesConfig(layers=[23, 0, 11]).resolved_layers == [0, 11, 23]


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_esmc_sae_features_align_with_residues():
    """Features are returned per residue, per layer, with exactly k active each.

    The SAE stack emits one row per unpadded token concatenated across the batch,
    so a batch of unequal-length sequences is the case that catches mis-splitting.
    """
    sequences = ["MKTAYIAKQRQISFVKSHFSRQ", "MVLSPADKTNVKAAW"]
    config = ESMCSAEFeaturesConfig(layers=[11, 23], batch_size=2)

    result = run_esmc_sae_features(inputs=ESMCSAEFeaturesInput(sequences=sequences), config=config)
    validate_output(result)

    assert result.tool_id == "esmc-sae-features"
    assert len(result.results) == len(sequences)

    for sequence, per_sequence in zip(sequences, result.results, strict=True):
        assert [layer.layer for layer in per_sequence.layers] == [11, 23]
        for layer in per_sequence.layers:
            assert len(layer.feature_indices) == len(sequence)
            assert len(layer.feature_magnitudes) == len(sequence)
            assert {len(row) for row in layer.feature_indices} == {config.k}
            # Magnitudes arrive ordered so callers can take the strongest features.
            for row in layer.feature_magnitudes:
                assert row == sorted(row, reverse=True)


@pytest.mark.uses_gpu
def test_esmc_sae_distinct_sequences_produce_distinct_features():
    """Unrelated sequences do not collapse to the same active features."""
    inputs = ESMCSAEFeaturesInput(sequences=["MVLSPADKTNVKAAW", "AAAAAAAAAAAAAAA"])
    result = run_esmc_sae_features(inputs=inputs, config=ESMCSAEFeaturesConfig())

    first = result.results[0].layers[0].feature_indices
    second = result.results[1].layers[0].feature_indices
    assert any(a != b for a, b in zip(first, second, strict=True))


@pytest.mark.uses_gpu
def test_esmc_sae_export_writes_every_active_feature(tmp_path):
    """Export writes per-residue features to both supported formats."""
    sequence = "MKTAYIAKQR"
    inputs = ESMCSAEFeaturesInput(sequences=[sequence])
    config = ESMCSAEFeaturesConfig()
    result = run_esmc_sae_features(inputs=inputs, config=config)

    result.export(name="features", export_path=tmp_path, file_format="json")
    result.export(name="features", export_path=tmp_path, file_format="csv")

    exported = json.loads((tmp_path / "features.json").read_text())
    assert exported[0]["layers"][0]["layer"] == 23
    assert exported[0]["sequence"] == sequence

    rows = (tmp_path / "features.csv").read_text().splitlines()
    # One header plus one row per (residue, active feature).
    assert len(rows) == 1 + len(sequence) * config.k
    assert rows[0] == "sequence_index,layer,position,residue,feature_index,magnitude"

    # position is 1-indexed and residue names the amino acid at that position.
    import csv

    parsed = list(csv.DictReader(rows))
    assert parsed[0]["position"] == "1"
    assert parsed[0]["residue"] == sequence[0]
    last = parsed[-1]
    assert last["position"] == str(len(sequence))
    assert last["residue"] == sequence[-1]


# ── Benchmarks ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("esmc-sae-features")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmc_sae_features_benchmark(request):
    """Benchmark esmc-sae-features on 50 sequences of length 200 (cold + warm)."""
    sequences = random_protein_sequences(n=50, length=200, seed=0)
    inputs = ESMCSAEFeaturesInput(sequences=sequences)
    config = ESMCSAEFeaturesConfig(batch_size=8)

    result = benchmark_twice(request, "esmc", lambda: run_esmc_sae_features(inputs=inputs, config=config))

    assert result.tool_id == "esmc-sae-features"
    assert len(result.results) == 50
    assert len(result.results[0].layers[0].feature_indices) == 200


# ── Backbone comparison (exploratory) ─────────────────────────────────────────


def _top_k_overlap(a, b) -> float:
    """Mean fraction of shared active features per residue between two results."""
    per_residue = [
        len(set(x) & set(y)) / len(x)
        for layer_a, layer_b in zip(a.layers, b.layers, strict=True)
        for x, y in zip(layer_a.feature_indices, layer_b.feature_indices, strict=True)
    ]
    return sum(per_residue) / len(per_residue)


@pytest.mark.uses_gpu
def test_esm_backbone_layer_offset():
    """The esm stack omits the embedding output, so layer{N} reads hidden_states[N-1].

    Pinning this guards the one assumption the esm backbone path rests on. If upstream
    reindexes ``hidden_states``, agreement collapses and this fails rather than silently
    returning features from the wrong layer.
    """
    inputs = ESMCSAEFeaturesInput(sequences=["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQ"])
    config = ESMCSAEFeaturesConfig(layers=[23])

    reference = run_esmc_sae_features(inputs=inputs, config=config)
    candidate = run_esmc_sae_features(inputs=inputs, config=ESMCSAEFeaturesConfig(layers=[23], backbone="esm"))

    overlap = _top_k_overlap(reference.results[0], candidate.results[0])
    assert overlap > 0.95, (
        f"esm backbone agrees on only {overlap:.3f} of active features; the "
        f"hidden_states[N-1] offset may no longer hold"
    )


@pytest.mark.uses_gpu
def test_esm_backbone_returns_same_shape_as_transformers():
    """Both backbones produce identically shaped per-residue features."""
    sequences = ["MKTAYIAKQRQISFVKSHFSRQ", "MVLSPADKTNVKAAW"]
    inputs = ESMCSAEFeaturesInput(sequences=sequences)

    for backbone in ("transformers", "esm"):
        result = run_esmc_sae_features(
            inputs=inputs, config=ESMCSAEFeaturesConfig(layers=[11, 23], batch_size=2, backbone=backbone)
        )
        assert len(result.results) == len(sequences), backbone
        for sequence, per_sequence in zip(sequences, result.results, strict=True):
            for layer in per_sequence.layers:
                assert len(layer.feature_indices) == len(sequence), f"{backbone} layer{layer.layer}"


# ── MLP-output SAEs ───────────────────────────────────────────────────────────


def test_estimated_download_tracks_codebook_and_layer_count():
    """The offline size estimate matches the published per-layer file sizes.

    Pinned against real repo sizes so the download warning stays meaningful: a
    hidden-state 300M layer is 0.13 GB, an MLP-output layer 1.01 GB, and a 6B
    MLP-output layer 2.69 GB.
    """
    hidden = ESMCSAEFeaturesConfig()
    assert round(hidden.estimated_download_gb, 2) == 0.13

    mlp = ESMCSAEFeaturesConfig(sae_target="mlp_outputs", codebook_size=131072)
    assert round(mlp.estimated_download_gb, 2) == 1.01

    # Scales linearly with the number of requested layers.
    mlp_three = ESMCSAEFeaturesConfig(sae_target="mlp_outputs", codebook_size=131072, layers=[5, 15, 23])
    assert mlp_three.estimated_download_gb == pytest.approx(3 * mlp.estimated_download_gb)


@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmc_sae_mlp_outputs_features_align_with_residues():
    """MLP-output SAEs run end to end and return per-residue features like hidden states.

    Separate from the hidden-state tests because this family is published only at a
    131072 codebook, so each layer file is ~1 GB.
    """
    sequence = "MKTAYIAKQRQISFVKSHFSRQ"
    config = ESMCSAEFeaturesConfig(sae_target="mlp_outputs", codebook_size=131072)
    assert config.sae_repo == "biohub/ESMC-300M-sae-mlp-k64-codebook131072"

    result = run_esmc_sae_features(inputs=ESMCSAEFeaturesInput(sequences=[sequence]), config=config)
    validate_output(result)

    layer = result.results[0].layers[0]
    assert layer.layer == 23
    assert len(layer.feature_indices) == len(sequence)
    assert {len(row) for row in layer.feature_indices} == {config.k}
    # Indices address the wider codebook this family uses.
    assert max(max(row) for row in layer.feature_indices) < config.codebook_size


# ── Feature descriptions ──────────────────────────────────────────────────────


def test_described_sae_is_reachable_from_config():
    """The 6B defaults resolve to the one SAE whose features have descriptions.

    The all-layer and single-layer 6B SAEs are independently trained, so an index from
    the wrong one resolves to an unrelated concept. This pins that the described repo is
    what a user actually gets.
    """
    assert ESMCSAEFeaturesConfig(model_checkpoint="esmc_6b").sae_repo == DESCRIBED_SAE_REPO


def test_describe_sae_features_rejects_out_of_codebook_indices():
    """Indices outside the described SAE's codebook cannot be described."""
    with pytest.raises(ValueError, match="16384-feature codebook"):
        describe_sae_features([16384])


@pytest.mark.integration
def test_describe_sae_features_returns_labels_and_statistics():
    """Descriptions carry a label plus the statistics needed to normalize activations."""
    described = describe_sae_features([3995, 1251])

    assert set(described) == {3995, 1251}
    assert "Ubiquitin-like" in described[3995]["label"]
    for record in described.values():
        # Both are required to compute (activation / max) * idf.
        assert record["uniref90_max_activation"] > 0
        assert record["uniref90_idf"] >= 0

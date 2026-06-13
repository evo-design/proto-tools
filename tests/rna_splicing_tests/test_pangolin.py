"""tests/rna_splicing_tests/test_pangolin.py.

Tests for Pangolin splice-site prediction and variant scoring.
"""

import pytest

from proto_tools.tools.rna_splicing.pangolin import (
    PangolinPredictConfig,
    PangolinPredictInput,
    PangolinScoreVariantsConfig,
    PangolinScoreVariantsInput,
    PangolinVariant,
    run_pangolin_predict,
    run_pangolin_score_variants,
)
from proto_tools.tools.rna_splicing.pangolin.shared_data_models import PANGOLIN_FLANK
from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_predict_and_check(device: str) -> None:
    """Run Pangolin prediction on one sequence and verify output shape."""
    seq = random_dna_sequences(n=1, length=10100, seed=0)[0]
    result = run_pangolin_predict(
        PangolinPredictInput(sequences=[seq]),
        PangolinPredictConfig(device=device),
    )

    assert result.success is True, f"Pangolin predict failed: {result}"
    assert result.tool_id == "pangolin-predict"
    assert len(result.results) == 1
    prediction = result.results[0]
    assert len(prediction.scores) == 10100 - 2 * PANGOLIN_FLANK
    assert all(len(row) == 4 for row in prediction.scores)
    assert prediction.output_start == PANGOLIN_FLANK


def _run_score_variants_and_check(device: str) -> None:
    """Run Pangolin variant scoring on one SNV and verify scores and metrics."""
    seq = random_dna_sequences(n=1, length=10200, seed=0)[0]
    pos = 5100
    ref = seq[pos]
    alt = "A" if ref != "A" else "C"
    variant = PangolinVariant(
        sequence=seq,
        variant_position=pos,
        reference_bases=ref,
        alternate_bases=alt,
    )
    result = run_pangolin_score_variants(
        PangolinScoreVariantsInput(variants=[variant]),
        PangolinScoreVariantsConfig(device=device, distance=50),
    )

    assert result.success is True, f"Pangolin score-variants failed: {result}"
    assert result.tool_id == "pangolin-score-variants"
    assert len(result.results) == 1
    effect = result.results[0]
    assert len(effect.gain_scores) == len(effect.loss_scores)
    assert len(effect.gain_scores) == 2 * 50 + 1
    assert isinstance(effect.increase_score, float)

    assert_metrics_in_spec(result)


# ── Validator coverage ─────────────────────────────────────────────────────────


def test_pangolin_predict_normalizes_single_sequence() -> None:
    """A single sequence string is normalized to a one-item list."""
    seq = "ACGT" * 2501  # 10,004 bp
    inp = PangolinPredictInput(sequences=seq)
    assert inp.sequences == [seq]


def test_pangolin_predict_rejects_short_sequence() -> None:
    """A sequence below 2*PANGOLIN_FLANK+1 bp is rejected pre-dispatch."""
    short = "ACGT" * 100  # 400 bp, far below the minimum
    with pytest.raises(ValueError, match="minimum"):
        PangolinPredictInput(sequences=[short])


def test_pangolin_variant_rejects_allele_mismatch() -> None:
    """A reference allele that disagrees with the window is rejected."""
    seq = random_dna_sequences(n=1, length=10200, seed=1)[0]
    pos = 5100
    correct = seq[pos]
    wrong = "A" if correct != "A" else "C"
    with pytest.raises(ValueError, match="does not match"):
        PangolinVariant(
            sequence=seq,
            variant_position=pos,
            reference_bases=wrong,
            alternate_bases="G",
        )


def test_pangolin_variant_rejects_insufficient_flank() -> None:
    """A variant with less than PANGOLIN_FLANK bp of 5' flank is rejected."""
    seq = random_dna_sequences(n=1, length=10200, seed=2)[0]
    pos = 10
    ref = seq[pos]
    alt = "A" if ref != "A" else "C"
    with pytest.raises(ValueError, match="flank"):
        PangolinVariant(
            sequence=seq,
            variant_position=pos,
            reference_bases=ref,
            alternate_bases=alt,
        )


def test_pangolin_variant_rejects_unsupported_indel() -> None:
    """A complex indel (both alleles >1 bp and unequal length) is rejected."""
    seq = random_dna_sequences(n=1, length=10200, seed=4)[0]
    pos = 5100
    ref = seq[pos : pos + 2]  # length 2
    alt = ref + "GG"  # length 4 — both alleles > 1 bp and unequal length
    with pytest.raises(ValueError, match="unsupported variant format"):
        PangolinVariant(
            sequence=seq,
            variant_position=pos,
            reference_bases=ref,
            alternate_bases=alt,
        )


def test_pangolin_variant_rejects_all_n_allele() -> None:
    """An allele with no A/C/G/T base (e.g. 'N') is rejected (upstream skips these)."""
    seq = random_dna_sequences(n=1, length=10200, seed=5)[0]
    pos = 5100
    with pytest.raises(ValueError, match="at least one"):
        PangolinVariant(sequence=seq, variant_position=pos, reference_bases=seq[pos], alternate_bases="N")


def test_pangolin_predict_rejects_rna_sequence() -> None:
    """A sequence containing U (RNA) is rejected; Pangolin scores DNA only."""
    rna = "U" + "ACGT" * 2600  # 10,401 bp, contains a U
    with pytest.raises(ValueError, match="Invalid nucleotide"):
        PangolinPredictInput(sequences=[rna])


def test_pangolin_variant_rejects_3prime_flank() -> None:
    """A variant with less than PANGOLIN_FLANK bp of 3' flank is rejected."""
    seq = random_dna_sequences(n=1, length=10200, seed=7)[0]
    pos = len(seq) - 10  # only 9 bp of 3' flank
    ref = seq[pos]
    alt = "A" if ref != "A" else "C"
    with pytest.raises(ValueError, match="3' flank"):
        PangolinVariant(sequence=seq, variant_position=pos, reference_bases=ref, alternate_bases=alt)


def test_pangolin_variant_accepts_simple_indels() -> None:
    """Equal-length MNV and a 1-to-many insertion are accepted (supported formats)."""
    seq = random_dna_sequences(n=1, length=10200, seed=8)[0]
    pos = 5100
    mnv = PangolinVariant(sequence=seq, variant_position=pos, reference_bases=seq[pos : pos + 2], alternate_bases="AC")
    insertion = PangolinVariant(sequence=seq, variant_position=pos, reference_bases=seq[pos], alternate_bases="ACG")
    assert len(mnv.reference_bases) == 2 and len(insertion.alternate_bases) == 3


def test_pangolin_variant_normalizes_single_variant() -> None:
    """A single PangolinVariant is normalized to a one-item list."""
    seq = random_dna_sequences(n=1, length=10200, seed=3)[0]
    pos = 5100
    ref = seq[pos]
    alt = "A" if ref != "A" else "C"
    variant = PangolinVariant(
        sequence=seq,
        variant_position=pos,
        reference_bases=ref,
        alternate_bases=alt,
    )
    inp = PangolinScoreVariantsInput(variants=variant)
    assert inp.variants == [variant]


# ── Integration tests ──────────────────────────────────────────────────────────


@pytest.mark.integration
def test_pangolin_predict_cpu() -> None:
    """Test Pangolin prediction on CPU."""
    _run_predict_and_check(device="cpu")


@pytest.mark.uses_gpu
def test_pangolin_predict_gpu() -> None:
    """Test Pangolin prediction on GPU."""
    _run_predict_and_check(device="cuda")


@pytest.mark.integration
def test_pangolin_score_variants_cpu() -> None:
    """Test Pangolin variant scoring on CPU."""
    _run_score_variants_and_check(device="cpu")


@pytest.mark.uses_gpu
def test_pangolin_score_variants_gpu() -> None:
    """Test Pangolin variant scoring on GPU."""
    _run_score_variants_and_check(device="cuda")


@pytest.mark.integration
def test_pangolin_predict_tissue_subset() -> None:
    """A tissue subset yields one score column per requested tissue, in order."""
    seq = random_dna_sequences(n=1, length=10100, seed=0)[0]
    tissues = ["BRAIN", "HEART"]
    result = run_pangolin_predict(
        PangolinPredictInput(sequences=[seq]),
        PangolinPredictConfig(device="cpu", tissues=tissues),
    )

    assert result.success is True, f"Pangolin predict failed: {result}"
    prediction = result.results[0]
    assert prediction.tissues == tissues
    assert all(len(row) == len(tissues) for row in prediction.scores)


@pytest.mark.integration
def test_pangolin_predict_batch() -> None:
    """Multiple sequences yield one result each (1:1 batch cardinality)."""
    seqs = random_dna_sequences(n=2, length=10100, seed=0)
    result = run_pangolin_predict(PangolinPredictInput(sequences=seqs), PangolinPredictConfig(device="cpu"))

    assert result.success is True, f"Pangolin predict failed: {result}"
    assert len(result.results) == 2
    assert all(len(r.scores) == 10100 - 2 * PANGOLIN_FLANK for r in result.results)


@pytest.mark.integration
def test_pangolin_score_variants_rejects_large_deletion() -> None:
    """A deletion wider than 2*distance is rejected before dispatch (matches upstream)."""
    seq = random_dna_sequences(n=1, length=10400, seed=9)[0]
    pos = 5100
    ref = seq[pos : pos + 150]  # 150 bp simple deletion (> 2*50)
    variant = PangolinVariant(sequence=seq, variant_position=pos, reference_bases=ref, alternate_bases=ref[0])
    with pytest.raises(ValueError, match="exceeds 2"):
        run_pangolin_score_variants(
            PangolinScoreVariantsInput(variants=[variant]),
            PangolinScoreVariantsConfig(device="cpu", distance=50),
        )


@pytest.mark.integration
def test_pangolin_score_variants_minus_strand() -> None:
    """Scoring on the '-' strand exercises the reverse-complement and score-reversal paths."""
    seq = random_dna_sequences(n=1, length=10200, seed=11)[0]
    pos = 5100
    ref = seq[pos]
    alt = "A" if ref != "A" else "C"
    variant = PangolinVariant(sequence=seq, variant_position=pos, reference_bases=ref, alternate_bases=alt, strand="-")
    result = run_pangolin_score_variants(
        PangolinScoreVariantsInput(variants=[variant]),
        PangolinScoreVariantsConfig(device="cpu", distance=50),
    )
    assert result.success is True, f"Pangolin score-variants failed: {result}"
    assert len(result.results[0].gain_scores) == 2 * 50 + 1


@pytest.mark.integration
def test_pangolin_score_variants_deletion() -> None:
    """A simple deletion is scored end-to-end (exercises the indel alignment in _compute_score)."""
    seq = random_dna_sequences(n=1, length=10300, seed=12)[0]
    pos = 5100
    ref = seq[pos : pos + 3]  # 3 bp deletion
    variant = PangolinVariant(sequence=seq, variant_position=pos, reference_bases=ref, alternate_bases=ref[0])
    result = run_pangolin_score_variants(
        PangolinScoreVariantsInput(variants=[variant]),
        PangolinScoreVariantsConfig(device="cpu", distance=50),
    )
    assert result.success is True, f"Pangolin score-variants failed: {result}"
    # Deletion score arrays span len(ref) + 2*distance positions.
    assert len(result.results[0].gain_scores) == len(ref) + 2 * 50


@pytest.mark.integration
def test_pangolin_score_variants_batch() -> None:
    """Multiple variants yield one result each (1:1 batch cardinality)."""
    seq = random_dna_sequences(n=1, length=10200, seed=13)[0]
    variants = [
        PangolinVariant(
            sequence=seq,
            variant_position=p,
            reference_bases=seq[p],
            alternate_bases="A" if seq[p] != "A" else "C",
        )
        for p in (5100, 5150)
    ]
    result = run_pangolin_score_variants(
        PangolinScoreVariantsInput(variants=variants),
        PangolinScoreVariantsConfig(device="cpu", distance=50),
    )
    assert result.success is True, f"Pangolin score-variants failed: {result}"
    assert len(result.results) == 2


# ── Export ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_pangolin_predict_export(tmp_path) -> None:
    """Prediction output exports to json and npy files."""
    seq = random_dna_sequences(n=1, length=10100, seed=0)[0]
    result = run_pangolin_predict(PangolinPredictInput(sequences=[seq]), PangolinPredictConfig(device="cpu"))
    assert result.success is True, f"Pangolin predict failed: {result}"

    out_dir = tmp_path / "out"
    for file_format in ("json", "npy"):
        result.export("pangolin", export_path=out_dir, file_format=file_format)
        path = out_dir / f"pangolin.{file_format}"
        assert path.exists() and path.stat().st_size > 0


@pytest.mark.integration
def test_pangolin_predict_export_ragged_npy(tmp_path) -> None:
    """A ragged batch (sequences of differing length) exports to npy as a 1-D object array."""
    import numpy as np

    seqs = [
        random_dna_sequences(n=1, length=10100, seed=0)[0],
        random_dna_sequences(n=1, length=10200, seed=1)[0],
    ]
    result = run_pangolin_predict(PangolinPredictInput(sequences=seqs), PangolinPredictConfig(device="cpu"))
    assert result.success is True, f"Pangolin predict failed: {result}"

    out_dir = tmp_path / "out"
    result.export("ragged", export_path=out_dir, file_format="npy")
    loaded = np.load(out_dir / "ragged.npy", allow_pickle=True)
    assert loaded.shape == (2,)
    assert loaded[0].shape == (10100 - 2 * PANGOLIN_FLANK, 4)
    assert loaded[1].shape == (10200 - 2 * PANGOLIN_FLANK, 4)


@pytest.mark.integration
def test_pangolin_score_variants_export(tmp_path) -> None:
    """Variant-scoring output exports to both json and csv files."""
    seq = random_dna_sequences(n=1, length=10200, seed=0)[0]
    pos = 5100
    ref = seq[pos]
    alt = "A" if ref != "A" else "C"
    variant = PangolinVariant(
        sequence=seq,
        variant_position=pos,
        reference_bases=ref,
        alternate_bases=alt,
    )
    result = run_pangolin_score_variants(
        PangolinScoreVariantsInput(variants=[variant]),
        PangolinScoreVariantsConfig(device="cpu", distance=50),
    )
    assert result.success is True, f"Pangolin score-variants failed: {result}"

    out_dir = tmp_path / "out"
    for file_format in ("json", "csv"):
        result.export("pangolin", export_path=out_dir, file_format=file_format)
        path = out_dir / f"pangolin.{file_format}"
        assert path.exists() and path.stat().st_size > 0
    header = (out_dir / "pangolin.csv").read_text().splitlines()[0]
    assert "increase_position" in header and "max_gain" in header


# ── Benchmark ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("pangolin-predict")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_pangolin_predict_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pangolin-predict on 64 sequences of length 12000 bp across 4 tissues (cold + warm)."""
    n = 64
    length = 12000
    sequences = random_dna_sequences(n=n, length=length, seed=0)
    inputs = PangolinPredictInput(sequences=sequences)
    config = PangolinPredictConfig(device="cuda")

    result = benchmark_twice(request, "pangolin", lambda: run_pangolin_predict(inputs, config))

    assert result.success is True, f"Pangolin predict failed: {result}"
    assert result.tool_id == "pangolin-predict"
    assert len(result.results) == n
    expected_positions = length - 2 * PANGOLIN_FLANK
    for prediction in result.results:
        assert len(prediction.scores) == expected_positions
        assert all(len(row) == 4 for row in prediction.scores)
        assert prediction.output_start == PANGOLIN_FLANK


@pytest.mark.benchmark("pangolin-score-variants")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_pangolin_score_variants_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pangolin-score-variants on 64 SNVs in 10,200 bp windows (cold + warm)."""
    seqs = random_dna_sequences(n=64, length=10200, seed=42)
    distance = 50
    variants = []
    for i, seq in enumerate(seqs):
        pos = 5100 - len(seqs) // 2 + i
        ref = seq[pos]
        alt = "A" if ref != "A" else "C"
        variants.append(
            PangolinVariant(
                sequence=seq,
                variant_position=pos,
                reference_bases=ref,
                alternate_bases=alt,
            )
        )
    inputs = PangolinScoreVariantsInput(variants=variants)
    config = PangolinScoreVariantsConfig(device="cuda", distance=distance)

    result = benchmark_twice(request, "pangolin", lambda: run_pangolin_score_variants(inputs, config))

    assert result.success is True, f"Pangolin score-variants failed: {result}"
    assert result.tool_id == "pangolin-score-variants"
    assert len(result.results) == 64
    for effect in result.results:
        assert len(effect.gain_scores) == 2 * distance + 1
        assert len(effect.gain_scores) == len(effect.loss_scores)
        assert isinstance(effect.increase_score, float)
    assert_metrics_in_spec(result)

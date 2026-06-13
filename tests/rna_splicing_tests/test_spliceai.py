"""Tests for SpliceAI variant scoring and raw splice-site prediction."""

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

from proto_tools.tools import (
    SpliceAIGeneScore,
    SpliceAIPredictConfig,
    SpliceAIPredictInput,
    SpliceAIPredictOutput,
    SpliceAIScoreConfig,
    SpliceAIScoreInput,
    SpliceAIScoreMetrics,
    SpliceAIScoreOutput,
    SpliceAIVariant,
    SpliceAIVariantResult,
    run_spliceai_predict,
    run_spliceai_score,
)
from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "proto_tools/tools/rna_splicing/spliceai/examples"


def _variant_result() -> SpliceAIVariantResult:
    """One SpliceAIVariantResult with a single gene score (built without a tool run)."""
    return SpliceAIVariantResult(
        chromosome="1",
        position=30000,
        ref="A",
        alt="C",
        scores=[
            SpliceAIGeneScore(
                allele="C",
                symbol="SPLICEAI_DEMO",
                ds_ag=0.91,
                ds_al=0.02,
                ds_dg=0.05,
                ds_dl=0.01,
                dp_ag=-12,
                dp_al=34,
                dp_dg=7,
                dp_dl=-3,
            )
        ],
        metrics=SpliceAIScoreMetrics(max_delta_score=0.91),
    )


# ── Validators ────────────────────────────────────────────────────────────────


def test_score_input_wraps_single_variant() -> None:
    """A bare SpliceAIVariant is normalized to a 1-element list (normalize_variants)."""
    variant = SpliceAIVariant(chromosome="1", position=30000, ref="A", alt="C")
    assert SpliceAIScoreInput(variants=variant).variants == [variant]


def test_variant_allele_validation() -> None:
    """validate_allele_bases uppercases DNA and rejects non-ACGTN characters."""
    variant = SpliceAIVariant(chromosome="1", position=30000, ref="a", alt="c")
    assert (variant.ref, variant.alt) == ("A", "C")
    with pytest.raises(ValueError, match="A/C/G/T/N"):
        SpliceAIVariant(chromosome="1", position=30000, ref="AX", alt="C")


def test_predict_input_wraps_single_sequence() -> None:
    """A bare sequence string is normalized to a 1-element list (normalize_sequences)."""
    assert SpliceAIPredictInput(sequences="ACGT").sequences == ["ACGT"]


# ── Export (custom serialization only) ──────────────────────────────────────


def test_score_export_vcf(tmp_path: Path) -> None:
    """VCF export emits the canonical SpliceAI INFO header + per-variant annotation."""
    SpliceAIScoreOutput(results=[_variant_result()]).export("scores", tmp_path, file_format="vcf")
    text = (tmp_path / "scores.vcf").read_text()
    assert "##INFO=<ID=SpliceAI" in text
    assert "SpliceAI=C|SPLICEAI_DEMO|0.91|0.02|0.05|0.01|-12|34|7|-3" in text


def test_score_export_vcf_unscored_mnv(tmp_path: Path) -> None:
    """A complex MNV (None scores, which SpliceAI emits as '.') renders as '.' in the VCF INFO."""
    gene = SpliceAIGeneScore(
        allele="GT",
        symbol="DEMO",
        ds_ag=None,
        ds_al=None,
        ds_dg=None,
        ds_dl=None,
        dp_ag=None,
        dp_al=None,
        dp_dg=None,
        dp_dl=None,
    )
    result = SpliceAIVariantResult(
        chromosome="1", position=30000, ref="AC", alt="GT", scores=[gene], metrics=SpliceAIScoreMetrics()
    )
    SpliceAIScoreOutput(results=[result]).export("scores", tmp_path, file_format="vcf")
    assert "SpliceAI=GT|DEMO|.|.|.|.|.|.|.|." in (tmp_path / "scores.vcf").read_text()


def test_score_export_csv_flattens_gene_rows(tmp_path: Path) -> None:
    """CSV export flattens one row per (variant, gene) with the score columns."""
    SpliceAIScoreOutput(results=[_variant_result()]).export("scores", tmp_path, file_format="csv")
    rows = list(csv.DictReader((tmp_path / "scores.csv").open()))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "SPLICEAI_DEMO"
    assert {"chromosome", "position", "ref", "alt", "ds_ag", "dp_dl"} <= rows[0].keys()


def test_predict_export_npy_ragged(tmp_path: Path) -> None:
    """Ragged-batch npy export round-trips (sequences of differing lengths)."""
    preds = [
        [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]],
        [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.1]],
    ]
    SpliceAIPredictOutput(predictions=preds).export("preds", tmp_path, file_format="npy")
    loaded = np.load(tmp_path / "preds.npy", allow_pickle=True)
    assert len(loaded) == 2
    assert len(loaded[0]) == 2 and len(loaded[1]) == 3


# ── Integration ───────────────────────────────────────────────────────────────


def _run_score(tmp_path: Path, device: str) -> None:
    """Score a scorable SNV and a complex MNV; verify real scores vs null (no crash)."""
    # Copy into tmp_path so pyfaidx writes its .fai index alongside the copy,
    # not into the committed source tree.
    genome = tmp_path / "example_genome.fa"
    annotation = tmp_path / "example_annotation.txt"
    shutil.copy(_EXAMPLES_DIR / "example_genome.fa", genome)
    shutil.copy(_EXAMPLES_DIR / "example_annotation.txt", annotation)

    # A complex MNV (multi-base ref AND alt) hits SpliceAI's '.'-null path; its
    # ref must match the genome, so read the two bases at the locus.
    gseq = "".join(ln.strip() for ln in genome.read_text().splitlines() if not ln.startswith(">"))
    mnv_ref = gseq[29999:30001]  # 1-based positions 30000-30001
    mnv_alt = "TT" if mnv_ref != "TT" else "GG"

    result = run_spliceai_score(
        SpliceAIScoreInput(
            variants=[
                SpliceAIVariant(chromosome="1", position=30000, ref="A", alt="C"),
                SpliceAIVariant(chromosome="1", position=30000, ref=mnv_ref, alt=mnv_alt),
            ]
        ),
        SpliceAIScoreConfig(reference_fasta=str(genome), annotation=str(annotation), device=device),
    )

    assert result.success is True, f"SpliceAI score failed: {result}"
    assert result.tool_id == "spliceai-score"
    assert len(result.results) == 2

    # SNV → real per-gene scores.
    snv = result.results[0]
    assert snv.scores, "Expected non-empty scores: the variant must overlap the synthetic gene"
    gene = snv.scores[0]
    assert gene.symbol == "SPLICEAI_DEMO"
    assert all(0.0 <= ds <= 1.0 for ds in (gene.ds_ag, gene.ds_al, gene.ds_dg, gene.ds_dl))
    assert all(isinstance(dp, int) for dp in (gene.dp_ag, gene.dp_al, gene.dp_dg, gene.dp_dl))
    assert_metrics_in_spec(result)
    assert 0.0 <= snv.metrics["max_delta_score"] <= 1.0

    # Complex MNV → SpliceAI emits '.', parsed to None; no crash, no max_delta.
    mnv = result.results[1]
    assert mnv.scores and mnv.scores[0].ds_ag is None
    assert "max_delta_score" not in mnv.metrics


def _run_predict(device: str) -> None:
    """Predict on a 400 bp sequence and verify the per-position probability shape."""
    result = run_spliceai_predict(
        SpliceAIPredictInput(sequences=["ACGT" * 100]),
        SpliceAIPredictConfig(device=device),
    )
    assert result.success is True, f"SpliceAI predict failed: {result}"
    assert result.tool_id == "spliceai-predict"
    assert len(result.predictions) == 1
    assert len(result.predictions[0]) == 400
    assert all(len(pos) == 3 and all(0.0 <= p <= 1.0 for p in pos) for pos in result.predictions[0])


@pytest.mark.integration
def test_spliceai_score_cpu(tmp_path: Path) -> None:
    """SpliceAI variant scoring on CPU."""
    _run_score(tmp_path, "cpu")


@pytest.mark.uses_gpu
def test_spliceai_score_gpu(tmp_path: Path) -> None:
    """SpliceAI variant scoring on GPU."""
    _run_score(tmp_path, "cuda")


@pytest.mark.integration
def test_spliceai_predict_cpu() -> None:
    """SpliceAI raw prediction on CPU."""
    _run_predict("cpu")


@pytest.mark.uses_gpu
def test_spliceai_predict_gpu() -> None:
    """SpliceAI raw prediction on GPU."""
    _run_predict("cuda")


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("spliceai-predict")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_spliceai_predict_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark spliceai-predict on 100 transcript-length (10 kb) sequences (cold + warm)."""
    n, length = 100, 10000
    sequences = random_dna_sequences(n=n, length=length, seed=0)
    inputs = SpliceAIPredictInput(sequences=sequences)
    config = SpliceAIPredictConfig(device="cuda")

    result = benchmark_twice(request, "spliceai", lambda: run_spliceai_predict(inputs, config))

    assert result.success is True, f"SpliceAI predict failed: {result}"
    assert result.tool_id == "spliceai-predict"
    assert len(result.predictions) == n, "Should have one prediction track per input sequence"
    for seq_pred in result.predictions:
        assert len(seq_pred) == length, "Per-position predictions should match input length"
        assert all(len(pos) == 3 and all(0.0 <= p <= 1.0 for p in pos) for pos in seq_pred)


@pytest.mark.benchmark("spliceai-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_spliceai_score_benchmark(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """Benchmark spliceai-score on 1000 SNVs spanning the demo gene (cold + warm)."""
    # Copy into tmp_path so pyfaidx writes its .fai index next to the copy, not the committed source tree.
    genome = tmp_path / "example_genome.fa"
    annotation = tmp_path / "example_annotation.txt"
    shutil.copy(_EXAMPLES_DIR / "example_genome.fa", genome)
    shutil.copy(_EXAMPLES_DIR / "example_annotation.txt", annotation)

    gseq = "".join(ln.strip() for ln in genome.read_text().splitlines() if not ln.startswith(">")).upper()
    n = 1000
    alt_map = {"A": "C", "C": "G", "G": "T", "T": "A"}
    margin = 1000
    lo, hi = 20000 + margin, 40000 - margin
    positions = [lo + i * ((hi - lo) // n) for i in range(n)]
    variants = [
        SpliceAIVariant(chromosome="1", position=p, ref=gseq[p - 1], alt=alt_map[gseq[p - 1]]) for p in positions
    ]

    inputs = SpliceAIScoreInput(variants=variants)
    config = SpliceAIScoreConfig(reference_fasta=str(genome), annotation=str(annotation), device="cuda")

    result = benchmark_twice(request, "spliceai", lambda: run_spliceai_score(inputs, config))
    validate_output(result)
    assert_metrics_in_spec(result)

    assert result.tool_id == "spliceai-score"
    assert len(result.results) == n
    first = result.results[0]
    assert first.scores, "Expected non-empty scores: variants must overlap the synthetic gene"
    gene = first.scores[0]
    assert gene.symbol == "SPLICEAI_DEMO"
    assert all(0.0 <= ds <= 1.0 for ds in (gene.ds_ag, gene.ds_al, gene.ds_dg, gene.ds_dl))
    assert 0.0 <= first.metrics["max_delta_score"] <= 1.0

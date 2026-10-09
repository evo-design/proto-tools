"""Tests for SpliceAI2 splice prediction and variant scoring."""

import csv
import functools
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

import proto_tools.utils.standalone_helpers_source as _shs
from proto_tools.tools import (
    SpliceAI2JunctionChange,
    SpliceAI2PredictConfig,
    SpliceAI2PredictInput,
    SpliceAI2Prediction,
    SpliceAI2PredictOutput,
    SpliceAI2ScoreConfig,
    SpliceAI2ScoreInput,
    SpliceAI2ScoreMetrics,
    SpliceAI2ScoreOutput,
    SpliceAI2SiteChange,
    SpliceAI2Variant,
    SpliceAI2VariantResult,
    run_spliceai2_predict,
    run_spliceai2_score,
)
from proto_tools.tools.rna_splicing.spliceai2.spliceai2_score import upstream_columns
from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

_STANDALONE_DIR = Path(__file__).resolve().parents[2] / "proto_tools/tools/rna_splicing/spliceai2/standalone"

# HBB locus on GRCh38 (Ensembl naming), minus strand; spans all three exons with flank.
_HBB_CHROM = "11"
_HBB_START, _HBB_END = 5_225_000, 5_227_600

# HBB IVS1-1 G>A (c.92+1G>A) abolishes the intron-1 donor; IVS1-110 G>A creates a cryptic acceptor.
_IVS1_1 = SpliceAI2Variant(chromosome=_HBB_CHROM, position=5226929, ref="C", alt="T", strand="-")
_IVS1_110 = SpliceAI2Variant(chromosome=_HBB_CHROM, position=5226820, ref="C", alt="T", strand="-")

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _variant_result() -> SpliceAI2VariantResult:
    """One SpliceAI2VariantResult with ten distinct changes per effect (built without a tool run)."""
    sites = [
        SpliceAI2SiteChange(delta_score=0.9 - k / 20, ref_score=0.9, alt_score=k / 20, distance=k - 5)
        for k in range(10)
    ]
    junctions = [
        SpliceAI2JunctionChange(
            delta_score=0.8 - k / 20, ref_score=0.8, alt_score=k / 20, donor_distance=-k, acceptor_distance=100 + k
        )
        for k in range(10)
    ]
    return SpliceAI2VariantResult(
        chromosome="11",
        position=5226929,
        ref="C",
        alt="T",
        strand="-",
        donor_gain=sites,
        donor_loss=sites,
        acceptor_gain=sites,
        acceptor_loss=sites,
        junction_gain=junctions,
        junction_loss=junctions,
        metrics=SpliceAI2ScoreMetrics(summary_score=0.9),
    )


def _load_standalone_inference() -> ModuleType:
    """Import the standalone ``inference.py`` with the worker-injected ``standalone_helpers`` on sys.path."""
    helpers_dir = str(Path(next(iter(_shs.__path__))))
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    spec = importlib.util.spec_from_file_location("_spliceai2_inference_for_tests", _STANDALONE_DIR / "inference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def _hbb_locus() -> str:
    """Plus-strand GRCh38 sequence of the HBB locus (1-based inclusive), read from the provisioned FASTA."""
    fasta = SpliceAI2ScoreConfig(reference_fasta="grch38").require_reference_fasta("spliceai2")
    start, end = _HBB_START - 1, _HBB_END
    fai = fasta.with_name(fasta.name + ".fai")
    if fai.exists():
        # Seek straight to the region using the samtools-style index.
        for line in fai.read_text().splitlines():
            name, _, offset, line_bases, line_width = line.split("\t")[:5]
            if name == _HBB_CHROM:
                offset, line_bases, line_width = int(offset), int(line_bases), int(line_width)
                with open(fasta, "rb") as handle:
                    handle.seek(offset + (start // line_bases) * line_width + start % line_bases)
                    raw = handle.read((end - start) + (end - start) // line_bases + 2).decode()
                return raw.replace("\n", "").replace("\r", "")[: end - start].upper()
    # No index: stream to the chromosome and stop once the region is covered.
    chunks: list[str] = []
    seen = 0
    in_chrom = False
    with open(fasta) as handle:
        for line in handle:
            if line.startswith(">"):
                if in_chrom:
                    break
                in_chrom = line[1:].split()[0] == _HBB_CHROM
                continue
            if in_chrom:
                chunks.append(line.strip())
                seen += len(chunks[-1])
                if seen >= end:
                    break
    return "".join(chunks)[start:end].upper()


# ── Validators ────────────────────────────────────────────────────────────────


def test_score_input_wraps_single_variant() -> None:
    """A bare SpliceAI2Variant is normalized to a 1-element list."""
    assert SpliceAI2ScoreInput(variants=_IVS1_1).variants == [_IVS1_1]


def test_score_input_rejects_empty_variants() -> None:
    """An empty variant list is rejected rather than dispatching a no-op."""
    with pytest.raises(ValidationError, match="variants cannot be empty"):
        SpliceAI2ScoreInput(variants=[])


def test_variant_rejects_missing_strand() -> None:
    """Strand is required: SpliceAI2 has no annotation to infer it from."""
    with pytest.raises(ValidationError, match="strand"):
        SpliceAI2Variant(chromosome="11", position=5226929, ref="C", alt="T")


def test_variant_rejects_invalid_strand() -> None:
    """Only '+' and '-' are valid strands."""
    with pytest.raises(ValidationError, match="strand"):
        SpliceAI2Variant(chromosome="11", position=5226929, ref="C", alt="T", strand=".")


def test_predict_input_wraps_single_sequence() -> None:
    """A bare sequence string is normalized to a 1-element list."""
    assert SpliceAI2PredictInput(sequences="ACGT").sequences == ["ACGT"]


def test_predict_input_rejects_empty_sequences() -> None:
    """An empty sequence list is rejected."""
    with pytest.raises(ValidationError, match="sequences must not be empty"):
        SpliceAI2PredictInput(sequences=[])


def test_predict_input_uppercases_sequences() -> None:
    """Lowercase (soft-masked) DNA is uppercased before dispatch."""
    assert SpliceAI2PredictInput(sequences=["acgtn"]).sequences == ["ACGTN"]


@pytest.mark.parametrize("sequence", ["ACGU", "ACGX"])
def test_predict_input_rejects_invalid_chars(sequence: str) -> None:
    """Non-DNA characters (RNA 'U', arbitrary letters) are rejected."""
    with pytest.raises(ValidationError, match="invalid nucleotide characters"):
        SpliceAI2PredictInput(sequences=[sequence])


# ── Remote device support ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "config",
    [SpliceAI2PredictConfig(), SpliceAI2ScoreConfig(reference_fasta="grch38")],
    ids=["predict", "score"],
)
def test_config_remote_unsupported_reason(config: SpliceAI2PredictConfig | SpliceAI2ScoreConfig) -> None:
    """SpliceAI2 is refused on device='proto' but allowed on a caller-owned Modal deployment."""
    reason = config.remote_unsupported_reason("proto")
    assert reason is not None
    assert "device='modal'" in reason
    assert config.remote_unsupported_reason("modal") is None


def test_score_config_local_path_refused_remotely(tmp_path: Path) -> None:
    """A local FASTA path cannot travel to a Modal deployment."""
    fasta = tmp_path / "custom.fa"
    fasta.write_text(">1\nACGT\n")
    reason = SpliceAI2ScoreConfig(reference_fasta=str(fasta)).remote_unsupported_reason("modal")
    assert reason is not None
    assert "local path" in reason


@pytest.mark.parametrize("config_class", [SpliceAI2PredictConfig, SpliceAI2ScoreConfig], ids=["predict", "score"])
def test_config_defaults_to_gpu(config_class: type[SpliceAI2PredictConfig | SpliceAI2ScoreConfig]) -> None:
    """The GPU default from SpliceAI2Config survives mixing in the reference-genome config."""
    assert config_class().device == "cuda"


# ── Export (custom serialization only) ──────────────────────────────────────


def test_export_tsv(tmp_path: Path) -> None:
    """TSV export matches upstream's .spliceai2 columns and maps each change to its column."""
    result = _variant_result()
    SpliceAI2ScoreOutput(results=[result]).export("scores", tmp_path, file_format="tsv")
    with open(tmp_path / "scores.tsv", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        header = reader.fieldnames

    assert len(upstream_columns()) == 260
    assert header == ["chrom", "pos", "ref", "alt", "strand", *upstream_columns(), "spliceai2_summary_score"]
    assert len(rows) == 1
    row = rows[0]
    assert (row["chrom"], row["pos"], row["strand"]) == ("11", "5226929", "-")
    assert int(row["jxn_gain_donor_dist_0"]) == result.junction_gain[0].donor_distance
    assert int(row["jxn_loss_acceptor_dist_9"]) == result.junction_loss[9].acceptor_distance
    assert float(row["donor_loss_delta_score_3"]) == pytest.approx(result.donor_loss[3].delta_score)
    assert int(row["acceptor_gain_dist_7"]) == result.acceptor_gain[7].distance
    assert float(row["spliceai2_summary_score"]) == pytest.approx(0.9)


def test_export_json(tmp_path: Path) -> None:
    """JSON export round-trips through SpliceAI2ScoreOutput.model_validate."""
    output = SpliceAI2ScoreOutput(results=[_variant_result()])
    output.export("scores", tmp_path, file_format="json")
    restored = SpliceAI2ScoreOutput.model_validate(json.loads((tmp_path / "scores.json").read_text()))
    assert restored.results == output.results


def _prediction(length: int) -> SpliceAI2Prediction:
    """A SpliceAI2Prediction of the given length with no junctions or transcripts."""
    return SpliceAI2Prediction(
        donor_probabilities=[0.1] * length,
        acceptor_probabilities=[0.2] * length,
        junctions=[],
        transcripts=[],
    )


def test_export_npy_ragged(tmp_path: Path) -> None:
    """Sequences of differing lengths export as an object array of per-sequence (L, 2) arrays."""
    SpliceAI2PredictOutput(results=[_prediction(3), _prediction(5)]).export("preds", tmp_path, file_format="npy")
    loaded = np.load(tmp_path / "preds.npy", allow_pickle=True)
    assert loaded.dtype == object
    assert [a.shape for a in loaded] == [(3, 2), (5, 2)]


def test_export_npy_uniform(tmp_path: Path) -> None:
    """Equal-length sequences export as one (B, L, 2) [donor, acceptor] array."""
    SpliceAI2PredictOutput(results=[_prediction(4), _prediction(4)]).export("preds", tmp_path, file_format="npy")
    loaded = np.load(tmp_path / "preds.npy")
    assert loaded.shape == (2, 4, 2)
    assert np.allclose(loaded[..., 0], 0.1) and np.allclose(loaded[..., 1], 0.2)


# ── Standalone helpers ────────────────────────────────────────────────────────


def test_exon_intervals() -> None:
    """A 0/1 exon mask converts to 1-based inclusive intervals, including a run at the sequence end."""
    exon_intervals = _load_standalone_inference()._exon_intervals
    assert exon_intervals(np.array([1, 1, 0, 0, 1, 1, 1, 0, 1])) == [(1, 2), (5, 7), (9, 9)]
    assert exon_intervals(np.ones(6, dtype=int)) == [(1, 6)]


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.uses_gpu
def test_spliceai2_score_hbb_variants() -> None:
    """Known HBB splice variants: IVS1-1 abolishes the intron-1 donor, IVS1-110 gains an acceptor."""
    result = run_spliceai2_score(
        SpliceAI2ScoreInput(variants=[_IVS1_1, _IVS1_110]),
        SpliceAI2ScoreConfig(reference_fasta="grch38"),
    )

    assert result.success is True, f"SpliceAI2 score failed: {result}"
    assert result.tool_id == "spliceai2-score"
    assert len(result.results) == 2
    assert_metrics_in_spec(result)

    donor_site, cryptic_acceptor = result.results
    assert donor_site.metrics["summary_score"] > 0.5
    assert donor_site.donor_loss[0].delta_score > 0.5
    assert abs(donor_site.donor_loss[0].distance) <= 2
    assert cryptic_acceptor.acceptor_gain[0].delta_score > 0.1


@pytest.mark.uses_gpu
def test_spliceai2_predict_hbb_locus() -> None:
    """The sense-strand HBB locus yields strong donor/acceptor sites and a three-exon transcript."""
    sequence = _hbb_locus().translate(_COMPLEMENT)[::-1]
    result = run_spliceai2_predict(SpliceAI2PredictInput(sequences=[sequence]), SpliceAI2PredictConfig())

    assert result.success is True, f"SpliceAI2 predict failed: {result}"
    assert result.tool_id == "spliceai2-predict"
    assert len(result.results) == 1
    prediction = result.results[0]
    assert len(prediction.donor_probabilities) == len(sequence)
    assert len(prediction.acceptor_probabilities) == len(sequence)
    assert all(0.0 <= p <= 1.0 for p in prediction.donor_probabilities + prediction.acceptor_probabilities)
    assert max(prediction.donor_probabilities) > 0.5
    assert max(prediction.acceptor_probabilities) > 0.5

    assert prediction.junctions, "Expected junctions between HBB's annotated splice sites"
    assert all(j.donor < j.acceptor for j in prediction.junctions)
    assert prediction.transcripts, "Expected at least one decoded transcript"
    assert len(prediction.transcripts[0].exons) == 3


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("spliceai2-predict")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_spliceai2_predict_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark spliceai2-predict on 16 transcript-length (10 kb) sequences (cold + warm)."""
    n, length = 16, 10_000
    inputs = SpliceAI2PredictInput(sequences=random_dna_sequences(n=n, length=length, seed=0))
    # No device= so the harness can route --use-modal to the deployment.
    config = SpliceAI2PredictConfig()

    result = benchmark_twice(request, "spliceai2", lambda: run_spliceai2_predict(inputs, config))

    assert result.success is True, f"SpliceAI2 predict failed: {result}"
    assert result.tool_id == "spliceai2-predict"
    assert len(result.results) == n
    for prediction in result.results:
        assert len(prediction.donor_probabilities) == length
        assert len(prediction.acceptor_probabilities) == length


@pytest.mark.benchmark("spliceai2-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_spliceai2_score_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark spliceai2-score on 64 SNVs spanning the HBB locus on GRCh38 (cold + warm)."""
    locus = _hbb_locus()
    n = 64
    alt_map = {"A": "C", "C": "G", "G": "T", "T": "A"}
    offsets = [(i * len(locus)) // n for i in range(n)]
    variants = [
        SpliceAI2Variant(
            chromosome=_HBB_CHROM, position=_HBB_START + k, ref=locus[k], alt=alt_map[locus[k]], strand="-"
        )
        for k in offsets
    ]
    inputs = SpliceAI2ScoreInput(variants=variants)
    # A named assembly resolves on any worker, so device= is left to the harness.
    config = SpliceAI2ScoreConfig(reference_fasta="grch38")

    result = benchmark_twice(request, "spliceai2", lambda: run_spliceai2_score(inputs, config))
    validate_output(result)
    assert_metrics_in_spec(result)

    assert result.tool_id == "spliceai2-score"
    assert len(result.results) == n
    assert all(0.0 <= r.metrics["summary_score"] <= 1.0 for r in result.results)

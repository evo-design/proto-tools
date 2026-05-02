"""Tests for the CRISPRtracrRNA prediction tool."""

import importlib
import json
from pathlib import Path

import pytest

from proto_tools.tools.gene_annotation.crispr_tracr_rna import (
    CrisprTracrRNAConfig,
    CrisprTracrRNAInput,
    CrisprTracrRNAOutput,
    CrisprTracrRNAPrediction,
    CrisprTracrRNASequenceResult,
    run_crispr_tracr_rna,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
    validate_output,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "crispr_tracr_rna"
_SETUP_SH = (
    Path(__file__).resolve().parents[2] / "proto_tools/tools/gene_annotation/crispr_tracr_rna/standalone/setup.sh"
)
_RUN_MODULE = "proto_tools.tools.gene_annotation.crispr_tracr_rna.standalone.run"

_persistent_tool = make_persistent_fixture("crispr_tracr_rna", gpu=False)


def _read_fasta_sequence(fasta_path: Path) -> str:
    with open(fasta_path) as f:
        lines = [line.strip() for line in f if not line.startswith(">")]
    return "".join(lines)


# ── Schema ────────────────────────────────────────────────────────────────


def test_input_single_sequence_normalization():
    assert CrisprTracrRNAInput(sequences="ATCGATCG").sequences == ["ATCGATCG"]


def test_has_tracr_truth_table():
    """has_tracr fires on any candidate-detection field across both upstream modes."""
    cases = {
        "complete_run": CrisprTracrRNAPrediction(sequence_id="t", tracr_rna_sequence="GCAU", anti_repeat_start=100),
        "anti_repeat_only": CrisprTracrRNAPrediction(sequence_id="t", anti_repeat_start=42),
        "model_run_only": CrisprTracrRNAPrediction(sequence_id="t", hit_sequence="AAGGCTTT"),
    }
    for name, pred in cases.items():
        assert pred.has_tracr is True, name
    assert CrisprTracrRNAPrediction(sequence_id="t").has_tracr is False


def test_prediction_accepts_model_run_columns():
    """Schema must coerce upstream model_run column types."""
    pred = CrisprTracrRNAPrediction(
        sequence_id="x",
        start="12002",  # type: ignore[arg-type]
        end="12056",  # type: ignore[arg-type]
        e_value="4.50e-25",  # type: ignore[arg-type]
        best_e_value="4.50e-25",  # type: ignore[arg-type]
        hit_sequence="AAGGCTTT",
    )
    assert (pred.start, pred.end, pred.e_value, pred.hit_sequence) == (12002, 12056, 4.50e-25, "AAGGCTTT")


def test_sequence_result_top_candidate_and_has_tracr():
    """SequenceResult exposes the top-ranked candidate and aggregates has_tracr."""
    top = CrisprTracrRNAPrediction(sequence_id="seq1", anti_repeat_start=100, score=0.9)
    runner_up = CrisprTracrRNAPrediction(sequence_id="seq1", anti_repeat_start=200, score=0.4)
    result = CrisprTracrRNASequenceResult(sequence_id="seq1", candidates=[top, runner_up])
    assert result.top_candidate is top
    assert result.has_tracr is True
    empty = CrisprTracrRNASequenceResult(sequence_id="seq2", candidates=[])
    assert empty.top_candidate is None
    assert empty.has_tracr is False


def test_num_with_tracr():
    output = CrisprTracrRNAOutput(
        results=[
            CrisprTracrRNASequenceResult(
                sequence_id="seq1",
                candidates=[CrisprTracrRNAPrediction(sequence_id="seq1", anti_repeat_start=100)],
            ),
            CrisprTracrRNASequenceResult(sequence_id="seq2", candidates=[]),
            CrisprTracrRNASequenceResult(
                sequence_id="seq3",
                candidates=[CrisprTracrRNAPrediction(sequence_id="seq3", tracr_rna_sequence="ACGUACGU")],
            ),
        ]
    )
    assert output.num_with_tracr == 2


# ── Export ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_output():
    """Two sequences: one with 2 candidates (exercises flattening), one with none."""
    return CrisprTracrRNAOutput(
        results=[
            CrisprTracrRNASequenceResult(
                sequence_id="seq1",
                candidates=[
                    CrisprTracrRNAPrediction(sequence_id="seq1", anti_repeat_start=100, score=0.84),
                    CrisprTracrRNAPrediction(sequence_id="seq1", anti_repeat_start=300, score=0.42),
                ],
            ),
            CrisprTracrRNASequenceResult(sequence_id="seq2", candidates=[]),
        ]
    )


@pytest.mark.parametrize("file_format", ["csv", "json"])
def test_export(sample_output, tmp_path, file_format):
    """Export emits one row per candidate; sequences with no candidates get a sequence_id-only row."""
    sample_output.export(name="tracr", export_path=str(tmp_path), file_format=file_format)
    out_path = tmp_path / f"tracr.{file_format}"
    assert validate_export_output(out_path)
    if file_format == "json":
        rows = json.loads(out_path.read_text())
        assert [r["sequence_id"] for r in rows] == ["seq1", "seq1", "seq2"]
        assert [r.get("score") for r in rows] == [0.84, 0.42, None]


# ── Standalone runner ─────────────────────────────────────────────────────


def test_parse_tracr_results_complete_run(tmp_path: Path):
    """complete_run header forwards every column verbatim, normalizing NA-style cells to None."""
    run_module = importlib.import_module(_RUN_MODULE)
    (tmp_path / "results.csv").write_text(
        "accession_number,anti_repeat_start,interaction_energy,terminator_presence_flag,score\n"
        "seq1,42,-7.5,True,0.84\n"
        "seq2,,,NA,\n"
    )
    by_id = {r["sequence_id"]: r for r in run_module._parse_tracr_results(tmp_path, ["seq1", "seq2", "seq3"])}
    seq1 = by_id["seq1"]["candidates"][0]
    assert seq1["score"] == "0.84"
    assert seq1["interaction_energy"] == "-7.5"
    seq2 = by_id["seq2"]["candidates"][0]
    assert seq2["anti_repeat_start"] is None
    assert seq2["terminator_presence_flag"] is None
    assert by_id["seq3"]["candidates"] == []


def test_parse_tracr_results_model_run(tmp_path: Path):
    """model_run header (acc_num,start,end,e_value,hit_sequence) round-trips."""
    run_module = importlib.import_module(_RUN_MODULE)
    (tmp_path / "complete_hits.csv").write_text(
        "acc_num,start,end,e_value,hit_sequence\nAAAABU010000051,12002,12056,4.50e-25,AAGGCTTT\n"
    )
    cand = run_module._parse_tracr_results(tmp_path, ["AAAABU010000051"])[0]["candidates"][0]
    assert (cand["start"], cand["end"], cand["e_value"], cand["hit_sequence"]) == (
        "12002",
        "12056",
        "4.50e-25",
        "AAGGCTTT",
    )


def test_parse_tracr_results_sorts_candidates_by_score(tmp_path: Path):
    """Parser sorts candidates score-descending regardless of CSV row order; NA / missing scores last."""
    run_module = importlib.import_module(_RUN_MODULE)
    (tmp_path / "CRISPRtracrRNA_result.csv").write_text(
        "accession_number,anti_repeat_start,score\n"
        "acc1,250,0.42\n"
        "acc1,400,NA\n"  # no score → last
        "acc1,100,0.84\n"  # highest → first
        "acc1,300,0.10\n"
    )
    candidates = run_module._parse_tracr_results(tmp_path, ["acc1"])[0]["candidates"]
    assert [c["score"] for c in candidates] == ["0.84", "0.42", "0.10", None]
    assert [c["anti_repeat_start"] for c in candidates] == ["100", "250", "300", "400"]


def test_parse_tracr_results_priority_file_wins(tmp_path: Path):
    """Per-fasta intermediates are skipped for accessions covered by the priority CSV."""
    run_module = importlib.import_module(_RUN_MODULE)
    (tmp_path / "AAAIBU010000002.csv").write_text("accession_number,anti_repeat_start,score\nAAAIBU010000002,74170,\n")
    (tmp_path / "CRISPRtracrRNA_result.csv").write_text(
        "accession_number,anti_repeat_start,score\nAAAIBU010000002,74170,0.84\n"
    )
    candidates = run_module._parse_tracr_results(tmp_path, ["AAAIBU010000002"])[0]["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["score"] == "0.84"


def test_build_upstream_flags():
    """Defaults are omitted; overrides emit --flag <value>; bool emits only when True."""
    run_module = importlib.import_module(_RUN_MODULE)
    assert (
        run_module._build_upstream_flags(
            {
                "anti_repeat_similarity_threshold": 0.7,
                "weight_interaction_score": 0.6,
                "perform_type_v_anti_repeat_analysis": False,
            }
        )
        == []
    )
    flags = run_module._build_upstream_flags(
        {
            "anti_repeat_similarity_threshold": 0.85,
            "weight_interaction_score": 1.0,
            "perform_type_v_anti_repeat_analysis": True,
        }
    )
    assert {"--anti_repeat_similarity_threshold", "0.85"} <= set(flags)
    assert {"--weight_interaction_score", "1.0"} <= set(flags)
    assert {"--perform_type_v_anti_repeat_analysis", "True"} <= set(flags)


def test_setup_sh_content():
    """setup.sh must install all upstream tools, download cas-id models, and pin Python 3.8 + sklearn 0.22."""
    content = _SETUP_SH.read_text()

    # Upstream repos cloned into the right relative paths.
    assert "tools/CRISPRidentify/CRISPRidentify" in content
    assert "CRISPRcasIdentifier" in content

    # cas-id Google Drive models — file IDs from upstream README. Without these the
    # cas-effector-detection leg of complete_run is a silent no-op.
    assert "proto_download_gdrive" in content
    assert "1YbTxkn9KuJP2D7U1-6kL1Yimu_4RqSl1" in content
    assert "1Nc5o6QVB6QxMxpQjmLQcbwQwkRLk-thM" in content
    # Canonical filenames CRISPRcasIdentifier.py looks up at runtime.
    for name in ("HMM_sets.tar.gz", "trained_models.tar.gz"):
        assert name in content
        assert f"tar -xzf {name}" in content

    # Pinned Python + sklearn for CRISPRidentify's pickled models.
    assert "python=3.8" in content
    assert "scikit-learn=0.22" in content

    # Bioinformatics tools and Python deps installed via conda.
    for pkg in (
        "intarna",
        "infernal",
        "prodigal",
        "hmmer",
        "viennarna",
        "vmatch",
        "clustalo",
        "blast",
        "fasta3",
        "h5py",
        "dill",
        "networkx",
        "pyyaml",
        "regex",
        "requests",
    ):
        assert pkg in content, f"setup.sh missing {pkg}"


# ── Integration ───────────────────────────────────────────────────────────


@pytest.mark.integration
def test_run_complete_run_populates_score():
    """complete_run on Listeria locus must populate score + similarity*coverage > 0.5.

    Score-populates is the regression test for the silent --output_summary_file drop.
    """
    seq = _read_fasta_sequence(DATA_DIR / "AAAABU010000051.fasta")
    result = run_crispr_tracr_rna(
        CrisprTracrRNAInput(sequences=[seq], sequence_ids=["AAAABU010000051"]),
        CrisprTracrRNAConfig(model_type="II"),
    )
    assert result.success, result.errors
    top = result.results[0].top_candidate
    assert top is not None
    assert top.score is not None, "complete_run must populate the multi-evidence ranking score"
    assert top.anti_repeat_similarity_coverage_multiplication is not None
    assert top.anti_repeat_similarity_coverage_multiplication > 0.5


@pytest.mark.integration
def test_run_model_run_real_sequence():
    """model_run on the same locus populates the cmsearch-only schema fields on the top candidate."""
    seq = _read_fasta_sequence(DATA_DIR / "AAAABU010000051.fasta")
    result = run_crispr_tracr_rna(
        CrisprTracrRNAInput(sequences=[seq], sequence_ids=["AAAABU010000051"]),
        CrisprTracrRNAConfig(model_type="II", run_type="model_run"),
    )
    assert result.success, result.errors
    seq_result = result.results[0]
    assert seq_result.has_tracr
    top = seq_result.top_candidate
    assert top is not None
    assert top.hit_sequence is not None
    assert top.start is not None and top.end is not None
    assert top.start < top.end
    assert top.e_value is not None or top.best_e_value is not None


@pytest.mark.integration
def test_positive_control_spyogenes_sf370():
    """S. pyogenes SF370 — top tracrRNA carries the canonical sgRNA scaffold + multiple candidates ranked.

    This locus produces multiple candidate hits (verified against bare upstream).
    Asserts the wrapper preserves all of them and surfaces them score-descending.
    """
    seq = _read_fasta_sequence(DATA_DIR / "NC_002737_849000_875000.fasta")
    result = run_crispr_tracr_rna(
        CrisprTracrRNAInput(sequences=[seq], sequence_ids=["SpCas9_SF370"]),
        CrisprTracrRNAConfig(model_type="II"),
    )
    assert result.success
    assert result.num_with_tracr == 1
    seq_result = result.results[0]
    top = seq_result.top_candidate
    assert top is not None
    assert top.tracr_rna_sequence is not None
    assert "GTGGCACCGAGTCGGTGC" in top.tracr_rna_sequence
    assert top.intarna_anti_repeat_interaction is not None

    # Multi-hit invariant: upstream finds >1 candidate for this locus, sorted by score desc.
    assert len(seq_result.candidates) >= 2, (
        f"SF370 should produce multiple candidates, got {len(seq_result.candidates)}"
    )
    scores = [c.score for c in seq_result.candidates if c.score is not None]
    assert scores == sorted(scores, reverse=True), f"candidates must be score-descending, got {scores}"


# ── Benchmark ─────────────────────────────────────────────────────────────


@pytest.mark.benchmark("crispr-tracr-rna")
@pytest.mark.slow
def test_crispr_tracr_rna_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark: full S. pyogenes SF370 CRISPR-Cas9 locus (~26 kbp), model_type='II' (cold + warm)."""
    seq = _read_fasta_sequence(DATA_DIR / "NC_002737_849000_875000.fasta")
    inputs = CrisprTracrRNAInput(sequences=[seq], sequence_ids=["SpCas9_SF370"])
    config = CrisprTracrRNAConfig(model_type="II")
    result = benchmark_twice(request, "crispr_tracr_rna", lambda: run_crispr_tracr_rna(inputs, config))
    validate_output(result)
    assert result.tool_id == "crispr-tracr-rna"

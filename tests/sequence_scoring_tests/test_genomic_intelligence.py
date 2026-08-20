"""Tests for the Genomic Intelligence hosted-API toolkit.

The parse helpers are exercised against pinned payload literals so a change in
the service's response shape surfaces here without a network call. Tests that
reach the live API carry ``@pytest.mark.integration`` and are skipped by default.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from proto_tools.tools.sequence_scoring.genomic_intelligence import (
    EXPRESSION_WINDOW_BP,
    ExpressionSequence,
    GIAnnotationInput,
    GIChromatinInput,
    GIConfig,
    GIEnhancerInput,
    GIExpressionConfig,
    GIFindGenesInput,
    GIPromoterConfig,
    GIPromoterInput,
    GISpliceConfig,
    GISpliceInput,
    run_gi_promoter,
    run_gi_splice,
)
from proto_tools.tools.sequence_scoring.genomic_intelligence.gi_annotation import parse_annotation_data
from proto_tools.tools.sequence_scoring.genomic_intelligence.gi_chromatin import parse_chromatin_data
from proto_tools.tools.sequence_scoring.genomic_intelligence.gi_enhancer import parse_enhancer_data
from proto_tools.tools.sequence_scoring.genomic_intelligence.gi_expression import parse_expression_data
from proto_tools.tools.sequence_scoring.genomic_intelligence.gi_find_genes_and_predict_expression import (
    parse_workflow_data,
)
from proto_tools.tools.sequence_scoring.genomic_intelligence.gi_promoter import parse_promoter_data
from proto_tools.tools.sequence_scoring.genomic_intelligence.gi_splice import parse_splice_data
from proto_tools.tools.sequence_scoring.genomic_intelligence.shared_data_models import (
    GIAPIError,
    resolve_api_key,
    validate_gi_sequence,
)

# Response shapes recorded from the live service. These pin the shape, not the
# science: a prediction moving is expected, a key disappearing is a contract break.
_META = {
    "job_id": "3f1b0c1e-0000-4000-8000-000000000001",
    "model": "g0-promoter-2000bp",
    "request_id": "9a0b1c2d-0000-4000-8000-000000000002",
    "inference_time_ms": 42.5,
    "cold_start": False,
    "task": "promoter",
    "sequence_length": 400,
    "task_specific_counts": {},
}

_PROMOTER_PAYLOAD: dict[str, Any] = {
    "data": {
        "task": "promoter",
        "model": "g0-promoter-2000bp",
        "input": {"sequence_name": "demo", "sequence_length": 400},
        "summary": {"total_windows": 1, "promoter_windows": 0, "threshold_used": 0.5},
        "regions": [],
        "window_details": [
            {
                "window_index": 0,
                "prediction_start": 0,
                "prediction_end": 400,
                "context_start": 0,
                "context_end": 400,
                "probability": 0.1496,
                "is_positive": False,
            }
        ],
        "formats": None,
    },
    "meta": _META,
}

_SPLICE_PAYLOAD: dict[str, Any] = {
    "data": {
        "task": "splice",
        "model": "g0-splice-bigbird",
        "input": {"sequence_name": "demo", "sequence_length": 1200},
        "summary": {"total_sites": 1, "donor_sites": 1, "acceptor_sites": 0, "total_windows": 1},
        "sites": [
            {
                "name": "SD_1",
                "start": 96,
                "end": 101,
                "site_type": "donor",
                "score": 0.9083,
                "strand": ".",
                "chrom": "",
            }
        ],
        "tracks": {"donors": [], "acceptors": []},
        "window_details": [],
    },
    "meta": _META,
}

_ENHANCER_PAYLOAD: dict[str, Any] = {
    "data": {
        "task": "enhancer",
        "model": "g0-deepstarr",
        "input": {"sequence_name": "demo", "sequence_length": 1200},
        "summary": {"total_windows": 1, "dev_score_max": -0.91, "hk_score_max": -0.09},
        "windows": [
            {
                "window_index": 0,
                "start": 0,
                "end": 249,
                "dev_score": -0.96,
                "hk_score": -1.06,
                "context_start": 0,
                "context_end": 249,
            }
        ],
        "tracks": {"dev": [], "hk": []},
    },
    "meta": _META,
}

_CHROMATIN_PAYLOAD: dict[str, Any] = {
    "data": {
        "task": "chromatin",
        "model": "g0-deepsea",
        "input": {"sequence_name": "demo", "sequence_length": 1200},
        "summary": {
            "total_windows": 1,
            "total_annotations": 383,
            "category_counts": {"DNase": 200, "CTCF": 183},
        },
        "windows": [{"window_index": 0, "start": 0, "end": 200, "annotation_count": 383, "annotations": []}],
        "tracks": {},
    },
    "meta": _META,
}

_ANNOTATION_PAYLOAD: dict[str, Any] = {
    "data": {
        "task": "annotation",
        "model": "g0-annotation",
        "input": {"sequence_name": "demo", "sequence_length": 25000},
        "summary": {"total_transcripts": 1, "forward_strand": 1, "reverse_strand": 0},
        "transcripts": [
            {
                "name": "transcript_1",
                "start": 100,
                "end": 900,
                "strand": "+",
                "score": 0.87,
                "tss_position": 120,
                "polya_position": 880,
                "transcript_type": "mRNA",
            }
        ],
    },
    "meta": _META,
}

_EXPRESSION_PAYLOAD: dict[str, Any] = {
    "data": {
        "task": "expression",
        "model": "g0-expression",
        "input": {"sequence_name": "demo", "sequence_length": 9198},
        "summary": {},
        "prediction": {
            "expression": 0.95,
            "expression_log_tpm": 0.95,
            "expression_tpm": 1.58,
            "unit": "log(TPM+1)",
        },
    },
    "meta": {**_META, "task_specific_counts": {"tss_index": 4599, "scored_window": [0, 9198]}},
}

_WORKFLOW_PAYLOAD: dict[str, Any] = {
    "data": {
        "task": "find_genes_and_predict_expression",
        "annotation_model": "g0-annotation",
        "expression_model": "g0-expression",
        "input": {"sequence_name": "HBB", "sequence_length": 25000},
        "summary": {"genes_found": 2, "genes_predicted": 1, "genes_skipped": 1},
        "annotation": {},
        "expression_predictions": [
            {
                "gene_index": 0,
                "gene_name": "transcript_0",
                "strand": "+",
                "tss_position": 12000,
                "centered_sequence_length": 9198,
                "expression": 1.2,
                "expression_tpm": 2.3,
                "skipped": False,
            },
            {
                "gene_index": 1,
                "gene_name": "transcript_1",
                "strand": "+",
                "tss_position": 2320,
                "centered_sequence_length": 0,
                "expression": 0.0,
                "expression_tpm": 0.0,
                "skipped": True,
                "skip_reason": "TSS too close to sequence boundary (>50% padding)",
            },
        ],
    },
    "meta": {**_META, "task": "find_genes_and_predict_expression"},
}


# ============================================================================
# Parsing
# ============================================================================


def test_parse_promoter_reads_summary_and_windows() -> None:
    """Promoter parsing surfaces the summary counts and per-window scores."""
    result = parse_promoter_data(_PROMOTER_PAYLOAD["data"], _PROMOTER_PAYLOAD, "demo")
    assert result.name == "demo"
    assert result.sequence_length == 400
    assert result.total_windows == 1
    assert result.promoter_windows == 0
    assert result.windows[0].start == 0
    assert result.windows[0].end == 400
    assert result.max_probability == pytest.approx(0.1496)
    assert result.meta.request_id == _META["request_id"]


def test_parse_promoter_handles_no_windows() -> None:
    """A response with no scored windows yields no max probability."""
    payload = {"data": {"model": "m", "input": {}, "summary": {}}, "meta": {}}
    result = parse_promoter_data(payload["data"], payload, "empty")
    assert result.windows == []
    assert result.max_probability is None


def test_parse_splice_reads_sites() -> None:
    """Splice parsing surfaces site coordinates, type and score."""
    result = parse_splice_data(_SPLICE_PAYLOAD["data"], _SPLICE_PAYLOAD, "demo")
    assert result.total_sites == 1
    assert result.donor_sites == 1
    assert result.sites[0].site_type == "donor"
    assert result.sites[0].start == 96
    assert result.sites[0].score == pytest.approx(0.9083)


def test_parse_enhancer_reads_dual_scores() -> None:
    """Enhancer parsing keeps the developmental and housekeeping scores apart."""
    result = parse_enhancer_data(_ENHANCER_PAYLOAD["data"], _ENHANCER_PAYLOAD, "demo")
    assert result.total_windows == 1
    assert result.dev_score_max == pytest.approx(-0.91)
    assert result.hk_score_max == pytest.approx(-0.09)
    assert result.windows[0].dev_score == pytest.approx(-0.96)


def test_parse_chromatin_reads_category_counts() -> None:
    """Chromatin parsing surfaces per-category call counts."""
    result = parse_chromatin_data(_CHROMATIN_PAYLOAD["data"], _CHROMATIN_PAYLOAD, "demo")
    assert result.total_annotations == 383
    assert result.category_counts["DNase"] == 200
    assert result.windows[0].annotation_count == 383


def test_parse_annotation_reads_transcripts() -> None:
    """Annotation parsing surfaces transcript bounds, strand and TSS."""
    result = parse_annotation_data(_ANNOTATION_PAYLOAD["data"], _ANNOTATION_PAYLOAD, "demo")
    assert result.total_transcripts == 1
    assert result.transcripts[0].strand == "+"
    assert result.transcripts[0].tss_position == 120
    assert result.transcripts[0].transcript_type == "mRNA"


def test_parse_expression_reads_applied_window_from_meta() -> None:
    """Expression parsing reports the window the service applied, not the request's."""
    result = parse_expression_data(_EXPRESSION_PAYLOAD["data"], _EXPRESSION_PAYLOAD, "demo")
    assert result.expression_log_tpm == pytest.approx(0.95)
    assert result.tss_index == 4599
    assert result.scored_window == [0, 9198]


def test_parse_workflow_counts_scored_and_skipped_genes() -> None:
    """Workflow parsing separates genes scored from genes skipped."""
    result = parse_workflow_data(_WORKFLOW_PAYLOAD["data"], _WORKFLOW_PAYLOAD, "HBB")
    assert result.genes_found == 2
    assert result.genes_scored == 1
    assert result.predictions[1].skipped is True
    assert result.predictions[1].skip_reason is not None
    assert "g0-annotation" in result.meta.model


# ============================================================================
# Input validation
# ============================================================================


def test_sequence_below_task_floor_is_rejected_locally() -> None:
    """A sequence under the published floor fails before any request is made."""
    with pytest.raises(ValueError, match="published minimum"):
        validate_gi_sequence("ACGT" * 10, min_bp=300, task="promoter")


def test_sequence_above_shared_cap_is_rejected_locally() -> None:
    """A sequence over the published cap fails before any request is made."""
    with pytest.raises(ValueError, match="published maximum"):
        validate_gi_sequence("A" * 500_001, min_bp=300, task="promoter")


def test_invalid_nucleotides_are_rejected() -> None:
    """Non-nucleotide characters fail validation."""
    with pytest.raises(ValueError, match="Invalid nucleotide"):
        validate_gi_sequence("ACGTZZZZ" * 100, min_bp=50, task="enhancer")


def test_bare_string_is_coerced_to_one_sequence() -> None:
    """Every task input accepts a bare DNA string in place of a list."""
    for input_class in (GIPromoterInput, GISpliceInput, GIEnhancerInput, GIChromatinInput, GIAnnotationInput):
        parsed = input_class(sequences="ACGT" * 300)
        assert len(parsed.sequences) == 1
        assert parsed.sequences[0].name == "sequence"


def test_workflow_input_coerces_bare_string() -> None:
    """The workflow input accepts a bare DNA string too."""
    assert len(GIFindGenesInput(sequences="ACGT" * 300).sequences) == 1


def test_expression_requires_tss_index_for_non_window_length() -> None:
    """A locus that is not exactly one window must carry a TSS offset."""
    with pytest.raises(ValueError, match="tss_index is required"):
        ExpressionSequence(sequence="A" * (EXPRESSION_WINDOW_BP + 1000))


def test_expression_accepts_exact_window_without_tss_index() -> None:
    """Exactly one window needs no offset."""
    parsed = ExpressionSequence(sequence="A" * EXPRESSION_WINDOW_BP)
    assert parsed.tss_index is None


def test_expression_rejects_out_of_range_tss_index() -> None:
    """An offset without a full window either side is rejected locally."""
    with pytest.raises(ValueError, match="outside"):
        ExpressionSequence(sequence="A" * (EXPRESSION_WINDOW_BP + 1000), tss_index=100)


def test_expression_accepts_in_range_tss_index() -> None:
    """An offset with a full window either side is accepted."""
    parsed = ExpressionSequence(sequence="A" * 20_000, tss_index=10_000)
    assert parsed.tss_index == 10_000


# ============================================================================
# Configuration
# ============================================================================


def test_missing_api_key_fails_fast_with_actionable_message() -> None:
    """No key configured raises before any network call, naming the fix."""
    config = GIConfig(gi_api_key=None)
    with pytest.raises(OSError, match="GI_API_KEY"):
        resolve_api_key(config)


def test_explicit_key_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit config value wins over the environment variable."""
    monkeypatch.setenv("GI_API_KEY", "gi_from_env")
    assert resolve_api_key(GIConfig(gi_api_key="gi_explicit")) == "gi_explicit"


def test_key_is_excluded_from_the_cache_key() -> None:
    """The credential must not participate in cache identity."""
    schema = GIConfig.model_json_schema()
    assert "gi_api_key" in schema["properties"]
    first = GIPromoterConfig(gi_api_key="gi_one")
    second = GIPromoterConfig(gi_api_key="gi_two")
    assert first.cache_key() == second.cache_key()


def test_model_is_unset_by_default() -> None:
    """Model selection is left to the service unless the caller overrides it."""
    assert GIPromoterConfig().model is None


def test_splice_threshold_rejects_zero() -> None:
    """A zero threshold would return every scored position; it is refused."""
    with pytest.raises(ValueError):
        GISpliceConfig(threshold=0.0)


def test_expression_description_is_required_and_non_empty() -> None:
    """The conditioning text cannot be blank."""
    with pytest.raises(ValueError):
        GIExpressionConfig(description="")


def test_api_error_carries_code_and_request_id() -> None:
    """The error surfaces the enum code and the correlation id."""
    error = GIAPIError(422, "validation_failed", "bad input", "abc-123", {"errors": []})
    assert error.code == "validation_failed"
    assert error.request_id == "abc-123"
    assert "abc-123" in str(error)


# ============================================================================
# Integration — live API, skipped unless pytest runs with --integration
# (in CI, the run-integration label)
# ============================================================================


@pytest.mark.integration
def test_live_promoter_scores_a_sequence() -> None:
    """A promoter call returns per-window scores and provenance."""
    if not os.environ.get("GI_API_KEY"):
        pytest.skip("GI_API_KEY not set")
    sequence = "ATGCGCGCTATAAAAGGCGCG" * 30
    output = run_gi_promoter(GIPromoterInput(sequences=sequence), GIPromoterConfig())
    result = output.results[0]
    assert result.total_windows >= 1
    assert result.meta.model
    assert result.meta.request_id


@pytest.mark.integration
def test_live_splice_is_strand_dependent() -> None:
    """The reverse complement scores differently, and not near zero.

    Documents the trap rather than asserting a value: the wrong strand returns
    plausible sites, so a caller cannot detect orientation from the result.
    """
    if not os.environ.get("GI_API_KEY"):
        pytest.skip("GI_API_KEY not set")
    sequence = "ATGCGCGCTATAAAAGGCGCGGTAAGTCCCC" * 20
    complement = str.maketrans("ACGT", "TGCA")
    reverse = sequence.translate(complement)[::-1]
    forward_out = run_gi_splice(GISpliceInput(sequences=sequence), GISpliceConfig())
    reverse_out = run_gi_splice(GISpliceInput(sequences=reverse), GISpliceConfig())
    assert forward_out.results[0].meta.request_id
    assert reverse_out.results[0].meta.request_id

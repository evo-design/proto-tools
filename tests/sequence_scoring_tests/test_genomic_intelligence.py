"""Tests for the Genomic Intelligence hosted-API toolkit.

The parse helpers are exercised against pinned payload literals so a change in
the service's response shape surfaces here without a network call. Tests that
reach the live API carry ``@pytest.mark.integration`` and are skipped by default.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
import requests

from proto_tools.tools.sequence_scoring.genomic_intelligence import (
    EXPRESSION_WINDOW_BP,
    ExpressionSequence,
    GIAnnotationConfig,
    GIAnnotationInput,
    GIChromatinConfig,
    GIChromatinInput,
    GIConfig,
    GIEnhancerConfig,
    GIEnhancerInput,
    GIExpressionConfig,
    GIExpressionInput,
    GIFindGenesConfig,
    GIFindGenesInput,
    GIPromoterConfig,
    GIPromoterInput,
    GISpliceConfig,
    GISpliceInput,
    run_gi_annotation,
    run_gi_chromatin,
    run_gi_enhancer,
    run_gi_expression,
    run_gi_find_genes_and_predict_expression,
    run_gi_promoter,
    run_gi_splice,
    shared_data_models,
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
    GIResponseShapeError,
    call_predict,
    call_workflow,
    resolve_api_key,
    validate_gi_sequence,
)
from proto_tools.utils.tool_io import ToolExecutionError
from tests.tool_infra_tests.test_export_functionality import validate_output

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


class TestWrappedSequencesMatchServerSemantics:
    """The API strips whitespace and measures the stripped length; so do we.

    The shared validate_dna_sequence upper-cases but does not strip, so a
    line-wrapped body used to be refused as an invalid nucleotide character —
    stricter than the service, which scores it.
    """

    def test_a_wrapped_body_is_accepted_and_measured_stripped(self) -> None:
        bases = "ACGT" * 2300
        wrapped = "\n".join(bases[i : i + 60] for i in range(0, len(bases), 60))
        assert len(validate_gi_sequence(wrapped, min_bp=9198, task="expression")) == 9200

    def test_length_is_judged_on_bases_not_characters(self) -> None:
        """9,000 bases plus newlines exceeds 9,198 characters but is still short."""
        bases = "ACGT" * 2250
        wrapped = "\n".join(bases[i : i + 45] for i in range(0, len(bases), 45))
        assert len(wrapped) > 9198
        with pytest.raises(ValueError, match="9,000 bp"):
            validate_gi_sequence(wrapped, min_bp=9198, task="expression")

    def test_ambiguity_codes_are_still_refused(self) -> None:
        with pytest.raises(ValueError, match="Invalid nucleotide"):
            validate_gi_sequence("ACGTRACGT" * 20, min_bp=50, task="enhancer")


# ============================================================================
# Response shape — a malformed 2xx is refused and named, never substituted
# ============================================================================
#
# The idiom these guards replace was `x or {}` / `x or []`, which handles null
# and absent but passes a truthy wrong type straight through. Two outcomes
# followed and both are wrong: an AttributeError raised deep inside a parse
# helper, which reads as a client bug, or -- if the wrong value is replaced
# with an empty one -- a result object of zeros returned to the caller with
# real-looking provenance beside it, indistinguishable from a prediction of
# nothing. The tests below assert the third option, a typed refusal that names
# the offending field, and they assert it at the level the caller sees.


class _FakeResponse:
    """The parts of ``requests.Response`` the client actually reads."""

    def __init__(
        self,
        status_code: int,
        body: Any = None,
        *,
        text: str | None = None,
        json_raises: bool = False,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self._json_raises = json_raises
        self.text = text if text is not None else json.dumps(body)
        self.headers = {"X-Request-Id": "req-abc-123"}

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    def json(self) -> Any:
        if self._json_raises:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code}", response=self)  # type: ignore[arg-type]


class _FakeSession:
    """A session that replays a scripted POST and a queue of job polls."""

    def __init__(self, post: _FakeResponse, polls: list[_FakeResponse] | None = None) -> None:
        self._post = post
        self._polls = list(polls or [])
        self.headers: dict[str, str] = {}
        self.closed = False

    def post(self, *_args: Any, **_kwargs: Any) -> _FakeResponse:
        return self._post

    def get(self, *_args: Any, **_kwargs: Any) -> _FakeResponse:
        return self._polls.pop(0)

    def close(self) -> None:
        self.closed = True


def _install_session(
    monkeypatch: pytest.MonkeyPatch,
    post: _FakeResponse,
    polls: list[_FakeResponse] | None = None,
) -> _FakeSession:
    """Point the client at a fake session instead of the network."""
    session = _FakeSession(post, polls)
    monkeypatch.setattr(shared_data_models, "_build_session", lambda _config: session)
    return session


_GOOD_DATA = {"model": "g0-promoter-2000bp", "input": {"sequence_length": 400}, "summary": {"total_windows": 1}}

_BAD_SUMMARY_PAYLOAD: dict[str, Any] = {"data": {**_GOOD_DATA, "summary": "all good"}, "meta": _META}


def _predict(config: GIConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Issue one promoter predict call through the shared client."""
    return call_predict(config, "promoter", "ATGC" * 100, "demo")


class TestTheEnvelopeIsRequired:
    """Every path that reads content out of ``data`` requires ``data``.

    There was no envelope check of any kind before this: ``call_predict`` and
    ``call_workflow`` both ended in ``payload.get("data") or {}`` and handed the
    result to the parse helpers, so a 2xx with no ``data``, an empty ``data``,
    or a ``data`` of the wrong type all produced a zero-valued result rather
    than an error. These drive the client against fake sessions rather than
    inspecting its source, so the check has to be reachable to pass.
    """

    def test_a_synchronous_result_must_carry_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_session(monkeypatch, _FakeResponse(200, {"meta": _META}))
        with pytest.raises(GIAPIError, match="'data' was missing or null"):
            _predict(GIConfig(gi_api_key="gi_test"))

    def test_an_empty_data_object_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point: ``{"data": {}}`` is malformed for every caller."""
        _install_session(monkeypatch, _FakeResponse(200, {"data": {}, "meta": _META}))
        with pytest.raises(GIAPIError, match="'data' was an empty object"):
            _predict(GIConfig(gi_api_key="gi_test"))

    def test_a_wrong_typed_data_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_session(monkeypatch, _FakeResponse(200, {"data": "all good", "meta": _META}))
        with pytest.raises(GIAPIError, match="'data' was str"):
            _predict(GIConfig(gi_api_key="gi_test"))

    def test_a_non_object_body_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_session(monkeypatch, _FakeResponse(200, ["not", "an", "envelope"]))
        with pytest.raises(GIAPIError, match="body was list"):
            _predict(GIConfig(gi_api_key="gi_test"))

    def test_a_non_json_two_hundred_is_named_rather_than_decoded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A raw decode error here reads as a client bug.

        Worse on the polling path, where ``JSONDecodeError`` is a
        ``RequestException`` and ``poll_until_complete`` retries it to the
        wall-clock deadline before failing with ``TimeoutError``.
        """
        _install_session(monkeypatch, _FakeResponse(200, None, text="<html>502</html>", json_raises=True))
        with pytest.raises(GIAPIError, match="non-JSON response body"):
            _predict(GIConfig(gi_api_key="gi_test"))

    def test_the_error_carries_the_correlation_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The refusal names the request the way a support ticket needs.

        That is why the envelope check lives in the client rather than in the
        parse helpers: the response, and its ``X-Request-Id``, are still in hand.
        """
        _install_session(monkeypatch, _FakeResponse(200, {"data": {}, "meta": _META}))
        with pytest.raises(GIAPIError) as caught:
            _predict(GIConfig(gi_api_key="gi_test"))
        assert caught.value.request_id == "req-abc-123"

    def test_an_accepted_job_must_carry_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``data.job_id`` is read out of the 202, so the 202 is enveloped too."""
        _install_session(monkeypatch, _FakeResponse(202, {"data": {}, "meta": _META}))
        with pytest.raises(GIAPIError, match="'data' was an empty object"):
            _predict(GIConfig(gi_api_key="gi_test", respond_async=True))

    def test_a_finished_job_result_must_carry_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_session(
            monkeypatch,
            _FakeResponse(202, {"data": {"job_id": "job-1"}, "meta": _META}),
            [_FakeResponse(200, {"data": {}, "meta": _META})],
        )
        with pytest.raises(GIAPIError, match="'data' was an empty object"):
            _predict(GIConfig(gi_api_key="gi_test", respond_async=True))

    def test_the_workflow_endpoint_is_enveloped_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_session(monkeypatch, _FakeResponse(200, {"data": {}, "meta": _META}))
        with pytest.raises(GIAPIError, match="'data' was an empty object"):
            call_workflow(GIConfig(gi_api_key="gi_test"), "ATGC" * 3000, "demo", {})

    def test_a_running_job_is_not_enveloped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job still running legitimately has nothing in ``data`` yet.

        So the 202 polls in between are exempt from the envelope check on
        purpose. Only the result the caller is handed is enveloped.
        """
        _install_session(
            monkeypatch,
            _FakeResponse(202, {"data": {"job_id": "job-1"}, "meta": _META}),
            [
                _FakeResponse(202, {"data": {}, "meta": _META}),
                _FakeResponse(200, {"data": _GOOD_DATA, "meta": _META}),
            ],
        )
        data, _ = _predict(GIConfig(gi_api_key="gi_test", respond_async=True, poll_interval_seconds=1.0))
        assert data == _GOOD_DATA

    def test_a_well_formed_envelope_still_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _install_session(monkeypatch, _FakeResponse(200, {"data": _GOOD_DATA, "meta": _META}))
        data, payload = _predict(GIConfig(gi_api_key="gi_test"))
        assert data == _GOOD_DATA
        assert payload["meta"]["request_id"] == _META["request_id"]
        assert session.closed


class TestNestedFieldsOfTheWrongType:
    """The fields nested inside ``data`` are checked per task.

    ``require_envelope`` guarantees ``data`` itself, and stops there: it is
    shared by six predict endpoints and one workflow, so it must not encode any
    single task's schema.

    Absent and null stay legitimate throughout -- a task that reports no regions
    omits the member -- so only a field that is *present with the wrong type* is
    a refusal.
    """

    @pytest.mark.parametrize(
        ("data", "field"),
        [
            ({"summary": "all good"}, "data.summary"),
            ({"regions": "none"}, "data.regions"),
            ({"window_details": {"probability": 0.9}}, "data.window_details"),
            ({"input": "400 bp"}, "data.input"),
        ],
    )
    def test_promoter_names_the_offending_field(self, data: dict[str, Any], field: str) -> None:
        with pytest.raises(GIResponseShapeError, match=re.escape(field)):
            parse_promoter_data(data, {"data": data, "meta": _META}, "demo")

    def test_an_array_element_of_the_wrong_type_names_its_index(self) -> None:
        """A wrong-typed element is named by index, not dropped.

        ``or []`` let a list of strings through and the row loop built rows out
        of them. Skipping them instead would lose rows the caller needed, which
        is the same quiet wrong answer in a different place.
        """
        data = {"window_details": [{"window_index": 0}, "second"]}
        with pytest.raises(GIResponseShapeError, match=re.escape("data.window_details[1]")):
            parse_promoter_data(data, {"data": data, "meta": _META}, "demo")

    def test_splice_sites_of_the_wrong_type(self) -> None:
        data = {"sites": "SD_1"}
        with pytest.raises(GIResponseShapeError, match=re.escape("data.sites")):
            parse_splice_data(data, {"data": data, "meta": _META}, "demo")

    def test_enhancer_windows_of_the_wrong_type(self) -> None:
        data = {"windows": 3}
        with pytest.raises(GIResponseShapeError, match=re.escape("data.windows")):
            parse_enhancer_data(data, {"data": data, "meta": _META}, "demo")

    def test_chromatin_category_counts_of_the_wrong_type(self) -> None:
        """One level deeper than the others, and named to that depth."""
        data = {"summary": {"category_counts": "promoter=3"}}
        with pytest.raises(GIResponseShapeError, match=re.escape("data.summary.category_counts")):
            parse_chromatin_data(data, {"data": data, "meta": _META}, "demo")

    def test_annotation_transcripts_of_the_wrong_type(self) -> None:
        data = {"transcripts": "transcript_0"}
        with pytest.raises(GIResponseShapeError, match=re.escape("data.transcripts")):
            parse_annotation_data(data, {"data": data, "meta": _META}, "demo")

    def test_expression_prediction_of_the_wrong_type(self) -> None:
        data = {"prediction": "3.2 log TPM"}
        with pytest.raises(GIResponseShapeError, match=re.escape("data.prediction")):
            parse_expression_data(data, {"data": data, "meta": _META}, "demo")

    def test_expression_reads_its_window_through_a_checked_meta(self) -> None:
        """Expression's applied window is provenance, and it lives in ``meta``.

        It is read back rather than assumed from the request because an in-range
        but wrong ``tss_index`` scores a different window and still returns 200,
        so a wrong-typed ``meta`` there is not something to shrug at.
        """
        data = {"prediction": {"expression_log_tpm": 3.2}}
        payload = {"data": data, "meta": {"task_specific_counts": "scored_window=[0, 9198]"}}
        with pytest.raises(GIResponseShapeError, match=re.escape("meta.task_specific_counts")):
            parse_expression_data(data, payload, "demo")

    def test_workflow_predictions_of_the_wrong_type(self) -> None:
        data = {"expression_predictions": "gene_0"}
        with pytest.raises(GIResponseShapeError, match=re.escape("data.expression_predictions")):
            parse_workflow_data(data, {"data": data, "meta": _META}, "demo")

    def test_a_wrong_typed_meta_is_refused(self) -> None:
        """``meta`` carries the provenance a caller cites in a support request."""
        data = dict(_PROMOTER_PAYLOAD["data"])
        with pytest.raises(GIResponseShapeError, match=re.escape("meta")):
            parse_promoter_data(data, {"data": data, "meta": "ok"}, "demo")

    @pytest.mark.parametrize("value", [None, "absent"])
    def test_absent_and_null_are_still_legitimate(self, value: Any) -> None:
        """A task with nothing to report omits the member, or sends null."""
        data: dict[str, Any] = {"model": "m", "summary": {"total_windows": 0}}
        if value is None:
            data["regions"] = None
            data["window_details"] = None
            data["input"] = None
        result = parse_promoter_data(data, {"data": data, "meta": _META}, "demo")
        assert result.regions == []
        assert result.windows == []
        assert result.sequence_length == 0

    def test_a_well_formed_payload_is_untouched(self) -> None:
        """The recorded live shapes still parse, so the guard is not overreaching."""
        assert parse_promoter_data(_PROMOTER_PAYLOAD["data"], _PROMOTER_PAYLOAD, "demo").total_windows == 1
        assert parse_splice_data(_SPLICE_PAYLOAD["data"], _SPLICE_PAYLOAD, "demo").total_sites == 1


class TestTheCallerSeesTheRefusal:
    """The durable half: what the tool hands back, not what a helper raised.

    Asserting only that a helper raised does not show what the caller ends up
    with, which is the mistake GI-057 made -- it pinned *survival* as the
    specification and made the quiet wrong answer official.

    ``proto_tools`` has two caller-visible outcomes and both are asserted here.
    By default a tool re-raises, so the refusal reaches the caller as an
    exception naming the field. Under ``PROTO_CAPTURE_ERRORS=1`` the framework
    captures it into the output instead, and what matters there is that the
    output says ``success=False`` and carries the field name -- not that it came
    back holding a plausible-looking result of zeros.
    """

    def test_a_wrong_typed_summary_raises_rather_than_returning_zeros(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_session(monkeypatch, _FakeResponse(200, _BAD_SUMMARY_PAYLOAD))
        monkeypatch.delenv("PROTO_CAPTURE_ERRORS", raising=False)
        with pytest.raises(GIResponseShapeError, match=re.escape("data.summary")):
            run_gi_promoter(
                GIPromoterInput(sequences="ATGC" * 100),
                GIPromoterConfig(gi_api_key="gi_test"),
            )

    def test_capture_mode_reports_a_failure_not_a_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_session(monkeypatch, _FakeResponse(200, _BAD_SUMMARY_PAYLOAD))
        monkeypatch.setenv("PROTO_CAPTURE_ERRORS", "1")
        output = run_gi_promoter(
            GIPromoterInput(sequences="ATGC" * 100),
            GIPromoterConfig(gi_api_key="gi_test"),
        )
        assert output.success is False
        assert "data.summary" in " ".join(output.errors)
        # And the results are not merely empty, they are unreadable: the
        # framework refuses the attribute on a failed output, so a caller
        # cannot mistake this for a prediction that found nothing.
        with pytest.raises(ToolExecutionError, match=re.escape("data.summary")):
            _ = output.results

    def test_a_sound_response_still_produces_a_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same path with the one field corrected.

        Without this, a passing refusal test could be the fake session failing
        rather than the guard firing.
        """
        _install_session(monkeypatch, _FakeResponse(200, _PROMOTER_PAYLOAD))
        monkeypatch.delenv("PROTO_CAPTURE_ERRORS", raising=False)
        output = run_gi_promoter(
            GIPromoterInput(sequences="ATGC" * 100),
            GIPromoterConfig(gi_api_key="gi_test"),
        )
        assert output.results[0].total_windows == 1
        assert output.results[0].meta.request_id == _META["request_id"]


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


# ============================================================================
# Benchmarks — live API, run only under --benchmark
# ============================================================================
#
# Every tool here is a client for a hosted HTTP API, so the only realistic
# workload is a real request: there is no local model to warm up and nothing
# offline to measure. Benchmarks are deselected unless pytest runs with
# --benchmark, and each one skips when GI_API_KEY is unset, which is the same
# guard the live tests above use.
#
# Workload is the HBB locus shipped with the toolkit (GRCh38
# chr11:5,220,000-5,245,000, 25,000 bp, gene-sense) rather than random DNA, so
# the timings reflect sequence the models were built for. The per-window tools
# score a batch of three 5,000 bp windows, the shape a caller scoring a
# population submits; the locus-level tools take the whole 25,000 bp.

_GI_EXAMPLES_DIR = (
    Path(__file__).resolve().parents[2] / "proto_tools/tools/sequence_scoring/genomic_intelligence/examples"
)

_BENCH_WINDOW_BP = 5_000
_BENCH_WINDOW_OFFSETS = (0, 6_000, 12_000)


def _hbb_locus() -> str:
    """Return the HBB locus shipped beside the toolkit's example notebook."""
    return (_GI_EXAMPLES_DIR / "hbb_locus.txt").read_text().strip()


def _bench_windows() -> list[dict[str, str]]:
    """Three non-overlapping 5,000 bp windows of the HBB locus."""
    locus = _hbb_locus()
    return [
        {"sequence": locus[offset : offset + _BENCH_WINDOW_BP], "name": f"window_{offset}"}
        for offset in _BENCH_WINDOW_OFFSETS
    ]


def _require_live_key() -> None:
    """Skip when no key is configured; a hosted-API benchmark cannot run without one."""
    if not os.environ.get("GI_API_KEY"):
        pytest.skip("GI_API_KEY not set")


@pytest.mark.benchmark("gi-promoter")
@pytest.mark.slow
def test_gi_promoter_benchmark() -> None:
    """Benchmark gi-promoter: 3 x 5,000 bp HBB windows."""
    _require_live_key()
    output = run_gi_promoter(GIPromoterInput(sequences=_bench_windows()), GIPromoterConfig())
    validate_output(output)

    assert output.tool_id == "gi-promoter"
    assert len(output.results) == len(_BENCH_WINDOW_OFFSETS)
    for result in output.results:
        assert result.total_windows >= 1
        assert result.meta.request_id


@pytest.mark.benchmark("gi-splice")
@pytest.mark.slow
def test_gi_splice_benchmark() -> None:
    """Benchmark gi-splice: 3 x 5,000 bp HBB windows, gene-sense strand."""
    _require_live_key()
    output = run_gi_splice(GISpliceInput(sequences=_bench_windows()), GISpliceConfig())
    validate_output(output)

    assert output.tool_id == "gi-splice"
    assert len(output.results) == len(_BENCH_WINDOW_OFFSETS)
    for result in output.results:
        assert result.total_sites == result.donor_sites + result.acceptor_sites
        assert result.meta.request_id


@pytest.mark.benchmark("gi-enhancer")
@pytest.mark.slow
def test_gi_enhancer_benchmark() -> None:
    """Benchmark gi-enhancer: 3 x 5,000 bp HBB windows."""
    _require_live_key()
    output = run_gi_enhancer(GIEnhancerInput(sequences=_bench_windows()), GIEnhancerConfig())
    validate_output(output)

    assert output.tool_id == "gi-enhancer"
    assert len(output.results) == len(_BENCH_WINDOW_OFFSETS)
    for result in output.results:
        assert result.total_windows >= 1
        assert result.meta.request_id


@pytest.mark.benchmark("gi-chromatin")
@pytest.mark.slow
def test_gi_chromatin_benchmark() -> None:
    """Benchmark gi-chromatin: 3 x 5,000 bp HBB windows."""
    _require_live_key()
    output = run_gi_chromatin(GIChromatinInput(sequences=_bench_windows()), GIChromatinConfig())
    validate_output(output)

    assert output.tool_id == "gi-chromatin"
    assert len(output.results) == len(_BENCH_WINDOW_OFFSETS)
    for result in output.results:
        assert result.total_windows >= 1
        assert result.meta.request_id


@pytest.mark.benchmark("gi-annotation")
@pytest.mark.slow
def test_gi_annotation_benchmark() -> None:
    """Benchmark gi-annotation: the whole 25,000 bp HBB locus."""
    _require_live_key()
    locus = _hbb_locus()
    output = run_gi_annotation(
        GIAnnotationInput(sequences=[{"sequence": locus, "name": "HBB"}]),
        GIAnnotationConfig(),
    )
    validate_output(output)

    assert output.tool_id == "gi-annotation"
    result = output.results[0]
    assert result.sequence_length == len(locus)
    assert result.total_transcripts >= 0
    assert result.meta.request_id


@pytest.mark.benchmark("gi-expression")
@pytest.mark.slow
def test_gi_expression_benchmark() -> None:
    """Benchmark gi-expression: the 25,000 bp HBB locus cut to one window around its midpoint."""
    _require_live_key()
    locus = _hbb_locus()
    # The model scores one 9,198 bp window, so a longer locus needs the TSS
    # offset. The midpoint keeps the required flank on both sides without
    # depending on an annotation call to place it.
    output = run_gi_expression(
        GIExpressionInput(sequences=[{"sequence": locus, "name": "HBB", "tss_index": len(locus) // 2}]),
        GIExpressionConfig(),
    )
    validate_output(output)

    assert output.tool_id == "gi-expression"
    result = output.results[0]
    assert result.sequence_length == len(locus)
    assert result.expression_log_tpm is not None
    assert result.meta.request_id


@pytest.mark.benchmark("gi-find-genes-and-predict-expression")
@pytest.mark.slow
def test_gi_find_genes_and_predict_expression_benchmark() -> None:
    """Benchmark the workflow: annotation plus per-gene expression over the 25,000 bp HBB locus."""
    _require_live_key()
    locus = _hbb_locus()
    output = run_gi_find_genes_and_predict_expression(
        GIFindGenesInput(sequences=[{"sequence": locus, "name": "HBB"}]),
        GIFindGenesConfig(),
    )
    validate_output(output)

    assert output.tool_id == "gi-find-genes-and-predict-expression"
    result = output.results[0]
    assert result.sequence_length == len(locus)
    assert result.genes_scored <= result.genes_found
    assert result.meta.request_id

"""A deployment records the result format its envelope returns, and drift is reported as data.

A caller that stores results (the hosted gateway) must refuse a deployment whose envelope it cannot
read before dispatching to it, rather than fail after the run has been paid for. No hash moves when
the envelope changes, because framework code is not fingerprinted, so the format is recorded on its
own and read back as a number.
"""

from __future__ import annotations

import contextlib
import io
import json
from typing import Any

import modal
import pytest

from proto_tools.modal import fingerprint as fp
from proto_tools.modal.utils import RESULT_FORMAT

TOOL_KEY = "esm2-embedding"
SERVICE = "ESM2Service"


class _Volume:
    """Captures what ``write_manifest`` uploads."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    @contextlib.contextmanager
    def batch_upload(self, force: bool = False):
        volume = self

        class _Batch:
            def put_file(self, handle: io.BytesIO, path: str) -> None:
                volume.files[path] = handle.getvalue()

        yield _Batch()


@pytest.fixture
def volume(monkeypatch: pytest.MonkeyPatch) -> _Volume:
    captured = _Volume()
    monkeypatch.setattr(modal.Volume, "from_name", lambda *_args, **_kwargs: captured)
    return captured


def _recorded(**overrides: Any) -> dict[str, Any]:
    """A manifest recording exactly the local fingerprint of ``TOOL_KEY``, with entry overrides."""
    entry = {**fp.fingerprint(TOOL_KEY).model_dump(), **overrides}
    return {"algorithm": fp.ALGORITHM, "result_format": RESULT_FORMAT, "tools": {TOOL_KEY: entry}}


def test_a_deploy_records_the_result_format(volume: _Volume) -> None:
    fp.write_manifest(SERVICE)

    manifest = json.loads(volume.files[fp.manifest_path(SERVICE)])

    assert manifest["result_format"] == RESULT_FORMAT
    assert fp.recorded_result_format(manifest) == RESULT_FORMAT


def test_a_manifest_written_before_the_field_existed_reads_as_the_first_format() -> None:
    """Every deployment recorded before this field predates JSON-mode results."""
    manifest = _recorded()
    del manifest["result_format"]

    assert fp.recorded_result_format(manifest) == 1


def test_an_aligned_deployment_has_no_drift() -> None:
    assert fp.drifted_fields(TOOL_KEY, _recorded()) == {}


def test_each_moved_hash_is_named_with_both_values() -> None:
    local = fp.fingerprint(TOOL_KEY)

    drifted = fp.drifted_fields(TOOL_KEY, _recorded(code_hash="000000000000", env_hash="111111111111"))

    assert drifted == {
        "code_hash": ("000000000000", local.code_hash),
        "env_hash": ("111111111111", local.env_hash),
    }


def test_a_different_hashing_algorithm_cannot_be_compared_hash_by_hash() -> None:
    manifest = {**_recorded(), "algorithm": fp.ALGORITHM - 1}

    assert fp.drifted_fields(TOOL_KEY, manifest) == {"algorithm": (fp.ALGORITHM - 1, fp.ALGORITHM)}


def test_a_tool_the_deployment_never_recorded_reports_nothing() -> None:
    assert fp.drifted_fields("esmfold-prediction", _recorded()) == {}


def test_warnings_describe_exactly_the_drifted_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prose and the structured answer come from one comparison, so they cannot disagree."""
    manifest = _recorded(schema_hash="000000000000")
    monkeypatch.setattr(fp, "read_manifest", lambda *_args, **_kwargs: manifest)

    warnings = fp.drift_warnings(TOOL_KEY, SERVICE)

    assert len(warnings) == 1
    assert "schema differs" in warnings[0]


class _ReadVolume:
    """Answers ``read_file`` with a manifest, or raises what Modal would."""

    def __init__(self, payload: bytes | None = None, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    def read_file(self, _path: str):
        if self._error is not None:
            raise self._error
        yield self._payload


@pytest.mark.parametrize(
    "error",
    [
        # What Volume.read_file raises for a missing file: it re-raises Modal's NotFoundError as this.
        FileNotFoundError("/_fingerprints/ESM2Service.json"),
        # A missing volume.
        modal.exception.NotFoundError("no such volume"),
    ],
)
def test_an_absent_manifest_reads_as_none(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    volume = _ReadVolume(error=error)
    monkeypatch.setattr(modal.Volume, "from_name", lambda *_args, **_kwargs: volume)

    assert fp.read_manifest(SERVICE) is None


def test_a_manifest_that_is_not_an_object_is_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    volume = _ReadVolume(payload=b"[]")
    monkeypatch.setattr(modal.Volume, "from_name", lambda *_args, **_kwargs: volume)

    with pytest.raises(fp.ManifestUnreadable):
        fp.read_manifest(SERVICE)


def test_an_unreadable_manifest_is_not_mistaken_for_an_absent_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller deciding whether to rebuild must not read a transient failure as "never recorded"."""
    volume = _ReadVolume(error=ConnectionError("modal unreachable"))
    monkeypatch.setattr(modal.Volume, "from_name", lambda *_args, **_kwargs: volume)

    with pytest.raises(fp.ManifestUnreadable):
        fp.read_manifest(SERVICE)


def test_drift_warnings_still_report_nothing_for_an_unreadable_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    volume = _ReadVolume(error=ConnectionError("modal unreachable"))
    monkeypatch.setattr(modal.Volume, "from_name", lambda *_args, **_kwargs: volume)

    assert fp.drift_warnings(TOOL_KEY, SERVICE) == []


@pytest.mark.parametrize("value", [True, 2.0, "2", None])
def test_only_an_integer_counts_as_a_recorded_format(value: object) -> None:
    recorded = fp.recorded_result_format({"result_format": value})

    assert recorded == 1 and type(recorded) is int


@pytest.mark.parametrize("tools", [[], "esm2-embedding", None])
def test_a_malformed_tool_table_reads_as_no_drift(tools: object) -> None:
    assert fp.drifted_fields(TOOL_KEY, {"algorithm": fp.ALGORITHM, "tools": tools}) == {}


def test_a_deployment_newer_than_this_client_says_upgrade_not_redeploy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redeploying from an older client would downgrade a deployment a newer proto-tools built."""
    manifest = {**_recorded(code_hash="000000000000"), "result_format": RESULT_FORMAT + 1}
    monkeypatch.setattr(fp, "read_manifest", lambda *_args, **_kwargs: manifest)

    warnings = fp.drift_warnings(TOOL_KEY, SERVICE)

    assert len(warnings) == 1
    assert "newer proto-tools" in warnings[0]
    assert "Redeploy" not in warnings[0]

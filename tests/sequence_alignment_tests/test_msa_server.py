"""tests/sequence_alignment_tests/test_msa_server.py.

The ColabFold MSA server conversation, driven against a faked transport.

The real server is a third party and must not be called from a test, so every exchange here is
scripted. What is worth pinning is the protocol itself — the endpoints and mode strings the server
expects, and which statuses are worth waiting on — since getting those wrong surfaces only as
intermittent failures against a service nobody controls.
"""

import io
import tarfile
from pathlib import Path
from typing import Any

import pytest

from proto_tools.tools.sequence_alignment.mmseqs2 import msa_server


class _Reply:
    """One scripted HTTP reply."""

    def __init__(self, payload: dict[str, Any] | None = None, body: bytes = b""):
        """Hold a JSON payload, or raw bytes for a download."""
        self._payload = payload
        self._body = body
        self.text = str(payload or "")

    def json(self) -> dict[str, Any]:
        """Return the scripted payload, or refuse as an unreadable reply would."""
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def iter_content(self, chunk_size: int = 0) -> Any:
        """Yield the archive body."""
        yield self._body


def _tar_bytes(name: str = "uniref.a3m", content: bytes = b">101\nMKT\n") -> bytes:
    """A gzipped tar shaped like the server's result archive."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture
def transport(monkeypatch):
    """Script the server's replies and record every request made."""
    calls: list[tuple[str, str, dict[str, Any]]] = []
    replies: list[_Reply] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> _Reply:
        calls.append((method, url, kwargs))
        return replies.pop(0) if replies else _Reply({"status": "ERROR"})

    monkeypatch.setattr(msa_server.requests, "request", fake_request)
    monkeypatch.setattr(msa_server, "_sleep_with_jitter", lambda _s: None)
    return {"calls": calls, "replies": replies}


def test_a_completed_search_lands_its_alignments(transport, tmp_path):
    """The plain path: submit, poll once, download, unpack."""
    transport["replies"].extend(
        [
            _Reply({"status": "PENDING", "id": "tkt-1"}),
            _Reply({"status": "COMPLETE", "id": "tkt-1"}),
            _Reply(body=_tar_bytes()),
        ]
    )

    out = msa_server.run_remote_msa_search(["MKT"], tmp_path / "q")

    assert out == tmp_path / "q_all", "results land in a directory named for the mode"
    assert (out / "uniref.a3m").read_bytes() == b">101\nMKT\n"
    assert not (out / "out.tar.gz").exists(), "the archive is cleaned up after unpacking"


@pytest.mark.parametrize(
    ("use_env", "use_pairing", "strategy", "mode", "endpoint"),
    [
        (False, False, "greedy", "all", "ticket/msa"),
        (True, False, "greedy", "env", "ticket/msa"),
        (False, True, "greedy", "pairgreedy", "ticket/pair"),
        (True, True, "greedy", "pairgreedy-env", "ticket/pair"),
        # `complete` only pairs a species present in every chain, and the server has its own word
        # for that. Sending greedy regardless would quietly give the caller the other answer.
        (False, True, "complete", "paircomplete", "ticket/pair"),
        (True, True, "complete", "paircomplete-env", "ticket/pair"),
        # Without pairing there is nothing to pair, so the strategy must not reach the wire.
        (False, False, "complete", "all", "ticket/msa"),
    ],
)
def test_the_server_is_asked_for_the_right_search(transport, tmp_path, use_env, use_pairing, strategy, mode, endpoint):
    """Mode strings and endpoints are the server's vocabulary; a wrong one is a silent wrong answer."""
    transport["replies"].extend([_Reply({"status": "COMPLETE", "id": "t"}), _Reply(body=_tar_bytes())])

    out = msa_server.run_remote_msa_search(
        ["MKT"], tmp_path / "q", use_env=use_env, use_pairing=use_pairing, pairing_strategy=strategy
    )

    method, url, kwargs = transport["calls"][0]
    assert method == "POST"
    assert url.endswith(endpoint), f"a {mode} search goes to {endpoint}"
    assert kwargs["data"]["mode"] == mode
    assert out.name.endswith(mode), "the directory is named for the mode the server was asked for"


def test_sequences_are_sent_as_the_fasta_the_server_expects(transport, tmp_path):
    """Identifiers are the server's own counter, starting at 101."""
    transport["replies"].extend([_Reply({"status": "COMPLETE", "id": "t"}), _Reply(body=_tar_bytes())])

    msa_server.run_remote_msa_search(["MKT", "GGS"], tmp_path / "q")

    assert transport["calls"][0][2]["data"]["q"] == ">101\nMKT\n>102\nGGS\n"


def test_every_request_identifies_proto_tools(transport, tmp_path):
    """The server asks callers to identify themselves and warns anonymous use will stop working."""
    transport["replies"].extend([_Reply({"status": "COMPLETE", "id": "t"}), _Reply(body=_tar_bytes())])

    msa_server.run_remote_msa_search(["MKT"], tmp_path / "q")

    assert all("proto-tools" in kwargs["headers"]["User-Agent"] for _m, _u, kwargs in transport["calls"])


def test_a_queued_ticket_is_polled_rather_than_resubmitted(transport, tmp_path):
    """The work is already accepted, so asking again would queue it twice."""
    transport["replies"].extend(
        [
            _Reply({"status": "PENDING", "id": "tkt-1"}),
            _Reply({"status": "RUNNING", "id": "tkt-1"}),
            _Reply({"status": "COMPLETE", "id": "tkt-1"}),
            _Reply(body=_tar_bytes()),
        ]
    )

    msa_server.run_remote_msa_search(["MKT"], tmp_path / "q")

    submits = [url for method, url, _k in transport["calls"] if method == "POST"]
    polls = [url for method, url, _k in transport["calls"] if method == "GET" and "/ticket/" in url]
    assert len(submits) == 1, "a queued ticket must not be submitted again"
    assert polls == [f"{msa_server.MSA_SERVER_URL}/ticket/tkt-1"] * 2


def test_rate_limiting_resubmits_because_the_work_was_refused(transport, tmp_path):
    """A rate-limited request was not accepted, so there is no ticket to poll."""
    transport["replies"].extend(
        [
            _Reply({"status": "RATELIMIT"}),
            _Reply({"status": "COMPLETE", "id": "tkt-2"}),
            _Reply(body=_tar_bytes()),
        ]
    )

    msa_server.run_remote_msa_search(["MKT"], tmp_path / "q")

    submits = [url for method, url, _k in transport["calls"] if method == "POST"]
    assert len(submits) == 2, "the refused work has to be offered again"


def test_bad_input_fails_at_once_rather_than_being_retried(transport, tmp_path):
    """An error will not clear, so waiting only delays the message and spends the server's time."""
    transport["replies"].append(_Reply({"status": "ERROR", "detail": "invalid residue"}))

    with pytest.raises(RuntimeError, match="ERROR"):
        msa_server.run_remote_msa_search(["!!!"], tmp_path / "q")

    assert len(transport["calls"]) == 1, "one round trip, not several"


def test_an_unreadable_reply_is_treated_as_an_error(transport, tmp_path):
    """A gateway page instead of JSON should surface, not raise a parse error from inside."""
    transport["replies"].append(_Reply(payload=None))

    with pytest.raises(RuntimeError, match="ERROR"):
        msa_server.run_remote_msa_search(["MKT"], tmp_path / "q")


def test_a_network_fault_is_retried_but_a_reply_is_not(monkeypatch, tmp_path):
    """A dropped connection is not an answer; a status is."""
    import requests

    attempts: list[int] = []

    def flaky(method: str, url: str, **kwargs: Any) -> _Reply:
        attempts.append(1)
        if len(attempts) == 1:
            raise requests.ConnectionError("reset")
        return _Reply({"status": "COMPLETE", "id": "t"}) if method == "POST" else _Reply(body=_tar_bytes())

    monkeypatch.setattr(msa_server.requests, "request", flaky)
    monkeypatch.setattr(msa_server, "_sleep_with_jitter", lambda _s: None)

    msa_server.run_remote_msa_search(["MKT"], tmp_path / "q")

    assert len(attempts) == 3, "one failed connection, one retry that submits, one download"


def test_an_archive_cannot_write_outside_its_directory(transport, tmp_path):
    """The server is trusted, but an archive is still attacker-shaped input."""
    transport["replies"].extend(
        [_Reply({"status": "COMPLETE", "id": "t"}), _Reply(body=_tar_bytes(name="../escaped.a3m"))]
    )

    with pytest.raises(Exception, match=r"(?i)outside|absolute|escap|traversal|link"):
        msa_server.run_remote_msa_search(["MKT"], tmp_path / "q")

    assert not (tmp_path / "escaped.a3m").exists(), "nothing was written above the destination"


def test_a_ticket_that_never_finishes_gives_up(transport, monkeypatch, tmp_path):
    """A hung ticket must not wait forever holding the call open."""
    transport["replies"].extend([_Reply({"status": "PENDING", "id": "t"}), _Reply({"status": "PENDING", "id": "t"})])

    with pytest.raises(RuntimeError, match="did not finish"):
        msa_server.run_remote_msa_search(["MKT"], Path(tmp_path) / "q", timeout=0.0)


# ── which alignments a search returns ───────────────────────────────────────


def _search_returning(monkeypatch, tmp_path, files: dict[str, bytes], use_metagenomic_db: bool) -> str:
    """Run one unpaired search whose server returns ``files``, and give back what was written."""
    from proto_tools.tools.sequence_alignment.mmseqs2 import remote_search

    def fake_search(sequences, prefix, *, use_env=False, use_pairing=False, client_identity=None, timeout=None):
        landed = Path(f"{prefix}_x")
        landed.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (landed / name).write_bytes(content)
        return landed

    monkeypatch.setattr(remote_search, "run_remote_msa_search", fake_search)
    msas = tmp_path / "msas"
    msas.mkdir()
    written = remote_search._write_unpaired_batch([(0, "MKT")], 0, tmp_path, msas, use_metagenomic_db, None, None)["0"]
    return Path(written).read_text()


def test_a_metagenomic_search_keeps_both_alignments(monkeypatch, tmp_path):
    """The two databases together are the deeper alignment the caller asked for.

    Taking whichever the filesystem listed first would drop UniRef, because ``bfd.*`` sorts ahead
    of ``uniref.*`` — so asking for more depth would quietly return less.
    """
    written = _search_returning(
        monkeypatch,
        tmp_path,
        {"uniref.a3m": b">101\nMKT\n>u1\nMKS\n", "bfd.mgnify30.metaeuk30.smag30.a3m": b">101\nMKT\n>e1\nMKA\n"},
        use_metagenomic_db=True,
    )

    assert ">u1" in written, "the UniRef alignment must survive"
    assert ">e1" in written, "so must the environmental one"
    assert written.index(">u1") < written.index(">e1"), "UniRef first, as the server's own client reads them"


def test_a_plain_search_takes_only_uniref(monkeypatch, tmp_path):
    """Without the metagenomic database there is one alignment, and anything else is not ours."""
    written = _search_returning(
        monkeypatch,
        tmp_path,
        {"uniref.a3m": b">101\nMKT\n>u1\nMKS\n", "bfd.mgnify30.metaeuk30.smag30.a3m": b">101\nMKT\n>e1\nMKA\n"},
        use_metagenomic_db=False,
    )

    assert ">u1" in written
    assert ">e1" not in written, "an environmental alignment nobody asked for must not be folded in"


def test_a_renamed_alignment_still_comes_back(monkeypatch, tmp_path):
    """The server naming its output differently should degrade, not fail the search outright."""
    written = _search_returning(
        monkeypatch, tmp_path, {"something_else.a3m": b">101\nMKT\n>x1\nMKS\n"}, use_metagenomic_db=False
    )

    assert ">x1" in written


# ── one submission for every unpaired query ───────────────────────────────────


def _batched_search(monkeypatch, tmp_path, alignment: bytes, queries: list[dict]) -> tuple[dict, list[list[str]]]:
    """Run ``search_remote_msas`` against a server returning ``alignment``, recording submissions."""
    from proto_tools.tools.sequence_alignment.mmseqs2 import remote_search

    submissions: list[list[str]] = []

    def fake_search(
        sequences,
        prefix,
        *,
        use_env=False,
        use_pairing=False,
        pairing_strategy="greedy",
        client_identity=None,
        timeout=None,
    ):
        submissions.append(list(sequences))
        landed = Path(f"{prefix}_x")
        landed.mkdir(parents=True, exist_ok=True)
        (landed / "uniref.a3m").write_bytes(alignment)
        return landed

    monkeypatch.setattr(remote_search, "run_remote_msa_search", fake_search)
    result = remote_search.search_remote_msas(queries, tmp_path / "out")
    return result, submissions


def test_every_unpaired_query_goes_up_in_one_submission(monkeypatch, tmp_path):
    """Three queries must cost one ticket, not three."""
    alignment = b">101\nMKT\n>u1\nMKS\n\x00>102\nAAA\n>u2\nAAG\n\x00>103\nCCC\n>u3\nCCG\n"
    result, submissions = _batched_search(
        monkeypatch,
        tmp_path,
        alignment,
        [{"sequences": "MKT"}, {"sequences": "AAA"}, {"sequences": "CCC"}],
    )

    assert len(submissions) == 1, f"expected a single submission, got {len(submissions)}"
    assert submissions[0] == ["MKT", "AAA", "CCC"], "all queries go up together, in input order"
    assert sorted(result["msa_paths"]) == ["0", "1", "2"], "each query still gets its own alignment"


def test_each_query_gets_the_block_answering_it(monkeypatch, tmp_path):
    """Blocks come back in one file, so each query must receive its own — not the first one."""
    alignment = b">101\nMKT\n>u1\nMKS\n\x00>102\nAAA\n>u2\nAAG\n"
    result, _ = _batched_search(monkeypatch, tmp_path, alignment, [{"sequences": "MKT"}, {"sequences": "AAA"}])

    assert ">u1" in Path(result["msa_paths"]["0"]).read_text()
    assert ">u2" in Path(result["msa_paths"]["1"]).read_text()
    assert ">u2" not in Path(result["msa_paths"]["0"]).read_text(), "queries must not receive each other's blocks"


def test_duplicate_queries_share_one_deduplicated_block(monkeypatch, tmp_path):
    """The submission deduplicates, so two identical queries come back as one block.

    Mapping blocks by position would leave the second query unanswered; mapping by sequence
    gives both the alignment that answers them.
    """
    alignment = b">101\nMKT\n>u1\nMKS\n\x00>102\nAAA\n>u2\nAAG\n"
    result, submissions = _batched_search(
        monkeypatch,
        tmp_path,
        alignment,
        [{"sequences": "MKT"}, {"sequences": "AAA"}, {"sequences": "MKT"}],
    )

    assert len(submissions) == 1
    assert sorted(result["msa_paths"]) == ["0", "1", "2"], "the repeated query is still answered"
    assert Path(result["msa_paths"]["0"]).read_text() == Path(result["msa_paths"]["2"]).read_text()


def test_a_failed_batch_is_recorded_against_every_query_in_it(monkeypatch, tmp_path):
    """Sharing a submission means sharing its failure, and each query must still be accounted for."""
    from proto_tools.tools.sequence_alignment.mmseqs2 import remote_search

    def failing(*args, **kwargs):
        raise RuntimeError("server said no")

    monkeypatch.setattr(remote_search, "run_remote_msa_search", failing)
    result = remote_search.search_remote_msas([{"sequences": "MKT"}, {"sequences": "AAA"}], tmp_path / "out")

    assert result["success"] is False
    assert result["num_failed"] == 2, "both queries are reported, not just the batch"
    assert set(result["errors"]) == {"query_0", "query_1"}


def test_a_large_batch_is_split_across_submissions(monkeypatch, tmp_path):
    """Past the cap the queries are split, so one submission never grows without bound."""
    from proto_tools.tools.sequence_alignment.mmseqs2 import remote_search

    cap = remote_search.MAX_SEQUENCES_PER_SUBMISSION
    sequences = [f"MKT{i:04d}" for i in range(cap + 5)]
    submissions: list[list[str]] = []

    def fake_search(
        seqs, prefix, *, use_env=False, use_pairing=False, pairing_strategy="greedy", client_identity=None, timeout=None
    ):
        submissions.append(list(seqs))
        landed = Path(f"{prefix}_x")
        landed.mkdir(parents=True, exist_ok=True)
        blocks = [f">1\n{s}\n>h\n{s}\n".encode() for s in seqs]
        (landed / "uniref.a3m").write_bytes(b"\x00".join(blocks))
        return landed

    monkeypatch.setattr(remote_search, "run_remote_msa_search", fake_search)
    result = remote_search.search_remote_msas([{"sequences": s} for s in sequences], tmp_path / "out")

    assert len(submissions) == 2, f"expected the batch to split at {cap}, got {len(submissions)} submission(s)"
    assert [len(s) for s in submissions] == [cap, 5]
    assert len(result["msa_paths"]) == cap + 5, "every query is still answered"


def test_one_failed_submission_leaves_the_others_alone(monkeypatch, tmp_path):
    """Splitting bounds a failure too: only the queries sharing that submission are lost."""
    from proto_tools.tools.sequence_alignment.mmseqs2 import remote_search

    cap = remote_search.MAX_SEQUENCES_PER_SUBMISSION
    sequences = [f"MKT{i:04d}" for i in range(cap + 3)]
    calls = {"n": 0}

    def fake_search(
        seqs, prefix, *, use_env=False, use_pairing=False, pairing_strategy="greedy", client_identity=None, timeout=None
    ):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second submission refused")
        landed = Path(f"{prefix}_x")
        landed.mkdir(parents=True, exist_ok=True)
        blocks = [f">1\n{s}\n>h\n{s}\n".encode() for s in seqs]
        (landed / "uniref.a3m").write_bytes(b"\x00".join(blocks))
        return landed

    monkeypatch.setattr(remote_search, "run_remote_msa_search", fake_search)
    result = remote_search.search_remote_msas([{"sequences": s} for s in sequences], tmp_path / "out")

    assert len(result["msa_paths"]) == cap, "the first submission's queries survive"
    assert result["num_failed"] == 3, "only the failed submission's queries are lost"


def test_a_submission_missing_one_query_writes_nothing(monkeypatch, tmp_path):
    """A submission lands whole or not at all — no alignments for queries reported as failed.

    Writing as the loop goes would leave files behind for the queries resolved before the missing
    one, which anything reading the directory rather than ``msa_paths`` would read as successes.
    """
    from proto_tools.tools.sequence_alignment.mmseqs2 import remote_search

    def fake_search(
        seqs, prefix, *, use_env=False, use_pairing=False, pairing_strategy="greedy", client_identity=None, timeout=None
    ):
        landed = Path(f"{prefix}_x")
        landed.mkdir(parents=True, exist_ok=True)
        # The server answers the first two queries but not the third.
        blocks = [b">1\nAAA\n>h\nAAG\n", b">2\nCCC\n>h\nCCG\n"]
        (landed / "uniref.a3m").write_bytes(b"\x00".join(blocks))
        return landed

    monkeypatch.setattr(remote_search, "run_remote_msa_search", fake_search)
    out = tmp_path / "out"
    result = remote_search.search_remote_msas([{"sequences": "AAA"}, {"sequences": "CCC"}, {"sequences": "TTT"}], out)

    assert result["num_failed"] == 3, "the whole submission fails together"
    assert list((out / "msas").glob("*.a3m")) == [], "no alignment survives a submission that failed"


def test_a_metagenomic_batch_joins_each_query_across_both_alignments(monkeypatch, tmp_path):
    """With the environmental database, both files are NUL-separated per query and joined per query.

    The single-alignment tests cannot catch a mismatch here: each query's alignment is the UniRef
    block *and* the environmental block for that same query, so a wrong pairing would silently give
    a query someone else's environmental depth.
    """
    from proto_tools.tools.sequence_alignment.mmseqs2 import remote_search

    def fake_search(
        seqs, prefix, *, use_env=False, use_pairing=False, pairing_strategy="greedy", client_identity=None, timeout=None
    ):
        assert use_env, "this batch asked for the environmental database"
        landed = Path(f"{prefix}_x")
        landed.mkdir(parents=True, exist_ok=True)
        uniref = [f">1\n{s}\n>u_{s}\n{s}\n".encode() for s in seqs]
        env = [f">1\n{s}\n>e_{s}\n{s}\n".encode() for s in seqs]
        (landed / "uniref.a3m").write_bytes(b"\x00".join(uniref))
        (landed / "bfd.mgnify30.metaeuk30.smag30.a3m").write_bytes(b"\x00".join(env))
        return landed

    monkeypatch.setattr(remote_search, "run_remote_msa_search", fake_search)
    result = remote_search.search_remote_msas(
        [{"sequences": "AAA"}, {"sequences": "CCC"}], tmp_path / "out", use_metagenomic_db=True
    )

    first = Path(result["msa_paths"]["0"]).read_text()
    second = Path(result["msa_paths"]["1"]).read_text()

    assert ">u_AAA" in first and ">e_AAA" in first, "query 0 keeps both of its own alignments"
    assert ">u_CCC" in second and ">e_CCC" in second, "query 1 keeps both of its own alignments"
    assert ">e_CCC" not in first, "a query must not receive another query's environmental depth"
    assert first.index(">u_AAA") < first.index(">e_AAA"), "UniRef first, as the server's client reads them"

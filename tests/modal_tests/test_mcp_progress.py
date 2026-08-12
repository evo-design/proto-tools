"""A caller with no terminal still gets the worker's progress.

Modal workers publish progress records the whole time a tool runs, and a local session replays
them into its spinner. An MCP server has no spinner, so the same records went to a logger and out
to stderr, which no MCP client renders: every call was silent until it returned, including the
cold starts and multi-minute runs the tool descriptions warn about.

Supplying a consumer both redirects the records and turns streaming on, because none of the
conditions that enable it can be true in a process nobody is watching. These cover that the
switch works, and -- just as important -- that it stays off for everyone who did not ask.
"""

from __future__ import annotations

import logging
from typing import Any

from proto_tools.utils.base_config import BaseConfig


def _headless(monkeypatch) -> None:
    """Make the process look like an MCP server: no spinner, no verbosity, no queue."""
    from proto_tools.modal import client

    monkeypatch.setattr(client, "has_active_progress_bar", lambda: False)
    monkeypatch.setattr(client, "verbose_level_from_env", lambda: 0)
    monkeypatch.setattr(client, "open_progress_queue", lambda **_kwargs: None)
    monkeypatch.setattr(client, "set_substatus", lambda *_a, **_k: None)


def _record_tailer(monkeypatch) -> list[dict[str, Any]]:
    """Replace the tailer thread's target, capturing how it was called."""
    from proto_tools.modal import client

    calls: list[dict[str, Any]] = []

    def fake_stream(partition, expected_ends, stop, **kwargs):
        calls.append({"partition": partition, "expected_ends": expected_ends, **kwargs})

    monkeypatch.setattr(client, "stream_modal_progress", fake_stream)
    return calls


def test_a_consumer_turns_streaming_on_with_no_terminal(monkeypatch):
    """The load-bearing case: without this the whole path stays dark for an MCP caller.

    No spinner and verbose=0 is exactly an MCP server, and it is the combination that used to
    return before starting anything. The stamp matters as much as the thread: the partition is
    only sent to the worker when a config carries it, so an unstamped config means the worker
    publishes nothing to tail in the first place.
    """
    from proto_tools.modal import client

    _headless(monkeypatch)
    calls = _record_tailer(monkeypatch)
    config = BaseConfig(verbose=0)

    with client._live_progress([config], expected_ends=1, on_record=lambda _record: None):
        # Read inside the block: the partition is cleared again on the way out.
        stamped = config._progress_partition

    assert stamped, "config was not stamped, so the worker would publish nothing"
    assert len(calls) == 1, f"tailer was not started: {calls}"
    assert calls[0]["partition"] == stamped, "tailer listened on a different partition than the worker writes"


def test_records_reach_the_consumer_rather_than_the_local_replay(monkeypatch):
    """The consumer has to be handed down to the thread, or it tails into the local logger."""
    from proto_tools.modal import client

    _headless(monkeypatch)
    calls = _record_tailer(monkeypatch)
    sink: list[dict[str, Any]] = []

    def consume(record: dict[str, Any]) -> None:
        sink.append(record)

    with client._live_progress([BaseConfig(verbose=0)], expected_ends=1, on_record=consume):
        pass

    assert calls[0]["on_record"] is consume, "the tailer would replay locally instead of forwarding"


def test_the_cold_start_is_reported_to_the_consumer(monkeypatch):
    """The container wait is the longest silence, and it is announced before anything is tailed.

    Locally that phase goes to the spinner. A consumer replacing the replay never sees the
    spinner, so it has to be handed the same phase directly or the longest wait is the one part
    that still says nothing.
    """
    from proto_tools.modal import client

    _headless(monkeypatch)
    _record_tailer(monkeypatch)
    sink: list[dict[str, Any]] = []

    with client._live_progress([BaseConfig(verbose=0)], expected_ends=1, on_record=sink.append):
        pass

    assert sink, "the cold start was never reported"
    assert sink[0].get("m"), "the opening record carries no message"


def test_without_a_consumer_nothing_changes(monkeypatch):
    """The isolation guard: this must alter the MCP surface and nothing else.

    Everything outside the MCP calls with no consumer, so a headless, quiet, unwatched dispatch
    has to behave exactly as it did -- no thread, no stamp, no queue. Fails the moment someone
    makes streaming unconditional.
    """
    from proto_tools.modal import client

    _headless(monkeypatch)
    calls = _record_tailer(monkeypatch)
    config = BaseConfig(verbose=0)

    with client._live_progress([config], expected_ends=1):
        stamped = config._progress_partition

    assert stamped is None, "a caller that asked for nothing had its config stamped"
    assert calls == [], "a caller that asked for nothing started a tailer"


async def test_run_tool_forwards_records_as_mcp_progress(monkeypatch):
    """End of the chain: records produced mid-call arrive as progress notifications.

    They are emitted from a plain thread on purpose. The tailer is one, and a task scheduled onto
    the loop from a thread is created outside this request's context -- where the progress token
    lives -- so reporting from there reaches nobody while still reporting success. Emitting from
    the test's own thread is what keeps that from coming back unnoticed.
    """
    import threading

    from proto_tools.mcp import server as server_module

    sent: list[tuple[float, str | None]] = []

    class _Ctx:
        async def report_progress(self, progress: float, message: str | None = None) -> None:
            sent.append((progress, message))

    def fake_run_tool(*_args: Any, on_record=None, **_kwargs: Any) -> dict[str, Any]:
        def tail() -> None:
            on_record({"m": "loading weights", "l": logging.INFO})
            on_record({"m": "generating", "l": logging.INFO})
            on_record({"l": logging.INFO})  # no message: nothing to show, so nothing is sent

        thread = threading.Thread(target=tail)
        thread.start()
        thread.join()
        return {"ok": True, "tool": "evo2-sample"}

    monkeypatch.setattr(server_module.impl, "run_tool", fake_run_tool)

    mcp = server_module.build_server("modal")
    run_tool = (await mcp.get_tool("run_tool")).fn
    answer = await run_tool(_Ctx(), "evo2-sample", inputs={"prompts": ["ATG"]})

    assert answer["ok"] is True
    assert [message for _progress, message in sent] == [
        "Running evo2-sample",
        "loading weights",
        "generating",
    ]
    # Counted rather than fractional: records say what is happening, never how much is left.
    assert [progress for progress, _message in sent] == [1, 2, 3]


async def test_the_tool_is_named_before_it_starts(monkeypatch):
    """The first thing a client can show should say which tool is running.

    A client displays the latest progress message as the state of the call, so with nothing sent
    up front the label is whatever the client invents. It also has to come from here rather than
    from a worker's records: a tool answered in this process produces none at all.
    """
    from proto_tools.mcp import server as server_module

    sent: list[str | None] = []

    class _Ctx:
        async def report_progress(self, progress: float, message: str | None = None) -> None:
            sent.append(message)

    monkeypatch.setattr(
        server_module.impl,
        "run_tool",
        lambda *_a, on_record=None, **_k: {"ok": True},
    )

    mcp = server_module.build_server("local")
    run_tool = (await mcp.get_tool("run_tool")).fn
    await run_tool(_Ctx(), "viennarna-prediction", inputs={"sequences": ["GCGC"]})

    assert sent == ["Running viennarna-prediction"], "a silent tool never named itself"

"""``local_only`` and the deployment map must not disagree about what can run remotely.

``local_only`` is the declaration that a tool must never have a deployment. ``local_cpu`` is a
weaker, derived statement — no GPU and no standalone environment — which means a deployment is
*unnecessary*, not forbidden. A local_cpu tool with a large iterable input can still be worth
fanning out, so the two must not be conflated.

The rules enforced here run in both directions: a local_only tool must have no deployment, and
a tool that is not local_only must have one.
"""

from __future__ import annotations

from pathlib import Path

import proto_tools
from proto_tools.modal.tool_map import TOOL_MAP
from proto_tools.tools import ToolRegistry, ToolSpec

MOCK_TOOLKITS_DIR = Path(proto_tools.__file__).resolve().parent / "tools" / "testing"


def real_tools() -> list[ToolSpec]:
    """Registered tools, minus the mock toolkits a fixture imports into the session part-way through.

    ``proto_tools.tools.testing`` holds stand-ins used to exercise the runtime itself. They are
    registered only once some test imports them, so they are absent from a plain registry listing
    and present during a full run. Neither is a deployment question.
    """
    return [
        spec for spec in ToolRegistry.list_all() if MOCK_TOOLKITS_DIR not in Path(spec.source_file).resolve().parents
    ]


# Tools we deliberately do not host, though nothing stops them running remotely. This is a
# statement about our own deployment, not about the tool: a caller with their own Modal workspace
# can still deploy one, which is why these are not local_only.
EXCEPTIONS: dict[str, str] = {
    "blast-search": "The reference databases are far too large to host.",
}


def test_local_only_tools_have_no_deployment() -> None:
    """A tool that declares it can never run remotely must not be in the dispatch table."""
    deployed = sorted(spec.key for spec in ToolRegistry.list_all() if spec.local_only and spec.key in TOOL_MAP)
    assert not deployed, (
        f"Tools declare local_only but are in TOOL_MAP: {deployed}. Either remove the deployment "
        f"(delete its service and manifest entries, then regenerate the tool map) or drop the "
        f"local_only declaration if the tool really can run remotely."
    )


def test_every_remotely_runnable_tool_has_a_deployment() -> None:
    """A tool that does not declare local_only must be reachable in the dispatch table.

    Nothing else notices a missing deployment. The tool registers, runs in process, and passes
    its own tests, while ``device="modal"`` and every hosted caller have nowhere to send it.
    Twelve gradient tools accumulated that way before anyone looked.

    A tool that must never run remotely says so with ``local_only``. One we simply choose not to
    host belongs in ``EXCEPTIONS``, which keeps that choice deliberate and visible.
    """
    missing = sorted(
        spec.key
        for spec in real_tools()
        if not spec.local_only and spec.key not in TOOL_MAP and spec.key not in EXCEPTIONS
    )
    assert not missing, (
        f"Registered tools have no Modal deployment: {missing}. Add a method to the toolkit's "
        f"service and regenerate the tool map (scripts/generate_tool_map.py), declare local_only "
        f"if the tool must never run remotely, or add it to EXCEPTIONS with the reason."
    )


def test_deployment_exceptions_are_still_exceptions() -> None:
    """A tool we exempt from hosting must still be registered, and still be undeployed."""
    deployed = sorted(key for key in EXCEPTIONS if key in TOOL_MAP)
    assert not deployed, f"EXCEPTIONS entries now have deployments and must be removed: {deployed}."

    registered = {spec.key for spec in real_tools()}
    unregistered = sorted(key for key in EXCEPTIONS if key not in registered)
    assert not unregistered, f"EXCEPTIONS names tools that no longer exist: {unregistered}."


def test_deployed_local_cpu_tools_are_worth_fanning_out() -> None:
    """A deployed local_cpu tool must be iterable with a real chunk size, or the hop buys nothing.

    Such a tool has no model to keep warm and no environment to build, so the only thing a container
    offers is a share of a batch. Chunking one item at a time would spend a round trip per item and
    lose to running in-process, which is the outcome this guards against.

    ``pdockq2`` is the deliberate exception: it takes a single structure and is deployed so a caller
    on a remote device can reach it beside the tools that do fan out.
    """
    exempt = {"pdockq2"}
    bad = sorted(
        f"{spec.key} (iterable={bool(spec.iterable_input_fields)}, chunk={spec.max_chunk_size})"
        for spec in ToolRegistry.list_all()
        if spec.local_cpu
        and spec.key in TOOL_MAP
        and spec.key not in exempt
        and not (spec.iterable_input_fields and (spec.max_chunk_size or 1) > 1)
    )
    assert not bad, (
        f"Deployed local_cpu tools that cannot fan out: {bad}. Give the tool iterable_input_fields "
        f"and a max_chunk_size above 1, or do not deploy it — a container offers it nothing else."
    )

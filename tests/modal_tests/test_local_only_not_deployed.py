"""``local_only`` and the deployment map must not disagree about what can run remotely.

``local_only`` is the declaration that a tool must never have a deployment. ``local_cpu`` is a
weaker, derived statement — no GPU and no standalone environment — which means a deployment is
*unnecessary*, not forbidden. A local_cpu tool with a large iterable input can still be worth
fanning out, so the two must not be conflated: the only rule enforced here is the first one.
"""

from __future__ import annotations

from proto_tools.modal.tool_map import TOOL_MAP
from proto_tools.tools import ToolRegistry


def test_local_only_tools_have_no_deployment() -> None:
    """A tool that declares it can never run remotely must not be in the dispatch table."""
    deployed = sorted(spec.key for spec in ToolRegistry.list_all() if spec.local_only and spec.key in TOOL_MAP)
    assert not deployed, (
        f"Tools declare local_only but are in TOOL_MAP: {deployed}. Either remove the deployment "
        f"(delete its service and manifest entries, then regenerate the tool map) or drop the "
        f"local_only declaration if the tool really can run remotely."
    )


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

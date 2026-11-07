"""proto_tools/utils/outage.py.

Upstream outages declared on a tool: a whole tool, or one mode of it, that cannot be served
because a service it depends on is down or has restricted access. A matching call is refused
before dispatch with :class:`OutageError`, raised the way a validation error is.

The check belongs to the entry point (the caller's wrapper, the MCP, the hosted API). A Modal
container skips it (:func:`outage_checks_skipped`): it cannot see the caller's override, and an
exception raised there would come back opaque and be retried rather than refused.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

IGNORE_OUTAGES_ENV_VAR = "PROTO_IGNORE_OUTAGES"

_checks_skipped: ContextVar[bool] = ContextVar("outage_checks_skipped", default=False)


class Outage(BaseModel):
    """An upstream outage affecting a tool, or the calls that reach one mode of it.

    Attributes:
        reason (str): What is down, in the upstream's terms.
        since (date): When the outage began.
        tracking_url (str): Where its status is tracked (upstream issue or status page).
        affects (str | None): Label for the affected mode (e.g. ``"include_msa"``);
            ``None`` when the whole tool is down.
        when (Callable[[Any, Any], bool] | None): Given ``(inputs, config)``, whether a call
            reaches the affected mode. Requires ``affects``; never serialized, so an outage
            rebuilt from a dump carries the label without it and matches nothing.
    """

    reason: str = Field(title="Reason", description="What is down, in the upstream's terms")
    since: date = Field(title="Since", description="When the outage began")
    tracking_url: str = Field(title="Tracking URL", description="Upstream issue or status page tracking the outage")
    affects: str | None = Field(
        default=None, title="Affects", description="Affected mode of the tool; None when the whole tool is down"
    )
    when: Callable[[Any, Any], bool] | None = Field(
        default=None,
        exclude=True,
        title="When",
        description="Given (inputs, config), whether a call reaches the affected mode",
    )

    @model_validator(mode="after")
    def _predicate_is_labelled(self) -> Outage:
        """Require ``affects`` with ``when``, so a catalogue can name the mode a predicate tests.

        Not the converse: ``when`` is excluded from dumps, and a dump must rebuild. A declaration
        missing its predicate is caught by ``tests/style_consistency_tests/test_outage_declarations.py``.
        """
        if self.when is not None and self.affects is None:
            raise ValueError("Outage.when requires Outage.affects")
        return self

    def describe(self) -> str:
        """One-line description shared by the error message, MCP notes, and CLI."""
        scope = f"affects {self.affects}" if self.affects else "affects every call"
        return f"{self.reason} (since {self.since.isoformat()}; {scope}). Tracking: {self.tracking_url}"


class OutageError(ValueError):
    """Raised before dispatch when a call matches a declared upstream outage.

    A ``ValueError``, like pydantic's ``ValidationError``: the request cannot be served as asked,
    and it is refused before anything is built or started.

    Attributes:
        tool_key (str): Registry key of the refused tool.
        outage (Outage): The outage the call matched.
    """

    def __init__(self, tool_key: str, outage: Outage):
        """Initialize OutageError."""
        self.tool_key = tool_key
        self.outage = outage
        super().__init__(f"{tool_key}: {outage.describe()}. Set {IGNORE_OUTAGES_ENV_VAR}=1 to try anyway.")

    def __reduce__(self) -> tuple[type[OutageError], tuple[str, Outage]]:
        """Pickle with the outage rebuilt from its dump, since the ``when`` predicate cannot be pickled."""
        return type(self), (self.tool_key, Outage(**self.outage.model_dump()))


@contextmanager
def outage_checks_skipped() -> Iterator[None]:
    """Skip outage checks within this context, for a Modal container running a call already checked."""
    token = _checks_skipped.set(True)
    try:
        yield
    finally:
        _checks_skipped.reset(token)


def _reaches(tool_key: str, outage: Outage, inputs: Any, config: Any) -> bool:
    """Whether a call reaches ``outage``; a predicate that raises is logged and counts as no.

    A broken predicate is a bug in the declaration, so it must not refuse a call the tool could
    serve. Letting the call through means the worst case is the upstream's own error.
    """
    if outage.affects is None:
        return True
    if outage.when is None:
        return False
    try:
        return bool(outage.when(inputs, config))
    except Exception:
        logger.exception(
            "Tool %s: outage predicate raised; treating the call as unaffected (%s)", tool_key, outage.affects
        )
        return False


def active_outage(tool_key: str, outages: Iterable[Outage], inputs: Any, config: Any) -> Outage | None:
    """Return the first outage this call matches, or ``None``.

    With ``PROTO_IGNORE_OUTAGES=1`` a matching outage is logged and ignored, so a caller can
    check whether the upstream has recovered before the declaration is lifted. Inside
    :func:`outage_checks_skipped` nothing is checked.

    Args:
        tool_key (str): Registry key, for the log message.
        outages (Iterable[Outage]): The tool's declared outages.
        inputs (Any): The call's inputs.
        config (Any): The call's config.

    Returns:
        Outage | None: The matching outage, or ``None`` when none matches or outages are ignored.
    """
    if _checks_skipped.get():
        return None
    matched = next((o for o in outages if _reaches(tool_key, o, inputs, config)), None)
    if matched is not None and os.environ.get(IGNORE_OUTAGES_ENV_VAR) == "1":
        logger.warning(
            "Tool %s: ignoring declared outage (%s=1): %s", tool_key, IGNORE_OUTAGES_ENV_VAR, matched.describe()
        )
        return None
    return matched

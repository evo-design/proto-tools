"""proto_tools/utils/outage.py.

Upstream outages declared on a tool: a whole tool, or one mode of it, that cannot be served
because a service it depends on is down or has restricted access. A matching call is refused
before dispatch with :class:`OutageError`, raised the way a validation error is.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from datetime import date
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

IGNORE_OUTAGES_ENV_VAR = "PROTO_IGNORE_OUTAGES"


class Outage(BaseModel):
    """An upstream outage affecting a tool, or the calls that reach one mode of it.

    Attributes:
        reason (str): What is down, in the upstream's terms.
        since (date): When the outage began.
        tracking_url (str): Where its status is tracked (upstream issue or status page).
        affects (str | None): Label for the affected mode (e.g. ``"include_msa"``);
            ``None`` when the whole tool is down.
        when (Callable[[Any, Any], bool] | None): Given ``(inputs, config)``, whether a call
            reaches the affected mode. Set exactly when ``affects`` is.
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
    def _scope_is_labelled_and_testable(self) -> Outage:
        """Require ``affects`` and ``when`` together: a scoped outage needs a label and a test."""
        if (self.affects is None) != (self.when is None):
            raise ValueError("Outage.affects and Outage.when must be set together")
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


def active_outage(tool_key: str, outages: Iterable[Outage], inputs: Any, config: Any) -> Outage | None:
    """Return the first outage this call matches, or ``None``.

    With ``PROTO_IGNORE_OUTAGES=1`` a matching outage is logged and ignored, so a caller can
    check whether the upstream has recovered before the declaration is lifted.

    Args:
        tool_key (str): Registry key, for the log message.
        outages (Iterable[Outage]): The tool's declared outages.
        inputs (Any): The call's inputs.
        config (Any): The call's config.

    Returns:
        Outage | None: The matching outage, or ``None`` when none matches or outages are ignored.
    """
    matched = next((o for o in outages if o.when is None or o.when(inputs, config)), None)
    if matched is not None and os.environ.get(IGNORE_OUTAGES_ENV_VAR) == "1":
        logger.warning(
            "Tool %s: ignoring declared outage (%s=1): %s", tool_key, IGNORE_OUTAGES_ENV_VAR, matched.describe()
        )
        return None
    return matched

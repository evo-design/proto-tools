"""tests/style_consistency_tests/test_outage_declarations.py.

Every declared outage must be checkable and traceable.

An outage's ``when`` predicate only runs when a call arrives, so a typo in it (a misspelled
config field) would surface as a crash in a user's call instead of a refusal. Evaluating it
here on the tool's example input and default config catches that at review time. The tracking
link and start date are what someone lifting the outage later needs to check its status.
"""

from __future__ import annotations

from datetime import date

import pytest

from proto_tools.tools import ToolRegistry

_DECLARED = [(spec.key, i) for spec in ToolRegistry.list_all() for i in range(len(spec.outages))]


@pytest.mark.parametrize("key,index", _DECLARED, ids=[f"{k}[{i}]" for k, i in _DECLARED])
def test_outage_predicate_evaluates_on_the_example_call(key: str, index: int):
    """``when`` returns a bool for the example input and default config, rather than raising."""
    spec = ToolRegistry.get(key)
    outage = spec.outages[index]
    if outage.when is None:
        return
    example = ToolRegistry.get_example_input(key)
    assert example is not None, f"{key}: a scoped outage needs an example_input to check its predicate against"
    result = outage.when(example, spec.config_model())
    assert isinstance(result, bool), f"{key}: Outage.when returned {type(result).__name__}, expected bool"


@pytest.mark.parametrize("key,index", _DECLARED, ids=[f"{k}[{i}]" for k, i in _DECLARED])
def test_outage_is_traceable(key: str, index: int):
    """A tracking link to check before lifting it, and a start date that has happened."""
    outage = ToolRegistry.get(key).outages[index]
    assert outage.tracking_url.startswith(("https://", "http://")), f"{key}: tracking_url is not a URL"
    assert outage.since <= date.today(), f"{key}: outage starts in the future ({outage.since})"

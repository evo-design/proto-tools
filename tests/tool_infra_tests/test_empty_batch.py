"""An empty batch is refused before anything is built, started, or billed.

Only the primary iterable field is required. The parallel siblings that follow it — ``msas``
alongside ``complexes`` — are optional by design, and requiring them would reject an ordinary
call from someone with no alignments to supply.
"""

from __future__ import annotations

import pytest

from proto_tools.tools import ToolRegistry
from proto_tools.tools.tool_registry import _reject_empty_batch


def _spec(key: str):
    return ToolRegistry.get(key)


def test_an_empty_primary_field_is_refused():
    """Otherwise it reaches the tool and fails there, after a container start on a remote device."""
    spec = _spec("esm2-embedding")
    with pytest.raises(ValueError) as caught:
        _reject_empty_batch("esm2-embedding", spec, spec.input_model(sequences=[]))
    assert "'sequences' is empty" in str(caught.value)


def test_an_empty_parallel_sibling_is_allowed():
    """A structure prediction with no supplied alignments is a normal call, not an error."""
    spec = _spec("protenix-prediction")
    inputs = spec.input_model(complexes=["MKTAYLLIGLLAIAAFSPQVLA"], msas=[])
    _reject_empty_batch("protenix-prediction", spec, inputs)  # must not raise


def test_a_populated_batch_passes():
    spec = _spec("esm2-embedding")
    _reject_empty_batch("esm2-embedding", spec, spec.input_model(sequences=["MKTL"]))


def test_a_tool_with_no_iterable_fields_is_untouched():
    """Single-input tools have no batch to be empty."""
    spec = _spec("tmalign-alignment")
    assert not spec.iterable_input_fields
    _reject_empty_batch("tmalign-alignment", spec, ToolRegistry.get_example_input("tmalign-alignment"))


def test_an_unregistered_tool_is_untouched():
    """``spec`` is None for a tool called outside the registry; the guard must not assume one."""
    _reject_empty_batch("whatever", None, object())


def test_the_guard_runs_through_the_wrapper():
    """The check has to fire on the real call path, not only when invoked directly."""
    from proto_tools import ESM2EmbeddingsConfig, ESM2EmbeddingsInput, run_esm2_embeddings

    with pytest.raises(ValueError, match="is empty"):
        run_esm2_embeddings(ESM2EmbeddingsInput(sequences=[]), ESM2EmbeddingsConfig(device="cpu"))

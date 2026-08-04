"""What a config carries to a Modal worker.

The worker re-enters the proto-tools tool wrapper, so it would repeat any ``preprocess`` the
caller already ran. proto-tools records that on the config it executes with, and the client has to
forward it, which ``model_dump`` does not do. Callers that never preprocess, notably the MCP
surface, must report ``preprocess_completed`` as ``False``, since worker-side preprocess is the
only preprocess they get.

Offline: no Modal calls, no network.
"""

from __future__ import annotations

import pytest

from proto_tools.tools import ToolRegistry
from proto_tools.utils.base_config import INTERNAL_STATE_KEY, run_preprocess

TOOL_KEY = "esmfold-prediction"


@pytest.fixture
def models():
    """The config and input models of a deployed, preprocessing tool."""
    spec = ToolRegistry.get(TOOL_KEY)
    return spec.config_model, spec.input_model


def test_a_prepared_config_carries_its_state(models):
    """A config the wrapper already preprocessed tells the worker so."""
    config_model, input_model = models

    _, prepared = run_preprocess(config_model(), input_model(complexes=[]))

    assert prepared.to_transport_dict()[INTERNAL_STATE_KEY]["preprocess_completed"] is True


def test_an_unprepared_config_says_so(models):
    """A caller that never preprocessed reports it, so the worker still does the work."""
    config_model, _ = models

    assert config_model().to_transport_dict()[INTERNAL_STATE_KEY]["preprocess_completed"] is False


def test_model_dump_would_lose_the_state(models):
    """``model_dump`` drops it, which is why the client must not use it for dispatch.

    Pins the reason for the call in ``client.py``: switching back would silently reintroduce a
    second preprocess on the worker rather than fail.
    """
    config_model, input_model = models

    _, prepared = run_preprocess(config_model(), input_model(complexes=[]))

    assert INTERNAL_STATE_KEY not in prepared.model_dump(mode="json")
    assert INTERNAL_STATE_KEY in prepared.to_transport_dict()


def test_the_client_dispatches_with_the_transport_encoding():
    """Both dispatch forms send what ``to_transport_dict`` produces, not a plain dump.

    Checks the two functions rather than counting occurrences of a literal, since the batch form
    serializes inside a comprehension and its spelling changes with the surrounding code.
    """
    import inspect
    import re

    from proto_tools.modal import client

    for fn in (client.dispatch_to_modal, client.dispatch_batch_to_modal):
        body = inspect.getsource(fn)
        assert "to_transport_dict(" in body, f"{fn.__name__} must carry framework state"
        # Only a *config* dumped this way is the regression; inputs are dumped plainly on purpose.
        dumped_configs = re.findall(r"\b(config|cfg|one)\.model_dump\(", body)
        assert not dumped_configs, f"{fn.__name__} serializes a config with model_dump, which drops framework state"


# ============================================================================
# Live progress rides the same envelope
# ============================================================================
def test_a_progress_partition_travels_and_comes_back(models):
    """The worker learns where to stream without any tool's method signature changing."""
    import logging

    config_model, _ = models
    config = config_model()
    config._progress_partition = "abc123"
    config._progress_level = logging.DEBUG

    restored = config_model.model_validate(config.to_transport_dict())

    assert restored._progress_partition == "abc123"
    assert restored._progress_level == logging.DEBUG


def test_a_config_nobody_is_tailing_sends_no_progress_keys(models):
    """Absent rather than null, so a worker predating the key reads an unchanged envelope."""
    config_model, _ = models

    envelope = config_model().to_transport_dict()[INTERNAL_STATE_KEY]

    assert "progress_partition" not in envelope
    assert "progress_level" not in envelope


def test_a_worker_predating_the_key_is_unaffected(models):
    """Unknown envelope keys are ignored, which is what lets the client roll out first."""
    config_model, _ = models

    payload = config_model().to_transport_dict()
    payload[INTERNAL_STATE_KEY]["some_future_key"] = "value"

    restored = config_model.model_validate(payload)  # must not raise
    assert restored._progress_partition is None


def test_the_partition_never_reaches_the_cache_key(models):
    """Two identical calls must share a cache entry regardless of who was watching."""
    config_model, _ = models
    watched, unwatched = config_model(), config_model()
    watched._progress_partition = "abc123"

    assert watched.cache_key() == unwatched.cache_key()

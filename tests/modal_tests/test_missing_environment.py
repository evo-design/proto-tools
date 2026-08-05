"""A Modal environment that was never created says so, instead of reading as "not deployed".

Modal reports a missing environment and a missing app identically. Left undistinguished, a
workspace that has not run the one-time setup is told its tools are undeployed and sent to deploy
things that cannot land anywhere. The two need opposite actions, so they get separate errors.

proto-tools names the environment but does not create it: changing the shape of someone's Modal
workspace is theirs to decide, so the error offers the command rather than running it.
"""

from __future__ import annotations

import pytest

from proto_tools.modal import ModalEnvironmentNotFoundError, ToolNotDeployedError
from proto_tools.modal.client import _bound_method

WORKSPACE_HAS = ["main", "proto-env", "staging"]


@pytest.fixture
def modal_misses(monkeypatch):
    """Make every ``Cls.from_name`` lookup raise ``NotFoundError``, as Modal does for both causes."""
    import modal

    def _raise(*_args, **_kwargs):
        raise modal.exception.NotFoundError("nothing here")

    monkeypatch.setattr(modal.Cls, "from_name", staticmethod(_raise))


def _names(available):
    return lambda _client=None: available


def test_missing_environment_names_the_create_command(modal_misses, monkeypatch):
    """An environment the workspace does not have must point at the command that makes it."""
    monkeypatch.setattr("proto_tools.modal.app.environment_names", _names(WORKSPACE_HAS))

    with pytest.raises(ModalEnvironmentNotFoundError) as exc_info:
        _bound_method("proto-tools-tmalign", "TMalign", "run", "tmalign-alignment", environment="typoo")

    message = str(exc_info.value)
    assert "typoo" in message
    assert "proto-tools deploy --create-env --env typoo" in message
    assert "main, proto-env, staging" in message, "the error should say which environments do exist"


def test_missing_app_in_a_real_environment_still_says_not_deployed(modal_misses, monkeypatch):
    """When the environment exists, the app is the thing that is missing."""
    monkeypatch.setattr("proto_tools.modal.app.environment_names", _names(WORKSPACE_HAS))

    with pytest.raises(ToolNotDeployedError) as exc_info:
        _bound_method("proto-tools-tmalign", "TMalign", "run", "tmalign-alignment", environment="proto-env")

    assert "proto-tools deploy --apps tmalign" in str(exc_info.value)


def test_unreadable_workspace_falls_back_to_not_deployed(modal_misses, monkeypatch):
    """A listing that fails must not convert a tool error into an environment error."""
    monkeypatch.setattr("proto_tools.modal.app.environment_names", _names(None))

    with pytest.raises(ToolNotDeployedError):
        _bound_method("proto-tools-tmalign", "TMalign", "run", "tmalign-alignment", environment="proto-env")


def test_ambient_environment_is_not_second_guessed(modal_misses, monkeypatch):
    """With no environment named, there is nothing to diagnose, so no listing call is made."""
    called = []
    monkeypatch.setattr("proto_tools.modal.app.environment_names", lambda _client=None: called.append(1) or [])

    with pytest.raises(ToolNotDeployedError):
        _bound_method("proto-tools-tmalign", "TMalign", "run", "tmalign-alignment")

    assert not called, "an unnamed environment must not cost a network round-trip"


def test_workspace_info_reports_the_missing_environment(monkeypatch):
    """The MCP surface must not report a missing environment as "nothing deployed yet"."""
    import modal

    from proto_tools.mcp import tools as mcp_tools

    # ``workspace_info`` reports unauthenticated before it looks at anything else, so a machine
    # without credentials never reaches the environment check. Stand one in, since the question
    # here is what an authenticated caller is told about an environment that does not exist.
    monkeypatch.setattr(modal.Client, "from_env", staticmethod(lambda *_a, **_k: object()))
    monkeypatch.setattr("proto_tools.modal.app.environment_exists", lambda _name, _client=None: False)
    monkeypatch.setenv("MODAL_ENVIRONMENT", "never-made")

    info = mcp_tools.workspace_info("modal")

    assert info["environment_exists"] is False
    assert info["deployable"] is False
    assert "apps_deployed" not in info, "a count implies the environment exists"
    assert "--create-env" in info["hint"]


def test_create_env_is_a_no_op_when_it_already_exists(monkeypatch, capsys):
    """Re-running setup must be safe, so an existing name reports success without creating."""
    from proto_tools.modal import deploy

    monkeypatch.setattr("proto_tools.modal.app.environment_exists", lambda _name, _client=None: True)

    def _explode(*_args, **_kwargs):
        raise AssertionError("must not create an environment that already exists")

    monkeypatch.setattr("modal.environments.create_environment", _explode)

    assert deploy.create_env("proto-env") == 0
    assert "already exists" in capsys.readouterr().out

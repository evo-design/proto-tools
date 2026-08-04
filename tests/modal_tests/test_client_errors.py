"""The three ``device='modal'`` dispatch errors.

Each names a distinct human action — deploy a tool, authenticate, or pick a
different tool — so an agent (or a person) is never left guessing why a call
failed. All offline: no Modal calls, no network.
"""

from __future__ import annotations

import pytest

from proto_tools.modal import (
    ModalCredentialsError,
    ModalDispatchError,
    ToolNotDeployedError,
    ToolNotShippedError,
)
from proto_tools.modal.client import PROTO_TOOLS_REPO, _require_modal_credentials, resolve_tool


def test_unshipped_tool_is_distinct_from_undeployed():
    """A tool that is not shipped must not read as merely 'not deployed yet'."""
    with pytest.raises(ToolNotShippedError) as exc_info:
        resolve_tool("no-such-tool-anywhere")
    message = str(exc_info.value)
    assert "no-such-tool-anywhere" in message
    assert "proto-tools deploy --list" in message
    assert PROTO_TOOLS_REPO in message


def test_unshipped_error_stays_key_error_compatible():
    """resolve_tool historically raised KeyError; callers catching it must still work."""
    assert issubclass(ToolNotShippedError, KeyError)
    with pytest.raises(KeyError):
        resolve_tool("no-such-tool-anywhere")


def test_missing_credentials_names_the_setup_command(monkeypatch, tmp_path):
    """No token env vars and no ~/.modal.toml must point at ``modal token new``."""
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))

    with pytest.raises(ModalCredentialsError) as exc_info:
        _require_modal_credentials()
    message = str(exc_info.value)
    assert "modal token new" in message
    assert PROTO_TOOLS_REPO in message


def test_credentials_accepted_from_env(monkeypatch, tmp_path):
    """Token env vars satisfy the check even without a ~/.modal.toml."""
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
    _require_modal_credentials()  # must not raise


def test_credentials_accepted_from_config_file(monkeypatch, tmp_path):
    """A ~/.modal.toml satisfies the check even without env vars."""
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    (tmp_path / ".modal.toml").write_text("[default]\n")
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
    _require_modal_credentials()  # must not raise


def test_not_deployed_names_the_deploy_command():
    """The deploy error must carry a runnable command with the app slug and an env."""
    error = ToolNotDeployedError("esm2-embedding", "proto-tools-esm2")
    message = str(error)
    assert "proto-tools deploy --apps esm2 --env" in message
    assert PROTO_TOOLS_REPO in message
    assert error.tool_key == "esm2-embedding"


@pytest.mark.parametrize(
    "error",
    [
        ToolNotShippedError("x", 57),
        ModalCredentialsError("none are configured"),
        ToolNotDeployedError("x", "proto-tools-x"),
    ],
)
def test_all_dispatch_errors_share_the_actionable_base(error):
    """The MCP layer catches ModalDispatchError to flag needs_human; every error must be one."""
    assert isinstance(error, ModalDispatchError)


def test_mcp_run_tool_flags_dispatch_errors_as_needs_human(monkeypatch):
    """An error a human must resolve is surfaced to the agent as needs_human, not a raw crash."""
    from proto_tools.mcp import tools as mcp_tools

    def _raise_not_deployed(*_args, **_kwargs):
        raise ToolNotDeployedError("tmalign-alignment", "proto-tools-tmalign")

    monkeypatch.setattr("proto_tools.modal.client.dispatch_to_modal", _raise_not_deployed)
    result = mcp_tools.run_tool("tmalign-alignment", use_example=True)
    assert result["ok"] is False
    assert result["needs_human"] is True
    assert "proto-tools deploy" in result["error"]

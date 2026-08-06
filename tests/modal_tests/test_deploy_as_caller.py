"""Deploying into someone else's Modal workspace hands them nothing of the deploying process's.

``modal deploy`` is a subprocess, so unlike every other Modal call here it authenticates by
environment rather than by a client object. A subprocess inherits its parent's environment by
default, which for a server deploying on a caller's behalf means handing over whatever that server
holds: object-store credentials, a database URL, an API key.

Worse than the secrets is what the inherited variables would *build*. proto-tools reads several of
them to extend a deployment -- worker plugins that run arbitrary code in the container, extra
source and packages added to the image, and named secrets attached to the service. A deploy that
inherited those would build the deploying process's private variant of a tool inside the caller's
workspace.

So a credentialed deploy gets an allowlist and nothing else, and these pin that: what crosses, what
does not, and that the subprocess is actually run under it.

Every variable name below is a literal on purpose. Reading them back from the module under test
would make it its own oracle: moving a name into the allowlist -- the exact regression these exist
to catch -- would shrink the test set instead of failing it.

Offline throughout: no deploy is run, only the environment one would run under.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from proto_tools.modal.deploy import ModalTokens, deploy_app, deploy_environ, record_fingerprints

TOKENS = ModalTokens("ak-tokenid", "as-tokensecret")

#: Variables that turn a deployment into something other than stock proto-tools. Spelled out
#: rather than imported: these names are the requirement, not whatever the module currently lists.
EXTENSIONS = (
    "PROTO_MODAL_WORKER_PLUGINS",
    "PROTO_MODAL_EXTRA_SOURCE",
    "PROTO_MODAL_EXTRA_PACKAGES",
    "PROTO_MODAL_SECRETS",
    "PROTO_MODAL_PROTO_TOOLS",
)

#: Credentials a server running this genuinely has set, none of them a caller's business.
HOST_SECRETS = {
    "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
    "AWS_SECRET_ACCESS_KEY": "wJalrEXAMPLEKEY",
    "REDIS_URL": "redis://cache:6379",
    "DATABASE_URL": "postgres://db/records",
    "PROTO_API_KEY": "pk-hostkey",
}


class _Process:
    """A ``modal deploy`` that produces no output and succeeds, so only its environment matters."""

    def __init__(self) -> None:
        self.stdout: list[str] = []
        self.returncode = 0

    def wait(self) -> None:
        return None


@pytest.fixture
def hosted_environ(monkeypatch):
    """An environment shaped like a server's: its credentials, and every deployment extension set.

    ``PROTO_MODAL_HF_SECRET`` is left deliberately *unset*, which is the dangerous value: it is
    the one that makes proto-tools fall back to the deploying machine's own HuggingFace token.
    """
    for name, value in HOST_SECRETS.items():
        monkeypatch.setenv(name, value)
    for name in EXTENSIONS:
        monkeypatch.setenv(name, f"{name.lower()}-value")
    monkeypatch.delenv("PROTO_MODAL_HF_SECRET", raising=False)
    monkeypatch.setenv("HOME", "/home/app")


def test_the_cli_still_inherits_its_own_environment():
    """No tokens means the deploying workspace is the caller's own, so nothing is withheld."""
    assert deploy_environ(None) is None


def test_a_credentialed_deploy_authenticates_as_the_caller(hosted_environ):
    """The tokens are how the deploy lands in the caller's workspace rather than the server's."""
    env = deploy_environ(TOKENS)
    assert env is not None
    assert env["MODAL_TOKEN_ID"] == TOKENS.token_id
    assert env["MODAL_TOKEN_SECRET"] == TOKENS.token_secret


def test_the_hosts_credentials_do_not_cross(hosted_environ):
    """The reason this is an allowlist rather than a denylist."""
    env = deploy_environ(TOKENS)
    assert env is not None
    leaked = sorted(name for name in HOST_SECRETS if name in env)
    assert not leaked, f"reached a caller's deploy subprocess: {leaked}"


@pytest.mark.parametrize("name", EXTENSIONS)
def test_the_deployment_extensions_do_not_cross(hosted_environ, name):
    """A caller's deploy builds stock proto-tools: no plugins, no extra source, no added secrets."""
    env = deploy_environ(TOKENS)
    assert env is not None
    assert name not in env, f"{name} would have extended a caller's deployment with the server's"


def test_the_hosts_huggingface_token_is_never_minted_into_a_callers_workspace(hosted_environ):
    """The subtlest leak here, and the one that survives an allowlist.

    Unset, ``PROTO_MODAL_HF_SECRET`` makes proto-tools read the deploying machine's own HuggingFace
    token -- from ``HF_TOKEN``, or from files under ``HOME``, which the allowlist carries -- and
    attach it as a plaintext Modal secret on an app in the caller's workspace, where they can read
    it back. So it is set here rather than inherited, and inheriting nothing is not enough.
    """
    env = deploy_environ(TOKENS)
    assert env is not None
    assert env["PROTO_MODAL_HF_SECRET"] == "none", "an unset value falls back to the machine's token"
    assert "HF_TOKEN" not in env
    assert "HUGGING_FACE_HUB_TOKEN" not in env


def test_a_caller_can_bring_their_own_huggingface_token(hosted_environ):
    """Gated weights still work, without the caller's token passing through the deploying process.

    A name, not a token: ``Secret.from_name`` resolves in the workspace being deployed into, so
    the secret it finds is the caller's own.
    """
    env = deploy_environ(ModalTokens("ak-tokenid", "as-tokensecret", hf_secret="my-hf"))
    assert env is not None
    assert env["PROTO_MODAL_HF_SECRET"] == "my-hf"


def test_what_the_build_needs_does_cross(hosted_environ):
    """Withholding everything would be safe and useless."""
    env = deploy_environ(TOKENS)
    assert env is not None
    assert env["HOME"] == "/home/app"
    assert env["PATH"] == os.environ["PATH"]


def test_the_subprocess_actually_runs_under_it(hosted_environ, monkeypatch, tmp_path):
    """Building the environment and then not passing it would leave every test above vacuous."""
    seen: dict[str, object] = {}

    def _popen(cmd, **kwargs):
        seen["env"] = kwargs.get("env")
        return _Process()

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr("proto_tools.modal.deploy.record_fingerprints", lambda *a, **k: None)

    deploy_app("proto-tools-tmalign", "proto-env", tokens=TOKENS, log_dir=tmp_path)

    env = seen["env"]
    assert isinstance(env, dict), "Popen inherited the parent environment despite being given tokens"
    assert env["MODAL_TOKEN_ID"] == TOKENS.token_id
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "PROTO_MODAL_WORKER_PLUGINS" not in env


def test_fingerprints_are_recorded_into_the_workspace_that_was_deployed(monkeypatch, tmp_path):
    """Recorded elsewhere, the caller's deployment stays unfingerprinted and exempt from drift."""
    recorded: list[object] = []
    monkeypatch.setattr(
        "proto_tools.modal.fingerprint.write_manifest",
        lambda service, environment=None, client=None: recorded.append(client) or 1,
    )

    sentinel = object()
    monkeypatch.setattr(ModalTokens, "client", lambda self: sentinel)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: _Process())

    deploy_app("proto-tools-tmalign", "proto-env", tokens=TOKENS, log_dir=tmp_path)

    assert recorded, "a successful deploy recorded no fingerprints at all"
    assert all(c is sentinel for c in recorded), "fingerprints were written as the deploying process"


def test_a_server_can_supply_the_client_rather_than_have_one_opened(monkeypatch, tmp_path):
    """``Client.from_credentials`` registers a shutdown hook instead of closing.

    One opened per deploy stays open for the life of the process, which is why a long-running
    server reuses them. Given a client, this must use it rather than open another.
    """
    recorded: list[object] = []
    monkeypatch.setattr(
        "proto_tools.modal.fingerprint.write_manifest",
        lambda service, environment=None, client=None: recorded.append(client) or 1,
    )

    def _refuse(self):
        raise AssertionError("a client was opened despite one being supplied")

    monkeypatch.setattr(ModalTokens, "client", _refuse)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: _Process())

    cached = object()
    deploy_app("proto-tools-tmalign", "proto-env", tokens=TOKENS, client=cached, log_dir=tmp_path)

    assert recorded == [cached]


def test_a_deploy_that_succeeded_is_not_lost_to_a_fingerprint_failure(monkeypatch, tmp_path):
    """The caller has already paid for the build. Bookkeeping must not turn that into an error."""

    def _unreachable(self):
        raise ConnectionError("transient gRPC failure opening the client")

    monkeypatch.setattr(ModalTokens, "client", _unreachable)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: _Process())

    assert deploy_app("proto-tools-tmalign", "proto-env", tokens=TOKENS, log_dir=tmp_path) is True


def test_fingerprints_are_skipped_rather_than_recorded_in_the_wrong_workspace(monkeypatch, capsys):
    """Falling back to the deploying process's own client would record against the wrong volume.

    The caller's drift check would never look there, and the deploying process would read the
    record as though it described a deployment of its own.
    """
    written: list[object] = []
    monkeypatch.setattr(
        "proto_tools.modal.fingerprint.write_manifest",
        lambda service, environment=None, client=None: written.append(client) or 1,
    )

    def _unreachable(self):
        raise ConnectionError("transient gRPC failure opening the client")

    monkeypatch.setattr(ModalTokens, "client", _unreachable)
    record_fingerprints(["proto-tools-tmalign"], "proto-env", tokens=TOKENS)

    assert written == [], "a caller's fingerprints were written against the wrong workspace"
    assert "could not record fingerprints" in capsys.readouterr().out


def test_a_local_deploy_records_as_itself(monkeypatch, tmp_path):
    """The CLI has no tokens, and passing a client it never opened would break it."""
    recorded: list[object] = []
    monkeypatch.setattr(
        "proto_tools.modal.fingerprint.write_manifest",
        lambda service, environment=None, client=None: recorded.append(client) or 1,
    )
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: _Process())

    deploy_app("proto-tools-tmalign", "proto-env", log_dir=tmp_path)

    assert recorded == [None]


def test_the_token_pair_is_not_printable():
    """A deploy failure raises CalledProcessError, whose string form carries what it was given."""
    rendered = f"{TOKENS!r} {TOKENS}"
    assert TOKENS.token_secret not in rendered
    assert "<redacted>" in rendered


def test_a_failed_deploy_returns_the_build_output_not_a_path(tmp_path):
    """A caller who is not on this machine cannot read ``logs/deploy.tmalign.log``."""
    from proto_tools.mcp.tools import _log_tail

    log = tmp_path / "deploy.tmalign.log"
    log.write_text(
        "\x1b[38;5;39m⠋\x1b[0m \x1b[1mBuilding image\x1b[0m\n"
        "\x1b[38;5;39m⠙\x1b[0m \x1b[1mBuilding image\x1b[0m\n"
        "\x1b[38;5;196mModuleNotFoundError\x1b[0m: no module named 'x'\n"
    )

    tail = _log_tail(Path(tmp_path))
    assert "ModuleNotFoundError: no module named 'x'" in tail
    assert "⠋" not in tail, "spinner frames should be stripped, or the tail is all redraw noise"
    assert "\x1b[" not in tail, "colour codes are noise around the traceback an agent has to read"

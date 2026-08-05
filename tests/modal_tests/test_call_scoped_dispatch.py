"""A dispatch names its own Modal environment and client instead of publishing them.

``dispatch_to_modal`` used to write ``os.environ["MODAL_ENVIRONMENT"]`` and let Modal read it
back, which made the environment shared process state: two concurrent calls raced on it, and a
call had no way to name credentials at all. Both are now arguments.

One dispatch resolves three separate Modal objects — the service, the progress queue, and the
fingerprint volume — and the single ``os.environ`` write used to pin all three at once. Passing
the environment to only some of them splits a call across environments, so these check all three
together rather than the service alone.

Offline throughout: every Modal object is replaced, and clients are plain sentinels.
"""

from __future__ import annotations

import os
import queue as queue_module
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from proto_tools.modal import dispatch_batch_to_modal, dispatch_to_modal
from proto_tools.modal.client import _require_modal_credentials

TOOL_KEY = "esm2-embedding"


class _Probe:
    """Every Modal lookup one dispatch performs, with the environment and client it used."""

    def __init__(self) -> None:
        self.cls: list[tuple[str | None, object]] = []
        self.queue: list[tuple[str | None, object]] = []
        self.volume: list[tuple[str | None, object]] = []

    @property
    def everywhere(self) -> list[tuple[str | None, object]]:
        return self.cls + self.queue + self.volume


@pytest.fixture
def probe(monkeypatch):
    """Replace every Modal object a dispatch resolves, recording how each was addressed."""
    import modal

    from proto_tools.modal import client as client_module

    # A dispatch checks that credentials exist before resolving anything. These tests are about
    # where a call is addressed, not whether the machine is set up, so supply a pair rather than
    # inheriting one: CI has none, and a developer's own would make the result machine-dependent.
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")

    record = _Probe()

    class _Method:
        @staticmethod
        def remote(**_kwargs):
            return {"embeddings": []}

        @staticmethod
        def starmap(args, **_kwargs):
            return [{"embeddings": []} for _ in args]

    class _Service:
        def __call__(self):
            return self

        def hydrate(self):
            return None

        def with_options(self, **_kwargs):
            return self

        def __getattr__(self, _name):
            return _Method

    class _Queue:
        def hydrate(self):
            return None

        def get_many(self, *_args, **_kwargs):
            raise queue_module.Empty

    class _Volume:
        def read_file(self, _path):
            raise FileNotFoundError("no manifest recorded")

    def cls_from_name(_app, _name, *, environment_name=None, client=None):
        record.cls.append((environment_name, client))
        return _Service()

    def queue_from_name(_name, *, environment_name=None, client=None, **_kwargs):
        record.queue.append((environment_name, client))
        return _Queue()

    def volume_from_name(_name, *, environment_name=None, client=None, **_kwargs):
        record.volume.append((environment_name, client))
        return _Volume()

    monkeypatch.setattr(modal.Cls, "from_name", staticmethod(cls_from_name))
    monkeypatch.setattr(modal.Queue, "from_name", staticmethod(queue_from_name))
    monkeypatch.setattr(modal.Volume, "from_name", staticmethod(volume_from_name))
    # The identity plumbing is under test; output validation is covered elsewhere.
    monkeypatch.setattr(client_module, "_validated_output", lambda _key, result: result)
    client_module._DRIFT_WARNED.clear()
    return record


class _Inputs:
    """Minimal stand-in for a tool input payload."""

    @staticmethod
    def model_dump(**_kwargs):
        return {}


def _verbose_config():
    """A real config for the tool, verbose enough that live progress actually engages."""
    from proto_tools.tools import ToolRegistry

    return ToolRegistry.get(TOOL_KEY).config_model(verbose=1)


# --------------------------------------------------------------------------
# The regression the change exists for
# --------------------------------------------------------------------------


def test_dispatch_does_not_mutate_the_process_environment(probe, monkeypatch):
    """The environment must reach Modal as an argument, leaving the process untouched."""
    monkeypatch.setenv("MODAL_ENVIRONMENT", "untouched")

    dispatch_to_modal(TOOL_KEY, _Inputs(), _verbose_config(), environment="theirs")

    assert os.environ["MODAL_ENVIRONMENT"] == "untouched", "dispatch mutated the process environment"


def test_concurrent_dispatches_do_not_cross(probe):
    """Two callers dispatching at once must each reach Modal with their own environment.

    The barrier holds both inside the dispatch until both have entered, which is the interleaving
    the old ``os.environ`` write could not survive: one call's write landed while the other was
    between its own write and its lookup.
    """
    barrier = Barrier(2)

    def dispatch(name: str) -> None:
        barrier.wait(timeout=5)
        dispatch_to_modal(TOOL_KEY, _Inputs(), _verbose_config(), environment=name, client=f"client-{name}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(dispatch, ["alice-env", "bob-env"]))

    per_client = {client: env for env, client in probe.cls}
    assert set(per_client.values()) == {"alice-env", "bob-env"}, f"environments crossed: {probe.cls}"
    assert len(per_client) == 2, "both callers resolved through the same Modal client"


# --------------------------------------------------------------------------
# All three Modal objects, not just the service
# --------------------------------------------------------------------------


def test_every_modal_object_resolves_in_the_dispatch_environment(probe):
    """The service, the progress queue, and the fingerprint volume must all agree.

    They were pinned together by the one ``os.environ`` write. Threading the environment into
    only some of them means the container writes progress to a queue nobody tails, and drift is
    compared against another deployment.
    """
    dispatch_to_modal(TOOL_KEY, _Inputs(), _verbose_config(), environment="theirs")

    assert probe.cls, "the service was never resolved"
    assert probe.queue, "the progress queue was never opened"
    assert probe.volume, "the fingerprint volume was never read"
    seen = {env for env, _client in probe.everywhere}
    assert seen == {"theirs"}, f"dispatch split across environments: {seen}"


def test_every_modal_object_resolves_with_the_caller_client(probe):
    """A dispatch made on someone's behalf must not reach any object as the process."""
    dispatch_to_modal(TOOL_KEY, _Inputs(), _verbose_config(), environment="theirs", client="their-client")

    used = {client for _env, client in probe.everywhere}
    assert used == {"their-client"}, "some lookups fell back to the process's own credentials"


def test_batch_dispatch_matches_the_single_path(probe):
    """The batch entry point must resolve identity exactly as the single one does."""
    dispatch_batch_to_modal(
        TOOL_KEY, [_Inputs(), _Inputs()], _verbose_config(), environment="theirs", client="their-client"
    )

    assert {env for env, _client in probe.everywhere} == {"theirs"}
    assert {client for _env, client in probe.everywhere} == {"their-client"}


# --------------------------------------------------------------------------
# The ambient path, which every existing caller still uses
# --------------------------------------------------------------------------


def test_ambient_environment_is_still_honoured(probe, monkeypatch):
    """proto-tools-api pins itself by publishing MODAL_ENVIRONMENT; that must keep working.

    It sets the variable at startup and relies on proto-tools reading it. Deleting the write was
    safe; deleting the read would silently send its traffic to proto-tools' own default.
    """
    monkeypatch.setenv("MODAL_ENVIRONMENT", "staging")

    dispatch_to_modal(TOOL_KEY, _Inputs(), _verbose_config())

    assert {env for env, _client in probe.everywhere} == {"staging"}


def test_no_client_leaves_credential_resolution_to_modal(probe, monkeypatch):
    """A caller that names no client must hand that half back to Modal, as it always did."""
    monkeypatch.setenv("MODAL_ENVIRONMENT", "staging")

    dispatch_to_modal(TOOL_KEY, _Inputs(), _verbose_config())

    assert {client for _env, client in probe.everywhere} == {None}


def test_deployed_apps_asks_about_the_dispatch_environment(monkeypatch):
    """Reporting what is deployed must ask about the environment a call would actually reach.

    Asked ambiently it reports on the Modal profile's environment, which is not the one
    ``resolve_environment`` picks. That made ``list_tools(deployed_only=True)`` describe a
    different environment's deployments, and hid apps that were in fact deployed.
    """
    import modal

    from proto_tools.utils.modal_status import deployed_apps

    seen: list[str | None] = []

    class _Hydrated:
        def hydrate(self):
            return None

    def cls_from_name(_app, _name, *, environment_name=None, **_kwargs):
        seen.append(environment_name)
        return _Hydrated()

    monkeypatch.setattr(modal.Cls, "from_name", staticmethod(cls_from_name))
    monkeypatch.delenv("MODAL_ENVIRONMENT", raising=False)

    deployed_apps()

    assert seen, "no apps were probed"
    assert set(seen) == {"proto-env"}, f"probed the wrong environment: {set(seen)}"


def test_supplied_client_satisfies_the_credential_check(monkeypatch, tmp_path):
    """A caller acting for someone else must not need the process to have credentials."""
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))

    _require_modal_credentials("their-client")  # must not raise


def test_no_client_still_requires_process_credentials(monkeypatch, tmp_path):
    """Without a client and without ambient credentials, the actionable error still fires."""
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))

    from proto_tools.modal import ModalCredentialsError

    with pytest.raises(ModalCredentialsError):
        _require_modal_credentials()

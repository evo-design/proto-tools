"""tests/test_deploy_helpers.py.

The pure parts of the deploy CLI: slug resolution, output shaping, and the filters that keep a
live display's redraws out of a captured log.

These run without Modal. Everything here is decided before a network call, so a mistake shows up
as a confusing command rather than a failed deploy, which is exactly the kind of thing that goes
unnoticed without tests.
"""

from __future__ import annotations

import pytest

from proto_tools.modal.deploy import (
    RecentLines,
    _display_path,
    _in_columns,
    apps_to_smoke_test,
    describe_progress,
    for_display,
    modal_command,
    render_entrypoint,
    resolve_targets,
)
from proto_tools.modal.manifest import (
    APP_BUCKETS,
    app_name_for_slug,
    app_slug,
    get_app_name_for_service,
    module_name,
    physical_device_for_service,
)


# ============================================================================
# Naming
# ============================================================================
def test_a_slug_round_trips_to_its_app_name():
    """``--apps esmc`` has to reach ``proto-tools-esmc`` and nothing else."""
    for app_name in APP_BUCKETS:
        assert app_name_for_slug(app_slug(app_name)) == app_name


def test_an_unknown_slug_is_rejected_by_name():
    with pytest.raises(KeyError) as caught:
        app_name_for_slug("no-such-tool")
    assert "no-such-tool" in str(caught.value)


def test_a_module_name_is_importable():
    """The rendered entrypoint is written to this filename, so it must be a valid identifier."""
    for app_name in APP_BUCKETS:
        assert module_name(app_name).isidentifier()


def test_every_service_resolves_to_its_app():
    for app_name, services in APP_BUCKETS.items():
        for service in services:
            assert get_app_name_for_service(service) == app_name


def test_a_gpu_service_reports_a_gpu_device():
    """Dispatch rewrites a logical device to this, so a wrong answer runs a model on the CPU."""
    assert physical_device_for_service("ESMCService") == "cuda"
    assert physical_device_for_service("TMalignService") == "cpu"


# ============================================================================
# Target resolution
# ============================================================================
def test_resolving_targets():
    assert resolve_targets("esmc") == [app_name_for_slug("esmc")]
    assert resolve_targets("esmc,tmalign") == [app_name_for_slug("esmc"), app_name_for_slug("tmalign")]
    assert resolve_targets("all") == sorted(APP_BUCKETS)
    assert resolve_targets(None) == sorted(APP_BUCKETS)


def test_a_failed_deploy_is_not_smoke_tested():
    """A failed deploy leaves the previous one running.

    Testing it would report on code that never shipped.
    """
    testable, skipped = apps_to_smoke_test(["a", "b"], {"a": True, "b": False})
    assert (testable, skipped) == (["a"], ["b"])


def test_skipping_the_deploy_tests_whatever_is_already_there():
    """``--skip-deploy --test`` means "check what is deployed", so nothing is excluded."""
    testable, skipped = apps_to_smoke_test(["a", "b"], None)
    assert (testable, skipped) == (["a", "b"], [])


# ============================================================================
# The rendered entrypoint
# ============================================================================
def test_the_entrypoint_imports_every_service_in_the_app():
    """Importing a service binds its ``@app.cls`` to the app being deployed.

    A service left out of the render deploys as an app with nothing in it.
    """
    app_name = "proto-tools-esm2"
    source = render_entrypoint(app_name, APP_BUCKETS[app_name])
    for service in APP_BUCKETS[app_name]:
        assert f"import {service}" in source
    assert f'get_app("{app_name}")' in source


def test_modal_is_invoked_through_this_interpreter(tmp_path):
    """Modal runs through this interpreter, not whichever one is first on PATH.

    A bare ``modal`` can belong to a different environment, where the entrypoint's imports
    do not resolve.
    """
    import sys
    from pathlib import Path

    cmd = modal_command(Path(tmp_path / "x.py"), "proto-env")
    assert cmd[:4] == [sys.executable, "-m", "modal", "deploy"]
    assert cmd[-2:] == ["--env", "proto-env"]
    assert "--env" not in modal_command(Path(tmp_path / "x.py"), None)


def test_a_path_outside_the_working_directory_stays_absolute():
    from pathlib import Path

    assert _display_path(Path.cwd() / "logs" / "a.log") == "logs/a.log"
    assert _display_path(Path("/etc/hosts")) == "/etc/hosts"


# ============================================================================
# Output shaping
# ============================================================================
def test_columns_widen_to_the_longest_entry():
    """A slug wider than a fixed column would run into its neighbour."""
    rows = _in_columns(["a", "splice-transformer", "b", "c", "d"], per_row=4)
    assert len(rows) == 2
    assert "splice-transformer  b" in rows[0]


def test_columns_of_nothing():
    assert _in_columns([]) == []


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Building image im-abc", "building image"),
        ("=> Step 1: COPY . /", "COPY . /"),
        ("Creating objects...", "creating Modal objects"),
        ("✓ App deployed in 12s! 🎉", "deployed"),
        ("Collecting numpy==1.26", None),
    ],
)
def test_phases_are_recognised_from_the_line_that_marks_them(line, expected):
    assert describe_progress(line) == expected


# ============================================================================
# Redraw suppression
# ============================================================================
def test_a_redraw_frame_is_shown_once():
    """A live display repeats its whole frame on every refresh."""
    seen = RecentLines()
    first = for_display("\x1b[2K\x1b[1A├── Created mount\n", seen)
    again = for_display("\x1b[2K\x1b[1A├── Created mount\n", seen)
    assert first == "├── Created mount"
    assert again is None


def test_a_spinner_counts_as_the_same_line_whatever_frame_it_is_on():
    """Each frame carries a different braille glyph, so a plain repeat check misses them all."""
    seen = RecentLines()
    shown = [for_display(f"{frame} Creating objects...\n", seen) for frame in "⠋⠙⠹⠸⠼"]
    assert sum(s is not None for s in shown) == 1


def test_blank_lines_are_dropped():
    assert for_display("\x1b[2K\n", RecentLines()) is None


def test_the_window_is_bounded():
    """Progress lines can each be unique, so an unbounded record would grow without limit."""
    seen = RecentLines(limit=10)
    for i in range(1000):
        seen.add_if_new(f"Uploaded {i}/1000 files")
    assert len(seen._keys) == 10


def test_a_line_returning_after_the_window_is_shown_again():
    """The window is deliberately not a full history: a repeat long afterwards is a real one."""
    seen = RecentLines(limit=2)
    assert for_display("same\n", seen) == "same"
    for filler in ("a\n", "b\n"):
        for_display(filler, seen)
    assert for_display("same\n", seen) == "same"


def _fake_run(outcomes):
    """A ``_run_modal_deploy`` stand-in that replays ``outcomes``, recording each call."""
    calls = []

    def run(app_name, entrypoint, environment, log_file, on_progress, verbose, tokens):
        calls.append(app_name)
        return outcomes[len(calls) - 1]

    return run, calls


def test_a_lost_publish_race_is_retried(monkeypatch, tmp_path, capsys):
    """Losing a publish race must not surface as a failed deploy.

    The same app reaches Modal from this CLI, from a server deploying on someone's behalf, and
    from their own machine, and none of those can see the others. Modal shares the image build
    between them, so a collision costs only the final publish, and it says to try again.
    """
    from proto_tools.modal import deploy as deploy_module

    run, calls = _fake_run([(False, True), (True, False)])
    monkeypatch.setattr(deploy_module, "_run_modal_deploy", run)
    monkeypatch.setattr(deploy_module, "record_fingerprints", lambda *a, **k: None)

    assert deploy_module.deploy_app("proto-tools-esm2", "proto-env", log_dir=tmp_path) is True
    assert len(calls) == 2, "the losing attempt was not retried"
    assert "retrying" in capsys.readouterr().out


def test_an_ordinary_failure_is_not_retried(monkeypatch, tmp_path):
    """A build that genuinely fails must fail once, not three times.

    Retrying it would triple the wait before a user sees a broken image, and each attempt costs
    them a build.
    """
    from proto_tools.modal import deploy as deploy_module

    run, calls = _fake_run([(False, False)])
    monkeypatch.setattr(deploy_module, "_run_modal_deploy", run)
    monkeypatch.setattr(deploy_module, "record_fingerprints", lambda *a, **k: None)

    assert deploy_module.deploy_app("proto-tools-esm2", "proto-env", log_dir=tmp_path) is False
    assert len(calls) == 1


def test_giving_up_on_a_race_still_reports_the_failure(monkeypatch, tmp_path, capsys):
    """A run stays quiet about a race, so exhausting the retries must not be silent."""
    from proto_tools.modal import deploy as deploy_module

    run, calls = _fake_run([(False, True)] * deploy_module.DEPLOY_RACE_ATTEMPTS)
    monkeypatch.setattr(deploy_module, "_run_modal_deploy", run)
    monkeypatch.setattr(deploy_module, "record_fingerprints", lambda *a, **k: None)

    assert deploy_module.deploy_app("proto-tools-esm2", "proto-env", log_dir=tmp_path) is False
    assert len(calls) == deploy_module.DEPLOY_RACE_ATTEMPTS
    assert "kept losing" in capsys.readouterr().out


def test_the_race_marker_matches_what_modal_actually_says(monkeypatch, tmp_path):
    """Pinned against Modal's wording, observed from a real concurrent deploy.

    The marker is the whole mechanism: if Modal rephrases it, a race silently stops being
    retried, and there is nothing else in the output that identifies one.
    """
    from proto_tools.modal import deploy as deploy_module

    observed = (
        "modal.exception.AlreadyExistsError: A deployment with this name already exists "
        "and should be updated. Possible race between two concurrent deploys. Try again."
    )
    assert deploy_module.DEPLOY_RACE_MARKER in observed


def test_the_run_recognises_a_race_in_modal_output(monkeypatch, tmp_path):
    """The detection itself, over the stream rather than a stubbed return value.

    This is the half a stubbed ``_run_modal_deploy`` cannot check, and the half that breaks if
    the marker or the ANSI stripping is wrong.
    """
    import subprocess

    from proto_tools.modal import deploy as deploy_module

    class _Process:
        # Modal colours its errors, so the marker arrives wrapped in escape codes.
        stdout = iter(["Building image\n", f"\x1b[31m{deploy_module.DEPLOY_RACE_MARKER}. Try again.\x1b[0m\n"])
        returncode = 1

        def wait(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Process())

    deployed, raced = deploy_module._run_modal_deploy(
        "proto-tools-esm2", tmp_path / "e.py", "proto-env", tmp_path / "d.log", None, False, None
    )

    assert deployed is False
    assert raced is True, "the race went unrecognised, so it would not be retried"

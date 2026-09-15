"""Tests for generic container executable and bind handling."""

from pathlib import Path

import pytest

from proto_tools.utils.standalone_helpers_source.standalone_helpers.containers import resolve_container_runtime


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_configured_runtime_precedes_path_and_records_hash(tmp_path: Path, monkeypatch) -> None:
    executable = _executable(tmp_path / "apptainer")
    image = tmp_path / "model.sif"
    image.write_bytes(b"image")
    monkeypatch.setattr("shutil.which", lambda _name: "/ignored/path")

    runtime = resolve_container_runtime(
        image,
        configured_executables=(executable,),
    )
    provenance = runtime.provenance()

    assert runtime.executable.path == executable.resolve()
    assert provenance["runtime"]["resolution_source"] == "configured_path"
    assert len(provenance["runtime"]["sha256"]) == 64
    assert len(provenance["image"]["sha256"]) == 64


def test_singularity_is_supported_as_ordered_path_fallback(tmp_path: Path, monkeypatch) -> None:
    singularity = _executable(tmp_path / "singularity")
    image = tmp_path / "model.sif"
    image.write_bytes(b"image")
    monkeypatch.setattr(
        "shutil.which",
        lambda name: str(singularity) if name == "singularity" else None,
    )

    runtime = resolve_container_runtime(image)

    assert runtime.executable.name == "singularity"
    assert runtime.executable.source == "PATH"


def test_missing_container_executable_is_actionable(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "model.sif"
    image.write_bytes(b"image")
    monkeypatch.setattr("shutil.which", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="no supported executable found"):
        resolve_container_runtime(image)


def test_container_command_normalizes_bind_paths(tmp_path: Path, monkeypatch) -> None:
    executable = _executable(tmp_path / "apptainer")
    image = tmp_path / "model.sif"
    image.write_bytes(b"image")
    source = tmp_path / "relative-source"
    destination = tmp_path / "relative-destination"
    monkeypatch.chdir(tmp_path)
    runtime = resolve_container_runtime(
        image,
        configured_executables=(executable,),
    )

    command = runtime.build_exec_command(
        arguments=["--flag=value"],
        binds=[(source.name, destination.name)],
        entrypoint=["python", "tool.py"],
        environment={"EXAMPLE": "1"},
        gpu=True,
    )

    assert command[:3] == [str(executable.resolve()), "exec", "--nv"]
    assert command[3:5] == ["--env", "EXAMPLE=1"]
    bind_index = command.index("--bind")
    assert command[bind_index + 1] == f"{source.resolve()}:{destination.resolve()}"
    assert command[-3:] == ["python", "tool.py", "--flag=value"]

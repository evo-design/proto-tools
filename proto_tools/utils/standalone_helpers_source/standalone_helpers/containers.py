"""Generic executable resolution and argv construction for container runtimes."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .proto_logging import get_logger
from .runtime_provenance import file_identity

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResolvedExecutable:
    """One executable resolved from a configured path or ``PATH``."""

    path: Path
    name: str
    source: str

    def provenance(self) -> dict[str, Any]:
        """Return the executable's exact file identity and resolution source."""
        result = file_identity(
            self.path,
            role="container_runtime_executable",
            logical_id=self.name,
        )
        result["resolution_source"] = self.source
        return result


def resolve_executable(
    *,
    configured_paths: tuple[str | Path, ...] = (),
    path_names: tuple[str, ...],
) -> ResolvedExecutable:
    """Resolve an executable from explicit candidates, then ordered ``PATH`` names."""
    checked: list[str] = []
    for candidate in configured_paths:
        path = Path(candidate).expanduser().resolve()
        checked.append(str(path))
        if path.is_file() and os.access(path, os.X_OK):
            return ResolvedExecutable(path=path, name=path.name, source="configured_path")
    for name in path_names:
        checked.append(f"PATH:{name}")
        discovered = shutil.which(name)
        if discovered:
            path = Path(discovered).resolve()
            return ResolvedExecutable(path=path, name=name, source="PATH")
    raise FileNotFoundError("no supported executable found; checked " + ", ".join(checked))


@dataclass(frozen=True)
class ContainerRuntime:
    """A resolved container executable bound to one immutable image file."""

    executable: ResolvedExecutable
    image_path: Path

    def __post_init__(self) -> None:
        """Normalize and validate the container image path."""
        image = self.image_path.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"container image does not exist: {image}")
        object.__setattr__(self, "image_path", image)

    def build_exec_command(
        self,
        *,
        arguments: list[str],
        binds: list[tuple[str | Path, str | Path]],
        entrypoint: list[str],
        environment: dict[str, str] | None = None,
        gpu: bool = False,
    ) -> list[str]:
        """Build an Apptainer/Singularity-compatible ``exec`` argv."""
        command = [str(self.executable.path), "exec"]
        if gpu:
            command.append("--nv")
        for name, value in (environment or {}).items():
            command.extend(["--env", f"{name}={value}"])
        for source, destination in binds:
            command.extend(
                [
                    "--bind",
                    f"{Path(source).expanduser().resolve()}:{Path(destination).expanduser().resolve()}",
                ]
            )
        return [*command, str(self.image_path), *entrypoint, *arguments]

    def provenance(self) -> dict[str, Any]:
        """Return exact executable and image identities."""
        return {
            "mode": "container",
            "runtime": self.executable.provenance(),
            "image": file_identity(
                self.image_path,
                role="container_image",
                logical_id=self.image_path.name,
            ),
        }


def resolve_container_runtime(
    image_path: str | Path,
    *,
    configured_executables: tuple[str | Path, ...] = (),
    runtime_names: tuple[str, ...] = ("apptainer", "singularity"),
) -> ContainerRuntime:
    """Resolve a supported runtime and bind it to an existing image."""
    executable = resolve_executable(
        configured_paths=configured_executables,
        path_names=runtime_names,
    )
    return ContainerRuntime(executable=executable, image_path=Path(image_path))

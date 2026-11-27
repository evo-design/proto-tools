"""Generic executable resolution and argv construction for container runtimes."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .proto_logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResolvedExecutable:
    """One executable resolved from a configured path or ``PATH``.

    Attributes:
        path (Path): Absolute path to the resolved executable.
        name (str): The executable's name as resolved.
        source (str): Where it came from: ``"configured_path"`` or ``"PATH"``.
    """

    path: Path
    name: str
    source: str


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

"""Shared Modal base images, and the layers that add proto-tools to them."""

import importlib.metadata
import os
from pathlib import Path

import modal

from proto_tools.modal.proto_tools_source import PATH_ENV, checkout_or_none

# ``add_local_python_source`` places packages in /root, which precedes site-packages on the
# container's PYTHONPATH. Mounting there rather than installing a second copy elsewhere means
# the tree that gets patched with standalone overrides is the one imports actually resolve to.
CONTAINER_ROOT = "/root"

#: Extra pip requirements every service image installs, separated by whitespace. For dependencies
#: a deployment needs that proto-tools itself does not.
EXTRA_PACKAGES_ENV = "PROTO_MODAL_EXTRA_PACKAGES"

#: Extra local directories every service image carries, separated by :data:`os.pathsep`. Each is
#: mounted under :data:`CONTAINER_ROOT` by its own name, so a container imports it as that name.
EXTRA_SOURCE_ENV = "PROTO_MODAL_EXTRA_SOURCE"

# Resolving a source tree is a deploy-time concern; a container never builds an image.
_CHECKOUT = checkout_or_none()


def container_package_root() -> str:
    """Return the ``proto_tools`` package path inside a built image."""
    return f"{CONTAINER_ROOT}/proto_tools"


def _runtime_requirements() -> list[str]:
    """Return the installed distribution's dependencies, excluding optional extras."""
    try:
        requires = importlib.metadata.requires("proto-tools") or []
    except importlib.metadata.PackageNotFoundError:
        return []
    return [r for r in requires if "extra ==" not in r]


def with_dependencies(image: modal.Image) -> modal.Image:
    """Install proto-tools' dependencies, from the checkout's pyproject or the install's metadata.

    Args:
        image (modal.Image): The image to add the dependency layer to.

    Returns:
        modal.Image: The image carrying proto-tools' dependencies.
    """
    if _CHECKOUT is not None:
        return image.pip_install_from_pyproject(str(_CHECKOUT / "pyproject.toml"))
    requirements = _runtime_requirements()
    return image.pip_install(*requirements) if requirements else image


# Benchmark results recorded beside each deployment. They describe what a service costs, which is
# useful in the repository and useless inside the image: no container reads them, and rewriting one
# after a benchmark run would otherwise invalidate the mount layer and force every image to rebuild.
#
# Anything but the empty list here is load-bearing. The default for add_local_python_source keeps
# only .py files, which would drop every setup.sh and requirements.txt a standalone environment
# needs — hence an explicit list naming exactly the reports rather than a broader pattern.
#
# Scoped to `*_deployment/` because that is the only place a report is written, and because
# `proto_tools/modal/README.md` is the package's own setup documentation, which does ship.
BENCHMARK_REPORT_PATTERNS: list[str] = ["**/modal/*/*_deployment/README.md"]


def extra_packages() -> list[str]:
    """Return the requirements named in :data:`EXTRA_PACKAGES_ENV`.

    Returns:
        list[str]: Requirement specifiers, empty when the variable is unset. Split on whitespace
            rather than commas, which a specifier may contain.
    """
    return os.environ.get(EXTRA_PACKAGES_ENV, "").split()


def extra_source_dirs() -> list[Path]:
    """Return the directories named in :data:`EXTRA_SOURCE_ENV`.

    Returns:
        list[Path]: Resolved directories, empty when the variable is unset.

    Raises:
        ValueError: If a named path is not a directory. Building an image without the code a
            deployment asked for would fail later, inside a container, on the first call.
    """
    resolved = []
    for entry in os.environ.get(EXTRA_SOURCE_ENV, "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry).resolve()
        if not path.is_dir():
            raise ValueError(f"{EXTRA_SOURCE_ENV} names {entry!r}, which is not a directory")
        resolved.append(path)
    return resolved


def _with_extras(image: modal.Image) -> modal.Image:
    """Add the packages and source a deployment asked for, below proto-tools' own layers."""
    if packages := extra_packages():
        image = image.pip_install(*packages)
    for path in extra_source_dirs():
        image = image.add_local_dir(str(path), f"{CONTAINER_ROOT}/{path.name}", copy=True)
    return image


def with_proto_tools(
    image: modal.Image, *, overrides: str | None = None, overrides_dir: Path | str | None = None
) -> modal.Image:
    """Add proto-tools itself to ``image``, on top of a base that already carries its dependencies.

    Resolved the way Python resolves the import, so an editable install, a plain install, and a
    clone on ``sys.path`` all mount the tree the deploying process is actually running.

    Args:
        image (modal.Image): A base image built from :data:`GPU_BASE` or :data:`CPU_BASE`.
        overrides (str | None): Toolkit whose standalone directory the overlay targets.
        overrides_dir (Path | str | None): Directory holding ``standalone_overrides/``,
            normally the service module's own directory.

    Returns:
        modal.Image: The image with proto-tools importable, and whatever :data:`EXTRA_PACKAGES_ENV`
            and :data:`EXTRA_SOURCE_ENV` name.
    """
    from proto_tools.modal.utils import apply_standalone_overrides

    # Applied first, so editing proto-tools does not rebuild a deployment's own layers.
    image = _with_extras(image)

    # ignore=[] is required: the default keeps only .py, dropping every setup.sh and
    # requirements.txt a standalone environment needs.
    override = os.environ.get(PATH_ENV)
    if override:
        # An explicit tree is not what `import proto_tools` resolves to, so mount it by path.
        image = image.add_local_dir(
            str(Path(override) / "proto_tools"),
            container_package_root(),
            copy=True,
            ignore=BENCHMARK_REPORT_PATTERNS,
        )
    else:
        image = image.add_local_python_source("proto_tools", copy=True, ignore=BENCHMARK_REPORT_PATTERNS)
    if overrides and overrides_dir:
        image = apply_standalone_overrides(image, overrides, overrides_dir)
    return image


GPU_BASE = with_dependencies(
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12").apt_install("curl", "git")
)

CPU_BASE = with_dependencies(modal.Image.debian_slim(python_version="3.12").apt_install("curl", "git", "g++"))

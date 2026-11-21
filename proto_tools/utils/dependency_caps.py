"""Caps on GPU-, driver- and compiler-coupled packages at the newest versions we have tested.

``dependency_caps.txt`` holds one upper bound per package, each with a comment recording the
version tested, when, on what hardware and in which run. Every env build hands the file to
``uv pip`` and ``pip`` as a constraints file, so a release newer than a cap cannot reach an env,
even as an indirect dependency, until someone tests it and raises the cap. Everything older still
resolves, so the resolver keeps choosing what fits each machine.

Caps are set and raised by hand only, in a reviewed PR with test evidence; see
``notes/tool-environments.md``. ``PROTO_DEPENDENCY_CAPS`` overrides them:

- unset: use the packaged ``dependency_caps.txt``
- ``0``: build without caps
- a path: use that constraints file instead
"""

import os
from collections.abc import Mapping
from pathlib import Path

CAPS_FILE = Path(__file__).with_name("dependency_caps.txt")
CAPS_ENV_VAR = "PROTO_DEPENDENCY_CAPS"
CONSTRAINT_ENV_VARS = ("UV_CONSTRAINT", "PIP_CONSTRAINT")


def resolve_caps_file() -> Path | None:
    """Return the constraints file env builds should use, or ``None`` when caps are turned off.

    Returns:
        Path | None: The packaged caps, the file ``PROTO_DEPENDENCY_CAPS`` names, or ``None``
            when it is ``0``.

    Raises:
        FileNotFoundError: If ``PROTO_DEPENDENCY_CAPS`` names a file that does not exist.
    """
    value = os.environ.get(CAPS_ENV_VAR, "").strip()
    if value == "0":
        return None
    if not value:
        return CAPS_FILE
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"{CAPS_ENV_VAR}={value!r} is not a file. Set it to a constraints file, "
            f"or to 0 to build without dependency caps."
        )
    return path


def constraint_env(caps_file: Path | None, parent_env: Mapping[str, str]) -> dict[str, str]:
    """Return ``UV_CONSTRAINT`` and ``PIP_CONSTRAINT`` for an env build: the caps plus the caller's own.

    A tool's subprocess env copies only allowlisted variables, so a user's own constraints are read
    from the caller's environment and passed through here. Both variables take a space-separated
    list, so the caps and the user's constraints apply together.

    Args:
        caps_file (Path | None): The caps to apply, or ``None`` when they are turned off.
        parent_env (Mapping[str, str]): The caller's environment, normally ``os.environ``.

    Returns:
        dict[str, str]: The variables that end up non-empty, ready to merge into the build's env.
    """
    constraints = {var: f"{caps_file or ''} {parent_env.get(var, '')}".strip() for var in CONSTRAINT_ENV_VARS}
    return {var: value for var, value in constraints.items() if value}


def caps_hint(caps_file: Path) -> str:
    """Guidance appended to env-build failures when the build ran with dependency caps.

    Args:
        caps_file (Path): The constraints file the build used.

    Returns:
        str: A note naming the caps and how to build without them.
    """
    return (
        f"\n\nThis env was built with dependency caps ({caps_file}). If the failure is a version "
        "conflict involving a capped package (for example this machine needs a newer torch or "
        f"triton than we have tested), retry with {CAPS_ENV_VAR}=0, or point {CAPS_ENV_VAR} at "
        "your own constraints file, and please tell us which version worked."
    )

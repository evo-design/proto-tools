"""Caps on releases known to break our envs, applied only on the machines they break.

``dependency_caps.txt`` holds one upper bound per line, with the machines it applies to and the
documented break behind it, as in
``triton<3.9  # driver<580: 3.9 builds sm_90+ kernels that CUDA 12 drivers reject (https://...)``.
The condition is ``all``, or ``driver<N`` for machines whose NVIDIA driver major version is below
``N``. No pip or uv marker can test the driver, so each env build writes the caps that apply to its
machine into the env and hands that file to ``uv pip`` and ``pip`` as a constraints file, which caps
indirect dependencies too. A release nobody has seen break stays uncapped, so every machine gets
the newest version that works on it.

See ``notes/tool-environments.md``.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

CAPS_FILE = Path(__file__).with_name("dependency_caps.txt")
ENV_CAPS_NAME = "dependency_caps.txt"

_CAP_LINE = re.compile(
    r"^(?P<requirement>[a-z0-9][a-z0-9-]*<\d+(\.\d+)*)\s+#\s+(?P<condition>all|driver<\d+):\s+(?P<reason>.+)$"
)


class Cap(NamedTuple):
    """One line of ``dependency_caps.txt``."""

    requirement: str
    condition: str
    reason: str


def parse_caps(text: str) -> list[Cap]:
    """Parse the caps file, skipping blank and comment lines.

    Args:
        text (str): The contents of ``dependency_caps.txt``.

    Returns:
        list[Cap]: Each cap with its condition and reason.

    Raises:
        ValueError: If a line is not ``<package><<version>  # <condition>: <reason>``.
    """
    caps = []
    for line in text.splitlines():
        line = line.strip()  # noqa: PLW2901
        if not line or line.startswith("#"):
            continue
        match = _CAP_LINE.match(line)
        if not match:
            raise ValueError(f"dependency caps: {line!r} is not `<package><<version>  # <condition>: <reason>`")
        caps.append(Cap(match["requirement"], match["condition"], match["reason"]))
    return caps


def cap_applies(condition: str, driver_major: int | None) -> bool:
    """Return whether a cap's condition holds on a machine.

    Args:
        condition (str): ``all`` or ``driver<N``.
        driver_major (int | None): The NVIDIA driver major version, or ``None`` with no NVIDIA GPU.

    Returns:
        bool: ``True`` when the cap should constrain this machine's builds.
    """
    if condition == "all":
        return True
    return driver_major is not None and driver_major < int(condition.removeprefix("driver<"))


def write_env_caps(env_path: Path, build_env: Mapping[str, str]) -> Path | None:
    """Write the caps that apply to this machine into the env, for its setup.sh to install under.

    The caps are filtered by the driver ``compute_deps`` detected. The env keeps the file as a record
    of the caps it was built under.

    Args:
        env_path (Path): The env being built.
        build_env (Mapping[str, str]): setup.sh's environment, carrying ``DETECTED_DRIVER_VERSION``.

    Returns:
        Path | None: The file to constrain installs with, or ``None`` when no caps apply.
    """
    driver = build_env.get("DETECTED_DRIVER_VERSION")
    driver_major = int(driver) if driver and driver.isdigit() else None
    applied = [cap.requirement for cap in parse_caps(CAPS_FILE.read_text()) if cap_applies(cap.condition, driver_major)]
    machine = f"NVIDIA driver {driver_major}" if driver_major is not None else "no NVIDIA driver"
    env_caps = (env_path / ENV_CAPS_NAME).absolute()
    env_caps.write_text(
        f"# Caps from {CAPS_FILE.name} that apply with {machine}\n" + "".join(f"{r}\n" for r in applied)
    )
    return env_caps if applied else None

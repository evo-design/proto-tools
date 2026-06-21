"""X3DNA fiber standalone runner for ToolInstance venv execution."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)


def _resolve_x3dna_dir(x3dna_dir: str | None) -> str:
    """Resolve a local X3DNA install root containing ``bin/fiber``.

    Resolution order: explicit ``x3dna_dir`` (config), the standard ``X3DNA`` environment
    variable, then the resolved tool cache (``PROTO_X3DNA_WEIGHTS_DIR`` / ``PROTO_MODEL_CACHE``).
    X3DNA is user-provisioned (CC-BY-NC-4.0) and not auto-downloaded.
    """
    candidates: list[str] = []
    if x3dna_dir:
        candidates.append(x3dna_dir)
    if os.environ.get("X3DNA"):
        candidates.append(os.environ["X3DNA"])
    try:
        from standalone_helpers import resolve_weights_dir

        resolved = resolve_weights_dir("x3dna")
        if resolved:
            candidates.append(resolved)
    except Exception as exc:  # resolve_weights_dir is best-effort here
        logger.debug("resolve_weights_dir('x3dna') unavailable: %s", exc)

    for candidate in candidates:
        if candidate and (Path(candidate) / "bin" / "fiber").is_file():
            return candidate
    raise FileNotFoundError(
        "x3dna-fiber: could not find an X3DNA install with bin/fiber. Install X3DNA v2.4 "
        "(https://x3dna.org, CC-BY-NC-4.0) and set the X3DNA environment variable (or the "
        "tool's x3dna_dir config) to its root. Searched: " + ", ".join(candidates or ["<none>"])
    )


def _build_fiber_structure(sequence: str, form_flag: str, single_stranded: bool, x3dna_root: str) -> str:
    """Run X3DNA ``fiber`` for one sequence and return the generated PDB content."""
    env = os.environ.copy()
    env["X3DNA"] = x3dna_root
    env["PATH"] = f"{Path(x3dna_root) / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    fiber_bin = str(Path(x3dna_root) / "bin" / "fiber")

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_pdb = Path(tmp_dir) / "fiber.pdb"
        cmd = [fiber_bin, form_flag]
        if single_stranded:
            cmd.append("-single")
        cmd += [f"-seq={sequence}", str(out_pdb)]
        result = subprocess.run(cmd, env=env, cwd=tmp_dir, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not out_pdb.is_file():
            raise RuntimeError(
                f"x3dna fiber failed (rc={result.returncode}) for sequence {sequence!r}:\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return out_pdb.read_text()


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    x3dna_root = _resolve_x3dna_dir(input_dict.get("x3dna_dir"))
    form_flag = input_dict["form_flag"]
    single_stranded = bool(input_dict.get("single_stranded", False))
    structures = [
        _build_fiber_structure(sequence, form_flag, single_stranded, x3dna_root) for sequence in input_dict["sequences"]
    ]
    return {"structures": structures}


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CPU-only fiber generation."""
    return {"success": True, "device": device, "note": "CPU-only tool"}


def get_memory_stats() -> dict[str, Any]:
    """X3DNA fiber is a CPU-only external binary; no framework memory to report."""
    return {"available": False, "framework": "cpu", "note": "CPU tool"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"x3dna fiber: usage: {sys.argv[0]} <input_json_path> <output_json_path>", file=sys.stderr)
        sys.exit(1)
    input_path, output_path = sys.argv[1], sys.argv[2]
    with open(input_path) as handle:
        payload = json.load(handle)
    output = dispatch(payload)
    with open(output_path, "w") as handle:
        json.dump(output, handle)

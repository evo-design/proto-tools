"""IPSAE standalone inference — wraps DunbrackLab/IPSAE CLI for interface scoring."""

import json
import os
import subprocess
import sys
import tempfile
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)


def _resolve_ipsae_script() -> str:
    """Return the path to ipsae.py, which setup.sh installs into the venv.

    Check the override first, then the venv location. Finally, check beside
    this module so an ejected standalone can still locate the script.
    """
    override = os.environ.get("IPSAE_SCRIPT_PATH")
    candidates = [
        *([override] if override else []),
        os.path.join(sys.prefix, "share", "ipsae", "ipsae.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ipsae.py"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"ipsae: ipsae.py not found (looked in {candidates}); "
        "set IPSAE_SCRIPT_PATH or rebuild the environment so setup.sh downloads it"
    )


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run IPSAE scoring on a cofolded complex.

    Write temporary PDB and PAE JSON files, invoke ipsae.py, parse the
    tab-delimited output file, and return metrics for each chain pair.
    """
    # Ensure that the requested operation is supported
    operation = input_dict["operation"]
    if operation != "score":
        raise ValueError(f"ipsae: unknown operation {operation!r}; valid: ['score']")

    # Unpack the structure, PAE matrix, pLDDT and cutoffs
    pdb_content: str = input_dict["pdb_content"]
    pae: list[list[float]] = input_dict["pae"]
    plddt: list[float] | None = input_dict.get("plddt")
    pae_cutoff: float = float(input_dict.get("pae_cutoff", 10))
    dist_cutoff: float = float(input_dict.get("dist_cutoff", 10))

    ipsae_script = _resolve_ipsae_script()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write the PDB and the PAE (plus pLDDT) JSON that ipsae.py reads
        pdb_path = os.path.join(tmp_dir, "complex.pdb")
        pae_path = os.path.join(tmp_dir, "complex_scores.json")

        with open(pdb_path, "w") as f:
            f.write(pdb_content)

        pae_json: dict[str, Any] = {"pae": pae}
        if plddt is not None:
            pae_json["plddt"] = plddt
        with open(pae_path, "w") as f:
            json.dump(pae_json, f)

        # Run ipsae.py on the complex
        result = subprocess.run(
            [sys.executable, ipsae_script, pae_path, pdb_path, str(pae_cutoff), str(dist_cutoff)],
            capture_output=True,
            text=True,
            cwd=tmp_dir,
            check=False,
        )

        if result.returncode != 0:
            stderr_tail = " | ".join((result.stderr or "").strip().splitlines()) or "<no stderr>"
            raise RuntimeError(f"ipsae: ipsae.py failed (exit {result.returncode}); last stderr: {stderr_tail}")

        # Locate the output file, which ipsae.py names after the two-digit cutoffs
        pae_str = f"{int(pae_cutoff):02d}"
        dist_str = f"{int(dist_cutoff):02d}"
        output_file = os.path.join(tmp_dir, f"complex_{pae_str}_{dist_str}.txt")

        if not os.path.isfile(output_file):
            candidates = [f for f in os.listdir(tmp_dir) if f.endswith(".txt")]
            raise FileNotFoundError(
                f"ipsae: expected output file {output_file} not found; tmp_dir contents: {candidates}"
            )

        # Parse the per-chain-pair rows from the output table
        chain_pair_results = _parse_output(output_file)

    return {"chain_pair_results": chain_pair_results}


def _try_float(value: str) -> str | float:
    """Try to convert a string to float, return the string if it fails."""
    try:
        return float(value)
    except ValueError:
        return value


def _parse_output(output_file: str) -> list[dict[str, str | float]]:
    """Parse IPSAE tab-delimited output file into list of chain-pair dicts."""
    results: list[dict[str, str | float]] = []
    with open(output_file) as f:
        header: list[str] | None = None
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts, strict=False))
            results.append({k: _try_float(v) for k, v in row.items()})
    return results


def to_device(device: str) -> dict[str, Any]:
    """CPU tool — no device management needed."""
    return {"success": True, "device": device, "note": "CPU tool, no persistent model"}


def get_memory_stats() -> dict[str, Any]:
    """CPU tool — no GPU memory tracking."""
    return {"available": False, "framework": "cpu", "reason": "CPU tool"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("ipsae: usage: python inference.py <input_json_path> <output_json_path>")
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    with open(input_path) as f:
        input_data = json.load(f)
    output = dispatch(input_data)
    with open(output_path, "w") as f:
        json.dump(output, f)

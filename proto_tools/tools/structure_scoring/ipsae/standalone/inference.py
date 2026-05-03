"""IPSAE standalone inference — wraps DunbrackLab/IPSAE CLI for interface scoring."""

import json
import os
import subprocess
import sys
import tempfile
from typing import Any


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run IPSAE scoring on a cofolded complex.

    Writes temporary PDB + PAE JSON files, invokes ipsae.py, parses the
    tab-delimited output file, and returns per-chain-pair metrics.
    """
    operation = input_dict["operation"]
    if operation != "score":
        raise ValueError(f"ipsae: unknown operation {operation!r}; valid: ['score']")

    pdb_content: str = input_dict["pdb_content"]
    pae_matrix: list[list[float]] = input_dict["pae_matrix"]
    plddt: list[float] | None = input_dict.get("plddt")
    pae_cutoff: float = float(input_dict.get("pae_cutoff", 10))
    dist_cutoff: float = float(input_dict.get("dist_cutoff", 10))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ipsae_script = os.path.join(script_dir, "ipsae.py")

    if not os.path.isfile(ipsae_script):
        raise FileNotFoundError(f"ipsae: ipsae.py not found at {ipsae_script}; run setup.sh first")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdb_path = os.path.join(tmp_dir, "complex.pdb")
        pae_path = os.path.join(tmp_dir, "complex_scores.json")

        with open(pdb_path, "w") as f:
            f.write(pdb_content)

        pae_json: dict[str, Any] = {"pae": pae_matrix}
        if plddt is not None:
            pae_json["plddt"] = plddt
        with open(pae_path, "w") as f:
            json.dump(pae_json, f)

        result = subprocess.run(
            [sys.executable, ipsae_script, pae_path, pdb_path, str(pae_cutoff), str(dist_cutoff)],
            capture_output=True,
            text=True,
            cwd=tmp_dir,
            check=False,
        )

        if result.returncode != 0:
            stderr_tail = " | ".join((result.stderr or "").strip().splitlines()[-10:]) or "<no stderr>"
            raise RuntimeError(f"ipsae: ipsae.py failed (exit {result.returncode}); last stderr: {stderr_tail}")

        pae_str = f"{int(pae_cutoff):02d}"
        dist_str = f"{int(dist_cutoff):02d}"
        output_file = os.path.join(tmp_dir, f"complex_{pae_str}_{dist_str}.txt")

        if not os.path.isfile(output_file):
            candidates = [f for f in os.listdir(tmp_dir) if f.endswith(".txt")]
            raise FileNotFoundError(
                f"ipsae: expected output file {output_file} not found; tmp_dir contents: {candidates}"
            )

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
    return {"available": False, "framework": "cpu", "note": "CPU tool"}


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

"""DSSP standalone runner for ToolInstance venv execution."""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


def _find_dssp_binary() -> str:
    """Find DSSP in the tool venv, accepting either modern or legacy binary names."""
    bin_dir = Path(sys.executable).parent
    for name in ("mkdssp", "dssp"):
        candidate = bin_dir / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"DSSP binary not found in {bin_dir}. Recreate the dssp standalone environment.")


def _secondary_structure_percentages(pdb_content: str, chain_id: str, dssp_binary: str) -> dict[str, float]:
    """Compute BindCraft-style helix/sheet/loop percentages for one chain."""
    from Bio.PDB.DSSP import DSSP
    from Bio.PDB.PDBParser import PDBParser

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "input.pdb"
        tmp_path.write_text(pdb_content)

        parser = PDBParser(QUIET=True)  # type: ignore[no-untyped-call]
        model = parser.get_structure("protein", str(tmp_path))[0]  # type: ignore[no-untyped-call]
        if chain_id not in model:
            raise ValueError(f"Chain {chain_id!r} not found in DSSP input PDB")

        dssp = DSSP(model, str(tmp_path), dssp=dssp_binary)  # type: ignore[no-untyped-call]
        ss_counts = {"helix": 0, "sheet": 0, "loop": 0}
        for residue in model[chain_id]:
            if residue.id[0].strip():
                continue
            key = (chain_id, residue.id)
            if key not in dssp:
                continue

            ss_code = dssp[key][2]
            if ss_code in ("H", "G", "I"):
                ss_counts["helix"] += 1
            elif ss_code == "E":
                ss_counts["sheet"] += 1
            else:
                ss_counts["loop"] += 1

        total = sum(ss_counts.values())
        if total == 0:
            raise ValueError(f"DSSP produced no residue assignments for chain {chain_id!r}")
        return {
            "helix_pct": round(ss_counts["helix"] / total * 100.0, 2),
            "sheet_pct": round(ss_counts["sheet"] / total * 100.0, 2),
            "loop_pct": round(ss_counts["loop"] / total * 100.0, 2),
        }


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    dssp_binary = _find_dssp_binary()
    return {
        "results": [
            _secondary_structure_percentages(pdb_content, chain_id, dssp_binary)
            for pdb_content, chain_id in zip(input_dict["pdb_contents"], input_dict["chain_ids"], strict=True)
        ]
    }


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CPU-only DSSP assignment."""
    return {"success": True, "device": device, "note": "CPU-only tool"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_json_path> <output_json_path>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    output_data = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(output_data, f)

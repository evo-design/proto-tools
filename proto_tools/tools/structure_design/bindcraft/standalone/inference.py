"""BindCraft binder-design CLI subprocess wrapper for ToolInstance dispatch."""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from standalone_helpers import get_jax_memory_stats, get_subprocess_device_env, resolve_weights_dir

# ============================================================================
# Resolved at import time — point at the BindCraft repo cloned by setup.sh
# ============================================================================

_VENV_PATH = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VIRTUAL_ENV") or os.environ.get("VENV_PATH")
if not _VENV_PATH:
    raise RuntimeError(
        "BindCraft inference requires TOOL_VENV_PATH / VIRTUAL_ENV / VENV_PATH to be set. "
        "This script must be invoked via ToolInstance."
    )
_BINDCRAFT_DIR = Path(_VENV_PATH) / "data" / "BindCraft"
_DEFAULT_FILTERS_PATH = Path(__file__).parent / "default_filters.json"


# ============================================================================
# Upstream final_design_stats.csv → BindCraftMetrics field map
# ============================================================================
# Mirrors `Average_*` columns the upstream filter check evaluates against.
# Keys are the raw CSV headers; values are the snake_case names used in
# proto_tools' BindCraftMetrics.metric_spec.

_METRIC_COLUMN_MAP: dict[str, str] = {
    "Average_pLDDT": "avg_plddt",
    "Average_pTM": "avg_ptm",
    "Average_i_pTM": "avg_iptm",
    "Average_pAE": "avg_pae",
    "Average_i_pAE": "avg_ipae",
    "Average_i_pLDDT": "avg_iplddt",
    "Average_ss_pLDDT": "avg_ss_plddt",
    "Average_Binder_pLDDT": "avg_binder_plddt",
    "Average_Binder_pTM": "avg_binder_ptm",
    "Average_Binder_pAE": "avg_binder_pae",
    "Average_Binder_Energy_Score": "binder_energy_score",
    "Average_dG": "dG",
    "Average_dSASA": "dSASA",
    "Average_dG/dSASA": "dG_per_dSASA",
    "Average_Interface_SASA_%": "interface_sasa_pct",
    "Average_Interface_Hydrophobicity": "interface_hydrophobicity",
    "Average_Surface_Hydrophobicity": "surface_hydrophobicity",
    "Average_ShapeComplementarity": "shape_complementarity",
    "Average_PackStat": "packstat",
    "Average_n_InterfaceHbonds": "n_interface_hbonds",
    "Average_InterfaceHbondsPercentage": "interface_hbonds_pct",
    "Average_n_InterfaceUnsatHbonds": "n_interface_unsat_hbonds",
    "Average_InterfaceUnsatHbondsPercentage": "interface_unsat_hbonds_pct",
    "Average_n_InterfaceResidues": "n_interface_residues",
    "Average_Binder_Helix%": "binder_helix_pct",
    "Average_Binder_BetaSheet%": "binder_betasheet_pct",
    "Average_Binder_Loop%": "binder_loop_pct",
    "Average_Interface_Helix%": "interface_helix_pct",
    "Average_Interface_BetaSheet%": "interface_betasheet_pct",
    "Average_Interface_Loop%": "interface_loop_pct",
    "Average_Hotspot_RMSD": "hotspot_rmsd",
    "Average_Target_RMSD": "target_rmsd",
    "Average_Binder_RMSD": "binder_rmsd",
    "Average_Unrelaxed_Clashes": "unrelaxed_clashes",
    "Average_Relaxed_Clashes": "relaxed_clashes",
}

_MISSING_VALUES = {"", "None", "nan", "NaN", "NA", "null"}


# ============================================================================
# Settings builders
# ============================================================================


def _materialize_pdb(pdb: str, work_dir: Path) -> str:
    """Return a filesystem path to the target PDB.

    If ``pdb`` is already a path to an existing file, return it unchanged.
    If it's PDB-format content (starts with HEADER/ATOM/MODEL/CRYST1/REMARK),
    write it into ``work_dir/target.pdb`` and return that path.
    """
    candidate = pdb.lstrip()
    if candidate.startswith(("HEADER", "ATOM", "MODEL", "CRYST1", "REMARK", "HETATM")):
        out_path = work_dir / "target.pdb"
        out_path.write_text(pdb)
        return str(out_path)
    if os.path.isfile(pdb):
        return os.path.abspath(pdb)
    raise FileNotFoundError(f"target_pdb is neither a PDB-format string nor an existing file path: {pdb!r}")


def _build_target_settings(payload: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    """Build BindCraft's target_settings.json from the dispatch payload."""
    return {
        "design_path": str(work_dir / "designs"),
        "binder_name": payload["binder_name"],
        "starting_pdb": _materialize_pdb(payload["target_pdb"], work_dir),
        "chains": payload["target_chain"],
        "target_hotspot_residues": payload.get("target_hotspot_residues") or "",
        "lengths": list(payload["binder_lengths"]),
        "number_of_final_designs": payload["number_of_final_designs"],
    }


def _build_advanced_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Build BindCraft's advanced_settings.json from the dispatch payload.

    Pulls all upstream knobs (51 user-facing + 10 hardcoded internals) from
    ``advanced_settings`` and injects the three infrastructure paths
    (af_params_dir, dssp_path, dalphaball_path).
    """
    advanced: dict[str, Any] = dict(payload["advanced_settings"])
    af2_weights = resolve_weights_dir("alphafold2")
    if not af2_weights:
        raise RuntimeError(
            "AlphaFold2 weights directory could not be resolved. "
            "Re-run setup.sh or set PROTO_ALPHAFOLD2_WEIGHTS_DIR / PROTO_BINDCRAFT_WEIGHTS_DIR."
        )
    # ColabDesign's get_model_haiku_params() looks for params at
    # ``{data_dir}/params/params_{model}.npz``. Our setup.sh writes the
    # weights to ``{af2_weights}/params/*.npz`` (matching the alphafold2
    # toolkit), so af_params_dir is the parent — not the params/ subdir.
    advanced["af_params_dir"] = str(af2_weights)
    advanced["dssp_path"] = str(_BINDCRAFT_DIR / "functions" / "dssp")
    advanced["dalphaball_path"] = str(_BINDCRAFT_DIR / "functions" / "DAlphaBall.gcc")
    return advanced


def _build_filters(payload: dict[str, Any]) -> dict[str, Any]:
    """Load upstream default_filters.json and merge per-metric overrides on top."""
    base: dict[str, Any] = json.loads(_DEFAULT_FILTERS_PATH.read_text())
    overrides = payload.get("filter_overrides") or {}
    base.update(overrides)
    return base


# ============================================================================
# Output parsers
# ============================================================================


def _row_to_metrics(row: dict[str, str]) -> dict[str, float]:
    """Extract metric columns from a final_design_stats.csv row.

    Skips empty / sentinel-missing entries (some metrics may not be set,
    e.g. when an upstream filter step short-circuits).
    """
    out: dict[str, float] = {}
    for csv_col, metric_key in _METRIC_COLUMN_MAP.items():
        val = row.get(csv_col, "")
        if val.strip() in _MISSING_VALUES:
            continue
        try:
            out[metric_key] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _parse_int_list(s: str) -> list[int]:
    """Parse 'A1,A2,A3' or '1,2,3' style strings into 1-indexed residue positions."""
    if not s:
        return []
    out: list[int] = []
    for tok in s.split(","):
        digits = re.sub(r"[^0-9]", "", tok)
        if digits:
            out.append(int(digits))
    return out


def _parse_interface_aas(s: str) -> dict[str, int]:
    """Parse a Python-dict-as-string or JSON dict cell into {aa: count}."""
    if not s:
        return {}
    # CSV cells from upstream are stringified Python dicts (single quotes).
    # JSON requires double quotes — flip them, then parse.
    try:
        return {str(k): int(v) for k, v in json.loads(s.replace("'", '"')).items()}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _resolve_accepted_pdb(designs_dir: Path, design_name: str) -> Path | None:
    """Find the accepted PDB for a design.

    Upstream stores the best-model PDB at ``Accepted/{design_name}_model{N}.pdb``
    where N is the AF2 model index (1-5) chosen by best pLDDT. The CSV row
    only carries the design_name (without the model suffix), so we glob for it.
    """
    accepted_dir = designs_dir / "Accepted"
    if not accepted_dir.is_dir():
        return None
    matches = sorted(accepted_dir.glob(f"{design_name}_model*.pdb"))
    if matches:
        return matches[0]
    # Fallback: some pipelines may write the file without the model suffix.
    direct = accepted_dir / f"{design_name}.pdb"
    return direct if direct.exists() else None


def _count_trajectories(designs_dir: Path) -> int:
    """Count trajectories attempted from trajectory_stats.csv (one row per trajectory)."""
    stats_csv = designs_dir / "trajectory_stats.csv"
    if not stats_csv.exists():
        return 0
    with stats_csv.open() as f:
        # Subtract 1 for the header row; clamp at zero.
        return max(0, sum(1 for _ in f) - 1)


def _parse_outputs(designs_dir: Path) -> dict[str, Any]:
    """Read final_design_stats.csv + Accepted/ PDBs into the dispatch result dict."""
    designs: list[dict[str, Any]] = []
    final_csv = designs_dir / "final_design_stats.csv"
    if final_csv.exists():
        with final_csv.open() as f:
            for row in csv.DictReader(f):
                design_name = (row.get("Design") or "").strip()
                if not design_name:
                    continue
                pdb_path = _resolve_accepted_pdb(designs_dir, design_name)
                if pdb_path is None:
                    # No PDB on disk — skip; the row may be a header artifact
                    # or a rejected-design footer in some upstream variants.
                    continue
                designs.append(
                    {
                        "design_name": design_name,
                        "binder_sequence": (row.get("Sequence") or "").strip(),
                        "pdb": pdb_path.read_text(),
                        "metrics": _row_to_metrics(row),
                        "seed": int((row.get("Seed") or "0").strip() or 0),
                        "interface_aas": _parse_interface_aas(row.get("Average_InterfaceAAs", "")),
                        "interface_residues": _parse_int_list(row.get("InterfaceResidues", "")),
                    }
                )
    n_trajectories = _count_trajectories(designs_dir)
    # If trajectory_stats.csv is missing or empty, fall back to the accepted
    # count so the report is monotonically sane (you can't have accepted more
    # designs than you ran trajectories for).
    n_trajectories = max(n_trajectories, len(designs))
    return {
        "designs": designs,
        "n_trajectories_run": n_trajectories,
        "n_designs_accepted": len(designs),
    }


# ============================================================================
# Dispatch entry point
# ============================================================================


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for ToolInstance.dispatch.

    Materializes BindCraft's three JSON settings files in a temp dir, invokes
    ``bindcraft.py`` as a subprocess, and parses ``final_design_stats.csv`` +
    ``Accepted/*.pdb`` back into the design list expected by
    ``run_bindcraft_design``.
    """
    operation = input_dict["operation"]
    if operation != "design":
        raise ValueError(f"Unknown BindCraft operation: {operation}")

    if not _BINDCRAFT_DIR.is_dir():
        raise FileNotFoundError(f"BindCraft repo not found at {_BINDCRAFT_DIR}. Re-run setup.sh.")

    with tempfile.TemporaryDirectory(prefix="bindcraft_") as work_dir_str:
        work_dir = Path(work_dir_str)
        (work_dir / "designs").mkdir(parents=True, exist_ok=True)

        target = _build_target_settings(input_dict, work_dir)
        advanced = _build_advanced_settings(input_dict)
        filters = _build_filters(input_dict)

        target_path = work_dir / "target_settings.json"
        advanced_path = work_dir / "advanced_settings.json"
        filters_path = work_dir / "filters.json"
        target_path.write_text(json.dumps(target, indent=2))
        advanced_path.write_text(json.dumps(advanced, indent=2))
        filters_path.write_text(json.dumps(filters, indent=2))

        env = get_subprocess_device_env(input_dict.get("device", "cuda"))
        # PYTHONHASHSEED gives the bindcraft.py subprocess deterministic hash randomization
        # when a seed is supplied (matters for set/dict iteration in upstream's design loop).
        seed = input_dict.get("seed")
        if seed is not None:
            env["PYTHONHASHSEED"] = str(seed)

        cmd = [
            sys.executable,
            str(_BINDCRAFT_DIR / "bindcraft.py"),
            "--settings",
            str(target_path),
            "--advanced",
            str(advanced_path),
            "--filters",
            str(filters_path),
        ]

        # cwd = the BindCraft repo; bindcraft.py uses os.path.dirname(__file__) for
        # asset resolution (functions/, settings_*/), so launching from elsewhere
        # works, but staying in-repo avoids any hidden relative-path surprises.
        verbose = bool(input_dict.get("verbose", False))
        try:
            subprocess.run(
                cmd,
                env=env,
                cwd=str(_BINDCRAFT_DIR),
                check=True,
                text=True,
                encoding="utf-8",
                stdout=sys.stdout if verbose else subprocess.PIPE,
                stderr=sys.stderr if verbose else subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            stderr_tail = " | ".join((e.stderr or "").strip().splitlines()[-10:]) or "<no stderr>"
            raise RuntimeError(f"bindcraft: failed (exit {e.returncode}): {stderr_tail}") from e

        return _parse_outputs(work_dir / "designs")


# ============================================================================
# DeviceManager protocol (passthrough — CLI subprocess auto-unloads)
# ============================================================================


def to_device(device: str) -> dict[str, Any]:
    """DeviceManager callback. CLI subprocess flows manage GPU per call.

    Returns:
        dict[str, Any]: Status dict noting the passthrough.
    """
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """Report JAX memory stats for the worker's first visible device."""
    stats: dict[str, Any] = get_jax_memory_stats(device_index=0)
    return stats


# ============================================================================
# Worker protocol entry point
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")
    with open(sys.argv[1]) as f:
        result = dispatch(json.load(f))
    with open(sys.argv[2], "w") as f:
        json.dump(result, f)

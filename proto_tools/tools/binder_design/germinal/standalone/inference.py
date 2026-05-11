"""Germinal antibody design — Hydra subprocess driver."""

from __future__ import annotations

import csv as _csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)

# Typed Config threshold fields → filter YAML keys (shape: {value, operator}).
_TYPED_FILTER_KEY_MAP: dict[str, str] = {
    "plddt_threshold": "external_plddt",
    "iptm_threshold": "external_iptm",
    "ipae_threshold": "external_pae",
    "ptm_threshold": "external_ptm",
    "pdockq2_threshold": "pdockq2",
}

_KNOWN_INT_METRICS: frozenset[str] = frozenset(
    {
        "clashes",
        "clashes_unrelaxed",
        "interface_hbonds",
        "interface_nres",
        "interface_delta_unsat_hbonds",
        "cdr3_hotspot_contacts",
        "cdr_hotspot_contacts",
        "hydrophobic_patches_binder",
        "hydrophobic_patches_struct",
        "n_framework_mutations",
    }
)
_KNOWN_BOOL_METRICS: frozenset[str] = frozenset({"binder_near_hotspot"})

# `<target>_<run_type>_s<seed>` or `<...>_s<seed>_abmpnn_<j>` (j=0 when absent).
_DESIGN_NAME_INDICES = re.compile(r"_s(\d+)(?:_abmpnn_(\d+))?$")


class GerminalRunner:
    """Stateless runner that locates the Germinal repo + weights and dispatches one campaign."""

    def __init__(self) -> None:
        """Lazy-init; setup happens in `load()` on first call."""
        self._loaded = False
        self.germinal_repo: Path | None = None
        self.weights_dir: Path | None = None
        self.af_params_dir: Path | None = None

    def load(self) -> None:
        """Resolve the cloned Germinal repo and AF-Multimer params directory."""
        from standalone_helpers import resolve_weights_dir

        venv = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VIRTUAL_ENV")
        if not venv:
            raise RuntimeError(
                "Neither TOOL_VENV_PATH nor VIRTUAL_ENV is set. Germinal must be invoked via ToolInstance."
            )
        repo = Path(venv) / "data" / "germinal"
        if not (repo / "run_germinal.py").exists():
            raise FileNotFoundError(f"Germinal repository not found at {repo}. Re-run standalone/setup.sh.")
        self.germinal_repo = repo

        weights_dir = resolve_weights_dir("germinal")
        if weights_dir is None:
            raise RuntimeError("resolve_weights_dir('germinal') returned None — weights not provisioned.")
        self.weights_dir = Path(weights_dir)

        # Resolve af_params_dir: prefer the .params_redirect file written by
        # setup.sh when reusing the AF2 standalone's params; otherwise use
        # $WEIGHTS_DIR/params (where setup.sh downloads them).
        redirect = self.weights_dir / ".params_redirect"
        if redirect.exists():
            for line in redirect.read_text().splitlines():
                if "=" in line:
                    key, _, val = line.partition("=")
                    if key.strip() == "PROTO_GERMINAL_AF_PARAMS_DIR":
                        self.af_params_dir = Path(val.strip())
                        break
        if self.af_params_dir is None:
            self.af_params_dir = self.weights_dir / "params"

        if not self.af_params_dir.exists() or not list(self.af_params_dir.glob("*.npz")):
            raise FileNotFoundError(
                f"AlphaFold-Multimer params not found at {self.af_params_dir}. "
                "Re-run standalone/setup.sh to download them."
            )

        self._loaded = True
        logger.debug(
            "Germinal loaded: repo=%s weights=%s af_params_dir=%s",
            self.germinal_repo,
            self.weights_dir,
            self.af_params_dir,
        )

    def __call__(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Run one Germinal campaign end-to-end and return parsed outputs."""
        from standalone_helpers import set_torch_seed

        if not self._loaded:
            self.load()

        seed = input_dict.get("seed")
        set_torch_seed(seed)

        target_pdb_in = input_dict["target_pdb"]
        target_name = input_dict.get("target_name") or self._hash_pdb(target_pdb_in)

        persistent = input_dict.get("output_dir")
        if persistent:
            workdir = Path(persistent)
            workdir.mkdir(parents=True, exist_ok=True)
            return self._run_in(workdir, target_name, target_pdb_in, input_dict)

        with tempfile.TemporaryDirectory(prefix="germinal_") as tmp:
            return self._run_in(Path(tmp), target_name, target_pdb_in, input_dict)

    def _run_in(
        self,
        workdir: Path,
        target_name: str,
        target_pdb_in: str,
        input_dict: dict[str, Any],
    ) -> dict[str, Any]:
        from standalone_helpers import get_subprocess_device_env

        assert self.germinal_repo is not None
        assert self.af_params_dir is not None

        self._write_target_pdb(workdir, target_name, target_pdb_in)
        self._write_overlay_configs(workdir, target_name, input_dict)
        cmd = self._build_cmd(workdir, target_name, input_dict)
        env = {**os.environ, **get_subprocess_device_env(input_dict.get("device", "cuda"))}

        logger.debug("Spawning Germinal subprocess: %s", " ".join(cmd))
        repo_cwd = str(self.germinal_repo)
        verbose = input_dict.get("verbose", False)
        try:
            subprocess.run(
                cmd,
                cwd=repo_cwd,
                env=env,
                check=True,
                text=True,
                encoding="utf-8",
                stdout=sys.stdout if verbose else subprocess.PIPE,
                stderr=sys.stderr if verbose else subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            stderr_tail = " | ".join((e.stderr or "").strip().splitlines()[-10:]) or "<no stderr>"
            raise RuntimeError(f"germinal: failed (exit {e.returncode}): {stderr_tail}") from e
        return self._collect_outputs(workdir, target_name, input_dict)

    @staticmethod
    def _hash_pdb(pdb: str) -> str:
        """Derive a stable target name from PDB content (file path or PDB string)."""
        body = Path(pdb).read_text() if Path(pdb).is_file() else pdb
        return "target_" + hashlib.sha256(body.encode()).hexdigest()[:8]

    @staticmethod
    def _write_target_pdb(workdir: Path, target_name: str, source: str) -> None:
        out = workdir / "pdbs" / f"{target_name}.pdb"
        out.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).is_file():
            shutil.copy(source, out)
        else:
            out.write_text(source)

    def _write_overlay_configs(self, workdir: Path, target_name: str, d: dict[str, Any]) -> None:
        """Write target + per-call filter overlay YAMLs into ``{workdir}/configs/``.

        Filter overlays use a unique ``<design_type>_<target_name>`` stem because
        Hydra's ``+hydra.searchpath=`` is appended AFTER the primary config_path
        — writing under the upstream stem would silently lose to the bundled
        preset. ``_build_cmd`` selects the unique stem via ``filter/initial=``.
        """
        import yaml

        assert self.germinal_repo is not None
        cfg_root = workdir / "configs"
        for sub in ("target", "filter/initial", "filter/final"):
            (cfg_root / sub).mkdir(parents=True, exist_ok=True)

        # target/<name>.yaml — Germinal expects target_hotspots as a comma-separated string
        target_yaml: dict[str, Any] = {
            "target_name": target_name,
            "target_pdb_path": str(workdir / "pdbs" / f"{target_name}.pdb"),
            "target_chain": d["target_chain"],
            "binder_chain": d["binder_chain"],
            "target_hotspots": ",".join(d.get("hotspots", []) or []),
        }
        if d.get("hotspot_residue"):
            target_yaml["hotspot_residue"] = d["hotspot_residue"]
        (cfg_root / "target" / f"{target_name}.yaml").write_text(yaml.safe_dump(target_yaml, sort_keys=False))

        # filter overlays — read upstream preset, apply typed thresholds + free-form dict overrides
        design_type = d["design_type"]
        overlay_stem = f"{design_type}_{target_name}"
        for stage in ("initial", "final"):
            preset = self.germinal_repo / "configs" / "filter" / stage / f"{design_type}.yaml"
            if not preset.exists():
                logger.warning("Germinal preset filter not found at %s; skipping overlay", preset)
                continue
            base: dict[str, Any] = yaml.safe_load(preset.read_text()) or {}

            if stage == "final":
                for cfg_field, filter_key in _TYPED_FILTER_KEY_MAP.items():
                    val = d.get(cfg_field)
                    if val is None:
                        continue
                    entry = base.get(filter_key)
                    if isinstance(entry, dict) and "value" in entry:
                        entry["value"] = val
                    else:
                        # Filter not in preset — synthesize the canonical operator
                        operator = "<" if filter_key == "external_pae" else ">"
                        base[filter_key] = {"value": val, "operator": operator}
                    logger.info("Applied typed threshold override: filter.final.%s.value = %s", filter_key, val)

            for key, override in (d.get("filter_overrides") or {}).get(stage, {}).items():
                base[key] = override
                logger.info("Applied %s filter override: %s = %s", stage, key, override)

            (cfg_root / "filter" / stage / f"{overlay_stem}.yaml").write_text(
                yaml.safe_dump(base, sort_keys=False),
            )

    def _build_cmd(self, workdir: Path, target_name: str, d: dict[str, Any]) -> list[str]:
        """Compose the ``python run_germinal.py`` Hydra invocation."""
        assert self.af_params_dir is not None

        overlay_stem = f"{d['design_type']}_{target_name}"
        run_dir = workdir / "runs" / target_name
        cmd: list[str] = [
            sys.executable,
            "run_germinal.py",
            f"hydra.searchpath=[file://{workdir / 'configs'}]",
            f"hydra.run.dir={run_dir}",
            f"project_dir={workdir}",
            "results_dir=runs",
            "run_config=",
            f"target={target_name}",
            f"run={d['design_type']}",
            f"filter/initial={overlay_stem}",
            f"filter/final={overlay_stem}",
            f"experiment_name={target_name}",
            f"max_trajectories={d['max_trajectories']}",
            f"max_hallucinated_trajectories={d['max_hallucinated_trajectories']}",
            f"max_passing_designs={d['max_passing_designs']}",
            f"structure_model={d['structure_model']}",
            f"af_params_dir={self.af_params_dir}",
        ]
        # cfg.seed is not in upstream's struct config; numpy/torch seeded via set_torch_seed.
        for key, val in (d.get("germinal_overrides") or {}).items():
            cmd.append(f"{key}={val}")
        return cmd

    def _collect_outputs(
        self,
        workdir: Path,
        target_name: str,
        d: dict[str, Any],
    ) -> dict[str, Any]:
        """Walk Germinal's output tree and assemble a JSON-serialisable result dict."""
        run_dir = workdir / "runs" / target_name
        designs: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Higher-priority stages first so a design promoted to `accepted`
        # appears once with the highest stage label.
        stages: tuple[tuple[str, str], ...] = (
            ("accepted", "accepted"),
            ("redesign_candidates", "redesign_candidate"),
            ("trajectories", "trajectory"),
        )
        for stage_dir, stage_label in stages:
            stage_root = run_dir / stage_dir
            if not stage_root.is_dir():
                continue

            csv_path = stage_root / "designs.csv"
            metrics_by_id = self._read_csv(csv_path) if csv_path.exists() else {}

            structures_dir = stage_root / "structures"
            if not structures_dir.is_dir():
                continue

            for pdb in sorted(structures_dir.glob("*.pdb")):
                design_id = pdb.stem
                if design_id in seen or design_id not in metrics_by_id:
                    continue
                seen.add(design_id)
                row = metrics_by_id[design_id]
                seq_h, seq_l = self._extract_binder_sequences(pdb, d["binder_chain"], d["design_type"])
                traj_idx, mpnn_idx = self._parse_design_indices(design_id)
                designs.append(
                    {
                        "sequence_heavy": seq_h,
                        "sequence_light": seq_l,
                        "structure_content": pdb.read_text(),
                        "metrics": self._normalize_metrics(row),
                        "stage_passed": stage_label,
                        "design_id": design_id,
                        "trajectory_index": traj_idx,
                        "mpnn_index": mpnn_idx,
                    }
                )

        # failure_counts.csv is a single-row "header → counts" CSV with no
        # design_name key column, so it's parsed flat (NOT through _read_csv).
        fail_csv = run_dir / "failure_counts.csv"
        pipeline_stats = self._read_flat_counts(fail_csv) if fail_csv.exists() else {}

        return {"designs": designs, "pipeline_stats": pipeline_stats}

    @staticmethod
    def _read_csv(path: Path) -> dict[str, dict[str, str]]:
        """Return ``{design_name: {col: val}}`` indexed by Germinal's design_name column."""
        out: dict[str, dict[str, str]] = {}
        with path.open(newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                key = row.get("design_name") or row.get("design_id") or row.get("name")
                if key:
                    out[key] = row
        return out

    @staticmethod
    def _read_flat_counts(path: Path) -> dict[str, int]:
        """Parse a one-row failure_counts.csv with header → value layout."""
        out: dict[str, int] = {}
        with path.open(newline="") as f:
            reader = _csv.reader(f)
            try:
                header = next(reader)
                values = next(reader)
            except StopIteration:
                return out
            for k, v in zip(header, values, strict=False):
                if v in (None, ""):
                    continue
                try:
                    out[k] = int(float(v))
                except (TypeError, ValueError):
                    continue
        return out

    @staticmethod
    def _normalize_metrics(row: dict[str, str]) -> dict[str, Any]:
        """Cast Germinal CSV string values to floats / ints / bools where appropriate.

        Only emits keys with non-empty values. Non-metric identity columns
        (design_name, experiment_name, cdr_lengths, target_hotspots,
        final_structure_path) are dropped — they're recovered from the PDB path
        and the original input.
        """
        skip = {
            "design_name",
            "experiment_name",
            "cdr_lengths",
            "target_hotspots",
            "final_structure_path",
        }
        out: dict[str, Any] = {}
        for k, v in row.items():
            if k in skip or v in (None, ""):
                continue
            if k in _KNOWN_BOOL_METRICS:
                out[k] = str(v).lower() in {"true", "1", "yes"}
                continue
            if k in _KNOWN_INT_METRICS:
                try:
                    out[k] = int(float(v))
                    continue
                except (TypeError, ValueError):
                    pass
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                # Fall through: keep as raw string if not numeric
                out[k] = v
        return out

    @staticmethod
    def _parse_design_indices(design_name: str) -> tuple[int, int]:
        """Extract (trajectory_index, mpnn_index) from a Germinal design_name.

        Germinal names designs ``<target>_<type>_s<seed>`` (trajectory-only)
        or ``<target>_<type>_s<seed>_abmpnn_<j+1>`` (AbMPNN-redesigned).
        Returns (seed, mpnn_idx); mpnn_idx is 0 for trajectory-only designs.
        Returns (0, 0) if the name doesn't match the upstream convention.
        """
        match = _DESIGN_NAME_INDICES.search(design_name)
        if not match:
            return 0, 0
        seed = int(match.group(1))
        mpnn_idx = int(match.group(2)) if match.group(2) else 0
        return seed, mpnn_idx

    @staticmethod
    def _extract_binder_sequences(
        pdb: Path,
        binder_chain: str,
        design_type: str,
    ) -> tuple[str, str | None]:
        """Read the binder chain sequence from the PDB.

        For VHH (single-domain), returns (heavy, None).
        For scFv, returns (heavy, light) by splitting on the largest backbone
        gap in the binder chain (>= 5 Å between consecutive CA atoms). If no
        gap is detected, falls back to (full_sequence, None).
        """
        # NOTE: Bio.PDB.Polypeptide.three_to_one was removed in BioPython 1.82;
        # protein_letters_3to1 is the supported replacement (a plain dict).
        from Bio import PDB
        from Bio.PDB.Polypeptide import protein_letters_3to1  # type: ignore[attr-defined]

        parser = PDB.PDBParser(QUIET=True)  # type: ignore[attr-defined,no-untyped-call]
        structure = parser.get_structure("design", str(pdb))  # type: ignore[no-untyped-call]
        model = next(iter(structure))
        if binder_chain not in [c.id for c in model.get_chains()]:
            return "", None
        chain = model[binder_chain]

        residues = [
            res for res in chain.get_residues() if res.id[0] == " " and res.get_resname() in protein_letters_3to1
        ]
        if not residues:
            return "", None

        full_seq = "".join(protein_letters_3to1.get(res.get_resname(), "X") for res in residues)

        if design_type != "scfv" or len(residues) < 2:
            return full_seq, None

        # Find the largest CA-CA gap to split the H + L chains
        max_gap = 0.0
        split_idx: int | None = None
        for i in range(1, len(residues)):
            try:
                ca_prev = residues[i - 1]["CA"]
                ca_curr = residues[i]["CA"]
            except KeyError:
                continue
            dist = ca_prev - ca_curr  # BioPython Atom subtraction returns the distance
            if dist > max_gap:
                max_gap = dist
                split_idx = i
        if split_idx is None or max_gap < 5.0:
            return full_seq, None
        return full_seq[:split_idx], full_seq[split_idx:]


# ============================================================================
# Module-level dispatch + device protocol (REQUIRED by ToolInstance)
# ============================================================================
_runner: GerminalRunner | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point invoked by ToolInstance. Returns a JSON-serialisable dict."""
    global _runner
    if _runner is None:
        _runner = GerminalRunner()
    return _runner(input_dict)


def to_device(device: str) -> dict[str, Any]:
    """Germinal is a CLI subprocess pattern — auto-unloads after each call."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """Report PyTorch memory; Germinal uses both PyTorch + JAX inside the subprocess."""
    from standalone_helpers import get_pytorch_memory_stats

    stats: dict[str, Any] = get_pytorch_memory_stats(device=0)
    return stats


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input.json> <output.json>")
    with open(sys.argv[1]) as fh:
        in_data = json.load(fh)
    out = dispatch(in_data)
    with open(sys.argv[2], "w") as fh:
        json.dump(out, fh)

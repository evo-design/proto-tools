"""RoseTTAFold3 (RF3) inference implementation."""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)


class RF3Model:
    """RF3 model wrapper that drives the ``rf3 fold`` CLI."""

    def __init__(self) -> None:
        """Initialize the wrapper; resolve weights cache directory."""
        self._loaded = False
        from standalone_helpers import resolve_weights_dir

        weights_dir = resolve_weights_dir("rf3")
        if weights_dir is None:
            raise RuntimeError("rf3: cannot determine cache directory; set PROTO_HOME or PROTO_MODEL_CACHE")
        self.cache_dir = Path(weights_dir)
        self.rf3_executable: str | None = None

    def __call__(
        self,
        input_json_path: str,
        output_dir: str,
        device: str = "cuda",
        n_recycles: int = 10,
        diffusion_batch_size: int = 5,
        num_steps: int = 50,
        cyclic_chains: list[str] | None = None,
        seed: int | None = None,
        verbose: int = 0,
    ) -> dict[str, Any]:
        """Run RF3 structure prediction via the ``rf3 fold`` CLI.

        Args:
            input_json_path: Path to the RF3 input JSON file.
            output_dir: Directory where ``rf3 fold`` writes its outputs.
            device: ``"cuda"`` or ``"cpu"``; controls subprocess GPU visibility.
            n_recycles: Upstream ``n_recycles``; iterative refinement passes.
            diffusion_batch_size: Independent diffusion samples per call (best
                by ``ranking_score`` is returned).
            num_steps: Denoising steps in the diffusion process.
            cyclic_chains: Chain IDs to mark as cyclic.
            seed: Random seed forwarded to ``rf3 fold`` as ``seed=<int>``.
            verbose: Stream CLI output to stderr.

        Returns:
            ``{"structure_cif_output": <cif_str>, "metrics": <metrics_dict>}``.
        """
        if not self._loaded:
            self.load(verbose)

        # early_stopping_plddt_threshold=0 overrides upstream's 0.5 default: a fired abort writes no
        # structure, which would break this tool's one-structure-per-complex contract.
        cmd: list[str] = [
            str(self.rf3_executable),
            "fold",
            f"inputs={input_json_path}",
            f"out_dir={output_dir}",
            f"n_recycles={n_recycles}",
            f"diffusion_batch_size={diffusion_batch_size}",
            f"num_steps={num_steps}",
            "early_stopping_plddt_threshold=0",
            "annotate_b_factor_with_plddt=true",
            "dump_predictions=true",
        ]
        if cyclic_chains:
            cmd.append(f"cyclic_chains=[{','.join(cyclic_chains)}]")
        if seed is not None:
            cmd.append(f"seed={seed}")

        from standalone_helpers import get_subprocess_device_env, is_cuda_oom, raise_oom, run_teed

        env = get_subprocess_device_env(device)
        existing = env.get("FOUNDRY_CHECKPOINT_DIRS", "")
        env["FOUNDRY_CHECKPOINT_DIRS"] = f"{self.cache_dir}{':' + existing if existing else ''}"
        logger.debug("Running RF3 command: %s", " ".join(cmd))
        sys.stdout.flush()

        returncode, stdout, stderr = run_teed(cmd, env=env, verbose=verbose, encoding="utf-8")
        if returncode != 0:
            stderr_tail = " | ".join(stderr.strip().splitlines()) or "<no stderr>"
            if is_cuda_oom(stderr_tail):
                raise_oom("rf3", hint="Reduce diffusion_batch_size or complex size, or use a GPU with more memory.")
            raise RuntimeError(f"rf3: rf3 fold failed (exit {returncode}): {stderr_tail}")

        captured = "\n".join(s for s in (stdout, stderr) if s) or None
        return self._extract_output(output_dir, captured=captured)

    def _extract_output(self, output_dir: str, captured: str | None = None) -> dict[str, Any]:
        """Locate the best-ranked sample's mmCIF and summary confidences."""
        # rf3 writes outputs under output_dir/<input_stem>/...; rglob locates them.
        summary_files = sorted(Path(output_dir).rglob("*_summary_confidences.json"))
        if not summary_files:
            hint = f" rf3 exited 0 but wrote no summary; output: {captured}" if captured else ""
            raise FileNotFoundError(f"rf3: summary_confidences.json not found under {output_dir}.{hint}")

        # Pick the best sample by ranking_score.
        best_score = float("-inf")
        best_summary: Path | None = None
        for path in summary_files:
            try:
                data = json.loads(path.read_text())
                score = float(data.get("ranking_score", float("-inf")))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if score > best_score:
                best_score = score
                best_summary = path

        if best_summary is None:
            raise RuntimeError(f"rf3: could not select best sample from {len(summary_files)} candidates")

        summary = json.loads(best_summary.read_text())

        sample_dir = best_summary.parent
        prefix = best_summary.name.removesuffix("_summary_confidences.json")
        cif_candidates = list(sample_dir.glob(f"{prefix}*model.cif")) + list(sample_dir.glob(f"{prefix}*model.cif.gz"))
        if not cif_candidates:
            raise FileNotFoundError(f"rf3: mmCIF not found in {sample_dir} for prefix {prefix!r}")
        cif_path = cif_candidates[0]
        if cif_path.suffix == ".gz":
            import gzip

            cif_text = gzip.decompress(cif_path.read_bytes()).decode("utf-8")
        else:
            cif_text = cif_path.read_text()

        n_chains = len(summary["chain_ptm"])
        metrics = {
            "avg_plddt": float(summary["overall_plddt"]),
            "avg_pae": float(summary["overall_pae"]),
            "pde": float(summary["overall_pde"]),
            "ptm": float(summary["ptm"]),
            "ranking_score": float(summary["ranking_score"]),
            # Upstream's `chain_ptm` key carries per-chain pLDDT, not pTM (https://github.com/RosettaCommons/foundry/blob/d8b0be6/models/rf3/src/rf3/utils/predicted_error.py#L372). Kept under the same name for consistency with the upstream JSON.
            "chain_ptm": [float(x) for x in summary["chain_ptm"]],
            "chain_pair_pae": _as_float_matrix_or_empty(summary["chain_pair_pae"], n_chains),
            "chain_pair_pae_min": _as_float_matrix_or_empty(summary["chain_pair_pae_min"], n_chains),
            "chain_pair_pde": _as_float_matrix_or_empty(summary["chain_pair_pde"], n_chains),
            "chain_pair_pde_min": _as_float_matrix_or_empty(summary["chain_pair_pde_min"], n_chains),
            "has_clash": bool(summary["has_clash"]),
        }
        # Surface iptm only for multi-chain complexes; upstream emits 0.0 for single-chain.
        if n_chains > 1 and summary.get("iptm") is not None:
            metrics["iptm"] = float(summary["iptm"])

        return {"structure_cif_output": cif_text, "metrics": metrics}

    def load(self, verbose: int = 0) -> None:  # noqa: ARG002 — required by tool interface
        """Resolve the ``rf3`` executable path."""
        venv_rf3 = Path(sys.executable).parent / "rf3"
        exe = str(venv_rf3) if venv_rf3.exists() else shutil.which("rf3")
        if not exe:
            raise ImportError("rf3: 'rf3' executable not found in current environment")
        self.rf3_executable = exe
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = True
        logger.debug("RF3 initialized; executable: %s", self.rf3_executable)


def _as_float_matrix_or_empty(value: Any, n_chains: int) -> list[list[float]]:
    """Coerce a chain-pair matrix into a finite ``list[list[float]]``; ``[]`` for single chain.

    Empirically upstream RF3 emits an upper-triangular representation: only
    ``(i, j)`` for ``i < j`` is populated; the diagonal and lower triangle
    come back as ``None``. The chain-pair aggregates (PAE, PDE, and their
    ``*_min`` variants) are symmetric per-pair quantities by construction,
    so we mirror the upper triangle into the lower triangle and coerce the
    diagonal to ``0.0`` (a chain compared to itself has zero aggregate error).
    Single-chain inputs yield ``[[None]]`` upstream, which we collapse to
    ``[]`` since "the PAE between a chain and itself" is meaningless.
    """
    import math

    if value is None or n_chains < 2:
        return []
    matrix: list[list[float]] = [[float(x) if x is not None else float("nan") for x in row] for row in value]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if not math.isnan(matrix[i][j]):
                continue
            if i == j:
                matrix[i][j] = 0.0
            elif i > j and not math.isnan(matrix[j][i]):
                matrix[i][j] = matrix[j][i]
    leftover = [(i, j) for i in range(len(matrix)) for j in range(len(matrix[i])) if math.isnan(matrix[i][j])]
    if leftover:
        logger.warning("rf3: %d non-finite chain-pair cells coerced to 0.0 (paths=%s)", len(leftover), leftover[:5])
        for i, j in leftover:
            matrix[i][j] = 0.0
    return matrix


# ============================================================================
# Dispatch
# ============================================================================
_model: RF3Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = RF3Model()

    operation = input_dict["operation"]
    if operation == "predict":
        return _model(
            input_json_path=input_dict["input_json_path"],
            output_dir=input_dict["output_dir"],
            device=input_dict["device"],
            n_recycles=input_dict["n_recycles"],
            diffusion_batch_size=input_dict["diffusion_batch_size"],
            num_steps=input_dict["num_steps"],
            cyclic_chains=input_dict.get("cyclic_chains"),
            seed=input_dict.get("seed"),
            verbose=input_dict["verbose"],
        )
    raise ValueError(f"rf3: unknown operation {operation!r}; valid: ['predict']")


def to_device(device: str) -> dict[str, Any]:
    """CLI passthrough — RF3 spawns subprocesses and auto-unloads after each call."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(device=0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("rf3: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)

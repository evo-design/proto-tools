"""Cherimoya standalone inference: scores, profiles, and gradients for CATv1 and custom checkpoints."""

import hashlib
import json
import os
import sys
from collections import OrderedDict
from typing import Any

from standalone_helpers import get_logger, serialize_output

logger = get_logger(__name__)

# Deterministic cuBLAS needs this before CUDA initializes, so it is set for every call.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

CONTEXT = 2114
OUTPUT = 1000
TRIMMING = (CONTEXT - OUTPUT) // 2
VOCAB = ["A", "C", "G", "T"]
# 64 models (or loss terms) x 5 folds is the most one call can request; all stay loaded (~2.4 MB each in fp32).
MAX_CACHED_CHECKPOINTS = 512


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _one_hot(sequences: list[str], device: str) -> Any:
    """Encode DNA as a ``(B, 4, L)`` float tensor in A,C,G,T order; N and other bases are all zeros."""
    import numpy
    import torch

    table = numpy.full(256, -1, dtype=numpy.int64)
    for index, base in enumerate(VOCAB):
        table[ord(base)] = index
    codes = table[numpy.frombuffer("".join(sequences).encode("ascii"), dtype=numpy.uint8)]
    codes = codes.reshape(len(sequences), -1)
    encoded = numpy.zeros((codes.shape[0], 4, codes.shape[1]), dtype=numpy.float32)
    rows, positions = numpy.nonzero(codes >= 0)
    encoded[rows, codes[rows, positions], positions] = 1.0
    return torch.from_numpy(encoded).to(device)


def _relaxed_dna_probs(logits: Any, *, temperature: float, soft: float, hard: float) -> Any:
    """Convert ``B x L x 4`` logits to relaxed probabilities, with an optional straight-through estimator."""
    import torch.nn.functional as F

    probs = F.softmax(logits / temperature, dim=-1)
    hard_probs = F.one_hot(probs.argmax(dim=-1), num_classes=len(VOCAB)).to(probs.dtype)
    relaxed = (1.0 - soft) * hard_probs + soft * probs
    if hard > 0.0:
        straight_through = hard_probs + probs - probs.detach()
        relaxed = hard * straight_through + (1.0 - hard) * relaxed
    return relaxed


def _forward(model: Any, X: Any, reverse_complement: bool) -> tuple[Any, Any]:
    """Return ``(probabilities (B, 1000), log_counts (B,))`` for one checkpoint."""
    import torch

    profile, log_counts = model(X)
    probs = torch.softmax(profile.float(), dim=-1)[:, 0]
    counts = log_counts.float()[:, 0]
    if reverse_complement:
        # Flipping both axes reverse-complements an A,C,G,T one-hot.
        rc_profile, rc_log_counts = model(X.flip(dims=[1, 2]))
        probs = (probs + torch.softmax(rc_profile.float(), dim=-1)[:, 0].flip(-1)) / 2.0
        counts = (counts + rc_log_counts.float()[:, 0]) / 2.0
    return probs, counts


def _aggregate(models: list[Any], X: Any, reverse_complement: bool) -> tuple[Any, Any, Any]:
    """Average folds in probability and log-count space.

    Returns:
        tuple: ``(probabilities (B, 1000), mean log counts (B,), per-fold log counts (F, B))``.
    """
    import torch

    outputs = [_forward(model, X, reverse_complement) for model in models]
    probs = torch.stack([p for p, _ in outputs]).mean(dim=0)
    fold_counts = torch.stack([c for _, c in outputs])
    return probs, fold_counts.mean(dim=0), fold_counts


class CherimoyaWorker:
    """Holds loaded Cherimoya checkpoints and runs the toolkit's operations.

    Checkpoints are swapped in-process: the selecting config fields are not
    ``reload_on_change``, so this worker is the only reloader. Each cached checkpoint keeps
    the source file and SHA-256 it was loaded from, and every call reports those for the
    wrapper to check against the request.
    """

    def __init__(self) -> None:
        """Initialize an empty checkpoint cache."""
        self._checkpoints: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.device: str | None = None
        self._triton_available: bool | None = None

    @staticmethod
    def _key(spec: dict[str, Any]) -> str:
        if spec.get("path"):
            # A checkpoint retrained into the same path is a different model.
            stat = os.stat(spec["path"])
            return f"{spec['path']}:{stat.st_mtime_ns}:{stat.st_size}"
        return f"{spec['repo_id']}@{spec['revision']}:{spec['filename']}"

    def _load(self, spec: dict[str, Any], device: str, verbose: bool) -> dict[str, Any]:
        from cherimoya import Cherimoya

        if spec.get("path"):
            path = source = str(spec["path"])
        else:
            from huggingface_hub import hf_hub_download

            logger.update_status(f"Fetching {spec['filename']}")
            path = hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"])
            source = f"{spec['repo_id']}@{spec['revision']}:{spec['filename']}"
        if verbose:
            logger.info("Loading Cherimoya checkpoint %s", source)
        model = Cherimoya.load(path, device=device, compile=False).eval()
        model.requires_grad_(False)
        if model.n_control_tracks != 0 or model.n_outputs != 1 or model.trimming != TRIMMING:
            raise ValueError(
                f"cherimoya: {spec['model']!r} has {model.n_control_tracks} control tracks, {model.n_outputs} "
                f"outputs and trimming {model.trimming}; this toolkit runs models with no control tracks, one "
                f"output and trimming {TRIMMING} ({CONTEXT} bp in, {OUTPUT} bp out)"
            )
        return {"model": model, "source": source, "sha256": _sha256(path)}

    def checkpoints(
        self, specs: list[dict[str, Any]], device: str, verbose: bool = False
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Load (or reuse) every requested checkpoint on ``device``.

        Returns:
            tuple: The models in spec order, and ``{"model", "fold", "source", "sha256"}``
                per spec, where source and SHA-256 are those recorded when it was loaded.
        """
        self.to_device(device)
        models, applied = [], []
        for spec in specs:
            key = self._key(spec)
            if key in self._checkpoints:
                self._checkpoints.move_to_end(key)
            else:
                self._checkpoints[key] = self._load(spec, device, verbose)
                while len(self._checkpoints) > MAX_CACHED_CHECKPOINTS:
                    self._checkpoints.popitem(last=False)
            entry = self._checkpoints[key]
            models.append(entry["model"])
            applied.append(
                {"model": spec["model"], "fold": spec["fold"], "source": entry["source"], "sha256": entry["sha256"]}
            )
        return models, applied

    def to_device(self, device: str) -> None:
        """Move every cached checkpoint to ``device``."""
        from standalone_helpers import move_model_to_device

        if self.device is not None and self.device != device:
            for entry in self._checkpoints.values():
                entry["model"] = move_model_to_device(entry["model"], self.device, device)
        self.device = device

    def set_deterministic(self, deterministic: bool) -> None:
        """Choose Cherimoya's Triton kernels or plain PyTorch, which repeats bit-identically."""
        import torch
        from cherimoya import cheri

        if self._triton_available is None:
            self._triton_available = bool(cheri.HAS_TRITON)
        cheri.HAS_TRITON = self._triton_available and not deterministic
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        # `import cherimoya` sets "high" (TF32); the deterministic path uses full fp32 matmuls.
        torch.set_float32_matmul_precision("highest" if deterministic else "high")

    def _groups(self, input_dict: dict[str, Any], loaded: list[Any]) -> list[tuple[str, list[Any]]]:
        """Group loaded checkpoints by requested model, in ``models`` order."""
        return [
            (
                ref,
                [model for model, spec in zip(loaded, input_dict["checkpoints"], strict=True) if spec["model"] == ref],
            )
            for ref in input_dict["models"]
        ]

    def score(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Return log counts (and target counts) per sequence and model."""
        import torch

        self.set_deterministic(False)
        device = input_dict["device"]
        loaded, applied = self.checkpoints(input_dict["checkpoints"], device, bool(input_dict.get("verbose")))
        groups = self._groups(input_dict, loaded)
        sequences, spans = input_dict["sequences"], input_dict["target_spans"]
        batch_size, rc = int(input_dict["batch_size"]), bool(input_dict["reverse_complement_average"])

        rows: list[list[dict[str, Any]]] = [[] for _ in sequences]
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                X = _one_hot(sequences[start : start + batch_size], device)
                for ref, models in groups:
                    probs, counts, fold_counts = _aggregate(models, X, rc)
                    expected = probs * torch.expm1(counts)[:, None]
                    fold_values = fold_counts.T.cpu().tolist()
                    for j, mean in enumerate(counts.cpu().tolist()):
                        target = None
                        if spans is not None:
                            begin, end = spans[start + j]
                            target = float(expected[j, begin:end].sum())
                        rows[start + j].append(
                            {
                                "model": ref,
                                "log_counts": mean,
                                "fold_log_counts": fold_values[j],
                                "target_counts": target,
                            }
                        )
        return {"scores": rows, "applied": applied}

    def predict(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Return a 1,000 bp profile per sequence and model."""
        import torch

        self.set_deterministic(False)
        device = input_dict["device"]
        loaded, applied = self.checkpoints(input_dict["checkpoints"], device, bool(input_dict.get("verbose")))
        groups = self._groups(input_dict, loaded)
        sequences = input_dict["sequences"]
        batch_size, rc = int(input_dict["batch_size"]), bool(input_dict["reverse_complement_average"])
        as_counts = input_dict["values"] == "expected_counts"

        rows: list[list[dict[str, Any]]] = [[] for _ in sequences]
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                X = _one_hot(sequences[start : start + batch_size], device)
                for ref, models in groups:
                    probs, counts, fold_counts = _aggregate(models, X, rc)
                    values = probs * torch.expm1(counts)[:, None] if as_counts else probs
                    fold_values = fold_counts.T.cpu().tolist()
                    for j, (mean, profile) in enumerate(zip(counts.cpu().tolist(), values.cpu().tolist(), strict=True)):
                        rows[start + j].append(
                            {"model": ref, "log_counts": mean, "fold_log_counts": fold_values[j], "values": profile}
                        )
        return {"profiles": rows, "applied": applied}

    def gradient(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Return the summed loss terms and their gradient with respect to the logits."""
        import torch

        self.set_deterministic(bool(input_dict["deterministic"]))
        device = input_dict["device"]
        loaded, applied = self.checkpoints(input_dict["checkpoints"], device, bool(input_dict.get("verbose")))
        compute_gradient = bool(input_dict["compute_gradient"])
        rc = bool(input_dict["reverse_complement_average"])
        span = input_dict["target_span"]

        with torch.set_grad_enabled(compute_gradient):
            logits = torch.tensor(
                input_dict["logits"], device=device, dtype=torch.float32, requires_grad=compute_gradient
            )
            relaxed = _relaxed_dna_probs(
                logits, temperature=float(input_dict["temperature"]), soft=input_dict["soft"], hard=input_dict["hard"]
            )
            batch = int(relaxed.shape[0])
            parts = [relaxed.transpose(1, 2)]
            if input_dict["context_left"]:
                parts.insert(0, _one_hot([input_dict["context_left"]], device).expand(batch, -1, -1))
            if input_dict["context_right"]:
                parts.append(_one_hot([input_dict["context_right"]], device).expand(batch, -1, -1))
            X = torch.cat(parts, dim=2)

            sample_losses = torch.zeros(batch, device=device)
            term_rows: list[list[dict[str, Any]]] = [[] for _ in range(batch)]
            for term in input_dict["loss_terms"]:
                probs, counts, _ = _aggregate([loaded[i] for i in term["checkpoints"]], X, rc)
                if term["quantity"] == "log_counts":
                    value = counts
                else:
                    value = torch.log1p(probs[:, span[0] : span[1]].sum(dim=-1) * torch.expm1(counts))
                sigmoid = torch.sigmoid((value - term["sigmoid_center"]) / term["sigmoid_scale"])
                score = 1.0 - sigmoid if term["direction"] == "max" else sigmoid
                weighted = score * float(term["weight"])
                sample_losses = sample_losses + weighted
                for j, (v, s, w) in enumerate(
                    zip(
                        value.detach().cpu().tolist(),
                        score.detach().cpu().tolist(),
                        weighted.detach().cpu().tolist(),
                        strict=True,
                    )
                ):
                    term_rows[j].append(
                        {
                            "model": term["model"],
                            "folds": term["folds"],
                            "quantity": term["quantity"],
                            "direction": term["direction"],
                            "value": v,
                            "sigmoid_center": term["sigmoid_center"],
                            "sigmoid_scale": term["sigmoid_scale"],
                            "score": s,
                            "weight": term["weight"],
                            "weighted_score": w,
                        }
                    )

            total = sample_losses.sum()
            gradient = None
            if compute_gradient:
                total.backward()
                if logits.grad is None:
                    raise RuntimeError("cherimoya: missing logits gradient")
                gradient = logits.grad.detach().cpu().tolist()

        losses = [float(value) for value in sample_losses.detach().cpu().tolist()]
        return {
            "gradient": gradient,
            "loss": float(sum(losses)),
            "metrics": {
                "losses": losses,
                "loss_terms": term_rows,
                "batch_size": batch,
                "temperature": input_dict["temperature"],
                "soft": input_dict["soft"],
                "hard": input_dict["hard"],
                "deterministic": bool(input_dict["deterministic"]),
                "objective": "cherimoya_accessibility",
            },
            "vocab": VOCAB,
            "applied": applied,
        }


_worker = CherimoyaWorker()


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Route a request to the worker operation it names."""
    operation = input_dict["operation"]
    if operation == "score":
        return _worker.score(input_dict)
    if operation == "predict":
        return _worker.predict(input_dict)
    if operation == "gradient":
        return _worker.gradient(input_dict)
    raise ValueError(f"cherimoya: unknown operation {operation!r}; valid: ['score', 'predict', 'gradient']")


def to_device(device: str) -> dict[str, Any]:
    """Move every cached checkpoint to the specified device (called by DeviceManager)."""
    if not _worker._checkpoints:
        return {"success": True, "device": device, "note": "model not loaded yet"}
    _worker.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(_worker.device or 0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("cherimoya: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)

"""Malinois standalone inference implementation for venv execution."""

import contextlib
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import malinois_scoring
import torch
import torch.nn.functional as F
from standalone_helpers import get_logger, move_model_to_device, resolve_weights_dir, serialize_output, set_torch_seed

logger = get_logger(__name__)

CELL_TYPE_INDEX = {"K562": 0, "HepG2": 1, "SKNSH": 2}
DNA_VOCAB = ["A", "C", "G", "T"]
DEFAULT_ARTIFACT_FILENAME = "MODELS-malinois_artifacts__20211113_021200__287348.tar.gz"
DEFAULT_ARTIFACT_MD5 = "375142a714e7df73c463b46113a65210"
LOCK_TIMEOUT_SECONDS = 600


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Small cross-process lock using O_EXCL so concurrent workers do not race downloads."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as err:  # noqa: PERF203 - lock acquisition is intentionally retry-based.
            if time.monotonic() - started > LOCK_TIMEOUT_SECONDS:
                raise TimeoutError(f"malinois: timed out waiting for lock {lock_path}") from err
            time.sleep(1)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weights_dir() -> Path:
    resolved = resolve_weights_dir("malinois")
    if resolved:
        return Path(resolved)
    fallback = Path(tempfile.gettempdir()) / "proto_malinois_weights"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    return name or DEFAULT_ARTIFACT_FILENAME


def _download_artifact(url: str, expected_md5: str | None, cache_dir: Path) -> Path:
    if not url:
        raise FileNotFoundError("malinois: artifact_path is unavailable and artifact_url is empty")

    dest = cache_dir / _filename_from_url(url)
    if dest.is_file() and (not expected_md5 or _md5(dest) == expected_md5):
        return dest

    with _file_lock(cache_dir / ".malinois_artifact_download.lock"):
        if dest.is_file() and (not expected_md5 or _md5(dest) == expected_md5):
            return dest
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        logger.info("Downloading Malinois artifact from %s to %s", url, dest)
        try:
            request = urllib.request.Request(url)
            if urllib.parse.urlparse(url).netloc.endswith("huggingface.co") and os.environ.get("HF_TOKEN"):
                request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
            with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if expected_md5:
                observed = _md5(tmp)
                if observed != expected_md5:
                    raise RuntimeError(
                        f"malinois: artifact checksum mismatch for {url}; expected {expected_md5}, got {observed}"
                    )
            os.replace(tmp, dest)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
            raise
    return dest


def _extract_artifact(artifact_path: Path, cache_dir: Path) -> Path:
    extracted_dir = cache_dir / "malinois_artifacts"
    checkpoint = extracted_dir / "artifacts" / "torch_checkpoint.pt"
    if checkpoint.is_file():
        return extracted_dir

    with _file_lock(cache_dir / ".malinois_artifact_extract.lock"):
        if checkpoint.is_file():
            return extracted_dir
        tmp_dir = cache_dir / f".malinois_extract_{os.getpid()}"
        if tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting Malinois artifact %s to %s", artifact_path, extracted_dir)
        try:
            import shutil

            shutil.unpack_archive(str(artifact_path), str(tmp_dir))
            tmp_checkpoint = tmp_dir / "artifacts" / "torch_checkpoint.pt"
            if not tmp_checkpoint.is_file():
                raise FileNotFoundError(f"malinois: extracted artifact missing {tmp_checkpoint}")
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)
            os.replace(tmp_dir, extracted_dir)
        except Exception:
            import shutil

            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(tmp_dir)
            raise
    return extracted_dir


def _resolve_artifact_paths(
    *,
    artifact_path: str,
    artifact_url: str,
    artifact_md5: str,
    malinois_dir: str,
) -> tuple[str, str]:
    cache_dir = _weights_dir()
    artifact_candidate = Path(artifact_path).expanduser() if artifact_path else None
    if artifact_candidate is not None and artifact_candidate.is_file():
        resolved_artifact = artifact_candidate
    else:
        expected_md5 = artifact_md5 or DEFAULT_ARTIFACT_MD5
        resolved_artifact = _download_artifact(artifact_url, expected_md5, cache_dir)

    metadata_candidate = Path(malinois_dir).expanduser() if malinois_dir else None
    if metadata_candidate is not None and (metadata_candidate / "artifacts" / "torch_checkpoint.pt").is_file():
        resolved_dir = metadata_candidate
    else:
        resolved_dir = _extract_artifact(resolved_artifact, cache_dir)

    return str(resolved_artifact), str(resolved_dir)


class MalinoisModel:
    """Malinois model wrapper for regulatory DNA scoring."""

    def __init__(self) -> None:
        """Initialize an unloaded Malinois model wrapper."""
        self.model: Any = None
        self.flank_builder: Any = None
        self.device: str | None = None
        self.model_key: tuple[str, str, int] | None = None

    def load(
        self,
        *,
        artifact_path: str,
        artifact_url: str,
        artifact_md5: str,
        malinois_dir: str,
        seq_length: int,
        device: str,
        verbose: bool = False,
    ) -> None:
        """Load the Malinois model and MPRA flank builder."""
        artifact_path, malinois_dir = _resolve_artifact_paths(
            artifact_path=artifact_path,
            artifact_url=artifact_url,
            artifact_md5=artifact_md5,
            malinois_dir=malinois_dir,
        )
        model_key = (artifact_path, malinois_dir, seq_length)
        if self.model is not None and self.model_key == model_key:
            self.to_device(device)
            return

        artifact = Path(artifact_path)
        if not artifact.is_file():
            raise FileNotFoundError(f"malinois: artifact_path does not exist: {artifact_path}")

        checkpoint = Path(malinois_dir) / "artifacts" / "torch_checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"malinois: metadata checkpoint does not exist: {checkpoint}")

        if verbose:
            logger.info("Loading Malinois model from %s", artifact_path)

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="malinois_artifact_") as temp_dir:
            os.chdir(temp_dir)
            try:
                self.model, self.flank_builder = malinois_scoring.load_malinois(
                    artifact_path=artifact_path,
                    malinois_dir=malinois_dir,
                    seq_len=seq_length,
                )
            finally:
                os.chdir(cwd)

        self.model_key = model_key
        self.device = None
        self.model.requires_grad_(False)
        self.to_device(device)

    def to_device(self, device: str) -> None:
        """Move the loaded model and flank builder to a device."""
        if self.model is None or self.flank_builder is None:
            return
        if self.device == device:
            return
        previous_device = self.device or "cpu"
        self.model = move_model_to_device(self.model, previous_device, device)
        self.flank_builder = move_model_to_device(self.flank_builder, previous_device, device)
        self.device = device

    def score_sequences(
        self,
        *,
        sequences: list[str],
        cell_types: list[str],
        batch_size: int,
        device: str,
    ) -> list[dict[str, float]]:
        """Score DNA sequences and return selected Malinois outputs."""
        if self.model is None or self.flank_builder is None:
            raise ValueError("malinois: model is not loaded")
        self.to_device(device)

        seq_tensor = torch.stack([malinois_scoring.dna2tensor(sequence) for sequence in sequences], dim=0)
        dataset = torch.utils.data.TensorDataset(seq_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        chunks = []
        with torch.inference_mode():
            for (batch_tensor,) in loader:
                batch_on_device = batch_tensor.to(device)
                prepped = self.flank_builder(batch_on_device)
                preds = self.model(prepped) + self.model(prepped.flip(dims=[1, 2]))
                chunks.append(preds.div(2.0).detach().cpu())

        predictions = torch.cat(chunks, dim=0)
        indices = [CELL_TYPE_INDEX[cell_type] for cell_type in cell_types]
        return [
            {cell_type: float(row[index].item()) for cell_type, index in zip(cell_types, indices, strict=True)}
            for row in predictions
        ]

    def compute_gradient(
        self,
        *,
        logits_list: list[list[list[float]]],
        temperature: float,
        loss_terms: list[dict[str, Any]],
        soft: float,
        hard: float,
        compute_gradient: bool,
        device: str,
    ) -> dict[str, Any]:
        """Compute differentiable Malinois loss and optional gradient."""
        if self.model is None or self.flank_builder is None:
            raise ValueError("malinois: model is not loaded")
        self.to_device(device)

        with torch.set_grad_enabled(compute_gradient):
            logits = torch.tensor(logits_list, device=device, dtype=torch.float32, requires_grad=compute_gradient)
            if logits.ndim != 3:
                raise ValueError(f"malinois: logits must have rank 3 (B, L, 4), got rank {logits.ndim}")

            relaxed = _relaxed_dna_probs(logits, temperature=temperature, soft=soft, hard=hard)
            insert = relaxed.transpose(1, 2)
            prepped = self.flank_builder(insert)
            predictions = self.model(prepped)
            predictions = (predictions + self.model(prepped.flip(dims=[1, 2]))) / 2.0

            batch_size = int(predictions.shape[0])
            sample_losses = torch.zeros(batch_size, device=device, dtype=predictions.dtype)
            term_metrics_by_sample: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
            raw_scores_by_sample: list[dict[str, float]] = [{} for _ in range(batch_size)]
            for term in loss_terms:
                cell_type = term["cell_type"]
                direction = term["direction"]
                weight = float(term.get("weight", 1.0))
                center = float(term.get("sigmoid_center", 4.0))
                scale = float(term.get("sigmoid_scale", 1.0))
                raw_score = predictions[:, CELL_TYPE_INDEX[cell_type]]
                scaled_score = (raw_score - center) / scale
                sigmoid_value = torch.sigmoid(scaled_score)
                if direction == "max":
                    score = 1.0 - sigmoid_value
                elif direction == "min":
                    score = sigmoid_value
                else:
                    raise ValueError(f"malinois: invalid direction {direction!r}; expected 'max' or 'min'")
                weighted_score = score * weight
                sample_losses = sample_losses + weighted_score

                raw_score_values = raw_score.detach().cpu().tolist()
                scaled_score_values = scaled_score.detach().cpu().tolist()
                sigmoid_values = sigmoid_value.detach().cpu().tolist()
                score_values = score.detach().cpu().tolist()
                weighted_score_values = weighted_score.detach().cpu().tolist()
                for sample_idx in range(batch_size):
                    raw_scores_by_sample[sample_idx][cell_type] = float(raw_score_values[sample_idx])
                    term_metrics_by_sample[sample_idx].append(
                        {
                            "cell_type": cell_type,
                            "direction": direction,
                            "weight": weight,
                            "raw_score": float(raw_score_values[sample_idx]),
                            "scaled_score": float(scaled_score_values[sample_idx]),
                            "sigmoid_value": float(sigmoid_values[sample_idx]),
                            "score": float(score_values[sample_idx]),
                            "weighted_score": float(weighted_score_values[sample_idx]),
                            "sigmoid_center": center,
                            "sigmoid_scale": scale,
                        }
                    )

            total_loss = sample_losses.sum()
            sample_loss_values = [float(value) for value in sample_losses.detach().cpu().tolist()]

            gradient_value: list[list[list[float]]] | None = None
            if compute_gradient:
                backward = cast(Callable[[], None], total_loss.backward)
                backward()
                if logits.grad is None:
                    raise RuntimeError("malinois: missing logits gradient")
                gradient_value = logits.grad.detach().cpu().tolist()

        return {
            "gradient": gradient_value,
            "loss": float(sum(sample_loss_values)),
            "metrics": {
                "raw_scores": raw_scores_by_sample,
                "loss_terms": term_metrics_by_sample,
                "losses": sample_loss_values,
                "batch_size": batch_size,
                "temperature": temperature,
                "soft": soft,
                "hard": hard,
                "objective": "malinois_activity",
            },
            "vocab": DNA_VOCAB,
        }


_model = MalinoisModel()


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    set_torch_seed(input_dict.get("seed"))
    operation = input_dict["operation"]

    if operation == "compute_gradient":
        loss_terms = input_dict["loss_terms"]
        unsupported = sorted({term["cell_type"] for term in loss_terms} - set(CELL_TYPE_INDEX))
        if unsupported:
            raise ValueError(f"malinois: unsupported cell_types: {unsupported}")
        _model.load(
            artifact_path=input_dict["artifact_path"],
            artifact_url=input_dict.get("artifact_url", ""),
            artifact_md5=input_dict.get("artifact_md5", DEFAULT_ARTIFACT_MD5),
            malinois_dir=input_dict["malinois_dir"],
            seq_length=int(input_dict["seq_length"]),
            device=input_dict["device"],
            verbose=bool(input_dict.get("verbose", False)),
        )
        return _model.compute_gradient(
            logits_list=input_dict["logits"],
            temperature=float(input_dict["temperature"]),
            loss_terms=loss_terms,
            soft=float(input_dict.get("soft", 1.0)),
            hard=float(input_dict.get("hard", 0.0)),
            compute_gradient=bool(input_dict.get("compute_gradient", True)),
            device=input_dict["device"],
        )

    if operation != "score":
        raise ValueError(f"malinois: unknown operation {operation!r}; valid: ['score', 'compute_gradient']")

    cell_types = input_dict["cell_types"]
    unsupported = sorted(set(cell_types) - set(CELL_TYPE_INDEX))
    if unsupported:
        raise ValueError(f"malinois: unsupported cell_types: {unsupported}")

    _model.load(
        artifact_path=input_dict["artifact_path"],
        artifact_url=input_dict.get("artifact_url", ""),
        artifact_md5=input_dict.get("artifact_md5", DEFAULT_ARTIFACT_MD5),
        malinois_dir=input_dict["malinois_dir"],
        seq_length=int(input_dict["seq_length"]),
        device=input_dict["device"],
        verbose=bool(input_dict.get("verbose", False)),
    )
    return {
        "scores": _model.score_sequences(
            sequences=input_dict["sequences"],
            cell_types=cell_types,
            batch_size=max(1, int(input_dict.get("batch_size", 1))),
            device=input_dict["device"],
        )
    }


def _relaxed_dna_probs(
    logits: torch.Tensor,
    *,
    temperature: float,
    soft: float,
    hard: float,
) -> torch.Tensor:
    """Convert ``... x L x 4`` DNA logits to relaxed probabilities with optional STE."""
    probs = F.softmax(logits / temperature, dim=-1)
    hard_probs = F.one_hot(probs.argmax(dim=-1), num_classes=len(DNA_VOCAB)).to(probs.dtype)
    relaxed = (1.0 - soft) * hard_probs + soft * probs
    if hard > 0.0:
        straight_through = hard_probs + probs - probs.detach()
        relaxed = hard * straight_through + (1.0 - hard) * relaxed
    return relaxed


def to_device(device: str) -> dict[str, Any]:
    """Move model to the specified device."""
    _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(_model.device or 0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("malinois: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)

"""PARADE standalone inference implementation for venv execution."""

import contextlib
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import parade_scoring
from standalone_helpers import (
    get_logger,
    move_model_to_device,
    resolve_weights_dir,
    serialize_output,
    set_torch_seed,
)

logger = get_logger(__name__)

LOCK_TIMEOUT_SECONDS = 600


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Cross-process lock using O_EXCL so concurrent workers do not race downloads."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as err:  # noqa: PERF203 - lock acquisition is intentionally retry-based.
            if time.monotonic() - started > LOCK_TIMEOUT_SECONDS:
                raise TimeoutError(f"parade: timed out waiting for lock {lock_path}") from err
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
    resolved = resolve_weights_dir("parade")
    if resolved:
        return Path(resolved)
    fallback = Path(tempfile.gettempdir()) / "proto_parade_weights"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _download_checkpoint(url: str, expected_md5: str, filename: str, cache_dir: Path) -> Path:
    """Download a PARADE checkpoint into the cache, verifying its MD5 checksum."""
    if not url:
        raise FileNotFoundError("parade: checkpoint_path is unavailable and checkpoint_url is empty")

    dest = cache_dir / filename
    if dest.is_file() and (not expected_md5 or _md5(dest) == expected_md5):
        return dest

    with _file_lock(cache_dir / f".{filename}.download.lock"):
        if dest.is_file() and (not expected_md5 or _md5(dest) == expected_md5):
            return dest
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        logger.info("Downloading PARADE checkpoint from %s to %s", url, dest)
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
                        f"parade: checkpoint checksum mismatch for {url}; expected {expected_md5}, got {observed}"
                    )
            os.replace(tmp, dest)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
            raise
    return dest


def _resolve_checkpoint(input_dict: dict[str, Any]) -> str:
    """Resolve a local checkpoint path, downloading the pinned artifact if needed."""
    local = input_dict.get("checkpoint_path") or ""
    if local:
        candidate = Path(local).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"parade: checkpoint_path does not exist: {local}")

    checkpoint = _download_checkpoint(
        url=input_dict["checkpoint_url"],
        expected_md5=input_dict.get("checkpoint_md5", ""),
        filename=input_dict["checkpoint_filename"],
        cache_dir=_weights_dir(),
    )
    return str(checkpoint)


class ParadeModel:
    """Caches a loaded PARADE LegNet module and delegates inference to ``parade_scoring``."""

    def __init__(self) -> None:
        """Initialize an unloaded PARADE model wrapper."""
        self.model: Any = None
        self.device: str | None = None
        self.checkpoint_key: str | None = None

    def load(self, *, checkpoint_path: str, device: str, verbose: bool = False) -> None:
        """Load a PARADE checkpoint (once per checkpoint) and move it to the device."""
        if self.model is not None and self.checkpoint_key == checkpoint_path:
            self.to_device(device)
            return
        if verbose:
            logger.info("Loading PARADE checkpoint from %s", checkpoint_path)
        self.model = parade_scoring.load_legnet(checkpoint_path)
        self.checkpoint_key = checkpoint_path
        self.device = "cpu"
        self.to_device(device)

    def to_device(self, device: str) -> None:
        """Move the cached LegNet module to a device."""
        if self.model is None or self.device == device:
            return
        previous_device = self.device or "cpu"
        self.model = move_model_to_device(self.model, previous_device, device)
        self.device = device

    def score_activity(
        self, *, sequences: list[str], construct_type: str, cell_types: list[str], batch_size: int, device: str
    ) -> list[dict[str, float]]:
        """Predict per-cell-type activity for each sequence."""
        if self.model is None:
            raise ValueError("parade: model is not loaded")
        self.to_device(device)
        return parade_scoring.predict_activity(  # type: ignore[no-any-return]
            self.model,
            sequences=sequences,
            construct_type=construct_type,
            cell_types=cell_types,
            batch_size=batch_size,
            device=device,
        )

    def score_stability(self, *, sequences: list[str], batch_size: int, device: str) -> list[float]:
        """Predict the RNA/gDNA log-ratio for each 3' UTR sequence."""
        if self.model is None:
            raise ValueError("parade: model is not loaded")
        self.to_device(device)
        return parade_scoring.predict_stability(  # type: ignore[no-any-return]
            self.model, sequences=sequences, batch_size=batch_size, device=device
        )

    def compute_gradient(self, **kwargs: Any) -> dict[str, Any]:
        """Compute a differentiable PARADE activity loss and optional gradient."""
        if self.model is None:
            raise ValueError("parade: model is not loaded")
        self.to_device(kwargs["device"])
        return parade_scoring.compute_gradient(self.model, **kwargs)  # type: ignore[no-any-return]


_model = ParadeModel()


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    set_torch_seed(input_dict.get("seed"))
    operation = input_dict["operation"]
    if operation not in {"activity", "stability", "gradient"}:
        raise ValueError(f"parade: unknown operation {operation!r}; valid: ['activity', 'stability', 'gradient']")

    checkpoint_path = _resolve_checkpoint(input_dict)
    _model.load(
        checkpoint_path=checkpoint_path,
        device=input_dict["device"],
        verbose=bool(input_dict.get("verbose", False)),
    )

    if operation == "gradient":
        return _model.compute_gradient(
            logits_list=input_dict["logits"],
            temperature=float(input_dict["temperature"]),
            construct_type=input_dict["construct_type"],
            loss_terms=input_dict["loss_terms"],
            soft=float(input_dict.get("soft", 1.0)),
            hard=float(input_dict.get("hard", 0.0)),
            compute_gradient=bool(input_dict.get("compute_gradient", True)),
            device=input_dict["device"],
        )

    batch_size = max(1, int(input_dict.get("batch_size", 1)))
    if operation == "activity":
        return {
            "scores": _model.score_activity(
                sequences=input_dict["sequences"],
                construct_type=input_dict["construct_type"],
                cell_types=input_dict["cell_types"],
                batch_size=batch_size,
                device=input_dict["device"],
            )
        }
    return {
        "log_ratios": _model.score_stability(
            sequences=input_dict["sequences"],
            batch_size=batch_size,
            device=input_dict["device"],
        )
    }


def to_device(device: str) -> dict[str, Any]:
    """Move the model to the specified device."""
    _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(_model.device or 0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("parade: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)

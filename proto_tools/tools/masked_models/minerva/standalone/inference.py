"""Minerva checkpoint adapter for shared mixed-modality masked-model inference."""

from typing import Any, ClassVar, cast

from standalone_helpers import get_logger, get_pytorch_memory_stats
from standalone_helpers.mixed_mlm import MixedMLMAdapter, MixedMLMRuntime, run_json_main

logger = get_logger(__name__)


class MinervaAdapter(MixedMLMAdapter):  # type: ignore[misc]  # Helper is injected only in the isolated environment.
    """Load pinned public Minerva masked-language-model checkpoints."""

    toolkit = "minerva"
    default_checkpoint = "gbrixi/minerva-mlm"
    checkpoints: ClassVar[dict[str, str]] = {
        "gbrixi/minerva-mlm": "0403fd777b8803ce93b199923d4ba3905e7d533c",
        "gbrixi/minerva-mlm-8k": "df01967534e5414af838665f715fa2033f4c9012",
    }

    def _load_checkpoint(self, checkpoint: str) -> tuple[Any, Any]:
        import torch
        from minerva import MinervaForMaskedLM
        from transformers import AutoTokenizer

        revision = self.checkpoints[checkpoint]
        model = MinervaForMaskedLM.from_pretrained(checkpoint, revision=revision, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        return model, tokenizer


_MODEL = MinervaAdapter()
_RUNTIME = MixedMLMRuntime(_MODEL)


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a plain-data request to the persistent model adapter."""
    return cast(dict[str, Any], _RUNTIME.dispatch(input_dict))


def to_device(device: str) -> dict[str, Any]:
    """Move the loaded model for DeviceManager memory management."""
    _MODEL.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report device allocation through the shared PyTorch memory helper."""
    return cast(dict[str, Any], get_pytorch_memory_stats(_MODEL.device or "cpu"))


if __name__ == "__main__":
    run_json_main(dispatch)

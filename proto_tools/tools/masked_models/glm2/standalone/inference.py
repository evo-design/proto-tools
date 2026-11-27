"""GLM2 checkpoint adapter for shared mixed-modality masked-model inference."""

from typing import Any, ClassVar, cast

from standalone_helpers import get_logger, get_pytorch_memory_stats
from standalone_helpers.mixed_mlm import MixedMLMAdapter, MixedMLMRuntime, run_json_main

logger = get_logger(__name__)


class GLM2Adapter(MixedMLMAdapter):  # type: ignore[misc]  # Helper is injected only in the isolated environment.
    """Load pinned public GLM2 masked-language-model checkpoints."""

    toolkit = "glm2"
    default_checkpoint = "tattabio/gLM2_650M"
    checkpoints: ClassVar[dict[str, str]] = {
        "tattabio/gLM2_150M": "93c88529115476b27bdd85da14311779114a8a64",
        "tattabio/gLM2_650M": "08754cba59a1f97d517f873fad6c672d2b1abdc7",
    }

    def _load_checkpoint(self, checkpoint: str) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        revision = self.checkpoints[checkpoint]
        model = AutoModelForMaskedLM.from_pretrained(
            checkpoint, revision=revision, trust_remote_code=True, torch_dtype=torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
        return model, tokenizer


_MODEL = GLM2Adapter()
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

"""Checkpoint configuration shared by the GLM2 toolkit."""

from typing import Any, ClassVar, Literal, cast

from proto_tools.tools.masked_models.mixed_configs import MixedModelConfig
from proto_tools.utils import ConfigField


class GLM2Config(MixedModelConfig):
    """Shared GLM2 checkpoint selection and token limits.

    Attributes:
        model_checkpoint (Literal['tattabio/gLM2_150M', 'tattabio/gLM2_650M']): Public model checkpoint.
        batch_size (int): Sequences or masked PLL variants per forward pass.
    """

    context_limits: ClassVar[dict[str, int]] = {"tattabio/gLM2_150M": 4096, "tattabio/gLM2_650M": 4096}
    layer_counts: ClassVar[dict[str, int]] = {"tattabio/gLM2_150M": 30, "tattabio/gLM2_650M": 33}
    model_checkpoint: Literal["tattabio/gLM2_150M", "tattabio/gLM2_650M"] = ConfigField(
        default="tattabio/gLM2_650M",
        title="Model Checkpoint",
        description="gLM2 checkpoint; downloaded automatically from Hugging Face",
        reload_on_change=True,
    )

    @classmethod
    def minimal(cls, **kwargs: Any) -> "GLM2Config":
        """Smallest checkpoint, so smoke and device tests load the model in seconds."""
        kwargs.setdefault("model_checkpoint", "tattabio/gLM2_150M")
        return cast("GLM2Config", super().minimal(**kwargs))

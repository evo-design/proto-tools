"""Checkpoint configuration shared by the Minerva toolkit."""

from typing import ClassVar, Literal

from proto_tools.tools.masked_models.mixed_configs import MixedModelConfig
from proto_tools.utils import ConfigField


class MinervaConfig(MixedModelConfig):
    """Shared Minerva checkpoint selection and token limits.

    Attributes:
        model_checkpoint (Literal['gbrixi/minerva-mlm', 'gbrixi/minerva-mlm-8k']): Public model checkpoint.
        batch_size (int): Sequences or masked PLL variants per forward pass.
    """

    context_limits: ClassVar[dict[str, int]] = {"gbrixi/minerva-mlm": 4096, "gbrixi/minerva-mlm-8k": 8192}
    layer_counts: ClassVar[dict[str, int]] = {"gbrixi/minerva-mlm": 33, "gbrixi/minerva-mlm-8k": 33}
    model_checkpoint: Literal["gbrixi/minerva-mlm", "gbrixi/minerva-mlm-8k"] = ConfigField(
        default="gbrixi/minerva-mlm",
        title="Model Checkpoint",
        description="Minerva checkpoint; downloaded automatically from Hugging Face",
        reload_on_change=True,
    )

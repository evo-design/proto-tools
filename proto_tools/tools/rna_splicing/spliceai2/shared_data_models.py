"""Shared constants and config base for SpliceAI2 tools."""

from typing import Literal

from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.device import RemoteDevice

SPLICEAI2_HF_REPO = "illumina-ai/SpliceAI2"
SPLICEAI2_HF_URL = f"https://huggingface.co/{SPLICEAI2_HF_REPO}"

# Context the model crops from each side of its 196,608 bp input to predict the central 65,536 bp.
SPLICEAI2_FLANK = 65_536

# Genome assemblies of the ten training species, each one-hot encoded as a species channel.
SpliceAI2Assembly = Literal[
    "GRCh38",
    "NHGRI_mPanTro3-v2.0",
    "Panubis1.0",
    "T2T-MFA8v1.1",
    "Mmul_10",
    "ARS-UCD2.0",
    "ARS-UI_Ramb_v3.0",
    "Sscrofa11.1",
    "GRCm38",
    "GRCr8",
]


class SpliceAI2Config(BaseConfig):
    """Shared configuration for SpliceAI2 tools.

    Attributes:
        assembly (SpliceAI2Assembly): Species assembly whose channel conditions
            the model. ``GRCh38`` (human) for human sequence, including GRCh37
            coordinates.
        device (str): Device to run the model on. Override of ``BaseConfig.device``
            because SpliceAI2 requires a GPU (default ``cuda``).
    """

    assembly: SpliceAI2Assembly = ConfigField(
        title="Assembly",
        default="GRCh38",
        description="Species assembly conditioning the model (GRCh38 for any human sequence)",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g. 'cuda', 'cuda:0')",
        include_in_key=False,
    )

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """SpliceAI2 runs only where the caller owns the deployment: locally or on their own Modal account."""
        if device == "proto":
            return (
                "SpliceAI2 is not hosted on device='proto'. Run locally (device='cuda'), "
                "or deploy it on your own Modal account and use device='modal'."
            )
        return super().remote_unsupported_reason(device)

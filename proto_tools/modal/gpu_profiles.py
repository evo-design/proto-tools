"""Shared Modal GPU fallback profiles."""

from typing import Final

# The default choice for a new service.
GPU_DEFAULT: Final[list[str]] = ["H100:1", "H200:1", "A100-80GB:1"]

# Only where Hopper FP8 is required, such as Evo2 1B, 20B, and 40B.
GPU_HOPPER: Final[list[str]] = ["H100:1", "H200:1"]

# Only where an H100 would be plainly excessive.
GPU_BASIC: Final[list[str]] = ["T4:1", "L4:1", "A10:1"]

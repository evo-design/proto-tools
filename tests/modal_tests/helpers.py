"""Shared discovery helpers for the Modal deployment tests.

These run offline: no Modal API calls, no GPU, no deploys.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
MODAL_ROOT = REPO / "proto_tools" / "modal"


def service_modules() -> list[Path]:
    """Return every Modal service module."""
    return sorted(p for p in MODAL_ROOT.rglob("*_service.py") if "__pycache__" not in p.parts)

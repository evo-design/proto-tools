"""Build filenames using the shared export naming convention.

Composes ``{project}__{run_name}__{YYYY-MM-DD_HHMMSS}{discriminator}.{ext}``. Empty fields drop out
with their separator. Discriminators (e.g. ``_stage-0``) must start with ``_``.
"""

import re
import unicodedata
from datetime import datetime

_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x08\x0E-\x1F\x7F]')
_WS_RUN = re.compile(r"\s+")
_WIN_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
)


def sanitize_field(raw: str | None) -> str:
    """Normalize text to NFC and replace unsafe filename characters with ``_``.

    Collapse repeated whitespace and trim dots and whitespace from both ends.
    """
    if not raw:
        return ""
    s = _BAD_CHARS.sub("_", unicodedata.normalize("NFC", str(raw)))
    return _WS_RUN.sub(" ", s).strip(" .\t")


def build_export_name(
    *,
    project: str | None = None,
    run_name: str | None = None,
    timestamp: str | datetime | None = None,
    discriminator: str | None = None,
    ext: str | None = None,
) -> str:
    """Build an export filename, omitting empty fields and their ``__`` separators."""
    if isinstance(timestamp, datetime):
        ts = timestamp.strftime("%Y-%m-%d_%H%M%S")
    elif timestamp is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    else:
        ts = timestamp
    # Clean each field before joining the nonempty parts.
    parts = [sanitize_field(project), sanitize_field(run_name), sanitize_field(ts)]
    base = "__".join(p for p in parts if p) or "export"
    if discriminator:
        base += sanitize_field(discriminator)
    # Prefix names that Windows reserves for devices.
    if base.split(".", 1)[0].upper() in _WIN_RESERVED:
        base = "_" + base
    safe_ext = sanitize_field(ext) if ext else ""
    return f"{base}.{safe_ext}" if safe_ext else base

"""MEME Suite FIMO standalone runner for ToolInstance venv execution.

Runs FIMO motif scanning via ``pymemesuite`` (no host MEME install needed).
Communicates via JSON input/output files (ToolInstance pattern).

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import json
import sys
from typing import Any

from pymemesuite.common import MotifFile, Sequence
from pymemesuite.fimo import FIMO
from standalone_helpers import get_logger

logger = get_logger(__name__)


# ============================================================================
# Helpers
# ============================================================================
def _decode(value: Any) -> str:
    """Decode pymemesuite bytes fields to str ('' for empty/None)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_motifs(motifs_path: str) -> tuple[list[Any], Any]:
    """Read every motif and the background from a MEME-format file.

    Motif objects and the background outlive the file handle, matching
    pymemesuite's documented usage.
    """
    motifs = []
    with open(motifs_path, "rb") as handle:
        motif_file = MotifFile(handle)
        while (motif := motif_file.read()) is not None:
            motifs.append(motif)
        background = motif_file.background
    return motifs, background


def _is_complementable(motif: Any) -> bool:
    """Whether a motif's alphabet supports reverse-complement scanning.

    The FIMO CLI only scans the reverse strand for complementable (nucleotide)
    alphabets; protein and other non-complementable alphabets are forward-only.
    Detected from the motif alphabet: nucleotide alphabets have 4 core symbols
    including A/C/G and T or U, whereas protein has 20.
    """
    symbols = set(motif.alphabet.symbols.upper())
    return motif.alphabet.size == 4 and {"A", "C", "G"} <= symbols and ("T" in symbols or "U" in symbols)


# ============================================================================
# Implementation
# ============================================================================
def run_fimo_scan(input_data: dict[str, Any]) -> dict[str, Any]:
    """Scan sequences against MEME-format motifs with FIMO."""
    sequences = [Sequence(seq, name=f"seq_{i}".encode()) for i, seq in enumerate(input_data["sequences"])]
    motifs, background = _read_motifs(input_data["motifs_path"])

    # FIMO only reverse-complements nucleotide alphabets; mirror the CLI by scanning
    # the given strand only for protein / non-complementable motifs, even if both_strands
    # was requested (otherwise pymemesuite emits spurious reverse-strand hits on protein).
    both_strands = input_data["both_strands"] and bool(motifs) and _is_complementable(motifs[0])
    if input_data["both_strands"] and motifs and not both_strands:
        logger.info("Motif alphabet is not complementable; scanning the given strand only (both_strands ignored).")

    fimo = FIMO(both_strands=both_strands, threshold=input_data["threshold"])

    matches: list[dict[str, Any]] = []
    for motif in motifs:
        accession = _decode(motif.accession)
        name = _decode(motif.name)
        motif_id = accession or name
        motif_alt_id = name if (name and name != motif_id) else "-"

        # FIMO reports start <= stop with strand separate; pymemesuite gives
        # strand-oriented coordinates, so normalize to (min, max).
        pattern = fimo.score_motif(motif, sequences, background)
        matches.extend(
            {
                "motif_id": motif_id,
                "motif_alt_id": motif_alt_id,
                "sequence_name": _decode(element.source.name),
                "start": int(min(element.start, element.stop)),
                "stop": int(max(element.start, element.stop)),
                "strand": element.strand,
                "score": float(element.score),
                "pvalue": float(element.pvalue),
                "qvalue": float(element.qvalue),
                "matched_sequence": _decode(element.sequence),
            }
            for element in pattern.matched_elements
        )

    return {
        "matches": matches,
        "num_sequences": len(sequences),
        "num_motifs": len(motifs),
    }


# ============================================================================
# Device protocol (CPU tool — no persistent GPU state)
# ============================================================================
def to_device(device: str) -> dict[str, Any]:
    """Passthrough — FIMO is CPU-only and keeps no persistent device state."""
    return {"success": True, "device": device, "note": "CPU tool"}


def get_memory_stats() -> dict[str, Any]:
    """FIMO is CPU-only; no GPU memory to report."""
    return {"available": False, "framework": "cpu", "note": "CPU tool"}


# ============================================================================
# Entry point (called by ToolInstance)
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"meme: usage: python {sys.argv[0]} <input_json_path> <output_json_path>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    operation = input_data["operation"]
    if operation == "fimo_scan":
        output_data = run_fimo_scan(input_data)
    else:
        raise ValueError(f"meme: unknown operation {operation!r}; valid: ['fimo_scan']")

    with open(sys.argv[2], "w") as f:
        json.dump(output_data, f)

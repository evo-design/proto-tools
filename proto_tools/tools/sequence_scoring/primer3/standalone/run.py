"""Primer3 thermodynamics standalone runner for ToolInstance venv execution.

Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>
"""

import json
import sys
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)

# primer3-py returns dimer/hairpin ΔG in cal/mol; divide by this for kcal/mol.
_CAL_PER_KCAL = 1000.0


# =============================================================================
# Implementation
# =============================================================================
def _gc_content(sequence: str) -> float:
    """Fraction of G/C bases in the oligo (0–1)."""
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def _gc_clamp(sequence: str) -> bool:
    """True if either of the last two 3' bases is G or C."""
    return any(base in "GC" for base in sequence[-2:])


def _score_oligo(oligo: dict[str, Any], oligo_id: str, thermo_kwargs: dict[str, float]) -> dict[str, Any]:
    """Compute Primer3 thermodynamic scores for a single oligo (and optional partner)."""
    import primer3

    sequence = oligo["sequence"]
    partner = oligo.get("partner")

    # calc_tm takes ionic/oligo conditions but not temp_c (it computes the melting point itself).
    tm_kwargs = {k: v for k, v in thermo_kwargs.items() if k != "temp_c"}

    tm = primer3.calc_tm(sequence, **tm_kwargs)
    hairpin = primer3.calc_hairpin(sequence, **thermo_kwargs)
    homodimer = primer3.calc_homodimer(sequence, **thermo_kwargs)

    result: dict[str, Any] = {
        "oligo_id": oligo_id,
        "length": len(sequence),
        "tm": float(tm),
        "hairpin_dg": hairpin.dg / _CAL_PER_KCAL,
        "homodimer_dg": homodimer.dg / _CAL_PER_KCAL,
        "gc_content": round(_gc_content(sequence), 4),
        "gc_clamp": _gc_clamp(sequence),
        "hairpin_structure_found": bool(hairpin.structure_found),
        "homodimer_structure_found": bool(homodimer.structure_found),
        "heterodimer_dg": None,
        "heterodimer_structure_found": None,
    }

    if partner:
        heterodimer = primer3.calc_heterodimer(sequence, partner, **thermo_kwargs)
        result["heterodimer_dg"] = heterodimer.dg / _CAL_PER_KCAL
        result["heterodimer_structure_found"] = bool(heterodimer.structure_found)

    return result


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Score each input oligo and emit per-oligo thermodynamic results."""
    oligos = input_dict["oligos"]
    oligo_ids = input_dict["oligo_ids"]
    config = input_dict.get("config", {})
    thermo_kwargs = {
        "mv_conc": float(config["mv_conc"]),
        "dv_conc": float(config["dv_conc"]),
        "dntp_conc": float(config["dntp_conc"]),
        "dna_conc": float(config["dna_conc"]),
        "temp_c": float(config["temp_c"]),
    }

    logger.debug("Scoring %d oligo(s) with primer3 thermodynamics", len(oligos))
    results = [
        _score_oligo(oligo, oligo_id, thermo_kwargs)
        for oligo, oligo_id in zip(oligos, oligo_ids, strict=True)
    ]
    return {"results": results}


# =============================================================================
# Device protocol (CPU-only tool)
# =============================================================================
def to_device(device: str) -> dict[str, Any]:
    """No-op for CPU tools — workers unload between calls."""
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """CPU-only tool — no GPU memory to report."""
    return {"available": False, "framework": "cpu", "note": "CPU tool"}


# =============================================================================
# Entry point (called by ToolInstance)
# =============================================================================
if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        input_data = json.load(f)
    output = dispatch(input_data)
    with open(sys.argv[2], "w") as f:
        json.dump(output, f)

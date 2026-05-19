"""proto_tools/tools/mutagenesis/codons.py.

Self-contained codon infrastructure for random mutagenesis tools.
No BioPython dependency.
"""

import functools
import random

from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS

# ============================================================================
# Standard genetic code (codon → amino acid)
# ============================================================================

CODON_TO_AA: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# ============================================================================
# IUPAC ambiguity codes for DNA
# ============================================================================

IUPAC_DNA: dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "GC",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}

STANDARD_AMINO_ACIDS = PROTEIN_AMINO_ACIDS

# Named codon schemes supported by the protein sampler
COMMON_CODON_SCHEMES: list[str] = [
    "UNIFORM",
    "NNN",
    "NNK",
    "NNS",
    "NDT",
    "DBK",
    "NRT",
]


# ============================================================================
# Codon scheme expansion
# ============================================================================


def _expand_degenerate_codon(codon: str) -> list[str]:
    """Expand a 3-letter degenerate codon into all concrete codons."""
    pools = [list(IUPAC_DNA[base]) for base in codon]
    results = [""]
    for pool in pools:
        results = [r + b for r in results for b in pool]
    return results


@functools.cache
def get_codon_scheme(name: str, include_stop: bool = False) -> dict[str, list[str] | dict[str, float]]:
    """Expand a codon scheme name into its codons and amino acid weights.

    Args:
        name (str): Codon scheme name (e.g., ``"NNK"``, ``"NNS"``, ``"UNIFORM"``).
            Case-insensitive. Must be a valid IUPAC degenerate codon or ``"UNIFORM"``.
        include_stop (bool): If True, include the stop symbol ``"*"`` in the
            returned amino acids and weights. For degenerate schemes ``"*"`` is
            weighted by its stop-codon count, exactly like amino acids; for
            ``"UNIFORM"`` it is added as an equally weighted 21st symbol.

    Returns a dict with:
        - ``"codons"``: list of concrete DNA codons in the scheme
        - ``"amino_acids"``: list of reachable amino acids (``"*"`` included
          only when ``include_stop`` is True)
        - ``"weights"``: dict mapping each amino acid to its sampling weight
          (proportional to the number of codons encoding it)

    For ``"UNIFORM"``, all 20 standard amino acids have equal weight.
    """
    name = name.upper()
    if name == "UNIFORM":
        weights = dict.fromkeys(STANDARD_AMINO_ACIDS, 1.0)
        if include_stop:
            weights["*"] = 1.0
        return {
            "codons": [],
            "amino_acids": sorted(weights),
            "weights": weights,
        }

    codons = _expand_degenerate_codon(name)
    weights: dict[str, float] = {}  # type: ignore[no-redef]
    for codon in codons:
        aa = CODON_TO_AA.get(codon)
        if aa is None or (aa == "*" and not include_stop):
            continue
        weights[aa] = weights.get(aa, 0.0) + 1.0

    if not weights:
        raise ValueError(f"Codon scheme '{name}' produces no amino acids (all stop codons).")

    return {
        "codons": codons,
        "amino_acids": sorted(weights),
        "weights": weights,
    }


def sample_amino_acid(scheme: str, rng: random.Random | None = None, include_stop: bool = False) -> str:
    """Sample a single amino acid from a codon scheme.

    Stops are excluded unless ``include_stop`` is True, in which case ``"*"``
    may be drawn with the weight defined by :func:`get_codon_scheme`.
    """
    info = get_codon_scheme(scheme, include_stop=include_stop)
    weights = info["weights"]
    assert isinstance(weights, dict)
    aas = list(weights.keys())
    ws = list(weights.values())
    r = rng or random
    return r.choices(aas, weights=ws, k=1)[0]


# ============================================================================
# IUPAC nucleotide expansion and sampling
# ============================================================================


@functools.cache
def get_substitution_pool(iupac: str) -> list[str]:
    """Expand an IUPAC ambiguity code into a list of concrete nucleotides."""
    iupac = iupac.upper()
    if iupac not in IUPAC_DNA:
        raise ValueError(f"Unknown IUPAC code '{iupac}'. Valid codes: {', '.join(sorted(IUPAC_DNA))}")
    return list(IUPAC_DNA[iupac])


def sample_nucleotide(iupac: str, rng: random.Random | None = None) -> str:
    """Sample a single nucleotide from an IUPAC ambiguity code."""
    pool = get_substitution_pool(iupac)
    r = rng or random
    return r.choice(pool)

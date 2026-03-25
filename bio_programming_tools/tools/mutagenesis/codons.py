"""Codon tables, IUPAC ambiguity codes, and sampling utilities.

Self-contained codon infrastructure for random mutagenesis tools.
No BioPython dependency.
"""
from __future__ import annotations

import functools
import random

# ============================================================================
# Standard genetic code (codon → amino acid)
# ============================================================================

CODON_TO_AA: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
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

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Named codon schemes supported by the protein sampler
COMMON_CODON_SCHEMES: list[str] = [
    "UNIFORM", "NNN", "NNK", "NNS", "NDT", "DBK", "NRT",
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


@functools.lru_cache(maxsize=None)
def get_codon_scheme(name: str) -> dict[str, list[str] | dict[str, float]]:
    """Expand a codon scheme name into its codons and amino acid weights.

    Returns a dict with:
        - ``"codons"``: list of concrete DNA codons in the scheme
        - ``"amino_acids"``: list of reachable amino acids (stops excluded)
        - ``"weights"``: dict mapping each amino acid to its sampling weight
          (proportional to the number of codons encoding it)

    For ``"UNIFORM"``, all 20 standard amino acids have equal weight.
    """
    name = name.upper()
    if name == "UNIFORM":
        weights = {aa: 1.0 for aa in STANDARD_AMINO_ACIDS}
        return {
            "codons": [],
            "amino_acids": sorted(weights),
            "weights": weights,
        }

    codons = _expand_degenerate_codon(name)
    weights: dict[str, float] = {}
    for codon in codons:
        aa = CODON_TO_AA.get(codon)
        if aa is None or aa == "*":
            continue
        weights[aa] = weights.get(aa, 0.0) + 1.0

    if not weights:
        raise ValueError(
            f"Codon scheme '{name}' produces no amino acids (all stop codons)."
        )

    return {
        "codons": codons,
        "amino_acids": sorted(weights),
        "weights": weights,
    }


def sample_amino_acid(scheme: str, rng: random.Random | None = None) -> str:
    """Sample a single amino acid from a codon scheme (stops excluded)."""
    info = get_codon_scheme(scheme)
    weights = info["weights"]
    aas = list(weights.keys())
    ws = list(weights.values())
    r = rng or random
    return r.choices(aas, weights=ws, k=1)[0]


# ============================================================================
# IUPAC nucleotide expansion and sampling
# ============================================================================

@functools.lru_cache(maxsize=None)
def get_substitution_pool(iupac: str) -> list[str]:
    """Expand an IUPAC ambiguity code into a list of concrete nucleotides."""
    iupac = iupac.upper()
    if iupac not in IUPAC_DNA:
        raise ValueError(
            f"Unknown IUPAC code '{iupac}'. "
            f"Valid codes: {', '.join(sorted(IUPAC_DNA))}"
        )
    return list(IUPAC_DNA[iupac])


def sample_nucleotide(iupac: str, rng: random.Random | None = None) -> str:
    """Sample a single nucleotide from an IUPAC ambiguity code."""
    pool = get_substitution_pool(iupac)
    r = rng or random
    return r.choice(pool)

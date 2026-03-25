"""Tests for codon tables, IUPAC codes, and sampling utilities."""
import random

import pytest

from bio_programming_tools.tools.mutagenesis.codons import (
    CODON_TO_AA,
    COMMON_CODON_SCHEMES,
    IUPAC_DNA,
    STANDARD_AMINO_ACIDS,
    get_codon_scheme,
    get_substitution_pool,
    sample_amino_acid,
    sample_nucleotide,
)

# ============================================================================
# Codon table
# ============================================================================

def test_codon_table_has_64_entries():
    assert len(CODON_TO_AA) == 64


def test_codon_table_covers_all_amino_acids():
    aas = {v for v in CODON_TO_AA.values() if v != "*"}
    assert aas == set(STANDARD_AMINO_ACIDS)


def test_codon_table_has_three_stop_codons():
    stops = [k for k, v in CODON_TO_AA.items() if v == "*"]
    assert set(stops) == {"TAA", "TAG", "TGA"}


# ============================================================================
# IUPAC codes
# ============================================================================

def test_iupac_single_bases():
    for base in "ACGT":
        assert IUPAC_DNA[base] == base


# ============================================================================
# get_codon_scheme
# ============================================================================

def test_uniform_scheme_equal_weights():
    info = get_codon_scheme("UNIFORM")
    weights = list(info["weights"].values())
    assert all(w == 1.0 for w in weights)


def test_nnk_scheme_codon_count():
    info = get_codon_scheme("NNK")
    # NNK: N=ACGT, N=ACGT, K=GT -> 4*4*2 = 32 codons
    assert len(info["codons"]) == 32
    assert "*" not in info["amino_acids"]


def test_ndt_scheme_codon_count():
    info = get_codon_scheme("NDT")
    # NDT: N=ACGT, D=AGT, T=T -> 4*3*1 = 12 codons, 12 amino acids
    assert len(info["codons"]) == 12
    assert len(info["amino_acids"]) == 12


@pytest.mark.parametrize("scheme", COMMON_CODON_SCHEMES)
def test_all_common_schemes_valid(scheme):
    info = get_codon_scheme(scheme)
    assert len(info["amino_acids"]) > 0
    assert "*" not in info["amino_acids"]
    assert all(w > 0 for w in info["weights"].values())


def test_case_insensitive():
    info_upper = get_codon_scheme("NNK")
    info_lower = get_codon_scheme("nnk")
    assert info_upper["amino_acids"] == info_lower["amino_acids"]


# ============================================================================
# sample_amino_acid
# ============================================================================

def test_sample_amino_acid_nnk_no_stops():
    rng = random.Random(42)
    for _ in range(100):
        aa = sample_amino_acid("NNK", rng=rng)
        assert aa in STANDARD_AMINO_ACIDS
        assert aa != "*"


def test_sample_amino_acid_reproducible():
    a1 = sample_amino_acid("NNK", rng=random.Random(123))
    a2 = sample_amino_acid("NNK", rng=random.Random(123))
    assert a1 == a2


# ============================================================================
# get_substitution_pool
# ============================================================================

@pytest.mark.parametrize("code, expected", [
    ("N", {"A", "C", "G", "T"}),
    ("R", {"A", "G"}),
    ("Y", {"C", "T"}),
    ("S", {"G", "C"}),
    ("W", {"A", "T"}),
])
def test_substitution_pool(code, expected):
    assert set(get_substitution_pool(code)) == expected


def test_substitution_pool_invalid():
    with pytest.raises(ValueError, match="Unknown IUPAC code"):
        get_substitution_pool("Z")


# ============================================================================
# sample_nucleotide
# ============================================================================

def test_sample_nucleotide_reproducible():
    b1 = sample_nucleotide("N", rng=random.Random(99))
    b2 = sample_nucleotide("N", rng=random.Random(99))
    assert b1 == b2


def test_sample_nucleotide_invalid_code():
    with pytest.raises(ValueError, match="Unknown IUPAC code"):
        sample_nucleotide("X")

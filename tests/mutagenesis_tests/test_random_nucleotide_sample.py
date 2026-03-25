"""Tests for random nucleotide sampling tool."""
import pytest

from bio_programming_tools.tools.mutagenesis.codons import IUPAC_DNA
from bio_programming_tools.tools.mutagenesis.random_nucleotide import (
    RandomNucleotideSampleConfig,
    RandomNucleotideSampleInput,
    run_random_nucleotide_sample,
)

# ============================================================================
# Substitution scheme variations
# ============================================================================

@pytest.mark.parametrize("scheme", ["N", "R", "Y", "S", "W", "K", "M"])
def test_substitution_scheme(scheme):
    valid_bases = set(IUPAC_DNA[scheme])
    config = RandomNucleotideSampleConfig(
        substitution_scheme=scheme, seed=42,
    )
    result = run_random_nucleotide_sample(
        RandomNucleotideSampleInput(sequences=["____"]),
        config=config,
    )
    for ch in result.sequences[0]:
        assert ch in valid_bases


# ============================================================================
# DNA vs RNA auto-detection
# ============================================================================

def test_dna_auto_detection():
    config = RandomNucleotideSampleConfig(seed=42)
    result = run_random_nucleotide_sample(
        RandomNucleotideSampleInput(sequences=["ACGT_CGT"]),
        config=config,
    )
    assert "U" not in result.sequences[0]


def test_rna_auto_detection():
    config = RandomNucleotideSampleConfig(seed=42)
    result = run_random_nucleotide_sample(
        RandomNucleotideSampleInput(sequences=["AUGC_CGU"]),
        config=config,
    )
    seq = result.sequences[0]
    assert seq[4] in "ACGU"
    assert "T" not in seq[4]


def test_explicit_rna():
    config = RandomNucleotideSampleConfig(
        sequence_type="rna", seed=42,
    )
    result = run_random_nucleotide_sample(
        RandomNucleotideSampleInput(sequences=["____"]),
        config=config,
    )
    for ch in result.sequences[0]:
        assert ch in "ACGU"


# ============================================================================
# Reproducibility
# ============================================================================

def test_seed_reproducibility():
    inp = RandomNucleotideSampleInput(sequences=["____"])
    config = RandomNucleotideSampleConfig(seed=42)
    r1 = run_random_nucleotide_sample(inp, config)
    r2 = run_random_nucleotide_sample(inp, config)
    assert r1.sequences == r2.sequences


# ============================================================================
# Output export
# ============================================================================

def test_export_fasta(tmp_path):
    result = run_random_nucleotide_sample(
        RandomNucleotideSampleInput(sequences=["AC_T"]),
    )
    path = tmp_path / "out.fasta"
    result._export_output(str(path), "fasta")
    content = path.read_text()
    assert content.startswith(">seq_0")


def test_export_txt(tmp_path):
    result = run_random_nucleotide_sample(
        RandomNucleotideSampleInput(sequences=["AC_T", "G_C"]),
    )
    path = tmp_path / "out.txt"
    result._export_output(str(path), "txt")
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_export_json(tmp_path):
    import json

    result = run_random_nucleotide_sample(
        RandomNucleotideSampleInput(sequences=["AC_T"]),
    )
    path = tmp_path / "out.json"
    result._export_output(str(path), "json")
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1


def test_export_unsupported_format(tmp_path):
    result = run_random_nucleotide_sample(
        RandomNucleotideSampleInput(sequences=["AC_T"]),
    )
    with pytest.raises(ValueError, match="Unsupported format"):
        result._export_output(str(tmp_path / "out"), "csv")

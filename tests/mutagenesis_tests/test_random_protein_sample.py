"""tests/mutagenesis_tests/test_random_protein_sample.py

Tests for random protein sampling tool."""
import pytest

from proto_tools.tools.mutagenesis.codons import get_codon_scheme
from proto_tools.tools.mutagenesis.random_protein import (
    RandomProteinSampleConfig,
    RandomProteinSampleInput,
    run_random_protein_sample,
)

# ============================================================================
# Codon scheme variations
# ============================================================================

@pytest.mark.parametrize("scheme", ["UNIFORM", "NNK", "NNS", "NDT", "DBK", "NRT"])
def test_codon_scheme(scheme):
    info = get_codon_scheme(scheme)
    valid_aas = set(info["amino_acids"])
    config = RandomProteinSampleConfig(codon_scheme=scheme, seed=42)
    result = run_random_protein_sample(
        RandomProteinSampleInput(sequences=["____"]),
        config=config,
    )
    for ch in result.sequences[0]:
        assert ch in valid_aas


# ============================================================================
# Reproducibility
# ============================================================================

def test_seed_reproducibility():
    inp = RandomProteinSampleInput(sequences=["____"])
    config = RandomProteinSampleConfig(seed=42)
    r1 = run_random_protein_sample(inp, config)
    r2 = run_random_protein_sample(inp, config)
    assert r1.sequences == r2.sequences


# ============================================================================
# Output export
# ============================================================================

def test_export_fasta(tmp_path):
    result = run_random_protein_sample(
        RandomProteinSampleInput(sequences=["M_TL"]),
    )
    path = tmp_path / "out.fasta"
    result._export_output(str(path), "fasta")
    content = path.read_text()
    assert content.startswith(">seq_0")
    assert len(content.strip().split("\n")) == 2


def test_export_txt(tmp_path):
    result = run_random_protein_sample(
        RandomProteinSampleInput(sequences=["M_TL", "A_C"]),
    )
    path = tmp_path / "out.txt"
    result._export_output(str(path), "txt")
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_export_json(tmp_path):
    import json

    result = run_random_protein_sample(
        RandomProteinSampleInput(sequences=["M_TL"]),
    )
    path = tmp_path / "out.json"
    result._export_output(str(path), "json")
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1


def test_export_unsupported_format(tmp_path):
    result = run_random_protein_sample(
        RandomProteinSampleInput(sequences=["M_TL"]),
    )
    with pytest.raises(ValueError, match="Unsupported format"):
        result._export_output(str(tmp_path / "out"), "csv")

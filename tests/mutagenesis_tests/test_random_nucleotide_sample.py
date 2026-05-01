"""tests/mutagenesis_tests/test_random_nucleotide_sample.py.

Tests for random nucleotide sampling tool.
"""

import pytest

from proto_tools.tools.mutagenesis.codons import IUPAC_DNA
from proto_tools.tools.mutagenesis.random_nucleotide import (
    RandomNucleotideSampleConfig,
    RandomNucleotideSampleInput,
    run_random_nucleotide_sample,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

# ============================================================================
# Substitution scheme variations
# ============================================================================


@pytest.mark.parametrize("scheme", ["N", "R", "Y", "S", "W", "K", "M"])
def test_substitution_scheme(scheme):
    valid_bases = set(IUPAC_DNA[scheme])
    config = RandomNucleotideSampleConfig(
        substitution_scheme=scheme,
        seed=42,
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
        sequence_type="rna",
        seed=42,
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


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("random-nucleotide-sample")
@pytest.mark.slow
def test_random_nucleotide_sample_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark random-nucleotide-sample: 1000 fully-masked 1000-bp sequences, IUPAC scheme=N (cold + warm).

    Pure-Python sampler with no persistent worker, so we time both passes directly.
    """
    import time

    masked = ["_" * 1000] * 1000
    inputs = RandomNucleotideSampleInput(sequences=masked)
    config = RandomNucleotideSampleConfig(substitution_scheme="N", seed=0)
    runner = lambda: run_random_nucleotide_sample(inputs, config)  # noqa: E731

    t0 = time.perf_counter()
    _ = runner()
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = runner()
    warm = time.perf_counter() - t0
    request.node.user_properties.append(("cold_seconds", cold))
    request.node.user_properties.append(("warm_seconds", warm))

    validate_output(result)
    assert result.tool_id == "random-nucleotide-sample"
    assert len(result.sequences) == 1000
    assert all(len(s) == 1000 for s in result.sequences)

"""tests/gene_annotation_tests/test_meme_fimo_scan.py.

Tests for the MEME Suite FIMO motif scanning tool.
"""

from pathlib import Path

import pytest

from proto_tools.tools import (
    MEMEFimoScanConfig,
    MEMEFimoScanInput,
    run_meme_fimo_scan,
)
from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

_MEME_DIR = Path(__file__).parent.parent.parent / "proto_tools" / "tools" / "gene_annotation" / "meme"
EXAMPLE_MEME_FILE = _MEME_DIR / "examples" / "example.meme"

# Consensus GAGCTGGTCA appears twice on the forward strand.
SAMPLE_SEQUENCE = "GTTGAGCTGGTCAACAAGTTGAGCTGGTCAAC"


# ── Input validation ─────────────────────────────────────────────────────


def test_input_single_sequence_normalized_to_list():
    """A single sequence string is normalized to a one-element list."""
    inputs = MEMEFimoScanInput(sequences=SAMPLE_SEQUENCE, motifs=str(EXAMPLE_MEME_FILE))
    assert inputs.sequences == [SAMPLE_SEQUENCE]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fimo_scan_basic_execution():
    """FIMO finds the bundled motif in the sample DNA sequence."""
    inputs = MEMEFimoScanInput(sequences=SAMPLE_SEQUENCE, motifs=str(EXAMPLE_MEME_FILE))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig())

    assert result.success is True
    assert result.tool_id == "meme-fimo-scan"
    assert len(result.results) == 1  # one bundle per input sequence
    assert result.num_matches >= 1

    match = result.results[0].matches[0]
    assert match.motif_id == "MA0TEST"
    assert match.strand in {"+", "-"}
    assert 0 < match.pvalue <= 1e-4
    assert match.start <= match.stop


@pytest.mark.integration
def test_fimo_scan_results_align_to_inputs():
    """Output is 1:1 with inputs; a sequence with no occurrences yields an empty bundle."""
    inputs = MEMEFimoScanInput(sequences=[SAMPLE_SEQUENCE, "AAAAAAAAAAAAAAAA"], motifs=str(EXAMPLE_MEME_FILE))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig())

    assert result.success is True
    assert len(result.results) == 2  # positionally aligned to the two inputs
    assert len(result.results[0].matches) >= 1  # motif present in sequence 0
    assert result.results[1].matches == []  # no motif in sequence 1 -> empty bundle


@pytest.mark.integration
def test_fimo_scan_single_strand_finds_forward_matches():
    """Disabling reverse-strand scanning still recovers the forward occurrences."""
    inputs = MEMEFimoScanInput(sequences=SAMPLE_SEQUENCE, motifs=str(EXAMPLE_MEME_FILE))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig(both_strands=False))

    assert result.success is True
    assert result.tool_id == "meme-fimo-scan"
    assert result.num_matches >= 1
    assert all(m.strand == "+" for r in result.results for m in r.matches)


def _write_protein_motif(path: Path) -> None:
    """Write a minimal 3-position protein motif (consensus MKL) in MEME format."""
    aa = "ACDEFGHIKLMNPQRSTVWY"
    rows = "\n".join(" ".join("0.81" if c == dom else "0.01" for c in aa) for dom in "MKL")
    path.write_text(
        f"MEME version 4\n\nALPHABET= {aa}\n\n"
        f"MOTIF P1 prot\nletter-probability matrix: alength= 20 w= 3 nsites= 20 E= 0\n{rows}\n"
    )


@pytest.mark.integration
def test_fimo_scan_protein_motif_is_forward_only(tmp_path):
    """both_strands=True is ignored for a protein (non-complementable) motif.

    Without the alphabet guard, pymemesuite emits spurious reverse-strand hits on
    protein; the tool must match the FIMO CLI and scan the given strand only.
    """
    motif = tmp_path / "prot.meme"
    _write_protein_motif(motif)
    inputs = MEMEFimoScanInput(sequences="AAAMKLAAAMKLAAA", motifs=str(motif))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig(both_strands=True, threshold=1e-2))

    assert result.success is True
    assert result.num_matches >= 1
    assert all(m.strand == "+" for r in result.results for m in r.matches)


@pytest.mark.integration
def test_fimo_scan_export_csv(tmp_path):
    """Results export to a CSV file on disk."""
    inputs = MEMEFimoScanInput(sequences=SAMPLE_SEQUENCE, motifs=str(EXAMPLE_MEME_FILE))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig())

    assert result.success is True
    assert result.num_matches >= 1

    result.export(name="fimo_matches", export_path=tmp_path, file_format="csv")

    written = tmp_path / "fimo_matches"
    assert written.exists()
    assert written.stat().st_size > 0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark("meme-fimo-scan")
@pytest.mark.slow
def test_meme_fimo_scan_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark meme-fimo-scan: scan 15000 random 1 kb DNA sequences against the bundled motif (cold + warm)."""
    sequences = random_dna_sequences(n=15000, length=1000, seed=0)
    inputs = MEMEFimoScanInput(sequences=sequences, motifs=str(EXAMPLE_MEME_FILE))
    config = MEMEFimoScanConfig()

    result = benchmark_twice(request, "meme", lambda: run_meme_fimo_scan(inputs, config))
    validate_output(result)

    assert result.tool_id == "meme-fimo-scan"
    assert result.success is True
    assert len(result.results) == 15000  # one bundle per input sequence
    assert result.num_matches > 0

"""tests/sequence_scoring_tests/test_primer3_thermodynamics.py.

Tests for the Primer3 thermodynamics tool.
"""

import pytest

from proto_tools.tools.sequence_scoring.primer3 import (
    Primer3Oligo,
    Primer3OligoResult,
    Primer3ThermodynamicsConfig,
    Primer3ThermodynamicsInput,
    Primer3ThermodynamicsOutput,
    run_primer3_thermodynamics,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_export_output, validate_output

# A validated GAPDH-style qPCR primer pair (forward + reverse).
FWD = "ACCCACTCCTCCACCTTTGA"
REV = "CTGTTGCTGTAGCCAAATTCGT"


# ── Input validation (custom validators) ────────────────────────────────────


def test_bare_string_is_coerced_to_oligo_list():
    """A bare DNA string becomes a one-element list of Primer3Oligo."""
    inp = Primer3ThermodynamicsInput(oligos=FWD)
    assert isinstance(inp.oligos, list)
    assert len(inp.oligos) == 1
    assert inp.oligos[0].sequence == FWD
    assert inp.oligos[0].partner is None


def test_dict_with_partner_is_coerced():
    """A {'sequence', 'partner'} dict is coerced into a Primer3Oligo."""
    inp = Primer3ThermodynamicsInput(oligos=[{"sequence": FWD, "partner": REV}])
    assert inp.oligos[0].partner == REV


def test_sequence_is_uppercased():
    """Lowercase input is normalized to uppercase."""
    assert Primer3Oligo(sequence="acgt").sequence == "ACGT"


def test_non_acgt_sequence_is_rejected():
    """Degenerate/ambiguous bases are rejected — thermodynamics needs concrete bases."""
    with pytest.raises(ValueError, match="non-ACGT"):
        Primer3Oligo(sequence="ACGTN")


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        Primer3Oligo(sequence="   ")


# ── Export ───────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_output() -> Primer3ThermodynamicsOutput:
    result = Primer3OligoResult(
        oligo_id="oligo_0",
        length=len(FWD),
        tm=60.4,
        hairpin_dg=0.0,
        homodimer_dg=-0.98,
        heterodimer_dg=-2.57,
        gc_content=0.55,
        gc_clamp=True,
        hairpin_structure_found=False,
        homodimer_structure_found=True,
        heterodimer_structure_found=True,
    )
    return Primer3ThermodynamicsOutput(results=[result])


def test_export_csv(sample_output: Primer3ThermodynamicsOutput, tmp_path):
    sample_output.export(name="primer3", export_path=str(tmp_path), file_format="csv")
    assert validate_export_output(tmp_path / "primer3.csv")


def test_export_json(sample_output: Primer3ThermodynamicsOutput, tmp_path):
    sample_output.export(name="primer3", export_path=str(tmp_path), file_format="json")
    assert validate_export_output(tmp_path / "primer3.json")


# ── Integration ──────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_scores_primer_pair_with_heterodimer():
    """A real qPCR pair scores a sensible Tm and computes heterodimer ΔG for the partner."""
    inputs = Primer3ThermodynamicsInput(oligos=[{"sequence": FWD, "partner": REV}])
    result = run_primer3_thermodynamics(inputs, Primer3ThermodynamicsConfig())

    assert result.success
    assert result.tool_id == "primer3-thermodynamics"
    assert len(result.results) == 1

    r = result.results[0]
    assert r.oligo_id == "oligo_0"
    assert 55.0 < r.tm < 65.0  # forward primer designed for ~60 °C
    assert 0.4 < r.gc_content < 0.6
    assert r.gc_clamp is True
    assert r.heterodimer_dg is not None  # partner supplied → heterodimer computed


@pytest.mark.integration
def test_no_partner_leaves_heterodimer_null():
    """Without a partner, heterodimer fields are null but the oligo is still scored."""
    inputs = Primer3ThermodynamicsInput(oligos=FWD)
    result = run_primer3_thermodynamics(inputs, Primer3ThermodynamicsConfig())

    r = result.results[0]
    assert r.heterodimer_dg is None
    assert r.heterodimer_structure_found is None
    assert r.tm > 0


@pytest.mark.integration
def test_batch_scores_in_input_order():
    """Multiple oligos are scored and returned index-aligned with the input."""
    inputs = Primer3ThermodynamicsInput(oligos=[FWD, REV])
    result = run_primer3_thermodynamics(inputs, Primer3ThermodynamicsConfig())

    assert [r.oligo_id for r in result.results] == ["oligo_0", "oligo_1"]
    assert result.results[0].length == len(FWD)
    assert result.results[1].length == len(REV)


# ── Benchmark ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("primer3-thermodynamics")
@pytest.mark.slow
def test_primer3_thermodynamics_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark primer3-thermodynamics: 100 distinct 20-mers scored (cold + warm)."""
    # 100 distinct oligos: the first four bases encode the index, the rest is fixed.
    bases = "ACGT"
    oligos = ["".join(bases[(i // 4**j) % 4] for j in range(4)) + FWD[4:] for i in range(100)]
    inputs = Primer3ThermodynamicsInput(oligos=oligos)

    result = benchmark_twice(
        request, "primer3", lambda: run_primer3_thermodynamics(inputs, Primer3ThermodynamicsConfig())
    )
    validate_output(result)

    assert result.tool_id == "primer3-thermodynamics"
    assert len(result.results) == 100

"""tests/gene_annotation_tests/test_minced.py.

Tests for the MinCED CRISPR array detection tool.
"""

import pytest
from pydantic import ValidationError

from proto_tools.tools.gene_annotation.minced import (
    CrisprArray,
    CrisprRepeatSpacer,
    MincedConfig,
    MincedInput,
    MincedOutput,
    MincedSequenceResult,
    run_minced,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
    validate_output,
)

_persistent_tool = make_persistent_fixture("minced", gpu=False)

# Known CRISPR-containing sequence fragment (synthetic, for testing)
_CRISPR_SEQUENCE = (
    "ATCGATCGATCGATCGATCGATCGATCG"  # Leader
    + (
        "GTTTTAGAGCTATGCTGTTTTGAATGGTCCCAAAAC"  # Repeat (36nt)
        + "AAAAAAAACCCCCCCCTTTTTTTTGGGGGGGG"  # Spacer (31nt)
    )
    * 5  # 5 repeat-spacer units
    + "GTTTTAGAGCTATGCTGTTTTGAATGGTCCCAAAAC"  # Final repeat
    + "ATCGATCGATCGATCGATCGATCGATCG"  # Trailer
)


# ── Input validation ─────────────────────────────────────────────────────


def test_input_single_sequence_normalization():
    """Single string should be normalized to list."""
    inp = MincedInput(sequences="ATCGATCG")
    assert isinstance(inp.sequences, list)
    assert len(inp.sequences) == 1
    assert inp.sequences[0] == "ATCGATCG"


def test_input_list_of_sequences():
    inp = MincedInput(sequences=["ATCG", "GCTA"])
    assert len(inp.sequences) == 2


def test_config_min_num_repeats_validation():
    """min_num_repeats must be >= 2."""
    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        MincedConfig(min_num_repeats=1)


def test_config_min_repeat_length_validation():
    """min_repeat_length must be >= 10."""
    with pytest.raises(ValidationError, match="greater than or equal to 10"):
        MincedConfig(min_repeat_length=5)


def test_inverted_ranges_rejected():
    """MinCED silently accepts inverted ranges; the wrapper rejects them at construction."""
    with pytest.raises(ValidationError, match="max_repeat_length"):
        MincedConfig(min_repeat_length=30, max_repeat_length=20)
    with pytest.raises(ValidationError, match="max_spacer_length"):
        MincedConfig(min_spacer_length=40, max_spacer_length=20)


# ── Data model properties ────────────────────────────────────────────────


def test_crispr_array_num_repeats():
    rs1 = CrisprRepeatSpacer(position=0, repeat="ATCG", spacer="GCTA")
    rs2 = CrisprRepeatSpacer(position=50, repeat="ATCG", spacer="TTAA")
    rs3 = CrisprRepeatSpacer(position=100, repeat="ATCG")
    array = CrisprArray(repeats_and_spacers=[rs1, rs2, rs3])
    assert array.num_repeats == 3


def test_crispr_array_spacers_property():
    rs1 = CrisprRepeatSpacer(position=0, repeat="ATCG", spacer="GCTA")
    rs2 = CrisprRepeatSpacer(position=50, repeat="ATCG", spacer="TTAA")
    rs3 = CrisprRepeatSpacer(position=100, repeat="ATCG")  # Last repeat, no spacer
    array = CrisprArray(repeats_and_spacers=[rs1, rs2, rs3])
    spacers = array.spacers
    assert len(spacers) == 2
    assert "GCTA" in spacers
    assert "TTAA" in spacers


def test_crispr_array_empty():
    array = CrisprArray()
    assert array.num_repeats == 0
    assert array.spacers == []


def test_sequence_result_has_crispr_with_arrays():
    rs = CrisprRepeatSpacer(position=0, repeat="ATCG")
    array = CrisprArray(repeats_and_spacers=[rs])
    result = MincedSequenceResult(sequence_id="test", crispr_arrays=[array])
    assert result.has_crispr is True
    assert result.num_arrays == 1


def test_sequence_result_has_crispr_without_arrays():
    result = MincedSequenceResult(sequence_id="test", crispr_arrays=[])
    assert result.has_crispr is False
    assert result.num_arrays == 0


def test_num_sequences_with_crispr():
    rs = CrisprRepeatSpacer(position=0, repeat="ATCG")
    array = CrisprArray(repeats_and_spacers=[rs])

    r1 = MincedSequenceResult(sequence_id="seq1", crispr_arrays=[array])
    r2 = MincedSequenceResult(sequence_id="seq2", crispr_arrays=[])
    r3 = MincedSequenceResult(sequence_id="seq3", crispr_arrays=[array])

    output = MincedOutput(results=[r1, r2, r3])
    assert output.num_sequences_with_crispr == 2


def test_num_sequences_with_crispr_empty():
    output = MincedOutput(results=[])
    assert output.num_sequences_with_crispr == 0


# ── Export ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_output():
    rs1 = CrisprRepeatSpacer(
        position=100,
        repeat="ATCGATCG",
        spacer="GCTAGCTA",
        repeat_length=8,
        spacer_length=8,
    )
    rs2 = CrisprRepeatSpacer(
        position=200,
        repeat="ATCGATCG",
        repeat_length=8,
    )
    array = CrisprArray(repeats_and_spacers=[rs1, rs2])
    result = MincedSequenceResult(sequence_id="test_seq", crispr_arrays=[array])
    return MincedOutput(results=[result])


def test_export_csv(sample_output, tmp_path):
    sample_output.export(name="minced", export_path=str(tmp_path), file_format="csv")
    csv_path = tmp_path / "minced.csv"
    assert validate_export_output(csv_path)


def test_export_json(sample_output, tmp_path):
    sample_output.export(name="minced", export_path=str(tmp_path), file_format="json")
    json_path = tmp_path / "minced.json"
    assert validate_export_output(json_path)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_run_minced_with_crispr_sequence():
    """Run MinCED on a sequence with known CRISPR arrays."""
    inputs = MincedInput(sequences=[_CRISPR_SEQUENCE])
    config = MincedConfig(min_num_repeats=3, min_repeat_length=23)
    result = run_minced(inputs, config)

    assert isinstance(result, MincedOutput)
    assert len(result.results) == 1


@pytest.mark.integration
def test_run_minced_no_crispr():
    """Run MinCED on a sequence without CRISPR arrays."""
    random_seq = "ATCGATCG" * 500
    inputs = MincedInput(sequences=[random_seq])
    config = MincedConfig()
    result = run_minced(inputs, config)

    assert isinstance(result, MincedOutput)
    assert len(result.results) == 1
    assert result.num_sequences_with_crispr == 0


# A 25-bp variant of the E. coli K-12 CRISPR-2 repeat (originally 29 bp,
# trimmed to 25). Falls between the old wrapper default (27) and the upstream
# default (23). MinCED detects it with `-minRL 23` but not with `-minRL 27`.
_SHORT_REPEAT_E_COLI = "CGGTTTATCCCCGCTGGCGCGGGGA"  # 25 bp
_SHORT_REPEAT_SPACERS = [
    "GTGGAATTCGCAGCCATGAACGCCATTGTC",
    "TTGCAATGCTGAATAATTGCAATCGCAGAA",
    "AAATCGCATCATTGCAATGCTGAATAATTG",
    "CAATGCTGAATAATTGCAATCGCAGAACAT",
    "GAATAATTGCAATCGCAGAACATAATGCTA",
]
_SHORT_REPEAT_SEQUENCE = (
    "ATGAATTGTGCATTGGCATCATTAATTATTGTGCAATGCTAGAA"  # Leader
    + "".join(_SHORT_REPEAT_E_COLI + s for s in _SHORT_REPEAT_SPACERS)
    + _SHORT_REPEAT_E_COLI  # Final repeat
    + "CACATCAATAATCGTGAGCGTAATTGCAA"  # Trailer
)


@pytest.mark.integration
def test_run_minced_default_finds_short_repeat_after_default_change():
    """Regression test for the 27→23 default change.

    Source: ``minced --help`` reports ``-minRL 23`` as upstream. Wrapper used to
    default to 27, silently missing CRISPR arrays with 23-26 bp repeats. This
    test pins the new default with a 25-bp E. coli-derived repeat: ``MincedConfig()``
    detects it; ``MincedConfig(min_repeat_length=27)`` (the old default) does not.
    """
    inputs = MincedInput(sequences=[_SHORT_REPEAT_SEQUENCE])
    new_default = run_minced(inputs, MincedConfig())
    old_default = run_minced(inputs, MincedConfig(min_repeat_length=27))
    assert new_default.success and old_default.success
    new_arrays = new_default.results[0].num_arrays
    old_arrays = old_default.results[0].num_arrays
    assert new_arrays > 0, "MincedConfig() must detect 25-bp CRISPR repeats with the new minRL=23 default"
    assert old_arrays == 0, (
        "MincedConfig(min_repeat_length=27) must NOT detect 25-bp repeats (proves the change matters)"
    )


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("minced-crispr")
@pytest.mark.slow
def test_minced_crispr_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark minced-crispr: 100 distinct CRISPR-bearing sequences with varied flanking regions (cold + warm)."""
    sequences = [_CRISPR_SEQUENCE + f"AAAA{i:08d}AAAA" for i in range(100)]
    inputs = MincedInput(sequences=sequences)
    config = MincedConfig()

    result = benchmark_twice(request, "minced", lambda: run_minced(inputs, config))
    validate_output(result)

    assert result.tool_id == "minced-crispr"
    assert len(result.results) == 100

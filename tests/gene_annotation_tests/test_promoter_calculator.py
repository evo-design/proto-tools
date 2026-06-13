"""tests/gene_annotation_tests/test_promoter_calculator.py.

Tests for the Salis Lab Promoter Calculator tool.
"""

import pytest

from proto_tools.tools.gene_annotation.promoter_calculator import (
    PromoterCalculatorConfig,
    PromoterCalculatorInput,
    PromoterCalculatorOutput,
    PromoterCalculatorSequenceResult,
    PromoterPrediction,
    run_promoter_calculator,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
    validate_output,
)

_persistent_tool = make_persistent_fixture("promoter_calculator", gpu=False)

# Consensus E. coli sigma70 promoter padded with neutral context.
CONSENSUS_PROMOTER = "A" * 20 + "TTGACAATGATACTTAGATTCACTTATAATACTAGTAGGAGGAACTTTATGAAA" + "A" * 20


# ── Input validation ─────────────────────────────────────────────────────


def test_input_single_sequence_normalization():
    """Single string should be normalized to list."""
    inp = PromoterCalculatorInput(sequences="ATCG")
    assert isinstance(inp.sequences, list)
    assert len(inp.sequences) == 1
    assert inp.sequences[0] == "ATCG"


def test_input_list_of_sequences():
    inp = PromoterCalculatorInput(sequences=["ATCG", "GCTA"])
    assert len(inp.sequences) == 2


# ── Data model properties ────────────────────────────────────────────────


def _make_prediction(tss_name: str = "Fwd45", tss: int = 45) -> PromoterPrediction:
    return PromoterPrediction(
        tss_name=tss_name,
        tss=tss,
        strand="+",
        dG_total=-3.5,
        Tx_rate=8500.0,
        promoter_sequence="TTGACAATGATACTTAGATTCACTTATAATACTAGTAGGAGGAACTTTATGAAA",
        length=70,
        UP_position=[0, 10],
        hex35_position=[10, 16],
        spacer_position=[16, 33],
        hex10_position=[33, 39],
        disc_position=[39, 45],
    )


def test_sequence_result_has_promoter_with_predictions():
    pred = _make_prediction()
    result = PromoterCalculatorSequenceResult(sequence_id="test", predictions=[pred])
    assert result.has_promoter is True
    assert result.num_promoters == 1


def test_sequence_result_has_promoter_without():
    result = PromoterCalculatorSequenceResult(sequence_id="test", predictions=[])
    assert result.has_promoter is False
    assert result.num_promoters == 0


def test_num_sequences_with_promoter():
    pred = _make_prediction()

    r1 = PromoterCalculatorSequenceResult(sequence_id="seq1", predictions=[pred])
    r2 = PromoterCalculatorSequenceResult(sequence_id="seq2", predictions=[])
    r3 = PromoterCalculatorSequenceResult(sequence_id="seq3", predictions=[pred])

    output = PromoterCalculatorOutput(results=[r1, r2, r3])
    assert output.num_sequences_with_promoter == 2


def test_num_sequences_with_promoter_empty():
    output = PromoterCalculatorOutput(results=[])
    assert output.num_sequences_with_promoter == 0


# ── Export ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_output():
    pred1 = _make_prediction(tss_name="Fwd45", tss=45)
    pred2 = _make_prediction(tss_name="Rev120", tss=120)
    pred2 = pred2.model_copy(update={"strand": "-", "dG_total": -2.1, "Tx_rate": 4200.0})
    result = PromoterCalculatorSequenceResult(
        sequence_id="test_seq",
        predictions=[pred1, pred2],
    )
    return PromoterCalculatorOutput(results=[result])


def test_export_csv(sample_output, tmp_path):
    sample_output.export(name="promoter_calculator", export_path=str(tmp_path), file_format="csv")
    csv_path = tmp_path / "promoter_calculator.csv"
    assert validate_export_output(csv_path)


def test_export_json(sample_output, tmp_path):
    sample_output.export(name="promoter_calculator", export_path=str(tmp_path), file_format="json")
    json_path = tmp_path / "promoter_calculator.json"
    assert validate_export_output(json_path)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_run_promoter_calculator_with_consensus_promoter():
    """Run the calculator on a sigma70 consensus-style promoter."""
    inputs = PromoterCalculatorInput(sequences=[CONSENSUS_PROMOTER])
    result = run_promoter_calculator(inputs, PromoterCalculatorConfig())

    assert isinstance(result, PromoterCalculatorOutput)
    assert len(result.results) == 1
    assert result.results[0].sequence_id == "seq_0"


@pytest.mark.integration
def test_run_promoter_calculator_homopolymer():
    """A homopolymer should still return a structurally valid result."""
    inputs = PromoterCalculatorInput(sequences=["A" * 200])
    result = run_promoter_calculator(inputs, PromoterCalculatorConfig())

    assert isinstance(result, PromoterCalculatorOutput)
    assert len(result.results) == 1


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("promoter-calculator")
@pytest.mark.slow
def test_promoter_calculator_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark promoter-calculator: 100 distinct sigma70 consensus-bearing sequences (cold + warm)."""
    sequences = [CONSENSUS_PROMOTER + f"AAAA{i:08d}AAAA" for i in range(100)]
    inputs = PromoterCalculatorInput(sequences=sequences)

    result = benchmark_twice(
        request, "promoter_calculator", lambda: run_promoter_calculator(inputs, PromoterCalculatorConfig())
    )
    validate_output(result)

    assert result.tool_id == "promoter-calculator"
    assert len(result.results) == 100

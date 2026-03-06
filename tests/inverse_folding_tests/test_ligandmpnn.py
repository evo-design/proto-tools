"""Tests for LigandMPNN sampling."""
from pathlib import Path

import pytest

from bio_programming_tools.entities.structures.structure import Structure
from bio_programming_tools.tools.inverse_folding import (
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingStructureInput,
    run_ligandmpnn_sample,
)
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_CIF_FILE = Path(__file__).parent.parent / "dummy_data" / "renin.cif"
DEFAULT_CHECKPOINT = (
    Path.home() / ".foundry" / "checkpoints" / "ligandmpnn_v_32_010_25.pt"
)


_persistent_tool = make_persistent_fixture("ligandmpnn")


@pytest.fixture(scope="module")
def cif_structure():
    return Structure(structure_filepath_or_content=TEST_CIF_FILE)


@pytest.mark.include_in_env_report(category="inverse_folding")
@pytest.mark.uses_gpu
def test_ligandmpnn_sample_simple(cif_structure: Structure):
    """Basic LigandMPNN sampling with a single structure."""
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=cif_structure, chain_ids=["A"])
        ]
    )
    config = InverseFoldingConfig(num_sequences_per_structure=2, temperature=0.1, seed=42)

    output = run_ligandmpnn_sample(inp, config)

    validate_output(output)

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 2
    assert all(isinstance(sequence, str) for sequence in designed.sequences)
    assert all(len(seq) > 0 for seq in designed.sequences)
    assert len(designed.ligandmpnn_metrics) == 2
    assert all(isinstance(score, dict) for score in designed.ligandmpnn_metrics)


@pytest.mark.uses_gpu
def test_ligandmpnn_sample_chunked_batching(cif_structure: Structure):
    """Chunked batching produces the correct number of sequences."""
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=cif_structure, chain_ids=["A"])
        ]
    )
    config = InverseFoldingConfig(
        num_sequences_per_structure=6,
        batch_size=2,
        temperature=0.1,
        seed=42,
    )
    output = run_ligandmpnn_sample(inp, config)
    assert output.success, f"Chunked batching failed: {output}"

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 6
    assert all(isinstance(seq, str) for seq in designed.sequences)
    assert all(len(seq) > 0 for seq in designed.sequences)
    assert len(designed.ligandmpnn_metrics) == 6
    assert all(isinstance(m, dict) for m in designed.ligandmpnn_metrics)


@pytest.mark.uses_gpu
def test_ligandmpnn_sample_multiple_structures(cif_structure: Structure):
    """Sampling with multiple input structures."""
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=cif_structure, chain_ids=["A"]),
            InverseFoldingStructureInput(structure=cif_structure, chain_ids=["A"]),
        ]
    )
    config = InverseFoldingConfig(num_sequences_per_structure=3, temperature=0.1, seed=42)

    output = run_ligandmpnn_sample(inp, config)

    validate_output(output)

    assert len(output.designed_sequences) == 2
    for designed in output.designed_sequences:
        assert len(designed.sequences) == 3
        assert all(isinstance(sequence, str) for sequence in designed.sequences)

"""tests/inverse_folding_tests/test_if_schemas.py.

Tests for inverse folding shared data models.
"""

from pathlib import Path

import pytest

from proto_tools.entities.complex import Chain
from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedComplex,
    DesignSet,
    InverseFoldingInput,
    InverseFoldingOutput,
    InverseFoldingScoringMetrics,
    InverseFoldingScoringOutput,
    InverseFoldingStructureInput,
    SequenceStructurePair,
)

TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"


@pytest.fixture(scope="module")
def pdb_file_content() -> str:
    with open(TEST_PDB_FILE) as f:
        return f.read()


# ── Structure input validation ───────────────────────────────────────────


def test_structure_from_pdb_filepath():
    structure = InverseFoldingStructureInput(structure=TEST_PDB_FILE)
    assert isinstance(structure.structure, Structure)


def test_structure_from_pdb_content(pdb_file_content: str):
    structure = InverseFoldingStructureInput(structure=pdb_file_content)
    assert isinstance(structure.structure, Structure)


def test_structure_with_designed():
    structure = InverseFoldingStructureInput(structure=TEST_PDB_FILE, chains_to_redesign=["A"])
    assert structure.chains_to_redesign is not None
    assert structure.chains_to_redesign.chains == ["A"]


def test_structure_designed_defaults_to_none_resolved_chains_returns_all():
    """Designed defaults to None; chain_ids_to_redesign resolves to all structure chains."""
    structure = InverseFoldingStructureInput(structure=TEST_PDB_FILE)
    assert structure.chains_to_redesign is None
    expected_chains = structure.structure.get_chain_ids()
    assert structure.chain_ids_to_redesign == expected_chains


def test_structure_with_fixed():
    structure = InverseFoldingStructureInput(structure=TEST_PDB_FILE, fixed_positions={"A": [1, 2, 3]})
    assert structure.fixed_positions is not None
    assert structure.fixed_positions.chains == {"A": [1, 2, 3]}


def _offset_multichain_pdb() -> str:
    """Two chains whose residues are numbered off 1 with gaps: A = 27,28,29,35,36; B = 10,11,12."""

    def residue(chain: str, num: int, base_id: int) -> str:
        return "".join(
            f"ATOM  {base_id + k:>5}  {atom:<3} MET {chain}{num:>4}      "
            f"27.340  24.430   2.614  1.00  9.67           {atom[0]}\n"
            for k, atom in enumerate(["N", "CA", "C", "O"], start=1)
        )

    lines, base_id = "", 0
    for chain, nums in (("A", [27, 28, 29, 35, 36]), ("B", [10, 11, 12])):
        for num in nums:
            lines += residue(chain, num, base_id)
            base_id += 4
    return lines + "END\n"


def test_structure_with_fixed_positions_1indexed_on_offset_numbering():
    """1-indexed fixed positions across chains validate on offset/gapped numbering and resolve to residue numbers."""
    structure = InverseFoldingStructureInput(
        structure=_offset_multichain_pdb(),
        fixed_positions={"A": [1, 2, 4], "B": [1, 3]},
    )
    assert structure.fixed_positions is not None
    # Position i targets the i-th residue: A[4] skips the 30-34 gap to residue 35; B is offset at 10.
    assert structure.fixed_positions.to_residue_numbers(structure.structure) == {"A": [27, 28, 35], "B": [10, 12]}


def test_structure_rejects_fixed_position_past_chain_length():
    """A fixed position beyond the chain's residue count is rejected."""
    with pytest.raises(ValueError, match="invalid positions"):
        InverseFoldingStructureInput(structure=_offset_multichain_pdb(), fixed_positions={"A": [6]})


def test_structure_rejects_invalid_pdb_content():
    with pytest.raises(ValueError):
        InverseFoldingStructureInput(structure="not a pdb file")


def test_structure_rejects_missing_file():
    with pytest.raises((FileNotFoundError, ValueError)):
        InverseFoldingStructureInput(structure="/not/a/real/file.pdb")


# ── InverseFoldingInput ──────────────────────────────────────────────────


def test_input_from_pdb_filepath():
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=TEST_PDB_FILE),
            InverseFoldingStructureInput(structure=TEST_PDB_FILE),
            InverseFoldingStructureInput(structure=TEST_PDB_FILE),
        ]
    )
    assert len(inp.inputs) == 3
    assert all(isinstance(i.structure, Structure) for i in inp.inputs)


def test_input_from_pdb_content(pdb_file_content: str):
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=pdb_file_content),
            InverseFoldingStructureInput(structure=pdb_file_content),
            InverseFoldingStructureInput(structure=pdb_file_content),
        ]
    )
    assert len(inp.inputs) == 3
    assert all(isinstance(i.structure, Structure) for i in inp.inputs)


# ── DesignedComplex / DesignSet ──────────────────────────────────────────


class _MockDesignSet(DesignSet):
    """Concrete DesignSet over plain DesignedComplex instances for testing."""

    complexes: list[DesignedComplex]


def _mock_complex() -> DesignedComplex:
    return DesignedComplex(
        chains=[
            Chain(id="A", sequence="MVLSP"),
            Chain(id="B", sequence="GGGS"),
        ],
        designed=[True, False],
        metrics=InverseFoldingScoringMetrics(perplexity=1.5, log_likelihood=-3.2, avg_log_likelihood=-0.64),
    )


def test_design_set_routes_metrics():
    """``get_design_metrics(i)`` delegates to ``self.complexes[i].design_metrics()``."""
    design_set = _MockDesignSet(complexes=[_mock_complex()])
    assert design_set.get_design_metrics(0)["perplexity"] == 1.5


def test_designed_complex_helpers():
    """chain_sequences, chain_sequence_map, designed_chains, design_metrics."""
    complex_ = _mock_complex()
    assert complex_.chain_sequences == ["MVLSP", "GGGS"]
    assert complex_.chain_sequence_map() == {"A": "MVLSP", "B": "GGGS"}
    redesigned = complex_.designed_chains
    assert [c.id for c in redesigned] == ["A"]
    assert complex_.design_metrics()["perplexity"] == 1.5


# ── InverseFoldingOutput ─────────────────────────────────────────────────


def test_output_iter():
    design_set = _MockDesignSet(complexes=[_mock_complex()])
    output = InverseFoldingOutput(design_sets=[design_set])
    assert list(output) == [design_set]


# ── Validation: invalid chains_to_redesign and fixed_positions selections ───────────────────────


def test_structure_rejects_invalid_designed():
    with pytest.raises(ValueError, match="not in structure"):
        InverseFoldingStructureInput(structure=TEST_PDB_FILE, chains_to_redesign=["Z"])


def test_structure_rejects_invalid_fixed_chain():
    with pytest.raises(ValueError, match="not in structure"):
        InverseFoldingStructureInput(structure=TEST_PDB_FILE, fixed_positions={"Z": [1]})


# ── Export ──────────────────────────────────────────────────────────────────


def test_output_export_fasta(tmp_path):
    output = InverseFoldingOutput(design_sets=[_MockDesignSet(complexes=[_mock_complex()])])
    output.export("complexes", export_path=tmp_path, file_format="fasta")
    fasta_files = list((tmp_path / "complexes").glob("*.fasta"))
    assert len(fasta_files) == 1
    assert "MVLSP" in fasta_files[0].read_text()


def test_output_export_fasta_includes_ligand_smiles(tmp_path):
    """Ligand Fragments are emitted with a `_ligand_{id}_{ccd}` header and their canonical SMILES."""
    from proto_tools.entities.ligands import Fragment

    design = DesignedComplex(
        chains=[
            Chain(id="A", sequence="MVLSP"),
            Fragment(id="B", ccd_code="HEM"),
        ],
        designed=[True, False],
        metrics=InverseFoldingScoringMetrics(perplexity=1.5, log_likelihood=-3.2, avg_log_likelihood=-0.64),
    )
    output = InverseFoldingOutput(design_sets=[_MockDesignSet(complexes=[design])])
    output.export("complexes", export_path=tmp_path, file_format="fasta")
    text = (tmp_path / "complexes" / "input_0.fasta").read_text()
    assert ">input_0_design_0_chain_A\nMVLSP\n" in text
    assert ">input_0_design_0_ligand_B_HEM\n" in text
    # SMILES for HEM starts with C=CC1=C(C)... in RDKit's canonical form.
    assert "C=CC1=C(C)" in text


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_scoring_output_export(fmt, tmp_path):
    output = InverseFoldingScoringOutput(
        scores=[InverseFoldingScoringMetrics(perplexity=1.5, log_likelihood=-3.2)],
    )
    output.export("scores", export_path=tmp_path, file_format=fmt)
    assert (tmp_path / f"scores.{fmt}").stat().st_size > 0


# ── Validator error paths ────────────────────────────────────────────────────────


def test_structure_rejects_unsupported_type():
    with pytest.raises(ValueError):
        InverseFoldingStructureInput(structure=12345)


def test_structure_rejects_invalid_fixed_residues():
    with pytest.raises(ValueError, match="invalid positions"):
        InverseFoldingStructureInput(structure=TEST_PDB_FILE, fixed_positions={"A": [99999]})


# ── InverseFoldingScoringMetrics ─────────────────────────────────────────────────


def test_scoring_metrics_attribute_and_dict_access():
    score = InverseFoldingScoringMetrics(perplexity=1.5)
    assert score.perplexity == 1.5
    assert score["perplexity"] == 1.5
    # Set via mapping interface
    score["new_metric"] = 2.0
    assert score.new_metric == 2.0
    assert score["new_metric"] == 2.0


# ── Export edge cases ────────────────────────────────────────────────────────────


def test_output_export_json(tmp_path):
    output = InverseFoldingOutput(design_sets=[_MockDesignSet(complexes=[_mock_complex()])])
    output.export("complexes", export_path=tmp_path, file_format="json")
    json_files = list((tmp_path / "complexes").glob("*.json"))
    assert len(json_files) == 1


# ── JSON round-trip (HTTP transport contract) ─────────────────────────────────────


def test_proteinmpnn_sample_input_roundtrip():
    """Validator dict branch + constraint preservation through JSON round-trip."""
    from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_sample import ProteinMPNNSampleInput

    original = ProteinMPNNSampleInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=TEST_PDB_FILE,
                chains_to_redesign=["A"],
                fixed_positions={"A": [1, 2, 3]},
            )
        ]
    )
    restored = ProteinMPNNSampleInput(**original.model_dump(mode="json"))
    assert restored.inputs[0].structure.structure == original.inputs[0].structure.structure
    assert restored.inputs[0].chains_to_redesign is not None
    assert restored.inputs[0].chains_to_redesign.chains == ["A"]
    assert restored.inputs[0].fixed_positions is not None
    assert restored.inputs[0].fixed_positions.chains == {"A": [1, 2, 3]}


def test_sequence_structure_pair_roundtrip():
    """Native Pydantic coercion + fixed_positions survive a JSON round-trip."""
    original = SequenceStructurePair(
        sequence="MVLSP",
        structure=Structure.from_file(TEST_PDB_FILE),
        fixed_positions={"A": [1, 2, 3]},
    )
    restored = SequenceStructurePair(**original.model_dump(mode="json"))
    assert restored.sequence == "MVLSP"
    assert restored.structure.structure == original.structure.structure
    assert restored.fixed_positions is not None
    assert restored.fixed_positions.chains == {"A": [1, 2, 3]}


def test_sequence_structure_pair_rejects_invalid_fixed_chain():
    with pytest.raises(ValueError, match="not in structure"):
        SequenceStructurePair(
            sequence="MVLSP",
            structure=Structure.from_file(TEST_PDB_FILE),
            fixed_positions={"Z": [1]},
        )


def test_sequence_structure_pair_rejects_invalid_fixed_residues():
    with pytest.raises(ValueError, match="invalid positions"):
        SequenceStructurePair(
            sequence="MVLSP",
            structure=Structure.from_file(TEST_PDB_FILE),
            fixed_positions={"A": [99999]},
        )


def test_sequence_structure_pair_fixed_positions_distinguishes_cache_key():
    """Per-pair fixed_positions enters the per-item cache key, preventing collisions."""
    from proto_tools.utils.tool_cache import _serialize_for_cache_key

    structure = Structure.from_file(TEST_PDB_FILE)
    base = SequenceStructurePair(sequence="MVLSP", structure=structure)
    fixed = SequenceStructurePair(sequence="MVLSP", structure=structure, fixed_positions={"A": [1, 2, 3]})
    assert _serialize_for_cache_key(base) != _serialize_for_cache_key(fixed)


# ── Per-tool concrete output round-trip ───────────────────────────────────────


def _proteinmpnn_design():
    from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_sample import (
        ProteinMPNNDesign,
        ProteinMPNNDesignMetrics,
    )

    return ProteinMPNNDesign(
        chains=[
            Chain(id="A", sequence="MVLSP"),
            Chain(id="B", sequence="GGGS"),
        ],
        designed=[True, False],
        metrics=ProteinMPNNDesignMetrics(perplexity=1.5, sequence_recovery=0.42),
    )


def test_sample_output_polymorphism_roundtrip():
    """Concrete ProteinMPNNSampleOutput must reconstruct its subclasses on model_validate."""
    from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_sample import (
        ProteinMPNNDesign,
        ProteinMPNNDesignSet,
        ProteinMPNNSampleOutput,
    )

    original = ProteinMPNNSampleOutput(design_sets=[ProteinMPNNDesignSet(complexes=[_proteinmpnn_design()])])
    restored = ProteinMPNNSampleOutput.model_validate(original.model_dump(mode="json"))

    restored_set = restored.design_sets[0]
    assert isinstance(restored_set, ProteinMPNNDesignSet)
    restored_design = restored_set.complexes[0]
    assert isinstance(restored_design, ProteinMPNNDesign)
    assert restored_design.chain_sequences == ["MVLSP", "GGGS"]
    assert restored_design.metrics["perplexity"] == 1.5
    assert restored_design.metrics["sequence_recovery"] == 0.42


def test_metrics_round_trip_preserves_values():
    """ProteinMPNNDesign.model_validate(model_dump()) preserves metric values."""
    from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_sample import ProteinMPNNDesign

    design = _proteinmpnn_design()
    restored = ProteinMPNNDesign.model_validate(design.model_dump())
    assert restored.metrics["perplexity"] == 1.5
    assert restored.metrics.primary_value == 1.5


# ── Bridge to structure prediction ────────────────────────────────────────────


def test_designed_complex_feeds_structure_predictor():
    """LSP: ``DesignedComplex`` is-a ``Complex``, so SP tool inputs accept it directly."""
    from proto_tools.tools.structure_prediction import ESMFoldInput

    designed = _mock_complex()
    inp = ESMFoldInput(complexes=[designed])
    assert inp.complexes[0] is designed
    assert inp.complexes[0].chain_sequences == designed.chain_sequences


def test_complex_accepts_mixed_chain_list():
    """``Complex`` chains list still accepts mixed ``Chain``/string entries via the lenient validator."""
    from proto_tools.entities.complex import Complex

    mixed = Complex(chains=[Chain(id="A", sequence="MVLSP"), "GGGG"])
    assert [c.sequence for c in mixed.chains] == ["MVLSP", "GGGG"]

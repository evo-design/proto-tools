"""tests/structure_prediction_tests/test_shared_data_models.py.

Tests for structure prediction shared data models: MSA preprocessing
orchestration, the unwrap helper, and synthesized paired-MSA headers.
"""

import re
from typing import ClassVar

import pytest

from proto_tools.entities.msa import MSA
from proto_tools.tools.sequence_alignment.colabfold_search import colabfold_search as cfs
from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import ColabfoldSearchConfig
from proto_tools.tools.structure_prediction import shared_data_models as sdm
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    ChainModification,
    Complex,
    ComplexMSAs,
    Fragment,
    StructurePredictionInput,
    _row_to_base36_5,
    count_structure_tokens,
    unwrap_complex_msas,
    write_paired_a3m_with_uniprot_headers,
)

# Distinct protein sequences (valid amino acids) for orchestration tests.
_SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
_SEQ_B = "MQTQNNSREKQAAALERLFLSCFLKDPVPKPLQE"
_SEQ_C = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF"

# AF3/Protenix species regex (kept in sync with upstream msa_features/msa_utils).
_UNIPROT_SPECIES_RE = re.compile(r"(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|[A-Z0-9]{1,10}_(?P<sp>[A-Z0-9]{1,5})")


class _SPInput(StructurePredictionInput):
    """Minimal concrete input for exercising shared preprocessing."""

    SUPPORTED_ENTITY_TYPES: ClassVar[set] = {"protein"}
    ALLOWS_CHAIN_MODIFICATIONS: ClassVar[bool] = False


def _protein_complex(*sequences):
    return Complex(chains=[Chain(sequence=s, entity_type="protein") for s in sequences])


def _stub_colabfold_search(monkeypatch):
    """Replace run_colabfold_search with a stub echoing 3-row MSAs; capture submitted queries."""
    captured = {}

    def fake(colabfold_input, config):
        captured["queries"] = list(colabfold_input.queries)
        results = []
        for q in colabfold_input.queries:
            # One 3-row MSA per chain (equal depth → row-aligned for paired groups).
            chain_msas = [MSA(aligned_sequences=[s, s, s]) for s in q.sequences]
            results.append(cfs.ColabfoldSearchResult(query_sequences=q.sequences, msas=chain_msas))
        return cfs.ColabfoldSearchOutput(results=results)

    monkeypatch.setattr(cfs, "run_colabfold_search", fake)
    return captured


# ── unwrap_complex_msas ───────────────────────────────────────────────────────


def test_unwrap_complex_msas_none():
    assert unwrap_complex_msas(None) == ({}, False)


def test_unwrap_complex_msas_unpaired():
    msa = MSA(aligned_sequences=["AA", "AA"])
    per_chain, is_paired = unwrap_complex_msas(ComplexMSAs(per_chain={0: msa}, paired=False))
    assert per_chain == {0: msa}
    assert is_paired is False


def test_unwrap_complex_msas_paired():
    msa = MSA(aligned_sequences=["AA", "AA"])
    per_chain, is_paired = unwrap_complex_msas(ComplexMSAs(per_chain={0: msa, 1: msa}, paired=True))
    assert set(per_chain) == {0, 1}
    assert is_paired is True


# ── Synthesized paired-MSA headers ────────────────────────────────────────────


def test_row_to_base36_5_padding_and_range():
    assert _row_to_base36_5(0) == "00000"
    assert _row_to_base36_5(1) == "00001"
    assert _row_to_base36_5(35) == "0000Z"
    assert _row_to_base36_5(36) == "00010"
    assert _row_to_base36_5(36**5 - 1) == "ZZZZZ"


def test_write_paired_a3m_headers_match_species_regex(tmp_path):
    """Non-query rows get distinct, regex-matching species tokens by row index; query is inert."""
    msa = MSA(aligned_sequences=["MKTL", "MITL", "MKAL", "MQTL"])
    path = tmp_path / "chain.paired.a3m"
    write_paired_a3m_with_uniprot_headers(msa, str(path))

    lines = path.read_text().splitlines()
    assert lines[0] == ">query"
    assert lines[1] == "MKTL"
    species = []
    for i in range(2, len(lines), 2):
        # Predictors match the FASTA description (header with the leading '>' stripped).
        match = _UNIPROT_SPECIES_RE.match(lines[i].removeprefix(">"))
        assert match is not None, f"header does not match species regex: {lines[i]}"
        species.append(match.group("sp"))
    assert species == ["00001", "00002", "00003"]


def test_write_paired_a3m_species_tokens_align_across_chains(tmp_path):
    """Row N gets the same species token in every chain, so predictors pair row N across chains."""
    chain_a = MSA(aligned_sequences=["MK", "MI", "ML"])
    chain_b = MSA(aligned_sequences=["AC", "AS", "AT"])
    path_a, path_b = tmp_path / "a.a3m", tmp_path / "b.a3m"
    write_paired_a3m_with_uniprot_headers(chain_a, str(path_a))
    write_paired_a3m_with_uniprot_headers(chain_b, str(path_b))

    def species(path):
        lines = path.read_text().splitlines()
        return [_UNIPROT_SPECIES_RE.match(lines[i].removeprefix(">")).group("sp") for i in range(2, len(lines), 2)]

    assert species(path_a) == species(path_b)


# ── MSA preprocessing orchestration ───────────────────────────────────────────


def test_preprocess_single_chain_is_unpaired(monkeypatch):
    """A single protein chain issues one unpaired query and yields a plain per-chain dict."""
    captured = _stub_colabfold_search(monkeypatch)
    inputs = _SPInput(complexes=[_protein_complex(_SEQ_A)])

    out = sdm._preprocess_structure_prediction_msas(inputs, ColabfoldSearchConfig(), verbose=0)

    assert len(captured["queries"]) == 1
    assert not captured["queries"][0].is_paired
    entry = out.msas[0]
    assert isinstance(entry, ComplexMSAs)
    assert not entry.paired
    assert set(entry.per_chain) == {0}


def test_preprocess_homomultimer_broadcasts_single_msa(monkeypatch):
    """A homodimer issues one unpaired query (deduped) and broadcasts the MSA to both chains."""
    captured = _stub_colabfold_search(monkeypatch)
    inputs = _SPInput(complexes=[_protein_complex(_SEQ_A, _SEQ_A)])

    out = sdm._preprocess_structure_prediction_msas(inputs, ColabfoldSearchConfig(), verbose=0)

    assert len(captured["queries"]) == 1
    assert not captured["queries"][0].is_paired
    entry = out.msas[0]
    assert not entry.paired
    assert set(entry.per_chain) == {0, 1}
    assert entry.per_chain[0].aligned_sequences == entry.per_chain[1].aligned_sequences


def test_preprocess_heterocomplex_is_paired(monkeypatch):
    """A heterodimer issues one paired query and yields a paired ComplexMSAs over both chains."""
    captured = _stub_colabfold_search(monkeypatch)
    inputs = _SPInput(complexes=[_protein_complex(_SEQ_A, _SEQ_B)])

    out = sdm._preprocess_structure_prediction_msas(inputs, ColabfoldSearchConfig(), verbose=0)

    paired = [q for q in captured["queries"] if q.is_paired]
    assert len(paired) == 1
    assert len(paired[0].sequences) == 2
    entry = out.msas[0]
    assert entry.paired
    assert set(entry.per_chain) == {0, 1}
    assert entry.as_paired_msa().row_count == 3


def test_preprocess_two_heterocomplexes_not_deduped_across_complexes(monkeypatch):
    """Two heterocomplexes sharing a chain sequence still submit two separate paired groups."""
    captured = _stub_colabfold_search(monkeypatch)
    inputs = _SPInput(complexes=[_protein_complex(_SEQ_A, _SEQ_B), _protein_complex(_SEQ_A, _SEQ_C)])

    out = sdm._preprocess_structure_prediction_msas(inputs, ColabfoldSearchConfig(), verbose=0)

    paired = [q for q in captured["queries"] if q.is_paired]
    assert len(paired) == 2
    assert len(out.msas) == 2
    assert all(entry.paired for entry in out.msas)


def test_preprocess_skips_search_when_msas_presupplied(monkeypatch):
    """Pre-supplied msas short-circuit preprocessing without calling ColabFold."""
    calls = {"n": 0}

    def fail(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("run_colabfold_search must not be called when msas are pre-supplied")

    monkeypatch.setattr(cfs, "run_colabfold_search", fail)

    msa = MSA(aligned_sequences=[_SEQ_A, _SEQ_A])
    inputs = _SPInput(complexes=[_protein_complex(_SEQ_A)], msas=[{0: msa}])

    out = sdm._preprocess_structure_prediction_msas(inputs, ColabfoldSearchConfig(), verbose=0)

    assert calls["n"] == 0
    assert out.msas[0].per_chain[0].aligned_sequences == [_SEQ_A, _SEQ_A]


def test_preprocess_rejects_presupplied_length_mismatch(monkeypatch):
    """Pre-supplied msas must be parallel to complexes."""
    _stub_colabfold_search(monkeypatch)
    msa = MSA(aligned_sequences=[_SEQ_A, _SEQ_A])
    inputs = _SPInput(complexes=[_protein_complex(_SEQ_A), _protein_complex(_SEQ_B)], msas=[{0: msa}])

    with pytest.raises(ValueError, match="does not match"):
        sdm._preprocess_structure_prediction_msas(inputs, ColabfoldSearchConfig(), verbose=0)


# ── AF3-style token counting ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("chains", "expected"),
    [
        pytest.param([Chain(sequence="MKTL", entity_type="protein")], 4, id="protein_residues"),
        pytest.param(
            [
                Chain(sequence="MKTL", entity_type="protein"),
                Fragment(ccd_code="ATP"),  # 31 heavy atoms
                Fragment(ccd_code="MG"),  # 1 heavy atom
                Fragment(smiles="CCO"),  # ethanol: 3 heavy atoms
            ],
            4 + 31 + 1 + 3,
            id="ligand_heavy_atoms",
        ),
        pytest.param(
            [
                Chain(
                    sequence="MVLSPADKTN",
                    entity_type="protein",
                    modifications=[ChainModification(position=4, modification_code="SEP")],  # SEP: 11 heavy atoms
                )
            ],
            9 + 11,
            id="modified_residue",
        ),
    ],
)
def test_count_structure_tokens(chains, expected):
    """1 token per residue/nucleotide; heavy-atom count per ligand and modified residue."""
    assert count_structure_tokens(chains) == expected

"""tests/structure_prediction_tests/test_sp_inputs.py.

Tests for structure prediction shared input data models.
"""

from pathlib import Path
from typing import ClassVar

import pytest

from proto_tools.entities.ligands import Fragment, Ligands
from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_prediction.shared_data_models import (
    CHAIN_IDS,
    Chain,
    ChainModification,
    Complex,
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)

# ── Module-level test sequences ────────────────────────────────────────────────

_PROTEIN_SEQ = "MVLSPADKTN"
_PROTEIN_SEQ_SHORT = "MVLSPAD"
_PROTEIN_SEQ_TYPED = "MVLTPADKTN"  # used for modification compatibility tests ('T' at pos 5)
_DNA_SEQ = "ATCGATCG"
_RNA_SEQ = "AUGCAUGC"

# ── Concrete subclasses used as test fixtures ───────────────────────────────────
# These define SUPPORTED_ENTITY_TYPES and ALLOWS_CHAIN_MODIFICATIONS to exercise
# the base-class validators without coupling to any real tool implementation.


class _AllTypesInput(StructurePredictionInput):
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class _ProteinOnlyInput(StructurePredictionInput):
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class _ProteinDNAInput(StructurePredictionInput):
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class _NoModificationsInput(StructurePredictionInput):
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = False


# ── Complex ─────────────────────────────────────────────────


def test_complex_single_chain_sum_of_lengths():
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT])
    assert complex_obj.sum_of_chain_lengths() == len(_PROTEIN_SEQ_SHORT)


def test_complex_multi_chain_sum_of_lengths():
    chains = [_PROTEIN_SEQ_SHORT, "GSSGSSG"]
    complex_obj = Complex(chains=chains)
    assert complex_obj.sum_of_chain_lengths() == sum(len(c) for c in chains)


def test_complex_rejects_empty_chain_string():
    with pytest.raises(ValueError, match="cannot be empty"):
        Complex(chains=[_PROTEIN_SEQ_SHORT, ""])


def test_complex_rejects_nonstring_chain():
    with pytest.raises(ValueError, match="must be a string, dict, Chain, Fragment, or Ligands"):
        Complex(chains=[_PROTEIN_SEQ_SHORT, 123])


# ── StructurePredictionInput normalisation ─────────────────────────────────────


@pytest.mark.parametrize(
    "complexes,expected_chain_sequences",
    [
        (
            [
                Complex(chains=[_PROTEIN_SEQ_SHORT]),
                Complex(chains=["GSSGSSG"]),
            ],
            [[_PROTEIN_SEQ_SHORT], ["GSSGSSG"]],
        ),
        (
            [_PROTEIN_SEQ_SHORT, "GSSGSSG"],
            [[_PROTEIN_SEQ_SHORT], ["GSSGSSG"]],
        ),
        (
            [[_PROTEIN_SEQ_SHORT], ["GSSGSSG", _DNA_SEQ]],
            [[_PROTEIN_SEQ_SHORT], ["GSSGSSG", _DNA_SEQ]],
        ),
    ],
)
def test_input_normalises_complexes(complexes, expected_chain_sequences):
    input_obj = _AllTypesInput(complexes=complexes)
    assert len(input_obj.complexes) == len(expected_chain_sequences)
    for comp, expected_seqs in zip(input_obj.complexes, expected_chain_sequences, strict=False):
        assert comp.chain_sequences == expected_seqs


def test_input_rejects_none():
    with pytest.raises(ValueError, match="cannot be None"):
        _AllTypesInput(complexes=None)


def test_input_rejects_invalid_element_format():
    with pytest.raises(ValueError, match="Unsupported input format"):
        _AllTypesInput(complexes=[123, _PROTEIN_SEQ_SHORT])


# ── validate_supported_types ───────────────────────────────────────────────────


def test_all_supported_types_accepted():
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, _DNA_SEQ, _RNA_SEQ])
    input_obj = _AllTypesInput(complexes=[complex_obj])
    assert len(input_obj.complexes) == 1


@pytest.mark.parametrize(
    "chain,expected_type",
    [
        (_DNA_SEQ, "dna"),
        (_RNA_SEQ, "rna"),
        ("CC(C)C", "ligand"),
    ],
)
def test_protein_only_input_rejects_non_protein(chain, expected_type):
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, chain])
    with pytest.raises(ValueError, match=f"unsupported entity types: {expected_type}"):
        _ProteinOnlyInput(complexes=[complex_obj])


def test_protein_only_input_accepts_protein():
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, "GSSGSSG"])
    input_obj = _ProteinOnlyInput(complexes=[complex_obj])
    assert len(input_obj.complexes) == 1


def test_protein_dna_input_rejects_rna():
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, _RNA_SEQ])
    with pytest.raises(ValueError, match="unsupported entity types: rna"):
        _ProteinDNAInput(complexes=[complex_obj])


def test_protein_dna_input_accepts_protein_and_dna():
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, _DNA_SEQ])
    input_obj = _ProteinDNAInput(complexes=[complex_obj])
    assert len(input_obj.complexes) == 1


def test_multiple_unsupported_types_all_listed_in_error():
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, _RNA_SEQ, "CC(C)C"])
    with pytest.raises(ValueError, match="ligand, rna"):
        _ProteinDNAInput(complexes=[complex_obj])


def test_unsupported_type_error_includes_complex_index():
    complex1 = Complex(chains=[_PROTEIN_SEQ_SHORT])
    complex2 = Complex(chains=[_RNA_SEQ])
    with pytest.raises(ValueError, match="Complex 1"):
        _ProteinOnlyInput(complexes=[complex1, complex2])


def test_unsupported_type_error_includes_input_class_name():
    complex_obj = Complex(chains=[_RNA_SEQ])
    with pytest.raises(ValueError, match="_ProteinOnlyInput only supports"):
        _ProteinOnlyInput(complexes=[complex_obj])


def test_unsupported_type_error_lists_supported_types():
    complex_obj = Complex(chains=[_RNA_SEQ])
    with pytest.raises(ValueError, match="only supports: protein"):
        _ProteinOnlyInput(complexes=[complex_obj])


def test_supported_type_validation_checks_all_complexes():
    complex1 = Complex(chains=[_PROTEIN_SEQ_SHORT])
    complex2 = Complex(chains=[_DNA_SEQ])
    complex3 = Complex(chains=[_PROTEIN_SEQ_SHORT])
    input_obj = _ProteinDNAInput(complexes=[complex1, complex2, complex3])
    assert len(input_obj.complexes) == 3


# ── ChainModification ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("position", [0, -1])
def test_chain_modification_rejects_non_positive_position(position):
    with pytest.raises(ValueError, match="Position must be 1-based"):
        ChainModification(position=position, modification_code="SEP")


def test_chain_modification_rejects_empty_code():
    with pytest.raises(ValueError, match="Invalid CCD code"):
        ChainModification(position=1, modification_code="")


def test_chain_modification_strips_whitespace_from_code():
    mod = ChainModification(position=1, modification_code="  SEP  ")
    assert mod.modification_code == "SEP"


# ── Chain ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sequence,expected_type",
    [
        ("ACDEFGHIKLMNPQRSTVWY", "protein"),
        (_DNA_SEQ, "dna"),
        (_RNA_SEQ, "rna"),
    ],
)
def test_chain_auto_infers_entity_type(sequence, expected_type):
    chain = Chain(sequence=sequence)
    assert chain.entity_type == expected_type


def test_chain_defaults_to_empty_modifications():
    chain = Chain(sequence=_PROTEIN_SEQ)
    assert len(chain.modifications) == 0


def test_chain_explicit_entity_type_is_preserved():
    chain = Chain(sequence=_PROTEIN_SEQ, entity_type="protein")
    assert chain.entity_type == "protein"


def test_chain_with_single_modification():
    mod = ChainModification(position=4, modification_code="SEP")
    chain = Chain(sequence=_PROTEIN_SEQ, modifications=[mod])
    assert len(chain.modifications) == 1
    assert chain.modifications[0].position == 4
    assert chain.modifications[0].modification_code == "SEP"


def test_chain_with_multiple_modifications():
    mods = [
        ChainModification(position=4, modification_code="SEP"),
        ChainModification(position=5, modification_code="HY3"),
        ChainModification(position=9, modification_code="TPO"),
    ]
    chain = Chain(sequence="MVLSPADKTNVKAAW", modifications=mods)
    assert len(chain.modifications) == 3
    assert chain.modifications[0].modification_code == "SEP"
    assert chain.modifications[1].modification_code == "HY3"
    assert chain.modifications[2].modification_code == "TPO"


def test_chain_rejects_modification_position_beyond_sequence():
    mod = ChainModification(position=20, modification_code="SEP")
    with pytest.raises(ValueError, match="exceeds sequence length"):
        Chain(sequence=_PROTEIN_SEQ_SHORT, modifications=[mod])


def test_chain_accepts_modification_at_last_position():
    mod = ChainModification(position=4, modification_code="SEP")
    chain = Chain(sequence=_PROTEIN_SEQ_SHORT, modifications=[mod])
    assert len(chain.modifications) == 1


@pytest.mark.parametrize("sequence", ["", "   "])
def test_chain_rejects_empty_or_whitespace_sequence(sequence):
    with pytest.raises(ValueError, match="cannot be empty"):
        Chain(sequence=sequence)


def test_chain_accepts_tuple_modifications():
    chain = Chain(sequence=_PROTEIN_SEQ, modifications=[(4, "SEP"), (9, "TPO")])
    assert len(chain.modifications) == 2
    assert chain.modifications[0].position == 4
    assert chain.modifications[0].modification_code == "SEP"
    assert chain.modifications[1].position == 9
    assert chain.modifications[1].modification_code == "TPO"


def test_chain_accepts_mixed_tuple_and_object_modifications():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        modifications=[
            (4, "SEP"),
            ChainModification(position=9, modification_code="TPO"),
        ],
    )
    assert len(chain.modifications) == 2
    assert chain.modifications[0].position == 4
    assert chain.modifications[1].modification_code == "TPO"


@pytest.mark.parametrize("bad_tuple", [(5,), (5, "SEP", "extra")])
def test_chain_rejects_malformed_modification_tuple(bad_tuple):
    with pytest.raises(ValueError, match="must have exactly 2 elements"):
        Chain(sequence=_PROTEIN_SEQ, modifications=[bad_tuple])


def test_chain_rejects_invalid_modification_type():
    with pytest.raises(ValueError, match="must be a ChainModification object, a dict, or a tuple"):
        Chain(sequence=_PROTEIN_SEQ, modifications=["invalid"])


# ── Complex with Chain objects ──────────────────────────────


def test_complex_with_chain_objects_preserves_entity_types():
    chain1 = Chain(sequence=_PROTEIN_SEQ, entity_type="protein")
    chain2 = Chain(sequence=_DNA_SEQ, entity_type="dna")
    complex_obj = Complex(chains=[chain1, chain2])
    assert complex_obj.entity_types == ["protein", "dna"]


def test_complex_with_mixed_strings_and_chain_objects():
    chain1 = Chain(sequence=_PROTEIN_SEQ, entity_type="protein")
    complex_obj = Complex(chains=[chain1, _DNA_SEQ])
    assert complex_obj.chains[0].sequence == _PROTEIN_SEQ
    assert complex_obj.chains[1].sequence == _DNA_SEQ
    assert complex_obj.entity_types == ["protein", "dna"]


def test_complex_preserves_chain_modifications():
    mod = ChainModification(position=4, modification_code="SEP")
    chain = Chain(sequence=_PROTEIN_SEQ, modifications=[mod])
    complex_obj = Complex(chains=[chain])
    assert complex_obj.chains[0].modifications[0].modification_code == "SEP"


def test_complex_with_multiple_modified_chains():
    chain1 = Chain(
        sequence=_PROTEIN_SEQ,
        entity_type="protein",
        modifications=[ChainModification(position=4, modification_code="SEP")],
    )
    chain2 = Chain(
        sequence=_RNA_SEQ,
        entity_type="rna",
        modifications=[ChainModification(position=3, modification_code="2MG")],
    )
    complex_obj = Complex(chains=[chain1, chain2])
    assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
    assert complex_obj.chains[1].modifications[0].modification_code == "2MG"


def test_complex_chain_sequences_property():
    chain1 = Chain(sequence=_PROTEIN_SEQ)
    chain2 = Chain(sequence=_DNA_SEQ)
    complex_obj = Complex(chains=[chain1, chain2])
    assert complex_obj.chain_sequences == [_PROTEIN_SEQ, _DNA_SEQ]


def test_extract_protein_chains_filters_by_entity_type():
    complex_obj = Complex(
        chains=[
            Chain(sequence=_PROTEIN_SEQ, entity_type="protein"),
            Chain(sequence=_DNA_SEQ, entity_type="dna"),
            Chain(sequence=_PROTEIN_SEQ_SHORT, entity_type="protein"),
        ]
    )
    seqs, chain_ids = complex_obj.extract_protein_chains()
    # Returns only proteins; chain IDs reflect original position (A=0, C=2)
    assert seqs == [_PROTEIN_SEQ, _PROTEIN_SEQ_SHORT]
    assert chain_ids == ["A", "C"]


def test_extract_protein_chains_empty_when_no_proteins():
    complex_obj = Complex(chains=[_DNA_SEQ, _RNA_SEQ])
    assert complex_obj.extract_protein_chains() == ([], [])


def test_complex_accepts_more_than_26_chains():
    """``Complex`` carries no chain-count cap; per-tool input validators are the right place to enforce limits."""
    chains = [Chain(sequence="M", entity_type="protein") for _ in range(27)]
    complex_obj = Complex(chains=chains)
    assert complex_obj.num_chains() == 27


def test_extract_protein_chains_at_26_chains():
    chains = [Chain(sequence="M", entity_type="protein") for _ in range(26)]
    complex_obj = Complex(chains=chains)
    seqs, chain_ids = complex_obj.extract_protein_chains()
    assert seqs == ["M"] * 26
    assert chain_ids == CHAIN_IDS


def test_extract_protein_chains_past_26_uses_two_letter_labels():
    """No-id chains past index 25 get spreadsheet-style labels (Z → AA → AB → ...) instead of crashing."""
    chains = [Chain(sequence="M", entity_type="protein") for _ in range(30)]
    complex_obj = Complex(chains=chains)
    _, chain_ids = complex_obj.extract_protein_chains()
    assert len(chain_ids) == 30
    assert chain_ids[25] == "Z"
    assert chain_ids[26] == "AA"
    assert chain_ids[27] == "AB"
    assert chain_ids[29] == "AD"


def test_chain_label_helper():
    """``chain_label`` returns spreadsheet-style positional IDs and rejects negative input."""
    from proto_tools.entities.complex import chain_label

    assert chain_label(0) == "A"
    assert chain_label(25) == "Z"
    assert chain_label(26) == "AA"
    assert chain_label(51) == "AZ"
    assert chain_label(52) == "BA"
    assert chain_label(701) == "ZZ"
    assert chain_label(702) == "AAA"
    with pytest.raises(ValueError, match="non-negative"):
        chain_label(-1)


def test_complex_accepts_dictionary_chain():
    complex_obj = Complex(chains=[{"sequence": _PROTEIN_SEQ, "entity_type": "protein"}])
    assert complex_obj.chains[0].sequence == _PROTEIN_SEQ
    assert complex_obj.chains[0].entity_type == "protein"


def test_complex_accepts_dictionary_chain_with_modifications():
    complex_obj = Complex(
        chains=[
            {
                "sequence": _PROTEIN_SEQ,
                "entity_type": "protein",
                "modifications": [(4, "SEP"), (9, "TPO")],
            }
        ]
    )
    assert len(complex_obj.chains[0].modifications) == 2
    assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
    assert complex_obj.chains[0].modifications[1].modification_code == "TPO"


def test_complex_accepts_mixed_string_dict_and_chain_formats():
    complex_obj = Complex(
        chains=[
            _PROTEIN_SEQ,
            {"sequence": _DNA_SEQ, "entity_type": "dna"},
            Chain(sequence=_RNA_SEQ, entity_type="rna"),
        ]
    )
    assert len(complex_obj.chains) == 3
    assert complex_obj.entity_types == ["protein", "dna", "rna"]


def test_complex_rejects_invalid_dictionary_chain():
    with pytest.raises(ValueError, match="Chain 0 is invalid"):
        Complex(chains=[{"invalid_field": _PROTEIN_SEQ}])


def test_complex_dictionary_chain_infers_entity_type_when_absent():
    complex_obj = Complex(chains=[{"sequence": _PROTEIN_SEQ}])
    assert complex_obj.chains[0].entity_type == "protein"
    assert len(complex_obj.chains[0].modifications) == 0


def test_complex_iteration_yields_chain_objects():
    chain1 = Chain(sequence=_PROTEIN_SEQ)
    chain2 = Chain(sequence=_DNA_SEQ)
    complex_obj = Complex(chains=[chain1, chain2])
    chains_list = list(complex_obj)
    assert all(isinstance(c, Chain) for c in chains_list)


def test_complex_indexing_returns_chain_object():
    chain1 = Chain(sequence=_PROTEIN_SEQ)
    chain2 = Chain(sequence=_DNA_SEQ)
    complex_obj = Complex(chains=[chain1, chain2])
    assert isinstance(complex_obj[0], Chain)
    assert complex_obj[0].sequence == _PROTEIN_SEQ
    assert complex_obj[1].sequence == _DNA_SEQ


# ── Chain modification helper methods ─────────────────────────────────────────


def test_chain_add_modification():
    chain = Chain(sequence=_PROTEIN_SEQ)
    chain.add_modification(4, "SEP")
    assert len(chain.modifications) == 1
    assert chain.modifications[0].position == 4
    assert chain.modifications[0].modification_code == "SEP"


def test_chain_add_multiple_modifications_sequentially():
    chain = Chain(sequence=_PROTEIN_SEQ)
    chain.add_modification(4, "SEP")
    chain.add_modification(9, "TPO")
    assert len(chain.modifications) == 2
    assert chain.modifications[0].modification_code == "SEP"
    assert chain.modifications[1].modification_code == "TPO"


def test_chain_add_modification_returns_self_for_chaining():
    chain = Chain(sequence=_PROTEIN_SEQ)
    result = chain.add_modification(4, "SEP").add_modification(9, "TPO")
    assert result is chain
    assert len(chain.modifications) == 2


def test_chain_add_modification_rejects_position_beyond_sequence():
    chain = Chain(sequence=_PROTEIN_SEQ_SHORT)
    with pytest.raises(ValueError, match="exceeds sequence length"):
        chain.add_modification(20, "SEP")


def test_chain_clear_modifications():
    chain = Chain(sequence=_PROTEIN_SEQ)
    chain.add_modification(4, "SEP")
    chain.add_modification(9, "TPO")
    chain.clear_modifications()
    assert len(chain.modifications) == 0


def test_chain_clear_modifications_returns_self_for_chaining():
    chain = Chain(sequence=_PROTEIN_SEQ)
    chain.add_modification(4, "SEP")
    result = chain.clear_modifications()
    assert result is chain
    assert len(chain.modifications) == 0


def test_complex_add_modification_to_specific_chain():
    complex_obj = Complex(chains=[_PROTEIN_SEQ, _DNA_SEQ])
    complex_obj.add_modification_to_chain(0, 4, "SEP")
    assert len(complex_obj.chains[0].modifications) == 1
    assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
    assert len(complex_obj.chains[1].modifications) == 0


def test_complex_add_modifications_to_multiple_chains():
    complex_obj = Complex(chains=[_PROTEIN_SEQ, _RNA_SEQ])
    complex_obj.add_modification_to_chain(0, 4, "SEP")
    complex_obj.add_modification_to_chain(1, 3, "2MG")
    assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
    assert complex_obj.chains[1].modifications[0].modification_code == "2MG"


def test_complex_add_modification_to_chain_returns_self_for_chaining():
    complex_obj = Complex(chains=[_PROTEIN_SEQ])
    result = complex_obj.add_modification_to_chain(0, 4, "SEP").add_modification_to_chain(0, 9, "TPO")
    assert result is complex_obj
    assert len(complex_obj.chains[0].modifications) == 2


def test_complex_add_modification_rejects_out_of_bounds_chain_index():
    complex_obj = Complex(chains=[_PROTEIN_SEQ])
    with pytest.raises(IndexError, match="Chain index 5 out of bounds"):
        complex_obj.add_modification_to_chain(5, 5, "SEP")


def test_complex_add_modification_rejects_negative_chain_index():
    complex_obj = Complex(chains=[_PROTEIN_SEQ])
    with pytest.raises(IndexError, match="Chain index -1 out of bounds"):
        complex_obj.add_modification_to_chain(-1, 5, "SEP")


def test_complex_clear_all_modifications():
    complex_obj = Complex(chains=[_PROTEIN_SEQ, _DNA_SEQ])
    complex_obj.add_modification_to_chain(0, 4, "SEP")
    complex_obj.add_modification_to_chain(1, 1, "6MA")
    complex_obj.clear_all_modifications()
    assert len(complex_obj.chains[0].modifications) == 0
    assert len(complex_obj.chains[1].modifications) == 0


def test_complex_clear_all_modifications_returns_self_for_chaining():
    complex_obj = Complex(chains=[_PROTEIN_SEQ])
    complex_obj.add_modification_to_chain(0, 4, "SEP")
    result = complex_obj.clear_all_modifications()
    assert result is complex_obj
    assert len(complex_obj.chains[0].modifications) == 0


# ── ALLOWS_CHAIN_MODIFICATIONS validation ─────────────────────────────────────


def test_allows_modifications_input_accepts_modified_chains():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        modifications=[ChainModification(position=4, modification_code="SEP")],
    )
    complex_obj = Complex(chains=[chain])
    input_obj = _AllTypesInput(complexes=[complex_obj])
    assert len(input_obj.complexes[0].chains[0].modifications) == 1


def test_allows_modifications_input_accepts_unmodified_chains():
    complex_obj = Complex(chains=[_PROTEIN_SEQ])
    input_obj = _AllTypesInput(complexes=[complex_obj])
    assert len(input_obj.complexes[0].chains[0].modifications) == 0


def test_no_modifications_input_rejects_modified_chain():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        modifications=[ChainModification(position=4, modification_code="SEP")],
    )
    complex_obj = Complex(chains=[chain])
    with pytest.raises(ValueError, match="contains modifications"):
        _NoModificationsInput(complexes=[complex_obj])


def test_no_modifications_input_accepts_unmodified_chain():
    complex_obj = Complex(chains=[_PROTEIN_SEQ])
    input_obj = _NoModificationsInput(complexes=[complex_obj])
    assert len(input_obj.complexes) == 1


def test_no_modifications_input_rejects_single_modified_chain_in_multi_chain_complex():
    chain1 = Chain(sequence=_PROTEIN_SEQ)
    chain2 = Chain(
        sequence=_DNA_SEQ,
        modifications=[ChainModification(position=1, modification_code="6MA")],
    )
    complex_obj = Complex(chains=[chain1, chain2])
    with pytest.raises(ValueError, match="contains modifications"):
        _NoModificationsInput(complexes=[complex_obj])


def test_no_modifications_input_error_includes_complex_index():
    complex1 = Complex(chains=[_PROTEIN_SEQ])
    complex2 = Complex(
        chains=[
            Chain(
                sequence=_DNA_SEQ,
                modifications=[ChainModification(position=1, modification_code="6MA")],
            )
        ]
    )
    with pytest.raises(ValueError, match="Complex 1"):
        _NoModificationsInput(complexes=[complex1, complex2])


def test_no_modifications_input_error_includes_class_name():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        modifications=[ChainModification(position=4, modification_code="SEP")],
    )
    complex_obj = Complex(chains=[chain])
    with pytest.raises(ValueError, match="_NoModificationsInput does not allow"):
        _NoModificationsInput(complexes=[complex_obj])


def test_no_modifications_input_rejects_chain_with_multiple_modifications():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        modifications=[
            ChainModification(position=4, modification_code="SEP"),
            ChainModification(position=9, modification_code="TPO"),
        ],
    )
    complex_obj = Complex(chains=[chain])
    with pytest.raises(ValueError, match="contains modifications"):
        _NoModificationsInput(complexes=[complex_obj])


def test_no_modifications_input_accepts_string_sequences():
    input_obj = _NoModificationsInput(complexes=[_PROTEIN_SEQ, _DNA_SEQ])
    assert len(input_obj.complexes) == 2


def test_no_modifications_input_accepts_list_of_lists():
    input_obj = _NoModificationsInput(complexes=[[_PROTEIN_SEQ, _DNA_SEQ]])
    assert len(input_obj.complexes) == 1
    assert len(input_obj.complexes[0].chains) == 2


def test_no_modifications_input_validates_all_complexes():
    complex1 = Complex(chains=[_PROTEIN_SEQ])
    complex2 = Complex(chains=[_DNA_SEQ])
    complex3 = Complex(chains=[_RNA_SEQ])
    input_obj = _NoModificationsInput(complexes=[complex1, complex2, complex3])
    assert len(input_obj.complexes) == 3


def test_allows_modifications_input_accepts_dictionary_chain_with_modifications():
    complex_obj = Complex(
        chains=[
            {
                "sequence": _PROTEIN_SEQ,
                "entity_type": "protein",
                "modifications": [(4, "SEP")],
            }
        ]
    )
    input_obj = _AllTypesInput(complexes=[complex_obj])
    assert len(input_obj.complexes[0].chains[0].modifications) == 1


def test_no_modifications_input_rejects_dictionary_chain_with_modifications():
    complex_obj = Complex(
        chains=[
            {
                "sequence": _PROTEIN_SEQ,
                "entity_type": "protein",
                "modifications": [(4, "SEP")],
            }
        ]
    )
    with pytest.raises(ValueError, match="contains modifications"):
        _NoModificationsInput(complexes=[complex_obj])


# ── Modification compatibility validation ──────────────────────────────────────


@pytest.mark.parametrize(
    "sequence,entity_type,position,code",
    [
        (_PROTEIN_SEQ, "protein", 4, "SEP"),  # position 4 is 'S' (serine)
        (_RNA_SEQ, "rna", 3, "2MG"),  # position 3 is 'G' (guanosine)
        (_DNA_SEQ, "dna", 1, "1AP"),  # position 1 is 'A' (adenine)
    ],
)
def test_valid_modification_accepted(sequence, entity_type, position, code):
    chain = Chain(sequence=sequence, entity_type=entity_type, modifications=[(position, code)])
    assert chain.modifications[0].modification_code == code


def test_invalid_protein_modification_raises_with_position():
    # Position 5 is 'P' in "MVLTPADKTN", SEP is for serine ('S')
    with pytest.raises(ValueError, match="Invalid modification 'SEP' at position 5"):
        Chain(
            sequence=_PROTEIN_SEQ_TYPED,
            entity_type="protein",
            modifications=[(5, "SEP")],
        )


def test_invalid_modification_error_shows_expected_residue():
    with pytest.raises(ValueError, match="This modification is for residue or base 'S'"):
        Chain(
            sequence=_PROTEIN_SEQ_TYPED,
            entity_type="protein",
            modifications=[(5, "SEP")],
        )


def test_invalid_modification_error_shows_actual_residue():
    # Position 5 is 'P' in "MVLTPADKTN"
    with pytest.raises(ValueError, match="but position 5 contains 'P'"):
        Chain(
            sequence=_PROTEIN_SEQ_TYPED,
            entity_type="protein",
            modifications=[(5, "SEP")],
        )


def test_invalid_modification_error_lists_allowed_modifications_for_actual_residue():
    # Position 5 is 'P' — error should suggest HYP and other proline modifications
    with pytest.raises(ValueError, match=r"Allowed modifications for 'P' in protein: .*HYP"):
        Chain(
            sequence=_PROTEIN_SEQ_TYPED,
            entity_type="protein",
            modifications=[(5, "SEP")],
        )


def test_all_valid_modifications_pass_in_batch():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        entity_type="protein",
        modifications=[(4, "SEP"), (9, "TPO")],
    )
    assert len(chain.modifications) == 2


def test_one_invalid_modification_among_valid_ones_raises():
    with pytest.raises(ValueError, match="Invalid modification 'SEP' at position 10"):
        Chain(
            sequence=_PROTEIN_SEQ,
            entity_type="protein",
            modifications=[
                (4, "SEP"),  # Valid: position 4 is 'S'
                (10, "SEP"),  # Invalid: position 10 is 'N', not 'S'
            ],
        )


def test_rna_modification_on_wrong_base_raises():
    # 2MG is for guanosine (G), but position 1 is adenosine (A)
    with pytest.raises(ValueError, match="Invalid modification '2MG' at position 1"):
        Chain(sequence=_RNA_SEQ, entity_type="rna", modifications=[(1, "2MG")])


def test_rna_modification_rejected_on_dna_same_base():
    # 2MG is an RNA modification for G; should not work on DNA G
    with pytest.raises(ValueError, match="Invalid modification '2MG'"):
        Chain(
            sequence=_DNA_SEQ,
            entity_type="dna",
            modifications=[(7, "2MG")],  # position 7 is 'G' but this is DNA
        )


def test_add_modification_method_validates_compatibility():
    chain = Chain(sequence=_PROTEIN_SEQ, entity_type="protein")
    with pytest.raises(ValueError, match="Invalid modification 'TPO'"):
        chain.add_modification(4, "TPO")  # position 4 is 'S', TPO is for threonine


def test_complex_with_invalid_modification_raises():
    with pytest.raises(ValueError, match="Invalid modification 'SEP'"):
        Complex(
            chains=[
                {
                    "sequence": _PROTEIN_SEQ_TYPED,
                    "entity_type": "protein",
                    "modifications": [(5, "SEP")],  # position 5 is 'P', not 'S'
                }
            ]
        )


# ── Validator error paths ────────────────────────────────────────────────────────


def test_chain_modifications_must_be_list():
    with pytest.raises(ValueError, match="modifications must be a list"):
        Chain(sequence=_PROTEIN_SEQ, entity_type="protein", modifications="not a list")


def test_chain_modification_invalid_dict():
    with pytest.raises(ValueError, match="Modification dict at index 0 is invalid"):
        Chain(sequence=_PROTEIN_SEQ, entity_type="protein", modifications=[{"bad_key": "value"}])


def test_ligand_rejects_modifications():
    with pytest.raises(ValueError, match="Ligands cannot have modifications"):
        Chain(sequence="CCO", entity_type="ligand", modifications=[(1, "SEP")])


def test_chain_modification_zero_based_position():
    with pytest.raises(ValueError, match="1-based"):
        Chain(sequence=_PROTEIN_SEQ, entity_type="protein", modifications=[(0, "TPO")])


def test_chains_single_string_autowraps_to_one_chain():
    """``Complex(chains="MVLS...")`` is shorthand for a one-chain complex."""
    complex_obj = Complex(chains=_PROTEIN_SEQ_SHORT)
    assert complex_obj.num_chains() == 1
    assert isinstance(complex_obj.chains[0], Chain)
    assert complex_obj.chains[0].sequence == _PROTEIN_SEQ_SHORT
    assert complex_obj.chains[0].entity_type == "protein"


def test_chains_rejects_non_list_non_string():
    """Non-list, non-string ``chains`` values are rejected."""
    with pytest.raises(ValueError, match="chains must be a list"):
        Complex(chains=42)  # type: ignore[arg-type]


def test_input_rejects_unsupported_entity_types():
    with pytest.raises(ValueError, match="unsupported entity types"):
        _ProteinOnlyInput(complexes=[Complex(chains=[Chain(sequence=_DNA_SEQ, entity_type="dna")])])


def test_input_rejects_modifications_when_disallowed():
    chain = Chain(sequence=_PROTEIN_SEQ_TYPED, entity_type="protein", modifications=[(4, "TPO")])
    with pytest.raises(ValueError, match="does not allow chain modifications"):
        _NoModificationsInput(complexes=[Complex(chains=[chain])])


def test_input_get_entity_type_set():
    inp = _AllTypesInput(
        complexes=[
            Complex(chains=[Chain(sequence=_PROTEIN_SEQ, entity_type="protein")]),
            Complex(chains=[Chain(sequence=_DNA_SEQ, entity_type="dna")]),
        ]
    )
    assert inp.get_entity_type_set() == {"protein", "dna"}


@pytest.mark.parametrize(
    "attrs,match",
    [
        ({"ALLOWS_CHAIN_MODIFICATIONS": True}, "SUPPORTED_ENTITY_TYPES"),
        ({"SUPPORTED_ENTITY_TYPES": {"protein"}}, "ALLOWS_CHAIN_MODIFICATIONS"),
    ],
    ids=["missing-entity-types", "missing-allows-modifications"],
)
def test_init_subclass_missing_class_attrs(attrs, match):
    with pytest.raises(TypeError, match=match):
        type("_Bad", (StructurePredictionInput,), attrs)


def test_normalize_complexes_wraps_non_list():
    inp = _AllTypesInput(complexes=Complex(chains=[_PROTEIN_SEQ]))
    assert len(inp) == 1


# ── MSAStructurePredictionConfig ─────────────────────────────────────────────────


def test_msa_config_minimal():
    config = MSAStructurePredictionConfig.minimal()
    assert config.use_msa is False


# ── StructurePredictionOutput ────────────────────────────────────────────────────

_TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"


@pytest.mark.parametrize("fmt", ["pdb", "cif"], ids=["pdb", "cif"])
def test_output_export(fmt, tmp_path):
    struct = Structure.from_file(_TEST_PDB_FILE)
    output = StructurePredictionOutput.model_construct(structures=[struct], success=True)
    output._export_output(tmp_path / "out", fmt)
    assert (tmp_path / "out" / f"structure_0.{fmt}").exists()


def test_output_export_invalid_format(tmp_path):
    struct = Structure.from_file(_TEST_PDB_FILE)
    output = StructurePredictionOutput.model_construct(structures=[struct], success=True)
    with pytest.raises(ValueError, match="Invalid file format"):
        output._export_output(tmp_path / "out", "xyz")


# ── Fragment-as-chain integration ──────────────────────────────────────────────


def test_complex_accepts_fragment_directly():
    """A Fragment passed in chains list is used as-is (no Chain wrapping)."""
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, Fragment(ccd_code="ATP")])
    assert len(complex_obj.chains) == 2
    assert isinstance(complex_obj.chains[0], Chain)
    assert isinstance(complex_obj.chains[1], Fragment)
    assert complex_obj.chains[1].ccd_code == "ATP"
    assert complex_obj.chains[1].entity_type == "ligand"


def test_complex_expands_ligands_collection():
    """A Ligands collection in chains list expands to one Fragment per fragment."""
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, Ligands(ccd_codes=["ATP", "MG", "MG"])])
    assert len(complex_obj.chains) == 4
    assert all(isinstance(c, Fragment) for c in complex_obj.chains[1:])
    assert [f.ccd_code for f in complex_obj.chains[1:]] == ["ATP", "MG", "MG"]


def test_complex_auto_splits_multi_fragment_smiles_string():
    """A dot-separated SMILES string in chains list expands to N Fragments."""
    complex_obj = Complex(chains=["CCO.CO"])
    assert len(complex_obj.chains) == 2
    assert all(isinstance(c, Fragment) for c in complex_obj.chains)
    assert {f.smiles for f in complex_obj.chains} == {"CCO", "CO"}


def test_complex_converts_ligand_chain_to_fragment():
    """Chain(entity_type='ligand') is auto-converted to Fragment in normalize_chains."""
    chain = Chain(sequence="CCO", entity_type="ligand")
    complex_obj = Complex(chains=[chain])
    assert isinstance(complex_obj.chains[0], Fragment)
    assert complex_obj.chains[0].smiles == "CCO"
    assert complex_obj.chains[0].entity_type == "ligand"


def test_complex_dict_with_ccd_code_builds_fragment():
    """A dict with ccd_code (no sequence) builds a Fragment."""
    complex_obj = Complex(chains=[{"ccd_code": "ATP"}])
    assert isinstance(complex_obj.chains[0], Fragment)
    assert complex_obj.chains[0].ccd_code == "ATP"


def test_complex_dict_with_entity_type_ligand_builds_fragment():
    """A dict with entity_type='ligand' builds a Fragment, not a Chain."""
    complex_obj = Complex(chains=[{"sequence": "CCO", "entity_type": "ligand"}])
    assert isinstance(complex_obj.chains[0], Fragment)
    assert complex_obj.chains[0].smiles == "CCO"


def test_complex_dict_with_implicit_ligand_smiles_builds_fragment():
    """A dict with only a SMILES-shaped 'sequence' (no entity_type) auto-converts to Fragment.

    Mirrors the behavior of bare-string and Chain-object inputs: SMILES auto-detects
    to entity_type='ligand', and ligand-typed entries are normalized to Fragment.
    """
    complex_obj = Complex(chains=[{"sequence": "CCO"}])
    assert isinstance(complex_obj.chains[0], Fragment)
    assert complex_obj.chains[0].smiles == "CCO"
    assert complex_obj.chains[0].entity_type == "ligand"


def test_complex_entity_types_uniform_for_chain_and_fragment():
    """entity_types property returns 'ligand' for both Fragment and ligand-converted chains."""
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, Fragment(ccd_code="ATP"), "CC(C)C"])
    assert complex_obj.entity_types == ["protein", "ligand", "ligand"]


def test_complex_sum_of_chain_lengths_counts_fragments_as_one():
    """Each ligand Fragment contributes 1 to sum_of_chain_lengths."""
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, Fragment(ccd_code="ATP"), Fragment(ccd_code="MG")])
    assert complex_obj.sum_of_chain_lengths() == len(_PROTEIN_SEQ_SHORT) + 2


def test_complex_sum_of_chain_lengths_with_ligands_collection():
    """A Ligands collection expands to N Fragments and each contributes 1."""
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, Ligands(ccd_codes=["ATP", "MG"])])
    assert complex_obj.num_chains() == 3
    assert complex_obj.sum_of_chain_lengths() == len(_PROTEIN_SEQ_SHORT) + 2


def test_complex_chain_sequences_returns_smiles_for_fragments():
    """chain_sequences exposes SMILES for Fragments and sequence for Chains."""
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, Fragment(smiles="CCO")])
    assert complex_obj.chain_sequences == [_PROTEIN_SEQ_SHORT, "CCO"]


def test_complex_extract_protein_chains_skips_fragments():
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, Fragment(ccd_code="ATP"), "GSSGSSG"])
    seqs, chain_ids = complex_obj.extract_protein_chains()
    assert seqs == [_PROTEIN_SEQ_SHORT, "GSSGSSG"]
    assert chain_ids == ["A", "C"]


def test_complex_add_modification_to_fragment_chain_raises():
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, Fragment(ccd_code="ATP")])
    with pytest.raises(ValueError, match="ligand Fragment and cannot have modifications"):
        complex_obj.add_modification_to_chain(1, 1, "SEP")


def test_complex_round_trip_with_fragment():
    """Heterogeneous chains list survives model_dump → model_validate."""
    complex_obj = Complex(chains=[_PROTEIN_SEQ_SHORT, Fragment(ccd_code="ATP")])
    dumped = complex_obj.model_dump()
    reconstructed = Complex.model_validate(dumped)
    assert len(reconstructed.chains) == 2
    assert isinstance(reconstructed.chains[0], Chain)
    assert isinstance(reconstructed.chains[1], Fragment)
    assert reconstructed.chains[1].ccd_code == "ATP"

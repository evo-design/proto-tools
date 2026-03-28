"""tests/structure_prediction_tests/test_sp_inputs.py

Tests for structure prediction shared input data models."""
from __future__ import annotations

import pytest

from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    ChainModification,
    StructurePredictionComplex,
    StructurePredictionInput,
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
    SUPPORTED_ENTITY_TYPES = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class _ProteinOnlyInput(StructurePredictionInput):
    SUPPORTED_ENTITY_TYPES = {"protein"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class _ProteinDNAInput(StructurePredictionInput):
    SUPPORTED_ENTITY_TYPES = {"protein", "dna"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class _NoModificationsInput(StructurePredictionInput):
    SUPPORTED_ENTITY_TYPES = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = False


# ── StructurePredictionComplex ─────────────────────────────────────────────────


def test_complex_single_chain_sum_of_lengths():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT])
    assert complex_obj.sum_of_chain_lengths() == len(_PROTEIN_SEQ_SHORT)


def test_complex_multi_chain_sum_of_lengths():
    chains = [_PROTEIN_SEQ_SHORT, "GSSGSSG"]
    complex_obj = StructurePredictionComplex(chains=chains)
    assert complex_obj.sum_of_chain_lengths() == sum(len(c) for c in chains)


def test_complex_rejects_empty_chain_string():
    with pytest.raises(ValueError, match="cannot be empty"):
        StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT, ""])


def test_complex_rejects_nonstring_chain():
    with pytest.raises(ValueError, match="must be a string, dictionary, or Chain object"):
        StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT, 123])


# ── StructurePredictionInput normalisation ─────────────────────────────────────


@pytest.mark.parametrize("complexes,expected_chain_sequences", [
    (
        [StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT]),
         StructurePredictionComplex(chains=["GSSGSSG"])],
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
])
def test_input_normalises_complexes(complexes, expected_chain_sequences):
    input_obj = _AllTypesInput(complexes=complexes)
    assert len(input_obj.complexes) == len(expected_chain_sequences)
    for comp, expected_seqs in zip(input_obj.complexes, expected_chain_sequences):
        assert comp.chain_sequences == expected_seqs


def test_input_rejects_none():
    with pytest.raises(ValueError, match="cannot be None"):
        _AllTypesInput(complexes=None)


def test_input_rejects_invalid_element_format():
    with pytest.raises(ValueError, match="Unsupported input format"):
        _AllTypesInput(complexes=[123, _PROTEIN_SEQ_SHORT])


# ── validate_supported_types ───────────────────────────────────────────────────


def test_all_supported_types_accepted():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT, _DNA_SEQ, _RNA_SEQ])
    input_obj = _AllTypesInput(complexes=[complex_obj])
    assert len(input_obj.complexes) == 1


@pytest.mark.parametrize("chain,expected_type", [
    (_DNA_SEQ, "dna"),
    (_RNA_SEQ, "rna"),
    ("CC(C)C", "ligand"),
])
def test_protein_only_input_rejects_non_protein(chain, expected_type):
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT, chain])
    with pytest.raises(ValueError, match=f"unsupported entity types: {expected_type}"):
        _ProteinOnlyInput(complexes=[complex_obj])


def test_protein_only_input_accepts_protein():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT, "GSSGSSG"])
    input_obj = _ProteinOnlyInput(complexes=[complex_obj])
    assert len(input_obj.complexes) == 1


def test_protein_dna_input_rejects_rna():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT, _RNA_SEQ])
    with pytest.raises(ValueError, match="unsupported entity types: rna"):
        _ProteinDNAInput(complexes=[complex_obj])


def test_protein_dna_input_accepts_protein_and_dna():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT, _DNA_SEQ])
    input_obj = _ProteinDNAInput(complexes=[complex_obj])
    assert len(input_obj.complexes) == 1


def test_multiple_unsupported_types_all_listed_in_error():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT, _RNA_SEQ, "CC(C)C"])
    with pytest.raises(ValueError, match="ligand, rna"):
        _ProteinDNAInput(complexes=[complex_obj])


def test_unsupported_type_error_includes_complex_index():
    complex1 = StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT])
    complex2 = StructurePredictionComplex(chains=[_RNA_SEQ])
    with pytest.raises(ValueError, match="Complex 1"):
        _ProteinOnlyInput(complexes=[complex1, complex2])


def test_unsupported_type_error_includes_input_class_name():
    complex_obj = StructurePredictionComplex(chains=[_RNA_SEQ])
    with pytest.raises(ValueError, match="_ProteinOnlyInput only supports"):
        _ProteinOnlyInput(complexes=[complex_obj])


def test_unsupported_type_error_lists_supported_types():
    complex_obj = StructurePredictionComplex(chains=[_RNA_SEQ])
    with pytest.raises(ValueError, match="only supports: protein"):
        _ProteinOnlyInput(complexes=[complex_obj])


def test_supported_type_validation_checks_all_complexes():
    complex1 = StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT])
    complex2 = StructurePredictionComplex(chains=[_DNA_SEQ])
    complex3 = StructurePredictionComplex(chains=[_PROTEIN_SEQ_SHORT])
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


@pytest.mark.parametrize("sequence,expected_type", [
    ("ACDEFGHIKLMNPQRSTVWY", "protein"),
    (_DNA_SEQ, "dna"),
    (_RNA_SEQ, "rna"),
])
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
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        modifications=[(4, "SEP"), (9, "TPO")]
    )
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
            ChainModification(position=9, modification_code="TPO")
        ]
    )
    assert len(chain.modifications) == 2
    assert chain.modifications[0].position == 4
    assert chain.modifications[1].modification_code == "TPO"


@pytest.mark.parametrize("bad_tuple", [(5,), (5, "SEP", "extra")])
def test_chain_rejects_malformed_modification_tuple(bad_tuple):
    with pytest.raises(ValueError, match="must have exactly 2 elements"):
        Chain(sequence=_PROTEIN_SEQ, modifications=[bad_tuple])


def test_chain_rejects_invalid_modification_type():
    with pytest.raises(ValueError, match="must be a ChainModification object or a tuple"):
        Chain(sequence=_PROTEIN_SEQ, modifications=["invalid"])


# ── StructurePredictionComplex with Chain objects ──────────────────────────────


def test_complex_with_chain_objects_preserves_entity_types():
    chain1 = Chain(sequence=_PROTEIN_SEQ, entity_type="protein")
    chain2 = Chain(sequence=_DNA_SEQ, entity_type="dna")
    complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
    assert complex_obj.entity_types == ["protein", "dna"]


def test_complex_with_mixed_strings_and_chain_objects():
    chain1 = Chain(sequence=_PROTEIN_SEQ, entity_type="protein")
    complex_obj = StructurePredictionComplex(chains=[chain1, _DNA_SEQ])
    assert complex_obj.chains[0].sequence == _PROTEIN_SEQ
    assert complex_obj.chains[1].sequence == _DNA_SEQ
    assert complex_obj.entity_types == ["protein", "dna"]


def test_complex_preserves_chain_modifications():
    mod = ChainModification(position=4, modification_code="SEP")
    chain = Chain(sequence=_PROTEIN_SEQ, modifications=[mod])
    complex_obj = StructurePredictionComplex(chains=[chain])
    assert complex_obj.chains[0].modifications[0].modification_code == "SEP"


def test_complex_with_multiple_modified_chains():
    chain1 = Chain(
        sequence=_PROTEIN_SEQ,
        entity_type="protein",
        modifications=[ChainModification(position=4, modification_code="SEP")]
    )
    chain2 = Chain(
        sequence=_RNA_SEQ,
        entity_type="rna",
        modifications=[ChainModification(position=3, modification_code="2MG")]
    )
    complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
    assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
    assert complex_obj.chains[1].modifications[0].modification_code == "2MG"


def test_complex_chain_sequences_property():
    chain1 = Chain(sequence=_PROTEIN_SEQ)
    chain2 = Chain(sequence=_DNA_SEQ)
    complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
    assert complex_obj.chain_sequences == [_PROTEIN_SEQ, _DNA_SEQ]


def test_complex_accepts_dictionary_chain():
    complex_obj = StructurePredictionComplex(
        chains=[{"sequence": _PROTEIN_SEQ, "entity_type": "protein"}]
    )
    assert complex_obj.chains[0].sequence == _PROTEIN_SEQ
    assert complex_obj.chains[0].entity_type == "protein"


def test_complex_accepts_dictionary_chain_with_modifications():
    complex_obj = StructurePredictionComplex(
        chains=[
            {
                "sequence": _PROTEIN_SEQ,
                "entity_type": "protein",
                "modifications": [(4, "SEP"), (9, "TPO")]
            }
        ]
    )
    assert len(complex_obj.chains[0].modifications) == 2
    assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
    assert complex_obj.chains[0].modifications[1].modification_code == "TPO"


def test_complex_accepts_mixed_string_dict_and_chain_formats():
    complex_obj = StructurePredictionComplex(
        chains=[
            _PROTEIN_SEQ,
            {"sequence": _DNA_SEQ, "entity_type": "dna"},
            Chain(sequence=_RNA_SEQ, entity_type="rna"),
        ]
    )
    assert len(complex_obj.chains) == 3
    assert complex_obj.entity_types == ["protein", "dna", "rna"]


def test_complex_rejects_invalid_dictionary_chain():
    with pytest.raises(ValueError, match="dictionary is invalid"):
        StructurePredictionComplex(chains=[{"invalid_field": _PROTEIN_SEQ}])


def test_complex_dictionary_chain_infers_entity_type_when_absent():
    complex_obj = StructurePredictionComplex(chains=[{"sequence": _PROTEIN_SEQ}])
    assert complex_obj.chains[0].entity_type == "protein"
    assert len(complex_obj.chains[0].modifications) == 0


def test_complex_iteration_yields_chain_objects():
    chain1 = Chain(sequence=_PROTEIN_SEQ)
    chain2 = Chain(sequence=_DNA_SEQ)
    complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
    chains_list = list(complex_obj)
    assert all(isinstance(c, Chain) for c in chains_list)


def test_complex_indexing_returns_chain_object():
    chain1 = Chain(sequence=_PROTEIN_SEQ)
    chain2 = Chain(sequence=_DNA_SEQ)
    complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
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
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ, _DNA_SEQ])
    complex_obj.add_modification_to_chain(0, 4, "SEP")
    assert len(complex_obj.chains[0].modifications) == 1
    assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
    assert len(complex_obj.chains[1].modifications) == 0


def test_complex_add_modifications_to_multiple_chains():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ, _RNA_SEQ])
    complex_obj.add_modification_to_chain(0, 4, "SEP")
    complex_obj.add_modification_to_chain(1, 3, "2MG")
    assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
    assert complex_obj.chains[1].modifications[0].modification_code == "2MG"


def test_complex_add_modification_to_chain_returns_self_for_chaining():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ])
    result = complex_obj.add_modification_to_chain(0, 4, "SEP").add_modification_to_chain(0, 9, "TPO")
    assert result is complex_obj
    assert len(complex_obj.chains[0].modifications) == 2


def test_complex_add_modification_rejects_out_of_bounds_chain_index():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ])
    with pytest.raises(IndexError, match="Chain index 5 out of bounds"):
        complex_obj.add_modification_to_chain(5, 5, "SEP")


def test_complex_add_modification_rejects_negative_chain_index():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ])
    with pytest.raises(IndexError, match="Chain index -1 out of bounds"):
        complex_obj.add_modification_to_chain(-1, 5, "SEP")


def test_complex_clear_all_modifications():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ, _DNA_SEQ])
    complex_obj.add_modification_to_chain(0, 4, "SEP")
    complex_obj.add_modification_to_chain(1, 1, "6MA")
    complex_obj.clear_all_modifications()
    assert len(complex_obj.chains[0].modifications) == 0
    assert len(complex_obj.chains[1].modifications) == 0


def test_complex_clear_all_modifications_returns_self_for_chaining():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ])
    complex_obj.add_modification_to_chain(0, 4, "SEP")
    result = complex_obj.clear_all_modifications()
    assert result is complex_obj
    assert len(complex_obj.chains[0].modifications) == 0


# ── ALLOWS_CHAIN_MODIFICATIONS validation ─────────────────────────────────────


def test_allows_modifications_input_accepts_modified_chains():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        modifications=[ChainModification(position=4, modification_code="SEP")]
    )
    complex_obj = StructurePredictionComplex(chains=[chain])
    input_obj = _AllTypesInput(complexes=[complex_obj])
    assert len(input_obj.complexes[0].chains[0].modifications) == 1


def test_allows_modifications_input_accepts_unmodified_chains():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ])
    input_obj = _AllTypesInput(complexes=[complex_obj])
    assert len(input_obj.complexes[0].chains[0].modifications) == 0


def test_no_modifications_input_rejects_modified_chain():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        modifications=[ChainModification(position=4, modification_code="SEP")]
    )
    complex_obj = StructurePredictionComplex(chains=[chain])
    with pytest.raises(ValueError, match="contains modifications"):
        _NoModificationsInput(complexes=[complex_obj])


def test_no_modifications_input_accepts_unmodified_chain():
    complex_obj = StructurePredictionComplex(chains=[_PROTEIN_SEQ])
    input_obj = _NoModificationsInput(complexes=[complex_obj])
    assert len(input_obj.complexes) == 1


def test_no_modifications_input_rejects_single_modified_chain_in_multi_chain_complex():
    chain1 = Chain(sequence=_PROTEIN_SEQ)
    chain2 = Chain(
        sequence=_DNA_SEQ,
        modifications=[ChainModification(position=1, modification_code="6MA")]
    )
    complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
    with pytest.raises(ValueError, match="contains modifications"):
        _NoModificationsInput(complexes=[complex_obj])


def test_no_modifications_input_error_includes_complex_index():
    complex1 = StructurePredictionComplex(chains=[_PROTEIN_SEQ])
    complex2 = StructurePredictionComplex(
        chains=[
            Chain(
                sequence=_DNA_SEQ,
                modifications=[ChainModification(position=1, modification_code="6MA")]
            )
        ]
    )
    with pytest.raises(ValueError, match="Complex 1"):
        _NoModificationsInput(complexes=[complex1, complex2])


def test_no_modifications_input_error_includes_class_name():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        modifications=[ChainModification(position=4, modification_code="SEP")]
    )
    complex_obj = StructurePredictionComplex(chains=[chain])
    with pytest.raises(ValueError, match="_NoModificationsInput does not allow"):
        _NoModificationsInput(complexes=[complex_obj])


def test_no_modifications_input_rejects_chain_with_multiple_modifications():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        modifications=[
            ChainModification(position=4, modification_code="SEP"),
            ChainModification(position=9, modification_code="TPO")
        ]
    )
    complex_obj = StructurePredictionComplex(chains=[chain])
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
    complex1 = StructurePredictionComplex(chains=[_PROTEIN_SEQ])
    complex2 = StructurePredictionComplex(chains=[_DNA_SEQ])
    complex3 = StructurePredictionComplex(chains=[_RNA_SEQ])
    input_obj = _NoModificationsInput(complexes=[complex1, complex2, complex3])
    assert len(input_obj.complexes) == 3


def test_allows_modifications_input_accepts_dictionary_chain_with_modifications():
    complex_obj = StructurePredictionComplex(
        chains=[
            {
                "sequence": _PROTEIN_SEQ,
                "entity_type": "protein",
                "modifications": [(4, "SEP")]
            }
        ]
    )
    input_obj = _AllTypesInput(complexes=[complex_obj])
    assert len(input_obj.complexes[0].chains[0].modifications) == 1


def test_no_modifications_input_rejects_dictionary_chain_with_modifications():
    complex_obj = StructurePredictionComplex(
        chains=[
            {
                "sequence": _PROTEIN_SEQ,
                "entity_type": "protein",
                "modifications": [(4, "SEP")]
            }
        ]
    )
    with pytest.raises(ValueError, match="contains modifications"):
        _NoModificationsInput(complexes=[complex_obj])


# ── Modification compatibility validation ──────────────────────────────────────


@pytest.mark.parametrize("sequence,entity_type,position,code", [
    (_PROTEIN_SEQ, "protein", 4, "SEP"),   # position 4 is 'S' (serine)
    (_RNA_SEQ, "rna", 3, "2MG"),           # position 3 is 'G' (guanosine)
    (_DNA_SEQ, "dna", 1, "1AP"),           # position 1 is 'A' (adenine)
])
def test_valid_modification_accepted(sequence, entity_type, position, code):
    chain = Chain(sequence=sequence, entity_type=entity_type, modifications=[(position, code)])
    assert chain.modifications[0].modification_code == code


def test_invalid_protein_modification_raises_with_position():
    # Position 5 is 'P' in "MVLTPADKTN", SEP is for serine ('S')
    with pytest.raises(ValueError, match="Invalid modification 'SEP' at position 5"):
        Chain(
            sequence=_PROTEIN_SEQ_TYPED,
            entity_type="protein",
            modifications=[(5, "SEP")]
        )


def test_invalid_modification_error_shows_expected_residue():
    with pytest.raises(ValueError, match="This modification is for residue or base 'S'"):
        Chain(
            sequence=_PROTEIN_SEQ_TYPED,
            entity_type="protein",
            modifications=[(5, "SEP")]
        )


def test_invalid_modification_error_shows_actual_residue():
    # Position 5 is 'P' in "MVLTPADKTN"
    with pytest.raises(ValueError, match="but position 5 contains 'P'"):
        Chain(
            sequence=_PROTEIN_SEQ_TYPED,
            entity_type="protein",
            modifications=[(5, "SEP")]
        )


def test_invalid_modification_error_lists_allowed_modifications_for_actual_residue():
    # Position 5 is 'P' — error should suggest HYP and other proline modifications
    with pytest.raises(ValueError, match="Allowed modifications for 'P' in protein: .*HYP"):
        Chain(
            sequence=_PROTEIN_SEQ_TYPED,
            entity_type="protein",
            modifications=[(5, "SEP")]
        )


def test_all_valid_modifications_pass_in_batch():
    chain = Chain(
        sequence=_PROTEIN_SEQ,
        entity_type="protein",
        modifications=[(4, "SEP"), (9, "TPO")]
    )
    assert len(chain.modifications) == 2


def test_one_invalid_modification_among_valid_ones_raises():
    with pytest.raises(ValueError, match="Invalid modification 'SEP' at position 10"):
        Chain(
            sequence=_PROTEIN_SEQ,
            entity_type="protein",
            modifications=[
                (4, "SEP"),   # Valid: position 4 is 'S'
                (10, "SEP")   # Invalid: position 10 is 'N', not 'S'
            ]
        )


def test_rna_modification_on_wrong_base_raises():
    # 2MG is for guanosine (G), but position 1 is adenosine (A)
    with pytest.raises(ValueError, match="Invalid modification '2MG' at position 1"):
        Chain(
            sequence=_RNA_SEQ,
            entity_type="rna",
            modifications=[(1, "2MG")]
        )


def test_rna_modification_rejected_on_dna_same_base():
    # 2MG is an RNA modification for G; should not work on DNA G
    with pytest.raises(ValueError, match="Invalid modification '2MG'"):
        Chain(
            sequence=_DNA_SEQ,
            entity_type="dna",
            modifications=[(7, "2MG")]  # position 7 is 'G' but this is DNA
        )


def test_add_modification_method_validates_compatibility():
    chain = Chain(sequence=_PROTEIN_SEQ, entity_type="protein")
    with pytest.raises(ValueError, match="Invalid modification 'TPO'"):
        chain.add_modification(4, "TPO")  # position 4 is 'S', TPO is for threonine


def test_complex_with_invalid_modification_raises():
    with pytest.raises(ValueError, match="Invalid modification 'SEP'"):
        StructurePredictionComplex(
            chains=[
                {
                    "sequence": _PROTEIN_SEQ_TYPED,
                    "entity_type": "protein",
                    "modifications": [(5, "SEP")]  # position 5 is 'P', not 'S'
                }
            ]
        )

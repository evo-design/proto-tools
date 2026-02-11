"""
test_sp_inputs.py

Test the structure prediction input classes.
"""

import pytest

from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    ChainModification,
    StructurePredictionComplex,
    StructurePredictionInput,
)


class StructurePredictionInputTestClass(StructurePredictionInput):
    SUPPORTED_ENTITY_TYPES = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class ProteinOnlyInput(StructurePredictionInput):
    """Test input class that only supports protein entities."""
    SUPPORTED_ENTITY_TYPES = {"protein"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class ProteinDNAInput(StructurePredictionInput):
    """Test input class that only supports protein and DNA entities."""
    SUPPORTED_ENTITY_TYPES = {"protein", "dna"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class NoModificationsInput(StructurePredictionInput):
    """Test input class that does not allow modifications."""
    SUPPORTED_ENTITY_TYPES = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = False


class TestStructurePredictionComplex:
    def test_single_chain_complex(self):
        complex_obj = StructurePredictionComplex(chains=["MVLSPAD"])
        assert len(complex_obj.chains) == 1
        assert len(complex_obj.entity_types) == 1
        assert complex_obj.sum_of_chain_lengths() == len("MVLSPAD")

    def test_multi_chain_complex(self):
        chains = ["MVLSPAD", "GSSGSSG"]
        complex_obj = StructurePredictionComplex(chains=chains)
        assert len(complex_obj.chains) == 2
        assert len(complex_obj.entity_types) == 2
        assert complex_obj.sum_of_chain_lengths() == sum(len(c) for c in chains)

    def test_invalid_chain_empty_string(self):
        with pytest.raises(ValueError):
            StructurePredictionComplex(chains=["MVLSPAD", ""])

    def test_invalid_chain_nonstring(self):
        with pytest.raises(ValueError):
            StructurePredictionComplex(chains=["MVLSPAD", 123])


class TestStructurePredictionInput:

    def test_input_with_complex_instances(self):
        complex1 = StructurePredictionComplex(chains=["MVLSPAD"])
        complex2 = StructurePredictionComplex(chains=["GSSGSSG"])
        input_obj = StructurePredictionInputTestClass(complexes=[complex1, complex2])
        assert len(input_obj.complexes) == 2

    def test_input_with_list_of_strings(self):
        sequences = ["MVLSPAD", "GSSGSSG"]
        input_obj = StructurePredictionInputTestClass(complexes=sequences)
        assert len(input_obj.complexes) == 2
        assert input_obj.complexes[0].chains[0].sequence == "MVLSPAD"
        assert input_obj.complexes[1].chains[0].sequence == "GSSGSSG"

    def test_input_with_list_of_lists(self):
        sequences = [["MVLSPAD"], ["GSSGSSG", "ATCG"]]
        input_obj = StructurePredictionInputTestClass(complexes=sequences)
        assert len(input_obj.complexes) == 2
        assert input_obj.complexes[0].chain_sequences == ["MVLSPAD"]
        assert input_obj.complexes[1].chain_sequences == ["GSSGSSG", "ATCG"]

    def test_input_none_raises(self):
        with pytest.raises(ValueError):
            StructurePredictionInputTestClass(complexes=None)

    def test_input_invalid_format_raises(self):
        with pytest.raises(ValueError):
            StructurePredictionInputTestClass(complexes=[123, "MVLSPAD"])


class TestValidateSupportedTypes:
    """Test suite for the validate_supported_types validator."""

    def test_all_supported_types_pass(self):
        """Test that validation passes when all entity types are supported."""
        # Protein only
        complex1 = StructurePredictionComplex(
            chains=["MVLSPAD"],
        )
        input_obj = StructurePredictionInputTestClass(complexes=[complex1])
        assert len(input_obj.complexes) == 1

        # Multiple supported types
        complex2 = StructurePredictionComplex(
            chains=["MVLSPAD", "ATCG", "AUGC"],
        )
        input_obj = StructurePredictionInputTestClass(complexes=[complex2])
        assert len(input_obj.complexes) == 1

    def test_protein_only_rejects_dna(self):
        """Test that ProteinOnlyInput rejects DNA."""
        complex1 = StructurePredictionComplex(
            chains=["MVLSPAD", "ATCG"],
        )
        with pytest.raises(ValueError, match="unsupported entity types: dna"):
            ProteinOnlyInput(complexes=[complex1])

    def test_protein_only_rejects_rna(self):
        """Test that ProteinOnlyInput rejects RNA."""
        complex1 = StructurePredictionComplex(
            chains=["MVLSPAD", "AUGC"],
        )
        with pytest.raises(ValueError, match="unsupported entity types: rna"):
            ProteinOnlyInput(complexes=[complex1])

    def test_protein_only_rejects_ligand(self):
        """Test that ProteinOnlyInput rejects ligands."""
        complex1 = StructurePredictionComplex(
            chains=["MVLSPAD", "CC(C)C"],
        )
        with pytest.raises(ValueError, match="unsupported entity types: ligand"):
            ProteinOnlyInput(complexes=[complex1])

    def test_protein_only_accepts_protein(self):
        """Test that ProteinOnlyInput accepts protein-only complexes."""
        complex1 = StructurePredictionComplex(
            chains=["MVLSPAD", "GSSGSSG"],
        )
        input_obj = ProteinOnlyInput(complexes=[complex1])
        assert len(input_obj.complexes) == 1

    def test_protein_dna_rejects_rna(self):
        """Test that ProteinDNAInput rejects RNA."""
        complex1 = StructurePredictionComplex(
            chains=["MVLSPAD", "AUGC"],
        )
        with pytest.raises(ValueError, match="unsupported entity types: rna"):
            ProteinDNAInput(complexes=[complex1])

    def test_protein_dna_accepts_protein_and_dna(self):
        """Test that ProteinDNAInput accepts protein and DNA."""
        complex1 = StructurePredictionComplex(
            chains=["MVLSPAD", "ATCG"],
        )
        input_obj = ProteinDNAInput(complexes=[complex1])
        assert len(input_obj.complexes) == 1

    def test_multiple_unsupported_types_listed(self):
        """Test that multiple unsupported types are all listed in error."""
        complex1 = StructurePredictionComplex(
            chains=["MVLSPAD", "AUGC", "CC(C)C"],
        )
        with pytest.raises(ValueError, match="ligand, rna"):
            ProteinDNAInput(complexes=[complex1])

    def test_error_includes_complex_index(self):
        """Test that error message includes the complex index."""
        complex1 = StructurePredictionComplex(chains=["MVLSPAD"])
        complex2 = StructurePredictionComplex(chains=["AUGC"])
        with pytest.raises(ValueError, match="Complex 1"):
            ProteinOnlyInput(complexes=[complex1, complex2])

    def test_error_includes_tool_name(self):
        """Test that error message includes the input class name."""
        complex1 = StructurePredictionComplex(chains=["AUGC"])
        with pytest.raises(ValueError, match="ProteinOnlyInput only supports"):
            ProteinOnlyInput(complexes=[complex1])

    def test_error_lists_supported_types(self):
        """Test that error message lists all supported types."""
        complex1 = StructurePredictionComplex(chains=["AUGC"])
        with pytest.raises(ValueError, match="only supports: protein"):
            ProteinOnlyInput(complexes=[complex1])

    def test_validation_across_multiple_complexes(self):
        """Test that validation checks all complexes in the input."""
        complex1 = StructurePredictionComplex(
            chains=["MVLSPAD"],
        )
        complex2 = StructurePredictionComplex(
            chains=["ATCG"],
        )
        complex3 = StructurePredictionComplex(
            chains=["MVLSPAD"],
        )
        # All should pass for ProteinDNAInput
        input_obj = ProteinDNAInput(complexes=[complex1, complex2, complex3])
        assert len(input_obj.complexes) == 3


class TestChainModification:
    def test_create_modification_with_valid_data(self):
        mod = ChainModification(position=5, modification_code="SEP")
        assert mod.position == 5
        assert mod.modification_code == "SEP"

    def test_modification_position_must_be_positive(self):
        with pytest.raises(ValueError, match="Position must be 1-based"):
            ChainModification(position=0, modification_code="SEP")

    def test_modification_position_cannot_be_negative(self):
        with pytest.raises(ValueError, match="Position must be 1-based"):
            ChainModification(position=-1, modification_code="SEP")

    def test_modification_code_cannot_be_empty(self):
        with pytest.raises(ValueError, match="Value error, Invalid CCD code"):
            ChainModification(position=1, modification_code="")

    def test_modification_code_strips_whitespace(self):
        mod = ChainModification(position=1, modification_code="  SEP  ")
        assert mod.modification_code == "SEP"


class TestChain:
    def test_create_chain_with_sequence_only(self):
        chain = Chain(sequence="MVLSPADKTN")
        assert chain.sequence == "MVLSPADKTN"
        assert chain.entity_type == "protein"
        assert len(chain.modifications) == 0

    def test_chain_auto_infers_protein(self):
        chain = Chain(sequence="ACDEFGHIKLMNPQRSTVWY")
        assert chain.entity_type == "protein"

    def test_chain_auto_infers_dna(self):
        chain = Chain(sequence="ATCGATCG")
        assert chain.entity_type == "dna"

    def test_chain_auto_infers_rna(self):
        chain = Chain(sequence="AUGCAUGC")
        assert chain.entity_type == "rna"

    def test_chain_explicit_entity_type(self):
        chain = Chain(sequence="MVLSPADKTN", entity_type="protein")
        assert chain.entity_type == "protein"

    def test_chain_with_single_modification(self):
        mod = ChainModification(position=4, modification_code="SEP")
        chain = Chain(sequence="MVLSPADKTN", modifications=[mod])
        assert len(chain.modifications) == 1
        assert chain.modifications[0].position == 4
        assert chain.modifications[0].modification_code == "SEP"

    def test_chain_with_multiple_modifications(self):
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

    def test_modification_position_exceeds_sequence_length_raises(self):
        mod = ChainModification(position=20, modification_code="SEP")
        with pytest.raises(ValueError, match="exceeds sequence length"):
            Chain(sequence="MVLSPAD", modifications=[mod])

    def test_modification_at_last_position_is_valid(self):
        mod = ChainModification(position=4, modification_code="SEP")
        chain = Chain(sequence="MVLSPAD", modifications=[mod])
        assert len(chain.modifications) == 1

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Chain(sequence="")

    def test_whitespace_only_sequence_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Chain(sequence="   ")

    def test_chain_with_tuple_modifications(self):
        chain = Chain(
            sequence="MVLSPADKTN",
            modifications=[(4, "SEP"), (9, "TPO")]
        )
        assert len(chain.modifications) == 2
        assert chain.modifications[0].position == 4
        assert chain.modifications[0].modification_code == "SEP"
        assert chain.modifications[1].position == 9
        assert chain.modifications[1].modification_code == "TPO"

    def test_chain_with_mixed_tuple_and_object_modifications(self):
        chain = Chain(
            sequence="MVLSPADKTN",
            modifications=[
                (4, "SEP"),
                ChainModification(position=9, modification_code="TPO")
            ]
        )
        assert len(chain.modifications) == 2
        assert chain.modifications[0].position == 4
        assert chain.modifications[0].modification_code == "SEP"
        assert chain.modifications[1].position == 9
        assert chain.modifications[1].modification_code == "TPO"

    def test_invalid_tuple_length_raises(self):
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            Chain(sequence="MVLSPADKTN", modifications=[(5,)])

    def test_invalid_tuple_three_elements_raises(self):
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            Chain(sequence="MVLSPADKTN", modifications=[(5, "SEP", "extra")])

    def test_invalid_modification_type_raises(self):
        with pytest.raises(ValueError, match="must be a ChainModification object or a tuple"):
            Chain(sequence="MVLSPADKTN", modifications=["invalid"])


class TestComplexWithChainObjects:
    def test_complex_with_chain_objects(self):
        chain1 = Chain(sequence="MVLSPADKTN", entity_type="protein")
        chain2 = Chain(sequence="ATCGATCG", entity_type="dna")
        complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
        assert len(complex_obj.chains) == 2
        assert complex_obj.entity_types == ["protein", "dna"]

    def test_complex_with_mixed_strings_and_chain_objects(self):
        chain1 = Chain(sequence="MVLSPADKTN", entity_type="protein")
        complex_obj = StructurePredictionComplex(chains=[chain1, "ATCGATCG"])
        assert len(complex_obj.chains) == 2
        assert complex_obj.chains[0].sequence == "MVLSPADKTN"
        assert complex_obj.chains[1].sequence == "ATCGATCG"
        assert complex_obj.entity_types == ["protein", "dna"]

    def test_complex_with_chain_with_modifications(self):
        mod = ChainModification(position=4, modification_code="SEP")
        chain = Chain(sequence="MVLSPADKTN", modifications=[mod])
        complex_obj = StructurePredictionComplex(chains=[chain])
        assert len(complex_obj.chains) == 1
        assert len(complex_obj.chains[0].modifications) == 1
        assert complex_obj.chains[0].modifications[0].modification_code == "SEP"

    def test_complex_with_multiple_modified_chains(self):
        chain1 = Chain(
            sequence="MVLSPADKTN",
            entity_type="protein",
            modifications=[ChainModification(position=4, modification_code="SEP")]
        )
        chain2 = Chain(
            sequence="AUGCAUGC",
            entity_type="rna",
            modifications=[ChainModification(position=3, modification_code="2MG")]
        )
        complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
        assert len(complex_obj.chains) == 2
        assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
        assert complex_obj.chains[1].modifications[0].modification_code == "2MG"

    def test_chain_sequences_property(self):
        chain1 = Chain(sequence="MVLSPADKTN")
        chain2 = Chain(sequence="ATCGATCG")
        complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
        assert complex_obj.chain_sequences == ["MVLSPADKTN", "ATCGATCG"]

    def test_complex_with_dictionary_chain(self):
        complex_obj = StructurePredictionComplex(
            chains=[
                {"sequence": "MVLSPADKTN", "entity_type": "protein"}
            ]
        )
        assert len(complex_obj.chains) == 1
        assert complex_obj.chains[0].sequence == "MVLSPADKTN"
        assert complex_obj.chains[0].entity_type == "protein"

    def test_complex_with_dictionary_chain_with_modifications(self):
        complex_obj = StructurePredictionComplex(
            chains=[
                {
                    "sequence": "MVLSPADKTN",
                    "entity_type": "protein",
                    "modifications": [(4, "SEP"), (9, "TPO")]
                }
            ]
        )
        assert len(complex_obj.chains) == 1
        assert len(complex_obj.chains[0].modifications) == 2
        assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
        assert complex_obj.chains[0].modifications[1].modification_code == "TPO"

    def test_complex_with_mixed_formats(self):
        complex_obj = StructurePredictionComplex(
            chains=[
                "MVLSPADKTN",  # String
                {"sequence": "ATCGATCG", "entity_type": "dna"},  # Dictionary
                Chain(sequence="AUGCAUGC", entity_type="rna"),  # Chain object
            ]
        )
        assert len(complex_obj.chains) == 3
        assert complex_obj.chains[0].sequence == "MVLSPADKTN"
        assert complex_obj.chains[1].sequence == "ATCGATCG"
        assert complex_obj.chains[2].sequence == "AUGCAUGC"
        assert complex_obj.entity_types == ["protein", "dna", "rna"]

    def test_complex_with_invalid_dictionary_raises(self):
        with pytest.raises(ValueError, match="dictionary is invalid"):
            StructurePredictionComplex(
                chains=[
                    {"invalid_field": "MVLSPADKTN"}  # Missing required "sequence" field
                ]
            )

    def test_complex_with_dictionary_only_sequence(self):
        complex_obj = StructurePredictionComplex(
            chains=[
                {"sequence": "MVLSPADKTN"}  # Only sequence, should auto-infer entity_type
            ]
        )
        assert len(complex_obj.chains) == 1
        assert complex_obj.chains[0].sequence == "MVLSPADKTN"
        assert complex_obj.chains[0].entity_type == "protein"
        assert len(complex_obj.chains[0].modifications) == 0

    def test_complex_with_dictionary_no_modifications_field(self):
        complex_obj = StructurePredictionComplex(
            chains=[
                {"sequence": "MVLSPADKTN", "entity_type": "protein"}
            ]
        )
        assert len(complex_obj.chains) == 1
        assert complex_obj.chains[0].sequence == "MVLSPADKTN"
        assert complex_obj.chains[0].entity_type == "protein"
        assert len(complex_obj.chains[0].modifications) == 0

    def test_complex_iteration_returns_chain_objects(self):
        chain1 = Chain(sequence="MVLSPADKTN")
        chain2 = Chain(sequence="ATCGATCG")
        complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
        chains_list = list(complex_obj)
        assert len(chains_list) == 2
        assert isinstance(chains_list[0], Chain)
        assert isinstance(chains_list[1], Chain)

    def test_complex_indexing_returns_chain_objects(self):
        chain1 = Chain(sequence="MVLSPADKTN")
        chain2 = Chain(sequence="ATCGATCG")
        complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
        assert isinstance(complex_obj[0], Chain)
        assert complex_obj[0].sequence == "MVLSPADKTN"
        assert complex_obj[1].sequence == "ATCGATCG"


class TestModificationMethods:
    def test_chain_add_modification(self):
        chain = Chain(sequence="MVLSPADKTN")
        chain.add_modification(4, "SEP")
        assert len(chain.modifications) == 1
        assert chain.modifications[0].position == 4
        assert chain.modifications[0].modification_code == "SEP"

    def test_chain_add_multiple_modifications(self):
        chain = Chain(sequence="MVLSPADKTN")
        chain.add_modification(4, "SEP")
        chain.add_modification(9, "TPO")
        assert len(chain.modifications) == 2
        assert chain.modifications[0].modification_code == "SEP"
        assert chain.modifications[1].modification_code == "TPO"

    def test_chain_add_modification_method_chaining(self):
        chain = Chain(sequence="MVLSPADKTN")
        result = chain.add_modification(4, "SEP").add_modification(9, "TPO")
        assert result is chain
        assert len(chain.modifications) == 2

    def test_chain_add_modification_exceeds_length_raises(self):
        chain = Chain(sequence="MVLSPAD")
        with pytest.raises(ValueError, match="exceeds sequence length"):
            chain.add_modification(20, "SEP")

    def test_chain_clear_modifications(self):
        chain = Chain(sequence="MVLSPADKTN")
        chain.add_modification(4, "SEP")
        chain.add_modification(9, "TPO")
        chain.clear_modifications()
        assert len(chain.modifications) == 0

    def test_chain_clear_modifications_method_chaining(self):
        chain = Chain(sequence="MVLSPADKTN")
        chain.add_modification(4, "SEP")
        result = chain.clear_modifications()
        assert result is chain
        assert len(chain.modifications) == 0

    def test_complex_add_modification_to_chain(self):
        complex_obj = StructurePredictionComplex(chains=["MVLSPADKTN", "ATCGATCG"])
        complex_obj.add_modification_to_chain(0, 4, "SEP")
        assert len(complex_obj.chains[0].modifications) == 1
        assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
        assert len(complex_obj.chains[1].modifications) == 0

    def test_complex_add_modifications_to_multiple_chains(self):
        complex_obj = StructurePredictionComplex(chains=["MVLSPADKTN", "AUGCAUGC"])
        complex_obj.add_modification_to_chain(0, 4, "SEP")
        complex_obj.add_modification_to_chain(1, 3, "2MG")
        assert len(complex_obj.chains[0].modifications) == 1
        assert complex_obj.chains[0].modifications[0].modification_code == "SEP"
        assert len(complex_obj.chains[1].modifications) == 1
        assert complex_obj.chains[1].modifications[0].modification_code == "2MG"

    def test_complex_add_modification_method_chaining(self):
        complex_obj = StructurePredictionComplex(chains=["MVLSPADKTN"])
        result = complex_obj.add_modification_to_chain(0, 4, "SEP").add_modification_to_chain(0, 9, "TPO")
        assert result is complex_obj
        assert len(complex_obj.chains[0].modifications) == 2

    def test_complex_add_modification_invalid_chain_index_raises(self):
        complex_obj = StructurePredictionComplex(chains=["MVLSPADKTN"])
        with pytest.raises(IndexError, match="Chain index 5 out of bounds"):
            complex_obj.add_modification_to_chain(5, 5, "SEP")

    def test_complex_add_modification_negative_chain_index_raises(self):
        complex_obj = StructurePredictionComplex(chains=["MVLSPADKTN"])
        with pytest.raises(IndexError, match="Chain index -1 out of bounds"):
            complex_obj.add_modification_to_chain(-1, 5, "SEP")

    def test_complex_clear_all_modifications(self):
        complex_obj = StructurePredictionComplex(chains=["MVLSPADKTN", "ATCGATCG"])
        complex_obj.add_modification_to_chain(0, 4, "SEP")
        complex_obj.add_modification_to_chain(1, 1, "6MA")
        complex_obj.clear_all_modifications()
        assert len(complex_obj.chains[0].modifications) == 0
        assert len(complex_obj.chains[1].modifications) == 0

    def test_complex_clear_all_modifications_method_chaining(self):
        complex_obj = StructurePredictionComplex(chains=["MVLSPADKTN"])
        complex_obj.add_modification_to_chain(0, 4, "SEP")
        result = complex_obj.clear_all_modifications()
        assert result is complex_obj
        assert len(complex_obj.chains[0].modifications) == 0


class TestChainModificationsValidation:
    """Test suite for the ALLOWS_CHAIN_MODIFICATIONS validator."""

    def test_allows_modifications_accepts_modified_chains(self):
        """Test that input class that allows modifications accepts modified chains."""
        chain = Chain(
            sequence="MVLSPADKTN",
            modifications=[ChainModification(position=4, modification_code="SEP")]
        )
        complex_obj = StructurePredictionComplex(chains=[chain])
        input_obj = StructurePredictionInputTestClass(complexes=[complex_obj])
        assert len(input_obj.complexes) == 1
        assert len(input_obj.complexes[0].chains[0].modifications) == 1

    def test_allows_modifications_accepts_unmodified_chains(self):
        """Test that input class that allows modifications also accepts unmodified chains."""
        complex_obj = StructurePredictionComplex(chains=["MVLSPADKTN"])
        input_obj = StructurePredictionInputTestClass(complexes=[complex_obj])
        assert len(input_obj.complexes) == 1
        assert len(input_obj.complexes[0].chains[0].modifications) == 0

    def test_disallows_modifications_rejects_modified_chains(self):
        """Test that input class that disallows modifications rejects modified chains."""
        chain = Chain(
            sequence="MVLSPADKTN",
            modifications=[ChainModification(position=4, modification_code="SEP")]
        )
        complex_obj = StructurePredictionComplex(chains=[chain])
        with pytest.raises(ValueError, match="contains modifications"):
            NoModificationsInput(complexes=[complex_obj])

    def test_disallows_modifications_accepts_unmodified_chains(self):
        """Test that input class that disallows modifications accepts unmodified chains."""
        complex_obj = StructurePredictionComplex(chains=["MVLSPADKTN"])
        input_obj = NoModificationsInput(complexes=[complex_obj])
        assert len(input_obj.complexes) == 1

    def test_disallows_modifications_rejects_single_modified_chain_in_complex(self):
        """Test that even a single modified chain in a multi-chain complex is rejected."""
        chain1 = Chain(sequence="MVLSPADKTN")  # No modifications
        chain2 = Chain(
            sequence="ATCGATCG",
            modifications=[ChainModification(position=1, modification_code="6MA")]
        )
        complex_obj = StructurePredictionComplex(chains=[chain1, chain2])
        with pytest.raises(ValueError, match="contains modifications"):
            NoModificationsInput(complexes=[complex_obj])

    def test_disallows_modifications_error_includes_complex_index(self):
        """Test that error message includes the complex index."""
        complex1 = StructurePredictionComplex(chains=["MVLSPADKTN"])
        complex2 = StructurePredictionComplex(
            chains=[
                Chain(
                    sequence="ATCGATCG",
                    modifications=[ChainModification(position=1, modification_code="6MA")]
                )
            ]
        )
        with pytest.raises(ValueError, match="Complex 1"):
            NoModificationsInput(complexes=[complex1, complex2])

    def test_disallows_modifications_error_includes_tool_name(self):
        """Test that error message includes the input class name."""
        chain = Chain(
            sequence="MVLSPADKTN",
            modifications=[ChainModification(position=4, modification_code="SEP")]
        )
        complex_obj = StructurePredictionComplex(chains=[chain])
        with pytest.raises(ValueError, match="NoModificationsInput does not allow"):
            NoModificationsInput(complexes=[complex_obj])

    def test_disallows_modifications_with_multiple_modifications(self):
        """Test rejection when chain has multiple modifications."""
        chain = Chain(
            sequence="MVLSPADKTN",
            modifications=[
                ChainModification(position=4, modification_code="SEP"),
                ChainModification(position=9, modification_code="TPO")
            ]
        )
        complex_obj = StructurePredictionComplex(chains=[chain])
        with pytest.raises(ValueError, match="contains modifications"):
            NoModificationsInput(complexes=[complex_obj])

    def test_disallows_modifications_with_string_input(self):
        """Test that unmodified strings are accepted by NoModificationsInput."""
        input_obj = NoModificationsInput(complexes=["MVLSPADKTN", "ATCGATCG"])
        assert len(input_obj.complexes) == 2

    def test_disallows_modifications_with_list_of_lists(self):
        """Test that unmodified list of lists is accepted by NoModificationsInput."""
        input_obj = NoModificationsInput(complexes=[["MVLSPADKTN", "ATCGATCG"]])
        assert len(input_obj.complexes) == 1
        assert len(input_obj.complexes[0].chains) == 2

    def test_validation_across_multiple_complexes(self):
        """Test that validation checks all complexes in the input."""
        complex1 = StructurePredictionComplex(chains=["MVLSPADKTN"])
        complex2 = StructurePredictionComplex(chains=["ATCGATCG"])
        complex3 = StructurePredictionComplex(chains=["AUGCAUGC"])
        # All should pass for NoModificationsInput
        input_obj = NoModificationsInput(complexes=[complex1, complex2, complex3])
        assert len(input_obj.complexes) == 3

    def test_allows_modifications_with_dictionary_chain(self):
        """Test that dictionary chains with modifications work with allowing input."""
        complex_obj = StructurePredictionComplex(
            chains=[
                {
                    "sequence": "MVLSPADKTN",
                    "entity_type": "protein",
                    "modifications": [(4, "SEP")]
                }
            ]
        )
        input_obj = StructurePredictionInputTestClass(complexes=[complex_obj])
        assert len(input_obj.complexes) == 1
        assert len(input_obj.complexes[0].chains[0].modifications) == 1

    def test_disallows_modifications_with_dictionary_chain(self):
        """Test that dictionary chains with modifications are rejected by disallowing input."""
        complex_obj = StructurePredictionComplex(
            chains=[
                {
                    "sequence": "MVLSPADKTN",
                    "entity_type": "protein",
                    "modifications": [(4, "SEP")]
                }
            ]
        )
        with pytest.raises(ValueError, match="contains modifications"):
            NoModificationsInput(complexes=[complex_obj])


class TestModificationCompatibilityValidation:
    """Test suite for validating that modifications are compatible with their residues or bases."""

    def test_valid_protein_modification_passes(self):
        """Test that a valid protein modification (SEP on serine) passes validation."""
        # Position 4 is 'S' (serine), SEP is phosphoserine
        chain = Chain(
            sequence="MVLSPADKTN",
            entity_type="protein",
            modifications=[(4, "SEP")]
        )
        assert chain.modifications[0].modification_code == "SEP"

    def test_valid_rna_modification_passes(self):
        """Test that a valid RNA modification (2MG on guanosine) passes validation."""
        # Position 3 is 'G' (guanosine), 2MG is 2'-O-methylguanosine
        chain = Chain(
            sequence="AUGCAUGC",
            entity_type="rna",
            modifications=[(3, "2MG")]
        )
        assert chain.modifications[0].modification_code == "2MG"

    def test_valid_dna_modification_passes(self):
        """Test that a valid DNA modification passes validation."""
        # Position 1 is 'A' (adenine), 1AP is 1-aminopurine (DNA adenine modification)
        chain = Chain(
            sequence="ATCGATCG",
            entity_type="dna",
            modifications=[(1, "1AP")]
        )
        assert chain.modifications[0].modification_code == "1AP"

    def test_invalid_protein_modification_raises(self):
        """Test that an invalid protein modification (SEP on threonine) raises error."""
        # Position 5 is 'T' (threonine), but SEP is for serine
        with pytest.raises(ValueError, match="Invalid modification 'SEP' at position 5"):
            Chain(
                sequence="MVLTPADKTN",
                entity_type="protein",
                modifications=[(5, "SEP")]
            )

    def test_invalid_modification_error_shows_expected_residue(self):
        """Test that error message shows which residue the modification expects."""
        # SEP expects 'S' (serine)
        with pytest.raises(ValueError, match="This modification is for residue or base 'S'"):
            Chain(
                sequence="MVLTPADKTN",
                entity_type="protein",
                modifications=[(5, "SEP")]  # Position 5 is 'T', not 'S'
            )

    def test_invalid_modification_error_shows_actual_residue(self):
        """Test that error message shows the actual residue at that position."""
        # Position 5 is 'P' in sequence "MVLTPADKTN"
        with pytest.raises(ValueError, match="but position 5 contains 'P'"):
            Chain(
                sequence="MVLTPADKTN",
                entity_type="protein",
                modifications=[(5, "SEP")]
            )

    def test_invalid_modification_error_lists_allowed_modifications(self):
        """Test that error message lists allowed modifications for the actual residue."""
        # Position 5 is 'P' in "MVLTPADKTN", should show HYP and other proline modifications
        with pytest.raises(ValueError, match="Allowed modifications for 'P' in protein: .*HYP"):
            Chain(
                sequence="MVLTPADKTN",
                entity_type="protein",
                modifications=[(5, "SEP")]
            )

    def test_multiple_valid_modifications_pass(self):
        """Test that multiple valid modifications all pass validation."""
        chain = Chain(
            sequence="MVLSPADKTN",
            entity_type="protein",
            modifications=[
                (4, "SEP"),
                (9, "TPO")
            ]
        )
        assert len(chain.modifications) == 2

    def test_one_invalid_among_multiple_modifications_raises(self):
        """Test that one invalid modification among valid ones still raises error."""
        with pytest.raises(ValueError, match="Invalid modification 'SEP' at position 10"):
            Chain(
                sequence="MVLSPADKTN",
                entity_type="protein",
                modifications=[
                    (4, "SEP"),   # Valid: position 4 is 'S'
                    (10, "SEP")   # Invalid: position 10 is 'N', not 'S'
                ]
            )

    def test_rna_modification_on_wrong_base_raises(self):
        """Test that RNA modification on wrong base raises error."""
        # 2MG is for guanosine (G), but position 1 is adenosine (A)
        with pytest.raises(ValueError, match="Invalid modification '2MG' at position 1"):
            Chain(
                sequence="AUGCAUGC",
                entity_type="rna",
                modifications=[(1, "2MG")]  # Position 1 is 'A', not 'G'
            )

    def test_dna_vs_rna_modification_distinction(self):
        """Test that DNA and RNA modifications are properly distinguished."""
        # 2MG is an RNA modification for G, should not work on DNA G
        with pytest.raises(ValueError, match="Invalid modification '2MG'"):
            Chain(
                sequence="ATCGATCG",
                entity_type="dna",
                modifications=[(7, "2MG")]  # Position 7 is 'G' but this is DNA
            )

    def test_add_modification_method_validates_compatibility(self):
        """Test that add_modification method also validates compatibility."""
        chain = Chain(sequence="MVLSPADKTN", entity_type="protein")
        with pytest.raises(ValueError, match="Invalid modification 'TPO'"):
            chain.add_modification(4, "TPO")

    def test_complex_with_invalid_modification_raises(self):
        """Test that creating a complex with invalid modifications raises error."""
        with pytest.raises(ValueError, match="Invalid modification 'SEP'"):
            StructurePredictionComplex(
                chains=[
                    {
                        "sequence": "MVLTPADKTN",
                        "entity_type": "protein",
                        "modifications": [(5, "SEP")]  # Position 5 is 'T', not 'S'
                    }
                ]
            )

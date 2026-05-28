"""tests/structure_prediction_tests/test_chain_ids.py.

Tests for the shared chain-ID helpers that every structure predictor uses to
keep predicted-structure chain IDs aligned with the input ``Complex``.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from proto_tools.entities.complex import Chain
from proto_tools.entities.ligands import Fragment
from proto_tools.tools.structure_prediction.shared_data_models import (
    normalize_output_chain_ids,
    resolve_chain_ids,
)

_PROTEIN_A = "EELLKKLEELLKKLEELLKK"
_PROTEIN_B = "KKLLEEKKLLEEKKLLEEKK"


# ── resolve_chain_ids ────────────────────────────────────────────────────────


def test_resolve_chain_ids_prefers_explicit_then_positional():
    """Explicit ids win; unlabeled chains fall back to positional A/B/... by full index."""
    chains = [
        Chain(id="T", sequence=_PROTEIN_A, entity_type="protein"),
        Chain(sequence=_PROTEIN_B, entity_type="protein"),
        Fragment(id="L", ccd_code="ATP"),
        Fragment(ccd_code="HEM"),
    ]
    assert resolve_chain_ids(chains) == ["T", "B", "L", "D"]


def test_resolve_chain_ids_all_positional():
    """With no explicit ids, every chain gets its positional label."""
    chains = [
        Chain(sequence=_PROTEIN_A, entity_type="protein"),
        Chain(sequence=_PROTEIN_B, entity_type="protein"),
    ]
    assert resolve_chain_ids(chains) == ["A", "B"]


def test_chain_and_fragment_reject_empty_id():
    """Empty-string ids are unrepresentable, so resolve_chain_ids' ``or`` fallback is safe."""
    with pytest.raises(ValidationError):
        Chain(id="", sequence=_PROTEIN_A, entity_type="protein")
    with pytest.raises(ValidationError):
        Fragment(id="", ccd_code="ATP")


def test_resolve_chain_ids_disambiguates_same_chain_ligand():
    """A ligand sharing its polymer's chain id (Complex.from_structure) takes the next free label."""
    chains = [
        Chain(id="A", sequence=_PROTEIN_A, entity_type="protein"),
        Fragment(id="A", ccd_code="HEM"),  # same chain id as its polymer
    ]
    assert resolve_chain_ids(chains) == ["A", "B"]


def test_resolve_chain_ids_disambiguated_ligand_avoids_later_chains():
    """A bumped ligand must skip labels later polymer chains will claim, not just earlier ones."""
    # from_structure of a 2-chain protein with a ligand bound to chain A.
    chains = [
        Chain(id="A", sequence=_PROTEIN_A, entity_type="protein"),
        Fragment(id="A", ccd_code="HEM"),  # collides with chain A -> must not become "B"
        Chain(id="B", sequence=_PROTEIN_B, entity_type="protein"),
    ]
    assert resolve_chain_ids(chains) == ["A", "C", "B"]


def test_resolve_chain_ids_rejects_duplicate_polymers():
    """Two polymer chains sharing an id is a caller error and must raise."""
    chains = [
        Chain(id="X", sequence=_PROTEIN_A, entity_type="protein"),
        Chain(id="X", sequence=_PROTEIN_B, entity_type="protein"),
    ]
    with pytest.raises(ValueError, match="duplicate chain ID"):
        resolve_chain_ids(chains)


def test_resolve_chain_ids_rejects_explicit_positional_collision():
    """An explicit id colliding with a later chain's positional fallback also raises."""
    chains = [
        Chain(id="B", sequence=_PROTEIN_A, entity_type="protein"),
        Chain(sequence=_PROTEIN_B, entity_type="protein"),  # positional fallback -> "B"
    ]
    with pytest.raises(ValueError, match="duplicate chain ID"):
        resolve_chain_ids(chains)


# ── normalize_output_chain_ids ───────────────────────────────────────────────


def _mock_structure(observed_ids: list[str]) -> Mock:
    structure = Mock()
    structure.get_chain_ids.return_value = observed_ids
    structure.with_renamed_chains.return_value = Mock(name="renamed")
    return structure


def test_normalize_no_op_when_ids_already_match():
    """When output IDs already equal the resolved input IDs, return the structure untouched."""
    chains = [Chain(sequence=_PROTEIN_A, entity_type="protein")]  # -> "A"
    structure = _mock_structure(["A"])

    assert normalize_output_chain_ids(structure, chains) is structure
    structure.with_renamed_chains.assert_not_called()


def test_normalize_remaps_polymer_chains_positionally():
    """Predictor-native names are remapped to the input IDs by order."""
    chains = [
        Chain(id="H", sequence=_PROTEIN_A, entity_type="protein"),
        Chain(id="L", sequence=_PROTEIN_B, entity_type="protein"),
    ]
    structure = _mock_structure(["0", "1"])

    result = normalize_output_chain_ids(structure, chains)

    structure.with_renamed_chains.assert_called_once_with({"0": "H", "1": "L"})
    assert result is structure.with_renamed_chains.return_value


def test_normalize_excludes_ligand_chains():
    """Ligand chains carry no polymer entry in get_chain_ids, so they're skipped."""
    chains = [
        Chain(id="A", sequence=_PROTEIN_A, entity_type="protein"),
        Fragment(id="L", ccd_code="ATP"),
        Chain(id="B", sequence=_PROTEIN_B, entity_type="protein"),
    ]
    structure = _mock_structure(["x", "y"])  # two polymer chains only

    normalize_output_chain_ids(structure, chains)

    structure.with_renamed_chains.assert_called_once_with({"x": "A", "y": "B"})


def test_normalize_no_op_on_count_mismatch():
    """A polymer-count mismatch leaves IDs unchanged rather than mis-mapping."""
    chains = [
        Chain(sequence=_PROTEIN_A, entity_type="protein"),
        Chain(sequence=_PROTEIN_B, entity_type="protein"),
    ]
    structure = _mock_structure(["A"])  # one observed vs two expected ("A","B")

    assert normalize_output_chain_ids(structure, chains) is structure
    structure.with_renamed_chains.assert_not_called()

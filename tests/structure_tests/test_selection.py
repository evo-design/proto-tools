"""Tests for ChainSelection, ResidueSelection, and StructureInputBase."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures import (
    ChainSelection,
    ResidueSelection,
    SingleChainSelection,
    Structure,
    StructureInputBase,
)

# Two-chain PDB used elsewhere in the test suite (chains A and B).
EXAMPLE_PDB = Path(__file__).parent.parent / "dummy_data" / "pdl1.pdb"


@pytest.fixture
def structure() -> Structure:
    """Load a real two-chain structure for validate_against tests."""
    return Structure.from_file(EXAMPLE_PDB)


# ============================================================================
# ChainSelection
# ============================================================================
def test_chain_selection_coerces_string() -> None:
    sel = ChainSelection.model_validate("A")
    assert sel.chains == ["A"]


def test_chain_selection_coerces_list() -> None:
    sel = ChainSelection.model_validate(["A", "B"])
    assert sel.chains == ["A", "B"]


def test_chain_selection_canonical_kwargs() -> None:
    sel = ChainSelection(chains=["A", "B"])
    assert sel.chains == ["A", "B"]


def test_chain_selection_passthrough() -> None:
    original = ChainSelection(chains=["A"])
    sel = ChainSelection.model_validate(original)
    assert sel.chains == ["A"]


def test_chain_selection_rejects_dict_of_positions() -> None:
    with pytest.raises(ValidationError, match="Cannot coerce dict to ChainSelection"):
        ChainSelection.model_validate({"A": [1, 2]})


def test_chain_selection_rejects_int() -> None:
    with pytest.raises(ValidationError, match="Cannot coerce int to ChainSelection"):
        ChainSelection.model_validate(42)


def test_chain_selection_rejects_list_with_non_strings() -> None:
    with pytest.raises(ValidationError, match="must contain only chain ID strings"):
        ChainSelection.model_validate(["A", 1])


def test_chain_selection_rejects_empty_list() -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        ChainSelection.model_validate([])


def test_chain_selection_rejects_empty_canonical() -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        ChainSelection(chains=[])


def test_chain_selection_validate_against_good(structure: Structure) -> None:
    ChainSelection(chains=["A", "B"]).validate_against(structure)


def test_chain_selection_validate_against_bad_chain(structure: Structure) -> None:
    with pytest.raises(ValueError, match=r"myfield: chain.*'Z'"):
        ChainSelection(chains=["Z"]).validate_against(structure, label="myfield")


# ============================================================================
# ResidueSelection
# ============================================================================
def test_residue_selection_coerces_dict_of_positions() -> None:
    sel = ResidueSelection.model_validate({"A": [1, 2, 3]})
    assert sel.chains == {"A": [1, 2, 3]}


def test_residue_selection_canonical_kwargs() -> None:
    sel = ResidueSelection(chains={"A": [1, 2]})
    assert sel.chains == {"A": [1, 2]}


def test_residue_selection_passthrough() -> None:
    original = ResidueSelection(chains={"A": [1]})
    sel = ResidueSelection.model_validate(original)
    assert sel.chains == {"A": [1]}


def test_residue_selection_rejects_string() -> None:
    with pytest.raises(ValidationError, match="Cannot coerce str to ResidueSelection"):
        ResidueSelection.model_validate("A")


def test_residue_selection_rejects_list() -> None:
    with pytest.raises(ValidationError, match="Cannot coerce list to ResidueSelection"):
        ResidueSelection.model_validate(["A"])


def test_residue_selection_rejects_empty_dict() -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        ResidueSelection.model_validate({})


def test_residue_selection_drops_only_chain_with_empty_positions(caplog: pytest.LogCaptureFixture) -> None:
    """Empty positions for the only chain → drop it, then cascade to parent-empty rejection."""
    with caplog.at_level("WARNING"):
        with pytest.raises(ValidationError, match=r"cannot be empty"):
            ResidueSelection.model_validate({"A": []})
    assert any("dropping chain with empty position list" in rec.message for rec in caplog.records)


def test_residue_selection_drops_empty_chain_entries(caplog: pytest.LogCaptureFixture) -> None:
    """Empty chain entries are dropped with a warning when other chains have positions."""
    with caplog.at_level("WARNING"):
        sel = ResidueSelection.model_validate({"A": [1, 2], "B": [], "C": [3]})
    assert sel.chains == {"A": [1, 2], "C": [3]}
    assert any("chain 'B': dropping chain with empty position list" in rec.message for rec in caplog.records)


def test_residue_selection_rejects_zero_position() -> None:
    """Position 0 is invalid because positions are 1-indexed; reject at construction."""
    with pytest.raises(ValidationError, match=r"positions must be >= 1.*1-indexed"):
        ResidueSelection.model_validate({"A": [0, 1, 2]})


def test_residue_selection_rejects_negative_position() -> None:
    with pytest.raises(ValidationError, match=r"positions must be >= 1.*1-indexed"):
        ResidueSelection.model_validate({"A": [-1, 5]})


def test_residue_selection_rejects_bool_positions() -> None:
    """Booleans are int subclasses but not valid positions; reject with a clear error."""
    with pytest.raises(ValidationError, match=r"positions must be int, not bool"):
        ResidueSelection.model_validate({"A": [True, False]})


def test_residue_selection_rejects_mixed_canonical_and_extra_keys() -> None:
    """Reject dicts that mix the canonical 'chains' key with extras up front.

    Without the explicit guard, this would fall through to the per-chain
    shorthand branch and produce a confusing nested-coercion error.
    """
    with pytest.raises(ValidationError, match=r"mixed keys"):
        ResidueSelection.model_validate({"chains": {"A": [1]}, "extra_key": "foo"})


def test_residue_selection_deduplicates_positions(caplog: pytest.LogCaptureFixture) -> None:
    """Duplicate positions are dropped silently; a warning is emitted."""
    with caplog.at_level("WARNING"):
        sel = ResidueSelection.model_validate({"A": [1, 1, 2, 3, 2]})
    assert sel.chains == {"A": [1, 2, 3]}
    assert any("dropped 2 duplicate position(s)" in rec.message for rec in caplog.records)


def test_residue_selection_dedup_silent_when_no_duplicates(caplog: pytest.LogCaptureFixture) -> None:
    """No warning when positions are already unique."""
    with caplog.at_level("WARNING"):
        ResidueSelection.model_validate({"A": [1, 2, 3]})
    assert not any("duplicate position" in rec.message for rec in caplog.records)


def test_chain_selection_rejects_chains_non_list_value() -> None:
    """Canonical kwargs form must use a list; non-list values get a clear error."""
    with pytest.raises(ValidationError, match=r"'chains' must be a list of chain IDs"):
        ChainSelection.model_validate({"chains": "A"})


def test_chain_selection_rejects_mixed_canonical_and_extra_keys() -> None:
    """Reject dicts that mix the canonical 'chains' key with extras up front."""
    with pytest.raises(ValidationError, match=r"mixed keys"):
        ChainSelection.model_validate({"chains": ["A"], "extra_key": "foo"})


def test_residue_selection_validate_against_good(structure: Structure) -> None:
    ResidueSelection(chains={"A": [1, 2, 3]}).validate_against(structure)


def test_residue_selection_validate_against_bad_chain(structure: Structure) -> None:
    with pytest.raises(ValueError, match=r"myfield: chain.*'Z'"):
        ResidueSelection(chains={"Z": [1]}).validate_against(structure, label="myfield")


def test_residue_selection_validate_against_bad_position(structure: Structure) -> None:
    with pytest.raises(ValueError, match="invalid positions"):
        ResidueSelection(chains={"A": [9999]}).validate_against(structure)


# ============================================================================
# SingleChainSelection
# ============================================================================
def test_single_chain_selection_coerces_string() -> None:
    sel = SingleChainSelection.model_validate("A")
    assert sel.chain == "A"


def test_single_chain_selection_canonical_kwargs() -> None:
    sel = SingleChainSelection(chain="A")
    assert sel.chain == "A"


def test_single_chain_selection_passthrough() -> None:
    original = SingleChainSelection(chain="A")
    sel = SingleChainSelection.model_validate(original)
    assert sel.chain == "A"


def test_single_chain_selection_rejects_list() -> None:
    """A list is multi-chain; point the caller at ChainSelection."""
    with pytest.raises(ValidationError, match="ChainSelection for more than one chain"):
        SingleChainSelection.model_validate(["A", "B"])


def test_single_chain_selection_rejects_int() -> None:
    with pytest.raises(ValidationError, match="Cannot coerce int to SingleChainSelection"):
        SingleChainSelection.model_validate(42)


def test_single_chain_selection_rejects_mixed_keys() -> None:
    with pytest.raises(ValidationError, match=r"must use only the 'chain' key"):
        SingleChainSelection.model_validate({"chain": "A", "extra_key": "foo"})


def test_single_chain_selection_rejects_empty_string() -> None:
    """An empty chain ID is never a real chain; reject it at construction."""
    with pytest.raises(ValidationError, match="non-empty"):
        SingleChainSelection.model_validate("")


def test_single_chain_selection_rejects_empty_canonical() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        SingleChainSelection(chain="")


def test_single_chain_selection_validate_against_good(structure: Structure) -> None:
    SingleChainSelection(chain="A").validate_against(structure)


def test_single_chain_selection_validate_against_bad_chain(structure: Structure) -> None:
    with pytest.raises(ValueError, match=r"myfield: chain 'Z' not in structure"):
        SingleChainSelection(chain="Z").validate_against(structure, label="myfield")


# ============================================================================
# StructureInputBase
# ============================================================================
class _SubInput(StructureInputBase):
    """Fixture subclass with one of each selection role for auto-validation tests."""

    chain_role: ChainSelection | None = None
    single_chain_role: SingleChainSelection | None = None
    residue_role: ResidueSelection | None = None


def test_base_resolves_path_string() -> None:
    inp = _SubInput(structure=str(EXAMPLE_PDB))
    assert isinstance(inp.structure, Structure)


def test_base_resolves_path_object() -> None:
    inp = _SubInput(structure=EXAMPLE_PDB)
    assert isinstance(inp.structure, Structure)


def test_base_passes_through_structure_instance() -> None:
    s = Structure.from_file(EXAMPLE_PDB)
    inp = _SubInput(structure=s)
    assert inp.structure is s


def test_base_resolves_dict_form() -> None:
    s = Structure.from_file(EXAMPLE_PDB)
    inp = _SubInput.model_validate({"structure": s.model_dump(mode="json")})
    assert isinstance(inp.structure, Structure)


def test_base_auto_validates_chain_role() -> None:
    with pytest.raises(ValidationError, match=r"chain_role.*'Z'"):
        _SubInput(structure=str(EXAMPLE_PDB), chain_role=["Z"])


def test_base_auto_validates_single_chain_role() -> None:
    with pytest.raises(ValidationError, match=r"single_chain_role: chain 'Z'"):
        _SubInput(structure=str(EXAMPLE_PDB), single_chain_role="Z")


def test_base_auto_validates_residue_role() -> None:
    with pytest.raises(ValidationError, match=r"residue_role.*invalid positions"):
        _SubInput(structure=str(EXAMPLE_PDB), residue_role={"A": [9999]})


def test_base_skips_none_fields() -> None:
    inp = _SubInput(structure=str(EXAMPLE_PDB))
    assert inp.chain_role is None
    assert inp.residue_role is None


def test_base_accepts_valid_selections() -> None:
    inp = _SubInput(
        structure=str(EXAMPLE_PDB),
        chain_role="A",
        single_chain_role="B",
        residue_role={"A": [1, 2, 3]},
    )
    assert inp.chain_role is not None
    assert inp.single_chain_role is not None
    assert inp.residue_role is not None
    assert inp.chain_role.chains == ["A"]
    assert inp.single_chain_role.chain == "B"
    assert inp.residue_role.chains == {"A": [1, 2, 3]}


# ============================================================================
# Single-chain shorthand: bare list[int] auto-promoted to {<chain>: [...]}
# ============================================================================
class _SingleChainSub(StructureInputBase):
    """Test subclass for the single-chain residue-list shorthand."""

    fixed_positions: ResidueSelection | None = None


SINGLE_CHAIN_PDB = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"


def test_single_chain_shorthand_promotes_bare_position_list() -> None:
    """A bare list[int] on a ResidueSelection field is promoted to {<chain>: [...]} when the structure has one chain."""
    s = Structure.from_file(SINGLE_CHAIN_PDB)
    assert len(s.get_chain_ids()) == 1
    only_chain = s.get_chain_ids()[0]

    inp = _SingleChainSub(structure=s, fixed_positions=[1, 2, 3])

    assert inp.fixed_positions is not None
    assert inp.fixed_positions.chains == {only_chain: [1, 2, 3]}


def test_single_chain_shorthand_rejected_for_multichain_structure() -> None:
    """Bare list[int] against a multi-chain structure raises a clear error pointing to the dict form."""
    with pytest.raises(ValidationError, match=r"requires a single-chain structure"):
        _SingleChainSub(structure=str(EXAMPLE_PDB), fixed_positions=[1, 2, 3])


def test_dict_form_still_works_on_single_chain() -> None:
    """Explicit dict form still works on a single-chain structure (no auto-promotion needed)."""
    s = Structure.from_file(SINGLE_CHAIN_PDB)
    only_chain = s.get_chain_ids()[0]
    inp = _SingleChainSub(structure=s, fixed_positions={only_chain: [5, 6]})
    assert inp.fixed_positions is not None
    assert inp.fixed_positions.chains == {only_chain: [5, 6]}

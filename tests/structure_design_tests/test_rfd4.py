"""tests/structure_design_tests/test_rfd4.py.

Tests for RFD4-Proteina design and inverse folding.
"""

import functools
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import proto_tools.utils.standalone_helpers_source as _shs
from proto_tools.entities.structures import Structure
from proto_tools.tools import (
    RFD4Design,
    RFD4DesignConfig,
    RFD4DesignSpec,
    RFD4Input,
    RFD4InverseFoldingConfig,
    RFD4Output,
    RFD4SequenceProbability,
    run_rfd4_design,
    run_rfd4_inverse_folding,
)
from proto_tools.tools.structure_design.rfd4 import rfd4_design
from proto_tools.tools.structure_design.rfd4.helpers import (
    chains_needing_translation,
    prepare_rfd4_cif,
    translate_contig,
    translate_selection,
)
from proto_tools.tools.structure_design.rfd4.rfd4_design import (
    EXAMPLE_INPUT_FIXTURE,
    build_designs,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("rfd4")

# The shipped fixture is a 65-residue single-chain monomer numbered 1-65.
_FIXTURE_LENGTH = 65
# Author numbering offset applied to make author ids differ from label_seq_id (1-65 -> 17-81).
_OFFSET = 16

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STANDALONE_DIR = _REPO_ROOT / "proto_tools/tools/structure_design/rfd4/standalone"
# A real PDB entry with a polymer chain and a zinc sharing one author chain id.
_HETERO_FIXTURE = _REPO_ROOT / "proto_tools/tools/structure_scoring/metal3d/example_input_fixture.pdb"


def _fixture_structure() -> Structure:
    """The 65-residue monomer that ships with the toolkit."""
    return Structure.from_file(EXAMPLE_INPUT_FIXTURE)


def _offset_cif_text(offset: int = _OFFSET) -> str:
    """CIF text for the fixture with every author residue id shifted by ``offset``."""
    import gemmi

    structure = gemmi.read_structure(str(EXAMPLE_INPUT_FIXTURE))
    for model in structure:
        for chain in model:
            for residue in chain:
                residue.seqid.num += offset
    return structure.make_mmcif_document().as_string()


def _insertion_code_cif_text() -> str:
    """Fixture CIF renumbered so chain A reads 1, 1A, 2, 3, ... -- an insertion code at position 2."""
    import gemmi

    structure = gemmi.read_structure(str(EXAMPLE_INPUT_FIXTURE))
    for model in structure:
        for chain in model:
            for index, residue in enumerate(chain):
                residue.seqid = gemmi.SeqId(1, "A") if index == 1 else gemmi.SeqId(max(index, 1), " ")
    return structure.make_mmcif_document().as_string()


def _offset_map(offset: int = _OFFSET) -> dict[str, dict[str, int]]:
    """Author->label map for the offset fixture (author 17-81 -> label 1-65)."""
    return prepare_rfd4_cif(_offset_cif_text(offset), "cif")[1]


def _identity_map() -> dict[str, dict[str, int]]:
    """Author->label map for the shipped fixture, whose numbering is already 1-65."""
    return prepare_rfd4_cif(EXAMPLE_INPUT_FIXTURE.read_text(), "cif")[1]


def _atom_site_rows(cif_text: str, *columns: str) -> list[tuple[str, ...]]:
    """Read the named ``_atom_site`` columns out of a CIF string."""
    import gemmi

    block = gemmi.cif.read_string(cif_text).sole_block()
    table = block.find("_atom_site.", list(columns))
    return [tuple(row[index] for index in range(len(columns))) for row in table]


def _design_payload(spec_keys: list[str]) -> dict[str, list[dict[str, object]]]:
    """A flat subprocess payload with one design per entry of ``spec_keys``."""
    content = EXAMPLE_INPUT_FIXTURE.read_text()
    return {"designs": [{"structure_content": content, "spec_key": key} for key in spec_keys]}


@functools.cache
def _standalone_inference() -> ModuleType:
    """Import the standalone ``inference.py`` with the worker-injected ``standalone_helpers`` on sys.path."""
    helpers_dir = str(Path(next(iter(_shs.__path__))))
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    spec = importlib.util.spec_from_file_location("_rfd4_inference_for_tests", _STANDALONE_DIR / "inference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_dispatch(monkeypatch: pytest.MonkeyPatch, design_keys: list[str]) -> dict[str, Any]:
    """Replace ``ToolInstance.dispatch`` with a recorder that answers with one design per key.

    The returned dict is filled with the subprocess payload on every call, plus the design
    ids actually written into the spec file (which lives in a tempdir that is gone by the
    time the run returns).
    """
    captured: dict[str, Any] = {}

    def fake_dispatch(tool_id: str, input_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        spec_file = yaml.safe_load(Path(input_data["spec_path"]).read_text())
        captured.clear()
        captured.update(input_data)
        captured["_spec_file_keys"] = list(spec_file)
        return _design_payload(design_keys)

    monkeypatch.setattr(rfd4_design.ToolInstance, "dispatch", fake_dispatch)
    return captured


# ── Numbering translation: CIF preparation ──────────────────────────────────


def test_prepare_rfd4_cif_assigns_label_numbering_and_author_chain_ids():
    """A prepared CIF carries label_seq_id and a label_asym_id equal to the author chain id.

    Neither holds for a plain PDB->CIF conversion: gemmi leaves both unset ('.'), and its own
    ``setup_entities`` invents subchain-derived ids such as 'Axp'. RFD4 addresses chains by
    ``label_asym_id`` and residues by ``label_seq_id``, so both must be pinned.
    """
    cif_text, auth_to_label = prepare_rfd4_cif(_fixture_structure().structure_pdb, "pdb")

    rows = _atom_site_rows(cif_text, "label_asym_id", "label_seq_id", "auth_seq_id", "auth_asym_id")
    assert rows, "prepared CIF has no _atom_site rows"
    assert {label_asym for label_asym, _, _, _ in rows} == {"A"}
    assert all(label_asym == auth_asym for label_asym, _, _, auth_asym in rows)
    assert all(label_seq not in (".", "?", "") for _, label_seq, _, _ in rows)
    # Author numbering already starts at 1 here, so label numbering must agree with it.
    assert all(label_seq == auth_seq for _, label_seq, auth_seq, _ in rows)

    assert sorted(auth_to_label) == ["A"]
    assert auth_to_label["A"] == {str(index): index for index in range(1, _FIXTURE_LENGTH + 1)}


def test_prepare_rfd4_cif_maps_offset_author_numbering_to_label_seq_ids():
    """Author ids that are offset from label numbering produce a non-identity map."""
    cif_text, auth_to_label = prepare_rfd4_cif(_offset_cif_text(), "cif")

    assert auth_to_label["A"] == {str(index + _OFFSET): index for index in range(1, _FIXTURE_LENGTH + 1)}
    rows = _atom_site_rows(cif_text, "label_seq_id", "auth_seq_id")
    assert all(int(auth_seq) - int(label_seq) == _OFFSET for label_seq, auth_seq in rows)
    assert chains_needing_translation(auth_to_label) == ["A"]


def test_prepare_rfd4_cif_keeps_non_polymer_residues_off_the_polymer_label_asym_id():
    """A hetero residue sharing the polymer's author chain must still get its own label_asym_id.

    atomworks refuses to read a CIF whose ``label_asym_id`` mixes residues with and without a
    ``label_seq_id`` ("standalone non-polymers must use a separate label_asym_id"), which is
    what pinning every subchain to the author chain id produced for any entry carrying a
    water, ion, or ligand inside a protein chain -- that is, most real PDB entries.
    """
    cif_text, auth_to_label = prepare_rfd4_cif(_HETERO_FIXTURE.read_text(), "pdb")

    rows = _atom_site_rows(cif_text, "label_asym_id", "label_seq_id", "auth_asym_id", "label_comp_id")
    assert rows, "prepared CIF has no _atom_site rows"
    numbered = {(label_asym, label_seq not in (".", "?", "")) for label_asym, label_seq, _, _ in rows}
    mixed = {asym for asym, _ in numbered if (asym, True) in numbered and (asym, False) in numbered}
    assert not mixed, f"label_asym_id(s) {sorted(mixed)} mix defined and missing label_seq_id"

    # The polymer keeps the author chain id, so sel('X') and X42-48 still address it.
    polymer_rows = [row for row in rows if row[1] not in (".", "?", "")]
    assert {label_asym for label_asym, _, _, _ in polymer_rows} == {"X"}
    assert all(label_asym == auth_asym for label_asym, _, auth_asym, _ in polymer_rows)

    zinc_rows = [row for row in rows if row[3] == "ZN"]
    assert zinc_rows, "hetero fixture lost its zinc"
    assert all(label_asym != "X" and label_seq in (".", "?") for label_asym, label_seq, _, _ in zinc_rows)

    # Non-polymers are addressed by res_name upstream, so they carry no author->label entry.
    assert list(auth_to_label) == ["X"]
    assert "1268" not in auth_to_label["X"]


def test_prepare_rfd4_cif_keys_insertion_codes_and_shifts_every_later_label_id():
    """An insertion code is part of the author residue key, and it offsets every residue after it.

    For a chain numbered 1, 1A, 2, ... the map must read ``{"1": 1, "1A": 2, "2": 3, ...}``.
    Keying on ``seqid.num`` alone would collide 1 with 1A, silently dropping one of them and
    conditioning the wrong residue for every id further down the chain.
    """
    auth_to_label = prepare_rfd4_cif(_insertion_code_cif_text(), "cif")[1]

    assert sorted(auth_to_label["A"].items(), key=lambda item: item[1])[:3] == [("1", 1), ("1A", 2), ("2", 3)]
    assert len(auth_to_label["A"]) == _FIXTURE_LENGTH
    # Author id N > 1 now sits on label N+1, so nothing downstream may forward author ids.
    assert auth_to_label["A"]["64"] == _FIXTURE_LENGTH
    assert chains_needing_translation(auth_to_label) == ["A"]
    assert translate_contig("A2-64", auth_to_label) == "A3-65"


# ── Numbering translation: contigs and selections ───────────────────────────


def test_translate_contig_rewrites_preserved_segment_ids():
    """Preserved segments translate; chainbreaks and expandable segments pass through."""
    auth_to_label = _offset_map()

    assert translate_contig("A17-81", auth_to_label) == "A1-65"
    assert translate_contig("A56", auth_to_label) == "A40"
    # Only the preserved segment changes: '50-120' (unnamed expandable), '/0' (chainbreak) and
    # 'E1:2-20R' (named expandable with a chain type) carry no input residue ids.
    assert translate_contig("50-120, /0, A17-81, /0, E1:2-20R", auth_to_label) == "50-120, /0, A1-65, /0, E1:2-20R"


def test_translate_selection_rewrites_path_residue_ids():
    """Scalar, bracketed-list, range, and atom-level sel() paths all get their res_id translated."""
    auth_to_label = _offset_map()

    assert translate_selection("sel('A/*/17')", auth_to_label) == "sel('A/*/1')"
    assert translate_selection("sel('A/*/[17-20, 30]')", auth_to_label) == "sel('A/*/[1-4, 14]')"
    # The atom name sits in the fourth field, so the res_id field is the only one rewritten.
    assert translate_selection("sel('A/*/56/[CG,OH]')", auth_to_label) == "sel('A/*/40/[CG,OH]')"
    # atomworks paths run to five fields (the trailing one is the alt/atom index).
    assert translate_selection("sel('A/*/17/CA/1')", auth_to_label) == "sel('A/*/1/CA/1')"
    # The rewrite reuses the matched quote character, so a double-quoted path stays double-quoted.
    assert translate_selection('sel("A/*/17")', auth_to_label) == 'sel("A/*/1")'
    # Whitespace inside a bracketed list is upstream-legal and must not reach the lookup.
    assert translate_selection("sel('A/*/[ 17 , 19 ]')", auth_to_label) == "sel('A/*/[1, 3]')"
    # Every sel() call in a compound query is rewritten.
    assert translate_selection("sel('A/*/17') or sel('A/*/81')", auth_to_label) == "sel('A/*/1') or sel('A/*/65')"


def test_gappy_author_numbering_is_translated_for_every_residue():
    """Numbering that is neither identical nor a constant offset still needs translating.

    ``_identity_mapping`` has to hold for *every* residue: a check that compared only the
    first key, the lowest key, or the chain's span would call this chain identity-numbered
    and forward author ids as label ids, silently conditioning the wrong residues.
    """
    gappy = {"A": {"1": 1, "2": 2, "10": 3}}

    assert chains_needing_translation(gappy) == ["A"]
    assert translate_contig("A1-10", gappy) == "A1-3"
    assert translate_selection("sel('A/*/[2, 10]')", gappy) == "sel('A/*/[2, 3]')"


def test_translate_selection_resolves_chain_wildcards_and_lists_across_chains():
    """A path's chain field may be a wildcard or a list, and each chain has its own offset."""
    two_chains = {"A": {"17": 1, "18": 2}, "B": {"100": 1, "101": 2}}

    assert translate_selection("sel('A/*/18') or sel('B/*/101')", two_chains) == "sel('A/*/2') or sel('B/*/2')"
    # A wildcard or list chain field must resolve to the same label id in every chain it covers.
    assert translate_selection("sel('*/*/18')", {"A": {"18": 2}, "B": {"18": 2}}) == "sel('*/*/2')"
    assert translate_selection("sel('[A,B]/*/18')", {"A": {"18": 2}, "B": {"18": 2}}) == "sel('[A,B]/*/2')"
    # A wildcard over chains with different offsets cannot be rewritten as one path.
    with pytest.raises(ValueError, match="maps to different label ids across chains") as excinfo:
        translate_selection("sel('*/*/17')", {"A": {"17": 1}, "B": {"17": 5}})
    assert "residue_numbering='label'" in str(excinfo.value)
    # An id present in one chain but not the other names the chain that lacks it.
    with pytest.raises(ValueError, match="residue B17 is not present"):
        translate_selection("sel('*/*/17')", two_chains)


def test_translate_selection_passes_through_queries_without_residue_ids():
    """Paths with no res_id field and pure annotation queries are untouched."""
    auth_to_label = _offset_map()

    assert translate_selection("sel('A')", auth_to_label) == "sel('A')"
    assert translate_selection("sel('A/MET')", auth_to_label) == "sel('A/MET')"
    assert translate_selection("sel('A/*/*')", auth_to_label) == "sel('A/*/*')"
    assert translate_selection("is_ligand", auth_to_label) == "is_ligand"
    assert translate_selection("is_ligand and not is_protein_backbone", auth_to_label) == (
        "is_ligand and not is_protein_backbone"
    )


def test_translation_is_noop_when_author_numbering_equals_label_numbering():
    """An identity-numbered structure leaves contigs and selections unchanged."""
    auth_to_label = _identity_map()
    assert chains_needing_translation(auth_to_label) == []

    assert translate_contig("A1-65", auth_to_label) == "A1-65"
    assert translate_contig("50-120, /0, A1-65", auth_to_label) == "50-120, /0, A1-65"
    assert translate_selection("sel('A/*/[1-4, 14]')", auth_to_label) == "sel('A/*/[1-4, 14]')"
    assert translate_selection("sel('A/*/56/[CG,OH]')", auth_to_label) == "sel('A/*/56/[CG,OH]')"
    # A bare res_id comparison is only rejected when a chain actually needs translating.
    assert translate_selection("res_id > 5", auth_to_label) == "res_id > 5"


def test_translate_contig_rejects_residue_absent_from_structure():
    """An out-of-range contig residue names the chain's span and points at residue_numbering."""
    with pytest.raises(ValueError, match=re.escape("residue A99 is not present in the input structure")) as excinfo:
        translate_contig("A17-99", _offset_map())
    assert "chain A spans 17-81" in str(excinfo.value)
    assert "residue_numbering='label'" in str(excinfo.value)


def test_translate_selection_rejects_bare_res_id_comparison_on_offset_chain():
    """A bare res_id comparison cannot be translated, so it must fail loudly with the workaround."""
    with pytest.raises(ValueError, match="compares 'res_id' directly") as excinfo:
        translate_selection("res_id >= 20 and chain_id == 'A'", _offset_map())
    message = str(excinfo.value)
    assert "residue_numbering='label'" in message
    assert "['A']" in message


@pytest.mark.parametrize(
    ("contig", "message"),
    [
        # An unknown chain lists the chains that do exist.
        ("B17-20", "chain 'B' has no polymer residues"),
        # A token that is not <chain><id>[-<id>] cannot be translated at all.
        ("A17_20", "Malformed preserved-segment contig token 'A17_20'"),
    ],
)
def test_translate_contig_rejects_unresolvable_preserved_segments(contig, message):
    """A preserved segment that cannot be resolved fails before dispatch, naming the cause."""
    with pytest.raises(ValueError, match=re.escape(message)):
        translate_contig(contig, _offset_map())


def test_label_residue_numbering_bypasses_translation():
    """residue_numbering='label' forwards contigs and selections verbatim, with no lookup."""
    auth_to_label = _offset_map()
    # These label ids do not exist in the structure's author numbering (17-81), so a
    # translating spec would raise on them.
    spec = RFD4DesignSpec(
        input_structure=_fixture_structure(),
        contig="A1-65",
        residue_numbering="label",
        hotspots=["sel('A/*/5/[CG,OH]')"],
    )

    # The same ids are not valid author ids for this structure, so the translating path rejects
    # them -- the bypass is what makes the spec below resolve at all.
    with pytest.raises(ValueError, match="residue A1 is not present"):
        translate_contig("A1-65", auth_to_label)

    spec_dict = spec.to_spec_dict(EXAMPLE_INPUT_FIXTURE, auth_to_label)
    assert spec_dict["contig"] == "A1-65"
    assert spec_dict["conditions"]["C_HOT"] == {True: [{"select": "sel('A/*/5/[CG,OH]')"}]}


# ── Conditions compiler ─────────────────────────────────────────────────────


def test_build_conditions_maps_every_typed_field_to_its_upstream_alias():
    """Each typed condition field compiles to {alias: {value: [{"select": ...}]}}."""
    spec = RFD4DesignSpec(
        input_structure=_fixture_structure(),
        fixed_coordinates=["sel('A') and is_protein_backbone"],
        free_coordinates=["sel('A/*/[1-5]')"],
        known_sequence=["sel('A')"],
        free_sequence=["sel('A/*/[10-20]')"],
        hotspots=["sel('A/*/56/[CG,OH]')"],
        rigid_groups={1: ["sel('A/*/*/[N,CA,C,O]')"], 7: ["sel('A/*/9')"]},
        unindexed_groups={"motif": ["sel('A/*/[1-3]')", "sel('A/*/[60-62]')"], "site": ["sel('A/*/30')"]},
        burial={"BURIED": ["sel('A/*/7')"], "EXPOSED": ["sel('A/*/8')"]},
        hbond={"DONOR": ["sel('A/*/11')"]},
        pi_stacking=["sel('A/*/12')"],
        pi_cation={"AROMATIC": ["sel('A/*/13')"]},
        secondary_structure={"HELIX": ["sel('A/*/[14-24]')"]},
        fold_class={"2.60.40": ["sel('A')"]},
        chain_centroid={"12.3,-4.5,0.0": ["sel('A')"]},
        sequence_probabilities=[RFD4SequenceProbability(token="CYS", probability=0.0, select=["sel('A')"])],
    )

    conditions = spec.build_conditions(_identity_map())

    assert conditions["C_CRD"] == {
        False: [{"select": "sel('A/*/[1-5]')"}],
        True: [{"select": "sel('A') and is_protein_backbone"}],
    }
    assert conditions["C_SEQ"] == {
        False: [{"select": "sel('A/*/[10-20]')"}],
        True: [{"select": "sel('A')"}],
    }
    assert conditions["C_HOT"] == {True: [{"select": "sel('A/*/56/[CG,OH]')"}]}
    assert conditions["C_CRDGRP"] == {
        1: [{"select": "sel('A/*/*/[N,CA,C,O]')"}],
        7: [{"select": "sel('A/*/9')"}],
    }
    # C_IDX is a single False bucket whose entries each carry their group label.
    assert conditions["C_IDX"] == {
        False: [
            {"select": "sel('A/*/[1-3]')", "group": "motif"},
            {"select": "sel('A/*/[60-62]')", "group": "motif"},
            {"select": "sel('A/*/30')", "group": "site"},
        ]
    }
    assert conditions["C_RAS"] == {
        "BURIED": [{"select": "sel('A/*/7')"}],
        "EXPOSED": [{"select": "sel('A/*/8')"}],
    }
    assert conditions["C_HB"] == {"DONOR": [{"select": "sel('A/*/11')"}]}
    assert conditions["C_PISTACK"] == {True: [{"select": "sel('A/*/12')"}]}
    assert conditions["C_PICATION"] == {"AROMATIC": [{"select": "sel('A/*/13')"}]}
    assert conditions["C_SEC"] == {"HELIX": [{"select": "sel('A/*/[14-24]')"}]}
    assert conditions["C_CATH"] == {"2.60.40": [{"select": "sel('A')"}]}
    assert conditions["C_CENTROID"] == {"12.3,-4.5,0.0": [{"select": "sel('A')"}]}
    assert conditions["C_SEQ_PROB_CYS"] == {0.0: [{"select": "sel('A')"}]}


def test_build_conditions_translates_selections_to_label_numbering():
    """Compiled condition selections are translated, not forwarded in author numbering."""
    spec = RFD4DesignSpec(
        input_structure=_fixture_structure(),
        hotspots=["sel('A/*/56/[CG,OH]')"],
        unindexed_groups={"motif": ["sel('A/*/[17-20, 30]')"]},
        sequence_probabilities=[RFD4SequenceProbability(token="CYS", probability=0.0, select=["sel('A/*/81')"])],
    )

    conditions = spec.build_conditions(_offset_map())

    assert conditions["C_HOT"] == {True: [{"select": "sel('A/*/40/[CG,OH]')"}]}
    assert conditions["C_IDX"][False] == [{"select": "sel('A/*/[1-4, 14]')", "group": "motif"}]
    assert conditions["C_SEQ_PROB_CYS"] == {0.0: [{"select": "sel('A/*/65')"}]}


def test_build_conditions_groups_sequence_probabilities_by_token_and_value():
    """One alias per token, with one entry list per probability value."""
    spec = RFD4DesignSpec(
        input_structure=_fixture_structure(),
        sequence_probabilities=[
            RFD4SequenceProbability(token="CYS", probability=0.0, select=["sel('A/*/1')"]),
            RFD4SequenceProbability(token="CYS", probability=0.0, select=["sel('A/*/2')"]),
            RFD4SequenceProbability(token="PRO", probability=0.5, select=["sel('A/*/3')"]),
        ],
    )

    conditions = spec.build_conditions(_identity_map())

    assert conditions["C_SEQ_PROB_CYS"] == {0.0: [{"select": "sel('A/*/1')"}, {"select": "sel('A/*/2')"}]}
    assert conditions["C_SEQ_PROB_PRO"] == {0.5: [{"select": "sel('A/*/3')"}]}


def test_build_conditions_emission_order_is_semantic():
    """Order matters upstream: clear-before-regroup, last-write-wins, unindex against settled masks."""
    spec = RFD4DesignSpec(
        input_structure=_fixture_structure(),
        fixed_coordinates=["sel('A/*/[10-20]')"],
        free_coordinates=["sel('A/*/[1-5]')"],
        rigid_groups={1: ["sel('A/*/*/[N,CA,C,O]')"]},
        unindexed_groups={"motif": ["sel('A/*/[1-3]')"]},
        hotspots=["sel('A/*/56')"],
        raw_conditions={"C_SALT": {True: [{"select": "is_protein"}]}},
    )

    conditions = spec.build_conditions(_identity_map())
    order = list(conditions)

    # Absolute coordinate targets must be cleared before atoms move into an IC group.
    assert order.index("C_CRD") < order.index("C_CRDGRP")
    # Within C_CRD, clearing (False) precedes fixing (True) for the same reason.
    assert list(conditions["C_CRD"]) == [False, True]
    # Unindexing is stamped against the settled index mask, so it is emitted last of the
    # typed conditions -- only the raw_conditions passthrough may follow it.
    assert order[-2:] == ["C_IDX", "C_SALT"]


def test_raw_conditions_merge_last_and_override_a_typed_condition():
    """raw_conditions is the escape hatch, so it wins over a typed field compiled earlier."""
    override = {"C_HOT": {True: [{"select": "is_ligand"}]}}
    spec = RFD4DesignSpec(
        input_structure=_fixture_structure(),
        hotspots=["sel('A/*/56/[CG,OH]')"],
        raw_conditions=override,
    )

    conditions = spec.build_conditions(_identity_map())

    assert conditions["C_HOT"] == override["C_HOT"]


def test_to_spec_dict_emits_only_upstream_keys_and_omits_empty_ones():
    """RFD4 accepts exactly input, input_as_annotated_cif, contig, and conditions."""
    upstream_keys = {"input", "input_as_annotated_cif", "contig", "conditions"}

    bare = RFD4DesignSpec(input_structure=_fixture_structure())
    bare_dict = bare.to_spec_dict(EXAMPLE_INPUT_FIXTURE, _identity_map())
    assert bare_dict == {"input": str(EXAMPLE_INPUT_FIXTURE)}

    full = RFD4DesignSpec(
        input_structure=_fixture_structure(),
        input_as_annotated_cif=True,
        contig="A17-81",
        known_sequence=["sel('A')"],
    )
    full_dict = full.to_spec_dict(EXAMPLE_INPUT_FIXTURE, _offset_map())
    assert set(full_dict) <= upstream_keys
    assert full_dict["input_as_annotated_cif"] is True
    assert full_dict["contig"] == "A1-65"
    assert full_dict["conditions"] == {"C_SEQ": {True: [{"select": "sel('A')"}]}}

    # input_as_annotated_cif=False is the default and must not be emitted.
    assert "input_as_annotated_cif" not in RFD4DesignSpec(
        input_structure=_fixture_structure(), input_as_annotated_cif=False
    ).to_spec_dict(EXAMPLE_INPUT_FIXTURE, _identity_map())


def test_write_spec_file_writes_yaml_with_native_condition_value_keys(tmp_path):
    """Typed specs are YAML so bool/int/float condition values stay native mapping keys.

    JSON would stringify them ("true", "1", "0.0"), which upstream's parser reads as labels
    rather than values -- that is the whole reason this path is not JSON.
    """
    inputs = RFD4Input(
        design_specs=[
            RFD4DesignSpec(
                input_structure=_fixture_structure(),
                contig="A1-65",
                fixed_coordinates=["sel('A')"],
                free_coordinates=["sel('A/*/1')"],
                rigid_groups={7: ["sel('A/*/*/[N,CA,C,O]')"]},
                sequence_probabilities=[RFD4SequenceProbability(token="CYS", probability=0.0, select=["sel('A')"])],
            ),
            RFD4DesignSpec(input_structure=_fixture_structure()),
        ]
    )

    spec_path, spec_keys = inputs.write_spec_file(tmp_path)

    assert spec_path.name == "rfd4_input.yaml"
    assert spec_keys == ["spec-0", "spec-1"]

    loaded = yaml.safe_load(spec_path.read_text())
    assert list(loaded) == spec_keys
    coordinate_keys = list(loaded["spec-0"]["conditions"]["C_CRD"])
    assert coordinate_keys == [False, True]
    assert all(isinstance(key, bool) for key in coordinate_keys)
    group_key = next(iter(loaded["spec-0"]["conditions"]["C_CRDGRP"]))
    assert group_key == 7 and isinstance(group_key, int) and not isinstance(group_key, bool)
    probability_key = next(iter(loaded["spec-0"]["conditions"]["C_SEQ_PROB_CYS"]))
    assert probability_key == 0.0 and isinstance(probability_key, float)

    # Each spec's structure is materialized as a prepared CIF that its 'input' points at.
    for spec_key in spec_keys:
        cif_path = tmp_path / f"{spec_key}_input.cif"
        assert loaded[spec_key]["input"] == str(cif_path)
        assert cif_path.is_file()
        assert {row[0] for row in _atom_site_rows(cif_path.read_text(), "label_asym_id")} == {"A"}


def test_write_spec_file_passes_an_annotated_cif_through_byte_for_byte(tmp_path):
    """``input_as_annotated_cif`` must copy the input text verbatim and skip translation.

    The flag exists to preserve the conditioning a previous RFD4 output carries in custom
    mmCIF columns (``_atom_site.condition_*``, ``_condition_coordinate_1_atom``). A gemmi
    round-trip silently drops every one of them, so routing this path through
    ``prepare_rfd4_cif`` would leave the caller with an unconditioned structure and no error.
    """
    annotated = (
        EXAMPLE_INPUT_FIXTURE.read_text()
        + "\nloop_\n_condition_coordinate_1_atom.id\n_condition_coordinate_1_atom.val\n1 1\n"
    )
    inputs = RFD4Input(
        design_specs=[
            RFD4DesignSpec(
                input_structure=Structure(structure=annotated, structure_format="cif"),
                input_as_annotated_cif=True,
                # Absent from the fixture's author numbering (1-65), so translating would raise.
                contig="A17-81",
            )
        ]
    )

    spec_path, spec_keys = inputs.write_spec_file(tmp_path)

    written = tmp_path / f"{spec_keys[0]}_input.cif"
    assert written.read_text() == annotated
    loaded = yaml.safe_load(spec_path.read_text())[spec_keys[0]]
    assert loaded["input_as_annotated_cif"] is True
    # Forwarded untranslated even at the default residue_numbering='auth'.
    assert loaded["contig"] == "A17-81"


def test_annotated_cif_flag_is_rejected_on_a_pdb_input():
    """PDB cannot carry the condition columns the flag promises to preserve."""
    as_pdb = Structure(structure=_fixture_structure().structure_pdb, structure_format="pdb")

    with pytest.raises(ValidationError, match="must be a CIF"):
        RFD4DesignSpec(input_structure=as_pdb, input_as_annotated_cif=True)


def test_write_spec_file_writes_raw_json_as_json_in_input_order(tmp_path):
    """raw_json is forwarded as JSON, and global_args is not a design id."""
    raw = json.dumps(
        {
            "global_args": {"num_replicates": 2},
            "binder": {"input": "target.cif", "contig": "50-120, /0, A1-65"},
            "monomer": {"input": "target.cif"},
        }
    )

    spec_path, spec_keys = RFD4Input(raw_json=raw).write_spec_file(tmp_path)

    assert spec_path.name == "rfd4_input.json"
    assert spec_keys == ["binder", "monomer"]
    assert json.loads(spec_path.read_text()) == json.loads(raw)


# ── Validators ──────────────────────────────────────────────────────────────


def test_design_spec_rejects_contig_without_preserved_segment():
    """RFD4 rejects an all-expandable contig: the input structure exists to be referenced."""
    with pytest.raises(ValidationError, match="has no preserved segment"):
        RFD4DesignSpec(input_structure=_fixture_structure(), contig="50-120")
    with pytest.raises(ValidationError, match="has no preserved segment"):
        RFD4DesignSpec(input_structure=_fixture_structure(), contig="50-120, /0, E1:2-20R")

    # One preserved segment is enough.
    RFD4DesignSpec(input_structure=_fixture_structure(), contig="50-120, /0, A1-65")


def test_design_spec_rejects_malformed_contig_tokens():
    r"""Duplicate expandable names, the reserved <M> name, empty tokens, and '\\0' all fail."""
    structure = _fixture_structure()

    with pytest.raises(ValidationError, match="reuses expandable-segment name"):
        RFD4DesignSpec(input_structure=structure, contig="A1-65, EX1:10-20, EX1:30-40")
    with pytest.raises(ValidationError, match=re.escape("'<M>' is reserved internally by RFD4")):
        RFD4DesignSpec(input_structure=structure, contig="A1-65, <M>:10-20")
    with pytest.raises(ValidationError, match="has an empty token"):
        RFD4DesignSpec(input_structure=structure, contig="A1-65,")
    with pytest.raises(ValidationError, match="has an empty token"):
        RFD4DesignSpec(input_structure=structure, contig="A1-65, , 20-30")
    with pytest.raises(ValidationError, match=re.escape("use '/0' for RFD4 chain breaks")):
        RFD4DesignSpec(input_structure=structure, contig=r"A1-65, \0, 20-30")


def test_design_spec_rejects_reserved_and_out_of_range_rigid_group_ids():
    """0 (absolute) and -1 (unknown) are reserved coordinate-group ids; the max is 32767."""
    structure = _fixture_structure()
    selection = ["sel('A/*/*/[N,CA,C,O]')"]

    for group_id in (0, -1, 32768):
        with pytest.raises(ValidationError, match=f"rigid_groups id {group_id} is out of range"):
            RFD4DesignSpec(input_structure=structure, rigid_groups={group_id: selection})

    # 32767 is upstream's own maximum group id, so it has to stay accepted.
    for group_id in (1, 32767):
        RFD4DesignSpec(input_structure=structure, rigid_groups={group_id: selection})


def test_design_spec_rejects_chain_centroid_keys_that_are_not_three_numbers():
    """A centroid key is an 'x,y,z' coordinate, so wrong arity or non-numbers must fail."""
    structure = _fixture_structure()

    with pytest.raises(ValidationError, match="must be 'x,y,z' with exactly three components"):
        RFD4DesignSpec(input_structure=structure, chain_centroid={"1.0,2.0": ["sel('A')"]})
    with pytest.raises(ValidationError, match="must be 'x,y,z' with exactly three components"):
        RFD4DesignSpec(input_structure=structure, chain_centroid={"1.0,2.0,3.0,4.0": ["sel('A')"]})
    with pytest.raises(ValidationError, match="must contain three numbers"):
        RFD4DesignSpec(input_structure=structure, chain_centroid={"1.0,2.0,here": ["sel('A')"]})

    RFD4DesignSpec(input_structure=structure, chain_centroid={"12.3,-4.5,0": ["sel('A')"]})


def test_design_spec_rejects_non_cath_fold_class_keys():
    """fold_class keys are three-level C.A.T labels with 'x' only as a trailing suffix."""
    structure = _fixture_structure()

    with pytest.raises(ValidationError, match=re.escape("must be a three-level C.A.T label")):
        RFD4DesignSpec(input_structure=structure, fold_class={"2.60": ["sel('A')"]})
    with pytest.raises(ValidationError, match=re.escape("may only use 'x' as a trailing suffix")):
        RFD4DesignSpec(input_structure=structure, fold_class={"1.x.40": ["sel('A')"]})
    with pytest.raises(ValidationError, match="non-numeric level"):
        RFD4DesignSpec(input_structure=structure, fold_class={"1.60.abc": ["sel('A')"]})

    # A concrete label, a partially cleared one, and the fully cleared one all stay legal.
    for key in ("2.60.40", "1.x.x", "x.x.x"):
        RFD4DesignSpec(input_structure=structure, fold_class={key: ["sel('A')"]})


def test_raw_conditions_rejects_unsupported_aliases_and_legacy_dialect_fields():
    """The escape hatch cannot reach saved payloads, 2-body conditions, or removed spec fields."""
    structure = _fixture_structure()
    payload = {True: [{"select": "sel('A')"}]}

    with pytest.raises(ValidationError, match="raw_conditions may not set C_INTCRD") as excinfo:
        RFD4DesignSpec(input_structure=structure, raw_conditions={"C_INTCRD": payload})
    assert "rigid_groups" in str(excinfo.value)

    for alias in ("C_BP", "C_DIS", "C_CHA"):
        with pytest.raises(ValidationError, match=f"raw_conditions may not set {alias}") as excinfo:
            RFD4DesignSpec(input_structure=structure, raw_conditions={alias: payload})
        assert "2-body" in str(excinfo.value)

    expected_replacements = {
        "length": "contig",
        "symmetry": "rfdiffusion3-design",
        "select_hotspots": "hotspots",
    }
    for field_name, replacement in expected_replacements.items():
        with pytest.raises(ValidationError, match="removed pre-RFD4 contig-JSON dialect") as excinfo:
            RFD4DesignSpec(input_structure=structure, raw_conditions={field_name: payload})
        assert replacement in str(excinfo.value)

    # A condition without a typed field still passes through.
    RFD4DesignSpec(input_structure=structure, raw_conditions={"C_SALT": payload})


def test_input_rejects_raw_json_naming_a_removed_dialect_field():
    """A pre-RFD4 dialect field makes upstream reject the whole file, so catch it here."""
    raw = json.dumps({"global_args": {"num_replicates": 1}, "design-0": {"input": "t.cif", "symmetry": "C3"}})

    with pytest.raises(ValidationError, match="names removed pre-RFD4 dialect field") as excinfo:
        RFD4Input(raw_json=raw)
    assert "symmetry" in str(excinfo.value)

    # global_args is not a design, so its keys are not checked against the dialect list.
    RFD4Input(raw_json=json.dumps({"global_args": {"length": 100}, "design-0": {"input": "t.cif", "contig": "A1-65"}}))


def test_design_spec_rejects_sequence_probabilities_summing_above_one():
    """Probabilities stacked on one selection must leave mass for the unconstrained identities."""
    structure = _fixture_structure()
    constraints = [
        RFD4SequenceProbability(token="CYS", probability=0.7, select=["sel('A/*/1')"]),
        RFD4SequenceProbability(token="PRO", probability=0.6, select=["sel('A/*/1')"]),
    ]

    with pytest.raises(ValidationError, match="sequence_probabilities sum above 1") as excinfo:
        RFD4DesignSpec(input_structure=structure, sequence_probabilities=constraints)
    assert "sel('A/*/1')" in str(excinfo.value)

    # Exactly 1 on one selection is legal, and the same tokens on disjoint selections never stack.
    RFD4DesignSpec(
        input_structure=structure,
        sequence_probabilities=[
            RFD4SequenceProbability(token="CYS", probability=0.4, select=["sel('A/*/1')"]),
            RFD4SequenceProbability(token="PRO", probability=0.6, select=["sel('A/*/1')"]),
        ],
    )
    RFD4DesignSpec(
        input_structure=structure,
        sequence_probabilities=[
            RFD4SequenceProbability(token="CYS", probability=0.7, select=["sel('A/*/1')"]),
            RFD4SequenceProbability(token="PRO", probability=0.6, select=["sel('A/*/2')"]),
        ],
    )


def test_config_rejects_custom_nsteps_with_a_recycle_strategy():
    """A recycle_strategy preset expands to exactly 60 steps, so it pins nsteps.

    Upstream only raises "recycle schedule expands to 60 steps; expected N" after the
    model has loaded, so the wrapper enforces the invariant before dispatch.
    """
    with pytest.raises(ValidationError, match="expands to a fixed 60-step schedule"):
        RFD4DesignConfig(nsteps=20, recycle_strategy="low/flat")
    with pytest.raises(ValidationError, match="expands to a fixed 60-step schedule"):
        RFD4InverseFoldingConfig(nsteps=30)

    # Switching recycling off is what makes a custom nsteps legal.
    RFD4DesignConfig(nsteps=20, recycle_strategy=None)
    # minimal() takes exactly that route, so the smoke path stays both cheap and valid.
    minimal = RFD4DesignConfig.minimal()
    assert minimal.recycle_strategy is None
    assert minimal.nsteps < 60


def test_config_rejects_trajectory_dump_without_output_dir(tmp_path):
    """Trajectory frames land beside the designs, so a tempdir run would discard them."""
    with pytest.raises(ValidationError, match="needs 'output_dir' set"):
        RFD4DesignConfig(save_trajectory_every=5)
    RFD4DesignConfig(save_trajectory_every=5, output_dir=str(tmp_path))


def test_design_config_rejects_conflicting_partial_diffusion_settings():
    """Every partial-diffusion rule upstream enforces is enforced here first.

    Upstream resolves these in ``rfproteina.search.vanilla`` after the checkpoint has
    loaded, so each invalid combination otherwise costs a full model load to discover.
    """
    # The simple fraction excludes both per-modality levels and the coupling flag.
    with pytest.raises(ValidationError, match="mutually exclusive") as excinfo:
        RFD4DesignConfig(partial_trajectory_fraction_steps=0.3, partial_t_backbone=0.8)
    assert "partial_t_backbone" in str(excinfo.value)
    with pytest.raises(ValidationError, match="mutually exclusive"):
        RFD4DesignConfig(partial_trajectory_fraction_steps=0.3, partial_t_latent=0.8)
    with pytest.raises(ValidationError, match="mutually exclusive") as excinfo:
        RFD4DesignConfig(partial_trajectory_fraction_steps=0.3, couple_partial_t=True)
    assert "couple_partial_t" in str(excinfo.value)

    # A fraction that rounds to zero denoising steps is rejected against the real nsteps.
    with pytest.raises(ValidationError, match="rounds to zero denoising steps"):
        RFD4DesignConfig(partial_trajectory_fraction_steps=0.005)

    # couple_partial_t is set exactly when one of the two levels is given.
    with pytest.raises(ValidationError, match=re.escape("only 'partial_t_backbone' is set")):
        RFD4DesignConfig(partial_t_backbone=0.8)
    with pytest.raises(ValidationError, match="requires exactly one"):
        RFD4DesignConfig(partial_t_backbone=0.8, partial_t_latent=0.8, couple_partial_t=True)
    with pytest.raises(ValidationError, match="requires 'partial_t_backbone' or 'partial_t_latent'"):
        RFD4DesignConfig(couple_partial_t=True)

    # The three valid shapes.
    RFD4DesignConfig(partial_t_backbone=0.8, partial_t_latent=0.8)
    RFD4DesignConfig(partial_t_backbone=0.8, couple_partial_t=True)
    RFD4DesignConfig(partial_trajectory_fraction_steps=0.3)


def test_input_requires_design_specs_or_raw_json():
    """An input with neither design_specs nor raw_json has nothing to dispatch."""
    with pytest.raises(ValidationError, match=re.escape("Either 'design_specs' (non-empty) or 'raw_json'")):
        RFD4Input()
    with pytest.raises(ValidationError, match=re.escape("Either 'design_specs' (non-empty) or 'raw_json'")):
        RFD4Input(design_specs=[])

    assert len(RFD4Input(design_specs=[RFD4DesignSpec(input_structure=_fixture_structure())])) == 1
    assert len(RFD4Input(raw_json=json.dumps({"a": {"input": "t.cif"}, "b": {"input": "t.cif"}}))) == 2


def test_input_len_counts_the_specs_that_are_actually_dispatched(tmp_path):
    """``len(inputs)`` is the design count, so ``global_args`` must not be counted as a design.

    ``write_spec_file`` already excludes it, and the dispatched ``spec_keys`` -- and therefore
    the per-spec output bundles -- come from there. A ``len()`` that disagrees misreports the
    run size to everything that sizes work from the input (chunking, progress, cardinality).
    """
    raw = json.dumps({"global_args": {"num_replicates": 2}, "a": {"input": "t.cif"}, "b": {"input": "t.cif"}})
    inputs = RFD4Input(raw_json=raw)

    _, spec_keys = inputs.write_spec_file(tmp_path)

    assert spec_keys == ["a", "b"]
    assert len(inputs) == len(spec_keys)


def test_remote_unsupported_reason_flags_local_adapters_and_bundles(tmp_path):
    """Local adapter and model-bundle directories do not exist on a hosted worker."""
    reason = RFD4DesignConfig(adapter_path=str(tmp_path)).remote_unsupported_reason("proto")
    assert reason is not None and "adapter_path" in reason

    bundle_reason = RFD4DesignConfig(model_checkpoint=str(tmp_path)).remote_unsupported_reason("proto")
    assert bundle_reason is not None and "local directory" in bundle_reason

    # The default pinned HuggingFace repo id is fine remotely.
    assert RFD4DesignConfig().remote_unsupported_reason("proto") is None
    assert RFD4InverseFoldingConfig().remote_unsupported_reason("modal") is None


# ── Config: CLI override construction ───────────────────────────────────────


def test_common_config_cli_kwargs_pins_wrapper_owned_overrides(tmp_path):
    """metrics_tags and skip_existing are hardcoded off, and compile/recycle use Hydra's 'null'."""
    kwargs = RFD4DesignConfig().get_cli_kwargs()

    # No AlphaFold oracle is provisioned, and the wrapper owns a fresh output dir per call.
    assert kwargs["metrics_tags"] == []
    assert kwargs["skip_existing"] is False
    # Upstream's untagged unknown_residue_filter still runs, and deleting its failures exits 0.
    assert kwargs["delete_failing_pre_oracle"] is False
    # Compilation is off by default, which Hydra expresses as a literal "null".
    assert kwargs["compile_lengths"] == "null"
    assert "compile_lengths" not in RFD4DesignConfig(use_torch_compile=True).get_cli_kwargs()

    # recycle_strategy is a Hydra config *group*: switching it off means deleting the
    # default (~recycle_strategy), because a group override rejects a null value.
    off = RFD4DesignConfig(recycle_strategy=None, nsteps=20).get_cli_kwargs()
    assert "~recycle_strategy" in off
    assert "recycle_strategy" not in off
    assert RFD4DesignConfig(recycle_strategy="low/flat").get_cli_kwargs()["recycle_strategy"] == "low/flat"
    assert RFD4DesignConfig(model_revision=None).get_cli_kwargs()["hf_ckpt_revision"] == "null"

    # adapter_path is only forwarded when set, so an unset adapter never reaches Hydra.
    assert "adapter_path" not in kwargs
    adapter = str(tmp_path / "adapter")
    assert RFD4DesignConfig(adapter_path=adapter).get_cli_kwargs()["adapter_path"] == adapter


def test_common_config_cli_kwargs_deterministic_sampling_switches_both_modalities():
    """ODE integration must be set on the backbone and the latent modality together."""
    ode_keys = ("generation.model.bb_ca.gt.mode", "generation.model.local_latents.gt.mode")

    kwargs = RFD4DesignConfig(deterministic_sampling=True).get_cli_kwargs()
    assert all(kwargs[key] == "ode" for key in ode_keys)

    default_kwargs = RFD4DesignConfig().get_cli_kwargs()
    assert not any(key in default_kwargs for key in ode_keys)


def test_sampler_tuning_emits_per_modality_simulation_step_params():
    """Each sampler block writes to its own modality's simulation_step_params path."""
    kwargs = RFD4DesignConfig(
        backbone_sampling={"tsr_k": 8.0, "tsr_sigma": 2.0},
        sequence_sampling={"tsr_k": 2.0, "noise_scale": 0.0},
    ).get_cli_kwargs()

    assert kwargs["generation.model.bb_ca.simulation_step_params.tsr_k"] == 8.0
    assert kwargs["generation.model.bb_ca.simulation_step_params.tsr_sigma"] == 2.0
    assert kwargs["generation.model.local_latents.simulation_step_params.tsr_k"] == 2.0
    # noise_scale is renamed upstream, and 0.0 is a meaningful value that must survive the
    # None-filtering applied to the override dict.
    assert kwargs["generation.model.local_latents.simulation_step_params.sc_scale_noise"] == 0.0

    # Unset sampler fields inherit the checkpoint's values rather than being forwarded.
    assert not any(".simulation_step_params." in key for key in RFD4DesignConfig().get_cli_kwargs())


def test_design_config_cli_kwargs_adds_guidance_and_partial_diffusion():
    """Design-only overrides land under generation.args.* and omit unset partial-diffusion knobs."""
    kwargs = RFD4DesignConfig(guidance_w=2.0, partial_trajectory_fraction_steps=0.3).get_cli_kwargs()

    assert kwargs["generation.args.guidance_w"] == 2.0
    assert kwargs["generation.args.partial_trajectory_fraction_steps"] == 0.3
    assert "generation.args.partial_t_backbone" not in kwargs
    assert "generation.args.couple_partial_t" not in kwargs

    coupled = RFD4DesignConfig(partial_t_backbone=0.8, couple_partial_t=True).get_cli_kwargs()
    assert coupled["generation.args.partial_t_backbone"] == 0.8
    assert coupled["generation.args.couple_partial_t"] is True


def test_inverse_folding_config_cli_kwargs_selects_preset_without_guidance():
    """The inverse_folding preset carries the mode; guidance is a design-only control."""
    kwargs = RFD4InverseFoldingConfig(num_replicates=4, sequence_sampling={"tsr_k": 2.0}).get_cli_kwargs()

    assert kwargs["generation"] == "inverse_folding"
    assert "generation.args.guidance_w" not in kwargs
    # The shared overrides still apply.
    assert kwargs["num_replicates"] == 4
    assert kwargs["metrics_tags"] == []
    assert kwargs["generation.model.local_latents.simulation_step_params.tsr_k"] == 2.0


# ── Dispatch payload ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("runner", "config_class", "operation"),
    [
        (run_rfd4_design, RFD4DesignConfig, "design"),
        (run_rfd4_inverse_folding, RFD4InverseFoldingConfig, "inverse_folding"),
    ],
)
def test_dispatch_payload_names_the_operation_and_the_written_spec_keys(monkeypatch, runner, config_class, operation):
    """The payload must name the operation and the spec keys the subprocess maps designs back onto.

    ``spec_keys`` is the only link between upstream's ``<spec_key>_<replicate>`` example ids and
    the input order the output bundles follow, so it has to match the keys written into the spec
    file exactly.
    """
    captured = _capture_dispatch(monkeypatch, ["spec-0", "spec-1"])
    inputs = RFD4Input(
        design_specs=[
            RFD4DesignSpec(input_structure=_fixture_structure()),
            RFD4DesignSpec(input_structure=_fixture_structure(), contig="A1-65"),
        ]
    )

    result = runner(inputs, config_class(recycle_strategy=None, nsteps=20))

    assert captured["operation"] == operation
    assert captured["spec_keys"] == captured["_spec_file_keys"] == ["spec-0", "spec-1"]
    # recycle_strategy=None is a group deletion, so the bare '~' key must reach the payload.
    assert "~recycle_strategy" in captured
    assert "recycle_strategy" not in captured
    assert [bundle.spec_key for bundle in result] == ["spec-0", "spec-1"]


def test_dispatched_operations_are_accepted_by_the_standalone_dispatcher():
    """The operation literals are a cross-file contract: a rename is otherwise GPU-only to find."""
    assert {"design", "inverse_folding"} <= set(_standalone_inference()._VALID_OPERATIONS)


def test_dispatch_nests_a_fresh_run_directory_under_a_reused_output_dir(tmp_path, monkeypatch):
    """Each run needs its own subdirectory of ``output_dir``, not ``output_dir`` itself.

    ``_collect_designs`` globs the whole output root it is given, so a second run against a
    reused ``output_dir`` would harvest the first run's designs and return them as its own.
    """
    captured = _capture_dispatch(monkeypatch, ["spec-0"])
    inputs = RFD4Input(design_specs=[RFD4DesignSpec(input_structure=_fixture_structure())])

    run_rfd4_design(inputs, RFD4DesignConfig(output_dir=str(tmp_path)))
    first = Path(captured["output_dir"])
    run_rfd4_design(inputs, RFD4DesignConfig(output_dir=str(tmp_path)))
    second = Path(captured["output_dir"])

    assert first.parent == second.parent == tmp_path
    assert first != second
    assert first.name.startswith("run-") and second.name.startswith("run-")


# ── Standalone inference module ─────────────────────────────────────────────


def test_standalone_format_override_renders_hydra_tokens():
    """``rfd4-generate`` is a Hydra app: every argument is a key=value token, never a --flag."""
    inference = _standalone_inference()

    # 'recycle_strategy=null' fails Hydra composition; deleting the group is the only way off.
    assert inference._format_override("~recycle_strategy", True) == "~recycle_strategy"
    # Lists are JSON so an empty one stays an empty list literal rather than a string.
    assert inference._format_override("metrics_tags", []) == "metrics_tags=[]"
    assert inference._format_override("metrics_tags", ["af2"]) == 'metrics_tags=["af2"]'
    # Hydra bools are lowercase; Python's str() would emit True/False.
    assert inference._format_override("prevalidate_inputs", True) == "prevalidate_inputs=true"
    assert inference._format_override("skip_existing", False) == "skip_existing=false"
    assert inference._format_override("generation.args.nsteps", 20) == "generation.args.nsteps=20"


def test_standalone_drops_ephemeral_paths_from_a_metrics_row():
    """Run-local paths must not reach design metadata, or two identical runs differ.

    The wrapper runs in a fresh temporary directory per call, so every path upstream records
    is unique to that run: leaking them breaks output comparison and seeded reproducibility
    (and hands callers host paths). Both rules are needed -- ``dataset_name`` is caught only
    by its absolute value, and a relative ``*_path`` only by its key.
    """
    inference = _standalone_inference()

    row = {
        "example_id": "spec-0_0",
        "designed_structure_path": "inference_samples/spec-0_0/spec-0_0_generated.cif",
        "trajectory_dir": "frames",
        "dataset_name": "/ephemeral/run-abc/inputs/rfd4_input.yaml",
        "plddt": "0.91",
    }

    assert inference._drop_ephemeral_paths(row) == {"example_id": "spec-0_0", "plddt": "0.91"}


def test_standalone_split_example_id_prefers_the_longest_matching_spec_key():
    """A caller's own spec key may end in '_3', so the split is longest-prefix, not rsplit('_', 1)."""
    inference = _standalone_inference()
    spec_keys = ["spec-0", "spec-0_3"]

    assert inference._split_example_id("spec-0_3_1", spec_keys) == ("spec-0_3", 1)
    assert inference._split_example_id("spec-0_3", spec_keys) == ("spec-0", 3)

    with pytest.raises(inference.RFD4ExecutionError, match="does not match any dispatched spec key"):
        inference._split_example_id("other-0_0", spec_keys)


def test_standalone_strip_structure_suffix_keeps_dots_inside_the_example_id():
    """Only a structure suffix is stripped, so a dotted example id survives it."""
    inference = _standalone_inference()

    assert inference._strip_structure_suffix("design.v2_0_generated.cif.zst") == "design.v2_0_generated"
    assert inference._strip_structure_suffix("design.v2_0_generated.cif") == "design.v2_0_generated"


def test_standalone_collect_designs_orders_designs_and_merges_metrics(tmp_path, caplog):
    """Designs are discovered from the output tree, since RFD4 writes no manifest.

    Covers the whole harvest contract at once: only ``*_generated`` files are designs, the
    result follows dispatched spec order rather than glob order, metrics merge in without
    their run-local paths, and an unpinned revision reported to Hydra as "null" comes back
    as ``None``.
    """
    inference = _standalone_inference()
    samples = tmp_path / "inference_samples"
    # Written out of spec order on purpose.
    for example_id in ("spec-1_0", "spec-0_1", "spec-0_0"):
        (samples / example_id).mkdir(parents=True)
        (samples / example_id / f"{example_id}_generated.cif").write_text(f"data_{example_id}\n")
    # Upstream also copies the run's input beside the design; it is not a design.
    (samples / "spec-0_0" / "spec-0_0_input.cif").write_text("data_decoy\n")
    metrics_dir = tmp_path / "inference_metrics"
    metrics_dir.mkdir()
    (metrics_dir / "per_example_metrics.csv").write_text(
        "example_id,designed_structure_path,plddt,seed\nspec-0_0,/ephemeral/run-abc/spec-0_0.cif,0.91,4242\n"
    )

    with caplog.at_level(logging.WARNING):
        designs = inference.RFD4Model()._collect_designs(
            tmp_path,
            ["spec-0", "spec-1"],
            None,
            {"hf_ckpt_path": "nvidia/RFD4-Proteina", "hf_ckpt_revision": "null", "num_replicates": 2},
        )

    assert [(design["spec_key"], design["design_index"]) for design in designs] == [
        ("spec-0", 0),
        ("spec-0", 1),
        ("spec-1", 0),
    ]
    assert [design["structure_content"] for design in designs] == [
        "data_spec-0_0\n",
        "data_spec-0_1\n",
        "data_spec-1_0\n",
    ]

    metadata = designs[0]["metadata"]
    assert metadata["plddt"] == "0.91"
    assert "designed_structure_path" not in metadata
    assert metadata["checkpoint"] == "nvidia/RFD4-Proteina"
    assert metadata["checkpoint_revision"] is None
    # seed=None means upstream drew its own and recorded it; that real seed must survive.
    assert metadata["seed"] == "4242"
    assert designs[1]["metadata"]["seed"] is None

    # 2 specs x 2 replicates were dispatched, so a 3-design harvest is a silent upstream drop.
    assert "harvested 3" in caplog.text


def test_standalone_failure_message_includes_stdout_as_well_as_stderr(monkeypatch, tmp_path):
    """``rfd4-generate`` logs through Hydra, which writes to stdout; stderr alone is usually empty.

    Reporting only stderr turns most upstream failures (a rejected spec, a missing checkpoint)
    into a bare exit code with no diagnosis.
    """
    inference = _standalone_inference()
    helpers = importlib.import_module("standalone_helpers")
    monkeypatch.setattr(helpers, "set_torch_seed", lambda seed: None)
    monkeypatch.setattr(helpers, "get_subprocess_device_env", lambda device: {})
    monkeypatch.setattr(
        helpers,
        "run_teed",
        lambda cmd, env=None, verbose=False: (1, "Error executing job: contig has no preserved segment", "Traceback"),
    )
    model = inference.RFD4Model()
    model._loaded = True
    model.rfd4_executable = "rfd4-generate"

    with pytest.raises(inference.RFD4ExecutionError, match="exit 1") as excinfo:
        model(spec_path=str(tmp_path / "rfd4_input.yaml"), output_dir=str(tmp_path), spec_keys=["spec-0"])

    message = str(excinfo.value)
    assert "contig has no preserved segment" in message
    assert "Traceback" in message


def test_standalone_collect_designs_rejects_an_output_tree_with_no_designs(tmp_path):
    """Upstream can exit 0 having written nothing, which must be an error rather than an empty run."""
    inference = _standalone_inference()

    with pytest.raises(inference.RFD4ExecutionError, match=re.escape("no '*_generated.cif' outputs found")):
        inference.RFD4Model()._collect_designs(tmp_path, ["spec-0"], None, {})


# ── Output bundling and export ──────────────────────────────────────────────


def test_build_designs_buckets_designs_per_spec_and_backfills_empty_bundles():
    """Bundles follow input order, and a spec that produced nothing gets an empty bundle.

    Without the backfill, a barren spec would shift every later bundle, breaking the 1:1
    cardinality the framework relies on between design_specs and designed_structures.
    """
    payload = _design_payload(["spec-2", "spec-0", "spec-2"])
    payload["designs"][1]["metadata"] = {"seed": 7}

    bundles = build_designs(payload, ["spec-0", "spec-1", "spec-2"], "rfd4-design")

    assert [(bundle.spec_key, len(bundle)) for bundle in bundles] == [("spec-0", 1), ("spec-1", 0), ("spec-2", 2)]
    assert bundles[0][0].metadata == {"seed": 7}
    assert bundles[2][0].metadata == {}
    # Each design carries a structure and a predictor-ready Complex with its sequence.
    design = bundles[0][0]
    assert design.structure.source == "rfd4-design"
    assert len(design.complex.chains[0].sequence) == _FIXTURE_LENGTH


def test_rfd4_output_iterates_per_spec_bundles():
    """len/getitem/iter walk bundles, not individual designs."""
    bundles = build_designs(_design_payload(["spec-0", "spec-1", "spec-1"]), ["spec-0", "spec-1"], "rfd4-design")
    output = RFD4Output(designed_structures=bundles)

    assert len(output) == 2
    assert output[1].spec_key == "spec-1"
    assert [bundle.spec_key for bundle in output] == ["spec-0", "spec-1"]
    assert [len(bundle) for bundle in output] == [1, 2]
    assert isinstance(output[1][0], RFD4Design)
    assert "2 bundles, 3 designs" in repr(output)


@pytest.mark.parametrize("file_format", ["cif", "pdb"])
def test_export_formats(tmp_path, file_format):
    """Export writes one <spec_key>_design_<index>.<format> file per design."""
    bundles = build_designs(_design_payload(["spec-0", "spec-1", "spec-1"]), ["spec-0", "spec-1"], "rfd4-design")
    output = RFD4Output(designed_structures=bundles)

    output._export_output(tmp_path, file_format)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"spec-0_design_0.{file_format}",
        f"spec-1_design_0.{file_format}",
        f"spec-1_design_1.{file_format}",
    ]
    assert all(path.stat().st_size > 0 for path in tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
def test_rfd4_design_scaffolds_a_motif():
    """Design extends a preserved motif with a de novo segment and returns a sequenced design."""
    inputs = RFD4Input(
        design_specs=[
            RFD4DesignSpec(input_structure=_fixture_structure(), contig="A1-20, 20-30", known_sequence=["sel('A')"])
        ]
    )
    # low/flat is the cheapest in-distribution preset (120 trunk passes). nsteps stays at
    # its default 60: a preset pins it, and a shorter de novo trajectory collapses to
    # poly-alanine, which would make the sequence assertions below meaningless.
    config = RFD4DesignConfig(num_replicates=1, batch_size=1, recycle_strategy="low/flat")

    result = run_rfd4_design(inputs, config)

    validate_output(result)
    assert result.success is True
    assert result.tool_id == "rfd4-design"
    # One bundle per input spec, in input order.
    assert len(result) == len(inputs.design_specs) == 1
    bundle = result[0]
    assert bundle.spec_key == "spec-0"
    assert len(bundle) == 1
    design = bundle[0]
    assert design.structure is not None
    assert design.complex.chains
    sequence = design.complex.chains[0].sequence
    # Topology: 20 preserved residues plus a 20-30 residue de novo segment.
    assert 40 <= len(sequence) <= 50
    # A real co-designed sequence, not the poly-alanine an off-distribution run returns.
    assert len(set(sequence)) > 3, f"degenerate design sequence: {sequence}"


@pytest.mark.uses_gpu
def test_rfd4_inverse_folding_preserves_backbone_length():
    """Inverse folding redesigns the sequence of the fixed input backbone."""
    inputs = RFD4Input(design_specs=[RFD4DesignSpec(input_structure=_fixture_structure())])
    config = RFD4InverseFoldingConfig(num_replicates=1, batch_size=1, recycle_strategy="low/flat")

    result = run_rfd4_inverse_folding(inputs, config)

    assert result.success is True
    assert result.tool_id == "rfd4-inverse-folding"
    assert len(result) == 1
    design = result[0][0]
    # The backbone is held fixed, so only the sequence changes.
    assert sum(len(chain.sequence) for chain in design.complex.chains) == _FIXTURE_LENGTH


# ── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark("rfd4-design")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_rfd4_design_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark rfd4-design: 4 motif-scaffolded designs from the 65-residue monomer (cold + warm)."""
    inputs = RFD4Input(
        design_specs=[
            RFD4DesignSpec(
                input_structure=_fixture_structure(),
                contig="A1-20, /0, 60-80",
                known_sequence=["sel('A')"],
                hotspots=["sel('A/*/[10-14]')"],
            )
        ]
    )
    config = RFD4DesignConfig(num_replicates=4, batch_size=4)

    result = benchmark_twice(request, "rfd4", lambda: run_rfd4_design(inputs, config))
    validate_output(result)

    assert result.tool_id == "rfd4-design"
    assert len(result) == 1
    assert len(result[0]) == 4
    assert all(design.complex.chains[0].sequence for design in result[0])


@pytest.mark.benchmark("rfd4-inverse-folding")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_rfd4_inverse_folding_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark rfd4-inverse-folding: 4 sequences for the 65-residue backbone (cold + warm)."""
    inputs = RFD4Input(design_specs=[RFD4DesignSpec(input_structure=_fixture_structure())])
    config = RFD4InverseFoldingConfig(num_replicates=4, batch_size=4, sequence_sampling={"tsr_k": 2.0})

    result = benchmark_twice(request, "rfd4", lambda: run_rfd4_inverse_folding(inputs, config))
    validate_output(result)

    assert result.tool_id == "rfd4-inverse-folding"
    assert len(result) == 1
    assert len(result[0]) == 4
    assert all(len(design.complex.chains[0].sequence) == _FIXTURE_LENGTH for design in result[0])

"""tests/structure_tests/test_structure.py.

Tests for the Structure entity.
"""

import warnings
from pathlib import Path

import gemmi
import numpy as np
import pytest
from pydantic import BaseModel

from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.entities.structures.structure import _approx_equal_metric
from proto_tools.entities.structures.utils import (
    adjacent_distances,
    convert_cif_str_to_pdb_str,
    convert_pdb_str_to_cif_str,
    detect_structure_format,
    distances_to_centroid,
    get_backbone_atoms,
    get_centroid,
    is_valid_structure,
    load_structure_file,
    pairwise_distances,
    pdb_file_to_atomarray,
)
from proto_tools.utils.tool_io import Metrics
from tests._structure_fixtures import synthetic_cif

_TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"
_TEST_CIF_FILE = Path(__file__).parent.parent / "dummy_data" / "renin.cif"


@pytest.fixture(scope="module")
def test_pdb_file_content() -> str:
    with open(_TEST_PDB_FILE) as f:
        return f.read()


@pytest.fixture(scope="module")
def test_cif_file_content() -> str:
    with open(_TEST_CIF_FILE) as f:
        return f.read()


@pytest.fixture
def protein_from_pdb_file():
    return Structure.from_file(_TEST_PDB_FILE)


@pytest.fixture
def protein_from_cif_file():
    return Structure.from_file(_TEST_CIF_FILE)


# ── Initialization ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "source,expected_format",
    [
        ("pdb_file", "pdb"),
        ("cif_file", "cif"),
        ("pdb_content", "pdb"),
        ("cif_content", "cif"),
    ],
)
def test_init_detects_format(source, expected_format, test_pdb_file_content, test_cif_file_content):
    sources = {
        "pdb_file": lambda: Structure.from_file(_TEST_PDB_FILE),
        "cif_file": lambda: Structure.from_file(_TEST_CIF_FILE),
        "pdb_content": lambda: Structure(structure=test_pdb_file_content),
        "cif_content": lambda: Structure(structure=test_cif_file_content),
    }
    s = sources[source]()
    assert s.structure_format == expected_format


def test_init_with_invalid_structure():
    with pytest.raises(ValueError, match="Structure content is invalid"):
        Structure(structure="invalid structure content", structure_format="pdb")


def test_init_with_nonexistent_file():
    with pytest.raises(FileNotFoundError, match="File not found"):
        Structure.from_file(Path("/nonexistent/file.pdb"))


@pytest.mark.parametrize("path_type", [str, Path])
def test_init_accepts_path_in_structure_field(path_type):
    """Structure(structure=<path>) loads the file transparently and sets source."""
    s = Structure(structure=path_type(_TEST_PDB_FILE))
    assert s.structure_format == "pdb"
    assert s.source == str(_TEST_PDB_FILE)
    assert "ATOM" in s.structure


def test_init_path_does_not_override_explicit_source():
    s = Structure(structure=str(_TEST_PDB_FILE), source="custom-source")
    assert s.source == "custom-source"


def test_init_content_string_not_treated_as_path(test_pdb_file_content):
    """Multi-line content strings must not be mistaken for filesystem paths."""
    s = Structure(structure=test_pdb_file_content)
    assert s.source is None
    assert s.structure == test_pdb_file_content


def test_model_validate_accepts_bare_content_string(test_pdb_file_content):
    """Bare PDB/CIF string is accepted as shorthand for {"structure": value}.

    Used by nested Pydantic fields typed ``Structure`` (SequenceStructurePair,
    MutationInput) so HTTP callers don't have to hand-roll the envelope.
    """
    s = Structure.model_validate(test_pdb_file_content)
    assert s.structure_format == "pdb"
    assert s.structure == test_pdb_file_content
    assert s.source is None


@pytest.mark.parametrize("path_type", [str, Path])
def test_model_validate_accepts_bare_path(path_type):
    """Bare path is also accepted and loaded from disk, matching from_file."""
    s = Structure.model_validate(path_type(_TEST_PDB_FILE))
    assert s.structure_format == "pdb"
    assert s.source == str(_TEST_PDB_FILE)
    assert "ATOM" in s.structure


def test_nested_field_typed_structure_accepts_bare_string(test_pdb_file_content):
    """A nested Pydantic field typed ``Structure`` accepts a bare content string.

    Regression for the SequenceStructurePair / MutationInput contract — callers
    had to wrap payloads as {"structure": {"structure": <pdb>}} before the
    ``Structure`` before-validator learned to coerce bare strings.
    """

    class _Pair(BaseModel):
        sequence: str
        structure: Structure

    p = _Pair.model_validate({"sequence": "MKTL", "structure": test_pdb_file_content})
    assert p.structure.structure_format == "pdb"
    assert p.structure.structure == test_pdb_file_content


# ── Format conversion ────────────────────────────────────────────────────────


def test_pdb_to_cif_conversion(protein_from_pdb_file):
    cif_content = protein_from_pdb_file.structure_cif
    assert isinstance(cif_content, str)
    assert len(cif_content) > 0
    assert "data_" in cif_content or "_atom_site" in cif_content


def test_cif_to_pdb_conversion(protein_from_cif_file):
    pdb_content = protein_from_cif_file.structure_pdb
    assert isinstance(pdb_content, str)
    assert "ATOM" in pdb_content


def test_cif_to_pdb_no_warnings_for_clean_cif(test_cif_file_content):
    """A normal single-chain CIF (renin) should convert without any lossy-data warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        convert_cif_str_to_pdb_str(test_cif_file_content)
    lossy_warnings = [w for w in caught if "CIF→PDB conversion" in str(w.message)]
    assert lossy_warnings == []


def test_cif_to_pdb_warns_on_long_chain_id(test_cif_file_content):
    """Multi-character chain IDs cannot fit PDB's 1-char column — must warn."""
    # Rename chain A to AA, re-emit CIF, then convert.
    struct = gemmi.make_structure_from_block(gemmi.cif.read_string(test_cif_file_content)[0])
    for model in struct:
        for chain in model:
            chain.name = chain.name + chain.name  # "A" -> "AA"
            break
        break
    modified_cif = struct.make_mmcif_document().as_string()

    with pytest.warns(UserWarning, match="chain ID"):
        convert_cif_str_to_pdb_str(modified_cif)


@pytest.mark.parametrize("fixture_name", ["protein_from_pdb_file", "protein_from_cif_file"])
def test_same_format_returns_original(fixture_name, request):
    """Accessing the same format as stored returns the original string."""
    protein = request.getfixturevalue(fixture_name)
    if protein.structure_format == "pdb":
        assert protein.structure_pdb == protein.structure
    else:
        assert protein.structure_cif == protein.structure


def test_sequences_preserved_through_format_conversion(protein_from_pdb_file):
    original_sequences = protein_from_pdb_file.get_chain_sequences()
    cif_content = protein_from_pdb_file.structure_cif
    converted = Structure(structure=cif_content)
    assert converted.get_chain_sequences() == original_sequences


# ── Gemmi integration ────────────────────────────────────────────────────────


def test_gemmi_struct_lazy_loading_and_caching(protein_from_pdb_file):
    assert protein_from_pdb_file._gemmi_struct is None
    struct1 = protein_from_pdb_file.gemmi_struct
    assert struct1 is not None
    assert len(struct1) > 0
    assert protein_from_pdb_file.gemmi_struct is struct1


@pytest.mark.parametrize("fixture_name", ["protein_from_pdb_file", "protein_from_cif_file"])
def test_gemmi_struct_parses_both_formats(fixture_name, request):
    protein = request.getfixturevalue(fixture_name)
    assert len(protein.gemmi_struct) > 0


# ── File I/O ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "write_method,suffix,content_check",
    [
        ("write_cif", ".cif", lambda c: "data_" in c or "_atom_site" in c),
        ("write_pdb", ".pdb", lambda c: "ATOM" in c),
    ],
)
def test_write_and_round_trip(protein_from_pdb_file, tmp_path, write_method, suffix, content_check):
    out = tmp_path / f"out{suffix}"
    getattr(protein_from_pdb_file, write_method)(out)
    assert out.exists()
    content = out.read_text()
    assert content_check(content)
    Structure.from_file(out)


# ── Sequence extraction ───────────────────────────────────────────────────────


def test_get_chain_sequences_and_ids(protein_from_pdb_file):
    sequences = protein_from_pdb_file.get_chain_sequences()
    assert len(sequences) > 0
    for chain_id, sequence in sequences.items():
        assert isinstance(chain_id, str)
        assert len(sequence) > 0

    chain_ids = protein_from_pdb_file.get_chain_ids()
    assert set(chain_ids) == set(sequences.keys())

    # Default (no chain_id) returns first chain
    assert protein_from_pdb_file.get_chain_sequence() == next(iter(sequences.values()))

    # Specific chain
    assert len(protein_from_pdb_file.get_chain_sequence(chain_ids[0])) > 0


def test_get_chain_sequence_invalid_chain(protein_from_pdb_file):
    with pytest.raises(ValueError, match=r"Chain .* not found"):
        protein_from_pdb_file.get_chain_sequence("INVALID_CHAIN_XYZ")


# ── Pydantic serialization ────────────────────────────────────────────────────


def test_model_dump_and_validate_round_trip(protein_from_pdb_file):
    dumped = protein_from_pdb_file.model_dump()
    assert dumped["structure_format"] == "pdb"
    assert dumped["b_factor_type"] == "unspecified"

    reconstructed = Structure.model_validate(dumped)
    assert reconstructed.structure_format == "pdb"
    assert reconstructed.b_factor_type == protein_from_pdb_file.b_factor_type


def test_nested_pydantic_model_round_trip(protein_from_pdb_file):
    """Structure survives serialization when nested in another Pydantic model."""

    class _StructureModel(BaseModel):
        structure: Structure

    original = _StructureModel(structure=protein_from_pdb_file)
    reconstructed = _StructureModel.model_validate(original.model_dump())

    assert reconstructed.structure.b_factor_type == protein_from_pdb_file.b_factor_type
    assert reconstructed.structure.get_chain_sequences() == protein_from_pdb_file.get_chain_sequences()


def test_model_validate_with_b_factor_type():
    protein = Structure.from_file(_TEST_PDB_FILE, b_factor_type=BFactorType.PLDDT)
    dumped = protein.model_dump()
    assert Structure.model_validate(dumped).b_factor_type == BFactorType.PLDDT


def test_model_validate_missing_structure():
    with pytest.raises(ValueError):
        Structure.model_validate({"b_factor_type": "unspecified", "structure_format": "pdb"})


def test_model_validate_auto_detects_format(test_pdb_file_content):
    s = Structure.model_validate({"structure": test_pdb_file_content, "b_factor_type": "unspecified"})
    assert s.structure_format == "pdb"


def test_visualize(protein_from_pdb_file):
    _ = protein_from_pdb_file.visualize(show_legend=False)


def test_metrics_survive_round_trip():
    protein = Structure.from_file(_TEST_PDB_FILE, metrics={"plddt": 85.2, "ptm": 0.9})
    reconstructed = Structure.model_validate(protein.model_dump())
    assert reconstructed.metrics["plddt"] == 85.2
    assert reconstructed.metrics["ptm"] == 0.9
    assert set(reconstructed.metrics.keys()) == {"plddt", "ptm"}


def test_structure_approx_equal_matching():
    a = Structure.from_file(_TEST_PDB_FILE)
    b = Structure.from_file(_TEST_PDB_FILE)
    a.approx_equal(b)


def test_structure_metrics_access_goes_through_metrics_field():
    """Metric access must go through ``.metrics`` (attribute or mapping style).

    The old ``Structure.__getattr__`` delegation that allowed ``structure.ptm``
    as a shortcut for ``structure.metrics["ptm"]`` is deliberately removed.
    """
    protein = Structure.from_file(_TEST_PDB_FILE, metrics={"ptm": 0.9})
    assert protein.metrics.ptm == 0.9
    assert protein.metrics["ptm"] == 0.9
    # no bypass
    with pytest.raises(AttributeError):
        _ = protein.ptm
    with pytest.raises(AttributeError):
        _ = protein.nonexistent_field


def test_detect_structure_format_empty():
    with pytest.raises(ValueError, match="Empty structure content"):
        detect_structure_format("")


def test_load_structure_file_bad_extension(tmp_path):
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("content")
    with pytest.raises(ValueError, match="Invalid structure file extension"):
        load_structure_file(bad_file)


def test_convert_empty_strings():
    assert convert_pdb_str_to_cif_str("") == ""
    assert convert_cif_str_to_pdb_str("") == ""


# ── to_pdb_with_chain_mapping ─────────────────────────────────────────────────


def test_to_pdb_with_chain_mapping_pdb_input_is_identity(protein_from_pdb_file):
    """PDB-backed Structure returns identity mapping and unchanged content."""
    pdb_content, mapping = protein_from_pdb_file.to_pdb_with_chain_mapping()

    assert pdb_content == protein_from_pdb_file.structure
    assert mapping == {cid: cid for cid in protein_from_pdb_file.get_chain_ids()}


def test_to_pdb_with_chain_mapping_multichar_cif_shortens_and_maps():
    """CIF with multi-char chains yields a valid PDB and a populated mapping."""
    s = Structure(structure=synthetic_cif(["Heavy", "Light"]))
    assert s.get_chain_ids() == ["Heavy", "Light"]

    pdb_content, mapping = s.to_pdb_with_chain_mapping()

    # Mapping covers every original chain with a single-character target.
    assert set(mapping.keys()) == {"Heavy", "Light"}
    assert all(len(v) == 1 for v in mapping.values())
    # Targets are distinct so PyRosetta can disambiguate.
    assert len(set(mapping.values())) == len(mapping)
    # Emitted content is re-parseable PDB (round-trips through gemmi cleanly).
    round_tripped = gemmi.read_pdb_string(pdb_content)
    round_trip_chains = {chain.name for model in round_tripped for chain in model}
    assert round_trip_chains == set(mapping.values())


def test_to_pdb_with_chain_mapping_does_not_mutate_cached_gemmi_struct():
    """Calling the helper must not touch the lazy-loaded gemmi cache."""
    s = Structure(structure=synthetic_cif(["Heavy", "Light"]))
    # Warm the cache first.
    _ = s.gemmi_struct
    chains_before = [chain.name for model in s.gemmi_struct for chain in model]

    s.to_pdb_with_chain_mapping()

    chains_after = [chain.name for model in s.gemmi_struct for chain in model]
    assert chains_before == chains_after == ["Heavy", "Light"]


# ── Metrics container (from tool_io) ─────────────────────────────────────────


def test_metrics_init_strips_none():
    m = Metrics(ptm=0.9, plddt=None)
    assert "ptm" in m
    assert "plddt" not in m


def test_metrics_dict_protocol():
    m = Metrics(ptm=0.9, iptm=0.8)
    assert m["ptm"] == 0.9
    assert len(m) == 2
    assert set(m) == {"ptm", "iptm"}
    assert 42 not in m
    assert len(Metrics()) == 0
    with pytest.raises(KeyError):
        m["missing"]


def test_metrics_primary_value():
    assert Metrics(ptm=0.9, primary_metric="ptm").primary_value == 0.9
    assert Metrics(ptm=0.9).primary_value is None
    # primary_metric set but the named metric isn't in the container
    assert Metrics(primary_metric="missing").primary_value is None


def test_metrics_update():
    m = Metrics(ptm=0.9, iptm=0.7)
    m.update({"iptm": 0.8, "new_key": 0.5})
    assert m["iptm"] == 0.8
    assert m["new_key"] == 0.5
    # update from another Metrics instance
    m.update(Metrics(another=1.0))
    assert m["another"] == 1.0


# ── Structure methods ────────────────────────────────────────────────────────


def test_structure_add_metric(protein_from_pdb_file):
    protein_from_pdb_file.add_metric("new_metric", 42.0)
    assert protein_from_pdb_file.metrics["new_metric"] == 42.0


def test_structure_properties(protein_from_pdb_file):
    assert protein_from_pdb_file.num_chains > 0
    assert protein_from_pdb_file.num_residues > 0
    chain_types = protein_from_pdb_file.get_chain_types()
    assert chain_types and all(v in ("polymer", "ligand") for v in chain_types.values())
    pos_map = protein_from_pdb_file.get_residue_position_map()
    assert pos_map
    assert all(isinstance(t, tuple) and len(t) == 2 for positions in pos_map.values() for t in positions)


def test_get_chain_positions(protein_from_pdb_file):
    chain_id = protein_from_pdb_file.get_chain_ids()[0]
    positions = protein_from_pdb_file.get_chain_positions(chain_id)
    assert positions and all(isinstance(p, int) for p in positions)
    with pytest.raises(ValueError, match="not found"):
        protein_from_pdb_file.get_chain_positions("NONEXISTENT")


def test_approx_equal_different_metrics():
    a = Structure.from_file(_TEST_PDB_FILE, metrics={"ptm": 0.9})
    b = Structure.from_file(_TEST_PDB_FILE, metrics={"iptm": 0.8})
    with pytest.raises(AssertionError, match="Metric keys differ"):
        a.approx_equal(b)


@pytest.mark.parametrize(
    "a,b,error_match",
    [
        (1.0, 2.0, "Metric"),
        ([1.0], [1.0, 2.0], "length"),
        ([1.0, 2.0], [1.0, 2.0], None),
        ("a", "b", "Metric"),
    ],
    ids=["float-mismatch", "list-length", "list-recursive-ok", "non-equal"],
)
def test_approx_equal_metric(a, b, error_match):
    if error_match is None:
        _approx_equal_metric("k", a, b, 1e-4, 1e-6)
    else:
        with pytest.raises(AssertionError, match=error_match):
            _approx_equal_metric("k", a, b, 1e-4, 1e-6)


# ── Geometry utilities ──────────────────────────────────────────────────────────

_COORDS = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])


def test_pairwise_distances():
    result = pairwise_distances(_COORDS)
    assert result.shape == (3,)
    np.testing.assert_allclose(sorted(result), [3.0, 4.0, 5.0])


def test_adjacent_distances():
    result = adjacent_distances(_COORDS)
    assert result.shape == (3,)
    assert result[1] == pytest.approx(3.0)
    assert result[2] == pytest.approx(5.0)


def test_centroid_and_distances():
    centroid = get_centroid(_COORDS)
    assert centroid.shape == (1, 3)
    np.testing.assert_allclose(centroid[0], [1.0, 4.0 / 3, 0.0])
    dists = distances_to_centroid(_COORDS)
    assert dists.shape == (3,)
    assert all(d > 0 for d in dists)


def test_pdb_file_to_atomarray_and_backbone():
    atoms = pdb_file_to_atomarray(str(_TEST_PDB_FILE))
    assert len(atoms) > 0
    assert "CA" in atoms.atom_name
    backbone = get_backbone_atoms(atoms)
    assert len(backbone) > 0
    assert set(backbone.atom_name).issubset({"CA", "N", "C"})


# ── Format detection edge cases ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content,expected",
    [
        ("# comment\nloop_\n_atom_site.id\nATOM 1\n", "cif"),
        ("# comment\n" * 20 + "ATOM      1  N   ALA A   1\n", "pdb"),
        ("# filler\n" * 15 + "_atom_site.id 1\n", "cif"),
    ],
    ids=["cif-loop-marker", "pdb-late-atom", "cif-atom-site-deep"],
)
def test_detect_format_edge_cases(content, expected):
    assert detect_structure_format(content) == expected


# ── is_valid_structure ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "input_val,expected",
    [
        ("HEADER\nEND\n", False),
        (_TEST_PDB_FILE, True),
        ("not a structure at all", False),
    ],
    ids=["no-atoms", "file-path", "garbage"],
)
def test_is_valid_structure(input_val, expected):
    assert is_valid_structure(input_val) is expected


# ── CIF→PDB warning branches ───────────────────────────────────────────────────


def test_warn_cif_to_pdb_long_atom_names(test_cif_file_content):
    struct = gemmi.make_structure_from_block(gemmi.cif.read_string(test_cif_file_content)[0])
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.name = "LONGNAME"
                    break
                break
            break
        break
    modified_cif = struct.make_mmcif_document().as_string()
    with pytest.warns(UserWarning, match="atom name"):
        convert_cif_str_to_pdb_str(modified_cif)


def test_warn_cif_to_pdb_long_residue_names(test_cif_file_content):
    struct = gemmi.make_structure_from_block(gemmi.cif.read_string(test_cif_file_content)[0])
    for model in struct:
        for chain in model:
            for residue in chain:
                residue.name = "LONGRES"
                break
            break
        break
    modified_cif = struct.make_mmcif_document().as_string()
    with pytest.warns(UserWarning, match="residue name"):
        convert_cif_str_to_pdb_str(modified_cif)


def test_warn_cif_to_pdb_metadata_markers(test_cif_file_content):
    injected = test_cif_file_content + "\n_struct_ncs_oper.id 1\n"
    with pytest.warns(UserWarning, match="extension metadata"):
        convert_cif_str_to_pdb_str(injected)


# ── CIF→PDB conversion error paths ─────────────────────────────────────────────


def test_convert_cif_to_pdb_unparseable():
    with pytest.raises(ValueError, match="Failed to convert CIF to PDB"):
        convert_cif_str_to_pdb_str("this is not valid CIF content at all")


# ── per_residue_plddt property ────────────────────────────────────────────────


def _pdb_line(serial: int, chain: str, resseq: int, bfactor: float) -> str:
    x = float((serial - 1) * 3.8)
    return (
        f"ATOM  {serial:5d}  CA  ALA {chain}{resseq:4d}    {x:8.3f}   0.000   0.000  1.00{bfactor:6.2f}           C  "
    )


def test_per_residue_plddt_normalizes_and_spans_chains():
    """PLDDT (0-100) B-factors are normalized to 0-1; all chains included."""
    pdb = "\n".join(
        [
            _pdb_line(1, "A", 1, 95.0),
            _pdb_line(2, "A", 2, 80.0),
            _pdb_line(3, "B", 1, 60.0),
            "END",
        ]
    )
    s = Structure(structure=pdb, b_factor_type=BFactorType.PLDDT)
    assert s.per_residue_plddt == pytest.approx([0.95, 0.80, 0.60], abs=1e-2)

    # NORMALIZED_PLDDT (already 0-1) should not rescale
    pdb_norm = "\n".join(
        [
            _pdb_line(1, "A", 1, 0.95),
            _pdb_line(2, "A", 2, 0.80),
            "END",
        ]
    )
    s_norm = Structure(structure=pdb_norm, b_factor_type=BFactorType.NORMALIZED_PLDDT)
    assert s_norm.per_residue_plddt == pytest.approx([0.95, 0.80], abs=1e-2)


def test_per_residue_plddt_none_for_non_plddt():
    """Returns None when B-factors don't represent pLDDT."""
    pdb = "\n".join([_pdb_line(1, "A", 1, 15.0), "END"])
    assert Structure(structure=pdb, b_factor_type=BFactorType.UNSPECIFIED).per_residue_plddt is None


# ── chain selection ───────────────────────────────────────────────────────────


@pytest.fixture
def three_chain_structure() -> Structure:
    pdb = "\n".join(
        [
            _pdb_line(1, "A", 1, 95.0),
            _pdb_line(2, "A", 2, 80.0),
            _pdb_line(3, "B", 1, 60.0),
            _pdb_line(4, "B", 2, 55.0),
            _pdb_line(5, "C", 1, 40.0),
            "END",
        ]
    )
    return Structure(structure=pdb, b_factor_type=BFactorType.PLDDT, source="test")


@pytest.mark.parametrize(
    ("method_name", "chain_ids", "expected_chains", "expected_plddt"),
    [
        ("select_chain", "B", ["B"], [0.60, 0.55]),
        ("select_chains", "A,C", ["A", "C"], [0.95, 0.80, 0.40]),
        ("select_chains", ["B", "C"], ["B", "C"], [0.60, 0.55, 0.40]),
    ],
)
def test_chain_selection_keeps_requested_chains_and_metadata(
    method_name, chain_ids, expected_chains, expected_plddt, three_chain_structure
):
    selected = getattr(three_chain_structure, method_name)(chain_ids)

    assert [chain.name for model in selected.gemmi_struct for chain in model] == expected_chains
    assert selected.per_residue_plddt == pytest.approx(expected_plddt, abs=1e-2)
    assert selected.b_factor_type == BFactorType.PLDDT
    assert selected.source == "test"
    assert selected.structure_format == "pdb"
    assert three_chain_structure.per_residue_plddt == pytest.approx([0.95, 0.80, 0.60, 0.55, 0.40], abs=1e-2)


@pytest.mark.parametrize(
    ("method_name", "chain_ids"),
    [("select_chain", "B"), ("select_chains", "A,C")],
)
def test_chain_selection_metrics_are_deep_copied(method_name, chain_ids, three_chain_structure):
    three_chain_structure.add_metric("pae_matrix", [[0.0]])
    selected = getattr(three_chain_structure, method_name)(chain_ids)
    selected.add_metric("pae_matrix", [[1.0]])
    assert three_chain_structure.metrics["pae_matrix"] == [[0.0]]
    assert selected.metrics["pae_matrix"] == [[1.0]]


@pytest.mark.parametrize(
    ("method_name", "chain_ids", "match"),
    [
        ("select_chain", "Z", "not present"),
        ("select_chains", "A,Z", "not present"),
        ("select_chains", "", "At least one"),
    ],
)
def test_chain_selection_rejects_missing_or_empty_request(method_name, chain_ids, match):
    pdb = "\n".join([_pdb_line(1, "A", 1, 95.0), "END"])
    s = Structure(structure=pdb, b_factor_type=BFactorType.PLDDT)
    with pytest.raises(ValueError, match=match):
        getattr(s, method_name)(chain_ids)


@pytest.mark.parametrize("method_name", ["select_chain", "select_chains"])
def test_chain_selection_preserves_cif_format_for_multi_char_chains(method_name, test_cif_file_content):
    """CIF-sourced Structures round-trip without dropping multi-char chain names."""
    s = Structure(structure=test_cif_file_content, structure_format="cif")
    chain_name = next(c.name for m in s.gemmi_struct for c in m)
    out = getattr(s, method_name)(chain_name if method_name == "select_chain" else [chain_name])
    assert out.structure_format == "cif"
    assert any(c.name == chain_name for m in out.gemmi_struct for c in m)


# ============================================================================
# Interface & Clash Analysis
# ============================================================================


@pytest.fixture
def pdl1_complex() -> Structure:
    """PDL1 target + VHH binder: two-chain crystal structure, no clashes at 2.5 Å."""
    return Structure.from_file(Path(__file__).parent.parent / "dummy_data" / "pdl1.pdb")


@pytest.fixture
def multi_char_chain_structure(test_cif_file_content) -> Structure:
    """CIF-backed Structure with a multi-char chain ID — triggers the PDB-conversion truncation guard."""
    struct = gemmi.make_structure_from_block(gemmi.cif.read_string(test_cif_file_content)[0])
    struct[0][0].name = "Heavy"
    return Structure(structure=struct.make_mmcif_document().as_string(), structure_format="cif")


def _synthetic_pdb(*atom_lines: str) -> Structure:
    return Structure(structure="\n".join([*atom_lines, "END", ""]), structure_format="pdb")


@pytest.mark.parametrize(
    ("atoms", "expected"),
    [
        pytest.param(
            (
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C",
                "ATOM      2  CA  ALA A   2       1.500   0.000   0.000  1.00 50.00           C",
            ),
            0,
            id="bonded-neighbors-in-same-chain-excluded",
        ),
        pytest.param(
            (
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C",
                "ATOM      2  CA  ALA B   1       1.500   0.000   0.000  1.00 50.00           C",
            ),
            1,
            id="cross-chain-pair-counted",
        ),
    ],
)
def test_ca_clash_score_exclusion_rules(atoms, expected):
    """Two minimal cases isolate the exclusion rule (bonded) vs. the positive case (cross-chain)."""
    assert _synthetic_pdb(*atoms).ca_clash_score(threshold=2.5) == expected


def test_ca_clash_score_clean_complex_is_zero(pdl1_complex):
    """Regression baseline: a real, well-resolved complex has no Ca-Ca clashes at 2.5 Å."""
    assert pdl1_complex.ca_clash_score(threshold=2.5) == 0


def test_interface_contact_residues_shape_and_monotonicity(pdl1_complex):
    """Return contract: 1-indexed sorted keys, single-letter AA values; tighter cutoff ⊆ looser."""
    loose = pdl1_complex.interface_contact_residues(binder_chain="B", target_chains=["A"], cutoff=4.0)
    tight = pdl1_complex.interface_contact_residues(binder_chain="B", target_chains=["A"], cutoff=3.0)
    assert loose, "PDL1 binder should contact target at 4.0 Å"
    assert list(loose) == sorted(loose) and all(pos >= 1 for pos in loose)
    assert all(len(aa) == 1 and aa.isalpha() for aa in loose.values())
    assert set(tight) <= set(loose)


def test_interface_contact_residues_missing_target_is_empty(pdl1_complex):
    """A nonexistent target chain yields {} rather than raising."""
    assert pdl1_complex.interface_contact_residues(binder_chain="B", target_chains=["Z"], cutoff=4.0) == {}


def test_interface_contact_residues_include_hydrogens_toggle():
    """Heavy atoms 5 Å apart (no heavy-only contact) but a binder H is within 4 Å of the target CA.

    ``include_hydrogens=False`` (default) → heavy-only check, no contact.
    ``include_hydrogens=True`` → H-inclusive, matches Germinal's ``hotspot_residues`` which
    uses ``Selection.unfold_entities("A")`` on PyRosetta-relaxed PDBs that carry hydrogens.
    """
    struct = _synthetic_pdb(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C",
        "ATOM      2  H   ALA A   1       1.500   0.000   0.000  1.00 50.00           H",
        "ATOM      3  CA  ALA B   1       5.000   0.000   0.000  1.00 50.00           C",
    )
    assert struct.interface_contact_residues(binder_chain="A", target_chains=["B"], cutoff=4.0) == {}
    assert struct.interface_contact_residues(
        binder_chain="A", target_chains=["B"], cutoff=4.0, include_hydrogens=True
    ) == {1: "A"}


@pytest.mark.parametrize("target_chains", [["B", "C"], ("B", "C"), "B,C"])
def test_interface_contact_residues_pools_multi_chain_target(target_chains):
    """``target_chains`` unions atoms across explicit and comma-separated chain inputs."""
    # A at origin; B at (10,10,10) is out of range; C at (2,0,0) is 2 Å from A.
    contacts = _synthetic_pdb(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C",
        "ATOM      2  CA  ALA B   1      10.000  10.000  10.000  1.00 50.00           C",
        "ATOM      3  CA  ALA C   1       2.000   0.000   0.000  1.00 50.00           C",
    ).interface_contact_residues(binder_chain="A", target_chains=target_chains, cutoff=4.0)
    assert contacts == {1: "A"}


@pytest.mark.parametrize(
    ("binder", "target", "match"),
    [
        pytest.param("Heavy", "A", "single character", id="multi-char-binder"),
        pytest.param("B", "AA,BB", "single character", id="multi-char-target"),
        pytest.param("A", "A", "must not also appear", id="self-contact"),
        pytest.param("A", "B,A", "must not also appear", id="binder-in-target-list"),
    ],
)
def test_interface_contact_residues_rejects_invalid_chain_args(pdl1_complex, binder, target, match):
    """Multi-char chain IDs and binder∈target are both silent-nonsense — fail fast."""
    with pytest.raises(ValueError, match=match):
        pdl1_complex.interface_contact_residues(binder_chain=binder, target_chains=target, cutoff=4.0)


def test_hotspot_contacts_subset_relations(pdl1_complex):
    """Hits ⊆ interface contacts, germinal_mode ⊆ default, and str/list hotspot specs parse identically."""
    interface = pdl1_complex.interface_contact_residues(binder_chain="B", target_chains=["A"], cutoff=6.0)
    default = pdl1_complex.hotspot_contacts(binder_chain="B", target_hotspots="A56,A66")
    germinal = pdl1_complex.hotspot_contacts(binder_chain="B", target_hotspots="A56,A66", germinal_mode=True)
    assert default == pdl1_complex.hotspot_contacts(binder_chain="B", target_hotspots=["A56", "A66"])
    assert default and set(germinal) <= set(default) <= set(interface)


def test_hotspot_contacts_two_step_widens_beyond_direct_hotspot():
    """A binder touching a hotspot-neighbor (but not the hotspot itself) is included by the expansion step.

    Geometry: binder A:1 at (0,0,0), hotspot-neighbor B:20 at (4,0,0), declared hotspot B:10 at (8,0,0).
    Binder-to-hotspot = 8 Å (direct miss), but B:20 is 4 Å from both → two-step catches A:1;
    ``expansion_cutoff=0`` (direct-only) misses it.
    """
    struct = _synthetic_pdb(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C",
        "ATOM      2  CA  ALA B  20       4.000   0.000   0.000  1.00 50.00           C",
        "ATOM      3  CA  ALA B  10       8.000   0.000   0.000  1.00 50.00           C",
    )
    assert set(struct.hotspot_contacts(binder_chain="A", target_hotspots="B10")) == {1}
    assert (
        struct.hotspot_contacts(binder_chain="A", target_hotspots="B10", expansion_cutoff=0.0, contact_cutoff=5.0) == {}
    )


def test_hotspot_contacts_filters_by_binder_positions():
    """``binder_positions`` restricts the binder side; unmatched hotspots → graceful empty."""
    struct = _synthetic_pdb(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C",
        "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00 50.00           C",
        "ATOM      3  CA  ALA B  10       3.000   0.000   0.000  1.00 50.00           C",
    )
    kw = {"binder_chain": "A", "target_hotspots": "B10", "expansion_cutoff": 0.0, "contact_cutoff": 4.0}
    assert set(struct.hotspot_contacts(**kw)) == {1, 2}
    assert set(struct.hotspot_contacts(**kw, binder_positions=[2])) == {2}
    assert struct.hotspot_contacts(binder_chain="A", target_hotspots="Z99") == {}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"binder_chain": "B", "target_hotspots": "45"}, "chain-prefixed", id="no-chain-prefix"),
        pytest.param({"binder_chain": "B", "target_hotspots": "A45.5"}, "chain-prefixed", id="non-integer-residue"),
        pytest.param({"binder_chain": "Heavy", "target_hotspots": "A45"}, "single character", id="multi-char-binder"),
        pytest.param({"binder_chain": "A", "target_hotspots": "A45"}, "must not also appear", id="binder-in-hotspots"),
    ],
)
def test_hotspot_contacts_rejects_invalid_inputs(pdl1_complex, kwargs, match):
    """Malformed tokens, multi-char binder chains, and self-contact all fail fast."""
    with pytest.raises(ValueError, match=match):
        pdl1_complex.hotspot_contacts(**kwargs)


def test_ca_clash_score_rejects_multi_char_chain_id_on_structure(multi_char_chain_structure):
    """Guard protects the bonded-neighbor exclusion from silently mis-firing on collided chain IDs."""
    with pytest.raises(ValueError, match="single character"):
        multi_char_chain_structure.ca_clash_score(threshold=2.5)


# ============================================================================
# Structure.concat
# ============================================================================


def test_concat_roundtrips_split_chains(pdl1_complex):
    """A complex split by select_chain → concat preserves chain set, residue count, clash score, and b_factor_type."""
    merged = Structure.concat([pdl1_complex.select_chain("A"), pdl1_complex.select_chain("B")])
    assert set(merged.get_chain_ids()) == {"A", "B"}
    assert merged.num_residues == pdl1_complex.num_residues
    assert merged.ca_clash_score() == pdl1_complex.ca_clash_score()
    assert merged.b_factor_type == pdl1_complex.b_factor_type


def test_concat_rejects_duplicate_chain_ids(pdl1_complex):
    """Two inputs with the same chain ID collide — concat raises rather than producing an invalid PDB."""
    a = pdl1_complex.select_chain("A")
    with pytest.raises(ValueError, match="Duplicate chain ID"):
        Structure.concat([a, a])


def test_concat_rejects_b_factor_type_mismatch():
    """Mixing b_factor_types (pLDDT + temperature factors) would misbrand the result."""
    a = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C\nEND\n"
    b = "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00 50.00           C\nEND\n"
    with pytest.raises(ValueError, match="b_factor_type mismatch"):
        Structure.concat(
            [
                Structure(structure=a, structure_format="pdb", b_factor_type=BFactorType.PLDDT),
                Structure(structure=b, structure_format="pdb", b_factor_type=BFactorType.TEMPERATURE_FACTOR),
            ]
        )


def test_concat_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        Structure.concat([])


def test_concat_accepts_multi_char_chain_id_when_all_inputs_are_cif():
    """All-CIF concat preserves CIF format and supports multi-char chain IDs.

    When every input is CIF, the concatenation emits a CIF result that can hold
    multi-character chain labels like "Heavy" / "Light" without any PDB-shortening.
    """
    pdb_a = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n"
    pdb_b = "ATOM      1  CA  ALA B   1     100.000   0.000   0.000  1.00 20.00           C\nEND\n"
    # Build CIF inputs with multi-char chain names: PDB → CIF → rename to multi-char.
    s_a_cif = Structure(structure=Structure(structure=pdb_a).structure_cif, structure_format="cif")
    s_b_cif = Structure(structure=Structure(structure=pdb_b).structure_cif, structure_format="cif")
    heavy = s_a_cif.with_renamed_chains({"A": "Heavy"})
    light = s_b_cif.with_renamed_chains({"B": "Light"})

    out = Structure.concat([heavy, light])
    assert out.structure_format == "cif"
    assert sorted(out.get_chain_ids()) == ["Heavy", "Light"]


def test_concat_rejects_mixed_formats():
    """Mixed PDB+CIF inputs raise — caller must coerce explicitly to one format.

    Avoids the ambiguity of "should we promote to CIF or demote to PDB?" by
    forcing the choice into the call site, where the caller has the context.
    """
    pdb_struct = Structure(
        structure="ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C\nEND\n",
        structure_format="pdb",
    )
    cif_struct = Structure(structure=pdb_struct.structure_cif, structure_format="cif")
    with pytest.raises(ValueError, match="share structure_format"):
        Structure.concat([pdb_struct, cif_struct])


def test_concat_preserves_pdb_format():
    """All-PDB concat produces a PDB output (no surprise format change)."""
    a = Structure(
        structure="ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C\nEND\n",
        structure_format="pdb",
    )
    b = Structure(
        structure="ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00 50.00           C\nEND\n",
        structure_format="pdb",
    )
    out = Structure.concat([a, b])
    assert out.structure_format == "pdb"
    assert sorted(out.get_chain_ids()) == ["A", "B"]


# ── with_renamed_chains ───────────────────────────────────────────────────────


_SINGLE_CHAIN_PDB = (
    "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
    "ATOM      2  CA  GLY A   1       1.500   0.000   0.000  1.00 20.00           C\n"
    "ATOM      3  C   GLY A   1       2.000   1.500   0.000  1.00 20.00           C\n"
    "ATOM      4  O   GLY A   1       1.300   2.500   0.000  1.00 20.00           O\n"
    "END\n"
)

_TWO_CHAIN_PDB = (
    "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
    "ATOM      2  CA  GLY A   1       1.500   0.000   0.000  1.00 20.00           C\n"
    "ATOM      3  C   GLY A   1       2.000   1.500   0.000  1.00 20.00           C\n"
    "ATOM      4  O   GLY A   1       1.300   2.500   0.000  1.00 20.00           O\n"
    "TER\n"
    "ATOM      5  N   GLY B   1     100.000   0.000   0.000  1.00 20.00           N\n"
    "ATOM      6  CA  GLY B   1     101.500   0.000   0.000  1.00 20.00           C\n"
    "ATOM      7  C   GLY B   1     102.000   1.500   0.000  1.00 20.00           C\n"
    "ATOM      8  O   GLY B   1     101.300   2.500   0.000  1.00 20.00           O\n"
    "END\n"
)


def test_with_renamed_chains_identity_returns_unchanged():
    """Identity mapping (and empty mapping) short-circuit and return self."""
    s = Structure(structure=_SINGLE_CHAIN_PDB)
    assert s.with_renamed_chains({"A": "A"}) is s
    assert s.with_renamed_chains({}) is s


def test_with_renamed_chains_single_char_preserves_pdb_format():
    """Single-char rename on a PDB Structure stays PDB."""
    s = Structure(structure=_SINGLE_CHAIN_PDB)
    renamed = s.with_renamed_chains({"A": "B"})
    assert renamed.get_chain_ids() == ["B"]
    assert renamed.structure_format == "pdb"


def test_with_renamed_chains_pdb_source_multi_char_target_raises():
    """PDB col 22 is single-char only; multi-char target on a PDB Structure must raise.

    The error message points the caller at the explicit fix (convert to CIF first)
    rather than silently switching format.
    """
    s = Structure(structure=_SINGLE_CHAIN_PDB)
    with pytest.raises(ValueError, match="multi-character chain ID"):
        s.with_renamed_chains({"A": "Heavy"})


def test_with_renamed_chains_multi_char_works_when_source_is_cif():
    """Multi-char rename succeeds when the source is CIF (CIF can hold any chain ID)."""
    s = Structure(structure=_SINGLE_CHAIN_PDB)
    s_cif = Structure(structure=s.structure_cif, structure_format="cif")
    renamed = s_cif.with_renamed_chains({"A": "Heavy"})
    assert renamed.get_chain_ids() == ["Heavy"]
    assert renamed.structure_format == "cif"


def test_with_renamed_chains_partial_mapping_leaves_other_chains():
    """Chain IDs not in the mapping pass through unchanged."""
    s = Structure(structure=_TWO_CHAIN_PDB)
    renamed = s.with_renamed_chains({"A": "C"})
    # A → C, B unchanged. Both single-char so PDB stays PDB.
    assert sorted(renamed.get_chain_ids()) == ["B", "C"]
    assert renamed.structure_format == "pdb"


def test_with_renamed_chains_rejects_duplicate_targets():
    """Mapping that collapses two chains to the same target raises."""
    s = Structure(structure=_TWO_CHAIN_PDB)
    with pytest.raises(ValueError, match="duplicate chain ID"):
        s.with_renamed_chains({"A": "X", "B": "X"})


def test_with_renamed_chains_preserves_metadata():
    """b_factor_type, source, and metrics carry through unchanged on a fresh Structure."""

    class _M(Metrics):
        plddt: float | None = None

    s = Structure(
        structure=_SINGLE_CHAIN_PDB,
        b_factor_type=BFactorType.PLDDT,
        source="esmfold",
        metrics=_M(plddt=0.85),
    )
    renamed = s.with_renamed_chains({"A": "B"})
    assert renamed is not s
    assert renamed.b_factor_type == BFactorType.PLDDT
    assert renamed.source == "esmfold"
    assert renamed.metrics.plddt == 0.85
    # Deep copy: mutating the renamed metrics doesn't touch the original.
    renamed.metrics.plddt = 0.0
    assert s.metrics.plddt == 0.85


def test_with_renamed_chains_preserves_atomic_content():
    """Renaming chains must not change residues, atoms, or coordinates.

    Covers the gemmi clone-mutate-reserialize round trip:
    - same-format rename (PDB → PDB via single-char target)
    - same-format rename (CIF → CIF via multi-char target — exercises gemmi's
      CIF writer code path, distinct from the PDB writer)
    - full round-trip back to the original chain labels via ``approx_equal``
    """
    from proto_tools.entities.structures.structure import _extract_atom_positions

    s = Structure(structure=_TWO_CHAIN_PDB)
    original_atoms = _extract_atom_positions(s)

    # Same-format PDB rename: A→C, B→D. Output stays PDB.
    pdb_renamed = s.with_renamed_chains({"A": "C", "B": "D"})
    assert pdb_renamed.structure_format == "pdb"
    assert _extract_atom_positions(pdb_renamed) == original_atoms

    # CIF source + multi-char rename: exercises the CIF writer path.
    s_cif = Structure(structure=s.structure_cif, structure_format="cif")
    cif_renamed = s_cif.with_renamed_chains({"A": "Heavy", "B": "Light"})
    assert cif_renamed.structure_format == "cif"
    assert _extract_atom_positions(cif_renamed) == original_atoms

    # Full round-trip: rename and rename back, content matches via approx_equal.
    round_tripped = cif_renamed.with_renamed_chains({"Heavy": "A", "Light": "B"})
    s_cif.approx_equal(round_tripped)


# ── _serialize_gemmi consolidation ────────────────────────────────────────────


def _make_lossy_cif() -> str:
    """Build a minimal CIF whose atoms have multi-char names.

    Triggers the 'long atom name' warning when converted to PDB.
    """
    return (
        "data_lossy\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.pdbx_PDB_model_num\n"
        # 5-char atom name "OVRLG" exceeds PDB's 4-char field.
        "ATOM 1 N OVRLG . GLY A 1 1 ? 0.000 0.000 0.000 1.00 20.00 1 GLY A OVRLG 1\n"
    )


def _read_gemmi_pdb(content: str):
    return gemmi.read_pdb_string(content)


def test_serialize_gemmi_pdb_to_pdb_no_warning():
    """Re-emission in source format never warns."""
    from proto_tools.entities.structures.utils import _serialize_gemmi

    struct = _read_gemmi_pdb(_SINGLE_CHAIN_PDB)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _serialize_gemmi(struct, "pdb", source_format="pdb")
    assert "ATOM" in out
    assert [w for w in caught if "CIF→PDB" in str(w.message)] == []


def test_serialize_gemmi_cif_to_cif_no_warning():
    """CIF re-emission also never warns (no conversion happens)."""
    from proto_tools.entities.structures.utils import _serialize_gemmi

    struct = _read_gemmi_pdb(_SINGLE_CHAIN_PDB)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _serialize_gemmi(struct, "cif", source_format="cif")
    assert out.startswith("data_")
    assert [w for w in caught if "CIF→PDB" in str(w.message)] == []


def test_serialize_gemmi_pdb_to_cif_no_warning():
    """PDB→CIF is lossless and never warns."""
    from proto_tools.entities.structures.utils import _serialize_gemmi

    struct = _read_gemmi_pdb(_SINGLE_CHAIN_PDB)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _serialize_gemmi(struct, "cif", source_format="pdb")
    assert out.startswith("data_")
    assert [w for w in caught if "CIF→PDB" in str(w.message)] == []


def test_serialize_gemmi_cif_to_pdb_lossy_warns():
    """CIF→PDB on a structure with long atom names emits the lossy warning."""
    from proto_tools.entities.structures.utils import _serialize_gemmi

    cif = _make_lossy_cif()
    doc = gemmi.cif.read_string(cif)
    struct = gemmi.make_structure_from_block(doc[0])
    with pytest.warns(UserWarning, match="atom name"):
        _serialize_gemmi(struct, "pdb", source_format="cif", cif_content_for_warnings=cif)


def test_concat_does_not_warn_on_all_cif_input():
    """Format-preserving concat (all-CIF → CIF output) skips CIF→PDB entirely, so no lossy warnings fire.

    The consolidation work originally added a warning here because concat used to
    always emit PDB. After the format-preserving redesign, concat no longer does
    a CIF→PDB conversion when inputs are all CIF — the path that triggered
    `_warn_cif_to_pdb_lossy` is gone, replaced by lossless CIF re-emission.
    """
    cif_struct = Structure(structure=_make_lossy_cif())
    assert cif_struct.structure_format == "cif"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = Structure.concat([cif_struct])
    assert out.structure_format == "cif"  # Format preserved — no conversion
    assert [w for w in caught if "CIF→PDB" in str(w.message)] == []


def test_to_pdb_with_chain_mapping_warns_on_lossy_cif():
    """to_pdb_with_chain_mapping on a CIF with long atom names warns.

    When the source CIF carries data that PDB cannot represent (e.g. atom names
    longer than 4 characters), emitting PDB must surface a lossy-conversion
    warning rather than silently truncate.
    """
    cif_struct = Structure(structure=_make_lossy_cif())
    with pytest.warns(UserWarning, match="atom name"):
        cif_struct.to_pdb_with_chain_mapping()


@pytest.mark.parametrize("method_name", ["select_chain", "select_chains"])
def test_chain_selection_does_not_warn_on_cif_re_emission(method_name):
    """Chain selection is pure re-emission in source format — no warnings, ever."""
    cif_struct = Structure(structure=_make_lossy_cif())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(cif_struct, method_name)("A")
    assert [w for w in caught if "CIF→PDB" in str(w.message)] == []


def test_with_renamed_chains_does_not_warn_on_cif_re_emission():
    """with_renamed_chains is pure re-emission in source format — no warnings, ever."""
    cif_struct = Structure(structure=_make_lossy_cif())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cif_struct.with_renamed_chains({"A": "Heavy"})
    assert [w for w in caught if "CIF→PDB" in str(w.message)] == []

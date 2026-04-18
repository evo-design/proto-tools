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


def _synthetic_cif(chain_names: list[str]) -> str:
    """Build a minimal valid mmCIF with one glycine residue per named chain.

    Each chain gets four atoms (N, CA, C, O) so gemmi accepts it as a real residue.
    Chains are spaced 100 A apart on the x-axis to avoid any overlap artifacts.
    """
    header = (
        "data_synthetic\n"
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
    )
    atom_records = []
    atom_id = 1
    for chain_idx, name in enumerate(chain_names):
        base_x = chain_idx * 100.0
        label_asym = chr(ord("A") + chain_idx % 26)
        for atom_name, dx, dy in [("N", 0.0, 0.0), ("CA", 1.5, 0.0), ("C", 2.0, 1.5), ("O", 1.3, 2.5)]:
            atom_records.append(
                f"ATOM {atom_id} {atom_name[0]} {atom_name} . GLY {label_asym} 1 1 ? "
                f"{base_x + dx:.3f} {dy:.3f} 0.000 1.00 20.00 1 GLY {name} {atom_name} 1"
            )
            atom_id += 1
    return header + "\n".join(atom_records) + "\n"


def test_to_pdb_with_chain_mapping_pdb_input_is_identity(protein_from_pdb_file):
    """PDB-backed Structure returns identity mapping and unchanged content."""
    pdb_content, mapping = protein_from_pdb_file.to_pdb_with_chain_mapping()

    assert pdb_content == protein_from_pdb_file.structure
    assert mapping == {cid: cid for cid in protein_from_pdb_file.get_chain_ids()}


def test_to_pdb_with_chain_mapping_multichar_cif_shortens_and_maps():
    """CIF with multi-char chains yields a valid PDB and a populated mapping."""
    s = Structure(structure=_synthetic_cif(["Heavy", "Light"]))
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
    s = Structure(structure=_synthetic_cif(["Heavy", "Light"]))
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


# ── select_chain ──────────────────────────────────────────────────────────────


def test_select_chain_keeps_only_requested_chain_and_preserves_metadata():
    pdb = "\n".join(
        [
            _pdb_line(1, "A", 1, 95.0),
            _pdb_line(2, "A", 2, 80.0),
            _pdb_line(3, "B", 1, 60.0),
            _pdb_line(4, "B", 2, 55.0),
            "END",
        ]
    )
    s = Structure(structure=pdb, b_factor_type=BFactorType.PLDDT, source="test")
    b = s.select_chain("B")
    assert b.per_residue_plddt == pytest.approx([0.60, 0.55], abs=1e-2)
    assert b.b_factor_type == BFactorType.PLDDT
    assert b.source == "test"
    assert b.structure_format == "pdb"  # PDB in → PDB out
    # Cloned — original not mutated.
    assert s.per_residue_plddt == pytest.approx([0.95, 0.80, 0.60, 0.55], abs=1e-2)


def test_select_chain_raises_on_missing_chain():
    pdb = "\n".join([_pdb_line(1, "A", 1, 95.0), "END"])
    with pytest.raises(ValueError, match="not present"):
        Structure(structure=pdb, b_factor_type=BFactorType.PLDDT).select_chain("Z")


def test_select_chain_metrics_are_deep_copied():
    pdb = "\n".join([_pdb_line(1, "A", 1, 90.0), _pdb_line(2, "B", 1, 80.0), "END"])
    s = Structure(structure=pdb, b_factor_type=BFactorType.PLDDT)
    s.add_metric("plddt", 0.90)
    b = s.select_chain("B")
    b.add_metric("plddt", 0.50)
    assert s.metrics["plddt"] == 0.90 and b.metrics["plddt"] == 0.50


def test_select_chain_preserves_cif_format_for_multi_char_chains(test_cif_file_content):
    """CIF-sourced Structures round-trip through select_chain without dropping multi-char chain names."""
    s = Structure(structure=test_cif_file_content, structure_format="cif")
    chain_name = next(c.name for m in s.gemmi_struct for c in m)
    out = s.select_chain(chain_name)
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
    loose = pdl1_complex.interface_contact_residues(binder_chain="B", target_chain="A", cutoff=4.0)
    tight = pdl1_complex.interface_contact_residues(binder_chain="B", target_chain="A", cutoff=3.0)
    assert loose, "PDL1 binder should contact target at 4.0 Å"
    assert list(loose) == sorted(loose) and all(pos >= 1 for pos in loose)
    assert all(len(aa) == 1 and aa.isalpha() for aa in loose.values())
    assert set(tight) <= set(loose)


def test_interface_contact_residues_missing_target_is_empty(pdl1_complex):
    """A nonexistent target chain yields {} rather than raising."""
    assert pdl1_complex.interface_contact_residues(binder_chain="B", target_chain="Z", cutoff=4.0) == {}


def test_interface_contact_residues_pools_multi_chain_target():
    """``target_chain="B,C"`` splits and unions atoms across both chains."""
    # A at origin; B at (10,10,10) is out of range; C at (2,0,0) is 2 Å from A.
    contacts = _synthetic_pdb(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C",
        "ATOM      2  CA  ALA B   1      10.000  10.000  10.000  1.00 50.00           C",
        "ATOM      3  CA  ALA C   1       2.000   0.000   0.000  1.00 50.00           C",
    ).interface_contact_residues(binder_chain="A", target_chain="B,C", cutoff=4.0)
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
        pdl1_complex.interface_contact_residues(binder_chain=binder, target_chain=target, cutoff=4.0)


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


def test_concat_rejects_multi_char_chain_id(multi_char_chain_structure):
    """Concat emits PDB, which can't represent multi-char chain IDs — fail fast."""
    with pytest.raises(ValueError, match="single character"):
        Structure.concat([multi_char_chain_structure])

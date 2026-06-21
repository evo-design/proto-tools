"""tests/structure_alignment_tests/test_superposition.py.

Unit tests for the alignment superposition transform: the shared model, the
TMalign/USalign rotation-matrix parser, the PyMOL object-matrix decomposition,
and that each wrapper packages rotation/translation into ``output.superposition``.
These don't run the binaries; the integration tests cover the real output.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_alignment.pymol_rmsd import (
    PyMOLRMSDConfig,
    PyMOLRMSDInput,
    run_pymol_rmsd_alignment,
)
from proto_tools.tools.structure_alignment.pymol_rmsd.standalone.run import _object_transform
from proto_tools.tools.structure_alignment.superposition import (
    SuperpositionTransform,
    build_superimposed_pdb,
)
from proto_tools.tools.structure_alignment.tmalign import TMalignConfig, TMalignInput, run_tmalign
from proto_tools.tools.structure_alignment.tmalign.standalone.run import (
    _parse_rotation_matrix as _parse_tmalign_matrix,
)
from proto_tools.tools.structure_alignment.usalign import USalignConfig, USalignInput, run_usalign
from proto_tools.tools.structure_alignment.usalign.standalone.run import (
    _parse_rotation_matrix as _parse_usalign_matrix,
)

_PDB_PATH = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"

# A TMalign/USalign ``-m`` matrix file: identity rotation, translation (10.5, -3.2, 0.7).
_MATRIX_TEXT = """
 -------- rotation matrix to rotate Chain_1 to Chain_2 ------
 m               t[m]        u[m][0]        u[m][1]        u[m][2]
 0      10.5000000000   1.0000000000   0.0000000000   0.0000000000
 1      -3.2000000000   0.0000000000   1.0000000000   0.0000000000
 2       0.7000000000   0.0000000000   0.0000000000   1.0000000000
"""

_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


# ── Model ───────────────────────────────────────────────────────────────────


def test_transform_rejects_non_3x3_rotation():
    with pytest.raises(ValidationError, match="rotation must be a 3x3 matrix"):
        SuperpositionTransform(rotation=[[1, 0], [0, 1]], translation=[0, 0, 0])


def test_transform_rejects_wrong_length_translation():
    with pytest.raises(ValidationError, match="translation must have length 3"):
        SuperpositionTransform(rotation=_IDENTITY, translation=[0, 0])


def test_from_optional_returns_none_when_either_part_missing():
    assert SuperpositionTransform.from_optional(None, [1, 2, 3]) is None
    assert SuperpositionTransform.from_optional(_IDENTITY, None) is None


def test_from_optional_builds_when_both_present():
    t = SuperpositionTransform.from_optional(_IDENTITY, [1, 2, 3])
    assert t is not None
    assert t.rotation == _IDENTITY
    assert t.translation == [1.0, 2.0, 3.0]


def test_apply_matches_matrix_math():
    # 90° rotation about z then +(1, 2, 3): (1, 0, 0) -> (0, 1, 0) -> (1, 3, 3).
    t = SuperpositionTransform(rotation=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], translation=[1, 2, 3])
    assert t.apply(1.0, 0.0, 0.0) == pytest.approx((1.0, 3.0, 3.0))


# ── Superimposed multi-model PDB ──────────────────────────────────────────────


def _atom_lines(text):
    return [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]


def test_build_superimposed_pdb_frames_two_models():
    pdb = _PDB_PATH.read_text()
    identity = SuperpositionTransform(rotation=_IDENTITY, translation=[0.0, 0.0, 0.0])
    out = build_superimposed_pdb(pdb, pdb, identity)

    assert "MODEL        1" in out
    assert "MODEL        2" in out
    assert out.count("ENDMDL") == 2
    assert out.rstrip().endswith("END")
    # Identity ⇒ MODEL 1 coordinates match the query's atoms unchanged.
    model1 = out.split("ENDMDL", 1)[0]
    for orig, moved in zip(_atom_lines(pdb)[:5], _atom_lines(model1)[:5], strict=True):
        for cols in (slice(30, 38), slice(38, 46), slice(46, 54)):
            assert float(moved[cols]) == pytest.approx(float(orig[cols]), abs=1e-3)


def test_build_superimposed_pdb_applies_translation_to_model_1_only():
    pdb = _PDB_PATH.read_text()
    shifted = SuperpositionTransform(rotation=_IDENTITY, translation=[10.0, 0.0, 0.0])
    out = build_superimposed_pdb(pdb, pdb, shifted)
    model1, model2 = out.split("ENDMDL")[0], out.split("ENDMDL")[1]

    orig_x = float(_atom_lines(pdb)[0][30:38])
    assert float(_atom_lines(model1)[0][30:38]) == pytest.approx(orig_x + 10.0, abs=1e-3)
    # Reference (MODEL 2) is untouched.
    assert float(_atom_lines(model2)[0][30:38]) == pytest.approx(orig_x, abs=1e-3)


# ── Parsers ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("parse", [_parse_tmalign_matrix, _parse_usalign_matrix])
def test_rotation_matrix_parses_rotation_and_translation(parse):
    parsed = parse(_MATRIX_TEXT)
    assert parsed == {"rotation": _IDENTITY, "translation": [10.5, -3.2, 0.7]}


@pytest.mark.parametrize("parse", [_parse_tmalign_matrix, _parse_usalign_matrix])
def test_rotation_matrix_returns_none_on_unparseable_text(parse):
    assert parse("no matrix here") is None
    assert parse("") is None


def test_pymol_object_transform_decomposes_4x4():
    class FakeCmd:
        def get_object_matrix(self, name):
            # row-major 4x4: identity rotation, translation (5, 6, 7)
            return (1, 0, 0, 5, 0, 1, 0, 6, 0, 0, 1, 7, 0, 0, 0, 1)

    assert _object_transform(FakeCmd(), "mobile") == {
        "rotation": _IDENTITY,
        "translation": [5.0, 6.0, 7.0],
    }


def test_pymol_object_transform_none_when_matrix_empty():
    class FakeCmd:
        def get_object_matrix(self, name):
            return ()

    assert _object_transform(FakeCmd(), "mobile") == {"rotation": None, "translation": None}


# ── Wrappers package the transform ────────────────────────────────────────────


def _structure():
    return Structure.from_file(_PDB_PATH)


def test_tmalign_wrapper_packages_superposition():
    inputs = TMalignInput(query_structure=_structure(), reference_structure=_structure())
    with patch("proto_tools.tools.structure_alignment.tmalign.tmalign.ToolInstance.dispatch") as mock:
        mock.return_value = {
            "tm_score_chain_1": 1.0,
            "tm_score_chain_2": 1.0,
            "rotation": _IDENTITY,
            "translation": [1.0, 2.0, 3.0],
        }
        result = run_tmalign(inputs, TMalignConfig())
    assert result.superposition is not None
    assert result.superposition.rotation == _IDENTITY
    assert result.superposition.translation == [1.0, 2.0, 3.0]


def test_usalign_wrapper_packages_superposition():
    inputs = USalignInput(query_structure=_structure(), reference_structure=_structure())
    with patch("proto_tools.tools.structure_alignment.usalign.usalign.ToolInstance.dispatch") as mock:
        mock.return_value = {
            "tm_score_structure_1": 1.0,
            "tm_score_structure_2": 1.0,
            "rotation": _IDENTITY,
            "translation": [1.0, 2.0, 3.0],
        }
        result = run_usalign(inputs, USalignConfig())
    assert result.superposition is not None
    assert result.superposition.translation == [1.0, 2.0, 3.0]


def test_pymol_wrapper_packages_superposition():
    inputs = PyMOLRMSDInput(target_structure=_structure(), mobile_structure=_structure())
    with patch("proto_tools.tools.structure_alignment.pymol_rmsd.pymol_rmsd.ToolInstance.dispatch") as mock:
        mock.return_value = {
            "method": "cealign",
            "rmsd": 0.0,
            "aligned_length": 42,
            "rotation": _IDENTITY,
            "translation": [4.0, 5.0, 6.0],
        }
        result = run_pymol_rmsd_alignment(inputs, PyMOLRMSDConfig())
    assert result.superposition is not None
    assert result.superposition.translation == [4.0, 5.0, 6.0]


def test_wrapper_builds_superimposed_pdb_when_enabled():
    inputs = TMalignInput(query_structure=_structure(), reference_structure=_structure())
    with patch("proto_tools.tools.structure_alignment.tmalign.tmalign.ToolInstance.dispatch") as mock:
        mock.return_value = {
            "tm_score_chain_1": 1.0,
            "tm_score_chain_2": 1.0,
            "rotation": _IDENTITY,
            "translation": [0.0, 0.0, 0.0],
        }
        result = run_tmalign(inputs, TMalignConfig(include_superimposed_pdb=True))
    assert result.superimposed_pdb is not None
    assert "MODEL        1" in result.superimposed_pdb
    assert "MODEL        2" in result.superimposed_pdb


def test_wrapper_omits_superimposed_pdb_by_default():
    inputs = TMalignInput(query_structure=_structure(), reference_structure=_structure())
    with patch("proto_tools.tools.structure_alignment.tmalign.tmalign.ToolInstance.dispatch") as mock:
        mock.return_value = {
            "tm_score_chain_1": 1.0,
            "tm_score_chain_2": 1.0,
            "rotation": _IDENTITY,
            "translation": [0.0, 0.0, 0.0],
        }
        result = run_tmalign(inputs, TMalignConfig())
    assert result.superimposed_pdb is None
    assert result.superposition is not None  # transform still returned


def test_pdb_export_format_offered_and_writes_overlay(tmp_path):
    inputs = TMalignInput(query_structure=_structure(), reference_structure=_structure())
    with patch("proto_tools.tools.structure_alignment.tmalign.tmalign.ToolInstance.dispatch") as mock:
        mock.return_value = {
            "tm_score_chain_1": 1.0,
            "tm_score_chain_2": 1.0,
            "rotation": _IDENTITY,
            "translation": [0.0, 0.0, 0.0],
        }
        result = run_tmalign(inputs, TMalignConfig(include_superimposed_pdb=True))

    # "pdb" is advertised only when an overlay exists, and exporting it writes the file.
    assert "pdb" in result.output_format_options
    result.export(name="overlay", export_path=tmp_path, file_format="pdb")
    written = tmp_path / "overlay.pdb"
    assert written.exists()
    assert "MODEL        1" in written.read_text()


def test_pdb_export_format_hidden_without_overlay():
    inputs = TMalignInput(query_structure=_structure(), reference_structure=_structure())
    with patch("proto_tools.tools.structure_alignment.tmalign.tmalign.ToolInstance.dispatch") as mock:
        mock.return_value = {"tm_score_chain_1": 1.0, "tm_score_chain_2": 1.0}
        result = run_tmalign(inputs, TMalignConfig())
    assert result.output_format_options == ["json"]


def test_wrapper_superposition_is_none_when_transform_absent():
    inputs = TMalignInput(query_structure=_structure(), reference_structure=_structure())
    with patch("proto_tools.tools.structure_alignment.tmalign.tmalign.ToolInstance.dispatch") as mock:
        mock.return_value = {"tm_score_chain_1": 1.0, "tm_score_chain_2": 1.0}
        result = run_tmalign(inputs, TMalignConfig())
    assert result.superposition is None

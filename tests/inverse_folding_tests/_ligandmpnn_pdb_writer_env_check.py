"""In-env check that the Foundry path's PDB writer survives non-finite annotations.

Foundry's output atom array keeps the input's annotations, so an RFD4 backbone, whose CIF
writes ``B_iso_or_equiv = nan`` for designed residues, reaches ``_atom_array_to_pdb`` with
NaN B-factors. Biotite's fixed-width writer rejects those. This builds such an array, first
confirms Biotite itself rejects it (so the check exercises the real failure), then requires
``_atom_array_to_pdb`` to write it with Biotite's absent-annotation defaults, keep every
finite value, drop the atom without coordinates, and leave the caller's array unchanged.

Run by ``test_ligandmpnn_pdb_writer_handles_non_finite_annotations`` in ``test_ligandmpnn.py``.
Plain asserts; a non-zero exit surfaces as ``RuntimeError``.
"""

import io

import biotite.structure as struc
import numpy as np
from biotite.structure.io.pdb import PDBFile


def _array() -> struc.AtomArray:
    """Six backbone-like atoms: NaN and inf annotations on some, one atom without coordinates."""
    atoms = struc.AtomArray(6)
    atoms.coord = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0], [3.5, 1.0, 0.0], [np.nan, np.nan, np.nan], [5.0, 2.0, 0.0]]
    )
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = [1, 1, 1, 2, 2, 2]
    atoms.res_name[:] = "GLY"
    atoms.atom_name[:] = ["N", "CA", "C", "N", "CA", "C"]
    atoms.element[:] = ["N", "C", "C", "N", "C", "C"]
    atoms.hetero[:] = False
    atoms.set_annotation("b_factor", np.array([12.5, np.nan, np.inf, 30.0, np.nan, np.nan]))
    atoms.set_annotation("occupancy", np.array([0.5, 1.0, np.nan, 1.0, np.nan, 0.25]))
    return atoms


def main() -> int:
    """Run the control, then the writer, and compare what was written."""
    import inference
    from biotite.structure.io.pdb.file import BadStructureError

    atoms = _array()
    finite = np.isfinite(atoms.coord).all(axis=-1)

    # Control: the same array without its NaN-coordinate atom must still crash Biotite's writer,
    # or this check would pass without ever reaching the failure it guards.
    try:
        PDBFile().set_structure(atoms[finite])
    except BadStructureError as exc:
        print(f"CONTROL biotite rejects the array: {exc}")
    else:
        raise AssertionError("Biotite wrote non-finite B-factors; the check no longer exercises the bug")

    before = {name: atoms.get_annotation(name).copy() for name in ("b_factor", "occupancy")}
    text = inference._atom_array_to_pdb(atoms)

    for name, values in before.items():
        np.testing.assert_array_equal(atoms.get_annotation(name), values, err_msg=f"caller's {name} was modified")

    written = PDBFile.read(io.StringIO(text)).get_structure(model=1, extra_fields=["b_factor", "occupancy"])
    assert written.array_length() == int(finite.sum()), f"wrote {written.array_length()} atoms"
    np.testing.assert_allclose(written.coord, atoms.coord[finite], atol=1e-3)
    np.testing.assert_allclose(written.b_factor, [12.5, 0.0, 0.0, 30.0, 0.0])
    np.testing.assert_allclose(written.occupancy, [0.5, 1.0, 1.0, 1.0, 0.25])
    print("PASS non-finite b_factor -> 0.00, occupancy -> 1.00, finite values kept")

    # An array without either annotation still takes Biotite's own defaults.
    bare = atoms[finite]
    bare.del_annotation("b_factor")
    bare.del_annotation("occupancy")
    bare_written = PDBFile.read(io.StringIO(inference._atom_array_to_pdb(bare))).get_structure(
        model=1, extra_fields=["b_factor", "occupancy"]
    )
    np.testing.assert_allclose(bare_written.b_factor, 0.0)
    np.testing.assert_allclose(bare_written.occupancy, 1.0)
    print("PASS absent annotations -> Biotite defaults")

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

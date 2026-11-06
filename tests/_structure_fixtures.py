"""Shared test-data builders for Structure / structure-scoring tests.

Used by structure, structure-scoring, and structure-prediction tests.
Lives at the ``tests/`` root so every test package can import it.
"""

from __future__ import annotations


def synthetic_cif(chain_names: list[str]) -> str:
    """Build a minimal valid mmCIF with one glycine residue per named chain.

    Each chain gets four atoms (N, CA, C, O) so gemmi accepts it as a real
    residue. Chains are spaced 100 A apart on the x-axis to avoid any overlap
    artifacts.

    Args:
        chain_names (list[str]): Chain labels to emit. Multi-character labels
            (e.g. ``"Heavy"``) are valid under mmCIF and are useful for testing
            the PDB-shortening / restoration path.

    Returns:
        str: mmCIF content as a single string.
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
    rows = []
    atom_id = 1
    for chain_idx, name in enumerate(chain_names):
        base_x = chain_idx * 100.0
        label_asym = chr(ord("A") + chain_idx % 26)
        for atom_name, dx, dy in [("N", 0.0, 0.0), ("CA", 1.5, 0.0), ("C", 2.0, 1.5), ("O", 1.3, 2.5)]:
            rows.append(
                f"ATOM {atom_id} {atom_name[0]} {atom_name} . GLY {label_asym} 1 1 ? "
                f"{base_x + dx:.3f} {dy:.3f} 0.000 1.00 20.00 1 GLY {name} {atom_name} 1"
            )
            atom_id += 1
    return header + "\n".join(rows) + "\n"


def single_chain_pdb(num_residues: int) -> str:
    """Build a single-chain PDB (chain A) of ``num_residues`` glycine residues.

    Args:
        num_residues (int): Residues to emit, four backbone atoms each.

    Returns:
        str: PDB content as a single string.
    """
    lines = []
    atom_id = 1
    for residue_id in range(1, num_residues + 1):
        base = float(residue_id * 3)
        for atom_name, dx, dy in [("N", 0.0, 0.0), ("CA", 1.4, 0.0), ("C", 2.1, 1.2), ("O", 1.7, 2.3)]:
            lines.append(
                f"ATOM  {atom_id:5d} {atom_name:<4} GLY A{residue_id:4d}    "
                f"{base + dx:8.3f}{dy:8.3f}{0.0:8.3f}{1.0:6.2f}{0.8:6.2f}           {atom_name[0]:>2}"
            )
            atom_id += 1
    lines.append("END")
    return "\n".join(lines) + "\n"

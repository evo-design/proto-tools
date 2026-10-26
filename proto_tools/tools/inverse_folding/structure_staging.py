"""proto_tools/tools/inverse_folding/structure_staging.py.

Staging a structure for an inverse-folding model and reading its output back.

Every one of these models addresses a residue by its number alone, so two residues that
share a number through an insertion code -- ``47`` and ``47A``, as antibody numbering
produces by design -- collapse onto one and the model silently returns fewer residues than
the chain holds. Staging renumbers each chain ``1..N``, which makes a residue's number equal
its 1-indexed position, and :meth:`StagedStructure.restore` puts the input's numbering back
on whatever the model writes.
"""

import contextlib
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from proto_tools.entities.structures import ResidueSelection, Structure


def chain_residue_ids(structure: Structure) -> dict[str, list[tuple[int, str]]]:
    """Per-chain polymer residue IDs as ``(number, insertion code)``, in chain order.

    Unlike ``Structure.get_chain_positions``, keeps the insertion code, so ``10`` and
    ``10A`` stay distinct residues.

    Args:
        structure (Structure): The structure to read.

    Returns:
        dict[str, list[tuple[int, str]]]: Chain ID to residue IDs, first model only.
    """
    gemmi_struct = structure.gemmi_struct
    gemmi_struct.setup_entities()
    return {
        chain.name: [(residue.seqid.num, residue.seqid.icode.strip()) for residue in chain.get_polymer()]  # type: ignore[misc]
        for chain in gemmi_struct[0]
        if chain.get_polymer()
    }


def sequential_pdb(structure: Structure) -> tuple[str, dict[str, str]]:
    """PDB content with each chain's residues renumbered 1..N and insertion codes cleared.

    These models address residues as ``<chain><number>`` with no insertion code, so ``10``
    and ``10A`` would resolve to the same residue. Numbering by ordinal makes a residue's
    number equal its 1-indexed position, which is how every selection in these toolkits is
    written. The models are unaffected, since their residue-index feature is already a
    within-chain ordinal. Non-polymer residues are numbered after the polymer so none can
    share a number with a polymer residue.

    Multi-character CIF chain IDs, which PDB cannot hold, are shortened to single
    characters; the returned map translates selections in and output chains back.

    Args:
        structure (Structure): The structure to renumber.

    Returns:
        tuple[str, dict[str, str]]: The renumbered structure as PDB content, and the map
            from each original chain ID to its chain ID in that content.
    """
    import gemmi

    pdb_content, chain_map = structure.to_pdb_with_chain_mapping()
    gemmi_struct = gemmi.read_pdb_string(pdb_content)
    gemmi_struct.setup_entities()
    for model in gemmi_struct:
        for chain in model:
            polymer = chain.get_polymer()
            polymer_subchain = polymer.subchain_id() if polymer else None
            polymer_count = len(polymer)
            polymer_index = other_index = 0
            for residue in chain:
                if polymer_subchain is not None and residue.subchain == polymer_subchain:
                    polymer_index += 1
                    residue.seqid = gemmi.SeqId(polymer_index, " ")
                else:
                    other_index += 1
                    residue.seqid = gemmi.SeqId(polymer_count + other_index, " ")
    return gemmi_struct.make_pdb_string(), chain_map


def restore_numbering(
    content: str,
    residue_ids: dict[str, list[tuple[int, str]]],
    source: str,
    chain_map: dict[str, str] | None = None,
    structure_format: Literal["pdb", "cif"] = "cif",
) -> Structure:
    """Wrap a model's output structure, giving its chains back their input IDs and numbering.

    A model writes the sequentially numbered structure it was given, so each chain's
    ``i``-th polymer residue takes the ``i``-th input residue ID, insertion code included.

    Args:
        content (str): The structure content the model wrote.
        residue_ids (dict[str, list[tuple[int, str]]]): Input residue IDs, as
            ``chain_residue_ids`` returns them.
        source (str): ``source`` label for the returned structure.
        chain_map (dict[str, str] | None): Original to staged chain IDs, as
            :func:`sequential_pdb` returns them. ``None`` means the IDs are unchanged.
        structure_format (Literal["pdb", "cif"]): Format of ``content``, which the returned
            structure keeps, except that restoring a chain ID too long for PDB forces CIF.

    Returns:
        Structure: The output structure in the input structure's numbering.

    Raises:
        ValueError: If an output chain is absent from the input or has a different
            number of polymer residues, so its residues cannot be matched back.
    """
    import gemmi

    gemmi_struct = Structure(structure=content, structure_format=structure_format, source=source).gemmi_struct
    gemmi_struct.setup_entities()
    original_chain_ids = {staged: original for original, staged in (chain_map or {}).items()}
    for model in gemmi_struct:
        for chain in model:
            chain.name = original_chain_ids.get(chain.name, chain.name)
            polymer = chain.get_polymer()
            if not polymer:
                continue
            original = residue_ids.get(chain.name)
            if original is None or len(original) != len(polymer):
                raise ValueError(
                    f"{source}: output chain {chain.name} has {len(polymer)} residues but the input chain has "
                    f"{len(original) if original is not None else 'none'}, so its residue numbering cannot be restored"
                )
            for residue, (number, icode) in zip(polymer, original, strict=True):
                residue.seqid = gemmi.SeqId(number, icode or " ")
    # A chain ID that had to be shortened for PDB staging cannot be written back to PDB.
    if structure_format == "pdb" and not any(len(original) > 1 for original in (chain_map or {})):
        return Structure(structure=gemmi_struct.make_pdb_string(), structure_format="pdb", source=source)
    return Structure(structure=gemmi_struct.make_mmcif_document().as_string(), structure_format="cif", source=source)


@dataclass(frozen=True)
class StagedStructure:
    """An input structure as a model sees it, plus what is needed to read its output back.

    Attributes:
        path (Path): The staged PDB file, renumbered ``1..N`` per chain.
        chain_map (dict[str, str]): Original chain ID to its ID in the staged file.
        residue_ids (dict[str, list[tuple[int, str]]]): The input's residue IDs per chain,
            as :func:`chain_residue_ids` returns them.
    """

    path: Path
    chain_map: dict[str, str]
    residue_ids: dict[str, list[tuple[int, str]]]

    def chains(self, chain_ids: Iterable[str]) -> list[str]:
        """Translate input chain IDs to the staged file's, in the order given.

        Args:
            chain_ids (Iterable[str]): Chain IDs as the caller's input names them.

        Returns:
            list[str]: The same chains as the staged file names them.
        """
        return [self.chain_map[chain_id] for chain_id in chain_ids]

    def original_chain(self, staged_chain_id: str) -> str:
        """Translate one chain ID in the staged file back to the input's.

        Args:
            staged_chain_id (str): Chain ID as the model reported it.

        Returns:
            str: The chain ID the input structure uses.
        """
        for original, staged in self.chain_map.items():
            if staged == staged_chain_id:
                return original
        return staged_chain_id

    def positions(self, selection: ResidueSelection | None) -> dict[str, list[int]] | None:
        """Resolve a position selection against the staged file's numbering.

        Staging numbers each chain by ordinal, so a 1-indexed position is already the
        residue number the model resolves; only the chain IDs need translating.

        Args:
            selection (ResidueSelection | None): The selection to resolve, or ``None``.

        Returns:
            dict[str, list[int]] | None: Per-chain residue numbers in the staged file, or
                ``None`` when no selection was given.
        """
        if selection is None:
            return None
        return {self.chain_map[chain_id]: list(positions) for chain_id, positions in selection.chains.items()}

    def restore(self, content: str, *, source: str, structure_format: Literal["pdb", "cif"] = "pdb") -> Structure:
        """Put the input structure's chain IDs and residue numbering back on model output.

        Args:
            content (str): The structure content the model wrote.
            source (str): ``source`` label for the returned structure.
            structure_format (Literal["pdb", "cif"]): Format of ``content``, which the
                returned structure keeps where it can.

        Returns:
            Structure: The output structure in the input structure's numbering.
        """
        return restore_numbering(
            content,
            self.residue_ids,
            source=source,
            chain_map=self.chain_map,
            structure_format=structure_format,
        )


@contextlib.contextmanager
def staged_structure(structure: Structure) -> Iterator[StagedStructure]:
    """Materialize one structure as a sequentially numbered temp PDB file.

    PDB rather than CIF because these models read chain IDs from a PDB's chain column, and
    ``Structure.write_cif`` leaves ``label_asym_id`` unset.

    Args:
        structure (Structure): The structure to stage.

    Yields:
        StagedStructure: The staged file and the maps needed to read output back.
    """
    with staged_ensemble([structure]) as (staged, _):
        yield staged


@contextlib.contextmanager
def staged_ensemble(structures: list[Structure]) -> Iterator[tuple[StagedStructure, list[Path]]]:
    """Materialize an ensemble to sequentially numbered temp PDB files, primary conformer first.

    Each file is renumbered by :func:`sequential_pdb`, so a 1-indexed position is also the
    residue number the model resolves, in every conformer.

    Args:
        structures (list[Structure]): The ensemble, primary conformer first.

    Yields:
        tuple[StagedStructure, list[Path]]: The primary conformer as a
            :class:`StagedStructure`, and the other members' paths in ensemble order
            (empty without an ensemble).

    Raises:
        ValueError: If a conformer's chain IDs shorten differently from the primary's.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        chain_maps = []
        for index, structure in enumerate(structures):
            pdb_content, chain_map = sequential_pdb(structure)
            path = Path(tmp_dir) / f"structure_{index}.pdb"
            path.write_text(pdb_content)
            paths.append(path)
            chain_maps.append(chain_map)
        for index, chain_map in enumerate(chain_maps[1:]):
            if chain_map != chain_maps[0]:
                raise ValueError(
                    f"conformers[{index}] chain IDs shorten to {chain_map} for PDB staging but the primary "
                    f"structure's shorten to {chain_maps[0]}; give the ensemble identical chain IDs"
                )
        staged = StagedStructure(
            path=paths[0],
            chain_map=chain_maps[0],
            residue_ids=chain_residue_ids(structures[0]),
        )
        yield staged, paths[1:]

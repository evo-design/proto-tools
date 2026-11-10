"""proto_tools/tools/inverse_folding/structure_staging.py.

Staging a structure for an inverse-folding model and reading its output back.

Every one of these models addresses a residue by its number alone, so two residues that
share a number through an insertion code -- ``47`` and ``47A``, as antibody numbering
produces by design -- collapse onto one and the model silently returns fewer residues than
the chain holds. Staging renumbers each chain ``1..N``, which makes a residue's number equal
its 1-indexed position, and :meth:`StagedStructure.restore` puts the input's numbering back
on whatever the model writes.

The structure operations themselves live on ``Structure``
(``get_chain_residue_ids``, ``renumbered_sequentially``, ``with_residue_ids``); this module
bundles them with the temp file and the selection translation each model run needs.
"""

import contextlib
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from proto_tools.entities.structures import ResidueSelection, Structure


@dataclass(frozen=True)
class StagedStructure:
    """An input structure as a model sees it, plus what is needed to read its output back.

    Attributes:
        path (Path): The staged PDB file, renumbered ``1..N`` per chain.
        chain_map (dict[str, str]): Original chain ID to its ID in the staged file.
        residue_ids (dict[str, list[tuple[int, str]]]): The input's residue IDs per chain,
            as ``Structure.get_chain_residue_ids`` returns them.
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

    def restore(
        self,
        content: str,
        *,
        source: str,
        structure_format: Literal["pdb", "cif"] = "pdb",
        readable_positions: dict[str, list[int]] | None = None,
    ) -> Structure:
        """Put the input structure's chain IDs and residue numbering back on model output.

        Args:
            content (str): The structure content the model wrote.
            source (str): ``source`` label for the returned structure.
            structure_format (Literal["pdb", "cif"]): Format of ``content``, which the
                returned structure keeps where it can.
            readable_positions (dict[str, list[int]] | None): Per chain, the 1-indexed input
                positions the model reads, from ``Config.readable_positions``. A model that
                skips a residue may write a chain covering only the positions it read, so a
                chain that comes back that short is matched against those IDs instead of the
                whole chain's. A chain written in full keeps the whole chain's IDs either way,
                which is what a model that reads a chain only as context does with it.
                ``None`` expects one output residue per input residue.

        Returns:
            Structure: The output structure in the input structure's numbering.
        """
        written = Structure(structure=content, structure_format=structure_format, source=source)
        residue_ids = self.residue_ids
        if readable_positions is not None:
            residue_ids = {
                chain_id: self._ids_matching_output(chain_id, ids, readable_positions, written)
                for chain_id, ids in residue_ids.items()
            }
        return written.with_residue_ids(residue_ids, chain_map=self.chain_map)

    def _ids_matching_output(
        self,
        chain_id: str,
        ids: list[tuple[int, str]],
        readable_positions: dict[str, list[int]],
        written: Structure,
    ) -> list[tuple[int, str]]:
        """Pick the ID list for one chain that matches what the model actually wrote.

        A chain can come back either way. FAMPNN packs only the residues it read, so its
        chains are as short as ``readable_positions`` says. LigandMPNN writes a chain it reads
        only as ligand context in full, even though it designs none of it. Choosing by the
        length the model wrote keeps both, and leaves a genuine mismatch to
        :meth:`Structure.with_residue_ids`, which reports it.

        Args:
            chain_id (str): The input's chain ID.
            ids (list[tuple[int, str]]): That chain's residue IDs, whole chain.
            readable_positions (dict[str, list[int]]): Positions the model reads, per chain.
            written (Structure): What the model wrote.

        Returns:
            list[tuple[int, str]]: The whole chain's IDs, or only the readable ones.
        """
        readable = readable_positions.get(chain_id)
        if readable is None or len(readable) == len(ids):
            return ids
        staged_id = self.chain_map.get(chain_id, chain_id)
        written_length = len(written.get_chain_residue_atoms().get(staged_id, ()))
        if written_length == len(readable) != len(ids):
            return [ids[position - 1] for position in readable]
        return ids


@contextlib.contextmanager
def staged_structure(
    structure: Structure, *, atom_record_residues: frozenset[str] = frozenset()
) -> Iterator[StagedStructure]:
    """Materialize one structure as a sequentially numbered temp PDB file.

    PDB rather than CIF because these models read chain IDs from a PDB's chain column, and
    ``Structure.write_cif`` leaves ``label_asym_id`` unset.

    Args:
        structure (Structure): The structure to stage.
        atom_record_residues (frozenset[str]): Residue names to write as ``ATOM`` rather than
            ``HETATM``; see :func:`staged_ensemble`.

    Yields:
        StagedStructure: The staged file and the maps needed to read output back.
    """
    with staged_ensemble([structure], atom_record_residues=atom_record_residues) as (staged, _):
        yield staged


def _as_atom_records(pdb_content: str, residue_names: frozenset[str]) -> str:
    """Rewrite the named residues' ``HETATM`` lines to ``ATOM``.

    A modified amino acid is part of the polymer, but the PDB convention writes it as
    ``HETATM``. A parser that recovers it from a table of known names does not care, while one
    that keys on the record type alone reads it as a ligand instead. This lets a toolkit stage
    the names its model would otherwise lose, so the same residue is read the same way across
    toolkits.

    Args:
        pdb_content (str): The staged PDB text.
        residue_names (frozenset[str]): Residue names to write as ``ATOM``.

    Returns:
        str: The same text with those residues' record type rewritten.
    """
    if not residue_names:
        return pdb_content
    # Columns 1-6 are the record name and 18-20 the residue name.
    return "".join(
        "ATOM  " + line[6:] if line.startswith("HETATM") and line[17:20].strip() in residue_names else line
        for line in pdb_content.splitlines(keepends=True)
    )


@contextlib.contextmanager
def staged_ensemble(
    structures: list[Structure], *, atom_record_residues: frozenset[str] = frozenset()
) -> Iterator[tuple[StagedStructure, list[Path]]]:
    """Materialize an ensemble to sequentially numbered temp PDB files, primary conformer first.

    Each file is renumbered by ``Structure.renumbered_sequentially``, so a 1-indexed position
    is also the residue number the model resolves, in every conformer.

    Args:
        structures (list[Structure]): The ensemble, primary conformer first.
        atom_record_residues (frozenset[str]): Residue names to write as ``ATOM`` rather than
            ``HETATM``, for a model whose parser reads the record type rather than a table of
            known modified residues.

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
            pdb_content, chain_map = structure.renumbered_sequentially()
            pdb_content = _as_atom_records(pdb_content, atom_record_residues)
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
            residue_ids=structures[0].get_chain_residue_ids(),
        )
        yield staged, paths[1:]

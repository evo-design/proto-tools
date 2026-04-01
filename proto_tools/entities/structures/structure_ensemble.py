"""proto_tools/entities/structures/structure_ensemble.py.

Contains base class for representing a protein structure ensemble.
"""

from __future__ import annotations

from pydantic import BaseModel

from proto_tools.entities.structures.structure import Structure


class StructureEnsemble(BaseModel):
    """Container for a conformational ensemble of structures.

    This class represents an ensemble of sampled conformations for a single
    input sequence, with each conformation stored as a Structure.

    Attributes:
        structures (list[Structure]): List of sampled conformational
            structures. Each Structure represents a single backbone
            conformation from the ensemble.

        sequence (str): The input protein sequence.
    """

    structures: list[Structure]
    sequence: str

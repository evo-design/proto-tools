"""proto_tools/entities/structures/structure_ensemble.py.

Contains base class for representing a protein structure ensemble.
"""

from pydantic import BaseModel, Field

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

    structures: list[Structure] = Field(
        title="Structures",
        description="Sampled conformational structures, one per frame in the ensemble.",
    )
    sequence: str = Field(
        title="Sequence",
        description="Input protein sequence shared by all conformations in the ensemble.",
    )

    def approx_equal(self, other: "StructureEnsemble", rtol: float = 1e-4, atol: float = 1e-6) -> None:
        """Assert that two structure ensembles are approximately equal.

        Compares sequences exactly, structure counts exactly, and delegates
        per-structure comparison to ``Structure.approx_equal``.

        Args:
            other (StructureEnsemble): The other ensemble to compare against.
            rtol (float): Relative tolerance for coordinate comparison.
            atol (float): Absolute tolerance for coordinate comparison.

        Raises:
            AssertionError: If the ensembles differ.
        """
        if self.sequence != other.sequence:
            raise AssertionError(f"Ensemble sequences differ: {self.sequence!r} != {other.sequence!r}")
        if len(self.structures) != len(other.structures):
            raise AssertionError(f"Ensemble structure count differs: {len(self.structures)} != {len(other.structures)}")
        for s, o in zip(self.structures, other.structures, strict=True):
            s.approx_equal(o, rtol=rtol, atol=atol)

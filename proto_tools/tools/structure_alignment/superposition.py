"""proto_tools/tools/structure_alignment/superposition.py.

Shared rigid-body superposition transform returned by the pairwise structure-alignment
tools (TMalign, USalign, PyMOL-RMSD). The underlying binaries compute the superposition;
this carries the transform out so a viewer can overlay the two structures.
"""

from pydantic import BaseModel, Field, model_validator


class SuperpositionTransform(BaseModel):
    """Rigid-body transform that superposes the query/mobile structure onto the reference/target.

    Apply it to each query/mobile coordinate to bring it into the reference frame::

        x_ref ≈ rotation · x_query + translation

    Attributes:
        rotation (list[list[float]]): 3x3 row-major rotation matrix applied to the
            query/mobile coordinates.
        translation (list[float]): Length-3 translation vector (Angstrom) added after
            the rotation.
    """

    rotation: list[list[float]] = Field(
        title="Rotation",
        description="3x3 row-major rotation matrix applied to the query/mobile coordinates.",
    )
    translation: list[float] = Field(
        title="Translation",
        description="Length-3 translation vector (Angstrom) added after the rotation.",
    )

    @model_validator(mode="after")
    def _check_shape(self) -> "SuperpositionTransform":
        if len(self.rotation) != 3 or any(len(row) != 3 for row in self.rotation):
            raise ValueError("rotation must be a 3x3 matrix")
        if len(self.translation) != 3:
            raise ValueError("translation must have length 3")
        return self

    @classmethod
    def from_optional(
        cls,
        rotation: list[list[float]] | None,
        translation: list[float] | None,
    ) -> "SuperpositionTransform | None":
        """Build a transform when both parts are present, else ``None``.

        The standalone runners return ``None`` for both when the binary didn't emit a
        parseable transform, so the alignment scores still succeed without it.
        """
        if rotation is None or translation is None:
            return None
        return cls(rotation=rotation, translation=translation)

    def apply(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Apply the transform to a single (x, y, z) coordinate."""
        r, t = self.rotation, self.translation
        return (
            r[0][0] * x + r[0][1] * y + r[0][2] * z + t[0],
            r[1][0] * x + r[1][1] * y + r[1][2] * z + t[1],
            r[2][0] * x + r[2][1] * y + r[2][2] * z + t[2],
        )


def _transform_pdb_atoms(pdb_text: str, transform: SuperpositionTransform) -> str:
    """Rewrite every ATOM/HETATM coordinate in ``pdb_text`` through ``transform``.

    Coordinates live in the fixed-width PDB columns 31-54 (x, y, z as ``%8.3f``); non-atom
    lines and any record too short to hold coordinates pass through untouched.
    """
    out: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                out.append(line)
                continue
            nx, ny, nz = transform.apply(x, y, z)
            out.append(f"{line[:30]}{nx:8.3f}{ny:8.3f}{nz:8.3f}{line[54:]}")
        else:
            out.append(line)
    return "\n".join(out)


def _atom_records(pdb_text: str) -> list[str]:
    """Keep only the coordinate/terminator records (drop MODEL/END framing and headers)."""
    return [line for line in pdb_text.splitlines() if line.startswith(("ATOM", "HETATM", "TER"))]


def build_superimposed_pdb(
    query_pdb: str,
    reference_pdb: str,
    transform: SuperpositionTransform,
) -> str:
    """Combine the two structures into a multi-MODEL PDB overlaying the alignment.

    MODEL 1 is the query/mobile structure moved by ``transform`` into the reference frame;
    MODEL 2 is the reference/target as-is. The ``MODEL`` records keep the two sets of chain
    IDs from colliding, and standard viewers overlay the models — a portable, downloadable
    form of the superposition (exported via the tool's ``pdb`` output format).
    """
    moved = _transform_pdb_atoms(query_pdb, transform)
    lines = [
        "REMARK   Superimposed structures (proto-tools structure alignment).",
        "REMARK   MODEL 1 = query/mobile (transformed); MODEL 2 = reference/target.",
        "MODEL        1",
        *_atom_records(moved),
        "ENDMDL",
        "MODEL        2",
        *_atom_records(reference_pdb),
        "ENDMDL",
        "END",
    ]
    return "\n".join(lines) + "\n"

"""Small-molecule ligand representations as Pydantic BaseModels."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from proto_tools.entities.ligands.utils import is_smiles_valid

if TYPE_CHECKING:
    from rdkit import Chem


class Fragment(BaseModel):
    """A single small molecule with optional CCD code.

    At least one of ``smiles`` or ``ccd_code`` must be provided. When only one
    is given, the other is resolved automatically from the CCD database. When
    both are given, they are validated to refer to the same molecule.

    A Fragment must contain exactly one molecule. For multi-component inputs
    (dot-separated SMILES, multiple CCD codes), use ``Ligands`` instead.

    The RDKit ``Mol`` object is lazy-loaded from the SMILES on first access
    via the ``mol`` property. Use ``Fragment.from_mol()`` to construct from an
    existing Mol.

    Attributes:
        smiles (str | None): SMILES string (canonicalized to RDKit form on construction).
        ccd_code (str | None): CCD code from the Chemical Component Dictionary.
        entity_type (Literal["ligand"]): Always ``"ligand"``. Lets Fragments be used
            interchangeably with ``Chain`` in structure-prediction inputs.
        name (str | None): Human-readable molecule name.
        metrics (dict[str, float]): Computed metrics for this fragment.
    """

    model_config = ConfigDict(extra="forbid")

    smiles: str | None = Field(default=None, description="SMILES string")
    ccd_code: str | None = Field(default=None, description="CCD code")
    entity_type: Literal["ligand"] = Field(default="ligand", description="Always 'ligand'")
    name: str | None = Field(default=None, description="Molecule name")
    metrics: dict[str, float] = Field(default_factory=dict, description="Computed metrics")

    _mol: Any = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_smiles_and_ccd(cls, data: Any) -> Any:
        """Resolve and cross-validate SMILES and CCD code."""
        if not isinstance(data, dict):
            return data

        from rdkit import Chem

        from proto_tools.entities.ligands.ccd_utils import (
            is_valid_ccd_code,
            map_ccd_code_to_smiles,
            map_smiles_to_ccd_code,
        )

        raw_smiles = data.get("smiles")
        raw_ccd = data.get("ccd_code")

        if raw_smiles is None and raw_ccd is None:
            raise ValueError("At least one of 'smiles' or 'ccd_code' must be provided")

        # Resolve SMILES → canonical form + attempt CCD lookup
        canonical_smiles: str | None = None
        if raw_smiles is not None:
            parsed = Chem.MolFromSmiles(raw_smiles)
            if parsed is None:
                raise ValueError(f"Invalid SMILES string: {raw_smiles}")
            num_frags = len(Chem.rdmolops.GetMolFrags(parsed, asMols=False))
            if num_frags > 1:
                raise ValueError(
                    f"Fragment must contain exactly one molecule; got {num_frags} "
                    f"in SMILES '{raw_smiles}'. Use `Ligands(smiles=...)` to "
                    "construct a multi-fragment collection."
                )
            canonical_smiles = Chem.MolToSmiles(parsed, canonical=True)
            data["smiles"] = canonical_smiles

        # Resolve CCD code → SMILES
        if raw_ccd is not None:
            if not is_valid_ccd_code(raw_ccd):
                raise ValueError(f"Invalid CCD code: {raw_ccd}")
            ccd_smiles = map_ccd_code_to_smiles(raw_ccd)
            if ccd_smiles is None:
                raise ValueError(f"CCD code '{raw_ccd}' has no parseable SMILES in the database")
            data["ccd_code"] = raw_ccd.upper()
            if canonical_smiles is None:
                data["smiles"] = ccd_smiles
            elif canonical_smiles != ccd_smiles:
                raise ValueError(
                    f"SMILES and CCD code refer to different molecules: "
                    f"smiles='{canonical_smiles}' but CCD {raw_ccd.upper()} resolves to '{ccd_smiles}'"
                )
        elif canonical_smiles is not None:
            data["ccd_code"] = map_smiles_to_ccd_code(canonical_smiles)

        return data

    # ============================================================================
    # Factory
    # ============================================================================

    @classmethod
    def from_mol(cls, mol: Chem.Mol, name: str | None = None) -> Fragment:
        """Create a Fragment from an RDKit Mol object.

        Args:
            mol (Chem.Mol): RDKit molecule (must contain exactly one fragment).
            name (str | None): Optional molecule name.

        Returns:
            Fragment: The constructed fragment.

        Raises:
            ValueError: If the Mol contains more than one fragment.
        """
        from rdkit import Chem

        fragments = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        if len(fragments) > 1:
            msg = "Invalid Mol as Fragment input: Mol must contain only one fragment"
            raise ValueError(msg)
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
        frag = cls(smiles=smiles, name=name)
        frag._mol = Chem.AddHs(mol)
        return frag

    # ============================================================================
    # Mol access
    # ============================================================================

    @property
    def mol(self) -> Chem.Mol:
        """Lazy-loaded RDKit Mol object with explicit hydrogens.

        Returns:
            Chem.Mol: The molecule with hydrogens added.
        """
        from rdkit import Chem

        if self._mol is None:
            self._mol = Chem.AddHs(Chem.MolFromSmiles(self.smiles))
        return self._mol  # type: ignore[no-any-return]

    @property
    def conformers(self) -> list[Chem.Conformer]:
        """All conformers present on the underlying Mol object.

        Returns:
            list[Chem.Conformer]: List of 3D conformers.
        """
        return [self.mol.GetConformer(i) for i in range(self.mol.GetNumConformers())]

    def generate_conformers(
        self, num_conformers: int = 1, random_seed: int | None = 42, prune_rms_threshold: float = 0.5
    ) -> None:
        """Generate 3D conformers for this molecule using RDKit ETKDGv3.

        Args:
            num_conformers (int): Number of conformers to generate.
            random_seed (int | None): Random seed for reproducibility.
            prune_rms_threshold (float): RMS threshold to prune similar conformers (Angstroms).

        Raises:
            RuntimeError: If conformer generation fails.
        """
        from rdkit.Chem import AllChem

        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        if random_seed is not None:
            params.randomSeed = random_seed
        params.pruneRmsThresh = prune_rms_threshold

        conf_ids = AllChem.EmbedMultipleConfs(self.mol, numConfs=num_conformers, params=params)  # type: ignore[attr-defined]
        if not conf_ids:
            raise RuntimeError("Failed to generate conformers for this molecule.")

        for conf_id in conf_ids:
            AllChem.UFFOptimizeMolecule(self.mol, confId=conf_id)  # type: ignore[attr-defined]

    # ============================================================================
    # Display
    # ============================================================================

    def __str__(self) -> str:
        return f"Fragment(smiles={self.smiles}, name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fragment):
            return NotImplemented
        return self.smiles == other.smiles

    def __hash__(self) -> int:
        return hash(self.smiles)

    def visualize(self, width: int = 400, height: int = 400, style: str = "stick") -> None:
        """Render an interactive 3D visualization of this molecule.

        Args:
            width (int): Viewer width in pixels.
            height (int): Viewer height in pixels.
            style (str): Visualization style ('stick', 'sphere', etc.).
        """
        import py3Dmol
        from rdkit import Chem

        if len(self.conformers) == 0:
            self.generate_conformers(num_conformers=1)

        mol_block = Chem.MolToMolBlock(self.mol)
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(mol_block, "mol")

        if style == "stick":
            viewer.setStyle({"stick": {}})
        elif style == "sphere":
            viewer.setStyle({"sphere": {"scale": 0.3}})
        else:
            viewer.setStyle({style: {}})

        viewer.zoomTo()
        viewer.show()


class Ligands(BaseModel):
    """Collection of small-molecule fragments.

    Construct from explicit fragments, a dot-separated SMILES string, or a list
    of CCD codes. All three styles can be mixed in a single call:

    >>> Ligands(fragments=[Fragment(ccd_code="ATP")])
    >>> Ligands(smiles="ATP.MG")
    >>> Ligands(ccd_codes=["ATP", "MG", "MG"])
    >>> Ligands(smiles="CCO", ccd_codes=["MG"])  # combined → 2 fragments

    Attributes:
        fragments (list[Fragment]): The fragment molecules in this collection.
    """

    model_config = ConfigDict(extra="forbid")

    fragments: list[Fragment] = Field(default_factory=list, description="Fragment molecules")

    @model_validator(mode="before")
    @classmethod
    def _expand_inputs(cls, data: Any) -> Any:
        """Translate ``smiles=`` and ``ccd_codes=`` shorthand kwargs into ``fragments``."""
        if not isinstance(data, dict):
            return data
        smiles = data.pop("smiles", None)
        ccd_codes = data.pop("ccd_codes", None)
        fragments = list(data.get("fragments") or [])
        if smiles is not None:
            fragments.extend(parse_smiles_string(smiles))
        if ccd_codes is not None:
            fragments.extend(Fragment(ccd_code=code) for code in ccd_codes)
        data["fragments"] = fragments
        return data

    # ============================================================================
    # Factories
    # ============================================================================

    @classmethod
    def from_smiles(cls, smiles: str) -> Ligands:
        """Create Ligands from a dot-separated SMILES string.

        Args:
            smiles (str): SMILES string (dots separate multiple fragments).

        Returns:
            Ligands: Collection of parsed fragments.
        """
        frags = parse_smiles_string(smiles)
        return cls(fragments=frags)

    @classmethod
    def from_file(cls, path: str | Path) -> Ligands:
        """Load Ligands from a ``.smi`` or ``.sdf`` file.

        Args:
            path (str | Path): Path to the file.

        Returns:
            Ligands: Collection of parsed fragments.

        Raises:
            ValueError: If the file format is not supported.
            FileNotFoundError: If the file does not exist.
        """
        frags = parse_fragments_from_string_or_path(path)
        return cls(fragments=frags)

    @classmethod
    def from_mols(cls, mols: list[Chem.Mol]) -> Ligands:
        """Create Ligands from a list of RDKit Mol objects.

        Args:
            mols (list[Chem.Mol]): RDKit molecules.

        Returns:
            Ligands: Collection of parsed fragments.
        """
        frags = parse_fragments_from_mols(mols)
        return cls(fragments=frags)

    @classmethod
    def from_ccd_codes(cls, ccd_codes: list[str]) -> Ligands:
        """Create Ligands from a list of CCD codes.

        Args:
            ccd_codes (list[str]): CCD codes (e.g., ``["ATP", "ZN"]``).

        Returns:
            Ligands: Collection of fragments with resolved SMILES.
        """
        return cls(fragments=[Fragment(ccd_code=code) for code in ccd_codes])

    # ============================================================================
    # Collection operations
    # ============================================================================

    def add_fragment(self, fragment: Fragment) -> None:
        """Append a fragment to this collection.

        Args:
            fragment (Fragment): Fragment to add.
        """
        self.fragments.append(fragment)

    def generate_conformers(
        self, num_conformers: int = 1, random_seed: int | None = 42, prune_rms_threshold: float = 0.5
    ) -> None:
        """Generate 3D conformers for all fragments.

        Args:
            num_conformers (int): Number of conformers to generate.
            random_seed (int | None): Random seed for reproducibility.
            prune_rms_threshold (float): RMS threshold to prune similar conformers (Angstroms).
        """
        for fragment in self.fragments:
            fragment.generate_conformers(
                num_conformers=num_conformers, random_seed=random_seed, prune_rms_threshold=prune_rms_threshold
            )

    def get_smiles_list(self) -> list[str | None]:
        """Return SMILES strings for all fragments."""
        return [fragment.smiles for fragment in self.fragments]

    def get_names_list(self) -> list[str | None]:
        """Return names for all fragments."""
        return [fragment.name for fragment in self.fragments]

    @property
    def smiles(self) -> str:
        """Dot-separated SMILES for all fragments."""
        return ".".join(s for s in self.get_smiles_list() if s is not None)

    @property
    def ccd_codes(self) -> list[str | None]:
        """CCD codes for all fragments (None for molecules not in CCD)."""
        return [fragment.ccd_code for fragment in self.fragments]

    def __len__(self) -> int:
        return len(self.fragments)

    def __iter__(self) -> Iterator[Fragment]:  # type: ignore[override]
        return iter(self.fragments)

    def __getitem__(self, index: int) -> Fragment:
        return self.fragments[index]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Ligands(fragments={self.fragments})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ligands):
            return NotImplemented
        return set(self.get_smiles_list()) == set(other.get_smiles_list())

    def __hash__(self) -> int:
        return hash(frozenset(self.get_smiles_list()))

    # ============================================================================
    # I/O
    # ============================================================================

    def to_smi(self, filepath: str | Path) -> None:
        """Write all fragments to a ``.smi`` file.

        Args:
            filepath (str | Path): Destination file path.
        """
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            f.writelines(f"{frag.smiles}\t{frag.name}\n" for frag in self.fragments)

    def to_sdf(self, filepath: str | Path) -> None:
        """Write all fragments to an SDF file.

        Args:
            filepath (str | Path): Destination file path.
        """
        from rdkit import Chem

        filepath = Path(filepath)
        writer = Chem.SDWriter(str(filepath))

        for frag in self.fragments:
            if frag.mol.GetNumConformers() == 0:
                frag.generate_conformers(num_conformers=1)
            for i in range(frag.mol.GetNumConformers()):
                writer.write(frag.mol, confId=i)
        writer.close()

    def to_pdb(self, filepath: str | Path | None = None, spacing: float = 5.0) -> str:
        """Generate a PDB string with all fragments arranged along the X-axis.

        Each fragment gets a different chain ID (A, B, C, ...).

        Args:
            filepath (str | Path | None): Optional path to write the PDB file.
            spacing (float): Additional spacing (Angstroms) between bounding boxes.

        Returns:
            str: PDB format string.

        Raises:
            ValueError: If no fragments exist.
        """
        from rdkit import Chem

        if len(self.fragments) == 0:
            raise ValueError("Cannot generate PDB: no fragments in Ligands object")

        for frag in self.fragments:
            if frag.mol.GetNumConformers() == 0:
                frag.generate_conformers(num_conformers=1)

        pdb_lines = []
        current_atom_num = 1
        current_x_offset = 0.0
        chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for frag_idx, frag in enumerate(self.fragments):
            conf = frag.mol.GetConformer(0)

            positions = conf.GetPositions()
            min_coords = np.min(positions, axis=0)
            max_coords = np.max(positions, axis=0)
            bbox_size = max_coords - min_coords

            mol_copy = Chem.Mol(frag.mol)
            conf_copy = mol_copy.GetConformer(0)

            fragment_center = (min_coords + max_coords) / 2
            translation = np.array(
                [current_x_offset + bbox_size[0] / 2 - fragment_center[0], -fragment_center[1], -fragment_center[2]]
            )

            for atom_idx in range(mol_copy.GetNumAtoms()):
                pos = conf_copy.GetAtomPosition(atom_idx)
                new_pos = np.array([pos.x, pos.y, pos.z]) + translation
                conf_copy.SetAtomPosition(atom_idx, new_pos.tolist())

            pdb_block = Chem.MolToPDBBlock(mol_copy, confId=0)

            chain_id = chain_ids[frag_idx % len(chain_ids)]
            for line in pdb_block.split("\n"):
                if line.startswith(("HETATM", "ATOM")):
                    atom_num_str = f"{current_atom_num:5d}"
                    modified_line = line[:6] + atom_num_str + line[11:21] + chain_id + line[22:]
                    pdb_lines.append(modified_line)
                    current_atom_num += 1
                elif line.startswith("CONECT"):
                    continue

            frag_label = (frag.name or "UNK")[:3]
            pdb_lines.append(f"TER   {current_atom_num:5d}      {frag_label:3s} {chain_id}")
            current_atom_num += 1

            current_x_offset += bbox_size[0] + spacing

        pdb_lines.append("END")

        pdb_string = "\n".join(pdb_lines)

        if filepath is not None:
            filepath = Path(filepath)
            with open(filepath, "w") as f:
                f.write(pdb_string)

        return pdb_string

    def visualize(
        self,
        width: int = 400,
        height: int = 400,
        style: Literal["stick", "sphere", "line", "cartoon", "licorice"] = "stick",
    ) -> None:
        """Visualize all fragments in 3D using py3Dmol.

        Args:
            width (int): Viewer width in pixels.
            height (int): Viewer height in pixels.
            style (Literal["stick", "sphere", "line", "cartoon", "licorice"]): Visualization style.
        """
        import py3Dmol

        pdb = self.to_pdb()
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(pdb, "pdb")
        viewer.setStyle({style: {}})
        viewer.zoomTo()
        viewer.show()


# ============================================================================
# Parsing utilities
# ============================================================================


def parse_smiles_string(smiles: str) -> list[Fragment]:
    """Parse a dot-separated SMILES string into Fragment objects.

    Args:
        smiles (str): SMILES string (dots separate fragments).

    Returns:
        list[Fragment]: Parsed fragments.
    """
    fragments = []
    for raw_fragment in smiles.split("."):
        fragment = raw_fragment.strip()
        if fragment:
            fragments.append(Fragment(smiles=fragment))
    return fragments


def parse_fragments_from_smiles_file(filepath: str | Path) -> list[Fragment]:
    """Load fragments from a ``.smi`` file.

    Args:
        filepath (str | Path): Path to the SMILES file.

    Returns:
        list[Fragment]: Parsed fragments.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    fragments = []
    with open(filepath) as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                fragments.extend(parse_smiles_string(line))
    return fragments


def parse_mols_from_sdf_file(filepath: str | Path) -> list[Chem.Mol]:
    """Load Mol objects from an SDF file.

    Args:
        filepath (str | Path): Path to the SDF file.

    Returns:
        list[Chem.Mol]: Parsed molecules.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    from rdkit import Chem

    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    mols = []
    suppl = Chem.SDMolSupplier(str(filepath))
    for mol in suppl:
        if mol is None:
            continue  # type: ignore[unreachable]
        mols.append(mol)
    return mols


def parse_fragments_from_mols(mols: list[Chem.Mol]) -> list[Fragment]:
    """Convert RDKit Mol objects to Fragment objects.

    Args:
        mols (list[Chem.Mol]): RDKit molecules (may contain multi-fragment Mols).

    Returns:
        list[Fragment]: One Fragment per molecular fragment.
    """
    from rdkit import Chem

    fragments: list[Fragment] = []
    for mol in mols:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        fragments.extend(Fragment.from_mol(frag) for frag in frags)
    return fragments


def parse_fragments_from_sdf_file(filepath: str | Path) -> list[Fragment]:
    """Load fragments from an SDF file.

    Args:
        filepath (str | Path): Path to the SDF file.

    Returns:
        list[Fragment]: Parsed fragments.
    """
    mols = parse_mols_from_sdf_file(filepath)
    return parse_fragments_from_mols(mols)


def parse_fragments_from_string_or_path(string_or_path: str | Path) -> list[Fragment]:
    """Parse a SMILES string or file path into Fragment objects.

    Args:
        string_or_path (str | Path): SMILES string, or path to a ``.smi``/``.sdf`` file.

    Returns:
        list[Fragment]: Parsed fragments.

    Raises:
        ValueError: If the input is neither valid SMILES nor a supported file.
    """
    if str(string_or_path).lower().endswith(".smi"):
        return parse_fragments_from_smiles_file(string_or_path)
    if str(string_or_path).lower().endswith(".sdf"):
        return parse_fragments_from_sdf_file(string_or_path)
    if isinstance(string_or_path, str) and is_smiles_valid(string_or_path):
        return parse_smiles_string(string_or_path)
    raise ValueError(f"Invalid input: {string_or_path}")

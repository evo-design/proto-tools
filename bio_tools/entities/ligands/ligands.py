"""
ligands.py

Contains base class for representing small-molecule ligands.
"""
from __future__ import annotations

from pathlib import Path
import py3Dmol
from typing import Optional, Dict, Union, List, Literal
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from bio_programming.bio_tools.entities.ligands.utils import is_smiles_valid, get_name_from_smiles



class Fragment:
    """Class representing a single small molecule."""

    def __init__(self, molecule: str | Chem.Mol | "Fragment", name: Optional[str] = None):
        """
        Initialize a Fragment from either a SMILES string, an RDKit Mol object,
        or a Fragment object.
        
        Args:
            molecule: SMILES string, RDKit Mol object, or Fragment object
            name: Optional name for the small molecule
        """
        # Initialize the primary representation of the ligand
        self.mol = None
        if isinstance(molecule, str):
            
            if not is_smiles_valid(molecule):
                raise ValueError(f"Invalid SMILES string: {molecule}")
            self.mol = Chem.AddHs(Chem.MolFromSmiles(molecule))
        elif isinstance(molecule, Chem.Mol):
            # ValueError if the Mol contains more than one fragment
            fragments = Chem.rdmolops.GetMolFrags(molecule, asMols=True)
            if len(fragments) > 1:
                raise ValueError("Invalid Mol as Fragment input: Mol must contain only one fragment")
            self.mol = Chem.AddHs(molecule)

        elif isinstance(molecule, Fragment):
            self.mol = Chem.Mol(molecule.mol)
            self.name = molecule.name
            self.metrics = dict(molecule.metrics)
            return

        else:
            raise TypeError("molecule must be a SMILES string, RDKit Mol object, or Fragment object")

        self._validate_mol()

        self.name = name if name is not None else get_name_from_smiles(self.smiles)
        self.metrics: Dict[str, float] = {}

    def _validate_mol(self) -> None:
        if self.mol is None:
            raise ValueError("Invalid Fragment: Mol is None")
        if not self.mol.GetNumAtoms() > 0:
            raise ValueError("Invalid Fragment: Fragment must have at least one atom")

    @property
    def smiles(self) -> str:
        return Chem.MolToSmiles(Chem.RemoveHs(self.mol), canonical=True)


    def generate_conformers(
    self,
    num_conformers: int = 1,
    random_seed: Optional[int] = 42,
    prune_rms_threshold: float = 0.5
    ) -> None:
        params = AllChem.ETKDGv3()
        if random_seed is not None:
            params.randomSeed = random_seed
        params.pruneRmsThresh = prune_rms_threshold

        conf_ids = AllChem.EmbedMultipleConfs(self.mol, numConfs=num_conformers, params=params)
        if not conf_ids:
            raise RuntimeError("Failed to generate conformers for this molecule.")

        for conf_id in conf_ids:
            AllChem.UFFOptimizeMolecule(self.mol, confId=conf_id)

    @property
    def conformers(self) -> List[Chem.Conformer]:
        """Return all conformers present in the Mol object."""
        return [self.mol.GetConformer(i) for i in range(self.mol.GetNumConformers())]


    def __str__(self) -> str:
        return f"Fragment(smiles={self.smiles}, name={self.name})"


    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: 'Fragment') -> bool:
        return self.smiles == other.smiles

    def visualize(self, width=400, height=400, style='stick'):
        # Ensure there is at least one conformer
        if len(self.conformers) == 0:
            self.generate_conformers(num_conformers=1)

        # Convert to mol block for py3Dmol
        mol_block = Chem.MolToMolBlock(self.mol)
        
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(mol_block, 'mol')
        
        if style == 'stick':
            viewer.setStyle({'stick': {}})
        elif style == 'sphere':
            viewer.setStyle({'sphere': {'scale': 0.3}})
        else:
            viewer.setStyle({style: {}})
        
        viewer.zoomTo()
        viewer.show()

class Ligands:
    """Class representing Ligands (which can contain multiple small molecule 'fragments')."""
    def __init__(self, input_data: List[Fragment | Chem.Mol | str | Path | "Ligands"] | Fragment | Chem.Mol | str | Path | "Ligands"):
        """
        Initialize a Ligands object from a single Fragment, a list of Fragments,
        a single Chem.Mol object, a list of Chem.Mol objects, a single SMILES string,
        a list of SMILES strings, a single path to a SMILES file, or a list of paths
        to SMILES files.
        """
        self.fragments = []

        # Normalize input data to a list
        if not isinstance(input_data, list):
            input_data = [input_data]

        # Parse input data to fragments
        for item in input_data:
            if isinstance(item, Fragment):
                self.fragments.append(item)
            elif isinstance(item, Ligands):
                self.fragments.extend(item.fragments)
            elif isinstance(item, Chem.Mol):
                self.fragments.extend(parse_fragments_from_mols([item]))
            elif isinstance(item, str) or isinstance(item, Path):
                self.fragments.extend(parse_fragments_from_string_or_path(item))
            else:
                raise ValueError(f"Invalid list item: {item}")

    def add_fragment(self, fragment: Fragment):
        self.fragments.append(fragment)

    def generate_conformers(self, num_conformers: int = 1, random_seed: Optional[int] = 42, prune_rms_threshold: float = 0.5):
        """
        Generate 3D conformers for all fragments in the collection.

        Args:
            num_conformers: Number of conformers to generate.
            random_seed: Random seed for reproducibility.
            prune_rms_threshold: RMS threshold to prune similar conformers (Angstroms).

        Notes:
            Generated conformers are added to the underlying RDKit Mol objects,
            and are automatically visible via the `conformers` property.
        """
        for fragment in self.fragments:
            fragment.generate_conformers(num_conformers=num_conformers, random_seed=random_seed, prune_rms_threshold=prune_rms_threshold)

    def get_smiles_list(self):
        return [fragment.smiles for fragment in self.fragments]

    def get_names_list(self):
        return [fragment.name for fragment in self.fragments]

    @property
    def smiles(self) -> str:
        """"." characters are used to separate multiple ligands in the collection."""
        return ".".join(self.get_smiles_list())

    def __len__(self):
        return len(self.fragments)

    def __iter__(self):
        return iter(self.fragments)

    def __getitem__(self, index: int) -> Fragment:
        return self.fragments[index]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Ligands(fragments={self.fragments})"

    def __eq__(self, other: 'Ligands') -> bool:
        self_smiles_set = set(self.get_smiles_list())
        other_smiles_set = set(other.get_smiles_list())
        return self_smiles_set == other_smiles_set

    def to_smi(self, filepath: str | Path) -> None:
        """
        Write all fragments in this collection to a .smi file.
        """
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            for frag in self.fragments:
                f.write(f"{frag.smiles}\t{frag.name}\n")

    def to_sdf(self, filepath: str | Path) -> None:
        """
        Write all fragments to an SDF file.
        Each fragment's conformers are written as separate entries.
        """
        filepath = Path(filepath)
        writer = Chem.SDWriter(str(filepath))

        for frag in self.fragments:
            if frag.mol.GetNumConformers() == 0:
                frag.generate_conformers(num_conformers=1)
            for i in range(frag.mol.GetNumConformers()):
                writer.write(frag.mol, confId=i)
        writer.close()

    def to_pdb(self, filepath: Optional[Union[str, Path]] = None, spacing: float = 5.0) -> str:
        """
        Generate a PDB string containing all fragments with non-overlapping positions.

        Fragments are arranged linearly along the X-axis with spacing based on their
        bounding box sizes to prevent overlap. Each fragment is assigned a different
        chain ID (A, B, C, etc.).

        Args:
            filepath: Optional path to write the PDB file. If None, only returns the string.
            spacing: Additional spacing (in Angstroms) between fragment bounding boxes. Default: 5.0

        Returns:
            PDB format string containing all fragments
        """
        if len(self.fragments) == 0:
            raise ValueError("Cannot generate PDB: no fragments in Ligands object")

        # Ensure all fragments have conformers
        for frag in self.fragments:
            if frag.mol.GetNumConformers() == 0:
                frag.generate_conformers(num_conformers=1)

        pdb_lines = []
        current_atom_num = 1
        current_x_offset = 0.0
        chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for frag_idx, frag in enumerate(self.fragments):
            # Get the first conformer
            conf = frag.mol.GetConformer(0)

            # Calculate bounding box for this fragment
            positions = conf.GetPositions()
            min_coords = np.min(positions, axis=0)
            max_coords = np.max(positions, axis=0)
            bbox_size = max_coords - min_coords

            # Create a copy of the molecule to modify coordinates
            mol_copy = Chem.Mol(frag.mol)
            conf_copy = mol_copy.GetConformer(0)

            # Translate fragment to avoid overlap
            # Center the fragment at origin in Y and Z, offset in X
            fragment_center = (min_coords + max_coords) / 2
            translation = np.array([
                current_x_offset + bbox_size[0] / 2 - fragment_center[0],
                -fragment_center[1],
                -fragment_center[2]
            ])

            for atom_idx in range(mol_copy.GetNumAtoms()):
                pos = conf_copy.GetAtomPosition(atom_idx)
                new_pos = np.array([pos.x, pos.y, pos.z]) + translation
                conf_copy.SetAtomPosition(atom_idx, new_pos.tolist())

            # Convert to PDB block
            pdb_block = Chem.MolToPDBBlock(mol_copy, confId=0)

            # Parse and modify PDB block
            chain_id = chain_ids[frag_idx % len(chain_ids)]
            for line in pdb_block.split('\n'):
                if line.startswith('HETATM') or line.startswith('ATOM'):
                    # Modify atom number and chain ID
                    atom_num_str = f"{current_atom_num:5d}"
                    modified_line = line[:6] + atom_num_str + line[11:21] + chain_id + line[22:]
                    pdb_lines.append(modified_line)
                    current_atom_num += 1
                elif line.startswith('CONECT'):
                    # Skip CONECT records for now - could be added with renumbering if needed
                    continue

            # Add TER record after each fragment
            pdb_lines.append(f"TER   {current_atom_num:5d}      {frag.name[:3]:3s} {chain_id}")
            current_atom_num += 1

            # Update offset for next fragment
            current_x_offset += bbox_size[0] + spacing

        # Add END record
        pdb_lines.append("END")

        pdb_string = '\n'.join(pdb_lines)

        # Write to file if filepath provided
        if filepath is not None:
            filepath = Path(filepath)
            with open(filepath, 'w') as f:
                f.write(pdb_string)

        return pdb_string
    
    
    def visualize(self, width: int = 400, height: int = 400, style: Literal["stick", "sphere", "line", "cartoon", "licorice"] = "stick"):
        """
        Visualize all fragments in a Ligands object in 3D using py3Dmol.

        Args:
            ligands: Ligands object
            width, height: dimensions of the viewer
            style: visualization style ('stick', 'sphere', etc.)
        """
        pdb = self.to_pdb()
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(pdb, "pdb")
        viewer.setStyle({style: {}})
        viewer.zoomTo()
        viewer.show()


# -----------------------------
# SMILES utilities
# -----------------------------

def parse_smiles_string(smiles: str) -> List[Fragment]:
    """
    Parse a SMILES string (may contain '.' for multiple ligands)
    into a list of Fragment objects.
    """
    fragments = []
    for fragment in smiles.split("."):
        fragment = fragment.strip()
        if fragment:
            fragments.append(Fragment(fragment))
    return fragments

def parse_fragments_from_smiles_file(filepath: str | Path) -> List[Fragment]:
    """
    Load fragments from a .smi file. Each line can be a single or multi-fragment SMILES string.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    fragments = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                fragments.extend(parse_smiles_string(line))
    return fragments

# -----------------------------
# SDF utilities
# -----------------------------
def parse_mols_from_sdf_file(filepath: str | Path) -> List[Chem.Mol]:
    """Load mols from an SDF file."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    mols = []
    suppl = Chem.SDMolSupplier(str(filepath))
    for mol in suppl:
        if mol is None:
            continue
        mols.append(mol)
    return mols

def parse_fragments_from_mols(mols: List[Chem.Mol]) -> List[Fragment]:
    """Load fragments from a list of Chem.Mol objects."""
    fragments = []
    for mol in mols:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        for frag in frags:
            fragments.append(Fragment(frag))
    return fragments

def parse_fragments_from_sdf_file(filepath: str | Path) -> List[Fragment]:
    """Load ligands from an SDF file. Each Mol in the SDF may have multiple fragments."""
    mols = parse_mols_from_sdf_file(filepath)
    return parse_fragments_from_mols(mols)


# ===============================
# String utilities
# ===============================
def parse_fragments_from_string_or_path(string_or_path: str | Path) -> List[Fragment]:
    """Parse a string or path to a file into a list of Fragment objects."""
    if str(string_or_path).lower().endswith(".smi"):
        return parse_fragments_from_smiles_file(string_or_path)
    elif str(string_or_path).lower().endswith(".sdf"):
        return parse_fragments_from_sdf_file(string_or_path)
    elif isinstance(string_or_path, str) and is_smiles_valid(string_or_path):
        return parse_smiles_string(string_or_path)
    else:
        raise ValueError(f"Invalid input: {string_or_path}")
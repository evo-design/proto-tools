"""
proto_tools/entities/structures/structure.py

Contains base class for representing a protein structure.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import gemmi
import py3Dmol
from IPython.display import HTML, display
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from .utils import (
    SUPPORTED_EXTENSIONS,
    convert_cif_str_to_pdb_str,
    convert_pdb_str_to_cif_str,
    detect_structure_format,
    is_valid_structure,
    load_structure_file,
)

VISUALIZE_STYLE_OPTIONS = ["cartoon", "line", "stick", "sphere", "licorice"]

# Color palette for chain coloring (supports up to 20 chains with distinct colors)
CHAIN_COLORS = [
    'red', 'blue', 'green', 'yellow', 'orange', 'purple',
    'cyan', 'magenta', 'lime', 'pink', 'brown', 'gray',
    'darkred', 'darkblue', 'darkgreen', 'gold', 'coral', 'indigo',
    'turquoise', 'salmon'
]


def _create_bfactor_legend_html(b_factor_type: BFactorType, range_max: float) -> str:
    """Create an HTML legend for B-factor coloring.

    Args:
        b_factor_type (BFactorType): The type of B-factor data
        range_max (float): Maximum value of the B-factor range

    Returns:
        str: HTML string for the legend overlay
    """
    return f"""
    <div style="position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.9);
                padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                font-family: Arial, sans-serif; font-size: 12px; z-index: 1000; color: black;">
        <div style="font-weight: bold; margin-bottom: 8px;">{b_factor_type.value}</div>
        <div style="display: flex; align-items: center; gap: 0;">
            <div style="width: 30px; height: 100px;
                        background: linear-gradient(to bottom, blue, cyan, green, yellow, orange, red);
                        border: 1px solid #ccc; border-radius: 3px;"></div>
            <div style="display: flex; flex-direction: column; justify-content: space-between;
                        height: 100px; position: relative;">
                <div style="display: flex; align-items: center; height: 0;">
                    <div style="width: 8px; height: 1px; background-color: #333;"></div>
                    <span style="font-size: 10px; margin-left: 4px;">{range_max:.1f}</span>
                </div>
                <div style="display: flex; align-items: center; height: 0;">
                    <div style="width: 8px; height: 1px; background-color: #333;"></div>
                    <span style="font-size: 10px; margin-left: 4px;">{range_max/2:.1f}</span>
                </div>
                <div style="display: flex; align-items: center; height: 0;">
                    <div style="width: 8px; height: 1px; background-color: #333;"></div>
                    <span style="font-size: 10px; margin-left: 4px;">0</span>
                </div>
            </div>
        </div>
    </div>
    """


def _create_chain_legend_html(chain_color_map: Dict[str, str]) -> str:
    """Create an HTML legend for chain coloring.

    Args:
        chain_color_map (dict[str, str]): Dictionary mapping chain IDs to their assigned colors

    Returns:
        str: HTML string for the legend overlay
    """
    if not chain_color_map:
        return ""

    chain_items = []
    for chain_id, color in sorted(chain_color_map.items()):
        chain_items.append(
            f'<div style="display: flex; align-items: center; gap: 6px; margin: 4px 0;">'
            f'<div style="width: 16px; height: 16px; background-color: {color}; '
            f'border: 1px solid #ccc; border-radius: 2px;"></div>'
            f'<span>{chain_id}</span>'
            f'</div>'
        )

    items_html = ''.join(chain_items)

    return f"""
    <div style="position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.9);
                padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                font-family: Arial, sans-serif; font-size: 12px; z-index: 1000; max-height: 400px;
                overflow-y: auto; color: black;">
        <div style="font-weight: bold; margin-bottom: 8px;">Chains</div>
        {items_html}
    </div>
    """


class BFactorType(str, Enum):
    """What the B-factor column contains."""
    TEMPERATURE_FACTOR = "temperature_factor"
    PLDDT = "pLDDT"
    NORMALIZED_PLDDT = "normalized_pLDDT"
    CONFIDENCE = "confidence"
    UNKNOWN = "unknown"
    UNSPECIFIED = "unspecified"

class Structure:
    """Base class for representing macromolecular structures

    Standardized class for representing structures (proteins, nucleic acids, etc.).

    Attributes:
        structure_cif: Internal CIF format of the structure (converted from PDB if needed)
        b_factor_type: What the B-factor column contains (default is UNSPECIFIED)
    """

    def __init__(
        self,
        structure_filepath_or_content: Path | str,
        b_factor_type: BFactorType | str = BFactorType.UNSPECIFIED,
        metrics: Optional[Dict[str, float]] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Initializes a Structure object from a provided structure file or
        structure content string (can input PDB or CIF content strings directly).

        Args:
            structure_filepath_or_content (Path | str): Path to the structure file or string of
                structure file content (can input PDB or CIF content strings directly).
            b_factor_type (BFactorType | str): What the B-factor column contains (default is UNSPECIFIED).
                Can be a BFactorType enum or string value.
            metrics (dict[str, float] | None): Optional dictionary of metrics to associate with
                the structure (e.g., scores, confidence values). Default is None.
            source (str | None): Optional source identifier for the structure. If not provided
                and a filepath is given, will be set to the filepath. Default is None.
        """

        # Initialize the structure content and format strings
        structure_content = structure_filepath_or_content
        structure_format = None

        # If a file path is provided, load the structure content and format
        if str(structure_filepath_or_content).lower().endswith(SUPPORTED_EXTENSIONS):
            structure_content = load_structure_file(structure_filepath_or_content)
            if source is None:
                source = str(structure_filepath_or_content)

        # Validate the structure
        if not is_valid_structure(structure_filepath_or_content=structure_content):
            raise ValueError("Structure content is invalid")

        # Otherwise, detect the structure format from the content string
        structure_format = detect_structure_format(structure_content)

        # Save the structure content and format
        self.structure = structure_content
        self.structure_format = structure_format

        # Save other attributes
        self.b_factor_type = (
            BFactorType(b_factor_type) if isinstance(b_factor_type, str) else b_factor_type
        )
        self.source = source

        # Set up placeholder for lazy loading of gemmi structure object
        self._gemmi_struct = None

        # Set up metrics
        self.metrics = metrics if metrics is not None else {}

    # ===============================
    # Metrics
    # ===============================
    def __getattr__(self, name: str) -> Any:
        if name in self.metrics:
            return self.metrics[name]
        raise AttributeError(name)

    def add_metric(self, metric: str, value: float) -> None:
        """
        Add a metric to the structure.
        """
        self.metrics.update({metric: value})

    @property
    def gemmi_struct(self) -> gemmi.Structure:
        """
        Lazy loads the gemmi structure from the internal structure representation.

        Returns:
            gemmi.Structure: The parsed structure object
        """
        if self._gemmi_struct is None:
            if self.structure_format == "cif":
                doc = gemmi.cif.read_string(self.structure)

                # Find first valid structure block
                for block in doc:
                    struct = gemmi.make_structure_from_block(block)
                    if struct is not None and len(struct) > 0:
                        self._gemmi_struct = struct
                        break

                if self._gemmi_struct is None:
                    raise ValueError("No valid structure found in CIF content")
            else:
                self._gemmi_struct = gemmi.read_pdb_string(self.structure)

        return self._gemmi_struct

    @property
    def structure_pdb(self) -> str:
        """Converts the CIF representation of the structure to a PDB string."""
        if self.structure_format == "cif":
            return convert_cif_str_to_pdb_str(self.structure)
        else:
            return self.structure

    @property
    def structure_cif(self) -> str:
        """Converts the PDB representation of the structure to a CIF string."""
        if self.structure_format == "pdb":
            return convert_pdb_str_to_cif_str(self.structure)
        else:
            return self.structure

    # ===============================
    # File I/O
    # ===============================
    def write_cif(self, filepath: Path | str) -> None:
        """
        Write the structure to a CIF file.

        Args:
            filepath (Path | str): Path where to save the CIF file
        """
        Path(filepath).write_text(self.structure_cif)

    def write_pdb(self, filepath: Path | str) -> None:
        """
        Write the structure to a PDB file.

        WARNING: PDB format has limitations that may cause data loss.

        Args:
            filepath (Path | str): Path where to save the PDB file
        """
        Path(filepath).write_text(self.structure_pdb)

    # ===============================
    # Chain Related
    # ===============================
    def get_chain_sequence(
        self, chain_id: Optional[str] = None, remove_non_standard: bool = False
    ) -> str:
        """
        Extract the sequence of a specific chain from the structure.

        Args:
            chain_id (str | None): Chain ID to extract (e.g., 'A'). If None, returns the first chain.
            remove_non_standard (bool): If True, removes non-standard residues (X) and gaps (-)
                from the sequence. Default is False to preserve all residues.

        Returns:
            str: One-letter amino acid sequence of the chain

        Raises:
            ValueError: If specified chain_id is not found or no chains exist

        Examples:
            >>> protein.get_chain_sequence()  # First chain, all residues
            'MVLSE-GEWQX'
            >>> protein.get_chain_sequence('A')  # Chain A specifically
            'MVLSE-GEWQX'
            >>> protein.get_chain_sequence('A', remove_non_standard=True)  # Only standard residues
            'MVLSEGEWQ'
        """
        sequences = self.get_chain_sequences(remove_non_standard=remove_non_standard)

        if not sequences:
            raise ValueError("No protein chains found in structure")

        if chain_id is not None:
            if chain_id not in sequences:
                raise ValueError(
                    f"Chain '{chain_id}' not found. Available chains: {list(sequences.keys())}"
                )
            return sequences[chain_id]

        # Return first chain
        return next(iter(sequences.values()))

    def get_chain_sequences(self, remove_non_standard: bool = False) -> Dict[str, str]:
        """
        Extract the sequences of all chains in the structure.

        Args:
            remove_non_standard (bool): If True, removes non-standard residues (X) and gaps (-)
                from the sequences. Default is False to preserve all residues.

        Returns:
            Dict[str, str]: Dictionary mapping chain ID to sequence

        Examples:
            >>> protein.get_chain_sequences()
            {'A': 'MVLSE-GEWQX', 'B': 'ACDEFGHIK'}
            >>>
            >>> # Iterate over chains
            >>> for chain_id, sequence in protein.get_chain_sequences().items():
            ...     print(f"Chain {chain_id}: {len(sequence)} residues")
            Chain A: 11 residues
            Chain B: 9 residues
            >>>
            >>> # Remove non-standard residues
            >>> protein.get_chain_sequences(remove_non_standard=True)
            {'A': 'MVLSEGEWQ', 'B': 'ACDEFGHIK'}
        """
        sequences = {}
        for model in self.gemmi_struct:
            for chain in model:
                polymer = chain.whole()
                if polymer:
                    seq = polymer.make_one_letter_sequence()
                    if remove_non_standard:
                        # Remove non-standard residues (X) and gaps (-)
                        seq = seq.replace("X", "").replace("-", "")
                    sequences[chain.name] = seq
        return sequences

    def get_chain_ids(self) -> List[str]:
        """
        Extract the IDs of all chains in the structure.

        Returns:
            List[str]: List of chain IDs
        """
        return list(self.get_chain_sequences().keys())

    def get_chain_types(self) -> Dict[str, str]:
        """
        Classify each chain as either 'polymer' or 'ligand' based on entity type.

        Returns:
            Dict[str, str]: Dictionary mapping chain IDs to their type ('polymer' or 'ligand')

        Examples:
            >>> protein.get_chain_types()
            {'A': 'polymer', 'B': 'polymer', 'C': 'ligand'}
        """
        # Ensure entities are properly set up
        self.gemmi_struct.setup_entities()

        chain_types = {}
        for model in self.gemmi_struct:
            for chain in model:
                # Check if chain has polymer residues
                polymer = chain.get_polymer()
                ligands = chain.get_ligands()

                # Classify based on what the chain contains
                if polymer.length() > 0:
                    chain_types[chain.name] = 'polymer'
                elif ligands.length() > 0:
                    chain_types[chain.name] = 'ligand'
                # If neither (e.g., only waters), default to polymer style
                else:
                    chain_types[chain.name] = 'polymer'

        return chain_types

    @property
    def num_chains(self) -> int:
        """
        Get the number of residues in the structure.
        """
        return len(self.get_chain_sequences())

    # ===============================
    # Residue Related
    # ===============================

    def get_residue_position_map(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Gets a dictionary mapping chain IDs to lists of tuples of (residue_id, position)
        in the chain. Residue ID is the 1-letter code of the residue.
        """
        position_map = {}
        for model in self._gemmi_struct:
            for chain in model:
                chain_id = chain.name
                position_map[chain_id] = []
                chain_sequence = chain.whole()
                residue_id_list = gemmi.one_letter_code(
                    [residue.name for residue in chain_sequence]
                )
                position_list = [residue.seqid.num for residue in chain_sequence]
                position_map[chain_id] = list(zip(residue_id_list, position_list))
        return position_map

    def get_chain_positions(self, chain_id: str) -> List[int]:
        """
        Get the list of residue positions (1-indexed) for a specific chain.

        Args:
            chain_id (str): The chain identifier (e.g., "A", "B").

        Returns:
            list[int]: List of residue position numbers from the PDB file.

        Raises:
            ValueError: If the chain_id is not found in the structure.
        """
        residue_map = self.get_residue_position_map()
        if chain_id not in residue_map:
            raise ValueError(
                f"Chain '{chain_id}' not found in structure. "
                f"Available chains: {list(residue_map.keys())}"
            )
        return [pos for _, pos in residue_map[chain_id]]

    @property
    def num_residues(self) -> int:
        """
        Get the number of residues in the structure.

        TODO: Determine if we should differentiate different types of chains
        """
        return sum(len(chain) for chain in self.get_chain_sequences().values())

    # ===============================
    # Visualization
    # ===============================
    def visualize(
        self,
        style: Literal["cartoon", "line", "stick", "sphere", "licorice"] = "cartoon",
        color_by: Optional[Literal["bfactor", "chain"]] = None,
        show_legend: bool = True,
        width: int = 400,
        height: int = 400,
        ligand_style: Literal["stick", "sphere", "line", "licorice"] = "stick"
    ):
        """
        Visualize the structure using py3Dmol with optional coloring modes and legends.

        Supports two coloring modes:
        - "bfactor": Colors by B-factor values with a gradient (red=low to blue=high)
        - "chain": Colors each chain with a distinct color

        Automatically determines the appropriate B-factor range from `b_factor_type`:
        - "normalized_pLDDT": 0-1 scale
        - "pLDDT": 0-100 scale
        - Others: 0-100 scale (default)

        Args:
            style (Literal['cartoon', 'line', 'stick', 'sphere', 'licorice']): Visualization style for polymer chains (default: "cartoon"). Must be one of:
                "cartoon", "line", "stick", "sphere", "licorice"
            color_by (Literal['bfactor', 'chain'] | None): Coloring mode (default: "chain" if b_factor_type is UNSPECIFIED, otherwise "bfactor")
                - "bfactor": Color by B-factor values with gradient
                - "chain": Color each chain with a distinct color
            show_legend (bool): Whether to display a legend/colorbar (default: True)
                For "bfactor": Shows a horizontal colorbar with the B-factor scale
                For "chain": Shows a legend listing chain IDs and their colors
            width (int): Width of the viewer in pixels (default: 400)
            height (int): Height of the viewer in pixels (default: 400)
            ligand_style (Literal['stick', 'sphere', 'line', 'licorice']): Visualization style for ligand (non-polymer) chains (default: "stick").
                Must be one of: "stick", "sphere", "line", "licorice"
                Note: Ligands don't work with "cartoon" style as they lack backbone structure.

        Returns:
            py3Dmol view object

        Examples:
            >>> # Default B-factor coloring with legend (if b_factor_type is specified)
            >>> protein.visualize()
            >>>
            >>> # Chain coloring with legend (default if b_factor_type is UNSPECIFIED)
            >>> protein.visualize()
            >>>
            >>> # Explicitly set chain coloring
            >>> protein.visualize(color_by="chain")
            >>>
            >>> # Chain coloring without legend
            >>> protein.visualize(color_by="chain", show_legend=False)
            >>>
            >>> # B-factor with custom style
            >>> protein.visualize(style="sphere", color_by="bfactor")
            >>>
            >>> # Protein as cartoon, ligands as spheres
            >>> protein.visualize(style="cartoon", ligand_style="sphere")
        """
        # Default color_by based on b_factor_type if not explicitly provided
        if color_by is None:
            color_by = "chain" if self.b_factor_type == BFactorType.UNSPECIFIED else "bfactor"

        # Validate color_by parameter
        valid_color_modes = ["bfactor", "chain"]
        if color_by not in valid_color_modes:
            raise ValueError(
                f"Invalid color_by value: '{color_by}'. "
                f"Must be one of: {', '.join(valid_color_modes)}"
            )

        # Create a new py3Dmol viewer
        viewer = py3Dmol.view(width=width, height=height)

        # Add the structure to the viewer
        if self.structure_format == "cif":
            viewer.addModel(self.structure, "cif")
        elif self.structure_format == "pdb":
            viewer.addModel(self.structure, "pdb")

        legend_html = ""

        if color_by == "bfactor":
            # B-factor gradient coloring
            range_max = 1.0 if self.b_factor_type == BFactorType.NORMALIZED_PLDDT else 100.0
            chain_types = self.get_chain_types()

            # Apply styles based on chain type
            for chain_id, chain_type in chain_types.items():
                chain_style = ligand_style if chain_type == 'ligand' else style
                viewer.setStyle(
                    {'chain': chain_id},
                    {
                        chain_style: {
                            "colorscheme": {
                                "prop": "b",
                                "gradient": "roygb",
                                "min": 0.0,
                                "max": range_max,
                            }
                        }
                    }
                )

            # Prepare B-factor legend HTML
            if show_legend:
                legend_html = _create_bfactor_legend_html(self.b_factor_type, range_max)

        elif color_by == "chain":
            # Chain-based coloring with manual color assignment
            chain_ids = self.get_chain_ids()
            chain_types = self.get_chain_types()
            chain_color_map = {}

            for idx, chain_id in enumerate(chain_ids):
                color = CHAIN_COLORS[idx % len(CHAIN_COLORS)]
                chain_color_map[chain_id] = color

                # Use appropriate style based on chain type
                chain_style = ligand_style if chain_types.get(chain_id) == 'ligand' else style
                viewer.setStyle({'chain': chain_id}, {chain_style: {'color': color}})

            # Prepare chain legend HTML
            if show_legend:
                legend_html = _create_chain_legend_html(chain_color_map)

        viewer.zoomTo()

        # If legend is needed, wrap the viewer with HTML overlay
        if show_legend and legend_html:
            # Get the viewer's HTML
            viewer_html = viewer._make_html()

            # Wrap it with a container that includes the legend
            combined_html = f"""
            <div style="position: relative; width: {width}px; height: {height}px; display: inline-block;">
                {viewer_html}
                {legend_html}
            </div>
            """

            display(HTML(combined_html))
        else:
            # Show the viewer normally
            viewer.show()

    # ===============================
    # Pydantic Serialization
    # ===============================
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        Tells Pydantic how to validate and serialize Structure objects.
        """
        return core_schema.no_info_after_validator_function(
            cls._validate_from_dict,
            core_schema.union_schema(
                [
                    # Allow creating from a dict (for deserialization)
                    core_schema.dict_schema(),
                    # Allow passing an existing Structure instance
                    core_schema.is_instance_schema(cls),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize_to_dict,
                info_arg=False,
                return_schema=core_schema.dict_schema(),
            ),
        )

    @classmethod
    def _validate_from_dict(
        cls, value: Dict[str, Any] | "Structure"
    ) -> Structure:
        """
        Create a Structure from a dictionary (used during deserialization).
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, dict):
            raise ValueError(f"Expected dict or Structure, got {type(value)}")

        # Reconstruct from serialized format
        structure = value.get("structure")
        structure_format = value.get("structure_format")
        b_factor_type_str = value.get("b_factor_type", "unspecified")

        if structure is None:
            raise ValueError("Missing 'structure' in serialized data")
        if b_factor_type_str is None:
            raise ValueError("Missing 'b_factor_type' in serialized data")
        source = value.get("source")

        if structure_format is None:
            structure_format = detect_structure_format(structure)

        # Create new instance directly with CIF content
        instance = cls.__new__(cls)
        instance.structure = structure
        instance.structure_format = structure_format
        instance.b_factor_type = BFactorType(b_factor_type_str)
        instance._gemmi_struct = None
        instance.source = source
        instance.metrics = value.get("metrics", {})
        return instance

    def _serialize_to_dict(self) -> Dict[str, Any]:
        """
        Serialize Structure to a dictionary (for Pydantic models).
        """
        return {
            "structure": self.structure,
            "structure_format": self.structure_format,
            "b_factor_type": self.b_factor_type.value,
            "metrics": self.metrics,
            "source": self.source,
        }

    def __str__(self) -> str:
        return f"Structure(structure_format={self.structure_format}, b_factor_type={self.b_factor_type}, source={self.source})"

    def __repr__(self) -> str:
        return self.__str__()

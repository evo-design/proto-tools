"""Macromolecular structure representation as a Pydantic BaseModel."""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import gemmi
import py3Dmol
from IPython.display import HTML, display
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from proto_tools.entities.structures.utils import (
    convert_cif_str_to_pdb_str,
    convert_pdb_str_to_cif_str,
    detect_structure_format,
    is_valid_structure,
    load_structure_file,
    looks_like_structure_path,
)
from proto_tools.utils.tool_io import Metrics, MetricValue

VISUALIZE_STYLE_OPTIONS = ["cartoon", "line", "stick", "sphere", "licorice"]


# Color palette for chain coloring (supports up to 20 chains with distinct colors)
CHAIN_COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "lime",
    "pink",
    "brown",
    "gray",
    "darkred",
    "darkblue",
    "darkgreen",
    "gold",
    "coral",
    "indigo",
    "turquoise",
    "salmon",
]


def _create_bfactor_legend_html(b_factor_type: BFactorType, range_max: float) -> str:
    """Create an HTML legend for B-factor coloring.

    Args:
        b_factor_type (BFactorType): The type of B-factor data.
        range_max (float): Maximum value of the B-factor range.

    Returns:
        str: HTML string for the legend overlay.
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
                    <span style="font-size: 10px; margin-left: 4px;">{range_max / 2:.1f}</span>
                </div>
                <div style="display: flex; align-items: center; height: 0;">
                    <div style="width: 8px; height: 1px; background-color: #333;"></div>
                    <span style="font-size: 10px; margin-left: 4px;">0</span>
                </div>
            </div>
        </div>
    </div>
    """


def _create_chain_legend_html(chain_color_map: dict[str, str]) -> str:
    """Create an HTML legend for chain coloring.

    Args:
        chain_color_map (dict[str, str]): Dictionary mapping chain IDs to their assigned colors.

    Returns:
        str: HTML string for the legend overlay.
    """
    if not chain_color_map:
        return ""

    chain_items = []
    for chain_id, color in sorted(chain_color_map.items()):
        chain_items.append(
            f'<div style="display: flex; align-items: center; gap: 6px; margin: 4px 0;">'
            f'<div style="width: 16px; height: 16px; background-color: {color}; '
            f'border: 1px solid #ccc; border-radius: 2px;"></div>'
            f"<span>{chain_id}</span>"
            f"</div>"
        )

    items_html = "".join(chain_items)

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


class Structure(BaseModel):
    """Standardized representation of a macromolecular structure (protein, nucleic acid, etc.).

    A Pydantic model storing structure content as a PDB or CIF format string.
    The ``structure`` field accepts either the raw content string or a path to a
    ``.pdb``/``.cif``/``.mmcif`` file — paths are loaded transparently at construction
    time. ``Structure.from_file()`` is also available as an explicit factory.
    Heavy objects (gemmi parsed structure) are lazy-loaded via ``PrivateAttr``.

    Attributes:
        structure (str): Raw structure content in PDB or CIF format.
        structure_format (Literal["pdb", "cif"] | None): Format of the content string (auto-detected if omitted).
        b_factor_type (BFactorType): What the B-factor column represents.
        source (str | None): Optional source identifier (filepath or tool name).
        metrics (Metrics): Associated metrics (e.g., pLDDT, pTM scores,
            per-chain lists, pairwise matrices). None values are stripped at construction.
    """

    model_config = ConfigDict(extra="forbid")

    structure: str = Field(description="Structure content (PDB or CIF format string)")
    structure_format: Literal["pdb", "cif"] | None = Field(default=None, description="Format of the structure content")
    b_factor_type: BFactorType = Field(
        default=BFactorType.UNSPECIFIED, description="What the B-factor column represents"
    )
    source: str | None = Field(default=None, description="Source identifier for the structure")
    metrics: Metrics = Field(default_factory=Metrics, description="Associated metrics")

    _gemmi_struct: Any = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _handle_construction(cls, data: Any) -> Any:
        """Load ``structure`` from disk if it looks like a path, then auto-detect format.

        This lets callers write ``Structure(structure="foo.pdb")`` in addition to
        ``Structure.from_file("foo.pdb")`` — the old plain-class shortcut survives.
        """
        if not isinstance(data, dict):
            return data

        # If structure looks like a path to an existing file, load it transparently.
        structure_value = data.get("structure")
        if structure_value is not None and looks_like_structure_path(structure_value):
            path = Path(structure_value)
            data["structure"] = load_structure_file(path)
            if not data.get("source"):
                data["source"] = str(path)

        # Auto-detect structure_format when not provided or explicitly None
        if "structure" in data and not data.get("structure_format"):
            data["structure_format"] = detect_structure_format(data["structure"])

        return data

    @model_validator(mode="after")
    def _validate_structure(self) -> Structure:
        """Validate that the structure content is parseable and format is resolved."""
        if self.structure_format is None:
            msg = "structure_format could not be determined"
            raise ValueError(msg)
        if not is_valid_structure(structure_filepath_or_content=self.structure):
            msg = "Structure content is invalid"
            raise ValueError(msg)
        return self

    # ============================================================================
    # Factory
    # ============================================================================

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        b_factor_type: BFactorType = BFactorType.UNSPECIFIED,
        metrics: Metrics | dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Structure:
        """Load a Structure from a PDB or CIF file.

        Args:
            path (str | Path): Path to a ``.pdb``, ``.cif``, or ``.mmcif`` file.
            b_factor_type (BFactorType): What the B-factor column represents.
            metrics (Metrics | dict[str, Any] | None): Optional metrics to attach.
            source (str | None): Source identifier. Defaults to the filepath.

        Returns:
            Structure: The loaded structure.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        content = load_structure_file(path)
        fmt = detect_structure_format(content)
        if metrics is None:
            metrics = Metrics()
        elif isinstance(metrics, dict):
            metrics = Metrics(**metrics)
        return cls(
            structure=content,
            structure_format=fmt,  # type: ignore[arg-type]
            b_factor_type=b_factor_type,
            source=source or str(path),
            metrics=metrics,
        )

    # ============================================================================
    # Metrics
    # ============================================================================

    def add_metric(self, metric: str, value: MetricValue) -> None:
        """Add a metric to the structure.

        Args:
            metric (str): Name of the metric.
            value (MetricValue): Value of the metric.
        """
        self.metrics[metric] = value

    def attach_metrics(self, metrics: Metrics) -> None:
        """Merge tool-emitted metrics into this structure's metrics container.

        Tools return metrics parallel to their input structures without
        mutating the inputs (so caching and reproducibility invariants hold).
        Use this helper to carry tool-emitted metrics back onto the original
        ``Structure`` in place; existing keys are overwritten.

        Args:
            metrics (Metrics): The metrics container to merge into ``self.metrics``.
        """
        self.metrics.update(metrics)

    # ============================================================================
    # Gemmi / Format Conversion
    # ============================================================================

    @property
    def gemmi_struct(self) -> gemmi.Structure:
        """Lazy-load the gemmi structure from the internal content string.

        Returns:
            gemmi.Structure: The parsed structure object.
        """
        if self._gemmi_struct is None:
            if self.structure_format == "cif":
                doc = gemmi.cif.read_string(self.structure)
                for block in doc:
                    struct = gemmi.make_structure_from_block(block)
                    if struct is not None and len(struct) > 0:  # type: ignore[redundant-expr]
                        self._gemmi_struct = struct
                        break
                if self._gemmi_struct is None:
                    raise ValueError("No valid structure found in CIF content")
            else:
                self._gemmi_struct = gemmi.read_pdb_string(self.structure)
        return self._gemmi_struct  # type: ignore[no-any-return]

    @property
    def per_residue_plddt(self) -> list[float] | None:
        """Per-residue pLDDT values extracted from the B-factor column, normalized to 0-1.

        Returns ``None`` when ``b_factor_type`` does not represent pLDDT (e.g.,
        experimental structures with temperature factors). Values are normalized
        to the 0-1 range regardless of the original tool scale (ESMFold uses 0-1
        natively; AlphaFold3 uses 0-100).

        Returns:
            list[float] | None: Per-residue pLDDT values in 0-1 range, or None if
                B-factors do not represent pLDDT.
        """
        if self.b_factor_type not in (BFactorType.PLDDT, BFactorType.NORMALIZED_PLDDT):
            return None
        values: list[float] = []
        for model in self.gemmi_struct:
            for chain in model:
                for residue in chain:
                    b_factors = [atom.b_iso for atom in residue]
                    if b_factors:
                        values.append(sum(b_factors) / len(b_factors))
        if self.b_factor_type == BFactorType.PLDDT:
            values = [v / 100.0 for v in values]
        return values or None

    @property
    def structure_pdb(self) -> str:
        """Get the structure content as a PDB string, converting from CIF if needed."""
        if self.structure_format == "cif":
            return convert_cif_str_to_pdb_str(self.structure)
        return self.structure

    def to_pdb_with_chain_mapping(self) -> tuple[str, dict[str, str]]:
        """Convert to PDB content, shortening multi-character chain IDs for PDB compatibility.

        PDB format restricts chain IDs to a single character, while mmCIF permits
        arbitrary-length chain labels. When a CIF structure contains multi-character
        chain IDs (e.g., ``"Heavy"``, ``"Light"``, ``"AA"``), a naive CIF→PDB
        conversion either fails outright (≥3 chars) or produces non-standard PDB
        output that downstream tools mis-parse (2 chars — the chain name overflows
        into the residue name column). This method pre-shortens chain names via
        gemmi's ``shorten_chain_names`` so that the resulting PDB is spec-compliant,
        and returns the mapping from original chain IDs to their single-character
        equivalents so callers can translate user-supplied chain IDs before
        dispatching to PDB-only tools and reconstruct original labels in outputs.

        For structures already in PDB format, or CIF structures whose chain IDs
        are already single-character, the mapping is the identity and the PDB
        content is emitted without shortening.

        Returns:
            tuple[str, dict[str, str]]: A ``(pdb_content, chain_id_map)`` pair where
                ``chain_id_map`` maps each original chain ID to its PDB-compatible
                single-character equivalent.

        Raises:
            ValueError: If the structure content cannot be parsed or if PDB
                conversion cannot represent the structure even after chain shortening
                (e.g., more chains than available single-character slots).
        """
        # PDB input: no conversion, identity mapping.
        if self.structure_format == "pdb":
            identity = {cid: cid for cid in self.get_chain_ids()}
            return self.structure, identity

        # CIF input: parse fresh (do not mutate the cached _gemmi_struct),
        # shorten chain names in place on the copy, then emit PDB.
        try:
            doc = gemmi.cif.read_string(self.structure)
            gemmi_struct: gemmi.Structure | None = None
            for block in doc:
                try:
                    candidate = gemmi.make_structure_from_block(block)
                    if candidate is not None and len(candidate) > 0:  # type: ignore[redundant-expr]
                        gemmi_struct = candidate
                        break
                except Exception:  # noqa: S112
                    continue
            if gemmi_struct is None:
                raise ValueError("No valid structure found in CIF content")

            before = [chain.name for model in gemmi_struct for chain in model]
            gemmi_struct.shorten_chain_names()
            after = [chain.name for model in gemmi_struct for chain in model]
            chain_id_map = dict(zip(before, after, strict=True))

            return gemmi_struct.make_pdb_string(), chain_id_map
        except Exception as e:
            raise ValueError(f"Failed to convert CIF to PDB with chain mapping: {e}") from e

    @property
    def structure_cif(self) -> str:
        """Get the structure content as a CIF string, converting from PDB if needed."""
        if self.structure_format == "pdb":
            return convert_pdb_str_to_cif_str(self.structure)
        return self.structure

    # ============================================================================
    # File I/O
    # ============================================================================

    def write_cif(self, filepath: Path | str) -> None:
        """Write the structure to a CIF file.

        Args:
            filepath (Path | str): Path where to save the CIF file.
        """
        Path(filepath).write_text(self.structure_cif)

    def write_pdb(self, filepath: Path | str) -> None:
        """Write the structure to a PDB file.

        WARNING: PDB format has limitations that may cause data loss.

        Args:
            filepath (Path | str): Path where to save the PDB file.
        """
        Path(filepath).write_text(self.structure_pdb)

    # ============================================================================
    # Chain Related
    # ============================================================================

    def get_chain_sequence(self, chain_id: str | None = None, remove_non_standard: bool = False) -> str:
        """Extract the sequence of a specific chain from the structure.

        Args:
            chain_id (str | None): Chain ID to extract (e.g., 'A'). If None, returns the first chain.
            remove_non_standard (bool): If True, removes non-standard residues (X) and gaps (-)
                from the sequence. Default is False to preserve all residues.

        Returns:
            str: One-letter amino acid sequence of the chain.

        Raises:
            ValueError: If specified chain_id is not found or no chains exist.

        Examples:
            >>> protein.get_chain_sequence()  # First chain, all residues
            'MVLSE-GEWQX'
            >>> protein.get_chain_sequence("A")  # Chain A specifically
            'MVLSE-GEWQX'
            >>> protein.get_chain_sequence("A", remove_non_standard=True)  # Only standard residues
            'MVLSEGEWQ'
        """
        sequences = self.get_chain_sequences(remove_non_standard=remove_non_standard)

        if not sequences:
            raise ValueError("No protein chains found in structure")

        if chain_id is not None:
            if chain_id not in sequences:
                raise ValueError(f"Chain '{chain_id}' not found. Available chains: {list(sequences.keys())}")
            return sequences[chain_id]

        return next(iter(sequences.values()))

    def get_chain_sequences(self, remove_non_standard: bool = False) -> dict[str, str]:
        """Extract the sequences of all chains in the structure.

        Args:
            remove_non_standard (bool): If True, removes non-standard residues (X) and gaps (-)
                from the sequences. Default is False to preserve all residues.

        Returns:
            dict[str, str]: Dictionary mapping chain ID to sequence.

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
                        seq = seq.replace("X", "").replace("-", "")
                    sequences[chain.name] = seq
        return sequences

    def get_chain_ids(self) -> list[str]:
        """Extract the IDs of all chains in the structure.

        Returns:
            list[str]: List of chain IDs.
        """
        return list(self.get_chain_sequences().keys())

    def get_chain_types(self) -> dict[str, str]:
        """Classify each chain as either 'polymer' or 'ligand' based on entity type.

        Returns:
            dict[str, str]: Dictionary mapping chain IDs to their type ('polymer' or 'ligand').

        Examples:
            >>> protein.get_chain_types()
            {'A': 'polymer', 'B': 'polymer', 'C': 'ligand'}
        """
        self.gemmi_struct.setup_entities()

        chain_types = {}
        for model in self.gemmi_struct:
            for chain in model:
                polymer = chain.get_polymer()
                ligands = chain.get_ligands()

                if polymer.length() > 0:
                    chain_types[chain.name] = "polymer"
                elif ligands.length() > 0:
                    chain_types[chain.name] = "ligand"
                else:
                    chain_types[chain.name] = "polymer"

        return chain_types

    @property
    def num_chains(self) -> int:
        """Number of chains in the structure."""
        return len(self.get_chain_sequences())

    # ============================================================================
    # Residue Related
    # ============================================================================

    def get_residue_position_map(self) -> dict[str, list[tuple[str, int]]]:
        """Get a dictionary mapping chain IDs to lists of (residue_id, position) tuples.

        Returns:
            dict[str, list[tuple[str, int]]]: Chain ID to (one-letter code, position) mapping.
        """
        position_map: dict[str, list[tuple[str, int]]] = {}
        for model in self.gemmi_struct:
            for chain in model:
                chain_id = chain.name
                position_map[chain_id] = []
                chain_sequence = chain.whole()
                residue_id_list = gemmi.one_letter_code([residue.name for residue in chain_sequence])
                position_list: list[int] = [residue.seqid.num for residue in chain_sequence]  # type: ignore[misc]
                position_map[chain_id] = list(zip(residue_id_list, position_list, strict=False))
        return position_map

    def get_chain_positions(self, chain_id: str) -> list[int]:
        """Get the list of residue positions (1-indexed) for a specific chain.

        Args:
            chain_id (str): The chain identifier (e.g., "A", "B").

        Returns:
            list[int]: List of residue position numbers from the PDB file.

        Raises:
            ValueError: If the chain_id is not found in the structure.
        """
        residue_map = self.get_residue_position_map()
        if chain_id not in residue_map:
            raise ValueError(f"Chain '{chain_id}' not found in structure. Available chains: {list(residue_map.keys())}")
        return [pos for _, pos in residue_map[chain_id]]

    @property
    def num_residues(self) -> int:
        """Total number of residues across all chains."""
        return sum(len(chain) for chain in self.get_chain_sequences().values())

    # ============================================================================
    # Approximate Comparison
    # ============================================================================

    def approx_equal(self, other: Structure, rtol: float = 1e-4, atol: float = 1e-6) -> None:
        """Assert that two structures are approximately equal.

        Compares topology (chain IDs, sequences) exactly and atom coordinates
        approximately. Used by seed reproducibility tests where CUDA non-determinism
        causes bit-level float differences in coordinates.

        Args:
            other (Structure): The other structure to compare against.
            rtol (float): Relative tolerance for coordinate comparison.
            atol (float): Absolute tolerance for coordinate comparison.

        Raises:
            AssertionError: If the structures differ, with details about the first mismatch.
        """
        # Compare topology exactly
        self_seqs = self.get_chain_sequences()
        other_seqs = other.get_chain_sequences()
        if self_seqs != other_seqs:
            raise AssertionError(f"Chain sequences differ: {self_seqs} != {other_seqs}")

        # Extract and compare atom coordinates approximately
        self_atoms = _extract_atom_positions(self)
        other_atoms = _extract_atom_positions(other)

        if len(self_atoms) != len(other_atoms):
            raise AssertionError(f"Atom count differs: {len(self_atoms)} != {len(other_atoms)}")

        for i, ((s_name, s_pos), (o_name, o_pos)) in enumerate(zip(self_atoms, other_atoms, strict=True)):
            if s_name != o_name:
                raise AssertionError(f"Atom {i} name differs: {s_name!r} != {o_name!r}")
            for axis, (s_val, o_val) in zip("xyz", zip(s_pos, o_pos, strict=True), strict=True):
                if not math.isclose(s_val, o_val, rel_tol=rtol, abs_tol=atol):
                    raise AssertionError(
                        f"Atom {i} ({s_name}) {axis} coordinate: {s_val} != {o_val} (rtol={rtol}, atol={atol})"
                    )

        # Compare metrics approximately
        if self.metrics.keys() != other.metrics.keys():
            raise AssertionError(f"Metric keys differ: {set(self.metrics.keys()) ^ set(other.metrics.keys())}")
        for key in self.metrics:
            _approx_equal_metric(key, self.metrics[key], other.metrics[key], rtol, atol)

    # ============================================================================
    # Visualization
    # ============================================================================

    def visualize(
        self,
        style: Literal["cartoon", "line", "stick", "sphere", "licorice"] = "cartoon",
        color_by: Literal["bfactor", "chain"] | None = None,
        show_legend: bool = True,
        width: int = 400,
        height: int = 400,
        ligand_style: Literal["stick", "sphere", "line", "licorice"] = "stick",
    ) -> None:
        """Visualize the structure using py3Dmol with optional coloring modes and legends.

        Supports two coloring modes:
        - "bfactor": Colors by B-factor values with a gradient (red=low to blue=high)
        - "chain": Colors each chain with a distinct color

        Automatically determines the appropriate B-factor range from ``b_factor_type``:
        - "normalized_pLDDT": 0-1 scale
        - "pLDDT": 0-100 scale
        - Others: 0-100 scale (default)

        Args:
            style (Literal["cartoon", "line", "stick", "sphere", "licorice"]): Visualization style
                for polymer chains (default: "cartoon").
            color_by (Literal["bfactor", "chain"] | None): Coloring mode. Defaults to "chain" if
                b_factor_type is UNSPECIFIED, otherwise "bfactor".
            show_legend (bool): Whether to display a legend/colorbar (default: True).
            width (int): Width of the viewer in pixels (default: 400).
            height (int): Height of the viewer in pixels (default: 400).
            ligand_style (Literal["stick", "sphere", "line", "licorice"]): Visualization style for
                ligand (non-polymer) chains (default: "stick").
        """
        if color_by is None:
            color_by = "chain" if self.b_factor_type == BFactorType.UNSPECIFIED else "bfactor"

        valid_color_modes = ["bfactor", "chain"]
        if color_by not in valid_color_modes:
            raise ValueError(f"Invalid color_by value: '{color_by}'. Must be one of: {', '.join(valid_color_modes)}")

        viewer = py3Dmol.view(width=width, height=height)

        if self.structure_format == "cif":
            viewer.addModel(self.structure, "cif")
        elif self.structure_format == "pdb":
            viewer.addModel(self.structure, "pdb")

        legend_html = ""

        if color_by == "bfactor":
            range_max = 1.0 if self.b_factor_type == BFactorType.NORMALIZED_PLDDT else 100.0
            chain_types = self.get_chain_types()

            for chain_id, chain_type in chain_types.items():
                chain_style = ligand_style if chain_type == "ligand" else style
                viewer.setStyle(
                    {"chain": chain_id},
                    {
                        chain_style: {
                            "colorscheme": {
                                "prop": "b",
                                "gradient": "roygb",
                                "min": 0.0,
                                "max": range_max,
                            }
                        }
                    },
                )

            if show_legend:
                legend_html = _create_bfactor_legend_html(self.b_factor_type, range_max)

        elif color_by == "chain":
            chain_ids = self.get_chain_ids()
            chain_types = self.get_chain_types()
            chain_color_map = {}

            for idx, chain_id in enumerate(chain_ids):
                color = CHAIN_COLORS[idx % len(CHAIN_COLORS)]
                chain_color_map[chain_id] = color

                chain_style = ligand_style if chain_types.get(chain_id) == "ligand" else style
                viewer.setStyle({"chain": chain_id}, {chain_style: {"color": color}})

            if show_legend:
                legend_html = _create_chain_legend_html(chain_color_map)

        viewer.zoomTo()

        if show_legend and legend_html:
            viewer_html = viewer._make_html()
            combined_html = f"""
            <div style="position: relative; width: {width}px; height: {height}px; display: inline-block;">
                {viewer_html}
                {legend_html}
            </div>
            """
            display(HTML(combined_html))  # type: ignore[no-untyped-call]
        else:
            viewer.show()

    # ============================================================================
    # Display
    # ============================================================================

    def __str__(self) -> str:
        return f"Structure(structure_format={self.structure_format}, b_factor_type={self.b_factor_type}, source={self.source})"

    def __repr__(self) -> str:
        return self.__str__()


def _extract_atom_positions(structure: Structure) -> list[tuple[str, tuple[float, float, float]]]:
    """Extract atom names and (x, y, z) positions from a Structure via gemmi.

    Args:
        structure (Structure): The structure to extract atoms from.

    Returns:
        list[tuple[str, tuple[float, float, float]]]: List of (atom_name, (x, y, z)) tuples.
    """
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for model in structure.gemmi_struct:
        for chain in model:
            for residue in chain:
                atoms.extend((atom.name, (atom.pos.x, atom.pos.y, atom.pos.z)) for atom in residue)
    return atoms


def _approx_equal_metric(key: str, a: Any, b: Any, rtol: float, atol: float) -> None:
    """Compare two metric values with float tolerance, recursing into lists."""
    if isinstance(a, float) and isinstance(b, float):
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"Metric {key!r}: {a} != {b} (rtol={rtol}, atol={atol})")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError(f"Metric {key!r} length: {len(a)} != {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b, strict=True)):
            _approx_equal_metric(f"{key}[{i}]", ai, bi, rtol, atol)
    elif a != b:
        raise AssertionError(f"Metric {key!r}: {a!r} != {b!r}")

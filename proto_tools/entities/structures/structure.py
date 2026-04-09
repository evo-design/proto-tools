"""Macromolecular structure representation as a Pydantic BaseModel."""

from __future__ import annotations

from collections.abc import Iterator
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

VISUALIZE_STYLE_OPTIONS = ["cartoon", "line", "stick", "sphere", "licorice"]

StructureMetricValue = float | int | list[float] | list[list[float]] | bool


class StructureMetrics(BaseModel):
    """Metrics container with attribute and dict-style access. None values are excluded at construction.

    Wraps a dict of metric values, automatically stripping any ``None`` entries so that
    only metrics the model actually produced are stored. Supports both attribute access
    (``metrics.ptm``) and dict-style access (``metrics["ptm"]``, ``"ptm" in metrics``).

    Attributes:
        primary_metric (str | None): Key of the primary quality metric for the prediction.
    """

    model_config = ConfigDict(extra="allow")

    primary_metric: str | None = Field(default=None, description="Key of the primary quality metric")

    def __init__(self, *, primary_metric: str | None = None, **kwargs: StructureMetricValue | None) -> None:
        """Accept arbitrary metric keyword arguments alongside primary_metric."""
        super().__init__(primary_metric=primary_metric, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def _exclude_none_values(cls, data: Any) -> Any:
        """Strip None-valued entries so absent metrics are truly absent."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data

    # ── Dict-like interface ──────────────────────────────────────────────────

    def _metric_data(self) -> dict[str, StructureMetricValue]:
        """Return the metric key-value pairs (excludes primary_metric metadata)."""
        return dict(self.__pydantic_extra__) if self.__pydantic_extra__ else {}

    def __contains__(self, key: object) -> bool:
        """Check if a metric key is present."""
        if not isinstance(key, str):
            return False
        return key in self._metric_data()

    def __getitem__(self, key: str) -> StructureMetricValue:
        """Get a metric value by key."""
        data = self._metric_data()
        if key in data:
            return data[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a metric value by key with a default."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        """Return all metric keys."""
        return list(self._metric_data().keys())

    def values(self) -> list[StructureMetricValue]:
        """Return all metric values."""
        return list(self._metric_data().values())

    def items(self) -> list[tuple[str, StructureMetricValue]]:
        """Return all metric key-value pairs."""
        return list(self._metric_data().items())

    def __len__(self) -> int:
        """Return the number of metrics."""
        return len(self._metric_data())

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over metric keys."""
        return iter(self._metric_data())

    def __eq__(self, other: object) -> bool:
        """Compare metric data, supporting comparison with dicts."""
        if isinstance(other, StructureMetrics):
            return self._metric_data() == other._metric_data()
        if isinstance(other, dict):
            return self._metric_data() == other
        return NotImplemented

    def __repr__(self) -> str:
        """Return a string representation."""
        data = self._metric_data()
        items = ", ".join(f"{k}={v!r}" for k, v in data.items())
        primary = f", primary_metric={self.primary_metric!r}" if self.primary_metric else ""
        return f"StructureMetrics({items}{primary})"

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def primary_value(self) -> StructureMetricValue | None:
        """Value of the primary quality metric, or None if not set."""
        if self.primary_metric and self.primary_metric in self:
            return self[self.primary_metric]
        return None

    # ── Validation ───────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_scalars(value: Any) -> list[float | int]:
        """Recursively extract all numeric scalars from a (possibly nested) list."""
        if isinstance(value, (int, float)):
            return [value]
        if isinstance(value, list):
            result: list[float | int] = []
            for item in value:
                result.extend(StructureMetrics._flatten_scalars(item))
            return result
        return []

    def _check_value(self, name: str, value: Any, spec: dict[str, Any]) -> None:
        """Validate a single metric value against its spec."""
        expected_type = spec["type"]
        if expected_type is float:
            if not isinstance(value, (int, float)):
                raise AssertionError(f"Metric '{name}': expected numeric, got {type(value).__name__}")
            if spec["min"] is not None and value < spec["min"]:
                raise AssertionError(f"Metric '{name}'={value} below min {spec['min']}")
            if spec["max"] is not None and value > spec["max"]:
                raise AssertionError(f"Metric '{name}'={value} above max {spec['max']}")
        elif expected_type is list:
            if not isinstance(value, list):
                raise AssertionError(f"Metric '{name}': expected list, got {type(value).__name__}")
            for scalar in self._flatten_scalars(value):
                if spec["min"] is not None and scalar < spec["min"]:
                    raise AssertionError(f"Metric '{name}' element {scalar} below min {spec['min']}")
                if spec["max"] is not None and scalar > spec["max"]:
                    raise AssertionError(f"Metric '{name}' element {scalar} above max {spec['max']}")
        elif expected_type is int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise AssertionError(f"Metric '{name}': expected int, got {type(value).__name__}")
            if spec["min"] is not None and value < spec["min"]:
                raise AssertionError(f"Metric '{name}'={value} below min {spec['min']}")
            if spec["max"] is not None and value > spec["max"]:
                raise AssertionError(f"Metric '{name}'={value} above max {spec['max']}")
        elif expected_type is bool:
            if not isinstance(value, bool):
                raise AssertionError(f"Metric '{name}': expected bool, got {type(value).__name__}")

    def validate_against_spec(self, spec: dict[str, Any]) -> None:
        """Validate all metrics against a declared spec.

        Checks that all "always" metrics are present, every present metric has the
        correct type and is within its declared range, and no undeclared metrics exist.

        Args:
            spec (dict[str, Any]): Mapping of metric names to ``MetricSpec`` dicts.

        Raises:
            AssertionError: If any validation check fails.
        """
        for metric_name, metric_spec in spec.items():
            if metric_spec["availability"] == "always" and metric_name not in self:
                raise AssertionError(f"Guaranteed metric '{metric_name}' not found")
            if metric_name in self:
                self._check_value(metric_name, self[metric_name], metric_spec)
        for key in self:
            if key not in spec:
                raise AssertionError(f"Undeclared metric '{key}'")


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
        metrics (StructureMetrics): Associated metrics (e.g., pLDDT, pTM scores,
            per-chain lists, pairwise matrices). None values are stripped at construction.
    """

    model_config = ConfigDict(extra="forbid")

    structure: str = Field(description="Structure content (PDB or CIF format string)")
    structure_format: Literal["pdb", "cif"] | None = Field(default=None, description="Format of the structure content")
    b_factor_type: BFactorType = Field(
        default=BFactorType.UNSPECIFIED, description="What the B-factor column represents"
    )
    source: str | None = Field(default=None, description="Source identifier for the structure")
    metrics: StructureMetrics = Field(default_factory=StructureMetrics, description="Associated metrics")

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
        metrics: StructureMetrics | dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Structure:
        """Load a Structure from a PDB or CIF file.

        Args:
            path (str | Path): Path to a ``.pdb``, ``.cif``, or ``.mmcif`` file.
            b_factor_type (BFactorType): What the B-factor column represents.
            metrics (StructureMetrics | dict[str, Any] | None): Optional metrics to attach.
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
            metrics = StructureMetrics()
        elif isinstance(metrics, dict):
            metrics = StructureMetrics(**metrics)
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

    def __getattr__(self, name: str) -> Any:
        """Access metrics as attributes, delegating to Pydantic for private attrs."""
        # Let Pydantic handle private attributes (e.g., _gemmi_struct) first
        try:
            return super().__getattr__(name)  # type: ignore[misc]
        except AttributeError:
            pass
        metrics = self.__dict__.get("metrics")
        if metrics is not None and name in metrics:
            return metrics[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def add_metric(self, metric: str, value: StructureMetricValue) -> None:
        """Add a metric to the structure.

        Args:
            metric (str): Name of the metric.
            value (StructureMetricValue): Value of the metric.
        """
        extra = self.metrics.__pydantic_extra__
        if extra is None:
            self.metrics.__pydantic_extra__ = {metric: value}
        else:
            extra[metric] = value

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
    def structure_pdb(self) -> str:
        """Get the structure content as a PDB string, converting from CIF if needed."""
        if self.structure_format == "cif":
            return convert_cif_str_to_pdb_str(self.structure)
        return self.structure

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

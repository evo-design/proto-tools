"""Macromolecular structure representation as a Pydantic BaseModel."""

from __future__ import annotations

import logging
import math
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import gemmi
import numpy as np
import py3Dmol
from biotite.structure import AtomArray, CellList, annotate_sse, superimpose
from biotite.structure import gyration_radius as _biotite_gyration_radius
from biotite.structure import rmsd as _biotite_rmsd
from IPython.display import HTML, display
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from proto_tools.entities.structures.utils import (
    _serialize_gemmi,
    convert_cif_str_to_pdb_str,
    convert_pdb_str_to_cif_str,
    detect_structure_format,
    is_valid_structure,
    load_structure_file,
    looks_like_structure_path,
    pdb_file_to_atomarray,
)
from proto_tools.utils.tool_io import Metrics, MetricValue

VISUALIZE_STYLE_OPTIONS = ["cartoon", "line", "stick", "sphere", "licorice"]


# Per-residue VDW "key atoms" used by ``hotspot_contacts`` with ``germinal_mode=True``.
_KEY_SIDE_CHAIN_ATOMS: dict[str, list[str]] = {
    "VAL": ["CG1", "CG2"],
    "ILE": ["CG1", "CG2", "CD1"],
    "LEU": ["CG", "CD1", "CD2"],
    "MET": ["CG", "SD", "CE"],
    "ALA": [],
    "PRO": ["CG", "CD"],
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "ASP": ["CG", "OD1", "OD2"],
    "GLU": ["CG", "CD", "OE1", "OE2"],
    "LYS": ["CG", "CD", "CE", "NZ"],
    "ARG": ["CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "SER": ["OG"],
    "THR": ["OG1", "CG2"],
    "ASN": ["CG", "OD1", "ND2"],
    "GLN": ["CG", "CD", "OE1", "NE2"],
    "GLY": [],
    "CYS": ["SG"],
}


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

    ``Structure.model_validate`` also accepts a bare ``str`` or ``Path`` as shorthand
    for ``{"structure": <value>}``, so any nested field typed ``Structure`` (e.g.
    ``SequenceStructurePair.structure``, ``MutationInput.structure``) takes a raw
    PDB/CIF string or file path directly — no ``{"structure": ...}`` envelope required.

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
        """Coerce bare input, load paths, then auto-detect format.

        Accepts ``str`` / ``Path`` input as shorthand for ``{"structure": <value>}`` so any
        field typed ``Structure`` on a nested Pydantic model (e.g. ``SequenceStructurePair``,
        ``MutationInput``) takes a raw PDB/CIF string or a file path without the caller
        having to spell out the envelope. This matches ``Structure(structure=<content-or-path>)``
        — the plain-class shortcut — and ``Structure.from_file(<path>)``.
        """
        if isinstance(data, (str, Path)):
            data = {"structure": data}

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
                    if len(struct) > 0:
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

    def select_chain(self, chain_id: str) -> Structure:
        """Return a new Structure with only the requested chain.

        Preserves ``b_factor_type``, ``source``, a deep copy of ``metrics``, and
        the original ``structure_format``.

        Args:
            chain_id (str): Chain identifier to retain.

        Returns:
            Structure: New Structure containing only the requested chain.

        Raises:
            ValueError: If ``chain_id`` is not present in this Structure.
        """
        return self.select_chains([chain_id])

    def select_chains(self, chain_ids: str | list[str] | tuple[str, ...]) -> Structure:
        """Return a new Structure with only the requested chains.

        ``chain_ids`` may be a comma-separated string (for example ``"A,B"``)
        or an explicit sequence of chain identifiers. Output preserves
        ``b_factor_type``, ``source``, a deep copy of ``metrics``, and the
        original ``structure_format``.

        Args:
            chain_ids (str | list[str] | tuple[str, ...]): Chain identifiers to retain.

        Returns:
            Structure: New Structure containing only the requested chains.

        Raises:
            ValueError: If no chain IDs are requested or any requested chain is
                not present in this Structure.
        """
        if isinstance(chain_ids, str):
            requested = [chain_id.strip() for chain_id in chain_ids.split(",") if chain_id.strip()]
        else:
            requested = [chain_id.strip() for chain_id in chain_ids if chain_id.strip()]
        if not requested:
            raise ValueError("At least one chain ID must be requested")

        available = {chain.name for model in self.gemmi_struct for chain in model}
        missing = [chain_id for chain_id in requested if chain_id not in available]
        if missing:
            raise ValueError(f"Chain(s) {missing!r} not present in Structure")

        keep = set(requested)
        struct = self.gemmi_struct.clone()
        for model in struct:
            for i in range(len(model) - 1, -1, -1):
                if model[i].name not in keep:
                    del model[i]
        new_format: Literal["pdb", "cif"] = self.structure_format or "pdb"
        serialized = _serialize_gemmi(struct, new_format, source_format=new_format)
        return Structure(
            structure=serialized,
            structure_format=new_format,
            b_factor_type=self.b_factor_type,
            source=self.source,
            metrics=self.metrics.model_copy(deep=True),
        )

    def with_renamed_chains(self, mapping: dict[str, str]) -> Structure:
        """Return a new Structure with chain IDs renamed via the given mapping.

        Output preserves the source format (PDB stays PDB, CIF stays CIF).
        Chain IDs not in the mapping pass through unchanged. Mirrors
        :meth:`select_chain` — preserves ``b_factor_type``, ``source``, and a
        deep copy of ``metrics``.

        Args:
            mapping (dict[str, str]): Old chain ID → new chain ID. Identity
                and unmatched entries are no-ops.

        Returns:
            Structure: New Structure with renamed chains, in the same format as
                ``self``. Returns ``self`` unchanged when the mapping is empty
                or fully identity.

        Raises:
            ValueError: On duplicate target chain IDs, or on a multi-character
                target on a PDB Structure (PDB col 22 is single-char only —
                convert via ``Structure(structure=self.structure_cif,
                structure_format="cif")`` first).
        """
        if not mapping or all(old == new for old, new in mapping.items()):
            return self

        struct = self.gemmi_struct.clone()
        new_names_seen: set[str] = set()
        for model in struct:
            for chain in model:
                new_name = mapping.get(chain.name, chain.name)
                if new_name in new_names_seen:
                    raise ValueError(f"Renaming would produce duplicate chain ID {new_name!r}")
                new_names_seen.add(new_name)
                chain.name = new_name

        if self.structure_format == "pdb":
            multi_char = sorted(name for name in new_names_seen if len(name) > 1)
            if multi_char:
                raise ValueError(
                    f"Cannot rename to multi-character chain ID(s) {multi_char} on a PDB-format "
                    "Structure (PDB column 22 is single-character only). Convert to CIF first via "
                    "Structure(structure=self.structure_cif, structure_format='cif'), then rename."
                )

        new_format: Literal["pdb", "cif"] = self.structure_format or "pdb"
        serialized = _serialize_gemmi(struct, new_format, source_format=new_format)

        return Structure(
            structure=serialized,
            structure_format=new_format,
            b_factor_type=self.b_factor_type,
            source=self.source,
            metrics=self.metrics.model_copy(deep=True),
        )

    @classmethod
    def concat(cls, structures: list[Structure]) -> Structure:
        """Merge Structures with distinct chain IDs into a single multi-chain Structure.

        Coordinates are preserved as-is. Callers are responsible for ensuring inputs
        share a coordinate frame — e.g., Structures produced by ``select_chain`` on a
        common source trivially do, as do chains predicted jointly by the same model.
        Uses model 0 from each input; additional models are ignored.

        **Format-preserving:** the output format follows the inputs. All-PDB inputs
        produce a PDB output; all-CIF inputs produce a CIF output (which can hold
        multi-character chain IDs). Mixed-format inputs raise — the caller should
        coerce explicitly via ``Structure(structure=s.structure_cif,
        structure_format="cif")`` (or its PDB analog) to opt into the format change.

        Args:
            structures (list[Structure]): Non-empty list to merge, in order.

        Returns:
            Structure: New Structure with all chains combined, in the same format
                as the inputs. ``b_factor_type`` is inherited from the first input;
                ``source`` and ``metrics`` are not carried over (merging them
                ambiguously would hide provenance).

        Raises:
            ValueError: If ``structures`` is empty, ``b_factor_type`` differs across
                inputs, ``structure_format`` is mixed (some PDB, some CIF), any
                chain ID appears in more than one input, or any chain ID is
                multi-character when the output format is PDB (PDB column 22 is
                single-character only — convert inputs to CIF first to allow it).
        """
        if not structures:
            raise ValueError("concat requires at least one Structure")
        formats = {s.structure_format for s in structures if s.structure_format is not None}
        if len(formats) > 1:
            raise ValueError(
                f"concat requires all inputs to share structure_format; got {sorted(formats)}. "
                "Coerce explicitly via Structure(structure=s.structure_cif, structure_format='cif') "
                "(or .structure_pdb / 'pdb') first."
            )
        target_format: Literal["pdb", "cif"] = next(iter(formats), "pdb")
        b_factor_type = structures[0].b_factor_type
        merged = gemmi.Structure()
        merged.add_model(gemmi.Model(1))
        seen: set[str] = set()
        for struct in structures:
            if struct.b_factor_type != b_factor_type:
                raise ValueError(f"concat b_factor_type mismatch: {struct.b_factor_type} vs {b_factor_type}")
            if len(struct.gemmi_struct) == 0:
                continue
            for chain in struct.gemmi_struct[0]:
                if target_format == "pdb" and len(chain.name) != 1:
                    raise ValueError(
                        f"Chain {chain.name!r} must be a single character when concat output is PDB. "
                        "Convert inputs to CIF (Structure(structure=s.structure_cif, structure_format='cif')) "
                        "to allow multi-char chain IDs."
                    )
                if chain.name in seen:
                    raise ValueError(f"Duplicate chain ID {chain.name!r} across concat inputs")
                seen.add(chain.name)
                new_chain = gemmi.Chain(chain.name)
                for residue in chain:
                    new_chain.add_residue(residue)
                merged[0].add_chain(new_chain)
        # source_format == target_format (mixed-format raised above); re-emission
        # never warns — caller already chose the format by passing matching inputs.
        return cls(
            structure=_serialize_gemmi(merged, target_format, source_format=target_format),
            structure_format=target_format,
            b_factor_type=b_factor_type,
        )

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
                    if len(candidate) > 0:
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

            # Chain shortening is intentional and silent (callers consume the
            # map). Other lossy aspects of CIF→PDB (long atom names, atom-count
            # caps, NCS metadata) still warn via _serialize_gemmi.
            pdb_content = _serialize_gemmi(
                gemmi_struct,
                "pdb",
                source_format="cif",
                cif_content_for_warnings=self.structure,
            )
            return pdb_content, chain_id_map
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
    # Interface & Clash Analysis
    # ============================================================================

    def ca_clash_score(self, threshold: float = 2.5) -> int:
        """Count Ca-Ca atom pairs with distance ≤ threshold, excluding bonded neighbors.

        "Bonded" means same chain, ``|res_id_i - res_id_j| == 1`` — numeric, so a
        chain break with sequential numbering is mis-excluded.

        Args:
            threshold (float): Distance cutoff in Å. Defaults to 2.5 (Germinal VHH clash gate).

        Returns:
            int: Number of clashing Ca-Ca pairs.

        Raises:
            ValueError: If any chain in the structure has a multi-character chain ID
                (the PDB conversion path truncates these, which could silently collide
                distinct chains onto one ID and mis-fire the bonded-neighbor exclusion).
        """
        for chain in self.gemmi_struct[0]:  # Guard the first model — same one pdb_file_to_atomarray reads.
            if len(chain.name) != 1:
                raise ValueError(f"Chain {chain.name!r} must be a single character.")
        atom_array = pdb_file_to_atomarray(StringIO(self.structure_pdb))
        ca_mask = atom_array.atom_name == "CA"
        ca_coords = atom_array.coord[ca_mask]
        ca_chain = atom_array.chain_id[ca_mask]
        ca_res = atom_array.res_id[ca_mask]
        if len(ca_coords) < 2:
            return 0

        cells = CellList(ca_coords, cell_size=threshold)
        clashes = 0
        for i, coord in enumerate(ca_coords):
            for j in cells.get_atoms(coord, radius=threshold):
                if j <= i:
                    continue  # triu — skips self-pair (get_atoms includes i at d=0) and double-counting.
                if ca_chain[i] == ca_chain[j] and abs(int(ca_res[i]) - int(ca_res[j])) == 1:
                    continue  # bonded neighbors
                clashes += 1
        return clashes

    def interface_contact_residues(
        self,
        binder_chain: str,
        target_chains: str | list[str] | tuple[str, ...],
        cutoff: float = 4.0,
        include_hydrogens: bool = False,
    ) -> dict[int, str]:
        """Binder residues with any atom within ``cutoff`` Å of a target atom.

        ``target_chains`` may be a comma-separated string or an explicit sequence of
        chain IDs.

        Args:
            binder_chain (str): Chain ID of the binder.
            target_chains (str | list[str] | tuple[str, ...]): Chain ID(s) of the target.
            cutoff (float): Atom distance cutoff in Å. Defaults to 4.0 (Germinal
                ``hotspot_residues`` default; the VHH pipeline uses 3.0 via
                ``vhh.yaml:122 atom_distance_cutoff``).
            include_hydrogens (bool): Include hydrogens in the distance check. Defaults to
                ``False`` (heavy-only). Set ``True`` for Germinal parity on PyRosetta-relaxed
                inputs, which carry hydrogens that upstream ``hotspot_residues`` counts.

        Returns:
            dict[int, str]: Map from 1-indexed binder residue number to single-letter AA
                code for every residue with at least one atom within ``cutoff``.

        Raises:
            ValueError: If any chain ID is more than 1 character, or if ``binder_chain``
                appears in ``target_chains`` (self-contact is not meaningful).
        """
        raw_chain_ids = [target_chains] if isinstance(target_chains, str) else target_chains
        targets = [chain.strip() for raw in raw_chain_ids for chain in raw.split(",") if chain.strip()]
        if not targets:
            raise ValueError("target_chains must contain at least one chain ID.")
        for cid in [binder_chain, *targets]:
            if len(cid) != 1:
                raise ValueError(f"Chain ID {cid!r} must be a single character.")
        if binder_chain in targets:
            raise ValueError(f"binder_chain {binder_chain!r} must not also appear in target_chains.")
        atom_array = pdb_file_to_atomarray(StringIO(self.structure_pdb))
        atoms = atom_array if include_hydrogens else atom_array[atom_array.element != "H"]
        binder_atoms = atoms[atoms.chain_id == binder_chain]
        target_atoms = atoms[np.isin(atoms.chain_id, targets)]
        if len(binder_atoms) == 0 or len(target_atoms) == 0:
            return {}

        cells = CellList(binder_atoms.coord, cell_size=cutoff)
        touched: set[int] = set()
        for coord in target_atoms.coord:
            for idx in cells.get_atoms(coord, radius=cutoff):
                touched.add(int(binder_atoms.res_id[idx]))

        # get_residue_position_map is polymer-only — HETATM/ligand residues in `touched` are dropped here.
        residue_map = self.get_residue_position_map()[binder_chain]
        aa_by_pos = {pos: aa for aa, pos in residue_map}
        return {pos: aa_by_pos[pos] for pos in sorted(touched) if pos in aa_by_pos}

    def hotspot_contacts(
        self,
        binder_chain: str,
        target_hotspots: str | list[str],
        expansion_cutoff: float = 5.3,
        contact_cutoff: float = 6.0,
        binder_positions: list[int] | None = None,
        germinal_mode: bool = False,
    ) -> dict[int, str]:
        """Binder residues near declared target hotspots via a two-step filter.

        1. Expand: target residues within ``expansion_cutoff`` Å of any hotspot atom.
        2. Contact: binder residues within ``contact_cutoff`` Å of any atom in that expanded region.

        ``germinal_mode=True`` switches to CA expansion + key-atom contacts; default is all-heavy.

        Raises:
            ValueError: ``binder_chain`` not single-char, in hotspot chains, or bad token format.
        """
        if len(binder_chain) != 1:
            raise ValueError(f"Chain ID {binder_chain!r} must be a single character.")

        # Parse "A45,A47" or ["A45", "A47"] → {chain: {residue_num, ...}}.
        tokens = target_hotspots.split(",") if isinstance(target_hotspots, str) else target_hotspots
        hotspots: dict[str, set[int]] = {}
        for raw in (t.strip() for t in tokens):
            if not raw:
                continue
            chain, residue_str = raw[0], raw[1:]
            if not chain.isalpha() or not residue_str.lstrip("-").isdigit():
                raise ValueError(f"Hotspot {raw!r} must be chain-prefixed like 'A45'.")
            hotspots.setdefault(chain, set()).add(int(residue_str))
        if binder_chain in hotspots:
            raise ValueError(f"binder_chain {binder_chain!r} must not also appear in target_hotspots.")
        if not hotspots:
            return {}

        atoms = pdb_file_to_atomarray(StringIO(self.structure_pdb))
        if germinal_mode:
            # Germinal's PyRosetta-era semantics: CA-CA distance for expansion, "key atoms"
            # (CA + CB-if-not-GLY + per-residue sidechain atoms) for VDW contacts.
            expansion_atom_mask = atoms.atom_name == "CA"
            contact_atom_mask = atoms.atom_name == "CA"
            contact_atom_mask |= (atoms.atom_name == "CB") & (atoms.res_name != "GLY")
            for res_name, atom_names in _KEY_SIDE_CHAIN_ATOMS.items():
                if atom_names:
                    contact_atom_mask |= (atoms.res_name == res_name) & np.isin(atoms.atom_name, atom_names)
        else:
            expansion_atom_mask = atoms.element != "H"
            contact_atom_mask = atoms.element != "H"

        binder_chain_mask = atoms.chain_id == binder_chain
        if binder_positions is not None:
            binder_chain_mask &= np.isin(atoms.res_id, binder_positions)

        # Step 1: per-chain expansion (neighbor search is restricted to each hotspot's own chain).
        # expansion_cutoff=0 skips the widening (declared hotspots only).
        expanded_by_chain: dict[str, set[int]] = {}
        for chain, residues in hotspots.items():
            if expansion_cutoff <= 0:
                expanded_by_chain[chain] = set(residues)
                continue
            chain_mask = atoms.chain_id == chain
            chain_hotspot_atoms = atoms[expansion_atom_mask & chain_mask & np.isin(atoms.res_id, list(residues))]
            chain_target_atoms = atoms[expansion_atom_mask & chain_mask]
            if len(chain_hotspot_atoms) == 0 or len(chain_target_atoms) == 0:
                continue
            cells = CellList(chain_target_atoms.coord, cell_size=expansion_cutoff)
            expanded: set[int] = set()
            for coord in chain_hotspot_atoms.coord:
                for idx in cells.get_atoms(coord, radius=expansion_cutoff):
                    expanded.add(int(chain_target_atoms.res_id[idx]))
            if expanded:
                expanded_by_chain[chain] = expanded

        if not expanded_by_chain:
            return {}

        expanded_residue_mask = np.logical_or.reduce(
            [(atoms.chain_id == c) & np.isin(atoms.res_id, list(r)) for c, r in expanded_by_chain.items()]
        )
        expanded_contact_atoms = atoms[contact_atom_mask & expanded_residue_mask]
        binder_contact_atoms = atoms[contact_atom_mask & binder_chain_mask]
        if len(expanded_contact_atoms) == 0 or len(binder_contact_atoms) == 0:
            return {}

        # Step 2: binder atoms within contact_cutoff of any expanded-region atom.
        cells = CellList(binder_contact_atoms.coord, cell_size=contact_cutoff)
        touched: set[int] = set()
        for coord in expanded_contact_atoms.coord:
            for idx in cells.get_atoms(coord, radius=contact_cutoff):
                touched.add(int(binder_contact_atoms.res_id[idx]))

        residue_map = self.get_residue_position_map()[binder_chain]
        aa_by_pos = {pos: aa for aa, pos in residue_map}
        return {pos: aa_by_pos[pos] for pos in sorted(touched) if pos in aa_by_pos}

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
    # Structural Analysis (biotite)
    # ============================================================================

    def _get_atom_array(self, chain_id: str | None = None) -> AtomArray:
        """Load biotite AtomArray, optionally filtered to a single chain."""
        array = pdb_file_to_atomarray(StringIO(self.structure_pdb))
        if chain_id is not None:
            array = array[array.chain_id == chain_id]
        return array

    def secondary_structure_percentages(self, chain_id: str | None = None) -> dict[str, float]:
        """Helix/sheet/loop percentages via biotite P-SEA algorithm.

        Args:
            chain_id (str | None): Chain to analyze. None uses the full structure.

        Returns:
            dict[str, float]: Keys ``"helix"``, ``"sheet"``, ``"loop"`` with values 0-100.
        """
        sse = annotate_sse(self._get_atom_array(chain_id))
        sse = sse[sse != ""]  # exclude non-amino-acid residues
        total = max(len(sse), 1)
        helix = int(np.sum(sse == "a"))
        sheet = int(np.sum(sse == "b"))
        loop = total - helix - sheet
        return {"helix": 100.0 * helix / total, "sheet": 100.0 * sheet / total, "loop": 100.0 * loop / total}

    def gyration_radius(self, chain_id: str | None = None) -> float:
        """Radius of gyration in Angstroms.

        Args:
            chain_id (str | None): Chain to analyze. None uses the full structure.

        Returns:
            float: Radius of gyration.
        """
        return float(_biotite_gyration_radius(self._get_atom_array(chain_id)))

    def longest_alpha_helix(self, chain_id: str | None = None) -> int:
        """Length of longest contiguous alpha helix segment.

        Args:
            chain_id (str | None): Chain to analyze. None uses the full structure.

        Returns:
            int: Residue count of the longest helix.
        """
        sse = annotate_sse(self._get_atom_array(chain_id))
        max_len = 0
        cur = 0
        for s in sse:
            if s == "a":
                cur += 1
                max_len = max(max_len, cur)
            else:
                cur = 0
        return max_len

    def backbone_rmsd(self, other: Structure, chain_id: str | None = None) -> float:
        """CA-atom RMSD after optimal superposition against another structure.

        CA atoms are paired by array index, not residue ID. Structures must share
        the same residue ordering for meaningful results. If lengths differ, only
        the first ``min(len_self, len_other)`` CA atoms are compared.

        Args:
            other (Structure): Reference structure to align against.
            chain_id (str | None): Chain to compare. None uses all chains.

        Returns:
            float: RMSD in Angstroms after superposition.
        """
        self_array = self._get_atom_array(chain_id)
        other_array = other._get_atom_array(chain_id)
        ca_self = self_array[self_array.atom_name == "CA"]
        ca_other = other_array[other_array.atom_name == "CA"]
        min_len = min(len(ca_self), len(ca_other))
        if min_len == 0:
            return float("inf")
        if len(ca_self) != len(ca_other):
            logging.getLogger(__name__).warning(
                "CA atom count mismatch (%d vs %d); RMSD computed on first %d atoms",
                len(ca_self),
                len(ca_other),
                min_len,
            )
        ca_self = ca_self[:min_len]
        ca_other = ca_other[:min_len]
        ca_self_sup, _ = superimpose(ca_other, ca_self)
        return float(_biotite_rmsd(ca_other, ca_self_sup))

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

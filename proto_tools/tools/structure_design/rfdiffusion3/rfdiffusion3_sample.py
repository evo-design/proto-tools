"""proto_tools/tools/structure_design/rfdiffusion3/rfdiffusion3_sample.py.

Example:
    >>> from proto_tools.tools.structure_design.rfdiffusion3 import (
    ...     run_rfdiffusion3,
    ...     RFdiffusion3Input,
    ...     RFdiffusion3Config,
    ...     RFdiffusion3DesignSpec,
    ... )
    >>> # Simple unconditional design
    >>> inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="100")])
    >>> config = RFdiffusion3Config(diffusion_batch_size=4)
    >>> result = run_rfdiffusion3(inputs, config)
    >>> n_designs = sum(len(b) for b in result.designed_structures)
    >>> print(f"Designed {n_designs} structures across {len(result)} input specs")
"""

import json
import logging
import math
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proto_tools.entities.complex import Complex, chain_label
from proto_tools.entities.structures import Structure, detect_structure_format
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.file_inputs import STRUCTURE_FILES

logger = logging.getLogger(__name__)


def _pdb_coordinate(line: str, start: int, end: int) -> float:
    """Parse one fixed-width PDB coordinate, naming the record when it is not a number."""
    field = line[start:end]
    try:
        return float(field)
    except ValueError as exc:
        raise ValueError(f"rfdiffusion3: input PDB record has a non-numeric coordinate {field!r}: {line!r}") from exc


def _drop_unresolved_pdb_atoms(content: str) -> tuple[str, dict[str, list[str]], int]:
    """Remove ``ATOM``/``HETATM`` records (and their ``ANISOU``) whose coordinates are not finite.

    Returns the text, the dropped atom names by residue, and the number of atom records kept.
    """
    kept: list[str] = []
    dropped: dict[str, list[str]] = defaultdict(list)
    n_kept = 0
    skip_anisou = False
    for line in content.splitlines(keepends=True):
        record = line[:6]
        if record in ("ATOM  ", "HETATM"):
            skip_anisou = False
            coords = (_pdb_coordinate(line, 30, 38), _pdb_coordinate(line, 38, 46), _pdb_coordinate(line, 46, 54))
            if not all(math.isfinite(c) for c in coords):
                residue = f"{line[21].strip()}{line[22:27].strip()} {line[17:20].strip()}"
                dropped[residue].append(line[12:16].strip())
                skip_anisou = True
                continue
            n_kept += 1
        elif record == "ANISOU" and skip_anisou:
            skip_anisou = False
            continue
        kept.append(line)
    return "".join(kept), dict(dropped), n_kept


def _drop_unresolved_cif_atoms(content: str) -> tuple[str, dict[str, list[str]], int]:
    """Remove ``_atom_site`` rows whose ``Cartn_x/y/z`` are ``nan``, ``?`` or ``.``; keep every other row and column.

    Returns the text, the dropped atom names by residue, and the number of atom rows kept.
    """
    import gemmi

    doc = gemmi.cif.read_string(content)
    dropped: dict[str, list[str]] = defaultdict(list)
    n_kept = 0
    for block in doc:
        table = block.find(
            "_atom_site.",
            ["Cartn_x", "Cartn_y", "Cartn_z", "?auth_asym_id", "?auth_seq_id", "?label_comp_id", "?label_atom_id"],
        )
        unresolved = [
            index
            for index, row in enumerate(table)
            if not all(math.isfinite(gemmi.cif.as_number(row[i])) for i in range(3))
        ]
        for index in unresolved:
            row = table[index]
            chain, seq, comp, atom = (gemmi.cif.as_string(row[i]) if row.has(i) else "?" for i in range(3, 7))
            dropped[f"{chain}{seq} {comp}"].append(atom)
        # Remove from the end so earlier indices stay valid.
        for index in reversed(unresolved):
            table.remove_row(index)
        n_kept += len(table)
    if not dropped:
        return content, {}, n_kept
    return doc.as_string(), dict(dropped), n_kept


def _drop_unresolved_atoms(content: str, structure_format: str, label: str) -> str:
    """Return ``content`` without atoms that have no coordinates, warning with the atoms dropped.

    rfd3 reads every atom in its input as fixed context and, at inference, has no
    representation for an unresolved one: its loader keeps an atom written with ``nan``
    coordinates (``add_missing_atoms=False``), the transforms that place unresolved atoms
    run only in training, and the NaN reaches the denoiser input, which fails with
    ``Tensor contains NaNs!``. An atom absent from the file is what rfd3 expects for one
    without coordinates, so those atoms are removed, as a PDB entry omits atoms it did
    not resolve. Atom records with coordinates, and every other category, are kept.

    Args:
        content (str): PDB or CIF text.
        structure_format (str): ``"pdb"`` or ``"cif"``.
        label (str): Spec key the structure belongs to, named in the warning.

    Returns:
        str: ``content`` unchanged when every atom has coordinates, else the text without the
            unresolved atoms.

    Raises:
        ValueError: If no atom in the structure has coordinates.
    """
    if structure_format == "cif":
        cleaned, dropped, n_kept = _drop_unresolved_cif_atoms(content)
    else:
        cleaned, dropped, n_kept = _drop_unresolved_pdb_atoms(content)
    if not dropped:
        return content
    if n_kept == 0:
        raise ValueError(f"rfdiffusion3: {label}: no atom in the input structure has coordinates.")
    count = sum(len(atoms) for atoms in dropped.values())
    logger.warning(
        "rfdiffusion3: %s: removing %d input atoms without coordinates, which rfd3 cannot "
        "condition on (it reads every input atom as fixed context): %s",
        label,
        count,
        "; ".join(f"{residue} {','.join(atoms)}" for residue, atoms in dropped.items()),
    )
    return cleaned


def _stage_raw_json_input(value: str, label: str, stem: str, staging_dir: Callable[[], Path]) -> str:
    """Return a ``raw_json`` ``"input"`` that names a structure file as an absolute path; leave content as is.

    A file with atoms that have no coordinates is staged as ``{stem}_input.{ext}`` without
    them (see ``_drop_unresolved_atoms``); the caller's file is never modified.
    """
    path = STRUCTURE_FILES.resolve(value)
    if path is None:
        return value
    content = path.read_text()
    structure_format = detect_structure_format(content)
    cleaned = _drop_unresolved_atoms(content, structure_format, label)
    if cleaned is content:
        return str(path.resolve())
    staged = staging_dir() / f"{stem}_input.{structure_format}"
    staged.write_text(cleaned)
    return str(staged)


# ============================================================================
# Data Models
# ============================================================================
# Input Data Models


class RFdiffusion3DesignSpec(BaseModel):
    r"""Single design specification for RFdiffusion3.

    Specify a design task and its constraints. Provide multiple specifications
    to generate designs under different conditions.

    For the full list of available InputSpecification fields, see:
    https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md

    Attributes:
        input_structure (Structure | None): Input structure (``Structure`` or PDB/CIF file path)
            for motif/binder design; written to a file before rfd3 (reads a path). Omit for de novo.
            Atoms without coordinates are left out of that file, with a warning naming them.

        contig (str | None): Contig string specifying the design topology.
            Format: comma-separated segments with chain breaks as ``/0``. Examples:
            ``"50-80"`` (design 50-80 residue monomer);
            ``"A1-100,50,A150-200"`` (scaffold around residues A1-100 and A150-200);
            ``"50,/0,B1-50"`` (design 50 residues, chain break, then keep B1-50).

        length (str | None): Per-asymmetric-unit length (int or
            ``"min-max"``); per-protomer under ``symmetry``.

        ligand (str | None): Ligand selection by residue name from input structure.
            Comma-separated list of 3-letter codes (e.g., ``"HAX,OAA"``).

        unindex (str | dict[str, str] | None): Unindexed motif components whose
            sequence position is unknown to the model. Useful for active site scaffolding
            where catalytic residues should be placed but their position in the final
            sequence is flexible. Components must not overlap with ``contig``.
            Example: ``"A244,A274,A320"`` lists multiple unindexed components.

        select_fixed_atoms (bool | str | dict[str, str] | None): Atoms to fix
            in 3D space during diffusion. Accepts InputSelection format:
            - ``True``: Fix all atoms from input
            - ``False``: Unfix all atoms
            - Contig string: Specify residues to fix (e.g., ``"A1-10,B1-3"``)
            - Dict: Map chain/residue to atom selection using ``"BKBN"`` (backbone),
              ``"TIP"`` (tip atom), ``"ALL"`` (all atoms), or explicit atom names
              (e.g., ``{"A1": "N,CA,C,O,CB", "A2-10": "BKBN"}``)

        select_unfixed_sequence (bool | str | dict[str, str] | None): Residues
            whose sequence can change during design. Accepts InputSelection format:
            - ``True``: All input regions have fixed sequences (upstream default)
            - ``False``: All atoms have unfixed/diffused sequences
            - Contig string: Components to unfix sequence for (e.g., ``"A5-10,B1-3"``)
            Note: Ligands and DNA always have fixed sequences.

        select_hotspots (bool | str | dict[str, str] | None): Atom or residue
            hotspots for binder/PPI design (typically <=4.5 Angstroms to any
            heavy atom in the designed structure).

        symmetry (str | dict[str, Any] | None): Symmetry for homo-oligomer design.
            A group-id string (e.g. ``"C3"``) is wrapped as ``{"id": "C3"}``; a
            full ``SymmetryConfig`` dict is passed through. Pair with
            ``RFdiffusion3Config.sampler_kind="symmetry"``.

        select_buried (bool | str | dict[str, str] | None): RASA selector for
            buried residues.

        select_partially_buried (bool | str | dict[str, str] | None): RASA
            selector for partially buried residues.

        select_exposed (bool | str | dict[str, str] | None): RASA selector for
            solvent-exposed residues.

        select_hbond_donor (dict[str, list[str]] | None): Atom-wise H-bond
            donor flags, e.g. ``{"A40": ["NE2"]}``.

        select_hbond_acceptor (dict[str, list[str]] | None): Atom-wise H-bond
            acceptor flags, e.g. ``{"A45": ["OD1"]}``.

        redesign_motif_sidechains (bool | None): Keep motif backbone fixed,
            redesign side-chains.

        plddt_enhanced (bool | None): Enable pLDDT-based denoising enhancement
            (upstream default ``True``; ``None`` keeps it).

        infer_ori_strategy (Literal['com', 'hotspots'] | None): Origin
            placement — ``com`` (center of mass) or ``hotspots``.

        ori_token (list[float] | None): ``[x, y, z]`` origin override (Angstroms).

        partial_t (float | None): Noise level (in Angstroms) for partial diffusion, which
            resamples backbones near the input. Lower values preserve more of the input
            structure. Recommended values are 5.0-15.0 Angstroms. Requires
            ``select_fixed_atoms`` to unfix the atoms to re-noise.

        is_non_loopy (bool | None): If ``True``/``False``, produces output
            structures with fewer/more loops. ``None`` uses the model's
            native default.

    Note:
        Only the typed fields above are accepted; unknown keys are rejected.
        For exotic InputSpecification fields (e.g. ``dialect``,
        ``cif_parser_args``), pass the full spec through
        ``RFdiffusion3Input.raw_json``. See
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
    """

    model_config = ConfigDict(extra="forbid")

    input_structure: Structure | None = Field(
        default=None,
        title="Input Structure",
        description="Input structure (Structure or PDB/CIF file path) for motif/binder design. Omit for de novo.",
    )
    contig: str | None = Field(
        default=None,
        title="Contig",
        description="Contig string specifying design topology (e.g., '50-80,/0,A1-100')",
        examples=["50-80", "A1-100,50,A150-200"],
    )
    length: str | None = Field(
        default=None,
        title="Length",
        description="Per-asymmetric-unit length (int or 'min-max'); per-protomer under symmetry",
        examples=["100", "80-120"],
    )
    ligand: str | None = Field(
        default=None,
        title="Ligand",
        description="Ligand selection by residue name (e.g., 'HAX,OAA')",
    )
    unindex: str | dict[str, str] | None = Field(
        default=None,
        title="Unindexed Motifs",
        description="Unindexed motif components for flexible positioning (e.g., 'A244,A274,A320')",
    )
    select_fixed_atoms: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Fixed Atoms",
        description="Atoms to fix in 3D space (True/False, contig string, or dict with BKBN/TIP/ALL)",
    )
    select_unfixed_sequence: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Unfixed Sequence",
        description="Residues whose sequence can change (True/False, contig string, or dict)",
    )
    select_hotspots: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Hotspots",
        description="Atom/residue-level hotspots for binder design (True/False, contig string, or dict)",
        examples=["A24,A35,A50"],
    )
    symmetry: str | dict[str, Any] | None = Field(
        default=None,
        title="Symmetry",
        description="Homo-oligomer symmetry: group-id or SymmetryConfig dict; pair with sampler_kind='symmetry'",
        examples=["C3", {"id": "C3"}],
    )
    select_buried: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Buried Residues",
        description="RASA selector for residues that should be buried (True/False, contig string, or dict)",
    )
    select_partially_buried: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Partially Buried Residues",
        description="RASA selector for partially buried residues (True/False, contig string, or dict)",
    )
    select_exposed: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Exposed Residues",
        description="RASA selector for solvent-exposed residues (True/False, contig string, or dict)",
    )
    select_hbond_donor: dict[str, list[str]] | None = Field(
        default=None,
        title="H-bond Donor Flags",
        description="Atom-wise hydrogen-bond donor flags, e.g. {'A40': ['NE2']}",
    )
    select_hbond_acceptor: dict[str, list[str]] | None = Field(
        default=None,
        title="H-bond Acceptor Flags",
        description="Atom-wise hydrogen-bond acceptor flags, e.g. {'A45': ['OD1']}",
    )
    redesign_motif_sidechains: bool | None = Field(
        default=None,
        title="Redesign Motif Sidechains",
        description="Keep motif backbone fixed but redesign side-chains",
    )
    plddt_enhanced: bool | None = Field(
        default=None,
        title="pLDDT-Enhanced Denoising",
        description="Enable pLDDT-based denoising enhancement (upstream default: True)",
    )
    infer_ori_strategy: Literal["com", "hotspots"] | None = Field(
        default=None,
        title="Origin Strategy",
        description="Origin placement strategy: 'com' (center of mass) or 'hotspots'",
    )
    ori_token: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        title="Origin (x, y, z)",
        description="Origin override for center-of-mass placement, in Ångstroms (shape [x, y, z])",
        examples=[[0.0, 0.0, 0.0]],
    )
    partial_t: float | None = Field(
        default=None,
        ge=0.0,
        title="Partial Diffusion Noise",
        description="Noise level in Ångstroms for partial diffusion; 5.0-15.0 recommended",
        examples=[5.0, 10.0, 15.0],
    )
    is_non_loopy: bool | None = Field(
        default=None,
        title="Suppress Loops",
        description="When True, produces structures with fewer loops; when False, more loops",
    )

    @field_validator("symmetry", mode="before")
    @classmethod
    def _normalize_symmetry(cls, value: Any) -> Any:
        """Wrap a group-id string (e.g. ``"c3"``) as the ``{"id": "C3"}`` dict rfd3 requires."""
        if isinstance(value, str):
            return {"id": value.strip().upper()}
        return value

    @field_validator("contig")
    @classmethod
    def _reject_literal_nul_in_contig(cls, value: str | None) -> str | None:
        if value is not None and "\0" in value:
            raise ValueError("contig contains a literal NUL byte; use '/0' for RFdiffusion3 chain breaks")
        if value is not None and "\\0" in value:
            raise ValueError("contig contains '\\0'; use '/0' for RFdiffusion3 chain breaks")
        return value

    @model_validator(mode="after")
    def validate_has_design_params(self) -> Any:
        """Reject specs with no constraints (at least one typed design field is required)."""
        typed_design_fields = (
            self.contig,
            self.length,
            self.input_structure,
            self.ligand,
            self.unindex,
            self.select_fixed_atoms,
            self.select_unfixed_sequence,
            self.select_hotspots,
            self.symmetry,
            self.select_buried,
            self.select_partially_buried,
            self.select_exposed,
            self.select_hbond_donor,
            self.select_hbond_acceptor,
            self.redesign_motif_sidechains,
            self.plddt_enhanced,
            self.infer_ori_strategy,
            self.ori_token,
            self.partial_t,
            self.is_non_loopy,
        )
        if not any(f is not None for f in typed_design_fields):
            raise ValueError(
                "At least one design parameter (contig, length, symmetry, select_*, partial_t, etc.) must be provided"
            )
        return self

    @model_validator(mode="after")
    def validate_selections_require_input_structure(self) -> Any:
        """Reject ``contig``, ``unindex``, and ``select_*`` fields without ``input_structure``."""
        # Upstream rfd3 resolves these against an atom array built from input_structure (even
        # chain-referencing contigs fail without it). ``length`` is the only input-free path.
        requires_input = (
            "contig",
            "unindex",
            "select_fixed_atoms",
            "select_unfixed_sequence",
            "select_hotspots",
            "select_buried",
            "select_partially_buried",
            "select_exposed",
            "select_hbond_donor",
            "select_hbond_acceptor",
        )
        offending = [name for name in requires_input if getattr(self, name) is not None]
        if offending and self.input_structure is None:
            raise ValueError(
                f"Fields {offending} require 'input_structure' (PDB/CIF) to resolve "
                "against an atom array. For unconditional design with no per-residue "
                "constraints, use 'length' instead."
            )
        return self

    @model_validator(mode="after")
    def validate_contig_xor_length(self) -> Any:
        """Reject ``contig`` and ``length`` together; a contig already encodes its lengths."""
        if self.contig is not None and self.length is not None:
            raise ValueError(
                "Pass either 'contig' or 'length', not both: a contig already encodes "
                "its designed-segment lengths, while 'length' is for unconditional "
                "design with no contig."
            )
        return self

    @model_validator(mode="after")
    def validate_composition_spec(self) -> Any:
        """Mirror rfd3's own composition rules so a bad spec fails here, not after GPU dispatch."""
        # Upstream ``DesignInputSpecification.validate_input_schema`` (rfd3
        # ``inference/input_parsing.py``) enforces the four rules below; each otherwise
        # surfaces only inside the subprocess, after weights load and a real dispatch.

        # Rule 1: something must establish what to build. Every other typed field
        # (symmetry, ligand, partial_t, plddt_enhanced, ...) only modifies a composition.
        if self.input_structure is None and self.contig is None and self.length is None:
            raise ValueError(
                "Provide at least one of 'input_structure', 'contig', or 'length': the other "
                "fields modify a composition but cannot define one."
            )

        # Rule 2: an input structure must be consumed by the composition itself. Only these
        # four fields place input atoms into the design; every select_* field, hotspots
        # included, merely annotates atoms that one of the four already placed -- naming
        # hotspots on a target says nothing about how the design composes with that target.
        consumers = ("contig", "unindex", "ligand", "partial_t")
        if self.input_structure is not None and all(getattr(self, f) is None for f in consumers):
            raise ValueError(
                "'input_structure' is unused: pass 'contig', 'unindex', 'ligand', or 'partial_t' "
                "to state how it enters the design (e.g. contig 'A1-100/0,80-120' for a binder)."
            )

        if self.partial_t is None:
            # Rule 3: unindexed motifs float inside a design of known extent.
            if self.unindex is not None and self.contig is None and self.length is None:
                raise ValueError("'unindex' requires 'contig' or 'length' to size the design it is placed into.")
        elif self.length is not None:
            # Rule 4: partial diffusion re-noises the input, so its extent is already fixed.
            # Upstream's companion check (partial diffusion requires an input) needs no mirror:
            # rule 1 and ``validate_selections_require_input_structure`` already reject every
            # input-free spec that could reach it.
            raise ValueError(
                "'length' must not be combined with 'partial_t': partial diffusion re-noises "
                "'input_structure', so the design's extent comes from the input."
            )
        if self.partial_t is not None and self.select_fixed_atoms in (None, True):
            # Rule 5: upstream fixes every input atom by default, leaving nothing to re-noise.
            raise ValueError(
                "'partial_t' requires 'select_fixed_atoms' (the atoms to hold, e.g. a target chain, "
                "or False to re-noise everything): with every input atom fixed, the input is returned unchanged."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert design spec to RFdiffusion3 JSON format.

        Returns a dict in RFdiffusion3's InputSpecification format. ``input_structure`` is
        emitted by ``to_json_spec`` (as a file path), not here.
        See https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
        """
        spec: dict[str, Any] = {}

        # input_structure is emitted by to_json_spec as a file path, not here.
        # Emit only the fields that are set, so RFdiffusion3 keeps its defaults for the rest
        # Copy the design topology: contig, length, ligand and unindexed residues
        if self.contig is not None:
            spec["contig"] = self.contig
        if self.length is not None:
            spec["length"] = self.length
        if self.ligand is not None:
            spec["ligand"] = self.ligand
        if self.unindex is not None:
            spec["unindex"] = self.unindex
        # Copy the fixed-atom, sequence, hotspot, symmetry, burial and H-bond conditioning
        if self.select_fixed_atoms is not None:
            spec["select_fixed_atoms"] = self.select_fixed_atoms
        if self.select_unfixed_sequence is not None:
            spec["select_unfixed_sequence"] = self.select_unfixed_sequence
        if self.select_hotspots is not None:
            spec["select_hotspots"] = self.select_hotspots
        if self.symmetry is not None:
            spec["symmetry"] = self.symmetry
        if self.select_buried is not None:
            spec["select_buried"] = self.select_buried
        if self.select_partially_buried is not None:
            spec["select_partially_buried"] = self.select_partially_buried
        if self.select_exposed is not None:
            spec["select_exposed"] = self.select_exposed
        if self.select_hbond_donor is not None:
            spec["select_hbond_donor"] = self.select_hbond_donor
        if self.select_hbond_acceptor is not None:
            spec["select_hbond_acceptor"] = self.select_hbond_acceptor
        # Copy the remaining design options
        if self.redesign_motif_sidechains is not None:
            spec["redesign_motif_sidechains"] = self.redesign_motif_sidechains
        if self.plddt_enhanced is not None:
            spec["plddt_enhanced"] = self.plddt_enhanced
        if self.infer_ori_strategy is not None:
            spec["infer_ori_strategy"] = self.infer_ori_strategy
        if self.ori_token is not None:
            spec["ori_token"] = self.ori_token
        if self.partial_t is not None:
            spec["partial_t"] = self.partial_t
        if self.is_non_loopy is not None:
            spec["is_non_loopy"] = self.is_non_loopy

        return spec


class RFdiffusion3Input(BaseToolInput):
    """Input for RFdiffusion3 structure design.

    Specify one or more RFdiffusion3 design tasks using structured Pydantic models
    for common use cases or raw JSON for options beyond those models.

    Attributes:
        design_specs (list[RFdiffusion3DesignSpec]): List of design specifications. Each
            spec represents an independent design task with its own constraints.
            Multiple specs will be processed in a single run.

        raw_json (str | None): Raw JSON string for advanced users who need
            full RFdiffusion3 flexibility. If provided, ``design_specs`` will be ignored
            and this JSON will be passed directly to RFdiffusion3. An ``"input"`` file with
            atoms that have no coordinates is staged as a copy without them.

    Examples:
        >>> # Simple unconditional design
        >>> inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="100-150")])
        >>>
        >>> # Motif scaffolding
        >>> inputs = RFdiffusion3Input(
        ...     design_specs=[
        ...         RFdiffusion3DesignSpec(
        ...             input_structure="path/to/motif.pdb",
        ...             contig="50-80,A10-25,30-50",
        ...         )
        ...     ]
        ... )
        >>>
        >>> # Raw JSON for advanced use
        >>> inputs = RFdiffusion3Input(raw_json='{"spec-1": {"length": "100", "select_buried": "ALA"}}')
    """

    design_specs: list[RFdiffusion3DesignSpec] = InputField(
        default_factory=list,
        title="Design Specs",
        description="List of design specifications",
    )
    raw_json: str | None = InputField(
        default=None,
        title="Raw JSON",
        description="Raw JSON string for advanced RFdiffusion3 configuration",
    )

    @model_validator(mode="after")
    def validate_has_input(self) -> Any:
        """Validate that either design_specs or raw_json is provided."""
        if not self.design_specs and not self.raw_json:
            raise ValueError("Either 'design_specs' (non-empty) or 'raw_json' must be provided")
        return self

    def to_json_spec(self, input_dir: Path | None = None) -> str:
        """Convert input to RFdiffusion3 JSON spec, materializing any ``input_structure`` to a file path."""
        dest: Path | None = None  # created only when a structure must be materialized

        def staging_dir() -> Path:
            nonlocal dest
            if dest is None:
                dest = Path(input_dir) if input_dir is not None else Path(tempfile.mkdtemp(prefix="rfd3_input_"))
                dest.mkdir(parents=True, exist_ok=True)
            return dest

        if self.raw_json:
            raw_spec = json.loads(self.raw_json)
            for i, (key, item) in enumerate(raw_spec.items() if isinstance(raw_spec, dict) else []):
                if isinstance(item, dict) and isinstance(item.get("input"), str):
                    item["input"] = _stage_raw_json_input(item["input"], str(key), f"raw-{i}", staging_dir)
            return json.dumps(raw_spec, indent=2)

        spec_dict: dict[str, Any] = {}
        for i, spec in enumerate(self.design_specs):
            spec_key = f"spec-{i}"
            spec_dict[spec_key] = spec.to_dict()
            if spec.input_structure is not None:
                # rfd3 needs a path: write the structure to a file, set "input" to it.
                ext = spec.input_structure.structure_format or "pdb"
                structure_path = staging_dir() / f"{spec_key}_input.{ext}"
                structure_path.write_text(_drop_unresolved_atoms(spec.input_structure.structure, ext, spec_key))
                spec_dict[spec_key]["input"] = str(structure_path)

        return json.dumps(spec_dict, indent=2)

    def __len__(self) -> int:
        """Get the number of design specs."""
        if self.raw_json:
            try:
                parsed = json.loads(self.raw_json)
                return len(parsed)
            except json.JSONDecodeError:
                return 1
        return len(self.design_specs)


# Config Data Models


class RFdiffusion3SamplerTuning(BaseModel):
    """Finer rfd3 ``inference_sampler`` settings forwarded as Hydra overrides.

    The settings below are the commonly tuned sampler parameters; each is validated,
    defaults to ``None``, and is forwarded to ``inference_sampler.<name>`` only
    when set, otherwise inheriting the checkpoint's upstream default. Any other
    valid ``inference_sampler`` field (e.g. ``s_min`` or ``s_max``)
    may be passed as an extra keyword and is forwarded verbatim. See
    https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md

    Attributes:
        noise_scale (float | None): Diffusion noise scale (upstream default 1.003).
        p (int | None): Noise-schedule exponent (upstream default 7).
        gamma_min (float | None): Lower stochasticity bound, paired with ``gamma_0`` (upstream default 1.0).
        s_trans (float | None): Translational noise scale for inference augmentation (upstream default 1.0).
        allow_realignment (bool | None): Allow motif realignment during sampling (upstream default False).
        s_jitter_origin (float | None): Gaussian sigma jittering the motif offset (upstream default 0.0).
    """

    # extra="allow" forwards niche inference_sampler settings not typed above, verbatim.
    model_config = ConfigDict(extra="allow")

    noise_scale: float | None = Field(
        default=None,
        ge=0.0,
        title="Noise Scale",
        description="Diffusion noise scale (upstream default 1.003); higher = noisier sampling",
    )
    p: int | None = Field(
        default=None,
        ge=1,
        title="Schedule Exponent",
        description="Noise-schedule exponent p (upstream default 7)",
    )
    gamma_min: float | None = Field(
        default=None,
        ge=0.0,
        title="Gamma Min",
        description="Lower stochasticity bound paired with gamma_0 (upstream default 1.0)",
    )
    s_trans: float | None = Field(
        default=None,
        ge=0.0,
        title="Translational Noise",
        description="Translational noise scale for inference augmentation (upstream default 1.0)",
    )
    allow_realignment: bool | None = Field(
        default=None,
        title="Allow Realignment",
        description="Allow motif realignment during sampling (upstream default False)",
    )
    s_jitter_origin: float | None = Field(
        default=None,
        ge=0.0,
        title="Origin Jitter Sigma",
        description="Gaussian sigma jittering the motif offset (upstream default 0.0)",
    )

    def to_overrides(self) -> dict[str, Any]:
        """Return set settings as ``inference_sampler.<key>`` Hydra overrides (extras included)."""
        # Untyped extras bypass validation; surface them so a typo is visible, not silent.
        if self.model_extra:
            logger.warning(
                "RFdiffusion3 sampler_tuning is forwarding unvalidated inference_sampler override(s) %s; "
                "a misspelled key will fail at rfd3 runtime. Typed settings: %s",
                sorted(self.model_extra),
                sorted(type(self).model_fields),
            )
        return {f"inference_sampler.{key}": value for key, value in self.model_dump(exclude_none=True).items()}


class RFdiffusion3Config(BaseConfig):
    """Configuration for RFdiffusion3 structure design.

    Choose the number of designs, diffusion parameters, and execution settings.

    Attributes:
        n_batches (int): Independent batches per spec
            (total designs = ``n_batches * diffusion_batch_size * num_specs``).
        diffusion_batch_size (int): Designs sampled in parallel per batch.
        num_timesteps (int): Diffusion timesteps; at least two are required to
            define one denoising transition, and more are slower but generally
            higher quality.
        step_scale (float): Step size scale; higher = less diverse, more designable.
        sampler_kind (Literal['default', 'symmetry']): Sampler kind;
            ``'symmetry'`` for homo-oligomer design (paired with ``DesignSpec.symmetry``).
        center_option (Literal['all', 'motif', 'diffuse']): Coordinate-frame
            centering — ``all`` (whole structure), ``motif`` (input motif),
            ``diffuse`` (diffused region only).
        use_classifier_free_guidance (bool): Enable CFG sampling.
        cfg_scale (float): CFG scale factor (typical 1.0-3.0); no-op when CFG is off.
        cfg_t_max (float | None): Noise level below which CFG stops being applied.
            Compared against the sampler's noise schedule, not a normalized time:
            with upstream defaults that schedule runs from about 2560 down to
            0.0064, so a small value such as ``0.5`` leaves CFG on for nearly the
            whole rollout. ``None`` (the default) applies CFG at every step.
        gamma_0 (float): Sampler stochasticity; lower = more designable,
            less diverse; ``0.0`` = deterministic ODE. Must be ``> 0.5``
            when ``sampler_kind="symmetry"``.
        sampler_tuning (RFdiffusion3SamplerTuning): Finer ``inference_sampler``
            settings (noise schedule, motif noise); see that class for the fields.
        low_memory_mode (bool): Memory-efficient tokenization (slower);
            enable only if GPU RAM is tight.
        dump_trajectories (bool): Save diffusion trajectory frames (debugging).
        align_trajectory_structures (bool): Align trajectory frames across
            timesteps (only when ``dump_trajectories=True``).
        prevalidate_inputs (bool): Fail-fast input JSON validation.
        ckpt_path (str): Checkpoint path or alias (``"rfd3"`` = production preset).
        input_dir (str | None): Local-execution input directory; ``None`` uses a tempdir.
        output_dir (str | None): Local-execution output directory; ``None`` uses a tempdir.
        device (str): ``"cuda"`` or ``"cpu"``.

    Note:
        Headline sampler controls (``sampler_kind``, ``cfg_scale``, ``cfg_t_max``,
        ``gamma_0``, ``center_option``, ``use_classifier_free_guidance``) are top-level fields.
        Finer ``inference_sampler`` settings live under ``sampler_tuning`` (a typed
        :class:`RFdiffusion3SamplerTuning`), e.g.
        ``RFdiffusion3Config(sampler_tuning={"noise_scale": 1.003})``. See
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
    """

    n_batches: int = ConfigField(
        title="Number of Batches",
        default=1,
        ge=1,
        description="Independent batches per spec (total designs per spec = n_batches * diffusion_batch_size)",
    )
    diffusion_batch_size: int = ConfigField(
        title="Diffusion Batch Size",
        default=8,
        ge=1,
        description="Designs sampled in parallel per batch",
    )
    num_timesteps: int = ConfigField(
        title="Number of Timesteps",
        default=200,
        ge=2,
        description="Diffusion steps (minimum 2; more is slower and generally higher quality)",
    )
    step_scale: float = ConfigField(
        title="Step Scale",
        default=1.5,
        ge=0.1,
        description="Diffusion step size scaling; higher = less diverse, more designable (typical: 1.0-2.0)",
    )
    sampler_kind: Literal["default", "symmetry"] = ConfigField(
        title="Sampler Kind",
        default="default",
        description="Inference sampler kind ('symmetry' for homo-oligomer design)",
        reload_on_change=True,
    )
    center_option: Literal["all", "motif", "diffuse"] = ConfigField(
        title="Center Option",
        default="all",
        description="Coordinate-frame centering mode (all/motif/diffuse)",
    )
    use_classifier_free_guidance: bool = ConfigField(
        title="Use Classifier-Free Guidance",
        default=False,
        description="Enable CFG sampling (cfg_scale is a no-op when False)",
    )
    cfg_scale: float = ConfigField(
        title="CFG Scale",
        default=1.5,
        ge=0.0,
        description="CFG scale (typical: 1.0-3.0); requires use_classifier_free_guidance=True",
    )
    cfg_t_max: float | None = ConfigField(
        title="CFG Noise Level Floor",
        default=None,
        ge=0.0,
        description="Noise level below which CFG stops being applied; None applies it throughout",
    )
    gamma_0: float = ConfigField(
        title="Gamma 0",
        default=0.6,
        ge=0.0,
        description="Sampler stochasticity (lower = more designable, less diverse); >0.5 for symmetry sampler",
    )
    sampler_tuning: RFdiffusion3SamplerTuning = ConfigField(
        default_factory=RFdiffusion3SamplerTuning,
        title="Sampler Tuning",
        description="Finer inference_sampler settings (noise schedule, motif noise); see RFdiffusion3SamplerTuning",
    )
    low_memory_mode: bool = ConfigField(
        title="Low Memory Mode",
        default=False,
        description="Memory-efficient tokenization (slower); set True only if GPU RAM is tight",
    )
    dump_trajectories: bool = ConfigField(
        title="Dump Trajectories",
        default=False,
        description="Save diffusion trajectory frames to the output directory",
        include_in_key=False,
    )
    align_trajectory_structures: bool = ConfigField(
        title="Align Trajectory Structures",
        default=False,
        description="Align trajectory structures across timesteps",
        include_in_key=False,
    )
    prevalidate_inputs: bool = ConfigField(
        title="Prevalidate Inputs",
        default=False,
        description="Validate the full input JSON before launching diffusion",
        include_in_key=False,
    )
    ckpt_path: str = ConfigField(
        title="Checkpoint Path",
        default="rfd3",
        description="Checkpoint path or canonical alias (default: rfd3 production preset)",
    )
    input_dir: str | None = ConfigField(
        title="Input Directory",
        default=None,
        description="Optional input directory for local execution inputs",
        include_in_key=False,
    )
    output_dir: str | None = ConfigField(
        title="Output Directory",
        default=None,
        description="Optional output directory for local execution outputs",
        include_in_key=False,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        include_in_key=False,
    )

    @model_validator(mode="after")
    def validate_gamma_0_with_symmetry_sampler(self) -> Any:
        """Upstream rfd3 invariant: ``gamma_0 > 0.5`` is required when ``sampler_kind == "symmetry"``."""
        if self.sampler_kind == "symmetry" and self.gamma_0 <= 0.5:
            raise ValueError(f"gamma_0 must be > 0.5 when sampler_kind='symmetry'; got {self.gamma_0}")
        return self

    def get_cli_kwargs(self) -> dict[str, Any]:
        """Build CLI args for the rfd3 inference script.

        Sampler settings use dotted Hydra paths (``inference_sampler.<key>``). On
        collision, the headline top-level fields win over ``sampler_tuning``.
        """
        cli_kwargs: dict[str, Any] = {}
        # sampler_tuning first so the headline fields below win on collision.
        cli_kwargs.update(self.sampler_tuning.to_overrides())
        cli_kwargs.update(
            {
                "n_batches": self.n_batches,
                "diffusion_batch_size": self.diffusion_batch_size,
                "num_timesteps": self.num_timesteps,
                "step_scale": self.step_scale,
                "low_memory_mode": self.low_memory_mode,
                "dump_trajectories": self.dump_trajectories,
                "align_trajectory_structures": self.align_trajectory_structures,
                "prevalidate_inputs": self.prevalidate_inputs,
                "ckpt_path": self.ckpt_path,
                "inference_sampler.kind": self.sampler_kind,
                "inference_sampler.center_option": self.center_option,
                "inference_sampler.use_classifier_free_guidance": self.use_classifier_free_guidance,
                "inference_sampler.cfg_scale": self.cfg_scale,
                "inference_sampler.gamma_0": self.gamma_0,
            }
        )
        if self.cfg_t_max is not None:
            cli_kwargs["inference_sampler.cfg_t_max"] = self.cfg_t_max
        return cli_kwargs


# Output Data Models


class RFdiffusion3Structure(BaseModel):
    """A single designed structure from RFdiffusion3.

    One per generated design. Multiple ``RFdiffusion3Structure``s are bundled
    inside an ``RFdiffusion3Designs`` (one bundle per input spec). The
    bundle's position in ``RFdiffusion3Output.designed_structures`` matches
    the position of the originating spec in ``RFdiffusion3Input.design_specs``.

    Attributes:
        structure (Structure): The designed 3D structure (PDB/CIF coordinates).
        complex (Complex): Predictor-ready chains + per-chain sequences; feed
            directly to a structure predictor.
        metadata (dict[str, Any]): Additional RFdiffusion3 output metadata.
    """

    structure: Structure = Field(
        title="Designed Structure",
        description="The designed 3D structure",
    )
    complex: Complex = Field(
        title="Designed Complex",
        description="Predictor-ready chains and per-chain sequences.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        title="Design Metadata",
        description="Additional metadata from RFdiffusion3 output",
    )


class RFdiffusion3Designs(BaseModel):
    """All designs produced for a single input spec.

    Bundling N designs into one wrapper preserves the framework's 1:1
    cardinality between ``iterable_input_fields`` and ``iterable_output_field``
    even though RFdiffusion3 fans out (N = ``n_batches * diffusion_batch_size``
    designs per spec). Mirrors the per-input bundling pattern used by
    ``DesignSet``.

    Attributes:
        spec_key (str): Identifier of the input specification that produced
            these designs (e.g. ``"spec-0"``). For ``raw_json`` callers this
            is the user-supplied key; otherwise it is positional
            (``f"spec-{i}"`` from ``RFdiffusion3Input.to_json_spec``).

        structures (list[RFdiffusion3Structure]): The designs generated for
            this spec, in the order RFdiffusion3 emitted them. List length
            equals ``n_batches * diffusion_batch_size``.
    """

    spec_key: str = Field(
        title="Spec Key",
        description="Identifier of the input spec that produced these designs",
    )
    structures: list[RFdiffusion3Structure] = Field(
        default_factory=list,
        title="Designs",
        description="Designs generated for this spec",
    )

    def __len__(self) -> int:
        """Get the number of designs for this spec."""
        return len(self.structures)

    def __getitem__(self, index: int) -> RFdiffusion3Structure:
        """Get one designed structure within this bundle by index."""
        return self.structures[index]

    def __iter__(self) -> Iterator[RFdiffusion3Structure]:  # type: ignore[override]
        """Iterate over the designs in this bundle."""
        return iter(self.structures)


class RFdiffusion3Output(BaseToolOutput):
    """Output from RFdiffusion3 structure design.

    Designs are grouped into ``RFdiffusion3Designs`` bundles — one per input
    spec — so the field cardinality matches ``RFdiffusion3Input.design_specs``
    1:1, consistent with the framework's iterable-tool contract.

    Attributes:
        designed_structures (list[RFdiffusion3Designs]): One bundle per input
            spec. Total design count is
            ``len(design_specs) * n_batches * diffusion_batch_size``.

    Note:
        This class supports list-like operations that iterate over the
        per-spec bundles: ``output[i]`` returns the i-th bundle,
        ``for designs in output`` walks bundles, ``len(output)`` is the
        number of bundles. To walk every individual design, use
        ``for bundle in output: for s in bundle: ...``.
    """

    designed_structures: list[RFdiffusion3Designs] = Field(
        default_factory=list,
        title="Designed Structures",
        description="Per-spec bundles of designed structures",
    )

    def __len__(self) -> int:
        """Get the number of per-spec design bundles."""
        return len(self.designed_structures)

    def __getitem__(self, index: int) -> RFdiffusion3Designs:
        """Get the design bundle for the i-th input spec."""
        return self.designed_structures[index]

    def __iter__(self) -> Iterator[RFdiffusion3Designs]:  # type: ignore[override]
        """Iterate over per-spec design bundles."""
        return iter(self.designed_structures)

    def __repr__(self) -> str:
        """Get string representation."""
        n_bundles = len(self.designed_structures)
        n_designs = sum(len(b) for b in self.designed_structures)
        return f"RFdiffusion3Output(designed_structures=[{n_bundles} bundles, {n_designs} designs])"

    def __str__(self) -> str:
        """Get human-readable string representation."""
        return self.__repr__()

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["pdb", "cif"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "pdb"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        if file_format not in ["pdb", "cif"]:
            raise ValueError(f"Unsupported format: {file_format}")

        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)
        for bundle in self.designed_structures:
            for i, struct in enumerate(bundle.structures):
                out_file = path / f"{bundle.spec_key}_design_{i}.{file_format}"
                if file_format == "pdb":
                    struct.structure.write_pdb(out_file)
                elif file_format == "cif":
                    struct.structure.write_cif(out_file)


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="100")])


def _canonicalize_design_chain_ids(structure: Structure) -> Structure:
    """Relabel polymer chains to positional IDs (``A``, ``B``, ...) by emission order.

    Symmetric rfd3 outputs suffix chain IDs with the transformation index (e.g.
    ``A1, A2, A3`` for a C3 trimer); this restores a predictable ``A, B, C`` /
    ``select_chain`` contract. No-op when already canonical.
    """
    observed = structure.get_chain_ids()
    target = [chain_label(i) for i in range(len(observed))]
    if observed == target:
        return structure
    return structure.with_renamed_chains(dict(zip(observed, target, strict=True)))


# Spec fields that feed the three upstream ``cfg_features`` tensors: select_hbond_donor and
# select_hbond_acceptor set ``active_donor``/``active_acceptor``, and the three RASA selectors
# set ``rasa_bin``, which becomes ``ref_atomwise_rasa``. select_hotspots is not among them --
# it sets ``is_atom_level_hotspot``, which CFG never strips.
_CFG_FEATURE_FIELDS = (
    "select_buried",
    "select_partially_buried",
    "select_exposed",
    "select_hbond_donor",
    "select_hbond_acceptor",
)

# ``unindex`` carries no CFG feature tensor, but upstream ``strip_f`` crops unindexed-motif atoms
# out of the reference pass regardless of ``cfg_features``, so CFG still guides away from that
# conditioning and is not a no-op for such a spec.
_CFG_CONDITIONING_FIELDS = (*_CFG_FEATURE_FIELDS, "unindex")


def _unguided_cfg_specs(inputs: RFdiffusion3Input) -> list[str]:
    """Return every spec key for which classifier-free guidance would be a no-op."""
    specs: Iterable[tuple[str, Any]]
    if inputs.raw_json:
        payload = json.loads(inputs.raw_json)
        if not isinstance(payload, dict):
            raise ValueError("RFdiffusion3 raw_json must contain a top-level object of named design specs")
        specs = ((str(key), value) for key, value in payload.items())
    else:
        specs = ((f"spec-{index}", spec.to_dict()) for index, spec in enumerate(inputs.design_specs))

    unguided: list[str] = []
    for spec_key, spec in specs:
        # A falsy selector carries no conditioning: ``None`` and empty str/list/dict leave the
        # feature unset, and an explicit ``False`` switches it off, so neither enables CFG.
        if not isinstance(spec, dict) or not any(spec.get(field) for field in _CFG_CONDITIONING_FIELDS):
            unguided.append(str(spec_key))
    return unguided


@tool(
    key="rfdiffusion3-design",
    label="RFdiffusion3 Structure Design",
    category="structure_design",
    input_class=RFdiffusion3Input,
    config_class=RFdiffusion3Config,
    output_class=RFdiffusion3Output,
    description="De novo protein structure design using RFdiffusion3",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["design_specs"],
    iterable_output_field="designed_structures",
    max_chunk_size=1,
    cacheable=True,
    stochastic=True,
)
def run_rfdiffusion3(inputs: RFdiffusion3Input, config: RFdiffusion3Config, instance: Any = None) -> RFdiffusion3Output:
    """Design protein structures using RFdiffusion3.

    RFdiffusion3 generates protein structures and sequences under the supplied
    constraints. It supports the following design tasks.

    - De novo protein design (unconditional generation)
    - Motif scaffolding (design around fixed structural motifs)
    - Protein binder design (design proteins that bind to targets)
    - Enzyme design (scaffold around catalytic sites)
    - Symmetric protein design (design homo-oligomers)

    Run the model on a local GPU in an isolated Python environment.

    Args:
        inputs (RFdiffusion3Input): Validated input containing one or more design
            specifications. Each spec defines constraints for a design task.
        config (RFdiffusion3Config): Validated configuration specifying diffusion
            parameters and execution options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        RFdiffusion3Output: Structured output containing:
            - ``designed_structures``: One ``RFdiffusion3Designs`` bundle per
              input spec, in input order. Each bundle has a ``spec_key`` and
              a list of ``RFdiffusion3Structure``s (one per generated design).
            - Each ``RFdiffusion3Structure`` includes:
                - 3D coordinates (as Structure)
                - Predictor-ready complex (as Complex)
                - Metadata (sampled contig, chain info, etc.)

    See Also:
        - RFdiffusion3 GitHub: https://github.com/RosettaCommons/foundry
        - RFdiffusion3 paper: https://doi.org/10.1101/2025.09.18.676967

    Example:
        >>> # Unconditional design of 100-residue protein, 4 designs in one batch
        >>> inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="100")])
        >>> config = RFdiffusion3Config(diffusion_batch_size=4, num_timesteps=200, verbose=True)
        >>> result = run_rfdiffusion3(inputs, config)
        >>> bundle = result[0]  # one bundle per input spec
        >>> print(f"Spec {bundle.spec_key}: {len(bundle)} designs")
        >>> print(f"First sequence (chain A): {bundle[0].complex.chains[0].sequence}")

    Note:
        - Each design includes a structure and sequence.
        - Memory usage scales with batch size and protein length
        - Dependency isolation is handled automatically via venv subprocess
    """
    if config.use_classifier_free_guidance:
        unguided_cfg_specs = _unguided_cfg_specs(inputs)
        if unguided_cfg_specs:
            logger.warning(
                "use_classifier_free_guidance=True, but specs %s set no native CFG feature "
                "(select_buried/select_partially_buried/select_exposed/select_hbond_*) and no "
                "'unindex'. Hotspot controls are distinct from CFG features, so guidance cancels "
                "to an identical prediction for these specs while doubling the forward passes. "
                "Set a feature selector, or disable CFG for unconditional and hotspot-only generation.",
                unguided_cfg_specs,
            )

    logger.debug("Using local GPU for RFdiffusion3 structure design...")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_dir = Path(config.input_dir) if config.input_dir else temp_dir
        output_dir = Path(config.output_dir) if config.output_dir else temp_dir / "rfdiffusion3_output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Inside the tempdir so input_structure files land in input_dir and outlive the dispatch.
        json_spec = inputs.to_json_spec(input_dir=input_dir)
        input_json_path = input_dir / "rfdiffusion3_input.json"

        with open(input_json_path, "w") as f:
            f.write(json_spec)

        input_data = {
            "operation": "design",
            "input_json_path": str(input_json_path),
            "output_dir": str(output_dir),
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
            **config.get_cli_kwargs(),
        }
        output_data = ToolInstance.dispatch(
            "rfdiffusion3",
            input_data,
            instance=instance,
            config=config,
        )

    # Group designs by originating spec so the output has 1:1 cardinality with
    # inputs.design_specs. Subprocess emits a flat list; the spec-key prefix on
    # each design (set in to_json_spec) lets us bucket them.
    buckets: dict[str, list[RFdiffusion3Structure]] = defaultdict(list)
    for design_data in output_data.get("designs", []):
        structure = Structure(
            structure=design_data["structure_content"],
            source="rfdiffusion3-design",
        )
        structure = _canonicalize_design_chain_ids(structure)
        buckets[design_data["spec_key"]].append(
            RFdiffusion3Structure(
                structure=structure,
                complex=Complex.from_structure(structure),
                metadata=design_data.get("metadata", {}),
            )
        )

    # One bundle per spec sent (json_spec keys), in input order, backfilling
    # empties so a zero-design spec can't shift later bundles out of alignment.
    sent_spec_keys = list(json.loads(json_spec).keys())
    designed_structures = [RFdiffusion3Designs(spec_key=key, structures=buckets.get(key, [])) for key in sent_spec_keys]
    return RFdiffusion3Output(designed_structures=designed_structures)

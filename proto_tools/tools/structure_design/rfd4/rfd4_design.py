"""proto_tools/tools/structure_design/rfd4/rfd4_design.py.

Example:
    >>> from proto_tools.entities.structures import Structure
    >>> from proto_tools.tools.structure_design.rfd4 import (
    ...     run_rfd4_design,
    ...     RFD4Input,
    ...     RFD4DesignConfig,
    ...     RFD4DesignSpec,
    ... )
    >>> inputs = RFD4Input(
    ...     design_specs=[
    ...         RFD4DesignSpec(
    ...             input_structure=Structure.from_file("target.pdb"),
    ...             contig="50-120, /0, A17-131",
    ...             hotspots=["sel('A/*/56/[CG,OH]')"],
    ...         )
    ...     ]
    ... )
    >>> result = run_rfd4_design(inputs, RFD4DesignConfig(num_replicates=4))
    >>> print(f"{sum(len(b) for b in result)} designs across {len(result)} specs")
"""

import json
import logging
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proto_tools.entities.complex import Complex
from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_design.rfd4.helpers import (
    CHAINBREAK,
    MASKED_SEGMENT_NAME,
    contig_segment_names,
    prepare_rfd4_cif,
    translate_contig,
    translate_selection,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.device import RemoteDevice

logger = logging.getLogger(__name__)

EXAMPLE_INPUT_FIXTURE = Path(__file__).parent / "example_input_fixture.cif"

# Pinned release bundle; see configs/inference.yaml in the upstream repo.
DEFAULT_MODEL_CHECKPOINT = "nvidia/RFD4-Proteina"
DEFAULT_MODEL_REVISION = "a06e2509c7033b010fc43614e9fd840303588502"

MAX_COORDINATE_GROUP_ID = 32767

# Every recycle_strategy preset's bands sum to exactly this, and upstream asserts the match.
RECYCLE_STRATEGY_NSTEPS = 60

RasaBin = Literal["BURIED", "INTERMEDIATE", "EXPOSED", "UNSPECIFIED"]
HBondRole = Literal["DONOR", "ACCEPTOR", "BOTH", "NONE"]
PiCationRole = Literal["AROMATIC", "CATION", "BOTH", "NONE"]
SecondaryStructureState = Literal["HELIX", "SHEET", "LOOP", "NONE"]

# The 28 canonical polymer tokens carrying a C_SEQ_PROB_<TOKEN> condition: 20 AA, 4 RNA, 4 DNA.
SequenceToken = Literal[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "A", "C", "G", "U", "DA", "DC", "DG", "DT",
]  # fmt: skip

RecycleStrategy = Literal[
    "low/flat", "low/front_loaded", "low/midpeak",
    "medium/flat", "medium/front_loaded", "medium/midpeak",
    "high/flat", "high/front_loaded", "high/midpeak",
    "thinking_low", "thinking_medium", "thinking_high",
]  # fmt: skip

# Rejected upstream, or 2-body with no single-selection grammar; guarded so a typo fails here.
_REJECTED_CONDITION_ALIASES = {
    "C_INTCRD": "a saved payload; set internal-coordinate conditioning through 'rigid_groups' instead",
    "C_BP": "a 2-body (base-pair) condition with no selection-pair grammar in the spec format",
    "C_DIS": "a 2-body (distance) condition with no selection-pair grammar in the spec format",
    "C_CHA": "a 2-body (chain) condition with no selection-pair grammar in the spec format",
}

# Removed pre-RFD4 dialect fields: naming one makes upstream reject the file, so name the fix.
_LEGACY_SPEC_FIELDS = {
    "length": "RFD4 sizes designs through 'contig' expandable segments (e.g. '80-120')",
    "ligand": 'select ligands with a condition query such as "is_ligand"',
    "symmetry": "not available in RFD4; use rfdiffusion3-design for symmetric oligomers",
    "unindex": "use 'unindexed_groups'",
    "select_hotspots": "use 'hotspots'",
    "select_fixed_atoms": "use 'fixed_coordinates'",
    "select_unfixed_sequence": "use 'free_sequence'",
    "select_buried": "use 'burial' with the BURIED bin",
    "select_exposed": "use 'burial' with the EXPOSED bin",
    "select_partially_buried": "use 'burial' with the INTERMEDIATE bin",
    "select_hbond_donor": "use 'hbond' with the DONOR role",
    "select_hbond_acceptor": "use 'hbond' with the ACCEPTOR role",
    "is_non_loopy": "use 'secondary_structure'",
    "plddt_enhanced": "not an RFD4 parameter",
    "infer_ori_strategy": "use 'chain_centroid'",
    "redesign_motif_sidechains": "condition the motif backbone only, e.g. fixed_coordinates with 'is_protein_backbone'",
}


# ============================================================================
# Data Models
# ============================================================================
# Input Data Models


class RFD4SequenceProbability(BaseModel):
    """One sequence-probability constraint on a set of residues.

    Attributes:
        token (SequenceToken): Residue identity the probability applies to, as a
            canonical three-letter amino acid or one/two-letter nucleotide code.
        probability (float): Target probability in ``[0, 1]``. ``0.0`` actively
            excludes the identity; an omitted identity stays flexible and shares the
            remaining probability mass.
        select (list[str]): Selection queries naming the residues to constrain.
    """

    model_config = ConfigDict(extra="forbid")

    token: SequenceToken = Field(
        title="Residue Token",
        description="Residue identity the probability applies to (e.g. 'CYS', 'U', 'DA')",
    )
    probability: float = Field(
        title="Probability",
        ge=0.0,
        le=1.0,
        description="Target probability in [0, 1]; 0.0 excludes the identity outright",
    )
    select: list[str] = Field(
        min_length=1,
        title="Selections",
        description="Selection queries naming the residues to constrain",
    )


class RFD4DesignSpec(BaseModel):
    r"""Single design specification for RFD4-Proteina.

    One spec is one design task. The typed condition fields below compile into the
    upstream ``conditions`` mapping; see
    https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/reference/yaml-json.md

    Every ``select``-style value is an atomworks query string, in either path syntax
    (``sel('<chain>/<res_name>/<res_id>/<atom_name>')``, trailing fields default to
    ``*``) or boolean query syntax over annotations (``is_ligand``,
    ``is_protein_backbone``, ``chain_id == 'A'``). Note that path fields are
    positional: ``sel('B/*/N')`` reads ``N`` as a residue id and fails — write
    ``sel('B/*/*/N')`` to reach an atom name.

    Attributes:
        input_structure (Structure): Starting structure. RFD4 requires one even for
            unconditional generation, and consumes bonds and formal charges, so the
            implementation always hands it a CIF (PDB input is converted automatically).

        input_as_annotated_cif (bool): When ``True``, load ``input_structure``
            as-is and preserve condition annotations it already carries (e.g. a prior
            RFD4 output, for refinement or partial diffusion). When ``False`` (default),
            treat it as a raw structure: add heavy atoms, convert MSE to MET, and ignore
            any existing conditioning.

        contig (str | None): Design topology as one comma-separated token string, e.g.
            ``"50-120, /0, A17-131"`` (a de novo 50-120 residue binder chain, then the
            whole target). Tokens: ``/0`` chainbreak; ``A42-48`` preserved input segment;
            ``A52`` a single preserved residue; ``20-30`` an unnamed expandable segment;
            ``E1:2-20`` a named expandable segment (the name becomes its ``res_name`` so
            conditions can select it). An expandable segment may carry a trailing
            ``P``/``R``/``D`` chain type, so ``10R`` is a ten-residue RNA segment rather
            than ten repeats; it defaults to the run's anchor chain type and is rejected
            when it disagrees with it. Omit to keep the whole input. A contig must contain
            at least one preserved segment, since there is no ``length`` field.
            The token forms match RFdiffusion3's, but the semantics do not: residue ids
            resolve against ``label_seq_id`` and a preserved segment is mandatory.
            RFdiffusion1's slash-separated contigs are an unrelated grammar.

        residue_numbering (Literal['auth', 'label']): How residue ids in ``contig``
            and selections are read. ``'auth'`` (default) is the author numbering shown
            in PDB files and used elsewhere in proto-tools; the implementation translates it to
            the mmCIF ``label_seq_id`` RFD4 resolves against. ``'label'`` passes ids
            through untouched, for annotated-CIF inputs already in label numbering.

        fixed_coordinates (list[str] | None): Selections whose real coordinates are
            given to the model (indexed motif scaffolding). Pair with
            ``is_protein_backbone`` to fix a backbone without revealing its sequence.

        free_coordinates (list[str] | None): Selections whose coordinate conditioning
            is cleared. Needed before moving atoms into a ``rigid_groups`` entry.

        known_sequence (list[str] | None): Selections whose residue identity is
            revealed to the model (folding, or a known target chain).

        free_sequence (list[str] | None): Selections whose residue identity is
            designed rather than revealed.

        hotspots (list[str] | None): Target-interface atoms the designed binder should
            engage. Trained as target atoms within 4.5 Å of any binder atom, so select
            the specific sidechain atoms to contact.

        rigid_groups (dict[int, list[str]] | None): Internal-coordinate groups keyed by
            group id (``1``-``32767``). Atoms in one group have known relative geometry
            but no fixed global pose — the standard way to give a binder target its
            backbone shape while leaving the dock free. Geometry is unknown across groups.

        unindexed_groups (dict[str, list[str]] | None): Unindexed motif groups keyed by
            a label you choose. Selections sharing a label keep their relative offsets
            but the model chooses where they sit in the chain — active-site scaffolding.

        burial (dict[RasaBin, list[str]] | None): Target burial per selection, keyed by
            ``BURIED`` / ``INTERMEDIATE`` / ``EXPOSED`` / ``UNSPECIFIED``.

        hbond (dict[HBondRole, list[str]] | None): Requested hydrogen-bond role, keyed
            by ``DONOR`` / ``ACCEPTOR`` / ``BOTH`` / ``NONE``. ``BOTH`` asks for donation
            and acceptance in different bonds; separate entries do not combine.

        pi_stacking (list[str] | None): Selections requesting aromatic stacking. Select
            every atom of the intended ring.

        pi_cation (dict[PiCationRole, list[str]] | None): Requested π-cation role, keyed
            by ``AROMATIC`` / ``CATION`` / ``BOTH`` / ``NONE``. ``AROMATIC`` describes the
            selected atoms, not the desired partner.

        secondary_structure (dict[SecondaryStructureState, list[str]] | None): Requested
            secondary structure, keyed by ``HELIX`` / ``SHEET`` / ``LOOP`` / ``NONE``.

        fold_class (dict[str, list[str]] | None): CATH class/architecture/topology label
            keyed by a quoted C.A.T string (``"2.60.40"``, ``"1.x.x"``; ``"x.x.x"``
            clears it). Labels are checked against the checkpoint's vocabulary at runtime.
            Select a whole chain or domain.

        chain_centroid (dict[str, list[str]] | None): Target chain centre of mass keyed
            by an ``"x,y,z"`` string in Ångströms. Per chain and all-or-nothing: each
            selection must cover an entire chain, and either every chain gets a centroid
            or none does.

        sequence_probabilities (list[RFD4SequenceProbability] | None): Per-identity
            sequence probability constraints, e.g. ``probability=0.0`` on ``CYS`` to
            request a cysteine-free design. Values specified on one residue may not sum
            above 1.

        raw_conditions (dict[str, Any] | None): Escape hatch merged last into the
            compiled ``conditions`` mapping, for conditions without a typed field above
            (e.g. ``C_SALT``, ``C_HPHOB``, ``C_MCOORD``, ``C_NTR``). Uses the upstream
            shape ``{alias: {value: [{"select": query}]}}``. Residue ids in raw
            conditions are **not** translated — write them in label numbering.

    Note:
        Conditions are not independent: a sidechain coordinate forces sequence
        conditioning, and atomized tokens (ligands, PTMs, non-canonicals) are always
        sequence-conditioned. Overlapping entries for one condition resolve
        last-write-wins. Ligands and modified residues must carry Kekulized bonds
        (``AROMATIC_SINGLE`` / ``AROMATIC_DOUBLE``, never ``AROMATIC``) for their bond
        orders to reach the model.

    Note:
        Symmetric oligomer design is documented as *Planned* upstream and is not
        available here. Use ``rfdiffusion3-design`` for symmetric homo-oligomer design.
    """

    model_config = ConfigDict(extra="forbid")

    input_structure: Structure = Field(
        title="Input Structure",
        description="Starting structure; converted to CIF automatically (RFD4 rejects PDB)",
    )
    input_as_annotated_cif: bool = Field(
        default=False,
        title="Load As Annotated CIF",
        description="Preserve condition annotations already on the input instead of starting fresh",
    )
    contig: str | None = Field(
        default=None,
        title="Contig",
        description="Design topology tokens, e.g. '50-120, /0, A17-131'; omit to keep the whole input",
        examples=["50-120, /0, A17-131", "A1-150, /0, EX1:15-30"],
    )
    residue_numbering: Literal["auth", "label"] = Field(
        default="auth",
        title="Residue Numbering",
        description="Whether contig/selection residue ids are author numbering (default) or mmCIF label_seq_id",
    )
    fixed_coordinates: list[str] | None = Field(
        default=None,
        title="Fixed Coordinates",
        description="Selections whose real coordinates are given to the model (indexed motif)",
        examples=[["sel('B') and is_protein_backbone"]],
    )
    free_coordinates: list[str] | None = Field(
        default=None,
        title="Free Coordinates",
        description="Selections whose coordinate conditioning is cleared before regrouping",
    )
    known_sequence: list[str] | None = Field(
        default=None,
        title="Known Sequence",
        description="Selections whose residue identity is revealed to the model",
        examples=[["sel('A')"]],
    )
    free_sequence: list[str] | None = Field(
        default=None,
        title="Free Sequence",
        description="Selections whose residue identity is designed rather than revealed",
    )
    hotspots: list[str] | None = Field(
        default=None,
        title="Hotspots",
        description="Target-interface atoms the designed binder should engage",
        examples=[["sel('A/*/56/[CG,OH]')"]],
    )
    rigid_groups: dict[int, list[str]] | None = Field(
        default=None,
        title="Rigid Groups",
        description="Internal-coordinate groups by id (1-32767): known relative geometry, free global pose",
        examples=[{1: ["sel('A/*/*/[N,CA,C,O]')"]}],
    )
    unindexed_groups: dict[str, list[str]] | None = Field(
        default=None,
        title="Unindexed Groups",
        description="Unindexed motif groups by label; relative offsets kept, chain position chosen by the model",
        examples=[{"motif": ["sel('A/*/[1-30]')", "sel('A/*/[120-150]')"]}],
    )
    burial: dict[RasaBin, list[str]] | None = Field(
        default=None,
        title="Burial",
        description="Target solvent accessibility per selection (BURIED/INTERMEDIATE/EXPOSED/UNSPECIFIED)",
    )
    hbond: dict[HBondRole, list[str]] | None = Field(
        default=None,
        title="Hydrogen Bond Roles",
        description="Requested H-bond role per selection (DONOR/ACCEPTOR/BOTH/NONE)",
    )
    pi_stacking: list[str] | None = Field(
        default=None,
        title="Pi Stacking",
        description="Selections requesting aromatic stacking; select whole rings",
    )
    pi_cation: dict[PiCationRole, list[str]] | None = Field(
        default=None,
        title="Pi-Cation Roles",
        description="Requested pi-cation role per selection (AROMATIC/CATION/BOTH/NONE)",
    )
    secondary_structure: dict[SecondaryStructureState, list[str]] | None = Field(
        default=None,
        title="Secondary Structure",
        description="Requested secondary structure per selection (HELIX/SHEET/LOOP/NONE)",
        examples=[{"HELIX": ["sel('*/EX1')"]}],
    )
    fold_class: dict[str, list[str]] | None = Field(
        default=None,
        title="CATH Fold Class",
        description="CATH C.A.T label per selection (e.g. '2.60.40', '1.x.x'; 'x.x.x' clears)",
        examples=[{"2.60.40": ["sel('A')"]}],
    )
    chain_centroid: dict[str, list[str]] | None = Field(
        default=None,
        title="Chain Centroid",
        description="Target chain centre of mass as an 'x,y,z' key in Angstroms; must cover whole chains",
        examples=[{"12.3,-4.5,0.0": ["sel('B')"]}],
    )
    sequence_probabilities: list[RFD4SequenceProbability] | None = Field(
        default=None,
        title="Sequence Probabilities",
        description="Per-identity sequence probability constraints (e.g. 0.0 on CYS to exclude cysteine)",
    )
    raw_conditions: dict[str, Any] | None = Field(
        default=None,
        title="Raw Conditions",
        description="Upstream-shaped conditions merged last, for conditions without a typed field",
    )

    @field_validator("contig")
    @classmethod
    def _validate_contig_tokens(cls, value: str | None) -> str | None:
        """Mirror upstream's contig checks that do not need the structure to evaluate."""
        if value is None:
            return value
        if "\\0" in value:
            raise ValueError("contig contains '\\0'; use '/0' for RFD4 chain breaks")
        segments = [segment.strip() for segment in value.split(",")]
        if any(not segment for segment in segments):
            raise ValueError(f"contig {value!r} has an empty token (check for stray or trailing commas)")
        if not any(seg != CHAINBREAK and ":" not in seg and seg[0].isalpha() for seg in segments):
            raise ValueError(
                f"contig {value!r} has no preserved segment (e.g. 'A17-131'). RFD4's input structure "
                "exists to be referenced by a preserved segment, so an all-expandable contig is "
                "rejected upstream. Omit 'contig' to keep the whole input instead."
            )
        named = [segment.split(":", 1)[0] for segment in segments if ":" in segment]
        duplicates = sorted({name for name in named if named.count(name) > 1})
        if duplicates:
            raise ValueError(f"contig {value!r} reuses expandable-segment name(s) {duplicates}; names must be unique")
        if MASKED_SEGMENT_NAME in named:
            raise ValueError(f"contig segment name {MASKED_SEGMENT_NAME!r} is reserved internally by RFD4")
        return value

    @field_validator("rigid_groups")
    @classmethod
    def _validate_rigid_group_ids(cls, value: dict[int, list[str]] | None) -> dict[int, list[str]] | None:
        """Reject the reserved coordinate-group ids (0 absolute, -1 unknown) and out-of-range ids."""
        if value is None:
            return value
        for group_id in value:
            if not 0 < group_id <= MAX_COORDINATE_GROUP_ID:
                raise ValueError(
                    f"rigid_groups id {group_id} is out of range: use 1-{MAX_COORDINATE_GROUP_ID} "
                    "(0 and -1 are reserved for absolute and unknown coordinates)"
                )
        return value

    @field_validator("chain_centroid")
    @classmethod
    def _validate_centroid_keys(cls, value: dict[str, list[str]] | None) -> dict[str, list[str]] | None:
        """Reject centroid keys that are not three comma-separated numbers."""
        if value is None:
            return value
        for key in value:
            parts = key.split(",")
            if len(parts) != 3:
                raise ValueError(f"chain_centroid key {key!r} must be 'x,y,z' with exactly three components")
            try:
                [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"chain_centroid key {key!r} must contain three numbers") from exc
        return value

    @field_validator("fold_class")
    @classmethod
    def _validate_cath_keys(cls, value: dict[str, list[str]] | None) -> dict[str, list[str]] | None:
        """Reject CATH labels that are not C.A.T strings with ``x`` only as a trailing suffix."""
        if value is None:
            return value
        for key in value:
            levels = key.split(".")
            if len(levels) != 3:
                raise ValueError(f"fold_class key {key!r} must be a three-level C.A.T label (e.g. '2.60.40', '1.x.x')")
            seen_x = False
            for level in levels:
                if level == "x":
                    seen_x = True
                elif seen_x:
                    raise ValueError(f"fold_class key {key!r} may only use 'x' as a trailing suffix")
                elif not level.isdigit():
                    raise ValueError(f"fold_class key {key!r} has non-numeric level {level!r}")
        return value

    @field_validator("raw_conditions")
    @classmethod
    def _validate_raw_conditions(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject condition aliases the upstream spec parser refuses, plus legacy-dialect names."""
        if value is None:
            return value
        for alias in value:
            if alias in _REJECTED_CONDITION_ALIASES:
                raise ValueError(f"raw_conditions may not set {alias}: it is {_REJECTED_CONDITION_ALIASES[alias]}")
            if alias in _LEGACY_SPEC_FIELDS:
                raise ValueError(
                    f"{alias!r} is a field of the removed pre-RFD4 contig-JSON dialect, not a condition: "
                    f"{_LEGACY_SPEC_FIELDS[alias]}"
                )
        return value

    @model_validator(mode="after")
    def validate_annotated_cif_is_cif(self) -> "RFD4DesignSpec":
        """Reject ``input_as_annotated_cif`` on a PDB, which cannot carry condition annotations."""
        if self.input_as_annotated_cif and (self.input_structure.structure_format or "pdb") != "cif":
            raise ValueError(
                "'input_as_annotated_cif' preserves the condition annotations stored in mmCIF "
                "columns, so the input must be a CIF; PDB cannot represent them. Pass the CIF "
                "written by a previous RFD4 run, or leave the flag unset to start fresh."
            )
        return self

    @model_validator(mode="after")
    def validate_sequence_probability_sums(self) -> "RFD4DesignSpec":
        """Reject constraints whose probabilities already exceed 1 on an identical selection."""
        if not self.sequence_probabilities:
            return self
        totals: dict[str, float] = defaultdict(float)
        for constraint in self.sequence_probabilities:
            for select in constraint.select:
                totals[select] += constraint.probability
        offending = {select: total for select, total in totals.items() if total > 1.0 + 1e-6}
        if offending:
            raise ValueError(
                f"sequence_probabilities sum above 1 for selection(s) {sorted(offending)}; "
                "probabilities on one residue must sum to at most 1"
            )
        return self

    def _translate(self, select: str, auth_to_label: dict[str, dict[str, int]]) -> str:
        """Translate one selection's residue ids when the spec is in author numbering.

        An empty ``auth_to_label`` means the input was passed through as an annotated
        CIF, whose ids are already the model's own numbering.
        """
        if self.residue_numbering == "label" or not auth_to_label:
            return select
        # A named expandable segment's residues are generated, so its ids are not author ids.
        return translate_selection(select, auth_to_label, contig_segment_names(self.contig))

    def _entries(self, selections: list[str], auth_to_label: dict[str, dict[str, int]]) -> list[dict[str, str]]:
        """Build the upstream ``[{"select": ...}]`` entry list for a set of selections."""
        return [{"select": self._translate(select, auth_to_label)} for select in selections]

    def build_conditions(self, auth_to_label: dict[str, dict[str, int]]) -> dict[str, Any]:
        """Compile the typed condition fields into RFD4's ``conditions`` mapping.

        Emission order is deliberate: overlapping entries resolve last-write-wins
        upstream, coordinate conditioning must be cleared before atoms move into an
        internal-coordinate group, and unindexing is applied against settled masks.

        Args:
            auth_to_label (dict[str, dict[str, int]]): Per-chain author→label residue map.

        Returns:
            dict[str, Any]: The ``conditions`` mapping, empty when nothing is conditioned.
        """
        conditions: dict[str, Any] = {}

        # C_CRD before C_CRDGRP: clearing absolute targets is a precondition for regrouping.
        coordinate: dict[Any, Any] = {}
        if self.free_coordinates:
            coordinate[False] = self._entries(self.free_coordinates, auth_to_label)
        if self.fixed_coordinates:
            coordinate[True] = self._entries(self.fixed_coordinates, auth_to_label)
        if coordinate:
            conditions["C_CRD"] = coordinate

        sequence: dict[Any, Any] = {}
        if self.free_sequence:
            sequence[False] = self._entries(self.free_sequence, auth_to_label)
        if self.known_sequence:
            sequence[True] = self._entries(self.known_sequence, auth_to_label)
        if sequence:
            conditions["C_SEQ"] = sequence

        if self.rigid_groups:
            conditions["C_CRDGRP"] = {
                group_id: self._entries(selections, auth_to_label) for group_id, selections in self.rigid_groups.items()
            }

        if self.hotspots:
            conditions["C_HOT"] = {True: self._entries(self.hotspots, auth_to_label)}
        if self.pi_stacking:
            conditions["C_PISTACK"] = {True: self._entries(self.pi_stacking, auth_to_label)}

        for field_name, alias in (
            ("burial", "C_RAS"),
            ("hbond", "C_HB"),
            ("pi_cation", "C_PICATION"),
            ("secondary_structure", "C_SEC"),
            ("fold_class", "C_CATH"),
            ("chain_centroid", "C_CENTROID"),
        ):
            value = getattr(self, field_name)
            if value:
                conditions[alias] = {key: self._entries(selections, auth_to_label) for key, selections in value.items()}

        for constraint in self.sequence_probabilities or []:
            alias = f"C_SEQ_PROB_{constraint.token}"
            entries = self._entries(constraint.select, auth_to_label)
            conditions.setdefault(alias, {}).setdefault(constraint.probability, []).extend(entries)

        # Unindexing last: it is stamped against the settled index mask.
        if self.unindexed_groups:
            conditions["C_IDX"] = {
                False: [
                    {"select": self._translate(select, auth_to_label), "group": label}
                    for label, selections in self.unindexed_groups.items()
                    for select in selections
                ]
            }

        if self.raw_conditions:
            conditions.update(self.raw_conditions)

        return conditions

    def to_spec_dict(self, cif_path: Path, auth_to_label: dict[str, dict[str, int]]) -> dict[str, Any]:
        """Build this spec's upstream entry: only ``input``, ``input_as_annotated_cif``, ``contig``, ``conditions``.

        Args:
            cif_path (Path): Path to the prepared input CIF.
            auth_to_label (dict[str, dict[str, int]]): Per-chain author→label residue map.

        Returns:
            dict[str, Any]: The per-design spec entry.
        """
        spec: dict[str, Any] = {"input": str(cif_path)}
        if self.input_as_annotated_cif:
            spec["input_as_annotated_cif"] = True
        if self.contig is not None:
            translate = self.residue_numbering == "auth" and bool(auth_to_label)
            spec["contig"] = translate_contig(self.contig, auth_to_label) if translate else self.contig
        conditions = self.build_conditions(auth_to_label)
        if conditions:
            spec["conditions"] = conditions
        return spec


class RFD4Input(BaseToolInput):
    """Input for RFD4-Proteina generation.

    Attributes:
        design_specs (list[RFD4DesignSpec]): Independent design tasks, each with its own
            input structure and constraints. Processed in a single run.

        raw_json (str | None): Raw upstream condition-spec JSON for callers who need
            conditions without a typed field. When set, ``design_specs`` is ignored and
            the JSON is forwarded as-is, so ``input`` paths must already point at CIF
            files and residue ids must already be in label numbering.

    Examples:
        >>> # De novo binder against a target's hotspots
        >>> from proto_tools.entities.structures import Structure
        >>> inputs = RFD4Input(
        ...     design_specs=[
        ...         RFD4DesignSpec(
        ...             input_structure=Structure.from_file("target.cif"),
        ...             contig="50-120, /0, A17-131",
        ...             rigid_groups={1: ["sel('A/*/*/[N,CA,C,O]')"]},
        ...             known_sequence=["sel('A')"],
        ...             hotspots=["sel('A/*/56/[CG,OH]')"],
        ...         )
        ...     ]
        ... )
    """

    design_specs: list[RFD4DesignSpec] = InputField(
        default_factory=list,
        title="Design Specs",
        description="Independent design tasks, each with its own input structure and constraints",
    )
    raw_json: str | None = InputField(
        default=None,
        title="Raw JSON",
        description="Raw upstream condition-spec JSON, forwarded as-is (ignores design_specs)",
    )

    @model_validator(mode="after")
    def validate_has_input(self) -> "RFD4Input":
        """Require either a non-empty ``design_specs`` or ``raw_json``."""
        if not self.design_specs and not self.raw_json:
            raise ValueError("Either 'design_specs' (non-empty) or 'raw_json' must be provided")
        return self

    @field_validator("raw_json")
    @classmethod
    def _validate_raw_json(cls, value: str | None) -> str | None:
        """Reject raw JSON that names a field from the removed pre-RFD4 dialect."""
        if value is None:
            return value
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("raw_json must be a JSON object mapping design ids to their specs")
        for design_id, spec in parsed.items():
            if design_id == "global_args" or not isinstance(spec, dict):
                continue
            legacy = sorted(set(spec) & set(_LEGACY_SPEC_FIELDS))
            if legacy:
                raise ValueError(
                    f"raw_json design {design_id!r} names removed pre-RFD4 dialect field(s) {legacy}; "
                    "RFD4 accepts only 'input', 'input_as_annotated_cif', 'contig', and 'conditions'"
                )
        return value

    def write_spec_file(self, input_dir: Path) -> tuple[Path, list[str]]:
        """Materialize each spec's structure as an RFD4-ready CIF and write the condition spec.

        Typed specs are written as YAML because condition *values* are mapping keys:
        YAML preserves the native bool, int, and float keys upstream expects, whereas
        JSON would stringify them. ``raw_json`` is written as JSON, as given.

        Args:
            input_dir (Path): Directory to write the prepared CIFs and spec file into.

        Returns:
            tuple[Path, list[str]]: The spec file path, and the design ids in input order.
        """
        if self.raw_json:
            spec_path = input_dir / "rfd4_input.json"
            parsed = json.loads(self.raw_json)
            spec_path.write_text(json.dumps(parsed, indent=2))
            return spec_path, [key for key in parsed if key != "global_args"]

        spec_dict: dict[str, Any] = {}
        for index, spec in enumerate(self.design_specs):
            spec_key = f"spec-{index}"
            cif_text: str
            auth_to_label: dict[str, dict[str, int]]
            if spec.input_as_annotated_cif:
                # Byte-for-byte: a gemmi round-trip drops the condition_* columns this flag preserves.
                cif_text, auth_to_label = spec.input_structure.structure, {}
            else:
                cif_text, auth_to_label = prepare_rfd4_cif(
                    spec.input_structure.structure,
                    spec.input_structure.structure_format or "pdb",
                )
            cif_path = input_dir / f"{spec_key}_input.cif"
            cif_path.write_text(cif_text)
            spec_dict[spec_key] = spec.to_spec_dict(cif_path, auth_to_label)

        spec_path = input_dir / "rfd4_input.yaml"
        spec_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False, default_flow_style=False))
        return spec_path, list(spec_dict)

    def __len__(self) -> int:
        """Get the number of design specs."""
        if self.raw_json:
            # global_args is a shared settings block, not a design, and is not a dispatched spec key.
            return len([key for key in json.loads(self.raw_json) if key != "global_args"])
        return len(self.design_specs)


# Config Data Models


class RFD4SamplerTuning(BaseModel):
    """Per-modality sampling temperature for RFD4's two generative modalities.

    RFD4 denoises a backbone modality (``bb_ca``) and a sidechain/sequence latent
    modality (``local_latents``) with separate schedules, so temperature is set per
    modality. Every field defaults to ``None``, which inherits the checkpoint's value.

    Attributes:
        tsr_k (float | None): Temporal score-rescaling sharpness; higher ``k`` is a
            lower temperature. Upstream defaults are ``16.0`` for the backbone and
            ``4.0`` for the latent modality.
        tsr_sigma (float | None): How quickly sharpening takes effect; higher is faster.
            Upstream defaults are ``3.0`` for the backbone and ``12.0`` for the latents.
        noise_scale (float | None): Injected-noise variance multiplier (upstream
            default ``1.0``). Note that ``0.0`` removes noise but retains stochastic
            drift, so it is not equivalent to deterministic sampling.
    """

    model_config = ConfigDict(extra="forbid")

    tsr_k: float | None = Field(
        default=None,
        gt=0.0,
        title="TSR Sharpness",
        description="Score-rescaling sharpness; higher = lower temperature (defaults: 16.0 backbone, 4.0 latent)",
    )
    tsr_sigma: float | None = Field(
        default=None,
        gt=0.0,
        title="TSR Sigma",
        description="How fast sharpening takes effect; higher = faster (defaults: 3.0 backbone, 12.0 latent)",
    )
    noise_scale: float | None = Field(
        default=None,
        ge=0.0,
        title="Noise Scale",
        description="Injected-noise variance multiplier (upstream default 1.0)",
    )

    def to_overrides(self, modality: str) -> dict[str, Any]:
        """Return set fields as ``generation.model.<modality>.simulation_step_params.*`` overrides."""
        prefix = f"generation.model.{modality}.simulation_step_params"
        mapping = {"tsr_k": "tsr_k", "tsr_sigma": "tsr_sigma", "noise_scale": "sc_scale_noise"}
        return {f"{prefix}.{mapping[name]}": value for name, value in self.model_dump(exclude_none=True).items()}


class RFD4CommonConfig(BaseConfig):
    """Settings shared by every RFD4-Proteina tool.

    Attributes:
        num_replicates (int): Designs generated per input spec.
        batch_size (int): Designs denoised in parallel on the GPU. Confirm the longest
            input fits at ``1`` before raising it.
        nsteps (int): Integration steps per design (upstream default 60). A
            ``recycle_strategy`` preset expands to exactly 60 steps and upstream requires
            the two to agree, so changing this requires ``recycle_strategy=None``.
        recycle_strategy (RecycleStrategy | None): Trunk recycling schedule as
            ``<low|medium|high>/<flat|midpeak|front_loaded>``, trading compute for
            quality. ``midpeak`` totals 120/150/180 trunk passes; ``flat`` and
            ``front_loaded`` total 120/180/300. The ``thinking_*`` aliases are the
            front-loaded schedules. Setting any preset pins ``nsteps`` to 60. ``None``
            disables the schedule and runs a fixed recycle depth of 0, which is the only
            way to use a custom ``nsteps``.
        backbone_sampling (RFD4SamplerTuning): Temperature for the backbone modality.
        sequence_sampling (RFD4SamplerTuning): Temperature for the sidechain/sequence
            latent modality. Lowering ``tsr_k`` raises sequence diversity.
        deterministic_sampling (bool): Switch both modalities to ODE integration, with
            no mid-trajectory noise injection.
        model_checkpoint (str): HuggingFace repo id or a local model-bundle directory.
            The denoiser and its matching autoencoder always load together.
        model_revision (str | None): HuggingFace commit to pin, for reproducible loads.
            Ignored when ``model_checkpoint`` is a local directory.
        adapter_path (str | None): Local PEFT/LoRA adapter directory, loaded onto the
            base and autoencoder pair it was fine-tuned against.
        use_torch_compile (bool): Compile the trunk across the upstream length buckets.
            Off by default: the first call otherwise pays a long compile across 15
            buckets, which only pays off for large runs.
        save_trajectory_every (int): Snapshot the trajectory every N steps. ``0``
            disables. Rejected unless ``output_dir`` is set.
        output_dir (str | None): Directory for run outputs; ``None`` uses a tempdir. Each
            call writes into a fresh ``run-<id>`` subdirectory of it, so a reused
            ``output_dir`` cannot return a previous run's designs, and the trajectory
            frames for a run land in that run's own subdirectory. The ``<id>`` is random
            and is deliberately not returned, since a per-run path in the output would
            break seeded reproducibility; locate a run by directory mtime, or pass a
            fresh ``output_dir`` per call.
        device (str): Execution device. ``rfd4-generate`` builds its Lightning trainer
            with ``accelerator="gpu"`` unconditionally, so a GPU is required.
    """

    num_replicates: int = ConfigField(
        title="Number of Replicates",
        default=1,
        ge=1,
        description="Designs generated per input spec",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Designs denoised in parallel on the GPU; confirm the longest input fits at 1 first",
    )
    nsteps: int = ConfigField(
        title="Integration Steps",
        default=60,
        ge=1,
        description="Integration steps per design; must stay 60 unless recycle_strategy is None",
    )
    recycle_strategy: RecycleStrategy | None = ConfigField(
        title="Recycle Strategy",
        default="thinking_medium",
        description="Trunk recycling schedule, e.g. 'medium/front_loaded'; pins nsteps to 60. None = no recycling",
    )
    backbone_sampling: RFD4SamplerTuning = ConfigField(
        default_factory=RFD4SamplerTuning,
        title="Backbone Sampling",
        description="Sampling temperature for the backbone modality; see RFD4SamplerTuning",
    )
    sequence_sampling: RFD4SamplerTuning = ConfigField(
        default_factory=RFD4SamplerTuning,
        title="Sequence Sampling",
        description="Sampling temperature for the sidechain/sequence latent modality",
    )
    deterministic_sampling: bool = ConfigField(
        title="Deterministic Sampling",
        default=False,
        description="Integrate both modalities as an ODE, with no mid-trajectory noise injection",
    )
    model_checkpoint: str = ConfigField(
        title="Model Checkpoint",
        default=DEFAULT_MODEL_CHECKPOINT,
        description="HuggingFace repo id or a local model-bundle directory",
        reload_on_change=True,
    )
    model_revision: str | None = ConfigField(
        title="Model Revision",
        default=DEFAULT_MODEL_REVISION,
        description="HuggingFace commit to pin for reproducible loads; ignored for local bundles",
        reload_on_change=True,
    )
    adapter_path: str | None = ConfigField(
        title="Adapter Path",
        default=None,
        description="Local PEFT/LoRA adapter directory to load onto the base model",
        reload_on_change=True,
    )
    use_torch_compile: bool = ConfigField(
        title="Use Torch Compile",
        default=False,
        description="Compile the trunk across upstream length buckets; pays a long first-call compile",
        include_in_key=False,
        # reserve_shape_cache() gates the held model's buckets on, so this cannot change under a
        # worker that already loaded one.
        reload_on_change=True,
    )
    save_trajectory_every: int = ConfigField(
        title="Save Trajectory Every",
        default=0,
        ge=0,
        description="Snapshot the trajectory every N steps (0 disables); requires output_dir to be set",
        include_in_key=False,
    )
    output_dir: str | None = ConfigField(
        title="Output Directory",
        default=None,
        description="Directory for run outputs, one fresh run-<id> subdir per call; None uses a tempdir",
        include_in_key=False,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on; RFD4 requires a GPU",
        include_in_key=False,
    )

    @model_validator(mode="after")
    def validate_recycle_strategy_matches_nsteps(self) -> "RFD4CommonConfig":
        """Mirror upstream's requirement that a recycle schedule expands to exactly ``nsteps``.

        Each preset's ``n_recycle_schedule`` bands sum to 60, and upstream raises
        "recycle schedule expands to 60 steps; expected N" after the model has loaded.
        """
        if self.recycle_strategy is not None and self.nsteps != RECYCLE_STRATEGY_NSTEPS:
            raise ValueError(
                f"recycle_strategy={self.recycle_strategy!r} expands to a fixed "
                f"{RECYCLE_STRATEGY_NSTEPS}-step schedule, so nsteps must be "
                f"{RECYCLE_STRATEGY_NSTEPS}; got {self.nsteps}. Set recycle_strategy=None to "
                "run a custom nsteps without trunk recycling."
            )
        return self

    @model_validator(mode="after")
    def validate_trajectories_are_reachable(self) -> "RFD4CommonConfig":
        """Reject trajectory dumping without an ``output_dir`` to leave the frames in.

        Trajectories are written beside the designs rather than returned in the output
        object, so with the default tempdir they would be deleted before the caller
        could read them -- a silently useless setting. With ``output_dir`` set, they
        land in that call's own ``run-<id>`` subdirectory of it.
        """
        if self.save_trajectory_every > 0 and self.output_dir is None:
            raise ValueError(
                "'save_trajectory_every' writes trajectory frames to disk beside the designs, "
                "so it needs 'output_dir' set; otherwise the frames land in a tempdir that is "
                "deleted as soon as the run finishes."
            )
        return self

    @classmethod
    def minimal(cls, **kwargs: Any) -> "RFD4CommonConfig":
        """Cheapest path through RFD4 for smoke testing.

        Disables trunk recycling and cuts the integration steps, which is far cheaper
        than any preset: 20 trunk passes against ``low/flat``'s 120 and
        ``thinking_medium``'s 180. Recycling has to go off for a custom ``nsteps`` to be
        legal at all. Used by the parametrized env-report and smoke-test infrastructure;
        production callers never invoke this.

        Args:
            **kwargs (Any): Field values passed to the config constructor. Take
                precedence over the cheap-mode defaults set here.

        Returns:
            RFD4CommonConfig: Config with cheap-mode sampling settings applied.
        """
        kwargs.setdefault("recycle_strategy", None)
        kwargs.setdefault("nsteps", 20)
        kwargs.setdefault("num_replicates", 1)
        kwargs.setdefault("batch_size", 1)
        return super().minimal(**kwargs)  # type: ignore[return-value]

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:  # noqa: ARG002 - registry contract
        """Reject local model bundles and adapters, which are not present on a hosted worker."""
        if self.adapter_path:
            return (
                "adapter_path points at a local PEFT adapter directory that is not available on "
                "device='proto'. Unset it, or run locally."
            )
        if "/" in self.model_checkpoint and Path(self.model_checkpoint).expanduser().exists():
            return (
                f"model_checkpoint {self.model_checkpoint!r} is a local directory that is not available "
                "on device='proto'. Use a HuggingFace repo id, or run locally."
            )
        return None

    def get_cli_kwargs(self) -> dict[str, Any]:
        """Build the Hydra ``key=value`` overrides shared by every RFD4 tool."""
        cli_kwargs: dict[str, Any] = {
            "num_replicates": self.num_replicates,
            "batch_size": self.batch_size,
            "generation.args.nsteps": self.nsteps,
            "hf_ckpt_path": self.model_checkpoint,
            # Always validate up front, so a bad spec fails before the checkpoint loads.
            "prevalidate_inputs": True,
            "generation.args.save_trajectory_every": self.save_trajectory_every,
            # Each call owns a fresh output dir, so upstream's resume shortcut could only hide a re-run.
            "skip_existing": False,
            # Inline metrics need an Apptainer AlphaFold2/3 oracle that this tool does not provision.
            "metrics_tags": [],
            # metrics_tags gates only tagged metrics: untagged unknown_residue_filter would rmtree a design.
            "delete_failing_pre_oracle": False,
        }
        if self.recycle_strategy is not None:
            cli_kwargs["recycle_strategy"] = self.recycle_strategy
        else:
            # A Hydra config group is switched off by deleting the default; null fails composition.
            cli_kwargs["~recycle_strategy"] = True
        cli_kwargs["hf_ckpt_revision"] = self.model_revision if self.model_revision is not None else "null"
        cli_kwargs["compile_lengths"] = "null" if not self.use_torch_compile else None
        if self.adapter_path is not None:
            cli_kwargs["adapter_path"] = self.adapter_path
        if self.deterministic_sampling:
            cli_kwargs["generation.model.bb_ca.gt.mode"] = "ode"
            cli_kwargs["generation.model.local_latents.gt.mode"] = "ode"
        cli_kwargs.update(self.backbone_sampling.to_overrides("bb_ca"))
        cli_kwargs.update(self.sequence_sampling.to_overrides("local_latents"))
        return {key: value for key, value in cli_kwargs.items() if value is not None}


class RFD4DesignConfig(RFD4CommonConfig):
    """Configuration for RFD4-Proteina structure and sequence design.

    Attributes:
        guidance_w (float): Classifier-free guidance weight. ``1.0`` is no guidance;
            ``0.0`` excludes the main model.
        partial_trajectory_fraction_steps (float | None): Run only the final fraction of
            the trajectory, to refine or diversify an input rather than generate from
            scratch. ``0.3`` runs the last 30% and stays close to the input; higher
            values drift further. Mutually exclusive with the ``partial_t_*`` settings.
        partial_t_backbone (float | None): Starting corruption level for the backbone,
            in ``(0, 1)``. Note the inverted sense relative to
            ``partial_trajectory_fraction_steps``: values near ``1`` stay close to the
            input, near ``0`` approach full generation.
        partial_t_latent (float | None): Starting corruption level for the
            sidechain/sequence latents, in ``(0, 1)``, with the same inverted sense.
        couple_partial_t (bool): Derive the unset modality's corruption level from the
            one that is set, through the default schedule coupling.
        num_replicates (int): Designs generated per input spec.
        batch_size (int): Designs denoised in parallel on the GPU. Confirm the longest
            input fits at ``1`` before raising it.
        nsteps (int): Integration steps per design (upstream default 60). A
            ``recycle_strategy`` preset expands to exactly 60 steps and upstream requires
            the two to agree, so changing this requires ``recycle_strategy=None``.
        recycle_strategy (RecycleStrategy | None): Trunk recycling schedule as
            ``<low|medium|high>/<flat|midpeak|front_loaded>``, trading compute for
            quality. ``midpeak`` totals 120/150/180 trunk passes; ``flat`` and
            ``front_loaded`` total 120/180/300. The ``thinking_*`` aliases are the
            front-loaded schedules. Setting any preset pins ``nsteps`` to 60. ``None``
            disables the schedule and runs a fixed recycle depth of 0, which is the only
            way to use a custom ``nsteps``.
        backbone_sampling (RFD4SamplerTuning): Temperature for the backbone modality.
        sequence_sampling (RFD4SamplerTuning): Temperature for the sidechain/sequence
            latent modality. Lowering ``tsr_k`` raises sequence diversity.
        deterministic_sampling (bool): Switch both modalities to ODE integration, with
            no mid-trajectory noise injection.
        model_checkpoint (str): HuggingFace repo id or a local model-bundle directory.
            The denoiser and its matching autoencoder always load together.
        model_revision (str | None): HuggingFace commit to pin, for reproducible loads.
            Ignored when ``model_checkpoint`` is a local directory.
        adapter_path (str | None): Local PEFT/LoRA adapter directory, loaded onto the
            base and autoencoder pair it was fine-tuned against.
        use_torch_compile (bool): Compile the trunk across the upstream length buckets.
            Off by default: the first call otherwise pays a long compile across 15
            buckets, which only pays off for large runs.
        save_trajectory_every (int): Snapshot the trajectory every N steps. ``0``
            disables. Rejected unless ``output_dir`` is set.
        output_dir (str | None): Directory for run outputs; ``None`` uses a tempdir. Each
            call writes into a fresh ``run-<id>`` subdirectory of it, so a reused
            ``output_dir`` cannot return a previous run's designs, and the trajectory
            frames for a run land in that run's own subdirectory. The ``<id>`` is random
            and is deliberately not returned, since a per-run path in the output would
            break seeded reproducibility; locate a run by directory mtime, or pass a
            fresh ``output_dir`` per call.
        device (str): Execution device. ``rfd4-generate`` builds its Lightning trainer
            with ``accelerator="gpu"`` unconditionally, so a GPU is required.

    Note:
        Partial diffusion needs the input's conditioning to survive the round trip, so
        it is normally paired with ``RFD4DesignSpec.input_as_annotated_cif=True`` on a
        previous RFD4 output.
    """

    guidance_w: float = ConfigField(
        title="Guidance Weight",
        default=1.0,
        ge=0.0,
        description="Classifier-free guidance weight; 1.0 = no guidance, 0.0 excludes the main model",
    )
    partial_trajectory_fraction_steps: float | None = ConfigField(
        title="Partial Trajectory Fraction",
        default=None,
        gt=0.0,
        le=1.0,
        description="Run only the final fraction of the trajectory; 0.3 stays close to the input",
    )
    partial_t_backbone: float | None = ConfigField(
        title="Partial T (Backbone)",
        default=None,
        gt=0.0,
        lt=1.0,
        description="Backbone starting corruption; near 1 stays close to the input, near 0 regenerates",
    )
    partial_t_latent: float | None = ConfigField(
        title="Partial T (Latent)",
        default=None,
        gt=0.0,
        lt=1.0,
        description="Sequence/sidechain starting corruption; near 1 stays close to the input",
    )
    couple_partial_t: bool = ConfigField(
        title="Couple Partial T",
        default=False,
        description="Derive the unset modality's corruption level from the one that is set",
    )

    @model_validator(mode="after")
    def validate_partial_diffusion(self) -> "RFD4DesignConfig":
        """Mirror upstream's partial-diffusion rules so a bad combination fails before the GPU.

        Upstream resolves these in ``rfproteina.search.vanilla`` *after* the model has
        loaded, so every rule below otherwise costs a full checkpoint load to discover.
        """
        advanced = [name for name in ("partial_t_backbone", "partial_t_latent") if getattr(self, name) is not None]

        if self.partial_trajectory_fraction_steps is not None:
            conflicting = advanced + (["couple_partial_t"] if self.couple_partial_t else [])
            if conflicting:
                raise ValueError(
                    "'partial_trajectory_fraction_steps' is mutually exclusive with "
                    f"{conflicting}: pass either the simple fraction or the per-modality corruption levels."
                )
            # Upstream rounds the fraction to a step count and rejects a zero-step trajectory.
            if round(self.partial_trajectory_fraction_steps * self.nsteps) < 1:
                raise ValueError(
                    f"partial_trajectory_fraction_steps={self.partial_trajectory_fraction_steps} with "
                    f"nsteps={self.nsteps} rounds to zero denoising steps; raise the fraction or nsteps."
                )
            return self

        # couple_partial_t derives the missing modality, so it is set exactly when one is given.
        if len(advanced) == 1 and not self.couple_partial_t:
            raise ValueError(
                f"only '{advanced[0]}' is set: either provide both 'partial_t_backbone' and "
                "'partial_t_latent', or set 'couple_partial_t=True' to derive the missing one."
            )
        if len(advanced) == 2 and self.couple_partial_t:
            raise ValueError(
                "'couple_partial_t=True' derives one modality's corruption level from the other, so it "
                "requires exactly one of 'partial_t_backbone' / 'partial_t_latent'; both were given."
            )
        if self.couple_partial_t and not advanced:
            raise ValueError(
                "'couple_partial_t' derives one modality's corruption level from the other, so it "
                "requires 'partial_t_backbone' or 'partial_t_latent' to be set."
            )
        return self

    def get_cli_kwargs(self) -> dict[str, Any]:
        """Add the design-only guidance and partial-diffusion overrides."""
        cli_kwargs = super().get_cli_kwargs()
        cli_kwargs["generation.args.guidance_w"] = self.guidance_w
        if self.partial_trajectory_fraction_steps is not None:
            cli_kwargs["generation.args.partial_trajectory_fraction_steps"] = self.partial_trajectory_fraction_steps
        if self.partial_t_backbone is not None:
            cli_kwargs["generation.args.partial_t_backbone"] = self.partial_t_backbone
        if self.partial_t_latent is not None:
            cli_kwargs["generation.args.partial_t_latent"] = self.partial_t_latent
        if self.couple_partial_t:
            cli_kwargs["generation.args.couple_partial_t"] = True
        return cli_kwargs


# Output Data Models


class RFD4Design(BaseModel):
    """A single design produced by RFD4-Proteina.

    RFD4 co-generates structure and sequence, so a design carries both without a
    separate inverse-folding step.

    Attributes:
        structure (Structure): The generated all-atom structure.
        complex (Complex): Predictor-ready chains; feed directly to a structure
            predictor for validation. Every polymer chain is a ``Chain`` with a
            ``sequence``, whatever its ``entity_type``; only a ligand becomes a
            ``Fragment``, which carries a CCD code or SMILES instead. Read sequences with
            ``chain_sequence_map()`` or ``chain_sequences``, which cover every entity --
            ``extract_protein_chains()`` returns only the ``entity_type == "protein"``
            chains, so a DNA or RNA design's nucleic chains are omitted from it.
        metadata (dict[str, Any]): Per-design metadata from RFD4's metrics table
            (resolved generation settings, checkpoint, seed).
    """

    structure: Structure = Field(
        title="Generated Structure",
        description="The generated all-atom structure",
    )
    complex: Complex = Field(
        title="Generated Complex",
        description="Predictor-ready chains and per-chain sequences",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        title="Design Metadata",
        description="Per-design metadata from RFD4's metrics table",
    )


class RFD4Designs(BaseModel):
    """All designs produced for a single input spec.

    Bundling the ``num_replicates`` designs for one spec into a single container keeps the
    framework's 1:1 cardinality between ``iterable_input_fields`` and
    ``iterable_output_field`` while RFD4 fans out per spec.

    Attributes:
        spec_key (str): Identifier of the spec that produced these designs. Positional
            (``"spec-0"``) for typed specs, or the caller's key under ``raw_json``.
        designs (list[RFD4Design]): The designs generated for this spec, in the order
            RFD4 emitted them. Length is ``num_replicates`` for a spec that generated
            successfully, and ``0`` for a spec that produced nothing.
    """

    spec_key: str = Field(
        title="Spec Key",
        description="Identifier of the input spec that produced these designs",
    )
    designs: list[RFD4Design] = Field(
        default_factory=list,
        title="Designs",
        description="Designs generated for this spec",
    )

    def __len__(self) -> int:
        """Get the number of designs for this spec."""
        return len(self.designs)

    def __getitem__(self, index: int) -> RFD4Design:
        """Get one design within this bundle by index."""
        return self.designs[index]

    def __iter__(self) -> Iterator[RFD4Design]:  # type: ignore[override]
        """Iterate over the designs in this bundle."""
        return iter(self.designs)


class RFD4Output(BaseToolOutput):
    """Output from an RFD4-Proteina run.

    Designs are grouped into ``RFD4Designs`` bundles — one per input spec — so the field
    cardinality matches ``RFD4Input.design_specs`` 1:1.

    Attributes:
        designed_structures (list[RFD4Designs]): One bundle per input spec, in input
            order. Total design count is at most ``len(design_specs) * num_replicates``;
            a spec that produced nothing contributes an empty bundle.

    Note:
        Iteration walks the per-spec bundles: ``output[i]`` is the i-th bundle and
        ``len(output)`` is the number of bundles. To walk individual designs, iterate a
        bundle: ``for bundle in output: for design in bundle: ...``.
    """

    designed_structures: list[RFD4Designs] = Field(
        default_factory=list,
        title="Designed Structures",
        description="Per-spec bundles of generated designs",
    )

    def __len__(self) -> int:
        """Get the number of per-spec design bundles."""
        return len(self.designed_structures)

    def __getitem__(self, index: int) -> RFD4Designs:
        """Get the design bundle for the i-th input spec."""
        return self.designed_structures[index]

    def __iter__(self) -> Iterator[RFD4Designs]:  # type: ignore[override]
        """Iterate over per-spec design bundles."""
        return iter(self.designed_structures)

    def __repr__(self) -> str:
        """Get string representation."""
        n_designs = sum(len(bundle) for bundle in self.designed_structures)
        return f"RFD4Output(designed_structures=[{len(self.designed_structures)} bundles, {n_designs} designs])"

    def __str__(self) -> str:
        """Get human-readable string representation."""
        return self.__repr__()

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["cif", "pdb"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "cif"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        if file_format not in ("cif", "pdb"):
            raise ValueError(f"Unsupported format: {file_format}")

        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)
        for bundle in self.designed_structures:
            for index, design in enumerate(bundle.designs):
                out_file = path / f"{bundle.spec_key}_design_{index}.{file_format}"
                if file_format == "cif":
                    design.structure.write_cif(out_file)
                else:
                    design.structure.write_pdb(out_file)


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> RFD4Input:
    """Minimal valid input for testing and examples."""
    return RFD4Input(
        design_specs=[
            RFD4DesignSpec(
                input_structure=Structure.from_file(EXAMPLE_INPUT_FIXTURE),
                contig="A1-20, 20-30",
            )
        ]
    )


def build_designs(output_data: dict[str, Any], spec_keys: list[str], source: str) -> list[RFD4Designs]:
    """Group RFD4's flat design list into one bundle per input spec, in input order.

    Args:
        output_data (dict[str, Any]): Raw subprocess payload with a ``designs`` list.
        spec_keys (list[str]): Design ids that were dispatched, in input order.
        source (str): Tool key recorded on each generated ``Structure``.

    Returns:
        list[RFD4Designs]: One bundle per dispatched spec, empty where none were produced.
    """
    buckets: dict[str, list[RFD4Design]] = defaultdict(list)
    for design_data in output_data.get("designs", []):
        structure = Structure(structure=design_data["structure_content"], source=source)
        buckets[design_data["spec_key"]].append(
            RFD4Design(
                structure=structure,
                complex=Complex.from_structure(structure),
                metadata=design_data.get("metadata", {}),
            )
        )
    # Backfill empties so a spec that produced nothing cannot shift later bundles.
    return [RFD4Designs(spec_key=key, designs=buckets.get(key, [])) for key in spec_keys]


@tool(
    key="rfd4-design",
    label="RFD4-Proteina Design",
    category="structure_design",
    input_class=RFD4Input,
    config_class=RFD4DesignConfig,
    output_class=RFD4Output,
    description="Joint sequence and all-atom structure design with RFD4-Proteina",
    uses_gpu=True,
    # Upstream builds its Lightning trainer with devices=N, which claims local cuda:0..N-1 and
    # cannot be redirected at a logical index in-process. Pinning makes the assigned GPU cuda:0.
    pin_visible_devices=True,
    example_input=example_input,
    iterable_input_fields=["design_specs"],
    iterable_output_field="designed_structures",
    max_chunk_size=1,
    cacheable=True,
    stochastic=True,
)
def run_rfd4_design(inputs: RFD4Input, config: RFD4DesignConfig, instance: Any = None) -> RFD4Output:
    """Design protein sequence and all-atom structure jointly with RFD4-Proteina.

    RFD4-Proteina is a flow-matching model that co-generates sequence and all-atom
    structure in one pass, so designs arrive with sequences already assigned. Supported
    design tasks include:

    - Unconditional and monomer generation
    - Motif scaffolding, indexed or unindexed (free placement)
    - Enzyme and atomic-motif (active site) design
    - Protein, small-molecule, and DNA binder design, with hotspot conditioning
    - Fold-class, burial, hydrogen-bond, and π-interaction conditioning
    - Sequence-identity constraints (e.g. cysteine-free designs)
    - Partial diffusion, to refine or diversify an existing design

    Args:
        inputs (RFD4Input): One or more design specifications.
        config (RFD4DesignConfig): Sampling, recycling, and checkpoint settings.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        RFD4Output: ``designed_structures`` holds one ``RFD4Designs`` bundle per input
            spec, in input order. Each ``RFD4Design`` carries the generated structure, a
            predictor-ready ``Complex`` with per-chain sequences, and per-design metadata.

    See Also:
        - RFD4-Proteina code: https://github.com/RosettaCommons/RFD4-Proteina
        - RFD4-Proteina docs: https://rfd4-proteina.org/latest/
        - ``rfdiffusion3-design`` for symmetric oligomers, which RFD4 does not support.

    Example:
        >>> # De novo binder against a target, with hotspots and a known target sequence
        >>> inputs = RFD4Input(
        ...     design_specs=[
        ...         RFD4DesignSpec(
        ...             input_structure=Structure.from_file("target.pdb"),
        ...             contig="50-120, /0, A17-131",
        ...             rigid_groups={1: ["sel('A/*/*/[N,CA,C,O]')"]},
        ...             known_sequence=["sel('A')"],
        ...             hotspots=["sel('A/*/56/[CG,OH]')"],
        ...         )
        ...     ]
        ... )
        >>> result = run_rfd4_design(inputs, RFD4DesignConfig(num_replicates=8))
        >>> sequences, chain_ids = result[0][0].complex.extract_protein_chains()
        >>> print(sequences[0])

    Note:
        - Residue ids are read in author numbering by default and translated to the
          mmCIF ``label_seq_id`` RFD4 resolves against; see
          ``RFD4DesignSpec.residue_numbering``.
        - Designs carry no built-in confidence score. Validate them by refolding with a
          structure predictor and comparing back to the generated backbone.
    """
    logger.debug("Using local GPU for RFD4-Proteina design")
    return _dispatch_rfd4(inputs, config, instance=instance, operation="design", source="rfd4-design")


def _dispatch_rfd4(
    inputs: RFD4Input,
    config: RFD4CommonConfig,
    *,
    instance: Any,
    operation: str,
    source: str,
) -> RFD4Output:
    """Materialize inputs, run RFD4 in its standalone environment, and bundle the designs."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_dir = temp_dir / "inputs"
        if config.output_dir:
            # Per-call subdir: _collect_designs globs the whole root, so a reused dir returns stale designs.
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=config.output_dir))
        else:
            output_dir = temp_dir / "rfd4_output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_path, spec_keys = inputs.write_spec_file(input_dir)

        input_data = {
            "operation": operation,
            "spec_path": str(spec_path),
            "output_dir": str(output_dir),
            # RFD4 suffixes example ids with the replicate index, so map designs back by these keys.
            "spec_keys": spec_keys,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
            **config.get_cli_kwargs(),
        }
        output_data = ToolInstance.dispatch("rfd4", input_data, instance=instance, config=config)

    return RFD4Output(designed_structures=build_designs(output_data, spec_keys, source))

"""Rich CCD result via pdbeccdutils (standalone subprocess)."""

import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from proto_tools.entities.ligands import Fragment, Ligands
from proto_tools.entities.ligands.ccd_utils import is_valid_ccd_code
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _silence_rdkit_logger() -> Iterator[None]:
    """Silence RDKit warnings around Fragment construction (organometallics, etc.)."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.*")  # type: ignore[attr-defined]


# ============================================================================
# Per-identifier result record (CCD-specific metadata that doesn't fit
# on Fragment). Pairs 1:1 with the corresponding Fragment in `ligands`.
# ============================================================================
# Fields that only a matched CCD entry can fill in; a record carrying any of them matched an entry.
_CCD_METADATA_FIELDS = ("formula", "formula_weight", "inchi", "inchikey", "release_status", "parent_ccd_code")


def _matched_ccd_entry(record: dict[str, Any]) -> bool:
    """Return whether a standalone record came from a CCD entry, rather than a failed lookup."""
    return any(record.get(name) is not None for name in _CCD_METADATA_FIELDS)


class CcdLookupResult(BaseModel):
    """One resolved ligand and its CCD-specific metadata.

    Carries the ``Fragment`` built for this identifier alongside the
    pdbeccdutils-derived fields that don't fit on ``Fragment`` (formula strings,
    descriptors, release status, optional network data). ``status`` says which of three
    outcomes the identifier had.

    Attributes:
        fragment (Fragment | None): The ligand built for this identifier. When resolved it
            carries a CCD code, which can be another entry with the same structure
            (``warnings`` says so). Otherwise it is the structure without a code, or None
            when no structure could be built (``errors`` says why).
        formula (str | None): Chemical formula (e.g. ``"C10 H16 N5 O13 P3"``).
        formula_weight (float | None): Formula weight in Daltons.
        inchi (str | None): InChI string as recorded in the CCD.
        inchikey (str | None): InChIKey as recorded in the CCD.
        released (bool): True iff release status is ``REL``.
        release_status (str | None): Raw ``pdbx_release_status`` value.
        parent_ccd_code (str | None): CCD code of the canonical parent
            (``mon_nstd_parent_comp_id``), if this component is a derivative.
        physchem_properties (dict[str, float]): Snake_case RDKit descriptors
            (``molecular_weight``, ``logp``, ``tpsa``, ``num_h_donors``, etc.).
        cross_references (dict[str, list[str]] | None): UniChem cross-references
            keyed by source database (e.g. ``{"chembl": ["CHEMBL14249"], ...}``).
            Populated only when ``include_cross_references=True``; requires network.
        pdb_structures (list[str] | None): PDB IDs of structures containing this
            ligand. Populated only when ``include_pdb_usage=True``; requires network.
        status (Literal["resolved", "no_ccd_code", "not_found"]): ``"resolved"`` when the
            fragment carries a CCD code; ``"no_ccd_code"`` when the identifier matched a CCD
            entry the fragment can't carry, with the CCD metadata kept and the reason in
            ``errors``; ``"not_found"`` when it matched no entry.
        warnings (list[str]): Non-fatal warnings emitted while reading the entry.
        errors (list[str]): Errors that prevented full parsing.
    """

    model_config = ConfigDict(extra="forbid")

    fragment: Fragment | None = Field(
        title="Fragment",
        description="The ligand built for this identifier, or None when no structure could be built",
    )

    formula: str | None = Field(default=None, title="Formula", description="Chemical formula string")
    formula_weight: float | None = Field(default=None, title="Formula Weight", description="Formula weight in Daltons")
    inchi: str | None = Field(default=None, title="InChI", description="InChI from CCD")
    inchikey: str | None = Field(default=None, title="InChIKey", description="InChIKey from CCD")
    released: bool = Field(default=False, title="Released", description="Whether release status is REL")
    release_status: str | None = Field(
        default=None, title="Release Status", description="Raw pdbx_release_status value"
    )
    parent_ccd_code: str | None = Field(
        default=None, title="Parent CCD Code", description="Canonical parent CCD code, if derivative"
    )
    physchem_properties: dict[str, float] = Field(
        default_factory=dict,
        title="Physicochemical Properties",
        description="RDKit-derived physicochemical properties",
    )
    cross_references: dict[str, list[str]] | None = Field(
        default=None,
        title="Cross-References",
        description="UniChem cross-references (network, opt-in)",
    )
    pdb_structures: list[str] | None = Field(
        default=None,
        title="PDB Structures",
        description="PDB IDs containing this ligand (network, opt-in)",
    )
    status: Literal["resolved", "no_ccd_code", "not_found"] = Field(
        title="Status",
        description="Whether the identifier matched a CCD entry, and whether the fragment carries its code",
    )
    warnings: list[str] = Field(default_factory=list, title="Warnings", description="Non-fatal parsing warnings")
    errors: list[str] = Field(default_factory=list, title="Errors", description="Parsing errors")

    @model_validator(mode="after")
    def _check_status_matches_contents(self) -> "CcdLookupResult":
        """The fragment carries a code only when resolved, and CCD metadata comes only from a match."""
        code = self.fragment.ccd_code if self.fragment is not None else None
        if (code is not None) != (self.status == "resolved"):
            raise ValueError(
                f"status is {self.status!r} but the fragment's CCD code is {code!r}; "
                "a resolved result carries a CCD code and no other result does"
            )
        metadata = [name for name in _CCD_METADATA_FIELDS if getattr(self, name) is not None]
        if self.status == "not_found" and metadata:
            raise ValueError(f"status is 'not_found' but CCD metadata {metadata} is set")
        if self.status == "no_ccd_code" and not metadata:
            raise ValueError(
                "status is 'no_ccd_code' but no CCD metadata is set; a result without a match is not_found"
            )
        return self


# ============================================================================
# Tool Input / Config / Output
# ============================================================================
class CcdLookupInput(BaseToolInput):
    """Input for CCD result.

    Each identifier is auto-detected as either a 1- to 5-character CCD code or
    a SMILES string. Mixed batches are supported.

    Attributes:
        identifiers (list[str]): CCD codes (e.g. ``"ATP"``) or SMILES strings.
            A single string is normalized to a list.
    """

    identifiers: list[str] = InputField(
        title="Identifiers", description="CCD codes (e.g. 'ATP') or SMILES strings to enrich"
    )

    @field_validator("identifiers", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: Any) -> list[str]:
        """Normalize a single string to a list."""
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[no-any-return]


class CcdLookupConfig(BaseConfig):
    """Configuration for CCD result.

    Enable network features explicitly when needed. They are off by default. For
    2D / 3D ligand visualization use
    :meth:`proto_tools.entities.ligands.Fragment.visualize` on the resolved
    ``result.ligands.fragments[i]`` instead.

    Attributes:
        include_cross_references (bool): Fetch UniChem cross-references
            (DrugBank/ChEMBL/PubChem/etc. IDs). Requires network. Default: False.
        include_pdb_usage (bool): Fetch PDB structures containing each ligand
            from the RCSB search API. Requires network. Default: False.
        sanitize (bool): Sanitize the parsed RDKit molecule. Disable only for
            CCD entries with unusual valences. Default: True.
        identifier_type (Literal["auto", "ccd_code", "smiles"]): How to read each identifier.
            ``"auto"`` reads a 1- to 5-character alphanumeric string as a CCD code and anything
            else as a SMILES, warning when a string reads both ways (``"CCO"`` is a CCD entry,
            not ethanol). ``"ccd_code"`` and ``"smiles"`` read every identifier one way.
            Default: ``"auto"``.
    """

    include_cross_references: bool = ConfigField(
        title="Include Cross-References", default=False, description="Fetch UniChem cross-references (network, opt-in)"
    )
    include_pdb_usage: bool = ConfigField(
        title="Include PDB Usage",
        default=False,
        description="Fetch PDB structures containing this ligand (network, opt-in)",
    )
    sanitize: bool = ConfigField(title="Sanitize", default=True, description="Sanitize the parsed RDKit molecule")
    identifier_type: Literal["auto", "ccd_code", "smiles"] = ConfigField(
        title="Identifier Type",
        default="auto",
        description="Read identifiers as CCD codes, as SMILES, or decide per identifier (auto)",
    )


class CcdLookupOutput(BaseToolOutput):
    """Output from CCD result.

    Each entry holds one resolved ligand together with its pdbeccdutils metadata.
    ``ligands`` gathers those fragments into the standard proto_tools ``Ligands``
    collection, ready to feed downstream tools that take ligands as input.

    Attributes:
        results (list[CcdLookupResult]): One entry per input identifier, in input
            order, each holding the resolved ``Fragment`` and its CCD-specific
            metadata (formula, descriptors, release status, optional network data).
        ligands (Ligands): Every result's fragment gathered into a ``Ligands`` collection, in
            input order. Raises when any result has no structure; not part of the export.
        num_resolved (int): Count of results with status ``"resolved"``.
        num_unresolved (int): Count of results with status ``"no_ccd_code"`` or ``"not_found"``.
    """

    results: list[CcdLookupResult] = Field(
        default_factory=list,
        title="Enrichments",
        description="Resolved ligand plus CCD metadata, one entry per input identifier",
    )

    @property
    def ligands(self) -> Ligands:
        """Every result's fragment as a ``Ligands`` collection, in input order.

        Raises rather than skip a result with no structure, so a complex can't be predicted with a
        ligand missing. To go on without those inputs, use the ``results`` whose ``fragment`` is set.

        Raises:
            ValueError: If any result has no fragment.
        """
        missing = [i for i, result in enumerate(self.results) if result.fragment is None]
        if missing:
            reasons = "; ".join(
                f"results[{i}]: {', '.join(self.results[i].errors) or 'no reason given'}" for i in missing
            )
            raise ValueError(
                f"No structure could be built for {len(missing)} of {len(self.results)} identifiers ({reasons}). "
                "To continue without them, build Ligands from the results whose fragment is set."
            )
        return Ligands(fragments=[result.fragment for result in self.results if result.fragment is not None])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_resolved(self) -> int:
        """Number of results whose fragment carries a CCD code."""
        return sum(1 for e in self.results if e.status == "resolved")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_unresolved(self) -> int:
        """Number of results with no CCD code on the fragment, whether matched or not."""
        return sum(1 for e in self.results if e.status != "resolved")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        path = Path(export_path).with_suffix(".json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> CcdLookupInput:
    """Minimal valid input for testing and examples."""
    return CcdLookupInput(identifiers=["ATP"])


def _status(
    fragment: Fragment | None, record: dict[str, Any], errors: list[str], warnings: list[str]
) -> Literal["resolved", "no_ccd_code", "not_found"]:
    """Classify a lookup, noting in ``errors`` or ``warnings`` where the fragment and the match disagree.

    The standalone sets ``ccd_code`` for any identifier shaped like a code, found or not, so a match
    is read from the CCD metadata only an entry can fill in.

    Args:
        fragment (Fragment | None): The fragment built for the identifier, if any.
        record (dict[str, Any]): The standalone record for the identifier.
        errors (list[str]): The result's errors, edited in place.
        warnings (list[str]): The result's warnings, appended to in place.

    Returns:
        Literal["resolved", "no_ccd_code", "not_found"]: The result's status.
    """
    matched = _matched_ccd_entry(record)
    code = record.get("ccd_code")
    if fragment is not None and fragment.ccd_code is not None:
        if not matched:
            # The fragment resolved from proto-tools' own CCD table; the lookup's failures no longer apply
            warnings.extend(errors)
            errors.clear()
        elif fragment.ccd_code != code:
            warnings.append(
                f"CCD entry {code!r} has the same structure as {fragment.ccd_code!r}, which the fragment carries"
            )
        return "resolved"
    if not matched:
        return "not_found"
    if not code or not is_valid_ccd_code(code):
        errors.append(f"CCD entry {code!r} is not in proto-tools' CCD table, so the fragment can't carry its code")
    else:
        errors.append(
            f"proto-tools' CCD table has no SMILES RDKit can parse for {code!r}, so the fragment can't carry its code"
        )
    return "no_ccd_code"


def _build_fragment(record: dict[str, Any], original_identifier: str, read_as_smiles: bool = True) -> Fragment | None:
    """Construct a Fragment from a standalone result record.

    Tries (smiles, ccd_code), then the code alone so a resolved entry survives a record
    whose SMILES will not parse, then (smiles only), in order of decreasing fidelity, and
    last the identifier itself as a SMILES. That last step is skipped when the identifier
    was looked up as the code of an entry that matched: read as a SMILES, a code names a
    different molecule (``FCO``, an iron complex, would be fluoromethanol). Returns None when
    nothing parses, rather than a stand-in molecule that a structure predictor would model
    in the ligand's place.

    Args:
        record (dict[str, Any]): The standalone record for the identifier.
        original_identifier (str): The identifier as the caller gave it.
        read_as_smiles (bool): Whether the identifier may be read as a SMILES; False when the
            caller said every identifier is a CCD code.

    Returns:
        Fragment | None: The built fragment, or None when no structure could be built.
    """
    smiles = record.get("smiles")
    ccd_code = record.get("ccd_code")
    name = record.get("name")
    identifier = original_identifier.strip()

    attempts: list[tuple[str | None, str | None]] = [(smiles, ccd_code), (None, ccd_code), (smiles, None)]
    looked_up_as_matched_code = _matched_ccd_entry(record) and identifier.upper() == ccd_code
    if read_as_smiles and identifier and not looked_up_as_matched_code:
        attempts.append((identifier, None))
    for s, c in attempts:
        if s is None and c is None:
            continue
        try:
            return Fragment(smiles=s, ccd_code=c, name=name)
        except ValueError:
            continue

    logger.warning("Could not build a structure for %r", original_identifier)
    return None


@tool(
    key="ccd-lookup",
    label="CCD Lookup (pdbeccdutils)",
    category="database_retrieval",
    input_class=CcdLookupInput,
    config_class=CcdLookupConfig,
    output_class=CcdLookupOutput,
    description=(
        "Rich result for wwPDB Chemical Component Dictionary entries via "
        "pdbeccdutils: returns Fragment objects plus formula, descriptors, "
        "parent component, physchem properties, and optional UniChem "
        "cross-references and PDB usage."
    ),
    example_input=example_input,
    iterable_input_fields=["identifiers"],
    iterable_output_field="results",
    max_chunk_size=256,
    cacheable=True,
    uses_gpu=False,
)
def run_ccd_lookup(
    inputs: CcdLookupInput,
    config: CcdLookupConfig,
    instance: Any = None,
) -> CcdLookupOutput:
    """Look up CCD codes or SMILES and return chemical metadata with pdbeccdutils.

    Inputs may mix 1- to 5-character CCD codes and SMILES strings. The worker
    detects the type of each identifier. Default operation is fully offline
    (uses bundled CCD mmCIF data). Network features (UniChem cross-references,
    PDB usage) are opt-in via config flags. See ``examples/example.ipynb`` and
    ``README.md`` for usage. For lookups within the calling process, see
    :mod:`proto_tools.entities.ligands.ccd_utils`.

    Args:
        inputs (CcdLookupInput): Identifiers to enrich.
        config (CcdLookupConfig): Configuration controlling optional network
            results (UniChem cross-refs, PDB usage).
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CcdLookupOutput: A ``Ligands`` collection plus a parallel list of
            CCD-specific ``CcdLookupResult`` records.
    """
    logger.debug("Running ccd-lookup for %d identifier(s)", len(inputs.identifiers))

    output_data = ToolInstance.dispatch(
        "ccd_lookup",
        {
            "identifiers": inputs.identifiers,
            "config": {
                "include_cross_references": config.include_cross_references,
                "include_pdb_usage": config.include_pdb_usage,
                "sanitize": config.sanitize,
                "identifier_type": config.identifier_type,
            },
        },
        instance=instance,
        config=config,
    )

    results: list[CcdLookupResult] = []
    # Fragment's validator re-runs RDKit canonicalization and InChIKey
    # generation; silence the chemistry logger across the build to keep
    # stderr clean for unusual user-supplied SMILES.
    with _silence_rdkit_logger():
        for original, record in zip(inputs.identifiers, output_data["records"], strict=True):
            fragment = _build_fragment(record, original, read_as_smiles=config.identifier_type != "ccd_code")
            errors = list(record.get("errors") or [])
            warnings = list(record.get("warnings") or [])
            status = _status(fragment, record, errors, warnings)
            if fragment is None:
                errors.append(f"No structure could be built for {original!r}")
            results.append(
                CcdLookupResult(
                    fragment=fragment,
                    formula=record.get("formula"),
                    formula_weight=record.get("formula_weight"),
                    inchi=record.get("inchi"),
                    inchikey=record.get("inchikey"),
                    released=bool(record.get("released", False)),
                    release_status=record.get("release_status"),
                    parent_ccd_code=record.get("parent_ccd_code"),
                    physchem_properties=record.get("physchem_properties") or {},
                    cross_references=record.get("cross_references"),
                    pdb_structures=record.get("pdb_structures"),
                    status=status,
                    warnings=warnings,
                    errors=errors,
                )
            )

    return CcdLookupOutput(
        metadata={
            "include_cross_references": config.include_cross_references,
            "include_pdb_usage": config.include_pdb_usage,
        },
        results=results,
    )

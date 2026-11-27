"""Rich CCD result via pdbeccdutils (standalone subprocess)."""

import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from proto_tools.entities.ligands import Fragment, Ligands
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
class CcdLookupResult(BaseModel):
    """One resolved ligand and its CCD-specific metadata.

    Carries the ``Fragment`` built for this identifier alongside the
    pdbeccdutils-derived fields that don't fit on ``Fragment`` (formula strings,
    descriptors, release status, optional network data). When the input was a
    SMILES with no CCD match, the fragment carries no ``ccd_code`` and the descriptive
    fields are also None.

    Attributes:
        fragment (Fragment): The ligand built for this identifier, carrying the CCD code
            when one could be resolved to a usable structure. ``fragment.ccd_code`` is the
            single source of truth for which CCD entry this is.
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
        resolved_without_structure (bool): True when the identifier matched a CCD entry
            whose structure could not be rebuilt. The CCD metadata below is populated,
            but ``fragment.ccd_code`` is None and the entry counts as unresolved.
        warnings (list[str]): Non-fatal warnings emitted while reading the entry.
        errors (list[str]): Errors that prevented full parsing.
    """

    model_config = ConfigDict(extra="forbid")

    fragment: Fragment = Field(
        title="Resolved Fragment",
        description="The ligand built for this identifier",
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
    resolved_without_structure: bool = Field(
        default=False,
        title="Resolved Without Structure",
        description="CCD entry matched but its structure could not be rebuilt",
    )
    warnings: list[str] = Field(default_factory=list, title="Warnings", description="Non-fatal parsing warnings")
    errors: list[str] = Field(default_factory=list, title="Errors", description="Parsing errors")

    @model_validator(mode="after")
    def _check_unresolved_metadata_is_explained(self) -> "CcdLookupResult":
        """CCD metadata without a resolved code must say why.

        These fields can only come from a CCD record, so their presence beside a fragment
        that carries no code is either a leak from another record or the narrow legitimate
        case where the lookup resolved an entry whose structure could not be rebuilt. The
        second is declared by ``resolved_without_structure``; the first is a bug and is
        still caught. The flag is checked rather than the presence of a warning, since
        unrelated warnings would otherwise disable the check.
        """
        if self.fragment.ccd_code is None and not self.resolved_without_structure:
            stale = [
                k
                for k, v in (
                    ("formula", self.formula),
                    ("formula_weight", self.formula_weight),
                    ("inchi", self.inchi),
                    ("inchikey", self.inchikey),
                    ("release_status", self.release_status),
                    ("parent_ccd_code", self.parent_ccd_code),
                )
                if v is not None
            ]
            if stale:
                raise ValueError(
                    f"fragment carries no ccd_code but {stale} are set, and resolved_without_structure "
                    "is False; a CcdLookupResult must be resolved, fully unresolved, or declare that its "
                    "CCD entry matched without a rebuildable structure"
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

    Network features are off by default; enable explicitly when needed. For
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


class CcdLookupOutput(BaseToolOutput):
    """Output from CCD result.

    Each entry holds one resolved ligand together with its pdbeccdutils metadata.
    ``ligands`` gathers those fragments into the standard proto_tools ``Ligands``
    collection, ready to feed downstream tools that take ligands as input.

    Attributes:
        results (list[CcdLookupResult]): One entry per input identifier, in input
            order, each holding the resolved ``Fragment`` and its CCD-specific
            metadata (formula, descriptors, release status, optional network data).
        ligands (Ligands): The resolved fragments gathered into a ``Ligands`` collection,
            in input order. Derived from ``results``.
        num_resolved (int): Count of identifiers that correspond to a known CCD component,
            whether the lookup returned the code or the fragment matched one by structure.
        num_unresolved (int): Count of identifiers with no usable CCD component.
    """

    results: list[CcdLookupResult] = Field(
        default_factory=list,
        title="Enrichments",
        description="Resolved ligand plus CCD metadata, one entry per input identifier",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ligands(self) -> Ligands:
        """The resolved fragments as a ``Ligands`` collection, in input order."""
        return Ligands(fragments=[result.fragment for result in self.results])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_resolved(self) -> int:
        """Number of identifiers carrying a CCD code, by lookup or by structural match."""
        return sum(1 for e in self.results if e.fragment.ccd_code is not None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_unresolved(self) -> int:
        """Number of identifiers with no usable CCD component.

        Includes entries whose CCD code matched but whose structure could not be
        rebuilt. Those carry ``resolved_without_structure=True`` and retain their
        CCD metadata, so resolution here means a usable structure, not merely a match.
        """
        return sum(1 for e in self.results if e.fragment.ccd_code is None)

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


def _build_fragment(record: dict[str, Any], original_identifier: str) -> Fragment:
    """Construct a Fragment from a standalone result record.

    Tries (smiles, ccd_code), then the code alone so a resolved entry survives a record
    whose SMILES will not parse, then (smiles only), then (original_identifier as smiles),
    in order of decreasing fidelity. Falls back to a wildcard-atom placeholder if none
    parse; the result carries the failure reason in ``errors``.
    """
    smiles = record.get("smiles")
    ccd_code = record.get("ccd_code")
    name = record.get("name")

    for s, c in ((smiles, ccd_code), (None, ccd_code), (smiles, None), (original_identifier, None)):
        if s is None and c is None:
            continue
        try:
            return Fragment(smiles=s, ccd_code=c, name=name)
        except ValueError:
            continue

    # Placeholder: helium has no CCD entry, so ccd_code stays None across
    # JSON round-trip ("[*]" would canonicalize to "*" → auto-resolve to "DUM").
    logger.warning("Could not build Fragment for %r; using placeholder", original_identifier)
    return Fragment(smiles="[He]", name=name or original_identifier)


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
    """Enrich CCD codes or SMILES with rich metadata via pdbeccdutils.

    Inputs may be a mix of 1- to 5-character CCD codes and SMILES strings; the
    standalone auto-detects per identifier. Default operation is fully offline
    (uses bundled CCD mmCIF data). Network features (UniChem cross-references,
    PDB usage) are opt-in via config flags. See ``examples/example.ipynb`` and
    ``README.md`` for usage patterns; for the microsecond-latency hot path see
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
            fragment = _build_fragment(record, original)
            warnings = list(record.get("warnings") or [])
            # Resolved a code no molecule could be built for: say so rather than leaving it unexplained.
            resolved_without_structure = fragment.ccd_code is None and bool(record.get("ccd_code"))
            if resolved_without_structure:
                warnings.append(
                    f"CCD entry {record['ccd_code']!r} resolved, but no usable structure could be built "
                    f"for it; the fragment carries no ccd_code and counts as unresolved."
                )
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
                    resolved_without_structure=resolved_without_structure,
                    warnings=warnings,
                    errors=list(record.get("errors") or []),
                )
            )

    return CcdLookupOutput(
        metadata={
            "include_cross_references": config.include_cross_references,
            "include_pdb_usage": config.include_pdb_usage,
        },
        results=results,
    )

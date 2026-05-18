"""Rich CCD enrichment via pdbeccdutils (standalone subprocess)."""

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
# Per-identifier enrichment record (CCD-specific metadata that doesn't fit
# on Fragment). Pairs 1:1 with the corresponding Fragment in `ligands`.
# ============================================================================
class CcdEnrichment(BaseModel):
    """CCD-specific metadata for a single ligand.

    Attached one-to-one to a ``Fragment`` in the tool's output. Holds the
    pdbeccdutils-derived fields that don't fit on ``Fragment`` (formula strings,
    descriptors, release status, optional network data). When the input was a
    SMILES with no CCD match, ``ccd_code`` is None and the descriptive fields
    are also None.

    Attributes:
        ccd_code (str | None): Three-letter CCD identifier; None when a SMILES
            input has no matching CCD entry.
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
        warnings (list[str]): Non-fatal warnings emitted while reading the entry.
        errors (list[str]): Errors that prevented full parsing.
    """

    model_config = ConfigDict(extra="forbid")

    ccd_code: str | None = Field(default=None, description="CCD code; None if SMILES has no match")
    formula: str | None = Field(default=None, description="Chemical formula string")
    formula_weight: float | None = Field(default=None, description="Formula weight in Daltons")
    inchi: str | None = Field(default=None, description="InChI from CCD")
    inchikey: str | None = Field(default=None, description="InChIKey from CCD")
    released: bool = Field(default=False, description="Whether release status is REL")
    release_status: str | None = Field(default=None, description="Raw pdbx_release_status value")
    parent_ccd_code: str | None = Field(default=None, description="Canonical parent CCD code, if derivative")
    physchem_properties: dict[str, float] = Field(
        default_factory=dict, description="RDKit-derived physicochemical properties"
    )
    cross_references: dict[str, list[str]] | None = Field(
        default=None, description="UniChem cross-references (network, opt-in)"
    )
    pdb_structures: list[str] | None = Field(
        default=None, description="PDB IDs containing this ligand (network, opt-in)"
    )
    warnings: list[str] = Field(default_factory=list, description="Non-fatal parsing warnings")
    errors: list[str] = Field(default_factory=list, description="Parsing errors")

    @model_validator(mode="after")
    def _check_resolved_or_fully_unresolved(self) -> "CcdEnrichment":
        """When ccd_code is None, the descriptive fields must also be None."""
        if self.ccd_code is None:
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
                    f"ccd_code is None but {stale} are set; CcdEnrichment must be fully resolved or fully None"
                )
        return self


# ============================================================================
# Tool Input / Config / Output
# ============================================================================
class CcdLookupInput(BaseToolInput):
    """Input for CCD enrichment.

    Each identifier is auto-detected as either a 1- to 5-character CCD code or
    a SMILES string. Mixed batches are supported.

    Attributes:
        identifiers (list[str]): CCD codes (e.g. ``"ATP"``) or SMILES strings.
            A single string is normalized to a list.
    """

    identifiers: list[str] = InputField(description="CCD codes (e.g. 'ATP') or SMILES strings to enrich")

    @field_validator("identifiers", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: Any) -> list[str]:
        """Normalize a single string to a list."""
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[no-any-return]


class CcdLookupConfig(BaseConfig):
    """Configuration for CCD enrichment.

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
    """Output from CCD enrichment.

    The output exposes both the standard proto_tools ligand types (``Ligands``
    collection of ``Fragment`` objects ready to feed downstream tools that take
    ligands as input) and a parallel list of CCD-specific enrichment records
    that hold the pdbeccdutils metadata.

    Attributes:
        ligands (Ligands): Collection of Fragment objects, one per input
            identifier in input order. SMILES inputs with no CCD match still
            produce a Fragment (with ``ccd_code=None``); see ``enrichments[i]``
            for the resolution status.
        enrichments (list[CcdEnrichment]): Per-identifier CCD-specific metadata
            (formula, descriptors, release status, optional network data).
            Same length and order as ``ligands.fragments``.
        num_resolved (int): Count of identifiers that resolved to a CCD record.
            Computed from ``enrichments``.
        num_unresolved (int): Count of identifiers that did not resolve.
            Computed from ``enrichments``.
    """

    ligands: Ligands = Field(
        default_factory=Ligands,
        description="Resolved fragments, one per input identifier (in input order)",
    )
    enrichments: list[CcdEnrichment] = Field(
        default_factory=list,
        description="Per-identifier CCD enrichment metadata, parallel to ligands.fragments",
    )

    @model_validator(mode="after")
    def _check_parallel_arrays(self) -> "CcdLookupOutput":
        """Enforce ligands.fragments and enrichments are parallel arrays."""
        if len(self.ligands.fragments) != len(self.enrichments):
            raise ValueError(
                f"ligands ({len(self.ligands.fragments)}) and enrichments ({len(self.enrichments)}) "
                "must be parallel arrays of the same length"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_resolved(self) -> int:
        """Number of identifiers that resolved to a CCD record."""
        return sum(1 for e in self.enrichments if e.ccd_code is not None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_unresolved(self) -> int:
        """Number of identifiers that did not resolve to any CCD entry."""
        return sum(1 for e in self.enrichments if e.ccd_code is None)

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
    """Construct a Fragment from a standalone enrichment record.

    Tries (smiles, ccd_code), then (smiles only), then (original_identifier as
    smiles), in order of decreasing fidelity. Falls back to a wildcard-atom
    placeholder if none parse — the caller relies on a 1:1 ``ligands`` /
    ``enrichments`` parallel array, and the corresponding ``CcdEnrichment``
    already carries the failure reason in ``errors``.
    """
    smiles = record.get("smiles")
    ccd_code = record.get("ccd_code")
    name = record.get("name")

    for s, c in ((smiles, ccd_code), (smiles, None), (original_identifier, None)):
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
        "Rich enrichment for wwPDB Chemical Component Dictionary entries via "
        "pdbeccdutils: returns Fragment objects plus formula, descriptors, "
        "parent component, physchem properties, and optional UniChem "
        "cross-references and PDB usage."
    ),
    example_input=example_input,
    iterable_input_field="identifiers",
    iterable_output_field="enrichments",
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
            enrichments (UniChem cross-refs, PDB usage).
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CcdLookupOutput: A ``Ligands`` collection plus a parallel list of
            CCD-specific ``CcdEnrichment`` records.
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

    fragments: list[Fragment] = []
    enrichments: list[CcdEnrichment] = []
    # Fragment's validator re-runs RDKit canonicalization and InChIKey
    # generation; silence the chemistry logger across the build to keep
    # stderr clean for unusual user-supplied SMILES.
    with _silence_rdkit_logger():
        for original, record in zip(inputs.identifiers, output_data["records"], strict=True):
            fragments.append(_build_fragment(record, original))
            enrichments.append(
                CcdEnrichment(
                    ccd_code=record.get("ccd_code"),
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
                    warnings=list(record.get("warnings") or []),
                    errors=list(record.get("errors") or []),
                )
            )

    return CcdLookupOutput(
        metadata={
            "include_cross_references": config.include_cross_references,
            "include_pdb_usage": config.include_pdb_usage,
        },
        ligands=Ligands(fragments=fragments),
        enrichments=enrichments,
    )

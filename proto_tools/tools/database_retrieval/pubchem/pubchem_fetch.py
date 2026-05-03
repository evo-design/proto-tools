"""proto_tools/tools/database_retrieval/pubchem/pubchem_fetch.py.

Resolves small-molecule identifiers (CID, name, SMILES, InChIKey) against
PubChem PUG REST and returns canonical structure data plus optional synonyms.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import requests
from pydantic import Field, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
)

logger = logging.getLogger(__name__)

_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

PubChemProperty = Literal[
    "MolecularFormula",
    "MolecularWeight",
    "SMILES",
    "ConnectivitySMILES",
    "InChI",
    "InChIKey",
    "IUPACName",
    "ExactMass",
    "MonoisotopicMass",
    "XLogP",
    "TPSA",
    "Complexity",
    "Charge",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "AtomStereoCount",
    "BondStereoCount",
    "DefinedAtomStereoCount",
    "DefinedBondStereoCount",
    "UndefinedAtomStereoCount",
    "UndefinedBondStereoCount",
    "IsotopeAtomCount",
    "CovalentUnitCount",
    "Volume3D",
    "XStericQuadrupole3D",
    "YStericQuadrupole3D",
    "ZStericQuadrupole3D",
    "FeatureCount3D",
    "FeatureAcceptorCount3D",
    "FeatureDonorCount3D",
    "FeatureAnionCount3D",
    "FeatureCationCount3D",
    "FeatureRingCount3D",
    "FeatureHydrophobeCount3D",
    "ConformerModelRMSD3D",
    "EffectiveRotorCount3D",
    "ConformerCount3D",
    "Fingerprint2D",
]

_DEFAULT_PROPERTIES: list[PubChemProperty] = [
    "MolecularFormula",
    "MolecularWeight",
    "SMILES",
    "ConnectivitySMILES",
    "InChI",
    "InChIKey",
    "IUPACName",
    "ExactMass",
    "TPSA",
    "Complexity",
    "Charge",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
]


# ============================================================================
# Data Models
# ============================================================================


class PubChemFetchInput(BaseToolInput):
    """Input for PubChem fetch.

    Exactly one of `cid`, `name`, `smiles`, `inchikey` must be provided.

    Attributes:
        cid (int | None): PubChem Compound Identifier (e.g. 2244 for aspirin).
        name (str | None): Common or systematic name (e.g. 'aspirin').
        smiles (str | None): SMILES string (e.g. 'CC(=O)Oc1ccccc1C(=O)O').
        inchikey (str | None): Standard InChIKey (e.g. 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N').
    """

    cid: int | None = InputField(default=None, ge=1, description="PubChem Compound Identifier")
    name: str | None = InputField(default=None, description="Common or systematic name")
    smiles: str | None = InputField(default=None, description="SMILES string")
    inchikey: str | None = InputField(default=None, description="Standard InChIKey")

    @model_validator(mode="after")
    def validate_exactly_one_identifier(self) -> "PubChemFetchInput":
        """Require exactly one of cid / name / smiles / inchikey."""
        provided = [v for v in (self.cid, self.name, self.smiles, self.inchikey) if v is not None]
        if len(provided) != 1:
            raise ValueError("Provide exactly one of: cid, name, smiles, inchikey")
        return self


class PubChemFetchConfig(BaseConfig):
    """Configuration for PubChem fetch.

    Attributes:
        properties (list[PubChemProperty]): PubChem property names to request.
            Defaults to a 15-property bundle covering structure (SMILES, InChI),
            mass, and basic descriptor counts (TPSA, HBA, HBD, etc.).
        include_synonyms (bool): If True, also fetch the compound's synonyms
            (one extra HTTP call).
        max_synonyms (int): Maximum number of synonyms to return (PubChem can
            return hundreds; the wrapper truncates client-side).
        request_timeout_seconds (int): HTTP timeout per request.
        http_retries (int): Number of retries for failed requests.
        backoff_seconds (float): Seconds to wait between retries (doubles after
            each attempt).
        user_agent (str): Identifier string sent to the PubChem API with each
            request.
    """

    properties: list[PubChemProperty] = ConfigField(
        title="Properties",
        default_factory=lambda: list(_DEFAULT_PROPERTIES),
        description="PubChem property names to request",
    )
    include_synonyms: bool = ConfigField(
        title="Include Synonyms",
        default=False,
        description="If True, also fetch the compound's synonyms list",
    )
    max_synonyms: int = ConfigField(
        title="Max Synonyms",
        default=50,
        ge=1,
        description="Maximum number of synonyms to return",
    )
    request_timeout_seconds: int = ConfigField(
        title="Request Timeout",
        default=15,
        ge=1,
        description="HTTP timeout in seconds",
        advanced=True,
    )
    http_retries: int = ConfigField(
        title="HTTP Retries",
        default=2,
        ge=0,
        description="Retries for HTTP requests",
        advanced=True,
    )
    backoff_seconds: float = ConfigField(
        title="Backoff Seconds",
        default=1.0,
        ge=0.0,
        description="Seconds to wait between retries (doubles after each attempt)",
        advanced=True,
    )
    user_agent: str = ConfigField(
        title="User Agent",
        default="proto-tools/pubchem-fetch-v1",
        description="Identifier string sent to the PubChem API with each request",
        advanced=True,
    )


class PubChemFetchOutput(BaseToolOutput):
    """Output from PubChem fetch.

    Attributes:
        cid (int): Resolved PubChem CID.
        all_matched_cids (list[int]): All CIDs returned by the resolver
            (length 1 for unambiguous queries; may be longer for ambiguous names).
        molecular_formula (str | None): Hill-system molecular formula.
        molecular_weight (float | None): Average molecular weight in g/mol.
        smiles (str | None): PubChem canonical SMILES with stereochemistry
            (the API field formerly named IsomericSMILES).
        connectivity_smiles (str | None): Connectivity-only SMILES, with
            stereochemistry stripped (the API field formerly named
            CanonicalSMILES).
        inchi (str | None): Standard InChI string.
        inchikey (str | None): Standard InChIKey hash.
        iupac_name (str | None): IUPAC systematic name.
        exact_mass (float | None): Exact (monoisotopic) mass in Da.
        tpsa (float | None): Topological polar surface area in angstroms-squared.
        complexity (int | None): Bertz / Hendrickson / Ihlenfeldt complexity.
        charge (int | None): Net formal charge.
        hbond_donor_count (int | None): Number of hydrogen-bond donors.
        hbond_acceptor_count (int | None): Number of hydrogen-bond acceptors.
        rotatable_bond_count (int | None): Number of rotatable bonds.
        heavy_atom_count (int | None): Number of non-hydrogen atoms.
        synonyms (list[str]): Up to `max_synonyms` synonyms (empty when
            `include_synonyms` is False).
        source_url (str): URL of the PubChem property request.
        raw_property_record (dict[str, Any]): Complete property record from
            PubChem for advanced programmatic access.
    """

    cid: int = Field(description="Resolved PubChem CID", ge=1)
    all_matched_cids: list[int] = Field(default_factory=list, description="All CIDs returned by the resolver")
    molecular_formula: str | None = Field(default=None, description="Molecular formula in Hill order")
    molecular_weight: float | None = Field(default=None, description="Molecular weight (g/mol)")
    smiles: str | None = Field(default=None, description="Canonical SMILES")
    connectivity_smiles: str | None = Field(default=None, description="Connectivity-only SMILES")
    inchi: str | None = Field(default=None, description="Standard InChI")
    inchikey: str | None = Field(default=None, description="Standard InChIKey")
    iupac_name: str | None = Field(default=None, description="IUPAC systematic name")
    exact_mass: float | None = Field(default=None, description="Exact (monoisotopic) mass in Da")
    tpsa: float | None = Field(default=None, description="Topological polar surface area")
    complexity: int | None = Field(default=None, description="Bertz/Hendrickson/Ihlenfeldt complexity")
    charge: int | None = Field(default=None, description="Net formal charge")
    hbond_donor_count: int | None = Field(default=None, description="Hydrogen-bond donor count")
    hbond_acceptor_count: int | None = Field(default=None, description="Hydrogen-bond acceptor count")
    rotatable_bond_count: int | None = Field(default=None, description="Rotatable bond count")
    heavy_atom_count: int | None = Field(default=None, description="Non-hydrogen atom count")
    synonyms: list[str] = Field(default_factory=list, description="Compound synonyms")
    source_url: str = Field(description="URL of the property request")
    raw_property_record: dict[str, Any] = Field(
        default_factory=dict, description="Complete property record from PubChem for advanced access"
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format == "json":
            path = Path(export_path).with_suffix(".json")
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return PubChemFetchInput(name="aspirin")


@tool(
    key="pubchem-fetch",
    label="PubChem Fetch",
    category="database_retrieval",
    input_class=PubChemFetchInput,
    config_class=PubChemFetchConfig,
    output_class=PubChemFetchOutput,
    description=(
        "Resolve small-molecule identifiers (CID, name, SMILES, InChIKey) against "
        "PubChem PUG REST and return canonical structure data plus optional synonyms"
    ),
    uses_gpu=False,
    example_input=example_input,
)
def run_pubchem_fetch(
    inputs: PubChemFetchInput,
    config: PubChemFetchConfig,
    instance: Any = None,
) -> PubChemFetchOutput:
    """Fetch a compound record from PubChem.

    Resolves the input identifier to a PubChem CID (skipped when `inputs.cid`
    is given directly), then fetches the configured property bundle and
    optionally synonyms.

    Args:
        inputs (PubChemFetchInput): Exactly one identifier (cid / name / smiles
            / inchikey).
        config (PubChemFetchConfig): Properties to request, synonym toggle, HTTP
            retry settings.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PubChemFetchOutput: Resolved CID, requested properties, optional
            synonyms, and the source URL.
    """
    del instance

    session = build_http_session(
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
    )

    try:
        all_cids = _resolve_to_cids(inputs, config, session)
        if not all_cids:
            raise ValueError(f"PubChem returned no CIDs for the supplied identifier: {_describe_input(inputs)}")
        if len(all_cids) > 1:
            logger.warning(
                "PubChem returned %d CIDs for %s; using the first (%d)",
                len(all_cids),
                _describe_input(inputs),
                all_cids[0],
            )
        cid = all_cids[0]

        property_record, property_url = _fetch_properties(cid, config, session)
        synonyms = _fetch_synonyms(cid, config, session) if config.include_synonyms else []

        return PubChemFetchOutput(
            cid=cid,
            all_matched_cids=all_cids,
            molecular_formula=property_record.get("MolecularFormula"),
            molecular_weight=_opt_float(property_record.get("MolecularWeight")),
            smiles=property_record.get("SMILES"),
            connectivity_smiles=property_record.get("ConnectivitySMILES"),
            inchi=property_record.get("InChI"),
            inchikey=property_record.get("InChIKey"),
            iupac_name=property_record.get("IUPACName"),
            exact_mass=_opt_float(property_record.get("ExactMass")),
            tpsa=_opt_float(property_record.get("TPSA")),
            complexity=_opt_int(property_record.get("Complexity")),
            charge=_opt_int(property_record.get("Charge")),
            hbond_donor_count=_opt_int(property_record.get("HBondDonorCount")),
            hbond_acceptor_count=_opt_int(property_record.get("HBondAcceptorCount")),
            rotatable_bond_count=_opt_int(property_record.get("RotatableBondCount")),
            heavy_atom_count=_opt_int(property_record.get("HeavyAtomCount")),
            synonyms=synonyms,
            source_url=property_url,
            raw_property_record=property_record,
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _describe_input(inputs: PubChemFetchInput) -> str:
    """Build a human-readable description of which identifier was provided."""
    if inputs.cid is not None:
        return f"cid={inputs.cid}"
    if inputs.name is not None:
        return f"name={inputs.name!r}"
    if inputs.smiles is not None:
        return f"smiles={inputs.smiles!r}"
    return f"inchikey={inputs.inchikey!r}"


def _resolve_to_cids(
    inputs: PubChemFetchInput,
    config: PubChemFetchConfig,
    session: requests.Session,
) -> list[int]:
    """Resolve any identifier to a list of matching PubChem CIDs.

    Returns an empty list on 404 (compound not found). Any other malformed
    response shape raises KeyError via direct dict access -- a real schema
    regression that the @tool decorator should surface.
    """
    if inputs.cid is not None:
        return [inputs.cid]

    if inputs.name is not None:
        url = f"{_PUBCHEM_BASE}/compound/name/{quote(inputs.name, safe='')}/cids/JSON"
    elif inputs.smiles is not None:
        url = f"{_PUBCHEM_BASE}/compound/smiles/{quote(inputs.smiles, safe='')}/cids/JSON"
    else:
        url = f"{_PUBCHEM_BASE}/compound/inchikey/{quote(inputs.inchikey or '', safe='')}/cids/JSON"

    response = session.get(url, timeout=config.request_timeout_seconds)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    payload = response.json()
    return [int(c) for c in payload["IdentifierList"]["CID"] if c]


def _fetch_properties(
    cid: int,
    config: PubChemFetchConfig,
    session: requests.Session,
) -> tuple[dict[str, Any], str]:
    """Fetch the requested property bundle for a CID. Returns (record, url)."""
    props = ",".join(config.properties)
    url = f"{_PUBCHEM_BASE}/compound/cid/{cid}/property/{props}/JSON"
    response = session.get(url, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    properties = response.json()["PropertyTable"]["Properties"]
    if not properties:
        raise ValueError(f"PubChem property response for CID {cid} contained no Properties entries at {url}")
    return properties[0], url


def _fetch_synonyms(
    cid: int,
    config: PubChemFetchConfig,
    session: requests.Session,
) -> list[str]:
    """Fetch the synonym list for a CID, truncated to `max_synonyms`.

    A 404 indicates PubChem has no synonyms for this CID -> empty list. Any
    other malformed shape raises KeyError via direct dict access.
    """
    url = f"{_PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
    response = session.get(url, timeout=config.request_timeout_seconds)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    info = response.json()["InformationList"]["Information"]
    if not info:
        return []
    return [str(s) for s in info[0]["Synonym"][: config.max_synonyms]]


def _opt_float(value: Any) -> float | None:
    """Coerce to float; pass through None unchanged.

    `None` is legitimate (PubChem omits properties absent from the request
    bundle); any other non-numeric value is a real schema regression and
    raises ValueError.
    """
    return None if value is None else float(value)


def _opt_int(value: Any) -> int | None:
    """Coerce to int; pass through None unchanged. Same loud-failure rules as `_opt_float`."""
    return None if value is None else int(value)

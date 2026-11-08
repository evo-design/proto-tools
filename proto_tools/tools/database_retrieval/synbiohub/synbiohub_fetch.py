"""Fetch a design from a SynBioHub registry as SBOL and parse it into a typed model."""

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    DeadlineAdapter,
    DeadlineRetry,
    InputField,
    build_http_session,
    request_with_retry,
)

SynBioHubMolecule = Literal["dna", "rna", "protein", "other"]
SynBioHubTopology = Literal["linear", "circular"]

_REQUEST_TIMEOUT_SECONDS = 30
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/synbiohub-fetch-v1"
# A read timeout or gateway error usually means the registry is still building the document; retrying adds load
_RETRY = DeadlineRetry(
    total=_HTTP_RETRIES,
    connect=_HTTP_RETRIES,
    read=0,
    backoff_factor=_BACKOFF_SECONDS,
    status_forcelist=[429, 500, 503],
    allowed_methods=["GET"],
)

_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_SBOL = "{http://sbols.org/v2#}"
_DCTERMS = "{http://purl.org/dc/terms/}"
_SO_PREFIX = "http://identifiers.org/so/"
_REVERSE_COMPLEMENT = "http://sbols.org/v2#reverseComplement"
_BIOPAX_MOLECULES: dict[str, SynBioHubMolecule] = {"DnaRegion": "dna", "RnaRegion": "rna", "Protein": "protein"}
_SO_TOPOLOGIES: dict[str, SynBioHubTopology] = {"SO:0000987": "linear", "SO:0000988": "circular"}

_EXPORT_SUFFIX = {"json": ".json", "sbol": ".sbol", "fasta": ".fasta", "genbank": ".gb"}
_GENBANK_MOLECULE = {"dna": "DNA", "rna": "RNA", "protein": "protein"}
_GENBANK_FEATURE_KEYS = {
    "SO:0000139": "RBS",
    "SO:0000316": "CDS",
    "SO:0000141": "terminator",
    "SO:0000167": "promoter",
    "SO:0000296": "rep_origin",
    "SO:0000553": "polyA_site",
    "SO:0000313": "stem_loop",
    "SO:0000204": "5'UTR",
    "SO:0000205": "3'UTR",
}
_FASTA_LINE_WIDTH = 70
_DATETIME = TypeAdapter(datetime)
# Links and SBOL identities on the public website resolve on its API host
_PUBLIC_API_HOST = "api.synbiohub.org"
_PUBLIC_WEB_HOSTS = {"synbiohub.org", "www.synbiohub.org"}
_PUBLIC_IDENTITY_ORIGIN = "https://synbiohub.org"
# Path segments of a versioned SynBioHub URI: /public/{collection}/{id}/{version}, /user/{user}/{collection}/{id}/{version}
_VERSIONED_PATH_SEGMENTS = {"public": 4, "user": 5}
_SPARQL_SAFE_PATH = re.compile(r"[A-Za-z0-9_./-]+")
# A bare SBOL displayId names a part in the registry's iGEM collection
_DISPLAY_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IGEM_COLLECTION_PATH = "/public/igem"
_IGEM_PART_LINKS = {
    "parts.igem.org": re.compile(r"/Part:([A-Za-z0-9_]+)"),
    "registry.igem.org": re.compile(r"^/parts/([A-Za-z0-9_-]+)"),
}


class SynBioHubFetchInput(BaseToolInput):
    """Input identifying one SynBioHub design.

    Attributes:
        identifier (str): A SynBioHub link to a ComponentDefinition, with or without its version, an
            iGEM Registry part link, or an iGEM part name such as ``BBa_I13504``.
    """

    identifier: str = InputField(
        title="Identifier",
        description="SynBioHub or iGEM Registry part link, or iGEM part name (e.g. 'BBa_I13504')",
    )

    @field_validator("identifier")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        identifier = value.strip().rstrip("/")
        if _DISPLAY_ID.fullmatch(identifier):
            return identifier
        if "://" not in identifier:
            identifier = f"https://{identifier}"
        parsed = urlsplit(identifier)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("identifier must be an HTTP or HTTPS link or an iGEM part name")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("identifier must not contain credentials, query parameters, or fragments")
        return identifier


class SynBioHubFetchConfig(BaseConfig):
    """Configuration for SynBioHub design retrieval.

    Attributes:
        registry_url (str): Registry origin allowed for object retrieval.
        synbiohub_email (str | None): SynBioHub account email or username.
        synbiohub_password (str | None): SynBioHub account password.
    """

    registry_url: str = ConfigField(
        default="https://api.synbiohub.org",
        title="Registry URL",
        description="SynBioHub registry origin allowed for object retrieval",
    )
    synbiohub_email: str | None = ConfigField(
        title="SynBioHub Email",
        description="SynBioHub email or username; required by the public instance.",
        credential="SYNBIOHUB_EMAIL",
    )
    synbiohub_password: str | None = ConfigField(
        title="SynBioHub Password",
        description="SynBioHub password; required by the public instance.",
        credential="SYNBIOHUB_PASSWORD",
    )

    @field_validator("registry_url")
    @classmethod
    def _validate_registry_url(cls, value: str) -> str:
        url = value.rstrip("/")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError("registry_url must be an HTTP or HTTPS origin without a path")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("registry_url must not contain credentials, query parameters, or fragments")
        return url


class SynBioHubLocation(BaseModel):
    """One contiguous range of a feature on its part's sequence.

    Attributes:
        start (int): First position, 1-indexed and inclusive.
        end (int): Last position, 1-indexed and inclusive.
        strand (Literal['+', '-']): Strand of the range relative to the part's sequence.
    """

    start: int = Field(title="Start", description="First position, 1-indexed and inclusive")
    end: int = Field(title="End", description="Last position, 1-indexed and inclusive")
    strand: Literal["+", "-"] = Field(title="Strand", description="Strand relative to the part's sequence")


class SynBioHubFeature(BaseModel):
    """One annotated feature on a part's sequence.

    Attributes:
        display_id (str): SBOL displayId of the sequence annotation.
        name (str | None): Annotation title, else the title of the subpart it places.
        roles (list[str]): Sequence Ontology roles, from the annotation or else its subpart.
        locations (list[SynBioHubLocation]): Ranges the feature covers; more than one is a split feature.
        subpart_uri (str | None): URI of the subpart this feature places, if any.
    """

    display_id: str = Field(title="Display ID", description="SBOL displayId of the sequence annotation")
    name: str | None = Field(default=None, title="Name", description="Annotation title, else its subpart's title")
    roles: list[str] = Field(
        default_factory=list,
        title="Roles",
        description="Sequence Ontology terms such as SO:0000139, inherited from the subpart if unset",
    )
    locations: list[SynBioHubLocation] = Field(
        title="Locations",
        description="Ranges the feature covers; more than one is a split feature",
    )
    subpart_uri: str | None = Field(
        default=None,
        title="Subpart URI",
        description="URI of the subpart this feature places, matching an entry in subparts",
    )


class SynBioHubPart(BaseModel):
    """One SBOL ComponentDefinition: a design or one of its parts.

    Attributes:
        uri (str): SBOL identity of the part.
        display_id (str): SBOL displayId.
        name (str | None): Part title.
        description (str | None): Part description.
        molecule (SynBioHubMolecule): Molecule type of the part.
        topology (SynBioHubTopology | None): Linear or circular, when the registry records it.
        roles (list[str]): Sequence Ontology roles of the part.
        sequence (str | None): Part sequence, uppercase, if the registry records one.
        features (list[SynBioHubFeature]): Features annotated on the sequence; ranged annotations only.
        modified (datetime | None): Last modification time in UTC, else creation time.
    """

    uri: str = Field(title="URI", description="SBOL identity of the part")
    display_id: str = Field(title="Display ID", description="SBOL displayId")
    name: str | None = Field(default=None, title="Name", description="Part title")
    description: str | None = Field(default=None, title="Description", description="Part description")
    molecule: SynBioHubMolecule = Field(title="Molecule", description="Molecule type of the part")
    topology: SynBioHubTopology | None = Field(
        default=None,
        title="Topology",
        description="Linear or circular, when the registry records it",
    )
    roles: list[str] = Field(
        default_factory=list,
        title="Roles",
        description="Sequence Ontology terms such as SO:0000141",
    )
    sequence: str | None = Field(default=None, title="Sequence", description="Part sequence, uppercase")
    features: list[SynBioHubFeature] = Field(
        default_factory=list,
        title="Features",
        description="Features annotated on the sequence; annotations without a range are omitted",
    )
    modified: datetime | None = Field(
        default=None,
        title="Modified",
        description="Last modification time in UTC (dcterms:modified), else creation time",
    )


class SynBioHubFetchOutput(BaseToolOutput):
    """One SynBioHub design, parsed from its recursive SBOL.

    Attributes:
        identifier (str): Requested identifier, as given.
        registry_url (str): Registry origin used for retrieval.
        source_url (str): SBOL download endpoint.
        design (SynBioHubPart): The requested design.
        subparts (list[SynBioHubPart]): Every other part in the SBOL document, at any depth.
        sbol (str): The downloaded SBOL document, verbatim.
    """

    # Keep the SBOL byte-exact; parsed fields carry no surrounding whitespace
    model_config = ConfigDict(str_strip_whitespace=False)

    identifier: str = Field(title="Identifier", description="Requested identifier, as given")
    registry_url: str = Field(title="Registry URL", description="Registry origin used for retrieval")
    source_url: str = Field(title="Source URL", description="SBOL download endpoint")
    design: SynBioHubPart = Field(title="Design", description="The requested design")
    subparts: list[SynBioHubPart] = Field(
        default_factory=list,
        title="Subparts",
        description="Every other part in the SBOL document, at any depth",
    )
    sbol: str = Field(repr=False, title="SBOL", description="The downloaded recursive SBOL document, verbatim")

    @property
    def output_format_options(self) -> list[str]:
        """Return supported output formats."""
        return ["json", "sbol", "fasta", "genbank"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format not in _EXPORT_SUFFIX:
            raise ValueError(f"Unsupported format: {file_format}")
        path = Path(export_path).with_suffix(_EXPORT_SUFFIX[file_format])
        if file_format == "json":
            path.write_text(json.dumps(self.model_dump(mode="json"), indent=2), encoding="utf-8")
        elif file_format == "sbol":
            path.write_text(self.sbol, encoding="utf-8")
        elif file_format == "fasta":
            path.write_text(_to_fasta(self.design), encoding="utf-8")
        else:
            _write_genbank(self.design, path)


def example_input() -> Any:
    """Return minimal valid example input."""
    return SynBioHubFetchInput(identifier="https://api.synbiohub.org/public/igem/BBa_I13504/1")


@tool(
    key="synbiohub-fetch",
    local_only="synbiohub-fetch is an HTTP-only CPU tool and needs no isolated environment",
    label="SynBioHub Fetch",
    category="database_retrieval",
    input_class=SynBioHubFetchInput,
    config_class=SynBioHubFetchConfig,
    output_class=SynBioHubFetchOutput,
    description="Fetch a SynBioHub design as typed parts and features, exportable to SBOL, FASTA, and GenBank",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_synbiohub_fetch(
    inputs: SynBioHubFetchInput,
    config: SynBioHubFetchConfig,
    instance: Any = None,
) -> SynBioHubFetchOutput:
    """Fetch one SynBioHub design and parse its recursive SBOL.

    Args:
        inputs (SynBioHubFetchInput): Design URI.
        config (SynBioHubFetchConfig): Registry and account settings.
        instance (Any): Optional ToolInstance; unused for this HTTP-only tool.

    Returns:
        SynBioHubFetchOutput: The parsed design, its subparts, and the verbatim SBOL.
    """
    del instance
    download_uri = _download_uri(inputs.identifier, config.registry_url)
    source_url = f"{download_uri}/sbol"
    if bool(config.synbiohub_email) != bool(config.synbiohub_password):
        raise ValueError("Set both SYNBIOHUB_EMAIL and SYNBIOHUB_PASSWORD, or neither")
    headers = {"Accept": "text/plain"}
    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )
    session.mount("https://", DeadlineAdapter(max_retries=_RETRY))
    try:
        if config.synbiohub_email and config.synbiohub_password:
            headers["X-authorization"] = _login(
                config.registry_url, config.synbiohub_email, config.synbiohub_password, session=session
            )
        # Without a version, SynBioHub serves the recursive document from /sbolnr too, so resolve the version first
        if not _is_versioned(download_uri):
            download_uri = _resolve_version(download_uri, config.registry_url, headers=headers, session=session)
            source_url = f"{download_uri}/sbol"
        # The non-recursive document is cheap for the registry to build, so check the type before the recursive one
        nonrecursive = _download(f"{download_uri}/sbolnr", headers=headers, session=session)
        design_uri = _design_uri(nonrecursive, download_uri)
        sbol = _download(source_url, headers=headers, session=session)
    finally:
        # SynBioHub keeps every token in memory until logout, so release this one
        if "X-authorization" in headers:
            session.post(f"{config.registry_url}/logout", headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS)
        session.close()
    design, subparts = _parse_sbol(sbol, design_uri)
    return SynBioHubFetchOutput(
        identifier=inputs.identifier,
        registry_url=config.registry_url,
        source_url=source_url,
        design=design,
        subparts=subparts,
        sbol=sbol,
    )


def _download_uri(identifier: str, registry_url: str) -> str:
    """Return the registry URI to download ``identifier`` from, rejecting objects outside ``registry_url``."""
    if _DISPLAY_ID.fullmatch(identifier):
        return f"{registry_url}{_IGEM_COLLECTION_PATH}/{_igem_part_name(identifier)}"
    parsed = urlsplit(identifier)
    host = parsed.netloc.lower()
    if host in _IGEM_PART_LINKS:
        match = _IGEM_PART_LINKS[host].search(parsed.path)
        if match is None:
            raise ValueError(f"{identifier} does not link to an iGEM part")
        return f"{registry_url}{_IGEM_COLLECTION_PATH}/{_igem_part_name(match[1])}"
    registry = urlsplit(registry_url)
    registry_host = registry.netloc.lower()
    if registry_host == _PUBLIC_API_HOST and host in {_PUBLIC_API_HOST, *_PUBLIC_WEB_HOSTS}:
        return f"{registry_url}{parsed.path}"
    if (parsed.scheme, host) != (registry.scheme, registry_host):
        raise ValueError("identifier must use the same scheme and host as registry_url")
    return f"{registry_url}{parsed.path}"


def _igem_part_name(name: str) -> str:
    """Return an iGEM part name as SynBioHub spells it, e.g. ``bba-i13504`` as ``BBa_I13504``."""
    prefix, rest = name[:4], name[4:]
    return f"BBa_{rest.upper()}" if prefix.lower() in {"bba_", "bba-"} else name


def _login(registry_url: str, email: str, password: str, *, session: requests.Session) -> str:
    """Log in to SynBioHub and return a token for this fetch."""
    response = session.post(
        f"{registry_url}/login",
        data={"email": email, "password": password},
        headers={"Accept": "text/plain"},
        timeout=_REQUEST_TIMEOUT_SECONDS,
        allow_redirects=False,
    )
    if response.status_code in (401, 403):
        raise ValueError(f"SynBioHub rejected the email or password: {response.text.strip()}")
    response.raise_for_status()
    return response.text.strip()


def _is_versioned(download_uri: str) -> bool:
    """Return whether a SynBioHub URI names one version of an object, judged by its path layout."""
    segments = urlsplit(download_uri).path.strip("/").split("/")
    return len(segments) >= _VERSIONED_PATH_SEGMENTS.get(segments[0], 0)


def _resolve_version(
    download_uri: str, registry_url: str, *, headers: dict[str, str], session: requests.Session
) -> str:
    """Return the download URI of the single version of an unversioned object, found by a SPARQL lookup."""
    path = urlsplit(download_uri).path
    if not _SPARQL_SAFE_PATH.fullmatch(path):
        raise ValueError(f"{download_uri} is not a SynBioHub object path")
    origin = _PUBLIC_IDENTITY_ORIGIN if urlsplit(registry_url).netloc.lower() == _PUBLIC_API_HOST else registry_url
    query = f"SELECT ?s WHERE {{ ?s <{_SBOL[1:-1]}persistentIdentity> <{origin}{path}> }}"
    response = _get(
        f"{registry_url}/sparql",
        headers={**headers, "Accept": "application/sparql-results+json"},
        session=session,
        params={"query": query},
    )
    versions = [binding["s"]["value"] for binding in response.json()["results"]["bindings"]]
    if not versions:
        raise ValueError(f"SynBioHub has no public object at {download_uri}; for other parts, pass a versioned link")
    if len(versions) > 1:
        raise ValueError(f"{download_uri} has several versions; include the version in the identifier")
    return f"{registry_url}{urlsplit(versions[0]).path}"


def _download(source_url: str, *, headers: dict[str, str], session: requests.Session) -> str:
    """Download one SynBioHub document, refusing redirects."""
    response = _get(source_url, headers=headers, session=session)
    return response.content.decode(response.encoding or "utf-8")


def _get(
    url: str, *, headers: dict[str, str], session: requests.Session, params: dict[str, str] | None = None
) -> requests.Response:
    """Send one GET to SynBioHub, refusing redirects and explaining authentication and lookup failures."""
    response = request_with_retry(
        lambda: session.get(
            url,
            headers=headers,
            params=params,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
        ),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    if 300 <= response.status_code < 400:
        raise ValueError(f"SynBioHub redirected {url}; verify the registry and identifier")
    if response.status_code == 404:
        raise ValueError(f"SynBioHub has no object at {url}")
    if response.status_code in (401, 403):
        if "X-authorization" not in headers:
            raise ValueError("SynBioHub requires a login; set SYNBIOHUB_EMAIL and SYNBIOHUB_PASSWORD")
        raise ValueError(f"SynBioHub refused access to {url} (HTTP {response.status_code})")
    response.raise_for_status()
    return response


def _design_uri(sbol: str, download_uri: str) -> str:
    """Return the SBOL identity of the ComponentDefinition at ``download_uri``, refusing any other object."""
    root = ET.fromstring(sbol)  # noqa: S314 -- expat refuses external entities and caps entity expansion
    # SBOL identities name the registry's public host, which can differ from the download host
    object_path = urlsplit(download_uri).path
    element = next((child for child in root if urlsplit(child.get(f"{_RDF}about", "")).path == object_path), None)
    if element is None or element.tag != f"{_SBOL}ComponentDefinition":
        kind = f"a {element.tag.rpartition('}')[2]}, not" if element is not None else "not"
        raise ValueError(f"{download_uri} is {kind} a ComponentDefinition; synbiohub-fetch retrieves designs")
    return element.get(f"{_RDF}about", "")


def _parse_sbol(sbol: str, design_uri: str) -> tuple[SynBioHubPart, list[SynBioHubPart]]:
    """Parse recursive SBOL into the design ``design_uri`` and every other part."""
    root = ET.fromstring(sbol)  # noqa: S314 -- expat refuses external entities and caps entity expansion
    definitions = {element.get(f"{_RDF}about", ""): element for element in root.findall(f"{_SBOL}ComponentDefinition")}
    sequences = {
        element.get(f"{_RDF}about", ""): "".join(element.findtext(f"{_SBOL}elements", "").split()).upper()
        for element in root.findall(f"{_SBOL}Sequence")
    }
    parts = {uri: _parse_part(uri, element, definitions, sequences) for uri, element in definitions.items()}
    design = parts.pop(design_uri)
    return design, list(parts.values())


def _parse_part(
    uri: str,
    element: ET.Element,
    definitions: dict[str, ET.Element],
    sequences: dict[str, str],
) -> SynBioHubPart:
    """Parse one ComponentDefinition."""
    molecule: SynBioHubMolecule = next(
        (
            _BIOPAX_MOLECULES[fragment]
            for type_uri in _resources(element, "type")
            if (fragment := type_uri.rpartition("#")[2]) in _BIOPAX_MOLECULES
        ),
        "other",
    )
    topology = next(
        (_SO_TOPOLOGIES[term] for term in _so_terms(_resources(element, "type")) if term in _SO_TOPOLOGIES), None
    )
    sequence = next((sequences[ref] for ref in _resources(element, "sequence") if ref in sequences), None)
    subpart_by_component = {
        component.get(f"{_RDF}about", ""): _resource(component, "definition")
        for component in element.findall(f"{_SBOL}component/{_SBOL}Component")
    }
    features = []
    for annotation in element.findall(f"{_SBOL}sequenceAnnotation/{_SBOL}SequenceAnnotation"):
        locations = [
            SynBioHubLocation(
                start=int(location.findtext(f"{_SBOL}start", "")),
                end=int(location.findtext(f"{_SBOL}end", "")),
                strand="-" if _resource(location, "orientation") == _REVERSE_COMPLEMENT else "+",
            )
            for location in annotation.iter(f"{_SBOL}Range")
        ]
        if not locations:
            continue
        subpart_uri = subpart_by_component.get(_resource(annotation, "component") or "")
        subpart = definitions.get(subpart_uri or "")
        features.append(
            SynBioHubFeature(
                display_id=annotation.findtext(f"{_SBOL}displayId", ""),
                name=_title(annotation) or (_title(subpart) if subpart is not None else None),
                roles=_so_roles(annotation) or (_so_roles(subpart) if subpart is not None else []),
                locations=locations,
                subpart_uri=subpart_uri,
            )
        )
    return SynBioHubPart(
        uri=uri,
        display_id=element.findtext(f"{_SBOL}displayId", ""),
        name=_title(element),
        description=element.findtext(f"{_DCTERMS}description") or None,
        molecule=molecule,
        topology=topology,
        roles=_so_roles(element),
        sequence=sequence,
        features=features,
        modified=_timestamp(element, "modified") or _timestamp(element, "created"),
    )


def _timestamp(element: ET.Element, tag: str) -> datetime | None:
    """Return an element's ``dcterms:{tag}`` time in UTC, or None when absent or malformed."""
    try:
        parsed = _DATETIME.validate_python((element.findtext(f"{_DCTERMS}{tag}") or "").strip())
    except ValidationError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _resources(element: ET.Element, tag: str) -> list[str]:
    """Return the rdf:resource URIs of an element's ``sbol:{tag}`` children."""
    return [child.get(f"{_RDF}resource", "") for child in element.findall(f"{_SBOL}{tag}")]


def _resource(element: ET.Element, tag: str) -> str | None:
    """Return the rdf:resource URI of an element's first ``sbol:{tag}`` child."""
    return next(iter(_resources(element, tag)), None)


def _title(element: ET.Element) -> str | None:
    """Return an element's dcterms:title, or None when absent or blank."""
    return element.findtext(f"{_DCTERMS}title") or None


def _so_roles(element: ET.Element) -> list[str]:
    """Return an element's Sequence Ontology roles as ``SO:NNNNNNN`` terms."""
    return _so_terms(_resources(element, "role"))


def _so_terms(uris: list[str]) -> list[str]:
    """Return the Sequence Ontology URIs among ``uris`` as ``SO:NNNNNNN`` terms."""
    return [uri.removeprefix(_SO_PREFIX) for uri in uris if uri.startswith(_SO_PREFIX)]


def _require_sequence(part: SynBioHubPart) -> str:
    """Return a part's sequence, or raise when the registry recorded none."""
    if part.sequence is None:
        raise ValueError(f"{part.display_id} has no sequence to export")
    return part.sequence


def _to_fasta(part: SynBioHubPart) -> str:
    """Render a part as a FASTA record."""
    sequence = _require_sequence(part)
    # A FASTA header is one line, so collapse any whitespace the description spans
    header = " ".join(f">{part.display_id} {part.description or ''}".split())
    lines = [sequence[i : i + _FASTA_LINE_WIDTH] for i in range(0, len(sequence), _FASTA_LINE_WIDTH)]
    return "\n".join([header, *lines]) + "\n"


def _write_genbank(part: SynBioHubPart, path: Path) -> None:
    """Write a part and its features as a GenBank record."""
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqFeature import CompoundLocation, SeqFeature, SimpleLocation
    from Bio.SeqRecord import SeqRecord

    sequence = _require_sequence(part)
    if part.molecule not in _GENBANK_MOLECULE:
        raise ValueError(f"{part.display_id} has molecule type {part.molecule!r}, which GenBank cannot record")
    record = SeqRecord(
        Seq(sequence),
        id=part.display_id,
        name=part.display_id,
        description=part.description or part.name or part.display_id,
        annotations={"molecule_type": _GENBANK_MOLECULE[part.molecule]},
    )
    if part.topology is not None:
        record.annotations["topology"] = part.topology
    if part.modified is not None:
        record.annotations["date"] = part.modified.strftime("%d-%b-%Y").upper()
    for feature in part.features:
        ranges = [
            SimpleLocation(location.start - 1, location.end, strand=1 if location.strand == "+" else -1)  # type: ignore[no-untyped-call]
            for location in feature.locations
        ]
        record.features.append(
            SeqFeature(  # type: ignore[no-untyped-call]
                CompoundLocation(ranges) if len(ranges) > 1 else ranges[0],  # type: ignore[no-untyped-call]
                type=next(
                    (_GENBANK_FEATURE_KEYS[r] for r in feature.roles if r in _GENBANK_FEATURE_KEYS), "misc_feature"
                ),
                qualifiers={"label": [feature.name or feature.display_id]},
            )
        )
    SeqIO.write(record, path, "genbank")

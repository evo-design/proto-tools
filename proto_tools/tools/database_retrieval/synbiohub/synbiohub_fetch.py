"""Fetch a design from a SynBioHub registry as SBOL and parse it into a typed model."""

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
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
_RETRY = Retry(
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


class SynBioHubFetchInput(BaseToolInput):
    """Input identifying one SynBioHub design.

    Attributes:
        object_uri (str): Absolute URI of a ComponentDefinition on the selected registry.
    """

    object_uri: str = InputField(
        title="Object URI",
        description="Absolute URI of a ComponentDefinition on the selected SynBioHub registry",
    )

    @field_validator("object_uri")
    @classmethod
    def _validate_object_uri(cls, value: str) -> str:
        uri = value.rstrip("/")
        parsed = urlsplit(uri)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("object_uri must be an absolute HTTP or HTTPS URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("object_uri must not contain credentials, query parameters, or fragments")
        return uri


class SynBioHubFetchConfig(BaseConfig):
    """Configuration for SynBioHub design retrieval.

    Attributes:
        registry_url (str): Registry origin allowed for object retrieval.
        synbiohub_token (str | None): SynBioHub login token.
    """

    registry_url: str = ConfigField(
        default="https://api.synbiohub.org",
        title="Registry URL",
        description="SynBioHub registry origin allowed for object retrieval",
    )
    synbiohub_token: str | None = ConfigField(
        title="SynBioHub Token",
        description="SynBioHub login token; required by the public instance.",
        credential="SYNBIOHUB_TOKEN",
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
        modified (datetime | None): Last modification time, else creation time.
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
        description="Last modification time (dcterms:modified), else creation time",
    )


class SynBioHubFetchOutput(BaseToolOutput):
    """One SynBioHub design, parsed from its recursive SBOL.

    Attributes:
        object_uri (str): Requested object URI.
        registry_url (str): Registry origin used for retrieval.
        source_url (str): SBOL download endpoint.
        design (SynBioHubPart): The requested design.
        subparts (list[SynBioHubPart]): Every other part in the SBOL document, at any depth.
        sbol (str): The downloaded SBOL document, verbatim.
    """

    # Keep the SBOL byte-exact; parsed fields carry no surrounding whitespace
    model_config = ConfigDict(str_strip_whitespace=False)

    object_uri: str = Field(title="Object URI", description="Requested object URI")
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
    return SynBioHubFetchInput(object_uri="https://api.synbiohub.org/public/igem/BBa_I13504/1")


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
        config (SynBioHubFetchConfig): Registry and token settings.
        instance (Any): Optional ToolInstance; unused for this HTTP-only tool.

    Returns:
        SynBioHubFetchOutput: The parsed design, its subparts, and the verbatim SBOL.
    """
    del instance
    _validate_same_origin(inputs.object_uri, config.registry_url)
    source_url = f"{inputs.object_uri}/sbol"
    headers = {"Accept": "text/plain"}
    if config.synbiohub_token:
        headers["X-authorization"] = config.synbiohub_token
    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )
    session.mount("https://", HTTPAdapter(max_retries=_RETRY))
    try:
        # The non-recursive document is cheap for the registry to build, so check the type before the recursive one
        nonrecursive = _download(f"{inputs.object_uri}/sbolnr", headers=headers, session=session)
        design_uri = _design_uri(nonrecursive, inputs.object_uri)
        sbol = _download(source_url, headers=headers, session=session)
    finally:
        session.close()
    design, subparts = _parse_sbol(sbol, design_uri)
    return SynBioHubFetchOutput(
        object_uri=inputs.object_uri,
        registry_url=config.registry_url,
        source_url=source_url,
        design=design,
        subparts=subparts,
        sbol=sbol,
    )


def _validate_same_origin(object_uri: str, registry_url: str) -> None:
    """Reject cross-origin object URLs to bound caller-selected network access."""
    object_origin = urlsplit(object_uri)
    registry_origin = urlsplit(registry_url)
    if (object_origin.scheme, object_origin.netloc) != (registry_origin.scheme, registry_origin.netloc):
        raise ValueError("object_uri must use the same scheme and host as registry_url")


def _download(source_url: str, *, headers: dict[str, str], session: requests.Session) -> str:
    """Download one SynBioHub document, refusing redirects."""
    response = request_with_retry(
        lambda: session.get(
            source_url,
            headers=headers,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
        ),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    if 300 <= response.status_code < 400:
        raise ValueError(f"SynBioHub redirected {source_url}; verify the registry and object URI")
    if response.status_code in (401, 403):
        if "X-authorization" not in headers:
            raise ValueError("SynBioHub requires a login token; set SYNBIOHUB_TOKEN")
        raise ValueError(
            f"SynBioHub rejected the token (HTTP {response.status_code}); it may have expired, so request a new one"
        )
    response.raise_for_status()
    return response.content.decode(response.encoding or "utf-8")


def _design_uri(sbol: str, object_uri: str) -> str:
    """Return the SBOL identity of the ComponentDefinition at ``object_uri``, refusing any other object."""
    root = ET.fromstring(sbol)  # noqa: S314 -- expat refuses external entities and caps entity expansion
    # SBOL identities name the registry's public host, which can differ from the download host
    object_path = urlsplit(object_uri).path
    element = next((child for child in root if urlsplit(child.get(f"{_RDF}about", "")).path == object_path), None)
    if element is None or element.tag != f"{_SBOL}ComponentDefinition":
        kind = f"a {element.tag.rpartition('}')[2]}, not" if element is not None else "not"
        raise ValueError(f"{object_uri} is {kind} a ComponentDefinition; synbiohub-fetch retrieves designs")
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
    timestamp = element.findtext(f"{_DCTERMS}modified") or element.findtext(f"{_DCTERMS}created")
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
        modified=datetime.fromisoformat(timestamp) if timestamp else None,
    )


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
    header = " ".join(filter(None, [f">{part.display_id}", part.description]))
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

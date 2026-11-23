"""proto_tools/tools/database_retrieval/europepmc/europepmc_fulltext.py.

Open-access full text from Europe PMC: fetches an article's JATS XML by PMCID and flattens it
into plain-text sections (title, abstracts, body sections with their heading path, figure and
table captions, and tables as text), so a passage can be searched for and quoted verbatim.
"""

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator

from proto_tools.tools.database_retrieval.europepmc.europepmc_search import _SEARCH_URL, _item_error
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    InputField,
    build_http_session,
    raise_for_status_with_body,
    request_with_retry,
)
from proto_tools.utils.standalone_helpers_source.standalone_helpers import StatusCounter

logger = logging.getLogger(__name__)

_REST_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_FULLTEXT_URL = _REST_URL + "/{pmcid}/fullTextXML"
_REQUEST_TIMEOUT_SECONDS = 60
# fullTextXML answers open-access articles with intermittent HTTP 500s.
_HTTP_RETRIES = 4
_LOOKUP_BATCH_SIZE = 100
# Europe PMC's maximum page size, so a PMCID with duplicate records cannot push another off the page.
_LOOKUP_PAGE_SIZE = 1000
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/europepmc-fulltext-v1"
_PMCID_RE = re.compile(r"^PMC\d+$")
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

# Elements whose content never belongs to running text.
_SKIPPED = frozenset({"fig", "fig-group", "table-wrap", "table-wrap-group", "ref-list", "graphic", "alt-text"})
# Elements that start a new block of text, so their content is separated by a space.
_BLOCKS = frozenset({"p", "title", "label", "caption", "list-item", "def-item", "break", "td", "th", "disp-formula"})
# Children of a section (or of <back>) that are themselves sections.
_SUBSECTIONS = frozenset({"sec", "ack", "app", "app-group", "notes", "glossary", "fn-group"})
_FLOATS = frozenset({"fig", "table-wrap"})

SectionKind = Literal["title", "abstract", "section", "figure", "table"]


# ============================================================================
# Data Models
# ============================================================================


class EuropePMCFullTextInput(BaseToolInput):
    """Articles to fetch from Europe PMC.

    Attributes:
        pmcids (list[str]): PubMed Central IDs (e.g. ``"PMC6034017"``; a bare number gets the ``PMC``
            prefix). A bare string is normalized to a one-item list.
    """

    pmcids: list[str] = InputField(
        min_length=1,
        title="PMCIDs",
        description="PubMed Central IDs of open-access articles, e.g. 'PMC6034017'",
    )

    @field_validator("pmcids", mode="before")
    @classmethod
    def _normalize_pmcids(cls, value: Any) -> Any:
        """Accept a bare string, upper-case and prefix bare numbers; reject anything else."""
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return value
        cleaned = []
        for item in value:
            pmcid = str(item).strip().upper()
            pmcid = f"PMC{pmcid}" if pmcid.isdigit() else pmcid
            if not _PMCID_RE.match(pmcid):
                raise ValueError(f"not a PMCID: {item!r} (expected e.g. 'PMC6034017')")
            cleaned.append(pmcid)
        return cleaned


class EuropePMCFullTextConfig(BaseConfig):
    """Configuration for Europe PMC full-text retrieval (no options beyond the shared ones)."""


class EuropePMCSection(BaseModel):
    """One block of article text.

    Attributes:
        kind (SectionKind): ``title``, ``abstract``, ``section`` (body or back matter), ``figure``
            (caption) or ``table`` (caption, rows, footnotes).
        heading (str | None): Heading path for sections (``"Results > Binding assays"``), or the
            label for figures and tables (``"Table 1"``).
        text (str): Plain text. Paragraphs are separated by newlines; table cells by `` | ``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: SectionKind = Field(title="Kind", description="title, abstract, section, figure or table")
    heading: str | None = Field(default=None, title="Heading", description="Heading path, or figure/table label")
    text: str = Field(title="Text", description="Plain text; paragraphs separated by newlines")


class EuropePMCFullTextDocument(BaseModel):
    """One article's full text.

    Attributes:
        pmcid (str): PubMed Central ID.
        pmid (str | None): PubMed ID, when the article lists one.
        doi (str | None): DOI, when the article lists one.
        title (str | None): Article title; None on error.
        license (str | None): License URL (or statement) from the article's permissions.
        sections (list[EuropePMCSection]): Title, abstracts, body and back sections, figures and
            tables, in document order (figures and tables follow the section citing them).
        source_url (str | None): Europe PMC full-text XML URL the text was read from; None on error.
        error (str | None): Why this article has no text, such as not being in Europe PMC or not
            being open access; None when it was retrieved.
    """

    model_config = ConfigDict(extra="forbid")

    pmcid: str = Field(title="PMCID", description="PubMed Central ID")
    pmid: str | None = Field(default=None, title="PMID", description="PubMed ID")
    doi: str | None = Field(default=None, title="DOI", description="Digital object identifier")
    title: str | None = Field(default=None, title="Title", description="Article title")
    license: str | None = Field(default=None, title="License", description="License URL or statement")
    sections: list[EuropePMCSection] = Field(
        default_factory=list, title="Sections", description="Text blocks in document order"
    )
    source_url: str | None = Field(
        default=None, title="Source URL", description="Full-text XML URL the text was read from"
    )
    error: str | None = Field(
        default=None, title="Error", description="Why this item has no data; None when it was retrieved"
    )

    @property
    def text(self) -> str:
        """The whole article as plain text, one block per section under its heading."""
        blocks = [
            f"## [{s.kind}] {s.heading}\n{s.text}" if s.heading else f"## [{s.kind}]\n{s.text}" for s in self.sections
        ]
        return "\n\n".join(blocks)


class EuropePMCFullTextOutput(BaseToolOutput):
    """Full texts, one per input PMCID.

    Attributes:
        documents (list[EuropePMCFullTextDocument]): Documents in input order.
        source_url (str): Europe PMC REST endpoint used.
    """

    documents: list[EuropePMCFullTextDocument] = Field(
        default_factory=list, title="Documents", description="Documents in input order"
    )
    source_url: str = Field(title="Source URL", description="Europe PMC REST endpoint used")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "txt"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        if file_format == "txt":
            with path.open("w", encoding="utf-8") as f:
                f.write(
                    "\n\n".join(
                        f"# {doc.pmcid} (error: {doc.error})"
                        if doc.error
                        else f"# {doc.pmcid} ({doc.source_url})\n\n{doc.text}"
                        for doc in self.documents
                    )
                )
                f.write("\n")
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return EuropePMCFullTextInput(pmcids=["PMC6034017"])


@tool(
    key="europepmc-fulltext",
    local_only="europepmc-fulltext does not use a gpu and does not need an environment, so it can run in process",
    label="Europe PMC Full Text",
    category="database_retrieval",
    input_class=EuropePMCFullTextInput,
    config_class=EuropePMCFullTextConfig,
    output_class=EuropePMCFullTextOutput,
    description=(
        "Fetch open-access article full text from Europe PMC by PMCID as plain-text sections, "
        "figure and table captions, and tables, to find and quote supporting passages"
    ),
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["pmcids"],
    iterable_output_field="documents",
    cacheable=True,
)
def run_europepmc_fulltext(
    inputs: EuropePMCFullTextInput,
    config: EuropePMCFullTextConfig,
    instance: Any = None,
) -> EuropePMCFullTextOutput:
    """Fetch and flatten the open-access full text of each article.

    Args:
        inputs (EuropePMCFullTextInput): PubMed Central IDs.
        config (EuropePMCFullTextConfig): Shared configuration only.
        instance (Any): Unused by this HTTP-only tool.

    Returns:
        EuropePMCFullTextOutput: One document per PMCID, in input order. A PMCID that is not in
            Europe PMC, is not open access, or has no full text gets an ``error`` and no sections.

    Raises:
        requests.HTTPError: If Europe PMC fails with a server error or keeps rate limiting.
    """
    del config, instance

    # fullTextXML answers every non-open-access PMCID with HTTP 500, so one search request
    # checks open access up front and only open-access articles are fetched.
    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )
    try:
        open_access = _open_access_status(inputs.pmcids, session)
        documents = [
            _document(pmcid, open_access.get(pmcid), session)
            for pmcid in StatusCounter(logger, "Fetching article %d/%d", inputs.pmcids)
        ]
        return EuropePMCFullTextOutput(documents=documents, source_url=_REST_URL)
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _open_access_status(pmcids: list[str], session: requests.Session) -> dict[str, bool]:
    """Map each PMCID Europe PMC holds to whether it is open access; unknown PMCIDs are absent."""
    unique = list(dict.fromkeys(pmcids))
    status: dict[str, bool] = {}
    for start in range(0, len(unique), _LOOKUP_BATCH_SIZE):
        for record in _lookup(unique[start : start + _LOOKUP_BATCH_SIZE], session):
            pmcid = record.get("pmcid")
            if pmcid:
                status[pmcid] = status.get(pmcid, False) or record.get("isOpenAccess") == "Y"
    return status


def _lookup(pmcids: list[str], session: requests.Session) -> list[dict[str, Any]]:
    """Return Europe PMC's ``lite`` search records for *pmcids* in one request."""
    params: dict[str, str | int] = {
        "query": " OR ".join(f"PMCID:{pmcid}" for pmcid in pmcids),
        "resultType": "lite",
        "format": "json",
        "pageSize": _LOOKUP_PAGE_SIZE,
    }
    response = request_with_retry(
        lambda: session.get(_SEARCH_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    raise_for_status_with_body(response, "Europe PMC open-access lookup")
    records: list[dict[str, Any]] = (response.json().get("resultList") or {}).get("result", [])
    return records


def _document(pmcid: str, open_access: bool | None, session: requests.Session) -> EuropePMCFullTextDocument:
    """Fetch an open-access article, or explain why *pmcid* has no full text."""
    if open_access is None:
        return EuropePMCFullTextDocument(pmcid=pmcid, error=f"{pmcid} is not in Europe PMC")
    if not open_access:
        return EuropePMCFullTextDocument(
            pmcid=pmcid,
            error=(
                f"{pmcid} is not open access; only open-access articles have full text "
                "(check is_open_access with europepmc-search)"
            ),
        )
    return _fetch(pmcid, session)


def _fetch(pmcid: str, session: requests.Session) -> EuropePMCFullTextDocument:
    """Download one article's JATS XML and parse it."""
    url = _FULLTEXT_URL.format(pmcid=pmcid)
    response = request_with_retry(
        lambda: session.get(url, timeout=_REQUEST_TIMEOUT_SECONDS),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    error = _item_error(response, f"Europe PMC full text for {pmcid}")
    if error:
        return EuropePMCFullTextDocument(pmcid=pmcid, error=error)
    return _parse_jats(response.content, pmcid=pmcid, source_url=url)


def _parse_jats(xml: bytes | str, pmcid: str, source_url: str) -> EuropePMCFullTextDocument:
    """Flatten a JATS article into plain-text sections.

    Args:
        xml (bytes | str): JATS XML of one article.
        pmcid (str): PubMed Central ID of the article.
        source_url (str): URL the XML was read from.

    Returns:
        EuropePMCFullTextDocument: The article's title, identifiers, license and sections.
    """
    root = ET.fromstring(xml)  # noqa: S314 -- expat refuses external entities and caps entity expansion
    meta = root.find("front/article-meta")
    ids = {} if meta is None else {el.get("pub-id-type"): (el.text or "").strip() for el in meta.findall("article-id")}
    title_el = None if meta is None else meta.find("title-group/article-title")
    title = _norm(_text(title_el)) if title_el is not None else ""

    sections: list[EuropePMCSection] = []
    if title:
        sections.append(EuropePMCSection(kind="title", text=title))
    for abstract in [] if meta is None else meta.findall("abstract"):
        kind_label = abstract.get("abstract-type")
        heading = f"Abstract ({kind_label})" if kind_label else "Abstract"
        _walk(abstract, [heading], "abstract", sections)
    for part in ("body", "back", "floats-group"):
        element = root.find(part)
        if element is not None:
            _walk(element, [], "section", sections)
    logger.debug("Parsed %s into %d sections", pmcid, len(sections))

    return EuropePMCFullTextDocument(
        pmcid=ids.get("pmcid") or pmcid,
        pmid=ids.get("pmid") or None,
        doi=ids.get("doi") or None,
        title=title,
        license=None if meta is None else _license(meta),
        sections=sections,
        source_url=source_url,
    )


def _walk(element: ET.Element, path: list[str], kind: SectionKind, out: list[EuropePMCSection]) -> None:
    """Emit one section for *element*'s own text, then its figures and tables, then its subsections."""
    title_el = element.find("title")
    heading = _norm(_text(title_el)) if title_el is not None else ""
    if heading and kind == "section":
        path = [*path, heading]
    paragraphs: list[str] = []
    floats: list[ET.Element] = []
    subsections: list[ET.Element] = []
    for child in element:
        tag = _local(child.tag)
        if tag in ("title", "label", "ref-list"):
            continue
        if tag in _SUBSECTIONS and kind == "section":
            subsections.append(child)
            continue
        floats.extend(el for el in child.iter() if _local(el.tag) in _FLOATS)
        if tag in _SKIPPED:
            continue
        text = _norm(_text(child))
        if text:
            paragraphs.append(text)
    if paragraphs:
        out.append(EuropePMCSection(kind=kind, heading=" > ".join(path) or None, text="\n".join(paragraphs)))
    out.extend(_float_section(float_el) for float_el in floats)
    for sub in subsections:
        _walk(sub, path, kind, out)


def _float_section(element: ET.Element) -> EuropePMCSection:
    """Turn a ``<fig>`` or ``<table-wrap>`` into a section: label, caption and, for tables, rows."""
    is_table = _local(element.tag) == "table-wrap"
    label_el = element.find("label")
    caption_el = element.find("caption")
    lines = [_norm(_text(caption_el))] if caption_el is not None else []
    if is_table:
        lines.extend(
            " | ".join(_norm(_text(cell)) for cell in row if _local(cell.tag) in ("td", "th"))
            for row in element.iter("tr")
        )
        foot = element.find("table-wrap-foot")
        if foot is not None:
            lines.append(_norm(_text(foot)))
    return EuropePMCSection(
        kind="table" if is_table else "figure",
        heading=(_norm(_text(label_el)) if label_el is not None else None) or element.get("id"),
        text="\n".join(line for line in lines if line),
    )


def _text(element: ET.Element) -> str:
    """Return an element's running text, skipping floats and graphics and one-of-many alternatives."""
    parts = [element.text or ""]
    for child in element:
        tag = _local(child.tag)
        if tag == "alternatives":
            parts.append(_text(_preferred_alternative(child)) if len(child) else "")
        elif tag not in _SKIPPED:
            inner = _text(child)
            parts.append(f" {inner} " if tag in _BLOCKS else inner)
        parts.append(child.tail or "")
    return "".join(parts)


def _preferred_alternative(element: ET.Element) -> ET.Element:
    """Pick one rendering from ``<alternatives>``: MathML, else a table, else the first child."""
    for wanted in ("math", "table", "tex-math"):
        for child in element:
            if _local(child.tag) == wanted:
                return child
    return element[0]


def _license(meta: ET.Element) -> str | None:
    """Return the license URL, else the license statement, from ``<permissions>``."""
    license_el = meta.find("permissions/license")
    if license_el is None:
        return None
    href = license_el.get(_XLINK_HREF)
    if href:
        return href
    for ext in license_el.iter("ext-link"):
        if ext.get(_XLINK_HREF):
            return ext.get(_XLINK_HREF)
    return _norm(_text(license_el)) or None


def _local(tag: Any) -> str:
    """Strip an XML namespace from a tag name."""
    return str(tag).rsplit("}", 1)[-1]


def _norm(text: str) -> str:
    """Collapse whitespace."""
    return " ".join(text.split())

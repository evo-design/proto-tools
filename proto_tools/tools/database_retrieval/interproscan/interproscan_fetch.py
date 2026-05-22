"""proto_tools/tools/database_retrieval/interproscan/interproscan_fetch.py.

Fetches protein domain annotations from InterPro — by UniProt accession via
the InterPro REST API (direct lookup) or by raw protein sequence via EBI's
iprscan5 submit-and-poll service. Both paths converge on a unified
``InterProDomain`` row schema covering Pfam, SMART, PROSITE, Gene3D /
CATH-Gene3D, Panther, and the rest of the InterPro member-DB catalog.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
    extract_text_status,
    poll_until_complete,
)

logger = logging.getLogger(__name__)

_INTERPRO_API_BASE = "https://www.ebi.ac.uk/interpro/api"
_IPRSCAN5_BASE = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
_REQUEST_TIMEOUT_SECONDS = 30
_RESULT_TIMEOUT_SECONDS = 120
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/interproscan-fetch-v1"
_IPRSCAN5_SUCCESS_STATES = frozenset({"FINISHED"})
_IPRSCAN5_FAILURE_STATES = frozenset({"ERROR", "FAILURE", "NOT_FOUND"})
_PAGE_SIZE = 200  # InterPro REST caps `?page_size` at 200; always request the max.
_DIRECT_LOOKUP_MAX_PAGES = 50  # 100x headroom over the largest real protein; bounds corrupted-cursor loops.
# Matches EBI's reference iprscan5 Python client (pollFreq=3); 30-min cap.
_IPRSCAN5_POLL_INTERVAL_SECONDS = 3.0
_IPRSCAN5_TIMEOUT_SECONDS = 1800.0


# Casing matches EBI's iprscan5 parameterdetails endpoint exactly: PfamA
# (not Pfam), Gene3d (not Gene3D). Validation of these strings happens at
# Pydantic-parse time, before they reach the server, so typos surface as
# pydantic ValidationError rather than HTTP 400.
InterProApp = Literal[
    "PfamA",
    "Panther",
    "Gene3d",
    "SuperFamily",
    "SMART",
    "PrositeProfiles",
    "PrositePatterns",
    "PRINTS",
    "PIRSF",
    "FunFam",
    "HAMAP",
    "CDD",
    "NCBIfam",
    "SFLD",
    "Coils",
    "MobiDBLite",
    "Phobius",
    "SignalP",
    "SignalP_EUK",
    "SignalP_GRAM_POSITIVE",
    "SignalP_GRAM_NEGATIVE",
    "AntiFam",
    "PIRSR",
    "TMHMM",
]

InterProDomainType = Literal[
    "family",
    "domain",
    "repeat",
    "active_site",
    "conserved_site",
    "homologous_superfamily",
    "binding_site",
    "ptm",
    "unknown",
]

# Mapping from InterPro's published `type` strings (lower-cased) to the
# canonical InterProDomainType labels. Values not seen here fall back to
# "unknown" so the wrapper never blows up on a new InterPro vocabulary entry.
_TYPE_MAP: dict[str, InterProDomainType] = {
    "family": "family",
    "domain": "domain",
    "repeat": "repeat",
    "active_site": "active_site",
    "active site": "active_site",
    "conserved_site": "conserved_site",
    "conserved site": "conserved_site",
    "homologous_superfamily": "homologous_superfamily",
    "homologous superfamily": "homologous_superfamily",
    "binding_site": "binding_site",
    "binding site": "binding_site",
    "ptm": "ptm",
}


# ============================================================================
# Data Models
# ============================================================================


class InterProDomain(BaseModel):
    """One InterPro hit row, sourced from a single member-DB match.

    Attributes:
        accession (str): Member-DB accession (e.g. ``"PF00870"``,
            ``"IPR011615"``, ``"G3DSA:1.10.10.10"``).
        name (str): Human-readable domain / family name.
        type (InterProDomainType): Category — ``family``, ``domain``,
            ``repeat``, ``active_site``, ``conserved_site``,
            ``homologous_superfamily``, ``binding_site``, ``ptm``, or
            ``unknown``.
        member_database (str): Source database (``"pfam"``, ``"panther"``,
            ``"cathgene3d"``, …).
        integrated_ipr (str | None): Parent InterPro accession; ``None``
            when the member-DB hit is not yet integrated.
        start (int): 1-indexed inclusive start residue.
        end (int): 1-indexed inclusive end residue.
        score (float | None): Per-DB score — e-value or bit-score depending
            on the member database. ``None`` when not reported.
        model (str | None): Underlying HMM / profile / model identifier.
        representative (bool): Whether this is InterPro's representative
            match for the protein (one per parent IPR entry).
        go_terms (list[str]): GO term IDs cross-referenced from this entry.
        pathways (list[str]): Pathway IDs (Reactome, MetaCyc, …)
            cross-referenced from this entry.
    """

    model_config = ConfigDict(extra="forbid")

    accession: str = Field(title="Accession", description="Member-DB accession (PfamID, IPR-ID, G3DSA-ID, ...)")
    name: str = Field(title="Name", description="Human-readable domain/family name")
    type: InterProDomainType = Field(title="Type", description="Category (family, domain, repeat, ...)")
    member_database: str = Field(
        title="Member Database", description="Source DB ('pfam', 'panther', 'cathgene3d', ...)"
    )
    integrated_ipr: str | None = Field(
        default=None,
        title="Parent InterPro ID",
        description="Parent InterPro accession; None for non-integrated member-DB hits",
    )
    start: int = Field(title="Start", description="1-indexed inclusive start residue", ge=1)
    end: int = Field(title="End", description="1-indexed inclusive end residue", ge=1)
    score: float | None = Field(
        default=None,
        title="Score",
        description="Per-database score: e-value (smaller is better) for HMM DBs, otherwise bit score (larger is better)",
    )
    model: str | None = Field(default=None, title="Model", description="HMM/profile model ID")
    representative: bool = Field(
        default=False, title="Representative", description="Whether this is the representative match"
    )
    go_terms: list[str] = Field(
        default_factory=list,
        title="GO Terms",
        description="GO term IDs cross-referenced from this entry",
    )
    pathways: list[str] = Field(
        default_factory=list, title="Pathways", description="Pathway IDs (Reactome, MetaCyc, ...)"
    )


class InterProScanFetchInput(BaseToolInput):
    """Input for InterPro fetch.

    Provide exactly one of ``uniprot_id`` (direct REST lookup) or
    ``sequence`` (submit-and-poll via iprscan5). Both paths return the
    same ``InterProDomain`` row schema.

    Attributes:
        uniprot_id (str | None): UniProt accession for direct entry lookup
            against ``interpro/api/entry/all/protein/uniprot/{acc}/``.
        sequence (str | None): Raw protein sequence for the iprscan5
            submit-and-scan path. Requires ``config.email``.
    """

    uniprot_id: str | None = InputField(
        default=None,
        title="UniProt Accession",
        description="UniProt accession for direct entry lookup",
    )
    sequence: str | None = InputField(
        default=None, title="Sequence", description="Protein sequence for submit-and-scan path"
    )

    @model_validator(mode="after")
    def validate_lookup_params(self) -> "InterProScanFetchInput":
        """Require exactly one of uniprot_id, sequence (after stripping whitespace)."""
        uid = (self.uniprot_id or "").strip()
        seq = (self.sequence or "").strip()
        if not uid and not seq:
            raise ValueError("Provide either uniprot_id or sequence")
        if uid and seq:
            raise ValueError("Provide exactly one of uniprot_id or sequence")
        return self


class InterProScanFetchConfig(BaseConfig):
    """Configuration for InterPro fetch.

    Attributes:
        email (str | None): Required by EBI's iprscan5 endpoint when
            submitting a sequence; ignored on the direct UniProt-lookup
            path. Defaults to the ``INTERPROSCAN_EMAIL`` environment
            variable; an explicit value passed to the config overrides
            the env var.
        applications (list[InterProApp] | None): Submit-only — restrict
            iprscan5 to a subset of member databases. ``None`` runs the
            EBI default set (every application enabled, matching upstream
            ``appl[]`` defaults).
        include_go_terms (bool): Include GO term cross-references in the
            output. Maps to iprscan5's ``goterms`` form param on the
            submit path; filters parser output on the direct path.
        include_pathways (bool): Fetch Reactome/KEGG/MetaCyc pathway
            cross-references after an iprscan5 sequence submission. Has
            no effect on the UniProt-id path — InterPro's UniProt-keyed
            endpoint does not return pathway data, so this stays empty
            on that path regardless of the flag.
        sequence_type (Literal['protein', 'nucleic']): Submit-only —
            ``nucleic`` tells iprscan5 to 6-frame translate the input.
    """

    email: str | None = ConfigField(
        title="Contact Email",
        default_factory=lambda: os.environ.get("INTERPROSCAN_EMAIL"),
        description="EBI contact email for the sequence-submit path. Defaults to the INTERPROSCAN_EMAIL env var.",
        include_in_key=False,
    )
    applications: list[InterProApp] | None = ConfigField(
        title="Applications",
        default=None,
        description="Submit-only — restrict to subset of InterPro member DBs; None runs the EBI default set",
    )
    include_go_terms: bool = ConfigField(
        title="Include GO Terms", default=True, description="Include GO term cross-references in the output"
    )
    include_pathways: bool = ConfigField(
        title="Include Pathways",
        default=True,
        description="Reactome/KEGG/MetaCyc xrefs on the sequence path; no-op on UniProt-id path",
    )
    sequence_type: Literal["protein", "nucleic"] = ConfigField(
        title="Sequence Type",
        default="protein",
        description="Sequence-submit path: 'protein' or 'nucleic' (6-frame translated server-side)",
    )


class InterProScanFetchOutput(BaseToolOutput):
    """Output from InterPro fetch.

    Attributes:
        accession (str | None): Resolved UniProt accession; ``None`` when
            the sequence path returns a result without a UniProt
            cross-reference.
        sequence_length (int | None): Length of the queried protein.
        domains (list[InterProDomain]): All hits across all member
            databases, in the order returned by the API.
        num_domains (int): ``len(domains)``.
        job_id (str): iprscan5 job ID for the submit path; empty string
            for the direct-lookup path.
        source_url (str): Canonical InterPro entry URL for the resolved
            accession (or the iprscan5 result URL on the sequence path).
        raw_entries (list[dict[str, Any]]): Raw API JSON entries — one
            per InterPro entry on the direct path, one per match on the
            sequence path — for advanced consumers.
    """

    accession: str | None = Field(default=None, title="UniProt Accession", description="Resolved UniProt accession")
    sequence_length: int | None = Field(default=None, title="Sequence Length", description="Queried protein length")
    domains: list[InterProDomain] = Field(
        default_factory=list, title="Domains", description="InterPro hits across all member DBs"
    )
    num_domains: int = Field(title="Number of Domains", description="Total number of hits", ge=0)
    job_id: str = Field(title="Job ID", description="iprscan5 job ID for the sequence path (empty for direct lookup)")
    source_url: str = Field(title="Source URL", description="InterPro entry / iprscan5 result URL")
    raw_entries: list[dict[str, Any]] = Field(
        default_factory=list, title="Raw Entries", description="Raw API JSON entries"
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

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
        if file_format == "csv":
            # One row per domain hit; nested fields (locations, go_terms,
            # pathway_xrefs) are JSON-encoded into single cells.
            rows: list[dict[str, Any]] = []
            for d in self.domains:
                row = d.model_dump()
                for k, v in list(row.items()):
                    if isinstance(v, (list, dict)):
                        row[k] = json.dumps(v, separators=(",", ":"))
                rows.append({"uniprot_accession": self.accession, **row})
            with path.open("w", encoding="utf-8", newline="") as f:
                if not rows:
                    return
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return InterProScanFetchInput(uniprot_id="P04637")


@tool(
    key="interproscan-fetch",
    label="InterProScan Fetch",
    category="database_retrieval",
    input_class=InterProScanFetchInput,
    config_class=InterProScanFetchConfig,
    output_class=InterProScanFetchOutput,
    description=(
        "Fetch InterPro domain annotations by UniProt accession (direct REST lookup) "
        "or by raw protein sequence (iprscan5 submit-and-scan)"
    ),
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_interproscan_fetch(
    inputs: InterProScanFetchInput,
    config: InterProScanFetchConfig,
    instance: Any = None,
) -> InterProScanFetchOutput:
    """Fetch InterPro domain annotations.

    Dispatches to the direct lookup path when ``inputs.uniprot_id`` is set,
    or the iprscan5 submit-and-poll path when ``inputs.sequence`` is set.
    The input validator guarantees exactly one is populated.

    Args:
        inputs (InterProScanFetchInput): UniProt accession or protein sequence.
        config (InterProScanFetchConfig): Optional fetch knobs (email,
            applications, paging, polling).
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        InterProScanFetchOutput: Domain hits across all InterPro member
            databases, plus the source URL and (for sequence-path) the
            iprscan5 job ID.

    Raises:
        ValueError: When the sequence path is requested without
            ``config.email`` set, or when the upstream API returns no
            data for the requested accession.
    """
    del instance

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        allowed_methods=["GET", "POST"],
    )

    try:
        if inputs.uniprot_id:
            return _direct_lookup(inputs.uniprot_id, config, session)
        sequence = inputs.sequence  # validator guarantees this is not None when uniprot_id is None
        email = config.email
        if email is None:
            raise ValueError(
                "config.email is required when input is a sequence (EBI iprscan5 mandates a contact email)"
            )
        if sequence is None:
            raise ValueError("Provide either uniprot_id or sequence")
        return _submit_and_poll(sequence, email, config, session)
    finally:
        session.close()


# ============================================================================
# Direct Lookup (UniProt accession → InterPro REST)
# ============================================================================


def _direct_lookup(
    uniprot_id: str,
    config: InterProScanFetchConfig,
    session: requests.Session,
) -> InterProScanFetchOutput:
    """Fetch InterPro entries for a UniProt accession via the REST API."""
    accession = uniprot_id.strip().upper()
    base_url = f"{_INTERPRO_API_BASE}/entry/all/protein/uniprot/{accession}/"
    next_url: str | None = f"{base_url}?page_size={_PAGE_SIZE}"

    domains: list[InterProDomain] = []
    raw_entries: list[dict[str, Any]] = []
    sequence_length: int | None = None
    pages_seen = 0
    seen_urls: set[str] = set()

    while next_url:
        if pages_seen >= _DIRECT_LOOKUP_MAX_PAGES:
            raise ValueError(
                f"InterPro pagination exceeded {_DIRECT_LOOKUP_MAX_PAGES} pages for '{accession}'; "
                "upstream `next` cursor may be corrupted"
            )
        if next_url in seen_urls:
            raise ValueError(f"InterPro pagination cursor revisited URL {next_url!r} for '{accession}'")
        seen_urls.add(next_url)
        response = session.get(next_url, timeout=_REQUEST_TIMEOUT_SECONDS)
        # 204 / 404 = "unknown accession" (InterPro's actual signals).
        if response.status_code in (204, 404):
            raise ValueError(f"InterPro has no entries for UniProt accession '{accession}'")
        # Surface other 4xx/5xx as HTTPError before checking for empty bodies, so a
        # 5xx with an empty body doesn't get misreported as "no entries".
        response.raise_for_status()
        if not response.text.strip():
            raise ValueError(f"InterPro has no entries for UniProt accession '{accession}'")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                f"InterPro returned non-JSON for '{accession}' at {next_url}; body[:200]={response.text[:200]!r}"
            ) from exc
        for result in payload.get("results", []):
            raw_entries.append(result)
            entry_domains, entry_sequence_length = _parse_direct_entry(
                result,
                include_go_terms=config.include_go_terms,
            )
            domains.extend(entry_domains)
            if entry_sequence_length is not None and sequence_length is None:
                sequence_length = entry_sequence_length
        next_url = payload.get("next") or None
        pages_seen += 1

    logger.debug(
        "interproscan-fetch direct lookup: %s → %d domains across %d pages", accession, len(domains), pages_seen
    )

    return InterProScanFetchOutput(
        accession=accession,
        sequence_length=sequence_length,
        domains=domains,
        num_domains=len(domains),
        job_id="",
        source_url=f"https://www.ebi.ac.uk/interpro/protein/UniProt/{accession}/",
        raw_entries=raw_entries,
    )


def _parse_direct_entry(
    result: dict[str, Any],
    *,
    include_go_terms: bool,
) -> "tuple[list[InterProDomain], int | None]":
    """Parse one direct-lookup result into domain rows + protein length."""
    metadata = result.get("metadata", {})
    proteins = result.get("proteins", [])
    if not metadata or not proteins:
        return [], None

    accession = str(metadata.get("accession", "")).upper()
    name = str(metadata.get("name", ""))
    type_raw = str(metadata.get("type", "")).strip().lower()
    domain_type: InterProDomainType = _TYPE_MAP.get(type_raw, "unknown")
    member_database = str(metadata.get("source_database", ""))
    integrated_raw = metadata.get("integrated")
    integrated_ipr: str | None = None
    if isinstance(integrated_raw, str) and integrated_raw.strip():
        integrated_ipr = integrated_raw.strip().upper()
    go_terms = _extract_xref_ids(metadata.get("go_terms")) if include_go_terms else []
    # Direct UniProt-lookup endpoint never surfaces pathway xrefs; iprscan5 path does.
    pathways: list[str] = []

    protein = proteins[0]
    sequence_length = protein.get("protein_length")
    sequence_length_int: int | None = int(sequence_length) if isinstance(sequence_length, int | float) else None

    domains: list[InterProDomain] = []
    for location in protein.get("entry_protein_locations") or []:
        score_raw = location.get("score")
        score = float(score_raw) if isinstance(score_raw, int | float) else None
        model = location.get("model")
        model_str = str(model) if isinstance(model, str) else None
        representative = bool(location.get("representative", False))
        domains.extend(
            InterProDomain(
                accession=accession,
                name=name,
                type=domain_type,
                member_database=member_database,
                integrated_ipr=integrated_ipr,
                start=int(fragment["start"]),
                end=int(fragment["end"]),
                score=score,
                model=model_str,
                representative=representative,
                go_terms=list(go_terms),
                pathways=list(pathways),
            )
            for fragment in location.get("fragments") or []
        )
    return domains, sequence_length_int


def _extract_xref_ids(raw: Any) -> list[str]:
    """Extract identifier strings from a cross-reference list.

    Tolerates the union of shapes both InterPro REST and iprscan5 use for
    GO/pathway cross-references: list-of-dicts keyed by ``id`` or
    ``identifier``, or a list of plain strings. Drops empty / missing
    values silently.

    Args:
        raw (Any): Potential xref list — accepts ``None``, non-list, or
            heterogeneous list contents without crashing.

    Returns:
        list[str]: Stripped, non-empty identifiers in source order.
    """
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            value = item.get("id") or item.get("identifier")
            if isinstance(value, str) and value.strip():
                ids.append(value.strip())
        elif isinstance(item, str) and item.strip():
            ids.append(item.strip())
    return ids


# ============================================================================
# Submit-and-Poll (raw protein sequence → iprscan5)
# ============================================================================


def _submit_and_poll(
    sequence: str,
    email: str,
    config: InterProScanFetchConfig,
    session: requests.Session,
) -> InterProScanFetchOutput:
    """Submit a sequence to iprscan5, poll to FINISHED, and parse the JSON result."""
    job_id = _submit_iprscan(sequence, email, config, session)
    poll_until_complete(
        session,
        f"{_IPRSCAN5_BASE}/status/{job_id}",
        poll_interval_seconds=_IPRSCAN5_POLL_INTERVAL_SECONDS,
        timeout_seconds=_IPRSCAN5_TIMEOUT_SECONDS,
        success_states=_IPRSCAN5_SUCCESS_STATES,
        failure_states=_IPRSCAN5_FAILURE_STATES,
        status_extractor=extract_text_status,
    )
    result_url = f"{_IPRSCAN5_BASE}/result/{job_id}/json"
    response = session.get(result_url, timeout=_RESULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(
            f"iprscan5 returned non-JSON for job {job_id} at {result_url}; body[:200]={response.text[:200]!r}"
        ) from exc
    return _parse_iprscan_payload(payload, job_id, result_url, config)


def _submit_iprscan(
    sequence: str,
    email: str,
    config: InterProScanFetchConfig,
    session: requests.Session,
) -> str:
    """POST a sequence to iprscan5/run/ and return the plain-text job ID."""
    data: list[tuple[str, str]] = [
        ("email", email),
        ("sequence", sequence),
        ("stype", "p" if config.sequence_type == "protein" else "n"),
        ("goterms", "true" if config.include_go_terms else "false"),
        ("pathways", "true" if config.include_pathways else "false"),
    ]
    if config.applications:
        data.extend(("appl", app) for app in config.applications)
    response = session.post(f"{_IPRSCAN5_BASE}/run/", data=data, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    job_id = response.text.strip()
    if not job_id:
        raise ValueError(f"iprscan5 submit returned an empty job ID; full response: {response.text!r}")
    return job_id


def _parse_iprscan_payload(
    payload: dict[str, Any],
    job_id: str,
    result_url: str,
    config: InterProScanFetchConfig,
) -> InterProScanFetchOutput:
    """Flatten an iprscan5 JSON result into the unified Output schema.

    The iprscan5 JSON ships ``results: [{sequence, sequenceLength, xref,
    matches: [...]}]`` — one element per submitted sequence. Each match
    has a ``signature`` block (the source InterPro / member-DB entry) and
    a ``locations`` list whose start/end fields populate the row.
    """
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"iprscan5 returned no results at {result_url}")

    first = results[0]
    sequence_length = first.get("sequenceLength")
    sequence_length_int: int | None = int(sequence_length) if isinstance(sequence_length, int | float) else None
    accession = _extract_iprscan_accession(first)

    domains: list[InterProDomain] = []
    raw_matches: list[dict[str, Any]] = []
    for match in first.get("matches") or []:
        raw_matches.append(match)
        signature = match.get("signature") or {}
        signature_library = signature.get("signatureLibraryRelease") or {}
        member_database = str(signature_library.get("library", "")).lower()
        accession_match = str(signature.get("accession", "")).upper()
        name = str(signature.get("name") or signature.get("description") or "")
        entry = signature.get("entry") or {}
        integrated_ipr_raw = entry.get("accession") if isinstance(entry, dict) else None
        integrated_ipr: str | None = None
        if isinstance(integrated_ipr_raw, str) and integrated_ipr_raw.strip():
            integrated_ipr = integrated_ipr_raw.strip().upper()
        type_raw = str(entry.get("type", "")).strip().lower() if isinstance(entry, dict) else ""
        domain_type: InterProDomainType = _TYPE_MAP.get(type_raw, "unknown")
        model = match.get("model-ac")
        model_str = str(model) if isinstance(model, str) else None
        go_terms = (
            _extract_xref_ids(entry.get("goXRefs") if isinstance(entry, dict) else None)
            if config.include_go_terms
            else []
        )
        pathways = (
            _extract_xref_ids(entry.get("pathwayXRefs") if isinstance(entry, dict) else None)
            if config.include_pathways
            else []
        )

        for location in match.get("locations") or []:
            score_raw = location.get("evalue") if location.get("evalue") is not None else location.get("score")
            score = float(score_raw) if isinstance(score_raw, int | float) else None
            domains.append(
                InterProDomain(
                    accession=accession_match,
                    name=name,
                    type=domain_type,
                    member_database=member_database,
                    integrated_ipr=integrated_ipr,
                    start=int(location["start"]),
                    end=int(location["end"]),
                    score=score,
                    model=model_str,
                    representative=False,
                    go_terms=list(go_terms),
                    pathways=list(pathways),
                )
            )

    return InterProScanFetchOutput(
        accession=accession,
        sequence_length=sequence_length_int,
        domains=domains,
        num_domains=len(domains),
        job_id=job_id,
        source_url=result_url,
        raw_entries=raw_matches,
    )


def _extract_iprscan_accession(result: dict[str, Any]) -> str | None:
    """Pull a UniProt-style accession from an iprscan5 result's xref list, if any."""
    xrefs = result.get("xref") or []
    for xref in xrefs:
        if not isinstance(xref, dict):
            continue
        identifier = xref.get("id") or xref.get("name")
        if isinstance(identifier, str) and identifier.strip():
            return identifier.strip().upper()
    return None

"""proto_tools/tools/database_retrieval/ensembl/shared_data_models.py.

Pydantic submodels + HTTP helpers shared by ensembl-fetch and ensembl-vep.
"""

from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from proto_tools.utils import build_http_session

_BASE_URL: dict[str, str] = {
    "GRCh38": "https://rest.ensembl.org",
    "GRCh37": "https://grch37.rest.ensembl.org",
}
_HTTP_RETRIES = 3
_BACKOFF_SECONDS = 1.0


EnsemblAssembly = Literal["GRCh38", "GRCh37"]
EnsemblSpecies = Literal[
    "homo_sapiens",
    "mus_musculus",
    "rattus_norvegicus",
    "danio_rerio",
    "saccharomyces_cerevisiae",
]


def build_session(tool_key: str) -> requests.Session:
    """Build an Ensembl-friendly HTTP session.

    Args:
        tool_key (str): Tool registry key, used in the User-Agent string.

    Returns:
        requests.Session: Session with retry adapter; urllib3's ``Retry()``
            honors ``Retry-After`` by default so Ensembl's 429s sleep for
            the server-directed interval.
    """
    return build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=f"proto-tools/{tool_key}-v1",
    )


def base_url_for(assembly: EnsemblAssembly) -> str:
    """Return the Ensembl REST base URL for the given assembly."""
    return _BASE_URL[assembly]


# ============================================================================
# Nested payload submodels (used by ensembl-fetch lookup_id / lookup_symbol)
# ============================================================================


class EnsemblExon(BaseModel):
    """One exon record returned by Ensembl REST.

    Attributes:
        id (str): Ensembl exon ID (ENSE...).
        seq_region_name (str): Chromosome / contig name (e.g. '17').
        start (int): 1-indexed inclusive start coordinate.
        end (int): 1-indexed inclusive end coordinate.
        strand (int): +1 or -1.
        assembly_name (str): Genome assembly name (e.g. 'GRCh38').
        version (int | None): Ensembl version of this exon.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    seq_region_name: str
    start: int
    end: int
    strand: int
    assembly_name: str
    version: int | None = None


class EnsemblTranslation(BaseModel):
    """Protein translation record returned by Ensembl REST.

    Attributes:
        id (str): Ensembl protein ID (ENSP...).
        start (int): Translation start in transcript coordinates.
        end (int): Translation end in transcript coordinates.
        length (int): Protein length in amino acids.
        Parent (str): Parent transcript ID (PascalCase preserved from API).
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    start: int
    end: int
    length: int
    Parent: str


class EnsemblTranscript(BaseModel):
    """Transcript record returned by Ensembl REST (with optional Exon / Translation expansion).

    Attributes:
        id (str): Ensembl transcript ID (ENST...).
        display_name (str | None): Human-readable transcript name.
        biotype (str): Transcript biotype (protein_coding, lncRNA, ...).
        is_canonical (bool): Whether this is the canonical transcript.
        start (int): 1-indexed inclusive start in genome coords.
        end (int): 1-indexed inclusive end in genome coords.
        strand (int): +1 or -1.
        seq_region_name (str): Chromosome / contig name.
        assembly_name (str): Genome assembly name.
        Exon (list[EnsemblExon]): Exons (PascalCase preserved from API).
        Translation (EnsemblTranslation | None): Translation block (PascalCase
            preserved from API); None for non-coding biotypes.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    display_name: str | None = None
    biotype: str
    is_canonical: bool = False
    start: int
    end: int
    strand: int
    seq_region_name: str
    assembly_name: str
    Exon: list[EnsemblExon] = Field(default_factory=list)
    Translation: EnsemblTranslation | None = None


class EnsemblGene(BaseModel):
    """Gene record returned by Ensembl REST lookup endpoints.

    Attributes:
        id (str): Ensembl gene ID (ENSG...).
        display_name (str | None): Human-readable gene symbol (e.g. 'BRCA1').
        description (str | None): Free-text description.
        biotype (str): Gene biotype (protein_coding, lncRNA, ...).
        species (str): Species slug (e.g. 'homo_sapiens').
        seq_region_name (str): Chromosome / contig name.
        start (int): 1-indexed inclusive genomic start.
        end (int): 1-indexed inclusive genomic end.
        strand (int): +1 or -1.
        assembly_name (str): Genome assembly name.
        canonical_transcript (str | None): Canonical transcript ID with version.
        Transcript (list[EnsemblTranscript]): Transcripts (PascalCase preserved
            from API); empty when ``expand=False``.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    display_name: str | None = None
    description: str | None = None
    biotype: str
    species: str
    seq_region_name: str
    start: int
    end: int
    strand: int
    assembly_name: str
    canonical_transcript: str | None = None
    Transcript: list[EnsemblTranscript] = Field(default_factory=list)


class EnsemblSequence(BaseModel):
    """Sequence record returned by /sequence/id/{id}.

    Attributes:
        id (str): Ensembl ID the sequence was fetched for.
        desc (str | None): Description string returned by the server.
        mol_type (str | None): Molecule type ('dna' / 'protein' / ...).
        seq (str): The raw sequence string.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    desc: str | None = None
    mol_type: str | None = None
    seq: str


class EnsemblXref(BaseModel):
    """One cross-reference returned by /xrefs/id or /xrefs/symbol.

    Attributes:
        dbname (str): External database name (e.g. 'Uniprot_gn', 'EntrezGene').
        db_display_name (str | None): Human-readable DB name.
        display_id (str): Display identifier in the external DB.
        primary_id (str): Primary identifier in the external DB.
        description (str | None): External-DB description.
        info_type (str | None): Cross-reference type ('DIRECT', 'DEPENDENT', ...).
    """

    model_config = ConfigDict(extra="ignore")

    dbname: str
    db_display_name: str | None = None
    display_id: str
    primary_id: str
    description: str | None = None
    info_type: str | None = None


class EnsemblOverlapFeatureRecord(BaseModel):
    """One feature returned by /overlap/id.

    Different ``feature`` query values (gene, regulatory, variation, ...)
    return divergent payload shapes; the typed fields below are the
    intersection across feature types, with ``raw`` carrying the full
    upstream record for type-specific access.

    Attributes:
        feature_type (str): Feature type ('gene', 'transcript', 'exon',
            'regulatory', 'motif', 'variation', ...).
        id (str | None): Feature ID where the API returns one.
        biotype (str | None): Biotype where applicable.
        start (int): 1-indexed inclusive genomic start.
        end (int): 1-indexed inclusive genomic end.
        strand (int): +1, -1, or 0 for unstranded features.
        seq_region_name (str): Chromosome / contig name.
        raw (dict[str, Any]): Full upstream record for feature-specific fields.
    """

    model_config = ConfigDict(extra="ignore")

    feature_type: str
    id: str | None = None
    biotype: str | None = None
    start: int
    end: int
    strand: int
    seq_region_name: str
    raw: dict[str, Any] = Field(default_factory=dict)

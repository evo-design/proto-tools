"""proto_tools/tools/database_retrieval/ensembl/shared_data_models.py.

Pydantic submodels + HTTP helpers shared by the ensembl-* tools (lookup,
sequence, overlap, xrefs, vep).
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
EnsemblSequenceType = Literal["genomic", "cdna", "cds", "protein"]
EnsemblOverlapFeature = Literal[
    "band",
    "gene",
    "transcript",
    "cds",
    "exon",
    "repeat",
    "simple",
    "misc",
    "variation",
    "somatic_variation",
    "structural_variation",
    "somatic_structural_variation",
    "constrained",
    "regulatory",
    "motif",
    "mane",
]

# Ensembl gene/transcript biotypes — canonical set from
# https://www.ensembl.org/info/genome/genebuild/biotypes.html
# Stable across releases; protein_coding / lncRNA / pseudogene families are
# the most common filters.
EnsemblBiotype = Literal[
    "protein_coding",
    "lncRNA",
    "lincRNA",  # legacy alias still served on older releases
    "miRNA",
    "snRNA",
    "snoRNA",
    "rRNA",
    "Mt_rRNA",
    "Mt_tRNA",
    "tRNA",
    "misc_RNA",
    "ribozyme",
    "scaRNA",
    "sRNA",
    "vault_RNA",
    "Y_RNA",
    "antisense_RNA",
    "non_coding",
    "processed_transcript",
    "pseudogene",
    "processed_pseudogene",
    "transcribed_processed_pseudogene",
    "transcribed_unitary_pseudogene",
    "transcribed_unprocessed_pseudogene",
    "unitary_pseudogene",
    "unprocessed_pseudogene",
    "translated_processed_pseudogene",
    "translated_unprocessed_pseudogene",
    "TR_C_gene",
    "TR_D_gene",
    "TR_J_gene",
    "TR_V_gene",
    "TR_J_pseudogene",
    "TR_V_pseudogene",
    "IG_C_gene",
    "IG_D_gene",
    "IG_J_gene",
    "IG_V_gene",
    "IG_C_pseudogene",
    "IG_J_pseudogene",
    "IG_V_pseudogene",
    "IG_pseudogene",
    "TEC",
    "nonsense_mediated_decay",
    "non_stop_decay",
    "retained_intron",
]

# Sequence Ontology consequence terms used by Ensembl VEP and the overlap
# variation filter. Source: https://www.ensembl.org/info/genome/variation/prediction/predicted_data.html
EnsemblSOTerm = Literal[
    "transcript_ablation",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
    "start_lost",
    "transcript_amplification",
    "feature_elongation",
    "feature_truncation",
    "inframe_insertion",
    "inframe_deletion",
    "missense_variant",
    "protein_altering_variant",
    "splice_region_variant",
    "splice_donor_5th_base_variant",
    "splice_donor_region_variant",
    "splice_polypyrimidine_tract_variant",
    "incomplete_terminal_codon_variant",
    "start_retained_variant",
    "stop_retained_variant",
    "synonymous_variant",
    "coding_sequence_variant",
    "mature_miRNA_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "intron_variant",
    "NMD_transcript_variant",
    "non_coding_transcript_variant",
    "coding_transcript_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "TFBS_ablation",
    "TFBS_amplification",
    "TF_binding_site_variant",
    "regulatory_region_ablation",
    "regulatory_region_amplification",
    "regulatory_region_variant",
    "intergenic_variant",
    "sequence_variant",
]

# Common Ensembl external-DB names accepted by /xrefs/id?external_db=...
# Source: live API at /xrefs/id with no filter. Not exhaustive — there are
# ~50 valid values across all dbs/assemblies; this covers the 27 most-used.
EnsemblExternalDB = Literal[
    "UniProtKB/Swiss-Prot",
    "UniProtKB/TrEMBL",
    "UniProtKB_all",
    "HGNC",
    "MGI",  # mouse
    "RGD",  # rat
    "ZFIN",  # zebrafish
    "EntrezGene",
    "RefSeq_mRNA",
    "RefSeq_mRNA_predicted",
    "RefSeq_ncRNA",
    "RefSeq_ncRNA_predicted",
    "RefSeq_peptide",
    "RefSeq_peptide_predicted",
    "RefSeq_genomic",
    "CCDS",
    "Pfam",
    "InterPro",
    "GO",
    "PDB",
    "STRING",
    "Reactome",
    "ArrayExpress",
    "WikiGene",
    "MIM_GENE",
    "MIM_MORBID",
    "ClinVar",
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
# Nested payload submodels (used by ensembl-lookup)
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

    id: str = Field(title="Exon ID", description="Stable Ensembl exon identifier (ENSE...)")
    seq_region_name: str = Field(title="Sequence Region", description="Chromosome or contig name (e.g. '17')")
    start: int = Field(title="Start", description="1-indexed inclusive genomic start coordinate")
    end: int = Field(title="End", description="1-indexed inclusive genomic end coordinate")
    strand: int = Field(title="Strand", description="Genomic strand (+1 or -1)")
    assembly_name: str = Field(title="Assembly Name", description="Genome assembly name (e.g. 'GRCh38')")
    version: int | None = Field(default=None, title="Version", description="Ensembl version of this exon")


class EnsemblTranslation(BaseModel):
    """Protein translation record returned by Ensembl REST.

    Attributes:
        id (str): Ensembl protein ID (ENSP...).
        start (int): 1-indexed inclusive genomic start of the CDS.
        end (int): 1-indexed inclusive genomic end of the CDS.
        length (int): Protein length in amino acids.
        Parent (str): Parent transcript ID (PascalCase preserved from API).
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(title="Protein ID", description="Stable Ensembl protein identifier (ENSP...)")
    start: int = Field(title="CDS Start", description="1-indexed inclusive start of the CDS on the parent transcript")
    end: int = Field(title="CDS End", description="1-indexed inclusive end of the CDS on the parent transcript")
    length: int = Field(title="Length", description="Protein length in amino acids")
    Parent: str = Field(title="Parent", description="Parent transcript ID (ENST...)")


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

    id: str = Field(title="Transcript ID", description="Stable Ensembl transcript identifier (ENST...)")
    display_name: str | None = Field(
        default=None, title="Display Name", description="Human-readable transcript name, if any"
    )
    biotype: str = Field(title="Biotype", description="Transcript biotype (protein_coding, lncRNA, ...)")
    is_canonical: bool = Field(
        default=False, title="Is Canonical", description="Whether this is the canonical transcript"
    )
    start: int = Field(title="Start", description="1-indexed inclusive genomic start coordinate")
    end: int = Field(title="End", description="1-indexed inclusive genomic end coordinate")
    strand: int = Field(title="Strand", description="Genomic strand (+1 or -1)")
    seq_region_name: str = Field(title="Sequence Region", description="Chromosome or contig name")
    assembly_name: str = Field(title="Assembly Name", description="Genome assembly name (e.g. 'GRCh38')")
    Exon: list[EnsemblExon] = Field(
        default_factory=list, title="Exons", description="Exon records (empty unless expand=True)"
    )
    Translation: EnsemblTranslation | None = Field(
        default=None, title="Translation", description="Protein translation; None for non-coding biotypes"
    )


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

    id: str = Field(title="Gene ID", description="Stable Ensembl gene identifier (ENSG...)")
    display_name: str | None = Field(
        default=None, title="Display Name", description="Human-readable gene symbol (e.g. 'BRCA1')"
    )
    description: str | None = Field(
        default=None, title="Description", description="Free-text gene description from Ensembl"
    )
    biotype: str = Field(title="Biotype", description="Gene biotype (protein_coding, lncRNA, ...)")
    species: str = Field(title="Species", description="Species slug (e.g. 'homo_sapiens')")
    seq_region_name: str = Field(title="Sequence Region", description="Chromosome or contig name")
    start: int = Field(title="Start", description="1-indexed inclusive genomic start coordinate")
    end: int = Field(title="End", description="1-indexed inclusive genomic end coordinate")
    strand: int = Field(title="Strand", description="Genomic strand (+1 or -1)")
    assembly_name: str = Field(title="Assembly Name", description="Genome assembly name (e.g. 'GRCh38')")
    canonical_transcript: str | None = Field(
        default=None, title="Canonical Transcript", description="Canonical transcript ID with version"
    )
    Transcript: list[EnsemblTranscript] = Field(
        default_factory=list, title="Transcripts", description="Transcript records (empty unless expand=True)"
    )


class EnsemblSequence(BaseModel):
    """Sequence record returned by /sequence/id/{id}.

    Attributes:
        id (str): Stable ID echoed by the server. May differ from the input
            ID — for example, an ENST input with ``type=protein`` resolves to
            the corresponding ENSP.
        desc (str | None): Description string returned by the server.
        mol_type (str | None): Molecule type ('dna' / 'protein' / ...).
        seq (str): The raw sequence string.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(title="ID", description="Stable Ensembl ID echoed by the server (may differ from input)")
    desc: str | None = Field(default=None, title="Description", description="Description string returned by the server")
    mol_type: str | None = Field(
        default=None,
        alias="molecule",
        title="Molecule Type",
        description="Molecule type ('dna', 'protein', ...)",
    )
    seq: str = Field(title="Sequence", description="Raw sequence string (DNA, cDNA, CDS, or protein)")


class EnsemblXref(BaseModel):
    """One cross-reference returned by /xrefs/id.

    Attributes:
        dbname (str): External database name (e.g. 'Uniprot_gn', 'EntrezGene').
        db_display_name (str | None): Human-readable DB name.
        display_id (str): Display identifier in the external DB.
        primary_id (str): Primary identifier in the external DB.
        description (str | None): External-DB description.
        info_type (str | None): Cross-reference type ('DIRECT', 'DEPENDENT', ...).
    """

    model_config = ConfigDict(extra="ignore")

    dbname: str = Field(title="Database Name", description="External database name (e.g. 'Uniprot_gn', 'EntrezGene')")
    db_display_name: str | None = Field(
        default=None, title="Database Display Name", description="Human-readable external database name"
    )
    display_id: str = Field(title="Display ID", description="Display identifier in the external database")
    primary_id: str = Field(title="Primary ID", description="Primary identifier in the external database")
    description: str | None = Field(
        default=None, title="Description", description="External-DB description for this entry"
    )
    info_type: str | None = Field(
        default=None, title="Info Type", description="Cross-reference type (e.g. 'DIRECT', 'DEPENDENT')"
    )


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

    feature_type: str = Field(
        title="Feature Type", description="Feature type ('gene', 'transcript', 'regulatory', 'variation', ...)"
    )
    id: str | None = Field(default=None, title="Feature ID", description="Feature identifier where the API returns one")
    biotype: str | None = Field(default=None, title="Biotype", description="Feature biotype where applicable")
    start: int = Field(title="Start", description="1-indexed inclusive genomic start coordinate")
    end: int = Field(title="End", description="1-indexed inclusive genomic end coordinate")
    strand: int = Field(title="Strand", description="Genomic strand (+1, -1, or 0 for unstranded)")
    seq_region_name: str = Field(title="Sequence Region", description="Chromosome or contig name")
    raw: dict[str, Any] = Field(
        default_factory=dict, title="Raw Record", description="Full upstream record for feature-specific fields"
    )

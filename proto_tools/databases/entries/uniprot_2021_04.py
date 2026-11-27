"""UniProt full (release 2021_04) — paired-MSA + template DB for AF3."""

from proto_tools.databases.mmseqs2 import MMseqs2DatabaseEntry, MMseqs2SearchFlags
from proto_tools.databases.registry import (
    BuildRecipe,
    BuildStep,
    DatasetRegistry,
    DownloadSpec,
)

ENTRY = MMseqs2DatabaseEntry(
    name="uniprot-2021-04",
    molecule_type="protein",
    display_name="UniProt (release 2021_04)",
    description="Full UniProt; AF3 paired-MSA + template hits source.",
    citation_doi="10.1093/nar/gkaa1100",  # UniProt — UniProt Consortium 2021
    downloads=[
        DownloadSpec(
            url="https://storage.googleapis.com/alphafold-databases/v3.0/uniprot_all_2021_04.fa.zst",
            filename="uniprot_all_2021_04.fa.zst",
        ),
    ],
    total_download_bytes=95_000_000_000,
    total_disk_bytes=165_000_000_000,
    build=BuildRecipe(
        steps=[
            BuildStep(
                command=["zstd", "-d", "--rm", "uniprot_all_2021_04.fa.zst"],
                description="Decompress source FASTA",
            ),
            BuildStep(
                command=["mmseqs", "createdb", "uniprot_all_2021_04.fa", "uniprot"],
                description="Build MMseqs2 DB from FASTA",
            ),
            BuildStep(
                command=["mmseqs", "makepaddedseqdb", "uniprot", "uniprot_padded"],
                description="Build GPU padded-sequence DB for MMseqs2-GPU",
            ),
        ],
        output_files=[
            "uniprot_padded.dbtype",
            "uniprot_padded.index",
        ],
    ),
    search_flags=MMseqs2SearchFlags(sensitivity=8.0, prefilter_mode=0, max_seqs=300),
    db_prefix="uniprot_padded",
    gpu_db="uniprot_padded",
    a3m_adapter="plain",
)

DatasetRegistry.register(ENTRY)

"""Rfam 15.1 — curated RNA family DB, full Rfam sequences.

Bundled fasta (~443 MB compressed) from EBI's versioned 15.1/fasta_files/Rfam.fa.gz.
Unlike :mod:`rfam_14_9_90_80` (AF3 pre-clustered subset), this is the full
Rfam 15.1 sequence set with no upstream linclust step. Slightly larger
indexed DB; still small in absolute terms.
"""

from proto_tools.databases.mmseqs2 import MMseqs2DatabaseEntry, MMseqs2SearchFlags
from proto_tools.databases.registry import (
    BuildRecipe,
    BuildStep,
    DatasetRegistry,
    DownloadSpec,
)

ENTRY = MMseqs2DatabaseEntry(
    name="rfam-15-1",
    molecule_type="rna",
    display_name="Rfam 15.1",
    description="Rfam 15.1 full RNA family sequences from EBI.",
    citation_doi="10.1093/nar/gkae1023",  # Rfam 15 — Kalvari et al. 2024
    downloads=[
        DownloadSpec(
            url="https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/fasta_files/Rfam.fa.gz",
            filename="Rfam.fa.gz",
        ),
    ],
    total_download_bytes=450_000_000,
    total_disk_bytes=3_000_000_000,
    build=BuildRecipe(
        steps=[
            BuildStep(
                command=["gunzip", "Rfam.fa.gz"],
                description="Decompress source FASTA",
            ),
            BuildStep(
                command=["mmseqs", "createdb", "Rfam.fa", "rfam"],
                description="Build MMseqs2 nucleotide DB from FASTA",
            ),
            BuildStep(
                command=[
                    "mmseqs",
                    "createindex",
                    "rfam",
                    "tmp_createindex",
                    "--search-type",
                    "3",
                    "--split-memory-limit",
                    "{split_memory_limit}",
                    "--remove-tmp-files",
                    "1",
                ],
                description="Build nucleotide-mode k-mer index (search-type 3)",
            ),
        ],
        output_files=[
            "rfam.dbtype",
            "rfam.index",
        ],
    ),
    search_flags=MMseqs2SearchFlags(sensitivity=7.0, prefilter_mode=0, max_seqs=300),
    db_prefix="rfam",
    a3m_adapter="rna",
)

DatasetRegistry.register(ENTRY)

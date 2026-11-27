"""Rfam 15.1 — curated RNA family DB, full Rfam sequences.

Bundled fasta (~443 MB compressed) from EBI's versioned 15.1/fasta_files/Rfam.fa.gz.
Unlike :mod:`rfam_14_9_90_80` (AF3 pre-clustered subset), this is the full
Rfam 15.1 sequence set with no upstream linclust step. Slightly larger
indexed DB; still small in absolute terms.
"""

from proto_tools.databases.registry import (
    DatasetEntry,
    DatasetRegistry,
    DownloadSpec,
    IndexRecipe,
    IndexStep,
    MmseqsFlags,
)

ENTRY = DatasetEntry(
    name="rfam-15-1",
    molecule_type="rna",
    display_name="Rfam 15.1",
    description="Rfam 15.1 full RNA family sequences from EBI.",
    citation_doi="10.1093/nar/gkae1023",  # Rfam 15 — Kalvari et al. 2024
    urls=[
        DownloadSpec(
            url="https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/fasta_files/Rfam.fa.gz",
            filename="Rfam.fa.gz",
        ),
    ],
    total_download_bytes=450_000_000,
    total_disk_bytes=3_000_000_000,
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["gunzip", "Rfam.fa.gz"],
                description="Decompress source FASTA",
            ),
            IndexStep(
                command=["mmseqs", "createdb", "Rfam.fa", "rfam"],
                description="Build MMseqs2 nucleotide DB from FASTA",
            ),
            IndexStep(
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
    mmseqs_flags=MmseqsFlags(sensitivity=7.0, prefilter_mode=0, max_seqs=300),
    db_prefix="rfam",
    supports_gpu=False,
    min_gpu_memory_gb=None,
    a3m_adapter="rna",
)

DatasetRegistry.register(ENTRY)

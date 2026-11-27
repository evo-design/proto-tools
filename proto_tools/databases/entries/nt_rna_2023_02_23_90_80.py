"""nt-rna (NCBI nt RNA-filtered subset, 2023_02_23, clustered 90/80) — RNA fallback DB for AF3.

RNA-only filtered subset of NCBI nt, clustered. Used as a deeper fallback
when RNAcentral + Rfam don't produce enough homologs. ~30 GB FASTA.
"""

from proto_tools.databases.mmseqs2 import MMseqs2DatabaseEntry, MMseqs2SearchFlags
from proto_tools.databases.registry import (
    BuildRecipe,
    BuildStep,
    DatasetRegistry,
    DownloadSpec,
)

ENTRY = MMseqs2DatabaseEntry(
    name="nt-rna-2023-02-23-90-80",
    molecule_type="rna",
    display_name="NCBI nt-rna (2023_02_23, clustered 90/80)",
    description="RNA-filtered NCBI nt subset, clustered — AF3 RNA fallback DB.",
    citation_doi="10.1093/nar/gkr1184",  # NCBI nt — Sayers et al.
    downloads=[
        DownloadSpec(
            url="https://storage.googleapis.com/alphafold-databases/v3.0/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
            filename="nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
        ),
    ],
    total_download_bytes=30_000_000_000,
    total_disk_bytes=30_000_000_000,
    build=BuildRecipe(
        steps=[
            BuildStep(
                command=["zstd", "-d", "--rm", "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst"],
                description="Decompress source FASTA",
            ),
            BuildStep(
                command=[
                    "mmseqs",
                    "createdb",
                    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
                    "nt_rna",
                ],
                description="Build MMseqs2 nucleotide DB from FASTA",
            ),
            BuildStep(
                command=[
                    "mmseqs",
                    "createindex",
                    "nt_rna",
                    "tmp_createindex",
                    "--search-type",
                    "3",
                    "--split-memory-limit",
                    "{split_memory_limit}",
                    "--remove-tmp-files",
                    "1",
                ],
                description="Build nucleotide-mode k-mer index (search-type 3, split for size)",
            ),
        ],
        output_files=[
            "nt_rna.dbtype",
            "nt_rna.index",
        ],
    ),
    search_flags=MMseqs2SearchFlags(sensitivity=7.0, prefilter_mode=0, max_seqs=300),
    db_prefix="nt_rna",
    a3m_adapter="rna",
)

DatasetRegistry.register(ENTRY)

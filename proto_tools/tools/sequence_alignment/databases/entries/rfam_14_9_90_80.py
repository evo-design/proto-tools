"""Rfam 14.9 (clustered 90% id, 80% cov) — curated RNA family DB for AF3.

Tiny DB (~10 MB FASTA). Curated noncoding RNA families; high-precision
seed alignments. Source from the AF3 mirror.
"""

from proto_tools.tools.sequence_alignment.databases.registry import (
    DatasetEntry,
    DatasetRegistry,
    DownloadSpec,
    IndexRecipe,
    IndexStep,
    MmseqsFlags,
)

ENTRY = DatasetEntry(
    name="rfam-14-9-90-80",
    molecule_type="rna",
    display_name="Rfam 14.9 (clustered 90/80)",
    description="Rfam 14.9 curated RNA families clustered at 90/80 — AF3 RNA component.",
    citation_doi="10.1093/nar/gkaa1047",  # Rfam — Kalvari et al. 2021
    urls=[
        DownloadSpec(
            url="https://storage.googleapis.com/alphafold-databases/v3.0/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
            filename="rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
        ),
    ],
    total_download_bytes=15_000_000,
    total_disk_bytes=20_000_000,
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["zstd", "-d", "--rm", "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst"],
                description="Decompress source FASTA",
            ),
            IndexStep(
                command=[
                    "mmseqs",
                    "createdb",
                    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
                    "rfam",
                ],
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
        paired_output_files=[],
    ),
    mmseqs_flags=MmseqsFlags(sensitivity=7.0, prefilter_mode=0, max_seqs=300),
    db_prefix="rfam",
    supports_gpu=False,
    supports_pairing=False,
    min_gpu_memory_gb=None,
    a3m_adapter="rna",
)

DatasetRegistry.register(ENTRY)

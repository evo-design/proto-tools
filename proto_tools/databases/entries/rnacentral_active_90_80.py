"""RNAcentral active sequences (clustered 90% id, 80% cov) — RNA MSA DB for AF3.

Source FASTA from the AF3 mirror at ``storage.googleapis.com/alphafold-databases/v3.0``.
Built locally as an MMseqs2 nucleotide DB with a search-type-3 k-mer
index (no GPU padded variant — AlphaFast uses CPU for RNA search).
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
    name="rnacentral-active-90-80",
    molecule_type="rna",
    display_name="RNAcentral active (clustered 90/80)",
    description="RNAcentral active sequences clustered at 90% identity, 80% coverage — AF3 RNA MSA primary.",
    citation_doi="10.1093/nar/gkaa921",  # RNAcentral — RNAcentral Consortium 2021
    urls=[
        DownloadSpec(
            url="https://storage.googleapis.com/alphafold-databases/v3.0/rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst",
            filename="rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst",
        ),
    ],
    total_download_bytes=30_000_000_000,
    total_disk_bytes=30_000_000_000,
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["zstd", "-d", "--rm", "rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst"],
                description="Decompress source FASTA",
            ),
            IndexStep(
                command=[
                    "mmseqs",
                    "createdb",
                    "rnacentral_active_seq_id_90_cov_80_linclust.fasta",
                    "rnacentral",
                ],
                description="Build MMseqs2 nucleotide DB from FASTA",
            ),
            IndexStep(
                command=[
                    "mmseqs",
                    "createindex",
                    "rnacentral",
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
            "rnacentral.dbtype",
            "rnacentral.index",
        ],
    ),
    mmseqs_flags=MmseqsFlags(sensitivity=7.0, prefilter_mode=0, max_seqs=300),
    db_prefix="rnacentral",
    supports_gpu=False,  # AlphaFast uses CPU for RNA search
    min_gpu_memory_gb=None,
    a3m_adapter="rna",
)

DatasetRegistry.register(ENTRY)

"""MGnify clusters (release 2022_05) — metagenomic protein DB for AF3."""

from proto_tools.tools.sequence_alignment.databases.registry import (
    DatasetEntry,
    DatasetRegistry,
    DownloadSpec,
    IndexRecipe,
    IndexStep,
    MmseqsFlags,
)

ENTRY = DatasetEntry(
    name="mgnify-2022-05",
    molecule_type="protein",
    display_name="MGnify clusters (release 2022_05)",
    description="Metagenomic protein clusters from MGnify 2022_05 — AF3 metagenomic component.",
    citation_doi="10.1093/nar/gkac1080",  # MGnify — Richardson et al. 2023
    urls=[
        DownloadSpec(
            url="https://storage.googleapis.com/alphafold-databases/v3.0/mgy_clusters_2022_05.fa.zst",
            filename="mgy_clusters_2022_05.fa.zst",
        ),
    ],
    total_download_bytes=60_000_000_000,
    total_disk_bytes=110_000_000_000,
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["zstd", "-d", "--rm", "mgy_clusters_2022_05.fa.zst"],
                description="Decompress source FASTA",
            ),
            IndexStep(
                command=["mmseqs", "createdb", "mgy_clusters_2022_05.fa", "mgnify"],
                description="Build MMseqs2 DB from FASTA",
            ),
            IndexStep(
                command=["mmseqs", "makepaddedseqdb", "mgnify", "mgnify_padded"],
                description="Build GPU padded-sequence DB for MMseqs2-GPU",
            ),
        ],
        output_files=[
            "mgnify_padded.dbtype",
            "mgnify_padded.index",
        ],
        paired_output_files=[],
    ),
    mmseqs_flags=MmseqsFlags(sensitivity=8.0, prefilter_mode=0, max_seqs=300),
    db_prefix="mgnify_padded",
    supports_gpu=True,
    supports_pairing=False,
    min_gpu_memory_gb=8.0,
    a3m_adapter="plain",
)

DatasetRegistry.register(ENTRY)

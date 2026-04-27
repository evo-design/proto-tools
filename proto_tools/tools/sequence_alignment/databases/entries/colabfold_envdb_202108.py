"""ColabFold environmental DB (release 202108) — metagenomic protein DB.

Companion to UniRef30 in the ColabFold MSA pipeline. Distributed as a
prebuilt MMseqs2 database via opendata.mmseqs.org. Optional even in the
ColabFold workflow (UniRef30 alone produces usable MSAs); pulling envdb
deepens MSAs at the cost of ~110 GB download + ~650 GB indexed.
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
    name="colabfold-envdb-202108",
    molecule_type="protein",
    display_name="ColabFold environmental DB (202108)",
    description="Metagenomic clustered protein DB from ColabFold release 202108.",
    citation_doi="10.1038/s41592-022-01488-1",  # ColabFold — Mirdita et al.
    urls=[
        DownloadSpec(
            url="https://opendata.mmseqs.org/colabfold/colabfold_envdb_202108.db.tar.gz",
            filename="colabfold_envdb_202108.tar.gz",
        ),
    ],
    total_download_bytes=110_000_000_000,
    total_disk_bytes=650_000_000_000,
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["tar", "-xzvf", "colabfold_envdb_202108.tar.gz"],
                description="Extract MMseqs2 DB tarball",
            ),
            IndexStep(
                command=[
                    "mmseqs",
                    "createindex",
                    "colabfold_envdb_202108_db",
                    "tmp_createindex",
                    "--remove-tmp-files",
                    "1",
                ],
                description="Build MMseqs2 sequence index for CPU search",
            ),
            IndexStep(
                command=["mmseqs", "makepaddedseqdb", "colabfold_envdb_202108_db", "colabfold_envdb_202108_db.idx_pad"],
                description="Build GPU padded-sequence DB for MMseqs2-GPU",
            ),
        ],
        output_files=[
            "colabfold_envdb_202108_db.dbtype",
            "colabfold_envdb_202108_db.idx",
            "colabfold_envdb_202108_db_h",
        ],
        paired_output_files=[
            "colabfold_envdb_202108_db.idx_pad",
        ],
    ),
    mmseqs_flags=MmseqsFlags(sensitivity=8.0, prefilter_mode=0, max_seqs=300),
    db_prefix="colabfold_envdb_202108_db",
    supports_gpu=True,
    gpu_padded_marker="colabfold_envdb_202108_db.idx_pad",
    supports_pairing=False,
    min_gpu_memory_gb=10.0,
    a3m_adapter="colabfold",
)

DatasetRegistry.register(ENTRY)

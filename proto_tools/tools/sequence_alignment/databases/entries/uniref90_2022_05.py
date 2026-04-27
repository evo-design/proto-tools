"""UniRef90 (release 2022_05) — primary protein MSA database for AF3.

Source FASTA from the AlphaFold 3 database mirror at
``storage.googleapis.com/alphafold-databases/v3.0``. Built locally into
an MMseqs2 GPU-padded DB the same way AlphaFast does it.
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
    name="uniref90-2022-05",
    molecule_type="protein",
    display_name="UniRef90 (release 2022_05)",
    description="UniProt clustered at 90% identity, 2022_05 — AF3 primary protein MSA DB.",
    citation_doi="10.1093/bioinformatics/btx019",  # UniRef — Suzek et al.
    urls=[
        DownloadSpec(
            url="https://storage.googleapis.com/alphafold-databases/v3.0/uniref90_2022_05.fa.zst",
            filename="uniref90_2022_05.fa.zst",
        ),
    ],
    total_download_bytes=70_000_000_000,
    total_disk_bytes=120_000_000_000,
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["zstd", "-d", "--rm", "uniref90_2022_05.fa.zst"],
                description="Decompress source FASTA",
            ),
            IndexStep(
                command=["mmseqs", "createdb", "uniref90_2022_05.fa", "uniref90"],
                description="Build MMseqs2 DB from FASTA",
            ),
            IndexStep(
                command=["mmseqs", "makepaddedseqdb", "uniref90", "uniref90_padded"],
                description="Build GPU padded-sequence DB for MMseqs2-GPU",
            ),
        ],
        output_files=[
            "uniref90_padded.dbtype",
            "uniref90_padded.index",
        ],
        paired_output_files=[],
    ),
    mmseqs_flags=MmseqsFlags(sensitivity=8.0, prefilter_mode=0, max_seqs=300),
    db_prefix="uniref90_padded",
    supports_gpu=True,
    supports_pairing=False,
    min_gpu_memory_gb=8.0,
    a3m_adapter="plain",
)

DatasetRegistry.register(ENTRY)

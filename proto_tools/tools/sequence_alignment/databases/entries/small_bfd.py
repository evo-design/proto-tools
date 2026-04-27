"""Small BFD (clustered, first-non-consensus) — protein DB for AF3.

Reduced version of BFD (Big Fantastic Database). Distinct from the full
BFD entry (~270 GB / ~1.8 TB); ``small-bfd`` is what AF3 / AlphaFast /
Lightning-Boltz use by default. Source FASTA from the AF3 database mirror.
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
    name="small-bfd",
    molecule_type="protein",
    display_name="Small BFD (first-non-consensus)",
    description="Reduced/clustered BFD; AF3 default protein DB component.",
    citation_doi="10.1101/2021.10.04.463034",  # AlphaFold MSA DB — Steinegger et al.
    urls=[
        DownloadSpec(
            url="https://storage.googleapis.com/alphafold-databases/v3.0/bfd-first_non_consensus_sequences.fasta.zst",
            filename="bfd-first_non_consensus_sequences.fasta.zst",
        ),
    ],
    total_download_bytes=14_000_000_000,
    total_disk_bytes=25_000_000_000,
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["zstd", "-d", "--rm", "bfd-first_non_consensus_sequences.fasta.zst"],
                description="Decompress source FASTA",
            ),
            IndexStep(
                command=["mmseqs", "createdb", "bfd-first_non_consensus_sequences.fasta", "small_bfd"],
                description="Build MMseqs2 DB from FASTA",
            ),
            IndexStep(
                command=["mmseqs", "makepaddedseqdb", "small_bfd", "small_bfd_padded"],
                description="Build GPU padded-sequence DB for MMseqs2-GPU",
            ),
        ],
        output_files=[
            "small_bfd_padded.dbtype",
            "small_bfd_padded.index",
        ],
        paired_output_files=[],
    ),
    mmseqs_flags=MmseqsFlags(sensitivity=8.0, prefilter_mode=0, max_seqs=300),
    db_prefix="small_bfd_padded",
    supports_gpu=True,
    supports_pairing=False,
    min_gpu_memory_gb=4.0,
    a3m_adapter="plain",
)

DatasetRegistry.register(ENTRY)

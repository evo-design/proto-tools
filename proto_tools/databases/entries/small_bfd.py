"""Small BFD (clustered, first-non-consensus) — protein DB for AF3.

Reduced version of BFD (Big Fantastic Database). Distinct from the full
BFD entry (~270 GB / ~1.8 TB); ``small-bfd`` is what AF3 / AlphaFast /
Lightning-Boltz use by default. Source FASTA from the AF3 database mirror.
"""

from proto_tools.databases.mmseqs2 import MMseqs2DatabaseEntry, MMseqs2SearchFlags
from proto_tools.databases.registry import (
    BuildRecipe,
    BuildStep,
    DatasetRegistry,
    DownloadSpec,
)

ENTRY = MMseqs2DatabaseEntry(
    name="small-bfd",
    molecule_type="protein",
    display_name="Small BFD (first-non-consensus)",
    description="Reduced/clustered BFD; AF3 default protein DB component.",
    citation_doi="10.1101/2021.10.04.463034",  # AlphaFold MSA DB — Steinegger et al.
    downloads=[
        DownloadSpec(
            url="https://storage.googleapis.com/alphafold-databases/v3.0/bfd-first_non_consensus_sequences.fasta.zst",
            filename="bfd-first_non_consensus_sequences.fasta.zst",
        ),
    ],
    total_download_bytes=14_000_000_000,
    total_disk_bytes=25_000_000_000,
    build=BuildRecipe(
        steps=[
            BuildStep(
                command=["zstd", "-d", "--rm", "bfd-first_non_consensus_sequences.fasta.zst"],
                description="Decompress source FASTA",
            ),
            BuildStep(
                command=["mmseqs", "createdb", "bfd-first_non_consensus_sequences.fasta", "small_bfd"],
                description="Build MMseqs2 DB from FASTA",
            ),
            BuildStep(
                command=["mmseqs", "makepaddedseqdb", "small_bfd", "small_bfd_padded"],
                description="Build GPU padded-sequence DB for MMseqs2-GPU",
            ),
        ],
        output_files=[
            "small_bfd_padded.dbtype",
            "small_bfd_padded.index",
        ],
    ),
    search_flags=MMseqs2SearchFlags(sensitivity=8.0, prefilter_mode=0, max_seqs=300),
    db_prefix="small_bfd_padded",
    gpu_db="small_bfd_padded",
    a3m_adapter="plain",
)

DatasetRegistry.register(ENTRY)

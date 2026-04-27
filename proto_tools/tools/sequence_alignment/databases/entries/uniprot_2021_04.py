"""UniProt full (release 2021_04) — paired-MSA + template DB for AF3."""

from proto_tools.tools.sequence_alignment.databases.registry import (
    DatasetEntry,
    DatasetRegistry,
    DownloadSpec,
    IndexRecipe,
    IndexStep,
    MmseqsFlags,
)

ENTRY = DatasetEntry(
    name="uniprot-2021-04",
    molecule_type="protein",
    display_name="UniProt (release 2021_04)",
    description="Full UniProt; AF3 paired-MSA + template hits source.",
    citation_doi="10.1093/nar/gkaa1100",  # UniProt — UniProt Consortium 2021
    urls=[
        DownloadSpec(
            url="https://storage.googleapis.com/alphafold-databases/v3.0/uniprot_all_2021_04.fa.zst",
            filename="uniprot_all_2021_04.fa.zst",
        ),
    ],
    total_download_bytes=95_000_000_000,
    total_disk_bytes=165_000_000_000,
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["zstd", "-d", "--rm", "uniprot_all_2021_04.fa.zst"],
                description="Decompress source FASTA",
            ),
            IndexStep(
                command=["mmseqs", "createdb", "uniprot_all_2021_04.fa", "uniprot"],
                description="Build MMseqs2 DB from FASTA",
            ),
            IndexStep(
                command=["mmseqs", "makepaddedseqdb", "uniprot", "uniprot_padded"],
                description="Build GPU padded-sequence DB for MMseqs2-GPU",
            ),
        ],
        output_files=[
            "uniprot_padded.dbtype",
            "uniprot_padded.index",
        ],
        paired_output_files=[],
    ),
    mmseqs_flags=MmseqsFlags(sensitivity=8.0, prefilter_mode=0, max_seqs=300),
    db_prefix="uniprot_padded",
    supports_gpu=True,
    supports_pairing=True,  # paired-MSA path uses UniProt for cross-chain coevolution
    min_gpu_memory_gb=10.0,
    a3m_adapter="plain",
)

DatasetRegistry.register(ENTRY)

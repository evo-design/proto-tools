"""UniRef50 (release 2026_01) — 50%-identity protein homology-search database.

Source FASTA from UniProt directly: UniRef50 is not mirrored on the AF3 or
ColabFold database hosts (unlike its siblings ``uniref90-2022-05`` and
``uniref30-2302``), so we pull UniProt's single-file ``uniref50.fasta.gz``.

NOTE: this is UniProt's *current-release* URL, which advances when UniProt
ships a new release. The ``2026_01`` in the name records the release this
entry was pinned against (60,315,044 clusters, observed 2026-05-26). Re-pin
the name + ``total_*_bytes`` when re-provisioning against a newer UniProt
release. Built into an MMseqs2 GPU-padded DB the same way [[uniref90-2022-05]].
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
    name="uniref50-2026-01",
    molecule_type="protein",
    display_name="UniRef50 (release 2026_01)",
    description="UniProt clustered at 50% identity, release 2026_01 — protein homology-search DB.",
    citation_doi="10.1093/bioinformatics/btx019",  # UniRef — Suzek et al.
    urls=[
        DownloadSpec(
            url="https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
            filename="uniref50_2026_01.fasta.gz",
        ),
    ],
    total_download_bytes=12_678_152_383,  # observed 2026-05-26
    total_disk_bytes=60_000_000_000,  # estimate post-index
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["gzip", "-d", "uniref50_2026_01.fasta.gz"],
                description="Decompress source FASTA",
            ),
            IndexStep(
                command=["mmseqs", "createdb", "uniref50_2026_01.fasta", "uniref50"],
                description="Build MMseqs2 DB from FASTA",
            ),
            IndexStep(
                command=["mmseqs", "makepaddedseqdb", "uniref50", "uniref50_padded"],
                description="Build GPU padded-sequence DB for MMseqs2-GPU",
            ),
        ],
        output_files=[
            "uniref50_padded.dbtype",
            "uniref50_padded.index",
        ],
        paired_output_files=[],
    ),
    mmseqs_flags=MmseqsFlags(sensitivity=8.0, prefilter_mode=0, max_seqs=300),
    db_prefix="uniref50_padded",
    supports_gpu=True,
    supports_pairing=False,
    min_gpu_memory_gb=8.0,
    a3m_adapter="plain",
)

DatasetRegistry.register(ENTRY)

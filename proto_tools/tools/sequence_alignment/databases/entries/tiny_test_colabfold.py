"""Tiny colabfold-style DB for tests (mini SwissProt, ~263 MB download).

Pulls ``mini_swissprot2503.tar.gz`` from the official MMseqs2/ColabFold
mirror and renames it into the standard ``uniref30_mini_db*`` layout. This
is a real clustered colabfold-style DB (with ``_seq`` cluster members,
prefilter ``.idx``, and ``.idx_pad`` for GPU), which the colabfold_search
pipeline's ``expandaln`` step requires — a freshly-built ``mmseqs createdb``
DB lacks the cluster expansion data.

Auto-provisions on first dispatch so smoke tests don't need a separate
``setup_databases.py`` invocation. The ``tests/dummy_data/create_mini_mmseqs_db.sh``
script uses the same source tarball and rename pattern.
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
    name="tiny-test-colabfold",
    molecule_type="protein",
    display_name="Tiny test DB (mini SwissProt, colabfold-style)",
    description="Mini SwissProt colabfold-style DB used by mmseqs2-homology-search smoke tests; not for production search.",
    citation_doi=None,
    urls=[
        DownloadSpec(
            url="https://opendata.mmseqs.org/colabfold/mini_swissprot2503.tar.gz",
            filename="mini_swissprot2503.tar.gz",
        ),
    ],
    total_download_bytes=263_000_000,
    total_disk_bytes=3_000_000_000,
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["tar", "-xzf", "mini_swissprot2503.tar.gz"],
                description="Extract mini SwissProt tarball",
            ),
            # Rename the extracted sprot2503* DBs into the colabfold-standard
            # uniref30_mini_db* layout that colabfold_search auto-detects.
            IndexStep(
                command=["mmseqs", "mvdb", "sprot2503_h", "uniref30_mini_db_h"],
                description="Rename header DB",
            ),
            IndexStep(
                command=["mmseqs", "mvdb", "sprot2503", "uniref30_mini_db"],
                description="Rename main sequence DB",
            ),
            IndexStep(
                command=["mmseqs", "mvdb", "sprot2503_aln", "uniref30_mini_db_aln"],
                description="Rename alignment DB",
            ),
            IndexStep(
                command=["mmseqs", "mvdb", "sprot2503_seq_h", "uniref30_mini_db_seq_h"],
                description="Rename cluster-member header DB",
            ),
            IndexStep(
                command=["mmseqs", "mvdb", "sprot2503_seq", "uniref30_mini_db_seq"],
                description="Rename cluster-member sequence DB",
            ),
            IndexStep(
                command=["mv", "-f", "sprot2503_taxonomy", "uniref30_mini_db_taxonomy"],
                description="Rename taxonomy file",
            ),
            IndexStep(
                command=["mv", "-f", "sprot2503_mapping", "uniref30_mini_db_mapping"],
                description="Rename taxonomy mapping file",
            ),
        ],
        output_files=[
            "uniref30_mini_db.dbtype",
            "uniref30_mini_db.idx",
            "uniref30_mini_db_h",
            "uniref30_mini_db_seq",
        ],
        paired_output_files=[],
    ),
    mmseqs_flags=MmseqsFlags(sensitivity=8.0, prefilter_mode=0, max_seqs=300),
    db_prefix="uniref30_mini_db",
    supports_gpu=True,
    gpu_padded_marker="uniref30_mini_db.idx_pad",
    supports_pairing=False,
    min_gpu_memory_gb=0.5,
    a3m_adapter="colabfold",
    auto_provision=True,
)

DatasetRegistry.register(ENTRY)

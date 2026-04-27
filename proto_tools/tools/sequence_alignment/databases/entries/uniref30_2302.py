"""UniRef30 release 2302 — primary protein MSA database for AF2/AF3/Chai-1/Protenix/Boltz-2.

Mirrors the files produced by
``proto_tools/tools/sequence_alignment/colabfold_search/setup_databases.sh``
when invoked with defaults — the official ColabFold tarball from
``opendata.mmseqs.org`` plus the companion taxonomy tarball, followed by
``mmseqs createindex`` and ``mmseqs makepaddedseqdb`` (GPU support) and
``mmseqs createbintaxmapping`` (taxonomy lookup for paired MSAs).
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
    name="uniref30-2302",
    molecule_type="protein",
    display_name="UniRef30 (release 2302)",
    description="Clustered UniProt at 30% sequence identity, ColabFold release 2302.",
    citation_doi="10.1038/s41592-019-0437-4",  # MMseqs2 Linclust/UniRef30 — Steinegger et al.
    urls=[
        DownloadSpec(
            url="https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz",
            filename="uniref30_2302.tar.gz",
        ),
        DownloadSpec(
            url="https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz",
            filename="uniref30_2302_newtaxonomy.tar.gz",
            required=False,  # only needed for paired MSAs
        ),
    ],
    total_download_bytes=106_729_528_496 + 1_975_608_472,  # main + taxonomy, observed 2026-04-24
    total_disk_bytes=650_000_000_000,  # ~620 GB post-index, observed 2026-04-24
    index_recipe=IndexRecipe(
        steps=[
            IndexStep(
                command=["tar", "-xzvf", "uniref30_2302.tar.gz"],
                description="Extract main MMseqs2 DB tarball",
            ),
            IndexStep(
                command=["tar", "-xzvf", "uniref30_2302_newtaxonomy.tar.gz"],
                description="Extract companion taxonomy tarball",
            ),
            IndexStep(
                command=["mmseqs", "createindex", "uniref30_2302_db", "tmp_createindex", "--remove-tmp-files", "1"],
                description="Build MMseqs2 sequence index (.idx) for CPU search",
            ),
            IndexStep(
                command=["mmseqs", "makepaddedseqdb", "uniref30_2302_db", "uniref30_2302_db.idx_pad"],
                description="Build GPU padded-sequence DB (.idx_pad) for MMseqs2-GPU",
            ),
            IndexStep(
                command=["mmseqs", "createbintaxmapping", "uniref30_2302_db_mapping", "uniref30_2302_db_mapping.bin"],
                description="Binary taxonomy mapping for paired-MSA species intersection",
            ),
        ],
        output_files=[
            "uniref30_2302_db.dbtype",
            "uniref30_2302_db.idx",
            "uniref30_2302_db_h",
            "uniref30_2302_db_seq",
        ],
        paired_output_files=[
            "uniref30_2302_db.idx_pad",
            "uniref30_2302_db_mapping.bin",
        ],
    ),
    mmseqs_flags=MmseqsFlags(
        sensitivity=8.0,
        prefilter_mode=0,
        max_seqs=300,
    ),
    db_prefix="uniref30_2302_db",
    supports_gpu=True,
    gpu_padded_marker="uniref30_2302_db.idx_pad",
    supports_pairing=True,
    min_gpu_memory_gb=10.0,
    a3m_adapter="colabfold",
)

DatasetRegistry.register(ENTRY)

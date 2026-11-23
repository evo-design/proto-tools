"""UniRef30 release 2302 — primary protein MSA database for AF2/AF3/Chai-1/Protenix/Boltz-2."""

from proto_tools.databases.mmseqs2 import MMseqs2DatabaseEntry, MMseqs2SearchFlags
from proto_tools.databases.registry import (
    BuildRecipe,
    BuildStep,
    DatasetRegistry,
    DownloadSpec,
)

ENTRY = MMseqs2DatabaseEntry(
    name="uniref30-2302",
    molecule_type="protein",
    display_name="UniRef30 (release 2302)",
    description="Clustered UniProt at 30% sequence identity, ColabFold release 2302.",
    citation_doi="10.1038/s41592-019-0437-4",  # MMseqs2 Linclust/UniRef30 — Steinegger et al.
    downloads=[
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
    build=BuildRecipe(
        steps=[
            BuildStep(
                command=["tar", "-xzvf", "uniref30_2302.tar.gz"],
                description="Extract main MMseqs2 DB tarball",
            ),
            BuildStep(
                command=[
                    "mmseqs",
                    "createindex",
                    "uniref30_2302_db",
                    "tmp_createindex",
                    "--split-memory-limit",
                    "{split_memory_limit}",
                    "--remove-tmp-files",
                    "1",
                ],
                description="Build MMseqs2 sequence index for search",
            ),
            BuildStep(
                command=["mmseqs", "makepaddedseqdb", "uniref30_2302_db", "uniref30_2302_db.idx_pad"],
                description="Build GPU-padded sequence DB for MMseqs2-GPU homology search",
            ),
            BuildStep(
                command=["tar", "-xzvf", "uniref30_2302_newtaxonomy.tar.gz"],
                description="Extract rebuilt taxonomy mapping (overwrites stale tarball mapping)",
            ),
        ],
        output_files=[
            "uniref30_2302_db.dbtype",
            "uniref30_2302_db_h",
            "uniref30_2302_db_seq",
            "uniref30_2302_db.idx_pad",
            "uniref30_2302_db.idx_pad.dbtype",
        ],
    ),
    search_flags=MMseqs2SearchFlags(
        sensitivity=8.0,
        prefilter_mode=0,
        max_seqs=300,
    ),
    db_prefix="uniref30_2302_db",
    gpu_db="uniref30_2302_db.idx_pad",
    a3m_adapter="colabfold",
)

DatasetRegistry.register(ENTRY)

"""PDB SeqRes (snapshot 2022_09_28) — template-hit DB for AF3.

Sequence-only PDB index used to identify candidate template structures
during AF3's structure-prediction pipeline. Tiny (~10 MB).
"""

from proto_tools.databases.mmseqs2 import MMseqs2DatabaseEntry, MMseqs2SearchFlags
from proto_tools.databases.registry import (
    BuildRecipe,
    BuildStep,
    DatasetRegistry,
    DownloadSpec,
)

ENTRY = MMseqs2DatabaseEntry(
    name="pdb-seqres-2022-09-28",
    molecule_type="protein",
    display_name="PDB SeqRes (2022_09_28)",
    description="PDB sequence-only index for template hit identification (AF3).",
    citation_doi="10.1093/nar/28.1.235",  # PDB — Berman et al.
    downloads=[
        DownloadSpec(
            url="https://storage.googleapis.com/alphafold-databases/v3.0/pdb_seqres_2022_09_28.fasta.zst",
            filename="pdb_seqres_2022_09_28.fasta.zst",
        ),
    ],
    total_download_bytes=15_000_000,
    total_disk_bytes=20_000_000,
    build=BuildRecipe(
        steps=[
            BuildStep(
                command=["zstd", "-d", "--rm", "pdb_seqres_2022_09_28.fasta.zst"],
                description="Decompress source FASTA",
            ),
            BuildStep(
                command=["mmseqs", "createdb", "pdb_seqres_2022_09_28.fasta", "pdb_seqres"],
                description="Build MMseqs2 DB from FASTA",
            ),
            BuildStep(
                command=["mmseqs", "makepaddedseqdb", "pdb_seqres", "pdb_seqres_padded"],
                description="Build GPU padded-sequence DB",
            ),
        ],
        output_files=[
            "pdb_seqres_padded.dbtype",
            "pdb_seqres_padded.index",
        ],
    ),
    search_flags=MMseqs2SearchFlags(sensitivity=8.0, prefilter_mode=0, max_seqs=300),
    db_prefix="pdb_seqres_padded",
    gpu_db="pdb_seqres_padded",
    a3m_adapter="plain",
)

DatasetRegistry.register(ENTRY)

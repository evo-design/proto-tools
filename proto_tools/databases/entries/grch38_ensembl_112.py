"""GRCh38 human reference genome, Ensembl release 112 primary assembly.

The wild-type sequence source for variant scoring. Ensembl naming ("1", not "chr1") matches the
GENCODE annotations bundled with SpliceAI; a caller's "chr1" is normalized to it at lookup time.
"""

from proto_tools.databases.registry import (
    DatasetEntry,
    DatasetRegistry,
    DownloadSpec,
    IndexRecipe,
    IndexStep,
)

FASTA = "Homo_sapiens.GRCh38.dna.primary_assembly.fa"

ENTRY = DatasetEntry(
    name="grch38",
    molecule_type="dna",
    display_name="GRCh38 (Ensembl 112)",
    description="Human reference genome GRCh38 primary assembly, Ensembl release 112.",
    citation_doi="10.1093/nar/gkad1049",  # Ensembl 2024 — Harrison et al.
    urls=[
        DownloadSpec(
            url=(
                "https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/"
                "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
            ),
            filename=f"{FASTA}.gz",
        ),
    ],
    total_download_bytes=881_964_081,
    total_disk_bytes=3_200_000_000,
    index_recipe=IndexRecipe(
        # Decompress only. pyfaidx builds the .fai on first open, and doing it here would need a
        # Python with pyfaidx on PATH, which a provisioning step is not guaranteed. The Modal
        # services build the index once at warmup instead, off the concurrent request path.
        steps=[
            IndexStep(
                command=["gunzip", "-f", f"{FASTA}.gz"],
                description="Decompress the reference FASTA",
            ),
        ],
        output_files=[FASTA],
    ),
    # ~3 GB, smaller than the AlphaFold2 parameters every structure-prediction toolkit already
    # fetches at env-build time, so it provisions on demand rather than requiring a separate step.
    auto_provision=True,
)

DatasetRegistry.register(ENTRY)

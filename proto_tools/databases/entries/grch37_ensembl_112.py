"""GRCh37 human reference genome, Ensembl GRCh37 archive release 112 primary assembly.

The wild-type sequence source for variant scoring against the older assembly. GRCh37 coordinates
differ from GRCh38, so this must agree with the annotation and with the variants being scored.
"""

from proto_tools.databases.registry import (
    DatasetEntry,
    DatasetRegistry,
    DownloadSpec,
    IndexRecipe,
    IndexStep,
)

FASTA = "Homo_sapiens.GRCh37.dna.primary_assembly.fa"

ENTRY = DatasetEntry(
    name="grch37",
    molecule_type="dna",
    display_name="GRCh37 (Ensembl 112 archive)",
    description="Human reference genome GRCh37 primary assembly, Ensembl GRCh37 archive release 112.",
    citation_doi="10.1093/nar/gkad1049",  # Ensembl 2024 — Harrison et al.
    urls=[
        DownloadSpec(
            url=(
                "https://ftp.ensembl.org/pub/grch37/release-112/fasta/homo_sapiens/dna/"
                "Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
            ),
            filename=f"{FASTA}.gz",
            expected_bytes=869_923_173,
        ),
    ],
    total_download_bytes=869_923_173,
    total_disk_bytes=3_100_000_000,
    index_recipe=IndexRecipe(
        # See the GRCh38 entry: decompress only, index built at warmup.
        steps=[
            IndexStep(
                command=["gunzip", "-f", f"{FASTA}.gz"],
                description="Decompress the reference FASTA",
            ),
        ],
        output_files=[FASTA],
    ),
    auto_provision=True,
)

DatasetRegistry.register(ENTRY)

"""BLAST database creation tool."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)


# ============================================================================
# Data Models
# ============================================================================
# Input:
class CreateBlastDbInput(BaseToolInput):
    """Input object for creating a BLAST database.

    This class defines the input parameters for creating a local BLAST database
    from a FASTA file using the ``makeblastdb`` utility from BLAST+.

    Attributes:
        fasta (str): Path to a FASTA file containing the sequences to be indexed into
            a BLAST database. The file must exist and contain valid FASTA-formatted
            sequences. For nucleotide databases, sequences should be DNA or RNA.
            For protein databases, sequences should be amino acids.
    """

    fasta: str = InputField(
        description="Path to the input FASTA file containing sequences to create a BLAST database from"
    )

    @field_validator("fasta")
    @classmethod
    def validate_fasta(cls, v: str) -> str:
        """Validate that FASTA file exists."""
        if not Path(v).exists():
            raise ValueError(f"FASTA file not found: {v}")
        return v


# Output:
class CreateBlastDbOutput(BaseToolOutput):
    """Output from BLAST database creation.

    This class encapsulates the results of creating a BLAST database,
    providing the path to the newly created database files.

    Attributes:
        db_path (str): The base path to the generated BLAST database files (without
            file extensions). This path can be used directly as the value for
            the ``local_db`` parameter in ``BlastSearchConfig``. For example, if
            ``db_path`` is ``"/data/mydb"``, ``makeblastdb`` will have created
            multiple files like ``"/data/mydb.nhr"``, ``"/data/mydb.nin"``,
            ``"/data/mydb.nsq"`` (for nucleotide databases) or similar extensions
            for protein databases.
    """

    db_path: str = Field(description="Path to the generated BLAST database")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return []

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return ""

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export output - No-op for database creation (output IS the DB path)."""


# Config:
class CreateBlastDbConfig(BaseConfig):
    """Configuration object for creating a BLAST database.

    Attributes:
        dbtype (Literal['nucl', 'prot']): ``"nucl"`` for DNA/RNA, ``"prot"``
            for protein. Must match the input FASTA.
        out_prefix (str | None): File-path prefix for generated DB files;
            ``None`` falls back to the input FASTA stem.
        title (str | None): Descriptive DB title shown in BLAST reports;
            ``makeblastdb`` falls back to the input file name when ``None``.
        parse_seqids (bool): Parse FASTA seq IDs so ``blastdbcmd`` can
            address sequences by ID; required for v5 taxonomy lookups.
        hash_index (bool): Create a hash index of seq IDs (faster ID lookups).
        blastdb_version (Literal[4, 5]): DB format version. ``5`` (taxonomy-
            aware) is the upstream default since BLAST+ 2.10.
        max_file_sz (str): Max size per DB volume with a unit suffix
            (e.g. ``"1GB"``); upstream caps at ``"4GB"``.
        taxid (int | None): NCBI taxonomy ID assigned to every sequence;
            set to tag a single-organism DB.
        extra_args (list[str]): Extra ``makeblastdb`` CLI tokens passed
            verbatim (e.g. ``["-mask_data", "/path/to/mask"]``). Escape
            hatch for flags not exposed as typed fields above.

    Raises:
        ValueError: If ``dbtype`` is not ``"nucl"`` or ``"prot"``.
    """

    dbtype: Literal["nucl", "prot"] = ConfigField(
        title="Database Type",
        default="nucl",
        description="Database type: `nucl` for DNA/RNA, `prot` for amino acid. Must match the input FASTA.",
    )
    out_prefix: str | None = ConfigField(
        title="Output Prefix",
        default=None,
        description="File-path prefix for the generated DB files; falls back to the input FASTA stem when None.",
        hidden=True,
    )
    title: str | None = ConfigField(
        title="Database Title",
        default=None,
        description="Descriptive DB title shown in search reports; `makeblastdb` falls back to the input file name.",
        advanced=True,
    )
    parse_seqids: bool = ConfigField(
        title="Parse Sequence IDs",
        default=False,
        description="Parse FASTA seq IDs so `blastdbcmd` can address sequences by ID; required for v5 taxonomy.",
        advanced=True,
    )
    hash_index: bool = ConfigField(
        title="Hash Index",
        default=False,
        description="Build a hash index of sequence IDs for faster ID lookups; usually paired with `parse_seqids`.",
        advanced=True,
    )
    blastdb_version: Literal[4, 5] = ConfigField(
        title="BLAST DB Version",
        default=5,
        description="BLAST DB format version: `5` (taxonomy-aware, default since BLAST+ 2.10) or `4` (legacy).",
        advanced=True,
    )
    max_file_sz: str = ConfigField(
        title="Max File Size",
        default="1GB",
        description="Max size per DB volume with unit suffix (e.g. `1GB`, `500MB`); `4GB` is the upstream max.",
        advanced=True,
    )
    taxid: int | None = ConfigField(
        title="Taxonomy ID",
        default=None,
        description="NCBI taxonomy ID assigned to every sequence; set to tag a single-organism DB.",
        advanced=True,
    )
    extra_args: list[str] = ConfigField(
        title="Extra CLI Arguments",
        default=[],
        description="Verbatim `makeblastdb` flags for fields not exposed above (e.g. `-mask_data /path`).",
        advanced=True,
    )

    @field_validator("dbtype")
    @classmethod
    def validate_dbtype(cls, v: str) -> str:
        """Validate database type."""
        if v not in {"nucl", "prot"}:
            raise ValueError('dbtype must be "nucl" or "prot"')
        return v


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CreateBlastDbInput(
        fasta=str(Path(__file__).parent / "example_input_fixture.fasta"),
    )


@tool(
    key="blast-create-db",
    label="Create BLAST Database",
    category="sequence_alignment",
    input_class=CreateBlastDbInput,
    config_class=CreateBlastDbConfig,
    output_class=CreateBlastDbOutput,
    description="Create a local BLAST database from a FASTA file",
    example_input=example_input,
)
def run_create_blast_db(
    inputs: CreateBlastDbInput,
    config: CreateBlastDbConfig,
    instance: Any = None,
) -> CreateBlastDbOutput:
    """Create a local BLAST database from a FASTA file.

    This is the standardized tool interface following the registry pattern.
    Returns structured output with database path.

    Args:
        inputs (CreateBlastDbInput): Validated BLAST database creation input
        config (CreateBlastDbConfig): Validated BLAST database creation configuration

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CreateBlastDbOutput: Structured output with database path

    Raises:
        ValueError: If the BLAST database creation fails.

    Examples:
        >>> from proto_tools.tools.sequence_alignment import (
        ...     run_create_blast_db,
        ...     CreateBlastDbConfig,
        ...     CreateBlastDbInput,
        ... )
        >>> inputs = CreateBlastDbInput(fasta="sequences.fasta")
        >>> config = CreateBlastDbConfig(dbtype="nucl", title="My Database")
        >>> result = run_create_blast_db(inputs, config)
        >>> print(f"Database created at: {result.db_path}")
    """
    fasta_path = Path(inputs.fasta)

    # Default prefix is the FASTA stem in the same directory
    if config.out_prefix is not None:
        out_prefix = str(Path(config.out_prefix).expanduser().resolve())
    else:
        out_prefix = str(fasta_path.with_suffix(""))

    output_data = ToolInstance.dispatch(
        "blast",
        {
            "device": "cpu",
            "operation": "create_blast_db",
            "fasta_path": str(fasta_path),
            "dbtype": config.dbtype,
            "out_prefix": out_prefix,
            "title": config.title,
            "parse_seqids": config.parse_seqids,
            "hash_index": config.hash_index,
            "blastdb_version": config.blastdb_version,
            "max_file_sz": config.max_file_sz,
            "taxid": config.taxid,
            "extra_args": list(config.extra_args),
        },
        instance=instance,
        config=config,
    )

    return CreateBlastDbOutput(
        metadata={
            "dbtype": config.dbtype,
            "fasta_file": str(fasta_path),
            "title": config.title,
        },
        db_path=output_data["db_path"],
    )

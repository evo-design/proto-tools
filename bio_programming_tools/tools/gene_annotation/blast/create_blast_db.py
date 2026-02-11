"""BLAST database creation tool."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator

from bio_programming_tools.tools.infra.tool_io import BaseToolInput, BaseToolOutput
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.tools.utils import BaseConfig, ConfigField


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

    fasta: str = Field(
        description="Path to the input FASTA file containing sequences to create a BLAST database from"
    )

    @field_validator("fasta")
    @classmethod
    def validate_fasta(cls, v: str) -> str:
        """Validate that FASTA file exists"""
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
            the ``local_db`` parameter in ``LocalBlastConfig``. For example, if
            ``db_path`` is ``"/data/mydb"``, ``makeblastdb`` will have created
            multiple files like ``"/data/mydb.nhr"``, ``"/data/mydb.nin"``,
            ``"/data/mydb.nsq"`` (for nucleotide databases) or similar extensions
            for protein databases.
    """
    db_path: str = Field(
        description="Path to the generated BLAST database"
    )

    @property
    def output_format_options(self) -> List[str]:
        return []

    @property
    def output_format_default(self) -> str:
        return ""

    def _export_output(self, export_path: str | Path, file_format: str):
        """Export output - No-op for database creation (output IS the DB path)."""
        pass


# Config:
class CreateBlastDbConfig(BaseConfig):
    """Configuration object for creating a BLAST database.

    This class defines all configuration parameters for creating a local BLAST
    database using ``makeblastdb``. It controls the database type, output location,
    metadata, and indexing options.

    Attributes:
        dbtype (str): The type of database to create:

            - ``"nucl"``: Nucleotide database (for DNA/RNA sequences)
            - ``"prot"``: Protein database (for amino acid sequences)

            This must match the sequence type in the input FASTA file.

        out_prefix (Optional[str]): Optional file path prefix for the generated
            database files. If not specified, the database files will be created
            in the same directory as the input FASTA file, using the FASTA filename
            (without extension) as the prefix. For example, if the input is
            ``"sequences.fasta"`` and ``out_prefix`` is ``None``, the database will
            be named ``"sequences"``. If ``out_prefix`` is ``"/data/mydb"``, the
            database files will be created as ``"/data/mydb.nhr"``,
            ``"/data/mydb.nin"``, ``"/data/mydb.nsq"``, etc.

        title (Optional[str]): Optional descriptive title for the database. This
            title will be displayed in BLAST search results and can help identify
            the database contents. If not specified, ``makeblastdb`` will use the
            input filename.

        additional_params (Dict[str, str | int | float | bool]): Dictionary
            of additional parameters for ``makeblastdb``. Common options include:

            - ``"parse_seqids"``: Parse sequence IDs in the FASTA file (boolean)
            - ``"hash_index"``: Create hash index for faster lookups (boolean)
            - ``"max_file_sz"``: Maximum file size for database volumes (string, e.g., ``"4GB"``)
            - ``"taxid"``: Assign this taxonomy ID to all sequences (integer)
            - ``"taxid_map"``: File mapping sequence IDs to taxonomy IDs (string path)
            - ``"logfile"``: Path to log file for ``makeblastdb`` output (string path)
            - ``"blastdb_version"``: BLAST database version (4 or 5, default: 5)

    Raises:
        ValueError: If ``dbtype`` is not ``"nucl"`` or ``"prot"``.
    """
    dbtype: Literal["nucl", "prot"] = ConfigField(
        title="Database Type",
        default="nucl",
        description="Specifies the type of database to create: Nucleotide or Protein",
    )
    # TODO: Determine how to handle this for the client.
    out_prefix: Optional[str] = ConfigField(
        title="Output Prefix",
        default=None,
        description="File-path prefix for database files (default: FASTA stem)",
    )
    title: Optional[str] = ConfigField(
        title="Database Title",
        default=None,
        description="Optional name for the database",
    )
    additional_params: Dict[str, str | int | float | bool] = ConfigField(
        title="Additional Parameter Dictionary",
        default_factory=dict,
        description="Extra flags for makeblastdb (e.g., parse_seqids, hash_index)",
        advanced=True,
    )

    @field_validator('dbtype')
    @classmethod
    def validate_dbtype(cls, v: str) -> str:
        """Validate database type"""
        if v not in {"nucl", "prot"}:
            raise ValueError('dbtype must be "nucl" or "prot"')
        return v


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="create-blast-db",
    label="Create BLAST Database",
    input=CreateBlastDbInput,
    config=CreateBlastDbConfig,
    output=CreateBlastDbOutput,
    description="Create a local BLAST database from a FASTA file",
)
def run_create_blast_db(
    inputs: CreateBlastDbInput, config: CreateBlastDbConfig
) -> CreateBlastDbOutput:
    """
    Create a local BLAST database from a FASTA file.

    This is the standardized tool interface following the registry pattern.
    Returns structured output with database path.

    Args:
        inputs (CreateBlastDbInput): Validated BLAST database creation input
        config (CreateBlastDbConfig): Validated BLAST database creation configuration

    Returns:
        Structured output with database path

    Raises:
        ValueError: If the BLAST database creation fails.

    Examples:
        >>> from bio_programming_tools.tools.gene_annotation import run_create_blast_db, CreateBlastDbConfig, CreateBlastDbInput
        >>> inputs = CreateBlastDbInput(fasta="sequences.fasta")
        >>> config = CreateBlastDbConfig(
        ...     dbtype="nucl",
        ...     title="My Database"
        ... )
        >>> result = run_create_blast_db(inputs, config)
        >>> print(f"Database created at: {result.db_path}")
    """

    from bio_programming_tools.tools.infra.env_manager import EnvManager

    fasta_path = Path(inputs.fasta)

    # Default prefix is the FASTA stem in the same directory
    if config.out_prefix is not None:
        out_prefix = str(Path(config.out_prefix).expanduser().resolve())
    else:
        out_prefix = str(fasta_path.with_suffix(""))

    venv_manager = EnvManager(model_name="blast")

    input_data = {
        "operation": "create_blast_db",
        "fasta_path": str(fasta_path),
        "dbtype": config.dbtype,
        "out_prefix": out_prefix,
        "title": config.title,
        "additional_params": config.additional_params,
    }

    output_data = venv_manager.call_standalone_script_in_venv(
        script_path=Path(__file__).parent / "standalone" / "run.py",
        input_dict=input_data,
        device="cpu",
    )

    return CreateBlastDbOutput(
        metadata={
            "dbtype": config.dbtype,
            "fasta_file": str(fasta_path),
            "title": config.title,
        },
        db_path=output_data["db_path"],
    )

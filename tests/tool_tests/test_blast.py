"""
test_blast.py

Tests for BLAST tools in bio_programming.bio_tools.tools.gene_annotation.blast
"""

import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError
from bio_programming.bio_tools.tools.gene_annotation import (
    run_online_blast_search,
    run_local_blast_search,
    run_create_blast_db,
    OnlineBlastInput,
    OnlineBlastConfig,
    LocalBlastInput,
    LocalBlastConfig,
    CreateBlastDbInput,
    CreateBlastDbConfig,
    BlastOutput,
)


# ============================================================================
# Config Validation Tests
# ============================================================================
def test_online_blast_config_invalid_program():
    """Test that invalid BLAST program raises ValueError."""
    with pytest.raises(ValidationError, match="Input should be"):
        OnlineBlastConfig(
            program="invalid_program",
            database="nt"
        )


def test_local_blast_input_validation():
    """Test that LocalBlastInput validates file existence."""
    # Create a temporary query file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">test\nATGCGTAAA\n")
        query_file = f.name

    try:
        # Valid input
        inputs = LocalBlastInput(
            query=query_file
        )
        assert Path(inputs.query).exists()
    finally:
        Path(query_file).unlink()

    # Invalid - file doesn't exist
    with pytest.raises(ValueError, match="Query file not found"):
        LocalBlastInput(
            query="/nonexistent/file.fasta"
        )


def test_local_blast_config_validation():
    """Test LocalBlastConfig validation."""
    _ = LocalBlastConfig(local_db="/data/blast/nr")


def test_create_blast_db_input_validation():
    """Test CreateBlastDbInput validation."""
    # Create temporary FASTA file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">seq1\nATGCGTAAA\n>seq2\nCCCGGGTTT\n")
        fasta_file = f.name

    try:
        inputs = CreateBlastDbInput(
            fasta=fasta_file
        )
        assert inputs.fasta == fasta_file
    finally:
        Path(fasta_file).unlink()


def test_create_blast_db_config_validation():
    """Test CreateBlastDbConfig validation."""
    config = CreateBlastDbConfig(
        dbtype="nucl"
    )
    assert config.dbtype == "nucl"
    assert config.title is None


def test_create_blast_db_config_invalid_dbtype():
    """Test that invalid dbtype raises ValueError."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">seq1\nATGCGTAAA\n")
        fasta_file = f.name

    try:
        with pytest.raises(ValidationError, match="Input should be"):
            CreateBlastDbConfig(
                dbtype="invalid"
            )
    finally:
        Path(fasta_file).unlink()


def test_create_blast_db_config_missing_file():
    """Test that missing FASTA file raises ValueError."""
    with pytest.raises(ValueError, match="FASTA file not found"):
        CreateBlastDbInput(
            fasta="/nonexistent/file.fasta"
        )


# ============================================================================
# Additional Params Tests
# ============================================================================

def test_additional_params_type_validation():
    """Test that additional_params validates value types."""
    # Valid types
    config = OnlineBlastConfig(
        additional_params={
            "word_size": 11,           # int
            "evalue": 0.001,          # float
            "ungapped": True,         # bool
            "matrix": "BLOSUM62"      # str
        }
    )
    assert config.additional_params["word_size"] == 11
    assert config.additional_params["evalue"] == 0.001
    assert config.additional_params["ungapped"] is True
    assert config.additional_params["matrix"] == "BLOSUM62"


# ============================================================================
# Registry Integration Tests
# ============================================================================

def test_blast_tools_registered():
    """Test that BLAST tools are registered in ToolRegistry."""
    from bio_programming.bio_tools.tools.tool_registry import ToolRegistry

    all_tools = ToolRegistry.list_all()

    # Check all BLAST tools are registered
    assert "online-blast" in {spec.key for spec in all_tools}
    assert "local-blast" in {spec.key for spec in all_tools}
    assert "create-blast-db" in {spec.key for spec in all_tools}

    # Convert to dict for easy access
    tools_dict = {spec.key: spec for spec in all_tools}

    # Check metadata
    online_spec = tools_dict["online-blast"]
    assert online_spec.description == "Submit query to online NCBI BLAST and return results"
    assert tools_dict.get("create-blast-db", None) is not None, "create-blast-db tool is not registered"


def test_blast_config_schema_generation():
    """Test that config schemas are properly generated."""
    from bio_programming.bio_tools.tools.tool_registry import ToolRegistry

    # Get schema for online BLAST (returns config schema)
    schema = ToolRegistry.get_schema("online-blast")

    assert "properties" in schema
    # Config fields should be in config schema
    assert "program" in schema["properties"]
    assert "database" in schema["properties"]
    assert "additional_params" in schema["properties"]


def test_create_blast_db_execution():
    """Test create_blast_db execution (requires BLAST+ installed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test FASTA
        fasta_path = Path(tmpdir) / "test.fasta"
        fasta_path.write_text(">seq1\nATGCGTAAA\n>seq2\nCCCGGGTTT\n")
        
        # Create database
        inputs = CreateBlastDbInput(
            fasta=str(fasta_path)
        )
        config = CreateBlastDbConfig(
            dbtype="nucl",
            title="Test Database"
        )

        result = run_create_blast_db(inputs, config)
        
        assert result.success is True
        assert Path(result.db_path).parent.resolve() == fasta_path.parent.resolve()
        assert result.execution_time > 0

@pytest.mark.skip(reason="Test hangs when contacting NCBI servers")
def test_online_blast_execution():
    """Test online_blast execution (requires internet)."""
    inputs = OnlineBlastInput(
        query="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"
    )
    config = OnlineBlastConfig(
        program="blastp",
        database="nr",
        additional_params={"hitlist_size": 5}
    )

    result = run_online_blast_search(inputs, config)
    
    assert isinstance(result, BlastOutput)
    assert result.tool_id == "online-blast"
    # May or may not have hits depending on database state
    assert result.num_hits >= 0

@pytest.mark.skip(reason="Test hangs when on Chimera")
def test_local_blast_execution():
    """Test local_blast execution (requires BLAST+ and database)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">test\nATGCGTAAA\n")
        query_file = f.name
    
    try:
        inputs = LocalBlastInput(
            query=query_file
        )
        config = LocalBlastConfig(
            db="nt",  # Assumes nt database is installed
            program="blastn",
            num_threads=2
        )

        result = run_local_blast_search(inputs, config)
        
        assert isinstance(result, BlastOutput)
        assert result.tool_id == "local-blast"
        assert result.execution_time >= 0
    finally:
        Path(query_file).unlink()

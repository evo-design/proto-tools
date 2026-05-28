"""tests/sequence_alignment_tests/test_remote_colabfold.py.

Tests for remote ColabFold MSA search.
"""

from pathlib import Path

import pytest

from proto_tools.tools.sequence_alignment.colabfold_search import colabfold_search as cfs
from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
    ColabfoldSearchInput,
    run_colabfold_search,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

# ============================================================================
# Test Data
# ============================================================================

# Small protein sequence that should have homologs
SAMPLE_PROTEIN_SEQ = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"

# Canonical UniRef homolog header — must survive the query-row rewrite.
HOMOLOG_HEADER = "UniRef100_Q43227"
HOMOLOG_SEQ = "MVLSAKDKTNIKTAWGKIGGHAAEYGAEALERMFVVYPTT"


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------


def test_remote_query_header_rewritten_to_sequence_id(tmp_path, monkeypatch):
    """Remote A3M query row echoes the resolved sequence_id, not ColabFold's ``101``."""
    seq_id = "my_protein"

    def fake_dispatch(toolkit, input_data, **kwargs):
        out_dir = Path(input_data["output_dir"]) / "msas"
        out_dir.mkdir(parents=True, exist_ok=True)
        a3m_path = out_dir / f"{seq_id}.a3m"
        a3m_path.write_text(f">101\n{SAMPLE_PROTEIN_SEQ}\n>{HOMOLOG_HEADER}\t101\t0.8\n{HOMOLOG_SEQ}\n")
        return {"msa_paths": {seq_id: str(a3m_path)}, "success": True, "num_successful": 1, "num_failed": 0}

    monkeypatch.setattr(cfs.ToolInstance, "dispatch", fake_dispatch)

    inputs = ColabfoldSearchInput(queries=[(SAMPLE_PROTEIN_SEQ, seq_id)])
    config = ColabfoldSearchConfig(search_mode="remote", output_dir=str(tmp_path))

    result = run_colabfold_search(inputs, config)
    validate_output(result)

    msa = result.results[0].msa
    assert msa is not None
    assert msa.sequence_ids[0] == seq_id
    assert msa.sequence_ids[1].startswith(HOMOLOG_HEADER)

    lines = (tmp_path / "msas" / f"{seq_id}.a3m").read_text().splitlines()
    assert lines[0] == f">{seq_id}"
    assert lines[2].startswith(f">{HOMOLOG_HEADER}")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skip_ci
@pytest.mark.integration
def test_remote_colabfold_search_no_metagenomic_db(tmp_path):
    """Test remote search with use_metagenomic_db=False."""
    inputs = ColabfoldSearchInput(queries=[(SAMPLE_PROTEIN_SEQ, "test_no_meta")])

    config = ColabfoldSearchConfig(
        search_mode="remote",
        output_dir=str(tmp_path),
        use_metagenomic_db=False,
        verbose=True,
    )

    result = run_colabfold_search(inputs, config)

    # Validate output and export functionality
    validate_output(result)

    # Verify it completes successfully
    assert len(result.results) == 1
    assert result.results[0].sequence_id == "test_no_meta"
    # Should find homologs for this sequence
    msa = result.results[0].msa
    assert msa is not None, f"MSA is None for sequence {result.results[0].sequence_id}"
    assert msa.num_sequences > 100
    assert msa.alignment_length == 40


@pytest.mark.skip(reason="Metagenomic DB hit count varies between API updates; needs range-based assertion")
@pytest.mark.skip_ci
@pytest.mark.integration
def test_remote_colabfold_search_with_metagenomic_db(tmp_path):
    """Test remote search with use_metagenomic_db=True."""
    inputs = ColabfoldSearchInput(queries=[(SAMPLE_PROTEIN_SEQ, "test_with_meta")])

    config = ColabfoldSearchConfig(
        search_mode="remote",
        output_dir=str(tmp_path),
        use_metagenomic_db=True,  # Enable metagenomic database
        verbose=True,
    )

    result = run_colabfold_search(inputs, config)

    # Verify it completes successfully with metagenomic DB
    assert len(result.results) == 1
    msa = result.results[0].msa
    assert msa.num_sequences > 100
    assert msa.alignment_length == 40


@pytest.mark.skip_ci
@pytest.mark.integration
def test_remote_colabfold_search_custom_output_dir(tmp_path):
    """Test remote search with custom output_dir."""
    custom_output = tmp_path / "my_custom_output"

    inputs = ColabfoldSearchInput(queries=[(SAMPLE_PROTEIN_SEQ, "test_output")])

    config = ColabfoldSearchConfig(
        search_mode="remote",
        output_dir=str(custom_output),
        use_metagenomic_db=False,
        verbose=True,
    )

    result = run_colabfold_search(inputs, config)

    # Validate output and export functionality
    validate_output(result)

    # Verify custom output directory was created and used
    assert custom_output.exists()
    assert (custom_output / "msas").exists()
    assert (custom_output / "msas" / "test_output.a3m").exists()

    # Verify result is valid
    assert len(result.results) == 1
    assert result.results[0].msa is not None

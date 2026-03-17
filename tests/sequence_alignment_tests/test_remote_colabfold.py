"""Tests for remote ColabFold MSA search."""

import pytest

from bio_programming_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
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


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skip_ci
@pytest.mark.integration
def test_remote_colabfold_search_no_metagenomic_db(tmp_path):
    """Test remote search with use_metagenomic_db=False."""
    inputs = ColabfoldSearchInput(
        queries=[(SAMPLE_PROTEIN_SEQ, "test_no_meta")]
    )

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
    assert (
        msa is not None
    ), f"MSA is None for sequence {result.results[0].sequence_id}"
    assert msa.num_sequences > 100
    assert msa.alignment_length == 40


@pytest.mark.skip(reason="Metagenomic DB hit count varies between API updates; needs range-based assertion")
@pytest.mark.skip_ci
@pytest.mark.integration
def test_remote_colabfold_search_with_metagenomic_db(tmp_path):
    """Test remote search with use_metagenomic_db=True."""
    inputs = ColabfoldSearchInput(
        queries=[(SAMPLE_PROTEIN_SEQ, "test_with_meta")]
    )

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

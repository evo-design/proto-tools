"""
test_remote_colabfold.py

Tests for Remote ColabFold MSA search configuration parameters.

These tests verify that the config parameters (use_metagenomic_db, output_dir, verbose)
function correctly with remote mode.
"""

import os
import tempfile

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


# ============================================================================
# Config Parameter Tests
# ============================================================================


@pytest.mark.skip_ci
class TestRemoteColabfoldSearch:
    """Tests for remote-specific configuration parameters."""

    def test_use_metagenomic_db_false(self):
        """Test remote search with use_metagenomic_db=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = ColabfoldSearchInput(
                queries=[(SAMPLE_PROTEIN_SEQ, "test_no_meta")]
            )

            config = ColabfoldSearchConfig(
                search_mode="remote",
                output_dir=tmpdir,
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
            assert msa.num_sequences == 2392
            assert msa.alignment_length == 40

    @pytest.mark.skip("metagenomic db num_sequences depends")
    def test_use_metagenomic_db_true(self):
        """Test remote search with use_metagenomic_db=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = ColabfoldSearchInput(
                queries=[(SAMPLE_PROTEIN_SEQ, "test_with_meta")]
            )

            config = ColabfoldSearchConfig(
                search_mode="remote",
                output_dir=tmpdir,
                use_metagenomic_db=True,  # Enable metagenomic database
                verbose=True,
            )

            result = run_colabfold_search(inputs, config)

            # Verify it completes successfully with metagenomic DB
            assert len(result.results) == 1
            msa = result.results[0].msa
            assert msa.num_sequences == 2392
            assert msa.alignment_length == 40

    def test_custom_output_dir(self):
        """Test remote search with custom output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_output = os.path.join(tmpdir, "my_custom_output")

            inputs = ColabfoldSearchInput(queries=[(SAMPLE_PROTEIN_SEQ, "test_output")])

            config = ColabfoldSearchConfig(
                search_mode="remote",
                output_dir=custom_output,
                use_metagenomic_db=False,
                verbose=True,
            )

            result = run_colabfold_search(inputs, config)

            # Validate output and export functionality
            validate_output(result)

            # Verify custom output directory was created and used
            assert os.path.exists(custom_output)
            assert os.path.exists(os.path.join(custom_output, "msas"))
            assert os.path.exists(
                os.path.join(custom_output, "msas", "test_output.a3m")
            )

            # Verify result is valid
            assert len(result.results) == 1
            assert result.results[0].msa is not None


# ============================================================================
# Standalone Test Runner
# ============================================================================


if __name__ == "__main__":
    """Run a simple manual test for debugging."""
    print("Running manual test of remote ColabFold search config parameters...\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        inputs = ColabfoldSearchInput(queries=[(SAMPLE_PROTEIN_SEQ, "manual_test")])

        print("Test 1: use_metagenomic_db=False")
        config = ColabfoldSearchConfig(
            search_mode="remote",
            output_dir=tmpdir,
            use_metagenomic_db=False,
            verbose=True,
        )

        try:
            result = run_colabfold_search(inputs, config)
            print(
                f"✓ Success - Found {result.results[0].num_homologs_found} homologs\n"
            )
        except Exception as e:
            print(f"✗ Failed: {e}\n")

        print("Test 2: use_metagenomic_db=True")
        config2 = ColabfoldSearchConfig(
            search_mode="remote",
            output_dir=tmpdir,
            use_metagenomic_db=True,
            verbose=True,
        )

        try:
            result2 = run_colabfold_search(inputs, config2)
            print(
                f"✓ Success - Found {result2.results[0].num_homologs_found} homologs\n"
            )
        except Exception as e:
            print(f"✗ Failed: {e}\n")

    print("All manual tests completed!")

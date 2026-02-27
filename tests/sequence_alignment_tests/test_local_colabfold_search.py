"""
test_local_colabfold_search.py

Tests for Local ColabFold MSA search tool in bio_programming_tools.tools.sequence_alignment.colabfold_search
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from bio_programming_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
    ColabfoldSearchInput,
    ColabfoldSearchQuery,
    run_colabfold_search,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

# ============================================================================
# Test Data
# ============================================================================

SAMPLE_PROTEIN_SEQ_1 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"
SAMPLE_PROTEIN_SEQ_2 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV"
SAMPLE_PROTEIN_SEQ_3 = "ACDEFGHIKLMNPQRSTVWY"

# Sample A3M content with insertions (lowercase)
SAMPLE_A3M_WITH_INSERTIONS = """>seq1
MVLSPADKTNVKAAWgkvGKVGAHAGEYGAEALERMFLSFPTT
>seq2
MKTAYIAKQRQISFVKSHFSrqlRQLEERLGLIEVQAPILSRV
>seq3
ACDEFGHIKLMNPQRSTVWYxyz
"""

SAMPLE_A3M_CLEAN = """>seq1
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT
>seq2
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV
>seq3
ACDEFGHIKLMNPQRSTVWY
"""


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestColabfoldSearchQuery:
    """Tests for ColabfoldSearchQuery validation."""

    def test_valid_query(self):
        query = ColabfoldSearchQuery(
            sequence=SAMPLE_PROTEIN_SEQ_1, sequence_id="test_seq"
        )
        assert query.sequence == SAMPLE_PROTEIN_SEQ_1
        assert query.sequence_id == "test_seq"

    def test_query_without_id(self):
        query = ColabfoldSearchQuery(sequence=SAMPLE_PROTEIN_SEQ_1)
        assert query.sequence == SAMPLE_PROTEIN_SEQ_1
        assert query.sequence_id is None

    def test_empty_sequence_fails(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            ColabfoldSearchQuery(sequence="")

    def test_whitespace_only_sequence_fails(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            ColabfoldSearchQuery(sequence="   ")

    def test_sequence_strips_whitespace(self):
        query = ColabfoldSearchQuery(sequence="  MVLS  ")
        assert query.sequence == "MVLS"


class TestColabfoldSearchInput:
    """Tests for ColabfoldSearchInput validation and normalization."""

    def test_single_sequence_string(self):
        """Test input with single sequence string."""
        inputs = ColabfoldSearchInput(queries=SAMPLE_PROTEIN_SEQ_1)
        assert len(inputs.queries) == 1
        assert inputs.queries[0].sequence == SAMPLE_PROTEIN_SEQ_1
        assert inputs.queries[0].sequence_id is not None  # Auto-generated

    def test_list_of_sequence_strings(self):
        """Test input with list of sequence strings."""
        inputs = ColabfoldSearchInput(
            queries=[SAMPLE_PROTEIN_SEQ_1, SAMPLE_PROTEIN_SEQ_2]
        )
        assert len(inputs.queries) == 2
        assert inputs.queries[0].sequence == SAMPLE_PROTEIN_SEQ_1
        assert inputs.queries[1].sequence == SAMPLE_PROTEIN_SEQ_2
        # IDs should be auto-generated and unique
        assert inputs.queries[0].sequence_id is not None
        assert inputs.queries[1].sequence_id is not None
        assert inputs.queries[0].sequence_id != inputs.queries[1].sequence_id

    def test_single_tuple_format(self):
        """Test input with single tuple (sequence, id)."""
        inputs = ColabfoldSearchInput(queries=(SAMPLE_PROTEIN_SEQ_1, "protein_A"))
        assert len(inputs.queries) == 1
        assert inputs.queries[0].sequence == SAMPLE_PROTEIN_SEQ_1
        assert inputs.queries[0].sequence_id == "protein_A"

    def test_list_of_tuples(self):
        """Test input with list of tuples."""
        inputs = ColabfoldSearchInput(
            queries=[
                (SAMPLE_PROTEIN_SEQ_1, "protein_A"),
                (SAMPLE_PROTEIN_SEQ_2, "protein_B"),
            ]
        )
        assert len(inputs.queries) == 2
        assert inputs.queries[0].sequence_id == "protein_A"
        assert inputs.queries[1].sequence_id == "protein_B"

    def test_list_of_query_objects(self):
        """Test input with explicit Query objects."""
        query1 = ColabfoldSearchQuery(sequence=SAMPLE_PROTEIN_SEQ_1, sequence_id="seq1")
        query2 = ColabfoldSearchQuery(sequence=SAMPLE_PROTEIN_SEQ_2, sequence_id="seq2")
        inputs = ColabfoldSearchInput(queries=[query1, query2])
        assert len(inputs.queries) == 2
        assert inputs.queries[0].sequence_id == "seq1"
        assert inputs.queries[1].sequence_id == "seq2"

    def test_single_query_object(self):
        """Test input with single Query object."""
        query = ColabfoldSearchQuery(sequence=SAMPLE_PROTEIN_SEQ_1, sequence_id="seq1")
        inputs = ColabfoldSearchInput(queries=query)
        assert len(inputs.queries) == 1
        assert inputs.queries[0].sequence_id == "seq1"

    def test_mixed_format_list(self):
        """Test input with mixed formats in list."""
        query1 = ColabfoldSearchQuery(
            sequence=SAMPLE_PROTEIN_SEQ_1, sequence_id="explicit"
        )
        inputs = ColabfoldSearchInput(
            queries=[
                query1,
                SAMPLE_PROTEIN_SEQ_2,
                (SAMPLE_PROTEIN_SEQ_3, "tuple_id"),
            ]
        )
        assert len(inputs.queries) == 3
        assert inputs.queries[0].sequence_id == "explicit"
        assert inputs.queries[1].sequence_id is not None  # Auto-generated
        assert inputs.queries[2].sequence_id == "tuple_id"

    def test_empty_queries_fails(self):
        """Test that empty queries list fails."""
        with pytest.raises(ValidationError, match="At least one query"):
            ColabfoldSearchInput(queries=[])

    def test_duplicate_sequence_ids_fails(self):
        """Test that duplicate sequence IDs fail validation."""
        with pytest.raises(ValidationError, match="not unique"):
            ColabfoldSearchInput(
                queries=[
                    (SAMPLE_PROTEIN_SEQ_1, "same_id"),
                    (SAMPLE_PROTEIN_SEQ_2, "same_id"),
                ]
            )

    def test_auto_generated_ids_are_unique(self):
        """Test that auto-generated IDs are unique."""
        inputs = ColabfoldSearchInput(
            queries=[SAMPLE_PROTEIN_SEQ_1, SAMPLE_PROTEIN_SEQ_2]
        )
        ids = {q.sequence_id for q in inputs.queries}
        assert len(ids) == 2  # All unique

    def test_auto_generated_ids_deterministic(self):
        """Test that auto-generated IDs are deterministic (hash-based)."""
        inputs1 = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ_1])
        inputs2 = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ_1])
        assert inputs1.queries[0].sequence_id == inputs2.queries[0].sequence_id

    def test_list_like_interface(self):
        """Test list-like operations on input."""
        inputs = ColabfoldSearchInput(
            queries=[SAMPLE_PROTEIN_SEQ_1, SAMPLE_PROTEIN_SEQ_2]
        )
        assert len(inputs) == 2
        assert inputs[0].sequence == SAMPLE_PROTEIN_SEQ_1
        assert inputs[1].sequence == SAMPLE_PROTEIN_SEQ_2
        # Test iteration
        sequences = [q.sequence for q in inputs]
        assert sequences == [SAMPLE_PROTEIN_SEQ_1, SAMPLE_PROTEIN_SEQ_2]


# ============================================================================
# Integration Tests (Full)
# ============================================================================

# Path to test database
TEST_DB_DIR = os.path.join(
    os.path.dirname(__file__), "..", "dummy_data", "mini_mmseqs_db"
)
TEST_DB_SETUP_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "dummy_data", "create_mini_mmseqs_db.py"
)


@pytest.fixture(scope="class", autouse=True)
def setup_mini_database():
    """Automatically setup mini database if it doesn't exist."""
    db_sentinel = os.path.join(TEST_DB_DIR, "uniref30_mini_db.dbtype")
    if not os.path.exists(db_sentinel):
        print(f"\n⚙️  Setting up mini MMseqs database...")
        print(f"   Running: {TEST_DB_SETUP_SCRIPT}")
        try:
            result = subprocess.run(
                [sys.executable, TEST_DB_SETUP_SCRIPT],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            print(f"✓ Mini database setup complete!")
            if result.stdout:
                print(f"   Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            pytest.fail(
                f"Failed to setup mini database:\n"
                f"Exit code: {e.returncode}\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Database setup timed out after 10 minutes")
    yield


@pytest.mark.skipif(
    not os.path.exists(TEST_DB_SETUP_SCRIPT),
    reason="Test database setup script not found.",
)
@pytest.mark.slow
class TestColabfoldSearchExecutionDebugDatabase:
    """End-to-end tests using real run_colabfold_search command with test database."""

    @pytest.mark.include_in_env_report(category="sequence_alignment")
    @pytest.mark.skip_ci
    def test_finding_self_in_database(self):
        """Test end-to-end search with a single sequence against test database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Bacillus subtilis YunC protein
            # NOTE: This sequence is in the database. Colabfold should be able to find it
            test_sequence = "MVNLTPIMIEGQPFTAVTVKLPKTNFMAVANDHGYIMCGALDVALLNEKLKERGIVAGRAVGVRTIDQLLDAPLESVTYA"

            config = ColabfoldSearchConfig(
                search_mode="local",
                msa_db_dir=TEST_DB_DIR,
                database_name="uniref30_mini_db",
                output_dir=tmpdir,
                use_metagenomic_db=False,
                num_threads=2,
                verbose=False,
            )

            inputs = ColabfoldSearchInput(queries=[(test_sequence, "test_query")])

            # Execute the search
            result = run_colabfold_search(inputs, config)

            # Validate output and export functionality
            validate_output(result)

            # Verify results
            assert len(result) == len(
                inputs.queries
            ), "Expected number of results to be equal to the number of queries"
            assert result.results[0].sequence_id == "test_query"

            # Verify MSA was created
            assert result.results[0].msa is not None
            assert result.results[0].num_homologs_found == 1  # query should find itself

            # Verify we can iterate through the MSA
            sequences = list(result.results[0].msa)
            assert len(sequences) == 2
            assert sequences[0] == sequences[1], "The two sequences should be the same"

            # Verify the output file exists
            msa_path = os.path.join(tmpdir, "msas", "test_query.a3m")
            assert os.path.exists(msa_path)

    @pytest.mark.skip_ci
    def test_finding_homologs(self):
        """Test end-to-end search with a sequence that has close homologs in the database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            srs = "MRAKKRFGQNFLIDQNIINKIVDSSEVENRNIIEIGPGKGALTKILVKKANKVLAYEIDQDMVNILNQQISSKNFVLINKDFLKEEFDKSQNYNIVANIPYYITSDIIFKIIENHQIFDQATLMVQKEVALRILAKQNDSEFSKLSLSVQFFFDVFLICDVSKNSFRPIPKVDSAVIKLVKKKNKDFSLWKEYFEFLKIAFSSRRKTLLNNLKYFFNEQKILKFFELKNYDPKVRAQNIKNEDFYALFLELR"

            config = ColabfoldSearchConfig(
                search_mode="local",
                msa_db_dir=TEST_DB_DIR,
                database_name="uniref30_mini_db",
                output_dir=tmpdir,
                use_metagenomic_db=False,
                num_threads=2,
                verbose=False,
            )

            inputs = ColabfoldSearchInput(
                queries=[(srs, "small_ribosomal_subunit_query")]
            )

            # Execute the search
            result = run_colabfold_search(inputs, config)

            # Validate output and export functionality
            validate_output(result)

            assert len(result) == len(
                inputs.queries
            ), "Expected number of results to be equal to the number of queries"

            msa = result.results[0].msa
            assert len(msa) == 439

    @pytest.mark.skip_ci
    def test_multiple_sequences_search(self):
        """Test end-to-end search with multiple sequences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # NOTE: The first two sequences are in the database and have close homologs
            # The third should not have any homologs
            test_sequences = [
                (
                    "MRAKKRFGQNFLIDQNIINKIVDSSEVENRNIIEIGPGKGALTKILVKKANKVLAYEIDQDMVNILNQQISSKNFVLINKDFLKEEFDKSQNYNIVANIPYYITSDIIFKIIENHQIFDQATLMVQKEVALRILAKQNDSEFSKLSLSVQFFFDVFLICDVSKNSFRPIPKVDSAVIKLVKKKNKDFSLWKEYFEFLKIAFSSRRKTLLNNLKYFFNEQKILKFFELKNYDPKVRAQNIKNEDFYALFLELR",
                    "Ribosomal RNA small subunit methyltransferase A OS=Hydrogenobaculum sp",
                ),
                (
                    "MVSASLDGGLRICVRASAPEVHDKAVAWASFLKAPLNPENPEQYFFHFFVEPEGVYVRDQEKRLLEIDFDKNHLDYERKGHRGKNELIAKALGVAKGARRILDLSVGMGIDSVFLTQLGFSVIGVERSPVLYALLKEAFARTKKDSLKSYELHFADSLQFLKQNKGLLEVDAIYFDPMYPHKKKSALPKQEMVVFRDLVGHDDDASLVLQEALTWPVKRVVVKRPMQAEELLPGVRHSYEGKVVRYDTYVVG",
                    "Ribosomal RNA small subunit methyltransferase A OS=Mycoplasmopsis pulmonis",
                ),
                ("AAAAAAAAAA", "bad_query"),
            ]

            config = ColabfoldSearchConfig(
                search_mode="local",
                msa_db_dir=TEST_DB_DIR,
                database_name="uniref30_mini_db",
                output_dir=tmpdir,
                use_metagenomic_db=False,
                num_threads=2,
                verbose=False,
            )

            inputs = ColabfoldSearchInput(queries=test_sequences)

            # Execute the search
            result = run_colabfold_search(inputs, config)

            # Validate output and export functionality
            validate_output(result)

            # Verify results
            assert len(result.results) == len(
                inputs.queries
            ), "Expected number of results to be equal to the number of queries"

            # First two MSAs should have a lot of homologs
            assert len(result.results[0].msa) == 439
            assert len(result.results[1].msa) == 107

            # Third MSA should have no homologs (returns None)
            assert result.results[2].msa is None
            assert result.results[2].num_homologs_found == 0

    @pytest.mark.skip_ci
    def test_with_sensitivity(self):
        """Test search with sensitivity parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_sequences = [
                (
                    "MRAKKRFGQNFLIDQNIINKIVDSSEVENRNIIEIGPGKGALTKILVKKANKVLAYEIDQDMVNILNQQISSKNFVLINKDFLKEEFDKSQNYNIVANIPYYITSDIIFKIIENHQIFDQATLMVQKEVALRILAKQNDSEFSKLSLSVQFFFDVFLICDVSKNSFRPIPKVDSAVIKLVKKKNKDFSLWKEYFEFLKIAFSSRRKTLLNNLKYFFNEQKILKFFELKNYDPKVRAQNIKNEDFYALFLELR",
                    "Ribosomal RNA small subunit methyltransferase A OS=Hydrogenobaculum sp",
                ),
            ]

            config = ColabfoldSearchConfig(
                search_mode="local",
                msa_db_dir=TEST_DB_DIR,
                database_name="uniref30_mini_db",
                output_dir=tmpdir,
                use_metagenomic_db=False,
                num_threads=2,
                sensitivity=1.0,
                verbose=False,
            )

            inputs = ColabfoldSearchInput(queries=test_sequences)

            # Execute the search
            result = run_colabfold_search(inputs, config)

            # Validate output and export functionality
            validate_output(result)

            # Verify results
            assert len(result.results) == len(
                inputs.queries
            ), "Expected number of results to be equal to the number of queries"

            assert len(result.results[0].msa) == 425


from bio_programming_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    CHIMERA_COLABFOLD_DB_LOCATION,
)


@pytest.mark.skipif(
    not (Path(CHIMERA_COLABFOLD_DB_LOCATION)).exists(),
    reason="Full database not found. Skipping",
)
class TestColabfoldSearchExecutionFullDatabase:

    @pytest.mark.only_chimera
    @pytest.mark.uses_gpu
    @pytest.mark.slow
    def test_gpu_acceleration(self):
        """
        Test GPU acceleration
        """
        # Erase this line to run the test
        pytest.skip("GPU acceleration is not currently supported for run_colabfold_search")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_sequences = [
                (
                    "MRAKKRFGQNFLIDQNIINKIVDSSEVENRNIIEIGPGKGALTKILVKKANKVLAYEIDQDMVNILNQQISSKNFVLINKDFLKEEFDKSQNYNIVANIPYYITSDIIFKIIENHQIFDQATLMVQKEVALRILAKQNDSEFSKLSLSVQFFFDVFLICDVSKNSFRPIPKVDSAVIKLVKKKNKDFSLWKEYFEFLKIAFSSRRKTLLNNLKYFFNEQKILKFFELKNYDPKVRAQNIKNEDFYALFLELR",
                    "Ribosomal RNA small subunit methyltransferase A OS=Hydrogenobaculum sp",
                ),
            ]

            config = ColabfoldSearchConfig(
                search_mode="local",
                msa_db_dir=CHIMERA_COLABFOLD_DB_LOCATION,
                output_dir=tmpdir,
                use_gpu=True,
                verbose=False,
            )

            inputs = ColabfoldSearchInput(queries=test_sequences)

            # Execute the search
            result = run_colabfold_search(inputs, config)

            # Verify results
            assert result.success, f"Search failed with errors: {result.errors}"
            assert len(result.results) == len(
                inputs.queries
            ), "Expected number of results to be equal to the number of queries"

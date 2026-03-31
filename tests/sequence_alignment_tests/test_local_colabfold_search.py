"""tests/sequence_alignment_tests/test_local_colabfold_search.py

Tests for local ColabFold MSA search tool."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    CHIMERA_COLABFOLD_DB_LOCATION,
    ColabfoldSearchConfig,
    ColabfoldSearchInput,
    ColabfoldSearchQuery,
    run_colabfold_search,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

logger = logging.getLogger(__name__)

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

# ── Path to test database ──────────────────────────────────────────────────
_TEST_DB_DIR = Path(__file__).parent.parent / "dummy_data" / "mini_mmseqs_db"
_TEST_DB_SETUP_SCRIPT = Path(__file__).parent.parent / "dummy_data" / "create_mini_mmseqs_db.py"


# ============================================================================
# Input Validation Tests
# ============================================================================


# ── ColabfoldSearchQuery tests ─────────────────────────────────────────────

def test_colabfold_query_valid():
    query = ColabfoldSearchQuery(
        sequence=SAMPLE_PROTEIN_SEQ_1, sequence_id="test_seq"
    )
    assert query.sequence == SAMPLE_PROTEIN_SEQ_1
    assert query.sequence_id == "test_seq"


def test_colabfold_query_without_id():
    query = ColabfoldSearchQuery(sequence=SAMPLE_PROTEIN_SEQ_1)
    assert query.sequence == SAMPLE_PROTEIN_SEQ_1
    assert query.sequence_id is None


def test_colabfold_query_empty_sequence_fails():
    with pytest.raises(ValidationError, match="cannot be empty"):
        ColabfoldSearchQuery(sequence="")


def test_colabfold_query_whitespace_only_sequence_fails():
    with pytest.raises(ValidationError, match="cannot be empty"):
        ColabfoldSearchQuery(sequence="   ")


def test_colabfold_query_sequence_strips_whitespace():
    query = ColabfoldSearchQuery(sequence="  MVLS  ")
    assert query.sequence == "MVLS"


# ── ColabfoldSearchInput tests ─────────────────────────────────────────────

def test_colabfold_input_single_sequence_string():
    """Test input with single sequence string."""
    inputs = ColabfoldSearchInput(queries=SAMPLE_PROTEIN_SEQ_1)
    assert len(inputs.queries) == 1
    assert inputs.queries[0].sequence == SAMPLE_PROTEIN_SEQ_1
    assert inputs.queries[0].sequence_id is not None  # Auto-generated


def test_colabfold_input_list_of_sequence_strings():
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


def test_colabfold_input_single_tuple_format():
    """Test input with single tuple (sequence, id)."""
    inputs = ColabfoldSearchInput(queries=(SAMPLE_PROTEIN_SEQ_1, "protein_A"))
    assert len(inputs.queries) == 1
    assert inputs.queries[0].sequence == SAMPLE_PROTEIN_SEQ_1
    assert inputs.queries[0].sequence_id == "protein_A"


def test_colabfold_input_list_of_tuples():
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


def test_colabfold_input_list_of_query_objects():
    """Test input with explicit Query objects."""
    query1 = ColabfoldSearchQuery(sequence=SAMPLE_PROTEIN_SEQ_1, sequence_id="seq1")
    query2 = ColabfoldSearchQuery(sequence=SAMPLE_PROTEIN_SEQ_2, sequence_id="seq2")
    inputs = ColabfoldSearchInput(queries=[query1, query2])
    assert len(inputs.queries) == 2
    assert inputs.queries[0].sequence_id == "seq1"
    assert inputs.queries[1].sequence_id == "seq2"


def test_colabfold_input_single_query_object():
    """Test input with single Query object."""
    query = ColabfoldSearchQuery(sequence=SAMPLE_PROTEIN_SEQ_1, sequence_id="seq1")
    inputs = ColabfoldSearchInput(queries=query)
    assert len(inputs.queries) == 1
    assert inputs.queries[0].sequence_id == "seq1"


def test_colabfold_input_mixed_format_list():
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


def test_colabfold_input_empty_queries_fails():
    """Test that empty queries list fails."""
    with pytest.raises(ValidationError, match="At least one query"):
        ColabfoldSearchInput(queries=[])


def test_colabfold_input_duplicate_sequence_ids_fails():
    """Test that duplicate sequence IDs fail validation."""
    with pytest.raises(ValidationError, match="not unique"):
        ColabfoldSearchInput(
            queries=[
                (SAMPLE_PROTEIN_SEQ_1, "same_id"),
                (SAMPLE_PROTEIN_SEQ_2, "same_id"),
            ]
        )


def test_colabfold_input_auto_generated_ids_are_unique():
    """Test that auto-generated IDs are unique."""
    inputs = ColabfoldSearchInput(
        queries=[SAMPLE_PROTEIN_SEQ_1, SAMPLE_PROTEIN_SEQ_2]
    )
    ids = {q.sequence_id for q in inputs.queries}
    assert len(ids) == 2  # All unique


def test_colabfold_input_auto_generated_ids_deterministic():
    """Test that auto-generated IDs are deterministic (hash-based)."""
    inputs1 = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ_1])
    inputs2 = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ_1])
    assert inputs1.queries[0].sequence_id == inputs2.queries[0].sequence_id


def test_colabfold_input_list_like_interface():
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
# Integration Tests (Debug Database)
# ============================================================================


@pytest.fixture(scope="module")
def setup_mini_database():
    """Setup mini database if it doesn't exist."""
    db_sentinel = _TEST_DB_DIR / "uniref30_mini_db.dbtype"
    if not db_sentinel.exists():
        logger.info("Setting up mini MMseqs database...")
        logger.info("Running: %s", _TEST_DB_SETUP_SCRIPT)
        try:
            result = subprocess.run(
                [sys.executable, str(_TEST_DB_SETUP_SCRIPT)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            if result.stdout:
                logger.info("Output: %s", result.stdout)
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


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _TEST_DB_SETUP_SCRIPT.exists(),
    reason="Test database setup script not found.",
)
@pytest.mark.skip_ci
@pytest.mark.integration
@pytest.mark.include_in_env_report(category="sequence_alignment")
def test_finding_self_in_database(setup_mini_database, tmp_path):
    """Test end-to-end search with a single sequence against test database."""
    # Bacillus subtilis YunC protein
    # NOTE: This sequence is in the database. Colabfold should be able to find it
    test_sequence = "MVNLTPIMIEGQPFTAVTVKLPKTNFMAVANDHGYIMCGALDVALLNEKLKERGIVAGRAVGVRTIDQLLDAPLESVTYA"

    config = ColabfoldSearchConfig(
        search_mode="local",
        msa_db_dir=str(_TEST_DB_DIR),
        database_name="uniref30_mini_db",
        output_dir=str(tmp_path),
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
    msa_path = tmp_path / "msas" / "test_query.a3m"
    assert msa_path.exists()


@pytest.mark.skipif(
    not _TEST_DB_SETUP_SCRIPT.exists(),
    reason="Test database setup script not found.",
)
@pytest.mark.skip_ci
@pytest.mark.slow
@pytest.mark.integration
def test_finding_homologs(setup_mini_database, tmp_path):
    """Test end-to-end search with a sequence that has close homologs in the database."""
    srs = "MRAKKRFGQNFLIDQNIINKIVDSSEVENRNIIEIGPGKGALTKILVKKANKVLAYEIDQDMVNILNQQISSKNFVLINKDFLKEEFDKSQNYNIVANIPYYITSDIIFKIIENHQIFDQATLMVQKEVALRILAKQNDSEFSKLSLSVQFFFDVFLICDVSKNSFRPIPKVDSAVIKLVKKKNKDFSLWKEYFEFLKIAFSSRRKTLLNNLKYFFNEQKILKFFELKNYDPKVRAQNIKNEDFYALFLELR"

    config = ColabfoldSearchConfig(
        search_mode="local",
        msa_db_dir=str(_TEST_DB_DIR),
        database_name="uniref30_mini_db",
        output_dir=str(tmp_path),
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


@pytest.mark.skipif(
    not _TEST_DB_SETUP_SCRIPT.exists(),
    reason="Test database setup script not found.",
)
@pytest.mark.skip_ci
@pytest.mark.slow
@pytest.mark.integration
def test_multiple_sequences_search(setup_mini_database, tmp_path):
    """Test end-to-end search with multiple sequences."""
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
        msa_db_dir=str(_TEST_DB_DIR),
        database_name="uniref30_mini_db",
        output_dir=str(tmp_path),
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


@pytest.mark.skipif(
    not _TEST_DB_SETUP_SCRIPT.exists(),
    reason="Test database setup script not found.",
)
@pytest.mark.skip_ci
@pytest.mark.slow
@pytest.mark.integration
def test_with_sensitivity(setup_mini_database, tmp_path):
    """Test search with sensitivity parameter."""
    test_sequences = [
        (
            "MRAKKRFGQNFLIDQNIINKIVDSSEVENRNIIEIGPGKGALTKILVKKANKVLAYEIDQDMVNILNQQISSKNFVLINKDFLKEEFDKSQNYNIVANIPYYITSDIIFKIIENHQIFDQATLMVQKEVALRILAKQNDSEFSKLSLSVQFFFDVFLICDVSKNSFRPIPKVDSAVIKLVKKKNKDFSLWKEYFEFLKIAFSSRRKTLLNNLKYFFNEQKILKFFELKNYDPKVRAQNIKNEDFYALFLELR",
            "Ribosomal RNA small subunit methyltransferase A OS=Hydrogenobaculum sp",
        ),
    ]

    config = ColabfoldSearchConfig(
        search_mode="local",
        msa_db_dir=str(_TEST_DB_DIR),
        database_name="uniref30_mini_db",
        output_dir=str(tmp_path),
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


# ============================================================================
# Integration Tests (Full Database)
# ============================================================================


@pytest.mark.skipif(
    not Path(CHIMERA_COLABFOLD_DB_LOCATION).exists(),
    reason="Full database not found. Skipping",
)
@pytest.mark.skip_ci
@pytest.mark.only_chimera
@pytest.mark.integration
@pytest.mark.slow
def test_full_database_search(tmp_path):
    """Test end-to-end search against the full ColabFold database."""
    test_sequences = [
        (
            "MRAKKRFGQNFLIDQNIINKIVDSSEVENRNIIEIGPGKGALTKILVKKANKVLAYEIDQDMVNILNQQISSKNFVLINKDFLKEEFDKSQNYNIVANIPYYITSDIIFKIIENHQIFDQATLMVQKEVALRILAKQNDSEFSKLSLSVQFFFDVFLICDVSKNSFRPIPKVDSAVIKLVKKKNKDFSLWKEYFEFLKIAFSSRRKTLLNNLKYFFNEQKILKFFELKNYDPKVRAQNIKNEDFYALFLELR",
            "Ribosomal RNA small subunit methyltransferase A OS=Hydrogenobaculum sp",
        ),
    ]

    config = ColabfoldSearchConfig(
        search_mode="local",
        msa_db_dir=CHIMERA_COLABFOLD_DB_LOCATION,
        output_dir=str(tmp_path),
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

"""tests/sequence_alignment_tests/test_local_colabfold_search.py.

Tests for local ColabFold MSA search tool.
"""

import logging
import platform
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
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


def _gpu_search_available() -> bool:
    """Check if GPU-accelerated search is available (GPU present, Linux, padded DB exists)."""
    from proto_tools.utils.device import number_of_visible_gpus

    if platform.system() != "Linux" or number_of_visible_gpus() < 1:
        return False
    return (_TEST_DB_DIR / "uniref30_mini_db.idx_pad").exists()


# ============================================================================
# Input Validation Tests
# ============================================================================


# ── ColabfoldSearchQuery tests ─────────────────────────────────────────────


def test_colabfold_query_valid():
    query = ColabfoldSearchQuery(sequences=SAMPLE_PROTEIN_SEQ_1)
    assert query.sequences == [SAMPLE_PROTEIN_SEQ_1]  # bare str normalized to a one-element list
    assert not query.is_paired
    assert query.chain_count == 1


def test_colabfold_query_paired_group():
    query = ColabfoldSearchQuery(sequences=[SAMPLE_PROTEIN_SEQ_1, SAMPLE_PROTEIN_SEQ_2])
    assert query.is_paired
    assert query.chain_count == 2


def test_colabfold_query_empty_sequence_fails():
    with pytest.raises(ValidationError, match="cannot be empty"):
        ColabfoldSearchQuery(sequences="")


def test_colabfold_query_whitespace_only_sequence_fails():
    with pytest.raises(ValidationError, match="cannot be empty"):
        ColabfoldSearchQuery(sequences="   ")


def test_colabfold_query_sequence_strips_whitespace():
    query = ColabfoldSearchQuery(sequences="  MVLS  ")
    assert query.sequences == ["MVLS"]


# ── ColabfoldSearchInput tests ─────────────────────────────────────────────


def test_colabfold_input_single_sequence_string():
    """A bare sequence string becomes one unpaired query."""
    inputs = ColabfoldSearchInput(queries=SAMPLE_PROTEIN_SEQ_1)
    assert len(inputs.queries) == 1
    assert inputs.queries[0].sequences == [SAMPLE_PROTEIN_SEQ_1]
    assert not inputs.queries[0].is_paired


def test_colabfold_input_list_of_sequence_strings():
    """A list of strings becomes one unpaired query each."""
    inputs = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ_1, SAMPLE_PROTEIN_SEQ_2])
    assert len(inputs.queries) == 2
    assert inputs.queries[0].sequences == [SAMPLE_PROTEIN_SEQ_1]
    assert inputs.queries[1].sequences == [SAMPLE_PROTEIN_SEQ_2]
    assert not any(q.is_paired for q in inputs.queries)


def test_colabfold_input_nested_list_is_paired_group():
    """A nested list of sequences becomes one paired query."""
    inputs = ColabfoldSearchInput(queries=[[SAMPLE_PROTEIN_SEQ_1, SAMPLE_PROTEIN_SEQ_2]])
    assert len(inputs.queries) == 1
    assert inputs.queries[0].is_paired
    assert inputs.queries[0].sequences == [SAMPLE_PROTEIN_SEQ_1, SAMPLE_PROTEIN_SEQ_2]


def test_colabfold_input_mixed_unpaired_and_paired():
    """Unpaired strings and paired groups can be mixed in one input."""
    inputs = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ_1, [SAMPLE_PROTEIN_SEQ_2, SAMPLE_PROTEIN_SEQ_3]])
    assert len(inputs.queries) == 2
    assert not inputs.queries[0].is_paired
    assert inputs.queries[1].is_paired


def test_colabfold_input_list_of_query_objects():
    """Explicit Query objects pass through unchanged."""
    query1 = ColabfoldSearchQuery(sequences=SAMPLE_PROTEIN_SEQ_1)
    query2 = ColabfoldSearchQuery(sequences=[SAMPLE_PROTEIN_SEQ_2, SAMPLE_PROTEIN_SEQ_3])
    inputs = ColabfoldSearchInput(queries=[query1, query2])
    assert len(inputs.queries) == 2
    assert inputs.queries[0].sequences == [SAMPLE_PROTEIN_SEQ_1]
    assert inputs.queries[1].is_paired


def test_colabfold_input_single_query_object():
    """A single Query object is wrapped into a one-element list."""
    query = ColabfoldSearchQuery(sequences=SAMPLE_PROTEIN_SEQ_1)
    inputs = ColabfoldSearchInput(queries=query)
    assert len(inputs.queries) == 1
    assert inputs.queries[0].sequences == [SAMPLE_PROTEIN_SEQ_1]


def test_colabfold_input_empty_queries_fails():
    """An empty queries list is rejected."""
    with pytest.raises(ValidationError, match="At least one query"):
        ColabfoldSearchInput(queries=[])


def test_colabfold_input_list_like_interface():
    """Input supports len, indexing, and iteration over queries."""
    inputs = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ_1, SAMPLE_PROTEIN_SEQ_2])
    assert len(inputs) == 2
    assert inputs[0].sequences == [SAMPLE_PROTEIN_SEQ_1]
    assert inputs[1].sequences == [SAMPLE_PROTEIN_SEQ_2]
    sequences = [q.sequences for q in inputs]
    assert sequences == [[SAMPLE_PROTEIN_SEQ_1], [SAMPLE_PROTEIN_SEQ_2]]


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
                f"Failed to setup mini database:\nExit code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Database setup timed out after 10 minutes")

    # Create GPU-compatible padded database if it doesn't exist
    idx_pad = _TEST_DB_DIR / "uniref30_mini_db.idx_pad"
    if not idx_pad.exists():
        from proto_tools.utils.proto_home import get_proto_home

        mmseqs = get_proto_home() / "proto_tool_envs" / "colabfold_search_env" / "bin" / "mmseqs"
        if mmseqs.exists():
            db = _TEST_DB_DIR / "uniref30_mini_db"
            logger.info("Creating GPU-compatible padded database for mini test DB...")
            subprocess.run(
                [str(mmseqs), "makepaddedseqdb", str(db), str(idx_pad)],
                check=True,
                capture_output=True,
            )

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
        use_gpu=_gpu_search_available(),
        verbose=False,
    )

    inputs = ColabfoldSearchInput(queries=[test_sequence])

    # Execute the search
    result = run_colabfold_search(inputs, config)

    # Validate output and export functionality
    validate_output(result)

    # Verify results
    assert len(result) == len(inputs.queries), "Expected number of results to be equal to the number of queries"
    assert result.results[0].query_sequences == [test_sequence]

    # Verify MSA was created (single-chain query → one-element msas list)
    assert result.results[0].msas[0] is not None
    assert result.results[0].num_homologs_found == 1  # query should find itself

    # Verify we can iterate through the MSA
    sequences = list(result.results[0].msas[0])
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
        use_gpu=_gpu_search_available(),
        verbose=False,
    )

    inputs = ColabfoldSearchInput(queries=[srs])

    # Execute the search
    result = run_colabfold_search(inputs, config)

    # Validate output and export functionality
    validate_output(result)

    assert len(result) == len(inputs.queries), "Expected number of results to be equal to the number of queries"

    msa = result.results[0].msas[0]
    expected = 493 if _gpu_search_available() else 439
    assert len(msa) == expected


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
        use_gpu=_gpu_search_available(),
        verbose=False,
    )

    inputs = ColabfoldSearchInput(queries=test_sequences)

    # Execute the search
    result = run_colabfold_search(inputs, config)

    # Validate output and export functionality
    validate_output(result)

    # Verify results
    assert len(result.results) == len(inputs.queries), "Expected number of results to be equal to the number of queries"

    # First two MSAs should have a lot of homologs (single-chain queries → msas[0])
    gpu = _gpu_search_available()
    assert len(result.results[0].msas[0]) == (493 if gpu else 439)
    assert len(result.results[1].msas[0]) == (117 if gpu else 107)

    # Third MSA should have no homologs (returns None)
    assert result.results[2].msas[0] is None
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
        use_gpu=_gpu_search_available(),
        verbose=False,
    )

    inputs = ColabfoldSearchInput(queries=test_sequences)

    # Execute the search
    result = run_colabfold_search(inputs, config)

    # Validate output and export functionality
    validate_output(result)

    # Verify results
    assert len(result.results) == len(inputs.queries), "Expected number of results to be equal to the number of queries"

    expected = 493 if _gpu_search_available() else 425
    assert len(result.results[0].msa) == expected

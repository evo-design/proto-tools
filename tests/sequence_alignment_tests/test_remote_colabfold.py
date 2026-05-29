"""tests/sequence_alignment_tests/test_remote_colabfold.py.

Tests for remote ColabFold MSA search.
"""

from pathlib import Path

import pytest

from proto_tools.tools.sequence_alignment.colabfold_search import colabfold_search as cfs
from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
    ColabfoldSearchConfig,
    ColabfoldSearchInput,
    ColabfoldSearchQuery,
    _remote_search_paired,
    run_colabfold_search,
)
from proto_tools.tools.sequence_alignment.colabfold_search.standalone.remote_msa_search import _parse_pair_a3m
from tests.tool_infra_tests.test_export_functionality import validate_output

# ============================================================================
# Test Data
# ============================================================================

# Small protein sequence that should have homologs
SAMPLE_PROTEIN_SEQ = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"

# Canonical UniRef homolog header — must survive the query-row rewrite.
HOMOLOG_HEADER = "UniRef100_Q43227"
HOMOLOG_SEQ = "MVLSAKDKTNIKTAWGKIGGHAAEYGAEALERMFVVYPTT"

# Full human hemoglobin alpha / beta — deep MSAs; used for the live paired query.
HBA_HUMAN = "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
HBB_HUMAN = "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------


def test_remote_query_header_normalized_to_query_label(tmp_path, monkeypatch):
    """Remote A3M query row is normalized from ColabFold's ``101`` to ``query``; homologs untouched."""

    def fake_dispatch(toolkit, input_data, **kwargs):
        out_dir = Path(input_data["output_dir"]) / "msas"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Standalone names unpaired outputs by query index.
        a3m_path = out_dir / "0.a3m"
        a3m_path.write_text(f">101\n{SAMPLE_PROTEIN_SEQ}\n>{HOMOLOG_HEADER}\t101\t0.8\n{HOMOLOG_SEQ}\n")
        return {"msa_paths": {"0": str(a3m_path)}, "success": True, "num_successful": 1, "num_failed": 0}

    monkeypatch.setattr(cfs.ToolInstance, "dispatch", fake_dispatch)

    inputs = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ])
    config = ColabfoldSearchConfig(search_mode="remote", output_dir=str(tmp_path))

    result = run_colabfold_search(inputs, config)
    validate_output(result)

    assert result.results[0].query_sequences == [SAMPLE_PROTEIN_SEQ]
    msa = result.results[0].msas[0]
    assert msa is not None
    assert msa.sequence_ids[0] == "query"
    assert msa.sequence_ids[1].startswith(HOMOLOG_HEADER)

    lines = (tmp_path / "msas" / "0.a3m").read_text().splitlines()
    assert lines[0] == ">query"
    assert lines[2].startswith(f">{HOMOLOG_HEADER}")


def test_parse_pair_a3m_splits_row_aligned_blocks(tmp_path):
    r"""A \x00-separated pair.a3m splits into per-chain blocks; trailing empty block dropped."""
    chain_a = b">101\nMKTL\n>UniRef100_X\nMITL\n>UniRef100_Y\nMKAL\n"
    chain_b = b">102\nACDE\n>UniRef100_X\nACSE\n>UniRef100_Y\nACDQ\n"
    pair = tmp_path / "pair.a3m"
    pair.write_bytes(chain_a + b"\x00" + chain_b + b"\x00")  # trailing delimiter → empty block

    blocks = _parse_pair_a3m(pair, num_chains=2)
    assert blocks == [chain_a, chain_b]


def test_parse_pair_a3m_rejects_wrong_block_count(tmp_path):
    """A file with fewer blocks than chains is a structural error."""
    pair = tmp_path / "pair.a3m"
    pair.write_bytes(b">1\nMK\n>h\nMA\n")  # single block
    with pytest.raises(RuntimeError, match="expected 2 chain blocks"):
        _parse_pair_a3m(pair, num_chains=2)


def test_parse_pair_a3m_rejects_unequal_row_counts(tmp_path):
    """Chain blocks must be row-aligned (equal sequence counts)."""
    chain_a = b">1\nMK\n>h\nMA\n>h2\nMM\n"  # 3 rows
    chain_b = b">2\nAC\n>h\nAS\n"  # 2 rows
    pair = tmp_path / "pair.a3m"
    pair.write_bytes(chain_a + b"\x00" + chain_b)
    with pytest.raises(RuntimeError, match="unequal row counts"):
        _parse_pair_a3m(pair, num_chains=2)


def test_remote_search_paired_rejects_chain_path_count_mismatch(tmp_path, monkeypatch):
    """A standalone returning the wrong number of chain paths is a contract violation, not 'no homologs'."""

    def fake_dispatch(_name, _input_data, **_kwargs):
        # 2-chain query, but the standalone returns only one chain path.
        return {"success": True, "paired_msa_paths": {"0": [str(tmp_path / "only_one.a3m")]}}

    monkeypatch.setattr(cfs.ToolInstance, "dispatch", fake_dispatch)
    query = ColabfoldSearchQuery(sequences=[SAMPLE_PROTEIN_SEQ, SAMPLE_PROTEIN_SEQ])
    config = ColabfoldSearchConfig(search_mode="remote", output_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="expected 2"):
        _remote_search_paired(query, config)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skip_ci
@pytest.mark.integration
def test_remote_colabfold_search_paired_two_chain(tmp_path):
    """Live paired query: hemoglobin alpha+beta return row-aligned per-chain MSAs of equal depth."""
    inputs = ColabfoldSearchInput(queries=[[HBA_HUMAN, HBB_HUMAN]])
    config = ColabfoldSearchConfig(
        search_mode="remote",
        output_dir=str(tmp_path),
        use_metagenomic_db=False,
        verbose=True,
    )

    result = run_colabfold_search(inputs, config)

    assert len(result.results) == 1
    res = result.results[0]
    assert res.query_sequences == [HBA_HUMAN, HBB_HUMAN]
    assert len(res.msas) == 2 and all(m is not None for m in res.msas)
    # Paired output is row-aligned: both per-chain MSAs share the same depth.
    assert res.msas[0].num_sequences == res.msas[1].num_sequences > 1


@pytest.mark.skip_ci
@pytest.mark.integration
def test_remote_colabfold_search_no_metagenomic_db(tmp_path):
    """Test remote search with use_metagenomic_db=False."""
    inputs = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ])

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
    assert result.results[0].query_sequences == [SAMPLE_PROTEIN_SEQ]
    # Should find homologs for this sequence
    msa = result.results[0].msas[0]
    assert msa is not None, "MSA is None for the query sequence"
    assert msa.num_sequences > 100
    assert msa.alignment_length == 40


@pytest.mark.skip(reason="Metagenomic DB hit count varies between API updates; needs range-based assertion")
@pytest.mark.skip_ci
@pytest.mark.integration
def test_remote_colabfold_search_with_metagenomic_db(tmp_path):
    """Test remote search with use_metagenomic_db=True."""
    inputs = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ])

    config = ColabfoldSearchConfig(
        search_mode="remote",
        output_dir=str(tmp_path),
        use_metagenomic_db=True,  # Enable metagenomic database
        verbose=True,
    )

    result = run_colabfold_search(inputs, config)

    # Verify it completes successfully with metagenomic DB
    assert len(result.results) == 1
    msa = result.results[0].msas[0]
    assert msa.num_sequences > 100
    assert msa.alignment_length == 40


@pytest.mark.skip_ci
@pytest.mark.integration
def test_remote_colabfold_search_custom_output_dir(tmp_path):
    """Test remote search with custom output_dir."""
    custom_output = tmp_path / "my_custom_output"

    inputs = ColabfoldSearchInput(queries=[SAMPLE_PROTEIN_SEQ])

    config = ColabfoldSearchConfig(
        search_mode="remote",
        output_dir=str(custom_output),
        use_metagenomic_db=False,
        verbose=True,
    )

    result = run_colabfold_search(inputs, config)

    # Validate output and export functionality
    validate_output(result)

    # Verify custom output directory was created and used (unpaired output named by query index).
    assert custom_output.exists()
    assert (custom_output / "msas").exists()
    assert (custom_output / "msas" / "0.a3m").exists()

    # Verify result is valid
    assert len(result.results) == 1
    assert result.results[0].msas[0] is not None

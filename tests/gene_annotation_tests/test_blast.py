"""Tests for BLAST tools."""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from bio_programming_tools.tools.gene_annotation import (
    BlastSearchConfig,
    BlastSearchInput,
    BlastSearchOutput,
    CreateBlastDbConfig,
    CreateBlastDbInput,
    run_blast_search,
    run_create_blast_db,
)
from bio_programming_tools.tools.gene_annotation.blast.blast_search import (
    _blast_results_to_df,
)

# ── Test data ──────────────────────────────────────────────────────────────

_NUCL_DB_FASTA = """\
>seq1 test_nucleotide_1
ATGCGTAAACCCGGGTTTTTTAAACCCGGGTTTATGCGTAAACCCGGGTTTTTTAAACCCGGGTTTATGCGTAAACCCGGGT
>seq2 test_nucleotide_2
CCCCGGGGAAAATTTTCCCCGGGGAAAATTTTCCCCGGGGAAAATTTTCCCCGGGGAAAATTTTCCCCGGGGAAAATTTTCCC
"""

_PROT_DB_FASTA = """\
>prot1 hemoglobin_alpha
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALS
>prot2 lysozyme
MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNA
"""


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def nucl_blast_db(tmp_path):
    """Create a temporary nucleotide BLAST database."""
    fasta = tmp_path / "nucl.fasta"
    fasta.write_text(_NUCL_DB_FASTA)
    result = run_create_blast_db(
        CreateBlastDbInput(fasta=str(fasta)),
        CreateBlastDbConfig(dbtype="nucl"),
    )
    return result.db_path


@pytest.fixture
def prot_blast_db(tmp_path):
    """Create a temporary protein BLAST database."""
    fasta = tmp_path / "prot.fasta"
    fasta.write_text(_PROT_DB_FASTA)
    result = run_create_blast_db(
        CreateBlastDbInput(fasta=str(fasta)),
        CreateBlastDbConfig(dbtype="prot"),
    )
    return result.db_path


# ── Mock classes for XML parsing tests ─────────────────────────────────────


class _MockHSP:
    """Minimal mock of Bio.Blast.Record.HSP."""

    def __init__(
        self,
        query="ATGC--GTAA",
        sbjct="ATGCAAGTAA",
        match="||||  ||||",
        identities=8,
        gaps=2,
        align_length=10,
        query_start=1,
        query_end=10,
        sbjct_start=1,
        sbjct_end=10,
        expect=1e-5,
        bits=42.0,
    ):
        self.query, self.sbjct, self.match = query, sbjct, match
        self.identities, self.gaps = identities, gaps
        self.align_length = align_length
        self.query_start, self.query_end = query_start, query_end
        self.sbjct_start, self.sbjct_end = sbjct_start, sbjct_end
        self.expect, self.bits = expect, bits


class _MockAlignment:
    """Minimal mock of Bio.Blast.Record.Alignment."""

    def __init__(
        self, hit_id="gi|123|ref|NM_001.1|", accession="NM_001.1", hsps=None
    ):
        self.hit_id, self.accession = hit_id, accession
        self.hsps = hsps or []


class _MockRecord:
    """Minimal mock of Bio.Blast.Record.Blast."""

    def __init__(self, query="query_1 test query", alignments=None):
        self.query = query
        self.alignments = alignments or []


# ── Input validation ───────────────────────────────────────────────────────


def test_input_raw_sequence():
    assert BlastSearchInput(query="ATGCGTAAA").query_type == "sequence"


def test_input_protein_sequence():
    inp = BlastSearchInput(
        query="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"
    )
    assert inp.query_type == "sequence"


def test_input_fasta_path(tmp_path):
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">test\nATGCGTAAA\n")
    inp = BlastSearchInput(query=str(fasta))
    assert inp.query_type == "fasta_path"


def test_input_rejects_invalid_chars():
    with pytest.raises(ValidationError, match="unexpected characters"):
        BlastSearchInput(query="ATGCGT123!@#")


def test_input_rejects_empty():
    with pytest.raises(ValidationError, match="cannot be empty"):
        BlastSearchInput(query="")


def test_input_rejects_nonexistent_path():
    with pytest.raises(ValidationError, match="unexpected characters"):
        BlastSearchInput(query="/nonexistent/file.fasta")


# ── Config validation ─────────────────────────────────────────────────────


def test_config_defaults():
    config = BlastSearchConfig()
    assert config.search_mode == "online"
    assert config.program == "blastn"
    assert config.database == "nt"
    assert config.num_threads == 4
    assert config.evalue is None
    assert config.local_db is None


def test_config_local_requires_db():
    with pytest.raises(ValidationError, match="local_db is required"):
        BlastSearchConfig(search_mode="local")


def test_config_local_with_db():
    config = BlastSearchConfig(search_mode="local", local_db="/data/blast/nr")
    assert config.search_mode == "local"
    assert config.local_db == "/data/blast/nr"


def test_config_online_only_warns_in_local(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="bio_programming_tools"):
        BlastSearchConfig(
            search_mode="local",
            local_db="/data/blast/nr",
            hitlist_size=10,
            entrez_query="Homo sapiens[Organism]",
        )

    msgs = [r.message for r in caplog.records]
    assert any("hitlist_size" in m and "online-only" in m for m in msgs)
    assert any("entrez_query" in m and "online-only" in m for m in msgs)


def test_config_local_only_warns_in_online(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="bio_programming_tools"):
        BlastSearchConfig(search_mode="online", num_threads=8)

    msgs = [r.message for r in caplog.records]
    assert any("num_threads" in m and "local-only" in m for m in msgs)


# ── XML → DataFrame parsing ───────────────────────────────────────────────


def test_results_to_df_empty():
    df = _blast_results_to_df([])
    expected_cols = [
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 0


def test_results_to_df_field_values():
    hsp = _MockHSP(
        query="ATGC--GTAA",
        sbjct="ATGCAAGTAA",
        identities=8,
        gaps=2,
        align_length=10,
        sbjct_start=101,
        sbjct_end=110,
    )
    aln = _MockAlignment(accession="NM_001.1", hsps=[hsp])
    rec = _MockRecord(query="query_seq description", alignments=[aln])

    row = _blast_results_to_df([rec]).iloc[0]

    assert row["qseqid"] == "query_seq"
    assert row["sseqid"] == "NM_001.1"
    assert row["pident"] == pytest.approx(80.0)  # 8/10 * 100
    assert row["length"] == 10
    assert row["mismatch"] == 0  # 10 - 8 - 2
    assert row["gapopen"] == 1  # one gap run in query
    assert row["qstart"] == 1
    assert row["qend"] == 10
    assert row["sstart"] == 101
    assert row["send"] == 110
    assert row["evalue"] == pytest.approx(1e-5)
    assert row["bitscore"] == pytest.approx(42.0)


def test_results_to_df_multiple_gap_opens():
    # query: 1 gap run of 2; subject: 1 gap run of 1 → total = 2
    hsp = _MockHSP(
        query="ATG--CGTAA",
        sbjct="ATGAAC-TAA",
        identities=7,
        gaps=3,
        align_length=10,
    )
    aln = _MockAlignment(hsps=[hsp])
    row = _blast_results_to_df([_MockRecord(alignments=[aln])]).iloc[0]

    assert row["gapopen"] == 2
    assert row["mismatch"] == 0  # 10 - 7 - 3


def test_results_to_df_no_gaps():
    hsp = _MockHSP(
        query="ATGCGTAACC",
        sbjct="ATGCTTAACC",
        identities=9,
        gaps=0,
        align_length=10,
    )
    aln = _MockAlignment(hsps=[hsp])
    row = _blast_results_to_df([_MockRecord(alignments=[aln])]).iloc[0]

    assert row["gapopen"] == 0
    assert row["mismatch"] == 1  # 10 - 9 - 0
    assert row["pident"] == pytest.approx(90.0)


def test_results_to_df_none_tuple_handling():
    """Biopython sets gaps/identities to (None, None) when XML omits them."""
    hsp = _MockHSP(
        query="ATGCGTAA",
        sbjct="ATGCGTAA",
        identities=8,
        gaps=0,
        align_length=8,
    )
    hsp.gaps = (None, None)
    hsp.identities = (None, None)

    aln = _MockAlignment(hsps=[hsp])
    row = _blast_results_to_df([_MockRecord(alignments=[aln])]).iloc[0]

    assert row["pident"] == pytest.approx(0.0)  # identities → 0
    assert row["mismatch"] == 8  # 8 - 0 - 0


def test_results_to_df_accession_fallback():
    hsp = _MockHSP()
    aln = _MockAlignment(hit_id="gi|456|ref|XM_002.1|", hsps=[hsp])
    del aln.accession

    row = _blast_results_to_df([_MockRecord(alignments=[aln])]).iloc[0]
    assert row["sseqid"] == "gi|456|ref|XM_002.1|"


def test_results_to_df_multiple_records():
    hsp1 = _MockHSP(identities=8, gaps=0, align_length=8)
    hsp2 = _MockHSP(identities=4, gaps=0, align_length=4)
    rec1 = _MockRecord(
        query="qA desc",
        alignments=[_MockAlignment(accession="A", hsps=[hsp1, hsp2])],
    )
    rec2 = _MockRecord(
        query="qB desc",
        alignments=[_MockAlignment(accession="B", hsps=[hsp1])],
    )

    df = _blast_results_to_df([rec1, rec2])
    assert len(df) == 3  # 2 HSPs from rec1 + 1 from rec2
    assert list(df["qseqid"]) == ["qA", "qA", "qB"]


# ── Output ─────────────────────────────────────────────────────────────────


def test_output_export_empty_warns():
    output = BlastSearchOutput(results_df=None, num_hits=0)
    with tempfile.TemporaryDirectory() as tmpdir:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output._export_output(Path(tmpdir) / "results", "csv")
            assert len(w) == 1
            assert "No BLAST results" in str(w[0].message)


def test_output_export_invalid_format():
    output = BlastSearchOutput(
        results_df=pd.DataFrame({"x": [1]}), num_hits=1
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unsupported format"):
            output._export_output(Path(tmpdir) / "results", "xlsx")


# ── Create BLAST DB validation ─────────────────────────────────────────────


def test_create_db_input_validates_path(tmp_path):
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">s1\nATG\n")
    assert CreateBlastDbInput(fasta=str(fasta)).fasta == str(fasta)


def test_create_db_input_rejects_missing_file():
    with pytest.raises(ValueError, match="FASTA file not found"):
        CreateBlastDbInput(fasta="/nonexistent/file.fasta")


def test_create_db_config_defaults():
    config = CreateBlastDbConfig()
    assert config.dbtype == "nucl"
    assert config.title is None


def test_create_db_config_rejects_invalid_dbtype():
    with pytest.raises(ValidationError, match="Input should be"):
        CreateBlastDbConfig(dbtype="invalid")


# ── Registry ──────────────────────────────────────────────────────────────


def test_blast_tools_registered():
    from bio_programming_tools.tools.tool_registry import ToolRegistry

    tool_keys = {spec.key for spec in ToolRegistry.list_all()}
    assert "blast-search" in tool_keys
    assert "blast-create-db" in tool_keys


def test_blast_config_schema():
    from bio_programming_tools.tools.tool_registry import ToolRegistry

    schema = ToolRegistry.get_config_schema("blast-search")
    assert "properties" in schema
    for field in ("program", "search_mode", "database", "evalue"):
        assert field in schema["properties"]


# ── Integration: local BLAST ──────────────────────────────────────────────


@pytest.mark.integration
class TestLocalBlastn:
    """End-to-end local blastn tests against a temporary nucleotide database."""

    def test_exact_match(self, nucl_blast_db):
        """Exact subsequence from the database → 100% identity hit."""
        query = (
            "ATGCGTAAACCCGGGTTTTTTAAACCCGGGTTT"
            "ATGCGTAAACCCGGGTTTTTTAAACCC"
        )
        result = run_blast_search(
            BlastSearchInput(query=query),
            BlastSearchConfig(
                search_mode="local",
                program="blastn",
                local_db=nucl_blast_db,
                task="blastn",
            ),
        )
        assert result.num_hits >= 1
        assert result.results_df.iloc[0]["pident"] == pytest.approx(100.0)

    def test_no_hits(self, nucl_blast_db):
        """Unrelated sequence with strict evalue → 0 hits."""
        result = run_blast_search(
            BlastSearchInput(query="TATATATATATATATATATATATATATATAT"),
            BlastSearchConfig(
                search_mode="local",
                program="blastn",
                local_db=nucl_blast_db,
                evalue=1e-10,
                task="blastn",
            ),
        )
        assert result.num_hits == 0

    def test_fasta_file_query(self, nucl_blast_db, tmp_path):
        """Query from a FASTA file instead of raw sequence."""
        fasta = tmp_path / "query.fasta"
        fasta.write_text(
            ">query\nATGCGTAAACCCGGGTTTTTTAAACCCGGGTTT\n"
        )
        result = run_blast_search(
            BlastSearchInput(query=str(fasta)),
            BlastSearchConfig(
                search_mode="local",
                program="blastn",
                local_db=nucl_blast_db,
                task="blastn",
            ),
        )
        assert result.num_hits >= 1

    def test_output_structure(self, nucl_blast_db):
        """Verify output metadata and DataFrame columns."""
        query = (
            "ATGCGTAAACCCGGGTTTTTTAAACCCGGGTTT"
            "ATGCGTAAACCCGGGTTTTTTAAACCC"
        )
        result = run_blast_search(
            BlastSearchInput(query=query),
            BlastSearchConfig(
                search_mode="local",
                program="blastn",
                local_db=nucl_blast_db,
                task="blastn",
            ),
        )
        assert isinstance(result, BlastSearchOutput)
        assert result.execution_time > 0
        assert result.metadata["search_mode"] == "local"
        assert result.metadata["program"] == "blastn"
        expected_cols = [
            "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
            "qstart", "qend", "sstart", "send", "evalue", "bitscore",
        ]
        assert list(result.results_df.columns) == expected_cols


@pytest.mark.integration
class TestLocalBlastp:
    """End-to-end local blastp tests against a temporary protein database."""

    def test_exact_match(self, prot_blast_db):
        """Exact protein subsequence → 100% identity hit."""
        query = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        result = run_blast_search(
            BlastSearchInput(query=query),
            BlastSearchConfig(
                search_mode="local",
                program="blastp",
                local_db=prot_blast_db,
            ),
        )
        assert result.num_hits >= 1
        assert result.results_df.iloc[0]["pident"] == pytest.approx(100.0)

    def test_no_hits(self, prot_blast_db):
        """Unrelated protein sequence with strict evalue → 0 hits."""
        result = run_blast_search(
            BlastSearchInput(
                query="WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
            ),
            BlastSearchConfig(
                search_mode="local",
                program="blastp",
                local_db=prot_blast_db,
                evalue=1e-10,
            ),
        )
        assert result.num_hits == 0


# ── Integration: online BLAST ─────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.skip_ci
class TestOnlineBlast:
    """End-to-end online BLAST tests against NCBI servers."""

    def test_online_blastp(self):
        """Search a known hemoglobin fragment against nr → expect hits."""
        result = run_blast_search(
            BlastSearchInput(
                query="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"
            ),
            BlastSearchConfig(
                search_mode="online",
                program="blastp",
                database="swissprot",
                hitlist_size=5,
            ),
        )
        assert isinstance(result, BlastSearchOutput)
        assert result.num_hits >= 1
        assert result.results_df is not None
        assert result.metadata["search_mode"] == "online"
        assert result.metadata["program"] == "blastp"
        # Hemoglobin is highly conserved — top hit should be high identity
        assert result.results_df.iloc[0]["pident"] > 80.0

    def test_online_blastn(self):
        """Search a short nucleotide sequence against nt → verify structure."""
        # Human beta-globin exon 1 fragment
        query = (
            "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTG"
        )
        result = run_blast_search(
            BlastSearchInput(query=query),
            BlastSearchConfig(
                search_mode="online",
                program="blastn",
                database="nt",
                hitlist_size=5,
                entrez_query="Homo sapiens[Organism]",
            ),
        )
        assert isinstance(result, BlastSearchOutput)
        assert result.num_hits >= 1
        expected_cols = [
            "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
            "qstart", "qend", "sstart", "send", "evalue", "bitscore",
        ]
        assert list(result.results_df.columns) == expected_cols


# ── Integration: create DB & pipeline ─────────────────────────────────────


@pytest.mark.integration
def test_create_blast_db(tmp_path):
    """Create a nucleotide BLAST database and verify output."""
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">seq1\nATGCGTAAA\n>seq2\nCCCGGGTTT\n")

    result = run_create_blast_db(
        CreateBlastDbInput(fasta=str(fasta)),
        CreateBlastDbConfig(dbtype="nucl", title="Test DB"),
    )

    assert result.success is True
    assert Path(result.db_path).parent.resolve() == fasta.parent.resolve()
    assert result.execution_time > 0


@pytest.mark.integration
def test_full_pipeline_create_db_search_export(tmp_path):
    """Full pipeline: create database → search → export results."""
    # Create database
    fasta = tmp_path / "db.fasta"
    fasta.write_text(_NUCL_DB_FASTA)
    db_result = run_create_blast_db(
        CreateBlastDbInput(fasta=str(fasta)),
        CreateBlastDbConfig(dbtype="nucl"),
    )

    # Search
    query = (
        "ATGCGTAAACCCGGGTTTTTTAAACCCGGGTTT"
        "ATGCGTAAACCCGGGTTTTTTAAACCC"
    )
    search_result = run_blast_search(
        BlastSearchInput(query=query),
        BlastSearchConfig(
            search_mode="local",
            program="blastn",
            local_db=db_result.db_path,
            task="blastn",
        ),
    )
    assert search_result.num_hits >= 1

    # Export to CSV and verify round-trip
    search_result._export_output(tmp_path / "results", "csv")
    csv_path = tmp_path / "results.csv"
    assert csv_path.exists()

    loaded = pd.read_csv(csv_path)
    assert len(loaded) == search_result.num_hits
    assert list(loaded.columns) == list(search_result.results_df.columns)

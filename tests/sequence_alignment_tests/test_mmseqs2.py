"""Tests for the MMseqs2 toolkit's protein search, genome search, and clustering tools."""

import random
from pathlib import Path

import pandas as pd
import pytest

from proto_tools.tools.sequence_alignment.mmseqs2 import (
    Mmseqs2ClusteringConfig,
    Mmseqs2ClusteringInput,
    Mmseqs2ClusteringOutput,
    Mmseqs2ClusterResult,
    Mmseqs2Hit,
    Mmseqs2SearchGenomesConfig,
    Mmseqs2SearchGenomesInput,
    Mmseqs2SearchGenomesOutput,
    Mmseqs2SearchProteinsConfig,
    Mmseqs2SearchProteinsInput,
    Mmseqs2SearchProteinsOutput,
    Mmseqs2SequenceSearchResult,
    run_mmseqs2_clustering,
    run_mmseqs2_search_genomes,
    run_mmseqs2_search_proteins,
)
from proto_tools.tools.sequence_alignment.mmseqs2.search_proteins import (
    _build_sequence_search_results,
    _filter_top_hits,
    _parse_m8_output,
    _resolve_gpu_db_stem,
    _resolve_registered_mmseqs_db,
)
from tests.conftest import benchmark_twice, random_dna_sequences, random_protein_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_DATA_DIR = Path("tests/dummy_data")
PROTEIN_FASTA = TEST_DATA_DIR / "test_protein_sequences.faa"
M8_FILE = TEST_DATA_DIR / "test_mmseqs_results.m8"


# ── Schema properties ────────────────────────────────────────────────────


def test_sequence_search_result_properties():
    """Computed properties: num_hits, has_hits, top_hit."""
    hits = [
        Mmseqs2Hit(target_id="db_1", pident=95.0, evalue=1e-50),
        Mmseqs2Hit(target_id="db_2", pident=85.0, evalue=1e-30),
    ]
    result = Mmseqs2SequenceSearchResult(
        query_id="seq_0",
        query_sequence="MVLSPADKTN",
        hits=hits,
    )
    assert result.num_hits == 2
    assert result.has_hits is True
    assert result.top_hit.pident == 95.0


def test_sequence_search_result_no_hits():
    result = Mmseqs2SequenceSearchResult(
        query_id="seq_0",
        query_sequence="MVLSPADKTN",
        hits=[],
    )
    assert result.num_hits == 0
    assert result.has_hits is False
    assert result.top_hit is None


def test_search_proteins_output_iteration():
    results = [
        Mmseqs2SequenceSearchResult(query_id="seq_0", query_sequence="SEQ1", hits=[]),
        Mmseqs2SequenceSearchResult(query_id="seq_1", query_sequence="SEQ2", hits=[]),
    ]
    output = Mmseqs2SearchProteinsOutput(results=results, metadata={})

    assert len(output) == 2
    assert output[0].query_sequence == "SEQ1"
    assert output[1].query_sequence == "SEQ2"
    assert list(output) == results
    assert output.total_hits == 0


def test_clustering_output_representative_indices():
    results = [
        Mmseqs2ClusterResult(sequence_id="seq_0", input_sequence="S1", cluster_id="seq_0", is_representative=True),
        Mmseqs2ClusterResult(sequence_id="seq_1", input_sequence="S2", cluster_id="seq_0", is_representative=False),
        Mmseqs2ClusterResult(sequence_id="seq_2", input_sequence="S3", cluster_id="seq_2", is_representative=True),
    ]
    output = Mmseqs2ClusteringOutput(results=results, metadata={})

    assert output.representative_indices == [0, 2]
    assert output.num_clusters == 2


# ── Helper functions ─────────────────────────────────────────────────────


def test_parse_m8_valid_string():
    if not M8_FILE.exists():
        pytest.skip("Test data file not found")

    raw_output = M8_FILE.read_text()
    df = _parse_m8_output(raw_output)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert list(df.columns) == ["query", "target", "pident", "evalue"]
    assert "protein_seq_1" in df["query"].values
    assert "protein_seq_2" in df["query"].values


def test_parse_m8_empty_string():
    df = _parse_m8_output("")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["query", "target", "pident", "evalue"]


def test_filter_top_hits_by_highest_pident():
    data = {
        "query": ["q1", "q1", "q2", "q2", "q3"],
        "target": ["t1", "t2", "t3", "t4", "t5"],
        "evalue": [1e-10, 1e-20, 1e-5, 1e-5, 1e-100],
        "pident": [90.0, 80.0, 95.0, 98.0, 99.0],
    }
    df = pd.DataFrame(data)
    filtered_df = _filter_top_hits(df)

    assert len(filtered_df) == 3
    assert filtered_df[filtered_df["query"] == "q1"].pident.iloc[0] == 90.0
    assert filtered_df[filtered_df["query"] == "q2"].pident.iloc[0] == 98.0
    assert filtered_df[filtered_df["query"] == "q3"].pident.iloc[0] == 99.0


def test_filter_top_hits_empty_dataframe():
    df = pd.DataFrame(columns=["query", "target", "pident", "evalue"])
    filtered_df = _filter_top_hits(df)
    assert isinstance(filtered_df, pd.DataFrame)
    assert len(filtered_df) == 0


def test_filter_top_hits_ties_in_pident():
    data = {
        "query": ["q1", "q1", "q1"],
        "target": ["t1", "t2", "t3"],
        "pident": [90.0, 90.0, 85.0],
        "evalue": [1e-10, 1e-15, 1e-20],
    }
    df = pd.DataFrame(data)
    filtered_df = _filter_top_hits(df)
    # Should keep one of the 90.0% hits (first by pandas default)
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["pident"] == 90.0


def test_build_results_with_hits():
    sequences = ["MVLSPADKTN", "MKLLVVAAAA"]
    sequence_ids = ["seq_0", "seq_1"]
    df = pd.DataFrame(
        {
            "query": ["seq_0", "seq_0", "seq_1"],
            "target": ["db_1", "db_2", "db_3"],
            "pident": [95.0, 85.0, 90.0],
            "evalue": [1e-50, 1e-30, 1e-40],
        }
    )

    results = _build_sequence_search_results(sequences, sequence_ids, df)

    assert len(results) == 2
    assert results[0].query_id == "seq_0"
    assert results[0].query_sequence == "MVLSPADKTN"
    assert results[0].num_hits == 2
    assert results[0].top_hit.pident == 95.0
    assert results[1].query_id == "seq_1"
    assert results[1].query_sequence == "MKLLVVAAAA"
    assert results[1].num_hits == 1


def test_build_results_no_hits():
    sequences = ["MVLSPADKTN", "MKLLVVAAAA"]
    sequence_ids = ["seq_0", "seq_1"]
    df = pd.DataFrame(columns=["query", "target", "pident", "evalue"])

    results = _build_sequence_search_results(sequences, sequence_ids, df)

    assert len(results) == 2
    assert all(r.num_hits == 0 for r in results)
    assert all(r.top_hit is None for r in results)


def test_build_results_partial_hits():
    """Only some sequences have hits; missing ones get empty results."""
    sequences = ["SEQ1", "SEQ2", "SEQ3"]
    sequence_ids = ["seq_0", "seq_1", "seq_2"]
    df = pd.DataFrame(
        {
            "query": ["seq_0", "seq_2"],
            "target": ["db_1", "db_2"],
            "pident": [95.0, 90.0],
            "evalue": [1e-50, 1e-40],
        }
    )

    results = _build_sequence_search_results(sequences, sequence_ids, df)

    assert len(results) == 3
    assert results[0].num_hits == 1
    assert results[1].num_hits == 0
    assert results[2].num_hits == 1


def test_build_results_with_custom_ids():
    sequences = ["MVLSPADKTN", "MKLLVVAAAA"]
    sequence_ids = ["protein_a", "protein_b"]
    df = pd.DataFrame(
        {
            "query": ["protein_a", "protein_b"],
            "target": ["db_1", "db_2"],
            "pident": [95.0, 90.0],
            "evalue": [1e-50, 1e-40],
        }
    )

    results = _build_sequence_search_results(sequences, sequence_ids, df)

    assert len(results) == 2
    assert results[0].query_id == "protein_a"
    assert results[1].query_id == "protein_b"


# ── GPU opt-in for mmseqs2-search-proteins ───────────────────────────────


def test_resolve_gpu_db_stem_finds_sibling(tmp_path):
    """``<db>.idx_pad`` sibling (proto-tools convention) is the canonical happy path."""
    db = tmp_path / "uniref30_2302_db"
    db.write_text("")
    (tmp_path / "uniref30_2302_db.dbtype").write_text("")
    padded = tmp_path / "uniref30_2302_db.idx_pad"
    padded.write_text("")
    (tmp_path / "uniref30_2302_db.idx_pad.dbtype").write_text("")
    assert _resolve_gpu_db_stem(str(db)) == str(padded)


def test_resolve_gpu_db_stem_accepts_padded_path_directly(tmp_path):
    """User passing the padded stem directly is recognized as GPU-ready (Case 1)."""
    padded = tmp_path / "foo.idx_pad"
    padded.write_text("")
    (tmp_path / "foo.idx_pad.dbtype").write_text("")
    assert _resolve_gpu_db_stem(str(padded)) == str(padded)


def test_resolve_gpu_db_stem_rejects_padded_data_without_dbtype(tmp_path):
    """A bare ``.idx_pad`` data file with no ``.dbtype`` companion is rejected."""
    db = tmp_path / "broken_db"
    db.write_text("")
    (tmp_path / "broken_db.idx_pad").write_text("")
    assert _resolve_gpu_db_stem(str(db)) is None


def test_run_protein_search_use_gpu_without_padded_db_fails(tmp_path):
    """End-to-end: ``use_gpu=True`` with no padded DB raises with remediation message."""
    fasta = tmp_path / "tiny.faa"
    fasta.write_text(">seq_0\nMKTL\n")
    with pytest.raises(Exception, match=r"(?s)GPU-padded MMseqs2 DB.*makepaddedseqdb"):
        run_mmseqs2_search_proteins(
            Mmseqs2SearchProteinsInput(query_sequences=["MKTL"]),
            Mmseqs2SearchProteinsConfig(mmseqs_db=str(fasta), use_gpu=True),
        )


def test_search_proteins_gpus_per_instance_reflects_use_gpu():
    """Per-call GPU need: 1 when ``use_gpu``, 0 otherwise."""
    assert Mmseqs2SearchProteinsConfig().gpus_per_instance == 0
    assert Mmseqs2SearchProteinsConfig(use_gpu=True).gpus_per_instance == 1


def test_standalone_dispatch_rejects_unknown_operation():
    """Unified ``dispatch()`` validates ``operation`` and lists valid choices on failure."""
    from proto_tools.tools.sequence_alignment.mmseqs2.standalone import run as standalone

    with pytest.raises(ValueError, match=r"unknown operation 'bogus'"):
        standalone.dispatch({"operation": "bogus"})


def test_genome_search_thread_fallback_retries_on_world_size_error(monkeypatch, tmp_path):
    """A split-index 'World Size > dbSize' abort retries once with --threads clamped to the reported dbSize."""
    from proto_tools.tools.sequence_alignment.mmseqs2.standalone import run as standalone

    calls: list[list[str]] = []

    def fake_run_cmd(cmd, description):
        calls.append(cmd)
        if len(calls) == 1:
            raise RuntimeError(
                "mmseqs2: mmseqs search failed (exit 1): World Size: 20 dbSize: 10 | World Size: 20 dbSize: 10"
            )
        return  # retry succeeds

    monkeypatch.setattr(standalone, "_run_cmd", fake_run_cmd)
    mmseqs_tmp = str(tmp_path / "mmseqs_tmp")
    res_dir = str(tmp_path / "results")

    def search_cmd_for(n: int) -> list[str]:
        return ["mmseqs", "search", "q", "t", res_dir, mmseqs_tmp, "--threads", str(n)]

    standalone._run_search_with_thread_fallback(search_cmd_for, 20, mmseqs_tmp, res_dir)

    assert len(calls) == 2  # one failed attempt, one successful retry
    assert calls[0][calls[0].index("--threads") + 1] == "20"
    assert calls[1][calls[1].index("--threads") + 1] == "10"  # clamped to dbSize


def test_genome_search_thread_fallback_handles_pipe_joined_error(monkeypatch, tmp_path):
    """The fallback still fires if World Size/dbSize land on separate stderr lines (joined by ' | ')."""
    from proto_tools.tools.sequence_alignment.mmseqs2.standalone import run as standalone

    calls: list[list[str]] = []

    def fake_run_cmd(cmd, description):
        calls.append(cmd)
        if len(calls) == 1:
            raise RuntimeError("mmseqs2: mmseqs search failed (exit 1): World Size: 20 | dbSize: 10")
        return

    monkeypatch.setattr(standalone, "_run_cmd", fake_run_cmd)
    tmp, res = str(tmp_path / "tmp"), str(tmp_path / "res")
    standalone._run_search_with_thread_fallback(
        lambda n: ["mmseqs", "search", "q", "t", res, tmp, "--threads", str(n)], 20, tmp, res
    )

    assert len(calls) == 2
    assert calls[1][calls[1].index("--threads") + 1] == "10"


def test_genome_search_thread_fallback_floors_threads_at_one(monkeypatch, tmp_path):
    """A degenerate dbSize of 0 retries with --threads 1, never --threads 0 (which mmseqs rejects)."""
    from proto_tools.tools.sequence_alignment.mmseqs2.standalone import run as standalone

    calls: list[list[str]] = []

    def fake_run_cmd(cmd, description):
        calls.append(cmd)
        if len(calls) == 1:
            raise RuntimeError("mmseqs2: mmseqs search failed (exit 1): World Size: 20 dbSize: 0")
        return

    monkeypatch.setattr(standalone, "_run_cmd", fake_run_cmd)
    tmp, res = str(tmp_path / "tmp"), str(tmp_path / "res")
    standalone._run_search_with_thread_fallback(
        lambda n: ["mmseqs", "search", "q", "t", res, tmp, "--threads", str(n)], 20, tmp, res
    )

    assert len(calls) == 2
    assert calls[1][calls[1].index("--threads") + 1] == "1"


def test_genome_search_thread_fallback_reraises_unrelated_error(monkeypatch, tmp_path):
    """A failure that isn't 'World Size > dbSize' propagates without a retry."""
    from proto_tools.tools.sequence_alignment.mmseqs2.standalone import run as standalone

    calls: list[list[str]] = []

    def fake_run_cmd(cmd, description):
        calls.append(cmd)
        raise RuntimeError("mmseqs2: mmseqs search failed (exit 1): some other error")

    monkeypatch.setattr(standalone, "_run_cmd", fake_run_cmd)

    with pytest.raises(RuntimeError, match="some other error"):
        standalone._run_search_with_thread_fallback(
            lambda n: ["mmseqs", "--threads", str(n)], 20, str(tmp_path / "t"), str(tmp_path / "r")
        )

    assert len(calls) == 1  # no retry


# ── M8 workflow (pure Python, local test data) ───────────────────────────


def test_workflow_with_real_test_data():
    """End-to-end parse → filter pipeline on local M8 file."""
    if not M8_FILE.exists() or not PROTEIN_FASTA.exists():
        pytest.skip("Test data files not found")

    raw_output = M8_FILE.read_text()
    df = _parse_m8_output(raw_output)
    filtered_df = _filter_top_hits(df)

    assert len(filtered_df) <= len(df)
    query_counts = filtered_df.groupby("query").size()
    assert all(count == 1 for count in query_counts)


def test_column_names_consistency():
    """Column names match MMseqs2 M8 format."""
    if not M8_FILE.exists():
        pytest.skip("Test data file not found")

    raw_output = M8_FILE.read_text()
    df = _parse_m8_output(raw_output)
    assert list(df.columns) == ["query", "target", "pident", "evalue"]


# ── Input validation ─────────────────────────────────────────────────────


def test_search_proteins_input_valid():
    """Sanity: a well-formed Input model accepts its required fields."""
    inputs = Mmseqs2SearchProteinsInput(query_sequences=["MVLSPADKTN", "MKLLVVAAAA"])
    assert len(inputs.query_sequences) == 2


@pytest.mark.parametrize(
    "input_cls, kwargs, msg",
    [
        (
            Mmseqs2SearchProteinsInput,
            {"query_sequences": []},
            "query_sequences list cannot be empty",
        ),
        (
            Mmseqs2SearchProteinsInput,
            {"query_sequences": "single"},
            "query_sequences must be a list",
        ),
        (
            Mmseqs2SearchGenomesInput,
            {"query_genomes": []},
            "query_genomes list cannot be empty",
        ),
        (Mmseqs2ClusteringInput, {"input_sequences": []}, "input_sequences list cannot be empty"),
    ],
)
def test_input_field_validator_rejects_bad_value(input_cls, kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        input_cls(**kwargs)


@pytest.mark.parametrize(
    "input_cls, base_kwargs, new_kwargs, cleared_field",
    [
        (Mmseqs2ClusteringInput, {}, {"mmseqs_db": "/p"}, "input_sequences"),
    ],
)
def test_new_xor_modality_accepted(input_cls, base_kwargs, new_kwargs, cleared_field):
    """Each tool's new XOR alternative (inline list or DB path) parses; the sibling field defaults to None."""
    inputs = input_cls(**base_kwargs, **new_kwargs)
    assert getattr(inputs, cleared_field) is None


@pytest.mark.parametrize(
    "input_cls, base_kwargs, both_kwargs, msg",
    [
        (
            Mmseqs2ClusteringInput,
            {},
            {"input_sequences": ["MKTL"], "mmseqs_db": "/p"},
            r"exactly one of `input_sequences` or `mmseqs_db`",
        ),
    ],
)
@pytest.mark.parametrize("violation", ["both", "neither"], ids=lambda v: v)
def test_xor_violation_raises(input_cls, base_kwargs, both_kwargs, msg, violation):
    """For every Input-level XOR-grouped tool, both `both fields set` and `neither set` raise a clear ValueError."""
    extra = both_kwargs if violation == "both" else {}
    with pytest.raises(ValueError, match=msg):
        input_cls(**base_kwargs, **extra)


# Proteins-specific Config-level XOR (target moved to Config in this refactor).
def test_proteins_config_target_xor_both_raises():
    """`mmseqs_db` and `target_sequences` are mutually exclusive on Mmseqs2SearchProteinsConfig."""
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        Mmseqs2SearchProteinsConfig(mmseqs_db="/p", target_sequences=["MKTL"])


def test_proteins_config_neither_target_accepted_at_construction():
    """`Mmseqs2SearchProteinsConfig()` constructs with both target fields unset; dispatch raises."""
    config = Mmseqs2SearchProteinsConfig()
    assert config.mmseqs_db is None
    assert config.target_sequences is None
    with pytest.raises(ValueError, match=r"exactly one of `mmseqs_db` or `target_sequences`"):
        run_mmseqs2_search_proteins(Mmseqs2SearchProteinsInput(query_sequences=["MKTL"]), config)


def test_proteins_config_rejects_empty_target_sequences():
    """`target_sequences=[]` on Config is rejected by the field validator."""
    with pytest.raises(ValueError, match=r"target_sequences list cannot be empty"):
        Mmseqs2SearchProteinsConfig(target_sequences=[])


# Genomes-specific Config-level XOR (target moved to Config in this refactor).
def test_genomes_config_target_xor_both_raises():
    """`target_genomes` and `target_db` are mutually exclusive on Mmseqs2SearchGenomesConfig."""
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        Mmseqs2SearchGenomesConfig(target_genomes=["ATCG"], target_db="/p")


def test_genomes_config_neither_target_accepted_at_construction():
    """`Mmseqs2SearchGenomesConfig()` constructs with both target fields unset; dispatch raises."""
    config = Mmseqs2SearchGenomesConfig()
    assert config.target_genomes is None
    assert config.target_db is None
    with pytest.raises(ValueError, match=r"exactly one of `target_genomes` or `target_db`"):
        run_mmseqs2_search_genomes(Mmseqs2SearchGenomesInput(query_genomes=["ATCG"]), config)


def test_genomes_config_rejects_empty_target_genomes():
    """`target_genomes=[]` on Config is rejected by the field validator."""
    with pytest.raises(ValueError, match=r"target_genomes list cannot be empty"):
        Mmseqs2SearchGenomesConfig(target_genomes=[])


# ── Cache-key separation (Config-moved targets now participate in the per-item key) ──


def test_proteins_per_item_cache_key_distinguishes_different_mmseqs_db():
    """Same query + different `mmseqs_db` on Config → distinct per-item cache keys.

    Regression test: before the Input→Config migration, `mmseqs_db` lived on Input as a
    non-iterable sibling field that bypassed the per-item key, so a cached result for
    one DB would silently be returned for queries against a different DB.
    """
    from proto_tools.utils.tool_cache import _generate_cache_key

    query = "MVLSPADKTN"
    cfg_a = Mmseqs2SearchProteinsConfig(mmseqs_db="/dbs/uniref30")
    cfg_b = Mmseqs2SearchProteinsConfig(mmseqs_db="/dbs/uniref90")
    key_a = _generate_cache_key("mmseqs2-search-proteins", input_item=query, config=cfg_a)
    key_b = _generate_cache_key("mmseqs2-search-proteins", input_item=query, config=cfg_b)
    assert key_a != key_b


def test_proteins_per_item_cache_key_distinguishes_different_target_sequences():
    """Same query + different inline `target_sequences` on Config → distinct per-item cache keys."""
    from proto_tools.utils.tool_cache import _generate_cache_key

    query = "MVLSPADKTN"
    cfg_a = Mmseqs2SearchProteinsConfig(target_sequences=["AAAA"])
    cfg_b = Mmseqs2SearchProteinsConfig(target_sequences=["BBBB"])
    key_a = _generate_cache_key("mmseqs2-search-proteins", input_item=query, config=cfg_a)
    key_b = _generate_cache_key("mmseqs2-search-proteins", input_item=query, config=cfg_b)
    assert key_a != key_b


def test_genomes_per_item_cache_key_distinguishes_different_target_db():
    """Same query + different `target_db` on Config → distinct per-item cache keys."""
    from proto_tools.utils.tool_cache import _generate_cache_key

    query = "ATCG"
    cfg_a = Mmseqs2SearchGenomesConfig(target_db="/dbs/genomes_a")
    cfg_b = Mmseqs2SearchGenomesConfig(target_db="/dbs/genomes_b")
    key_a = _generate_cache_key("mmseqs2-search-genomes", input_item=query, config=cfg_a)
    key_b = _generate_cache_key("mmseqs2-search-genomes", input_item=query, config=cfg_b)
    assert key_a != key_b


def test_resolve_to_mmseqs_db_classifies_inputs(tmp_path):
    """Atoms detect DB stems / FASTA; resolver raises FileNotFoundError on missing path and RuntimeError on unknown format."""
    from proto_tools.tools.sequence_alignment.mmseqs2.standalone.run import (
        _is_fasta,
        _is_mmseqs_db_stem,
        _resolve_to_mmseqs_db,
    )

    db_stem = tmp_path / "mydb"
    db_stem.write_text("")
    (tmp_path / "mydb.dbtype").write_text("")
    assert _is_mmseqs_db_stem(str(db_stem))

    fasta = tmp_path / "seq.fa"
    fasta.write_text(">seq_0\nMKTL\n")
    assert _is_fasta(str(fasta))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_to_mmseqs_db("mmseqs", str(tmp_path / "missing"), temp_db_path=str(tmp_path / "out"), label="test")

    plain = tmp_path / "plain.txt"
    plain.write_text("not a fasta or db\n")
    with pytest.raises(RuntimeError, match="neither a FASTA file nor an MMseqs2 DB stem"):
        _resolve_to_mmseqs_db("mmseqs", str(plain), temp_db_path=str(tmp_path / "out"), label="test")


def test_search_proteins_resolves_registered_dataset_names(monkeypatch, tmp_path):
    cache_dir = tmp_path / "pdb_seqres_2022_09_28"
    cache_dir.mkdir()
    stem = cache_dir / "pdb_seqres_padded"
    stem.write_text("")
    Path(f"{stem}.dbtype").write_text("")
    monkeypatch.setattr(
        "proto_tools.tools.sequence_alignment.mmseqs2.search_proteins.get_dataset_dir",
        lambda _name: cache_dir,
    )

    resolved, from_registry = _resolve_registered_mmseqs_db("pdb-seqres-2022-09-28", use_gpu=False)

    assert from_registry is True
    assert resolved == str(stem)


def test_search_proteins_registered_dataset_missing_gives_setup_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "proto_tools.tools.sequence_alignment.mmseqs2.search_proteins.get_dataset_dir",
        lambda _name: tmp_path / "missing",
    )

    with pytest.raises(FileNotFoundError, match="setup_databases pdb-seqres-2022-09-28"):
        _resolve_registered_mmseqs_db("pdb-seqres-2022-09-28", use_gpu=False)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_mmseqs_search_proteins_execution(tmp_path):
    db_file = tmp_path / "database.faa"
    db_file.write_text(">db1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT\n")

    inputs = Mmseqs2SearchProteinsInput(
        query_sequences=[
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
        ],
    )
    config = Mmseqs2SearchProteinsConfig(mmseqs_db=str(db_file), threads=2)
    result = run_mmseqs2_search_proteins(inputs, config)

    validate_output(result)

    assert isinstance(result, Mmseqs2SearchProteinsOutput)
    assert len(result) == 2
    assert result[0].num_hits >= 1
    assert result[1].num_hits >= 1


@pytest.mark.integration
def test_mmseqs_search_proteins_no_hits(tmp_path):
    db_file = tmp_path / "database.faa"
    db_file.write_text(">db1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")

    inputs = Mmseqs2SearchProteinsInput(query_sequences=["WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"])
    config = Mmseqs2SearchProteinsConfig(mmseqs_db=str(db_file), threads=2)
    result = run_mmseqs2_search_proteins(inputs, config)

    assert isinstance(result, Mmseqs2SearchProteinsOutput)
    assert result.success is True
    assert len(result) == 1
    assert result[0].num_hits == 0


@pytest.mark.integration
def test_mmseqs_clustering_execution():
    sequences = [
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",  # Identical to first
        "MKLLVVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # Different
    ]

    inputs = Mmseqs2ClusteringInput(input_sequences=sequences)
    config = Mmseqs2ClusteringConfig(min_seq_id=0.95)
    result = run_mmseqs2_clustering(inputs, config)

    validate_output(result)

    assert isinstance(result, Mmseqs2ClusteringOutput)
    assert len(result) == 3
    assert result.num_clusters == 2

    assert result[0].cluster_id == result[1].cluster_id
    assert result[2].cluster_id != result[0].cluster_id


@pytest.mark.integration
def test_mmseqs_clustering_all_identical():
    sequences = [
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
    ]

    inputs = Mmseqs2ClusteringInput(input_sequences=sequences)
    config = Mmseqs2ClusteringConfig(min_seq_id=0.95)
    result = run_mmseqs2_clustering(inputs, config)

    assert result.num_clusters == 1
    assert all(r.cluster_id == result[0].cluster_id for r in result)
    assert len(result.representative_indices) == 1


@pytest.mark.skip_ci
@pytest.mark.integration
def test_mmseqs_search_genomes_execution():
    query_seqs = [
        "ATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTCATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTC",
        "ATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCC",
    ]
    target_seqs = [
        "ATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTCATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTC",
        "ATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCC",
    ]

    inputs = Mmseqs2SearchGenomesInput(query_genomes=query_seqs)
    config = Mmseqs2SearchGenomesConfig(target_genomes=target_seqs)
    result = run_mmseqs2_search_genomes(inputs, config)

    validate_output(result)

    assert isinstance(result, Mmseqs2SearchGenomesOutput)
    assert len(result) == 2


@pytest.mark.integration
def test_search_proteins_inline_target_sequences_execution():
    """End-to-end: target_sequences (inline) routed via easy-search yields hits."""
    inputs = Mmseqs2SearchProteinsInput(query_sequences=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"])
    config = Mmseqs2SearchProteinsConfig(
        target_sequences=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT", "AAAAAAAAAAAAAAAAAAAA"],
        threads=2,
    )
    result = run_mmseqs2_search_proteins(inputs, config)
    assert result.success and result[0].num_hits >= 1


@pytest.mark.integration
def test_clustering_mmseqs_db_fasta_path_execution(tmp_path):
    """End-to-end: mmseqs_db pointing at a FASTA file → sniff routes to createdb → cluster."""
    fasta = tmp_path / "input.faa"
    fasta.write_text(
        ">a\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT\n"
        ">b\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT\n"
        ">c\nMKLLVVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
    )
    result = run_mmseqs2_clustering(
        Mmseqs2ClusteringInput(mmseqs_db=str(fasta)),
        Mmseqs2ClusteringConfig(min_seq_id=0.95),
    )
    assert result.num_clusters == 2 and all(r.input_sequence is None for r in result)


@pytest.mark.skip_ci
@pytest.mark.integration
def test_search_genomes_target_db_fasta_path_execution(tmp_path):
    """End-to-end: target_db pointing at a FASTA file → sniff routes to createdb → search."""
    target = tmp_path / "target.fna"
    target.write_text(">t1\nATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCC\n")
    result = run_mmseqs2_search_genomes(
        Mmseqs2SearchGenomesInput(query_genomes=["ATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCC"]),
        Mmseqs2SearchGenomesConfig(target_db=str(target)),
    )
    assert result.success and len(result) == 1


@pytest.mark.integration
def test_search_proteins_threads_zero_runs_successfully(tmp_path):
    """Behavior: ``threads=0`` (auto-detect) successfully runs end-to-end.

    Catches the regression where mmseqs rejected ``--threads 0`` directly and
    the standalone needs to omit the flag entirely to trigger auto-detection.
    """
    db_file = tmp_path / "database.faa"
    db_file.write_text(">db1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT\n")

    inputs = Mmseqs2SearchProteinsInput(query_sequences=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"])
    # Default threads=0 — verify the standalone correctly omits the flag
    # rather than passing `--threads 0` (which mmseqs rejects).
    result = run_mmseqs2_search_proteins(inputs, Mmseqs2SearchProteinsConfig(mmseqs_db=str(db_file)))
    assert result.success is True
    assert result[0].num_hits >= 1


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def _clustered_protein_sequences(
    n_families: int, per_family: int, length: int, mutation_rate: float, seed: int
) -> list[str]:
    """``n_families`` families of ``per_family`` ~mutation_rate-substituted variants each, so they actually cluster."""
    rng = random.Random(seed)
    bases = random_protein_sequences(n_families, length, seed)
    noise = random_protein_sequences(n_families * per_family, length, seed + 1)
    sequences = []
    for family, base in enumerate(bases):
        for variant in range(per_family):
            src = noise[family * per_family + variant]
            sequences.append("".join(src[i] if rng.random() < mutation_rate else base[i] for i in range(length)))
    return sequences


@pytest.mark.benchmark("mmseqs2-clustering")
@pytest.mark.slow
def test_mmseqs2_clustering_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark mmseqs2-clustering: cluster 1500 protein sequences in 50 homologous families of 200 aa (cold + warm)."""
    sequences = _clustered_protein_sequences(n_families=50, per_family=30, length=200, mutation_rate=0.2, seed=0)
    inputs = Mmseqs2ClusteringInput(input_sequences=sequences)
    config = Mmseqs2ClusteringConfig(min_seq_id=0.5)

    result = benchmark_twice(request, "mmseqs2", lambda: run_mmseqs2_clustering(inputs, config))
    validate_output(result)

    assert result.tool_id == "mmseqs2-clustering"
    assert isinstance(result, Mmseqs2ClusteringOutput)
    assert len(result) == 1500
    assert 1 < result.num_clusters <= 100
    assert len(result.representative_indices) == result.num_clusters


def _homolog_genome_db(queries: list[str], per_query: int, noise: int, mutation: float, seed: int) -> list[str]:
    """Each query gets ``per_query`` ~mutation-diverged homologs (real hits below 100% identity) plus ``noise`` random genomes."""
    rng = random.Random(seed)
    length = len(queries[0])
    variant_src = random_dna_sequences(per_query * len(queries), length, seed)
    targets = []
    for q_idx, base in enumerate(queries):
        for j in range(per_query):
            src = variant_src[q_idx * per_query + j]
            targets.append("".join(src[i] if rng.random() < mutation else base[i] for i in range(length)))
    targets += random_dna_sequences(noise, length, seed + 7)
    return targets


@pytest.mark.benchmark("mmseqs2-search-genomes")
@pytest.mark.slow
def test_mmseqs2_search_genomes_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark mmseqs2-search-genomes: 50 query genomes vs a 400-genome DB of ~90%-identity homologs + noise (cold + warm)."""
    query_genomes = random_dna_sequences(n=50, length=1000, seed=0)
    target_genomes = _homolog_genome_db(query_genomes, per_query=4, noise=200, mutation=0.10, seed=1)
    inputs = Mmseqs2SearchGenomesInput(query_genomes=query_genomes)
    config = Mmseqs2SearchGenomesConfig(target_genomes=target_genomes, threads=4)

    result = benchmark_twice(request, "mmseqs2", lambda: run_mmseqs2_search_genomes(inputs, config))
    validate_output(result)

    assert result.tool_id == "mmseqs2-search-genomes"
    assert isinstance(result, Mmseqs2SearchGenomesOutput)
    assert len(result) == 50
    assert result.total_hits >= len(query_genomes)

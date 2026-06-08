"""Tests for the mmseqs2-homology-search tool (protein; unpaired + taxonomy-paired)."""

import logging
import platform
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from proto_tools.databases import DatasetRegistry, dataset_slug, get_dataset_dir
from proto_tools.tools.sequence_alignment.mmseqs2 import (
    Mmseqs2HomologySearchConfig,
    Mmseqs2HomologySearchInput,
    Mmseqs2HomologySearchQuery,
    run_mmseqs2_homology_search,
)
from proto_tools.tools.sequence_alignment.mmseqs2.homology_search import (
    _assemble_paired_result,
    _check_dataset_provisioned,
    _rename_a3m_to_sequence_id,
)
from proto_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)


# ============================================================================
# Test Data
# ============================================================================

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
HEMOGLOBIN_ALPHA = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"

# Full-length human hemoglobin alpha/beta: a vertebrate-wide heterodimer, so the chains pair into deep, row-aligned MSAs.
HBA_HUMAN = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
HBB_HUMAN = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"


# ============================================================================
# Mocking helpers (no GPU / no provisioned DB needed)
# ============================================================================


def _write_a3m(path: Path, query_seq: str, n_rows: int) -> None:
    """Write an A3M with a query row plus ``n_rows - 1`` equal-length homolog rows."""
    lines = [f">query\n{query_seq}"]
    lines += [f">hom{i}\n{query_seq}" for i in range(n_rows - 1)]
    path.write_text("\n".join(lines) + "\n")


def _provision_fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal on-disk cache so the pre-dispatch provisioning check passes."""
    entry = DatasetRegistry.get("uniref30-2302")
    cache = tmp_path / dataset_slug("uniref30-2302")
    cache.mkdir()
    for out in entry.index_recipe.output_files or []:
        (cache / out.replace("{name}", dataset_slug("uniref30-2302"))).write_bytes(b"")
    monkeypatch.setattr(
        "proto_tools.tools.sequence_alignment.mmseqs2.homology_search.get_dataset_dir",
        lambda _: cache,
    )
    return cache


def _install_fake_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    captured: list[dict[str, Any]],
    *,
    paired_depth: int = 3,
) -> None:
    """Patch ``ToolInstance.dispatch`` to drop synthetic per-chain A3M files.

    A paired call (``pairing_strategy`` set) writes ``{i}.a3m`` (unpaired) and
    ``{i}.paired.a3m`` (row-aligned) per chain; an unpaired batch writes
    ``__q{idx}.a3m`` per sequence.
    """

    def fake(toolkit: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        captured.append(payload)
        out_dir = Path(payload["output_dir"])
        sequences = payload["sequences"]
        if payload.get("pairing_strategy") is not None:
            for i, seq in enumerate(sequences):
                _write_a3m(out_dir / f"{i}.a3m", seq, 4)
                _write_a3m(out_dir / f"{i}.paired.a3m", seq, paired_depth)
        else:
            for idx, seq in enumerate(sequences):
                _write_a3m(out_dir / f"__q{idx}.a3m", seq, 4)
        return {"success": True, "output_dir": payload["output_dir"], "db_name": "uniref30_2302_db"}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(fake))


def _uniref30_provisioned() -> bool:
    """Whether the UniRef30 dataset is fully provisioned on this host."""
    entry = DatasetRegistry.get("uniref30-2302")
    cache = get_dataset_dir("uniref30-2302")
    return (cache / f"{entry.db_prefix}.dbtype").is_file()


# ============================================================================
# Input validation
# ============================================================================


def test_string_query_sugar_creates_singleton_groups() -> None:
    """Plain strings become singleton query groups with auto-generated IDs."""
    inp = Mmseqs2HomologySearchInput(queries=[UBIQUITIN, HEMOGLOBIN_ALPHA])
    assert len(inp) == 2
    flat = inp.all_queries()
    assert len(flat) == 2
    assert all(q.sequence_id is not None and q.sequence_id.startswith("seq_") for q in flat)


def test_tuple_query_sugar_carries_id() -> None:
    """``(sequence, id)`` tuples carry through as the sequence's identifier."""
    inp = Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubi"), (HEMOGLOBIN_ALPHA, "hba")])
    ids = [q.sequence_id for q in inp.all_queries()]
    assert ids == ["ubi", "hba"]


def test_empty_queries_rejected() -> None:
    """An empty queries list is a validation error."""
    with pytest.raises(ValidationError, match="At least one query group"):
        Mmseqs2HomologySearchInput(queries=[])


def test_empty_sequence_rejected() -> None:
    """Whitespace-only sequences fail validation."""
    with pytest.raises(ValidationError, match="non-empty"):
        Mmseqs2HomologySearchQuery(sequence="   ")


def test_duplicate_sequence_ids_rejected() -> None:
    """Globally unique sequence_ids are required across all groups."""
    with pytest.raises(ValidationError, match="Duplicate sequence_id"):
        Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "x"), (HEMOGLOBIN_ALPHA, "x")])


def test_paired_group_accepted_as_nested_list() -> None:
    """A nested list of chains is accepted as one paired group."""
    inp = Mmseqs2HomologySearchInput(
        queries=[
            [
                Mmseqs2HomologySearchQuery(sequence=UBIQUITIN, sequence_id="a"),
                Mmseqs2HomologySearchQuery(sequence=HEMOGLOBIN_ALPHA, sequence_id="b"),
            ]
        ]
    )
    assert len(inp) == 1
    assert isinstance(inp.queries[0], list)
    assert [q.sequence_id for q in inp.queries[0]] == ["a", "b"]


def test_input_accepts_mixed_singleton_and_paired_groups() -> None:
    """One ``queries`` list may interleave singleton items and nested paired groups."""
    inp = Mmseqs2HomologySearchInput(
        queries=[
            (UBIQUITIN, "ubi"),
            [
                Mmseqs2HomologySearchQuery(sequence=HBA_HUMAN, sequence_id="hba"),
                Mmseqs2HomologySearchQuery(sequence=HBB_HUMAN, sequence_id="hbb"),
            ],
        ]
    )
    assert len(inp) == 2
    assert isinstance(inp.queries[0], Mmseqs2HomologySearchQuery)
    assert isinstance(inp.queries[1], list)
    assert [q.sequence_id for q in inp.all_queries()] == ["ubi", "hba", "hbb"]


# ============================================================================
# Config validation
# ============================================================================


def test_unknown_dataset_rejected() -> None:
    """A value outside the searchable product Literal is rejected at construction."""
    with pytest.raises(ValidationError, match="Input should be"):
        Mmseqs2HomologySearchConfig(dataset="does-not-exist")


def test_non_searchable_dataset_not_selectable() -> None:
    """A registered but non-ColabFold DB (a3m_adapter != 'colabfold') is not a valid ``dataset`` value."""
    with pytest.raises(ValidationError, match="Input should be"):
        Mmseqs2HomologySearchConfig(dataset="small-bfd", use_gpu=False)


def test_dataset_field_schema_carries_inline_enum_and_default() -> None:
    """The ``dataset`` Literal emits an INLINE JSON-Schema enum + default (drives the proto-ui dropdown).

    The proto-ui parser reads ``schemaProp.enum`` at the property level, so the
    enum must be inline. A Python ``Enum`` would instead emit a ``$ref`` into
    ``$defs`` that the parser doesn't resolve — a ``Literal`` stays inline.
    """
    prop = Mmseqs2HomologySearchConfig.model_json_schema()["properties"]["dataset"]
    assert prop["enum"] == ["colabfold-envdb-202108", "uniref30-2302"]
    assert prop["default"] == "uniref30-2302"
    assert "$ref" not in prop  # inline enum, not a $defs reference


def test_dataset_literal_matches_registry_searchable_set() -> None:
    """Drift guard: the ``dataset`` Literal must equal the registry's searchable product DBs.

    Searchable == ``a3m_adapter == "colabfold"`` minus the in-tree
    ``tiny-test-colabfold`` fixture. Registering a new searchable DB fails this
    assertion, prompting an update to the Literal (and thus the client dropdown).
    """
    searchable = {
        name
        for name in DatasetRegistry.list_all()
        if DatasetRegistry.get(name).a3m_adapter == "colabfold" and name != "tiny-test-colabfold"
    }
    literal_values = set(get_args(Mmseqs2HomologySearchConfig.model_fields["dataset"].annotation))
    assert literal_values == searchable


def test_default_search_mode_is_remote_and_claims_no_gpu() -> None:
    """The default config is remote: no local DB, and it claims no GPU even though use_gpu stays True."""
    cfg = Mmseqs2HomologySearchConfig()
    assert cfg.search_mode == "remote"
    assert cfg.gpus_per_instance == 0


def test_remote_mode_skips_gpu_platform_validation() -> None:
    """Remote mode skips the GPU-platform validator, so use_gpu=True doesn't raise even on non-Linux."""
    cfg = Mmseqs2HomologySearchConfig(search_mode="remote", use_gpu=True)
    assert cfg.search_mode == "remote"
    assert cfg.gpus_per_instance == 0


@pytest.mark.skipif(platform.system() != "Linux", reason="GPU validation is Linux-only")
def test_local_use_gpu_default_true() -> None:
    """In local mode, GPU is on by default and claims one GPU per instance."""
    cfg = Mmseqs2HomologySearchConfig(search_mode="local")
    assert cfg.use_gpu is True
    assert cfg.gpus_per_instance == 1


def test_gpus_per_instance_zero_when_cpu() -> None:
    """A local CPU-mode config reports zero GPU usage."""
    cfg = Mmseqs2HomologySearchConfig(search_mode="local", use_gpu=False)
    assert cfg.gpus_per_instance == 0


def test_missing_dataset_dir_gives_provisioning_hint(tmp_path: Path) -> None:
    """Dispatch-time check points users at the mmseqs2 setup_databases CLI when the DB isn't on disk."""
    entry = DatasetRegistry.get("uniref30-2302")
    bogus_cache = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match=r"setup_databases uniref30-2302"):
        _check_dataset_provisioned("uniref30-2302", entry, bogus_cache, require_idx_pad=True)


def test_missing_gpu_padded_marker_gives_use_gpu_false_hint(tmp_path: Path) -> None:
    """Dispatch-time check explains the missing GPU marker and suggests use_gpu=False."""
    entry = DatasetRegistry.get("uniref30-2302")
    # Provision dbtype but NOT the gpu_padded_marker — simulates a CPU-only DB build.
    (tmp_path / f"{entry.db_prefix}.dbtype").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"use_gpu=False"):
        _check_dataset_provisioned("uniref30-2302", entry, tmp_path, require_idx_pad=True)


def test_padded_marker_without_dbtype_is_rejected(tmp_path: Path) -> None:
    """Bare padded data file without sibling ``.dbtype`` is an incomplete build."""
    entry = DatasetRegistry.get("uniref30-2302")
    (tmp_path / f"{entry.db_prefix}.dbtype").write_bytes(b"")
    (tmp_path / entry.gpu_padded_marker).write_bytes(b"")
    # No <marker>.dbtype companion — must reject.
    with pytest.raises(FileNotFoundError, match=r"sibling \.dbtype"):
        _check_dataset_provisioned("uniref30-2302", entry, tmp_path, require_idx_pad=True)


def test_local_search_low_memory_reason_tracks_seq_db_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reason fires when RAM < sequence-DB size, and clears when RAM fits or the file is absent."""
    import proto_tools.tools.sequence_alignment.mmseqs2.homology_search as hs

    entry = DatasetRegistry.get("uniref30-2302")
    seq_db = tmp_path / f"{entry.db_prefix}_seq"
    with open(seq_db, "wb") as f:
        f.truncate(100 * 1024**3)  # 100 GiB, sparse (no real allocation)

    # RAM below the sequence DB: warn/skip with an actionable reason.
    monkeypatch.setattr(hs, "available_memory_bytes", lambda: 80 * 1024**3)
    reason = hs.local_search_low_memory_reason(entry, tmp_path)
    assert reason is not None and "remote" in reason

    # RAM above the sequence DB: fast path, no reason.
    monkeypatch.setattr(hs, "available_memory_bytes", lambda: 200 * 1024**3)
    assert hs.local_search_low_memory_reason(entry, tmp_path) is None

    # Sequence DB absent (size unknown): fall through rather than guess.
    seq_db.unlink()
    monkeypatch.setattr(hs, "available_memory_bytes", lambda: 1)
    assert hs.local_search_low_memory_reason(entry, tmp_path) is None


def test_dispatch_payload_carries_operation_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool layer must include ``operation="homology_search"`` in the dispatch payload.

    Regression: the unified ``mmseqs2`` standalone routes by ``operation`` key.
    A previous version of the merged tool layer omitted this key, which would
    have made the dispatcher raise ``unknown operation None`` at runtime.
    """
    from proto_tools.tools.sequence_alignment.mmseqs2 import (
        Mmseqs2HomologySearchConfig,
        Mmseqs2HomologySearchInput,
        run_mmseqs2_homology_search,
    )
    from proto_tools.utils.tool_instance import ToolInstance

    # Provision a minimal cache so the pre-dispatch ``_check_dataset_provisioned`` passes.
    entry = DatasetRegistry.get("uniref30-2302")
    cache = tmp_path / dataset_slug("uniref30-2302")
    cache.mkdir()
    for out in entry.index_recipe.output_files or []:
        (cache / out.replace("{name}", dataset_slug("uniref30-2302"))).write_bytes(b"")
    monkeypatch.setattr(
        "proto_tools.tools.sequence_alignment.mmseqs2.homology_search.get_dataset_dir",
        lambda _: cache,
    )

    captured: dict[str, object] = {}

    def fake_dispatch(toolkit: str, payload: dict[str, object], **_: object) -> dict[str, object]:
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        return {"success": True, "output_dir": payload["output_dir"], "db_name": entry.db_prefix}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(fake_dispatch))

    # The @tool wrapper catches the downstream A3M parse failure (output_dir is
    # empty) so the call returns rather than raising. We only need the payload.
    run_mmseqs2_homology_search(
        Mmseqs2HomologySearchInput(queries=["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLI"]),
        Mmseqs2HomologySearchConfig(search_mode="local", use_gpu=False),
    )

    assert captured["toolkit"] == "mmseqs2"
    assert captured["payload"]["operation"] == "homology_search"


# ============================================================================
# Databases root resolution (PROTO_DATABASES_DIR)
# ============================================================================


def test_databases_root_prefers_proto_databases_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PROTO_DATABASES_DIR is used verbatim as the databases root, overriding PROTO_MODEL_CACHE."""
    from proto_tools.databases.registry import get_databases_root

    monkeypatch.setenv("PROTO_DATABASES_DIR", str(tmp_path / "scratch_dbs"))
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path / "weights"))
    assert get_databases_root() == tmp_path / "scratch_dbs"


def test_databases_root_falls_back_to_model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without PROTO_DATABASES_DIR, the root is `$PROTO_MODEL_CACHE/databases/`."""
    from proto_tools.databases.registry import get_databases_root

    monkeypatch.delenv("PROTO_DATABASES_DIR", raising=False)
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path / "weights"))
    assert get_databases_root() == tmp_path / "weights" / "databases"


def test_get_dataset_dir_composes_with_databases_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_dataset_dir places the dataset's slug under the resolved databases root."""
    monkeypatch.setenv("PROTO_DATABASES_DIR", str(tmp_path / "dbs"))
    assert get_dataset_dir("uniref30-2302") == tmp_path / "dbs" / "uniref30_2302"


# ============================================================================
# Per-dataset provisioning + integration — auto-skip when not on device
# ============================================================================


def _provisioned_datasets() -> list[str]:
    """Return registry keys whose declared output_files are all present on disk.

    Used to parametrize per-dataset tests so they auto-skip when a dataset
    isn't installed; a fresh dev machine sees no parametrize entries skipped
    here unless the user has run ``setup_databases.py`` for that dataset.
    """
    found: list[str] = []
    for name in DatasetRegistry.list_all():
        entry = DatasetRegistry.get(name)
        cache = get_dataset_dir(name)
        if not entry.index_recipe.output_files:
            if (cache / f"{entry.db_prefix}.dbtype").is_file():
                found.append(name)
            continue
        if all((cache / out.replace("{name}", dataset_slug(name))).exists() for out in entry.index_recipe.output_files):
            found.append(name)
    return found


@pytest.mark.parametrize("dataset_name", DatasetRegistry.list_all())
def test_registry_entry_outputs_match_disk_when_provisioned(dataset_name: str) -> None:
    """If a dataset's cache dir has its dbtype file, all declared output_files exist.

    Catches drift between an entry's ``index_recipe.output_files`` and what
    its provisioning steps actually produce. Skips when not provisioned.
    """
    entry = DatasetRegistry.get(dataset_name)
    cache = get_dataset_dir(dataset_name)
    if not (cache / f"{entry.db_prefix}.dbtype").is_file():
        pytest.skip(f"{dataset_name} not provisioned at {cache}")
    missing = [
        out
        for out in entry.index_recipe.output_files
        if not (cache / out.replace("{name}", dataset_slug(dataset_name))).exists()
    ]
    assert not missing, (
        f"{dataset_name} declared output_files {missing} are missing from {cache}; "
        f"either provisioning failed or the entry's output_files list is stale."
    )


@pytest.mark.uses_gpu
@pytest.mark.parametrize("dataset_name", _provisioned_datasets() or ["__none_provisioned__"])
def test_e2e_search_against_provisioned_protein_dataset(dataset_name: str) -> None:
    """End-to-end GPU search against each provisioned protein dataset.

    Auto-parametrizes over whatever's on disk. RNA datasets are skipped
    (current tool surface is protein-only). Asserts the search returns
    a non-empty MSA for ubiquitin (universally conserved).
    """
    if dataset_name == "__none_provisioned__":
        pytest.skip("No datasets provisioned on this device")

    entry = DatasetRegistry.get(dataset_name)
    if entry.molecule_type != "protein":
        pytest.skip(f"{dataset_name} is {entry.molecule_type}; current tool is protein-only")
    if not entry.supports_gpu:
        pytest.skip(f"{dataset_name} declares supports_gpu=False")
    if entry.a3m_adapter != "colabfold":
        pytest.skip(f"{dataset_name} uses a3m_adapter={entry.a3m_adapter!r}; tool only handles colabfold-style DBs")
    if dataset_name == "tiny-test-colabfold":
        pytest.skip("test fixture excluded from the product Literal; exercised separately via model_construct")

    inp = Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubiquitin")])
    cfg = Mmseqs2HomologySearchConfig(search_mode="local", dataset=dataset_name, use_gpu=True)
    result = run_mmseqs2_homology_search(inp, cfg)

    assert result.success, f"Search against {dataset_name} failed: {result.errors}"
    assert len(result.results) == 1
    grp = result.results[0]
    assert grp.datasets_searched == [dataset_name]
    # Ubiquitin against any reasonable protein DB should return >0 homologs.
    # Use a low threshold (>5) so PDB-seqres-style tiny DBs still pass.
    assert grp.num_homologs_found[0] > 5, (
        f"{dataset_name} returned only {grp.num_homologs_found[0]} homologs for ubiquitin "
        "(expected >5 — universally conserved protein)"
    )


def _local_metagenomic_dbs_ready() -> bool:
    """True when UniRef30 + the ColabFold envdb are provisioned with GPU (idx_pad) indexes.

    Respects ``PROTO_DATABASES_DIR`` (the databases may live on scratch), so the
    test runs only where both DBs are actually present and skips everywhere else.
    """
    for name in ("uniref30-2302", "colabfold-envdb-202108"):
        entry = DatasetRegistry.get(name)
        cache = get_dataset_dir(name)
        if not (cache / f"{entry.db_prefix}.dbtype").is_file():
            return False
        if not (cache / entry.gpu_padded_marker).exists():
            return False
    return True


@pytest.mark.extensive
@pytest.mark.integration
@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.skipif(
    not _local_metagenomic_dbs_ready(),
    reason="UniRef30 + colabfold-envdb-202108 not provisioned with GPU indexes (set PROTO_DATABASES_DIR)",
)
def test_local_metagenomic_search_deepens_msa() -> None:
    """Local GPU search: ``use_metagenomic_db=True`` searches the envdb (``--db3``) and deepens the MSA.

    Runs the same query with metagenomic off (UniRef30 only) and on (UniRef30 + the
    ColabFold envdb). The envdb only *adds* environmental homologs, so the metagenomic
    MSA must be at least as deep — and for this query, strictly deeper. This is the
    end-to-end check that the local ``--db3`` / ``--use-env 1`` wiring actually runs.

    Marked ``extensive`` (run with ``--ext``): a full UniRef30 + ~650 GB envdb search
    takes many minutes. Requires both DBs provisioned with GPU indexes — point
    ``PROTO_DATABASES_DIR`` at wherever they live (e.g. scratch).
    """
    inp = Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubiquitin")])
    # Generous timeout: UniRef30 + 650 GB envdb is I/O-bound and slow when RAM < DB size.
    base = {"search_mode": "local", "use_gpu": True, "timeout": 21600}

    uniref_only = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(**base, use_metagenomic_db=False))
    with_env = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(**base, use_metagenomic_db=True))

    assert uniref_only.success and with_env.success, f"search failed: {uniref_only.errors} / {with_env.errors}"
    uniref_msa = uniref_only.results[0].msas[0]
    env_msa = with_env.results[0].msas[0]
    assert uniref_msa is not None and env_msa is not None, "both searches must return an MSA"
    assert uniref_msa.num_sequences > 1, "UniRef30 search returned no homologs"
    # The envdb only adds sequences; the metagenomic MSA must be strictly deeper for a query with environmental homology.
    assert env_msa.num_sequences >= uniref_msa.num_sequences
    assert env_msa.num_sequences > uniref_msa.num_sequences, (
        f"metagenomic DB did not deepen the MSA (uniref={uniref_msa.num_sequences}, env={env_msa.num_sequences})"
    )


# ============================================================================
# Paired (multi-chain) groups — mocked dispatch, no GPU / DB needed
# ============================================================================


def test_paired_group_produces_row_aligned_paired_msas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2-chain paired group yields per-chain unpaired + row-aligned paired MSAs."""
    _provision_fake_cache(tmp_path, monkeypatch)
    captured: list[dict[str, Any]] = []
    _install_fake_dispatch(monkeypatch, captured, paired_depth=3)

    inp = Mmseqs2HomologySearchInput(queries=[[(UBIQUITIN, "chainA"), (HEMOGLOBIN_ALPHA, "chainB")]])
    out = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(search_mode="local", use_gpu=False))

    assert len(out.results) == 1
    grp = out.results[0]
    assert grp.sequence_ids == ["chainA", "chainB"]
    assert all(m is not None for m in grp.msas)
    assert all(m is not None for m in grp.paired_msas)
    # Paired MSAs are row-aligned: equal depth across chains.
    depths = {m.num_sequences for m in grp.paired_msas if m is not None}
    assert depths == {3}
    # Exactly one paired dispatch (the group), submitted as a complex.
    assert len(captured) == 1
    assert captured[0]["pairing_strategy"] == 0  # greedy default
    assert captured[0]["sequences"] == [UBIQUITIN, HEMOGLOBIN_ALPHA]


def test_remote_paired_group_fetches_paired_and_deep_unpaired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote paired group returns row-aligned paired MSAs AND each chain's deeper unpaired MSA."""
    from proto_tools.tools.sequence_alignment.mmseqs2 import homology_search as hs

    members = [
        Mmseqs2HomologySearchQuery(sequence=UBIQUITIN, sequence_id="chainA"),
        Mmseqs2HomologySearchQuery(sequence=HEMOGLOBIN_ALPHA, sequence_id="chainB"),
    ]
    paired_dir = tmp_path / "paired"
    unpaired_dir = tmp_path / "unpaired"
    paired_dir.mkdir()
    unpaired_dir.mkdir()
    paired_paths: list[str] = []
    unpaired_map: dict[str, str] = {}
    for i, q in enumerate(members):
        pp = paired_dir / f"{i}.a3m"
        _write_a3m(pp, q.sequence, 3)
        paired_paths.append(str(pp))
        up = unpaired_dir / f"{i}.a3m"
        _write_a3m(up, q.sequence, 6)
        unpaired_map[str(i)] = str(up)

    def fake_dispatch(remote_queries, output_dir, config, instance, script_path):  # type: ignore[no-untyped-def]
        if isinstance(remote_queries[0]["sequences"], list):
            return {"success": True, "paired_msa_paths": {"0": paired_paths}}
        return {"success": True, "msa_paths": unpaired_map}

    monkeypatch.setattr(hs, "_dispatch_remote", fake_dispatch)
    result = hs._remote_paired_group(members, tmp_path / "grp", Mmseqs2HomologySearchConfig(), None, Path("x"))

    assert all(m is not None for m in result.paired_msas)
    assert all(m is not None for m in result.msas)
    assert {m.num_sequences for m in result.paired_msas if m is not None} == {3}
    assert {m.num_sequences for m in result.msas if m is not None} == {6}  # deeper unpaired


def test_remote_paired_group_degrades_when_unpaired_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unpaired-call failure leaves paired_msas intact and degrades msas to None (never breaks paired)."""
    from proto_tools.tools.sequence_alignment.mmseqs2 import homology_search as hs

    members = [
        Mmseqs2HomologySearchQuery(sequence=UBIQUITIN, sequence_id="chainA"),
        Mmseqs2HomologySearchQuery(sequence=HEMOGLOBIN_ALPHA, sequence_id="chainB"),
    ]
    paired_dir = tmp_path / "paired"
    paired_dir.mkdir()
    paired_paths: list[str] = []
    for i, q in enumerate(members):
        pp = paired_dir / f"{i}.a3m"
        _write_a3m(pp, q.sequence, 3)
        paired_paths.append(str(pp))

    def fake_dispatch(remote_queries, output_dir, config, instance, script_path):  # type: ignore[no-untyped-def]
        if isinstance(remote_queries[0]["sequences"], list):
            return {"success": True, "paired_msa_paths": {"0": paired_paths}}
        raise RuntimeError("unpaired API down")

    monkeypatch.setattr(hs, "_dispatch_remote", fake_dispatch)
    result = hs._remote_paired_group(members, tmp_path / "grp", Mmseqs2HomologySearchConfig(), None, Path("x"))

    assert all(m is not None for m in result.paired_msas)  # paired result intact
    assert all(m is None for m in result.msas)  # unpaired degraded, no exception


def test_paired_homomultimer_yields_per_chain_msas_with_correct_query_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A homomultimer paired group yields one MSA per chain index, each with the chain's query row.

    Counterpart to colabfold's ``test_local_paired_reorders_and_reuses_duplicate_chain_msas`` —
    mmseqs's standalone emits one A3M per chain index (no dedup), so the assertion is per-position
    rather than identity, but the spirit is the same: duplicate input chains map to coherent
    per-chain MSAs whose query rows are the requested chain's sequence.
    """
    _provision_fake_cache(tmp_path, monkeypatch)
    captured: list[dict[str, Any]] = []
    _install_fake_dispatch(monkeypatch, captured, paired_depth=3)

    inp = Mmseqs2HomologySearchInput(
        queries=[
            [
                (UBIQUITIN, "chainA"),
                (UBIQUITIN, "chainB"),
                (HBA_HUMAN, "chainC"),
                (HBA_HUMAN, "chainD"),
            ]
        ]
    )
    out = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(search_mode="local", use_gpu=False))

    grp = out.results[0]
    assert grp.sequence_ids == ["chainA", "chainB", "chainC", "chainD"]
    assert all(m is not None for m in grp.paired_msas)
    # Each chain's paired MSA's query row matches the chain's input sequence.
    queries = [m.original_sequences[0] for m in grp.paired_msas if m is not None]
    assert queries == [UBIQUITIN, UBIQUITIN, HBA_HUMAN, HBA_HUMAN]


def test_paired_assembly_raises_on_unmatched_query_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assembly raises when a paired A3M's query row doesn't match the requested chain.

    Defensive guard against a corrupt standalone output silently mis-keying MSAs to the wrong
    chain (matching colabfold's ``test_local_paired_raises_on_unmatched_msa_query_row``).
    """
    _provision_fake_cache(tmp_path, monkeypatch)

    def fake(toolkit: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        out_dir = Path(payload["output_dir"])
        sequences = payload["sequences"]
        _write_a3m(out_dir / "0.a3m", sequences[0], 3)
        # Chain 0's paired MSA carries chain 1's sequence as its query — the bug we want to catch.
        _write_a3m(out_dir / "0.paired.a3m", sequences[1], 3)
        _write_a3m(out_dir / "1.a3m", sequences[1], 3)
        _write_a3m(out_dir / "1.paired.a3m", sequences[1], 3)
        return {"success": True, "output_dir": payload["output_dir"], "db_name": "uniref30_2302_db"}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(fake))

    inp = Mmseqs2HomologySearchInput(queries=[[(UBIQUITIN, "a"), (HBB_HUMAN, "b")]])
    with pytest.raises(RuntimeError, match="paired MSA for chain 0"):
        run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(search_mode="local", use_gpu=False))


@pytest.mark.parametrize(("strategy", "expected_int"), [("greedy", 0), ("complete", 1)])
def test_pairing_strategy_maps_to_mmseqs_int(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy: str, expected_int: int
) -> None:
    """``pairing_strategy`` reaches the dispatch payload as the mmseqs pairing-mode int."""
    _provision_fake_cache(tmp_path, monkeypatch)
    captured: list[dict[str, Any]] = []
    _install_fake_dispatch(monkeypatch, captured)

    inp = Mmseqs2HomologySearchInput(queries=[[(UBIQUITIN, "a"), (HEMOGLOBIN_ALPHA, "b")]])
    run_mmseqs2_homology_search(
        inp, Mmseqs2HomologySearchConfig(search_mode="local", use_gpu=False, pairing_strategy=strategy)
    )

    assert captured[0]["pairing_strategy"] == expected_int


def test_singletons_batch_while_paired_group_dispatches_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Singletons share one unpaired batch; the paired group dispatches on its own, order preserved."""
    _provision_fake_cache(tmp_path, monkeypatch)
    captured: list[dict[str, Any]] = []
    _install_fake_dispatch(monkeypatch, captured)

    inp = Mmseqs2HomologySearchInput(
        queries=[
            (UBIQUITIN, "solo1"),
            [(UBIQUITIN, "pairA"), (HEMOGLOBIN_ALPHA, "pairB")],
            (HEMOGLOBIN_ALPHA, "solo2"),
        ]
    )
    out = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(search_mode="local", use_gpu=False))

    # Results stay parallel to input groups.
    assert [r.sequence_ids for r in out.results] == [["solo1"], ["pairA", "pairB"], ["solo2"]]
    # Singleton groups carry no paired MSAs; the paired group does.
    assert out.results[0].paired_msas == [None]
    assert out.results[2].paired_msas == [None]
    assert all(m is not None for m in out.results[1].paired_msas)
    # One batched unpaired dispatch (both singletons) + one paired dispatch.
    unpaired = [p for p in captured if p["pairing_strategy"] is None]
    paired = [p for p in captured if p["pairing_strategy"] is not None]
    assert len(unpaired) == 1 and unpaired[0]["sequences"] == [UBIQUITIN, HEMOGLOBIN_ALPHA]
    assert len(paired) == 1 and paired[0]["sequences"] == [UBIQUITIN, HEMOGLOBIN_ALPHA]


def test_assemble_paired_result_keeps_per_chain_paired_and_unpaired(tmp_path: Path) -> None:
    """Per-chain paired and unpaired MSAs are both returned; an empty paired chain stays None (no hard-fail)."""
    members = [
        Mmseqs2HomologySearchQuery(sequence=UBIQUITIN, sequence_id="a"),
        Mmseqs2HomologySearchQuery(sequence=HEMOGLOBIN_ALPHA, sequence_id="b"),
    ]
    # Chain 0 gets a paired MSA; chain 1 gets none. Both get unpaired MSAs.
    _write_a3m(tmp_path / "0.a3m", UBIQUITIN, 4)
    _write_a3m(tmp_path / "0.paired.a3m", UBIQUITIN, 3)
    _write_a3m(tmp_path / "1.a3m", HEMOGLOBIN_ALPHA, 4)
    result = _assemble_paired_result(tmp_path, members, "uniref30-2302")
    assert result.paired_msas[0] is not None and result.paired_msas[1] is None
    assert all(m is not None for m in result.msas)  # unpaired present for both chains


def test_assemble_paired_result_no_pairing_returns_all_none(tmp_path: Path) -> None:
    """No paired files at all (no shared species) → paired_msas all None, no error."""
    members = [
        Mmseqs2HomologySearchQuery(sequence=UBIQUITIN, sequence_id="a"),
        Mmseqs2HomologySearchQuery(sequence=HEMOGLOBIN_ALPHA, sequence_id="b"),
    ]
    _write_a3m(tmp_path / "0.a3m", UBIQUITIN, 4)
    _write_a3m(tmp_path / "1.a3m", HEMOGLOBIN_ALPHA, 4)
    result = _assemble_paired_result(tmp_path, members, "uniref30-2302")
    assert result.paired_msas == [None, None]
    assert all(m is not None for m in result.msas)  # unpaired still present


@pytest.mark.integration
@pytest.mark.uses_cpu
@pytest.mark.slow
@pytest.mark.skip_ci
def test_real_paired_search_against_mini_db() -> None:
    """Real taxonomy-paired search against the auto-provisioning mini SwissProt DB.

    Exercises the full ``--unpack 0`` + ``unpackdb`` paired pipeline end-to-end
    (no GPU, no manual DB setup — the ``tiny-test-colabfold`` fixture
    auto-provisions on first run). Hemoglobin alpha+beta pair into deep,
    row-aligned MSAs.
    """
    inp = Mmseqs2HomologySearchInput(queries=[[(HBA_HUMAN, "hba"), (HBB_HUMAN, "hbb")]])
    # tiny-test-colabfold is excluded from the product `dataset` Literal, so build
    # the config via model_construct to bypass enum validation — this test exercises
    # the search pipeline, not the field's product allowlist.
    cfg = Mmseqs2HomologySearchConfig.model_construct(search_mode="local", dataset="tiny-test-colabfold", use_gpu=False)
    out = run_mmseqs2_homology_search(inp, cfg)

    assert out.success, f"paired search failed: {out.errors}"
    assert len(out.results) == 1
    grp = out.results[0]
    assert grp.sequence_ids == ["hba", "hbb"]
    # Both chains paired and row-aligned: one depth shared across the group.
    assert all(m is not None for m in grp.paired_msas)
    depths = {m.num_sequences for m in grp.paired_msas if m is not None}
    assert len(depths) == 1 and next(iter(depths)) > 5, (
        f"expected deep equal-depth paired MSAs, got {[m.num_sequences for m in grp.paired_msas if m]}"
    )
    # Unpaired per-chain MSAs are present alongside the paired ones.
    assert all(m is not None for m in grp.msas)


@pytest.mark.integration
@pytest.mark.uses_cpu
@pytest.mark.slow
@pytest.mark.skip_ci
def test_local_sensitivity_actually_changes_msa_depth() -> None:
    """``sensitivity`` reaches the mmseqs CLI and changes hit counts end-to-end.

    Counterpart to colabfold's ``test_with_sensitivity``: low sensitivity
    (``-s 1.0``) finds strictly fewer homologs than high sensitivity (``-s 8.0``)
    against the same query / DB. Runs against the auto-provisioning mini SwissProt
    fixture; ubiquitin is short so each search is sub-second.
    """

    def search(sensitivity: float) -> int:
        cfg = Mmseqs2HomologySearchConfig.model_construct(
            search_mode="local",
            dataset="tiny-test-colabfold",
            use_gpu=False,
            sensitivity=sensitivity,
        )
        out = run_mmseqs2_homology_search(Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubi")]), cfg)
        assert out.success, f"sensitivity={sensitivity} failed: {out.errors}"
        return out.results[0].num_homologs_found[0]

    low = search(1.0)
    high = search(8.0)
    assert low < high, (
        f"expected fewer homologs at sensitivity=1.0 ({low}) than at sensitivity=8.0 ({high}); "
        "sensitivity flag may not be reaching mmseqs"
    )


# ============================================================================
# Remote mode (ColabFold API) — mocked dispatch, no network
# ============================================================================


def _write_remote_a3m(path: Path, query_seq: str, query_header: str, n_homologs: int) -> None:
    """Write a remote-style A3M whose query header is ``query_header`` (e.g. ColabFold's ``101``)."""
    lines = [f">{query_header}\n{query_seq}"]
    lines += [f">hom{i}\n{query_seq}" for i in range(n_homologs)]
    path.write_text("\n".join(lines) + "\n")


def _install_fake_remote_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    *,
    paired_homologs: int = 3,
    unpaired_homologs: int = 4,
) -> None:
    """Patch ``ToolInstance.dispatch`` to emulate the ColabFold remote standalone.

    A paired call (``sequences`` is a list) returns ``paired_msa_paths["0"]`` with
    ``paired_homologs`` homologs per chain (0 → query-only, triggers the fallback);
    an unpaired call returns ``msa_paths`` keyed by submission index. Query headers
    use ColabFold's numeric ``101`` so header normalization is exercised.
    """

    def fake(toolkit: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        out_dir = Path(payload["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        first = payload["queries"][0]["sequences"]
        if isinstance(first, list):
            calls.append("paired")
            paths = []
            for chain_idx, seq in enumerate(first):
                p = out_dir / f"chain_{chain_idx}.a3m"
                _write_remote_a3m(p, seq, "101", paired_homologs)
                paths.append(str(p))
            return {
                "paired_msa_paths": {"0": paths},
                "msa_paths": {},
                "success": True,
                "num_successful": 1,
                "num_failed": 0,
            }
        calls.append("unpaired")
        msa_paths = {}
        for idx, q in enumerate(payload["queries"]):
            p = out_dir / f"unpaired_{idx}.a3m"
            _write_remote_a3m(p, q["sequences"], "101", unpaired_homologs)
            msa_paths[str(idx)] = str(p)
        return {
            "msa_paths": msa_paths,
            "paired_msa_paths": {},
            "success": True,
            "num_successful": len(msa_paths),
            "num_failed": 0,
        }

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(fake))


def test_remote_singleton_normalizes_header_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote singleton returns an unpaired MSA whose query header is normalized to the sequence_id."""
    calls: list[str] = []
    _install_fake_remote_dispatch(monkeypatch, calls)

    inp = Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubi")])
    out = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig())  # default search_mode="remote"

    assert calls == ["unpaired"]
    grp = out.results[0]
    assert grp.sequence_ids == ["ubi"]
    assert grp.paired_msas == [None]
    assert grp.datasets_searched == ["colabfold-remote"]
    assert grp.msas[0] is not None
    assert grp.msas[0].sequence_ids[0] == "ubi"  # ColabFold's ">101" normalized to the sequence_id
    assert grp.num_homologs_found == [4]


def test_remote_paired_group_returns_row_aligned_paired_msas(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote paired group returns row-aligned paired MSAs AND each chain's deep unpaired MSA."""
    calls: list[str] = []
    _install_fake_remote_dispatch(monkeypatch, calls, paired_homologs=3, unpaired_homologs=4)

    inp = Mmseqs2HomologySearchInput(queries=[[(HBA_HUMAN, "hba"), (HBB_HUMAN, "hbb")]])
    out = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(search_mode="remote"))

    assert sorted(calls) == ["paired", "unpaired"]  # paired + per-chain unpaired fetched in parallel
    grp = out.results[0]
    assert grp.sequence_ids == ["hba", "hbb"]
    assert all(m is not None for m in grp.paired_msas)
    assert all(m is not None for m in grp.msas)  # per-chain unpaired depth now retained
    assert {m.num_sequences for m in grp.paired_msas if m is not None} == {4}  # 1 query + 3 homologs, row-aligned
    assert {m.num_sequences for m in grp.msas if m is not None} == {5}  # 1 query + 4 homologs (deeper unpaired)


def test_remote_paired_group_no_pairing_keeps_unpaired(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no shared taxonomy the paired set is empty; the per-chain unpaired MSAs still come back."""
    calls: list[str] = []
    _install_fake_remote_dispatch(monkeypatch, calls, paired_homologs=0, unpaired_homologs=4)

    inp = Mmseqs2HomologySearchInput(queries=[[(HBA_HUMAN, "hba"), (HBB_HUMAN, "hbb")]])
    out = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(search_mode="remote"))

    assert sorted(calls) == ["paired", "unpaired"]  # both always run, in parallel
    grp = out.results[0]
    assert grp.paired_msas == [None, None]
    assert all(m is not None for m in grp.msas)
    assert {m.num_sequences for m in grp.msas if m is not None} == {5}  # 1 query + 4 homologs


def test_remote_paired_chain_path_count_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A standalone returning the wrong number of chain paths is a contract violation, not 'no homologs'."""

    def fake(toolkit: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        # 2-chain query, but only one chain path returned.
        return {"paired_msa_paths": {"0": [str(tmp_path / "only_one.a3m")]}, "success": True}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(fake))
    inp = Mmseqs2HomologySearchInput(queries=[[(HBA_HUMAN, "hba"), (HBB_HUMAN, "hbb")]])
    with pytest.raises(RuntimeError, match="expected 2"):
        run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(search_mode="remote"))


def test_remote_mixed_groups_preserve_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed singleton + paired + singleton groups keep result order; singletons share one unpaired batch."""
    calls: list[str] = []
    _install_fake_remote_dispatch(monkeypatch, calls, paired_homologs=3)

    inp = Mmseqs2HomologySearchInput(
        queries=[
            (UBIQUITIN, "solo1"),
            [(HBA_HUMAN, "pairA"), (HBB_HUMAN, "pairB")],
            (HEMOGLOBIN_ALPHA, "solo2"),
        ]
    )
    out = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(search_mode="remote"))

    assert [r.sequence_ids for r in out.results] == [["solo1"], ["pairA", "pairB"], ["solo2"]]
    assert out.results[0].paired_msas == [None] and out.results[2].paired_msas == [None]
    assert all(m is not None for m in out.results[1].paired_msas)
    # Two unpaired calls (the singletons batch + the paired group's per-chain unpaired) and one paired call.
    assert calls.count("unpaired") == 2 and calls.count("paired") == 1


# ============================================================================
# Metagenomic DB (use_metagenomic_db) — toggle, remote pass-through, local gate
# ============================================================================


def test_use_metagenomic_db_defaults_false() -> None:
    """The metagenomic/environmental DB is off by default in both modes."""
    assert Mmseqs2HomologySearchConfig().use_metagenomic_db is False
    assert Mmseqs2HomologySearchConfig(search_mode="local", use_gpu=False).use_metagenomic_db is False


def test_remote_use_metagenomic_db_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote mode forwards use_metagenomic_db to the standalone (→ ColabFold API use_env)."""
    captured: dict[str, Any] = {}

    def fake(toolkit: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        captured["payload"] = payload
        out_dir = Path(payload["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "0.a3m"
        _write_remote_a3m(p, UBIQUITIN, "101", 4)
        return {
            "msa_paths": {"0": str(p)},
            "paired_msa_paths": {},
            "success": True,
            "num_successful": 1,
            "num_failed": 0,
        }

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(fake))
    inp = Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubi")])
    run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(search_mode="remote", use_metagenomic_db=True))

    assert captured["payload"]["use_metagenomic_db"] is True


def test_local_metagenomic_requires_envdb_provisioned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Local metagenomic search hard-errors with a provisioning hint when the envdb isn't on disk."""
    _provision_fake_cache(tmp_path, monkeypatch)  # provisions UniRef30 only; the envdb is absent
    inp = Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubi")])
    cfg = Mmseqs2HomologySearchConfig(search_mode="local", use_gpu=False, use_metagenomic_db=True)
    with pytest.raises(FileNotFoundError, match=r"colabfold-envdb-202108"):
        run_mmseqs2_homology_search(inp, cfg)


def test_local_metagenomic_dispatch_carries_env_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the envdb provisioned, the local dispatch payload carries the env DB dir + prefix."""
    cache = _provision_fake_cache(tmp_path, monkeypatch)
    (cache / "colabfold_envdb_202108_db.dbtype").write_bytes(b"")  # provision the envdb too
    captured: list[dict[str, Any]] = []
    _install_fake_dispatch(monkeypatch, captured)

    inp = Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubi")])
    run_mmseqs2_homology_search(
        inp, Mmseqs2HomologySearchConfig(search_mode="local", use_gpu=False, use_metagenomic_db=True)
    )

    assert captured[0]["use_metagenomic_db"] is True
    assert captured[0]["env_db_prefix"] == "colabfold_envdb_202108_db"
    assert captured[0]["env_dataset_dir"] is not None


@pytest.mark.integration
@pytest.mark.skip_ci
def test_remote_search_live_paired_hemoglobin() -> None:
    """Live remote paired query: hemoglobin alpha+beta return row-aligned per-chain MSAs of equal depth."""
    inp = Mmseqs2HomologySearchInput(queries=[[(HBA_HUMAN, "hba"), (HBB_HUMAN, "hbb")]])
    out = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(search_mode="remote"))

    assert len(out.results) == 1
    grp = out.results[0]
    assert grp.sequence_ids == ["hba", "hbb"]
    assert all(m is not None for m in grp.paired_msas)
    # Paired output is row-aligned: both per-chain MSAs share the same depth.
    assert grp.paired_msas[0].num_sequences == grp.paired_msas[1].num_sequences > 1


def test_rename_a3m_avoids_collision_with_adversarial_numeric_ids(tmp_path: Path) -> None:
    """Regression: numeric sequence_ids that look like ``__q{idx}`` indices must not collide.

    Pre-fix the FASTA used user sequence_ids as headers, so colabfold_search's
    output naming collided with user IDs (sequence_id="1" for the query at
    idx=0 would clobber the second query's "1.a3m" before our code ran). The
    fix uses internal ``__q{idx}`` source names that can't collide with user
    IDs. This test exercises that path directly with adversarial swapped IDs
    — fast, runs in CI, doesn't need a GPU or provisioned database.
    """
    # Two distinct internal A3Ms — each carries identifiable content so we can
    # verify no swap or clobber.
    (tmp_path / "__q0.a3m").write_text(">__q0\nUBIQUITIN_PLACEHOLDER\n>homolog_a\nA\n>homolog_b\nB\n")
    (tmp_path / "__q1.a3m").write_text(">__q1\nHEMOGLOBIN_PLACEHOLDER\n>homolog_x\nX\n")

    # Adversarial IDs: idx=0 wants "1", idx=1 wants "0" — exact swap.
    p0 = _rename_a3m_to_sequence_id(tmp_path, idx=0, sequence_id="1")
    p1 = _rename_a3m_to_sequence_id(tmp_path, idx=1, sequence_id="0")

    assert p0 == tmp_path / "1.a3m"
    assert p1 == tmp_path / "0.a3m"

    # Critical: each user-facing file holds the right query's content.
    assert "UBIQUITIN_PLACEHOLDER" in p0.read_text()
    assert "HEMOGLOBIN_PLACEHOLDER" in p1.read_text()

    # Query headers were rewritten; homolog headers untouched.
    p0_lines = p0.read_text().split("\n")
    p1_lines = p1.read_text().split("\n")
    assert p0_lines[0] == ">1"
    assert p1_lines[0] == ">0"
    assert ">homolog_a" in p0_lines
    assert ">homolog_x" in p1_lines

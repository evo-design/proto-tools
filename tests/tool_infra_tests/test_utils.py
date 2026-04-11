"""Tests for small utility modules."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tqdm import tqdm

from proto_tools.utils.auth import require_hf_token, resolve_hf_token
from proto_tools.utils.chemistry import validate_smiles
from proto_tools.utils.msa import extract_msa_sequences
from proto_tools.utils.progress import _is_disabled, progress_bar, set_substatus
from proto_tools.utils.proto_home import get_proto_home, show_first_run_notice
from proto_tools.utils.sequence import (
    calculate_gc_content,
    detect_sequence_type,
    resolve_sequence_ids,
    return_invalid_dna_chars,
    return_invalid_protein_chars,
    return_invalid_rna_chars,
)

# ── sequence.py ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sequence,expected", [("", 0.0), ("ATGC", 50.0)])
def test_calculate_gc_content(sequence, expected):
    assert calculate_gc_content(sequence) == pytest.approx(expected)


@pytest.mark.parametrize(
    "seqs,ids,expected",
    [
        (["A", "B"], ["x", "y"], ["x", "y"]),
        (["A", "B", "C"], None, ["seq_0", "seq_1", "seq_2"]),
    ],
    ids=["provided", "generated"],
)
def test_resolve_sequence_ids(seqs, ids, expected):
    assert resolve_sequence_ids(seqs, ids=ids) == expected


def test_resolve_sequence_ids_length_mismatch():
    with pytest.raises(ValueError, match="must match"):
        resolve_sequence_ids(["A"], ids=["x", "y"])


@pytest.mark.parametrize(
    "func,sequence,extra,expected",
    [
        (return_invalid_dna_chars, "ACGT", None, set()),
        (return_invalid_dna_chars, "ACGX", None, {"X"}),
        (return_invalid_dna_chars, "ACGTN", "N", set()),
        (return_invalid_rna_chars, "ACGU", None, set()),
        (return_invalid_protein_chars, "MVLSPADKTN", None, set()),
    ],
    ids=["dna-valid", "dna-invalid", "dna-extra-N", "rna-valid", "protein-valid"],
)
def test_return_invalid_chars(func, sequence, extra, expected):
    assert func(sequence, additional_valid_chars=extra) == expected


@pytest.mark.parametrize(
    "sequence,expected",
    [("ACGT", "dna"), ("ACGU", "rna"), ("MVLSPADKTN", "protein")],
)
def test_detect_sequence_type(sequence, expected):
    assert detect_sequence_type(sequence) == expected


# ── chemistry.py ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "smiles,expected",
    [("CCO", True), ("DEFINITELY_NOT_SMILES", False)],
    ids=["valid", "invalid"],
)
def test_validate_smiles(smiles, expected):
    assert validate_smiles(smiles, verbose=False) is expected


def test_validate_smiles_verbose_warns_on_invalid():
    with pytest.warns(UserWarning, match="could not parse SMILES"):
        validate_smiles("DEFINITELY_NOT_SMILES", verbose=True)


def test_validate_smiles_missing_rdkit(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _block_rdkit(name, *args, **kwargs):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_rdkit)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_smiles("CCO", verbose=True)

    assert result is False
    assert any("RDKit not installed" in str(w.message) for w in caught)


# ── msa.py ────────────────────────────────────────────────────────────────────


@dataclass
class _StubMSA:
    _entries: list[tuple[str, str]] = field(default_factory=list)

    @property
    def num_sequences(self) -> int:
        return len(self._entries)

    def iter_with_ids(self):
        yield from self._entries


_FIVE_SEQ_MSA = _StubMSA([("id_0", "AAAA"), ("id_1", "BBBB"), ("id_2", "CCCC"), ("id_3", "DDDD"), ("id_4", "EEEE")])


def test_extract_msa_no_swap():
    seqs, ids = extract_msa_sequences(_FIVE_SEQ_MSA, query_index=0)
    assert seqs == ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]
    assert ids == ["id_0", "id_1", "id_2", "id_3", "id_4"]


def test_extract_msa_with_swap():
    seqs, ids = extract_msa_sequences(_FIVE_SEQ_MSA, query_index=2)
    assert seqs[0] == "CCCC"
    assert ids[0] == "id_2"
    assert seqs[2] == "AAAA"
    assert ids[2] == "id_0"


@pytest.mark.parametrize("bad_index", [-1, 5])
def test_extract_msa_out_of_bounds(bad_index):
    with pytest.raises(IndexError, match="out of range"):
        extract_msa_sequences(_FIVE_SEQ_MSA, query_index=bad_index)


# ── proto_home.py ─────────────────────────────────────────────────────────────


@pytest.fixture()
def _clear_proto_home_cache():
    get_proto_home.cache_clear()
    yield
    get_proto_home.cache_clear()


def test_get_proto_home_from_env(monkeypatch, tmp_path, _clear_proto_home_cache):
    monkeypatch.setenv("PROTO_HOME", str(tmp_path / "custom"))
    assert get_proto_home() == (tmp_path / "custom").resolve()


def test_get_proto_home_default(monkeypatch, _clear_proto_home_cache):
    monkeypatch.delenv("PROTO_HOME", raising=False)
    assert get_proto_home() == Path.home() / ".proto"


def test_first_run_notice_sentinel_exists(monkeypatch, tmp_path, capsys, _clear_proto_home_cache):
    monkeypatch.setenv("PROTO_HOME", str(tmp_path))
    monkeypatch.delenv("PROTO_MODEL_CACHE", raising=False)
    (tmp_path / ".initialized").touch()
    show_first_run_notice()
    assert capsys.readouterr().err == ""


def test_first_run_notice_both_env_vars_silent(monkeypatch, tmp_path, capsys, _clear_proto_home_cache):
    monkeypatch.setenv("PROTO_HOME", str(tmp_path))
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path / "weights"))
    show_first_run_notice()
    assert (tmp_path / ".initialized").exists()
    assert capsys.readouterr().err == ""


def test_first_run_notice_shows_notice(monkeypatch, tmp_path, capsys, _clear_proto_home_cache):
    monkeypatch.setenv("PROTO_HOME", str(tmp_path))
    monkeypatch.delenv("PROTO_MODEL_CACHE", raising=False)
    show_first_run_notice()
    err = capsys.readouterr().err
    assert "first-run setup" in err
    assert "PROTO_MODEL_CACHE" in err
    assert (tmp_path / ".initialized").exists()


# ── progress.py ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("env_value,expected", [("1", True), ("true", True), ("yes", True), ("", False), ("0", False)])
def test_is_disabled(monkeypatch, env_value, expected):
    monkeypatch.setenv("PROTO_NO_SPINNER", env_value)
    assert _is_disabled() is expected


def test_progress_bar_disabled_returns_plain_tqdm(monkeypatch):
    monkeypatch.setenv("PROTO_NO_SPINNER", "1")
    bar = progress_bar(total=10, desc="test")
    assert type(bar) is tqdm
    bar.close()


def test_set_substatus_no_active_bar(caplog):
    with caplog.at_level("INFO", logger="proto_tools"):
        set_substatus("building environment")
    assert any("building environment" in r.message for r in caplog.records)


# ── auth.py ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("env_var", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
def test_resolve_hf_token_from_env(monkeypatch, env_var):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv(env_var, "hf_test_token_123")
    assert resolve_hf_token() == "hf_test_token_123"


def test_resolve_hf_token_none(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("proto_tools.utils.auth.os.path.isfile", lambda _: False)
    assert resolve_hf_token() is None


def test_require_hf_token_raises_when_missing(monkeypatch):
    monkeypatch.setattr("proto_tools.utils.auth.resolve_hf_token", lambda: None)
    with pytest.raises(OSError, match="requires a HuggingFace token"):
        require_hf_token("ESM3")

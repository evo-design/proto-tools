"""What each fingerprint hash is sensitive to, and what it deliberately ignores.

A hash that misses a real change reports "aligned" for a deployment that is not,
which is worse than no check. A hash that moves on an irrelevant change makes
every deployment look drifted at once, which reads the same as everything being
broken. Both directions are tested.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from proto_tools.modal import fingerprint as fp

TOOL = "esm2-embedding"
TOOLKIT = "esm2"


def _env_dir(toolkit: str) -> Path:
    from proto_tools.utils.tool_instance import ToolInstance

    env_dir, _name = ToolInstance._resolve_env_def(toolkit)
    return env_dir


def test_env_hash_covers_standalone_helpers():
    """Helpers hold real compute — scoring, sampling, seeding — so a change there changes results."""
    helpers = _env_dir(TOOLKIT) / "standalone_helpers"
    if not helpers.is_dir():
        pytest.skip(f"{TOOLKIT} ships no standalone_helpers/")

    hashed = {p.resolve() for p in _env_dir(TOOLKIT).rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert any(p.parent.name == "standalone_helpers" for p in hashed), (
        "standalone_helpers/ is excluded, so behaviour drift there would go unreported"
    )


def test_env_hash_ignores_pycache(tmp_path, monkeypatch):
    """Compiled bytecode is a local artefact; hashing it would drift on every import."""
    env = tmp_path / "standalone"
    (env / "__pycache__").mkdir(parents=True)
    (env / "setup.sh").write_text("echo hi\n")
    monkeypatch.setattr(fp, "_HASH_LENGTH", 12)
    monkeypatch.setattr("proto_tools.utils.tool_instance.ToolInstance._resolve_env_def", lambda _t: (env, "x"))

    before, _ = fp.env_hash("x")
    (env / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00\x01")
    after, _ = fp.env_hash("x")

    assert before == after


def test_env_hash_moves_when_a_helper_changes(tmp_path, monkeypatch):
    """The case the old non-recursive walk missed entirely."""
    env = tmp_path / "standalone"
    (env / "standalone_helpers").mkdir(parents=True)
    (env / "setup.sh").write_text("echo hi\n")
    helper = env / "standalone_helpers" / "scoring.py"
    helper.write_text("def score(): return 1\n")
    monkeypatch.setattr("proto_tools.utils.tool_instance.ToolInstance._resolve_env_def", lambda _t: (env, "x"))

    before, _ = fp.env_hash("x")
    helper.write_text("def score(): return 2\n")
    after, _ = fp.env_hash("x")

    assert before != after, "a helper edit changes results but left the fingerprint unmoved"


def test_env_hash_is_independent_of_tree_location(tmp_path, monkeypatch):
    """Hashing absolute paths would make two checkouts of the same code disagree."""
    digests = []
    for parent in ("a", "b"):
        env = tmp_path / parent / "standalone"
        (env / "standalone_helpers").mkdir(parents=True)
        (env / "setup.sh").write_text("echo hi\n")
        (env / "standalone_helpers" / "h.py").write_text("x = 1\n")
        monkeypatch.setattr("proto_tools.utils.tool_instance.ToolInstance._resolve_env_def", lambda _t, e=env: (e, "x"))
        digests.append(fp.env_hash("x")[0])

    assert digests[0] == digests[1]


def test_code_hash_finds_shared_bases_without_being_told():
    """A shared base can change behaviour without altering the emitted schema."""
    files = {p.name for p in fp._defining_files("esmfold-prediction")}

    assert "esmfold.py" in files
    assert "shared_data_models.py" in files, "an inherited base was missed, so edits to it are invisible"


def test_code_hash_excludes_framework_modules():
    """Including them would move every tool's fingerprint on one framework edit."""
    for key in (TOOL, "esmfold-prediction", "tmalign-alignment"):
        rels = [p.relative_to(fp._PACKAGE_ROOT).as_posix() for p in fp._defining_files(key)]
        assert not [r for r in rels if r.startswith(("proto_tools/utils/", "proto_tools/tools/tool_registry.py"))]


def test_code_hash_moves_when_a_defining_file_changes(tmp_path, monkeypatch):
    """The point of the hash: logic edited in place, schema and env untouched."""
    source = tmp_path / "tool.py"
    source.write_text("def run(): return 1\n")
    monkeypatch.setattr(fp, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(fp, "_defining_files", lambda _k: [source])

    before = fp.code_hash("anything")
    source.write_text("def run(): return 2\n")
    after = fp.code_hash("anything")

    assert before != after


def test_every_registered_tool_fingerprints():
    """A tool whose fingerprint raises would break the drift check for its whole service."""
    from proto_tools.tools import ToolRegistry

    failures = []
    for spec in ToolRegistry.list_all():
        try:
            fp.fingerprint(spec.key)
        except Exception as exc:
            failures.append(f"{spec.key}: {type(exc).__name__}: {exc}")
    assert not failures, "tools whose fingerprint raises:\n  " + "\n  ".join(failures[:10])


def test_hashes_are_independent():
    """Three separate questions; collapsing them would lose which kind of drift occurred."""
    one = fp.fingerprint(TOOL)

    assert len({one.schema_hash, one.code_hash, one.env_hash}) == 3
    assert all(len(h) == fp._HASH_LENGTH for h in (one.schema_hash, one.code_hash, one.env_hash))


def test_algorithm_is_recorded_so_a_rule_change_is_detectable():
    """Without it, redefining a hash makes every deployment report drift at once."""
    assert fp.fingerprint(TOOL).algorithm == fp.ALGORITHM


def test_digest_is_truncated_consistently():
    """The stored form must match what a comparison recomputes."""
    assert fp._digest("x") == hashlib.sha256(b"x").hexdigest()[: fp._HASH_LENGTH]

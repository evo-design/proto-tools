"""Tests for the per-toolkit model-cache layout."""

import os
import subprocess
from pathlib import Path

import pytest

from proto_tools.utils import weights_layout
from proto_tools.utils.persistent_worker import _build_subprocess_env
from proto_tools.utils.standalone_helpers_source.standalone_helpers import resolve_weights_dir
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.weights_layout import toolkit_hf_hub_cache, weights_key

_HELPERS_SH = Path(weights_layout.__file__).parent / "standalone_helpers_source" / "standalone_helpers.sh"


@pytest.fixture
def key_file(monkeypatch, tmp_path: Path):
    """Give ``mocktool`` a toolkit dir whose ``standalone/weights_key.txt`` holds the given text."""

    def _write(text: str) -> None:
        tool_dir = tmp_path / "tool"
        (tool_dir / "standalone").mkdir(parents=True)
        (tool_dir / "standalone" / "weights_key.txt").write_text(text)
        monkeypatch.setattr(ToolInstance, "_find_tool_dir", classmethod(lambda cls, name: tool_dir))

    return _write


# ── weights_key ──────────────────────────────────────────────────────────────


def test_weights_key_reads_shared_folder(key_file):
    """weights_key.txt points a toolkit at another toolkit's folder."""
    key_file("alphafold2\n")
    assert weights_key("mocktool") == "alphafold2"


def test_weights_key_empty_file_raises(key_file):
    """An empty weights_key.txt is a mistake, not a silent fallback to the toolkit name."""
    key_file("  \n")
    with pytest.raises(ValueError, match="empty"):
        weights_key("mocktool")


# ── subprocess env ───────────────────────────────────────────────────────────


def test_toolkit_caches_live_in_its_folder(monkeypatch, tmp_path: Path, key_file):
    """HF, torch, and JAX caches all resolve under {cache}/{weights_key}/, where borrowers find them too."""
    key_file("shared")
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("PROTO_JAX_COMPILATION_CACHE", raising=False)

    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path / "venv", toolkit="mocktool")

    folder = tmp_path / "shared"
    assert env["HF_HOME"] == str(folder / "huggingface")
    assert env["TORCH_HOME"] == str(folder / "torch")
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(folder / "jax_cache")
    assert toolkit_hf_hub_cache("mocktool") == str(folder / "huggingface" / "hub")


@pytest.mark.parametrize("mode", ["IN_ENV", "NONE"])
def test_in_env_and_none_modes_ignore_layout(monkeypatch, tmp_path: Path, mode):
    """IN_ENV and NONE modes keep their per-venv and pass-through caches."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", mode)
    monkeypatch.delenv("HF_HOME", raising=False)

    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path / "venv", toolkit="mocktool")

    expected_hf = str(tmp_path / "venv" / "cache" / "huggingface") if mode == "IN_ENV" else None
    assert env.get("HF_HOME") == expected_hf
    assert toolkit_hf_hub_cache("mocktool") is None


def test_user_jax_cache_beats_toolkit_layout(monkeypatch, tmp_path: Path):
    """A user-set JAX_COMPILATION_CACHE_DIR still wins over the toolkit folder."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/custom/jax")

    env = _build_subprocess_env(device="cpu", toolkit="mocktool")

    assert env["JAX_COMPILATION_CACHE_DIR"] == "/custom/jax"


# ── resolve_weights_dir ──────────────────────────────────────────────────────


def _shell_weights_dir(env: dict[str, str], name: str) -> str:
    script = f'source "{_HELPERS_SH}" && proto_resolve_weights_dir {name} && printf %s "$WEIGHTS_DIR"'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("explicit_cache", [True, False])
def test_resolve_weights_dir_toolkit_layout(monkeypatch, tmp_path: Path, explicit_cache):
    """Python and shell helpers both resolve {cache}/{name}/weights."""
    monkeypatch.setenv("PROTO_HOME", str(tmp_path / "home"))
    if explicit_cache:
        monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path / "cache"))
        cache = tmp_path / "cache"
    else:
        monkeypatch.delenv("PROTO_MODEL_CACHE", raising=False)
        cache = tmp_path / "home" / "proto_model_cache"
    monkeypatch.delenv("PROTO_FAMPNN_WEIGHTS_DIR", raising=False)

    expected = str(cache / "fampnn" / "weights")
    assert resolve_weights_dir("fampnn") == expected
    assert _shell_weights_dir(dict(os.environ), "fampnn") == expected


def test_resolve_weights_dir_override_ignores_layout(monkeypatch, tmp_path: Path):
    """PROTO_{TOOL}_WEIGHTS_DIR is used as given, with no weights/ appended."""
    monkeypatch.setenv("PROTO_FAMPNN_WEIGHTS_DIR", str(tmp_path / "mine"))

    assert resolve_weights_dir("fampnn") == str(tmp_path / "mine")
    assert _shell_weights_dir(dict(os.environ), "fampnn") == str(tmp_path / "mine")


# ── compile caches ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode, expected",
    [("{tmp}/cache", "{tmp}/cache/mocktool/compile_cache"), ("IN_ENV", "{tmp}/venv/compile_cache"), ("NONE", None)],
)
def test_compile_caches_follow_model_cache(monkeypatch, tmp_path: Path, mode, expected):
    """Triton, TorchInductor, CUDA, vLLM, ... caches go in the toolkit folder; NONE leaves them alone."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", mode.format(tmp=tmp_path))
    for var in weights_layout.COMPILE_CACHE_VARS:
        monkeypatch.delenv(var, raising=False)

    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path / "venv", toolkit="mocktool")

    for var, sub in weights_layout.COMPILE_CACHE_VARS.items():
        assert env.get(var) == (expected and f"{expected.format(tmp=tmp_path)}/{sub}")


def test_user_compile_cache_wins(monkeypatch, tmp_path: Path):
    """A compile-cache location the user set in their shell is passed through, in every mode."""
    monkeypatch.setenv("TRITON_CACHE_DIR", "/custom/triton")
    for mode in (str(tmp_path), "NONE"):
        monkeypatch.setenv("PROTO_MODEL_CACHE", mode)
        env = _build_subprocess_env(device="cpu", toolkit="mocktool")
        assert env["TRITON_CACHE_DIR"] == "/custom/triton"


@pytest.mark.parametrize("mode", ["{tmp}/cache", "IN_ENV", "NONE"])
def test_general_compile_caches_are_shared_under_proto_home(monkeypatch, tmp_path: Path, mode):
    """Driver-level caches (CUDA JIT) are one per-user folder, not per toolkit; NONE leaves them alone."""
    monkeypatch.setenv("PROTO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PROTO_MODEL_CACHE", mode.format(tmp=tmp_path))
    for var in weights_layout.GENERAL_COMPILE_CACHE_VARS:
        monkeypatch.delenv(var, raising=False)

    envs = [_build_subprocess_env(device="cpu", tool_env_path=tmp_path / "venv", toolkit=tk) for tk in ("a", "b")]

    for var, sub in weights_layout.GENERAL_COMPILE_CACHE_VARS.items():
        expected = None if mode == "NONE" else str((tmp_path / "home").resolve() / "compile_cache" / sub)
        assert envs[0].get(var) == envs[1].get(var) == expected

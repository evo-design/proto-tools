"""Tests for run_in_env: executing code inside a tool's isolated environment."""

import json
from pathlib import Path

import pytest

from proto_tools.utils import run_in_env


def test_requires_exactly_one_of_code_or_script() -> None:
    """Passing both or neither of code/script raises before any env is touched."""
    with pytest.raises(ValueError, match="exactly one"):
        run_in_env("esm2", code="print(1)", script="x.py")
    with pytest.raises(ValueError, match="exactly one"):
        run_in_env("esm2")


@pytest.mark.integration
def test_nonzero_exit_raises() -> None:
    """A non-zero exit in the tool env surfaces as a RuntimeError with stderr."""
    with pytest.raises(RuntimeError, match="failed"):
        run_in_env("esm2", code="import sys; sys.exit(3)")


@pytest.mark.integration
def test_logs_demux_to_parent_and_stdout_stays_clean(caplog: pytest.LogCaptureFixture) -> None:
    """In-env proto-logging re-emits on the parent logger; stdout stays pure data."""
    code = (
        "import json\n"
        "from standalone_helpers import get_logger\n"
        "get_logger('rie').info('hello from env')\n"
        "print(json.dumps({'ok': True}))\n"
    )
    with caplog.at_level("INFO", logger="proto_tools.worker.esm2"):
        out = run_in_env("esm2", code=code)
    assert json.loads(out) == {"ok": True}
    assert any("hello from env" in r.getMessage() for r in caplog.records)


@pytest.mark.integration
def test_runs_script_file_in_env(tmp_path: Path) -> None:
    """The script= path runs a file, forwards args as sys.argv[1:], and exports RUN_IN_ENV_DEVICE."""
    script = tmp_path / "prog.py"
    script.write_text(
        "import json, os, sys; print(json.dumps({'argv': sys.argv[1:], 'device': os.environ['RUN_IN_ENV_DEVICE']}))"
    )
    out = run_in_env("esm2", script=script, args=["a", "b"], device="cpu")
    assert json.loads(out) == {"argv": ["a", "b"], "device": "cpu"}


@pytest.mark.integration
def test_timeout_raises() -> None:
    """A run that exceeds timeout is killed and surfaces as TimeoutError."""
    with pytest.raises(TimeoutError, match="timed out"):
        run_in_env("esm2", code="import time; time.sleep(30)", timeout=3)


@pytest.mark.integration
def test_jax_compilation_cache_reaches_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The tool env's JAX reads the parent-chosen cache directory and writes compiled programs to it."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("PROTO_JAX_COMPILATION_CACHE", raising=False)
    code = (
        "import jax, jax.numpy as jnp\n"
        "jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)\n"
        "jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)\n"
        "jax.jit(lambda x: jnp.sin(x) @ x.T)(jnp.ones((8, 8))).block_until_ready()\n"
        "print(jax.config.jax_compilation_cache_dir)\n"
    )

    out = run_in_env("mock_jax_tool", code=code)

    cache_dir = tmp_path / "mock_jax_tool" / "jax_cache"
    assert out.strip() == str(cache_dir)
    assert any(cache_dir.iterdir())


_VISIBILITY = (
    "import json, os\n"
    "print(json.dumps({k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'RUN_IN_ENV_DEVICE')}))\n"
)


@pytest.mark.integration
def test_cpu_run_sees_no_gpu() -> None:
    """A cpu run hides every GPU, so JAX initializes on CPU instead of claiming GPUs a worker may hold."""
    code = _VISIBILITY + "import jax\nprint(jax.default_backend())\n"
    visibility, backend = run_in_env("mock_jax_tool", code=code).strip().splitlines()[-2:]
    assert json.loads(visibility) == {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "RUN_IN_ENV_DEVICE": "cpu"}
    assert backend == "cpu"


@pytest.mark.uses_gpu
def test_cuda_run_sees_only_its_leased_gpu() -> None:
    """A cuda run sees one GPU, renumbered cuda:0, so bare "cuda" lands on the lease."""
    code = _VISIBILITY + "import torch\nprint(torch.cuda.device_count())\n"
    visibility, count = run_in_env("esm2", code=code, device="cuda").strip().splitlines()[-2:]
    assert json.loads(visibility)["RUN_IN_ENV_DEVICE"] == "cuda:0"
    assert count == "1"


@pytest.mark.uses_gpu(2)
def test_cuda_run_lands_on_the_free_gpu_when_another_is_leased() -> None:
    """With cuda:0 leased elsewhere, the run gets the other physical GPU, still seen as its cuda:0."""
    from proto_tools.utils.device import determine_visible_devices
    from proto_tools.utils.device_manager import DeviceManager

    code = _VISIBILITY + "import torch\nprint(torch.cuda.device_count())\n"
    with DeviceManager.get_instance().lease("other_tool", device="cuda:0"):
        visibility, count = run_in_env("esm2", code=code, device="cuda").strip().splitlines()[-2:]
    env = json.loads(visibility)
    assert env["CUDA_VISIBLE_DEVICES"] == determine_visible_devices("cuda:1")
    assert env["RUN_IN_ENV_DEVICE"] == "cuda:0" and count == "1"


@pytest.mark.uses_gpu(2)
@pytest.mark.parametrize("device", ["cudax2", "cuda:1,0"])
def test_multi_gpu_run_sees_its_leased_gpus_renumbered(device: str) -> None:
    """A multi-GPU lease is visible as cuda:0..N-1, in the order it was leased."""
    from proto_tools.utils.device import determine_visible_devices

    code = _VISIBILITY + "import torch\nprint(torch.cuda.device_count())\n"
    visibility, count = run_in_env("esm2", code=code, device=device).strip().splitlines()[-2:]
    env = json.loads(visibility)
    assert env["RUN_IN_ENV_DEVICE"] == "cuda:0,cuda:1" and count == "2"
    if device == "cuda:1,0":
        assert env["CUDA_VISIBLE_DEVICES"] == determine_visible_devices("cuda:1,0")  # leased order kept

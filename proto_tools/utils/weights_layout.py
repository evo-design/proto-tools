"""proto_tools/utils/weights_layout.py.

Per-toolkit layout of the model cache.

Every framework cache a toolkit writes to lands in one folder named by its weights key::

    {PROTO_MODEL_CACHE}/{key}/
    ├── huggingface/   HF_HOME
    ├── torch/         TORCH_HOME
    ├── jax_cache/     JAX_COMPILATION_CACHE_DIR
    ├── compile_cache/ Triton, TorchInductor, vLLM, ... (COMPILE_CACHE_VARS)
    └── weights/       resolve_weights_dir()

The weights key is the toolkit name unless ``standalone/weights_key.txt`` names another
toolkit's folder, so toolkits that load the same checkpoints share one copy.

``IN_ENV`` and ``NONE`` modes keep caches in the tool's venv or the caller's environment instead.
"""

import os
from pathlib import Path

# JIT and compile caches that libraries keep in ~ or /tmp by default. Every entry is a cache:
# deleting it only costs a recompile.
#
# Toolkit-specific: artifacts compiled from the toolkit's own model code against its env's library
# versions, so they go in the toolkit's compile_cache/. Extensions are keyed by name, so sharing
# them across envs with different torch versions would load the wrong build.
COMPILE_CACHE_VARS: dict[str, str] = {
    "TRITON_CACHE_DIR": "triton",  # ~/.triton/cache
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",  # /tmp/torchinductor_$USER
    "TORCH_EXTENSIONS_DIR": "torch_extensions",  # ~/.cache/torch_extensions
    "VLLM_CACHE_ROOT": "vllm",  # ~/.cache/vllm
    "HUMMING_CACHE_DIR": "humming",  # ~/.humming/cache (humming-kernels JIT, used by evo2's vLLM)
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",  # ~/.cache/flashinfer
    "CUPY_CACHE_DIR": "cupy",  # ~/.cupy/kernel_cache
    "NUMBA_CACHE_DIR": "numba",  # __pycache__ beside the source
}

# General: depend on the GPU driver and generic kernels, not on any model, so one per-user copy
# under PROTO_HOME/compile_cache/ serves every toolkit (not PROTO_MODEL_CACHE, which teams share).
GENERAL_COMPILE_CACHE_VARS: dict[str, str] = {
    "CUDA_CACHE_PATH": "cuda",  # ~/.nv/ComputeCache: the driver's JIT cache for any CUDA code
    "PYTORCH_KERNEL_CACHE_PATH": "torch_kernels",  # ~/.cache/torch/kernels
}


def weights_key(toolkit: str) -> str:
    """Return the model-cache folder name for ``toolkit``.

    Args:
        toolkit (str): Toolkit name.

    Returns:
        str: The name in ``standalone/weights_key.txt``, or ``toolkit`` when there is none.

    Raises:
        ValueError: If ``weights_key.txt`` exists but is empty.
    """
    # Lazy import to avoid circular dep: tool_instance imports persistent_worker, which imports this
    from proto_tools.utils.tool_instance import ToolInstance

    tool_dir = ToolInstance._find_tool_dir(toolkit)
    marker = tool_dir / "standalone" / "weights_key.txt" if tool_dir is not None else None
    if marker is None or not marker.is_file():
        return toolkit
    key = marker.read_text().strip()
    if not key:
        raise ValueError(f"weights_key.txt for toolkit {toolkit!r} is empty ({marker})")
    return key


def compile_cache_env(base: Path, variables: dict[str, str] = COMPILE_CACHE_VARS) -> dict[str, str]:
    """Return compile-cache variables pointed at subfolders of ``base``.

    Args:
        base (Path): A ``compile_cache/`` folder (the toolkit's, or ``PROTO_HOME``'s for general ones).
        variables (dict[str, str]): Variable to subfolder name.

    Returns:
        dict[str, str]: Environment variable to directory.
    """
    return {var: str(base / sub) for var, sub in variables.items()}


def toolkit_cache_env(toolkit: str, cache_root: Path) -> dict[str, str]:
    """Return the env vars that point the HF and torch caches at ``toolkit``'s folder.

    The JAX compilation cache is resolved separately because a user-set value and
    ``PROTO_JAX_COMPILATION_CACHE=0`` both take precedence over it.

    Args:
        toolkit (str): Toolkit name.
        cache_root (Path): The model cache root (``PROTO_MODEL_CACHE`` path mode).

    Returns:
        dict[str, str]: ``HF_HOME`` and ``TORCH_HOME``.
    """
    folder = cache_root / weights_key(toolkit)
    return {
        "HF_HOME": str(folder / "huggingface"),
        "TORCH_HOME": str(folder / "torch"),
    }


def toolkit_hf_hub_cache(toolkit: str) -> str | None:
    """Return ``toolkit``'s Hugging Face hub cache, for a worker that loads one of its checkpoints.

    A toolkit that needs another toolkit's model passes this to its worker, which downloads into
    and reads from that folder, so both share one copy.

    Args:
        toolkit (str): The toolkit whose model is borrowed.

    Returns:
        str | None: ``{PROTO_MODEL_CACHE}/{key}/huggingface/hub``, or None in ``IN_ENV`` and
            ``NONE`` modes, where caches are not kept per toolkit and the worker's own cache applies.
    """
    from proto_tools.utils.proto_home import get_proto_home

    cache_mode = os.environ.get("PROTO_MODEL_CACHE") or str(get_proto_home() / "proto_model_cache")
    if cache_mode in ("IN_ENV", "NONE"):
        return None
    return str(Path(toolkit_cache_env(toolkit, Path(cache_mode))["HF_HOME"]) / "hub")

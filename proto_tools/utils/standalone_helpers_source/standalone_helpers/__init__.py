"""proto_tools/utils/standalone_helpers_source/standalone_helpers/__init__.py.

Shared helpers copied to each tool's standalone/ directory at runtime.

This package provides common utilities that standalone scripts need but cannot
import from the main ``proto_tools`` package (due to environment isolation).

DO NOT MODIFY THESE FILES INSIDE STANDALONE FOLDERS. CHANGES WILL BE OVERWRITTEN.
If you need to make changes, modify the source files which are located at
``proto-tools/proto_tools/utils/standalone_helpers_source/standalone_helpers/``.

Inside tool standalone directories, helpers can be imported either via the
package entry point::

    from standalone_helpers import get_subprocess_device_env, set_torch_seed
    from standalone_helpers import get_logger

or from specific submodules::

    from standalone_helpers.seeding import set_torch_seed
    from standalone_helpers.compression import compress_array

This file and the submodules are copied by the worker bootstrap. The source
tree is tracked by git, but the copies in ``tools/*/standalone/`` are not.
"""

import os

# Bridge install must happen before any submodule's module-level get_logger call so their loggers are ProtoLogger instances.
from .proto_logging import get_logger, install

# Gated on TOOL_VENV_PATH so parent-side imports of this package don't attach the bridge to the parent's logger tree.
if os.environ.get("TOOL_VENV_PATH"):
    install()

from .compression import (
    _COMPRESS_MIN_SIZE,
    _COMPRESSED_ARRAY_SENTINEL,
    compress_array,
    is_compressed_array,
)
from .device import (
    _apply_jax_subprocess_env,
    _parse_cuda_indices,
    get_subprocess_device_env,
    move_model_to_device,
    resolve_jax_device,
)
from .memory import get_jax_memory_stats, get_pytorch_memory_stats
from .oom import GpuOutOfMemoryError, is_cuda_oom, oom_guard, raise_oom, release_cuda_memory

# ``iterative_sampling`` is not re-exported: it imports torch at module level, so keeping it off the package init lets torch-less envs (CI) import this package. Standalones import the submodule directly.
from .scoring import log_likelihood_metrics
from .seeding import (
    enable_jax_compilation_cache,
    get_random_int,
    set_jax_seed,
    set_torch_seed,
)
from .serialization import (
    AMINO_ACIDS_LIST,
    DNA_NUCLEOTIDES,
    RNA_NUCLEOTIDES,
    serialize_output,
)
from .subprocess_utils import run_teed
from .weights import resolve_weights_dir

__all__ = [
    # logging
    "get_logger",
    # compression
    "_COMPRESS_MIN_SIZE",
    "_COMPRESSED_ARRAY_SENTINEL",
    "compress_array",
    "is_compressed_array",
    # device
    "_apply_jax_subprocess_env",
    "_parse_cuda_indices",
    "get_subprocess_device_env",
    "move_model_to_device",
    "resolve_jax_device",
    # memory
    "get_jax_memory_stats",
    "get_pytorch_memory_stats",
    # oom
    "GpuOutOfMemoryError",
    "is_cuda_oom",
    "oom_guard",
    "raise_oom",
    "release_cuda_memory",
    # scoring
    "log_likelihood_metrics",
    # seeding
    "enable_jax_compilation_cache",
    "get_random_int",
    "set_jax_seed",
    "set_torch_seed",
    # serialization
    "AMINO_ACIDS_LIST",
    "DNA_NUCLEOTIDES",
    "RNA_NUCLEOTIDES",
    "serialize_output",
    # subprocess
    "run_teed",
    # weights
    "resolve_weights_dir",
]

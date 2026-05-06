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

or from specific submodules::

    from standalone_helpers.seeding import set_torch_seed
    from standalone_helpers.compression import compress_array

This file and the submodules are copied by the worker bootstrap. The source
tree is tracked by git, but the copies in ``tools/*/standalone/`` are not.
"""

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

# ``iterative_sampling`` imports torch at module level and is NOT re-exported
# from the package init, so CI's slim test env (no torch) can import this
# package. Standalones import the submodule directly.
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
from .weights import resolve_weights_dir

__all__ = [
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
    # weights
    "resolve_weights_dir",
]

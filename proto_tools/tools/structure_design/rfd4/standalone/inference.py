"""RFD4-Proteina inference implementation.

Generation runs in-process so a persistent worker can hold the checkpoint across calls.
There is deliberately no module-level ``to_device()``: both tools register
``pin_visible_devices=True`` because upstream's Lightning trainer claims local
``cuda:0..N-1`` and cannot be retargeted at a logical index from inside the process, so
DeviceManager respawns a pinned worker on a device change instead of moving the model.

References:
    https://github.com/RosettaCommons/RFD4-Proteina
    https://rfd4-proteina.org/latest/
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)

# Both tools run the same model; the difference is already in the Hydra overrides.
_VALID_OPERATIONS = ("design", "inverse_folding")

# Matches the model_checkpoint default; only reached if an override omits hf_ckpt_path.
_DEFAULT_CHECKPOINT = "nvidia/RFD4-Proteina"

# Upstream layout: <out_dir>/inference_samples/<example_id>/<example_id>_generated.cif
_SAMPLES_SUBDIR = "inference_samples"
_METRICS_SUBDIR = "inference_metrics"
_GENERATED_STEM_SUFFIX = "_generated"

# compress_outputs is left off, but harvest zstd too so an override still yields structures.
_STRUCTURE_SUFFIXES = (".cif", ".cif.zst")


class RFD4ExecutionError(Exception):
    """Raised when the ``rfd4-generate`` CLI exits non-zero for a non-OOM reason."""


def _drop_ephemeral_paths(row: dict[str, Any]) -> dict[str, Any]:
    """Strip run-local filesystem paths out of a metrics row before it becomes output metadata.

    Upstream's ``per_example_metrics.csv`` records absolute paths for the run it came from
    (``designed_structure_path``, ``dataset_name``, ...). The implementation runs in a fresh
    temporary directory every call, so those values differ between two otherwise identical
    runs: they would leak host paths to callers and defeat output comparison and seeded
    reproducibility. The design's content is returned directly, so the paths carry nothing.

    Args:
        row (dict[str, Any]): One metrics row.

    Returns:
        dict[str, Any]: The row without path-valued entries.
    """
    return {
        key: value
        for key, value in row.items()
        if not key.endswith(("_path", "_dir")) and not (isinstance(value, str) and value.startswith("/"))
    }


def _format_override(key: str, value: Any) -> str:
    """Render one Hydra ``key=value`` override.

    ``rfd4-generate`` is a Hydra app, so every argument is a ``key=value`` token and
    never a ``--flag``. Lists and dicts are serialized as JSON, which is also valid
    Hydra override syntax (``metrics_tags=[]``), and bools use Hydra's literal form.

    A key prefixed with ``~`` is Hydra's config-group *deletion* syntax and renders as a
    bare token with no value. Removing a group default is the only way to switch one off:
    a config group takes a group name, so ``recycle_strategy=null`` fails composition with
    "Config group override must be a string or a list. Got NoneType".

    Args:
        key (str): Override key, e.g. ``generation.args.nsteps``, or ``~<group>``.
        value (Any): Value to render. Ignored for a ``~`` deletion key.

    Returns:
        str: The ``key=value`` token, or a bare ``~key`` for a deletion.
    """
    if key.startswith("~"):
        return key
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, (dict, list)):
        rendered = json.dumps(value, separators=(",", ":"))
    else:
        rendered = str(value)
    return f"{key}={rendered}"


def _strip_structure_suffix(name: str) -> str:
    """Strip a structure suffix, keeping any dots the example id itself carries.

    Args:
        name (str): Structure file name.

    Returns:
        str: The file name without its ``.cif`` / ``.cif.zst`` suffix.
    """
    for suffix in _STRUCTURE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _read_structure(path: Path) -> str:
    """Read a generated structure, transparently decompressing ``.cif.zst``.

    Args:
        path (Path): Path to the ``.cif`` or ``.cif.zst`` file.

    Returns:
        str: The mmCIF text.
    """
    if path.name.endswith(".zst"):
        import zstandard

        with path.open("rb") as handle:
            return str(zstandard.ZstdDecompressor().stream_reader(handle).read().decode())
    return path.read_text()


def _split_example_id(example_id: str, spec_keys: list[str]) -> tuple[str, int]:
    """Map an upstream example id back to its spec key and replicate index.

    ``ConditionSpecDataset`` renames every example to ``f"{spec_key}_{i}"`` — even at
    ``num_replicates=1`` — so the replicate index is always a trailing ``_<int>``.
    Matching takes the longest spec key that prefixes the id and leaves an all-digit
    remainder, because a caller's own ``raw_json`` key may itself end in ``_3``.

    Args:
        example_id (str): Example id upstream wrote outputs under.
        spec_keys (list[str]): Spec ids that were dispatched, in input order.

    Returns:
        tuple[str, int]: The originating spec key and the 0-based replicate index.

    Raises:
        RFD4ExecutionError: If no dispatched spec key explains the example id.
    """
    candidates = [key for key in spec_keys if example_id.startswith(f"{key}_") and example_id[len(key) + 1 :].isdigit()]
    if not candidates:
        raise RFD4ExecutionError(
            f"rfd4: output example id {example_id!r} does not match any dispatched spec key {spec_keys}; "
            "expected '<spec_key>_<replicate_index>'"
        )
    spec_key = max(candidates, key=len)
    return spec_key, int(example_id[len(spec_key) + 1 :])


def _read_per_example_metrics(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Read upstream's per-example metrics table when one was written.

    The implementation always passes ``metrics_tags=[]``, which gates only *tagged* metrics
    -- the oracle-scored set. Untagged metrics such as ``unknown_residue_filter``
    still run, so a table with a few composition columns is the normal case. The
    read stays best-effort because a run may write none at all.

    Args:
        output_dir (Path): Run output directory passed as ``out_dir``.

    Returns:
        dict[str, dict[str, Any]]: Metrics rows keyed by example id, empty when absent.
    """
    rows: dict[str, dict[str, Any]] = {}
    for csv_path in sorted((output_dir / _METRICS_SUBDIR).glob("*per_example_metrics*.csv")):
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                example_id = row.get("example_id")
                if example_id:
                    rows[example_id] = row
    return rows


class RFD4Model:
    """RFD4-Proteina generation with the checkpoint held in memory across calls.

    ``rfd4-generate`` is a thin Hydra wrapper over :func:`rfproteina.generate.run_generation`,
    which accepts an already-loaded model. Calling it in-process instead of shelling out to
    the console script is what lets a persistent worker pay the checkpoint load once.

    References:
        https://github.com/RosettaCommons/RFD4-Proteina
        https://rfd4-proteina.org/latest/
    """

    def __init__(self) -> None:
        """Initialize RFD4Model."""
        self._loaded = False
        self.device: str | None = None
        self.model: Any = None
        self._config_dir: str | None = None

    def load(
        self,
        model_checkpoint: str,
        model_revision: str | None,
        adapter_path: str | None,
        device: str,
        verbose: bool = False,
    ) -> None:
        """Load the RFD4-Proteina checkpoint and hold it for subsequent calls.

        The adapter is applied here rather than per call: upstream's
        ``load_ckpt_n_configure_inference`` re-applies ``adapter_path`` every time it is
        given one, which would stack LoRA layers onto a model that is being reused.
        ``adapter_path`` is ``reload_on_change``, so a different adapter respawns the worker.

        Args:
            model_checkpoint (str): HuggingFace repo id of the checkpoint bundle.
            model_revision (str | None): Pinned HuggingFace revision, or None for the default.
            adapter_path (str | None): Local PEFT/LoRA adapter directory, or None.
            device (str): Device to hold the model on (e.g. ``"cuda"``, ``"cpu"``).
            verbose (bool): Log the checkpoint being loaded.

        Raises:
            ImportError: If the rfd4-proteina package is not importable.
        """
        logger.debug("Initializing RFD4-Proteina")

        try:
            import torch
            from rfproteina.proteina import Proteina
        except ImportError as exc:
            raise ImportError("rfd4: the rfd4-proteina package is not importable; re-run setup.sh") from exc

        if verbose:
            logger.info(f"Loading RFD4-Proteina checkpoint {model_checkpoint} (revision={model_revision})")

        model = Proteina.from_pretrained(model_checkpoint, revision=model_revision, nn_overrides={})
        if adapter_path:
            from rfproteina.utils.lora import load_lora_adapter

            load_lora_adapter(model, adapter_path)

        self.model = model.to(torch.device(device)).eval()
        self.device = device
        # generate.py reads PROJECT_ROOT at import time; rfproteina.utils.env setdefault's it.
        self._config_dir = os.path.join(os.environ["PROJECT_ROOT"], "configs")
        self._loaded = True

        logger.debug(f"RFD4-Proteina initialized on {device}.")

    def _compose(self, spec_path: str, output_dir: str, seed: int | None, cli_kwargs: dict[str, Any]) -> Any:
        """Compose the inference config the ``rfd4-generate`` CLI would have composed.

        A fresh config is built per call because ``run_generation`` mutates what it is
        given (``cfg.seed``, ``cfg.out_dir``). ``adapter_path`` is dropped: the adapter is
        already on the held model, and passing it again would re-apply it.

        Args:
            spec_path (str): Path to the condition-spec YAML/JSON.
            output_dir (str): Directory passed to upstream as ``out_dir``.
            seed (int | None): Base seed; None lets upstream draw its own.
            cli_kwargs (dict[str, Any]): Hydra overrides as ``key: value`` pairs.

        Returns:
            Any: The composed Hydra ``DictConfig``.
        """
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        overrides = [f"inputs={spec_path}", f"out_dir={output_dir}"]
        if seed is not None:
            overrides.append(f"seed={seed}")
        overrides += [
            _format_override(key, value)
            for key, value in cli_kwargs.items()
            if value is not None and key != "adapter_path"
        ]

        logger.debug(f"Composing RFD4-Proteina config with overrides: {overrides}")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(self._config_dir), version_base="1.3"):
            return compose(config_name="inference", overrides=overrides)

    def unload(self) -> None:
        """Unload the model to free GPU memory."""
        import gc

        if not self._loaded:
            return

        self.model = None
        self._loaded = False
        self.device = None

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __call__(
        self,
        spec_path: str,
        output_dir: str,
        spec_keys: list[str],
        device: str = "cuda",
        seed: int | None = None,
        verbose: bool = False,
        **cli_kwargs: Any,
    ) -> dict[str, Any]:
        """Run RFD4-Proteina generation and harvest the designs it wrote.

        Args:
            spec_path (str): Path to the condition-spec YAML/JSON written by the implementation.
            output_dir (str): Directory passed to upstream as ``out_dir``.
            spec_keys (list[str]): Spec ids that were dispatched, in input order.
            device (str): Device string (e.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"``).
            seed (int | None): Base seed; ``None`` lets upstream draw its own.
            verbose (bool): Log the composed overrides and the checkpoint being loaded.
            **cli_kwargs (Any): Hydra overrides forwarded verbatim as ``key=value``.

        Returns:
            dict[str, Any]: ``{"designs": [{"structure_content", "spec_key",
                "design_index", "metadata"}, ...]}``.

        Raises:
            RFD4ExecutionError: If generation failed, or produced no designs.
            GpuOutOfMemoryError: If generation failed with a CUDA out-of-memory error.
        """
        # Seeds the Python-side RNG and no-ops on None; the Hydra override is what drives generation.
        from standalone_helpers import set_torch_seed

        set_torch_seed(seed)

        if not self._loaded:
            revision = cli_kwargs.get("hf_ckpt_revision")
            self.load(
                model_checkpoint=cli_kwargs.get("hf_ckpt_path", _DEFAULT_CHECKPOINT),
                # get_cli_kwargs renders an unpinned revision as Hydra's literal "null".
                model_revision=None if revision in (None, "null") else str(revision),
                adapter_path=cli_kwargs.get("adapter_path"),
                device=device,
                verbose=verbose,
            )

        from standalone_helpers import is_cuda_oom, raise_oom

        cfg = self._compose(spec_path, output_dir, seed, cli_kwargs)

        from rfproteina.generate import run_generation

        try:
            run_generation(cfg, job_id=0, model=self.model)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if is_cuda_oom(detail):
                raise_oom(
                    "rfd4",
                    hint="Lower batch_size, shorten the contig, or use a GPU with more memory.",
                )
            raise RFD4ExecutionError(f"rfd4: generation failed: {detail}") from exc

        logger.debug("RFD4-Proteina execution completed successfully.")
        return {"designs": self._collect_designs(Path(output_dir), spec_keys, seed, cli_kwargs)}

    def _collect_designs(
        self,
        output_dir: Path,
        spec_keys: list[str],
        seed: int | None,
        cli_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Collect the generated structures, grouped back onto their spec keys.

        RFD4 writes no output JSON and no run manifest, so the designs are discovered
        from the ``inference_samples/<example_id>/`` tree.

        Args:
            output_dir (Path): Run output directory passed as ``out_dir``.
            spec_keys (list[str]): Spec ids that were dispatched, in input order.
            seed (int | None): Base seed handed to upstream, recorded in metadata.
            cli_kwargs (dict[str, Any]): Hydra overrides, read for checkpoint identity.

        Returns:
            list[dict[str, Any]]: One entry per design, ordered by spec then replicate.

        Raises:
            RFD4ExecutionError: If no generated structure was written.
        """
        samples_dir = output_dir / _SAMPLES_SUBDIR
        structure_paths = [
            path
            for path in samples_dir.glob("*/*")
            if path.is_file()
            and path.name.endswith(_STRUCTURE_SUFFIXES)
            and _strip_structure_suffix(path.name).endswith(_GENERATED_STEM_SUFFIX)
        ]
        if not structure_paths:
            raise RFD4ExecutionError(
                f"rfd4: no '*_generated.cif' outputs found under {samples_dir}; the run exited cleanly "
                "but produced no designs (upstream also exits early when a results CSV for this run "
                "already exists next to out_dir)"
            )

        metrics_rows = _read_per_example_metrics(output_dir)
        spec_order = {key: index for index, key in enumerate(spec_keys)}
        # An unpinned revision reaches Hydra as the literal "null"; report it as absent.
        revision = cli_kwargs.get("hf_ckpt_revision")
        revision = None if revision == "null" else revision

        designs: list[dict[str, Any]] = []
        for path in structure_paths:
            example_id = path.parent.name
            spec_key, design_index = _split_example_id(example_id, spec_keys)
            metadata: dict[str, Any] = _drop_ephemeral_paths(metrics_rows.get(example_id, {}))
            metadata.update(
                {
                    "example_id": example_id,
                    "design_index": design_index,
                    "checkpoint": cli_kwargs.get("hf_ckpt_path"),
                    "checkpoint_revision": revision,
                }
            )
            # Upstream records the seed it drew, so never overwrite it with the caller's None.
            if seed is not None or "seed" not in metadata:
                metadata["seed"] = seed if seed is not None else metadata.get("seed")
            designs.append(
                {
                    "structure_content": _read_structure(path),
                    "spec_key": spec_key,
                    "design_index": design_index,
                    "metadata": metadata,
                }
            )

        designs.sort(key=lambda design: (spec_order[design["spec_key"]], int(design["design_index"])))

        # Upstream can drop a design and still exit 0, so never return a short result silently.
        expected = len(spec_keys) * int(cli_kwargs.get("num_replicates", 1))
        if len(designs) < expected:
            logger.warning(
                f"rfd4: expected {expected} design(s) ({len(spec_keys)} spec(s) x "
                f"{cli_kwargs.get('num_replicates', 1)} replicate(s)) but harvested {len(designs)}; "
                "upstream dropped the rest (a rejected design or a failed structure write) and still "
                "exited cleanly."
            )
        logger.debug(f"Collected {len(designs)} RFD4-Proteina design(s) from {samples_dir}")
        return designs


# ============================================================================
# Dispatch
# ============================================================================
_model: RFD4Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution.

    Args:
        input_dict (dict[str, Any]): Input parameters from ``ToolInstance.dispatch()``.

    Returns:
        dict[str, Any]: Dict with a ``designs`` list.

    Raises:
        ValueError: If the requested operation is unknown.
    """
    global _model
    if _model is None:
        _model = RFD4Model()

    kwargs = dict(input_dict)
    operation = kwargs.pop("operation")
    if operation not in _VALID_OPERATIONS:
        raise ValueError(f"rfd4: unknown operation {operation!r}; valid: {list(_VALID_OPERATIONS)}")

    return _model(
        spec_path=kwargs.pop("spec_path"),
        output_dir=kwargs.pop("output_dir"),
        spec_keys=kwargs.pop("spec_keys"),
        device=kwargs.pop("device"),
        seed=kwargs.pop("seed"),
        verbose=kwargs.pop("verbose"),
        **kwargs,
    )


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(device=0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("rfd4: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)

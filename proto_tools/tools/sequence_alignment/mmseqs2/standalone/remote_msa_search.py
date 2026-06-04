"""remote_msa_search.py.

ColabFold remote MSA search standalone script for isolated venv execution.

This script runs ColabFold's remote MMseqs2 API in an isolated environment
to avoid dependency conflicts with the main proto-language environment.
"""

import json
import random
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)

# Bounded retry: transient api.colabfold.com failures otherwise hard-fail predictions (#1027).
_MAX_ATTEMPTS = 3  # 1 initial try + 2 retries
_BASE_DELAY_SECONDS = 2.0  # doubles each retry: ~2s, ~4s
_MAX_DELAY_SECONDS = 30.0  # per-sleep cap
_MAX_JITTER_SECONDS = 1.0  # added per sleep to de-correlate concurrent jobs


def _run_with_retry(fn: Callable[..., Any], seq_id: str, *args: Any, **kwargs: Any) -> None:
    """Call ``fn(*args, **kwargs)`` with bounded backoff + jitter, retrying on failure.

    ColabFold reports transient and permanent failures with one opaque message, so we
    can't classify; the bounded cap keeps retrying genuinely-bad input cheap.
    """
    for attempt in range(_MAX_ATTEMPTS):
        try:
            fn(*args, **kwargs)
            return
        except Exception as exc:  # noqa: PERF203 -- retry loop
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            backoff = min(_BASE_DELAY_SECONDS * 2**attempt, _MAX_DELAY_SECONDS)
            jitter = random.uniform(0, _MAX_JITTER_SECONDS)
            delay = backoff + jitter
            logger.warning(
                f"{seq_id}: remote MSA error on attempt {attempt + 1}/{_MAX_ATTEMPTS}, retrying in {delay:.1f}s: {exc}"
            )
            time.sleep(delay)


class ColabFoldRemoteSearchWrapper:
    """Wrapper for ColabFold remote MSA search using the ColabFold API."""

    def __init__(self) -> None:
        """Initialize ColabFold remote search wrapper."""
        self._loaded = False
        self.run_mmseqs2: Any = None

    def __call__(
        self,
        queries: list[dict[str, Any]],
        output_dir: str | Path,
        use_metagenomic_db: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run ColabFold remote MSA search.

        Args:
            queries: List of query dicts. Each entry has:
                - ``sequences``: str (unpaired query) OR list[str] (paired group).
            output_dir: Directory to write MSA output files.
            use_metagenomic_db: Whether to include environmental sequences.
            verbose: Whether to print status messages.

        Returns:
            Dictionary with ``msa_paths`` (query-index str → path for unpaired
            queries), ``paired_msa_paths`` (query-index str → per-chain path list
            for paired queries), ``success``, ``num_successful``, ``num_failed``,
            optional ``errors``.
        """
        # Lazy load on first call
        if not self._loaded:
            self.load(verbose)

        # Convert paths
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        msas_dir = output_dir / "msas"
        msas_dir.mkdir(exist_ok=True)

        msa_paths: dict[str, str] = {}
        paired_msa_paths: dict[str, list[str]] = {}
        errors: list[tuple[str, str]] = []

        for q_idx, query in enumerate(queries):
            sequences = query["sequences"]
            label = f"query_{q_idx}"

            # Paired query: sequences is a list. One API call with use_pairing=True.
            if isinstance(sequences, list):
                msas_paired_dir = output_dir / "msas_paired"
                msas_paired_dir.mkdir(exist_ok=True)
                try:
                    paired_msa_paths[str(q_idx)] = self._run_paired_group(
                        sequences,
                        q_idx,
                        output_dir,
                        msas_paired_dir,
                        use_metagenomic_db=use_metagenomic_db,
                    )
                except Exception as e:
                    errors.append((label, f"Paired MSA failed: {e!s}"))
                    logger.debug(f"Paired MSA error in {label}: {e}")
                continue

            # Unpaired query: single string.
            seq = sequences
            try:
                logger.debug(f"Running remote MSA search for {label}...")

                # Use a temporary prefix for ColabFold output
                temp_output_prefix = str(output_dir / label)

                # Run remote MSA search, retrying transient API/network blips.
                _run_with_retry(
                    self.run_mmseqs2,
                    label,
                    seq,
                    temp_output_prefix,
                    use_env=use_metagenomic_db,
                )

                # ColabFold's MSA subfolder suffix depends on use_env: `_env` when True, `_all` when False.
                temp_results_dir = f"{temp_output_prefix}_env" if use_metagenomic_db else f"{temp_output_prefix}_all"

                temp_results_path = Path(temp_results_dir)

                # Debug: Check what directories and files exist
                logger.debug("Looking for MSA files...")
                logger.debug(f"  Temp prefix: {temp_output_prefix}")
                logger.debug(f"  Expected results dir: {temp_results_path}")
                logger.debug(f"  Results dir exists: {temp_results_path.exists()}")

                # Check parent directory
                parent_dir = Path(output_dir)
                if parent_dir.exists():
                    all_items = list(parent_dir.iterdir())
                    logger.debug(f"  Files/dirs in parent: {[item.name for item in all_items]}")

                if temp_results_path.exists():
                    all_files = list(temp_results_path.rglob("*"))
                    logger.debug(
                        f"  All files in results dir: {[str(f.relative_to(temp_results_path)) for f in all_files if f.is_file()]}"
                    )

                # Look for .a3m files in the results directory and subdirectories
                a3m_files = list(temp_results_path.rglob("*.a3m")) if temp_results_path.exists() else []

                if not a3m_files:
                    error_msg = f"Remote MSA generation completed but no .a3m files found in {temp_results_dir}"
                    logger.debug(f"Warning: {error_msg}")
                    errors.append((label, error_msg))

                    # Clean up temp directory
                    if Path(temp_results_dir).exists():
                        shutil.rmtree(temp_results_dir, ignore_errors=True)
                    continue

                # Use the first (or only) .a3m file found
                old_msa_path = a3m_files[0]

                logger.debug(f"Found MSA file: {old_msa_path.relative_to(temp_results_path)}")

                # Move to clean 'msas/' sub-directory, named by query index.
                new_msa_path = msas_dir / f"{q_idx}.a3m"
                shutil.copyfile(old_msa_path, new_msa_path)

                msa_paths[str(q_idx)] = str(new_msa_path)

                logger.debug(f"Successfully generated MSA for {label}")

                # Clean up temp directory after success
                if Path(temp_results_dir).exists():
                    shutil.rmtree(temp_results_dir, ignore_errors=True)

            except Exception as e:
                error_msg = f"Failed to generate MSA for {label}: {e!s}"
                logger.debug(f"Error: {error_msg}")
                errors.append((label, error_msg))

        success = (len(msa_paths) + len(paired_msa_paths)) > 0
        result: dict[str, Any] = {
            "msa_paths": msa_paths,
            "paired_msa_paths": paired_msa_paths,
            "success": success,
            "num_successful": len(msa_paths) + len(paired_msa_paths),
            "num_failed": len(errors),
        }

        if errors:
            result["errors"] = dict(errors)

        return result

    def _run_paired_group(
        self,
        group_seqs: list[str],
        group_idx: int,
        output_dir: Path,
        msas_paired_dir: Path,
        use_metagenomic_db: bool,
    ) -> list[str]:
        """Submit one paired query and parse the per-chain row-aligned blocks.

        Returns the per-chain A3M paths in input chain order. Each per-chain A3M
        preserves row order so downstream tools can pair rows by position. Raises
        on failure (caller catches and records in `errors`).
        """
        # Prefix lives under output_dir so it doesn't collide with unpaired runs.
        temp_prefix = str(output_dir / f"pair_group_{group_idx}")
        _run_with_retry(
            self.run_mmseqs2,
            f"pair_group_{group_idx}",
            group_seqs,
            temp_prefix,
            use_env=use_metagenomic_db,
            use_pairing=True,
        )

        # api.colabfold.com writes a single `pair.a3m` under `{prefix}_pairgreedy/`.
        pair_dir = Path(f"{temp_prefix}_pairgreedy")
        pair_a3m = pair_dir / "pair.a3m"
        if not pair_a3m.exists():
            candidates = list(pair_dir.rglob("*.a3m")) if pair_dir.exists() else []
            if not candidates:
                raise RuntimeError(f"colabfold paired query returned no .a3m file in {pair_dir}")
            pair_a3m = candidates[0]

        per_chain_blocks = _parse_pair_a3m(pair_a3m, num_chains=len(group_seqs))

        written: list[str] = []
        for chain_idx, block_bytes in enumerate(per_chain_blocks):
            chain_a3m = msas_paired_dir / f"{group_idx}_chain_{chain_idx}.a3m"
            chain_a3m.write_bytes(block_bytes)
            written.append(str(chain_a3m))

        # Clean up the temp pairgreedy dir.
        if pair_dir.exists():
            shutil.rmtree(pair_dir, ignore_errors=True)

        return written

    def load(self, verbose: bool = False) -> None:  # noqa: ARG002 — required by tool interface
        """Load ColabFold remote search module."""
        logger.debug("Initializing ColabFold remote search")

        try:
            from colabfold.colabfold import run_mmseqs2

            self.run_mmseqs2 = run_mmseqs2
        except ImportError as e:
            raise ImportError(
                "mmseqs2-homology-search: 'colabfold' module not installed; pip install 'colabfold[alphafold]' or re-run standalone/setup.sh"
            ) from e

        self._loaded = True

        logger.debug("ColabFold remote search initialized")


def _parse_pair_a3m(pair_a3m: Path, num_chains: int) -> list[bytes]:
    r"""Split a paired `pair.a3m` into per-chain row-aligned blocks.

    Format observed empirically against api.colabfold.com: a single file with
    `\x00`-separated chain blocks, each block a standard A3M with the chain
    query as the first sequence. Trailing empty block from the final delimiter
    is dropped. Asserts `num_chains` blocks were produced.
    """
    raw = pair_a3m.read_bytes()
    blocks = [b for b in raw.split(b"\x00") if b.strip()]
    if len(blocks) != num_chains:
        raise RuntimeError(f"colabfold pair.a3m: expected {num_chains} chain blocks, got {len(blocks)}")

    # Optional sanity check: equal row counts across blocks.
    row_counts = [b.count(b"\n>") + (1 if b.startswith(b">") else 0) for b in blocks]
    if len(set(row_counts)) != 1:
        raise RuntimeError(
            f"colabfold pair.a3m: chain blocks have unequal row counts {row_counts}; expected row-aligned paired output"
        )

    return blocks


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    wrapper = ColabFoldRemoteSearchWrapper()
    return wrapper(
        queries=input_dict["queries"],
        output_dir=input_dict["output_dir"],
        use_metagenomic_db=input_dict.get("use_metagenomic_db", False),
        verbose=input_dict.get("verbose", True),
    )


# Standalone script entry point for venv execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            "mmseqs2-homology-search: usage: python remote_msa_search.py <input_json_path> <output_json_path>"
        )

    # Get the input and output json paths
    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    # Read input json
    with open(input_json_path) as f:
        input_data = json.load(f)

    # Create wrapper and run search
    wrapper = ColabFoldRemoteSearchWrapper()
    output_data = wrapper(
        queries=input_data["queries"],
        output_dir=input_data["output_dir"],
        use_metagenomic_db=input_data.get("use_metagenomic_db", False),
        verbose=input_data.get("verbose", True),
    )

    # Write the output to a json file
    with open(output_json_path, "w") as f:
        json.dump(output_data, f)

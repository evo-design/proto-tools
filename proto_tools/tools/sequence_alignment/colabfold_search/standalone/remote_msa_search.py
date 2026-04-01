"""remote_msa_search.py.

ColabFold remote MSA search standalone script for isolated venv execution.

This script runs ColabFold's remote MMseqs2 API in an isolated environment
to avoid dependency conflicts with the main proto-language environment.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ColabFoldRemoteSearchWrapper:
    """Wrapper for ColabFold remote MSA search using the ColabFold API."""

    def __init__(self) -> None:
        """Initialize ColabFold remote search wrapper."""
        self._loaded = False
        self.run_mmseqs2 = None

    def __call__(
        self,
        sequences: list[str],
        sequence_ids: list[str],
        output_dir: str | Path,
        use_metagenomic_db: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run ColabFold remote MSA search.

        Args:
            sequences: List of protein sequences to search
            sequence_ids: List of sequence identifiers
            output_dir: Directory to write MSA output files
            use_metagenomic_db: Whether to include environmental sequences
            verbose: Whether to print status messages

        Returns:
            Dictionary containing output paths and success status
        """
        # Lazy load on first call
        if not self._loaded:
            self.load(verbose)

        # Convert paths
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create msas subdirectory
        msas_dir = output_dir / "msas"
        msas_dir.mkdir(exist_ok=True)

        msa_paths = {}
        errors = []

        for sequence, seq_id in zip(sequences, sequence_ids, strict=False):
            try:
                logger.debug(f"Running remote MSA search for {seq_id}...")

                # Use a temporary prefix for ColabFold output
                temp_output_prefix = str(output_dir / seq_id)

                # Run remote MSA search
                self.run_mmseqs2(  # type: ignore[misc]
                    sequence,
                    temp_output_prefix,
                    use_env=use_metagenomic_db,
                )

                # ColabFold saves the MSA in a specific subfolder
                # The suffix depends on whether environmental DB is used:
                # - use_env=True: {prefix}_env
                # - use_env=False: {prefix}_all
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
                    errors.append((seq_id, error_msg))

                    # Clean up temp directory
                    if Path(temp_results_dir).exists():
                        shutil.rmtree(temp_results_dir, ignore_errors=True)
                    continue

                # Use the first (or only) .a3m file found
                old_msa_path = a3m_files[0]

                logger.debug(f"Found MSA file: {old_msa_path.relative_to(temp_results_path)}")

                # Move to clean 'msas/' sub-directory
                new_msa_path = msas_dir / f"{seq_id}.a3m"
                shutil.copyfile(old_msa_path, new_msa_path)

                msa_paths[seq_id] = str(new_msa_path)

                logger.debug(f"Successfully generated MSA for {seq_id}")

                # Clean up temp directory after success
                if Path(temp_results_dir).exists():
                    shutil.rmtree(temp_results_dir, ignore_errors=True)

            except Exception as e:
                error_msg = f"Failed to generate MSA for {seq_id}: {e!s}"
                logger.debug(f"Error: {error_msg}")
                errors.append((seq_id, error_msg))

        success = len(msa_paths) > 0
        result = {
            "msa_paths": msa_paths,
            "success": success,
            "num_successful": len(msa_paths),
            "num_failed": len(errors),
        }

        if errors:
            result["errors"] = dict(errors)

        return result

    def load(self, verbose: bool = False) -> None:  # noqa: ARG002 — required by tool interface
        """Load ColabFold remote search module."""
        logger.debug("Initializing ColabFold remote search")

        try:
            from colabfold.colabfold import run_mmseqs2

            self.run_mmseqs2 = run_mmseqs2
        except ImportError as e:
            raise ImportError(
                "Error: The 'colabfold' module is missing.\nPlease install it using 'pip install colabfold[alphafold]'"
            ) from e

        self._loaded = True

        logger.debug("ColabFold remote search initialized")


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    wrapper = ColabFoldRemoteSearchWrapper()
    return wrapper(
        sequences=input_dict["sequences"],
        sequence_ids=input_dict["sequence_ids"],
        output_dir=input_dict["output_dir"],
        use_metagenomic_db=input_dict.get("use_metagenomic_db", False),
        verbose=input_dict.get("verbose", True),
    )


# Standalone script entry point for venv execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python remote_msa_search.py <input_json_path> <output_json_path>")

    # Get the input and output json paths
    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    # Read input json
    with open(input_json_path) as f:
        input_data = json.load(f)

    # Create wrapper and run search
    wrapper = ColabFoldRemoteSearchWrapper()
    output_data = wrapper(
        sequences=input_data["sequences"],
        sequence_ids=input_data["sequence_ids"],
        output_dir=input_data["output_dir"],
        use_metagenomic_db=input_data.get("use_metagenomic_db", False),
        verbose=input_data.get("verbose", True),
    )

    # Write the output to a json file
    with open(output_json_path, "w") as f:
        json.dump(output_data, f)

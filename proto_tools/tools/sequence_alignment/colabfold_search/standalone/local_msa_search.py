"""local_msa_search.py.

ColabFold MSA search standalone script for isolated venv execution.

This script runs colabfold_search in an isolated environment to avoid
dependency conflicts with the main proto-language environment.
"""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ColabFoldSearchWrapper:
    """Wrapper for colabfold_search command execution."""

    def __init__(self) -> None:
        """Initialize ColabFold search wrapper."""
        self._loaded = False
        self.colabfold_search_executable: str | None = None
        self.mmseqs_executable: str | None = None

    def _detect_database_name(
        self,
        msa_db_dir: Path,
        verbose: bool = False,  # noqa: ARG002 — required by tool interface
    ) -> str | None:
        """Auto-detect the database name by scanning for .dbtype files.

        Args:
            msa_db_dir: Path to the database directory
            verbose: Whether to print detection info

        Returns:
            Database name (without extension) or None if not found
        """
        # Look for files ending in .dbtype (but not _seq.dbtype, _aln.dbtype, _h.dbtype, etc.)
        # We want the main database file, which typically ends with just _db.dbtype
        dbtype_files = list(msa_db_dir.glob("*_db.dbtype"))

        if not dbtype_files:
            # Fallback: look for any .dbtype file that doesn't have a suffix
            all_dbtype = list(msa_db_dir.glob("*.dbtype"))
            if all_dbtype:
                # Filter out files with suffixes like _seq, _aln, _h
                main_dbs = [
                    f
                    for f in all_dbtype
                    if not any(f.stem.endswith(suffix) for suffix in ["_seq", "_aln", "_h", "_seq_h"])
                ]
                if main_dbs:
                    dbtype_files = main_dbs

        if not dbtype_files:
            logger.debug(f"Warning: No database files found in {msa_db_dir}")
            return None

        # Prioritize uniref databases over metagenome databases
        uniref_dbs = [f for f in dbtype_files if "uniref" in f.stem.lower()]

        if uniref_dbs:
            db_file = uniref_dbs[0]
            if len(dbtype_files) > 1:
                logger.debug("  Note: Multiple databases found, prioritizing uniref database")
        else:
            # Use the first database found if no uniref database is present
            db_file = dbtype_files[0]

        db_name = db_file.stem  # Remove .dbtype extension

        logger.debug(f"Auto-detected database: {db_name}")
        if len(dbtype_files) > 1:
            other_dbs = [f.stem for f in dbtype_files if f != db_file]
            logger.debug(f"  Other databases available: {', '.join(other_dbs)}")

        return db_name

    def __call__(
        self,
        query_fasta_path: str | Path,
        msa_db_dir: str | Path,
        output_dir: str | Path,
        num_threads: int = 8,
        use_metagenomic_db: bool = False,
        sensitivity: float | None = None,
        database_name: str | None = None,
        use_gpu: bool = False,
        verbose: bool = False,
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run ColabFold MSA search.

        Args:
            query_fasta_path: Path to query FASTA file
            msa_db_dir: Path to ColabFold/MMSeqs2 database directory
            output_dir: Directory to write MSA output files
            num_threads: Number of CPU threads
            use_metagenomic_db: Whether to use metagenomic databases
            sensitivity: MMseqs2 sensitivity parameter (1.0-9.0), None for default
            database_name: Name of the database to use. If None, auto-detects the database
            use_gpu: Whether to enable GPU acceleration
            verbose: Whether to print status messages
            extra_args: Verbatim colabfold_search CLI tokens appended after typed flags

        Returns:
            Dictionary containing output_dir and success status
        """
        # Lazy load on first call
        if not self._loaded:
            self.load(verbose)

        # Convert paths
        query_fasta_path = Path(query_fasta_path)
        msa_db_dir = Path(msa_db_dir)
        output_dir = Path(output_dir)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build the command
        cmd = [
            self.colabfold_search_executable,
            "--mmseqs",
            self.mmseqs_executable,
            "--threads",
            str(num_threads),
        ]

        # Add sensitivity parameter if provided
        if sensitivity is not None:
            cmd.extend(["-s", str(sensitivity)])

        # Add metagenomic database parameter if provided
        if not use_metagenomic_db:
            cmd.extend(["--use-env", "0"])

        # Add database name
        if database_name:
            db_name = database_name
            logger.debug(f"Using specified database: {db_name}")
        else:
            db_name = self._detect_database_name(msa_db_dir, verbose)  # type: ignore[assignment]
            if db_name is None:
                raise RuntimeError(
                    f"colabfold-search: no MMseqs2 database files found in {msa_db_dir} "
                    f"(expected *.dbtype); ensure the MSA database has been prepared, "
                    f"or pass database_name explicitly"
                )

        cmd.extend(["--db1", db_name])

        cmd.extend(
            [
                str(query_fasta_path),
                str(msa_db_dir),
                str(output_dir),
            ]
        )

        if use_gpu:
            cmd.extend(["--gpu", "1"])

        # Power-user escape hatch: append verbatim CLI tokens.
        if extra_args:
            cmd.extend(str(arg) for arg in extra_args)

        logger.debug(f"Running command: {' '.join(cmd)}")  # type: ignore[arg-type]

        # Run the command
        try:
            _ = subprocess.run(
                cmd,  # type: ignore[arg-type]
                check=True,
                capture_output=not verbose,
                text=True,
            )

            logger.debug("ColabFold search completed successfully")

            return {
                "output_dir": str(output_dir),
                "success": True,
                "returncode": 0,
            }

        except subprocess.CalledProcessError as e:
            stderr_tail = " | ".join((e.stderr or "").strip().splitlines()[-10:]) or "<no stderr>"
            error_msg = f"colabfold-search: failed (exit {e.returncode}): {stderr_tail}"

            # Check if this is a "prof_res does not exist" error
            # This happens when sequences have no hits at all
            stderr_text = e.stderr or ""
            if "prof_res does not exist" in stderr_text:
                logger.debug("Some sequences may have no homologs (prof_res error)")

                # Read the query FASTA to get sequence IDs and sequences
                with open(query_fasta_path) as f:
                    fasta_content = f.read()

                # Parse FASTA content
                sequences = []
                current_id = None
                current_seq: list[str] = []

                for line in fasta_content.strip().split("\n"):
                    if line.startswith(">"):
                        if current_id is not None:
                            sequences.append((current_id, "".join(current_seq)))  # type: ignore[arg-type]
                        current_id = line[1:].strip()
                        current_seq = []
                    else:
                        current_seq.append(line.strip())

                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq)))  # type: ignore[arg-type]

                # Create A3M files ONLY for sequences that don't already have one
                for idx, (seq_id, seq) in enumerate(sequences):
                    a3m_path = output_dir / f"{idx}.a3m"

                    # Only create if the file doesn't already exist
                    if not a3m_path.exists():
                        with open(a3m_path, "w") as f:
                            f.write(f">{seq_id}\n{seq}\n")

                        logger.debug(f"Created empty MSA for {seq_id} (no homologs found)")
                    else:
                        logger.debug(f"MSA already exists for {seq_id}")

                # Return success since we handled the no-results case
                return {
                    "output_dir": str(output_dir),
                    "success": True,
                    "returncode": 0,
                }

            return {
                "output_dir": str(output_dir),
                "success": False,
                "returncode": e.returncode,
                "error": error_msg,
            }

    def load(self, verbose: bool = False) -> None:  # noqa: ARG002 — required by tool interface
        """Load ColabFold search executable."""
        logger.debug("Initializing ColabFold search")

        # Get the venv's bin directory
        venv_bin_dir = Path(sys.executable).parent

        # First try to find colabfold_search in the current venv's bin directory
        venv_executable = venv_bin_dir / "colabfold_search"

        if venv_executable.exists():
            self.colabfold_search_executable = str(venv_executable)
        elif shutil.which("colabfold_search") is not None:
            self.colabfold_search_executable = shutil.which("colabfold_search")
        else:
            raise ImportError(
                "colabfold-search: 'colabfold_search' executable not found in venv; re-run standalone/setup.sh"
            )

        # Find mmseqs binary (required by colabfold_search)
        venv_mmseqs = venv_bin_dir / "mmseqs"
        if venv_mmseqs.exists():
            self.mmseqs_executable = str(venv_mmseqs)
        elif shutil.which("mmseqs") is not None:
            self.mmseqs_executable = shutil.which("mmseqs")
        else:
            raise ImportError("colabfold-search: 'mmseqs' executable not found in venv; re-run standalone/setup.sh")

        self._loaded = True

        logger.debug(f"ColabFold search initialized. Using executable: {self.colabfold_search_executable}")
        logger.debug(f"MMseqs2 executable: {self.mmseqs_executable}")


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for persistent-worker execution."""
    wrapper = ColabFoldSearchWrapper()
    return wrapper(
        query_fasta_path=input_dict["query_fasta_path"],
        msa_db_dir=input_dict["msa_db_dir"],
        output_dir=input_dict["output_dir"],
        num_threads=input_dict["num_threads"],
        use_metagenomic_db=input_dict["use_metagenomic_db"],
        sensitivity=input_dict.get("sensitivity"),
        database_name=input_dict.get("database_name"),
        use_gpu=input_dict.get("use_gpu", False),
        verbose=input_dict.get("verbose", True),
        extra_args=input_dict.get("extra_args"),
    )


# Standalone script entry point for venv execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("colabfold-search: usage: python msa_search.py <input_json_path> <output_json_path>")

    # Get the input and output json paths
    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    # Read input json
    with open(input_json_path) as f:
        input_data = json.load(f)

    # Create wrapper and run search
    wrapper = ColabFoldSearchWrapper()
    output_data = wrapper(
        query_fasta_path=input_data["query_fasta_path"],
        msa_db_dir=input_data["msa_db_dir"],
        output_dir=input_data["output_dir"],
        num_threads=input_data["num_threads"],
        use_metagenomic_db=input_data["use_metagenomic_db"],
        sensitivity=input_data.get("sensitivity"),
        database_name=input_data.get("database_name"),
        use_gpu=input_data.get("use_gpu", False),
        verbose=input_data.get("verbose", True),
        extra_args=input_data.get("extra_args"),
    )

    # Write the output to a json file
    with open(output_json_path, "w") as f:
        json.dump(output_data, f)

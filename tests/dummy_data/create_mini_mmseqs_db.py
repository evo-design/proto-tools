#!/usr/bin/env python3
"""
tests/dummy_data/create_mini_mmseqs_db.py

This script downloads and sets up a mini MMseqs2 database for testing purposes.
It uses ToolInstance to access mmseqs from the isolated tool environment.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# Add project root to path to import proto_tools
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Ensure project root is in path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from proto_tools.utils.tool_instance import ToolInstance
except ModuleNotFoundError:
    logger.error("Could not import proto_tools")
    logger.error("  Script location: %s", SCRIPT_DIR)
    logger.error("  Project root: %s", PROJECT_ROOT)
    logger.error("  sys.path: %s", sys.path)
    raise


def download_file(url: str, output_path: Path) -> None:
    """Download a file using available tools."""
    logger.info("Downloading %s...", url)

    # Try different download methods
    download_tools = [
        (["aria2c", "--max-connection-per-server=8", "--allow-overwrite=true", "-o", output_path.name, "-d", str(output_path.parent), url], "aria2c"),
        (["curl", "-L", "-o", str(output_path), url], "curl"),
        (["wget", "-O", str(output_path), url], "wget"),
    ]

    for cmd, tool_name in download_tools:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, check=True)
                logger.info("Downloaded using %s", tool_name)
                return
            except subprocess.CalledProcessError:
                continue

    # Fallback to Python's urllib
    try:
        logger.info("Using Python urllib to download...")
        urlretrieve(url, output_path)
        logger.info("Download complete")
        return
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def setup_mini_database():
    """Set up mini MMseqs2 debug database."""
    # Database directory
    db_dir = SCRIPT_DIR / "mini_mmseqs_db"
    db_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up mini MMseqs2 database in %s", db_dir)

    # Check if database already exists
    if (db_dir / "uniref30_mini_db.dbtype").exists():
        logger.info("Database already exists, skipping setup")
        return

    # We need the mmseqs binary directly (not via run_*), so explicitly
    # build the env; ToolInstance.get() alone defers env creation.
    logger.info("Initializing mmseqs environment...")
    tool = ToolInstance.get("mmseqs")
    tool.ensure_ready()
    mmseqs_bin = tool.env_path / "bin" / "mmseqs"

    if not mmseqs_bin.exists():
        raise RuntimeError(
            f"mmseqs binary not found at {mmseqs_bin}. "
            f"ToolInstance may have failed to set up the environment."
        )

    logger.info("Using mmseqs from: %s", mmseqs_bin)

    # Download mini SwissProt database
    archive_path = db_dir / "mini_swissprot2503.tar.gz"
    if not archive_path.exists():
        download_file(
            "https://opendata.mmseqs.org/colabfold/mini_swissprot2503.tar.gz",
            archive_path
        )

    # Extract archive
    logger.info("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=db_dir)
    logger.info("Extraction complete")

    # Move extracted files into MMseqs2 database structure using mmseqs mvdb
    logger.info("Renaming database files...")
    renames = [
        ("sprot2503_h", "uniref30_mini_db_h"),
        ("sprot2503", "uniref30_mini_db"),
        ("sprot2503_aln", "uniref30_mini_db_aln"),
        ("sprot2503_seq_h", "uniref30_mini_db_seq_h"),
        ("sprot2503_seq", "uniref30_mini_db_seq"),
    ]

    for old_name, new_name in renames:
        old_path = db_dir / old_name
        new_path = db_dir / new_name
        if old_path.with_suffix(".dbtype").exists():
            cmd = [str(mmseqs_bin), "mvdb", str(old_path), str(new_path)]
            logger.info("  Running: mmseqs mvdb %s %s", old_name, new_name)
            subprocess.run(cmd, check=True, cwd=db_dir)

    # Move taxonomy and mapping files (these are regular files, not databases)
    for file_suffix in ["taxonomy", "mapping"]:
        old_file = db_dir / f"sprot2503_{file_suffix}"
        new_file = db_dir / f"uniref30_mini_db_{file_suffix}"
        if old_file.exists():
            logger.info("  Moving %s to %s", old_file.name, new_file.name)
            old_file.rename(new_file)

    logger.info("Mini debug database setup complete in %s", db_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        setup_mini_database()
    except Exception as e:
        logger.error("ERROR: %s", e)
        sys.exit(1)

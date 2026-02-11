#!/usr/bin/env python3
"""
Setup script for MMseqs2 mini debug database.

This script downloads and sets up a mini MMseqs2 database for testing purposes.
It uses the EnvManager to access mmseqs from the isolated venv.
"""

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

# Add project root to path to import bio_programming_tools
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Ensure project root is in path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from bio_programming_tools.utils.env_manager import EnvManager
except ModuleNotFoundError:
    print(f"ERROR: Could not import bio_programming_tools", file=sys.stderr)
    print(f"  Script location: {SCRIPT_DIR}", file=sys.stderr)
    print(f"  Project root: {PROJECT_ROOT}", file=sys.stderr)
    print(f"  sys.path: {sys.path}", file=sys.stderr)
    raise


def download_file(url: str, output_path: Path) -> None:
    """Download a file using available tools."""
    print(f"Downloading {url}...")

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
                print(f"Downloaded using {tool_name}")
                return
            except subprocess.CalledProcessError:
                continue

    # Fallback to Python's urllib
    try:
        print("Using Python urllib to download...")
        urlretrieve(url, output_path)
        print("Download complete")
        return
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def setup_mini_database():
    """Set up mini MMseqs2 debug database."""
    # Database directory
    db_dir = SCRIPT_DIR / "mini_mmseqs_db"
    db_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up mini MMseqs2 database in {db_dir}")

    # Check if database already exists
    if (db_dir / "uniref30_mini_db.dbtype").exists():
        print("Database already exists, skipping setup")
        return

    # Initialize EnvManager for mmseqs
    print("Initializing mmseqs environment...")
    env_manager = EnvManager(model_name="mmseqs")
    mmseqs_bin = env_manager.env_path / "bin" / "mmseqs"

    if not mmseqs_bin.exists():
        raise RuntimeError(
            f"mmseqs binary not found at {mmseqs_bin}. "
            f"EnvManager may have failed to set up the environment."
        )

    print(f"Using mmseqs from: {mmseqs_bin}")

    # Download mini SwissProt database
    archive_path = db_dir / "mini_swissprot2503.tar.gz"
    if not archive_path.exists():
        download_file(
            "https://opendata.mmseqs.org/colabfold/mini_swissprot2503.tar.gz",
            archive_path
        )

    # Extract archive
    print("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=db_dir)
    print("Extraction complete")

    # Move extracted files into MMseqs2 database structure using mmseqs mvdb
    print("Renaming database files...")
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
            print(f"  Running: mmseqs mvdb {old_name} {new_name}")
            subprocess.run(cmd, check=True, cwd=db_dir)

    # Move taxonomy and mapping files (these are regular files, not databases)
    for file_suffix in ["taxonomy", "mapping"]:
        old_file = db_dir / f"sprot2503_{file_suffix}"
        new_file = db_dir / f"uniref30_mini_db_{file_suffix}"
        if old_file.exists():
            print(f"  Moving {old_file.name} to {new_file.name}")
            old_file.rename(new_file)

    print(f"\n✓ Mini debug database setup complete in {db_dir}")


if __name__ == "__main__":
    try:
        setup_mini_database()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

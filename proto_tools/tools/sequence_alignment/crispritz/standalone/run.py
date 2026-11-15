"""Standalone CRISPRitz index and search runner."""

import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from standalone_helpers import StatusCounter, get_logger

logger = get_logger(__name__)

VERSION = "2.8.1"
FASTA_SUFFIXES = {".fa", ".fasta", ".fna"}


def _reference_files(value: str) -> list[Path]:
    path = Path(value).expanduser().resolve()
    if path.is_file() and path.suffix.lower() in FASTA_SUFFIXES:
        return [path]
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in FASTA_SUFFIXES)
        if files:
            return files
    raise ValueError("reference_fasta must be an existing .fa, .fasta, or .fna file or containing directory")


def _hash_references(files: list[Path]) -> str:
    """Return one digest over every source FASTA's contents and size."""
    combined = hashlib.sha256()
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        combined.update(digest.hexdigest().encode())
        combined.update(str(path.stat().st_size).encode())
    return combined.hexdigest()


def _write_record(header: str | None, sequence: list[str], directory: Path, index: int) -> None:
    if header is None:
        return
    seq = "".join(sequence).replace(" ", "").upper()
    if not seq:
        raise ValueError(f"empty FASTA record: {header}")
    identifier = header.split()[0]
    destination = directory / f"record_{index:08d}.fa"
    destination.write_text(f">{identifier}\n{seq}\n")


def _split_fasta(files: list[Path], directory: Path) -> int:
    directory.mkdir(parents=True)
    count = 0
    for source in files:
        header = None
        sequence: list[str] = []
        with source.open() as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    _write_record(header, sequence, directory, count)
                    if header is not None:
                        count += 1
                    header, sequence = line[1:], []
                else:
                    if header is None:
                        raise ValueError(f"sequence data before FASTA header in {source}")
                    sequence.append(line)
        _write_record(header, sequence, directory, count)
        if header is not None:
            count += 1
    if not count:
        raise ValueError("reference FASTA contains no records")
    return count


def _binary(name: str) -> Path:
    path = Path(sys.prefix) / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(f"CRISPRitz executable is missing: {path}")
    return path


def _run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"CRISPRitz command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _ensure_index(
    files: list[Path],
    reference_sha256: str,
    guide_length: int,
    pam: str,
    pam_side: str,
    max_bulges: int,
    threads: int,
    cache_root: Path,
) -> tuple[Path, bool]:
    pam_spec = (
        ("N" * guide_length + pam, len(pam)) if pam_side == "three_prime" else (pam + "N" * guide_length, -len(pam))
    )
    key_material = json.dumps(
        {
            "version": VERSION,
            "reference_sha256": reference_sha256,
            "guide_length": guide_length,
            "pam": pam,
            "pam_side": pam_side,
            "max_bulges": max_bulges,
        },
        sort_keys=True,
    )
    index_key = hashlib.sha256(key_material.encode()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / index_key
    lock_path = cache_root / f".{index_key}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("key_material") == key_material:
            return destination / "index", True
        if destination.exists():
            shutil.rmtree(destination)
        staging = Path(tempfile.mkdtemp(prefix=f".{index_key}.", dir=cache_root))
        try:
            fasta_dir = staging / "fasta"
            index_dir = staging / "index"
            _split_fasta(files, fasta_dir)
            index_dir.mkdir()
            pam_path = staging / "pam.txt"
            pam_path.write_text(f"{pam_spec[0]} {pam_spec[1]}\n")
            fastas = sorted(fasta_dir.glob("*.fa"))
            counter = StatusCounter(logger, "Indexing reference sequence %d/%d", total=len(fastas))
            for index, fasta in enumerate(fastas, start=1):
                counter(index)
                _run(
                    [str(_binary("crispritz-buildTST")), str(fasta), str(pam_path), str(threads), str(max_bulges)],
                    index_dir,
                )
            if not any(index_dir.glob("*.bin")):
                raise RuntimeError("CRISPRitz index creation produced no .bin files")
            (staging / "manifest.json").write_text(json.dumps({"key_material": key_material}, indent=2))
            staging.replace(destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination / "index", False


def _guide_from_alignment(alignment: str, pam_length: int, pam_side: str) -> str:
    ungapped = alignment.replace("-", "")
    return ungapped[pam_length:] if pam_side == "five_prime" else ungapped[:-pam_length]


def _parse_hits(path: Path, pam_length: int, pam_side: str) -> list[dict[str, Any]]:
    hits = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {
            "#Bulge type",
            "crRNA",
            "DNA",
            "Chromosome",
            "Position",
            "Cluster Position",
            "Direction",
            "Mismatches",
            "Bulge Size",
            "Total",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(f"unexpected CRISPRitz target columns: {reader.fieldnames}")
        for row in reader:
            bulge = row["#Bulge type"].lower()
            normalized_bulge = {"x": "none", "dna": "dna", "rna": "rna", "rna,dna": "dna_rna"}.get(bulge)
            if normalized_bulge is None:
                raise ValueError(f"unexpected CRISPRitz bulge type: {row['#Bulge type']}")
            hits.append(
                {
                    "guide": _guide_from_alignment(row["crRNA"], pam_length, pam_side),
                    "bulge_type": normalized_bulge,
                    "crispritz_bulge_type": row["#Bulge type"],
                    "guide_alignment": row["crRNA"],
                    "target_alignment": row["DNA"],
                    "chromosome": row["Chromosome"],
                    "start": int(row["Position"]) + 1,
                    "end": int(row["Position"]) + len(row["DNA"].replace("-", "")),
                    "strand": row["Direction"],
                    "mismatches": int(row["Mismatches"]),
                    "bulge_size": int(row["Bulge Size"]),
                    "total_edits": int(row["Total"]),
                }
            )
    return hits


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Build or reuse the reference index and enumerate off-targets."""
    files = _reference_files(input_dict["reference_fasta"])
    logger.update_status("Hashing reference genome")
    reference_sha256 = _hash_references(files)
    guides = input_dict["guides"]
    pam = input_dict["pam"]
    pam_side = input_dict["pam_side"]
    max_bulges = max(input_dict["max_dna_bulges"], input_dict["max_rna_bulges"])
    default_cache = Path(os.environ.get("PROTO_HOME", Path.home() / ".proto")) / "proto_tool_cache/crispritz/indexes"
    cache_root = (
        Path(input_dict["index_cache_dir"]).expanduser().resolve()
        if input_dict.get("index_cache_dir")
        else default_cache
    )
    index_dir, index_reused = _ensure_index(
        files, reference_sha256, len(guides[0]), pam, pam_side, max_bulges, input_dict["threads"], cache_root
    )
    with tempfile.TemporaryDirectory(prefix="crispritz-search-") as temporary:
        work = Path(temporary)
        pam_length = len(pam)
        pam_pattern = "N" * len(guides[0]) + pam if pam_side == "three_prime" else pam + "N" * len(guides[0])
        pam_value = pam_length if pam_side == "three_prime" else -pam_length
        pam_path = work / "pam.txt"
        pam_path.write_text(f"{pam_pattern} {pam_value}\n")
        placeholder = "N" * pam_length
        guide_path = work / "guides.txt"
        guide_path.write_text(
            "\n".join(guide + placeholder if pam_side == "three_prime" else placeholder + guide for guide in guides)
            + "\n"
        )
        result_prefix = work / "results"
        command = [
            str(_binary("crispritz-searchTST")),
            str(index_dir) + "/",
            str(guide_path),
            str(input_dict["max_mismatches"]),
            str(input_dict["max_dna_bulges"]),
            str(input_dict["max_rna_bulges"]),
            str(pam_path),
            str(result_prefix),
            "r",
            str(input_dict["threads"]),
            str(max_bulges),
            str(input_dict["max_total_edits"] if input_dict["max_total_edits"] is not None else -1),
        ]
        logger.update_status(f"Searching off-targets for {len(guides)} guide{'s' if len(guides) != 1 else ''}")
        _run(command, work)
        hits = _parse_hits(result_prefix.with_suffix(".targets.txt"), pam_length, pam_side)
    return {"hits": hits, "reference_sha256": reference_sha256, "index_reused": index_reused}


def to_device(device: str) -> dict[str, Any]:
    """Confirm CPU placement."""
    del device
    return {"success": True, "device": "cpu"}


def get_memory_stats() -> dict[str, Any]:
    """Report that GPU accounting is inapplicable."""
    return {"available": False, "framework": "cpu"}


if __name__ == "__main__":
    with open(sys.argv[1]) as input_handle:
        payload = json.load(input_handle)
    with open(sys.argv[2], "w") as output_handle:
        json.dump(dispatch(payload), output_handle)

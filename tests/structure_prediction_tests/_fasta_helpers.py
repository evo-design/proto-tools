"""tests/structure_prediction_tests/_fasta_helpers.py.

Shared FASTA parsing helpers for structure prediction tests.

Loads ``tests/dummy_data/structure_prediction_test_examples/*.fasta`` into
``Complex`` objects. Used by both the cross-tool
integration test and the per-tool benchmarks.
"""

from __future__ import annotations

from pathlib import Path

from Bio import SeqIO

from proto_tools.tools.structure_prediction import Complex

FASTA_DIR = Path(__file__).parent.parent / "dummy_data" / "structure_prediction_test_examples"


def parse_modifications_from_header(description: str) -> list[tuple]:
    """Parse modifications from a FASTA header.

    Format: ``>name|entity_type|position:code,position:code``.

    Args:
        description (str): Full FASTA description line.

    Returns:
        list[tuple]: ``(position, code)`` pairs, or an empty list when the
        header carries no modifications.
    """
    parts = description.split("|")
    if len(parts) < 3:
        return []

    mod_string = parts[2].strip()
    if not mod_string:
        return []

    modifications = []
    for raw_mod in mod_string.split(","):
        mod = raw_mod.strip()
        if ":" in mod:
            pos_str, code = mod.split(":")
            modifications.append((int(pos_str), code.strip()))

    return modifications


def parse_fasta_to_complexes(fasta_file: Path) -> list[Complex]:
    """Parse a FASTA file into ``Complex`` objects.

    All sequences in a file are bundled into a single complex, except for
    ``two_complex.fasta`` which yields one complex per sequence.

    Args:
        fasta_file (Path): FASTA file to load.

    Returns:
        list[Complex]: Complexes parsed from the file.
    """
    chains_data = []

    for record in SeqIO.parse(str(fasta_file), "fasta"):
        sequence = str(record.seq).strip()
        parts = record.description.split("|")
        entity_type = parts[1].strip()
        modifications = parse_modifications_from_header(record.description)

        chain_dict = {
            "sequence": sequence,
            "entity_type": entity_type,
        }

        if modifications:
            chain_dict["modifications"] = modifications

        chains_data.append(chain_dict)

    if "two_complex" not in fasta_file.name:
        return [Complex(chains=chains_data)]
    return [Complex(chains=[chain_dict]) for chain_dict in chains_data]


def load_benchmark_complex(stem: str) -> Complex:
    """Load a single benchmark complex by FASTA stem.

    Args:
        stem (str): FASTA basename without ``.fasta`` (e.g. ``"trp_heterodimer"``).

    Returns:
        Complex: The first parsed complex from the file.
    """
    complexes = parse_fasta_to_complexes(FASTA_DIR / f"{stem}.fasta")
    return complexes[0]


def load_all_test_complexes() -> dict[str, list[Complex]]:
    """Pre-load every FASTA in ``FASTA_DIR`` keyed by file stem.

    Returns:
        dict[str, list[Complex]]: ``stem`` → complexes list.
    """
    return {f.stem: parse_fasta_to_complexes(f) for f in FASTA_DIR.glob("*.fasta")}

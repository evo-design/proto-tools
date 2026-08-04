"""proto_tools/tools/sequence_alignment/mmseqs2/remote_search.py.

Turning a batch of queries into alignments on disk, using the ColabFold MSA server.

The exchange with the server lives in :mod:`msa_server`; this decides what to ask for and where the
answers go. A query is either one sequence or a paired group submitted together, and the layout it
writes — ``msas/{index}.a3m`` and ``msas_paired/{index}_chain_{n}.a3m`` — is what the tool reads
back.

Based on ColabFold's own MSA search implementation
(https://github.com/sokrypton/ColabFold @ 1f8fd1a, ``colabfold/colabfold.py``, ``run_mmseqs2``),
MIT licensed.
"""

import logging
import shutil
from pathlib import Path
from typing import Any

from proto_tools.tools.sequence_alignment.mmseqs2.msa_server import run_remote_msa_search

logger = logging.getLogger(__name__)

# Unpaired sequences per submission. The endpoint accepts an unbounded batch, but one arrives as a
# single file held in memory, and an unknown server ceiling is better avoided than discovered.
MAX_SEQUENCES_PER_SUBMISSION = 10


def _a3m_query_sequence(block: bytes) -> str:
    """The block's first-record (query) sequence, uppercased and ungapped."""
    seq = bytearray()
    for line in block.split(b"\n")[1:]:  # skip the leading '>' header
        if line.startswith(b">"):
            break
        seq += line.strip()
    return seq.upper().replace(b"-", b"").decode()


def _parse_pair_a3m(pair_a3m: Path, group_seqs: list[str]) -> list[bytes]:
    r"""Split a paired `pair.a3m` into per-chain row-aligned blocks, in input chain order.

    Format observed empirically against api.colabfold.com: a single file with
    `\x00`-separated chain blocks, each block a standard A3M with the chain
    query as the first sequence. The API **deduplicates identical chains**, so a
    homo-oligomer query comes back with one block per *unique* sequence rather
    than one per chain. Map each block back onto every chain that shares its
    sequence (the blocks stay taxonomy-row-aligned, so duplicated chains pair
    correctly) and return one block per input chain.
    """
    raw = pair_a3m.read_bytes()
    blocks = [b for b in raw.split(b"\x00") if b.strip()]

    block_by_query = {_a3m_query_sequence(b): b for b in blocks}
    wanted = [s.upper() for s in group_seqs]
    unique_seqs = list(dict.fromkeys(wanted))
    if len(blocks) != len(unique_seqs) or any(seq not in block_by_query for seq in unique_seqs):
        raise RuntimeError(
            f"colabfold pair.a3m: {len(blocks)} block(s) do not match the "
            f"{len(unique_seqs)} unique chain sequence(s) in the query"
        )

    # Sanity check: equal row counts across blocks (taxonomy-paired output).
    row_counts = [b.count(b"\n>") + (1 if b.startswith(b">") else 0) for b in blocks]
    if len(set(row_counts)) != 1:
        raise RuntimeError(
            f"colabfold pair.a3m: chain blocks have unequal row counts {row_counts}; expected row-aligned paired output"
        )

    return [block_by_query[seq] for seq in wanted]


def search_remote_msas(
    queries: list[dict[str, Any]],
    output_dir: Path,
    *,
    use_metagenomic_db: bool = False,
    pairing_strategy: str = "greedy",
    client_identity: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Search alignments for a batch of queries and lay them out on disk.

    A query is either one sequence (unpaired) or a list of them (one paired group, submitted
    together so the server returns taxonomy-aligned rows). Unpaired queries go up together, in
    submissions of at most ``MAX_SEQUENCES_PER_SUBMISSION``; paired groups stay separate because
    each is already its own multi-sequence submission.

    A failure is recorded rather than raised, so a batch returns whatever it managed — the same
    contract the standalone had. Queries sharing a submission share its failure, so a failed
    submission is recorded against each of its own queries and leaves the others alone.

    Args:
        queries (list[dict[str, Any]]): Each ``{"sequences": str}`` or ``{"sequences": [str, ...]}``.
        output_dir (Path): Directory to write ``msas/`` and ``msas_paired/`` under.
        use_metagenomic_db (bool): Also search the metagenomic database.
        pairing_strategy (str): How paired groups are paired; ignored for single sequences.
        client_identity (str | None): Who asked, when the call came from elsewhere.
        timeout (float | None): Seconds to wait per submission; ``None`` waits indefinitely.

    Returns:
        dict[str, Any]: Paths per query, plus counts and any per-query errors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    msas_dir = output_dir / "msas"
    msas_dir.mkdir(exist_ok=True)

    msa_paths: dict[str, str] = {}
    paired_msa_paths: dict[str, list[str]] = {}
    errors: list[tuple[str, str]] = []

    unpaired: list[tuple[int, str]] = []
    for index, query in enumerate(queries):
        sequences = query["sequences"]
        if isinstance(sequences, list):
            label = f"query_{index}"
            try:
                paired_dir = output_dir / "msas_paired"
                paired_dir.mkdir(exist_ok=True)
                paired_msa_paths[str(index)] = _write_paired_group(
                    sequences,
                    index,
                    output_dir,
                    paired_dir,
                    use_metagenomic_db,
                    pairing_strategy,
                    client_identity,
                    timeout,
                )
            except Exception as exc:
                logger.debug("Remote MSA search failed for %s: %s", label, exc)
                errors.append((label, f"Failed to generate MSA for {label}: {exc!s}"))
        else:
            unpaired.append((index, sequences))

    for start in range(0, len(unpaired), MAX_SEQUENCES_PER_SUBMISSION):
        batch = unpaired[start : start + MAX_SEQUENCES_PER_SUBMISSION]
        try:
            msa_paths.update(
                _write_unpaired_batch(batch, start, output_dir, msas_dir, use_metagenomic_db, client_identity, timeout)
            )
        except Exception as exc:
            logger.debug("Remote MSA search failed for %d unpaired quer(ies): %s", len(batch), exc)
            for index, _ in batch:
                label = f"query_{index}"
                errors.append((label, f"Failed to generate MSA for {label}: {exc!s}"))

    result: dict[str, Any] = {
        "msa_paths": msa_paths,
        "paired_msa_paths": paired_msa_paths,
        "success": bool(msa_paths or paired_msa_paths),
        "num_successful": len(msa_paths) + len(paired_msa_paths),
        "num_failed": len(errors),
    }
    if errors:
        result["errors"] = dict(errors)
    return result


def _blocks_by_query(alignment: Path) -> dict[str, bytes]:
    r"""Split one returned alignment into per-query blocks, keyed by the query each one answers.

    A multi-sequence submission comes back as a single a3m whose queries are ``\x00``-separated,
    the same shape as a paired result. Keying by query sequence rather than position is what makes
    duplicates safe: the submission deduplicates, so two identical queries share one block.

    Keys are uppercased and ungapped, so two queries differing only in case would collide here while
    the submission treats them as distinct. Queries reaching this point are already uppercase amino
    acids, which is the same assumption :func:`_parse_pair_a3m` makes.
    """
    raw = alignment.read_bytes()
    return {_a3m_query_sequence(block): block for block in raw.split(b"\x00") if block.strip()}


def _write_unpaired_batch(
    items: list[tuple[int, str]],
    offset: int,
    output_dir: Path,
    msas_dir: Path,
    use_metagenomic_db: bool,
    client_identity: str | None,
    timeout: float | None,
) -> dict[str, str]:
    """Search one submission's worth of unpaired queries and write each alignment into ``msas/``.

    A metagenomic search returns **two** alignments — the UniRef one and the environmental one —
    which belong together as a single deeper alignment for the query. Taking whichever the
    filesystem listed first would silently return one of them, and since ``bfd.*`` sorts ahead of
    ``uniref.*`` that would mean discarding UniRef exactly when the caller asked for more depth.

    Args:
        items (list[tuple[int, str]]): ``(query index, sequence)`` for the queries in this submission.
        offset (int): Position of this submission within the unpaired queries; names its directory.
        output_dir (Path): Directory the search unpacks under.
        msas_dir (Path): Directory the per-query alignments are written to.
        use_metagenomic_db (bool): Whether the environmental database was searched too.
        client_identity (str | None): Who asked, when the call came from elsewhere.
        timeout (float | None): Seconds to wait for this submission; ``None`` waits indefinitely.

    Returns:
        dict[str, str]: Query index (as a string) to the alignment written for it.

    Raises:
        RuntimeError: If the server returned no block for one of the submitted queries.
    """
    results = run_remote_msa_search(
        [sequence for _, sequence in items],
        output_dir / f"unpaired_{offset}",
        use_env=use_metagenomic_db,
        client_identity=client_identity,
        timeout=timeout,
    )
    try:
        per_alignment = [_blocks_by_query(path) for path in _alignment_files(results, use_metagenomic_db)]

        # Resolve every query before writing any of them, so a submission either lands whole or
        # leaves nothing: alignments on disk for queries the caller is told failed would be
        # indistinguishable from successes to anything reading the directory rather than the paths.
        resolved: list[tuple[int, bytes]] = []
        for index, sequence in items:
            query = sequence.upper()
            blocks = []
            for by_query in per_alignment:
                if query not in by_query:
                    raise RuntimeError(f"remote search returned no alignment for query {index}")
                blocks.append(by_query[query])
            resolved.append((index, b"".join(blocks)))

        written: dict[str, str] = {}
        for index, payload in resolved:
            destination = msas_dir / f"{index}.a3m"
            destination.write_bytes(payload)
            written[str(index)] = str(destination)
        return written
    finally:
        shutil.rmtree(results, ignore_errors=True)


# What the server names its unpaired results, UniRef first: the order the alignments are
# concatenated in, and the order ColabFold's own client reads them.
_UNIREF_ALIGNMENT = "uniref.a3m"
_ENVIRONMENTAL_ALIGNMENT = "bfd.mgnify30.metaeuk30.smag30.a3m"


def _alignment_files(results: Path, use_metagenomic_db: bool) -> list[Path]:
    """The alignments a search produced, in the order they should be joined.

    Falls back to whatever ``.a3m`` files are present if the server has renamed them, so a rename
    degrades to the old behaviour rather than failing outright.

    Args:
        results (Path): Directory the search unpacked into.
        use_metagenomic_db (bool): Whether the environmental database was searched too.

    Returns:
        list[Path]: Alignment files, UniRef first.

    Raises:
        RuntimeError: If the search produced no alignment at all.
    """
    expected = [_UNIREF_ALIGNMENT, *([_ENVIRONMENTAL_ALIGNMENT] if use_metagenomic_db else [])]
    named = [results / name for name in expected if (results / name).is_file()]
    if named:
        return named
    if found := sorted(results.rglob("*.a3m")):
        logger.warning("Remote search returned no %s; falling back to %s", expected, [p.name for p in found])
        return found
    raise RuntimeError(f"remote search returned no .a3m file in {results}")


def _write_paired_group(
    sequences: list[str],
    index: int,
    output_dir: Path,
    paired_dir: Path,
    use_metagenomic_db: bool,
    pairing_strategy: str,
    client_identity: str | None,
    timeout: float | None,
) -> list[str]:
    """Search one paired group and split it into per-chain, row-aligned alignments."""
    results = run_remote_msa_search(
        sequences,
        output_dir / f"pair_group_{index}",
        use_env=use_metagenomic_db,
        use_pairing=True,
        pairing_strategy=pairing_strategy,
        client_identity=client_identity,
        timeout=timeout,
    )
    try:
        pair_a3m = results / "pair.a3m"
        if not pair_a3m.exists():
            candidates = sorted(results.rglob("*.a3m"))
            if not candidates:
                raise RuntimeError(f"paired search returned no .a3m file in {results}")
            pair_a3m = candidates[0]

        written: list[str] = []
        for chain_index, block in enumerate(_parse_pair_a3m(pair_a3m, sequences)):
            chain_a3m = paired_dir / f"{index}_chain_{chain_index}.a3m"
            chain_a3m.write_bytes(block)
            written.append(str(chain_a3m))
        return written
    finally:
        shutil.rmtree(results, ignore_errors=True)

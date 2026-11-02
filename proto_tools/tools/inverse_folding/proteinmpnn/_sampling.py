"""proto_tools/tools/inverse_folding/proteinmpnn/_sampling.py.

Shared sampling machinery for the ProteinMPNN sampling tools.

``proteinmpnn-sample`` and ``proteinmpnn-sample-homooligomer`` differ only in how
they resolve chains and fixed positions; the staging, chunked dispatch, and
segment handling below are common to both and live here so the two cannot drift.
"""

import logging
import random
from typing import Any

import numpy as np

from proto_tools.entities.structures import Structure
from proto_tools.entities.structures.selection import ResidueSelection
from proto_tools.utils import ToolInstance

logger = logging.getLogger(__name__)


def collect_samples(
    inp: Any,
    config: Any,
    *,
    chain_ids: list[str],
    fixed_positions: ResidueSelection | None,
    homooligomer: bool,
    seed_rng: random.Random,
    instance: Any,
) -> tuple[list[str], list[float], list[float]]:
    """Stage one structure and sample from it in ``batch_size`` chunks.

    Args:
        inp (Any): Structure input exposing ``staged()``.
        config (Any): Sampling config supplying temperature, model choice, and batching.
        chain_ids (list[str]): Chains to redesign, in output segment order.
        fixed_positions (ResidueSelection | None): Positions held at their input identity.
        homooligomer (bool): Whether to tie corresponding positions across ``chain_ids``.
        seed_rng (random.Random): Draws a fresh seed per chunk.
        instance (Any): ToolInstance, instance name, or None for one-shot execution.

    Returns:
        tuple[list[str], list[float], list[float]]: Raw sequences, perplexities, and
            sequence recoveries, each with one entry per generated design.
    """
    all_seqs: list[str] = []
    all_perp: list[float] = []
    all_seq_recovery: list[float] = []
    remaining = config.num_sequences_per_structure

    # Staged once per input and reused across chunks, numbered 1..N per chain so ProteinMPNN,
    # which addresses residues by number alone, sees every insertion-coded residue.
    with inp.staged() as staged:
        while remaining > 0:
            chunk = min(config.batch_size, remaining)
            input_dict = {
                "operation": "sample",
                "pdb_path": str(staged.path),
                "chain_ids": staged.chains(chain_ids),
                "batch_size": chunk,
                "temperature": config.temperature,
                "fixed_positions": staged.positions(fixed_positions),
                "excluded_amino_acids": config.excluded_amino_acids,
                "seed": seed_rng.randint(0, 2**31 - 1),
                "device": config.device,
                "model_choice": config.model_choice,
                "verbose": config.verbose,
                "return_logits": False,
                "backbone_noise": config.backbone_noise,
                "homooligomer": homooligomer,
            }
            result = ToolInstance.dispatch("proteinmpnn", input_dict, instance=instance, config=config)
            all_seqs.extend(result["seq"])
            all_perp.extend(np.exp(result["score"]).tolist())
            all_seq_recovery.extend(result["seqid"])
            remaining -= chunk

    return all_seqs, all_perp, all_seq_recovery


def split_chain_segments(raw_seq: str, chain_ids: list[str], *, tied: bool) -> list[str]:
    """Split one raw ProteinMPNN sequence into one segment per redesigned chain.

    Args:
        raw_seq (str): Slash-separated sequence as returned by the worker.
        chain_ids (list[str]): Redesigned chains, in segment order.
        tied (bool): Whether positions were tied across ``chain_ids``.

    Returns:
        list[str]: One sequence segment per entry of ``chain_ids``.

    Raises:
        ValueError: If the worker returned a segment count that cannot be matched
            to ``chain_ids``.
    """
    segments = raw_seq.split("/")
    if tied:
        # Tied decoding always collapses the oligomer to exactly one protomer segment
        # (ColabDesign's _get_seq keeps only seq.split("/")[0]), so anything else means the
        # tie did not happen. Accepting a per-chain split here would silently return
        # independently designed protomers from a tool whose whole promise is that they match.
        if len(segments) != 1:
            raise ValueError(
                f"Tied ProteinMPNN decoding must return exactly one protomer segment, but got "
                f"{len(segments)} for protomer chains {chain_ids}. Raw sequence: {raw_seq!r}."
            )
        return segments * len(chain_ids)
    if len(segments) != len(chain_ids):
        raise ValueError(
            f"ProteinMPNN returned {len(segments)} sequence segment(s) but "
            f"{len(chain_ids)} chain(s) were requested for redesign "
            f"({chain_ids}). Raw sequence: {raw_seq!r}."
        )
    return segments


def looks_like_homooligomer(structure: Structure, chain_ids: list[str]) -> bool:
    """Report whether the redesigned chains are a symmetric homooligomer.

    Args:
        structure (Structure): Structure the chains belong to.
        chain_ids (list[str]): Chains selected for redesign.

    Returns:
        bool: True when at least two chains are selected and all of them carry
            the same sequence. Equal length alone is not enough, since unrelated
            chains of matching length are not an oligomer.
    """
    if len(chain_ids) < 2:
        return False
    return len({structure.get_chain_sequence(cid) for cid in chain_ids}) == 1

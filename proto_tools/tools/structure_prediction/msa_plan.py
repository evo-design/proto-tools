"""Per-chain MSA inputs for predictor writers, and the record of what each writer sends.

Writers walk ``protein_chain_msas`` to decide what to write for every protein chain and return a
``SentMSA`` per chain describing the rows the backend actually receives. The conditioning report
is built from those records, so it cannot drift from what the writers do.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.msa import MSA
from proto_tools.tools.structure_prediction.shared_data_models import Chain, ComplexMSAs, resolve_chain_ids
from proto_tools.utils import extract_msa_sequences


@dataclass(frozen=True)
class ChainMSAs:
    """One protein chain's alignments from a ``ComplexMSAs`` bundle.

    Attributes:
        index (int): Zero-based position of the chain in the complex.
        chain_id (str): The chain's resolved ID.
        chain (Chain): The chain itself.
        primary (MSA | None): The chain's primary (paired, when the bundle is paired) alignment.
        unpaired (MSA | None): The chain's deep unpaired alignment, for paired bundles.
    """

    index: int
    chain_id: str
    chain: Chain
    primary: MSA | None
    unpaired: MSA | None


@dataclass(frozen=True)
class SentMSA:
    """The MSA rows a writer handed the backend for one protein chain.

    Attributes:
        rows (int): Primary rows sent, query included: the paired rows for a paired bundle, else
            the alignment.
        unpaired_rows (int): Unpaired homolog rows sent alongside the primary rows, query excluded.
        paired (bool): Whether the primary rows are paired across chains.
        ignored_reason (str | None): Why the backend does not use what it was sent, or why
            nothing was sent for a chain that has rows.
    """

    rows: int = 0
    unpaired_rows: int = 0
    paired: bool = False
    ignored_reason: str | None = None


def protein_chain_msas(chains: Sequence[Chain | Fragment], complex_msas: ComplexMSAs | None) -> list[ChainMSAs]:
    """Every protein chain with its primary and unpaired alignments (``None`` when absent)."""
    chain_ids = resolve_chain_ids(chains)
    per_chain = complex_msas.per_chain if complex_msas is not None else {}
    unpaired = (complex_msas.unpaired_per_chain or {}) if complex_msas is not None else {}
    return [
        ChainMSAs(index, chain_ids[index], chain, per_chain.get(index), unpaired.get(index))
        for index, chain in enumerate(chains)
        if isinstance(chain, Chain) and chain.entity_type == "protein"
    ]


def unpaired_depth_rows(rows: Sequence[str], unpaired: MSA | None) -> list[tuple[str, str]]:
    """``(sequence, id)`` unpaired rows to append beneath ``rows``: no query, no duplicate sequences."""
    if unpaired is None:
        return []
    sequences, ids = extract_msa_sequences(unpaired, 0)
    seen = set(rows)
    appended = []
    for sequence, sequence_id in zip(sequences[1:], ids[1:], strict=True):
        if sequence not in seen:
            seen.add(sequence)
            appended.append((sequence, sequence_id))
    return appended


def short_chain_reason(chain: Chain, backend: str, max_skipped_length: int = 4) -> str | None:
    """The ignored reason for backends that featurize no MSA for chains this short, else ``None``."""
    if len(chain.sequence) <= max_skipped_length:
        return f"{backend} skips the MSA of chains with {max_skipped_length} or fewer residues"
    return None


def has_homolog_rows(record: SentMSA) -> bool:
    """Whether a chain was sent any rows beyond its query."""
    return record.rows > 1 or record.unpaired_rows > 0


def conditions_on_msa(sent: Mapping[int, SentMSA]) -> bool:
    """Whether any chain was sent rows beyond its query that the backend uses."""
    return any(record.ignored_reason is None and has_homolog_rows(record) for record in sent.values())

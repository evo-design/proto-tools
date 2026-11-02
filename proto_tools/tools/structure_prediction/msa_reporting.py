"""Build truthful requested-versus-effective MSA conditioning reports."""

from collections.abc import Collection, Mapping, Sequence

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures.msa_conditioning import (
    MSA_SEARCH_SOURCES,
    ChainMSAConditioning,
    MSAConditioning,
    MSAMode,
    MSARequestedSource,
    MSAStatus,
    MSASubsampling,
)
from proto_tools.tools.structure_prediction.shared_data_models import Chain, ComplexMSAs, resolve_chain_ids


def build_msa_conditioning(
    chains: Sequence[Chain | Fragment],
    complex_msas: ComplexMSAs | None,
    *,
    consumed: bool | Collection[int] = True,
    subsampling: MSASubsampling | None = None,
    ignored_reasons: Mapping[int, str] | None = None,
    paired: bool | None = None,
) -> MSAConditioning:
    """Describe each protein chain's requested MSA and whether the backend consumes it.

    Requested chains are the bundle's ``requested_chain_indices``, else the chains it
    carries rows for. ``consumed`` names the chains whose rows the backend featurizes.
    ``paired`` overrides the bundle's flag for a backend that pairs rows its own way.
    """
    protein_indices = [
        index for index, chain in enumerate(chains) if isinstance(chain, Chain) and chain.entity_type == "protein"
    ]
    chain_ids = resolve_chain_ids(chains)
    present_indices: set[int] = set()
    if complex_msas is not None:
        present_indices.update(complex_msas.per_chain)
        present_indices.update((complex_msas.unpaired_per_chain or {}).keys())

    if complex_msas is not None and complex_msas.requested_chain_indices is not None:
        requested_indices = set(complex_msas.requested_chain_indices)
    else:
        requested_indices = present_indices

    consumed_indices = set(protein_indices) if consumed is True else set() if consumed is False else set(consumed)
    source: MSARequestedSource = complex_msas.source if complex_msas is not None else "none"
    ignored_reasons = ignored_reasons or {}

    evidence: list[ChainMSAConditioning] = []
    for index in protein_indices:
        primary = complex_msas.per_chain.get(index) if complex_msas is not None else None
        unpaired = (complex_msas.unpaired_per_chain or {}).get(index) if complex_msas is not None else None
        rows = len(primary) if primary is not None else 0
        unpaired_rows = len(unpaired) if unpaired is not None else 0
        is_requested = index in requested_indices
        is_consumed = index in consumed_indices

        status: MSAStatus
        if not is_requested:
            status = "not_requested"
        elif source in MSA_SEARCH_SOURCES and max(rows, unpaired_rows) <= 1:
            status = "no_hits"
        elif not is_consumed or max(rows, unpaired_rows) == 0:
            status = "ignored"
        else:
            status = "consumed"

        mode: MSAMode
        is_paired = paired if paired is not None else complex_msas is not None and complex_msas.paired
        if is_paired and rows:
            mode = "paired"
        elif rows or unpaired_rows:
            mode = "unpaired"
        else:
            mode = "none"

        evidence.append(
            ChainMSAConditioning(
                chain_index=index,
                chain_id=chain_ids[index],
                status=status,
                rows=rows,
                unpaired_rows=unpaired_rows,
                requested_source=source if is_requested else "none",
                mode=mode,
                ignored_reason=ignored_reasons.get(index) if status == "ignored" else None,
            )
        )

    return MSAConditioning(
        chains=evidence,
        subsampling=subsampling or MSASubsampling(mode="unknown" if requested_indices else "none"),
    )


def msa_consumption(
    chains: Sequence[Chain | Fragment],
    complex_msas: ComplexMSAs | None,
    *,
    backend: str,
    max_skipped_length: int | None = None,
    sends_unpaired_only_chains: bool = False,
) -> tuple[set[int], dict[int, str]]:
    """Consumed chain indices and ignored reasons for the alignments a backend's writer sends.

    Most writers emit one file per chain from ``per_chain``, so a chain with only unpaired rows
    is never seen; ``sends_unpaired_only_chains`` marks writers (Protenix, OpenDDE) that send
    those rows instead. AlphaFold3, Protenix, and OpenDDE also featurize no MSA for chains of
    ``max_skipped_length`` or fewer residues.
    """
    if complex_msas is None:
        return set(), {}
    reasons: dict[int, str] = {}
    with_rows = set(complex_msas.per_chain) | set(complex_msas.unpaired_per_chain or {})
    for index in with_rows:
        chain = chains[index]
        if index not in complex_msas.per_chain and not sends_unpaired_only_chains:
            reasons[index] = f"only unpaired rows were supplied; {backend} is sent a chain's primary alignment"
        elif max_skipped_length is not None and isinstance(chain, Chain) and len(chain.sequence) <= max_skipped_length:
            reasons[index] = f"{backend} skips the MSA of chains with {max_skipped_length} or fewer residues"
    sent = with_rows if sends_unpaired_only_chains else set(complex_msas.per_chain)
    return sent - set(reasons), reasons

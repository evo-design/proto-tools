"""Build MSA conditioning reports from what each predictor's writer sent its backend."""

import logging
from collections.abc import Mapping, Sequence

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures.msa_conditioning import (
    MSA_SEARCH_SOURCES,
    ChainMSAConditioning,
    MSAConditioning,
    MSAMode,
    MSAStatus,
    MSASubsampling,
)
from proto_tools.tools.structure_prediction.msa_plan import (
    SentMSA,
    conditions_on_msa,
    has_homolog_rows,
    protein_chain_msas,
)
from proto_tools.tools.structure_prediction.shared_data_models import Chain, ComplexMSAs

logger = logging.getLogger(__name__)

# (tool, reason) pairs already warned about; repeats log at DEBUG so batches and sweeps warn once.
_WARNED_UNUSED: set[tuple[str | None, str]] = set()


def build_msa_conditioning(
    chains: Sequence[Chain | Fragment],
    complex_msas: ComplexMSAs | None,
    sent: Mapping[int, SentMSA],
    *,
    subsampling: MSASubsampling | None = None,
    tool: str | None = None,
) -> MSAConditioning:
    """Describe each protein chain's MSA from the rows its writer sent the backend.

    A chain is ``consumed`` when the backend was sent rows beyond the query and uses them. A chain
    that was not is ``not_requested`` when the bundle asked nothing for it, ``no_hits`` when our own
    search found no homologs, and ``ignored`` otherwise. Requested chains are the bundle's
    ``requested_chain_indices``, else the chains it carries rows for. ``subsampling`` is the
    backend's row-selection policy; it is reported as ``none`` when nothing conditions on an MSA.
    Homolog rows a chain had but the backend did not use are logged as a warning naming ``tool``,
    once per tool and reason per process; repeats log at DEBUG.
    """
    # Find the chains the bundle carries rows for and the chains it asked an MSA for
    present = set(complex_msas.per_chain) | set(complex_msas.unpaired_per_chain or {}) if complex_msas else set()
    if complex_msas is not None and complex_msas.requested_chain_indices is not None:
        requested = set(complex_msas.requested_chain_indices)
    else:
        requested = present
    source = complex_msas.source if complex_msas is not None else "none"

    evidence: list[ChainMSAConditioning] = []
    unused: list[tuple[str, str]] = []
    # Pairing is across chains: a chain whose paired rows have no partner chain is not paired
    partnered = sum(r.paired and r.rows > 1 and r.ignored_reason is None for r in sent.values()) >= 2
    # Classify each protein chain by what its writer sent the backend
    for chain in protein_chain_msas(chains, complex_msas):
        record = sent.get(chain.index, SentMSA())
        # Count the rows the bundle held for this chain, query included
        bundle_rows = max(len(chain.primary or []), len(chain.unpaired or []))
        # Consumed when the backend used homolog rows; otherwise record why not
        status: MSAStatus
        if record.ignored_reason is None and has_homolog_rows(record):
            status = "consumed"
        elif chain.index not in requested:
            status = "not_requested"
        elif source in MSA_SEARCH_SOURCES and bundle_rows <= 1:
            status = "no_hits"
        else:
            status = "ignored"
        # Collect chains whose homolog rows went unused, to warn about below
        reason = record.ignored_reason or "no alignment rows reached the backend"
        if status == "ignored" and bundle_rows > 1:
            unused.append((chain.chain_id, reason))
        # Report paired only when a partner chain was also paired; other homolog rows count as unpaired
        mode: MSAMode = (
            "paired"
            if partnered and record.paired and record.rows > 1 and record.ignored_reason is None
            else "unpaired"
            if has_homolog_rows(record)
            else "none"
        )
        evidence.append(
            ChainMSAConditioning(
                chain_index=chain.index,
                chain_id=chain.chain_id,
                status=status,
                rows=record.rows,
                unpaired_rows=record.unpaired_rows,
                requested_source=source if status != "not_requested" else "none",
                mode=mode,
                ignored_reason=reason if status == "ignored" else None,
            )
        )
    # Warn once per tool and reason, naming every chain it applies to; repeats log at DEBUG
    for reason in dict.fromkeys(reason for _chain_id, reason in unused):
        chain_ids = ", ".join(chain_id for chain_id, chain_reason in unused if chain_reason == reason)
        level = logging.DEBUG if (tool, reason) in _WARNED_UNUSED else logging.WARNING
        _WARNED_UNUSED.add((tool, reason))
        logger.log(
            level, "%sMSA rows for chain(s) %s were not used: %s", f"{tool}: " if tool else "", chain_ids, reason
        )
    # A backend's row-selection policy only applies when something conditions on an MSA.
    if not conditions_on_msa(sent):
        subsampling = MSASubsampling(mode="none")
    return MSAConditioning(chains=evidence, subsampling=subsampling or MSASubsampling(mode="unknown"))


def describe_msa_conditioning(report: MSAConditioning) -> str:
    """One-line summary of what each protein chain sent the backend, for logs."""
    parts = []
    for chain in report.chains:
        if chain.status == "consumed":
            rows = f"{chain.rows}+{chain.unpaired_rows}" if chain.unpaired_rows else f"{chain.rows}"
            parts.append(f"{chain.chain_id}={rows} rows ({chain.requested_source}, {chain.mode})")
        else:
            parts.append(f"{chain.chain_id}={chain.status}")
    selection = report.subsampling
    cap = f" <= {selection.max_rows}" if selection.max_rows else ""
    return f"{'; '.join(parts) or 'no protein chains'}; selection {selection.mode}{cap}"

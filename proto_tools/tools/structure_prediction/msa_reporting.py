"""Build MSA conditioning reports from what each predictor's writer sent its backend."""

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


def build_msa_conditioning(
    chains: Sequence[Chain | Fragment],
    complex_msas: ComplexMSAs | None,
    sent: Mapping[int, SentMSA],
    *,
    subsampling: MSASubsampling | None = None,
) -> MSAConditioning:
    """Describe each protein chain's MSA from the rows its writer sent the backend.

    A chain is ``consumed`` when the backend was sent rows beyond the query and uses them. A chain
    that was not is ``not_requested`` when the bundle asked nothing for it, ``no_hits`` when our own
    search found no homologs, and ``ignored`` otherwise. Requested chains are the bundle's
    ``requested_chain_indices``, else the chains it carries rows for. ``subsampling`` is the
    backend's row-selection policy; it is reported as ``none`` when nothing conditions on an MSA.
    """
    present = set(complex_msas.per_chain) | set(complex_msas.unpaired_per_chain or {}) if complex_msas else set()
    if complex_msas is not None and complex_msas.requested_chain_indices is not None:
        requested = set(complex_msas.requested_chain_indices)
    else:
        requested = present
    source = complex_msas.source if complex_msas is not None else "none"

    evidence: list[ChainMSAConditioning] = []
    for chain in protein_chain_msas(chains, complex_msas):
        record = sent.get(chain.index, SentMSA())
        bundle_rows = max(len(chain.primary or []), len(chain.unpaired or []))
        status: MSAStatus
        if record.ignored_reason is None and has_homolog_rows(record):
            status = "consumed"
        elif chain.index not in requested:
            status = "not_requested"
        elif source in MSA_SEARCH_SOURCES and bundle_rows <= 1:
            status = "no_hits"
        else:
            status = "ignored"
        mode: MSAMode = (
            "paired" if record.paired and record.rows > 1 else "unpaired" if has_homolog_rows(record) else "none"
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
                ignored_reason=(record.ignored_reason or "no alignment rows reached the backend")
                if status == "ignored"
                else None,
            )
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

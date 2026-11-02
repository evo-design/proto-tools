"""Requested and backend-effective MSA conditioning evidence."""

from typing import Literal, get_args

from pydantic import BaseModel, Field, computed_field

MSAStatus = Literal["not_requested", "no_hits", "consumed", "ignored"]
# Where proto-tools obtained an alignment it searched for itself: ColabFold's hosted MSA API
# (mmseqs2-homology-search search_mode="remote") or MMseqs2 against a local database ("local").
MSASearchSource = Literal["colabfold_server", "local_mmseqs2"]
MSA_SEARCH_SOURCES: frozenset[str] = frozenset(get_args(MSASearchSource))
MSABundleSource = Literal["supplied", "colabfold_server", "local_mmseqs2"]
MSARequestedSource = Literal["none", "supplied", "colabfold_server", "local_mmseqs2"]
MSAMode = Literal["none", "paired", "unpaired"]


class ChainMSAConditioning(BaseModel):
    """The MSA rows sent to the backend for one protein chain, before its own subsampling.

    Attributes:
        chain_index (int): Zero-based position in the input complex.
        chain_id (str): Explicit input ID, or its positional chain label.
        status (MSAStatus): Whether an MSA was not requested, found no homologs, or was
            consumed or ignored by the backend.
        rows (int): Primary rows sent to the backend, including the query.
        unpaired_rows (int): Unpaired homolog rows sent alongside the primary rows, query excluded.
        requested_source (MSARequestedSource):
            Where rows came from: supplied, a ColabFold server or local MMseqs2 search, or none.
        mode (MSAMode): Whether the chain's rows are paired across chains, unpaired, or absent.
        ignored_reason (str | None): Why a requested alignment was not consumed.
    """

    chain_index: int = Field(ge=0, title="Chain Index", description="Zero-based input complex chain index.")
    chain_id: str = Field(title="Chain ID", description="Input chain ID or positional label.")
    status: MSAStatus = Field(
        title="Status", description="Whether the MSA was not requested, found no hits, or was consumed or ignored."
    )
    rows: int = Field(
        ge=0, title="Rows", description="Primary rows sent to the backend, including the query, before its subsampling."
    )
    unpaired_rows: int = Field(
        ge=0,
        title="Unpaired Rows",
        description="Unpaired homolog rows sent alongside the primary rows, query excluded.",
    )
    requested_source: MSARequestedSource = Field(
        title="Requested Source", description="Origin of the chain's MSA rows, or none when not requested."
    )
    mode: MSAMode = Field(title="Mode", description="Whether the chain's rows are paired, unpaired, or absent.")
    ignored_reason: str | None = Field(
        default=None,
        title="Ignored Reason",
        description="Reason requested alignment evidence was not consumed.",
    )


class MSASubsampling(BaseModel):
    """Backend MSA row-selection policy.

    Attributes:
        mode (Literal["none", "deterministic", "stochastic", "unknown"]): Whether rows are
            not selected, capped deterministically, drawn at random, or selected in an
            unreported way.
        max_rows (int | None): Primary row cap when one is known.
        max_extra_rows (int | None): Separate extra-row cap, for backends with an extra-MSA pool.
    """

    mode: Literal["none", "deterministic", "stochastic", "unknown"] = Field(
        default="unknown",
        title="Mode",
        description="Backend MSA row-selection policy.",
    )
    max_rows: int | None = Field(
        default=None,
        ge=1,
        title="Maximum Rows",
        description="Primary row cap when the backend's is known.",
    )
    max_extra_rows: int | None = Field(
        default=None,
        ge=1,
        title="Maximum Extra Rows",
        description="Extra-row cap for backends with a separate extra-MSA pool.",
    )


class MSAConditioning(BaseModel):
    """Per-complex MSA evidence, independent of transport logs or requested settings.

    Attributes:
        chains (list[ChainMSAConditioning]): Protein-chain evidence in input order.
        subsampling (MSASubsampling): Backend row-selection policy.
        coverage (Literal["full", "partial", "none", "not_applicable"]): Protein chains whose
            consumed alignment has rows beyond the query; derived from ``chains``.
        pairing_mode (Literal["none", "paired", "unpaired", "mixed"]): Pairing of the consumed
            alignments; derived from ``chains``.
    """

    chains: list[ChainMSAConditioning] = Field(title="Chains", description="Per-protein-chain alignment evidence.")
    subsampling: MSASubsampling = Field(
        default_factory=MSASubsampling,
        title="Subsampling",
        description="Backend MSA row-selection policy.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coverage(self) -> Literal["full", "partial", "none", "not_applicable"]:
        """Whether every, some, or no protein chain was conditioned on rows beyond its query."""
        if not self.chains:
            return "not_applicable"
        covered = sum(
            chain.status == "consumed" and (chain.rows > 1 or chain.unpaired_rows > 0) for chain in self.chains
        )
        return "none" if not covered else "full" if covered == len(self.chains) else "partial"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pairing_mode(self) -> Literal["none", "paired", "unpaired", "mixed"]:
        """The consumed chains' shared pairing mode, ``mixed`` when they differ, else ``none``."""
        modes = {chain.mode for chain in self.chains if chain.status == "consumed" and chain.mode != "none"}
        if not modes:
            return "none"
        return modes.pop() if len(modes) == 1 else "mixed"

"""InterPro domain annotation retrieval (UniProt accession or sequence)."""

from proto_tools.tools.database_retrieval.interproscan.interproscan_fetch import (
    InterProApp,
    InterProDomain,
    InterProDomainType,
    InterProScanFetchConfig,
    InterProScanFetchInput,
    InterProScanFetchOutput,
    run_interproscan_fetch,
)

__all__ = [
    "InterProApp",
    "InterProDomain",
    "InterProDomainType",
    "InterProScanFetchConfig",
    "InterProScanFetchInput",
    "InterProScanFetchOutput",
    "run_interproscan_fetch",
]

"""Log how many MSA rows a backend actually loaded, and warn when supplied rows never reach it.

Standalones count rows per chain from the backend's own featurized MSA (the rows that are not
all-gap within a chain's columns) and compare them with the rows proto-tools sent. A chain that
was sent an alignment but loaded only its query indicates that the MSA was dropped. Log this
at WARNING so it is visible in production. Log other row counts in a single DEBUG line.
"""

from __future__ import annotations

from collections.abc import Mapping

from .proto_logging import get_logger

logger = get_logger(__name__)


def count_a3m_rows(a3m: str | None) -> int:
    """Number of records in an A3M/FASTA string (0 for ``None`` or empty)."""
    if not a3m:
        return 0
    return sum(1 for line in a3m.splitlines() if line.startswith(">"))


def report_msa_rows_loaded(
    tool: str,
    loaded: Mapping[str, int],
    sent: Mapping[str, int],
    paired: Mapping[str, int] | None = None,
) -> None:
    """Log per-chain loaded MSA rows; warn for chains sent an alignment that loaded only the query.

    Args:
        tool (str): Toolkit name used as the message prefix.
        loaded (Mapping[str, int]): Chain ID to rows the backend featurized, query included.
        sent (Mapping[str, int]): Chain ID to rows proto-tools wrote for it, query included.
        paired (Mapping[str, int] | None): Chain ID to loaded rows paired across chains, query
            excluded, for backends that expose it.
    """
    if not sent and not loaded:
        return
    summary = ", ".join(
        f"{chain}={loaded.get(chain, 0)}/{sent.get(chain, 0)}" + (f" ({paired.get(chain, 0)} paired)" if paired else "")
        for chain in sorted({*sent, *loaded})
    )
    logger.debug("%s MSA rows loaded/sent per chain: %s", tool, summary)
    dropped = sorted(chain for chain, rows in sent.items() if rows > 1 and loaded.get(chain, 0) <= 1)
    if dropped:
        logger.warning(
            "%s: supplied MSA did not reach the model for chain(s) %s; it loaded only the query",
            tool,
            ", ".join(dropped),
        )

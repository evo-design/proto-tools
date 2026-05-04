#!/usr/bin/env python3
"""scripts/smoke_interproscan_fetch.py.

Real-user smoke test for the interproscan-fetch tool.

Calls ``run_interproscan_fetch`` against a panel of well-known UniProt
accessions, exercising the direct-lookup path against the live EBI API,
and prints a per-protein summary plus a small concurrency-stress block.

Optionally exercises the slow submit-and-scan path via
``--submit-path`` against the first 200 residues of TP53.

Usage::

    # Direct-lookup workload (fast: ~10s for 8 proteins sequentially)
    python scripts/smoke_interproscan_fetch.py

    # Add a small concurrent stress block (4 in parallel) to verify
    # urllib3 retry + EBI rate limits behave under burst load
    python scripts/smoke_interproscan_fetch.py --stress

    # Add the slow submit-and-scan probe (5-30 min depending on queue)
    python scripts/smoke_interproscan_fetch.py --submit-path

Exits non-zero on any failure.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from proto_tools.tools.database_retrieval import (
    InterProScanFetchConfig,
    InterProScanFetchInput,
    run_interproscan_fetch,
)


@dataclass(frozen=True)
class Probe:
    """One protein in the smoke panel."""

    uniprot_id: str
    label: str
    expected_length: int
    expected_member_db: str
    expected_name_substring: str  # case-insensitive substring of some hit's `name`


_PANEL: list[Probe] = [
    Probe("P04637", "TP53 (tumor suppressor)", 393, "pfam", "p53"),
    Probe("P38398", "BRCA1 (DNA repair)", 1863, "pfam", "brct"),
    Probe("P01116", "KRAS (Ras GTPase)", 189, "pfam", "ras"),
    Probe("P00533", "EGFR (RTK)", 1210, "pfam", "kinase"),
    Probe("P31749", "AKT1 (Ser/Thr kinase)", 480, "pfam", "kinase"),
    Probe("P10415", "BCL2 (apoptosis)", 239, "pfam", "bcl"),
    Probe("P01106", "MYC (transcription factor)", 454, "pfam", "helix-loop-helix"),
    Probe("P38936", "p21/CDKN1A (CDK inhibitor)", 164, "pfam", "cyclin-dependent kinase inhibitor"),
]

_SLOW_SEQ_LEN = 200
_TP53_FIRST_200 = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQK"
    "TYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGN"
)


def _summarize(probe: Probe, output, elapsed_s: float) -> tuple[bool, str]:
    """Return (passed, one-line summary) for a single probe result."""
    if not output.success:
        return False, f"FAIL  {probe.uniprot_id} ({probe.label}): {output.errors}"
    checks = [
        (output.accession == probe.uniprot_id, f"accession={output.accession!r}"),
        (output.sequence_length == probe.expected_length, f"length={output.sequence_length}"),
        (output.num_domains > 0, f"num_domains={output.num_domains}"),
        (
            any(d.member_database == probe.expected_member_db for d in output.domains),
            f"member_db={probe.expected_member_db}",
        ),
        (
            any(probe.expected_name_substring in d.name.lower() for d in output.domains),
            f"name~={probe.expected_name_substring!r}",
        ),
    ]
    failed = [reason for ok, reason in checks if not ok]
    if failed:
        return False, f"FAIL  {probe.uniprot_id} ({probe.label}): {', '.join(failed)}"
    member_dbs = sorted({d.member_database for d in output.domains})
    return True, (
        f"PASS  {probe.uniprot_id} ({probe.label}): {output.num_domains:>3d} domains, "
        f"{len(member_dbs)} member DBs, {elapsed_s * 1000:.0f} ms"
    )


def _run_probe(probe: Probe) -> tuple[bool, str, float]:
    start = time.monotonic()
    output = run_interproscan_fetch(InterProScanFetchInput(uniprot_id=probe.uniprot_id))
    elapsed = time.monotonic() - start
    ok, summary = _summarize(probe, output, elapsed)
    return ok, summary, elapsed


def run_sequential() -> bool:
    """Run the panel sequentially. Returns True if all probes pass."""
    print(f"=== Sequential workload ({len(_PANEL)} proteins) ===")
    all_ok = True
    total = 0.0
    for probe in _PANEL:
        ok, summary, elapsed = _run_probe(probe)
        total += elapsed
        all_ok &= ok
        print(summary)
    print(f"--- total: {total:.1f}s ({total / len(_PANEL):.2f}s avg) ---\n")
    return all_ok


def run_concurrent(workers: int = 4) -> bool:
    """Fire the panel against the live API in parallel; verify retries/rate-limit behave."""
    print(f"=== Concurrent stress ({len(_PANEL)} proteins, {workers} workers) ===")
    all_ok = True
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_probe, probe): probe for probe in _PANEL}
        for future in as_completed(futures):
            ok, summary, _ = future.result()
            all_ok &= ok
            print(summary)
    print(f"--- wall-clock: {time.monotonic() - start:.1f}s ---\n")
    return all_ok


def run_submit_path(email: str) -> bool:
    """Submit a 200-aa sequence to iprscan5; assert the canonical Pfam P53 hit comes back."""
    print("=== Submit-and-scan path (TP53[1:200], queue-dependent 2-30 min) ===")
    start = time.monotonic()
    output = run_interproscan_fetch(
        InterProScanFetchInput(sequence=_TP53_FIRST_200),
        InterProScanFetchConfig(email=email),
    )
    elapsed = time.monotonic() - start
    if not output.success:
        print(f"FAIL  submit-path: {output.errors}  ({elapsed:.0f}s)")
        return False
    pfam_hits = [d for d in output.domains if d.member_database == "pfam"]
    p53_hit = any("P53" in d.name for d in pfam_hits)
    member_dbs = sorted({d.member_database for d in output.domains})
    status = "PASS" if p53_hit else "FAIL"
    print(
        f"{status}  submit-path: job={output.job_id}, {output.num_domains} domains across {member_dbs}, "
        f"{elapsed:.0f}s wall-clock"
    )
    return p53_hit


def main() -> int:
    """Parse CLI args and run the requested probes; exit 0 on full pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress", action="store_true", help="Add a small concurrent stress block")
    parser.add_argument(
        "--submit-path",
        action="store_true",
        help="Add the slow iprscan5 submit probe (requires --email)",
    )
    parser.add_argument(
        "--email",
        default="noreply@example.org",
        help="Contact email for the iprscan5 submit path (EBI requirement)",
    )
    args = parser.parse_args()

    overall = run_sequential()
    if args.stress:
        overall &= run_concurrent()
    if args.submit_path:
        overall &= run_submit_path(args.email)

    print("=" * 60)
    print("OVERALL: PASS" if overall else "OVERALL: FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Build the bundled Cherimoya catalog CSV for a collection at a pinned Hugging Face revision.

The catalog is what the ``cherimoya`` toolkit validates model accessions against and what
``cherimoya-catalog`` searches, so the parent process never needs network access to check a
config. It joins the collection's experiment metadata, its per-fold validation metrics, and
the default experiment per (assay, biosample) that Chorus committed in
``chorus/oracles/cherimoya_source/catv1_defaults.py``.

Usage:
    python scripts/build_cherimoya_catalog.py --collection catv1
"""

import argparse
import ast
import csv
import io
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from proto_tools.tools.sequence_scoring.cherimoya.catalog import (  # noqa: E402 -- after the path insert
    CHERIMOYA_COLLECTIONS,
    CherimoyaCollection,
)

CHORUS_COMMIT = "7f0698e2631097dc835772f8c61723fa394e6948"
CHORUS_DEFAULTS_PATH = "chorus/oracles/cherimoya_source/catv1_defaults.py"
CHORUS_ASSAY = {"DNase-seq": "DNASE", "ATAC-seq": "ATAC"}
TOOLKIT = REPO / "proto_tools/tools/sequence_scoring/cherimoya"

COLUMNS = [
    "experiment_accession",
    "annotation_accession",
    "assay",
    "biosample",
    "biosample_term_id",
    "biosample_classification",
    "biosample_summary",
    "perturbed",
    "count_pearson_fold_0",
    "profile_pearson_fold_0",
    "count_pearson_mean",
    "profile_pearson_mean",
    "is_default",
    "default_reason",
]


def _get(url: str) -> str:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.text


def _hf_file(spec: CherimoyaCollection, filename: str) -> str:
    return _get(f"https://huggingface.co/{spec.hf_repo}/resolve/{spec.revision}/{filename}")


def _chorus_defaults() -> dict[tuple[str, str], tuple[str, int, str]]:
    source = _get(f"https://raw.githubusercontent.com/pinellolab/chorus/{CHORUS_COMMIT}/{CHORUS_DEFAULTS_PATH}")
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == "CATV1_DEFAULT_EXPERIMENT"
            and node.value is not None
        ):
            return ast.literal_eval(node.value)  # type: ignore[no-any-return]
    raise ValueError("CATV1_DEFAULT_EXPERIMENT not found in the Chorus defaults module")


def build_catv1(spec: CherimoyaCollection) -> list[dict[str, str]]:
    """Return one catalog row per CATv1 experiment."""
    metadata = list(csv.DictReader(io.StringIO(_hf_file(spec, "CATv1-metadata.tsv")), delimiter="\t"))
    performance = list(csv.DictReader(io.StringIO(_hf_file(spec, "performance.tsv")), delimiter="\t"))
    manifest = list(csv.DictReader(io.StringIO(_hf_file(spec, "manifest.csv"))))

    folds_by_experiment: dict[str, set[int]] = defaultdict(set)
    for row in manifest:
        expected = spec.checkpoint_filename(row["experiment"], int(row["fold"]))
        if row["checkpoint"] != expected:
            raise ValueError(f"unexpected checkpoint path {row['checkpoint']!r}; expected {expected!r}")
        folds_by_experiment[row["experiment"]].add(int(row["fold"]))

    metrics: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for row in performance:
        metrics[row["experiment_accession"]][int(row["fold"])] = {
            "count_pearson": float(row["count_pearson"]),
            "profile_pearson": float(row["profile_pearson"]),
        }

    defaults = {accession: reason for (accession, _, reason) in _chorus_defaults().values()}

    rows = []
    for record in metadata:
        accession = record["experiment_accession"]
        if folds_by_experiment[accession] != set(spec.folds):
            raise ValueError(f"{accession} has folds {sorted(folds_by_experiment[accession])} in manifest.csv")
        if set(metrics[accession]) != set(spec.folds):
            raise ValueError(f"{accession} has folds {sorted(metrics[accession])} in performance.tsv")
        if CHORUS_ASSAY.get(record["assay_term_name"]) is None:
            raise ValueError(f"{accession} has unknown assay {record['assay_term_name']!r}")
        by_fold = metrics[accession]
        rows.append(
            {
                "experiment_accession": accession,
                "annotation_accession": record["annotation_accession"],
                "assay": record["assay_term_name"],
                "biosample": record["experiment_biosample_term_name"],
                "biosample_term_id": record["biosample_term_id"],
                "biosample_classification": record["biosample_classification"],
                "biosample_summary": record["biosample_simple_summary"],
                "perturbed": str(record["experiment_peturbed"].upper() == "TRUE"),
                "count_pearson_fold_0": f"{by_fold[0]['count_pearson']:.4f}",
                "profile_pearson_fold_0": f"{by_fold[0]['profile_pearson']:.4f}",
                "count_pearson_mean": f"{statistics.fmean(m['count_pearson'] for m in by_fold.values()):.4f}",
                "profile_pearson_mean": f"{statistics.fmean(m['profile_pearson'] for m in by_fold.values()):.4f}",
                "is_default": str(accession in defaults),
                "default_reason": defaults.get(accession, ""),
            }
        )

    accessions = {row["experiment_accession"] for row in rows}
    missing = sorted(set(defaults) - accessions)
    if missing:
        raise ValueError(f"Chorus defaults name experiments absent from the metadata: {missing}")
    return sorted(rows, key=lambda row: (row["assay"], row["biosample"], row["experiment_accession"]))


def header(spec: CherimoyaCollection) -> str:
    """Return the provenance and license comment written above the catalog rows."""
    return f"""\
Cherimoya catalog for collection {spec.key}, generated by scripts/build_cherimoya_catalog.py.
Experiments, metadata and validation metrics: huggingface.co/{spec.hf_repo} at revision {spec.revision}
(CATv1-metadata.tsv, performance.tsv, manifest.csv). License CC-BY-4.0.
is_default and default_reason: the default experiment per (assay, biosample) committed by Chorus in
github.com/pinellolab/chorus/blob/{CHORUS_COMMIT}/{CHORUS_DEFAULTS_PATH}
Copyright (c) 2024 Pinello Lab. Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to
do so, subject to the following conditions: The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""


def main() -> None:
    """Write the catalog CSV for the requested collection."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", choices=sorted(CHERIMOYA_COLLECTIONS), default="catv1")
    parser.add_argument("--output", type=Path, help="defaults to the collection's catalog_file in the toolkit")
    args = parser.parse_args()

    spec = CHERIMOYA_COLLECTIONS[args.collection]
    output = args.output or TOOLKIT / spec.catalog_file
    rows = build_catv1(spec)
    with open(output, "w", newline="") as handle:
        handle.writelines(f"# {line}\n" for line in header(spec).splitlines())
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} experiments to {output}")


if __name__ == "__main__":
    main()

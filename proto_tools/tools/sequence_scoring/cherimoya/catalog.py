"""Cherimoya model collections and their bundled experiment catalogs."""

import csv
import difflib
import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CHERIMOYA_CONTEXT = 2114
CHERIMOYA_OUTPUT = 1000
CHERIMOYA_OUTPUT_FLANK = (CHERIMOYA_CONTEXT - CHERIMOYA_OUTPUT) // 2
DEFAULT_CHERIMOYA_MODEL = "ENCSR000EOT"

CherimoyaCollectionKey = Literal["catv1"]
CherimoyaFold = Literal[0, 1, 2, 3, 4]
CherimoyaAssay = Literal["DNase-seq", "ATAC-seq"]
CherimoyaBiosampleClassification = Literal[
    "cell line", "primary cell", "tissue", "in vitro differentiated cells", "organoid"
]

_ACCESSION = re.compile(r"^ENCSR\d{3}[A-Z]{3}$")


@dataclass(frozen=True)
class CherimoyaCollection:
    """A released set of Cherimoya checkpoints with one catalog row per experiment.

    Attributes:
        key (str): Collection name used in configs (``"catv1"``).
        hf_repo (str): Hugging Face model repository holding the checkpoints.
        revision (str): Pinned repository commit; every download uses it.
        checkpoint_template (str): Checkpoint path in the repository, formatted
            with ``accession`` and ``fold``.
        folds (tuple[int, ...]): Cross-validation folds released per experiment.
        catalog_file (str): Bundled catalog CSV next to this module.
    """

    key: str
    hf_repo: str
    revision: str
    checkpoint_template: str
    folds: tuple[int, ...]
    catalog_file: str

    def checkpoint_filename(self, accession: str, fold: int) -> str:
        """Return the repository path of one experiment's fold checkpoint."""
        return self.checkpoint_template.format(accession=accession, fold=fold)


CHERIMOYA_COLLECTIONS: dict[str, CherimoyaCollection] = {
    "catv1": CherimoyaCollection(
        key="catv1",
        hf_repo="programmable-genomics/CATv1",
        revision="dfdd7b6b0a9b20bd59427337e4244056b128bda9",
        checkpoint_template="models/{accession}/cherimoya.fold_{fold}.torch",
        folds=(0, 1, 2, 3, 4),
        catalog_file="catv1_catalog.csv",
    ),
}


class CherimoyaCatalogEntry(BaseModel):
    """One experiment in a Cherimoya collection.

    Attributes:
        collection (str): Collection the experiment belongs to.
        experiment_accession (str): ENCODE experiment the models were trained on.
        annotation_accession (str): ENCODE ChromBPNet model annotation for the same experiment.
        assay (str): ``"DNase-seq"`` or ``"ATAC-seq"``.
        biosample (str): ENCODE biosample term name.
        biosample_term_id (str): Ontology term ID of the biosample.
        biosample_classification (str): ENCODE biosample classification.
        biosample_summary (str): ENCODE biosample summary (donor sex and age, treatment).
        perturbed (bool): Whether ENCODE marks the experiment as perturbed.
        count_pearson_fold_0 (float): Fold 0 validation Pearson of log total counts.
        profile_pearson_fold_0 (float): Fold 0 validation Pearson of the base-resolution profile.
        count_pearson_mean (float): Count Pearson averaged over the five folds.
        profile_pearson_mean (float): Profile Pearson averaged over the five folds.
        is_default (bool): Whether this is the default experiment for its (assay, biosample).
        default_reason (str): Why it is the default (Chorus rule); empty when not a default.
        encode_url (str): ENCODE portal page of the experiment.
    """

    # Entries are shared process-wide by the cached catalog.
    model_config = ConfigDict(frozen=True)

    collection: str = Field(title="Collection", description="Collection the experiment belongs to")
    experiment_accession: str = Field(title="Experiment", description="ENCODE experiment accession")
    annotation_accession: str = Field(title="Annotation", description="ENCODE ChromBPNet annotation accession")
    assay: str = Field(title="Assay", description="DNase-seq or ATAC-seq")
    biosample: str = Field(title="Biosample", description="ENCODE biosample term name")
    biosample_term_id: str = Field(title="Biosample Term ID", description="Ontology term ID of the biosample")
    biosample_classification: str = Field(title="Classification", description="ENCODE biosample classification")
    biosample_summary: str = Field(title="Biosample Summary", description="ENCODE biosample summary")
    perturbed: bool = Field(title="Perturbed", description="Whether ENCODE marks the experiment as perturbed")
    count_pearson_fold_0: float = Field(
        title="Count Pearson (Fold 0)", description="Fold 0 validation Pearson of log total counts"
    )
    profile_pearson_fold_0: float = Field(
        title="Profile Pearson (Fold 0)", description="Fold 0 validation Pearson of the base-resolution profile"
    )
    count_pearson_mean: float = Field(title="Count Pearson (Mean)", description="Count Pearson averaged over folds")
    profile_pearson_mean: float = Field(
        title="Profile Pearson (Mean)", description="Profile Pearson averaged over folds"
    )
    is_default: bool = Field(title="Default", description="Default experiment for its (assay, biosample)")
    default_reason: str = Field(title="Default Reason", description="Chorus rule that chose the default, if any")
    encode_url: str = Field(title="ENCODE URL", description="ENCODE portal page of the experiment")


@functools.cache
def load_catalog(collection: str) -> dict[str, CherimoyaCatalogEntry]:
    """Return the bundled catalog of a collection, keyed by experiment accession.

    Args:
        collection (str): Collection key, e.g. ``"catv1"``.

    Returns:
        dict[str, CherimoyaCatalogEntry]: Catalog entries keyed by accession.
    """
    spec = CHERIMOYA_COLLECTIONS[collection]
    path = Path(__file__).with_name(spec.catalog_file)
    with open(path, newline="") as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        entries = [
            CherimoyaCatalogEntry(
                collection=collection,
                experiment_accession=row["experiment_accession"],
                annotation_accession=row["annotation_accession"],
                assay=row["assay"],
                biosample=row["biosample"],
                biosample_term_id=row["biosample_term_id"],
                biosample_classification=row["biosample_classification"],
                biosample_summary=row["biosample_summary"],
                perturbed=row["perturbed"] == "True",
                count_pearson_fold_0=float(row["count_pearson_fold_0"]),
                profile_pearson_fold_0=float(row["profile_pearson_fold_0"]),
                count_pearson_mean=float(row["count_pearson_mean"]),
                profile_pearson_mean=float(row["profile_pearson_mean"]),
                is_default=row["is_default"] == "True",
                default_reason=row["default_reason"],
                encode_url=f"https://www.encodeproject.org/experiments/{row['experiment_accession']}/",
            )
            for row in rows
        ]
    return {entry.experiment_accession: entry for entry in entries}


def is_checkpoint_path(model: str) -> bool:
    """Return whether a ``models`` entry names a local checkpoint file rather than an accession."""
    return not _ACCESSION.match(model)


def validate_model_reference(model: str, collection: str) -> str:
    """Validate one ``models`` entry: an accession in the collection or a ``.torch`` file.

    Args:
        model (str): ENCODE experiment accession or local checkpoint path.
        collection (str): Collection the accession must belong to.

    Returns:
        str: The accession unchanged, or the checkpoint path made absolute.

    Raises:
        ValueError: If the accession is not in the collection, or the path is not
            an existing ``.torch`` file.
    """
    if not is_checkpoint_path(model):
        catalog = load_catalog(collection)
        if model not in catalog:
            close = difflib.get_close_matches(model, list(catalog), n=3, cutoff=0.6)
            hint = f"; did you mean {', '.join(close)}?" if close else "; search with cherimoya-catalog"
            raise ValueError(f"{model!r} is not an experiment in collection {collection!r}{hint}")
        return model
    path = Path(model).expanduser()
    if path.suffix != ".torch":
        raise ValueError(f"{model!r} is neither an ENCODE experiment accession nor a .torch checkpoint path")
    if not path.is_file():
        raise ValueError(f"checkpoint {model!r} does not exist")
    return str(path.resolve())

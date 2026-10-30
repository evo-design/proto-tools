"""Shared input, config, and output models for the Cherimoya toolkit."""

from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from proto_tools.tools.sequence_scoring.cherimoya.catalog import (
    CHERIMOYA_COLLECTIONS,
    CHERIMOYA_CONTEXT,
    CHERIMOYA_OUTPUT,
    DEFAULT_CHERIMOYA_MODEL,
    CherimoyaCollectionKey,
    CherimoyaFold,
    is_checkpoint_path,
    load_catalog,
    validate_model_reference,
)
from proto_tools.tools.sequence_scoring.shared_data_models import (
    PreparedSequenceWindow,
    SequenceWindow,
    coerce_sequence_windows,
    prepare_model_windows,
    windows_target_ranges,
)
from proto_tools.utils import BaseConfig, BaseToolInput, ConfigField, InputField
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.tool_io import Metrics, MetricSpec

MAX_CHERIMOYA_MODELS = 64


def normalize_models(value: Any) -> Any:
    """Normalize a single model reference to a one-item list."""
    if isinstance(value, str):
        return [value]
    return value


def resolve_model_list(models: list[str], collection: str) -> list[str]:
    """Validate each model reference against the collection and reject duplicates."""
    resolved = [validate_model_reference(model, collection) for model in models]
    if len(set(resolved)) != len(resolved):
        raise ValueError("models must be unique")
    return resolved


def validate_folds(folds: list[int], collection: str) -> list[int]:
    """Reject a fold listed twice or one the collection did not release."""
    if len(set(folds)) != len(folds):
        raise ValueError("folds must be unique")
    released = CHERIMOYA_COLLECTIONS[collection].folds
    missing = [fold for fold in folds if fold not in released]
    if missing:
        raise ValueError(f"collection {collection!r} has folds {list(released)}, not {missing}")
    return folds


def checkpoint_specs(models: list[str], folds: Sequence[int], collection: str) -> list[dict[str, Any]]:
    """Expand validated model references into one worker checkpoint spec per (model, fold).

    A catalog accession expands to one spec per requested fold. A checkpoint path is a
    single model and expands to one spec with ``fold=None``. ``source`` names the file
    each spec must be loaded from, for the worker to report back.

    Args:
        models (list[str]): Validated accessions or absolute checkpoint paths.
        folds (Sequence[int]): Folds to load for catalog accessions.
        collection (str): Collection the accessions belong to.

    Returns:
        list[dict[str, Any]]: Specs the standalone worker loads, in model then fold order.
    """
    spec = CHERIMOYA_COLLECTIONS[collection]
    out: list[dict[str, Any]] = []
    for model in models:
        if is_checkpoint_path(model):
            out.append({"model": model, "fold": None, "path": model, "source": model})
            continue
        for fold in folds:
            filename = spec.checkpoint_filename(model, fold)
            out.append(
                {
                    "model": model,
                    "fold": fold,
                    "repo_id": spec.hf_repo,
                    "revision": spec.revision,
                    "filename": filename,
                    "source": f"{spec.hf_repo}@{spec.revision}:{filename}",
                }
            )
    return out


class CherimoyaModelInfo(BaseModel):
    """Provenance of one model a Cherimoya call ran.

    Attributes:
        model (str): ENCODE experiment accession, or the checkpoint path for a custom model.
        collection (str | None): Collection of a catalog model; ``None`` for a checkpoint path.
        revision (str | None): Hugging Face revision the checkpoints were downloaded at.
        assay (str | None): Assay the model was trained on, when known from the catalog.
        target (str | None): ChIP-seq target of a BATv1 model, when known from the catalog.
        biosample (str | None): Biosample the model was trained on, when known from the catalog.
        folds (list[int]): Folds loaded for this model; empty for a checkpoint path.
        checkpoint_sha256 (list[str]): SHA-256 of each checkpoint file, in fold order.
    """

    model: str = Field(title="Model", description="ENCODE experiment accession or checkpoint path")
    collection: str | None = Field(default=None, title="Collection", description="Collection of a catalog model")
    revision: str | None = Field(default=None, title="Revision", description="Hugging Face revision of the weights")
    assay: str | None = Field(default=None, title="Assay", description="Assay the model was trained on")
    target: str | None = Field(default=None, title="Target", description="ChIP-seq target of a BATv1 model")
    biosample: str | None = Field(default=None, title="Biosample", description="Biosample the model was trained on")
    folds: list[int] = Field(default_factory=list, title="Folds", description="Folds loaded for this model")
    checkpoint_sha256: list[str] = Field(
        default_factory=list, title="Checkpoint SHA-256", description="SHA-256 of each checkpoint, in fold order"
    )


def model_infos(
    models: list[str], collection: str, specs: list[dict[str, Any]], applied: list[dict[str, Any]]
) -> list[CherimoyaModelInfo]:
    """Check the worker ran one checkpoint per spec from the requested file, and describe each model.

    The worker swaps checkpoints in-process and reports, for each checkpoint it ran, the
    source file recorded when that checkpoint was loaded. A cached model that came from
    a different file than requested is raised rather than mislabeling its predictions.

    Args:
        models (list[str]): Validated model references in request order.
        collection (str): Collection key.
        specs (list[dict[str, Any]]): The checkpoint specs sent to the worker.
        applied (list[dict[str, Any]]): ``{"model", "fold", "source", "sha256"}`` per checkpoint run.

    Returns:
        list[CherimoyaModelInfo]: One entry per model, in request order.

    Raises:
        RuntimeError: If the worker ran other checkpoints than requested.
    """
    requested = [(spec["model"], spec["fold"], spec["source"]) for spec in specs]
    ran = [(entry["model"], entry["fold"], entry["source"]) for entry in applied]
    if ran != requested:
        raise RuntimeError(f"cherimoya worker ran checkpoints {ran} but {requested} were requested")

    catalog = load_catalog(collection)
    spec = CHERIMOYA_COLLECTIONS[collection]
    infos = []
    for model in models:
        entries = [entry for entry in applied if entry["model"] == model]
        record = catalog.get(model)
        infos.append(
            CherimoyaModelInfo(
                model=model,
                collection=None if record is None else collection,
                revision=None if record is None else spec.revision,
                assay=None if record is None else record.assay,
                target=None if record is None or not record.target else record.target,
                biosample=None if record is None else record.biosample,
                folds=[] if record is None else [int(entry["fold"]) for entry in entries],
                checkpoint_sha256=[str(entry["sha256"]) for entry in entries],
            )
        )
    return infos


def check_row_models(rows: list[list[dict[str, Any]]], models: list[str]) -> None:
    """Raise unless every per-sequence row lists one result per model, in request order."""
    for index, row in enumerate(rows):
        ran = [entry["model"] for entry in row]
        if ran != models:
            raise RuntimeError(f"cherimoya worker returned models {ran} for sequence {index}; expected {models}")


class CherimoyaSequenceInput(BaseToolInput):
    """DNA sequences for Cherimoya scoring and prediction.

    Exact-window mode takes sequences of exactly 2,114 bp, the model context. Target-range
    mode takes longer sequences, each with a ``target_range`` of at most 1,000 bp; the tool
    cuts the 2,114 bp window whose central 1,000 bp output starts at the range start (or as
    close as the sequence ends allow). There is no padding.

    Attributes:
        sequences (list[SequenceWindow]): DNA sequence(s), each with an optional
            ``target_range``; a bare string is accepted. ``N`` is encoded as all zeros.
    """

    sequences: list[SequenceWindow] = InputField(
        title="Sequences",
        description="2,114 bp DNA windows, or longer sequences each with a target_range of at most 1,000 bp",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def coerce_sequences(cls, value: Any) -> Any:
        """Coerce bare strings and dicts into SequenceWindow items."""
        return coerce_sequence_windows(value)

    @model_validator(mode="after")
    def validate_window_inputs(self) -> "CherimoyaSequenceInput":
        """Validate exact-context or target-aligned windows."""
        self.prepared_windows()
        return self

    def prepared_windows(self) -> list[PreparedSequenceWindow]:
        """Return the 2,114 bp model window and its coordinates for each sequence."""
        return prepare_model_windows(
            [window.sequence for window in self.sequences],
            context_length=CHERIMOYA_CONTEXT,
            output_length=CHERIMOYA_OUTPUT,
            target_ranges=windows_target_ranges(self.sequences),
        )

    def __len__(self) -> int:
        """Return the number of input sequences."""
        return len(self.sequences)


class CherimoyaWindowResult(BaseModel):
    """Coordinates of one input sequence's model window, shared by score and predict results.

    Coordinates are 0-based with exclusive ends, relative to ``sequence``.

    Attributes:
        sequence (str): Input DNA sequence.
        sequence_length (int): Length of the input sequence.
        context_start (int): Start of the 2,114 bp model input window.
        context_end (int): End of the model input window.
        output_start (int): Start of the 1,000 bp span the model predicts.
        output_end (int): End of the predicted span.
        target_start (int | None): Requested target start, if ``target_range`` was set.
        target_end (int | None): Requested target end, if ``target_range`` was set.
    """

    sequence: str = Field(title="Sequence", description="DNA sequence originally provided to the tool")
    sequence_length: int = Field(title="Sequence Length", description="Length of the provided DNA sequence")
    context_start: int = Field(title="Context Start", description="0-based start of the 2,114 bp model input window")
    context_end: int = Field(title="Context End", description="0-based exclusive end of the model input window")
    output_start: int = Field(title="Output Start", description="0-based start of the 1,000 bp predicted span")
    output_end: int = Field(title="Output End", description="0-based exclusive end of the predicted span")
    target_start: int | None = Field(default=None, title="Target Start", description="Requested target start")
    target_end: int | None = Field(default=None, title="Target End", description="Requested target end")


def window_fields(sequence: str, window: PreparedSequenceWindow) -> dict[str, Any]:
    """Return the CherimoyaWindowResult fields of one prepared window."""
    return {
        "sequence": sequence,
        "sequence_length": len(sequence),
        "context_start": window.context_start,
        "context_end": window.context_end,
        "output_start": window.output_start,
        "output_end": window.output_end,
        "target_start": window.target_start,
        "target_end": window.target_end,
    }


class CherimoyaModelsConfig(BaseConfig):
    """Model selection shared by the Cherimoya scoring and prediction tools.

    Attributes:
        collection (CherimoyaCollectionKey): Collection the accessions in ``models`` come from:
            ``"catv1"`` (chromatin accessibility) or ``"batv1"`` (ChIP-seq). The default
            ``models`` entry belongs to CATv1, so name the models when choosing BATv1.
        models (list[str]): ENCODE experiment accessions from the collection, or paths to
            Cherimoya ``.torch`` checkpoints with the same 2,114 bp in, 1,000 bp out geometry.
        folds (list[CherimoyaFold]): Folds to load for each accession; several are averaged.
            A checkpoint path is one model and ignores ``folds``.
        reverse_complement_average (bool): Average forward and reverse-complement predictions.
        batch_size (int): Sequences per forward pass.
        device (str): Device used for inference.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run Cherimoya inference on",
        include_in_key=False,
    )
    collection: CherimoyaCollectionKey = ConfigField(
        title="Collection",
        default="catv1",
        description="Model collection the accessions in models come from: catv1 (accessibility) or batv1 (ChIP-seq).",
    )
    models: list[str] = ConfigField(
        title="Models",
        default=[DEFAULT_CHERIMOYA_MODEL],
        min_length=1,
        max_length=MAX_CHERIMOYA_MODELS,
        description="ENCODE experiment accessions (see cherimoya-catalog) or local .torch checkpoint paths.",
    )
    folds: list[CherimoyaFold] = ConfigField(
        title="Folds",
        default=[0],
        min_length=1,
        description="Cross-validation folds to load per accession; several are averaged.",
    )
    reverse_complement_average: bool = ConfigField(
        title="Reverse-Complement Average",
        default=False,
        description="Average forward and reverse-complement predictions (twice the compute).",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=64,
        ge=1,
        description="Sequences per forward pass; the models are small, so 64 by default; lower if OOM",
        include_in_key=False,
    )

    @field_validator("models", mode="before")
    @classmethod
    def normalize_model_list(cls, value: Any) -> Any:
        """Normalize a single model reference to a list."""
        return normalize_models(value)

    @field_validator("models")
    @classmethod
    def resolve_models(cls, models: list[str], info: ValidationInfo) -> list[str]:
        """Resolve every model against the collection and reject duplicates."""
        return resolve_model_list(models, info.data.get("collection", "catv1"))

    @field_validator("folds")
    @classmethod
    def check_folds(cls, folds: list[int], info: ValidationInfo) -> list[int]:
        """Reject a repeated fold or one the collection did not release."""
        return validate_folds(folds, info.data.get("collection", "catv1"))

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """Checkpoint paths name local files that a remote worker does not have."""
        paths = [model for model in self.models if is_checkpoint_path(model)]
        if paths:
            return f"models {paths} are local checkpoint files not available on device='{device}'. Run locally."
        return None


class CherimoyaScoreMetrics(Metrics):
    """Cherimoya predictions for one model on one sequence.

    Metrics documented in ``metric_spec``:
        log_counts (float): Predicted ln(reads + 1) over the 1,000 bp output window, summed
            over both strands for a ChIP-seq model.
        target_counts (float): Expected reads inside ``target_range``.

    Attributes:
        model (str): Model these predictions come from.
        fold_log_counts (list[float]): ``log_counts`` of each fold before averaging.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "log_counts": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Predicted ln(reads + 1) over the 1,000 bp output window, mean over folds.",
            "better_values_are": "context-dependent",
        },
        "target_counts": {
            "availability": "when target_range is set",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Expected reads inside target_range: sum of the profile times expm1(log_counts).",
            "better_values_are": "context-dependent",
        },
    }

    model: str = Field(title="Model", description="Model these predictions come from")
    fold_log_counts: list[float] = Field(
        default_factory=list, title="Fold Log Counts", description="log_counts of each fold before averaging"
    )

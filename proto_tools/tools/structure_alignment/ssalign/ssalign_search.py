"""SSAlign two-stage protein structure search (SaProt embedding + FAISS prefilter + SAligner refine)."""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proto_tools.entities import Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.tool_io import MissingAssetError

logger = logging.getLogger(__name__)

SSAlignDim = Literal[64, 128, 256, 512, 1280]
SSAlignMode = Literal[0, 1]

# Shared 65-residue fixture reused across the category.
_EXAMPLE_PDB_PATH = str(Path(__file__).parents[1] / "example_input_fixture.pdb")

# Required filename suffixes inside a prebuilt SSAlignDB directory (mode 2).
_DB_INDEX_SUFFIX = "_IndexFlatIP_{dim}_faiss.index"
_DB_MU_SUFFIX = "_whitening_mu.npy"
_DB_W_SUFFIX = "_whitening_W.npy"
_DB_SEQS_SUFFIX = "_id_Seq.npz"


# ============================================================================
# Data Models
# ============================================================================


class SSAlignQuery(BaseModel):
    """One query structure plus an optional stable id echoed into its hits.

    Attributes:
        structure (Structure): Query structure (Structure/path/PDB/CIF); multi-chain uses the first chain only.
        query_id (str | None): Stable identifier echoed into output; defaults to positional 'query_{i}'.
    """

    model_config = ConfigDict(extra="forbid")

    structure: Structure = Field(
        title="Query Structure",
        description="Query structure (Structure/path/PDB/CIF); multi-chain uses the first chain only",
    )
    query_id: str | None = Field(
        default=None,
        title="Query ID",
        description="Stable id echoed into hits; defaults to positional 'query_{i}'",
    )


class SSAlignSearchInput(BaseToolInput):
    """Input for SSAlign structure search.

    Attributes:
        queries (list[SSAlignQuery]): Query structures; output is one ranked hit-bundle per query,
            in input order.
    """

    queries: list[SSAlignQuery] = InputField(
        title="Query Structures",
        description="Query structures to search; one ranked hit-bundle is returned per query, in order",
    )

    @model_validator(mode="after")
    def _validate_nonempty(self) -> "SSAlignSearchInput":
        """At least one query is required."""
        if not self.queries:
            raise ValueError("ssalign-search: at least one query is required.")
        return self


class SSAlignSearchConfig(BaseConfig):
    """Configuration for SSAlign search. Exactly one target source must be provided.

    Attributes:
        target_structures (list[Structure] | None): Mode 1 — targets embedded and indexed in-memory
            (build on the fly). Mutually exclusive with ssalign_db.
        ssalign_db (str | None): Mode 2 — path (or AssetRef) to a prebuilt SSAlignDB directory
            (FAISS index + whitening mu/W + id/seq npz). Mutually exclusive with target_structures.
        mode (SSAlignMode): 0 = cosine prefilter only; 1 = prefilter + SAligner 3Di refinement.
        dim (SSAlignDim): Prebuilt-SSAlignDB index to search (mode 2 only); ignored when building on the fly.
        prefilter_target (int): Top-K cosine candidates retained from the FAISS prefilter per query.
        prefilter_threshold (float): Candidates with cosine below this are refined by SAligner (mode 1).
        max_target (int): Final hits returned per query after ranking; must be <= prefilter_target.
        device (str): Device for SaProt embedding ('cuda' default, or 'cpu').
        batch_size (int): Structures per SaProt forward pass.
        num_threads (int): CPU threads for the FAISS search.
    """

    # ── target source (exactly one; enforced below) ──────────────────────────
    target_structures: list[Structure] | None = ConfigField(
        title="Target Structures",
        default=None,
        description="Mode 1: targets to embed and index in-memory. Provide this OR ssalign_db, not both",
    )
    ssalign_db: str | None = ConfigField(
        title="Prebuilt SSAlignDB Directory",
        default=None,
        description="Mode 2: path or AssetRef to a prebuilt SSAlignDB dir. Provide this OR target_structures",
    )

    # ── pipeline parameters (upstream defaults) ──────────────────────────────
    mode: SSAlignMode = ConfigField(
        title="Search Mode",
        default=1,
        description="0 = cosine prefilter only; 1 = prefilter + SAligner 3Di refine of below-threshold hits",
    )
    dim: SSAlignDim = ConfigField(
        title="Index Dimension",
        default=512,
        description="Prebuilt-SSAlignDB index to search (mode 2); ignored when building on the fly",
        reload_on_change=True,
    )
    prefilter_target: int = ConfigField(
        title="Prefilter Targets",
        default=2000,
        ge=1,
        description="Top-K cosine candidates kept from the FAISS prefilter per query",
    )
    prefilter_threshold: float = ConfigField(
        title="Prefilter Threshold",
        default=0.3,
        ge=-1.0,
        le=1.0,
        description="Cosine cutoff; below-threshold candidates are SAligner-refined in mode 1",
    )
    max_target: int = ConfigField(
        title="Max Targets per Query",
        default=1000,
        ge=1,
        description="Final hits returned per query after ranking; must be <= prefilter_target",
    )

    # ── device (GPU by default; SaProt embedding dominates runtime) ───────────
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device for SaProt embedding, e.g. 'cuda' (NVIDIA GPU) or 'cpu'",
        include_in_key=False,
        examples=["cuda", "cpu"],
    )

    # ── performance (not part of the cache key) ──────────────────────────────
    batch_size: int = ConfigField(
        title="Embedding Batch Size",
        default=8,
        ge=1,
        description="Structures per SaProt forward pass",
        include_in_key=False,
    )
    num_threads: int = ConfigField(
        title="Threads",
        default=4,
        ge=1,
        description="CPU threads for the FAISS search",
        include_in_key=False,
    )

    @model_validator(mode="after")
    def _validate_config(self) -> "SSAlignSearchConfig":
        """Enforce exactly one (non-empty) target source and max_target <= prefilter_target."""
        have_structs = self.target_structures is not None
        have_db = self.ssalign_db is not None
        if have_structs == have_db:
            raise ValueError(
                "ssalign-search: provide exactly one of target_structures (mode 1) or ssalign_db (mode 2)."
            )
        if have_structs and not self.target_structures:
            raise ValueError("ssalign-search: target_structures must contain at least one structure.")
        if self.max_target > self.prefilter_target:
            raise ValueError(
                f"ssalign-search: max_target ({self.max_target}) must be <= prefilter_target ({self.prefilter_target})."
            )
        return self


class SSAlignHit(BaseModel):
    """One ranked target hit for a query.

    Attributes:
        target_id (str): Identifier of the matched target structure.
        prefilter_score (float): FAISS cosine similarity (raw in mode 1, whitened in mode 2); in [-1, 1].
        saligner_score (float | None): SAligner 3Di global-alignment score (mode 1, refined hits only).
        ss_score (float): Predicted avg TM-score (SSAlign linear fit of cosine); noisy (r~0.69), can exceed 1.
        rank (int): 1-indexed rank within this query's results.
        refined (bool): True if this hit was re-ranked by SAligner (below prefilter_threshold).
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(title="Target ID", description="Identifier of the matched target structure")
    prefilter_score: float = Field(
        title="Prefilter Cosine Score",
        ge=-1.0,
        le=1.0,
        description="FAISS cosine similarity (raw in mode 1, whitened in mode 2); higher = closer",
    )
    saligner_score: float | None = Field(
        default=None,
        title="SAligner Score",
        description="SAligner 3Di global-alignment score (mode 1 refined hits only); None otherwise",
    )
    ss_score: float = Field(
        title="SS Score",
        description="Predicted avg TM-score (SSAlign linear fit of cosine); noisy (r~0.69), can exceed 1",
    )
    rank: int = Field(title="Rank", ge=1, description="1-indexed rank within this query's results")
    refined: bool = Field(
        title="Refined",
        description="True if re-ranked by SAligner (cosine was below prefilter_threshold)",
    )

    @model_validator(mode="after")
    def _validate_refined(self) -> "SSAlignHit":
        """'refined' is True exactly when SAligner produced a score (mode-1 below-threshold hits)."""
        if (self.saligner_score is not None) != self.refined:
            raise ValueError("ssalign: 'refined' must be True iff 'saligner_score' is set.")
        return self


class SSAlignQueryResult(BaseModel):
    """All ranked hits for a single query.

    Attributes:
        query_id (str): Echoed query identifier.
        hits (list[SSAlignHit]): Ranked hits, truncated to max_target.
        num_hits (int): ``len(hits)``.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(title="Query ID", description="Echoed query identifier")
    hits: list[SSAlignHit] = Field(default_factory=list, title="Hits", description="Ranked hits (<= max_target)")
    num_hits: int = Field(title="Number of Hits", ge=0, description="Number of hits returned for this query")

    @model_validator(mode="after")
    def _validate_num_hits(self) -> "SSAlignQueryResult":
        """num_hits must equal len(hits)."""
        if self.num_hits != len(self.hits):
            raise ValueError(f"ssalign: num_hits ({self.num_hits}) must equal len(hits) ({len(self.hits)}).")
        return self


class SSAlignSearchOutput(BaseToolOutput):
    """Output from SSAlign structure search.

    Attributes:
        results (list[SSAlignQueryResult]): One ranked hit-bundle per query, in input order.
    """

    results: list[SSAlignQueryResult] = Field(
        title="Results",
        description="One ranked hit-bundle per query, in input order",
    )

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> SSAlignQueryResult:
        return self.results[index]

    def __iter__(self) -> Iterator[SSAlignQueryResult]:  # type: ignore[override]
        return iter(self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format == "json":
            path = Path(export_path).with_suffix(".json")
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        if file_format == "csv":
            path = Path(export_path).with_suffix(".csv")
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["query_id", "rank", "target_id", "prefilter_score", "saligner_score", "ss_score", "refined"]
                )
                for result in self.results:
                    for hit in result.hits:
                        writer.writerow(
                            [
                                result.query_id,
                                hit.rank,
                                hit.target_id,
                                hit.prefilter_score,
                                hit.saligner_score,
                                hit.ss_score,
                                hit.refined,
                            ]
                        )
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return SSAlignSearchInput(queries=[SSAlignQuery(structure=Structure.from_file(_EXAMPLE_PDB_PATH), query_id="q0")])


def _require_ssalign_db(db_dir: str, dim: int) -> None:
    """Fail fast with MissingAssetError when a prebuilt SSAlignDB directory is absent or incomplete."""
    path = Path(db_dir)
    suffixes = [_DB_INDEX_SUFFIX.format(dim=dim), _DB_MU_SUFFIX, _DB_W_SUFFIX, _DB_SEQS_SUFFIX]
    if not path.is_dir() or any(not list(path.glob(f"*{suffix}")) for suffix in suffixes):
        raise MissingAssetError(
            "ssalign",
            "database",
            f"SSAlignDB directory {db_dir!r} is missing or lacks required files for dim={dim} "
            f"(expected *{_DB_INDEX_SUFFIX.format(dim=dim)}, *{_DB_MU_SUFFIX}, *{_DB_W_SUFFIX}, *{_DB_SEQS_SUFFIX}).",
        )


def _structure_payload(structure: Structure, item_id: str) -> dict[str, str]:
    """Serialize a Structure to the isolated standalone's expected {id, text, format} payload."""
    return {"id": item_id, "text": structure.structure_pdb, "format": "pdb"}


@tool(
    key="ssalign-search",
    label="SSAlign Structure Search",
    category="structure_alignment",
    input_class=SSAlignSearchInput,
    config_class=SSAlignSearchConfig,
    output_class=SSAlignSearchOutput,
    description=(
        "Two-stage protein structure search (SSAlign): SaProt embedding + FAISS cosine prefilter "
        "(whitened in mode 2), with optional SAligner 3Di global-alignment refinement."
    ),
    uses_gpu=True,
    cacheable=True,
    stochastic=False,
    example_input=example_input,
    iterable_input_field="queries",
    iterable_output_field="results",
)
def run_ssalign_search(
    inputs: SSAlignSearchInput,
    config: SSAlignSearchConfig,
    instance: Any = None,
) -> SSAlignSearchOutput:
    """Run an SSAlign two-stage structure search.

    Encodes each query (and each on-the-fly target) with mini3di 3Di + SaProt, prefilters with a
    FAISS cosine index (raw embeddings on the fly; the prebuilt DB's whitened index in mode 2), and
    (mode 1) refines below-threshold candidates with SAligner.

    Args:
        inputs (SSAlignSearchInput): Query structures.
        config (SSAlignSearchConfig): Target source (build-on-the-fly or prebuilt DB) + search options.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        SSAlignSearchOutput: One ranked hit-bundle per query, in input order.
    """
    if config.ssalign_db is not None:
        _require_ssalign_db(config.ssalign_db, config.dim)

    logger.debug("Using standalone venv for ssalign search (mode=%s, dim=%s)", config.mode, config.dim)

    input_dict: dict[str, Any] = {
        "mode": config.mode,
        "dim": config.dim,
        "prefilter_target": config.prefilter_target,
        "prefilter_threshold": config.prefilter_threshold,
        "max_target": config.max_target,
        "batch_size": config.batch_size,
        "num_threads": config.num_threads,
        "device": config.device,
        "seed": config.seed,
        "queries": [
            _structure_payload(query.structure, query.query_id or f"query_{i}")
            for i, query in enumerate(inputs.queries)
        ],
        "target_structures": (
            [_structure_payload(s, f"target_{i}") for i, s in enumerate(config.target_structures)]
            if config.target_structures is not None
            else None
        ),
        "ssalign_db": config.ssalign_db,
    }

    output_data = ToolInstance.dispatch("ssalign", input_dict, instance=instance, config=config)

    results = [
        SSAlignQueryResult(
            query_id=result["query_id"],
            hits=[SSAlignHit(**hit) for hit in result["hits"]],
            num_hits=len(result["hits"]),
        )
        for result in output_data["results"]
    ]
    return SSAlignSearchOutput(results=results, metadata={"tool": "ssalign", "mode": config.mode})

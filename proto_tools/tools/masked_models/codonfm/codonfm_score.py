"""CodonFM (Encodon) mutation-effect scoring (reference-vs-alternate codon log-likelihood ratio)."""

import csv
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from proto_tools.tools.masked_models.codonfm.shared_data_models import (
    CodonFMConfig,
    normalize_codon_sequence,
    resolve_checkpoint_source,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, InputField, ToolInstance


class CodonFMScoreConfig(CodonFMConfig):
    """Configuration for CodonFM codon-substitution scoring.

    Attributes:
        model_checkpoint (CodonFMCheckpoint): Encodon checkpoint to run.
        batch_size (int): Mutations processed per GPU forward pass.
    """


_DNA = frozenset("ACGT")


def _validate_codon(codon: str, field: str) -> str:
    """Uppercase + validate a single 3-nt DNA codon (RNA U mapped to T)."""
    c = codon.upper().replace("U", "T")
    if len(c) != 3 or any(ch not in _DNA for ch in c):
        raise ValueError(f"{field} must be a 3-nucleotide DNA codon (A/C/G/T); got {codon!r}")
    return c


class CodonFMMutation(BaseModel):
    """A single codon substitution to score against its reference coding sequence.

    Attributes:
        sequence (str): The reference coding sequence (codon-aligned DNA/RNA).
        codon_position (int): 1-based codon position of the substitution within ``sequence``.
        ref_codon (str): Reference codon at ``codon_position`` (must match ``sequence``).
        alt_codon (str): Alternate codon substituted at ``codon_position``.
    """

    sequence: str = Field(title="Sequence", description="Reference coding sequence (codon-aligned)")
    codon_position: int = Field(title="Codon Position", ge=1, description="1-based codon position of the substitution")
    ref_codon: str = Field(title="Reference Codon", description="Reference codon (3 nt) at codon_position")
    alt_codon: str = Field(title="Alternate Codon", description="Alternate codon (3 nt) substituted at codon_position")

    @model_validator(mode="after")
    def _validate(self) -> "CodonFMMutation":
        """Normalize the sequence/codons and check the position + reference agreement."""
        self.sequence = normalize_codon_sequence(self.sequence)
        self.ref_codon = _validate_codon(self.ref_codon, "ref_codon")
        self.alt_codon = _validate_codon(self.alt_codon, "alt_codon")
        n_codons = len(self.sequence) // 3
        if self.codon_position > n_codons:
            raise ValueError(f"codon_position {self.codon_position} is out of range for a {n_codons}-codon sequence")
        start = (self.codon_position - 1) * 3
        actual = self.sequence[start : start + 3]
        if actual != self.ref_codon:
            raise ValueError(
                f"ref_codon {self.ref_codon!r} does not match the sequence codon {actual!r} at position {self.codon_position}"
            )
        return self


class CodonFMScoreInput(BaseToolInput):
    """Codon substitutions to score with CodonFM.

    Attributes:
        mutations (list[CodonFMMutation]): One or more codon substitutions, each against its own
            reference coding sequence.
    """

    mutations: list[CodonFMMutation] = InputField(
        title="Mutations",
        description="Codon substitutions to score (each carries its reference sequence + ref/alt codon + position).",
        min_length=1,
    )

    def __len__(self) -> int:
        """Return the number of mutations."""
        return len(self.mutations)


class CodonFMMutationResult(BaseModel):
    """Per-mutation CodonFM scoring result.

    Attributes:
        sequence (str): Reference coding sequence used as model context.
        sequence_length (int): Length of the reference sequence in nucleotides.
        codon_position (int): 1-based codon position of the substitution.
        ref_codon (str): Reference codon.
        alt_codon (str): Alternate codon.
        ref_log_likelihood (float): Model log-likelihood of the reference codon at the site.
        alt_log_likelihood (float): Model log-likelihood of the alternate codon at the site.
        llr (float): Log-likelihood ratio ``ref - alt``; positive means the model favors the
            reference and the substitution is model-disfavored.
    """

    sequence: str = Field(title="Sequence", description="Reference coding sequence used as model context")
    sequence_length: int = Field(title="Sequence Length", description="Reference sequence length in nucleotides")
    codon_position: int = Field(title="Codon Position", description="1-based codon position of the substitution")
    ref_codon: str = Field(title="Reference Codon", description="Reference codon")
    alt_codon: str = Field(title="Alternate Codon", description="Alternate codon")
    ref_log_likelihood: float = Field(title="Ref Log-Likelihood", description="Log-likelihood of the reference codon")
    alt_log_likelihood: float = Field(title="Alt Log-Likelihood", description="Log-likelihood of the alternate codon")
    llr: float = Field(title="Log-Likelihood Ratio", description="ref - alt; positive favors the reference codon")


class CodonFMScoreOutput(BaseToolOutput):
    """Output from CodonFM mutation scoring.

    Attributes:
        results (list[CodonFMMutationResult]): Per-mutation log-likelihood-ratio scores.
    """

    results: list[CodonFMMutationResult] = Field(
        default_factory=list, title="Results", description="Per-mutation CodonFM log-likelihood-ratio scores"
    )

    def __len__(self) -> int:
        """Return the number of per-mutation results."""
        return len(self.results)

    def __getitem__(self, index: int) -> CodonFMMutationResult:
        """Return a per-mutation result."""
        return self.results[index]

    def __iter__(self) -> Iterator[CodonFMMutationResult]:  # type: ignore[override]
        """Iterate over per-mutation results."""
        return iter(self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Default export format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export per-mutation log-likelihood-ratio scores as JSON or CSV."""
        rows = [r.model_dump() for r in self.results]
        base = Path(export_path)
        if file_format == "json":
            (base.parent / f"{base.name}.json").write_text(json.dumps(rows, indent=2))
            return
        if file_format == "csv":
            fieldnames = [
                "sequence",
                "sequence_length",
                "codon_position",
                "ref_codon",
                "alt_codon",
                "ref_log_likelihood",
                "alt_log_likelihood",
                "llr",
            ]
            with open(base.parent / f"{base.name}.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CodonFMScoreInput(
        mutations=[
            CodonFMMutation(
                sequence="ATGGTGAGCAAGGGC",
                codon_position=2,
                ref_codon="GTG",
                alt_codon="GTA",
            ),
        ]
    )


@tool(
    key="codonfm-score",
    label="CodonFM Mutation Score",
    category="masked_models",
    input_class=CodonFMScoreInput,
    config_class=CodonFMScoreConfig,
    output_class=CodonFMScoreOutput,
    description="Score codon substitutions by ref-vs-alt log-likelihood ratio with the CodonFM/Encodon model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["mutations"],
    iterable_output_field="results",
    max_chunk_size=32,
    cacheable=True,
)
def run_codonfm_score(
    inputs: CodonFMScoreInput,
    config: CodonFMScoreConfig,
    instance: Any = None,
) -> CodonFMScoreOutput:
    """Score codon substitutions by CodonFM (Encodon) log-likelihood ratio.

    Args:
        inputs (CodonFMScoreInput): Codon substitutions to score.
        config (CodonFMScoreConfig): CodonFM runtime and checkpoint configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CodonFMScoreOutput: Per-mutation reference/alternate log-likelihoods and their ratio.
    """
    safetensors_url, config_url, filename, subdir = resolve_checkpoint_source(config.model_checkpoint)
    mutation_dicts = [
        {
            "sequence": m.sequence,
            "codon_position": m.codon_position - 1,
            "ref_codon": m.ref_codon,
            "alt_codon": m.alt_codon,
        }
        for m in inputs.mutations
    ]

    output_data = ToolInstance.dispatch(
        "codonfm",
        {
            "operation": "score",
            "mutations": mutation_dicts,
            "safetensors_url": safetensors_url,
            "config_url": config_url,
            "safetensors_filename": filename,
            "cache_subdir": subdir,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    scored = output_data["mutations"]
    if len(scored) != len(inputs.mutations):
        raise ValueError(f"Expected {len(inputs.mutations)} CodonFM mutation scores, got {len(scored)}")
    numeric_rows = [
        (float(row["ref_log_likelihood"]), float(row["alt_log_likelihood"]), float(row["llr"])) for row in scored
    ]
    if any(not all(math.isfinite(value) for value in row) for row in numeric_rows):
        raise ValueError("CodonFM returned a non-finite mutation score")
    if any(not math.isclose(ref_ll - alt_ll, llr, rel_tol=1e-5, abs_tol=1e-6) for ref_ll, alt_ll, llr in numeric_rows):
        raise ValueError("CodonFM returned an inconsistent mutation log-likelihood ratio")

    return CodonFMScoreOutput(
        metadata={"model_checkpoint": config.model_checkpoint, "num_mutations": len(inputs.mutations)},
        results=[
            CodonFMMutationResult(
                sequence=mutation.sequence,
                sequence_length=len(mutation.sequence),
                codon_position=mutation.codon_position,
                ref_codon=mutation.ref_codon,
                alt_codon=mutation.alt_codon,
                ref_log_likelihood=values[0],
                alt_log_likelihood=values[1],
                llr=values[2],
            )
            for mutation, values in zip(inputs.mutations, numeric_rows, strict=True)
        ],
    )

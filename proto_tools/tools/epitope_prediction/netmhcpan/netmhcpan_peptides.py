"""netMHCpan MHC class I binding prediction for explicit peptides."""

import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.tools.epitope_prediction.netmhcpan.shared_data_models import (
    NetMHCpanBinding,
    NetMHCpanConfig,
    NetMHCpanMetrics,
    binding_from_record,
    summarize,
    validate_peptide,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, InputField, ToolInstance

logger = logging.getLogger(__name__)

# One peptide against a panel of alleles is microseconds of arithmetic; the cost is process
# startup. Chunk generously so fan-out amortizes the binary's startup over many peptides.
_MAX_CHUNK_SIZE = 2048


class NetMHCpanPeptidesInput(BaseToolInput):
    """Input for netMHCpan prediction on explicit peptides.

    Attributes:
        peptides (list[str]): MHC class I peptides (8-14 standard uppercase amino acids),
            each scored against every configured allele. A single string is normalized to
            a one-element list.
    """

    peptides: list[str] = InputField(
        title="Peptides",
        description="MHC class I peptides (8-14 residues) to score against every allele.",
    )

    @field_validator("peptides", mode="before")
    @classmethod
    def _normalize_to_list(cls, value: Any) -> Any:
        """Normalize a single peptide string into a one-element list."""
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("peptides")
    @classmethod
    def _validate_peptides(cls, peptides: list[str]) -> list[str]:
        """Require a non-empty list of in-range, standard-alphabet peptides."""
        if not peptides:
            raise ValueError("peptides must contain at least one peptide")
        for index, peptide in enumerate(peptides):
            validate_peptide(peptide, index)
        return peptides


class NetMHCpanPeptidesConfig(NetMHCpanConfig):
    """Configuration for netMHCpan prediction on explicit peptides.

    Carries no peptide-length setting: netMHCpan ignores ``-l`` in peptide mode and scores
    each peptide at its own length.

    Attributes:
        alleles (list[str]): MHC class I alleles to predict against. Accepts the
            ``HLA-A02:01``, ``HLA-A*02:01`` and ``A*02:01`` spellings. Defaults to
            netMHCpan's own default allele.
        include_binding_affinity (bool): Also predict binding affinity (``-BA``), adding
            BA score, BA %rank and nM affinity alongside the eluted-ligand columns.
        strong_rank (float): %rank at or below which a peptide is a strong binder.
        weak_rank (float): %rank at or below which a peptide is a weak binder.
        netmhcpan_dir (str | None): Path to a local netMHCpan install root.
    """


class PeptidePrediction(NetMHCpanMetrics):
    """Every allele's prediction for one peptide, plus its summary metrics.

    Attributes:
        peptide (str): The scored peptide.
        bindings (list[NetMHCpanBinding]): One prediction per configured allele, in the
            order the alleles were configured.
    """

    peptide: str = Field(title="Peptide", description="The scored peptide sequence.")
    bindings: list[NetMHCpanBinding] = Field(
        title="Bindings",
        description="One prediction per configured allele, in configured order.",
    )


class NetMHCpanPeptidesOutput(BaseToolOutput):
    """Output from netMHCpan prediction on explicit peptides.

    Attributes:
        predictions (list[PeptidePrediction]): One entry per input peptide, index-aligned
            with ``inputs.peptides``.
        netmhcpan_version (str | None): Version string reported by the invoked binary.
    """

    predictions: list[PeptidePrediction] = Field(
        title="Predictions",
        description="One entry per input peptide, index-aligned with the inputs.",
    )
    netmhcpan_version: str | None = Field(
        default=None,
        title="netMHCpan Version",
        description="Version reported by the invoked binary; results differ across versions.",
    )

    def __len__(self) -> int:
        """Number of peptides scored."""
        return len(self.predictions)

    def __getitem__(self, index: int) -> PeptidePrediction:
        """Get one peptide's predictions by index."""
        return self.predictions[index]

    @property
    def output_format_options(self) -> list[str]:
        """Valid file formats for exporting the predictions."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Default export file format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Write one row per peptide-allele prediction to ``export_path``."""
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            path.write_text(self.model_dump_json(indent=2))
            return
        if file_format != "csv":
            raise ValueError(f"Invalid file format: {file_format}")

        import csv

        columns = [
            "peptide",
            "allele",
            "score_el",
            "rank_el",
            "binder_class",
            "score_ba",
            "rank_ba",
            "affinity_nm",
        ]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for prediction in self.predictions:
                for binding in prediction.bindings:
                    row = binding.model_dump(include=set(columns))
                    row["peptide"] = prediction.peptide
                    writer.writerow(row)


def example_input() -> NetMHCpanPeptidesInput:
    """Minimal valid input for testing and examples."""
    return NetMHCpanPeptidesInput(peptides=["AAAWYLWEV", "AAGLQDCTM"])


@tool(
    key="netmhcpan-peptides",
    local_only=(
        "netmhcpan-peptides requires a local netMHCpan install, whose academic license forbids "
        "transferring it to a remote worker. Run locally with device='cpu'."
    ),
    label="netMHCpan Peptides",
    category="epitope_prediction",
    input_class=NetMHCpanPeptidesInput,
    config_class=NetMHCpanPeptidesConfig,
    output_class=NetMHCpanPeptidesOutput,
    description="Predict MHC class I binding for explicit peptides with netMHCpan",
    metrics_class=NetMHCpanMetrics,
    example_input=example_input,
    uses_gpu=False,
    cacheable=True,
    iterable_input_fields=["peptides"],
    iterable_output_field="predictions",
    max_chunk_size=_MAX_CHUNK_SIZE,
)
def run_netmhcpan_peptides(
    inputs: NetMHCpanPeptidesInput,
    config: NetMHCpanPeptidesConfig,
    instance: Any = None,
) -> NetMHCpanPeptidesOutput:
    """Predict MHC class I binding for peptides that are already the right length.

    Each peptide is scored against every configured allele in a single netMHCpan
    invocation. Peptide length is taken from each peptide itself: netMHCpan ignores its
    ``-l`` flag in peptide mode, so this tool deliberately exposes no length setting (use
    ``netmhcpan-protein`` to enumerate k-mers from a protein instead).

    ``rank_el`` is the quantity to threshold on and to compare across alleles; raw scores
    sit on different per-allele background distributions.

    Args:
        inputs (NetMHCpanPeptidesInput): Peptides to score.
        config (NetMHCpanPeptidesConfig): Allele panel, thresholds, and install location.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        NetMHCpanPeptidesOutput: One prediction bundle per input peptide, index-aligned
            with ``inputs.peptides``.

    Example:
        >>> inputs = NetMHCpanPeptidesInput(peptides=["AAAWYLWEV"])
        >>> out = run_netmhcpan_peptides(inputs, NetMHCpanPeptidesConfig(alleles=["HLA-A*02:01"]))
        >>> out.predictions[0].bindings[0].rank_el < 1.0
        True
    """
    logger.debug("Using local venv for netmhcpan peptide prediction")
    input_data = {
        "operation": "predict_peptides",
        "peptides": inputs.peptides,
        "alleles": config.alleles,
        "include_binding_affinity": config.include_binding_affinity,
        "strong_rank": config.strong_rank,
        "weak_rank": config.weak_rank,
        "netmhcpan_dir": config.netmhcpan_dir,
        "device": "cpu",
    }

    output_data = ToolInstance.dispatch("netmhcpan", input_data, instance=instance, config=config)

    # Key by (allele, peptide) rather than relying on row order. Repeated peptides share a
    # row, which is correct because the prediction is identical, and a missing key is a
    # parser or allele mismatch that must surface rather than silently drop a peptide.
    by_key = {
        (binding.allele, binding.peptide): binding
        for binding in (
            binding_from_record(record, config.strong_rank, config.weak_rank) for record in output_data["records"]
        )
    }

    predictions = []
    for peptide in inputs.peptides:
        bindings = []
        for allele in config.alleles:
            binding = by_key.get((allele, peptide))
            if binding is None:
                raise ValueError(
                    f"netMHCpan returned no prediction for peptide {peptide!r} and allele {allele!r}; "
                    f"got {len(by_key)} rows covering {sorted({key[0] for key in by_key})}"
                )
            bindings.append(binding)
        predictions.append(
            PeptidePrediction(peptide=peptide, bindings=bindings, **summarize(bindings, config.weak_rank))
        )

    return NetMHCpanPeptidesOutput(
        predictions=predictions,
        netmhcpan_version=output_data["version"],
        metadata={
            "num_peptides": len(inputs.peptides),
            "alleles": config.alleles,
            "include_binding_affinity": config.include_binding_affinity,
            "netmhcpan_version": output_data["version"],
        },
    )

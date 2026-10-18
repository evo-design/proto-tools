"""netMHCpan MHC class I epitope scanning across protein sequences."""

import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.tools.epitope_prediction.netmhcpan.shared_data_models import (
    MAX_PEPTIDE_LENGTH,
    MIN_PEPTIDE_LENGTH,
    NetMHCpanBinding,
    NetMHCpanConfig,
    NetMHCpanProteinMetrics,
    binding_from_record,
    summarize,
    validate_protein,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, ConfigField, InputField, ToolInstance

logger = logging.getLogger(__name__)

# A protein scan is one row per window per allele, so a chunk covers far fewer sequences
# than the peptide tool covers peptides.
_MAX_CHUNK_SIZE = 64

# The standalone writes synthetic FASTA ids because netMHCpan truncates the Identity column
# to 15 characters, which would make user-supplied headers collide.
_SYNTHETIC_ID_PREFIX = "s"


class NetMHCpanProteinInput(BaseToolInput):
    """Input for netMHCpan epitope scanning across protein sequences.

    Attributes:
        sequences (list[str]): Protein sequences (standard uppercase amino acids) to
            enumerate k-mers from and score. A single string is normalized to a
            one-element list.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="Protein sequences to enumerate class I k-mers from and score.",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def _normalize_to_list(cls, value: Any) -> Any:
        """Normalize a single sequence string into a one-element list."""
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("sequences")
    @classmethod
    def _validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Require a non-empty list of standard-alphabet sequences long enough to window."""
        if not sequences:
            raise ValueError("sequences must contain at least one sequence")
        for index, sequence in enumerate(sequences):
            validate_protein(sequence, index)
        return sequences


class NetMHCpanProteinConfig(NetMHCpanConfig):
    """Configuration for netMHCpan epitope scanning across proteins.

    Attributes:
        peptide_lengths (list[int]): k-mer lengths to enumerate from each sequence
            (netMHCpan ``-l``). Defaults to the netMHCpan default of 8-11.
        alleles (list[str]): MHC class I alleles to predict against. Accepts the
            ``HLA-A02:01``, ``HLA-A*02:01`` and ``A*02:01`` spellings. Defaults to
            the netMHCpan default allele.
        include_binding_affinity (bool): Also predict binding affinity (``-BA``), adding
            BA score, BA %rank and nM affinity alongside the eluted-ligand columns.
        strong_rank (float): %rank at or below which a peptide is a strong binder.
        weak_rank (float): %rank at or below which a peptide is a weak binder.
        netmhcpan_dir (str | None): Path to a local netMHCpan install root.
    """

    peptide_lengths: list[int] = ConfigField(
        default=[8, 9, 10, 11],
        title="Peptide Lengths",
        description="k-mer lengths to enumerate from each sequence; netMHCpan's -l flag.",
    )

    @field_validator("peptide_lengths")
    @classmethod
    def _validate_lengths(cls, lengths: list[int]) -> list[int]:
        """Require a non-empty, in-range, de-duplicated set of k-mer lengths."""
        if not lengths:
            raise ValueError("peptide_lengths must contain at least one length")
        for length in lengths:
            if not MIN_PEPTIDE_LENGTH <= length <= MAX_PEPTIDE_LENGTH:
                raise ValueError(
                    f"peptide_lengths contains {length}; MHC class I peptide lengths must be "
                    f"{MIN_PEPTIDE_LENGTH}-{MAX_PEPTIDE_LENGTH}"
                )
        return sorted(set(lengths))


class ProteinScan(NetMHCpanProteinMetrics):
    """Every window's prediction for one protein, plus its summary metrics.

    Attributes:
        sequence_length (int): Length of the scanned protein, in residues.
        num_windows (int): Distinct peptide windows enumerated from the sequence.
        bindings (list[NetMHCpanBinding]): One prediction per window per allele.
    """

    sequence_length: int = Field(title="Sequence Length", description="Length of the scanned protein in residues.")
    num_windows: int = Field(title="Window Count", description="Distinct peptide windows enumerated.")
    bindings: list[NetMHCpanBinding] = Field(
        title="Bindings",
        description="One prediction per enumerated window per configured allele.",
    )


class NetMHCpanProteinOutput(BaseToolOutput):
    """Output from netMHCpan epitope scanning across proteins.

    Attributes:
        results (list[ProteinScan]): One entry per input sequence, index-aligned with
            ``inputs.sequences``.
        netmhcpan_version (str | None): Version string reported by the invoked binary.
    """

    results: list[ProteinScan] = Field(
        title="Results",
        description="One entry per input sequence, index-aligned with the inputs.",
    )
    netmhcpan_version: str | None = Field(
        default=None,
        title="netMHCpan Version",
        description="Version reported by the invoked binary; results differ across versions.",
    )

    def __len__(self) -> int:
        """Number of proteins scanned."""
        return len(self.results)

    def __getitem__(self, index: int) -> ProteinScan:
        """Get one protein's scan by index."""
        return self.results[index]

    @property
    def output_format_options(self) -> list[str]:
        """Valid file formats for exporting the scans."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Default export file format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Write one row per window-allele prediction to ``export_path``."""
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            path.write_text(self.model_dump_json(indent=2))
            return
        if file_format != "csv":
            raise ValueError(f"Invalid file format: {file_format}")

        import csv

        columns = [
            "sequence_index",
            "position",
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
            for index, scan in enumerate(self.results):
                for binding in scan.bindings:
                    row = binding.model_dump(include=set(columns))
                    row["sequence_index"] = index
                    writer.writerow(row)


def example_input() -> NetMHCpanProteinInput:
    """Minimal valid input for testing and examples."""
    return NetMHCpanProteinInput(sequences=["MKTIIALSYIFCLVFAQKLPGNDNSTATLCLGHHAV"])


@tool(
    key="netmhcpan-protein",
    local_only=(
        "netmhcpan-protein requires a local netMHCpan install, whose academic license forbids "
        "transferring it to a remote worker. Run locally with device='cpu'."
    ),
    label="netMHCpan Protein Scan",
    category="epitope_prediction",
    input_class=NetMHCpanProteinInput,
    config_class=NetMHCpanProteinConfig,
    output_class=NetMHCpanProteinOutput,
    description="Scan protein sequences for MHC class I binding peptides with netMHCpan",
    metrics_class=NetMHCpanProteinMetrics,
    example_input=example_input,
    uses_gpu=False,
    cacheable=True,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=_MAX_CHUNK_SIZE,
)
def run_netmhcpan_protein(
    inputs: NetMHCpanProteinInput,
    config: NetMHCpanProteinConfig,
    instance: Any = None,
) -> NetMHCpanProteinOutput:
    """Scan proteins for MHC class I binding peptides, windowing k-mers internally.

    Every ``peptide_lengths`` window of every sequence is scored against every configured
    allele in a single netMHCpan invocation. Doing the windowing inside the tool is the
    point: it is the common case for screening a design, and ``position`` is reported
    1-indexed against the supplied sequence.

    ``rank_el`` is the quantity to threshold on and to compare across alleles. Alongside
    the strong/weak counts, ``binding_propensity`` gives a smooth 0-1 summary suitable as
    an optimization target, since a hard threshold makes a piecewise-constant objective.

    Args:
        inputs (NetMHCpanProteinInput): Protein sequences to scan.
        config (NetMHCpanProteinConfig): Alleles, k-mer lengths, thresholds, install path.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        NetMHCpanProteinOutput: One scan per input sequence, index-aligned with
            ``inputs.sequences``.

    Example:
        >>> inputs = NetMHCpanProteinInput(sequences=["MKTIIALSYIFCLVFAQKLPGNDNSTATLCLGHHAV"])
        >>> out = run_netmhcpan_protein(inputs, NetMHCpanProteinConfig(peptide_lengths=[9]))
        >>> out.results[0].num_windows
        28
    """
    logger.debug("Using local venv for netmhcpan protein scanning")
    input_data = {
        "operation": "predict_protein",
        "sequences": inputs.sequences,
        "peptide_lengths": config.peptide_lengths,
        "alleles": config.alleles,
        "include_binding_affinity": config.include_binding_affinity,
        "strong_rank": config.strong_rank,
        "weak_rank": config.weak_rank,
        "netmhcpan_dir": config.netmhcpan_dir,
        "device": "cpu",
    }

    output_data = ToolInstance.dispatch("netmhcpan", input_data, instance=instance, config=config)

    # Group by the synthetic FASTA id the standalone wrote, which is the input index. Row
    # order is not assumed: netMHCpan groups its output by peptide length and allele.
    grouped: dict[str, list[NetMHCpanBinding]] = {}
    for record in output_data["records"]:
        grouped.setdefault(record["Identity"], []).append(
            binding_from_record(record, config.strong_rank, config.weak_rank)
        )

    results = []
    for index, sequence in enumerate(inputs.sequences):
        bindings = grouped.get(f"{_SYNTHETIC_ID_PREFIX}{index}", [])
        expected = sum(max(0, len(sequence) - length + 1) for length in config.peptide_lengths) * len(config.alleles)
        if len(bindings) != expected:
            raise ValueError(
                f"netMHCpan returned {len(bindings)} rows for sequences[{index}] but "
                f"{expected} were expected ({len(config.peptide_lengths)} length(s) x "
                f"{len(config.alleles)} allele(s) over {len(sequence)} residues)"
            )
        metrics = summarize(bindings, config.weak_rank)
        strong_per_100 = (
            metrics["strong_binder_count"] / (len(sequence) * len(config.alleles)) * 100.0 if sequence else 0.0
        )
        results.append(
            ProteinScan(
                sequence_length=len(sequence),
                num_windows=len({(binding.position, binding.peptide) for binding in bindings}),
                bindings=bindings,
                strong_binders_per_100_residues=strong_per_100,
                **metrics,
            )
        )

    return NetMHCpanProteinOutput(
        results=results,
        netmhcpan_version=output_data["version"],
        metadata={
            "num_sequences": len(inputs.sequences),
            "alleles": config.alleles,
            "peptide_lengths": config.peptide_lengths,
            "include_binding_affinity": config.include_binding_affinity,
            "netmhcpan_version": output_data["version"],
        },
    )

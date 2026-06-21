"""X3DNA fiber idealized nucleic-acid duplex builder."""

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)

# Canonical fiber form -> X3DNA `fiber` CLI flag. A/B/Z build the corresponding DNA
# helices; RNA builds an A-form RNA duplex (fiber's -rna).
_FORM_FLAGS: dict[str, str] = {"A-DNA": "-a", "B-DNA": "-b", "Z-DNA": "-z", "RNA": "-rna"}
_VALID_BASES = frozenset("ACGTU")


class X3DNAFiberInput(BaseToolInput):
    """Input for X3DNA fiber idealized-duplex generation.

    Attributes:
        sequences (list[str]): Nucleotide sequences (5'->3') for one strand of each
            duplex; the complementary strand is generated automatically. Bases must be
            A/C/G/T/U (any case). T and U are interconverted to match the requested
            ``form`` (T for DNA forms, U for RNA). A single string is normalized to a
            one-element list.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="Nucleotide sequences (one strand, 5'->3'); the complement is generated.",
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
        """Require a non-empty list of non-empty A/C/G/T/U sequences."""
        if not sequences:
            raise ValueError("sequences must contain at least one sequence")
        for index, sequence in enumerate(sequences):
            if not sequence:
                raise ValueError(f"sequence {index} is empty")
            invalid = set(sequence.upper()) - _VALID_BASES
            if invalid:
                raise ValueError(f"sequence {index} has invalid base(s) {sorted(invalid)}; only A/C/G/T/U are allowed")
        return sequences


class X3DNAFiberConfig(BaseConfig):
    """Configuration for X3DNA fiber idealized-duplex generation.

    Attributes:
        form (Literal['A-DNA', 'B-DNA', 'Z-DNA', 'RNA']): Canonical helix form to build:
            ``A-DNA``, ``B-DNA`` (default), ``Z-DNA``, or ``RNA`` (A-form RNA duplex).
        single_stranded (bool): Output only the single (sense) strand instead of the
            full duplex (fiber ``-single``). Default ``False``.
        x3dna_dir (str | None): Path to a local X3DNA v2.4 install root (the directory
            containing ``bin/fiber``). Overrides the ``X3DNA`` environment variable and
            the resolved tool cache. X3DNA is user-provisioned (CC-BY-NC-4.0); see the
            tool README.
    """

    form: Literal["A-DNA", "B-DNA", "Z-DNA", "RNA"] = ConfigField(
        default="B-DNA",
        title="Helix Form",
        description="Canonical fiber model: A-DNA, B-DNA, Z-DNA, or RNA (A-form RNA duplex).",
    )
    single_stranded: bool = ConfigField(
        default=False,
        title="Single Stranded",
        description="Output only the sense strand instead of the duplex (fiber -single).",
    )
    x3dna_dir: str | None = ConfigField(
        default=None,
        title="X3DNA Directory",
        description="Local X3DNA v2.4 install root (contains bin/fiber); overrides $X3DNA.",
    )

    def cloud_unsupported_reason(self) -> str | None:
        """X3DNA is a user-provisioned local binary not available on a hosted worker."""
        return (
            "x3dna-fiber requires a local X3DNA v2.4 install (bin/fiber) not available on "
            "device='cloud'. Run locally with device='cpu'."
        )


class X3DNAFiberOutput(BaseToolOutput):
    """Output from X3DNA fiber generation.

    Attributes:
        structures (list[Structure]): Idealized duplex (or single-strand) structures,
            one per input sequence, index-aligned with ``inputs.sequences``.
    """

    structures: list[Structure] = Field(
        title="Structures",
        description="Idealized fiber structures, one per input sequence.",
    )

    def __len__(self) -> int:
        """Number of generated structures."""
        return len(self.structures)

    def __getitem__(self, index: int) -> Structure:
        """Get a generated structure by index."""
        return self.structures[index]

    @property
    def output_format_options(self) -> list[str]:
        """Valid file formats for exporting the structures."""
        return ["pdb", "cif"]

    @property
    def output_format_default(self) -> str:
        """Default export file format."""
        return "pdb"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Write each structure to ``structure_{i}.{ext}`` under ``export_path``."""
        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)
        for index, structure in enumerate(self.structures):
            if file_format == "pdb":
                structure.write_pdb(path / f"structure_{index}.pdb")
            elif file_format == "cif":
                structure.write_cif(path / f"structure_{index}.cif")
            else:
                raise ValueError(f"Invalid file format: {file_format}")


def example_input() -> X3DNAFiberInput:
    """Minimal valid input for testing and examples."""
    return X3DNAFiberInput(sequences=["GGGCAAAATGCACTGCACTTTGGG"])


def _normalize_bases(sequence: str, form: str) -> str:
    """Upper-case and interconvert T/U so the sequence matches the requested form."""
    sequence = sequence.upper()
    if form == "RNA":
        return sequence.replace("T", "U")
    return sequence.replace("U", "T")


@tool(
    key="x3dna-fiber",
    label="X3DNA Fiber",
    category="structure_prediction",
    input_class=X3DNAFiberInput,
    config_class=X3DNAFiberConfig,
    output_class=X3DNAFiberOutput,
    description="Build idealized fiber DNA/RNA duplexes from base sequences with X3DNA fiber",
    example_input=example_input,
    uses_gpu=False,
    iterable_input_fields=["sequences"],
    iterable_output_field="structures",
)
def run_x3dna_fiber(
    inputs: X3DNAFiberInput,
    config: X3DNAFiberConfig,
    instance: Any = None,
) -> X3DNAFiberOutput:
    """Generate idealized fiber nucleic-acid structures from base sequences.

    Wraps X3DNA v2.4's ``fiber`` program, which builds canonical (Arnott) fiber models.
    For each input sequence the requested ``form`` duplex is generated (sense strand plus
    its generated complement, or just the sense strand when ``single_stranded`` is set).
    X3DNA must be installed locally (it is user-provisioned under CC-BY-NC-4.0); the
    runner resolves it from ``config.x3dna_dir``, then ``$X3DNA``, then the tool cache.

    Args:
        inputs (X3DNAFiberInput): Sequences to build idealized structures for.
        config (X3DNAFiberConfig): Form, single-strand, and X3DNA-path settings.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        X3DNAFiberOutput: Idealized structures, one per input sequence, index-aligned
            with ``inputs.sequences``.

    Example:
        >>> inputs = X3DNAFiberInput(sequences=["GGGCAAAATGCACTGCACTTTGGG"])
        >>> out = run_x3dna_fiber(inputs, X3DNAFiberConfig(form="B-DNA"))
        >>> duplex = out.structures[0]  # idealized B-form duplex
    """
    logger.debug("Using local venv for x3dna fiber generation")
    input_data = {
        "sequences": [_normalize_bases(s, config.form) for s in inputs.sequences],
        "form_flag": _FORM_FLAGS[config.form],
        "single_stranded": config.single_stranded,
        "x3dna_dir": config.x3dna_dir,
    }

    output_data = ToolInstance.dispatch("x3dna", input_data, instance=instance, config=config)

    structures = [
        Structure(structure=pdb, structure_format="pdb", source="x3dna-fiber") for pdb in output_data["structures"]
    ]
    return X3DNAFiberOutput(structures=structures, metadata={"num_sequences": len(structures), "form": config.form})

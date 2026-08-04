"""Primer3 oligonucleotide thermodynamics scoring tool."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

_VALID_BASES = set("ACGT")


def _validate_oligo_sequence(sequence: str) -> str:
    """Validate and normalize a DNA oligo: strip, uppercase, allow only A/C/G/T."""
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("oligo sequence must not be empty")
    invalid = sorted(set(normalized) - _VALID_BASES)
    if invalid:
        raise ValueError(f"oligo sequence has non-ACGT characters {invalid}; thermodynamics needs concrete bases")
    return normalized


# ============================================================================
# Data Models
# ============================================================================
class Primer3Oligo(BaseModel):
    """A DNA oligo to score, optionally paired with a partner for heterodimer ΔG.

    The partner is bundled with the oligo (rather than passed as a parallel
    list) so the two stay aligned through the framework's per-item batching.

    Attributes:
        sequence (str): DNA oligo (A/C/G/T only). Scored for Tm, hairpin,
            homodimer, GC content, and 3' GC-clamp.
        partner (str | None): Optional second oligo. When set, heterodimer ΔG
            is computed between ``sequence`` and ``partner`` (e.g. a forward
            primer with its reverse partner). None ⇒ heterodimer_dg is null.
    """

    sequence: str = Field(title="Sequence", description="DNA oligo to score (A/C/G/T only)")
    partner: str | None = Field(
        default=None,
        title="Partner",
        description="Optional second oligo for heterodimer scoring (e.g. the reverse primer)",
    )

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return _validate_oligo_sequence(value)

    @field_validator("partner")
    @classmethod
    def _validate_partner(cls, value: str | None) -> str | None:
        return None if value is None else _validate_oligo_sequence(value)


class Primer3OligoResult(BaseModel):
    """Thermodynamic scores for a single oligo.

    ΔG values are reported in kcal/mol at the configured temperature. A ΔG of
    0.0 with the corresponding ``*_structure_found`` False means no significant
    secondary structure was found — the favorable case.

    Attributes:
        oligo_id (str): Positional label of the input oligo (``oligo_0`` ...).
        length (int): Oligo length in nucleotides.
        tm (float): Melting temperature in °C.
        hairpin_dg (float): Hairpin ΔG in kcal/mol (more negative = more stable).
        homodimer_dg (float): Self-dimer ΔG in kcal/mol.
        heterodimer_dg (float | None): Cross-dimer ΔG with the partner, or None
            when no partner was supplied.
        gc_content (float): Fraction of G/C bases, 0-1.
        gc_clamp (bool): True if either of the last two 3' bases is G or C.
        hairpin_structure_found (bool): Whether a hairpin structure was found.
        homodimer_structure_found (bool): Whether a homodimer structure was found.
        heterodimer_structure_found (bool | None): Whether a heterodimer was
            found, or None when no partner was supplied.
    """

    oligo_id: str = Field(title="Oligo ID", description="Positional label of the input oligo")
    length: int = Field(title="Length", description="Oligo length in nucleotides")
    tm: float = Field(title="Tm", description="Melting temperature in °C")
    hairpin_dg: float = Field(title="Hairpin ΔG", description="Hairpin ΔG in kcal/mol")
    homodimer_dg: float = Field(title="Homodimer ΔG", description="Self-dimer ΔG in kcal/mol")
    heterodimer_dg: float | None = Field(
        default=None, title="Heterodimer ΔG", description="Cross-dimer ΔG with the partner in kcal/mol"
    )
    gc_content: float = Field(title="GC Content", description="Fraction of G/C bases (0-1)")
    gc_clamp: bool = Field(title="GC Clamp", description="True if a G/C is in the last two 3' bases")
    hairpin_structure_found: bool = Field(
        title="Hairpin Structure Found", description="Whether a hairpin structure was found"
    )
    homodimer_structure_found: bool = Field(
        title="Homodimer Structure Found", description="Whether a homodimer structure was found"
    )
    heterodimer_structure_found: bool | None = Field(
        default=None,
        title="Heterodimer Structure Found",
        description="Whether a heterodimer structure was found (None if no partner)",
    )


# Input:
class Primer3ThermodynamicsInput(BaseToolInput):
    """Input for Primer3 oligo thermodynamics scoring.

    Attributes:
        oligos (list[Primer3Oligo]): Oligos to score. A bare DNA string, a
            ``{"sequence": ..., "partner": ...}`` dict, or a single oligo are
            all accepted and coerced to a one-element list. Results are returned
            in input order.
    """

    oligos: list[Primer3Oligo] = InputField(
        title="Oligos",
        description="DNA oligos to score (each optionally paired with a partner for heterodimer ΔG)",
    )

    @field_validator("oligos", mode="before")
    @classmethod
    def _coerce_oligos(cls, value: Any) -> Any:
        """Accept a bare str/dict/Primer3Oligo (or a list of them) and normalize to a list."""
        if isinstance(value, (str, dict, Primer3Oligo)):
            value = [value]
        if not isinstance(value, list):
            return value
        return [Primer3Oligo(sequence=item) if isinstance(item, str) else item for item in value]


# Config:
class Primer3ThermodynamicsConfig(BaseConfig):
    """Configuration for Primer3 thermodynamics.

    Defaults match primer3-py's own defaults so results are reproducible against
    Primer3 directly. For qPCR, common practice is mv_conc≈50, dv_conc≈1.5-3,
    dntp_conc≈0.8, dna_conc≈200-250; see the README for target ranges.

    Attributes:
        mv_conc (float): Monovalent cation concentration in mM. Default 50.0.
        dv_conc (float): Divalent cation (Mg2+) concentration in mM. Default 1.5.
        dntp_conc (float): dNTP concentration in mM. Default 0.6.
        dna_conc (float): Oligo (DNA) concentration in nM. Default 50.0.
        temp_c (float): Temperature in °C for hairpin/dimer ΔG. Default 37.0.
    """

    mv_conc: float = ConfigField(
        title="Monovalent Cation (mM)",
        default=50.0,
        ge=0.0,
        description="Monovalent cation concentration in mM (e.g. Na+, K+)",
    )
    dv_conc: float = ConfigField(
        title="Divalent Cation (mM)",
        default=1.5,
        ge=0.0,
        description="Divalent cation concentration in mM (Mg2+); raises Tm and dimer stability",
    )
    dntp_conc: float = ConfigField(
        title="dNTP (mM)",
        default=0.6,
        ge=0.0,
        description="dNTP concentration in mM; sequesters Mg2+, lowering effective divalent",
    )
    dna_conc: float = ConfigField(
        title="Oligo (nM)",
        default=50.0,
        gt=0.0,
        description="Oligo/DNA concentration in nM; affects Tm and dimer ΔG",
    )
    temp_c: float = ConfigField(
        title="Temperature (°C)",
        default=37.0,
        description="Temperature in °C at which hairpin/homodimer/heterodimer ΔG is evaluated",
    )


# Output:
class Primer3ThermodynamicsOutput(BaseToolOutput):
    """Output from Primer3 thermodynamics scoring.

    Attributes:
        results (list[Primer3OligoResult]): Per-oligo scores, in input order.
    """

    results: list[Primer3OligoResult] = Field(
        default_factory=list,
        title="Results",
        description="Per-oligo thermodynamic scores",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        import pandas as pd

        path = Path(export_path).with_suffix(f".{file_format}")
        df = pd.DataFrame([r.model_dump() for r in self.results])
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input: a GAPDH qPCR forward primer paired with its reverse primer."""
    return Primer3ThermodynamicsInput(
        oligos=[Primer3Oligo(sequence="ACCCACTCCTCCACCTTTGA", partner="CTGTTGCTGTAGCCAAATTCGT")]
    )


@tool(
    key="primer3-thermodynamics",
    label="Primer3 Thermodynamics",
    category="sequence_scoring",
    input_class=Primer3ThermodynamicsInput,
    config_class=Primer3ThermodynamicsConfig,
    output_class=Primer3ThermodynamicsOutput,
    description="Score DNA oligos for Tm, hairpin/homodimer/heterodimer ΔG, GC content, and GC-clamp",
    example_input=example_input,
    iterable_input_fields=["oligos"],
    iterable_output_field="results",
    max_chunk_size=256,
    cacheable=True,
)
def run_primer3_thermodynamics(
    inputs: Primer3ThermodynamicsInput,
    config: Primer3ThermodynamicsConfig,
    instance: Any = None,
) -> Primer3ThermodynamicsOutput:
    """Score DNA oligos with Primer3's nearest-neighbor thermodynamics.

    Wraps primer3-py's ``calc_tm``/``calc_hairpin``/``calc_homodimer``/
    ``calc_heterodimer`` to report, per oligo, the melting temperature, the
    hairpin/homodimer (and optional heterodimer) ΔG, GC content, and whether the
    3' end has a GC clamp. These are the core filters for qPCR and general PCR
    primer design.

    Args:
        inputs (Primer3ThermodynamicsInput): Oligos to score.
        config (Primer3ThermodynamicsConfig): Ionic/oligo conditions.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        Primer3ThermodynamicsOutput: Per-oligo thermodynamic scores.

    Examples:
        >>> result = run_primer3_thermodynamics(
        ...     Primer3ThermodynamicsInput(
        ...         oligos=[{"sequence": "ACCCACTCCTCCACCTTTGA", "partner": "CTGTTGCTGTAGCCAAATTCGT"}]
        ...     ),
        ...     Primer3ThermodynamicsConfig(),
        ... )
        >>> print(f"Tm = {result.results[0].tm:.1f} °C")
    """
    input_data = {
        "oligos": [{"sequence": o.sequence, "partner": o.partner} for o in inputs.oligos],
        "oligo_ids": [f"oligo_{i}" for i in range(len(inputs.oligos))],
        "config": {
            "mv_conc": config.mv_conc,
            "dv_conc": config.dv_conc,
            "dntp_conc": config.dntp_conc,
            "dna_conc": config.dna_conc,
            "temp_c": config.temp_c,
        },
        "device": "cpu",
    }

    output_data = ToolInstance.dispatch(
        "primer3",
        input_data,
        instance=instance,
        config=config,
    )

    results = [Primer3OligoResult(**result_dict) for result_dict in output_data["results"]]

    return Primer3ThermodynamicsOutput(
        metadata={
            "num_oligos": len(inputs.oligos),
            "temp_c": config.temp_c,
        },
        results=results,
    )

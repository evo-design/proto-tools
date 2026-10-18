"""Shared netMHCpan allele handling, binding records, and summary metrics."""

from __future__ import annotations

import math
import re
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.tool_io import Metrics, MetricSpec

# netMHCpan accepts only the unstarred allele spelling ("HLA-A02:01"); "HLA-A*02:01" and
# "A*02:01" are rejected with exit status 1, while its own output table echoes the starred
# form. All three spellings are accepted here and normalized to the one the binary takes.
_STAR = "*"

# A bare HLA locus such as "A02:01" needs the species prefix restored. Non-human prefixes
# (BoLA, Patr, Mamu, Gogo, SLA, Eqca, DLA, H-2) are case-sensitive in MHC_pseudo.dat, so
# only this unambiguous human case is rewritten and nothing is upper-cased.
_BARE_HLA_LOCUS = re.compile(r"^[ABCE]\d")

# MHC class I peptides are 8-14mers; netMHCpan's networks are trained on that range even
# though the binary will score longer peptides. Enforced so out-of-range input fails loudly
# instead of returning predictions outside the model's domain.
MIN_PEPTIDE_LENGTH = 8
MAX_PEPTIDE_LENGTH = 14

# The 20 standard amino acids. netMHCpan silently rewrites any other residue to "X", which
# would mean scoring a peptide the caller never asked about, so off-alphabet input is rejected.
_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Steepness of the logistic used by ``binding_propensity``, applied in log-rank space so the
# curve is centred on the weak-binder threshold and decreases smoothly with %rank. This is
# not a netMHCpan parameter and is not configurable.
_PROPENSITY_STEEPNESS = 2.0

BinderClass = Literal["strong", "weak", "non-binder"]


def normalize_allele(allele: str) -> str:
    """Normalize an MHC allele name to the spelling netMHCpan accepts.

    Strips the ``*`` separator and restores the ``HLA-`` prefix on a bare human locus, so
    ``HLA-A*02:01``, ``A*02:01``, ``A02:01`` and ``HLA-A02:01`` all converge on
    ``HLA-A02:01``. Non-human prefixes are left untouched because MHC_pseudo.dat is
    case-sensitive. Whether the result actually exists is checked against the install's
    allele list at run time, which fails loudly naming the offending allele.

    Args:
        allele (str): Allele name in any of the common spellings.

    Returns:
        str: The allele in netMHCpan's accepted form.

    Raises:
        ValueError: If the name is empty or contains whitespace.
    """
    text = allele.strip().replace(_STAR, "")
    if not text:
        raise ValueError("allele name must not be empty")
    if any(character.isspace() for character in text):
        raise ValueError(f"allele name must not contain whitespace: {allele!r}")
    if _BARE_HLA_LOCUS.match(text):
        text = f"HLA-{text}"
    return text


def normalize_alleles(alleles: list[str]) -> list[str]:
    """Normalize an allele panel, rejecting an empty panel and preserving order.

    Args:
        alleles (list[str]): Allele names in any of the common spellings.

    Returns:
        list[str]: Normalized names, de-duplicated with first-seen order preserved.

    Raises:
        ValueError: If the panel is empty.
    """
    if not alleles:
        raise ValueError("alleles must contain at least one MHC allele")
    seen: dict[str, None] = {}
    for allele in alleles:
        seen.setdefault(normalize_allele(allele), None)
    return list(seen)


def validate_peptide(peptide: str, index: int) -> str:
    """Validate one peptide's alphabet and length.

    Args:
        peptide (str): Candidate peptide sequence.
        index (int): Position in the caller's list, used in error messages.

    Returns:
        str: The peptide unchanged.

    Raises:
        ValueError: If the peptide is off-alphabet or outside the class I length range.
    """
    if not peptide:
        raise ValueError(f"peptides[{index}] is empty")
    invalid = sorted(set(peptide) - _STANDARD_AMINO_ACIDS)
    if invalid:
        raise ValueError(
            f"peptides[{index}]={peptide!r} has non-standard residue(s) {invalid}; netMHCpan "
            "silently rewrites these to X, so only the 20 standard uppercase amino acids are accepted"
        )
    if not MIN_PEPTIDE_LENGTH <= len(peptide) <= MAX_PEPTIDE_LENGTH:
        raise ValueError(
            f"peptides[{index}]={peptide!r} has length {len(peptide)}; MHC class I peptides must be "
            f"{MIN_PEPTIDE_LENGTH}-{MAX_PEPTIDE_LENGTH} residues"
        )
    return peptide


def validate_protein(sequence: str, index: int) -> str:
    """Validate one protein sequence's alphabet and minimum length.

    Args:
        sequence (str): Candidate protein sequence.
        index (int): Position in the caller's list, used in error messages.

    Returns:
        str: The sequence unchanged.

    Raises:
        ValueError: If the sequence is off-alphabet or too short to window.
    """
    if not sequence:
        raise ValueError(f"sequences[{index}] is empty")
    invalid = sorted(set(sequence) - _STANDARD_AMINO_ACIDS)
    if invalid:
        raise ValueError(
            f"sequences[{index}] has non-standard residue(s) {invalid}; netMHCpan silently rewrites "
            "these to X, so only the 20 standard uppercase amino acids are accepted"
        )
    if len(sequence) < MIN_PEPTIDE_LENGTH:
        raise ValueError(
            f"sequences[{index}] has length {len(sequence)}, shorter than the shortest "
            f"class I peptide ({MIN_PEPTIDE_LENGTH})"
        )
    return sequence


def binding_propensity(rank: float, weak_rank: float) -> float:
    """Map a %rank to a smooth 0-1 binding propensity centred on the weak threshold.

    A binder count is piecewise-constant in the underlying sequence, which makes it
    uninformative to a sampler optimizing against immunogenicity. This is a logistic in
    log-rank space: it equals 0.5 exactly at ``weak_rank``, approaches 1 for very strong
    binders and 0 for very weak ones, and is strictly decreasing in ``rank``.

    Args:
        rank (float): netMHCpan %rank for one peptide-allele pair.
        weak_rank (float): Weak-binder %rank threshold the curve is centred on.

    Returns:
        float: Propensity in (0, 1).
    """
    if rank <= 0.0:
        return 1.0
    return 1.0 / (1.0 + (rank / weak_rank) ** _PROPENSITY_STEEPNESS)


class NetMHCpanBinding(BaseModel):
    """One netMHCpan prediction for a single peptide against a single allele.

    ``rank_el`` is the quantity to compare across alleles: raw scores sit on different
    per-allele background distributions, whereas %rank is a percentile against random
    peptides. The binding-affinity fields are populated only when
    ``include_binding_affinity`` is set, because netMHCpan's ``-BA`` flag *adds* them
    alongside the eluted-ligand columns rather than replacing them.

    Attributes:
        allele (str): Allele this prediction is for, in netMHCpan's accepted spelling.
        peptide (str): The scored peptide.
        position (int): 1-indexed start of the peptide in its source sequence.
        core (str): Predicted binding core.
        icore (str): Interaction core, including any inserted or deleted residues.
        score_el (float): Raw eluted-ligand likelihood score.
        rank_el (float): Eluted-ligand %rank against random peptides; lower binds better.
        binder_class (BinderClass): Classification from ``rank_el`` and the thresholds.
        score_ba (float | None): Raw binding-affinity score, when -BA was requested.
        rank_ba (float | None): Binding-affinity %rank, when -BA was requested.
        affinity_nm (float | None): Predicted affinity in nM, when -BA was requested.
    """

    allele: str = Field(title="Allele", description="MHC allele, in netMHCpan's accepted spelling.")
    peptide: str = Field(title="Peptide", description="The scored peptide sequence.")
    position: int = Field(title="Position", description="1-indexed peptide start in its source sequence.")
    core: str = Field(title="Binding Core", description="Predicted binding core of the peptide.")
    icore: str = Field(title="Interaction Core", description="Interaction core including inserted residues.")
    score_el: float = Field(title="EL Score", description="Raw eluted-ligand likelihood score.")
    rank_el: float = Field(title="EL %Rank", description="Eluted-ligand %rank; lower means stronger binding.")
    binder_class: BinderClass = Field(
        title="Binder Class",
        description="Binder strength from the EL %rank and the configured thresholds.",
    )
    score_ba: float | None = Field(
        default=None, title="BA Score", description="Raw binding-affinity score; set only with -BA."
    )
    rank_ba: float | None = Field(
        default=None, title="BA %Rank", description="Binding-affinity %rank; set only with -BA."
    )
    affinity_nm: float | None = Field(
        default=None, title="Affinity (nM)", description="Predicted binding affinity in nM; set only with -BA."
    )


_SHARED_METRIC_SPEC: dict[str, MetricSpec] = {
    "strong_binder_count": {
        "description": "Peptide-allele pairs at or below the strong %rank threshold.",
        "availability": "always",
        "type": "int",
        "min": 0,
        "max": None,
        "unit": None,
        # Fewer binders is better when deimmunizing a therapeutic; more is better when
        # selecting vaccine epitopes. The direction is the caller's, not the tool's.
        "better_values_are": "context-dependent",
    },
    "weak_binder_count": {
        "description": "Pairs between the strong and weak %rank thresholds.",
        "availability": "always",
        "type": "int",
        "min": 0,
        "max": None,
        "unit": None,
        "better_values_are": "context-dependent",
    },
    "strong_binder_fraction": {
        "description": "Strong binders over all pairs; comparable across panels and lengths.",
        "availability": "always",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "unit": None,
        "better_values_are": "context-dependent",
    },
    "best_rank": {
        "description": "Lowest EL %rank over all pairs; the single strongest predicted binder.",
        "availability": "always",
        "type": "float",
        "min": 0.0,
        "max": 100.0,
        "unit": "percent",
        "better_values_are": "context-dependent",
    },
    "binding_propensity": {
        "description": "Mean smooth 0-1 propensity over pairs; a continuous optimization target.",
        "availability": "always",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "unit": None,
        "better_values_are": "context-dependent",
    },
}


class NetMHCpanMetrics(Metrics):
    """Summary binding metrics for one peptide or one protein.

    Counts and the smooth propensity are reported together on purpose: counts are what
    immunology reads, while ``binding_propensity`` is differentiable-in-practice and so is
    the sensible objective for a sampler.

    Attributes:
        primary_metric (str | None): Headline metric used to rank results.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = _SHARED_METRIC_SPEC
    primary_metric: str | None = Field(
        default="binding_propensity",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class NetMHCpanProteinMetrics(NetMHCpanMetrics):
    """Summary binding metrics for one protein, including epitope density.

    Attributes:
        primary_metric (str | None): Headline metric used to rank results.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        **_SHARED_METRIC_SPEC,
        "strong_binders_per_100_residues": {
            "description": "Strong-binder density per 100 residues per allele.",
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
            "unit": None,
            "better_values_are": "context-dependent",
        },
    }


def classify(rank: float, strong_rank: float, weak_rank: float) -> BinderClass:
    """Classify a peptide-allele pair from its EL %rank and the configured thresholds.

    The binary's own ``BindLevel`` column is deliberately not trusted here: it is derived
    from the eluted-ligand %rank even when ``-BA`` is requested, which makes it ambiguous
    about which quantity produced it. Classifying explicitly from ``rank_el`` keeps the
    reported class and the reported rank consistent by construction.

    Args:
        rank (float): Eluted-ligand %rank.
        strong_rank (float): Strong-binder threshold.
        weak_rank (float): Weak-binder threshold.

    Returns:
        BinderClass: ``"strong"``, ``"weak"``, or ``"non-binder"``.
    """
    if rank <= strong_rank:
        return "strong"
    if rank <= weak_rank:
        return "weak"
    return "non-binder"


def _optional_float(record: dict[str, str], column: str) -> float | None:
    """Return a numeric column as a float, or None when the column is absent."""
    value = record.get(column)
    return None if value is None else float(value)


def binding_from_record(record: dict[str, str], strong_rank: float, weak_rank: float) -> NetMHCpanBinding:
    """Build a binding record from one parsed netMHCpan output row.

    The allele is restated in netMHCpan's accepted spelling rather than the starred form
    the output table prints, so it round-trips back into a later call unchanged.

    Args:
        record (dict[str, str]): One parsed output row, keyed by the table's column names.
        strong_rank (float): Strong-binder %rank threshold.
        weak_rank (float): Weak-binder %rank threshold.

    Returns:
        NetMHCpanBinding: The typed prediction.
    """
    rank_el = float(record["%Rank_EL"])
    return NetMHCpanBinding(
        allele=normalize_allele(record["MHC"]),
        peptide=record["Peptide"],
        position=int(record["Pos"]),
        core=record["Core"],
        icore=record["Icore"],
        score_el=float(record["Score_EL"]),
        rank_el=rank_el,
        binder_class=classify(rank_el, strong_rank, weak_rank),
        score_ba=_optional_float(record, "Score_BA"),
        rank_ba=_optional_float(record, "%Rank_BA"),
        affinity_nm=_optional_float(record, "Aff(nM)"),
    )


def summarize(bindings: list[NetMHCpanBinding], weak_rank: float) -> dict[str, float | int]:
    """Build the shared summary metric values for one item's predictions.

    Args:
        bindings (list[NetMHCpanBinding]): Every peptide-allele prediction for one item.
        weak_rank (float): Weak-binder threshold the propensity curve is centred on.

    Returns:
        dict[str, float | int]: Values for the shared metric spec.
    """
    total = len(bindings)
    strong = sum(1 for binding in bindings if binding.binder_class == "strong")
    weak = sum(1 for binding in bindings if binding.binder_class == "weak")
    propensities = [binding_propensity(binding.rank_el, weak_rank) for binding in bindings]
    return {
        "strong_binder_count": strong,
        "weak_binder_count": weak,
        "strong_binder_fraction": strong / total if total else 0.0,
        "best_rank": min((binding.rank_el for binding in bindings), default=100.0),
        "binding_propensity": math.fsum(propensities) / total if total else 0.0,
    }


class NetMHCpanConfig(BaseConfig):
    """Settings shared by every netMHCpan tool.

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

    alleles: list[str] = ConfigField(
        default=["HLA-A02:01"],
        title="Alleles",
        description="MHC class I alleles to predict against, e.g. ['HLA-A02:01', 'HLA-B07:02'].",
    )
    include_binding_affinity: bool = ConfigField(
        default=False,
        title="Include Binding Affinity",
        description="Also predict affinity (-BA): adds BA score, BA %rank and nM affinity.",
    )
    strong_rank: float = ConfigField(
        default=0.5,
        gt=0.0,
        le=100.0,
        title="Strong Binder %Rank",
        description="EL %rank at or below which a peptide counts as a strong binder.",
    )
    weak_rank: float = ConfigField(
        default=2.0,
        gt=0.0,
        le=100.0,
        title="Weak Binder %Rank",
        description="EL %rank at or below which a peptide counts as a weak binder.",
    )
    netmhcpan_dir: str | None = ConfigField(
        default=None,
        title="netMHCpan Directory",
        description="Local netMHCpan install root; overrides $NETMHCPAN_HOME and the tool cache.",
    )

    @model_validator(mode="after")
    def _validate_thresholds(self) -> NetMHCpanConfig:
        """Require the strong threshold to be at least as strict as the weak one."""
        if self.strong_rank > self.weak_rank:
            raise ValueError(
                f"strong_rank ({self.strong_rank}) must be <= weak_rank ({self.weak_rank}); "
                "a strong binder is a stricter %rank cutoff than a weak one"
            )
        return self

    @model_validator(mode="after")
    def _normalize_alleles(self) -> NetMHCpanConfig:
        """Normalize the allele panel to the spelling netMHCpan accepts."""
        normalized = normalize_alleles(self.alleles)
        if normalized != self.alleles:
            # Assignment re-triggers validation, so set via __dict__ to avoid recursion.
            self.__dict__["alleles"] = normalized
        return self

    @property
    def cpus_per_instance(self) -> int | None:
        """Opt in to ToolPool CPU fan-out -- netMHCpan is a single-threaded binary."""
        return 1

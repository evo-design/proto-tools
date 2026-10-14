"""Reference genome selection shared by the genome-coordinate splicing tools."""

from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator

from proto_tools.databases.assets import dataset_file, is_registered_dataset
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.tool_io import MissingAssetError

DEFAULT_ASSEMBLY = "grch38"

# Assemblies with a registered dataset, so naming one is all a caller needs to do.
ProvisionedGenome = Literal["grch37", "grch38"]

# The FASTA within each registered genome dataset.
GENOME_FASTA: dict[str, str] = {
    "grch37": "Homo_sapiens.GRCh37.dna.primary_assembly.fa",
    "grch38": "Homo_sapiens.GRCh38.dna.primary_assembly.fa",
}

# UCSC spellings of the same assemblies. Callers think in these; the registered datasets do not.
_GENOME_ALIASES: dict[str, str] = {"hg19": "grch37", "hg38": "grch38"}


class ReferenceGenomeConfig(BaseConfig):
    """Config mixin for tools that read variant context from a reference genome FASTA.

    Attributes:
        reference_fasta (ProvisionedGenome | str | None): The reference genome
            the wild-type sequence around each variant is read from. Either a
            provisioned assembly name (``'grch38'``/``'grch37'``, or their UCSC
            spellings ``'hg38'``/``'hg19'``), downloaded on first use and the
            only form a remote worker can resolve, or a path to a FASTA on this
            machine. Required at call time; ``None`` raises a
            ``MissingAssetError`` so un-provisioned hosts skip cleanly.
    """

    reference_fasta: ProvisionedGenome | str | None = ConfigField(
        title="Reference FASTA",
        default=None,
        description="Assembly name ('grch38'/'grch37', provisioned on demand) or a local FASTA path",
        reload_on_change=True,
    )

    @field_validator("reference_fasta", mode="before")
    @classmethod
    def _normalize_assembly_alias(cls, value: Any) -> Any:
        """Accept the UCSC spelling of a provisioned assembly, e.g. ``hg38`` for ``grch38``."""
        if isinstance(value, str):
            return _GENOME_ALIASES.get(value.strip().lower(), value)
        return value

    @field_validator("reference_fasta")
    @classmethod
    def _validate_genome_or_path(cls, value: str | None) -> str | None:
        """Require a value that is not a provisioned assembly to be a FASTA on this machine.

        Catches a typo where it is made rather than several minutes later, and keeps the two forms
        of the field distinguishable: anything not registered is a path, so it is local-only.

        Safe to check the filesystem here only because :meth:`remote_unsupported_reason` refuses a
        path on a remote device. Were that guard dropped, a container rebuilding this config from
        its transport dict would fail here on a path that never existed on its side.
        """
        if value is None or is_registered_dataset(value):
            return value
        if not Path(value).expanduser().exists():
            raise ValueError(
                f"reference_fasta: {value!r} is neither a provisioned assembly "
                f"({', '.join(sorted(GENOME_FASTA))}) nor an existing file on this machine"
            )
        return value

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """A genome given as a path lives on the caller's machine, so only a named assembly travels."""
        if self.reference_fasta is not None and not is_registered_dataset(self.reference_fasta):
            return (
                f"reference_fasta={self.reference_fasta!r} is a local path, which can't be staged to "
                f"device='{device}'. Name a provisioned assembly ({', '.join(sorted(GENOME_FASTA))}) "
                f"instead, or run it on this machine."
            )
        return super().remote_unsupported_reason(device)

    @classmethod
    def minimal(cls, **kwargs: Any) -> "BaseConfig":
        """Cheap-mode defaults: name the default assembly, since the tool cannot run without one.

        Naming it costs nothing where it is already staged, and provisions it where it is not,
        which is what lets parametrized infrastructure run this tool at all rather than reporting
        it as an unprovisioned asset it has no way to satisfy.
        """
        kwargs.setdefault("reference_fasta", DEFAULT_ASSEMBLY)
        return super().minimal(**kwargs)

    def resolved_reference_fasta(self) -> Path | None:
        """Return the FASTA to read, provisioning the assembly on first use.

        Returns:
            Path | None: The resolved FASTA, or ``None`` when nothing was configured. The path may
                not exist when provisioning was unavailable; the caller reports that as a
                ``MissingAssetError`` rather than treating it as a failure.
        """
        if self.reference_fasta is None:
            return None
        if is_registered_dataset(self.reference_fasta):
            return dataset_file(self.reference_fasta, GENOME_FASTA[self.reference_fasta])
        return Path(self.reference_fasta).expanduser()

    def require_reference_fasta(self, toolkit: str) -> Path:
        """Resolve the FASTA, raising when it is unset or could not be provisioned.

        Args:
            toolkit (str): Toolkit name reported in the error.

        Returns:
            Path: An existing FASTA file.

        Raises:
            MissingAssetError: If ``reference_fasta`` is None or missing (the test layer converts
                this to a skip).
        """
        resolved = self.resolved_reference_fasta()
        if resolved is None or not resolved.exists():
            raise MissingAssetError(
                toolkit,
                "reference",
                f"reference_fasta not provided or not found: {self.reference_fasta!r}. "
                f"A reference genome FASTA is required: name a provisioned assembly "
                f"({', '.join(sorted(GENOME_FASTA))}) or set reference_fasta to a local path.",
            )
        return resolved

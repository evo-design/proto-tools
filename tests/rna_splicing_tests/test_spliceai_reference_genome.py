"""``SpliceAIScoreConfig.reference_fasta`` accepting either a provisioned assembly or a local path.

The two forms differ in more than spelling: a named assembly resolves on any machine, a path only
on the one that wrote it. These cover the rules that keep them distinguishable.
"""

from __future__ import annotations

import pytest

from proto_tools.tools.rna_splicing.spliceai.spliceai_score import (
    _GENOME_FASTA,
    SpliceAIScoreConfig,
)

ASSEMBLIES = sorted(_GENOME_FASTA)


@pytest.mark.parametrize("assembly", ASSEMBLIES)
def test_named_assembly_is_registered(assembly: str) -> None:
    """Every assembly the field offers must resolve, or naming it is a dead end."""
    from proto_tools.databases.assets import is_registered_dataset

    assert is_registered_dataset(assembly)


@pytest.mark.parametrize("assembly", ASSEMBLIES)
def test_annotation_defaults_to_the_named_assembly(assembly: str) -> None:
    """Coordinates mean a different place per build, so the two must not disagree by default."""
    assert SpliceAIScoreConfig(reference_fasta=assembly).annotation == assembly


def test_explicit_annotation_beats_the_derived_one() -> None:
    """Deriving is a default, not a constraint; a caller pairing them deliberately still can."""
    config = SpliceAIScoreConfig(reference_fasta="grch38", annotation="grch37")
    assert config.annotation == "grch37"


@pytest.mark.parametrize(("alias", "canonical"), [("hg19", "grch37"), ("hg38", "grch38")])
def test_ucsc_alias_normalizes(alias: str, canonical: str) -> None:
    """Callers think in UCSC names; the bundled annotations are GENCODE-named."""
    config = SpliceAIScoreConfig(reference_fasta=alias)
    assert config.reference_fasta == canonical
    assert config.annotation == canonical


def test_unknown_value_that_is_not_a_file_is_rejected() -> None:
    """A typo'd assembly would otherwise read as a path and fail minutes later at dispatch."""
    with pytest.raises(ValueError, match="neither a provisioned assembly"):
        SpliceAIScoreConfig(reference_fasta="grch39")


def test_existing_local_path_is_accepted(tmp_path) -> None:
    """The escape hatch: any FASTA on this machine, including assemblies we do not register."""
    fasta = tmp_path / "custom.fa"
    fasta.write_text(">1\nACGT\n")
    config = SpliceAIScoreConfig(reference_fasta=str(fasta))
    assert config.reference_fasta == str(fasta)
    # A custom genome says nothing about which annotation matches it, so the default stands.
    assert config.annotation == "grch38"


@pytest.mark.parametrize("device", ["modal", "proto"])
def test_local_path_is_refused_remotely(tmp_path, device: str) -> None:
    """The path names a file on the caller's machine; a container would read something else or fail."""
    fasta = tmp_path / "custom.fa"
    fasta.write_text(">1\nACGT\n")
    reason = SpliceAIScoreConfig(reference_fasta=str(fasta)).remote_unsupported_reason(device)
    assert reason is not None
    assert "local path" in reason


def test_unset_reference_is_not_refused_remotely() -> None:
    """Absent is not wrong yet — dispatch reports it as a missing asset, which a smoke run skips."""
    assert SpliceAIScoreConfig().remote_unsupported_reason("modal") is None


def test_resolved_path_is_the_file_not_the_name(tmp_path) -> None:
    """The standalone caches its Annotator on this value, so a name and its path must not both appear."""
    fasta = tmp_path / "custom.fa"
    fasta.write_text(">1\nACGT\n")
    assert SpliceAIScoreConfig(reference_fasta=str(fasta)).resolved_reference_fasta() == fasta
    assert SpliceAIScoreConfig().resolved_reference_fasta() is None


def test_schema_still_advertises_the_assemblies() -> None:
    """The permissive type must not cost callers the discoverable list of common values."""
    prop = SpliceAIScoreConfig.model_json_schema()["properties"]["reference_fasta"]
    enums = [branch["enum"] for branch in prop["anyOf"] if "enum" in branch]
    assert enums == [ASSEMBLIES]

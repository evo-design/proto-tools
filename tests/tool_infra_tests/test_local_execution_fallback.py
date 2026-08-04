"""Configs that redirect a remote call to this machine instead of refusing it."""

from __future__ import annotations

import pytest

from proto_tools.tools.sequence_alignment.blast.blast_search import BlastSearchConfig
from proto_tools.tools.structure_alignment.foldseek.foldseek_multimer_search import FoldseekMultimerSearchConfig
from proto_tools.tools.structure_alignment.foldseek.foldseek_search import FoldseekSearchConfig
from proto_tools.utils import BaseConfig

_HTTP_MODE_CONFIGS = [
    pytest.param(BlastSearchConfig, "online", id="blast"),
    pytest.param(FoldseekSearchConfig, "remote", id="foldseek"),
    pytest.param(FoldseekMultimerSearchConfig, "remote", id="foldseek-multimer"),
]


def test_base_config_dispatches_remotely_by_default() -> None:
    """The hook is opt-in: a config that says nothing keeps its remote device."""
    assert BaseConfig().local_execution_reason("modal") is None


@pytest.mark.parametrize("config_class,http_mode", _HTTP_MODE_CONFIGS)
def test_http_search_mode_runs_here(config_class: type[BaseConfig], http_mode: str) -> None:
    """A search whose implementation is an HTTP call gains nothing from a container."""
    config = config_class(search_mode=http_mode)
    reason = config.local_execution_reason("modal")
    assert reason is not None
    assert http_mode in reason


@pytest.mark.parametrize("config_class,http_mode", _HTTP_MODE_CONFIGS)
def test_http_search_mode_is_redirected_not_refused(config_class: type[BaseConfig], http_mode: str) -> None:
    """The two hooks must not both fire: one raises, the other reroutes."""
    config = config_class(search_mode=http_mode)
    assert config.remote_unsupported_reason("modal") is None


@pytest.mark.parametrize("config_class,http_mode", _HTTP_MODE_CONFIGS)
def test_local_search_mode_still_dispatches(config_class: type[BaseConfig], http_mode: str, tmp_path) -> None:
    """Local-DB search is refused, not redirected, so the caller learns the database is unreachable."""
    config = config_class(search_mode="local", local_db=str(tmp_path / "db"))
    assert config.local_execution_reason("modal") is None
    assert config.remote_unsupported_reason("modal") is not None


def test_spliceai_path_genome_is_refused_rather_than_redirected(tmp_path) -> None:
    """A genome given as a path is a caller mistake on a remote device, not a pointless hop."""
    from proto_tools.tools.rna_splicing.spliceai.spliceai_score import SpliceAIScoreConfig

    genome = tmp_path / "hg38.fa"
    genome.write_text(">chr1\nACGT\n")
    config = SpliceAIScoreConfig(reference_fasta=str(genome))
    assert config.remote_unsupported_reason("modal") is not None
    assert config.local_execution_reason("modal") is None


def test_spliceai_named_assembly_dispatches_remotely() -> None:
    """A named assembly is staged in the container, so the hosted path is the real one."""
    from proto_tools.tools.rna_splicing.spliceai.spliceai_score import SpliceAIScoreConfig

    config = SpliceAIScoreConfig(reference_fasta="grch38")
    assert config.remote_unsupported_reason("modal") is None
    assert config.local_execution_reason("modal") is None


def test_mmseqs2_remote_search_is_redirected_despite_local_only() -> None:
    """A local_only tool still answers a remote device when the call needs no worker.

    ``mmseqs2-homology-search`` can never be deployed — its local mode searches a staged corpus —
    but its remote mode is an HTTP call to the ColabFold API. Asking the config first is what lets
    those two facts coexist.
    """
    from proto_tools.tools import ToolRegistry
    from proto_tools.tools.sequence_alignment import Mmseqs2HomologySearchConfig

    assert ToolRegistry.get("mmseqs2-homology-search").local_only is not None
    assert Mmseqs2HomologySearchConfig(search_mode="remote").local_execution_reason("modal") is not None


def test_mmseqs2_local_search_is_still_refused() -> None:
    """Local-corpus search keeps the local_only refusal, which is what the declaration is for."""
    from proto_tools.tools.sequence_alignment import Mmseqs2HomologySearchConfig

    assert Mmseqs2HomologySearchConfig(search_mode="local").local_execution_reason("modal") is None


def test_nested_msa_config_defaults_stay_local() -> None:
    """A structure predictor builds this config internally; its defaults must not reach for a worker."""
    from proto_tools.tools.sequence_alignment import Mmseqs2HomologySearchConfig

    config = Mmseqs2HomologySearchConfig()
    assert config.search_mode == "remote"
    assert config.device == "cpu"

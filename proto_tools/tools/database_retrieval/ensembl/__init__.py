"""Ensembl REST API wrappers (gene/transcript/sequence/overlap/xrefs + VEP)."""

from proto_tools.tools.database_retrieval.ensembl.ensembl_fetch import (
    EnsemblEndpoint,
    EnsemblFetchConfig,
    EnsemblFetchInput,
    EnsemblFetchOutput,
    EnsemblOverlapFeature,
    EnsemblSequenceType,
    run_ensembl_fetch,
)
from proto_tools.tools.database_retrieval.ensembl.ensembl_vep import (
    EnsemblVEPConfig,
    EnsemblVEPConsequence,
    EnsemblVEPInput,
    EnsemblVEPOutput,
    run_ensembl_vep,
)
from proto_tools.tools.database_retrieval.ensembl.shared_data_models import (
    EnsemblAssembly,
    EnsemblExon,
    EnsemblGene,
    EnsemblOverlapFeatureRecord,
    EnsemblSequence,
    EnsemblSpecies,
    EnsemblTranscript,
    EnsemblTranslation,
    EnsemblXref,
)

__all__ = [
    "EnsemblAssembly",
    "EnsemblEndpoint",
    "EnsemblExon",
    "EnsemblFetchConfig",
    "EnsemblFetchInput",
    "EnsemblFetchOutput",
    "EnsemblGene",
    "EnsemblOverlapFeature",
    "EnsemblOverlapFeatureRecord",
    "EnsemblSequence",
    "EnsemblSequenceType",
    "EnsemblSpecies",
    "EnsemblTranscript",
    "EnsemblTranslation",
    "EnsemblVEPConfig",
    "EnsemblVEPConsequence",
    "EnsemblVEPInput",
    "EnsemblVEPOutput",
    "EnsemblXref",
    "run_ensembl_fetch",
    "run_ensembl_vep",
]

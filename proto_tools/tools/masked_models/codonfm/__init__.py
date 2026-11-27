"""CodonFM (Encodon) toolkit: codon-level masked-language-model scoring of coding sequences.

Wraps the NVIDIA-BioNeMo/CodonFM Encodon models (Apache-2.0 code, NVIDIA Open Model License
weights) as five proto tools: mutation-effect scoring, upstream fitness, CLS embeddings,
masked pseudo-log-likelihood gradients, and masked-codon sampling.
"""

from proto_tools.tools.masked_models.codonfm.codonfm_embeddings import (
    CodonFMEmbeddingResult,
    CodonFMEmbeddingsConfig,
    CodonFMEmbeddingsInput,
    CodonFMEmbeddingsOutput,
    run_codonfm_embeddings,
)
from proto_tools.tools.masked_models.codonfm.codonfm_fitness import (
    CodonFMFitnessConfig,
    CodonFMFitnessInput,
    CodonFMFitnessOutput,
    CodonFMFitnessResult,
    run_codonfm_fitness,
)
from proto_tools.tools.masked_models.codonfm.codonfm_gradient import (
    CodonFMGradientConfig,
    CodonFMGradientInput,
    CodonFMGradientOutput,
    run_codonfm_gradient,
)
from proto_tools.tools.masked_models.codonfm.codonfm_sample import (
    CodonFMSampleConfig,
    CodonFMSampleInput,
    CodonFMSampleOutput,
    CodonFMSampleResult,
    run_codonfm_sample,
)
from proto_tools.tools.masked_models.codonfm.codonfm_score import (
    CodonFMMutation,
    CodonFMMutationResult,
    CodonFMScoreConfig,
    CodonFMScoreInput,
    CodonFMScoreOutput,
    run_codonfm_score,
)
from proto_tools.tools.masked_models.codonfm.shared_data_models import (
    CODONFM_CHECKPOINTS,
    CODONFM_CODON_VOCAB,
    CODONFM_NUM_CODONS,
    CodonFMCheckpoint,
    CodonFMConfig,
    CodonSequenceInput,
    normalize_codon_sequence,
    one_hot_codon_logits,
    resolve_checkpoint_source,
)

__all__ = [
    "CODONFM_CHECKPOINTS",
    "CODONFM_CODON_VOCAB",
    "CODONFM_NUM_CODONS",
    "CodonFMCheckpoint",
    "CodonFMConfig",
    "CodonFMEmbeddingResult",
    "CodonFMEmbeddingsConfig",
    "CodonFMEmbeddingsInput",
    "CodonFMEmbeddingsOutput",
    "CodonFMFitnessConfig",
    "CodonFMFitnessInput",
    "CodonFMFitnessOutput",
    "CodonFMFitnessResult",
    "CodonFMGradientConfig",
    "CodonFMGradientInput",
    "CodonFMGradientOutput",
    "CodonFMMutation",
    "CodonFMMutationResult",
    "CodonFMSampleConfig",
    "CodonFMSampleInput",
    "CodonFMSampleOutput",
    "CodonFMSampleResult",
    "CodonFMScoreConfig",
    "CodonFMScoreInput",
    "CodonFMScoreOutput",
    "CodonSequenceInput",
    "normalize_codon_sequence",
    "one_hot_codon_logits",
    "resolve_checkpoint_source",
    "run_codonfm_embeddings",
    "run_codonfm_fitness",
    "run_codonfm_gradient",
    "run_codonfm_sample",
    "run_codonfm_score",
]

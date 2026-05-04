"""FoldMason toolkit: multiple structure alignment + MSA-LDDT scoring."""

from proto_tools.tools.structure_alignment.foldmason.foldmason_msa import (
    FoldmasonMSAConfig,
    FoldmasonMSAInput,
    FoldmasonMSAOutput,
    run_foldmason_msa,
)
from proto_tools.tools.structure_alignment.foldmason.foldmason_score_msa import (
    FoldmasonScoreMSAConfig,
    FoldmasonScoreMSAInput,
    FoldmasonScoreMSAOutput,
    run_foldmason_score_msa,
)

__all__ = [
    "FoldmasonMSAConfig",
    "FoldmasonMSAInput",
    "FoldmasonMSAOutput",
    "FoldmasonScoreMSAConfig",
    "FoldmasonScoreMSAInput",
    "FoldmasonScoreMSAOutput",
    "run_foldmason_msa",
    "run_foldmason_score_msa",
]

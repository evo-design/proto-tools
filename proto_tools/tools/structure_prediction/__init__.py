"""Structure prediction tools."""

from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2Config,
    AlphaFold2Input,
    AlphaFold2Output,
    run_alphafold2,
)
from proto_tools.tools.structure_prediction.alphafold3 import (
    AlphaFold3Config,
    AlphaFold3Input,
    AlphaFold3Output,
    run_alphafold3,
)
from proto_tools.tools.structure_prediction.boltz2 import Boltz2Config, Boltz2Input, Boltz2Output, run_boltz2
from proto_tools.tools.structure_prediction.chai1 import Chai1Config, Chai1Input, Chai1Output, run_chai1
from proto_tools.tools.structure_prediction.dispatch import predict_structures
from proto_tools.tools.structure_prediction.esmfold import ESMFoldConfig, ESMFoldInput, ESMFoldOutput, run_esmfold
from proto_tools.tools.structure_prediction.protenix import ProtenixConfig, ProtenixInput, ProtenixOutput, run_protenix
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    ChainModification,
    MSAStructurePredictionConfig,
    StructurePredictionComplex,
    StructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
)
from proto_tools.tools.structure_prediction.viennarna import (
    ViennaRNAConfig,
    ViennaRNAInput,
    ViennaRNAOutput,
    run_viennarna,
)

__all__ = [
    # AlphaFold2
    "run_alphafold2",
    "AlphaFold2Input",
    "AlphaFold2Config",
    "AlphaFold2Output",
    # AlphaFold3
    "run_alphafold3",
    "AlphaFold3Input",
    "AlphaFold3Config",
    "AlphaFold3Output",
    # Boltz2
    "run_boltz2",
    "Boltz2Input",
    "Boltz2Config",
    "Boltz2Output",
    # Chai1
    "run_chai1",
    "Chai1Input",
    "Chai1Config",
    "Chai1Output",
    # ESMFold
    "run_esmfold",
    "ESMFoldInput",
    "ESMFoldConfig",
    "ESMFoldOutput",
    # Protenix
    "run_protenix",
    "ProtenixInput",
    "ProtenixConfig",
    "ProtenixOutput",
    # ViennaRNA
    "run_viennarna",
    "ViennaRNAInput",
    "ViennaRNAConfig",
    "ViennaRNAOutput",
    # Shared Data Models
    "Chain",
    "ChainModification",
    "MSAStructurePredictionConfig",
    "StructurePredictionComplex",
    "StructurePredictionConfig",
    "StructurePredictionInput",
    "StructurePredictionOutput",
    # Dispatch
    "predict_structures",
]

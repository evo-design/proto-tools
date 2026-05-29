"""Structure prediction tools."""

from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2BinderConfig,
    AlphaFold2BinderInput,
    AlphaFold2BinderOutput,
    AlphaFold2Config,
    AlphaFold2Input,
    AlphaFold2Output,
    run_alphafold2,
    run_alphafold2_binder,
)
from proto_tools.tools.structure_prediction.alphafold3 import (
    AlphaFold3Config,
    AlphaFold3Input,
    AlphaFold3Output,
    run_alphafold3,
)
from proto_tools.tools.structure_prediction.boltz2 import (
    Boltz2AffinityConfig,
    Boltz2AffinityInput,
    Boltz2AffinityMetrics,
    Boltz2AffinityOutput,
    Boltz2Config,
    Boltz2Input,
    Boltz2Output,
    run_boltz2,
    run_boltz2_affinity,
)
from proto_tools.tools.structure_prediction.chai1 import Chai1Config, Chai1Input, Chai1Output, run_chai1
from proto_tools.tools.structure_prediction.dispatch import predict_structures
from proto_tools.tools.structure_prediction.esmfold import (
    ESMFoldConfig,
    ESMFoldGradientConfig,
    ESMFoldGradientInput,
    ESMFoldGradientOutput,
    ESMFoldInput,
    ESMFoldOutput,
    run_esmfold,
    run_esmfold_gradient,
)
from proto_tools.tools.structure_prediction.esmfold2 import (
    ESMFold2Config,
    ESMFold2Input,
    ESMFold2Metrics,
    ESMFold2Output,
    run_esmfold2,
)
from proto_tools.tools.structure_prediction.protenix import ProtenixConfig, ProtenixInput, ProtenixOutput, run_protenix
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    ChainModification,
    Complex,
    ComplexMSAs,
    MSAStructurePredictionConfig,
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
    "run_alphafold2_binder",
    "AlphaFold2BinderInput",
    "AlphaFold2BinderConfig",
    "AlphaFold2BinderOutput",
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
    "run_boltz2_affinity",
    "Boltz2AffinityInput",
    "Boltz2AffinityConfig",
    "Boltz2AffinityOutput",
    "Boltz2AffinityMetrics",
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
    "run_esmfold_gradient",
    "ESMFoldGradientInput",
    "ESMFoldGradientConfig",
    "ESMFoldGradientOutput",
    # ESMFold2
    "run_esmfold2",
    "ESMFold2Input",
    "ESMFold2Config",
    "ESMFold2Output",
    "ESMFold2Metrics",
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
    "Complex",
    "ComplexMSAs",
    "StructurePredictionConfig",
    "StructurePredictionInput",
    "StructurePredictionOutput",
    # Dispatch
    "predict_structures",
]

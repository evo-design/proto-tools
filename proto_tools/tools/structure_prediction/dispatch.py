"""proto_tools/tools/structure_prediction/dispatch.py.

Router function for structure prediction tools.
"""

from __future__ import annotations

from typing import Any

from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2Config,
    AlphaFold2Input,
    run_alphafold2,
)
from proto_tools.tools.structure_prediction.alphafold3 import (
    AlphaFold3Config,
    AlphaFold3Input,
    run_alphafold3,
)
from proto_tools.tools.structure_prediction.boltz2 import (
    Boltz2Config,
    Boltz2Input,
    run_boltz2,
)
from proto_tools.tools.structure_prediction.chai1 import (
    Chai1Config,
    Chai1Input,
    run_chai1,
)
from proto_tools.tools.structure_prediction.esmfold import (
    ESMFoldConfig,
    ESMFoldInput,
    run_esmfold,
)
from proto_tools.tools.structure_prediction.protenix import (
    ProtenixConfig,
    ProtenixInput,
    run_protenix,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    StructurePredictionOutput,
)

SP_TOOL_MAP = {
    "esmfold": {"config": ESMFoldConfig, "input": ESMFoldInput, "run_func": run_esmfold},
    "alphafold2": {"config": AlphaFold2Config, "input": AlphaFold2Input, "run_func": run_alphafold2},
    "alphafold3": {"config": AlphaFold3Config, "input": AlphaFold3Input, "run_func": run_alphafold3},
    "boltz2": {"config": Boltz2Config, "input": Boltz2Input, "run_func": run_boltz2},
    "chai1": {"config": Chai1Config, "input": Chai1Input, "run_func": run_chai1},
    "protenix": {"config": ProtenixConfig, "input": ProtenixInput, "run_func": run_protenix},
}


def predict_structures(
    complexes: StructurePredictionComplex | list[StructurePredictionComplex],
    tool_name: str,
    tool_config: ESMFoldConfig
    | AlphaFold2Config
    | AlphaFold3Config
    | Boltz2Config
    | Chai1Config
    | ProtenixConfig
    | dict[str, Any]
    | None = None,
) -> StructurePredictionOutput:
    """Dispatch structure prediction to the specified tool.

    Dynamically imports tools to avoid circular dependencies. Used by
    structure-related constraints and optimizers.

    Args:
        complexes (StructurePredictionComplex | list[StructurePredictionComplex]): List of StructurePredictionComplex objects to predict.
        tool_name (str): Name of the structure prediction tool. Supported values:
            ``"esmfold"``, ``"alphafold2"``, ``"alphafold3"``, ``"boltz2"``, ``"chai1"``, ``"protenix"``.
        tool_config (ESMFoldConfig | AlphaFold2Config | AlphaFold3Config | Boltz2Config | Chai1Config | ProtenixConfig | dict[str, Any] | None): Tool-specific configuration dictionary.

    Returns:
        StructurePredictionOutput: StructurePredictionOutput containing predicted structures and metrics.

    Raises:
        ValueError: If tool_name is not recognized.
    """
    # If complexes is a single StructurePredictionComplex, normalize it to a list
    if isinstance(complexes, StructurePredictionComplex):
        complexes = [complexes]

    if tool_name not in SP_TOOL_MAP:
        raise ValueError(
            f"Unknown structure prediction tool: '{tool_name}'. Supported tools: {', '.join(SP_TOOL_MAP.keys())}"
        )

    # Collect the expected config class for the tool
    expected_config_class = SP_TOOL_MAP[tool_name]["config"]

    # If the tool_config is a dictionary, convert it to the expected config class
    if isinstance(tool_config, dict):
        tool_config = expected_config_class(**tool_config)  # type: ignore[operator]
    elif tool_config is None:
        tool_config = expected_config_class()  # type: ignore[operator]

    # Ensure that the tool_config is the expected config class
    if not isinstance(tool_config, expected_config_class):  # type: ignore[arg-type]
        raise ValueError(
            f"tool_config type {type(tool_config).__name__} doesn't match "
            f"tool_name '{tool_name}' (expected {expected_config_class.__name__})"  # type: ignore[attr-defined]
        )

    # Get the run function for the tool
    run_func = SP_TOOL_MAP[tool_name]["run_func"]

    # Wrap the config input in the expected input class
    inputs = SP_TOOL_MAP[tool_name]["input"](complexes=complexes)  # type: ignore[operator]

    # Run the prediction
    return run_func(inputs, tool_config)  # type: ignore[no-any-return, operator]

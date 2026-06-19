"""proto_tools/tools/structure_prediction/dispatch.py.

Router function for structure prediction tools.
"""

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
from proto_tools.tools.structure_prediction.esmfold2 import (
    ESMFold2Config,
    ESMFold2Input,
    run_esmfold2,
)
from proto_tools.tools.structure_prediction.protenix import (
    ProtenixConfig,
    ProtenixInput,
    run_protenix,
)
from proto_tools.tools.structure_prediction.rf3 import (
    RF3Config,
    RF3Input,
    run_rf3_prediction,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    Complex,
    ComplexMSAs,
    StructurePredictionOutput,
)

SP_TOOL_MAP = {
    "esmfold": {"config": ESMFoldConfig, "input": ESMFoldInput, "run_func": run_esmfold},
    "esmfold2": {"config": ESMFold2Config, "input": ESMFold2Input, "run_func": run_esmfold2},
    "alphafold2": {"config": AlphaFold2Config, "input": AlphaFold2Input, "run_func": run_alphafold2},
    "alphafold3": {"config": AlphaFold3Config, "input": AlphaFold3Input, "run_func": run_alphafold3},
    "boltz2": {"config": Boltz2Config, "input": Boltz2Input, "run_func": run_boltz2},
    "chai1": {"config": Chai1Config, "input": Chai1Input, "run_func": run_chai1},
    "protenix": {"config": ProtenixConfig, "input": ProtenixInput, "run_func": run_protenix},
    "rf3": {"config": RF3Config, "input": RF3Input, "run_func": run_rf3_prediction},
}


def predict_structures(
    complexes: Complex | list[Complex],
    toolkit: str,
    tool_config: ESMFoldConfig
    | ESMFold2Config
    | AlphaFold2Config
    | AlphaFold3Config
    | Boltz2Config
    | Chai1Config
    | ProtenixConfig
    | RF3Config
    | dict[str, Any]
    | None = None,
    msas: ComplexMSAs | list[ComplexMSAs] | None = None,
) -> StructurePredictionOutput:
    """Dispatch structure prediction to the specified tool.

    Maps ``toolkit`` to its Input/Config/run-function and invokes it. Used by
    structure-related constraints and optimizers.

    Args:
        complexes (Complex | list[Complex]): List of Complex objects to predict.
        toolkit (str): Name of the structure prediction tool. Supported values:
            ``"esmfold"``, ``"esmfold2"``, ``"alphafold2"``, ``"alphafold3"``, ``"boltz2"``, ``"chai1"``, ``"protenix"``, ``"rf3"``.
        tool_config (ESMFoldConfig | ESMFold2Config | AlphaFold2Config | AlphaFold3Config | Boltz2Config | Chai1Config | ProtenixConfig | RF3Config | dict[str, Any] | None): Tool-specific config object, a dict coerced to it, or None for defaults.
        msas (ComplexMSAs | list[ComplexMSAs] | None): Pre-computed MSAs, one per complex (a single ``ComplexMSAs`` is normalized to a one-element list, mirroring ``complexes``); chains omitted from a complex's ``per_chain`` map stay single-sequence. When supplied, the tool consumes them directly and skips MMseqs2 homology search.

    Returns:
        StructurePredictionOutput: StructurePredictionOutput containing predicted structures and metrics.

    Raises:
        ValueError: If toolkit is not recognized, or tool_config is an instance of the wrong config class.
    """
    # If complexes is a single Complex, normalize it to a list
    if isinstance(complexes, Complex):
        complexes = [complexes]

    # Mirror the above for a single ComplexMSAs so the convenience form stays symmetric.
    if isinstance(msas, ComplexMSAs):
        msas = [msas]

    if toolkit not in SP_TOOL_MAP:
        raise ValueError(f"predict_structures: unknown toolkit {toolkit!r}; supported: {sorted(SP_TOOL_MAP)}")

    # Collect the expected config class for the tool
    expected_config_class = SP_TOOL_MAP[toolkit]["config"]

    # If the tool_config is a dictionary, convert it to the expected config class
    if isinstance(tool_config, dict):
        tool_config = expected_config_class(**tool_config)  # type: ignore[operator]
    elif tool_config is None:
        tool_config = expected_config_class()  # type: ignore[operator]

    # Ensure that the tool_config is the expected config class
    if not isinstance(tool_config, expected_config_class):  # type: ignore[arg-type]
        raise ValueError(
            f"tool_config type {type(tool_config).__name__} doesn't match "
            f"toolkit '{toolkit}' (expected {expected_config_class.__name__})"  # type: ignore[attr-defined]
        )

    # Get the run function for the tool
    run_func = SP_TOOL_MAP[toolkit]["run_func"]

    # Wrap the config input in the expected input class
    inputs = SP_TOOL_MAP[toolkit]["input"](complexes=complexes, msas=msas)  # type: ignore[operator]

    # Run the prediction
    return run_func(inputs, tool_config)  # type: ignore[no-any-return, operator]

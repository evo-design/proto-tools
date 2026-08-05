"""PyRosetta Modal service.

Delegates to proto-tools' five PyRosetta entry points for parameter validation, environment
setup, and scoring. PyRosetta installs unauthenticated from ``conda.rosettacommons.org``, so the
build needs no credentials; the licence is held by whoever deploys, free for academic, non-profit
and government use and paid for commercial. CPU only, with no model weights to stage.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import (
    RUNTIME_ENV,
    env_for,
    run_tool_call,
    stage_all_excluded_fixtures,
)


def _warmup() -> None:
    """Deploy-time: build the env so the conda install is not paid for on first call."""
    from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_energy import example_input, run_pyrosetta_energy

    run_pyrosetta_energy(example_input())


image = (
    with_proto_tools(CPU_BASE)
    # PROTO_MODEL_CACHE unused for pure-CPU tools; set for env consistency with GPU services
    .env(env_for())
)
image = (
    stage_all_excluded_fixtures(image)
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("PyRosettaService")


@app.cls(
    include_source=False,
    image=image,
    cpu=4,
    timeout=SERVICE_MODAL_TIMEOUTS["PyRosettaService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "pyrosetta-energy": "energy",
        "pyrosetta-interface-analyzer": "interface_analyzer",
        "pyrosetta-relax": "relax",
        "pyrosetta-sap": "sap",
        "pyrosetta-sasa": "sasa",
    }
)
class PyRosettaService:
    """Modal service for PyRosetta structure scoring and relaxation."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so PyRosetta initializes once across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("pyrosetta")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def energy(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Score structures with the Rosetta energy function.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-structure energy terms.
        """
        from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_energy import (
            PyRosettaEnergyConfig,
            PyRosettaEnergyInput,
            run_pyrosetta_energy,
        )

        return run_tool_call(
            run_pyrosetta_energy,
            PyRosettaEnergyInput,
            PyRosettaEnergyConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def interface_analyzer(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Analyze a protein-protein interface.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-structure interface metrics.
        """
        from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_interface_analyzer import (
            PyRosettaInterfaceAnalyzerConfig,
            PyRosettaInterfaceAnalyzerInput,
            run_pyrosetta_interface_analyzer,
        )

        return run_tool_call(
            run_pyrosetta_interface_analyzer,
            PyRosettaInterfaceAnalyzerInput,
            PyRosettaInterfaceAnalyzerConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def relax(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Relax structures under the Rosetta force field.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Relaxed structures and their scores.
        """
        from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import (
            PyRosettaRelaxConfig,
            PyRosettaRelaxInput,
            run_pyrosetta_relax,
        )

        return run_tool_call(
            run_pyrosetta_relax,
            PyRosettaRelaxInput,
            PyRosettaRelaxConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def sap(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute spatial aggregation propensity.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-structure SAP scores.
        """
        from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_sap import (
            PyRosettaSAPConfig,
            PyRosettaSAPInput,
            run_pyrosetta_sap,
        )

        return run_tool_call(
            run_pyrosetta_sap,
            PyRosettaSAPInput,
            PyRosettaSAPConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

    @modal.method()
    def sasa(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute solvent-accessible surface area.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Per-structure SASA values.
        """
        from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_sasa import (
            PyRosettaSASAConfig,
            PyRosettaSASAInput,
            run_pyrosetta_sasa,
        )

        return run_tool_call(
            run_pyrosetta_sasa,
            PyRosettaSASAInput,
            PyRosettaSASAConfig,
            input_dict,
            config_dict,
            instance=self.instance,
        )

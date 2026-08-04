"""Foldseek Modal service.

Delegates to proto-tools' four structure-query Foldseek entry points for parameter validation,
environment setup, and search. Foldseek installs from bioconda at build time, so the warmup pays
for the env once and the image layer keeps it.

``foldseek-rbh`` is absent by design: it is declared ``local_only`` because it searches a database
on the caller's own disk. The two search tools here reach that same limit only in one mode —
``FoldseekSearchConfig`` and ``FoldseekMultimerSearchConfig`` refuse ``search_mode='local'`` on a
remote device, while their default ``'remote'`` mode queries the public server and works. The two
clustering tools resolve a directory input to inline structure content before transport, so nothing
of theirs is left behind on the caller's machine. CPU only, with no weights to stage.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, MODEL_CACHE, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, dispatch_tool_call, env_for


def _warmup() -> None:
    """Deploy-time: build the env so the conda install is not paid for on first call."""
    from proto_tools.tools.structure_alignment.foldseek.foldseek_cluster import example_input, run_foldseek_cluster

    run_foldseek_cluster(example_input())


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(_warmup, volumes={"/weights": MODEL_CACHE}, secrets=[HF_TOKEN_SECRET], include_source=False)
    .env(RUNTIME_ENV)
)


app = get_app_for_service("FoldseekService")


@app.cls(
    include_source=False,
    image=image,
    cpu=4,
    timeout=SERVICE_MODAL_TIMEOUTS["FoldseekService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools(
    {
        "foldseek-cluster": "cluster",
        "foldseek-multimer-search": "multimer_search",
        "foldseek-multimercluster": "multimer_cluster",
        "foldseek-search": "search",
    }
)
class FoldseekService:
    """Modal service for Foldseek structural search and clustering."""

    @modal.enter()
    def setup(self) -> None:
        """Start a persistent worker so the binary stays warm across requests."""
        from proto_tools.utils.tool_instance import ToolInstance

        self._persist_ctx = ToolInstance.persist_tool("foldseek")
        self.instance = self._persist_ctx.__enter__()

    @modal.exit()
    def teardown(self) -> None:
        """Close the persistent worker context manager on container shutdown."""
        self._persist_ctx.__exit__(None, None, None)

    @modal.method()
    def cluster(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Cluster structures by structural similarity.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Cluster assignments and representatives.
        """
        from proto_tools.tools.structure_alignment.foldseek.foldseek_cluster import (
            FoldseekClusterConfig,
            FoldseekClusterInput,
            run_foldseek_cluster,
        )

        inputs = FoldseekClusterInput(**input_dict)
        config = FoldseekClusterConfig(**config_dict)
        return dispatch_tool_call(run_foldseek_cluster, inputs, config, instance=self.instance)

    @modal.method()
    def multimer_search(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Search a multimeric query structure against a structure database.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Hits with structural alignment statistics.
        """
        from proto_tools.tools.structure_alignment.foldseek.foldseek_multimer_search import (
            FoldseekMultimerSearchConfig,
            FoldseekMultimerSearchInput,
            run_foldseek_multimer_search,
        )

        inputs = FoldseekMultimerSearchInput(**input_dict)
        config = FoldseekMultimerSearchConfig(**config_dict)
        return dispatch_tool_call(run_foldseek_multimer_search, inputs, config, instance=self.instance)

    @modal.method()
    def multimer_cluster(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Cluster multimeric structures by structural similarity.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Cluster assignments and representatives.
        """
        from proto_tools.tools.structure_alignment.foldseek.foldseek_multimercluster import (
            FoldseekMultimerClusterConfig,
            FoldseekMultimerClusterInput,
            run_foldseek_multimercluster,
        )

        inputs = FoldseekMultimerClusterInput(**input_dict)
        config = FoldseekMultimerClusterConfig(**config_dict)
        return dispatch_tool_call(run_foldseek_multimercluster, inputs, config, instance=self.instance)

    @modal.method()
    def search(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Search a query structure against a structure database.

        Args:
            input_dict (dict[str, Any]): Mapping of input names to their serialized values.
            config_dict (dict[str, Any]): Mapping of configuration parameter names to values.

        Returns:
            dict[str, Any]: Hits with structural alignment statistics.
        """
        from proto_tools.tools.structure_alignment.foldseek.foldseek_search import (
            FoldseekSearchConfig,
            FoldseekSearchInput,
            run_foldseek_search,
        )

        inputs = FoldseekSearchInput(**input_dict)
        config = FoldseekSearchConfig(**config_dict)
        return dispatch_tool_call(run_foldseek_search, inputs, config, instance=self.instance)

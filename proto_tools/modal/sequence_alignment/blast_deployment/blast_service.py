"""Portable BLAST Modal service for caller-owned workspaces.

Only online NCBI QBLAST searches are remotely portable. Local database searches are rejected by
``BlastSearchConfig.remote_unsupported_reason`` before dispatch because the database path belongs
to the caller's machine. Proto's hosted backend may override this service with its own deployment
that mounts managed databases; this definition deliberately has no Proto volumes or secrets.
"""

from typing import Any

import modal

from proto_tools.modal.app import HF_TOKEN_SECRET, SERVICE_RETRIES, get_app_for_service
from proto_tools.modal.base_images import CPU_BASE, with_proto_tools
from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
from proto_tools.modal.registry import register_tools
from proto_tools.modal.utils import RUNTIME_ENV, env_for, run_tool_call


def _warmup() -> None:
    """Validate the online-only runtime without submitting a request to NCBI during deployment."""
    from Bio.Blast import NCBIWWW, NCBIXML  # noqa: F401

    from proto_tools.tools.sequence_alignment.blast.blast_search import BlastSearchConfig, BlastSearchInput

    BlastSearchInput(query="ATGC")
    BlastSearchConfig(search_mode="online")


image = (
    with_proto_tools(CPU_BASE)
    .env(env_for())
    .run_function(
        _warmup,
        secrets=[HF_TOKEN_SECRET],
        include_source=False,
    )
    .env(RUNTIME_ENV)
)


app = get_app_for_service("BlastService")


@app.cls(
    include_source=False,
    image=image,
    cpu=2,
    timeout=SERVICE_MODAL_TIMEOUTS["BlastService"],
    retries=SERVICE_RETRIES,
    secrets=[HF_TOKEN_SECRET],
)
@register_tools({"blast-search": "search"})
class BlastService:
    """Modal service for online NCBI BLAST searches."""

    @modal.method()
    def search(self, input_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
        """Search an NCBI BLAST database through QBLAST.

        Args:
            input_dict (dict[str, Any]): Serialized BLAST query input.
            config_dict (dict[str, Any]): Serialized online-search configuration.

        Returns:
            dict[str, Any]: Structured BLAST hits and query metadata.

        Raises:
            ValueError: If a caller bypasses dispatch validation and requests local database mode.
        """
        from proto_tools.tools.sequence_alignment.blast.blast_search import (
            BlastSearchConfig,
            BlastSearchInput,
            run_blast_search,
        )

        config = BlastSearchConfig(**config_dict)
        if config.search_mode != "online":
            raise ValueError("BlastService only supports search_mode='online'; local databases are not portable.")
        return run_tool_call(
            run_blast_search,
            BlastSearchInput,
            BlastSearchConfig,
            input_dict,
            config_dict,
        )

"""Deployment of proto-tools onto Modal infrastructure, and dispatch to it.

Exports the client surface that ``config.device == "modal"`` routes through.
"""

from proto_tools.modal.client import (
    DeploymentDriftWarning,
    ModalCredentialsError,
    ModalDispatchError,
    ModalEnvironmentNotFoundError,
    ToolNotDeployedError,
    ToolNotShippedError,
    available_tools,
    dispatch_batch_to_modal,
    dispatch_to_modal,
    resolve_tool,
)

__all__ = [
    "DeploymentDriftWarning",
    "ModalCredentialsError",
    "ModalDispatchError",
    "ModalEnvironmentNotFoundError",
    "ToolNotDeployedError",
    "ToolNotShippedError",
    "available_tools",
    "dispatch_batch_to_modal",
    "dispatch_to_modal",
    "resolve_tool",
]

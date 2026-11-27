"""Dispatch helpers for operation-based masked-model workers."""

from typing import Any

from proto_tools.utils import BaseConfig, BaseToolInput, ToolInstance
from proto_tools.utils.compressed_array import decompress_result


def dispatch_masked_model(
    toolkit: str, operation: str, inputs: BaseToolInput, config: BaseConfig, instance: Any
) -> dict[str, Any]:
    """Dispatch an operation and decode dense outputs through the standard worker infrastructure."""
    payload = {**config.model_dump(), **inputs.model_dump(), "operation": operation}
    result = ToolInstance.dispatch(toolkit, payload, instance=instance, config=config)
    result = decompress_result(result, to_list=True)
    result["metadata"] = {"model_checkpoint": payload["model_checkpoint"]}
    return result  # type: ignore[no-any-return]

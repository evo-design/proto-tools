"""Standalone Paperclip client runner."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)


@contextmanager
def _paperclip_import_path(library_path: str | None) -> Iterator[None]:
    """Temporarily expose an optional managed Paperclip client path."""
    if not library_path:
        yield
        return
    path = str(Path(library_path).expanduser())
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if added:
            sys.path.remove(path)


def _client(config: dict[str, Any]) -> Any:
    """Create an authenticated client from the standalone request config."""
    try:
        client_module = importlib.import_module("gxl_paperclip")
    except ModuleNotFoundError as exc:
        if exc.name != "gxl_paperclip":
            raise
        with _paperclip_import_path(config.get("paperclip_library_path")):
            client_module = importlib.import_module("gxl_paperclip")
    client_options = {
        "base_url": config.get("paperclip_base_url"),
        "timeout": config["timeout_seconds"],
        "user_agent": "proto-tools/paperclip-literature-v1",
    }
    api_key = config.get("paperclip_api_key")
    if api_key:
        return client_module.PaperclipClient(auth=client_module.APIKeyAuth(api_key), **client_options)
    return client_module.PaperclipClient.from_env(**client_options)


def _require_success(result: Any, operation: str) -> None:
    """Raise an actionable error for a failed Paperclip operation."""
    if result.exit_code != 0:
        raise RuntimeError(f"Paperclip {operation} failed with exit code {result.exit_code}: {result.output}")


def run_paperclip(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Execute one search or exact-document retrieval operation."""
    operation = input_dict["operation"]
    if operation not in {"search", "document"}:
        raise ValueError(f"Unsupported Paperclip operation: {operation!r}")
    config = input_dict["config"]
    client = _client(config)
    if operation == "search":
        result = client.search(
            input_dict["query"],
            source=",".join(config["sources"]),
            limit=config["limit"],
            exact=config["exact"],
            since=config["since"],
            timeout=config["timeout_seconds"],
        )
        _require_success(result, "search")
        return {"result_id": result.result_id, "papers": result.papers}
    if operation == "document":
        base = f"/papers/{input_dict['document_id']}"
        metadata_result = client.papers.cat(f"{base}/meta.json", timeout=config["timeout_seconds"])
        _require_success(metadata_result, "metadata retrieval")
        output: dict[str, Any] = {
            "metadata_output": metadata_result.output,
            "content_exit_code": None,
            "content_output": None,
            "wc_exit_code": None,
            "wc_output": None,
        }
        if config["include_full_text"]:
            content_result = client.papers.cat(f"{base}/content.lines", timeout=config["timeout_seconds"])
            output["content_exit_code"] = content_result.exit_code
            output["content_output"] = content_result.output
            if content_result.exit_code == 0 and content_result.output.strip():
                wc_result = client.execute("wc", [f"{base}/content.lines"], timeout=config["timeout_seconds"])
                output["wc_exit_code"] = wc_result.exit_code
                output["wc_output"] = wc_result.output
        return output
    raise RuntimeError(f"Unhandled Paperclip operation: {operation!r}")


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run Paperclip through the persistent-worker protocol."""
    return run_paperclip(input_dict)


def to_device(device: str) -> dict[str, Any]:
    """Acknowledge CPU placement for the remote literature client."""
    return {"success": True, "device": device, "note": "CPU tool"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_json_path> <output_json_path>", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as input_handle:
        input_data = json.load(input_handle)
    output_data = run_paperclip(input_data)
    with open(sys.argv[2], "w", encoding="utf-8") as output_handle:
        json.dump(output_data, output_handle)

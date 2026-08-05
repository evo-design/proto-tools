"""Smoke tests that run each deployed tool's canonical example input."""

import time
import traceback

import modal

from proto_tools.modal.manifest import APP_BUCKETS, physical_device_for_service
from proto_tools.utils.base_config import BaseConfig


def _tool_keys_for_service(service_class: str) -> dict[str, str]:
    """Return ``{tool_key: method_name}`` registered by ``service_class``."""
    from proto_tools.modal.registry import get_registry, import_all_services

    import_all_services()
    return {key: method for key, (cls_name, method) in get_registry().items() if cls_name == service_class}


def run_tool(service_class: str, tool_key: str, method_name: str) -> tuple[bool, str]:
    """Run one tool's example input against its deployed service.

    Returns:
        tuple[bool, str]: ``(passed, detail)``. ``detail`` is a short status
            note — timing on success, the failure reason otherwise.
    """
    from proto_tools.modal.manifest import get_app_name_for_service
    from proto_tools.tools import ToolRegistry

    example = ToolRegistry.get_example_input(tool_key)
    if example is None:
        return True, "skipped — no example_input declared"

    config_model = ToolRegistry.get(tool_key).config_model
    if not issubclass(config_model, BaseConfig):
        return False, f"config model is {config_model.__name__}, not a BaseConfig"
    # ``minimal()`` rather than the bare defaults: it is the constructor that supplies whatever a
    # tool needs to run at all (miranda's microRNA queries) and reduces the work to the cheapest
    # path through it, which is what keeps a design pipeline's smoke test from billing a full run.
    config = config_model.minimal()
    config.device = physical_device_for_service(service_class)

    app_name = get_app_name_for_service(service_class)
    started = time.perf_counter()
    Service = modal.Cls.from_name(app_name, service_class)
    method = getattr(Service(), method_name)
    result = method.remote(
        input_dict=example.model_dump(mode="json"),
        config_dict=config.model_dump(mode="json"),
    )
    elapsed = time.perf_counter() - started

    if not isinstance(result, dict):
        return False, f"expected a dict result, got {type(result).__name__}"
    if result.get("success") is False:
        return False, f"tool reported failure: {result.get('errors')}"
    return True, f"{elapsed:.1f}s"


def smoke_app(app_name: str) -> tuple[int, int]:
    """Run every tool registered to ``app_name``. Returns ``(passed, failed)``."""
    passed = failed = 0
    for service_class in APP_BUCKETS[app_name]:
        tools = _tool_keys_for_service(service_class)
        if not tools:
            print(f"  ⚠️  {service_class}: no registered tools (license-gated or unregistered)")
            continue
        for tool_key, method_name in sorted(tools.items()):
            try:
                ok, detail = run_tool(service_class, tool_key, method_name)
            except Exception:
                print(f"  ❌ {tool_key}")
                traceback.print_exc()
                failed += 1
                continue
            if ok:
                print(f"  ✅ {tool_key} ({detail})")
                passed += 1
            else:
                print(f"  ❌ {tool_key} — {detail}")
                failed += 1
    return passed, failed

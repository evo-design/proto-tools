"""The manifest, the entrypoints rendered from it, and the generated tool map must agree.

Disagreement is silent at deploy time and confusing at call time: a wrong entry
sends calls to the wrong Modal method, or reports a deployed tool as missing.
"""

from tests.modal_tests.helpers import REPO


def test_every_app_renders_a_valid_entrypoint():
    """Every manifest app must render an entrypoint that imports and binds the right app.

    Rendered per deploy rather than committed, so the failure this guards against
    is a manifest entry naming a service module that cannot be imported.
    """
    import ast

    from proto_tools.modal.deploy import render_entrypoint
    from proto_tools.modal.manifest import APP_BUCKETS, SERVICE_TO_MODULE

    broken = []
    for app_name, services in sorted(APP_BUCKETS.items()):
        source = render_entrypoint(app_name, services)
        try:
            ast.parse(source)
        except SyntaxError as exc:
            broken.append(f"{app_name}: renders invalid Python ({exc})")
            continue
        if f'get_app("{app_name}")' not in source:
            broken.append(f"{app_name}: entrypoint does not bind its own app")
        broken.extend(
            f"{app_name}: does not import {service}" for service in services if SERVICE_TO_MODULE[service] not in source
        )
    assert not broken, "manifest apps that render a broken entrypoint:\n  " + "\n  ".join(broken)


def test_every_app_has_registered_tools():
    """An app with no tools is unreachable: deployable, but nothing can dispatch to it."""
    from proto_tools.modal import available_tools
    from proto_tools.modal.manifest import APP_BUCKETS, app_slug, get_app_name_for_service

    reachable = {get_app_name_for_service(svc) for svc, _ in available_tools().values()}
    orphans = sorted(app_slug(a) for a in APP_BUCKETS if a not in reachable)
    assert not orphans, f"apps with no registered tools: {orphans}"


def test_tool_map_matches_the_registry():
    """The generated dispatch table must match what the services actually register.

    A stale map sends calls to the wrong Modal method, or reports a tool as
    missing that is deployed — both silent at deploy time and confusing at
    call time.
    """
    from scripts.generate_tool_map import TOOL_MAP_PATH, render_tool_map

    assert TOOL_MAP_PATH.exists(), "run scripts/generate_tool_map.py"
    assert TOOL_MAP_PATH.read_text() == render_tool_map(), (
        "the generated tool map is out of date — run scripts/generate_tool_map.py"
    )


def test_client_resolves_without_importing_service_modules():
    """The whole point of the static map: a caller must not need proto_tools/modal/.

    Service modules build Modal images and require the proto-tools submodule
    on disk. If the client starts importing them again, a pip-installed
    client stops working and every dispatch pays that import.
    """
    import subprocess
    import sys

    probe = (
        "import sys; sys.path.insert(0, '.');"
        "from proto_tools.modal import resolve_tool, available_tools;"
        "assert resolve_tool('esm2-embedding')[0] == 'proto-tools-esm2';"
        "assert len(available_tools()) > 0;"
        "leaked = sorted(m for m in sys.modules if m.endswith('_service'));"
        "print('LEAKED:' + ','.join(leaked))"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr[-400:]
    leaked = out.stdout.strip().removeprefix("LEAKED:")
    assert not leaked, f"client imported service modules, which pull in Modal image definitions: {leaked}"


def test_tool_map_gpu_flags_match_the_manifest():
    """Wrong hardware flags silently run a GPU model on a CPU."""
    from proto_tools.modal.manifest import GPU_SERVICES
    from proto_tools.modal.tool_map import TOOL_MAP

    wrong = [k for k, e in TOOL_MAP.items() if e.gpu != (e.service in GPU_SERVICES)]
    assert not wrong, f"tool_map gpu flags disagree with GPU_SERVICES: {wrong}"

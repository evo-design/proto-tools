"""The manifest, the entrypoints rendered from it, and the generated tool map must agree.

Disagreement is silent at deploy time and confusing at call time: a wrong entry
sends calls to the wrong Modal method, or reports a deployed tool as missing.
"""

import pytest

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


# ── Wall tiers ───────────────────────────────────────────────────────────────
# Walls in force before tiers were named. A tier assignment may raise a service's wall but must
# never shorten one: work that used to finish would start being killed mid-flight.
_WALLS_BEFORE_TIERS: dict[str, int] = {
    "AbLangService": 3600,
    "AlphaFold2Service": 3600,
    "BioEmuService": 14400,
    "Boltz2Service": 3600,
    "BorzoiService": 3600,
    "Chai1Service": 3600,
    "CrisprTracrRNAService": 14400,
    "DSSPService": 600,
    "ESM2Service": 3600,
    "ESMCService": 3600,
    "ESMFold2Service": 3600,
    "ESMFoldService": 3600,
    "ESMIF1Service": 3600,
    "EnformerService": 1800,
    "Evo1Service": 3600,
    "Evo2Service": 3600,
    "FAMPNNService": 3600,
    "FreeBindCraftService": 86400,
    "IPSAEService": 600,
    "LigandMPNNService": 1800,
    "MafftAlignService": 1800,
    "MalinoisService": 1800,
    "Metal3DService": 3600,
    "MincedService": 1800,
    "OrfipyService": 600,
    "PangolinService": 3600,
    "ProGen2Service": 3600,
    "ProteinMPNNService": 3600,
    "ProtenixService": 3600,
    "PyMOLService": 600,
    "RF3Service": 3600,
    "RFdiffusion3Service": 3600,
    "SegmaskerService": 600,
    "SpliceTransformerService": 1800,
    "TMalignService": 600,
    "USalignService": 600,
}

# Floor on a GPU service's per-item budget (wall / max_chunk_size). Chosen to sit just under the
# 56 s/item that every correctly-sized GPU service already clears, so it catches a chunk size
# grown out of step with its wall without pinning the exact values.
_MIN_GPU_SECONDS_PER_ITEM = 50


def test_every_service_has_a_tier():
    """A service without a tier gets no wall, and its @app.cls would fail at deploy."""
    from proto_tools.modal.manifest import SERVICE_TIERS, SERVICE_TO_MODULE, TIER_SECONDS

    assert set(SERVICE_TIERS) == set(SERVICE_TO_MODULE), "SERVICE_TIERS and SERVICE_TO_MODULE disagree"
    unknown = {s: t for s, t in SERVICE_TIERS.items() if t not in TIER_SECONDS}
    assert not unknown, f"services assigned an undefined tier: {unknown}"


def test_no_service_wall_was_shortened():
    """Tiers may lengthen a wall; shortening one kills work that used to complete."""
    from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS

    shortened = {
        service: (before, SERVICE_MODAL_TIMEOUTS[service])
        for service, before in _WALLS_BEFORE_TIERS.items()
        if SERVICE_MODAL_TIMEOUTS[service] < before
    }
    assert not shortened, f"tier assignment shortened walls (service: before -> after): {shortened}"


def test_gpu_walls_leave_a_plausible_per_item_budget():
    """A chunk size grown out of step with its wall cannot finish; esmfold shipped at 3.5 s/sequence."""
    from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS
    from proto_tools.modal.tool_map import TOOL_MAP
    from proto_tools.tools import ToolRegistry

    specs = {spec.key: spec for spec in ToolRegistry.list_all()}
    too_tight = {}
    for key, entry in TOOL_MAP.items():
        if not entry.gpu:
            continue
        chunk = specs[key].max_chunk_size or 1
        budget = SERVICE_MODAL_TIMEOUTS[entry.service] / chunk
        if budget < _MIN_GPU_SECONDS_PER_ITEM:
            too_tight[key] = f"{chunk} items in {SERVICE_MODAL_TIMEOUTS[entry.service]}s = {budget:.1f}s/item"
    assert not too_tight, f"GPU tools whose chunk cannot finish inside their wall: {too_tight}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("2", 2.0, id="lengthens"),
        pytest.param("0.5", 1.0, id="would_shorten"),
        pytest.param("later", 1.0, id="unparseable"),
    ],
)
def test_timeout_scale_never_shortens_a_wall(monkeypatch, raw, expected):
    """The deploy-time override may lengthen a wall; a value below 1 is ignored, not honoured."""
    from proto_tools.modal.manifest import _resolve_timeout_scale

    monkeypatch.delenv("PROTO_MODAL_TIMEOUT_SCALE", raising=False)
    if raw is not None:
        monkeypatch.setenv("PROTO_MODAL_TIMEOUT_SCALE", raw)
    assert _resolve_timeout_scale() == expected

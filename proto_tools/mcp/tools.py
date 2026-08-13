"""Tool implementations behind the MCP server, kept free of MCP decorators."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

from proto_tools.mcp.device import Device, is_remote

if TYPE_CHECKING:  # imported for the annotation only; the runtime import stays inside deploy_tool
    from proto_tools.modal.deploy import ModalTokens
from proto_tools.utils.base_config import BaseConfig
from proto_tools.utils.modal_status import credentials_checked, deployed_apps

logger = logging.getLogger(__name__)

# Fields large enough to hurt an agent's context if returned inline. Anything
# longer than this is written to disk and replaced with a path.
INLINE_CHAR_LIMIT = 2_000

# Where run_tool writes large outputs when the caller does not choose.
DEFAULT_OUTPUT_DIR = Path.cwd() / "proto_tools_outputs"


def _registry() -> dict[str, tuple[str, str]]:
    from proto_tools.modal.client import available_tools

    return available_tools()


def _app_for(tool_key: str) -> str:
    from proto_tools.modal.manifest import get_app_name_for_service

    service, _method = _registry()[tool_key]
    return get_app_name_for_service(service)


def app_for_tool(tool_key: str) -> str | None:
    """Return the Modal app serving ``tool_key``, or ``None`` if it serves none."""
    try:
        return _app_for(tool_key)
    except KeyError:
        return None


def _hosted_catalogue() -> dict[str, dict[str, Any]]:
    """Tools Proto hosts, keyed by tool key, with an ``unhosted_reason`` when it does not."""
    from proto_tools.proto import _get_client

    entries = _get_client(None).tools.catalogue()
    return {entry["key"]: entry for entry in entries}


def _all_registered() -> set[str]:
    """Every tool key in the registry, deployable or not."""
    from proto_tools.tools import ToolRegistry

    return {spec.key for spec in ToolRegistry.list_all()}


def runs_in_process(tool_key: str) -> bool:
    """Report whether a call to ``tool_key`` on a remote device is answered here instead.

    A ``local_only`` tool never has a deployment, and its inputs are on this machine anyway. A
    ``local_cpu`` tool has no GPU and no environment, so it runs here only when nothing serves it
    — deploying one is a choice, and ``pdockq2`` and ``structure-metrics`` were deployed. That is
    the same question the tool wrapper asks, so the two paths agree on where a call landed.

    Config-dependent redirects — a search mode that is an HTTP call — are excluded, because
    whether they apply depends on a config the caller has not written yet.
    """
    from proto_tools.tools import ToolRegistry

    spec = ToolRegistry.get(tool_key)
    if spec.local_only:
        return True
    return bool(spec.local_cpu) and tool_key not in _registry()


def answered_in_process_keys() -> set[str]:
    """Return the tools a remote session answers here rather than dispatching."""
    from proto_tools.tools import ToolRegistry

    return {spec.key for spec in ToolRegistry.list_all() if runs_in_process(spec.key)}


def deployed_keys(device: Device, *, environment: str | None = None, client: Any | None = None) -> set[str]:
    """Return the tools ``device`` actually serves, ignoring anything answered in-process.

    ``environment`` and ``client`` name whose workspace to ask about. Omitted, the question is
    answered for this process, which is what a local session wants. A server answering for
    someone else must pass theirs, or it reports its own deployments as though they were the
    caller's.
    """
    if device == "local":
        return set()
    if device == "proto":
        return {key for key, entry in _hosted_catalogue().items() if entry.get("hosted")}
    live = deployed_apps(environment=environment, client=client)
    return {key for key in _registry() if _app_for(key) in live}


def available_keys(device: Device, *, environment: str | None = None, client: Any | None = None) -> set[str]:
    """Return the tool keys that can actually run on ``device``."""
    if device == "local":
        # Every registered tool runs here: a standalone env builds on first use, so availability
        # is a question of time and disk rather than of what has been provisioned in advance.
        return _all_registered()
    return deployed_keys(device, environment=environment, client=client) | answered_in_process_keys()


def workspace_info(
    device: Device = "modal", *, environment: str | None = None, client: Any | None = None
) -> dict[str, Any]:
    """Report where calls will land, and whether the caller can deploy there.

    ``environment`` and ``client`` describe whose workspace to report on. Omitted, this describes
    the process's own, which is what a local session wants.
    """
    if device == "local":
        from proto_tools.tools import ToolRegistry
        from proto_tools.utils.device import number_of_visible_gpus
        from proto_tools.utils.proto_home import get_proto_home

        gpus = number_of_visible_gpus()
        return {
            "device": "local",
            "authenticated": True,  # nothing to authenticate against
            "tools_total": len(ToolRegistry.list_all()),
            "gpus_visible": gpus,
            "proto_home": str(get_proto_home()),
            "deployable": False,
            "note": (
                "Tools run in this process. Each builds its standalone environment and downloads "
                "its weights on first use, so a first call can be slow."
                + ("" if gpus else " No GPU is visible, so GPU-only tools cannot run here.")
            ),
        }

    if device == "proto":
        catalogue = _hosted_catalogue()
        return {
            "device": "proto",
            "authenticated": bool(os.environ.get("PROTO_API_KEY")),
            "tools_hosted": sum(1 for e in catalogue.values() if e.get("hosted")),
            "tools_total": len(catalogue),
            "deployable": False,
            "note": (
                "Proto hosts a fixed catalogue. You cannot deploy to it; use device='modal' "
                "to run a tool in your own Modal workspace instead."
            ),
        }

    import modal

    from proto_tools.modal.app import resolve_environment
    from proto_tools.modal.manifest import APP_BUCKETS

    try:
        # A caller-supplied client already carries credentials; only the process needs checking.
        if client is None:
            modal.Client.from_env()  # raises when no credentials are configured
    except Exception as exc:
        return {
            "device": "modal",
            "authenticated": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": (
                "Interactive shell: run `modal token new` (writes ~/.modal.toml). "
                "Container, CI, or agent sandbox: set the MODAL_TOKEN_ID and "
                "MODAL_TOKEN_SECRET environment variables instead — a token file written "
                "outside this process is not visible to it."
            ),
            "credentials_checked": credentials_checked(),
        }

    # Modal exposes no public reader for the active profile: ``config_profiles()`` lists them all
    # and ``config_set_active_profile`` writes one, but nothing returns the current one. Read the
    # private name defensively and outside the block above, so a Modal rename costs this label
    # rather than reporting a working install as unauthenticated.
    workspace = getattr(modal.config, "_profile", None) or "(unknown)"

    from proto_tools.modal.app import environment_exists

    resolved = resolve_environment(environment)
    # Asked before counting, because an environment that does not exist counts zero apps and
    # reads as "nothing deployed yet" — which sends the caller off to deploy into a place that
    # cannot receive it. The one-time setup is the actual answer.
    if environment_exists(resolved, client) is False:
        return {
            "device": "modal",
            "authenticated": True,
            "workspace": workspace,
            "environment": resolved,
            "environment_exists": False,
            "deployable": False,
            "error": f"Modal environment {resolved!r} has not been created in this workspace.",
            "hint": f"Create it with: proto-tools deploy --create-env --env {resolved}",
        }

    deployed = deployed_apps(environment=resolved, client=client)
    return {
        "device": "modal",
        "authenticated": True,
        "workspace": workspace,
        "environment": resolved,
        "environment_exists": True,
        "apps_deployed": len(deployed),
        "apps_available": len(APP_BUCKETS),
        "tools_total": len(_registry()),
        "deployable": True,
    }


def list_tools(
    deployed_only: bool = True,
    category: str | None = None,
    device: Device = "modal",
    *,
    environment: str | None = None,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """List tools, flagged by whether they can actually run on ``device``.

    Defaults to available-only: an agent choosing from the full catalogue will
    pick tools that cannot run and hit an error it cannot resolve itself.

    Each entry carries what choosing between tools needs — category, summary, and whether it
    wants a GPU — so a caller can pick one without fetching a schema per candidate.

    "Available" is wider than "deployed". A tool needing no GPU and no environment, or one that
    can never be deployed, is answered in this process instead, so it runs on a remote device
    too. Those carry ``runs_in_process`` to explain why they are usable without a deployment.

    When using ``modal``, "deployed" means deployed in your workspace, which you can change. When
    using ``proto``, it means hosted by Proto, which you cannot.
    """
    from proto_tools.tools import ToolRegistry

    if category is not None:
        known = sorted({spec.category for spec in ToolRegistry.list_all()})
        if category not in known:
            return [{"ok": False, "error": f"no category named {category!r}", "categories": known}]

    available = available_keys(device, environment=environment, client=client)
    in_process = set() if device == "local" else answered_in_process_keys()
    # Resolved once: on ``modal`` this reads the live app list, which is a network call.
    deployed = deployed_keys(device, environment=environment, client=client)
    # The catalogue itself differs by device: the dispatch table lists what a container could
    # serve, which is the right universe for a remote backend but omits the tools answered here.
    catalogue = available if device == "local" else set(_registry()) | in_process
    out = []
    for key in sorted(catalogue):
        is_available = key in available
        if deployed_only and not is_available:
            continue
        spec = ToolRegistry.get(key)
        if category is not None and spec.category != category:
            continue
        entry: dict[str, Any] = {
            "tool_key": key,
            "category": spec.category,
            "summary": spec.description,
            "uses_gpu": spec.uses_gpu,
            "available": is_available,
        }
        # Both halves matter to a caller deciding what to do next: that no deployment is needed,
        # and why -- which is where a tool says it runs for hours.
        notes = ["Runs in this session; no deployment is needed."] if key in in_process else []
        if spec.local_only:
            notes.append(spec.local_only)
        if notes:
            entry["note"] = " ".join(notes)
        if key in in_process:
            entry["runs_in_process"] = True
        if device == "modal":
            entry["app"] = app_for_tool(key)
            entry["deployed"] = key in deployed
        out.append(entry)
    return out


# The vocabulary an agent reaches for rarely matches a tool's own wording:
# nobody searching for a structure comparison types "alignment". Substring
# matching cannot bridge that, so map the common cases explicitly.
_SYNONYMS = {
    "compare": "align",
    "comparison": "align",
    "superimpose": "align",
    "fold": "structure",
    "folding": "structure",
    "predict": "prediction",
    "design": "design",
    "embed": "embedding",
    "mutate": "mutagenesis",
    "similarity": "align",
}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "for",
        "to",
        "and",
        "or",
        "with",
        "that",
        "can",
        "how",
        "what",
        "which",
        "do",
        "does",
        "i",
        "my",
        "me",
        "two",
        "some",
        "any",
        "at",
        "by",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "on",
        "use",
        "using",
    }
)


def _stem(term: str) -> str:
    """Crudely singularise, so "structures" matches a description saying "structure"."""
    return term[:-1] if len(term) > 4 and term.endswith("s") and not term.endswith("ss") else term


def _tokens(text: str) -> set[str]:
    """Split a field into stemmed words, so matching is per word rather than per substring."""
    return {_stem(word) for word in re.split(r"[^a-z0-9]+", text.lower()) if word}


# Where a term matched, strongest first, and a whole word ahead of a fragment of one.
_KEY_WORD, _KEY_PART = 10, 6
_CATEGORY_WORD, _CATEGORY_PART = 6, 4
_SUMMARY_WORD, _SUMMARY_PART = 2, 1
# Bonus for a tool whose whole category the query named.
_CATEGORY_COVERAGE = 12
# Below this, a fragment match is noise: "dna" appears inside a dozen unrelated words.
_MIN_PART_LEN = 4


def _field_score(term: str, tokens: set[str], word: int, part: int) -> int:
    """Score one term against one field, by how much of a word it matched."""
    if term in tokens:
        return word
    if len(term) >= _MIN_PART_LEN and any(term in token for token in tokens):
        return part
    return 0


def _tool_score(forms: list[set[str]], key: set[str], category: set[str], summary: set[str]) -> int:
    """Score one tool against the query, as a relevance out of 100.

    Each term scores against the strongest field it matched. Normalising by the best a tool
    could have scored keeps the range wide enough to tell a near-exact match from a
    coincidental keyword overlap, which a sum of small integers does not.
    """
    total = sum(
        max(
            max(
                _field_score(term, key, _KEY_WORD, _KEY_PART),
                _field_score(term, category, _CATEGORY_WORD, _CATEGORY_PART),
                _field_score(term, summary, _SUMMARY_WORD, _SUMMARY_PART),
            )
            for term in form
        )
        for form in forms
    )
    if total and category:
        asked = sum(
            1 for token in category if any(_field_score(term, {token}, 1, 1) for form in forms for term in form)
        )
        total += _CATEGORY_COVERAGE * asked // len(category)
    return min(100, round(100 * total / (_KEY_WORD * len(forms))))


def _no_match_hint() -> str:
    """Say what to do next, so an empty result does not read as "no such capability"."""
    from proto_tools.tools import ToolRegistry

    categories = ", ".join(sorted({spec.category for spec in ToolRegistry.list_all()}))
    return f"Nothing matched. Browse instead with list_tools(category=...); the categories are: {categories}."


def search_tools(
    query: str,
    deployed_only: bool = True,
    limit: int = 10,
    device: Device = "modal",
    *,
    environment: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Find tools by keyword, best match first.

    Matches per term rather than on the whole string: agents ask in natural
    language ("compare two protein structures"), and a literal substring
    search returns nothing for those, which reads as "no such tool exists"
    rather than "rephrase".

    Returns the top ``limit`` under ``tools``, each carrying the ``score`` out of 100 it ranked
    on, with ``n_total`` for how many matched in all. Broad queries match half the catalogue,
    and an agent that cannot tell first place from fiftieth pays for the whole list to find out.

    Take the key to run from each entry's ``tool_key``. A display name is not a key.
    """
    terms = [_stem(t) for t in re.split(r"[^a-z0-9]+", query.lower()) if t not in _STOPWORDS and len(t) > 1]
    # Both the term and its expansion are scored: rewriting "fold" to "structure" otherwise
    # discards the literal token that matches esmfold and foldseek in a key.
    forms = [{t, _stem(_SYNONYMS.get(t, t))} for t in terms]
    if not forms:
        return {"tools": [], "n_total": 0, "hint": _no_match_hint()}

    scored = []
    for entry in list_tools(deployed_only=deployed_only, device=device, environment=environment, client=client):
        score = _tool_score(
            forms,
            _tokens(entry["tool_key"]),
            _tokens(entry.get("category") or ""),
            _tokens(entry.get("summary") or ""),
        )
        if score:
            scored.append((score, entry))

    # Key order after score, so a tie ranks the same way twice rather than by registry order.
    scored.sort(key=lambda pair: (-pair[0], pair[1]["tool_key"]))
    found = {"tools": [{**entry, "score": score} for score, entry in scored[:limit]], "n_total": len(scored)}
    if not scored:
        found["hint"] = _no_match_hint()
    return found


def _unknown_key(tool_key: str) -> dict[str, Any]:
    """Describe an unresolvable tool key as a result the caller can act on.

    Returned rather than raised: a raise becomes an MCP protocol error, which reads as
    "this call is broken" instead of "try another argument". Guessing a key is the most
    likely mistake an agent makes, because the key is the one thing it has to invent.
    """
    from proto_tools.tools import ToolRegistry

    return {
        "ok": False,
        "error": f"no tool with key {tool_key!r}",
        "did_you_mean": ToolRegistry.suggest_keys(tool_key),
        "hint": (
            "Tool keys are '<model>-<action>', for example 'esmfold-prediction'. Call search_tools() to find one."
        ),
    }


def get_tool_schema(tool_key: str) -> dict[str, Any]:
    """Return the input, config and output JSON schemas for one tool."""
    from proto_tools.tools import ToolRegistry

    try:
        spec = ToolRegistry.get(tool_key)
    except (ValueError, KeyError):
        return _unknown_key(tool_key)
    return {
        "tool_key": tool_key,
        "description": (spec.description or "").strip(),
        "input_schema": spec.input_model.model_json_schema(),
        "config_schema": spec.config_model.model_json_schema(),
        "output_schema": spec.output_model.model_json_schema(),
    }


def _elide(value: Any, key: str = "") -> Any:
    """Replace bulky leaves with a description of what they hold.

    An example exists to show *shape*. Structure tools carry entire PDB files
    in theirs — tens of thousands of characters that tell a caller nothing it
    could not infer from a placeholder, and that overflow a context window.

    Elided structure content says a path is accepted, because the placeholder otherwise reads
    as "inline this much text to call me", and an agent either does that or gives up on the
    tool. ``key`` carries the field the value sat under, so only the fields that really take a
    path say so.
    """
    if isinstance(value, dict):
        return {k: _elide(v, k) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 3:
            return [_elide(v, key) for v in value[:3]] + [f"<… {len(value) - 3} more items>"]
        return [_elide(v, key) for v in value]
    if isinstance(value, str) and len(value) > 200:
        if key == "structure":
            return (
                f"<{len(value):,} characters elided — a file path or http(s) URL is accepted "
                f"instead, here or in place of the whole field, e.g. '/path/to/structure.pdb'>"
            )
        return f"<{len(value):,} characters elided — see run_tool(use_example=True)>"
    return value


def get_tool_example(tool_key: str) -> dict[str, Any] | None:
    """Return the tool's canonical example input, or ``None`` if it declares none.

    Bulky values are elided: this shows the shape to build a call, not a
    payload to copy. To run the example as-is, use
    ``run_tool(tool_key, use_example=True)``.
    """
    from proto_tools.tools import ToolRegistry

    try:
        example = ToolRegistry.get_example_input(tool_key)
    except (ValueError, KeyError):
        return _unknown_key(tool_key)
    return None if example is None else _elide(example.model_dump(mode="json"))


def get_tool_citation(tool_key: str) -> dict[str, Any]:
    """Return the BibTeX citation and DOI for the work a tool implements.

    An agent that reports results is expected to attribute the method it used, and the
    reference is already in the tool's ``cite.bib``. ``bibtex`` is ``None`` for the few
    tools that register none.
    """
    from proto_tools.tools import ToolRegistry

    try:
        bibtex = ToolRegistry.get_citation(tool_key)
    except (ValueError, KeyError):
        return _unknown_key(tool_key)
    return {
        "tool_key": tool_key,
        "bibtex": bibtex,
        "doi": ToolRegistry.get_doi(tool_key),
        "docs_url": ToolRegistry.get_docs_url(tool_key),
    }


def _spill_path(output_dir: Path, key_path: str, suffix: str) -> Path:
    """Return the path to spill one oversized field to, creating the directory on first use.

    Created here rather than before the call, so a result that fits inline leaves no empty
    directory behind in the caller's working directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{key_path.replace('.', '_') or 'value'}.{suffix}"


def _summarize(value: Any, key_path: str, output_dir: Path) -> Any:
    """Replace oversized leaves with a file path, recursing through containers."""
    if isinstance(value, dict):
        return {k: _summarize(v, f"{key_path}.{k}" if key_path else k, output_dir) for k, v in value.items()}
    if isinstance(value, list):
        items = [
            _summarize(v, f"{key_path}.{i}" if key_path else str(i), output_dir) if isinstance(v, dict) else v
            for i, v in enumerate(value)
        ]
        rendered = json.dumps(items, default=str)
        if len(rendered) <= INLINE_CHAR_LIMIT:
            return items
        # Many small items: nothing left to spill individually, so the list goes out whole.
        path = _spill_path(output_dir, key_path, "json")
        path.write_text(rendered)
        return {"_saved_to": str(path), "_kind": "json", "_items": len(value), "_bytes": len(rendered)}
    if isinstance(value, str) and len(value) > INLINE_CHAR_LIMIT:
        path = _spill_path(output_dir, key_path, "txt")
        path.write_text(value)
        return {"_saved_to": str(path), "_kind": "text", "_bytes": len(value)}
    return value


def _setup_errors(device: Device) -> tuple[type[Exception], ...]:
    """Exceptions meaning "this cannot run as configured", rather than "the tool failed"."""
    if device == "local":
        # Nothing to configure: a failure here is the tool's own, reported as such.
        return ()
    if device == "proto":
        # No key configured, or a key the server will not accept.
        return (NotImplementedError, PermissionError)
    from proto_tools.modal.client import ModalDispatchError

    return (ModalDispatchError,)


def _dispatch(
    device: Device,
    tool_key: str,
    payload: Any,
    cfg: Any,
    *,
    environment: str | None = None,
    client: Any | None = None,
    progress_partition: str | None = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Any, Device]:
    """Route one call to the backend ``device`` names, and report where it ran.

    The server is a stdio process on the caller's own machine, so "run it here" is always an
    option and is sometimes the only correct one. Which tools that covers is a property of the
    tool rather than a choice for the caller, so the answer comes from the registry and the config,
    the same three questions the tool wrapper asks for a direct call.

    Returns:
        tuple[Any, Device]: The tool's output, and the device it actually ran on. The two differ
            whenever a tool that cannot run remotely was asked for on a remote device.
    """
    from proto_tools.tools import ToolRegistry

    spec = ToolRegistry.get(tool_key)
    if device == "local":
        return spec.function(payload, cfg), "local"

    # A local_cpu tool has no GPU and no environment, so a deployment is optional. The wrapper
    # decides between dispatching and running here on whether one exists; ``runs_in_process`` asks
    # the same question, so the reported device matches where the call actually went.
    if spec.local_cpu:
        ran_on: Device = "local" if runs_in_process(tool_key) else device
        return spec.function(payload, cfg), ran_on
    # A search mode whose implementation is an HTTP call is answered from here. Otherwise the
    # server would look for a deployment that such a tool has no reason to have.
    if (reason := cfg.local_execution_reason(device)) is not None:
        logger.info("Tool %s: %s Running in this process instead.", tool_key, reason)
        cfg.device = "cpu"
        return spec.function(payload, cfg), "local"
    # Declared undeployable. Its inputs live on this machine, which is where the server is, so
    # running it here answers the call instead of naming a deployment that will never exist.
    if spec.local_only is not None:
        logger.info("Tool %s: %s Running in this process instead.", tool_key, spec.local_only)
        cfg.device = "cpu"
        return spec.function(payload, cfg), "local"

    if device == "proto":
        from proto_tools.proto import dispatch_to_proto

        return dispatch_to_proto(tool_key, payload, cfg), device
    from proto_tools.modal.client import dispatch_to_modal

    return (
        dispatch_to_modal(
            tool_key,
            payload,
            cfg,
            environment=environment,
            client=client,
            progress_partition=progress_partition,
            on_record=on_record,
        ),
        device,
    )


def _unavailable(device: Device, tool_key: str, error: str) -> dict[str, Any]:
    """Explain a setup failure in terms the caller can act on, if anyone can.

    When using ``modal``, the remedy is a deployment the user can perform, and the
    message therefore carries the command and ``needs_human``. When using ``proto``,
    the catalogue is fixed: no such command exists, and offering one would direct
    the caller toward something impossible.
    """
    if device == "modal":
        return {"error": error, "needs_human": True}

    entry = _hosted_catalogue().get(tool_key, {})
    reason = entry.get("unhosted_reason") or error
    return {
        "error": f"{tool_key} is not available on Proto: {reason}",
        "needs_human": False,
        "fix": (
            "Proto's catalogue is fixed and cannot be deployed to. Run this tool in your own "
            "Modal workspace with device='modal', or ask Proto to host it."
        ),
    }


def run_tool(
    tool_key: str,
    inputs: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    output_dir: str | None = None,
    use_example: bool = False,
    device: Device = "modal",
    *,
    environment: str | None = None,
    client: Any | None = None,
    progress_partition: str | None = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a tool and return its result, with large fields written to disk.

    Blocks until the tool finishes. Most tools return in seconds once a
    container is warm, but a few are genuinely long-running — see the tool
    description before calling.

    ``use_example=True`` runs the tool's canonical example input, so a caller
    can exercise a tool without materialising inputs that may be very large.

    A structure input takes a file path or an http(s) URL where the schema shows an object,
    so chaining one tool's output file into the next never reads it into the call. This is not
    uniform across bulky inputs: an MSA takes its content, and a path is rejected.

    ``on_record`` receives the worker's progress records as they arrive, for a caller that has
    somewhere to put them. Without one the call is silent until it returns, which is what a
    terminal-less caller would otherwise get for a fold that takes minutes.
    """
    from proto_tools.tools import ToolRegistry

    try:
        spec = ToolRegistry.get(tool_key)
    except (ValueError, KeyError):
        return _unknown_key(tool_key)
    if use_example:
        example = ToolRegistry.get_example_input(tool_key)
        if example is None:
            return {"ok": False, "error": f"{tool_key!r} declares no example input; pass inputs explicitly."}
        inputs = example.model_dump(mode="json")
    elif inputs is None:
        return {"ok": False, "error": "provide inputs, or pass use_example=True to run the canonical example."}

    try:
        payload = spec.input_model(**inputs)
        cfg = spec.config_model(**(config or {}))
    except Exception as exc:
        return {
            "ok": False,
            "error": f"invalid arguments: {exc}",
            "hint": f"Call get_tool_schema({tool_key!r}) for the expected shape.",
        }

    if not isinstance(cfg, BaseConfig):
        return {"ok": False, "error": f"config model is {type(cfg).__name__}, not a BaseConfig"}
    cfg.device = device if is_remote(device) else "cpu"
    # The registry's wrapper asks this before dispatching, but this path does not go through it.
    # Without the check a setting the device cannot honour -- a custom checkpoint on Proto, a
    # local database on any remote worker -- reaches a container and fails there instead.
    #
    # Asked before _dispatch, where the wrapper asks it after local_only. The order is visible
    # only for a tool that is both local_only and refuses the device by config, which none is
    # today: such a tool would be refused here and run in-process by a direct call.
    if is_remote(device) and (reason := cfg.remote_unsupported_reason(device)) is not None:
        return {"ok": False, "error": reason, "not_supported_on": device}
    try:
        result, ran_on = _dispatch(
            device,
            tool_key,
            payload,
            cfg,
            environment=environment,
            client=client,
            progress_partition=progress_partition,
            on_record=on_record,
        )
    except _setup_errors(device) as exc:
        return {"ok": False, **_unavailable(device, tool_key, str(exc))}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    target = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    body = _summarize(result.model_dump(mode="json"), "", target)
    saved = [v["_saved_to"] for v in _walk_saved(body)]
    # ``ran_on`` because the two can differ: a tool that cannot run remotely is answered here even
    # in a Modal session, and the caller should not have to infer that from the timing.
    answer = {"ok": True, "tool_key": tool_key, "ran_on": ran_on, "result": body, "saved_files": saved}
    if drift := drift_for(tool_key, ran_on, environment=environment, client=client):
        answer["warnings"] = drift
    return answer


def drift_for(tool_key: str, ran_on: Device, *, environment: str | None = None, client: Any | None = None) -> list[str]:
    """Return drift warnings for a call that went to Modal, empty for one that did not.

    The dispatch path warns about drift too, but through ``warnings.warn`` and only once per
    process. That is right for a session on someone's laptop and wrong everywhere else: an MCP
    caller never sees the process's warnings, and on a server the first caller to reach a stale
    deployment consumes the warning for everyone after them. Returning it makes drift part of the
    answer, so every caller who is affected is told.

    Never raises -- :func:`drift_warnings` swallows its own failures, and a check that broke a
    successful run would be worse than a missed warning.
    """
    if ran_on != "modal":
        return []
    from proto_tools.modal.app import resolve_environment
    from proto_tools.modal.fingerprint import drift_warnings

    try:
        service, _method = _registry()[tool_key]
    except KeyError:
        return []
    # Resolved, because the dispatch resolved it too: a caller who named no environment ran in
    # proto-env, and reading the manifest from the ambient one compares against a deployment they
    # never called. Unresolved, this reports nothing in exactly the default configuration.
    return drift_warnings(tool_key, service, environment=resolve_environment(environment), client=client)


def _walk_saved(node: Any) -> list[dict[str, Any]]:
    """Collect the placeholder dicts written by :func:`_summarize`."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "_saved_to" in node:
            return [node]
        for v in node.values():
            found += _walk_saved(v)
    elif isinstance(node, list):
        for v in node:
            found += _walk_saved(v)
    return found


async def deploy_tool(
    tool_key: str,
    environment: str,
    report: Callable[[str], Coroutine[Any, Any, None]],
    *,
    tokens: ModalTokens | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Deploy the Modal app serving one tool, after the caller has approved the spend.

    Approval is the caller's to obtain -- this runs the deploy and reports phases
    through ``report``. One app per call, and the environment is required rather
    than taken from the ambient one, so a deploy cannot land somewhere the caller
    did not name.

    Args:
        tool_key (str): Tool whose serving app to deploy.
        environment (str): Modal environment to deploy into. Resolved the same way a dispatch
            resolves it, so a deploy and the call that follows it name the same place.
        report (Callable[[str], Coroutine[Any, Any, None]]): Awaited with each build phase.
        tokens (ModalTokens | None): Deploy into the workspace these tokens name rather than this
            process's own. A server deploying for a caller passes theirs; a local session leaves
            this unset, because its workspace is already the caller's.
        client (Any | None): Modal client for recording fingerprints, opened from ``tokens`` when
            not given.

    Returns:
        dict[str, Any]: ``ok`` plus the app deployed, or an error carrying the end of the build
            output, since a caller who is not on this machine cannot read the log themselves.
    """
    import asyncio
    import contextlib
    import tempfile

    from proto_tools.modal.app import resolve_environment
    from proto_tools.modal.deploy import deploy_app

    try:
        app = _app_for(tool_key)
    except KeyError:
        return {"ok": False, "error": f"{tool_key!r} is not a tool this deployment serves."}

    # ``modal deploy`` omits --env entirely for an empty or absent name, which lands the app in
    # whichever environment the tokens make active. A dispatch resolves to proto-env instead, so
    # unresolved the caller watches a deploy succeed and the next call miss it.
    environment = resolve_environment(environment)

    loop = asyncio.get_running_loop()

    def emit(phase: str) -> None:
        # Called from the reader thread; hop back to the loop to send the notification. A caller
        # who disconnected mid-build leaves a closed loop, and raising here would abandon the
        # subprocess unwaited rather than let the deploy they paid for finish.
        with contextlib.suppress(RuntimeError):
            asyncio.run_coroutine_threadsafe(report(phase), loop)

    await report(f"deploying {app} to {environment}")
    # A directory per deploy, so concurrent callers deploying the same app do not share a log file.
    with tempfile.TemporaryDirectory(prefix="proto-tools-deploy-log-") as logs:
        ok = await asyncio.to_thread(
            deploy_app, app, environment, emit, tokens=tokens, client=client, log_dir=Path(logs)
        )
        if not ok:
            return {"ok": False, "app": app, "error": "deploy failed", "build_output": _log_tail(Path(logs))}
    return {"ok": True, "app": app, "environment": environment, "tool_key": tool_key}


#: Lines of build output returned when a deploy fails. Enough for the traceback Modal ends on,
#: short enough not to bury the rest of the response.
_LOG_TAIL_LINES = 40


def _log_tail(logs: Path) -> str:
    """Return the end of a deploy log, stripped of the frames that redraw a build tree.

    A build renders an animated tree, so the raw log is mostly the same lines written over and
    over. Reading it back means dropping those, or the tail is entirely spinner frames.

    The colour codes and the spinner glyph go too. ``for_display`` keeps both, because in a
    terminal they are what makes the build readable and show it is alive. Nothing here is watching
    it animate, so they are only noise around the message an agent has to read.
    """
    from proto_tools.modal.deploy import _ANSI_ESCAPE, _SPINNER_FRAME, RecentLines, for_display

    lines: list[str] = []
    for log in sorted(logs.glob("deploy.*.log")):
        try:
            raw = log.read_text(errors="replace")
        except OSError:
            continue
        seen = RecentLines()
        shown = (for_display(line + "\n", seen) for line in raw.splitlines())
        plain = (_ANSI_ESCAPE.sub("", text) for text in shown if text is not None)
        lines += [_SPINNER_FRAME.sub("", text).rstrip() for text in plain]
    return "\n".join(line for line in lines[-_LOG_TAIL_LINES:] if line)

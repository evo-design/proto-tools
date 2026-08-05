"""Deployment of Modal apps, and the command line interface that drives it."""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from proto_tools.modal.app import DEFAULT_ENVIRONMENT, resolve_environment
from proto_tools.modal.manifest import APP_BUCKETS, SERVICE_TO_MODULE, app_name_for_slug, app_slug, module_name

USAGE = """Deploy proto-tools apps to your own Modal workspace.

Each tool is its own Modal app, so you deploy only what you need.

    proto-tools deploy --list                    what is available
    proto-tools deploy --apps tmalign            deploy one
    proto-tools deploy --apps esm2 --test        deploy and smoke-test
    proto-tools deploy --status                  what is deployed now
    proto-tools deploy --apps all                deploy everything

Apps go to the 'proto-env' environment unless --env names another one, or
MODAL_ENVIRONMENT is set. Create it with: modal environment create proto-env
"""


def render_entrypoint(app_name: str, services: list[str]) -> str:
    """Render the module ``modal deploy`` runs for one app.

    Importing a service module fires its ``@app.cls`` decorator, which is what
    binds the class to the app being deployed.
    """
    imports = "\n".join(
        f"from {SERVICE_TO_MODULE[service]} import {service}  # noqa: F401" for service in sorted(services)
    )
    return (
        f'"""Deploy entrypoint for Modal app ``{app_name}``."""\n'
        f"\n"
        f"from proto_tools.modal.app import get_app\n"
        f"{imports}\n"
        f"\n"
        f'app = get_app("{app_name}")\n'
    )


def _logs_dir() -> Path:
    """Return the directory for per-app deploy logs.

    Anchored to the working directory: a pip install has no repository to
    write into, and running from a clone puts logs in the same ``logs/`` as
    before.
    """
    return Path.cwd() / "logs"


def _display_path(path: Path) -> str:
    """Render a path relative to the working directory when it sits below it."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def modal_command(entrypoint: Path, environment: str | None) -> list[str]:
    """Build the ``modal deploy`` command for one entrypoint.

    Modal is invoked through *this* interpreter rather than whichever ``modal``
    comes first on PATH. The entrypoint imports ``proto_tools.modal``, which only
    resolves in the environment proto-tools is installed into; a bare ``modal``
    can belong to a different environment, and the deploy then fails on an
    import error that says nothing about why.
    """
    cmd = [sys.executable, "-m", "modal", "deploy", str(entrypoint)]
    if environment:
        cmd.extend(["--env", environment])
    return cmd


# Cursor movement and erase-line, which a live display uses to redraw in place. Colour (``m``)
# is deliberately left alone. IPython runs ``!`` commands under a pseudo-terminal, so Modal
# renders its animated build tree even in a notebook, where nothing interprets these and every
# frame lands as new lines instead of overwriting the last.
_CURSOR_ESCAPE = re.compile(r"\x1b\[\??[0-9;]*[A-HJKSTfhl]")

# Braille cells (U+2800-U+28FF) are the animation frames of a spinner. Every frame carries a
# different glyph, so the same status line differs byte for byte on each refresh and would defeat
# a plain repeat check.
_SPINNER_FRAME = re.compile(r"[⠀-⣿]+\s*")

# Every escape, colour included. :func:`describe_progress` matches on a line's opening words, so
# a leading colour code hides the phase from it.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class RecentLines:
    """Bounded memory of recently emitted lines, for suppressing a live display's redraws.

    A window rather than a full history, for two reasons. It cannot grow without limit on a
    build whose progress lines are each unique (``Uploaded 141/205 files``, and so on for
    thousands). And a redraw repeats within a few refreshes, so a line recurring long
    afterwards is usually a genuine second occurrence worth printing.
    """

    def __init__(self, limit: int = 100) -> None:
        """Remember at most ``limit`` lines, evicting the oldest.

        ``limit`` has a floor worth respecting: it must exceed the number of distinct lines in
        one redraw frame, or every line is evicted before the frame comes round again and none
        is recognised as a repeat. Against a 40-line build tree, 50 and above suppress it and 20
        lets all 12,300 lines through.
        """
        self._order: deque[str] = deque()
        self._keys: set[str] = set()
        self._limit = limit

    def add_if_new(self, key: str) -> bool:
        """Record ``key`` and report whether it had not been seen within the window."""
        if key in self._keys:
            return False
        self._keys.add(key)
        self._order.append(key)
        if len(self._order) > self._limit:
            self._keys.discard(self._order.popleft())
        return True


def for_display(line: str, seen: RecentLines) -> str | None:
    """Return the line to print for one line of Modal output, or ``None`` to drop it.

    Strips the redraw escapes, then suppresses anything printed recently for this app. A live
    display repeats its whole frame on every refresh, and the repeats carry nothing new. The
    spinner glyph is ignored when deciding what counts as a repeat, so an animated status
    survives as a single line rather than one line per frame.

    The log file still receives the raw stream, so nothing dropped here is lost.
    """
    text = _CURSOR_ESCAPE.sub("", line).rstrip()
    if not text.strip():
        return None
    return text if seen.add_if_new(_SPINNER_FRAME.sub("", text)) else None


def describe_progress(line: str) -> str | None:
    """Summarise one line of Modal's build output, or ``None`` if it says nothing new.

    A deploy emits hundreds of lines, most of them pip output. A caller showing a
    status somewhere it cannot scroll -- an agent's progress indicator -- wants the
    few that mark a change of phase, not the stream.
    """
    text = line.strip()
    if text.startswith("=> Step"):
        return text.split(":", 1)[-1].strip()[:80] or "building image"
    if text.startswith("Building image"):
        return "building image"
    if "Running function" in text or "run_function" in text:
        return "running warmup (this executes the tool once)"
    if text.startswith(("Creating objects", "✓ Created objects")):
        return "creating Modal objects"
    if "App deployed" in text:
        return "deployed"
    return None


def deploy_app(
    app_name: str,
    environment: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    verbose: bool = False,
) -> bool:
    """Deploy one app, rendering its entrypoint fresh from the manifest.

    ``modal deploy`` wants a file, but an entrypoint is only two absolute imports
    and a :func:`get_app` call -- nothing reads its own location. Writing it to a
    scratch directory therefore deploys exactly what the manifest currently says,
    and adding a service of your own needs no generated file in your tree.

    Args:
        app_name (str): Modal app to deploy.
        environment (str | None): Modal environment, or ``None`` for the ambient one.
        on_progress (Callable[[str], None] | None): Called with a short phase
            description as the build moves through it. A deploy takes minutes, so a
            caller with no other output needs this to show it has not hung.
        verbose (bool): Stream every line of Modal's output rather than one line per phase.

    Returns:
        bool: Whether the deploy succeeded.
    """
    logs_dir = _logs_dir()
    logs_dir.mkdir(exist_ok=True, parents=True)
    log_file = logs_dir / f"deploy.{app_slug(app_name)}.log"

    with tempfile.TemporaryDirectory(prefix="proto-tools-deploy-") as scratch:
        entrypoint = Path(scratch) / f"{module_name(app_name)}.py"
        entrypoint.write_text(render_entrypoint(app_name, APP_BUCKETS[app_name]))
        deployed = _run_modal_deploy(app_name, entrypoint, environment, log_file, on_progress, verbose)

    # Recorded here rather than by the caller, so every route that deploys gets drift detection.
    # An absent manifest reads as "nothing to report", so a path that skipped this would leave the
    # app permanently exempt from the check rather than visibly missing it.
    if deployed:
        record_fingerprints([app_name], environment)
    return deployed


def _run_modal_deploy(
    app_name: str,
    entrypoint: Path,
    environment: str | None,
    log_file: Path,
    on_progress: Callable[[str], None] | None = None,
    verbose: bool = False,
) -> bool:
    """Run ``modal deploy`` for one entrypoint, reporting its progress prefixed by app.

    Prints a line per phase by default. Modal renders an animated build tree, and IPython runs
    ``!`` commands under a pseudo-terminal, so the raw stream reaches a notebook as thousands of
    redraw frames. ``verbose`` streams every line instead; either way the log file has it all.
    """
    cmd = modal_command(entrypoint, environment)
    seen = RecentLines()
    last_phase: str | None = None
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout or ():
                f.write(line)  # the log keeps the raw stream, redraw frames and all
                phase = describe_progress(_ANSI_ESCAPE.sub("", line))
                if verbose:
                    # Prefix per-app stdout so parallel deploy output stays greppable.
                    if (shown := for_display(line, seen)) is not None:
                        sys.stdout.write(f"[{app_slug(app_name)}] {shown}\n")
                        sys.stdout.flush()
                elif phase and phase != last_phase:
                    sys.stdout.write(f"[{app_slug(app_name)}] {phase}\n")
                    sys.stdout.flush()
                    last_phase = phase
                if on_progress and phase:
                    on_progress(phase)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        return True
    except subprocess.CalledProcessError:
        print(f"[{app_slug(app_name)}] deploy failed — full log at {_display_path(log_file)}")
        return False


def deploy_apps(
    app_names: list[str], environment: str | None = None, max_parallel: int = 4, verbose: bool = False
) -> dict[str, bool]:
    """Deploy apps in parallel. Returns ``{app_name: success}``; one failure never blocks the rest."""
    results: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(deploy_app, name, environment, None, verbose): name for name in app_names}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results


def record_fingerprints(app_names: list[str], environment: str | None) -> None:
    """Record what each app was built from, so clients can detect drift later.

    Runs from the same source that was just deployed, so the recorded values
    describe what actually shipped. Best-effort: a failure here leaves a working
    deployment with no drift detection, which beats failing the deploy.
    """
    from proto_tools.modal.fingerprint import write_manifest

    for app_name in sorted(app_names):
        for service in APP_BUCKETS[app_name]:
            try:
                write_manifest(service, environment)
            except Exception as exc:
                # Only the failure is worth saying. Recording fingerprints is bookkeeping the
                # caller did not ask for, and announcing it pushes the deploy result off the end.
                print(f"  ⚠️  {app_slug(app_name)}: could not record fingerprints ({exc})")


def known_environments() -> list[str] | None:
    """Return the environment names in the active workspace, or ``None`` if they cannot be read.

    ``None`` is not "no environments" -- it means the lookup itself failed, so a caller must
    not treat a name as absent on the strength of it.
    """
    try:
        from modal.environments import list_environments

        return [env.name for env in list_environments()]
    except Exception:  # a Modal API change or an auth failure must not break reporting
        return None


def app_status(app_names: list[str]) -> dict[str, tuple[str, str]]:
    """Resolve each app in the active workspace. Returns ``{slug: (state, detail)}``.

    ``state`` is ``"deployed"``, ``"missing"``, or ``"error"``. A lookup failure that is not
    a plain absence is kept separate rather than folded into "not deployed", because the two
    call for different actions.
    """
    import modal

    states: dict[str, tuple[str, str]] = {}
    for app_name in app_names:
        service = APP_BUCKETS[app_name][0]
        try:
            modal.Cls.from_name(app_name, service).hydrate()
            states[app_slug(app_name)] = ("deployed", "")
        except modal.exception.NotFoundError:
            states[app_slug(app_name)] = ("missing", "")
        except modal.exception.Error as exc:
            states[app_slug(app_name)] = ("error", f"{type(exc).__name__}: {exc}")
    return states


def show_status(app_names: list[str], verbose: bool = False, environment: str | None = None) -> None:
    """Report which apps currently resolve in the active workspace.

    Summarised by default: what is deployed is the short list worth reading, and what is not
    is usually most of the catalogue, so it collapses to a count. ``verbose`` prints every app.

    An environment that does not exist is reported as such. Modal raises the same
    ``NotFoundError`` for a missing app and a missing environment, so without this check every
    app reads as "not deployed" and the real problem stays hidden.
    """
    if environment and (existing := known_environments()) is not None and environment not in existing:
        print(f"  ⚠️  environment {environment!r} does not exist in this workspace.")
        print(f"      Existing environments: {', '.join(existing)}")
        print(f"      Create it with: modal environment create {environment}")
        return

    states = app_status(app_names)
    deployed = sorted(s for s, (state, _) in states.items() if state == "deployed")
    missing = sorted(s for s, (state, _) in states.items() if state == "missing")
    errored = sorted((s, detail) for s, (state, detail) in states.items() if state == "error")

    if verbose:
        for slug in sorted(states):
            state, detail = states[slug]
            mark = {"deployed": "✅", "missing": "⬜", "error": "⚠️ "}[state]
            print(f"  {mark} {slug} — {detail or state.replace('missing', 'not deployed')}")
        return

    print(f"  ✅ deployed ({len(deployed)})")
    for row in _in_columns(deployed):
        print(f"       {row}")
    if not deployed:
        print("       none")
    if missing:
        print(f"  ⬜ not deployed ({len(missing)}) — list them with --verbose")
    for slug, detail in errored:
        print(f"  ⚠️  {slug} — {detail}")


def _in_columns(slugs: list[str], per_row: int = 4) -> list[str]:
    """Lay slugs out in fixed-width columns, so a long list stays scannable.

    Width follows the longest slug rather than a constant, since a slug wider than the column
    would otherwise run into its neighbour.
    """
    width = max((len(slug) for slug in slugs), default=0) + 2
    return [
        "".join(slug.ljust(width) for slug in slugs[i : i + per_row]).rstrip() for i in range(0, len(slugs), per_row)
    ]


def resolve_targets(spec: str | None) -> list[str]:
    """Resolve an ``--apps`` value to full app names. ``None`` or ``all`` means everything.

    Callers that spend money must require an explicit value — see ``main``.
    ``None`` meaning "everything" is only safe for read-only reporting.
    """
    if spec is None or spec.strip().lower() == "all":
        return sorted(APP_BUCKETS)
    return [app_name_for_slug(s.strip()) for s in spec.split(",") if s.strip()]


def apps_to_smoke_test(targets: list[str], results: dict[str, bool] | None) -> tuple[list[str], list[str]]:
    """Split targets into the apps worth smoke-testing and those to skip.

    A failed deploy leaves the *previous* deployment running, so testing it
    exercises code that never shipped and reports success for a run that
    failed. Only apps that deployed are tested.

    Args:
        targets (list[str]): Apps the user asked for.
        results (dict[str, bool] | None): Per-app deploy outcome, or ``None`` when no deploy ran — with
            ``--skip-deploy`` the intent is to test what is already deployed.

    Returns:
        tuple[list[str], list[str]]: ``(testable, skipped)``.
    """
    if results is None:
        return sorted(targets), []
    return sorted(t for t in targets if results.get(t)), sorted(t for t in targets if not results.get(t))


def require_proto_tools() -> None:
    """Fail early, with the fix, when proto-tools is not installed.

    Deploying imports service modules that need it. Without this the failure
    surfaces as a bare ImportError from inside a ``modal deploy`` subprocess,
    long after the command looked like it was working.

    Raises:
        SystemExit: If proto-tools cannot be imported.
    """
    if importlib.util.find_spec("proto_tools") is not None:
        return
    raise SystemExit(
        "proto-tools is not installed, and deploying needs it.\n"
        "  It ships as a dependency, so this usually means an install with --no-deps.\n"
        "  Working from a clone:  pip install -e ./proto-tools"
    )


def require_extra_source() -> None:
    """Fail early, on screen, when an extra source directory cannot be mounted.

    Resolved during service-module import, inside a ``modal deploy`` subprocess whose output is
    filtered down to recognised phase lines. Without this the reason reaches the log file only,
    once per app, and the console shows a bare deploy failure.

    Raises:
        SystemExit: If :data:`~proto_tools.modal.base_images.EXTRA_SOURCE_ENV` names a path that
            is not a mountable directory.
    """
    from proto_tools.modal.base_images import extra_source_dirs

    try:
        extra_source_dirs()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """Build the argument parser, named for however it was invoked."""
    parser = argparse.ArgumentParser(prog=prog, description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--env", type=str, default=None, help=f"Modal environment to deploy into (default: {DEFAULT_ENVIRONMENT})"
    )
    parser.add_argument(
        "--env-default",
        action="store_true",
        help="Deploy into your Modal profile's active environment instead of naming one with --env",
    )
    parser.add_argument(
        "--apps",
        type=str,
        default=None,
        help="Comma-separated app slugs to deploy or test. Required for both; pass 'all' to opt into every app.",
    )
    parser.add_argument("--skip-deploy", action="store_true", help="Skip deploy; useful with --test or --status")
    parser.add_argument("--test", action="store_true", help="Smoke-test the targeted apps after deploying")
    parser.add_argument("--status", action="store_true", help="Report which targeted apps are deployed, then exit")
    parser.add_argument(
        "--verbose", action="store_true", help="Stream Modal's full output; with --status, list every app"
    )
    parser.add_argument(
        "--create-env",
        action="store_true",
        help=f"Create the Modal environment named by --env (default: {DEFAULT_ENVIRONMENT}), then exit",
    )
    parser.add_argument("--list", action="store_true", help="List available apps and their services, then exit")
    parser.add_argument("--max-parallel", type=int, default=4, help="Max parallel app deploys (default: 4)")
    return parser


def create_env(name: str) -> int:
    """Create one Modal environment, reporting a name that already exists as success.

    Deliberately its own command rather than something a deploy does implicitly. Creating an
    environment changes the shape of someone's Modal workspace, which is theirs to decide;
    proto-tools only names the default and offers the command.

    Args:
        name (str): Environment to create.

    Returns:
        int: Process exit code.
    """
    from proto_tools.modal.app import environment_exists

    if environment_exists(name) is True:
        print(f"Modal environment {name!r} already exists; nothing to do.")
        return 0
    try:
        from modal.environments import create_environment

        create_environment(name)
    except Exception as exc:
        print(f"Could not create Modal environment {name!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Created Modal environment {name!r}. Deploy into it with:\n\n    proto-tools deploy --apps <slug>\n")
    return 0


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    """Parse arguments, deploy and/or smoke-test, and return a process exit code."""
    # Python block-buffers stdout when it is a pipe rather than a terminal, which
    # hides phase headers until the process exits — long builds then look hung to
    # anyone redirecting output or running this unattended.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

    parser = build_parser(prog)
    args = parser.parse_args(argv)

    if args.list:
        print(f"{len(APP_BUCKETS)} app(s):")
        for app_name, services in sorted(APP_BUCKETS.items()):
            print(f"  {app_slug(app_name):<14} {app_name:<28} {', '.join(services)}")
        return 0

    if args.env and args.env_default:
        parser.error("pass either --env <name> or --env-default, not both")

    # Deploying overwrites any app of the same name in the target environment, so an omitted
    # flag resolves to DEFAULT_ENVIRONMENT rather than the Modal profile's active one. The
    # profile ordinarily points at production, where a same-named app may belong to another
    # project entirely. ``--env-default`` is the deliberate way to accept it anyway.
    args.env = None if args.env_default else resolve_environment(args.env)

    # Answered before the --apps guard below, because creating an environment targets no app and
    # spends nothing. It is also the fix that guard's own error cannot offer.
    if args.create_env:
        if not args.env:
            parser.error("--create-env needs a name; drop --env-default or pass --env <name>")
        return create_env(args.env)

    # Both deploying and smoke-testing cost money — a build runs a real warmup
    # inference, and a smoke test runs the tool for real. Neither may default to
    # every app: naming the targets has to be deliberate.
    spends_money = not args.status and (not args.skip_deploy or args.test)
    if spends_money and not args.apps:
        parser.error(
            "--apps is required.\n"
            "  Deploying builds images (each ends in a real warmup inference) and\n"
            "  smoke-testing runs the tools, so both cost money on your Modal account.\n"
            "  Name what you want:   --apps tmalign        (or a comma-separated list)\n"
            "  See the options with: --list\n"
            f"  To build every app deliberately: --apps all   ({len(APP_BUCKETS)} images)"
        )

    if not args.status:
        require_proto_tools()
        require_extra_source()

    if args.env:
        os.environ["MODAL_ENVIRONMENT"] = args.env
        # A deploy into an environment that does not exist fails deep inside Modal with a
        # message about the app. Asked here, the answer names the missing environment and the
        # command that creates it, which is the one-time setup a new workspace has not run.
        from proto_tools.modal.app import environment_exists

        if environment_exists(args.env) is False:
            print(
                f"Modal environment {args.env!r} has not been created in this workspace.\n"
                f"Nothing can be deployed to it until it exists. Create it with:\n\n"
                f"    proto-tools deploy --create-env --env {args.env}\n",
                file=sys.stderr,
            )
            return 1

    try:
        targets = resolve_targets(args.apps)
    except KeyError as exc:
        parser.error(str(exc))

    if args.status:
        show_status(targets, verbose=args.verbose, environment=args.env)
        return 0

    deploy_ok = True
    results: dict[str, bool] | None = None
    if not args.skip_deploy:
        where = args.env or "your profile's active environment"
        if len(targets) == len(APP_BUCKETS) and len(targets) > 1:
            print(f"⚠️  Building all {len(targets)} apps. Each runs a warmup inference — GPU apps cost GPU time.\n")
        print(f"Deploying {len(targets)} app(s) to [{where}]: {', '.join(app_slug(t) for t in targets)}\n")
        results = deploy_apps(targets, environment=args.env, max_parallel=args.max_parallel, verbose=args.verbose)
        deploy_ok = all(results.values())
        print("\nDeploy results:")
        for name in sorted(results):
            print(f"  {'✅' if results[name] else '❌'} {app_slug(name)}")

    passed = failed = 0
    if args.test:
        from proto_tools.modal.smoke import smoke_app

        testable, skipped = apps_to_smoke_test(targets, results)
        print("\nSmoke tests:")
        for app_name in skipped:
            print(f"  ⏭️  {app_slug(app_name)} — not tested, deploy failed")
        for app_name in testable:
            p, f = smoke_app(app_name)
            passed += p
            failed += f
        if passed or failed:
            print(f"\n{'✅' if not failed else '❌'} {passed}/{passed + failed} smoke tests passed")

    return 0 if deploy_ok and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(prog="proto-tools deploy"))

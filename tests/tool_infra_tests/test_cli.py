"""tests/tool_infra_tests/test_cli.py.

Smoke tests for the ``proto-tools`` CLI entry point. Each verb maps to a
``ToolRegistry`` classmethod, so coverage here is intentionally thin —
just enough to catch breakage in argparse wiring, exit codes, and the
text-vs-JSON output toggle. Behavioral coverage of the underlying
functions lives in ``test_tool_docs.py``.

All tests invoke the CLI via the in-process ``main()`` rather than a
subprocess so they stay fast (no Python startup cost per call).
"""

from __future__ import annotations

import io
import json
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from proto_tools import cli as cli_mod
from proto_tools.cli import main
from proto_tools.utils.remote_execution import ExistingRemoteBatchNode, ExistingRemoteCommandNode, RemoteNode


def _run(*argv: str) -> tuple[int, str, str]:
    """Invoke ``main(argv)`` and capture (exit_code, stdout, stderr)."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


def _init_git_repo(path: Path) -> Path:
    """Create a minimal git checkout for deploy CLI tests."""
    path.mkdir(parents=True)
    subprocess.run(["git", "-C", str(path), "init"], check=True, text=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Proto Tests"], check=True)
    (path / "tracked.txt").write_text("clean\n")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "initial"], check=True, text=True, capture_output=True)
    return path


# ── Discovery verbs ─────────────────────────────────────────────────────────


def test_list_default_outputs_text() -> None:
    code, out, _ = _run("list")
    assert code == 0
    assert "esm2-embedding" in out
    assert "[masked_models]" in out


def test_list_category_filter() -> None:
    code, out, _ = _run("list", "--category", "masked_models")
    assert code == 0
    lines = [line for line in out.splitlines() if line.strip()]
    assert all("[masked_models]" in line for line in lines)
    assert any(line.startswith("esm2-embedding") for line in lines)


def test_list_json_payload_is_valid() -> None:
    code, out, _ = _run("list", "--category", "masked_models", "--json")
    assert code == 0
    payload = json.loads(out)
    keys = [item["key"] for item in payload]
    assert "esm2-embedding" in keys


def test_categories_outputs_known_value() -> None:
    code, out, _ = _run("categories")
    assert code == 0
    assert "masked_models" in out.splitlines()


def test_catalog_json_groups_by_category() -> None:
    code, out, _ = _run("catalog", "--json")
    assert code == 0
    payload = json.loads(out)
    assert "masked_models" in payload
    assert any(item["key"] == "esm2-embedding" for item in payload["masked_models"])


# ── Agent context ───────────────────────────────────────────────────────────


def test_agent_context_prints_primer() -> None:
    code, out, _ = _run("agent-context")
    assert code == 0
    assert "Input -> Config -> run_*() -> Output" in out
    assert "github.com/evo-design/proto-tools/tree/main/notes" in out


# ── Remote execution verbs ─────────────────────────────────────────────────


def test_remote_profile_list_filters_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty"),
        RemoteNode(name="h100", host="h100", work_dir="/remote/h100", scheduler="slurm", weight=3.0),
    ]
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    code, out, _ = _run("remote", "profile", "list", "--profile", "nodes.json", "--node", "h100")

    assert code == 0
    assert out.splitlines() == ["h100\th100\tslurm\tweight=3.0\tmax_concurrent_buckets=1"]


def test_remote_profile_scaffold_and_validate(tmp_path: Path) -> None:
    profile_path = tmp_path / "remote-nodes.json"

    code, out, _ = _run(
        "remote",
        "profile",
        "scaffold",
        "--output",
        str(profile_path),
        "--direct",
        "ty=/sfs/home/zengzhengpeng",
        "--direct",
        "ty3=/mnt/sfs/zengzhengpeng",
        "--slurm",
        "h100=/public/home/zengzhengpeng",
        "--slurm-weight",
        "3",
        "--slurm-max-active-jobs",
        "2",
        "--slurm-sbatch-arg=--partition=gpu",
        "--slurm-sbatch-arg=--gres=gpu:h100:1",
    )
    validate_code, validate_out, _ = _run(
        "remote",
        "profile",
        "validate",
        "--profile",
        str(profile_path),
        "--json",
    )

    payload = json.loads(profile_path.read_text())
    assert code == 0
    assert f"profile\t{profile_path}" in out
    assert [node["name"] for node in payload["nodes"]] == ["ty", "ty3", "h100"]
    assert payload["nodes"][0]["worker_device"] == "cuda:0"
    assert payload["nodes"][2]["scheduler"] == "slurm"
    assert payload["nodes"][2]["weight"] == 3.0
    assert payload["nodes"][2]["max_active_slurm_jobs"] == 2
    assert payload["nodes"][2]["sbatch_args"] == ["--partition=gpu", "--gres=gpu:h100:1"]
    assert validate_code == 0
    assert json.loads(validate_out) == {"ok": True, "node_count": 3, "issues": []}


def test_remote_profile_scaffold_ty_directs_and_h100_slurm_golden(tmp_path: Path) -> None:
    profile_path = tmp_path / "remote-nodes.json"

    code, out, _ = _run(
        "remote",
        "profile",
        "scaffold",
        "--output",
        str(profile_path),
        "--direct",
        "ty=/sfs/home/zengzhengpeng",
        "--direct",
        "ty2=/mnt/sfs/zengzhengpeng",
        "--direct",
        "ty3=/mnt/sfs/zengzhengpeng",
        "--work-name-template",
        "proto_remote_{node}",
        "--slurm",
        "h100=/public/home/zengzhengpeng",
        "--slurm-weight",
        "3",
        "--slurm-max-active-jobs",
        "2",
        "--slurm-sbatch-arg=--partition=gpu",
        "--slurm-sbatch-arg=--gres=gpu:h100:1",
    )
    validate_code, validate_out, _ = _run("remote", "profile", "validate", "--profile", str(profile_path), "--json")

    profile = json.loads(profile_path.read_text())
    nodes = {node["name"]: node for node in profile["nodes"]}
    issues = json.loads(validate_out)["issues"]
    warning_pairs = {(issue["field"], issue["node_name"]) for issue in issues}

    assert code == 0
    assert [node["name"] for node in profile["nodes"]] == ["ty", "ty2", "ty3", "h100"]
    assert nodes["ty"]["work_dir"] == "/sfs/home/zengzhengpeng/proto_remote_ty"
    assert nodes["ty2"]["work_dir"] == "/mnt/sfs/zengzhengpeng/proto_remote_ty2"
    assert nodes["ty3"]["work_dir"] == "/mnt/sfs/zengzhengpeng/proto_remote_ty3"
    assert nodes["h100"]["scheduler"] == "slurm"
    assert nodes["h100"]["weight"] == 3.0
    assert nodes["h100"]["max_active_slurm_jobs"] == 2
    assert nodes["h100"]["sbatch_args"] == ["--partition=gpu", "--gres=gpu:h100:1"]
    assert validate_code == 0
    assert warning_pairs == {("repo_dir", "ty2,ty3"), ("venv_dir", "ty2,ty3")}
    assert "warning\tty2,ty3\trepo_dir\tremote repo_dir is shared by multiple nodes" in out
    assert "warning\tty2,ty3\tvenv_dir\tremote venv_dir is shared by multiple nodes" in out


def test_remote_profile_scaffold_supports_work_name_template(tmp_path: Path) -> None:
    profile_path = tmp_path / "remote-nodes.json"

    code, _, _ = _run(
        "remote",
        "profile",
        "scaffold",
        "--output",
        str(profile_path),
        "--direct",
        "ty2=/mnt/sfs/zengzhengpeng",
        "--direct",
        "ty3=/mnt/sfs/zengzhengpeng",
        "--work-name-template",
        "proto_remote_{node}",
    )
    validate_code, validate_out, _ = _run("remote", "profile", "validate", "--profile", str(profile_path), "--json")

    payload = json.loads(profile_path.read_text())
    assert code == 0
    assert [node["work_dir"] for node in payload["nodes"]] == [
        "/mnt/sfs/zengzhengpeng/proto_remote_ty2",
        "/mnt/sfs/zengzhengpeng/proto_remote_ty3",
    ]
    assert validate_code == 0
    assert json.loads(validate_out)["ok"] is True


def test_remote_profile_validate_returns_one_for_static_errors(tmp_path: Path) -> None:
    profile_path = tmp_path / "bad-remote-nodes.json"
    profile_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"name": "ty", "host": "ty", "work_dir": "/remote/ty"},
                    {"name": "ty", "host": "ty2", "work_dir": "relative/work"},
                ]
            }
        )
    )

    code, out, _ = _run("remote", "profile", "validate", "--profile", str(profile_path), "--json")

    payload = json.loads(out)
    assert code == 1
    assert payload["ok"] is False
    assert {"level": "error", "node_name": "ty", "field": "name", "message": "duplicate node name appears 2 times"} in payload[
        "issues"
    ]


def test_remote_profile_validate_rejects_unsafe_node_name(tmp_path: Path) -> None:
    profile_path = tmp_path / "bad-remote-nodes.json"
    profile_path.write_text(json.dumps({"nodes": [{"name": "../escape", "host": "ty", "work_dir": "/remote/ty"}]}))

    code, _, err = _run("remote", "profile", "validate", "--profile", str(profile_path), "--json")

    assert code == 2
    assert "RemoteNode.name" in err


def test_remote_smoke_cli_uses_smoke_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty"),
        RemoteNode(name="h100", host="h100", work_dir="/remote/h100", scheduler="slurm"),
    ]
    calls = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    class FakeReport:
        def __init__(self) -> None:
            self.ok = True
            self.profile_issues = ()
            self.preflight = {"h100": {"ok": True}}

        def to_dict(self):
            return {
                "ok": True,
                "preflight": self.preflight,
                "manager_status": {"h100": "stopped"},
                "diagnostics": {},
                "rsync_pull_commands": [["rsync"]],
            }

    def fake_smoke(smoke_nodes, *, local_results_dir, timeout_sec):
        calls.append((smoke_nodes, local_results_dir, timeout_sec))
        return FakeReport()

    monkeypatch.setattr(cli_mod.remote_exec, "smoke_remote_nodes", fake_smoke)
    monkeypatch.setattr(cli_mod.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    code, out, _ = _run(
        "remote",
        "smoke",
        "--profile",
        "nodes.json",
        "--node",
        "h100",
        "--local-results-dir",
        str(tmp_path / "results"),
        "--timeout-sec",
        "45",
        "--json",
    )

    assert code == 0
    assert json.loads(out)["ok"] is True
    assert calls == [([nodes[1]], str(tmp_path / "results"), 45.0)]


def test_remote_existing_plan_cli_renders_existing_command_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.fa"
    input_path.write_text(">x\nAAAA\n")
    collect_dir = tmp_path / "collect"
    node = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/existing",
        command_argv=("/env/bin/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
    )
    monkeypatch.setattr(cli_mod.remote_exec, "load_existing_remote_command_nodes_json", lambda path: [node])

    code, out, _ = _run(
        "remote",
        "existing",
        "plan",
        "--profile",
        "existing.json",
        "--node",
        "ty",
        "--input",
        str(input_path),
        "--run-id",
        "run-cli",
        "--local-collect-dir",
        str(collect_dir),
        "--json",
    )

    payload = json.loads(out)
    assert code == 0
    assert payload["node_name"] == "ty"
    assert payload["remote_input_path"] == "/remote/existing/run-cli/input/input.fa"
    assert payload["remote_output_dir"] == "/remote/existing/run-cli/output"
    assert payload["commands"]["execute"][:4] == ["ssh", "ty", "bash", "-lc"]


def test_remote_existing_run_cli_uses_existing_command_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.fa"
    input_path.write_text(">x\nAAAA\n")
    collect_dir = tmp_path / "collect"
    node = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/existing",
        command_argv=("/env/bin/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
    )
    calls = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_existing_remote_command_nodes_json", lambda path: [node])

    class FakePlan:
        node_name = "ty"
        run_id = "run-cli"

        def to_dict(self):
            return {"node_name": self.node_name, "run_id": self.run_id}

    class FakeResult:
        plan = FakePlan()
        collected_files = ("scores.json",)

        def to_dict(self):
            return {"plan": self.plan.to_dict(), "collected_files": list(self.collected_files)}

    def fake_run(existing_node, *, local_input_path, run_id, local_collect_dir, timeout_sec):
        calls.append((existing_node, local_input_path, run_id, local_collect_dir, timeout_sec))
        return FakeResult()

    monkeypatch.setattr(cli_mod.remote_exec, "run_existing_remote_command", fake_run)

    code, out, _ = _run(
        "remote",
        "existing",
        "run",
        "--profile",
        "existing.json",
        "--node",
        "ty",
        "--input",
        str(input_path),
        "--run-id",
        "run-cli",
        "--local-collect-dir",
        str(collect_dir),
        "--timeout-sec",
        "12.5",
        "--json",
    )

    assert code == 0
    assert json.loads(out) == {"plan": {"node_name": "ty", "run_id": "run-cli"}, "collected_files": ["scores.json"]}
    assert calls == [(node, str(input_path), "run-cli", str(collect_dir), 12.5)]


def test_remote_existing_launch_cli_runs_manifest_backed_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One launch command should derive run paths and invoke the manager-backed API."""
    input_path = tmp_path / "random_protein.fasta"
    input_path.write_text(">one\nAAAA\n")
    run_dir = tmp_path / "af3-run"
    batch_node = ExistingRemoteBatchNode(
        manager=RemoteNode(
            name="ty",
            host="ty",
            work_dir="/remote/manager",
            worker_device="cuda:7",
        ),
        command=ExistingRemoteCommandNode(
            name="ty",
            host="ty",
            work_dir="/remote/af3",
            command_argv=("/env/python", "/scripts/007_run_af3_direct.py", "-i", "{input}", "-o", "{output}"),
            status="ready",
            software="alphafold3",
            required_artifacts=("**/*_model.cif",),
        ),
    )
    calls = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_existing_remote_batch_nodes_json", lambda path: [batch_node])

    class FakeResult:
        def to_dict(self):
            return {
                "run_id": "af3-run",
                "task_count": 1,
                "assignment": {"ty": 1},
                "summary": str(run_dir / "summary.json"),
            }

    def fake_launch(nodes, **kwargs):
        calls.append((nodes, kwargs))
        return FakeResult()

    monkeypatch.setattr(cli_mod.remote_exec, "launch_existing_remote_command_batch", fake_launch)

    code, out, _ = _run(
        "remote",
        "existing",
        "launch",
        "alphafold3",
        "--profile",
        "af3-batch.json",
        "--input",
        str(input_path),
        "--run-dir",
        str(run_dir),
        "--bucket-size",
        "1",
        "--slurm-group-size",
        "10",
        "--wait-poll-interval",
        "7",
        "--timeout-sec",
        "86400",
        "--remote-check-timeout-sec",
        "120",
        "--json",
    )

    assert code == 0
    assert json.loads(out) == {
        "run_id": "af3-run",
        "task_count": 1,
        "assignment": {"ty": 1},
        "summary": str(run_dir / "summary.json"),
    }
    assert calls == [
        (
            [batch_node],
            {
                "input_fasta": str(input_path),
                "software": "alphafold3",
                "run_dir": str(run_dir),
                "bucket_size": 1,
                "slurm_group_size": 10,
                "wait_poll_interval": 7.0,
                "timeout_sec": 86400.0,
                "remote_check_timeout_sec": 120.0,
            },
        )
    ]


def test_remote_compute_cli_resolves_profile_and_runs_manifest_backed_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The simple command should resolve the maintained AF3 profile and reuse the existing launcher."""
    input_path = tmp_path / "random_protein.fasta"
    input_path.write_text(">one\nAAAA\n")
    run_dir = tmp_path / "af3-run"
    profile_path = tmp_path / "profiles" / "af3.json"
    profile_path.parent.mkdir()
    profile_path.write_text('{"profile_schema_version": 1, "nodes": []}\n')
    registry_path = tmp_path / "runtime_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "software": {
                    "alphafold3": {
                        "aliases": ["af3"],
                        "status": "ready",
                        "profile": "profiles/af3.json",
                        "defaults": {
                            "bucket_size": 1,
                            "slurm_group_size": 10,
                            "wait_poll_interval": 5,
                            "timeout_sec": 7200,
                            "remote_check_timeout_sec": 60,
                        },
                    }
                },
            }
        )
    )
    batch_node = ExistingRemoteBatchNode(
        manager=RemoteNode(
            name="ty",
            host="ty",
            work_dir="/remote/manager",
            worker_device="cuda:7",
        ),
        command=ExistingRemoteCommandNode(
            name="ty",
            host="ty",
            work_dir="/remote/af3",
            command_argv=("/env/python", "/scripts/007_run_af3_direct.py", "-i", "{input}", "-o", "{output}"),
            status="ready",
            software="alphafold3",
            required_artifacts=("**/*_model.cif",),
        ),
    )
    calls = []
    receipts = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_existing_remote_batch_nodes_json", lambda path: [batch_node])
    monkeypatch.setattr(
        cli_mod.remote_exec,
        "write_remote_compute_receipt",
        lambda payload, path: receipts.append((payload, path)),
    )

    class FakeResult:
        def to_dict(self):
            return {
                "run_id": "af3-run",
                "task_count": 1,
                "assignment": {"ty": 1},
                "summary": str(run_dir / "summary.json"),
            }

    def fake_launch(nodes, **kwargs):
        calls.append((nodes, kwargs))
        return FakeResult()

    monkeypatch.setattr(cli_mod.remote_exec, "launch_existing_remote_command_batch", fake_launch)

    code, out, _ = _run(
        "remote",
        "compute",
        "af3",
        str(input_path),
        str(run_dir),
        "--registry",
        str(registry_path),
        "--json",
    )

    assert code == 0
    assert json.loads(out) == {
        "software": "alphafold3",
        "registry": str(registry_path.resolve()),
        "profile": str(profile_path.resolve()),
        "receipt": str(run_dir.resolve() / "compute.json"),
        "run_id": "af3-run",
        "task_count": 1,
        "assignment": {"ty": 1},
        "summary": str(run_dir / "summary.json"),
    }
    assert calls == [
        (
            [batch_node],
            {
                "input_fasta": input_path.resolve(),
                "software": "alphafold3",
                "run_dir": run_dir.resolve(),
                "bucket_size": 1,
                "slurm_group_size": 10,
                "wait_poll_interval": 5.0,
                "timeout_sec": 7200.0,
                "remote_check_timeout_sec": 60.0,
            },
        )
    ]
    assert receipts == [(json.loads(out), run_dir.resolve() / "compute.json")]


def test_remote_deploy_render_outputs_commands_without_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty", repo_dir="/remote/repo"),
        RemoteNode(name="h100", host="h100", work_dir="/remote/h100", repo_dir="/remote/repo-h100"),
    ]
    calls = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    def fake_render(node, local_repo_dir):
        calls.append((node.name, local_repo_dir))
        return [["ssh", node.host, "mkdir"], ["rsync", str(local_repo_dir), node.repo_dir]]

    class ForbiddenDeploymentClient:
        def __init__(self):
            raise AssertionError("render must not instantiate the deployment client")

    monkeypatch.setattr(cli_mod.remote_exec, "render_deploy_commands", fake_render)
    monkeypatch.setattr(cli_mod.remote_exec, "RemoteDeploymentClient", ForbiddenDeploymentClient)

    code, out, _ = _run(
        "remote",
        "deploy",
        "render",
        "--profile",
        "nodes.json",
        "--node",
        "ty",
        "--local-repo-dir",
        str(repo),
        "--json",
    )

    payload = json.loads(out)
    assert code == 0
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["local_repo_dir"] == str(repo.resolve())
    assert payload["node_count"] == 1
    assert payload["command_count"] == 2
    assert payload["nodes"][0]["name"] == "ty"
    assert [command["step"] for command in payload["nodes"][0]["commands"]] == ["mkdir", "rsync_upload"]
    assert [command["argv"] for command in payload["nodes"][0]["commands"]] == [
        ["ssh", "ty", "mkdir"],
        ["rsync", str(repo), "/remote/repo"],
    ]
    assert calls == [("ty", str(repo))]

    text_code, text_out, _ = _run(
        "remote",
        "deploy",
        "render",
        "--profile",
        "nodes.json",
        "--node",
        "ty",
        "--local-repo-dir",
        str(repo),
    )

    assert text_code == 0
    assert text_out.splitlines() == [
        "deploy-command\tty\tssh ty mkdir",
        f"deploy-command\tty\trsync {repo} /remote/repo",
    ]
    assert calls == [("ty", str(repo)), ("ty", str(repo))]


def test_remote_deploy_render_reports_dirty_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    (repo / "tracked.txt").write_text("dirty\n")
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", repo_dir="/remote/repo")
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: [node])
    monkeypatch.setattr(cli_mod.remote_exec, "render_deploy_commands", lambda node, local_repo_dir: [["ssh", "ty"]])

    code, out, _ = _run(
        "remote",
        "deploy",
        "render",
        "--profile",
        "nodes.json",
        "--local-repo-dir",
        str(repo),
        "--json",
    )

    payload = json.loads(out)
    assert code == 1
    assert payload["ok"] is False
    assert payload["command_count"] == 1
    assert payload["local_checkout"]["dirty_count"] == 1
    assert payload["blockers"] == ["local checkout has 1 tracked and 0 untracked changed paths"]


def test_remote_deploy_render_canonical_profile_reports_shared_ty2_ty3_install_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    nodes = [
        RemoteNode(
            name="ty",
            host="ty",
            work_dir="/sfs/home/zengzhengpeng/proto_remote_ty",
            repo_dir="/sfs/home/zengzhengpeng/proto-tools",
        ),
        RemoteNode(
            name="ty2",
            host="ty2",
            work_dir="/mnt/sfs/zengzhengpeng/proto_remote_ty2",
            repo_dir="/mnt/sfs/zengzhengpeng/proto-tools",
        ),
        RemoteNode(
            name="ty3",
            host="ty3",
            work_dir="/mnt/sfs/zengzhengpeng/proto_remote_ty3",
            repo_dir="/mnt/sfs/zengzhengpeng/proto-tools",
        ),
        RemoteNode(
            name="h100",
            host="h100",
            work_dir="/public/home/zengzhengpeng/proto_remote_h100",
            repo_dir="/public/home/zengzhengpeng/proto-tools",
            scheduler="slurm",
            sbatch_args=("--partition=gpu", "--gres=gpu:h100:1"),
        ),
    ]
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    code, out, _ = _run(
        "remote",
        "deploy",
        "render",
        "--profile",
        "nodes.json",
        "--local-repo-dir",
        str(repo),
        "--json",
    )

    payload = json.loads(out)
    assert code == 1
    assert payload["ok"] is False
    assert "repo_dir /mnt/sfs/zengzhengpeng/proto-tools shared by ty2,ty3" in payload["blockers"]
    assert "venv_dir /mnt/sfs/zengzhengpeng/proto-tools/.venv shared by ty2,ty3" in payload["blockers"]
    assert "select one shared-root node or pass --allow-shared-install-root before deploy apply" in payload["actions"]


def test_remote_deploy_apply_uses_deployment_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty", repo_dir="/remote/repo"),
        RemoteNode(name="h100", host="h100", work_dir="/remote/h100", repo_dir="/remote/repo-h100"),
    ]
    calls = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    class FakeCompleted:
        returncode = 0

    class FakeDeploymentClient:
        def deploy_node(self, node, local_repo_dir):
            calls.append((node.name, local_repo_dir))
            return [FakeCompleted(), FakeCompleted(), FakeCompleted()]

    monkeypatch.setattr(cli_mod.remote_exec, "RemoteDeploymentClient", FakeDeploymentClient)

    code, out, _ = _run(
        "remote",
        "deploy",
        "apply",
        "--profile",
        "nodes.json",
        "--node",
        "h100",
        "--local-repo-dir",
        str(repo),
        "--json",
    )

    assert code == 0
    assert json.loads(out) == {"h100": {"command_count": 3, "returncodes": [0, 0, 0]}}
    assert calls == [("h100", str(repo))]


def test_remote_deploy_apply_rejects_dirty_checkout_before_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    (repo / "untracked.txt").write_text("new\n")
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", repo_dir="/remote/repo")
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: [node])

    class ForbiddenDeploymentClient:
        def __init__(self):
            raise AssertionError("dirty checkout should be rejected before deployment")

    monkeypatch.setattr(cli_mod.remote_exec, "RemoteDeploymentClient", ForbiddenDeploymentClient)

    code, _, err = _run(
        "remote",
        "deploy",
        "apply",
        "--profile",
        "nodes.json",
        "--local-repo-dir",
        str(repo),
    )

    assert code == 2
    assert "local checkout has 0 tracked and 1 untracked changed paths" in err


def test_remote_deploy_apply_can_allow_dirty_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    (repo / "untracked.txt").write_text("new\n")
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", repo_dir="/remote/repo")
    calls = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: [node])

    class FakeCompleted:
        returncode = 0

    class FakeDeploymentClient:
        def deploy_node(self, deploy_node, local_repo_dir):
            calls.append((deploy_node.name, local_repo_dir))
            return [FakeCompleted()]

    monkeypatch.setattr(cli_mod.remote_exec, "RemoteDeploymentClient", FakeDeploymentClient)

    code, out, _ = _run(
        "remote",
        "deploy",
        "apply",
        "--profile",
        "nodes.json",
        "--local-repo-dir",
        str(repo),
        "--allow-dirty-checkout",
        "--json",
    )

    assert code == 0
    assert json.loads(out) == {"ty": {"command_count": 1, "returncodes": [0]}}
    assert calls == [("ty", str(repo))]


def test_remote_deploy_apply_rejects_shared_install_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    nodes = [
        RemoteNode(
            name="ty2",
            host="ty2",
            work_dir="/remote/ty2",
            repo_dir="/remote/shared-repo",
            venv_dir="/remote/shared-repo/.venv",
        ),
        RemoteNode(
            name="ty3",
            host="ty3",
            work_dir="/remote/ty3",
            repo_dir="/remote/shared-repo",
            venv_dir="/remote/shared-repo/.venv",
        ),
    ]
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    class ForbiddenDeploymentClient:
        def __init__(self):
            raise AssertionError("shared roots should be rejected before deployment")

    monkeypatch.setattr(cli_mod.remote_exec, "RemoteDeploymentClient", ForbiddenDeploymentClient)

    code, _, err = _run(
        "remote",
        "deploy",
        "apply",
        "--profile",
        "nodes.json",
        "--local-repo-dir",
        str(repo),
    )

    assert code == 2
    assert "shared install roots" in err
    assert "repo_dir /remote/shared-repo shared by ty2,ty3" in err
    assert "venv_dir /remote/shared-repo/.venv shared by ty2,ty3" in err


def test_remote_deploy_apply_can_allow_shared_install_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    nodes = [
        RemoteNode(
            name="ty2",
            host="ty2",
            work_dir="/remote/ty2",
            repo_dir="/remote/shared-repo",
            venv_dir="/remote/shared-repo/.venv",
        ),
        RemoteNode(
            name="ty3",
            host="ty3",
            work_dir="/remote/ty3",
            repo_dir="/remote/shared-repo",
            venv_dir="/remote/shared-repo/.venv",
        ),
    ]
    calls = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    class FakeCompleted:
        returncode = 0

    class FakeDeploymentClient:
        def deploy_node(self, node, local_repo_dir):
            calls.append((node.name, local_repo_dir))
            return [FakeCompleted()]

    monkeypatch.setattr(cli_mod.remote_exec, "RemoteDeploymentClient", FakeDeploymentClient)

    code, out, _ = _run(
        "remote",
        "deploy",
        "apply",
        "--profile",
        "nodes.json",
        "--local-repo-dir",
        str(repo),
        "--allow-shared-install-root",
        "--json",
    )

    assert code == 0
    assert json.loads(out) == {
        "ty2": {"command_count": 1, "returncodes": [0]},
        "ty3": {"command_count": 1, "returncodes": [0]},
    }
    assert calls == [("ty2", str(repo)), ("ty3", str(repo))]


def test_remote_manager_status_uses_manager_client(monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3"),
    ]
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    class FakeManagerClient:
        def status_node(self, node):
            return f"running:{node.name}"

    monkeypatch.setattr(cli_mod.remote_exec, "RemoteManagerClient", FakeManagerClient)

    code, out, _ = _run("remote", "manager", "status", "--profile", "nodes.json", "--json")

    assert code == 0
    assert json.loads(out) == {"ty": "running:ty", "ty3": "running:ty3"}


@pytest.mark.parametrize(
    ("verb", "extra_args", "expected_call", "expected_output"),
    [
        (
            "start",
            ["--poll-interval", "2", "--slurm-group-size", "7"],
            ("start", "ty", 2.0, 7, 12.5),
            {"ty": "started:ty"},
        ),
        ("status", [], ("status", "ty", 12.5), {"ty": "running:ty"}),
        ("preflight", [], ("preflight", "ty", 12.5), {"ty": {"ok": True, "node": "ty"}}),
        ("diagnostics", [], ("diagnostics", "ty", 12.5), {"ty": {"counts": {}, "node": "ty"}}),
    ],
)
def test_remote_manager_commands_pass_timeout_to_client(
    monkeypatch: pytest.MonkeyPatch,
    verb: str,
    extra_args: list[str],
    expected_call: tuple,
    expected_output: dict[str, object],
) -> None:
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")
    calls = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: [node])

    class FakeManagerClient:
        def start_node(self, node, *, poll_interval, slurm_group_size, timeout_sec):
            calls.append(("start", node.name, poll_interval, slurm_group_size, timeout_sec))
            return f"started:{node.name}"

        def status_node(self, node, *, timeout_sec):
            calls.append(("status", node.name, timeout_sec))
            return f"running:{node.name}"

        def preflight_node(self, node, *, timeout_sec):
            calls.append(("preflight", node.name, timeout_sec))
            return {"ok": True, "node": node.name}

        def diagnostics_node(self, node, *, timeout_sec):
            calls.append(("diagnostics", node.name, timeout_sec))
            return {"counts": {}, "node": node.name}

    monkeypatch.setattr(cli_mod.remote_exec, "RemoteManagerClient", FakeManagerClient)

    code, out, _ = _run(
        "remote",
        "manager",
        verb,
        "--profile",
        "nodes.json",
        "--timeout-sec",
        "12.5",
        *extra_args,
        "--json",
    )

    assert code == 0
    assert json.loads(out) == expected_output
    assert calls == [expected_call]


@pytest.mark.parametrize("verb", ["start", "preflight", "status", "diagnostics"])
def test_remote_manager_commands_reject_static_profile_errors_before_client(
    monkeypatch: pytest.MonkeyPatch,
    verb: str,
) -> None:
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/shared"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/shared"),
    ]
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    class ForbiddenManagerClient:
        def __init__(self) -> None:
            raise AssertionError("invalid profile should stop before manager client setup")

    monkeypatch.setattr(cli_mod.remote_exec, "RemoteManagerClient", ForbiddenManagerClient)

    code, _, err = _run("remote", "manager", verb, "--profile", "nodes.json")

    assert code == 2
    assert "Invalid remote profile: ty,ty3 work_dir" in err
    assert "remote work_dir is shared by multiple nodes" in err


def test_remote_rsync_pull_start_uses_local_client(monkeypatch: pytest.MonkeyPatch) -> None:
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")
    calls = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: [node])

    class FakeRsyncClient:
        def start(self, nodes, local_results_dir, *, pid_path, log_path, interval_sec):
            calls.append((nodes, local_results_dir, pid_path, log_path, interval_sec))
            return "started 123"

    monkeypatch.setattr(cli_mod.remote_exec, "LocalRsyncPullClient", FakeRsyncClient)

    code, out, _ = _run(
        "remote",
        "rsync",
        "pull",
        "start",
        "--profile",
        "nodes.json",
        "--local-results-dir",
        "remote_results",
        "--pid-file",
        "remote-rsync.pid",
        "--log-file",
        "remote-rsync.log",
        "--interval-sec",
        "3",
    )

    assert code == 0
    assert out == "started 123\n"
    assert calls == [([node], "remote_results", "remote-rsync.pid", "remote-rsync.log", 3)]


def test_remote_rsync_pull_once_runs_rendered_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")
    calls = []
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: [node])
    monkeypatch.setattr(
        cli_mod.remote_exec,
        "render_rsync_pull_commands",
        lambda nodes, local_results_dir: [["rsync", "ty:/remote/ty/results/", str(Path(local_results_dir) / "ty")]],
    )

    def fake_run(cmd, check=False):
        calls.append({"cmd": cmd, "check": check})

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    code, out, _ = _run(
        "remote",
        "rsync",
        "pull",
        "once",
        "--profile",
        "nodes.json",
        "--local-results-dir",
        "remote_results",
    )

    assert code == 0
    assert out == ""
    assert calls == [{"cmd": ["rsync", "ty:/remote/ty/results/", "remote_results/ty"], "check": True}]


def test_remote_run_plan_writes_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")
    calls = []
    input_json = tmp_path / "input.json"
    config_json = tmp_path / "config.json"
    manifest_path = tmp_path / "run.manifest.json"
    results_dir = tmp_path / "results"
    input_json.write_text('{"sequences":["AA"]}\n')
    config_json.write_text('{"device":"remote"}\n')
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: [node])

    class FakePlanner:
        def __init__(self, nodes, bucket_size):
            calls.append(("planner", nodes, bucket_size))

        def build_plan(self, tool, inputs, config, *, run_id=None):
            calls.append(("build", tool, inputs, config, run_id))
            return object()

    class FakeManifest:
        run_id = "run-cli"
        requests = (object(), object())
        local_results_dir = str(results_dir.resolve())

    def fake_from_plan(plan, *, nodes, bucket_size, local_results_dir):
        calls.append(("manifest", nodes, bucket_size, str(local_results_dir)))
        return FakeManifest()

    def fake_write(manifest, path, *, overwrite=False):
        calls.append(("write", manifest, path, overwrite))

    def forbidden_diagnostics(nodes):
        raise AssertionError("plan should not read diagnostics without the flag")

    monkeypatch.setattr(cli_mod.remote_exec, "RemoteDispatchPlanner", FakePlanner)
    monkeypatch.setattr(cli_mod.remote_exec.RemoteRunManifest, "from_plan", staticmethod(fake_from_plan))
    monkeypatch.setattr(cli_mod.remote_exec, "write_remote_run_manifest", fake_write)
    monkeypatch.setattr(cli_mod.remote_exec, "remote_run_diagnostics_backlog_loads", forbidden_diagnostics)

    code, out, _ = _run(
        "remote",
        "run",
        "plan",
        "remote-mock",
        "--profile",
        "nodes.json",
        "--input-json",
        str(input_json),
        "--config-json",
        str(config_json),
        "--bucket-size",
        "50",
        "--local-results-dir",
        str(results_dir),
        "--manifest",
        str(manifest_path),
        "--run-id",
        "run-cli",
    )

    assert code == 0
    assert "run_id\trun-cli" in out
    assert calls[0] == ("planner", [node], 50)
    assert calls[1] == ("build", "remote-mock", {"sequences": ["AA"]}, {"device": "remote"}, "run-cli")
    assert calls[2] == ("manifest", [node], 50, str(results_dir.resolve()))
    assert calls[3] == ("write", calls[3][1], str(manifest_path), False)


def test_remote_run_plan_can_use_diagnostics_backlog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3"),
    ]
    calls = []
    input_json = tmp_path / "input.json"
    manifest_path = tmp_path / "run.manifest.json"
    results_dir = tmp_path / "results"
    input_json.write_text('{"sequences":["AA","BB"]}\n')
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    def fake_backlog(backlog_nodes, *, timeout_sec=None):
        calls.append(("backlog", backlog_nodes, timeout_sec))
        return (
            {"ty": {"counts": {"queue": 10}}, "ty3": {"counts": {}}},
            {"ty": 10.0, "ty3": 0.0},
        )

    class FakePlanner:
        def __init__(self, planner_nodes, bucket_size, *, initial_node_loads=None):
            calls.append(("planner", planner_nodes, bucket_size, initial_node_loads))

        def build_plan(self, tool, inputs, config, *, run_id=None):
            calls.append(("build", tool, inputs, config, run_id))
            return object()

    class FakeManifest:
        run_id = "run-cli"
        requests = (object(),)
        local_results_dir = str(results_dir.resolve())

    def fake_from_plan(plan, *, nodes, bucket_size, local_results_dir):
        calls.append(("manifest", nodes, bucket_size, str(local_results_dir)))
        return FakeManifest()

    def fake_write(manifest, path, *, overwrite=False):
        calls.append(("write", manifest, path, overwrite))

    monkeypatch.setattr(cli_mod.remote_exec, "remote_run_diagnostics_backlog_loads", fake_backlog)
    monkeypatch.setattr(cli_mod.remote_exec, "RemoteDispatchPlanner", FakePlanner)
    monkeypatch.setattr(cli_mod.remote_exec.RemoteRunManifest, "from_plan", staticmethod(fake_from_plan))
    monkeypatch.setattr(cli_mod.remote_exec, "write_remote_run_manifest", fake_write)

    code, out, _ = _run(
        "remote",
        "run",
        "plan",
        "remote-mock",
        "--profile",
        "nodes.json",
        "--input-json",
        str(input_json),
        "--bucket-size",
        "50",
        "--local-results-dir",
        str(results_dir),
        "--manifest",
        str(manifest_path),
        "--run-id",
        "run-cli",
        "--use-diagnostics-backlog",
        "--remote-check-timeout-sec",
        "12.5",
        "--json",
    )

    payload = json.loads(out)
    assert code == 0
    assert payload["initial_node_loads"] == {"ty": 10.0, "ty3": 0.0}
    assert payload["diagnostics"] == {"ty": {"counts": {"queue": 10}}, "ty3": {"counts": {}}}
    assert calls[0] == ("backlog", nodes, 12.5)
    assert calls[1] == ("planner", nodes, 50, {"ty": 10.0, "ty3": 0.0})
    assert calls[2] == ("build", "remote-mock", {"sequences": ["AA", "BB"]}, None, "run-cli")
    assert calls[3] == ("manifest", nodes, 50, str(results_dir.resolve()))
    assert calls[4] == ("write", calls[4][1], str(manifest_path), False)


def test_remote_run_plan_rejects_static_profile_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nodes = [
        RemoteNode(name="ty2", host="ty2", work_dir="/remote/shared"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/shared"),
    ]
    input_json = tmp_path / "input.json"
    manifest_path = tmp_path / "run.manifest.json"
    input_json.write_text('{"sequences":["AA"]}\n')
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: nodes)

    class ForbiddenPlanner:
        def __init__(self, nodes, bucket_size):
            raise AssertionError("invalid profile should stop before planning")

    monkeypatch.setattr(cli_mod.remote_exec, "RemoteDispatchPlanner", ForbiddenPlanner)

    code, _, err = _run(
        "remote",
        "run",
        "plan",
        "remote-mock",
        "--profile",
        "nodes.json",
        "--input-json",
        str(input_json),
        "--bucket-size",
        "50",
        "--local-results-dir",
        str(tmp_path / "results"),
        "--manifest",
        str(manifest_path),
    )

    assert code == 2
    assert "Invalid remote profile: ty2,ty3 work_dir" in err
    assert "remote work_dir is shared by multiple nodes" in err
    assert not manifest_path.exists()


def test_remote_run_launch_uses_launch_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")
    calls = []
    input_json = tmp_path / "input.json"
    config_json = tmp_path / "config.json"
    manifest_path = tmp_path / "run.manifest.json"
    output_path = tmp_path / "output.json"
    results_dir = tmp_path / "results"
    input_json.write_text('{"sequences":["AA"]}\n')
    config_json.write_text('{"device":"remote"}\n')
    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_nodes_json", lambda path: [node])

    class FakeLaunchResult:
        def __init__(self) -> None:
            self.run_id = "run-cli"
            self.manifest_path = str(manifest_path)
            self.output_path = str(output_path)
            self.bucket_count = 1

        def to_dict(self):
            return {"run_id": self.run_id, "manifest": self.manifest_path, "output": self.output_path}

    def fake_launch(tool, nodes, inputs, config, **kwargs):
        calls.append((tool, nodes, inputs, config, kwargs))
        return FakeLaunchResult()

    monkeypatch.setattr(cli_mod.remote_exec, "launch_remote_run", fake_launch)

    code, out, _ = _run(
        "remote",
        "run",
        "launch",
        "remote-mock",
        "--profile",
        "nodes.json",
        "--input-json",
        str(input_json),
        "--config-json",
        str(config_json),
        "--bucket-size",
        "50",
        "--local-results-dir",
        str(results_dir),
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_path),
        "--run-id",
        "run-cli",
        "--overwrite",
        "--manager-poll-interval",
        "2",
        "--slurm-group-size",
        "7",
        "--wait-poll-interval",
        "3",
        "--timeout-sec",
        "30",
        "--rsync-mode",
        "background",
        "--rsync-pid-file",
        "remote-rsync.pid",
        "--rsync-log-file",
        "remote-rsync.log",
        "--rsync-interval-sec",
        "4",
        "--use-diagnostics-backlog",
        "--remote-check-timeout-sec",
        "12.5",
    )

    assert code == 0
    assert out.splitlines() == [
        "completed\trun-cli",
        f"manifest\t{manifest_path}",
        f"output\t{output_path}",
        "bucket_count\t1",
    ]
    assert calls == [
        (
            "remote-mock",
            [node],
            {"sequences": ["AA"]},
            {"device": "remote"},
            {
                "bucket_size": 50,
                "local_results_dir": str(results_dir),
                "manifest_path": str(manifest_path),
                "output_path": str(output_path),
                "run_id": "run-cli",
                "overwrite": True,
                "manager_poll_interval": 2.0,
                "slurm_group_size": 7,
                "wait_poll_interval": 3.0,
                "timeout_sec": 30.0,
                "rsync_mode": "background",
                "rsync_pid_path": "remote-rsync.pid",
                "rsync_log_path": "remote-rsync.log",
                "rsync_interval_sec": 4,
                "use_diagnostics_backlog": True,
                "overwrite_output": False,
                "remote_check_timeout_sec": 12.5,
            },
        )
    ]


def test_remote_run_resume_uses_manifest_resume_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []
    manifest_path = tmp_path / "run.manifest.json"
    output_path = tmp_path / "output.json"

    class FakeManifest:
        run_id = "run-cli"

    class FakeResumeResult:
        def __init__(self) -> None:
            self.run_id = "run-cli"
            self.manifest_path = str(manifest_path)
            self.output_path = str(output_path)
            self.bucket_count = 1

        def to_dict(self):
            return {"run_id": self.run_id, "manifest": self.manifest_path, "output": self.output_path}

    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_run_manifest", lambda path: FakeManifest())

    def fake_resume(manifest, **kwargs):
        calls.append((manifest.run_id, kwargs))
        return FakeResumeResult()

    monkeypatch.setattr(cli_mod.remote_exec, "resume_remote_run_manifest", fake_resume)

    code, out, _ = _run(
        "remote",
        "run",
        "resume",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_path),
        "--manager-poll-interval",
        "2",
        "--slurm-group-size",
        "7",
        "--wait-poll-interval",
        "3",
        "--timeout-sec",
        "30",
        "--rsync-mode",
        "background",
        "--rsync-pid-file",
        "remote-rsync.pid",
        "--rsync-log-file",
        "remote-rsync.log",
        "--rsync-interval-sec",
        "4",
        "--overwrite-output",
        "--remote-check-timeout-sec",
        "12.5",
    )

    assert code == 0
    assert out.splitlines() == [
        "completed\trun-cli",
        f"manifest\t{manifest_path}",
        f"output\t{output_path}",
        "bucket_count\t1",
    ]
    assert calls == [
        (
            "run-cli",
            {
                "manifest_path": str(manifest_path),
                "output_path": str(output_path),
                "manager_poll_interval": 2.0,
                "slurm_group_size": 7,
                "wait_poll_interval": 3.0,
                "timeout_sec": 30.0,
                "rsync_mode": "background",
                "rsync_pid_path": "remote-rsync.pid",
                "rsync_log_path": "remote-rsync.log",
                "rsync_interval_sec": 4,
                "overwrite_output": True,
                "remote_check_timeout_sec": 12.5,
            },
        )
    ]


def test_remote_run_submit_wait_collect_use_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []
    manifest_path = tmp_path / "run.manifest.json"
    output_path = tmp_path / "output.json"

    class FakeManifest:
        run_id = "run-cli"

    class FakeOutput:
        def model_dump_json(self, indent=None):
            assert indent == 2
            return '{"results":["ok"]}'

    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_run_manifest", lambda path: FakeManifest())

    def fake_submit(manifest):
        calls.append(("submit", manifest.run_id))
        return ["/remote/queue/bucket-00000.json"]

    def fake_wait(manifest, *, poll_interval, timeout_sec, pull_results, rsync_pid_path, rsync_log_path):
        calls.append(("wait", manifest.run_id, poll_interval, timeout_sec, pull_results, rsync_pid_path, rsync_log_path))

    def fake_collect(manifest):
        calls.append(("collect", manifest.run_id))
        return FakeOutput()

    monkeypatch.setattr(cli_mod.remote_exec, "submit_remote_run_manifest", fake_submit)
    monkeypatch.setattr(cli_mod.remote_exec, "wait_remote_run_manifest", fake_wait)
    monkeypatch.setattr(cli_mod.remote_exec, "collect_remote_run_manifest", fake_collect)

    submit_code, submit_out, _ = _run("remote", "run", "submit", "--manifest", str(manifest_path))
    wait_code, wait_out, _ = _run(
        "remote",
        "run",
        "wait",
        "--manifest",
        str(manifest_path),
        "--poll-interval",
        "2.5",
        "--timeout-sec",
        "10",
        "--no-pull",
        "--rsync-pid-file",
        "remote-rsync.pid",
        "--rsync-log-file",
        "remote-rsync.log",
    )
    collect_code, collect_out, _ = _run(
        "remote",
        "run",
        "collect",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_path),
    )

    assert submit_code == 0
    assert submit_out == "submitted\trun-cli\tbucket_count=1\n"
    assert wait_code == 0
    assert wait_out == "completed\trun-cli\n"
    assert collect_code == 0
    assert collect_out == f"output\t{output_path}\n"
    assert json.loads(output_path.read_text()) == {"results": ["ok"]}
    assert calls == [
        ("submit", "run-cli"),
        ("wait", "run-cli", 2.5, 10.0, False, "remote-rsync.pid", "remote-rsync.log"),
        ("collect", "run-cli"),
    ]


def test_remote_run_collect_rejects_existing_output_before_collect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run.manifest.json"
    output_path = tmp_path / "output.json"
    output_path.write_text("sentinel\n")

    def fake_collect(manifest):
        raise AssertionError("collect should not run")

    monkeypatch.setattr(cli_mod.remote_exec, "collect_remote_run_manifest", fake_collect)

    code, _, err = _run(
        "remote",
        "run",
        "collect",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_path),
    )

    assert code == 2
    assert "Remote run output already exists" in err
    assert output_path.read_text() == "sentinel\n"


def test_remote_run_collect_overwrite_output_replaces_existing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run.manifest.json"
    output_path = tmp_path / "output.json"
    output_path.write_text("sentinel\n")

    class FakeManifest:
        run_id = "run-cli"

    class FakeOutput:
        def model_dump_json(self, indent=None):
            assert indent == 2
            return '{"results":["ok"]}'

    monkeypatch.setattr(cli_mod.remote_exec, "load_remote_run_manifest", lambda path: FakeManifest())
    monkeypatch.setattr(cli_mod.remote_exec, "collect_remote_run_manifest", lambda manifest: FakeOutput())

    code, out, _ = _run(
        "remote",
        "run",
        "collect",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_path),
        "--overwrite-output",
    )

    assert code == 0
    assert out == f"output\t{output_path}\n"
    assert json.loads(output_path.read_text()) == {"results": ["ok"]}


# ── Per-tool docs ───────────────────────────────────────────────────────────


def test_docs_text_includes_canonical_sections() -> None:
    code, out, _ = _run("docs", "esm2-embedding")
    assert code == 0
    assert "ESM2 Embeddings" in out
    assert "Applications" in out
    assert "Usage Tips" in out
    assert "Toolkit Notes" in out


def test_docs_no_toolkit_notes_flag() -> None:
    code, out, _ = _run("docs", "esm2-embedding", "--no-toolkit-notes")
    assert code == 0
    assert "Toolkit Notes" not in out


def test_docs_accepts_run_function_name() -> None:
    code, out, _ = _run("docs", "run_esm2_embeddings")
    assert code == 0
    assert "esm2-embedding" in out


def test_docs_json_roundtrips_to_pydantic_payload() -> None:
    code, out, _ = _run("docs", "esm2-embedding", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["key"] == "esm2-embedding"
    assert payload["toolkit_notes"]


# ── Error paths ─────────────────────────────────────────────────────────────


def test_ambiguous_toolkit_exits_two() -> None:
    code, _, err = _run("docs", "esm2")
    assert code == 2
    assert "ambiguous" in err
    assert "esm2-embedding" in err  # candidate list should be in the message


def test_unknown_identifier_exits_two() -> None:
    code, _, err = _run("docs", "not-a-real-tool")
    assert code == 2
    assert "Could not resolve" in err


def test_unknown_section_exits_one() -> None:
    code, _, err = _run("section", "esm2", "Nonexistent Section")
    assert code == 1
    assert "not found" in err


# ── Schema / example-input ─────────────────────────────────────────────────


def test_schema_input_is_valid_json() -> None:
    code, out, _ = _run("schema", "esm2-embedding", "--input")
    assert code == 0
    payload = json.loads(out)
    assert "properties" in payload
    assert "sequences" in payload["properties"]


def test_example_input_is_valid_json() -> None:
    code, out, _ = _run("example-input", "esm2-embedding")
    assert code == 0
    payload = json.loads(out)
    assert "sequences" in payload


# ── Example notebook ───────────────────────────────────────────────────────


def test_example_renders_markdown_and_code_fences() -> None:
    code, out, _ = _run("example", "esm2-embedding")
    assert code == 0
    assert out.startswith("# example notebook:")
    assert "example.ipynb" in out
    assert "```python" in out


def test_example_missing_notebook_exits_one() -> None:
    code, _, err = _run("example", "mmseqs2-clustering")
    assert code == 1
    assert "No example notebook found" in err


# ── Model doc verbs ────────────────────────────────────────────────────────


@pytest.mark.parametrize("verb", ["input", "config", "output"])
def test_model_doc_verbs(verb: str) -> None:
    code, out, _ = _run(verb, "esm2-embedding")
    assert code == 0
    assert "ESM2Embeddings" in out

"""Tests for remote execution planning and manager helpers."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import Field

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils import remote_execution as remote_execution_mod
from proto_tools.utils.remote_execution import (
    ExistingRemoteBatchNode,
    ExistingRemoteCommandNode,
    LocalRsyncPullClient,
    RemoteDeploymentClient,
    RemoteDispatchPlanner,
    RemoteManagerClient,
    RemoteNode,
    RemoteRunManifest,
    RemoteSubmissionClient,
    RemoteToolDispatcher,
    build_deploy_render_report,
    collect_existing_remote_command_manifest,
    collect_plan_outputs,
    collect_remote_run_manifest,
    execute_request,
    execute_request_file,
    inspect_local_checkout,
    launch_remote_run,
    load_existing_remote_batch_nodes_json,
    load_existing_remote_command_nodes_json,
    load_remote_nodes_json,
    load_remote_run_manifest,
    plan_existing_remote_command_batch,
    read_fasta_records,
    reconcile_submitted_slurm_jobs,
    remote_run_diagnostics_backlog_loads,
    render_continuous_rsync_script,
    render_deploy_commands,
    render_existing_artifact_pull_commands,
    render_existing_remote_shell_command,
    render_manager_diagnostics_script,
    render_manager_loop_command,
    render_manager_status_script,
    render_preflight_script,
    render_rsync_pull_commands,
    render_rsync_pull_status_script,
    render_start_manager_script,
    render_start_rsync_pull_script,
    render_stop_rsync_pull_script,
    resume_remote_run_manifest,
    run_existing_remote_command,
    run_manager_loop,
    run_manager_once,
    scaffold_remote_node,
    select_existing_remote_command_node,
    smoke_remote_nodes,
    submit_remote_run_manifest,
    validate_remote_nodes,
    wait_remote_run_manifest,
    write_existing_remote_batch_nodes_json,
    write_existing_remote_command_nodes_json,
    write_remote_nodes_json,
    write_remote_run_manifest,
)
from proto_tools.utils.tool_io import BaseToolInput
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase


@pytest.fixture
def clean_registry():
    """Provide a clean registry for each test."""
    original_registry = ToolRegistry._registry.copy()
    original_backend = ToolRegistry._dispatch_backend
    ToolRegistry._registry.clear()
    ToolRegistry.clear_dispatch_backend()
    yield ToolRegistry
    ToolRegistry._registry = original_registry
    ToolRegistry._dispatch_backend = original_backend


class RemoteMockInput(BaseToolInput):
    """Mock iterable input with sequence-length scheduling cost."""

    sequences: list[str] = Field(description="Sequences to process")

    @classmethod
    def item_cost(cls, item):
        return float(len(item))


class RemoteMockConfig(BaseConfig):
    """Mock remote config."""

    device: str = ConfigField(default="remote", title="Device", description="Execution device")
    suffix: str = ConfigField(default="x", title="Suffix", description="Suffix to append")


class RemoteMockOutput(MockToolOutputBase):
    """Mock iterable output."""

    results: list[str] = Field(description="Processed results")


def _register_remote_mock_tool(registry):
    """Register a mock iterable tool."""

    @registry.register(
        key="remote-mock",
        label="Remote Mock",
        category="testing",
        input_class=RemoteMockInput,
        config_class=RemoteMockConfig,
        output_class=RemoteMockOutput,
        description="Mock remote iterable tool",
        iterable_input_fields=["sequences"],
        iterable_output_field="results",
    )
    def run_remote_mock(inputs, config=None, instance=None):
        del instance
        return RemoteMockOutput(results=[f"{seq}:{config.device}:{config.suffix}" for seq in inputs.sequences])

    return run_remote_mock


@pytest.mark.parametrize("name", ["", "../x", "/x", ".", "..", ".hidden", "a/b", r"a\b"])
def test_remote_node_rejects_unsafe_path_segment_names(name: str) -> None:
    """Node names become local result path components and must be single safe segments."""
    with pytest.raises(ValueError, match=r"RemoteNode\.name"):
        RemoteNode(name=name, host="ty", work_dir="/remote/ty")


def test_remote_node_accepts_safe_path_segment_name() -> None:
    """Node names can keep common run-label characters."""
    node = RemoteNode(name="ty-01.a_b", host="ty", work_dir="/remote/ty")

    assert node.name == "ty-01.a_b"


def test_existing_remote_command_profile_roundtrip(tmp_path: Path) -> None:
    """Existing-command profiles should roundtrip without using manager fields."""
    profile_path = tmp_path / "existing-nodes.json"
    nodes = [
        ExistingRemoteCommandNode(
            name="ty",
            host="ty",
            work_dir="/remote/existing",
            command_argv=("/env/bin/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
            env={"CUDA_VISIBLE_DEVICES": "0"},
            gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
            ssh_args=("-p", "2222"),
            rsync_ssh_args=("-i", "key.pem"),
        )
    ]

    write_existing_remote_command_nodes_json(nodes, profile_path)
    loaded = load_existing_remote_command_nodes_json(profile_path)

    assert loaded == nodes
    payload = json.loads(profile_path.read_text())
    assert payload["profile_schema_version"] == 1
    assert payload["nodes"][0]["command_argv"] == [
        "/env/bin/python",
        "/scripts/run.py",
        "-i",
        "{input}",
        "-o",
        "{output}",
    ]
    assert select_existing_remote_command_node(loaded, "ty") == nodes[0]


def test_existing_remote_batch_profile_roundtrip(tmp_path: Path) -> None:
    """One profile file should freeze both manager and existing-command contracts."""
    profile_path = tmp_path / "af3-batch.json"
    node = ExistingRemoteBatchNode(
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
            command_argv=("/env/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
            status="ready",
            software="alphafold3",
            required_artifacts=("**/*_model.cif",),
            required_stdout_patterns=("CudaDevice",),
            env={"CUDA_VISIBLE_DEVICES": "7"},
            gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
        ),
    )

    write_existing_remote_batch_nodes_json([node], profile_path)

    assert load_existing_remote_batch_nodes_json(profile_path) == [node]
    payload = json.loads(profile_path.read_text())
    assert payload["nodes"][0]["manager"]["worker_device"] == "cuda:7"
    assert payload["nodes"][0]["command"]["software"] == "alphafold3"
    assert payload["nodes"][0]["command"]["required_stdout_patterns"] == ["CudaDevice"]
    assert payload["nodes"][0]["command"]["gpu_admission"] == {
        "max_memory_used_mib": 1024,
        "max_utilization_percent": 5,
    }


def test_existing_remote_command_batch_plans_thirty_fasta_records_by_weight(tmp_path: Path) -> None:
    """The real 30-record shape should freeze one GPU command per weighted assignment."""
    fasta = tmp_path / "random_protein.fasta"
    fasta.write_text("".join(f">random_protein_{idx}\n{'A' * 50}\n" for idx in range(1, 31)))
    manager_nodes = [
        RemoteNode(
            name="ty",
            host="ty",
            work_dir="/sfs/home/user/proto_remote_af3_ty",
            weight=1,
            worker_device="cuda:7",
        ),
        RemoteNode(
            name="ty3",
            host="ty3",
            work_dir="/mnt/sfs/user/proto_remote_af3_ty3",
            weight=1,
            worker_device="cuda:0",
        ),
        RemoteNode(
            name="h100",
            host="h100",
            work_dir="/public/home/user/proto_remote_af3_h100",
            weight=3,
            scheduler="slurm",
            worker_device="cuda",
            sbatch_args=("--partition=gpu_special", "--gres=gpu:h100:1"),
            max_active_slurm_jobs=2,
        ),
    ]
    command_nodes = [
        ExistingRemoteCommandNode(
            name=node.name,
            host=node.host,
            work_dir=f"/{node.name}/remote_runs/alphafold3",
            command_argv=(f"/{node.name}/env/bin/python", f"/{node.name}/007_run_af3_direct.py", "-i", "{input}", "-o", "{output}"),
            env={} if node.name == "h100" else {"CUDA_VISIBLE_DEVICES": "7" if node.name == "ty" else "0"},
            gpu_admission=(
                None
                if node.name == "h100"
                else {"max_utilization_percent": 5, "max_memory_used_mib": 1024}
            ),
            status="ready",
            software="alphafold3",
            required_artifacts=("**/*_model.cif", "**/*_summary_confidences.json"),
            required_stdout_patterns=("CudaDevice",),
        )
        for node in manager_nodes
    ]
    nodes = [
        ExistingRemoteBatchNode(manager=manager, command=command)
        for manager, command in zip(manager_nodes, command_nodes, strict=True)
    ]

    records = read_fasta_records(fasta)
    manifest = plan_existing_remote_command_batch(
        nodes,
        input_fasta=fasta,
        run_id="af3-random30",
        bucket_size=1,
        local_results_dir=tmp_path / "results",
        software="alphafold3",
    )

    assert len(records) == 30
    assert records[0].header == "random_protein_1"
    assert records[0].sequence == "A" * 50
    counts = {
        node_name: sum(request.node_name == node_name for request in manifest.requests)
        for node_name in ("ty", "ty3", "h100")
    }
    assert counts == {"ty": 6, "ty3": 6, "h100": 18}
    assert all(request.request["request_kind"] == "existing_command" for request in manifest.requests)
    assert all("device" not in request.request for request in manifest.requests)
    assert len(
        {
            task["artifact_dir"]
            for request in manifest.requests
            for task in request.request["existing_command"]["tasks"]
        }
    ) == 30
    first = manifest.requests[0].request["existing_command"]
    assert first["env"] == {"CUDA_VISIBLE_DEVICES": "7"}
    assert first["resource_admission"] == {
        "kind": "cuda_idle",
        "gpu_id": "7",
        "max_utilization_percent": 5,
        "max_memory_used_mib": 1024,
    }
    assert first["required_stdout_patterns"] == ["CudaDevice"]
    assert first["tasks"][0]["input_text"] == f">random_protein_1\n{'A' * 50}\n"


def test_existing_remote_command_batch_rejects_partial_profile(tmp_path: Path) -> None:
    """A partial runtime must stop planning before any SSH side effect."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">one\nAAAA\n")
    manager = RemoteNode(name="ty3", host="ty3", work_dir="/remote/manager", worker_device="cuda:0")
    command = ExistingRemoteCommandNode(
        name="ty3",
        host="ty3",
        work_dir="/remote/af3",
        command_argv=("/env/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        status="partial",
        software="alphafold3",
        required_artifacts=("**/*_model.cif",),
    )

    with pytest.raises(ValueError, match="status='partial'"):
        plan_existing_remote_command_batch(
            [ExistingRemoteBatchNode(manager=manager, command=command)],
            input_fasta=fasta,
            run_id="partial-run",
            bucket_size=1,
            local_results_dir=tmp_path / "results",
            software="alphafold3",
        )


def test_existing_remote_command_batch_requires_cuda_stdout_evidence_for_af3(tmp_path: Path) -> None:
    """AF3 batch planning must reject profiles that cannot prove CUDA execution."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">one\nAAAA\n")
    manager = RemoteNode(name="ty", host="ty", work_dir="/remote/manager", worker_device="cuda:7")
    command = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/af3",
        command_argv=("/env/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        status="ready",
        software="alphafold3",
        required_artifacts=("**/*_model.cif",),
        env={"CUDA_VISIBLE_DEVICES": "7"},
        gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
    )

    with pytest.raises(ValueError, match=r"required_stdout_patterns.*CudaDevice"):
        plan_existing_remote_command_batch(
            [ExistingRemoteBatchNode(manager=manager, command=command)],
            input_fasta=fasta,
            run_id="unsafe-af3",
            bucket_size=1,
            local_results_dir=tmp_path / "results",
            software="alphafold3",
        )


def test_resolve_remote_compute_config_uses_alias_relative_profile_and_defaults(tmp_path: Path) -> None:
    """The hand-compute registry should resolve one ready profile without caller path knowledge."""
    profile = tmp_path / "profiles" / "af3.json"
    profile.parent.mkdir()
    profile.write_text('{"profile_schema_version": 1, "nodes": []}\n')
    registry = tmp_path / "runtime_registry.json"
    registry.write_text(
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
                            "slurm_group_size": 20,
                            "wait_poll_interval": 2,
                            "timeout_sec": 7200,
                            "remote_check_timeout_sec": 60,
                        },
                    }
                },
            }
        )
    )

    config = remote_execution_mod.resolve_remote_compute_config("af3", registry_path=registry)

    assert config.software == "alphafold3"
    assert config.registry_path == registry.resolve()
    assert config.profile_path == profile.resolve()
    assert config.bucket_size == 1
    assert config.slurm_group_size == 20
    assert config.wait_poll_interval == 2.0
    assert config.timeout_sec == 7200.0
    assert config.remote_check_timeout_sec == 60.0


def test_resolve_remote_compute_config_finds_bundled_codex_skill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / "codex"
    registry = (
        codex_home
        / "skills"
        / "02_molaison"
        / "hand-compute"
        / "references"
        / "runtime_registry.json"
    )
    registry.parent.mkdir(parents=True)
    profile = registry.parent / "profiles" / "af3.json"
    profile.parent.mkdir()
    profile.write_text('{"profile_schema_version": 1, "nodes": []}\n')
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "software": {
                    "alphafold3": {
                        "aliases": ["af3"],
                        "status": "ready",
                        "profile": "profiles/af3.json",
                    }
                },
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.delenv("PROTO_TOOLS_REMOTE_COMPUTE_REGISTRY", raising=False)

    config = remote_execution_mod.resolve_remote_compute_config("af3")

    assert config.registry_path == registry.resolve()
    assert config.profile_path == profile.resolve()


def test_resolve_remote_compute_config_rejects_non_ready_runtime(tmp_path: Path) -> None:
    """A configured runtime must never silently fall back from blocked to another execution mode."""
    registry = tmp_path / "runtime_registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "software": {
                    "alphafold3": {
                        "aliases": ["af3"],
                        "status": "blocked",
                        "profile": "profiles/af3.json",
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match=r"not ready.*status='blocked'"):
        remote_execution_mod.resolve_remote_compute_config("af3", registry_path=registry)


def test_resolve_remote_compute_config_rejects_non_finite_timeout(tmp_path: Path) -> None:
    """Registry timing values must be finite so foreground supervision cannot hang unpredictably."""
    profile = tmp_path / "af3.json"
    profile.write_text('{"profile_schema_version": 1, "nodes": []}\n')
    registry = tmp_path / "runtime_registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "software": {
                    "alphafold3": {
                        "aliases": ["af3"],
                        "status": "ready",
                        "profile": "af3.json",
                        "defaults": {"timeout_sec": float("nan")},
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match=r"defaults.timeout_sec must be positive or null"):
        remote_execution_mod.resolve_remote_compute_config("af3", registry_path=registry)


def test_existing_remote_command_batch_launch_rejects_cpu_gpu_profile(tmp_path: Path) -> None:
    """The old CPU substrate profile must never launch AlphaFold3."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">one\nAAAA\n")
    manager = RemoteNode(name="ty", host="ty", work_dir="/remote/manager", worker_device="cpu")
    command = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/af3",
        command_argv=("/env/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        status="ready",
        software="alphafold3",
        required_artifacts=("**/*_model.cif",),
    )

    with pytest.raises(ValueError, match="must use a CUDA worker_device"):
        remote_execution_mod.launch_existing_remote_command_batch(
            [ExistingRemoteBatchNode(manager=manager, command=command)],
            input_fasta=fasta,
            software="alphafold3",
            run_dir=tmp_path / "run",
        )


def test_existing_remote_command_batch_launch_rejects_cpu_gpu_profile_for_af3_alias(tmp_path: Path) -> None:
    """The short AF3 name must enforce the same GPU-only contract as alphafold3."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">one\nAAAA\n")
    manager = RemoteNode(name="ty", host="ty", work_dir="/remote/manager", worker_device="cpu")
    command = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/af3",
        command_argv=("/env/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        status="ready",
        software="af3",
        required_artifacts=("**/*_model.cif",),
    )

    with pytest.raises(ValueError, match="must use a CUDA worker_device"):
        remote_execution_mod.launch_existing_remote_command_batch(
            [ExistingRemoteBatchNode(manager=manager, command=command)],
            input_fasta=fasta,
            software="af3",
            run_dir=tmp_path / "run",
        )


def test_existing_remote_command_batch_launch_creates_result_pull_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline result rsync must receive an existing node destination directory."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">one\nAAAA\n")
    manager = RemoteNode(name="ty", host="ty", work_dir="/remote/manager", worker_device="cuda:7")
    command = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/af3",
        command_argv=("/env/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        status="ready",
        software="alphafold3",
        required_artifacts=("**/*_model.cif",),
        required_stdout_patterns=("CudaDevice",),
        env={"CUDA_VISIBLE_DEVICES": "7"},
        gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
    )
    monkeypatch.setattr(
        remote_execution_mod,
        "_manager_preflight_node",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        remote_execution_mod,
        "_manager_start_node",
        lambda *args, **kwargs: "running",
    )
    monkeypatch.setattr(remote_execution_mod, "submit_remote_run_manifest", lambda *args, **kwargs: [])

    def assert_pull_destination(manifest, **kwargs):
        del kwargs
        assert (Path(manifest.local_results_dir) / "ty").is_dir()
        raise RuntimeError("stop after result directory assertion")

    monkeypatch.setattr(remote_execution_mod, "wait_remote_run_manifest", assert_pull_destination)

    def successful_preflight_runner(command_argv, **kwargs):
        if any("utilization.gpu,memory.used" in part for part in command_argv):
            return subprocess.CompletedProcess(command_argv, 0, stdout="0, 0\n", stderr="")
        return subprocess.CompletedProcess(command_argv, 0, stdout="runtime-ready\n", stderr="")

    with pytest.raises(RuntimeError, match="stop after result directory assertion"):
        remote_execution_mod.launch_existing_remote_command_batch(
            [ExistingRemoteBatchNode(manager=manager, command=command)],
            input_fasta=fasta,
            software="alphafold3",
            run_dir=tmp_path / "run",
            runner=successful_preflight_runner,
        )


def test_existing_remote_command_batch_launch_resumes_matching_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running launch should resume the exact frozen manifest without duplicate work."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">one\nAAAA\n")
    run_dir = tmp_path / "run"
    manager = RemoteNode(name="ty", host="ty", work_dir="/remote/manager", worker_device="cuda:7")
    command = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/af3",
        command_argv=("/env/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        status="ready",
        software="alphafold3",
        required_artifacts=("**/*_model.cif",),
        required_stdout_patterns=("CudaDevice",),
        env={"CUDA_VISIBLE_DEVICES": "7"},
        gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
    )
    batch_nodes = [ExistingRemoteBatchNode(manager=manager, command=command)]
    frozen = plan_existing_remote_command_batch(
        batch_nodes,
        input_fasta=fasta,
        run_id="run",
        bucket_size=1,
        local_results_dir=run_dir / "remote_results",
        software="alphafold3",
    )
    write_remote_run_manifest(frozen, run_dir / "run.manifest.json")
    monkeypatch.setattr(remote_execution_mod, "_manager_preflight_node", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(remote_execution_mod, "_manager_start_node", lambda *args, **kwargs: "running")
    monkeypatch.setattr(remote_execution_mod, "submit_remote_run_manifest", lambda *args, **kwargs: [])

    def assert_resumed(manifest, **kwargs):
        del kwargs
        assert manifest.to_dict() == frozen.to_dict()
        raise RuntimeError("matching manifest resumed")

    monkeypatch.setattr(remote_execution_mod, "wait_remote_run_manifest", assert_resumed)

    def successful_preflight_runner(command_argv, **kwargs):
        if any("utilization.gpu,memory.used" in part for part in command_argv):
            return subprocess.CompletedProcess(command_argv, 0, stdout="0, 0\n", stderr="")
        return subprocess.CompletedProcess(command_argv, 0, stdout="runtime-ready\n", stderr="")

    with pytest.raises(RuntimeError, match="matching manifest resumed"):
        remote_execution_mod.launch_existing_remote_command_batch(
            batch_nodes,
            input_fasta=fasta,
            software="alphafold3",
            run_dir=run_dir,
            runner=successful_preflight_runner,
        )


def test_existing_remote_command_batch_launch_rejects_mismatched_existing_manifest(tmp_path: Path) -> None:
    """An existing run directory must not be resumed with changed scientific inputs."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">one\nAAAA\n")
    run_dir = tmp_path / "run"
    manager = RemoteNode(name="ty", host="ty", work_dir="/remote/manager", worker_device="cuda:7")
    command = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/af3",
        command_argv=("/env/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        status="ready",
        software="alphafold3",
        required_artifacts=("**/*_model.cif",),
        required_stdout_patterns=("CudaDevice",),
        env={"CUDA_VISIBLE_DEVICES": "7"},
        gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
    )
    batch_nodes = [ExistingRemoteBatchNode(manager=manager, command=command)]
    frozen = plan_existing_remote_command_batch(
        batch_nodes,
        input_fasta=fasta,
        run_id="run",
        bucket_size=1,
        local_results_dir=run_dir / "remote_results",
        software="alphafold3",
    )
    write_remote_run_manifest(frozen, run_dir / "run.manifest.json")
    fasta.write_text(">one\nBBBB\n")

    with pytest.raises(ValueError, match="does not match the existing manifest"):
        remote_execution_mod.launch_existing_remote_command_batch(
            batch_nodes,
            input_fasta=fasta,
            software="alphafold3",
            run_dir=run_dir,
            runner=lambda *args, **kwargs: pytest.fail("SSH must not run for a mismatched manifest"),
        )


def test_existing_batch_launch_adopts_empty_run_dir_and_freezes_manifest_before_manager(
    tmp_path: Path,
) -> None:
    """A preflight retry must use the manifest written before manager state changes."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">one\nAAAA\n")
    run_dir = tmp_path / "run"
    (run_dir / "remote_results").mkdir(parents=True)
    manager = RemoteNode(name="ty", host="ty", work_dir="/remote/manager", worker_device="cuda:0")
    command = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/af3",
        command_argv=("/env/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        software="alphafold3",
        required_artifacts=("**/*_model.cif",),
        required_stdout_patterns=("CudaDevice",),
        env={"CUDA_VISIBLE_DEVICES": "0"},
        gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
    )

    class FailingManager:
        def preflight_node(self, node, **kwargs):
            assert (run_dir / "run.manifest.json").is_file()
            raise RuntimeError("manager preflight stopped")

    def fake_runner(command_argv, **kwargs):
        if any("utilization.gpu,memory.used" in part for part in command_argv):
            return subprocess.CompletedProcess(command_argv, 0, stdout="0, 0\n", stderr="")
        return subprocess.CompletedProcess(command_argv, 0, stdout="runtime-ready\n", stderr="")

    with pytest.raises(RuntimeError, match="manager preflight stopped"):
        remote_execution_mod.launch_existing_remote_command_batch(
            [ExistingRemoteBatchNode(manager=manager, command=command)],
            input_fasta=fasta,
            software="alphafold3",
            run_dir=run_dir,
            manager_client=FailingManager(),
            runner=fake_runner,
        )

    frozen = load_remote_run_manifest(run_dir / "run.manifest.json")
    assert frozen.run_id == "run"
    assert len(frozen.requests) == 1


def test_existing_batch_launch_assigns_new_work_only_to_live_admitted_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A busy direct GPU must not receive new buckets when a Slurm GPU node is ready."""
    fasta = tmp_path / "four.fasta"
    fasta.write_text("".join(f">p{index}\nAAAA\n" for index in range(4)))
    direct = ExistingRemoteBatchNode(
        manager=RemoteNode(
            name="ty",
            host="ty",
            work_dir="/remote/manager-ty",
            worker_device="cuda:7",
        ),
        command=ExistingRemoteCommandNode(
            name="ty",
            host="ty",
            work_dir="/remote/af3-ty",
            command_argv=("/ty/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
            software="alphafold3",
            required_artifacts=("**/*_model.cif",),
            required_stdout_patterns=("CudaDevice",),
            env={"CUDA_VISIBLE_DEVICES": "7"},
            gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
        ),
    )
    slurm = ExistingRemoteBatchNode(
        manager=RemoteNode(
            name="h100",
            host="h100",
            work_dir="/remote/manager-h100",
            scheduler="slurm",
            worker_device="cuda",
            sbatch_args=("--partition=gpu_special", "--gres=gpu:h100:1"),
            max_active_slurm_jobs=2,
        ),
        command=ExistingRemoteCommandNode(
            name="h100",
            host="h100",
            work_dir="/remote/af3-h100",
            command_argv=("/h100/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
            software="alphafold3",
            required_artifacts=("**/*_model.cif",),
            required_stdout_patterns=("CudaDevice",),
        ),
    )

    class FakeManager:
        def preflight_node(self, node, **kwargs):
            return {"node": node.name, "ok": True}

        def start_node(self, node, **kwargs):
            return f"running {node.name}"

    def fake_runner(command, **kwargs):
        del kwargs
        if any("utilization.gpu,memory.used" in part for part in command):
            return subprocess.CompletedProcess(command, 0, stdout="100, 6302\n", stderr="")
        if "sbatch --test-only" in command[-1]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="sbatch: Job 1 to start at 2000-01-01T00:00:00 using h100\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="runtime-ready\n", stderr="")

    monkeypatch.setattr(remote_execution_mod, "submit_remote_run_manifest", lambda *args, **kwargs: [])

    def assert_live_assignment(manifest, **kwargs):
        del kwargs
        assert {request.node_name for request in manifest.requests} == {"h100"}
        raise RuntimeError("live assignment reached wait")

    monkeypatch.setattr(remote_execution_mod, "wait_remote_run_manifest", assert_live_assignment)

    with pytest.raises(RuntimeError, match="live assignment reached wait"):
        remote_execution_mod.launch_existing_remote_command_batch(
            [direct, slurm],
            input_fasta=fasta,
            software="alphafold3",
            run_dir=tmp_path / "run",
            manager_client=FakeManager(),
            runner=fake_runner,
        )


def test_existing_batch_launch_initializes_profile_owned_manager_state() -> None:
    """Launch setup should create manager state roots without deploying code or runtimes."""
    node = RemoteNode(
        name="f101-gpu0",
        host="f101",
        work_dir="/remote/manager-f101-gpu0",
        ssh_args=("-o", "BatchMode=yes"),
    )
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    remote_execution_mod._ensure_existing_batch_manager_state(
        node,
        timeout_sec=60,
        runner=fake_runner,
    )

    assert calls[0][0][:-1] == ["ssh", "-o", "BatchMode=yes", "f101"]
    assert shlex.split(calls[0][0][-1]) == [
        "mkdir",
        "-p",
        "/remote/manager-f101-gpu0",
        "/remote/manager-f101-gpu0/queue",
        "/remote/manager-f101-gpu0/results",
        "/remote/manager-f101-gpu0/logs",
        "/remote/manager-f101-gpu0/running",
        "/remote/manager-f101-gpu0/done",
        "/remote/manager-f101-gpu0/failed",
        "/remote/manager-f101-gpu0/submitted",
        "/remote/manager-f101-gpu0/cancelled",
    ]
    assert calls[0][1]["timeout"] == 60


def test_existing_batch_launch_rebalances_queued_work_without_command_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wait loop must move unclaimed work when live GPU availability changes."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">p0\nAAAA\n")

    def batch_node(name: str) -> ExistingRemoteBatchNode:
        return ExistingRemoteBatchNode(
            manager=RemoteNode(
                name=name,
                host=name,
                work_dir=f"/remote/manager-{name}",
                worker_device="cuda:0",
            ),
            command=ExistingRemoteCommandNode(
                name=name,
                host=name,
                work_dir=f"/remote/af3-{name}",
                command_argv=(
                    f"/{name}/python",
                    "/scripts/run.py",
                    "-i",
                    "{input}",
                    "-o",
                    "{output}",
                ),
                software="alphafold3",
                required_artifacts=("**/*_model.cif",),
                required_stdout_patterns=("CudaDevice",),
                env={"CUDA_VISIBLE_DEVICES": "0"},
                gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
            ),
        )

    source = batch_node("ty")
    target = batch_node("f101")
    live_phase = {"changed": False}
    submitted = []

    class FakeManager:
        def preflight_node(self, node, **kwargs):
            return {"node": node.name, "ok": True}

        def start_node(self, node, **kwargs):
            return f"running {node.name}"

    def fake_runner(command, **kwargs):
        del kwargs
        if any("utilization.gpu,memory.used" in part for part in command):
            host = command[command.index("nvidia-smi") - 1]
            ready = (host == "ty") != live_phase["changed"]
            usage = "0, 0\n" if ready else "100, 6302\n"
            return subprocess.CompletedProcess(command, 0, stdout=usage, stderr="")
        if command[-1].startswith(source.manager.python + " -c"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout='{"status": "withdrawn"}\n',
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="runtime-ready\n", stderr="")

    monkeypatch.setattr(
        remote_execution_mod,
        "submit_remote_run_manifest",
        lambda manifest, **kwargs: submitted.append(manifest) or [],
    )

    def assert_live_move(manifest, **kwargs):
        live_phase["changed"] = True
        updated = kwargs["poll_hook"](manifest)
        assert {request.node_name for request in updated.requests} == {"f101"}
        raise RuntimeError("same launcher followed live move")

    monkeypatch.setattr(remote_execution_mod, "wait_remote_run_manifest", assert_live_move)

    with pytest.raises(RuntimeError, match="same launcher followed live move"):
        remote_execution_mod.launch_existing_remote_command_batch(
            [source, target],
            input_fasta=fasta,
            software="alphafold3",
            run_dir=tmp_path / "run",
            manager_client=FakeManager(),
            runner=fake_runner,
        )

    assert [manifest.requests[0].node_name for manifest in submitted] == ["ty", "f101"]


def test_existing_batch_launch_reports_initial_and_latest_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The result must not overwrite initial planning evidence with the last live probe."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">p0\nAAAA\n")
    node = ExistingRemoteBatchNode(
        manager=RemoteNode(
            name="ty",
            host="ty",
            work_dir="/remote/manager-ty",
            worker_device="cuda:0",
        ),
        command=ExistingRemoteCommandNode(
            name="ty",
            host="ty",
            work_dir="/remote/af3-ty",
            command_argv=("/ty/python", "/run.py", "-i", "{input}", "-o", "{output}"),
            software="alphafold3",
            required_artifacts=("**/*_model.cif",),
            required_stdout_patterns=("CudaDevice",),
            env={"CUDA_VISIBLE_DEVICES": "0"},
            gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
        ),
    )
    probe_count = 0

    def fake_runner(command, **kwargs):
        nonlocal probe_count
        del kwargs
        if any("utilization.gpu,memory.used" in part for part in command):
            probe_count += 1
            usage = "0, 0\n" if probe_count == 1 else "100, 4096\n"
            return subprocess.CompletedProcess(command, 0, stdout=usage, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="runtime-ready\n", stderr="")

    monkeypatch.setattr(remote_execution_mod, "_manager_preflight_node", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(remote_execution_mod, "_manager_start_node", lambda *args, **kwargs: "running")
    monkeypatch.setattr(remote_execution_mod, "submit_remote_run_manifest", lambda *args, **kwargs: [])

    def finish_after_live_probe(manifest, **kwargs):
        kwargs["poll_hook"](manifest)
        return manifest

    monkeypatch.setattr(remote_execution_mod, "wait_remote_run_manifest", finish_after_live_probe)
    monkeypatch.setattr(remote_execution_mod, "render_existing_artifact_pull_commands", lambda *args: [])
    monkeypatch.setattr(
        remote_execution_mod,
        "collect_existing_remote_command_manifest",
        lambda *args: {"task_count": 1},
    )

    result = remote_execution_mod.launch_existing_remote_command_batch(
        [node],
        input_fasta=fasta,
        software="alphafold3",
        run_dir=tmp_path / "run",
        runner=fake_runner,
    )

    assert result.preflight["ty"]["admission"]["ready"] is True
    assert result.admission["ty"]["ready"] is False


def test_existing_batch_rebalance_fills_each_ready_direct_device_once(tmp_path: Path) -> None:
    """An idle target must claim one bucket before it can receive another move."""
    fasta = tmp_path / "three.fasta"
    fasta.write_text("".join(f">p{index}\nAAAA\n" for index in range(3)))

    def batch_node(name: str) -> ExistingRemoteBatchNode:
        return ExistingRemoteBatchNode(
            manager=RemoteNode(
                name=name,
                host=name,
                work_dir=f"/remote/manager-{name}",
                worker_device="cuda:0",
            ),
            command=ExistingRemoteCommandNode(
                name=name,
                host=name,
                work_dir=f"/remote/af3-{name}",
                command_argv=(f"/{name}/python", "/run.py", "-i", "{input}", "-o", "{output}"),
                software="alphafold3",
                required_artifacts=("**/*_model.cif",),
                required_stdout_patterns=("CudaDevice",),
                env={"CUDA_VISIBLE_DEVICES": "0"},
                gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
            ),
        )

    source = batch_node("source")
    target = batch_node("target")
    run_dir = tmp_path / "run"
    manifest_path = run_dir / "run.manifest.json"
    manifest = plan_existing_remote_command_batch(
        [source],
        input_fasta=fasta,
        run_id="run",
        bucket_size=1,
        local_results_dir=run_dir / "remote_results",
        software="alphafold3",
    )
    write_remote_run_manifest(manifest, manifest_path)
    kwargs = {
        "batch_nodes": [source, target],
        "input_fasta": fasta,
        "software": "alphafold3",
        "admission": {
            "source": {"ready": False, "kind": "cuda_idle"},
            "target": {"ready": True, "kind": "cuda_idle"},
        },
        "manifest_path": manifest_path,
        "withdrawer": lambda *args, **kwargs: {"status": "withdrawn"},
        "runner": lambda *args, **kwargs: None,
    }

    updated, first_moves = remote_execution_mod._rebalance_unclaimed_existing_batch_manifest(
        manifest,
        **kwargs,
    )
    unchanged, second_moves = remote_execution_mod._rebalance_unclaimed_existing_batch_manifest(
        updated,
        **kwargs,
    )

    assert len(first_moves) == 1
    assert [request.node_name for request in updated.requests].count("target") == 1
    assert unchanged == updated
    assert second_moves == []


def test_existing_batch_slurm_future_start_uses_direct_queue_as_holding_pool() -> None:
    """A far-future Slurm allocation must not absorb work from reassignable direct queues."""
    command = ExistingRemoteCommandNode(
        name="h100",
        host="h100",
        work_dir="/remote/af3-h100",
        command_argv=("/python", "/run.py", "-i", "{input}", "-o", "{output}"),
    )
    slurm = ExistingRemoteBatchNode(
        manager=RemoteNode(
            name="h100",
            host="h100",
            work_dir="/remote/manager-h100",
            scheduler="slurm",
            sbatch_args=("--partition=gpu_special", "--gres=gpu:h100:1"),
            max_active_slurm_jobs=2,
        ),
        command=command,
    )
    direct = ExistingRemoteBatchNode(
        manager=RemoteNode(name="ty", host="ty", work_dir="/remote/manager-ty"),
        command=ExistingRemoteCommandNode(
            name="ty",
            host="ty",
            work_dir="/remote/af3-ty",
            command_argv=("/python", "/run.py", "-i", "{input}", "-o", "{output}"),
        ),
    )

    def future_runner(command_argv, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            command_argv,
            0,
            stdout="sbatch: Job 1 to start at 2099-01-01T00:00:00 using h100\n",
            stderr="",
        )

    slurm_admission = remote_execution_mod._probe_existing_batch_admission(
        slurm,
        timeout_sec=60,
        runner=future_runner,
    )
    selected = remote_execution_mod._select_existing_batch_planning_nodes(
        [direct, slurm],
        {
            "ty": {"kind": "cuda_idle", "ready": False},
            "h100": slurm_admission,
        },
    )

    assert slurm_admission["ready"] is False
    assert selected == [direct]


def test_existing_batch_resume_rebalances_unclaimed_request_and_journals_move(tmp_path: Path) -> None:
    """Resume should recover after only part of a busy-node batch was moved."""
    fasta = tmp_path / "four.fasta"
    fasta.write_text("".join(f">p{index}\nAAAA\n" for index in range(4)))
    direct = ExistingRemoteBatchNode(
        manager=RemoteNode(name="ty", host="ty", work_dir="/remote/manager-ty", worker_device="cuda:7"),
        command=ExistingRemoteCommandNode(
            name="ty",
            host="ty",
            work_dir="/remote/af3-ty",
            command_argv=("/ty/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
            software="alphafold3",
            required_artifacts=("**/*_model.cif",),
            required_stdout_patterns=("CudaDevice",),
            env={"CUDA_VISIBLE_DEVICES": "7"},
            gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
        ),
    )
    slurm = ExistingRemoteBatchNode(
        manager=RemoteNode(
            name="h100",
            host="h100",
            work_dir="/remote/manager-h100",
            scheduler="slurm",
            worker_device="cuda",
            sbatch_args=("--partition=gpu_special", "--gres=gpu:h100:1"),
            max_active_slurm_jobs=2,
        ),
        command=ExistingRemoteCommandNode(
            name="h100",
            host="h100",
            work_dir="/remote/af3-h100",
            command_argv=("/h100/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
            software="alphafold3",
            required_artifacts=("**/*_model.cif",),
            required_stdout_patterns=("CudaDevice",),
        ),
    )
    run_dir = tmp_path / "run"
    manifest_path = run_dir / "run.manifest.json"
    manifest = plan_existing_remote_command_batch(
        [direct, slurm],
        input_fasta=fasta,
        run_id="run",
        bucket_size=1,
        local_results_dir=run_dir / "remote_results",
        software="alphafold3",
    )
    write_remote_run_manifest(manifest, manifest_path)
    direct_requests = [request for request in manifest.requests if request.node_name == "ty"]
    assert len(direct_requests) == 2
    for request in manifest.requests:
        if request.node_name != "h100":
            continue
        result_path = manifest.local_result_path(request)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "run_id": manifest.run_id,
                    "bucket_id": request.bucket_id,
                    "tool_key": manifest.tool_key,
                    "item_indices": list(request.item_indices),
                }
            )
        )
    moved = []

    def interrupted_withdraw(node, request, *, cancellation_id, timeout_sec, runner):
        del cancellation_id, timeout_sec, runner
        moved.append((node.name, request.bucket_id))
        if len(moved) == 1:
            return {"status": "withdrawn"}
        raise RuntimeError("local interruption after remote cancellation")

    kwargs = {
        "batch_nodes": [direct, slurm],
        "input_fasta": fasta,
        "software": "alphafold3",
        "admission": {
            "ty": {"ready": False, "kind": "cuda_idle"},
            "h100": {"ready": True, "kind": "slurm"},
        },
        "manifest_path": manifest_path,
        "runner": lambda *args, **kwargs: None,
    }
    with pytest.raises(RuntimeError, match="local interruption"):
        remote_execution_mod._rebalance_unclaimed_existing_batch_manifest(
            manifest,
            withdrawer=interrupted_withdraw,
            **kwargs,
        )

    completed_journal = (
        run_dir
        / "rebalances"
        / f"{direct_requests[0].bucket_id}.move-00001.ty-to-h100.json"
    )
    planned_journal = (
        run_dir
        / "rebalances"
        / f"{direct_requests[1].bucket_id}.move-00001.ty-to-h100.json"
    )
    assert json.loads(completed_journal.read_text())["status"] == "completed"
    assert json.loads(planned_journal.read_text())["status"] == "planned"
    partial = load_remote_run_manifest(manifest_path)
    assert partial != manifest
    remote_execution_mod._validate_existing_batch_resume_manifest(
        partial,
        batch_nodes=[direct, slurm],
        input_fasta=fasta,
        software="alphafold3",
    )

    def recovered_withdraw(node, request, *, cancellation_id, timeout_sec, runner):
        del node, request, cancellation_id, timeout_sec, runner
        return {"status": "already_cancelled"}

    updated, moves = remote_execution_mod._rebalance_unclaimed_existing_batch_manifest(
        partial,
        withdrawer=recovered_withdraw,
        **kwargs,
    )

    assert moved == [("ty", request.bucket_id) for request in direct_requests]
    assert {request.node_name for request in updated.requests} == {"h100"}
    assert moves == [
        {
            "bucket_id": direct_requests[1].bucket_id,
            "from_node": "ty",
            "to_node": "h100",
        }
    ]
    journal = json.loads(planned_journal.read_text())
    assert journal["status"] == "completed"
    assert load_remote_run_manifest(manifest_path) == updated


def test_existing_batch_queue_withdrawal_quotes_remote_python_command() -> None:
    """SSH must receive one quoted command so multiline Python reaches the remote interpreter."""
    node = RemoteNode(
        name="ty",
        host="ty",
        work_dir="/remote/manager",
        python="/remote/venv/bin/python",
        ssh_args=("-o", "BatchMode=yes"),
    )
    request = remote_execution_mod.RemoteRunRequest(
        node_name="ty",
        bucket_id="bucket-00000",
        item_indices=(0,),
        total_cost=4.0,
        request={"run_id": "run", "bucket_id": "bucket-00000"},
    )

    def fake_runner(command, **kwargs):
        assert command[:-1] == ["ssh", "-o", "BatchMode=yes", "ty"]
        remote_argv = shlex.split(command[-1])
        assert remote_argv[0:2] == ["/remote/venv/bin/python", "-c"]
        assert remote_argv[2] == remote_execution_mod._WITHDRAW_QUEUED_REQUEST_PY
        assert remote_argv[3:] == [
            "/remote/manager/queue/run/bucket-00000.json",
            "/remote/manager/cancelled/run/bucket-00000/move-00001-ty-to-h100.json",
        ]
        assert json.loads(kwargs["input"]) == request.request
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"status": "withdrawn"}\n',
            stderr="",
        )

    result = remote_execution_mod._withdraw_queued_manifest_request(
        node,
        request,
        cancellation_id="move-00001-ty-to-h100",
        runner=fake_runner,
    )

    assert result == {"status": "withdrawn"}


def test_existing_batch_queue_withdrawal_retries_ssh_transport_failure() -> None:
    """A transient SSH exit 255 must recover inside the one launch command."""
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/manager")
    request = remote_execution_mod.RemoteRunRequest(
        node_name="ty",
        bucket_id="bucket-00000",
        item_indices=(0,),
        total_cost=4.0,
        request={"run_id": "run", "bucket_id": "bucket-00000"},
    )
    attempts = []

    def flaky_runner(command, **kwargs):
        attempts.append((command, kwargs))
        if len(attempts) == 1:
            raise subprocess.CalledProcessError(
                255,
                command,
                output="",
                stderr="kex_exchange_identification: Connection closed by remote host",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"status": "withdrawn"}\n',
            stderr="",
        )

    result = remote_execution_mod._withdraw_queued_manifest_request(
        node,
        request,
        cancellation_id="move-00001-ty-to-h100",
        timeout_sec=10,
        retry_interval=0,
        runner=flaky_runner,
    )

    assert result == {"status": "withdrawn"}
    assert len(attempts) == 2


def test_existing_batch_queue_withdrawal_uses_fresh_tombstone_for_each_move(
    tmp_path: Path,
) -> None:
    """Returning a bucket to one node must not reuse an older cancellation tombstone."""
    queue_path = tmp_path / "queue" / "run" / "bucket-00000.json"
    request = {"run_id": "run", "bucket_id": "bucket-00000"}
    queue_path.parent.mkdir(parents=True)

    statuses = []
    for move_id in ("move-00001-ty-to-f101", "move-00002-ty-to-f102"):
        queue_path.write_text(json.dumps(request))
        cancelled_path = tmp_path / "cancelled" / "run" / "bucket-00000" / f"{move_id}.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                remote_execution_mod._WITHDRAW_QUEUED_REQUEST_PY,
                str(queue_path),
                str(cancelled_path),
            ],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=True,
        )
        statuses.append(json.loads(completed.stdout)["status"])
        assert cancelled_path.is_file()
        assert not queue_path.exists()

    assert statuses == ["withdrawn", "withdrawn"]


def test_existing_remote_command_manifest_pulls_and_validates_native_artifacts(tmp_path: Path) -> None:
    """Artifact collection must validate the local mirror, not trust process exit alone."""
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">one\nAAAA\n")
    manager = RemoteNode(name="ty", host="ty", work_dir="/remote/manager", worker_device="cuda:0")
    command = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/af3",
        command_argv=("/env/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        status="ready",
        software="alphafold3",
        required_artifacts=("**/*_model.cif",),
        required_stdout_patterns=("CudaDevice",),
        env={"CUDA_VISIBLE_DEVICES": "0"},
        gpu_admission={"max_utilization_percent": 5, "max_memory_used_mib": 1024},
    )
    manifest = plan_existing_remote_command_batch(
        [ExistingRemoteBatchNode(manager=manager, command=command)],
        input_fasta=fasta,
        run_id="artifact-run",
        bucket_size=1,
        local_results_dir=tmp_path / "results",
        software="alphafold3",
    )
    request = manifest.requests[0]
    remote_task = request.request["existing_command"]["tasks"][0]
    local_artifacts = tmp_path / "artifacts"
    local_task = local_artifacts / "ty" / "task_00000" / "attempt_ty"
    (local_task / "output" / "sample").mkdir(parents=True)
    (local_task / "output" / "sample" / "sample_model.cif").write_text("data_model\n")
    result_path = manifest.local_result_path(request)
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "request_kind": "existing_command",
                "tool_key": manifest.tool_key,
                "run_id": manifest.run_id,
                "bucket_id": request.bucket_id,
                "item_indices": [0],
                "existing_command_results": [
                    {
                        "task_id": "task_00000",
                        "artifact_dir": remote_task["artifact_dir"],
                        "input_path": f"{remote_task['artifact_dir']}/input/task_00000.fasta",
                        "output_dir": f"{remote_task['artifact_dir']}/output",
                        "stdout_path": f"{remote_task['artifact_dir']}/logs/stdout.log",
                        "stderr_path": f"{remote_task['artifact_dir']}/logs/stderr.log",
                        "artifacts": ["sample/sample_model.cif"],
                        "validated_stdout_patterns": ["CudaDevice"],
                    }
                ],
            }
        )
    )

    pull_commands = render_existing_artifact_pull_commands(manifest, local_artifacts)
    summary = collect_existing_remote_command_manifest(manifest, local_artifacts)

    assert pull_commands == [
        [
            "rsync",
            "-a",
            "ty:/remote/af3/artifact-run/artifacts/",
            str(local_artifacts / "ty") + "/",
        ]
    ]
    assert summary["task_count"] == 1
    assert summary["tasks"][0]["header"] == "one"
    assert summary["tasks"][0]["node"] == "ty"
    assert summary["tasks"][0]["local_artifact_dir"] == str(local_task)
    assert summary["tasks"][0]["artifacts"] == ["sample/sample_model.cif"]
    assert summary["tasks"][0]["validated_stdout_patterns"] == ["CudaDevice"]


def test_existing_remote_command_renders_direct_env_and_placeholders() -> None:
    """Direct existing-command rendering should preserve env and input/output paths."""
    node = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/existing",
        command_argv=("/env/bin/python", "/scripts/run.py", "predict", "-i", "{input}", "-o", "{output}"),
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )

    script = render_existing_remote_shell_command(
        node,
        remote_input_path="/remote/existing/run-1/input/input.fa",
        remote_output_dir="/remote/existing/run-1/output",
    )

    assert script.startswith("set -euo pipefail; mkdir -p /remote/existing/run-1/output")
    assert "export CUDA_VISIBLE_DEVICES=0" in script
    assert (
        "/env/bin/python /scripts/run.py predict -i /remote/existing/run-1/input/input.fa "
        "-o /remote/existing/run-1/output"
    ) in script


def test_existing_remote_command_renders_srun_wrapper() -> None:
    """Slurm-backed existing commands should run inside the rendered srun shell."""
    node = ExistingRemoteCommandNode(
        name="h100",
        host="h100",
        work_dir="/remote/existing",
        command_argv=("/env/bin/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        scheduler="srun",
        srun_args=("-p", "gpu_special", "--gres=gpu:h100:1", "--immediate=60"),
    )

    script = render_existing_remote_shell_command(
        node,
        remote_input_path="/remote/existing/run-1/input/input.fa",
        remote_output_dir="/remote/existing/run-1/output",
    )

    assert script.startswith("srun -p gpu_special --gres=gpu:h100:1 --immediate=60 bash -lc ")
    assert "set -euo pipefail" in script
    assert "/env/bin/python /scripts/run.py" in script


def test_existing_remote_command_plan_builds_stage_execute_collect_commands(tmp_path: Path) -> None:
    """Planning should render SSH prepare/execute plus rsync upload/collect."""
    input_path = tmp_path / "input.fa"
    collect_dir = tmp_path / "collect"
    input_path.write_text(">x\nAAAA\n")
    node = ExistingRemoteCommandNode(
        name="ty",
        host="ty-submit",
        rsync_host="ty-sync",
        work_dir="/remote/existing",
        command_argv=("/env/bin/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
        ssh_args=("-p", "2222"),
        rsync_ssh_args=("-i", "key.pem"),
    )

    plan = remote_execution_mod.plan_existing_remote_command(
        node,
        local_input_path=input_path,
        run_id="run-1",
        local_collect_dir=collect_dir,
    )

    assert plan.prepare_command[:6] == ("ssh", "-p", "2222", "ty-submit", "bash", "-lc")
    assert plan.execute_command[:6] == ("ssh", "-p", "2222", "ty-submit", "bash", "-lc")
    assert plan.remote_input_path == "/remote/existing/run-1/input/input.fa"
    assert plan.remote_output_dir == "/remote/existing/run-1/output"
    assert plan.upload_command == (
        "rsync",
        "-a",
        "-e",
        "ssh -i key.pem",
        str(input_path),
        "ty-sync:/remote/existing/run-1/input/input.fa",
    )
    assert plan.collect_command == (
        "rsync",
        "-a",
        "-e",
        "ssh -i key.pem",
        "ty-sync:/remote/existing/run-1/output/",
        str(collect_dir) + "/",
    )


def test_existing_remote_command_run_calls_steps_and_lists_collected_files(tmp_path: Path) -> None:
    """The run helper should execute prepare/upload/execute/collect in order."""
    input_path = tmp_path / "input.fa"
    collect_dir = tmp_path / "collect"
    input_path.write_text(">x\nAAAA\n")
    node = ExistingRemoteCommandNode(
        name="ty",
        host="ty",
        work_dir="/remote/existing",
        command_argv=("/env/bin/python", "/scripts/run.py", "-i", "{input}", "-o", "{output}"),
    )
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[:2] == ["rsync", "-a"] and str(cmd[-2]).endswith("/output/"):
            (collect_dir / "scores.json").write_text("{}\n")
            nested = collect_dir / "model"
            nested.mkdir()
            (nested / "structure.pdb").write_text("MODEL\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=f"stdout:{cmd[0]}", stderr=f"stderr:{cmd[0]}")

    result = run_existing_remote_command(
        node,
        local_input_path=input_path,
        run_id="run-1",
        local_collect_dir=collect_dir,
        runner=fake_runner,
        timeout_sec=12.5,
    )

    assert [cmd[0] for cmd, _ in calls] == ["ssh", "rsync", "ssh", "rsync"]
    assert all(kwargs["check"] is True for _, kwargs in calls)
    assert all(kwargs["text"] is True and kwargs["capture_output"] is True for _, kwargs in calls)
    assert [kwargs["timeout"] for _, kwargs in calls] == [12.5, 12.5, 12.5, 12.5]
    assert result.collected_files == ("model/structure.pdb", "scores.json")
    assert [step.step for step in result.command_results] == ["prepare", "upload", "execute", "collect"]
    assert result.command_results[2].stdout == "stdout:ssh"
    assert result.command_results[2].stderr == "stderr:ssh"
    assert result.to_dict()["command_results"][2]["step"] == "execute"


def _init_git_repo(path: Path) -> Path:
    """Create a minimal git checkout for local deployment tests."""
    path.mkdir(parents=True)
    subprocess.run(["git", "-C", str(path), "init"], check=True, text=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Proto Tests"], check=True)
    (path / "tracked.txt").write_text("clean\n")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "initial"], check=True, text=True, capture_output=True)
    return path


def test_remote_dispatch_planner_buckets_and_weights(clean_registry):
    """Planner should form contiguous buckets and send more cost to stronger nodes."""
    _register_remote_mock_tool(clean_registry)
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/proto-ty", weight=1.0, worker_device="cuda:0"),
        RemoteNode(
            name="h100",
            host="h100",
            work_dir="/remote/proto-h100",
            weight=3.0,
            scheduler="slurm",
            worker_device="cuda",
        ),
    ]
    inputs = RemoteMockInput(sequences=["A", "AA", "AAA", "AAAA", "AAAAA"])

    plan = RemoteDispatchPlanner(nodes, bucket_size=2).build_plan(
        "remote-mock",
        inputs,
        RemoteMockConfig(suffix="fold"),
        run_id="run-1",
    )

    assert [assignment.bucket.item_indices for assignment in plan.assignments] == [(0, 1), (2, 3), (4,)]
    h100_cost = sum(assignment.bucket.total_cost for assignment in plan.assignments_for_node("h100"))
    ty_cost = sum(assignment.bucket.total_cost for assignment in plan.assignments_for_node("ty"))
    assert h100_cost > ty_cost

    request = plan.request_for(plan.assignments[0])
    assert request["tool_key"] == "remote-mock"
    assert request["config"]["device"] == plan.assignments[0].node.worker_device
    assert request["inputs"]["sequences"] == ["A", "AA"]


def test_remote_dispatch_planner_uses_initial_node_loads(clean_registry):
    """Planner should avoid assigning new buckets to nodes with existing backlog."""
    _register_remote_mock_tool(clean_registry)
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty", weight=1.0, worker_device="cuda:0"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3", weight=1.0, worker_device="cuda:1"),
    ]

    plan = RemoteDispatchPlanner(nodes, bucket_size=1, initial_node_loads={"ty": 100.0}).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["A", "B"]),
        run_id="run-initial-load",
    )

    assert [assignment.node.name for assignment in plan.assignments] == ["ty3", "ty3"]


def test_remote_submission_stages_node_buckets_over_one_ssh(clean_registry):
    """Submission should stage every bucket for one node through one SSH command."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/proto", ssh_args=("-p", "2222"))
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["A", "B"]),
        run_id="run-2",
    )
    calls = []

    def fake_runner(cmd, input=None, text=False, check=False):
        calls.append({"cmd": cmd, "input": input, "text": text, "check": check})

    staged = RemoteSubmissionClient().submit_plan(plan, runner=fake_runner)

    assert staged == [
        node.queue_root / "run-2" / "bucket-00000.json",
        node.queue_root / "run-2" / "bucket-00001.json",
    ]
    assert len(calls) == 1
    assert calls[0]["cmd"][:4] == ["ssh", "-p", "2222", "ty"]
    assert "proto_tools.utils.remote_execution" not in calls[0]["cmd"][4]
    payload = json.loads(calls[0]["input"])
    assert [item["filename"] for item in payload["requests"]] == ["bucket-00000.json", "bucket-00001.json"]
    assert payload["requests"][0]["request"]["bucket_id"] == "bucket-00000"
    assert calls[0]["text"] is True
    assert calls[0]["check"] is True


def test_execute_request_runs_registered_tool(clean_registry):
    """A manager request should execute the registered tool locally on that node."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/proto", worker_device="cuda:1")
    plan = RemoteDispatchPlanner([node], bucket_size=2).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA", "BB"]),
        RemoteMockConfig(suffix="done"),
        run_id="run-3",
    )

    result = execute_request(plan.request_for(plan.assignments[0]))

    assert result["status"] == "completed"
    assert result["item_indices"] == [0, 1]
    assert result["output"]["results"] == ["AA:cuda:1:done", "BB:cuda:1:done"]


def test_execute_request_runs_existing_command_without_tool_registry(tmp_path: Path) -> None:
    """Existing-command requests should stage input, run argv, and inventory artifacts."""
    artifact_dir = (tmp_path / "artifacts" / "task_00000" / "attempt_ty").resolve()
    request = {
        "schema_version": 1,
        "request_kind": "existing_command",
        "tool_key": "existing-alphafold3",
        "run_id": "af3-existing-test",
        "bucket_id": "bucket-00000",
        "item_indices": [0],
        "existing_command": {
            "command_argv": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import sys; "
                    "inp=Path(sys.argv[1]); out=Path(sys.argv[2]); "
                    "out.mkdir(parents=True, exist_ok=True); "
                    "(out/'model.cif').write_text(inp.read_text()); "
                    "(out/'summary_confidences.json').write_text('{\"ptm\": 0.5}\\n'); "
                    "print('gpu command completed')"
                ),
                "{input}",
                "{output}",
            ],
            "env": {"CUDA_VISIBLE_DEVICES": "7"},
            "tasks": [
                {
                    "task_id": "task_00000",
                    "input_filename": "random_protein_1.fasta",
                    "input_text": ">random_protein_1\nAAAA\n",
                    "artifact_dir": str(artifact_dir),
                    "required_artifacts": ["model.cif", "summary_confidences.json"],
                }
            ],
        },
    }

    result = execute_request(request)

    assert result["status"] == "completed"
    assert result["request_kind"] == "existing_command"
    assert result["item_indices"] == [0]
    assert result["existing_command_results"] == [
        {
            "task_id": "task_00000",
            "artifact_dir": str(artifact_dir),
            "input_path": str(artifact_dir / "input" / "random_protein_1.fasta"),
            "output_dir": str(artifact_dir / "output"),
            "stdout_path": str(artifact_dir / "logs" / "stdout.log"),
            "stderr_path": str(artifact_dir / "logs" / "stderr.log"),
            "artifacts": ["model.cif", "summary_confidences.json"],
        }
    ]
    assert (artifact_dir / "input" / "random_protein_1.fasta").read_text() == ">random_protein_1\nAAAA\n"
    assert (artifact_dir / "logs" / "stdout.log").read_text() == "gpu command completed\n"
    assert (artifact_dir / "logs" / "stderr.log").read_text() == ""
    assert (artifact_dir / "output" / "model.cif").stat().st_size > 0


def test_execute_request_rejects_missing_required_stdout_device_evidence(tmp_path: Path) -> None:
    """A zero-exit command with artifacts must fail when CUDA log evidence is absent."""
    artifact_dir = (tmp_path / "artifacts" / "task_00000" / "attempt_h100").resolve()
    request = {
        "schema_version": 1,
        "request_kind": "existing_command",
        "tool_key": "existing-alphafold3",
        "run_id": "af3-device-proof-failed",
        "bucket_id": "bucket-00000",
        "item_indices": [0],
        "existing_command": {
            "command_argv": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import sys; "
                    "out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); "
                    "(out/'model.cif').write_text('model'); print('CpuDevice(id=0)')"
                ),
                "{input}",
                "{output}",
            ],
            "required_stdout_patterns": ["CudaDevice"],
            "tasks": [
                {
                    "task_id": "task_00000",
                    "input_filename": "random_protein_1.fasta",
                    "input_text": ">random_protein_1\nAAAA\n",
                    "artifact_dir": str(artifact_dir),
                    "required_artifacts": ["model.cif"],
                }
            ],
        },
    }

    with pytest.raises(RuntimeError, match=r"required stdout pattern.*CudaDevice"):
        execute_request(request)

    assert (artifact_dir / "logs" / "stdout.log").read_text() == "CpuDevice(id=0)\n"


def test_execute_request_file_writes_failed_existing_command_envelope_for_missing_artifact(
    tmp_path: Path,
) -> None:
    """A zero-exit command without required artifacts must remain a visible failure."""
    artifact_dir = (tmp_path / "artifacts" / "task_00000" / "attempt_ty").resolve()
    request = {
        "schema_version": 1,
        "request_kind": "existing_command",
        "tool_key": "existing-alphafold3",
        "run_id": "af3-existing-failed",
        "bucket_id": "bucket-00000",
        "item_indices": [0],
        "existing_command": {
            "command_argv": [sys.executable, "-c", "print('no model')", "{input}", "{output}"],
            "tasks": [
                {
                    "task_id": "task_00000",
                    "input_filename": "random_protein_1.fasta",
                    "input_text": ">random_protein_1\nAAAA\n",
                    "artifact_dir": str(artifact_dir),
                    "required_artifacts": ["model.cif"],
                }
            ],
        },
    }
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request))

    with pytest.raises(FileNotFoundError, match="required artifact"):
        execute_request_file(request_path, result_path)

    envelope = json.loads(result_path.read_text())
    assert envelope["status"] == "failed"
    assert envelope["request_kind"] == "existing_command"
    assert envelope["tool_key"] == "existing-alphafold3"
    assert "required artifact" in envelope["error"]
    assert (artifact_dir / "logs" / "stdout.log").read_text() == "no model\n"


def test_run_manager_once_direct_writes_result_and_done_request(clean_registry, tmp_path):
    """Direct manager mode should claim queued work, run it, and write a result."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/proto", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-4",
    )
    request_dir = tmp_path / "queue" / "run-4"
    request_dir.mkdir(parents=True)
    (request_dir / "bucket-00000.json").write_text(json.dumps(plan.request_for(plan.assignments[0])))

    events = run_manager_once(tmp_path / "queue", tmp_path / "results")

    assert [event.status for event in events] == ["completed"]
    result_path = tmp_path / "results" / "run-4" / "bucket-00000.json"
    assert json.loads(result_path.read_text())["output"]["results"] == ["AA:cuda:0:x"]
    assert (tmp_path / "done" / "run-4" / "bucket-00000.json").exists()
    assert not (tmp_path / "queue" / "run-4" / "bucket-00000.json").exists()


def test_run_manager_once_leaves_existing_request_queued_while_gpu_is_busy(tmp_path: Path) -> None:
    """A direct manager must not claim an existing GPU command until its GPU is idle."""
    request_dir = tmp_path / "queue" / "af3-busy"
    request_dir.mkdir(parents=True)
    request_path = request_dir / "bucket-00000.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request_kind": "existing_command",
                "tool_key": "existing-alphafold3",
                "run_id": "af3-busy",
                "bucket_id": "bucket-00000",
                "item_indices": [0],
                "existing_command": {
                    "command_argv": [sys.executable, "-c", "raise SystemExit('must not run')"],
                    "env": {"CUDA_VISIBLE_DEVICES": "7"},
                    "resource_admission": {
                        "kind": "cuda_idle",
                        "gpu_id": "7",
                        "max_utilization_percent": 5,
                        "max_memory_used_mib": 1024,
                    },
                    "tasks": [],
                },
            }
        )
    )
    calls: list[list[str]] = []

    def busy_gpu_runner(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="100, 6302\n", stderr="")

    events = run_manager_once(
        tmp_path / "queue",
        tmp_path / "results",
        runner=busy_gpu_runner,
    )

    assert events == []
    assert request_path.exists()
    assert not (tmp_path / "running").exists()
    assert not (tmp_path / "results").exists()
    assert calls == [
        [
            "nvidia-smi",
            "-i",
            "7",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    ]


def test_run_manager_once_fails_existing_request_when_gpu_probe_is_invalid(tmp_path: Path) -> None:
    """A broken admission probe must become a failed envelope instead of an endless queue."""
    request_dir = tmp_path / "queue" / "af3-probe-error"
    request_dir.mkdir(parents=True)
    request_path = request_dir / "bucket-00000.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request_kind": "existing_command",
                "tool_key": "existing-alphafold3",
                "run_id": "af3-probe-error",
                "bucket_id": "bucket-00000",
                "item_indices": [0],
                "existing_command": {
                    "command_argv": [sys.executable, "-c", "raise SystemExit('must not run')"],
                    "resource_admission": {
                        "kind": "cuda_idle",
                        "gpu_id": "7",
                        "max_utilization_percent": 5,
                        "max_memory_used_mib": 1024,
                    },
                    "tasks": [],
                },
            }
        )
    )

    def invalid_gpu_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="not-a-gpu-sample\n", stderr="")

    events = run_manager_once(
        tmp_path / "queue",
        tmp_path / "results",
        runner=invalid_gpu_runner,
    )

    assert [event.status for event in events] == ["failed"]
    envelope = json.loads(
        (tmp_path / "results" / "failed" / "bucket-00000.json.json").read_text()
    )
    assert envelope["status"] == "failed"
    assert "unexpected nvidia-smi admission output" in envelope["error"]
    assert (tmp_path / "failed" / "af3-probe-error" / "bucket-00000.json").exists()


def test_run_manager_once_rejects_unsafe_request_path_segments(clean_registry, tmp_path):
    """Malformed queue payloads must not write outside the results root."""
    _register_remote_mock_tool(clean_registry)
    request_dir = tmp_path / "queue" / "bad-run"
    request_dir.mkdir(parents=True)
    request_path = request_dir / "bad.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool_key": "remote-mock",
                "run_id": "../escape",
                "bucket_id": "bucket-00000",
                "item_indices": [0],
                "inputs": {"sequences": ["AA"]},
                "config": {"device": "cuda:0"},
            }
        )
    )

    events = run_manager_once(tmp_path / "queue", tmp_path / "results")

    assert [event.status for event in events] == ["failed"]
    assert not (tmp_path / "escape").exists()
    failed_result = tmp_path / "results" / "failed" / "bad.json.json"
    assert json.loads(failed_result.read_text())["status"] == "failed"
    assert (tmp_path / "failed" / "bad-run" / "bad.json").exists()


def test_run_manager_once_slurm_groups_requests_and_tracks_job_id(clean_registry, tmp_path):
    """SLURM mode should group queued buckets and record the sbatch job id."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA", "BB", "CC"]),
        run_id="run-5",
    )
    request_dir = tmp_path / "queue" / "run-5"
    request_dir.mkdir(parents=True)
    for assignment in plan.assignments:
        (request_dir / f"{assignment.bucket.bucket_id}.json").write_text(json.dumps(plan.request_for(assignment)))
    calls = []

    class FakeCompleted:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_runner(cmd, check=False, text=False, capture_output=False):
        calls.append({"cmd": cmd, "check": check, "text": text, "capture_output": capture_output})
        request_paths = [
            Path(args[index + 1])
            for args in [shlex.split(cmd[-1])]
            for index, arg in enumerate(args)
            if arg == "--request"
        ]
        assert request_paths
        assert all("submitted" in path.parts for path in request_paths)
        assert not any("running" in path.parts for path in request_paths)
        assert all(path.exists() for path in request_paths)
        return FakeCompleted(f"job-{len(calls)}\n")

    events = run_manager_once(
        tmp_path / "queue",
        tmp_path / "results",
        scheduler="slurm",
        python="/env/bin/python",
        sbatch_args=("--partition", "gpu", "--gres", "gpu:h100:1"),
        slurm_group_size=2,
        max_active_slurm_jobs=10,
        runner=fake_runner,
    )

    assert [event.status for event in events] == ["submitted", "submitted", "submitted"]
    assert [event.slurm_job_id for event in events] == ["job-1", "job-1", "job-2"]
    assert len(calls) == 2
    assert calls[0]["cmd"][:6] == ["sbatch", "--parsable", "--partition", "gpu", "--gres", "gpu:h100:1"]
    assert "proto_tools.utils.remote_execution run-request-batch" in calls[0]["cmd"][-1]
    assert calls[0]["cmd"][-1].count("--request") == 2
    assert calls[0]["check"] is True
    assert calls[0]["text"] is True
    assert calls[0]["capture_output"] is True
    assert (tmp_path / "submitted" / "run-5" / "bucket-00000.json").exists()
    assert not (tmp_path / "running" / "run-5" / "bucket-00000.json").exists()
    sidecar = tmp_path / "submitted" / "run-5" / "bucket-00000.json.slurm.json"
    sidecar_payload = json.loads(sidecar.read_text())
    assert sidecar_payload["slurm_job_id"] == "job-1"
    assert sidecar_payload["submitted_at_epoch"] > 0


def test_run_manager_once_slurm_submit_failure_moves_claims_to_failed(clean_registry, tmp_path):
    """SLURM submit failures should not strand claimed requests in running/submitted."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA", "BB"]),
        run_id="run-submit-fail",
    )
    request_dir = tmp_path / "queue" / "run-submit-fail"
    request_dir.mkdir(parents=True)
    for assignment in plan.assignments:
        (request_dir / f"{assignment.bucket.bucket_id}.json").write_text(json.dumps(plan.request_for(assignment)))

    def failing_runner(cmd, check=False, text=False, capture_output=False):
        request_paths = [
            Path(args[index + 1])
            for args in [shlex.split(cmd[-1])]
            for index, arg in enumerate(args)
            if arg == "--request"
        ]
        assert all("submitted" in path.parts for path in request_paths)
        assert all(path.exists() for path in request_paths)
        raise subprocess.CalledProcessError(1, cmd, stderr="sbatch unavailable")

    events = run_manager_once(
        tmp_path / "queue",
        tmp_path / "results",
        scheduler="slurm",
        python="/env/bin/python",
        slurm_group_size=2,
        max_active_slurm_jobs=10,
        runner=failing_runner,
    )

    assert [event.status for event in events] == ["failed", "failed"]
    assert not (tmp_path / "running" / "run-submit-fail" / "bucket-00000.json").exists()
    assert not (tmp_path / "submitted" / "run-submit-fail" / "bucket-00000.json").exists()
    assert (tmp_path / "failed" / "run-submit-fail" / "bucket-00000.json").exists()
    envelope = json.loads((tmp_path / "results" / "run-submit-fail" / "bucket-00000.json").read_text())
    assert envelope["status"] == "failed"
    assert "sbatch unavailable" in envelope["traceback"]


def test_run_manager_once_slurm_empty_sbatch_job_id_fails_visible(clean_registry, tmp_path):
    """Slurm submission must not leave an untrackable submitted sidecar."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-slurm-empty-job",
    )
    request_dir = tmp_path / "queue" / "run-slurm-empty-job"
    request_dir.mkdir(parents=True)
    (request_dir / "bucket-00000.json").write_text(json.dumps(plan.request_for(plan.assignments[0])))

    class FakeSubmittedWithoutJobId:
        stdout = "\n"

    events = run_manager_once(
        tmp_path / "queue",
        tmp_path / "results",
        scheduler="slurm",
        max_active_slurm_jobs=1,
        runner=lambda *args, **kwargs: FakeSubmittedWithoutJobId(),
    )

    assert [event.status for event in events] == ["failed"]
    result_path = tmp_path / "results" / "run-slurm-empty-job" / "bucket-00000.json"
    envelope = json.loads(result_path.read_text())
    assert envelope["status"] == "failed"
    assert "sbatch did not return a Slurm job id" in envelope["error"]
    assert (tmp_path / "failed" / "run-slurm-empty-job" / "bucket-00000.json").exists()
    assert not list((tmp_path / "submitted").glob("**/*.slurm.json"))


def test_run_manager_once_slurm_requires_active_job_cap_before_submit(clean_registry, tmp_path):
    """Slurm manager mode should fail before sbatch when active-job cap is missing."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-missing-cap",
    )
    request_dir = tmp_path / "queue" / "run-missing-cap"
    request_dir.mkdir(parents=True)
    (request_dir / "bucket-00000.json").write_text(json.dumps(plan.request_for(plan.assignments[0])))

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("missing cap should stop before sacct or sbatch")

    with pytest.raises(ValueError, match="max_active_slurm_jobs is required"):
        run_manager_once(
            tmp_path / "queue",
            tmp_path / "results",
            scheduler="slurm",
            runner=forbidden_runner,
        )

    assert (tmp_path / "queue" / "run-missing-cap" / "bucket-00000.json").exists()
    assert not (tmp_path / "running").exists()
    assert not (tmp_path / "submitted").exists()


def test_run_manager_once_slurm_caps_active_jobs_and_leaves_queue(clean_registry, tmp_path):
    """SLURM mode should not claim more queued groups than active-job capacity allows."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA", "BB", "CC"]),
        run_id="run-cap",
    )
    request_dir = tmp_path / "queue" / "run-cap"
    request_dir.mkdir(parents=True)
    for assignment in plan.assignments:
        (request_dir / f"{assignment.bucket.bucket_id}.json").write_text(json.dumps(plan.request_for(assignment)))

    active_request_dir = tmp_path / "submitted" / "run-active"
    active_request_dir.mkdir(parents=True)
    active_request = active_request_dir / "bucket-00000.json"
    active_request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool_key": "remote-mock",
                "run_id": "run-active",
                "bucket_id": "bucket-00000",
                "item_indices": [0],
                "inputs": {"sequences": ["DD"]},
                "config": {"device": "cuda"},
            }
        )
    )
    active_request.with_name("bucket-00000.json.slurm.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "submitted",
                "slurm_job_id": "active-1",
                "result_path": str(tmp_path / "results" / "run-active" / "bucket-00000.json"),
            }
        )
    )
    calls = []

    class FakeCompleted:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_runner(cmd, check=False, text=False, capture_output=False):
        calls.append(cmd)
        if cmd[0] == "sacct":
            return FakeCompleted("active-1|RUNNING\n")
        return FakeCompleted("new-1\n")

    events = run_manager_once(
        tmp_path / "queue",
        tmp_path / "results",
        scheduler="slurm",
        python="/env/bin/python",
        slurm_group_size=2,
        max_active_slurm_jobs=2,
        runner=fake_runner,
    )

    assert [event.status for event in events] == ["submitted", "submitted"]
    assert [event.slurm_job_id for event in events] == ["new-1", "new-1"]
    assert [call[0] for call in calls] == ["sacct", "sbatch"]
    assert calls[1][-1].count("--request") == 2
    assert (tmp_path / "queue" / "run-cap" / "bucket-00002.json").exists()
    assert not (tmp_path / "running" / "run-cap" / "bucket-00002.json").exists()
    assert (tmp_path / "submitted" / "run-active" / "bucket-00000.json.slurm.json").exists()


def test_run_manager_once_slurm_stale_missing_job_releases_active_slot(clean_registry, tmp_path):
    """Missing old Slurm jobs should fail visibly so queued buckets can use the freed slot."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    old_plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-slurm-stale",
    )
    old_request_dir = tmp_path / "submitted" / "run-slurm-stale"
    old_request_dir.mkdir(parents=True)
    old_request_path = old_request_dir / "bucket-00000.json"
    old_request = old_plan.request_for(old_plan.assignments[0])
    old_request_path.write_text(json.dumps(old_request))
    old_result_path = tmp_path / "results" / "run-slurm-stale" / "bucket-00000.json"
    old_request_path.with_name("bucket-00000.json.slurm.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "submitted",
                "slurm_job_id": "404",
                "submitted_at_epoch": 1.0,
                "result_path": str(old_result_path),
            }
        )
    )
    new_plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["BB"]),
        run_id="run-slurm-new",
    )
    new_request_dir = tmp_path / "queue" / "run-slurm-new"
    new_request_dir.mkdir(parents=True)
    (new_request_dir / "bucket-00000.json").write_text(json.dumps(new_plan.request_for(new_plan.assignments[0])))
    calls = []

    class EmptySacct:
        stdout = ""

    class Submitted:
        stdout = "200\n"

    def fake_runner(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "sacct":
            return EmptySacct()
        if cmd[0] == "sbatch":
            return Submitted()
        raise AssertionError(cmd)

    events = run_manager_once(
        tmp_path / "queue",
        tmp_path / "results",
        scheduler="slurm",
        max_active_slurm_jobs=1,
        slurm_accounting_grace_sec=10.0,
        runner=fake_runner,
    )

    assert [event.status for event in events] == ["failed", "submitted"]
    assert [event.slurm_job_id for event in events] == ["404", "200"]
    assert calls == ["sacct", "sbatch"]
    old_envelope = json.loads(old_result_path.read_text())
    assert old_envelope["status"] == "failed"
    assert "SLURM_JOB_NOT_FOUND_OR_STALE" in old_envelope["error"]
    assert (tmp_path / "failed" / "run-slurm-stale" / "bucket-00000.json").exists()
    assert (tmp_path / "submitted" / "run-slurm-new" / "bucket-00000.json").exists()


def test_run_manager_once_slurm_fresh_missing_job_keeps_active_slot(clean_registry, tmp_path):
    """Recent jobs missing from sacct should keep their active slot during the grace window."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    old_plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-slurm-fresh",
    )
    old_request_dir = tmp_path / "submitted" / "run-slurm-fresh"
    old_request_dir.mkdir(parents=True)
    old_request_path = old_request_dir / "bucket-00000.json"
    old_request_path.write_text(json.dumps(old_plan.request_for(old_plan.assignments[0])))
    old_result_path = tmp_path / "results" / "run-slurm-fresh" / "bucket-00000.json"
    old_request_path.with_name("bucket-00000.json.slurm.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "submitted",
                "slurm_job_id": "405",
                "submitted_at_epoch": remote_execution_mod.time.time(),
                "result_path": str(old_result_path),
            }
        )
    )
    new_plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["BB"]),
        run_id="run-slurm-waiting",
    )
    new_request_dir = tmp_path / "queue" / "run-slurm-waiting"
    new_request_dir.mkdir(parents=True)
    (new_request_dir / "bucket-00000.json").write_text(json.dumps(new_plan.request_for(new_plan.assignments[0])))

    class EmptySacct:
        stdout = ""

    events = run_manager_once(
        tmp_path / "queue",
        tmp_path / "results",
        scheduler="slurm",
        max_active_slurm_jobs=1,
        slurm_accounting_grace_sec=3600.0,
        runner=lambda *args, **kwargs: EmptySacct(),
    )

    assert events == []
    assert old_request_path.exists()
    assert not old_result_path.exists()
    assert (tmp_path / "queue" / "run-slurm-waiting" / "bucket-00000.json").exists()


def test_reconcile_submitted_slurm_failed_job_writes_failed_envelope(clean_registry, tmp_path):
    """Slurm reconciliation should expose terminal job failure as a failed bucket."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-slurm-failed",
    )
    request_dir = tmp_path / "submitted" / "run-slurm-failed"
    request_dir.mkdir(parents=True)
    request_path = request_dir / "bucket-00000.json"
    request_path.write_text(json.dumps(plan.request_for(plan.assignments[0])))
    sidecar = request_path.with_name("bucket-00000.json.slurm.json")
    result_path = tmp_path / "results" / "run-slurm-failed" / "bucket-00000.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "submitted",
                "slurm_job_id": "123",
                "run_id": "run-slurm-failed",
                "bucket_id": "bucket-00000",
                "result_path": str(result_path),
            }
        )
    )

    class FakeCompleted:
        stdout = "123|FAILED\n123.batch|COMPLETED\n"

    def fake_runner(cmd, check=False, text=False, capture_output=False):
        assert cmd[:4] == ["sacct", "--noheader", "--parsable2", "--format=JobIDRaw,State"]
        assert check is True
        assert text is True
        assert capture_output is True
        return FakeCompleted()

    events = reconcile_submitted_slurm_jobs(tmp_path / "submitted", tmp_path / "results", runner=fake_runner)

    assert [event.status for event in events] == ["failed"]
    assert events[0].slurm_job_id == "123"
    envelope = json.loads(result_path.read_text())
    assert envelope["status"] == "failed"
    assert "Slurm job 123 ended with state FAILED" in envelope["error"]
    assert (tmp_path / "failed" / "run-slurm-failed" / "bucket-00000.json").exists()
    assert (tmp_path / "failed" / "run-slurm-failed" / "bucket-00000.json.slurm.json").exists()
    assert not request_path.exists()
    assert not sidecar.exists()


def test_reconcile_submitted_slurm_rejects_unsafe_sidecar_result_path(clean_registry, tmp_path):
    """Slurm sidecars cannot redirect result reconciliation outside the results root."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-slurm-sidecar-path",
    )
    request_dir = tmp_path / "submitted" / "run-slurm-sidecar-path"
    request_dir.mkdir(parents=True)
    request_path = request_dir / "bucket-00000.json"
    request_path.write_text(json.dumps(plan.request_for(plan.assignments[0])))
    sidecar = request_path.with_name("bucket-00000.json.slurm.json")
    outside_path = tmp_path / "outside.json"
    safe_result_path = tmp_path / "results" / "run-slurm-sidecar-path" / "bucket-00000.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "submitted",
                "slurm_job_id": "126",
                "result_path": str(outside_path),
            }
        )
    )

    events = reconcile_submitted_slurm_jobs(
        tmp_path / "submitted",
        tmp_path / "results",
        runner=lambda *args, **kwargs: None,
    )

    assert [event.status for event in events] == ["failed"]
    assert not outside_path.exists()
    envelope = json.loads(safe_result_path.read_text())
    assert envelope["status"] == "failed"
    assert "result_path mismatch" in envelope["error"]
    assert (tmp_path / "failed" / "run-slurm-sidecar-path" / "bucket-00000.json").exists()


def test_run_manager_once_slurm_reconciles_completed_result(clean_registry, tmp_path):
    """Slurm manager should move submitted requests to done after result appears."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-slurm-completed",
    )
    request_dir = tmp_path / "submitted" / "run-slurm-completed"
    request_dir.mkdir(parents=True)
    request_path = request_dir / "bucket-00000.json"
    request = plan.request_for(plan.assignments[0])
    request_path.write_text(json.dumps(request))
    result_path = tmp_path / "results" / "run-slurm-completed" / "bucket-00000.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(execute_request(request)))
    request_path.with_name("bucket-00000.json.slurm.json").write_text(
        json.dumps({"schema_version": 1, "status": "submitted", "slurm_job_id": "124", "result_path": str(result_path)})
    )

    events = run_manager_once(
        tmp_path / "queue",
        tmp_path / "results",
        scheduler="slurm",
        max_active_slurm_jobs=1,
        runner=lambda *args, **kwargs: None,
    )

    assert [event.status for event in events] == ["completed"]
    assert (tmp_path / "done" / "run-slurm-completed" / "bucket-00000.json").exists()
    assert (tmp_path / "done" / "run-slurm-completed" / "bucket-00000.json.slurm.json").exists()
    assert not request_path.exists()


def test_run_manager_once_reconciles_running_completed_result(clean_registry, tmp_path):
    """Manager restart should finalize a running request when its completed result is visible."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-running-completed",
    )
    request = plan.request_for(plan.assignments[0])
    request_path = tmp_path / "running" / "run-running-completed" / "bucket-00000.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request))
    result_path = tmp_path / "results" / "run-running-completed" / "bucket-00000.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(execute_request(request)))

    events = run_manager_once(tmp_path / "queue", tmp_path / "results")

    assert [(event.status, event.request_path, event.result_path) for event in events] == [
        (
            "completed",
            tmp_path / "done" / "run-running-completed" / "bucket-00000.json",
            result_path,
        )
    ]
    assert not request_path.exists()
    assert (tmp_path / "done" / "run-running-completed" / "bucket-00000.json").exists()


def test_run_manager_once_reconciles_running_failed_result(clean_registry, tmp_path):
    """Manager restart should finalize a running request when its failed result is visible."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-running-failed",
    )
    request = plan.request_for(plan.assignments[0])
    request_path = tmp_path / "running" / "run-running-failed" / "bucket-00000.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request))
    result_path = tmp_path / "results" / "run-running-failed" / "bucket-00000.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "tool_key": "remote-mock",
                "run_id": "run-running-failed",
                "bucket_id": "bucket-00000",
                "item_indices": [0],
                "error": "boom",
            }
        )
    )

    events = run_manager_once(tmp_path / "queue", tmp_path / "results")

    assert [(event.status, event.request_path, event.result_path) for event in events] == [
        (
            "failed",
            tmp_path / "failed" / "run-running-failed" / "bucket-00000.json",
            result_path,
        )
    ]
    assert not request_path.exists()
    assert (tmp_path / "failed" / "run-running-failed" / "bucket-00000.json").exists()


def test_run_manager_once_leaves_running_request_without_result(clean_registry, tmp_path):
    """Manager restart should not guess that a result-less running request is stale."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-running-no-result",
    )
    request_path = tmp_path / "running" / "run-running-no-result" / "bucket-00000.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(plan.request_for(plan.assignments[0])))

    events = run_manager_once(tmp_path / "queue", tmp_path / "results")

    assert events == []
    assert request_path.exists()
    assert not (tmp_path / "done").exists()
    assert not (tmp_path / "failed").exists()


def test_reconcile_submitted_slurm_completed_result_does_not_query_sacct(clean_registry, tmp_path):
    """Visible result envelopes should reconcile without depending on Slurm accounting."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-slurm-result-first",
    )
    request_dir = tmp_path / "submitted" / "run-slurm-result-first"
    request_dir.mkdir(parents=True)
    request_path = request_dir / "bucket-00000.json"
    request = plan.request_for(plan.assignments[0])
    request_path.write_text(json.dumps(request))
    result_path = tmp_path / "results" / "run-slurm-result-first" / "bucket-00000.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(execute_request(request)))
    request_path.with_name("bucket-00000.json.slurm.json").write_text(
        json.dumps({"schema_version": 1, "status": "submitted", "slurm_job_id": "126", "result_path": str(result_path)})
    )

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("sacct should not run when the result envelope is already visible")

    events = reconcile_submitted_slurm_jobs(tmp_path / "submitted", tmp_path / "results", runner=forbidden_runner)

    assert [event.status for event in events] == ["completed"]
    assert (tmp_path / "done" / "run-slurm-result-first" / "bucket-00000.json").exists()
    assert (tmp_path / "done" / "run-slurm-result-first" / "bucket-00000.json.slurm.json").exists()


def test_run_manager_loop_logs_transient_slurm_reconcile_failure_and_continues(
    clean_registry,
    monkeypatch,
    tmp_path,
    capsys,
):
    """Persistent manager loops should log transient accounting failures and keep polling."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-slurm-sacct-down",
    )
    request_dir = tmp_path / "submitted" / "run-slurm-sacct-down"
    request_dir.mkdir(parents=True)
    request_path = request_dir / "bucket-00000.json"
    request_path.write_text(json.dumps(plan.request_for(plan.assignments[0])))
    result_path = tmp_path / "results" / "run-slurm-sacct-down" / "bucket-00000.json"
    request_path.with_name("bucket-00000.json.slurm.json").write_text(
        json.dumps({"schema_version": 1, "status": "submitted", "slurm_job_id": "127", "result_path": str(result_path)})
    )

    def failing_runner(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="sacct unavailable")

    def stop_after_first_tick(interval):
        assert interval == 0.01
        raise RuntimeError("stop-after-first-tick")

    monkeypatch.setattr(remote_execution_mod.time, "sleep", stop_after_first_tick)

    with pytest.raises(RuntimeError, match="stop-after-first-tick"):
        run_manager_loop(
            tmp_path / "queue",
            tmp_path / "results",
            scheduler="slurm",
            poll_interval=0.01,
            max_active_slurm_jobs=1,
            runner=failing_runner,
        )

    captured = capsys.readouterr()
    assert "CalledProcessError" in captured.err
    assert "sacct unavailable" in captured.err
    assert request_path.exists()
    assert not result_path.exists()


def test_reconcile_submitted_slurm_completed_without_result_fails_visible(clean_registry, tmp_path):
    """Completed Slurm jobs that write no envelope should become explicit failures."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/proto", scheduler="slurm")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-slurm-missing-result",
    )
    request_dir = tmp_path / "submitted" / "run-slurm-missing-result"
    request_dir.mkdir(parents=True)
    request_path = request_dir / "bucket-00000.json"
    request_path.write_text(json.dumps(plan.request_for(plan.assignments[0])))
    sidecar = request_path.with_name("bucket-00000.json.slurm.json")
    result_path = tmp_path / "results" / "run-slurm-missing-result" / "bucket-00000.json"
    sidecar.write_text(
        json.dumps({"schema_version": 1, "status": "submitted", "slurm_job_id": "125", "result_path": str(result_path)})
    )

    class FakeCompleted:
        stdout = "125|COMPLETED\n"

    events = reconcile_submitted_slurm_jobs(
        tmp_path / "submitted",
        tmp_path / "results",
        runner=lambda *args, **kwargs: FakeCompleted(),
    )

    assert [event.status for event in events] == ["failed"]
    envelope = json.loads(result_path.read_text())
    assert envelope["status"] == "failed"
    assert "COMPLETED_WITHOUT_RESULT" in envelope["error"]


def test_execute_request_file_writes_failed_envelope(clean_registry, tmp_path):
    """Request-file execution should leave a failed envelope before raising."""
    _register_remote_mock_tool(clean_registry)
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    request_path.write_text(
        json.dumps({"schema_version": 1, "tool_key": "missing-tool", "run_id": "r", "bucket_id": "b"})
    )

    with pytest.raises(ValueError):
        execute_request_file(request_path, output_path)

    envelope = json.loads(output_path.read_text())
    assert envelope["status"] == "failed"
    assert envelope["tool_key"] == "missing-tool"
    assert "missing-tool" in envelope["traceback"]


def test_execute_request_batch_rejects_unsafe_result_path_segments(clean_registry, tmp_path):
    """Batch workers must keep malformed request failures inside the results root."""
    _register_remote_mock_tool(clean_registry)
    request_path = tmp_path / "bad.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool_key": "remote-mock",
                "run_id": "run-safe",
                "bucket_id": "../escape",
                "item_indices": [0],
                "inputs": {"sequences": ["AA"]},
                "config": {"device": "cuda:0"},
            }
        )
    )

    with pytest.raises(RuntimeError, match=r"request\.bucket_id"):
        remote_execution_mod.execute_request_batch([request_path], tmp_path / "results")

    assert not (tmp_path / "escape.json").exists()
    failed_result = tmp_path / "results" / "failed" / "bad.json.json"
    envelope = json.loads(failed_result.read_text())
    assert envelope["status"] == "failed"
    assert "request.bucket_id" in envelope["error"]


def test_collect_plan_outputs_merges_node_results(clean_registry, tmp_path):
    """Collected result files should merge back in original input order."""
    _register_remote_mock_tool(clean_registry)
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty", weight=1.0, worker_device="cuda:0"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3", weight=1.0, worker_device="cuda:1"),
    ]
    plan = RemoteDispatchPlanner(nodes, bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA", "BB"]),
        run_id="run-6",
    )
    for assignment in plan.assignments:
        result = execute_request(plan.request_for(assignment))
        result_path = tmp_path / assignment.node.name / "run-6" / f"{assignment.bucket.bucket_id}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result))

    merged = collect_plan_outputs(plan, tmp_path)

    assert merged.results == ["AA:cuda:0:x", "BB:cuda:1:x"]
    assert merged.metadata["remote_dispatch"]["run_id"] == "run-6"


def test_remote_run_manifest_round_trips_and_collects_results(clean_registry, tmp_path):
    """Run manifests should survive process boundaries and collect results."""
    _register_remote_mock_tool(clean_registry)
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3", worker_device="cuda:1"),
    ]
    plan = RemoteDispatchPlanner(nodes, bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA", "BB"]),
        run_id="run-manifest",
    )
    manifest = RemoteRunManifest.from_plan(plan, nodes=nodes, bucket_size=1, local_results_dir=tmp_path)
    manifest_path = tmp_path / "run.manifest.json"

    write_remote_run_manifest(manifest, manifest_path)
    loaded = load_remote_run_manifest(manifest_path)

    assert loaded.to_dict() == manifest.to_dict()
    assert [request.bucket_id for request in loaded.requests] == ["bucket-00000", "bucket-00001"]
    for request in loaded.requests:
        result = execute_request(request.request)
        result_path = loaded.local_result_path(request)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result))

    merged = collect_remote_run_manifest(loaded)

    assert merged.results == ["AA:cuda:0:x", "BB:cuda:1:x"]
    assert merged.metadata["remote_dispatch"]["run_id"] == "run-manifest"


def test_launch_remote_run_preflights_starts_submits_waits_and_collects(clean_registry, tmp_path):
    """Launch should compose manager startup, manifest submission, wait, and collect."""
    _register_remote_mock_tool(clean_registry)
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3", worker_device="cuda:1"),
    ]
    events = []
    local_results_dir = tmp_path / "results"

    class FakeManager:
        def preflight_node(self, node, *, runner):
            del runner
            events.append(("preflight", node.name))
            return {"node": node.name, "ok": True, "checks": []}

        def start_node(self, node, *, poll_interval, slurm_group_size, runner):
            del runner
            events.append(("start", node.name, poll_interval, slurm_group_size))
            return f"started {node.name}"

    class FakeCompleted:
        stdout = ""

    def fake_runner(cmd, input=None, text=False, check=False, capture_output=False):
        del text, check, capture_output
        events.append(("runner", cmd[0], cmd[1]))
        payload = json.loads(input)
        for item in payload["requests"]:
            request = item["request"]
            result = execute_request(request)
            result_path = local_results_dir / cmd[1] / request["run_id"] / item["filename"]
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result))
        return FakeCompleted()

    result = launch_remote_run(
        "remote-mock",
        nodes,
        RemoteMockInput(sequences=["AA", "BB"]),
        RemoteMockConfig(device="remote"),
        bucket_size=1,
        local_results_dir=local_results_dir,
        manifest_path=tmp_path / "run.manifest.json",
        output_path=tmp_path / "output.json",
        run_id="run-launch",
        manager_poll_interval=2.0,
        slurm_group_size=7,
        wait_poll_interval=0.01,
        manager_client=FakeManager(),
        runner=fake_runner,
    )

    assert result.run_id == "run-launch"
    assert result.bucket_count == 2
    assert result.staged_count == 2
    assert result.managers == {"ty": "started ty", "ty3": "started ty3"}
    assert (tmp_path / "run.manifest.json").exists()
    assert json.loads((tmp_path / "output.json").read_text())["results"] == ["AA:cuda:0:x", "BB:cuda:1:x"]
    assert events[:4] == [
        ("preflight", "ty"),
        ("preflight", "ty3"),
        ("start", "ty", 2.0, 7),
        ("start", "ty3", 2.0, 7),
    ]
    assert ("runner", "ssh", "ty") in events
    assert ("runner", "ssh", "ty3") in events


def test_launch_remote_run_skips_completed_result_beside_running_request(clean_registry, tmp_path):
    """Launch should survive submit immediately after manager start when running already has a result."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(
        name="ty",
        host="ty",
        work_dir=str(tmp_path / "remote" / "ty"),
        worker_device="cuda:0",
        python=sys.executable,
    )
    local_results_dir = tmp_path / "results"
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        RemoteMockConfig(device="remote"),
        run_id="run-launch-running-result",
    )
    request = plan.request_for(plan.assignments[0])
    result_payload = execute_request(request)
    remote_running = Path(node.work_dir) / "running" / "run-launch-running-result" / "bucket-00000.json"
    remote_running.parent.mkdir(parents=True)
    remote_running.write_text(json.dumps(request, sort_keys=True) + "\n")
    remote_result = Path(node.work_dir) / "results" / "run-launch-running-result" / "bucket-00000.json"
    remote_result.parent.mkdir(parents=True)
    remote_result.write_text(json.dumps(result_payload, sort_keys=True) + "\n")
    local_result = local_results_dir / "ty" / "run-launch-running-result" / "bucket-00000.json"
    local_result.parent.mkdir(parents=True)
    local_result.write_text(json.dumps(result_payload, sort_keys=True) + "\n")
    events = []

    class FakeManager:
        def preflight_node(self, node, *, runner):
            del runner
            events.append(("preflight", node.name))
            return {"node": node.name, "ok": True, "checks": []}

        def start_node(self, node, *, poll_interval, slurm_group_size, runner):
            del poll_interval, slurm_group_size, runner
            events.append(("start", node.name))
            return f"started {node.name}"

    def fake_runner(cmd, input=None, text=False, check=False, capture_output=False):
        del capture_output
        events.append(("runner", cmd[0], cmd[1]))
        proc = subprocess.run(
            ["bash", "-lc", cmd[-1]],
            input=input,
            text=text,
            capture_output=True,
            check=False,
        )
        if check and proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
        return proc

    result = launch_remote_run(
        "remote-mock",
        [node],
        RemoteMockInput(sequences=["AA"]),
        RemoteMockConfig(device="remote"),
        bucket_size=1,
        local_results_dir=local_results_dir,
        manifest_path=tmp_path / "run.manifest.json",
        output_path=tmp_path / "output.json",
        run_id="run-launch-running-result",
        wait_poll_interval=0.01,
        manager_client=FakeManager(),
        runner=fake_runner,
    )

    assert result.run_id == "run-launch-running-result"
    assert json.loads((tmp_path / "output.json").read_text())["results"] == ["AA:cuda:0:x"]
    assert events == [("preflight", "ty"), ("start", "ty"), ("runner", "ssh", "ty")]
    assert remote_running.exists()
    assert not (Path(node.work_dir) / "queue" / "run-launch-running-result" / "bucket-00000.json").exists()


def test_launch_remote_run_rejects_existing_output_before_remote_side_effects(clean_registry, tmp_path):
    """Launch should not preflight, write a manifest, or submit when output exists."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    manifest_path = tmp_path / "run.manifest.json"
    output_path = tmp_path / "output.json"
    output_path.write_text("sentinel\n")

    class FakeManager:
        def preflight_node(self, node, *, runner):
            raise AssertionError("preflight should not run")

        def start_node(self, node, *, poll_interval, slurm_group_size, runner):
            raise AssertionError("manager start should not run")

    def fake_runner(*args, **kwargs):
        raise AssertionError("submission should not run")

    with pytest.raises(FileExistsError, match="Remote run output already exists"):
        launch_remote_run(
            "remote-mock",
            [node],
            RemoteMockInput(sequences=["AA"]),
            RemoteMockConfig(device="remote"),
            bucket_size=1,
            local_results_dir=tmp_path / "results",
            manifest_path=manifest_path,
            output_path=output_path,
            manager_client=FakeManager(),
            runner=fake_runner,
        )

    assert output_path.read_text() == "sentinel\n"
    assert not manifest_path.exists()


def test_launch_remote_run_overwrite_output_allows_existing_output(clean_registry, tmp_path):
    """Explicit overwrite_output should allow launch to replace an existing output."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    local_results_dir = tmp_path / "results"
    output_path = tmp_path / "output.json"
    output_path.write_text("sentinel\n")
    events = []

    class FakeManager:
        def preflight_node(self, node, *, runner):
            del runner
            return {"node": node.name, "ok": True, "checks": []}

        def start_node(self, node, *, poll_interval, slurm_group_size, runner):
            del poll_interval, slurm_group_size, runner
            return f"started {node.name}"

    class FakeCompleted:
        stdout = ""

    def fake_runner(cmd, input=None, text=False, check=False, capture_output=False):
        del text, check, capture_output
        events.append(("runner", cmd[0], cmd[1]))
        payload = json.loads(input)
        for item in payload["requests"]:
            request = item["request"]
            result = execute_request(request)
            result_path = local_results_dir / cmd[1] / request["run_id"] / item["filename"]
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result))
        return FakeCompleted()

    launch_remote_run(
        "remote-mock",
        [node],
        RemoteMockInput(sequences=["AA"]),
        RemoteMockConfig(device="remote"),
        bucket_size=1,
        local_results_dir=local_results_dir,
        manifest_path=tmp_path / "run.manifest.json",
        output_path=output_path,
        run_id="run-overwrite-output",
        overwrite_output=True,
        wait_poll_interval=0.01,
        manager_client=FakeManager(),
        runner=fake_runner,
    )

    assert json.loads(output_path.read_text())["results"] == ["AA:cuda:0:x"]
    assert ("runner", "ssh", "ty") in events


def test_remote_run_diagnostics_backlog_loads_counts_visible_backlog() -> None:
    """Diagnostics backlog loads should count only pending manager states."""
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3"),
    ]
    calls = []

    class FakeManager:
        def diagnostics_node(self, node, *, timeout_sec, runner):
            calls.append((node.name, timeout_sec, runner))
            counts = {"queue": 2, "running": 3, "submitted": 5, "done": 99} if node.name == "ty" else {}
            return {"node": node.name, "counts": counts}

    diagnostics, loads = remote_run_diagnostics_backlog_loads(
        nodes,
        manager_client=FakeManager(),
        timeout_sec=12.0,
        runner="runner",
    )

    assert diagnostics == {
        "ty": {"node": "ty", "counts": {"queue": 2, "running": 3, "submitted": 5, "done": 99}},
        "ty3": {"node": "ty3", "counts": {}},
    }
    assert loads == {"ty": 10.0, "ty3": 0.0}
    assert calls == [("ty", 12.0, "runner"), ("ty3", 12.0, "runner")]


def test_remote_dispatch_planner_rejects_unsafe_run_id(clean_registry) -> None:
    """User-provided run_id values should not become nested remote/local paths."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")

    with pytest.raises(ValueError, match="run_id"):
        RemoteDispatchPlanner([node], bucket_size=1).build_plan(
            "remote-mock",
            RemoteMockInput(sequences=["AA"]),
            run_id="../escape",
        )


@pytest.mark.parametrize(
    ("mutator", "pattern"),
    [
        (lambda data: data.update({"run_id": "../escape"}), "run_id"),
        (lambda data: data["requests"][0].update({"node_name": "../escape"}), "node_name"),
        (lambda data: data["requests"][0].update({"bucket_id": "../escape"}), "bucket_id"),
        (lambda data: data["requests"][0].update({"node_name": "missing"}), "unknown node"),
        (lambda data: data["requests"].append(dict(data["requests"][0])), "duplicate bucket"),
        (lambda data: data["requests"][0]["request"].update({"run_id": "other-run"}), "identity mismatch"),
        (lambda data: data["requests"][0]["request"].update({"bucket_id": "../escape"}), "identity mismatch"),
    ],
)
def test_remote_run_manifest_rejects_unsafe_or_mismatched_path_identity(
    clean_registry,
    tmp_path,
    mutator,
    pattern,
) -> None:
    """Loaded manifests must reject path traversal and inner/outer identity drift."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-safe",
    )
    manifest = RemoteRunManifest.from_plan(plan, nodes=[node], bucket_size=1, local_results_dir=tmp_path)
    data = json.loads(json.dumps(manifest.to_dict()))
    mutator(data)

    with pytest.raises(ValueError, match=pattern):
        RemoteRunManifest.from_dict(data)


def test_launch_remote_run_can_use_diagnostics_backlog_for_planning(clean_registry, tmp_path):
    """Launch can seed planning load from existing remote manager backlog."""
    _register_remote_mock_tool(clean_registry)
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty", weight=1.0, worker_device="cuda:0"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3", weight=1.0, worker_device="cuda:1"),
    ]
    events = []
    local_results_dir = tmp_path / "results"

    class FakeManager:
        def preflight_node(self, node, *, timeout_sec, runner):
            del runner
            events.append(("preflight", node.name, timeout_sec))
            return {"node": node.name, "ok": True, "checks": []}

        def start_node(self, node, *, poll_interval, slurm_group_size, timeout_sec, runner):
            del poll_interval, slurm_group_size, runner
            events.append(("start", node.name, timeout_sec))
            return f"started {node.name}"

        def diagnostics_node(self, node, *, timeout_sec, runner):
            del runner
            backlog = {"queue": 10, "running": 0, "submitted": 0} if node.name == "ty" else {}
            events.append(("diagnostics", node.name, timeout_sec, backlog))
            return {"node": node.name, "counts": backlog}

    class FakeCompleted:
        stdout = ""

    def fake_runner(cmd, input=None, text=False, check=False, capture_output=False):
        del text, check, capture_output
        events.append(("runner", cmd[0], cmd[1]))
        payload = json.loads(input)
        for item in payload["requests"]:
            request = item["request"]
            result = execute_request(request)
            result_path = local_results_dir / cmd[1] / request["run_id"] / item["filename"]
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result))
        return FakeCompleted()

    result = launch_remote_run(
        "remote-mock",
        nodes,
        RemoteMockInput(sequences=["A", "B"]),
        RemoteMockConfig(device="remote"),
        bucket_size=1,
        local_results_dir=local_results_dir,
        manifest_path=tmp_path / "run.manifest.json",
        output_path=tmp_path / "output.json",
        run_id="run-backlog",
        use_diagnostics_backlog=True,
        timeout_sec=30.0,
        remote_check_timeout_sec=12.5,
        wait_poll_interval=0.01,
        manager_client=FakeManager(),
        runner=fake_runner,
    )

    manifest = load_remote_run_manifest(tmp_path / "run.manifest.json")
    assert [request.node_name for request in manifest.requests] == ["ty3", "ty3"]
    assert result.initial_node_loads == {"ty": 10.0, "ty3": 0.0}
    assert ("preflight", "ty", 12.5) in events
    assert ("start", "ty", 12.5) in events
    assert ("diagnostics", "ty", 12.5, {"queue": 10, "running": 0, "submitted": 0}) in events


def test_resume_remote_run_manifest_reuses_existing_manifest_without_replanning(clean_registry, tmp_path):
    """Resume should drive an existing manifest through managers, submit, wait, and collect."""
    _register_remote_mock_tool(clean_registry)
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3", worker_device="cuda:1"),
    ]
    plan = RemoteDispatchPlanner(nodes, bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA", "BB"]),
        RemoteMockConfig(device="remote"),
        run_id="run-resume",
    )
    manifest = RemoteRunManifest.from_plan(plan, nodes=nodes, bucket_size=1, local_results_dir=tmp_path / "results")
    manifest_path = tmp_path / "run.manifest.json"
    write_remote_run_manifest(manifest, manifest_path)
    original_manifest_text = manifest_path.read_text()
    events = []

    class FakeManager:
        def preflight_node(self, node, *, timeout_sec, runner):
            del runner
            events.append(("preflight", node.name, timeout_sec))
            return {"node": node.name, "ok": True, "checks": []}

        def start_node(self, node, *, poll_interval, slurm_group_size, timeout_sec, runner):
            del runner
            events.append(("start", node.name, poll_interval, slurm_group_size, timeout_sec))
            return f"started {node.name}"

    class FakeCompleted:
        stdout = ""

    def fake_runner(cmd, input=None, text=False, check=False, capture_output=False):
        del text, check, capture_output
        events.append(("runner", cmd[0], cmd[1]))
        payload = json.loads(input)
        for item in payload["requests"]:
            request = item["request"]
            result = execute_request(request)
            result_path = Path(manifest.local_results_dir) / cmd[1] / request["run_id"] / item["filename"]
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result))
        return FakeCompleted()

    result = resume_remote_run_manifest(
        load_remote_run_manifest(manifest_path),
        manifest_path=manifest_path,
        output_path=tmp_path / "output.json",
        manager_poll_interval=2.0,
        slurm_group_size=7,
        timeout_sec=30.0,
        remote_check_timeout_sec=12.5,
        wait_poll_interval=0.01,
        manager_client=FakeManager(),
        runner=fake_runner,
    )

    assert result.run_id == "run-resume"
    assert result.manifest_path == str(manifest_path)
    assert result.bucket_count == 2
    assert result.staged_count == 2
    assert result.managers == {"ty": "started ty", "ty3": "started ty3"}
    assert manifest_path.read_text() == original_manifest_text
    assert json.loads((tmp_path / "output.json").read_text())["results"] == ["AA:cuda:0:x", "BB:cuda:1:x"]
    assert events[:4] == [
        ("preflight", "ty", 12.5),
        ("preflight", "ty3", 12.5),
        ("start", "ty", 2.0, 7, 12.5),
        ("start", "ty3", 2.0, 7, 12.5),
    ]


def test_resume_remote_run_manifest_skips_completed_result_beside_running_request(clean_registry, tmp_path):
    """Resume should not fail when a completed result is visible before running is reconciled."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(
        name="ty",
        host="ty",
        work_dir=str(tmp_path / "remote" / "ty"),
        worker_device="cuda:0",
        python=sys.executable,
    )
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        RemoteMockConfig(device="remote"),
        run_id="run-resume-running-result",
    )
    manifest = RemoteRunManifest.from_plan(
        plan,
        nodes=[node],
        bucket_size=1,
        local_results_dir=tmp_path / "results",
    )
    manifest_path = tmp_path / "run.manifest.json"
    write_remote_run_manifest(manifest, manifest_path)
    request = manifest.requests[0].request
    result_payload = execute_request(request)
    remote_running = Path(node.work_dir) / "running" / manifest.run_id / "bucket-00000.json"
    remote_running.parent.mkdir(parents=True)
    remote_running.write_text(json.dumps(request, sort_keys=True) + "\n")
    remote_result = Path(node.work_dir) / "results" / manifest.run_id / "bucket-00000.json"
    remote_result.parent.mkdir(parents=True)
    remote_result.write_text(json.dumps(result_payload, sort_keys=True) + "\n")
    local_result = manifest.local_result_path(manifest.requests[0])
    local_result.parent.mkdir(parents=True)
    local_result.write_text(json.dumps(result_payload, sort_keys=True) + "\n")
    events = []

    class FakeManager:
        def preflight_node(self, node, *, runner):
            del runner
            events.append(("preflight", node.name))
            return {"node": node.name, "ok": True, "checks": []}

        def start_node(self, node, *, poll_interval, slurm_group_size, runner):
            del poll_interval, slurm_group_size, runner
            events.append(("start", node.name))
            return f"started {node.name}"

    def fake_runner(cmd, input=None, text=False, check=False, capture_output=False):
        del capture_output
        events.append(("runner", cmd[0], cmd[1]))
        proc = subprocess.run(
            ["bash", "-lc", cmd[-1]],
            input=input,
            text=text,
            capture_output=True,
            check=False,
        )
        if check and proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
        return proc

    result = resume_remote_run_manifest(
        load_remote_run_manifest(manifest_path),
        manifest_path=manifest_path,
        output_path=tmp_path / "output.json",
        wait_poll_interval=0.01,
        manager_client=FakeManager(),
        runner=fake_runner,
    )

    assert result.run_id == "run-resume-running-result"
    assert json.loads((tmp_path / "output.json").read_text())["results"] == ["AA:cuda:0:x"]
    assert events == [("preflight", "ty"), ("start", "ty"), ("runner", "ssh", "ty")]
    assert remote_running.exists()
    assert not (Path(node.work_dir) / "queue" / manifest.run_id / "bucket-00000.json").exists()


def test_resume_remote_run_manifest_rejects_existing_output_before_remote_side_effects(clean_registry, tmp_path):
    """Resume should not preflight, submit, or rewrite evidence when output exists."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        RemoteMockConfig(device="remote"),
        run_id="run-resume-output-exists",
    )
    manifest = RemoteRunManifest.from_plan(plan, nodes=[node], bucket_size=1, local_results_dir=tmp_path / "results")
    manifest_path = tmp_path / "run.manifest.json"
    write_remote_run_manifest(manifest, manifest_path)
    original_manifest_text = manifest_path.read_text()
    output_path = tmp_path / "output.json"
    output_path.write_text("sentinel\n")

    class FakeManager:
        def preflight_node(self, node, *, runner):
            raise AssertionError("preflight should not run")

        def start_node(self, node, *, poll_interval, slurm_group_size, runner):
            raise AssertionError("manager start should not run")

    def fake_runner(*args, **kwargs):
        raise AssertionError("submission should not run")

    with pytest.raises(FileExistsError, match="Remote run output already exists"):
        resume_remote_run_manifest(
            load_remote_run_manifest(manifest_path),
            manifest_path=manifest_path,
            output_path=output_path,
            manager_client=FakeManager(),
            runner=fake_runner,
        )

    assert output_path.read_text() == "sentinel\n"
    assert manifest_path.read_text() == original_manifest_text


def test_launch_remote_run_preflight_failure_stops_before_manifest(clean_registry, tmp_path):
    """Launch should not submit or write a manifest when preflight fails."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")

    class FakeManager:
        def preflight_node(self, node, *, runner):
            del runner
            return {"node": node.name, "ok": False, "checks": [{"name": "import proto_tools", "ok": False}]}

        def start_node(self, node, *, poll_interval, slurm_group_size, runner):
            raise AssertionError("manager start should not run")

    def fake_runner(*args, **kwargs):
        raise AssertionError("submission should not run")

    with pytest.raises(ValueError, match="Remote preflight failed"):
        launch_remote_run(
            "remote-mock",
            [node],
            RemoteMockInput(sequences=["AA"]),
            RemoteMockConfig(device="remote"),
            bucket_size=1,
            local_results_dir=tmp_path / "results",
            manifest_path=tmp_path / "run.manifest.json",
            output_path=tmp_path / "output.json",
            manager_client=FakeManager(),
            runner=fake_runner,
        )

    assert not (tmp_path / "run.manifest.json").exists()
    assert not (tmp_path / "output.json").exists()


def test_launch_remote_run_background_rsync_failure_stops_wait(clean_registry, tmp_path):
    """Background launch should surface a stopped rsync loop instead of timing out."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    events = []
    log_path = tmp_path / "rsync.log"
    log_path.write_text("rsync failed\n")

    class FakeManager:
        def preflight_node(self, node, *, runner):
            del runner
            return {"node": node.name, "ok": True, "checks": []}

        def start_node(self, node, *, poll_interval, slurm_group_size, runner):
            del poll_interval, slurm_group_size, runner
            return f"started {node.name}"

    class FakeRsync:
        def start(self, nodes, local_results_dir, *, pid_path, log_path, interval_sec, runner):
            events.append(("start-rsync", [node.name for node in nodes], local_results_dir, pid_path, log_path, interval_sec))
            return "started 123"

        def status(self, pid_path, *, runner):
            events.append(("status-rsync", pid_path))
            return "stale 123"

    class FakeCompleted:
        stdout = ""

    def fake_runner(cmd, input=None, text=False, check=False, capture_output=False):
        del input, text, check, capture_output
        events.append(("runner", cmd[0], cmd[1]))
        return FakeCompleted()

    with pytest.raises(RuntimeError, match=r"rsync pull loop is not running.*rsync failed"):
        launch_remote_run(
            "remote-mock",
            [node],
            RemoteMockInput(sequences=["AA"]),
            RemoteMockConfig(device="remote"),
            bucket_size=1,
            local_results_dir=tmp_path / "results",
            manifest_path=tmp_path / "run.manifest.json",
            output_path=tmp_path / "output.json",
            run_id="run-background-rsync",
            rsync_mode="background",
            rsync_pid_path=tmp_path / "rsync.pid",
            rsync_log_path=log_path,
            manager_client=FakeManager(),
            rsync_client=FakeRsync(),
            runner=fake_runner,
        )

    assert ("status-rsync", tmp_path / "rsync.pid") in events


def test_submit_remote_run_manifest_stages_node_batches(clean_registry, tmp_path):
    """Manifest submission should stage requests grouped by node."""
    _register_remote_mock_tool(clean_registry)
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3", worker_device="cuda:1"),
    ]
    plan = RemoteDispatchPlanner(nodes, bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA", "BB"]),
        run_id="run-submit-manifest",
    )
    manifest = RemoteRunManifest.from_plan(plan, nodes=nodes, bucket_size=1, local_results_dir=tmp_path)
    calls = []

    def fake_runner(cmd, input=None, text=False, check=False):
        calls.append({"cmd": cmd, "input": input, "text": text, "check": check})

    staged = submit_remote_run_manifest(manifest, runner=fake_runner)

    assert staged == [
        manifest.node_for_request(request).queue_root / "run-submit-manifest" / f"{request.bucket_id}.json"
        for request in manifest.requests
    ]
    assert [call["cmd"][:2] for call in calls] == [["ssh", "ty"], ["ssh", "ty3"]]
    assert all(call["text"] is True and call["check"] is True for call in calls)
    payloads = [json.loads(call["input"]) for call in calls]
    assert [[item["filename"] for item in payload["requests"]] for payload in payloads] == [
        ["bucket-00000.json"],
        ["bucket-00001.json"],
    ]

    calls.clear()
    selected = submit_remote_run_manifest(
        manifest,
        bucket_ids={"bucket-00001"},
        runner=fake_runner,
    )

    assert selected == [nodes[1].queue_root / "run-submit-manifest" / "bucket-00001.json"]
    assert [call["cmd"][:2] for call in calls] == [["ssh", "ty3"]]
    assert [item["filename"] for item in json.loads(calls[0]["input"])["requests"]] == [
        "bucket-00001.json"
    ]


def _run_stage_many_locally(root: Path, request: dict[str, object]):
    """Run the remote staging script against a local state root."""
    payload = {
        "schema_version": 1,
        "run_id": request["run_id"],
        "requests": [{"filename": f"{request['bucket_id']}.json", "request": request}],
    }
    root.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, "-c", RemoteSubmissionClient._STAGE_MANY_PY, str(root), "--summary"],
        input=json.dumps(payload, sort_keys=True) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("state", ["queue", "submitted", "done"])
def test_manifest_submit_skips_identical_request_already_in_state(clean_registry, tmp_path, state):
    """Manifest submit should be idempotent for already staged or completed requests."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-idempotent",
    )
    request = plan.request_for(plan.assignments[0])
    existing = tmp_path / state / "run-idempotent" / "bucket-00000.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps(request, sort_keys=True) + "\n")

    proc = _run_stage_many_locally(tmp_path / "queue" / "run-idempotent", request)

    assert proc.returncode == 0
    assert json.loads(proc.stdout) == {
        "skipped": [
            {
                "filename": "bucket-00000.json",
                "path": str(existing),
                "state": state,
            }
        ],
        "staged": [],
    }
    assert json.loads(existing.read_text()) == request


def test_manifest_submit_skips_matching_completed_result(clean_registry, tmp_path):
    """Manifest submit should not requeue a bucket that already has a matching completed result."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-result-idempotent",
    )
    request = plan.request_for(plan.assignments[0])
    result = tmp_path / "results" / "run-result-idempotent" / "bucket-00000.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(execute_request(request), sort_keys=True) + "\n")

    proc = _run_stage_many_locally(tmp_path / "queue" / "run-result-idempotent", request)

    assert proc.returncode == 0
    assert json.loads(proc.stdout) == {
        "skipped": [
            {
                "filename": "bucket-00000.json",
                "path": str(result),
                "state": "result",
            }
        ],
        "staged": [],
    }
    assert not (tmp_path / "queue" / "run-result-idempotent" / "bucket-00000.json").exists()


def test_manifest_submit_skips_matching_running_request_without_requeue(clean_registry, tmp_path):
    """Exact resume should wait for a claimed request without staging a duplicate."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-running",
    )
    request = plan.request_for(plan.assignments[0])
    running = tmp_path / "running" / "run-running" / "bucket-00000.json"
    running.parent.mkdir(parents=True)
    running.write_text(json.dumps(request, sort_keys=True) + "\n")

    proc = _run_stage_many_locally(tmp_path / "queue" / "run-running", request)

    assert proc.returncode == 0
    assert json.loads(proc.stdout) == {
        "skipped": [
            {
                "filename": "bucket-00000.json",
                "path": str(running),
                "state": "running",
            }
        ],
        "staged": [],
    }
    assert not (tmp_path / "queue" / "run-running" / "bucket-00000.json").exists()


def test_manifest_submit_skips_running_request_with_matching_completed_result(clean_registry, tmp_path):
    """Manifest submit should tolerate a manager-restart race when the result is already complete."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-running-result",
    )
    request = plan.request_for(plan.assignments[0])
    running = tmp_path / "running" / "run-running-result" / "bucket-00000.json"
    running.parent.mkdir(parents=True)
    running.write_text(json.dumps(request, sort_keys=True) + "\n")
    result = tmp_path / "results" / "run-running-result" / "bucket-00000.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(execute_request(request), sort_keys=True) + "\n")

    proc = _run_stage_many_locally(tmp_path / "queue" / "run-running-result", request)

    assert proc.returncode == 0
    assert json.loads(proc.stdout) == {
        "skipped": [
            {
                "filename": "bucket-00000.json",
                "path": str(result),
                "state": "running_result",
            }
        ],
        "staged": [],
    }
    assert running.exists()
    assert not (tmp_path / "queue" / "run-running-result" / "bucket-00000.json").exists()


def test_manifest_submit_rejects_running_request_with_matching_failed_result(clean_registry, tmp_path):
    """Manifest submit should preserve failed-result evidence beside a running request."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-running-failed-result",
    )
    request = plan.request_for(plan.assignments[0])
    running = tmp_path / "running" / "run-running-failed-result" / "bucket-00000.json"
    running.parent.mkdir(parents=True)
    running.write_text(json.dumps(request, sort_keys=True) + "\n")
    failed_result = tmp_path / "results" / "run-running-failed-result" / "bucket-00000.json"
    failed_result.parent.mkdir(parents=True)
    failed_result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "tool_key": "remote-mock",
                "run_id": "run-running-failed-result",
                "bucket_id": "bucket-00000",
                "item_indices": [0],
                "error": "boom",
            },
            sort_keys=True,
        )
        + "\n"
    )

    proc = _run_stage_many_locally(tmp_path / "queue" / "run-running-failed-result", request)

    assert proc.returncode != 0
    assert "request previously failed" in proc.stderr
    assert json.loads(failed_result.read_text())["error"] == "boom"
    assert running.exists()
    assert not (tmp_path / "queue" / "run-running-failed-result" / "bucket-00000.json").exists()


def test_manifest_submit_rejects_different_running_request_even_with_result(clean_registry, tmp_path):
    """A completed result should not hide a conflicting running request payload."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-running-conflict",
    )
    request = plan.request_for(plan.assignments[0])
    conflict = dict(request)
    conflict["inputs"] = {"sequences": ["DIFFERENT"]}
    running = tmp_path / "running" / "run-running-conflict" / "bucket-00000.json"
    running.parent.mkdir(parents=True)
    running.write_text(json.dumps(conflict, sort_keys=True) + "\n")
    result = tmp_path / "results" / "run-running-conflict" / "bucket-00000.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(execute_request(request), sort_keys=True) + "\n")

    proc = _run_stage_many_locally(tmp_path / "queue" / "run-running-conflict", request)

    assert proc.returncode != 0
    assert "existing running request differs from manifest" in proc.stderr
    assert json.loads(running.read_text()) == conflict
    assert not (tmp_path / "queue" / "run-running-conflict" / "bucket-00000.json").exists()


def test_manifest_submit_rejects_failed_request_state(clean_registry, tmp_path):
    """Manifest submit should not silently requeue a previously failed request."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-failed-state",
    )
    request = plan.request_for(plan.assignments[0])
    failed = tmp_path / "failed" / "run-failed-state" / "bucket-00000.json"
    failed.parent.mkdir(parents=True)
    failed.write_text(json.dumps(request, sort_keys=True) + "\n")

    proc = _run_stage_many_locally(tmp_path / "queue" / "run-failed-state", request)

    assert proc.returncode != 0
    assert "request previously failed" in proc.stderr
    assert not (tmp_path / "queue" / "run-failed-state" / "bucket-00000.json").exists()


def test_manifest_submit_rejects_matching_failed_result(clean_registry, tmp_path):
    """Manifest submit should preserve an existing failed result instead of requeueing."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-failed-result",
    )
    request = plan.request_for(plan.assignments[0])
    failed_result = tmp_path / "results" / "run-failed-result" / "bucket-00000.json"
    failed_result.parent.mkdir(parents=True)
    failed_result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "tool_key": "remote-mock",
                "run_id": "run-failed-result",
                "bucket_id": "bucket-00000",
                "item_indices": [0],
                "error": "boom",
            },
            sort_keys=True,
        )
        + "\n"
    )

    proc = _run_stage_many_locally(tmp_path / "queue" / "run-failed-result", request)

    assert proc.returncode != 0
    assert "request previously failed" in proc.stderr
    assert json.loads(failed_result.read_text())["error"] == "boom"
    assert not (tmp_path / "queue" / "run-failed-result" / "bucket-00000.json").exists()


def test_manifest_submit_rejects_different_existing_request(clean_registry, tmp_path):
    """Manifest submit should fail visibly instead of overwriting a different request."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-conflict",
    )
    request = plan.request_for(plan.assignments[0])
    conflict = dict(request)
    conflict["inputs"] = {"sequences": ["DIFFERENT"]}
    existing = tmp_path / "queue" / "run-conflict" / "bucket-00000.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps(conflict, sort_keys=True) + "\n")

    proc = _run_stage_many_locally(tmp_path / "queue" / "run-conflict", request)

    assert proc.returncode != 0
    assert "existing queue request differs from manifest" in proc.stderr
    assert json.loads(existing.read_text()) == conflict


def test_wait_remote_run_manifest_raises_on_failed_envelope(clean_registry, tmp_path):
    """Manifest wait should fail as soon as a failed bucket is mirrored."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-manifest-fail",
    )
    manifest = RemoteRunManifest.from_plan(plan, nodes=[node], bucket_size=1, local_results_dir=tmp_path)
    failed_path = manifest.local_result_path(manifest.requests[0])
    failed_path.parent.mkdir(parents=True)
    failed_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "tool_key": "remote-mock",
                "run_id": "run-manifest-fail",
                "bucket_id": "bucket-00000",
                "item_indices": [0],
                "error": "boom",
            }
        )
    )

    with pytest.raises(RuntimeError, match=r"ty/bucket-00000.*boom"):
        wait_remote_run_manifest(manifest, pull_results=False, poll_interval=0.01, timeout_sec=0.1)


def test_wait_remote_run_manifest_rejects_non_completed_envelope(clean_registry, tmp_path):
    """Manifest wait should not treat a non-completed result envelope as done."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-manifest-submitted",
    )
    manifest = RemoteRunManifest.from_plan(plan, nodes=[node], bucket_size=1, local_results_dir=tmp_path)
    result_path = manifest.local_result_path(manifest.requests[0])
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "submitted",
                "tool_key": "remote-mock",
                "run_id": "run-manifest-submitted",
                "bucket_id": "bucket-00000",
                "item_indices": [0],
            }
        )
    )

    with pytest.raises(RuntimeError, match=r"non-completed result.*status='submitted'"):
        wait_remote_run_manifest(manifest, pull_results=False, poll_interval=0.01, timeout_sec=0.1)


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("run_id", "wrong-run", "run_id expected 'run-manifest-identity' got 'wrong-run'"),
        ("bucket_id", "wrong-bucket", "bucket_id expected 'bucket-00000' got 'wrong-bucket'"),
        ("tool_key", "wrong-tool", "tool_key expected 'remote-mock' got 'wrong-tool'"),
        ("item_indices", [1], r"item_indices expected \[0\] got \[1\]"),
    ],
)
def test_wait_remote_run_manifest_rejects_wrong_identity(clean_registry, tmp_path, field, value, pattern):
    """Manifest wait should reject result files that belong to a different bucket."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-manifest-identity",
    )
    manifest = RemoteRunManifest.from_plan(plan, nodes=[node], bucket_size=1, local_results_dir=tmp_path)
    result = execute_request(manifest.requests[0].request)
    result[field] = value
    result_path = manifest.local_result_path(manifest.requests[0])
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(result))

    with pytest.raises(RuntimeError, match=pattern):
        wait_remote_run_manifest(manifest, pull_results=False, poll_interval=0.01, timeout_sec=0.1)


def test_collect_remote_run_manifest_rejects_wrong_identity(clean_registry, tmp_path):
    """Collect should not merge a completed envelope from a different bucket identity."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-collect-identity",
    )
    manifest = RemoteRunManifest.from_plan(plan, nodes=[node], bucket_size=1, local_results_dir=tmp_path)
    result = execute_request(manifest.requests[0].request)
    result["run_id"] = "wrong-run"
    result_path = manifest.local_result_path(manifest.requests[0])
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(result))

    with pytest.raises(RuntimeError, match="wrong identity"):
        collect_remote_run_manifest(manifest)


def test_wait_remote_run_manifest_checks_background_rsync_health(clean_registry, tmp_path):
    """Background wait should fail fast when the local rsync loop is no longer running."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-rsync-health",
    )
    manifest = RemoteRunManifest.from_plan(plan, nodes=[node], bucket_size=1, local_results_dir=tmp_path / "results")
    log_path = tmp_path / "rsync.log"
    log_path.write_text("old line\nfatal rsync error\n")
    calls = []

    class FakeRsyncClient:
        def status(self, pid_path, *, runner):
            calls.append((pid_path, runner))
            return "stale 123"

    with pytest.raises(RuntimeError, match=r"rsync pull loop is not running[\s\S]*fatal rsync error"):
        wait_remote_run_manifest(
            manifest,
            pull_results=False,
            poll_interval=0.01,
            timeout_sec=10,
            rsync_pid_path=tmp_path / "rsync.pid",
            rsync_log_path=log_path,
            rsync_client=FakeRsyncClient(),
        )

    assert calls == [(tmp_path / "rsync.pid", subprocess.run)]


def test_wait_remote_run_manifest_retries_transient_inline_rsync_failure(
    clean_registry,
    tmp_path,
    caplog,
):
    """One SSH/rsync transport failure should stay visible and retry within the wait timeout."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-inline-rsync-retry",
    )
    manifest = RemoteRunManifest.from_plan(
        plan,
        nodes=[node],
        bucket_size=1,
        local_results_dir=tmp_path / "results",
    )
    calls = []

    def flaky_runner(cmd, *, check):
        calls.append((cmd, check))
        if len(calls) == 1:
            raise subprocess.CalledProcessError(255, cmd, stderr="connection closed")
        result_path = manifest.local_result_path(manifest.requests[0])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(execute_request(manifest.requests[0].request)))
        return subprocess.CompletedProcess(cmd, 0)

    wait_remote_run_manifest(
        manifest,
        poll_interval=0.01,
        timeout_sec=1,
        runner=flaky_runner,
    )

    assert len(calls) == 2
    assert "rsync pull attempt failed" in caplog.text


def test_wait_remote_run_manifest_follows_poll_hook_reassignment(clean_registry, tmp_path):
    """One launcher must keep waiting on the manifest returned by its rebalance hook."""
    _register_remote_mock_tool(clean_registry)
    original_node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    target_node = RemoteNode(name="f101", host="f101", work_dir="/remote/f101", worker_device="cuda:0")
    tool_input = RemoteMockInput(sequences=["AA"])
    original_plan = RemoteDispatchPlanner([original_node], bucket_size=1).build_plan(
        "remote-mock",
        tool_input,
        run_id="run-live-rebalance",
    )
    target_plan = RemoteDispatchPlanner([target_node], bucket_size=1).build_plan(
        "remote-mock",
        tool_input,
        run_id="run-live-rebalance",
    )
    original = RemoteRunManifest.from_plan(
        original_plan,
        nodes=[original_node],
        bucket_size=1,
        local_results_dir=tmp_path,
    )
    target = RemoteRunManifest.from_plan(
        target_plan,
        nodes=[target_node],
        bucket_size=1,
        local_results_dir=tmp_path,
    )
    calls = []

    def move_once(current):
        calls.append(current)
        result_path = target.local_result_path(target.requests[0])
        result_path.parent.mkdir(parents=True)
        result_path.write_text(json.dumps(execute_request(target.requests[0].request)))
        return target

    completed = wait_remote_run_manifest(
        original,
        pull_results=False,
        poll_interval=0.01,
        timeout_sec=1,
        poll_hook=move_once,
    )

    assert calls == [original]
    assert completed == target


def test_remote_tool_dispatcher_routes_explicit_remote_device(clean_registry, tmp_path):
    """RemoteToolDispatcher should plug into ToolRegistry for device='remote' calls."""
    run_remote_mock = _register_remote_mock_tool(clean_registry)
    nodes = [RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")]

    class InlineSubmitter:
        def submit_plan(self, plan, runner=None):
            del runner
            for assignment in plan.assignments:
                result = execute_request(plan.request_for(assignment))
                result_path = tmp_path / assignment.node.name / plan.run_id / f"{assignment.bucket.bucket_id}.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(result))

    with RemoteToolDispatcher(
        nodes,
        bucket_size=1,
        local_results_dir=tmp_path,
        submission_client=InlineSubmitter(),
        pull_results=False,
        poll_interval=0.01,
    ) as dispatcher:
        result = run_remote_mock(RemoteMockInput(sequences=["AA", "BB"]), RemoteMockConfig(device="remote"))

    assert result.results == ["AA:cuda:0:x", "BB:cuda:0:x"]
    assert result.tool_id == "remote-mock"
    assert dispatcher.last_plan is not None
    assert dispatcher.last_plan.run_id.startswith("remote-mock-")
    assert ToolRegistry.dispatch_backend_configured() is False


def test_dispatcher_raises_immediately_on_failed_envelope(clean_registry, tmp_path):
    """Dispatcher waiting should fail as soon as a failed envelope is visible."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA", "BB"]),
        run_id="run-fail-fast",
    )
    failed_path = tmp_path / "ty" / "run-fail-fast" / "bucket-00000.json"
    failed_path.parent.mkdir(parents=True)
    failed_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "run_id": "run-fail-fast",
                "bucket_id": "bucket-00000",
                "error": "boom",
            }
        )
    )

    dispatcher = RemoteToolDispatcher([node], bucket_size=1, local_results_dir=tmp_path, pull_results=False)

    with pytest.raises(RuntimeError, match=r"ty/bucket-00000.*boom"):
        dispatcher.wait_for_plan(plan)


def test_collect_plan_outputs_failed_envelope_includes_node_path(clean_registry, tmp_path):
    """Collect errors should include node, bucket, result path, and trace tail."""
    _register_remote_mock_tool(clean_registry)
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", worker_device="cuda:0")
    plan = RemoteDispatchPlanner([node], bucket_size=1).build_plan(
        "remote-mock",
        RemoteMockInput(sequences=["AA"]),
        run_id="run-collect-fail",
    )
    failed_path = tmp_path / "ty" / "run-collect-fail" / "bucket-00000.json"
    failed_path.parent.mkdir(parents=True)
    failed_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "run_id": "run-collect-fail",
                "bucket_id": "bucket-00000",
                "error": "worker died",
                "traceback": "line1\nlast-line",
            }
        )
    )

    with pytest.raises(RuntimeError, match=r"ty/bucket-00000.*worker died.*last-line"):
        collect_plan_outputs(plan, tmp_path)


def test_rsync_command_rendering():
    """Rsync helpers should render one pull command per node and a parallel loop."""
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty", ssh_args=("-p", "2222")),
        RemoteNode(
            name="h100",
            host="login",
            rsync_host="h100-data",
            rsync_ssh_args=("-J", "login"),
            work_dir="/remote/h100",
        ),
    ]

    commands = render_rsync_pull_commands(nodes, Path("/local/results"))
    script = render_continuous_rsync_script(nodes, Path("/local/results"), interval_sec=30)

    assert commands[0] == [
        "rsync",
        "-az",
        "--partial",
        "-e",
        "ssh -p 2222",
        "ty:/remote/ty/results/",
        "/local/results/ty/",
    ]
    assert commands[1] == [
        "rsync",
        "-az",
        "--partial",
        "-e",
        "ssh -J login",
        "h100-data:/remote/h100/results/",
        "/local/results/h100/",
    ]
    assert script.startswith("while true; do\n  pids=()")
    assert "ty:/remote/ty/results/" in script
    assert "h100-data:/remote/h100/results/" in script
    assert script.count('pids+=("$!")') == 2
    assert 'wait "$pid" || status=$?' in script
    assert "sleep 30" in script


def test_local_rsync_pull_scripts_manage_pid_and_log_paths():
    """Local rsync pull scripts should manage pid/log files around the parallel loop."""
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")

    start_script = render_start_rsync_pull_script(
        [node],
        Path("/local/results"),
        pid_path=Path("/state/proto-rsync.pid"),
        log_path=Path("/state/proto-rsync.log"),
        interval_sec=7,
    )
    status_script = render_rsync_pull_status_script(Path("/state/proto-rsync.pid"))
    stop_script = render_stop_rsync_pull_script(Path("/state/proto-rsync.pid"))

    assert "nohup bash -lc" in start_script
    assert "/state/proto-rsync.pid" in start_script
    assert "/state/proto-rsync.log" in start_script
    assert "pids=()" in start_script
    assert "sleep 7" in start_script
    assert "echo running" in status_script
    assert "echo stale" in status_script
    assert "kill \"$(cat /state/proto-rsync.pid)\"" in stop_script
    assert "stale-removed" in stop_script


def test_local_rsync_pull_client_uses_local_bash_runner():
    """LocalRsyncPullClient should run start/status/stop through local bash."""
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")
    calls = []
    outputs = ["started 123\n", "running 123\n", "stopped\n"]

    class FakeCompleted:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_runner(cmd, check=False, text=False, capture_output=False):
        calls.append({"cmd": cmd, "check": check, "text": text, "capture_output": capture_output})
        return FakeCompleted(outputs.pop(0))

    client = LocalRsyncPullClient()

    assert (
        client.start(
            [node],
            Path("/local/results"),
            pid_path=Path("/state/proto-rsync.pid"),
            log_path=Path("/state/proto-rsync.log"),
            runner=fake_runner,
        )
        == "started 123"
    )
    assert client.status(Path("/state/proto-rsync.pid"), runner=fake_runner) == "running 123"
    assert client.stop(Path("/state/proto-rsync.pid"), runner=fake_runner) == "stopped"

    assert [call["cmd"][:2] for call in calls] == [["bash", "-lc"], ["bash", "-lc"], ["bash", "-lc"]]
    assert "rsync -az --partial" in calls[0]["cmd"][2]
    assert "manager-loop" not in calls[0]["cmd"][2]
    assert all(call["check"] and call["text"] and call["capture_output"] for call in calls)


def test_manager_loop_command_uses_node_scheduler_fields():
    """Manager command rendering should carry scheduler and sbatch args from the node spec."""
    node = RemoteNode(
        name="h100",
        host="h100",
        work_dir="/remote/h100",
        scheduler="slurm",
        python="/env/bin/python",
        sbatch_args=(
            "--partition=gpu_special",
            "--gres=gpu:h100:1",
            "--output=/remote/h100/logs/slurm-%j.out",
        ),
        max_active_slurm_jobs=3,
    )

    cmd = render_manager_loop_command(node, poll_interval=2.5, once=True)
    parsed = remote_execution_mod._build_arg_parser().parse_args(cmd[3:])

    assert cmd[:5] == ["/env/bin/python", "-m", "proto_tools.utils.remote_execution", "manager-loop", "--queue-dir"]
    assert "--scheduler" in cmd
    assert cmd[cmd.index("--scheduler") + 1] == "slurm"
    assert cmd[cmd.index("--slurm-group-size") + 1] == "10"
    assert cmd[cmd.index("--max-active-slurm-jobs") + 1] == "3"
    assert parsed.sbatch_arg == list(node.sbatch_args)
    assert cmd[-1] == "--once"


def test_manager_loop_command_requires_slurm_active_job_cap():
    """Slurm manager rendering should not omit the active-job cap."""
    node = RemoteNode(name="h100", host="h100", work_dir="/remote/h100", scheduler="slurm")

    with pytest.raises(ValueError, match="Slurm manager requires max_active_slurm_jobs"):
        render_manager_loop_command(node)


def test_manager_start_and_status_scripts_use_pid_and_logs():
    """Manager shell snippets should use stable pid and log paths."""
    node = RemoteNode(
        name="ty",
        host="ty",
        work_dir="/remote/ty",
        repo_dir="/remote/proto-tools",
        python="/env/bin/python",
    )

    start_script = render_start_manager_script(node, poll_interval=1.5)
    status_script = render_manager_status_script(node)

    assert "cd /remote/proto-tools && export PYTHONPATH=/remote/proto-tools" in start_script
    assert "nohup bash -lc" in start_script
    assert "/env/bin/python -m proto_tools.utils.remote_execution manager-loop" in start_script
    assert "/remote/ty/manager.pid" in start_script
    assert "/remote/ty/logs/manager.log" in start_script
    assert "sleep 0.2" in start_script
    assert "tail -n 40" in start_script
    assert "kill -0" in status_script
    assert "echo stopped" in status_script


def test_remote_manager_client_uses_ssh_runner():
    """RemoteManagerClient should route start/status through SSH."""
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", ssh_args=("-p", "2222"))
    calls = []

    class FakeCompleted:
        stdout = "running 123\n"

    def fake_runner(cmd, check=False, text=False, capture_output=False):
        calls.append({"cmd": cmd, "check": check, "text": text, "capture_output": capture_output})
        return FakeCompleted()

    client = RemoteManagerClient()
    assert client.start_node(node, runner=fake_runner) == "running 123"
    assert client.status_node(node, runner=fake_runner) == "running 123"

    assert calls[0]["cmd"][:6] == ["ssh", "-p", "2222", "ty", "bash", "-lc"]
    assert "manager-loop" in calls[0]["cmd"][6]
    assert "manager.pid" in calls[1]["cmd"][6]
    assert calls[1]["cmd"][6].startswith("'set -euo pipefail;")
    assert all(call["check"] and call["text"] and call["capture_output"] for call in calls)


def test_remote_preflight_reports_python_import_and_slurm_checks():
    """Preflight should check Python, importability, directories, rsync, and Slurm."""
    node = RemoteNode(
        name="h100",
        host="h100",
        work_dir="/remote/h100",
        repo_dir="/remote/proto-tools",
        scheduler="slurm",
        python="/remote/proto-tools/.venv/bin/python",
    )

    script = render_preflight_script(node)

    assert "python>=3.10" in script
    assert "import proto_tools" in script
    assert "/remote/proto-tools" in script
    assert "/remote/h100/queue" in script
    assert "rsync available" in script
    assert "sbatch available" in script
    assert "sacct available" in script


def test_remote_manager_client_preflight_and_diagnostics_parse_json():
    """Manager client should parse JSON returned by preflight and diagnostics."""
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")
    calls = []

    class FakeCompleted:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_runner(cmd, check=False, text=False, capture_output=False):
        calls.append(cmd)
        if "recent_failed" in cmd[-1]:
            return FakeCompleted('{"node":"ty","counts":{"queue":1},"recent_failed":[]}\n')
        return FakeCompleted('{"node":"ty","ok":true,"checks":[]}\n')

    client = RemoteManagerClient()

    assert client.preflight_node(node, runner=fake_runner)["ok"] is True
    assert client.diagnostics_node(node, runner=fake_runner)["counts"]["queue"] == 1
    assert all(cmd[:4] == ["ssh", "ty", "bash", "-lc"] for cmd in calls)


def test_remote_manager_client_preflight_invalid_json_names_node_and_output():
    """Invalid preflight output should identify the node and preserve SSH evidence."""
    node = RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3")

    class FakeCompleted:
        stdout = "login banner\n"
        stderr = "remote warning\n"

    with pytest.raises(
        ValueError,
        match=r"ty3.*invalid JSON.*login banner.*remote warning",
    ):
        RemoteManagerClient().preflight_node(node, runner=lambda *args, **kwargs: FakeCompleted())


def test_remote_manager_client_preflight_accepts_final_json_after_visible_login_noise(caplog):
    """A remote profile warning may precede the final JSON but must remain visible."""
    node = RemoteNode(name="ty3", host="ty3", work_dir="/remote/ty3")

    class FakeCompleted:
        stdout = "gpu-ideal.sh: nvidia-modprobe: command not found\n{\"node\":\"ty3\",\"ok\":true}\n"
        stderr = "Shared connection closed.\n"

    payload = RemoteManagerClient().preflight_node(
        node,
        runner=lambda *args, **kwargs: FakeCompleted(),
    )

    assert payload == {"node": "ty3", "ok": True}
    assert "gpu-ideal.sh: nvidia-modprobe: command not found" in caplog.text


def test_remote_manager_client_passes_timeout_to_runner():
    """Manager client should pass timeout_sec through to each SSH subprocess."""
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")
    outputs = iter(
        [
            "started\n",
            "running\n",
            '{"node":"ty","ok":true,"checks":[]}\n',
            '{"node":"ty","counts":{},"recent_failed":[]}\n',
        ]
    )
    calls = []

    class FakeCompleted:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return FakeCompleted(next(outputs))

    client = RemoteManagerClient()

    assert client.start_node(node, timeout_sec=12.5, runner=fake_runner) == "started"
    assert client.status_node(node, timeout_sec=12.5, runner=fake_runner) == "running"
    assert client.preflight_node(node, timeout_sec=12.5, runner=fake_runner)["ok"] is True
    assert client.diagnostics_node(node, timeout_sec=12.5, runner=fake_runner)["counts"] == {}
    assert [kwargs["timeout"] for _, kwargs in calls] == [12.5, 12.5, 12.5, 12.5]
    assert all(kwargs["check"] is True and kwargs["text"] is True and kwargs["capture_output"] is True for _, kwargs in calls)


def test_manager_diagnostics_script_counts_roots_and_failed_items():
    """Diagnostics script should report queue state, log tail, and failed summaries."""
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", python="/env/bin/python")

    script = render_manager_diagnostics_script(node)

    assert "/remote/ty/queue" in script
    assert "/remote/ty/results" in script
    assert "/remote/ty/logs/manager.log" in script
    assert "recent_failed" in script
    assert "recent_running" in script
    assert "recent_submitted" in script
    assert "recent_tmp" in script
    assert "recent_failed_results" in script


def test_manager_diagnostics_script_reports_logical_counts_and_visibility(tmp_path):
    """Diagnostics should surface stuck requests, sidecars, tmp files, and fallback failures."""
    work_dir = tmp_path / "remote-work"
    node = RemoteNode(name="ty", host="ty", work_dir=str(work_dir), python=sys.executable)
    running_request = work_dir / "running" / "run-running" / "bucket-00000.json"
    submitted_request = work_dir / "submitted" / "run-submitted" / "bucket-00001.json"
    sidecar = submitted_request.with_suffix(submitted_request.suffix + ".slurm.json")
    queue_tmp = work_dir / "queue" / "run-queued" / ".bucket-00002.json.tmp"
    result_tmp = work_dir / "results" / "run-submitted" / ".bucket-00001.json.tmp"
    fallback_failed = work_dir / "results" / "failed" / "missing-request.json"
    for path in (running_request, submitted_request, sidecar, queue_tmp, result_tmp, fallback_failed):
        path.parent.mkdir(parents=True, exist_ok=True)

    running_request.write_text(
        json.dumps({"run_id": "run-running", "bucket_id": "bucket-00000", "tool_key": "remote-mock"})
    )
    submitted_request.write_text(
        json.dumps({"run_id": "run-submitted", "bucket_id": "bucket-00001", "tool_key": "remote-mock"})
    )
    sidecar.write_text(
        json.dumps(
            {
                "slurm_job_id": "12345",
                "bucket_id": "bucket-00001",
                "result_path": str(work_dir / "results" / "run-submitted" / "bucket-00001.json"),
            }
        )
    )
    queue_tmp.write_text("{}")
    result_tmp.write_text("{}")
    fallback_failed.write_text(
        json.dumps(
            {
                "status": "failed",
                "run_id": "run-failed",
                "bucket_id": "bucket-00999",
                "tool_key": "remote-mock",
                "error": "request file missing",
            }
        )
    )

    completed = subprocess.run(
        ["bash", "-lc", render_manager_diagnostics_script(node)],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["counts"]["queue"] == 0
    assert payload["counts"]["running"] == 1
    assert payload["counts"]["submitted"] == 1
    assert payload["counts"]["results"] == 1
    assert payload["sidecar_counts"] == {"submitted": 1}
    assert payload["recent_running"][0]["run_id"] == "run-running"
    assert payload["recent_running"][0]["bucket_id"] == "bucket-00000"
    assert payload["recent_submitted"][0]["request_exists"] is True
    assert payload["recent_submitted"][0]["slurm_job_id"] == "12345"
    assert payload["recent_submitted"][0]["run_id"] == "run-submitted"
    assert {item["root"] for item in payload["recent_tmp"]} == {"queue", "results"}
    assert payload["recent_failed_results"][0]["status"] == "failed"
    assert payload["recent_failed_results"][0]["error"] == "request file missing"


def test_remote_deploy_renders_rsync_and_manager_env(tmp_path):
    """Deployment rendering should make dirs, upload checkout, and install editable."""
    node = RemoteNode(
        name="ty",
        host="ty",
        work_dir="/remote/ty-work",
        repo_dir="/remote/proto-tools",
        venv_dir="/remote/proto-tools/.venv",
        bootstrap_python="/usr/bin/python3.12",
        python="/remote/proto-tools/.venv/bin/python",
        ssh_args=("-p", "2222"),
    )

    commands = render_deploy_commands(node, tmp_path)
    start_script = render_start_manager_script(node)

    assert commands[0][:6] == ["ssh", "-p", "2222", "ty", "bash", "-lc"]
    assert "mkdir -p /remote/proto-tools /remote/ty-work/queue" in commands[0][-1]
    assert commands[1][:4] == ["rsync", "-az", "--partial", "--delete"]
    assert "--delete-excluded" not in commands[1]
    assert str(tmp_path) + "/" in commands[1]
    assert commands[1][-1] == "ty:/remote/proto-tools/"
    assert "/usr/bin/python3.12 -m venv /remote/proto-tools/.venv" in commands[2][-1]
    assert "/remote/proto-tools/.venv/bin/python -m pip install -e ." in commands[2][-1]
    assert "cd /remote/proto-tools && export PYTHONPATH=/remote/proto-tools" in start_script
    assert "/remote/proto-tools/.venv/bin/python -m proto_tools.utils.remote_execution" in start_script


def test_inspect_local_checkout_counts_clean_dirty_and_untracked(tmp_path):
    """Local checkout inspection should expose tracked and untracked changes."""
    repo = _init_git_repo(tmp_path / "repo")

    clean = inspect_local_checkout(repo)

    assert clean.is_git_checkout is True
    assert clean.path == str(repo.resolve())
    assert clean.dirty is False
    assert clean.dirty_count == 0
    assert clean.untracked_count == 0
    assert clean.head_commit

    (repo / "tracked.txt").write_text("dirty\n")
    (repo / "untracked.txt").write_text("new\n")

    dirty = inspect_local_checkout(repo)

    assert dirty.is_git_checkout is True
    assert dirty.dirty is True
    assert dirty.dirty_count == 1
    assert dirty.untracked_count == 1
    assert any(line.endswith("tracked.txt") for line in dirty.status_lines)
    assert "?? untracked.txt" in dirty.status_lines


def test_remote_deploy_report_serializes_structured_commands(tmp_path):
    """Deployment report JSON should preserve argv plus stable command metadata."""
    repo = _init_git_repo(tmp_path / "repo")
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/work", repo_dir="/remote/repo")

    report = build_deploy_render_report([node], repo)
    payload = report.to_dict()
    commands = render_deploy_commands(node, repo)

    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["local_repo_dir"] == str(repo.resolve())
    assert payload["node_count"] == 1
    assert payload["command_count"] == len(commands)
    assert payload["nodes"][0]["name"] == "ty"
    assert payload["nodes"][0]["command_count"] == len(commands)
    assert [command["step"] for command in payload["nodes"][0]["commands"]] == [
        "mkdir",
        "rsync_upload",
        "install_editable",
    ]
    assert [command["index"] for command in payload["nodes"][0]["commands"]] == [0, 1, 2]
    assert [command["argv"] for command in payload["nodes"][0]["commands"]] == commands
    assert payload["nodes"][0]["commands"][0]["shell"] == shlex.join(commands[0])


def test_remote_deployment_client_executes_rendered_commands(tmp_path):
    """Deployment client should execute the rendered command sequence."""
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/work", repo_dir="/remote/repo")
    calls = []

    def fake_runner(cmd, check=False, text=False, capture_output=False):
        calls.append({"cmd": cmd, "check": check, "text": text, "capture_output": capture_output})

    RemoteDeploymentClient().deploy_node(node, tmp_path, runner=fake_runner)

    assert len(calls) == 3
    assert calls[0]["cmd"][0] == "ssh"
    assert calls[1]["cmd"][0] == "rsync"
    assert calls[2]["cmd"][0] == "ssh"
    assert all(call["check"] and call["text"] and call["capture_output"] for call in calls)


def test_load_remote_nodes_json(tmp_path):
    """JSON profiles should load into RemoteNode objects with tuple arguments."""
    profile = tmp_path / "nodes.json"
    profile.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "name": "ty",
                        "host": "ty",
                        "work_dir": "/remote/ty",
                        "repo_dir": "/remote/repo",
                        "venv_dir": "/remote/repo/.venv",
                        "bootstrap_python": "/usr/bin/python3.12",
                        "ssh_args": ["-p", "2222"],
                        "rsync_ssh_args": ["-J", "login"],
                    }
                ]
            }
        )
    )

    nodes = load_remote_nodes_json(profile)

    assert nodes == [
        RemoteNode(
            name="ty",
            host="ty",
            work_dir="/remote/ty",
            repo_dir="/remote/repo",
            venv_dir="/remote/repo/.venv",
            bootstrap_python="/usr/bin/python3.12",
            ssh_args=("-p", "2222"),
            rsync_ssh_args=("-J", "login"),
        )
    ]


def test_scaffold_remote_node_uses_home_layout_and_static_validation(tmp_path):
    """Profile scaffolding should produce deployable direct and Slurm nodes."""
    nodes = [
        scaffold_remote_node("ty", "/sfs/home/zengzhengpeng", scheduler="direct"),
        scaffold_remote_node(
            "h100",
            "/public/home/zengzhengpeng",
            scheduler="slurm",
            weight=3.0,
            max_active_slurm_jobs=2,
            sbatch_args=("--partition=gpu", "--gres=gpu:h100:1"),
        ),
    ]
    profile_path = tmp_path / "remote-nodes.json"

    write_remote_nodes_json(nodes, profile_path)
    loaded = load_remote_nodes_json(profile_path)

    assert loaded == nodes
    assert loaded[0].work_dir == "/sfs/home/zengzhengpeng/proto_remote"
    assert loaded[0].repo_dir == "/sfs/home/zengzhengpeng/proto-tools"
    assert loaded[0].venv_dir == "/sfs/home/zengzhengpeng/proto-tools/.venv"
    assert loaded[0].python == "/sfs/home/zengzhengpeng/proto-tools/.venv/bin/python"
    assert loaded[0].worker_device == "cuda:0"
    assert loaded[1].worker_device == "cuda"
    assert loaded[1].max_active_slurm_jobs == 2
    assert validate_remote_nodes(loaded) == []


def test_validate_remote_nodes_reports_duplicate_and_relative_paths():
    """Static validation should catch local profile mistakes before SSH use."""
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty"),
        RemoteNode(name="ty", host="ty2", work_dir="relative/work"),
        RemoteNode(name="h100", host="h100", work_dir="/remote/h100", scheduler="slurm"),
    ]

    issues = validate_remote_nodes(nodes)

    assert ("error", "ty", "name") in {(issue.level, issue.node_name, issue.field) for issue in issues}
    assert ("error", "ty", "work_dir") in {(issue.level, issue.node_name, issue.field) for issue in issues}
    assert ("error", "h100", "max_active_slurm_jobs") in {
        (issue.level, issue.node_name, issue.field) for issue in issues
    }
    assert ("warning", "h100", "sbatch_args") in {(issue.level, issue.node_name, issue.field) for issue in issues}


def test_validate_remote_nodes_warns_direct_max_concurrent_buckets_planner_only():
    """Direct node concurrency is planner capacity, not per-manager parallel execution."""
    node = RemoteNode(
        name="ty",
        host="ty",
        work_dir="/remote/ty",
        max_concurrent_buckets=2,
    )

    issues = validate_remote_nodes([node])

    assert ("warning", "ty", "max_concurrent_buckets") in {
        (issue.level, issue.node_name, issue.field) for issue in issues
    }
    assert any("only affects local planning capacity" in issue.message for issue in issues)


def test_validate_remote_nodes_rejects_shared_work_dir_and_scaffold_template():
    """Node-specific work templates should avoid shared queues on shared homes."""
    nodes = [
        scaffold_remote_node("ty2", "/mnt/sfs/zengzhengpeng", work_name="proto_remote"),
        scaffold_remote_node("ty3", "/mnt/sfs/zengzhengpeng", work_name="proto_remote"),
    ]

    issues = validate_remote_nodes(nodes)
    templated = scaffold_remote_node("ty3", "/mnt/sfs/zengzhengpeng", work_name="proto_remote_{node}")
    templated_nodes = [
        scaffold_remote_node("ty2", "/mnt/sfs/zengzhengpeng", work_name="proto_remote_{node}"),
        templated,
    ]
    templated_issues = validate_remote_nodes(templated_nodes)

    assert ("error", "ty2,ty3", "work_dir") in {(issue.level, issue.node_name, issue.field) for issue in issues}
    assert ("warning", "ty2,ty3", "repo_dir") in {
        (issue.level, issue.node_name, issue.field) for issue in templated_issues
    }
    assert ("warning", "ty2,ty3", "venv_dir") in {
        (issue.level, issue.node_name, issue.field) for issue in templated_issues
    }
    assert templated.work_dir == "/mnt/sfs/zengzhengpeng/proto_remote_ty3"


def test_smoke_remote_nodes_validates_profile_before_ssh(tmp_path):
    """Smoke should stop before SSH when static profile validation fails."""
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/shared"),
        RemoteNode(name="ty3", host="ty3", work_dir="/remote/shared"),
    ]

    class ForbiddenManager:
        def preflight_node(self, *args, **kwargs):
            raise AssertionError("preflight should not run")

    report = smoke_remote_nodes(nodes, local_results_dir=tmp_path, manager_client=ForbiddenManager())

    assert report.ok is False
    assert report.preflight == {}
    assert ("error", "ty,ty3", "work_dir") in {
        (issue.level, issue.node_name, issue.field) for issue in report.profile_issues
    }


def test_smoke_remote_nodes_preflight_status_diagnostics_without_submission(tmp_path):
    """Smoke should inspect nodes without starting managers or submitting Slurm."""
    nodes = [
        RemoteNode(name="ty", host="ty", work_dir="/remote/ty"),
        RemoteNode(
            name="h100",
            host="h100",
            work_dir="/remote/h100",
            scheduler="slurm",
            sbatch_args=("--partition=gpu", "--gres=gpu:h100:1"),
            max_active_slurm_jobs=2,
        ),
    ]
    calls = []

    def fake_runner(cmd, check=False, text=False, capture_output=False):
        joined = " ".join(cmd)
        calls.append(joined)
        assert check is True
        assert text is True
        assert capture_output is True
        assert cmd[0] == "ssh"
        assert "manager-loop" not in joined
        assert "run-request-batch" not in joined
        assert "--wrap" not in joined
        assert "nohup bash -lc" not in joined

        class FakeCompleted:
            def __init__(self, stdout):
                self.stdout = stdout

        script = cmd[-1]
        if "recent_failed" in script:
            return FakeCompleted('{"node":"x","manager":{"status":"stopped"},"counts":{},"recent_failed":[]}\n')
        if "manager.pid" in script and "kill -0" in script:
            return FakeCompleted("stopped\n")
        return FakeCompleted('{"node":"x","ok":true,"checks":[{"name":"rsync available","ok":true}]}\n')

    report = smoke_remote_nodes(nodes, local_results_dir=tmp_path / "results", runner=fake_runner)

    assert report.ok is True
    assert set(report.preflight) == {"ty", "h100"}
    assert report.manager_status == {"ty": "stopped", "h100": "stopped"}
    assert [cmd[0] for cmd in report.rsync_pull_commands] == ["rsync", "rsync"]
    assert len(calls) == 6


def test_smoke_remote_nodes_passes_timeout_to_remote_checks(tmp_path):
    """Smoke should bound each SSH check when timeout_sec is provided."""
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty")
    timeouts = []

    def fake_runner(cmd, check=False, text=False, capture_output=False, timeout=None):
        del check, text, capture_output
        timeouts.append(timeout)

        class FakeCompleted:
            def __init__(self, stdout):
                self.stdout = stdout

        script = cmd[-1]
        if "recent_failed" in script:
            return FakeCompleted('{"node":"ty","manager":{"status":"stopped"},"counts":{},"recent_failed":[]}\n')
        if "manager.pid" in script and "kill -0" in script:
            return FakeCompleted("stopped\n")
        return FakeCompleted('{"node":"ty","ok":true,"checks":[]}\n')

    report = smoke_remote_nodes([node], local_results_dir=tmp_path, timeout_sec=12.5, runner=fake_runner)

    assert report.ok is True
    assert timeouts == [12.5, 12.5, 12.5]


def test_smoke_remote_nodes_reports_remote_check_failures_without_raising(tmp_path):
    """Smoke should classify remote SSH failures per node instead of aborting the run."""
    node = RemoteNode(name="ty", host="ty", work_dir="/remote/ty", python="/missing/python")

    def fake_runner(cmd, check=False, text=False, capture_output=False, timeout=None):
        del timeout
        assert check is True
        assert text is True
        assert capture_output is True

        class FakeCompleted:
            stdout = "stopped\n"

        script = cmd[-1]
        if "manager.pid" in script and "kill -0" in script:
            return FakeCompleted()
        raise subprocess.CalledProcessError(
            127,
            cmd,
            output="",
            stderr="/missing/python: No such file or directory",
        )

    report = smoke_remote_nodes([node], local_results_dir=tmp_path, runner=fake_runner)

    assert report.ok is False
    assert report.preflight["ty"]["ok"] is False
    assert report.preflight["ty"]["error"]["returncode"] == 127
    assert "/missing/python" in report.preflight["ty"]["error"]["stderr"]
    assert report.manager_status == {"ty": "stopped"}
    assert report.diagnostics["ty"]["ok"] is False
    assert report.diagnostics["ty"]["error"]["returncode"] == 127

"""Remote queue and scheduling helpers for multi-node tool execution.

This module keeps remote execution deliberately small:

* local code plans contiguous input buckets across named nodes;
* SSH submission stages JSON requests into a per-node queue;
* a node-side manager consumes queued requests and writes JSON results; and
* result files can be merged back into the tool's normal output model.

The node manager runs ordinary registered proto-tools functions on the remote
host, so each node still uses its local ``DeviceManager`` and persistent-worker
machinery.
"""

import argparse
import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.base_config import BaseConfig
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput

RemoteScheduler = Literal["direct", "slurm"]
RemoteRsyncMode = Literal["inline", "background"]
ExistingCommandScheduler = Literal["direct", "srun"]
ExistingCommandStatus = Literal["ready", "partial", "blocked"]
_SCHEMA_VERSION = 1
_MANIFEST_SCHEMA_VERSION = 1
_PROFILE_SCHEMA_VERSION = 1
_EXISTING_COMMAND_PROFILE_SCHEMA_VERSION = 1
_DEPLOY_RENDER_SCHEMA_VERSION = 1
_REMOTE_COMPUTE_REGISTRY_SCHEMA_VERSION = 1
_REMOTE_COMPUTE_REGISTRY_ENV = "PROTO_TOOLS_REMOTE_COMPUTE_REGISTRY"
_DEPLOY_COMMAND_STEPS = ("mkdir", "rsync_upload", "install_editable")
_SLURM_ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RUNNING",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "SIGNALING",
    "SUSPENDED",
}
_SLURM_COMPLETED_STATES = {"COMPLETED"}
_SLURM_FAILED_STATES = {
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "BOOT_FAIL",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SLURM_ACCOUNTING_GRACE_SEC = 3600.0
_SLURM_ADMISSION_MAX_START_DELAY_SEC = 300.0
logger = logging.getLogger(__name__)


def _validate_path_segment(field: str, value: str) -> str:
    """Return ``value`` or raise when it is unsafe as one path component."""
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if not _PATH_SEGMENT_RE.fullmatch(value):
        raise ValueError(
            f"{field} must be a safe path segment using letters, digits, '_', '-', or '.', got {value!r}"
        )
    return value


def _request_path_segment(request: dict[str, Any], key: str) -> str:
    """Read and validate one path segment from a request payload."""
    return _validate_path_segment(f"request.{key}", str(request[key]))


@dataclass(frozen=True)
class RemoteNode:
    """A remote execution target managed through SSH."""

    name: str
    host: str
    work_dir: str
    repo_dir: str | None = None
    venv_dir: str | None = None
    bootstrap_python: str | None = None
    weight: float = 1.0
    max_concurrent_buckets: int = 1
    scheduler: RemoteScheduler = "direct"
    worker_device: str = "cuda"
    python: str = "python"
    ssh_args: tuple[str, ...] = ()
    sbatch_args: tuple[str, ...] = ()
    max_active_slurm_jobs: int | None = None
    rsync_host: str | None = None
    rsync_ssh_args: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate node scheduling fields."""
        _validate_path_segment("RemoteNode.name", self.name)
        if not self.host:
            raise ValueError(f"RemoteNode {self.name!r}: host must be non-empty")
        if not self.work_dir:
            raise ValueError(f"RemoteNode {self.name!r}: work_dir must be non-empty")
        if self.weight <= 0:
            raise ValueError(f"RemoteNode {self.name!r}: weight must be > 0")
        if self.max_concurrent_buckets < 1:
            raise ValueError(f"RemoteNode {self.name!r}: max_concurrent_buckets must be >= 1")
        if self.max_active_slurm_jobs is not None and self.max_active_slurm_jobs < 1:
            raise ValueError(f"RemoteNode {self.name!r}: max_active_slurm_jobs must be >= 1")
        if self.scheduler not in ("direct", "slurm"):
            raise ValueError(f"RemoteNode {self.name!r}: unsupported scheduler {self.scheduler!r}")

    @property
    def queue_root(self) -> PurePosixPath:
        """Remote queue directory for staged requests."""
        return PurePosixPath(self.work_dir) / "queue"

    @property
    def results_root(self) -> PurePosixPath:
        """Remote results directory written by the node manager."""
        return PurePosixPath(self.work_dir) / "results"

    @property
    def logs_root(self) -> PurePosixPath:
        """Remote logs directory for background manager processes."""
        return PurePosixPath(self.work_dir) / "logs"

    @property
    def manager_pid_path(self) -> PurePosixPath:
        """Remote pid file for the background manager process."""
        return PurePosixPath(self.work_dir) / "manager.pid"

    @property
    def manager_log_path(self) -> PurePosixPath:
        """Remote log file for the background manager process."""
        return self.logs_root / "manager.log"

    @property
    def sync_host(self) -> str:
        """Host string used by rsync pulls."""
        return self.rsync_host or self.host

    @property
    def sync_ssh_args(self) -> tuple[str, ...]:
        """SSH args used by rsync; defaults to the submission SSH args."""
        return self.ssh_args if self.rsync_ssh_args is None else self.rsync_ssh_args

    @property
    def effective_venv_dir(self) -> str | None:
        """Remote virtualenv path, defaulting under repo_dir when configured."""
        if self.venv_dir is not None:
            return self.venv_dir
        if self.repo_dir is not None:
            return str(PurePosixPath(self.repo_dir) / ".venv")
        return None

    @property
    def deploy_python(self) -> str:
        """Python used to create/install the remote environment."""
        return self.bootstrap_python or self.python

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RemoteNode":
        """Create a node from a JSON-compatible mapping."""
        values = dict(data)
        for key in ("ssh_args", "sbatch_args", "rsync_ssh_args"):
            if key in values and values[key] is not None:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Serialize a node to JSON-compatible data."""
        return {
            "name": self.name,
            "host": self.host,
            "work_dir": self.work_dir,
            "repo_dir": self.repo_dir,
            "venv_dir": self.venv_dir,
            "bootstrap_python": self.bootstrap_python,
            "weight": self.weight,
            "max_concurrent_buckets": self.max_concurrent_buckets,
            "scheduler": self.scheduler,
            "worker_device": self.worker_device,
            "python": self.python,
            "ssh_args": list(self.ssh_args),
            "sbatch_args": list(self.sbatch_args),
            "max_active_slurm_jobs": self.max_active_slurm_jobs,
            "rsync_host": self.rsync_host,
            "rsync_ssh_args": None if self.rsync_ssh_args is None else list(self.rsync_ssh_args),
        }


@dataclass(frozen=True)
class RemoteProfileIssue:
    """A static profile validation issue."""

    level: Literal["error", "warning"]
    node_name: str | None
    field: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize this issue to JSON-compatible data."""
        return {
            "level": self.level,
            "node_name": self.node_name,
            "field": self.field,
            "message": self.message,
        }


@dataclass(frozen=True)
class RemoteSmokeReport:
    """No-submit smoke report for remote node profiles."""

    ok: bool
    profile_issues: tuple[RemoteProfileIssue, ...]
    preflight: dict[str, Any]
    manager_status: dict[str, str]
    diagnostics: dict[str, Any]
    rsync_pull_commands: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this report to JSON-compatible data."""
        return {
            "ok": self.ok,
            "profile_issues": [issue.to_dict() for issue in self.profile_issues],
            "preflight": self.preflight,
            "manager_status": self.manager_status,
            "diagnostics": self.diagnostics,
            "rsync_pull_commands": [list(cmd) for cmd in self.rsync_pull_commands],
        }


@dataclass(frozen=True)
class ExistingRemoteCommandNode:
    """A remote node that runs an already-installed command instead of a proto-tools manager."""

    name: str
    host: str
    work_dir: str
    command_argv: tuple[str, ...]
    status: ExistingCommandStatus = "ready"
    software: str = "existing"
    required_artifacts: tuple[str, ...] = ()
    required_stdout_patterns: tuple[str, ...] = ()
    scheduler: ExistingCommandScheduler = "direct"
    env: dict[str, str] | None = None
    gpu_admission: dict[str, int] | None = None
    srun_args: tuple[str, ...] = ()
    ssh_args: tuple[str, ...] = ()
    rsync_host: str | None = None
    rsync_ssh_args: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the existing-command node shape."""
        _validate_path_segment("ExistingRemoteCommandNode.name", self.name)
        if not self.host:
            raise ValueError(f"ExistingRemoteCommandNode {self.name!r}: host must be non-empty")
        if not self.work_dir:
            raise ValueError(f"ExistingRemoteCommandNode {self.name!r}: work_dir must be non-empty")
        if not PurePosixPath(self.work_dir).is_absolute():
            raise ValueError(f"ExistingRemoteCommandNode {self.name!r}: work_dir must be absolute")
        if not self.command_argv:
            raise ValueError(f"ExistingRemoteCommandNode {self.name!r}: command_argv must be non-empty")
        if self.status not in ("ready", "partial", "blocked"):
            raise ValueError(f"ExistingRemoteCommandNode {self.name!r}: unsupported status {self.status!r}")
        _validate_path_segment("ExistingRemoteCommandNode.software", self.software)
        if not all(self.required_stdout_patterns):
            raise ValueError(
                f"ExistingRemoteCommandNode {self.name!r}: required_stdout_patterns must be non-empty strings"
            )
        if self.scheduler not in ("direct", "srun"):
            raise ValueError(
                f"ExistingRemoteCommandNode {self.name!r}: unsupported scheduler {self.scheduler!r}"
            )
        if self.scheduler == "srun" and not self.srun_args:
            raise ValueError(f"ExistingRemoteCommandNode {self.name!r}: srun_args are required for scheduler='srun'")
        for key in (self.env or {}):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"ExistingRemoteCommandNode {self.name!r}: invalid env var name {key!r}")
        if self.gpu_admission is not None:
            expected = {"max_utilization_percent", "max_memory_used_mib"}
            if set(self.gpu_admission) != expected:
                raise ValueError(
                    f"ExistingRemoteCommandNode {self.name!r}: gpu_admission keys must be {sorted(expected)!r}"
                )
            utilization = self.gpu_admission["max_utilization_percent"]
            memory = self.gpu_admission["max_memory_used_mib"]
            if not 0 <= utilization <= 100:
                raise ValueError(
                    f"ExistingRemoteCommandNode {self.name!r}: max_utilization_percent must be in [0, 100]"
                )
            if memory < 0:
                raise ValueError(
                    f"ExistingRemoteCommandNode {self.name!r}: max_memory_used_mib must be >= 0"
                )
            if not (self.env or {}).get("CUDA_VISIBLE_DEVICES"):
                raise ValueError(
                    f"ExistingRemoteCommandNode {self.name!r}: gpu_admission requires CUDA_VISIBLE_DEVICES"
                )

    @property
    def sync_host(self) -> str:
        """Host string used by rsync pulls."""
        return self.rsync_host or self.host

    @property
    def sync_ssh_args(self) -> tuple[str, ...]:
        """SSH args used by rsync; defaults to the command SSH args."""
        return self.ssh_args if self.rsync_ssh_args is None else self.rsync_ssh_args

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExistingRemoteCommandNode":
        """Create a node from a JSON-compatible mapping."""
        values = dict(data)
        values["command_argv"] = tuple(values.get("command_argv", ()))
        for key in (
            "required_artifacts",
            "required_stdout_patterns",
            "srun_args",
            "ssh_args",
            "rsync_ssh_args",
        ):
            if key in values and values[key] is not None:
                values[key] = tuple(values[key])
        if values.get("env") is not None:
            values["env"] = {str(k): str(v) for k, v in values["env"].items()}
        if values.get("gpu_admission") is not None:
            values["gpu_admission"] = {
                str(k): int(v) for k, v in values["gpu_admission"].items()
            }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this node to JSON-compatible data."""
        return {
            "name": self.name,
            "host": self.host,
            "work_dir": self.work_dir,
            "command_argv": list(self.command_argv),
            "status": self.status,
            "software": self.software,
            "required_artifacts": list(self.required_artifacts),
            "required_stdout_patterns": list(self.required_stdout_patterns),
            "scheduler": self.scheduler,
            "env": self.env or {},
            "gpu_admission": self.gpu_admission,
            "srun_args": list(self.srun_args),
            "ssh_args": list(self.ssh_args),
            "rsync_host": self.rsync_host,
            "rsync_ssh_args": None if self.rsync_ssh_args is None else list(self.rsync_ssh_args),
        }


@dataclass(frozen=True)
class ExistingRemoteBatchNode:
    """Pair one manager transport/scheduler with one maintained existing command."""

    manager: RemoteNode
    command: ExistingRemoteCommandNode

    def __post_init__(self) -> None:
        """Require both halves to describe the same node and one scheduler owner."""
        if self.manager.name != self.command.name:
            raise ValueError(
                f"Existing batch node name mismatch: manager={self.manager.name!r}, command={self.command.name!r}"
            )
        if self.manager.host != self.command.host:
            raise ValueError(
                f"Existing batch node host mismatch for {self.manager.name!r}: "
                f"manager={self.manager.host!r}, command={self.command.host!r}"
            )
        if self.command.scheduler != "direct":
            raise ValueError(
                f"Existing batch command {self.command.name!r} must use scheduler='direct'; "
                "the manager owns direct/Slurm allocation"
            )
        if self.manager.scheduler == "slurm" and self.command.gpu_admission is not None:
            raise ValueError(
                f"Existing batch command {self.command.name!r} must not set gpu_admission under Slurm; "
                "the Slurm allocation owns GPU admission"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExistingRemoteBatchNode":
        """Load one nested manager/command profile row."""
        manager = data.get("manager")
        command = data.get("command")
        if not isinstance(manager, dict) or not isinstance(command, dict):
            raise ValueError("Existing remote batch node must contain manager and command objects")
        return cls(
            manager=RemoteNode.from_dict(manager),
            command=ExistingRemoteCommandNode.from_dict(command),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the combined profile row."""
        return {
            "manager": self.manager.to_dict(),
            "command": self.command.to_dict(),
        }


@dataclass(frozen=True)
class RemoteComputeConfig:
    """Resolved default profile and launch settings for one maintained runtime."""

    software: str
    registry_path: Path
    profile_path: Path
    bucket_size: int
    slurm_group_size: int
    wait_poll_interval: float
    timeout_sec: float | None
    remote_check_timeout_sec: float | None


@dataclass(frozen=True)
class ExistingFastaRecord:
    """One FASTA record preserved as one logical existing-command task."""

    header: str
    sequence: str

    def to_text(self) -> str:
        """Render the normalized single-record FASTA payload."""
        return f">{self.header}\n{self.sequence}\n"


def read_fasta_records(path: Path | str) -> tuple[ExistingFastaRecord, ...]:
    """Read non-empty FASTA records while preserving full headers and order."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"FASTA input not found: {input_path}")
    records: list[ExistingFastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []
    for line_number, raw_line in enumerate(input_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                sequence = "".join(sequence_parts)
                if not sequence:
                    raise ValueError(f"FASTA record {header!r} has an empty sequence")
                records.append(ExistingFastaRecord(header=header, sequence=sequence))
            header = line[1:].strip()
            if not header:
                raise ValueError(f"FASTA header is empty at line {line_number}")
            sequence_parts = []
            continue
        if header is None:
            raise ValueError(f"FASTA sequence appears before a header at line {line_number}")
        sequence_parts.append(line)
    if header is not None:
        sequence = "".join(sequence_parts)
        if not sequence:
            raise ValueError(f"FASTA record {header!r} has an empty sequence")
        records.append(ExistingFastaRecord(header=header, sequence=sequence))
    if not records:
        raise ValueError(f"FASTA input contains no records: {input_path}")
    return tuple(records)


@dataclass(frozen=True)
class ExistingRemoteCommandPlan:
    """Concrete command plan for one existing remote installation run."""

    node_name: str
    run_id: str
    local_input_path: str
    remote_input_path: str
    remote_output_dir: str
    local_collect_dir: str
    prepare_command: tuple[str, ...]
    upload_command: tuple[str, ...]
    execute_command: tuple[str, ...]
    collect_command: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this plan to JSON-compatible data."""
        return {
            "node_name": self.node_name,
            "run_id": self.run_id,
            "local_input_path": self.local_input_path,
            "remote_input_path": self.remote_input_path,
            "remote_output_dir": self.remote_output_dir,
            "local_collect_dir": self.local_collect_dir,
            "commands": {
                "prepare": list(self.prepare_command),
                "upload": list(self.upload_command),
                "execute": list(self.execute_command),
                "collect": list(self.collect_command),
            },
        }


@dataclass(frozen=True)
class ExistingRemoteCommandResult:
    """Result metadata for one existing remote command run."""

    plan: ExistingRemoteCommandPlan
    collected_files: tuple[str, ...]
    command_results: tuple["ExistingRemoteCommandStepResult", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible data."""
        return {
            "plan": self.plan.to_dict(),
            "collected_files": list(self.collected_files),
            "command_results": [result.to_dict() for result in self.command_results],
        }


@dataclass(frozen=True)
class ExistingRemoteCommandStepResult:
    """Captured subprocess output for one existing-command step."""

    step: str
    returncode: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize this step result to JSON-compatible data."""
        return {
            "step": self.step,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class LocalCheckoutStatus:
    """Local git checkout state used before deployment."""

    path: str
    head_commit: str | None
    branch: str | None
    dirty_count: int
    untracked_count: int
    status_lines: tuple[str, ...]
    is_git_checkout: bool = True
    error: str | None = None

    @property
    def dirty(self) -> bool:
        """Whether tracked or untracked local changes are present."""
        return bool(self.dirty_count or self.untracked_count)

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkout state to JSON-compatible data."""
        return {
            "path": self.path,
            "is_git_checkout": self.is_git_checkout,
            "head_commit": self.head_commit,
            "branch": self.branch,
            "dirty": self.dirty,
            "dirty_count": self.dirty_count,
            "untracked_count": self.untracked_count,
            "status_lines": list(self.status_lines),
            "error": self.error,
        }


@dataclass(frozen=True)
class RemoteDeployCommandPlan:
    """One rendered deployment command."""

    index: int
    step: str
    argv: tuple[str, ...]

    @property
    def shell(self) -> str:
        """Shell-rendered command for display."""
        return shlex.join(self.argv)

    def to_dict(self) -> dict[str, Any]:
        """Serialize command plan to JSON-compatible data."""
        return {
            "index": self.index,
            "step": self.step,
            "argv": list(self.argv),
            "shell": self.shell,
        }


@dataclass(frozen=True)
class RemoteDeployNodePlan:
    """Rendered deployment plan for one node."""

    node_name: str
    host: str
    scheduler: RemoteScheduler
    work_dir: str
    repo_dir: str | None
    venv_dir: str | None
    commands: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize node deployment plan to JSON-compatible data."""
        command_plans = tuple(
            RemoteDeployCommandPlan(
                index=index,
                step=_DEPLOY_COMMAND_STEPS[index] if index < len(_DEPLOY_COMMAND_STEPS) else f"command_{index}",
                argv=command,
            )
            for index, command in enumerate(self.commands)
        )
        return {
            "name": self.node_name,
            "host": self.host,
            "scheduler": self.scheduler,
            "work_dir": self.work_dir,
            "repo_dir": self.repo_dir,
            "venv_dir": self.venv_dir,
            "command_count": len(self.commands),
            "commands": [command.to_dict() for command in command_plans],
        }


@dataclass(frozen=True)
class RemoteDeploymentRenderReport:
    """No-submit deployment render report."""

    ok: bool
    local_repo_dir: str
    local_checkout: LocalCheckoutStatus
    profile_issues: tuple[RemoteProfileIssue, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    actions: tuple[str, ...]
    nodes: tuple[RemoteDeployNodePlan, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize deployment render report to JSON-compatible data."""
        return {
            "schema_version": _DEPLOY_RENDER_SCHEMA_VERSION,
            "ok": self.ok,
            "local_repo_dir": self.local_repo_dir,
            "node_count": len(self.nodes),
            "command_count": sum(len(node.commands) for node in self.nodes),
            "local_checkout": self.local_checkout.to_dict(),
            "profile_issues": [issue.to_dict() for issue in self.profile_issues],
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "actions": list(self.actions),
            "nodes": [node.to_dict() for node in self.nodes],
        }


def load_remote_nodes_json(path: Path | str) -> list[RemoteNode]:
    """Load remote nodes from a JSON file with a top-level ``nodes`` list."""
    payload = json.loads(Path(path).read_text())
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Remote node profile must contain a top-level 'nodes' list")
    return [RemoteNode.from_dict(node) for node in nodes]


def write_remote_nodes_json(
    nodes: list[RemoteNode],
    path: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    """Write a remote node profile atomically."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Remote node profile already exists: {output_path}")
    _write_json_atomic(
        output_path,
        {
            "profile_schema_version": _PROFILE_SCHEMA_VERSION,
            "nodes": [node.to_dict() for node in nodes],
        },
    )


def load_existing_remote_command_nodes_json(path: Path | str) -> list[ExistingRemoteCommandNode]:
    """Load existing-command nodes from a JSON file with a top-level ``nodes`` list."""
    payload = json.loads(Path(path).read_text())
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Existing remote command profile must contain a top-level 'nodes' list")
    return [ExistingRemoteCommandNode.from_dict(node) for node in nodes]


def write_existing_remote_command_nodes_json(
    nodes: list[ExistingRemoteCommandNode],
    path: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    """Write an existing-command profile atomically."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Existing remote command profile already exists: {output_path}")
    _write_json_atomic(
        output_path,
        {
            "profile_schema_version": _EXISTING_COMMAND_PROFILE_SCHEMA_VERSION,
            "nodes": [node.to_dict() for node in nodes],
        },
    )


def load_existing_remote_batch_nodes_json(path: Path | str) -> list[ExistingRemoteBatchNode]:
    """Load a combined manager/existing-command profile."""
    payload = json.loads(Path(path).read_text())
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Existing remote batch profile must contain a top-level 'nodes' list")
    return [ExistingRemoteBatchNode.from_dict(node) for node in nodes]


def _default_remote_compute_registry() -> Path:
    """Find the hand-compute registry in direct or bundled Codex skill installs."""
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
    skills_dir = codex_home / "skills"
    direct = skills_dir / "hand-compute" / "references" / "runtime_registry.json"
    if direct.is_file():
        return direct.resolve()

    bundled = sorted(
        path
        for path in skills_dir.glob("*/hand-compute/references/runtime_registry.json")
        if path.is_file()
    )
    if len(bundled) == 1:
        return bundled[0].resolve()
    if len(bundled) > 1:
        candidates = ", ".join(str(path.resolve()) for path in bundled)
        raise ValueError(
            "Multiple bundled $hand-compute runtime registries found; "
            f"set {_REMOTE_COMPUTE_REGISTRY_ENV} explicitly: {candidates}"
        )
    return direct.resolve()


def resolve_remote_compute_config(
    software: str,
    *,
    registry_path: Path | str | None = None,
) -> RemoteComputeConfig:
    """Resolve one ready runtime from the hand-compute registry."""
    if registry_path is None:
        configured = os.environ.get(_REMOTE_COMPUTE_REGISTRY_ENV)
        registry = (
            Path(configured).expanduser().resolve()
            if configured
            else _default_remote_compute_registry()
        )
    else:
        registry = Path(registry_path).expanduser().resolve()
    if not registry.is_file():
        raise ValueError(
            f"Remote compute registry not found: {registry}. "
            f"Install $hand-compute or set {_REMOTE_COMPUTE_REGISTRY_ENV}."
        )

    payload = json.loads(registry.read_text())
    if payload.get("schema_version") != _REMOTE_COMPUTE_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"Remote compute registry {registry} must use "
            f"schema_version={_REMOTE_COMPUTE_REGISTRY_SCHEMA_VERSION}"
        )
    software_entries = payload.get("software")
    if not isinstance(software_entries, dict):
        raise ValueError(f"Remote compute registry {registry} must contain a top-level 'software' object")

    requested = software.strip().lower()
    matches: list[tuple[str, dict[str, Any]]] = []
    for canonical, raw_entry in software_entries.items():
        if not isinstance(canonical, str) or not isinstance(raw_entry, dict):
            raise ValueError(f"Remote compute registry {registry} has an invalid software entry")
        aliases = raw_entry.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            raise ValueError(f"Remote compute registry entry {canonical!r} must contain a string aliases list")
        if requested == canonical.lower() or requested in {alias.lower() for alias in aliases}:
            matches.append((canonical, raw_entry))
    if not matches:
        available = ", ".join(sorted(software_entries))
        raise ValueError(f"Remote compute software {software!r} is not configured; available: {available}")
    if len(matches) != 1:
        raise ValueError(f"Remote compute software alias {software!r} matches multiple registry entries")

    canonical, entry = matches[0]
    _validate_path_segment("remote compute software", canonical)
    status = entry.get("status")
    if status != "ready":
        raise ValueError(f"Remote compute software {canonical!r} is not ready; registry status={status!r}")
    profile_value = entry.get("profile")
    if not isinstance(profile_value, str) or not profile_value:
        raise ValueError(f"Remote compute registry entry {canonical!r} must define profile")
    profile = Path(profile_value).expanduser()
    if not profile.is_absolute():
        profile = registry.parent / profile
    profile = profile.resolve()
    if not profile.is_file():
        raise ValueError(f"Remote compute profile for {canonical!r} not found: {profile}")

    defaults = entry.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"Remote compute registry entry {canonical!r} defaults must be an object")

    def positive_int(name: str, fallback: int) -> int:
        value = defaults.get(name, fallback)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Remote compute registry {canonical!r} defaults.{name} must be a positive integer")
        return int(value)

    def positive_float(name: str, fallback: float | None) -> float | None:
        value = defaults.get(name, fallback)
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"Remote compute registry {canonical!r} defaults.{name} must be positive or null")
        return float(value)

    return RemoteComputeConfig(
        software=canonical,
        registry_path=registry,
        profile_path=profile,
        bucket_size=positive_int("bucket_size", 1),
        slurm_group_size=positive_int("slurm_group_size", 10),
        wait_poll_interval=positive_float("wait_poll_interval", 5.0) or 5.0,
        timeout_sec=positive_float("timeout_sec", None),
        remote_check_timeout_sec=positive_float("remote_check_timeout_sec", None),
    )


def write_existing_remote_batch_nodes_json(
    nodes: list[ExistingRemoteBatchNode],
    path: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    """Write a combined manager/existing-command profile atomically."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Existing remote batch profile already exists: {output_path}")
    _write_json_atomic(
        output_path,
        {
            "profile_schema_version": _EXISTING_COMMAND_PROFILE_SCHEMA_VERSION,
            "nodes": [node.to_dict() for node in nodes],
        },
    )


def select_existing_remote_command_node(
    nodes: list[ExistingRemoteCommandNode],
    node_name: str,
) -> ExistingRemoteCommandNode:
    """Return the selected existing-command node or raise with a clear message."""
    matches = [node for node in nodes if node.name == node_name]
    if not matches:
        available = ", ".join(node.name for node in nodes) or "<none>"
        raise ValueError(f"Existing remote command node {node_name!r} not found; available: {available}")
    if len(matches) > 1:
        raise ValueError(f"Existing remote command node {node_name!r} is duplicated")
    return matches[0]


def _render_existing_argv(
    node: ExistingRemoteCommandNode,
    *,
    remote_input_path: str,
    remote_output_dir: str,
) -> list[str]:
    """Render an existing-command argv template with remote input/output paths."""
    values = {"input": remote_input_path, "output": remote_output_dir}
    try:
        return [part.format(**values) for part in node.command_argv]
    except KeyError as exc:
        raise ValueError(
            f"ExistingRemoteCommandNode {node.name!r}: unknown command_argv placeholder {exc.args[0]!r}"
        ) from exc


def render_existing_remote_shell_command(
    node: ExistingRemoteCommandNode,
    *,
    remote_input_path: str,
    remote_output_dir: str,
) -> str:
    """Render the remote shell command for an already-installed remote tool."""
    exports = [
        f"export {key}={shlex.quote(value)}"
        for key, value in sorted((node.env or {}).items())
    ]
    inner_parts = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(remote_output_dir)}",
        *exports,
        shlex.join(
            _render_existing_argv(
                node,
                remote_input_path=remote_input_path,
                remote_output_dir=remote_output_dir,
            )
        ),
    ]
    inner_script = "; ".join(inner_parts)
    if node.scheduler == "direct":
        return inner_script
    return f"{shlex.join(['srun', *node.srun_args])} bash -lc {shlex.quote(inner_script)}"


def plan_existing_remote_command(
    node: ExistingRemoteCommandNode,
    *,
    local_input_path: Path | str,
    run_id: str,
    local_collect_dir: Path | str,
) -> ExistingRemoteCommandPlan:
    """Build a concrete stage/execute/collect plan for one existing remote command."""
    _validate_path_segment("run_id", run_id)
    input_path = Path(local_input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Local input file not found: {input_path}")
    input_name = input_path.name
    _validate_path_segment("input filename", input_name)
    remote_run_dir = PurePosixPath(node.work_dir) / run_id
    remote_input_path = str(remote_run_dir / "input" / input_name)
    remote_output_dir = str(remote_run_dir / "output")
    local_collect_path = Path(local_collect_dir)
    execute_script = render_existing_remote_shell_command(
        node,
        remote_input_path=remote_input_path,
        remote_output_dir=remote_output_dir,
    )
    prepare_script = (
        f"set -euo pipefail; mkdir -p {shlex.quote(str(PurePosixPath(remote_input_path).parent))} "
        f"{shlex.quote(remote_output_dir)}"
    )
    upload_target = f"{node.sync_host}:{remote_input_path}"
    upload_command: list[str] = ["rsync", "-a"]
    collect_source = f"{node.sync_host}:{remote_output_dir}/"
    collect_command: list[str] = ["rsync", "-a"]
    if node.sync_ssh_args:
        upload_command.extend(["-e", shlex.join(["ssh", *node.sync_ssh_args])])
        collect_command.extend(["-e", shlex.join(["ssh", *node.sync_ssh_args])])
    upload_command.extend([str(input_path), upload_target])
    collect_command.extend([collect_source, str(local_collect_path) + "/"])
    return ExistingRemoteCommandPlan(
        node_name=node.name,
        run_id=run_id,
        local_input_path=str(input_path),
        remote_input_path=remote_input_path,
        remote_output_dir=remote_output_dir,
        local_collect_dir=str(local_collect_path),
        prepare_command=("ssh", *node.ssh_args, node.host, "bash", "-lc", shlex.quote(prepare_script)),
        upload_command=tuple(upload_command),
        execute_command=("ssh", *node.ssh_args, node.host, "bash", "-lc", shlex.quote(execute_script)),
        collect_command=tuple(collect_command),
    )


def run_existing_remote_command(
    node: ExistingRemoteCommandNode,
    *,
    local_input_path: Path | str,
    run_id: str,
    local_collect_dir: Path | str,
    runner: Any = subprocess.run,
    timeout_sec: float | None = None,
) -> ExistingRemoteCommandResult:
    """Run one existing remote command and collect its output directory."""
    plan = plan_existing_remote_command(
        node,
        local_input_path=local_input_path,
        run_id=run_id,
        local_collect_dir=local_collect_dir,
    )
    Path(plan.local_collect_dir).mkdir(parents=True, exist_ok=True)
    runner_kwargs: dict[str, Any] = {"check": True, "text": True, "capture_output": True}
    if timeout_sec is not None:
        runner_kwargs["timeout"] = timeout_sec
    command_results: list[ExistingRemoteCommandStepResult] = []
    for step, command in (
        ("prepare", plan.prepare_command),
        ("upload", plan.upload_command),
        ("execute", plan.execute_command),
        ("collect", plan.collect_command),
    ):
        completed = runner(list(command), **runner_kwargs)
        command_results.append(
            ExistingRemoteCommandStepResult(
                step=step,
                returncode=int(getattr(completed, "returncode", 0)),
                stdout=str(getattr(completed, "stdout", "") or ""),
                stderr=str(getattr(completed, "stderr", "") or ""),
            )
        )
    collected_files = tuple(
        sorted(
            str(path.relative_to(plan.local_collect_dir))
            for path in Path(plan.local_collect_dir).rglob("*")
            if path.is_file()
        )
    )
    return ExistingRemoteCommandResult(
        plan=plan,
        collected_files=collected_files,
        command_results=tuple(command_results),
    )


def scaffold_remote_node(
    name: str,
    home_dir: str,
    *,
    host: str | None = None,
    scheduler: RemoteScheduler = "direct",
    repo_name: str = "proto-tools",
    work_name: str = "proto_remote",
    bootstrap_python: str = "python3",
    weight: float = 1.0,
    max_concurrent_buckets: int = 1,
    max_active_slurm_jobs: int | None = None,
    worker_device: str | None = None,
    sbatch_args: tuple[str, ...] = (),
) -> RemoteNode:
    """Create a conventional node profile entry from a remote home directory."""
    home = PurePosixPath(home_dir)
    if not home.is_absolute():
        raise ValueError(f"Remote node {name!r}: home_dir must be an absolute path")
    repo_dir = home / repo_name
    venv_dir = repo_dir / ".venv"
    rendered_work_name = work_name.format(node=name)
    return RemoteNode(
        name=name,
        host=host or name,
        work_dir=str(home / rendered_work_name),
        repo_dir=str(repo_dir),
        venv_dir=str(venv_dir),
        bootstrap_python=bootstrap_python,
        weight=weight,
        max_concurrent_buckets=max_concurrent_buckets,
        max_active_slurm_jobs=max_active_slurm_jobs,
        scheduler=scheduler,
        worker_device=worker_device or ("cuda" if scheduler == "slurm" else "cuda:0"),
        python=str(venv_dir / "bin" / "python"),
        sbatch_args=sbatch_args,
    )


def validate_remote_nodes(nodes: list[RemoteNode]) -> list[RemoteProfileIssue]:
    """Return static profile issues without opening SSH connections."""
    issues: list[RemoteProfileIssue] = []
    if not nodes:
        return [
            RemoteProfileIssue(
                level="error",
                node_name=None,
                field="nodes",
                message="profile must contain at least one node",
            )
        ]

    seen: dict[str, int] = {}
    for node in nodes:
        seen[node.name] = seen.get(node.name, 0) + 1
    for name, count in seen.items():
        if count > 1:
            issues.append(
                RemoteProfileIssue(
                    level="error",
                    node_name=name,
                    field="name",
                    message=f"duplicate node name appears {count} times",
                )
            )

    roots_by_path: dict[str, list[str]] = {}
    for node in nodes:
        roots_by_path.setdefault(str(PurePosixPath(node.work_dir)), []).append(node.name)
    for raw_path, names in roots_by_path.items():
        if len(names) > 1:
            issues.append(
                RemoteProfileIssue(
                    level="error",
                    node_name=",".join(sorted(names)),
                    field="work_dir",
                    message=f"remote work_dir is shared by multiple nodes: {raw_path}",
                )
            )

    for field in ("repo_dir", "venv_dir"):
        paths_by_value: dict[str, list[str]] = {}
        for node in nodes:
            candidate_path = node.repo_dir if field == "repo_dir" else node.effective_venv_dir
            if candidate_path is not None:
                paths_by_value.setdefault(str(PurePosixPath(candidate_path)), []).append(node.name)
        for shared_path, names in paths_by_value.items():
            if len(names) > 1:
                issues.append(
                    RemoteProfileIssue(
                        level="warning",
                        node_name=",".join(sorted(names)),
                        field=field,
                        message=f"remote {field} is shared by multiple nodes: {shared_path}",
                    )
                )

    for node in nodes:
        path_fields: tuple[tuple[str, str | None], ...] = (
            ("work_dir", node.work_dir),
            ("repo_dir", node.repo_dir),
            ("venv_dir", node.effective_venv_dir),
        )
        for field, candidate_path in path_fields:
            if candidate_path is not None and not PurePosixPath(candidate_path).is_absolute():
                issues.append(
                    RemoteProfileIssue(
                        level="error",
                        node_name=node.name,
                        field=field,
                        message=f"{field} must be an absolute remote path",
                    )
                )
        if node.repo_dir is None:
            issues.append(
                RemoteProfileIssue(
                    level="warning",
                    node_name=node.name,
                    field="repo_dir",
                    message="manager startup will not set a repo-local PYTHONPATH",
                )
            )
        if node.scheduler == "slurm" and not node.sbatch_args:
            issues.append(
                RemoteProfileIssue(
                    level="warning",
                    node_name=node.name,
                    field="sbatch_args",
                    message="Slurm node has no sbatch arguments; cluster defaults will be used",
                )
            )
        if node.scheduler == "slurm" and node.max_active_slurm_jobs is None:
            issues.append(
                RemoteProfileIssue(
                    level="error",
                    node_name=node.name,
                    field="max_active_slurm_jobs",
                    message="Slurm node requires max_active_slurm_jobs to avoid unbounded sbatch submission",
                )
            )
        if node.scheduler == "direct" and node.sbatch_args:
            issues.append(
                RemoteProfileIssue(
                    level="warning",
                    node_name=node.name,
                    field="sbatch_args",
                    message="direct node has sbatch arguments that manager-loop will ignore",
                )
            )
        if node.scheduler == "direct" and node.max_active_slurm_jobs is not None:
            issues.append(
                RemoteProfileIssue(
                    level="warning",
                    node_name=node.name,
                    field="max_active_slurm_jobs",
                    message="direct node has max_active_slurm_jobs that manager-loop will ignore",
                )
            )
        if node.scheduler == "direct" and node.max_concurrent_buckets > 1:
            issues.append(
                RemoteProfileIssue(
                    level="warning",
                    node_name=node.name,
                    field="max_concurrent_buckets",
                    message=(
                        "direct node max_concurrent_buckets only affects local planning capacity; "
                        "one node manager still executes buckets serially"
                    ),
                )
            )
    return issues


def _remote_bash_command(node: RemoteNode, script: str) -> list[str]:
    """Build an SSH command that runs ``script`` under remote bash."""
    return ["ssh", *node.ssh_args, node.host, "bash", "-lc", shlex.quote(script)]


def _remote_env_prefix(node: RemoteNode) -> str:
    """Shell exports required before running repo-local manager commands."""
    if node.repo_dir is None:
        return ""
    return f"cd {shlex.quote(node.repo_dir)} && export PYTHONPATH={shlex.quote(node.repo_dir)}${{PYTHONPATH:+:$PYTHONPATH}} && "


def render_rsync_upload_command(node: RemoteNode, local_repo_dir: Path | str) -> list[str]:
    """Render the rsync command that uploads a repo checkout to one node."""
    if node.repo_dir is None:
        raise ValueError(f"RemoteNode {node.name!r}: repo_dir is required for deployment")
    source = str(Path(local_repo_dir)) + "/"
    cmd = [
        "rsync",
        "-az",
        "--partial",
        "--delete",
        "--exclude",
        ".git",
        "--exclude",
        ".venv",
        "--exclude",
        ".pytest_cache",
        "--exclude",
        ".ruff_cache",
        "--exclude",
        "__pycache__",
        "--exclude",
        "logs",
    ]
    if node.sync_ssh_args:
        cmd.extend(["-e", shlex.join(["ssh", *node.sync_ssh_args])])
    cmd.extend([source, f"{node.host}:{node.repo_dir}/"])
    return cmd


def render_deploy_commands(node: RemoteNode, local_repo_dir: Path | str) -> list[list[str]]:
    """Render explicit deployment commands for a remote node.

    The commands create the manager directories, rsync the current checkout,
    create a virtualenv, and install the checkout editable. They are rendered
    only; callers decide when to execute them.
    """
    if node.repo_dir is None:
        raise ValueError(f"RemoteNode {node.name!r}: repo_dir is required for deployment")
    venv_dir = node.effective_venv_dir
    if venv_dir is None:
        raise ValueError(f"RemoteNode {node.name!r}: venv_dir is required for deployment")
    mkdir_script = (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(node.repo_dir)} "
        f"{shlex.quote(str(node.queue_root))} "
        f"{shlex.quote(str(node.results_root))} "
        f"{shlex.quote(str(node.logs_root))}"
    )
    install_script = (
        "set -euo pipefail; "
        f"cd {shlex.quote(node.repo_dir)}; "
        f"{shlex.quote(node.deploy_python)} -m venv {shlex.quote(venv_dir)}; "
        f"{shlex.quote(str(PurePosixPath(venv_dir) / 'bin' / 'python'))} -m pip install -e ."
    )
    return [
        _remote_bash_command(node, mkdir_script),
        render_rsync_upload_command(node, local_repo_dir),
        _remote_bash_command(node, install_script),
    ]


def inspect_local_checkout(
    local_repo_dir: Path | str,
    *,
    runner: Any = subprocess.run,
) -> LocalCheckoutStatus:
    """Inspect local git state before rendering or applying deployment."""
    requested = Path(local_repo_dir).resolve()

    def run_git(repo: Path, *args: str) -> str:
        result = runner(
            ["git", "-C", str(repo), *args],
            check=True,
            text=True,
            capture_output=True,
        )
        return str(getattr(result, "stdout", "") or "").strip()

    try:
        repo = Path(run_git(requested, "rev-parse", "--show-toplevel")).resolve()
    except subprocess.CalledProcessError as exc:
        return LocalCheckoutStatus(
            path=str(requested),
            head_commit=None,
            branch=None,
            dirty_count=0,
            untracked_count=0,
            status_lines=(),
            is_git_checkout=False,
            error=str(exc),
        )

    status_lines = tuple(
        line for line in run_git(repo, "status", "--porcelain=v1", "--untracked-files=all").splitlines() if line
    )
    untracked_count = sum(1 for line in status_lines if line.startswith("??"))
    dirty_count = len(status_lines) - untracked_count
    return LocalCheckoutStatus(
        path=str(repo),
        head_commit=run_git(repo, "rev-parse", "HEAD"),
        branch=run_git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        dirty_count=dirty_count,
        untracked_count=untracked_count,
        status_lines=status_lines,
    )


def shared_remote_install_roots(nodes: list[RemoteNode]) -> tuple[str, ...]:
    """Return shared repo/venv roots that would make concurrent deploy unsafe."""
    messages: list[str] = []
    for field in ("repo_dir", "venv_dir"):
        roots: dict[str, list[str]] = {}
        for node in nodes:
            root = node.repo_dir if field == "repo_dir" else node.effective_venv_dir
            if root is not None:
                roots.setdefault(root, []).append(node.name)
        for root, names in roots.items():
            if len(names) > 1:
                messages.append(f"{field} {root} shared by {','.join(sorted(names))}")
    return tuple(messages)


def build_deploy_render_report(
    nodes: list[RemoteNode],
    local_repo_dir: Path | str,
    *,
    allow_dirty_checkout: bool = False,
    runner: Any = subprocess.run,
) -> RemoteDeploymentRenderReport:
    """Build a no-submit deployment render report."""
    checkout = inspect_local_checkout(local_repo_dir, runner=runner)
    profile_issues = tuple(validate_remote_nodes(nodes))
    blockers = [issue.message for issue in profile_issues if issue.level == "error"]
    warnings = [issue.message for issue in profile_issues if issue.level == "warning"]
    actions: list[str] = []

    if not checkout.is_git_checkout:
        blockers.append(f"local repo dir is not a git checkout: {checkout.path}")
        actions.append("run deploy from a git checkout")
    elif checkout.dirty and allow_dirty_checkout:
        warnings.append(
            f"deploying local checkout with dirty_count={checkout.dirty_count} "
            f"untracked_count={checkout.untracked_count}"
        )
    elif checkout.dirty:
        blockers.append(
            f"local checkout has {checkout.dirty_count} tracked and "
            f"{checkout.untracked_count} untracked changed paths"
        )
        actions.append("commit, stash, or pass --allow-dirty-checkout")

    shared_roots = shared_remote_install_roots(nodes)
    blockers.extend(shared_roots)
    if shared_roots:
        actions.append("select one shared-root node or pass --allow-shared-install-root before deploy apply")

    node_plans: list[RemoteDeployNodePlan] = []
    for node in nodes:
        node_blocked = False
        if node.repo_dir is None:
            blockers.append(f"{node.name}: repo_dir is required for deployment")
            node_blocked = True
        if node.effective_venv_dir is None:
            blockers.append(f"{node.name}: venv_dir is required for deployment")
            node_blocked = True
        if node.scheduler == "slurm":
            joined_args = " ".join(node.sbatch_args).lower()
            if "gpu" not in joined_args:
                warnings.append(f"{node.name}: Slurm sbatch_args do not mention a GPU resource")
            if node.name.lower() == "h100" and "h100" not in joined_args:
                warnings.append(f"{node.name}: Slurm sbatch_args do not mention h100")

        commands: tuple[tuple[str, ...], ...] = ()
        if not node_blocked:
            commands = tuple(tuple(command) for command in render_deploy_commands(node, checkout.path))
        node_plans.append(
            RemoteDeployNodePlan(
                node_name=node.name,
                host=node.host,
                scheduler=node.scheduler,
                work_dir=node.work_dir,
                repo_dir=node.repo_dir,
                venv_dir=node.effective_venv_dir,
                commands=commands,
            )
        )

    return RemoteDeploymentRenderReport(
        ok=not blockers,
        local_repo_dir=checkout.path,
        local_checkout=checkout,
        profile_issues=profile_issues,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        actions=tuple(dict.fromkeys(actions)),
        nodes=tuple(node_plans),
    )


@dataclass(frozen=True)
class RemoteBucket:
    """A contiguous group of iterable input rows."""

    bucket_id: str
    item_indices: tuple[int, ...]
    total_cost: float

    def __post_init__(self) -> None:
        """Validate bucket identifiers before they become filenames."""
        _validate_path_segment("RemoteBucket.bucket_id", self.bucket_id)


@dataclass(frozen=True)
class RemoteAssignment:
    """One bucket assigned to one remote node."""

    node: RemoteNode
    bucket: RemoteBucket


@dataclass(frozen=True)
class RemoteDispatchPlan:
    """Complete bucket assignment for one tool invocation."""

    tool_key: str
    run_id: str
    input_fields: tuple[str, ...]
    iterable_output_field: str
    inputs: BaseToolInput
    config: BaseConfig
    assignments: tuple[RemoteAssignment, ...]

    def __post_init__(self) -> None:
        """Validate plan identifiers before queue/result paths are rendered."""
        _validate_path_segment("RemoteDispatchPlan.run_id", self.run_id)

    def request_for(self, assignment: RemoteAssignment) -> dict[str, Any]:
        """Build the JSON-serializable request for one assignment."""
        update = {
            field: [getattr(self.inputs, field)[idx] for idx in assignment.bucket.item_indices]
            for field in self.input_fields
        }
        bucket_inputs = self.inputs.model_copy(update=update)
        bucket_config = self.config.model_copy(update={"device": assignment.node.worker_device})
        return {
            "schema_version": _SCHEMA_VERSION,
            "tool_key": self.tool_key,
            "run_id": self.run_id,
            "bucket_id": assignment.bucket.bucket_id,
            "item_indices": list(assignment.bucket.item_indices),
            "inputs": bucket_inputs.model_dump(mode="json", exclude_none=True),
            "config": bucket_config.model_dump(mode="json", exclude_none=True),
        }

    def assignments_for_node(self, node_name: str) -> tuple[RemoteAssignment, ...]:
        """Return assignments for ``node_name`` in deterministic order."""
        return tuple(assignment for assignment in self.assignments if assignment.node.name == node_name)


@dataclass(frozen=True)
class RemoteRunRequest:
    """One manifest-backed bucket request."""

    node_name: str
    bucket_id: str
    item_indices: tuple[int, ...]
    total_cost: float
    request: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate manifest request path components."""
        _validate_path_segment("RemoteRunRequest.node_name", self.node_name)
        _validate_path_segment("RemoteRunRequest.bucket_id", self.bucket_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RemoteRunRequest":
        """Create a run request from JSON-compatible data."""
        return cls(
            node_name=str(data["node_name"]),
            bucket_id=str(data["bucket_id"]),
            item_indices=tuple(int(index) for index in data["item_indices"]),
            total_cost=float(data["total_cost"]),
            request=cast(dict[str, Any], data["request"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the run request to JSON-compatible data."""
        return {
            "node_name": self.node_name,
            "bucket_id": self.bucket_id,
            "item_indices": list(self.item_indices),
            "total_cost": self.total_cost,
            "request": self.request,
        }


@dataclass(frozen=True)
class RemoteRunManifest:
    """Disk-backed remote run plan for resumable submit, wait, and collect."""

    manifest_schema_version: int
    request_schema_version: int
    run_id: str
    tool_key: str
    bucket_size: int
    input_fields: tuple[str, ...]
    iterable_output_field: str
    local_results_dir: str
    nodes: tuple[RemoteNode, ...]
    requests: tuple[RemoteRunRequest, ...]

    def __post_init__(self) -> None:
        """Validate manifest path components and request identity."""
        _validate_path_segment("RemoteRunManifest.run_id", self.run_id)
        node_names = [node.name for node in self.nodes]
        duplicate_nodes = sorted({name for name in node_names if node_names.count(name) > 1})
        if duplicate_nodes:
            raise ValueError(f"Remote run manifest has duplicate node name(s): {duplicate_nodes}")
        known_nodes = set(node_names)
        bucket_ids = [request.bucket_id for request in self.requests]
        duplicate_buckets = sorted({bucket_id for bucket_id in bucket_ids if bucket_ids.count(bucket_id) > 1})
        if duplicate_buckets:
            raise ValueError(f"Remote run manifest has duplicate bucket id(s): {duplicate_buckets}")
        for request in self.requests:
            if request.node_name not in known_nodes:
                raise ValueError(f"Remote run manifest references unknown node {request.node_name!r}")
            payload = request.request
            identity_checks = (
                ("run_id", self.run_id, payload.get("run_id")),
                ("bucket_id", request.bucket_id, payload.get("bucket_id")),
                ("tool_key", self.tool_key, payload.get("tool_key")),
                ("item_indices", list(request.item_indices), payload.get("item_indices")),
            )
            mismatches = [
                f"{field} expected {expected!r} got {actual!r}"
                for field, expected, actual in identity_checks
                if actual != expected
            ]
            if mismatches:
                raise ValueError(
                    f"Remote run manifest request identity mismatch for {request.node_name}/{request.bucket_id}: "
                    + "; ".join(mismatches)
                )

    @classmethod
    def from_plan(
        cls,
        plan: RemoteDispatchPlan,
        *,
        nodes: list[RemoteNode],
        bucket_size: int,
        local_results_dir: Path | str,
    ) -> "RemoteRunManifest":
        """Create a manifest from an in-memory dispatch plan."""
        nodes_by_name = {node.name: node for node in nodes}
        missing_nodes = sorted({assignment.node.name for assignment in plan.assignments} - set(nodes_by_name))
        if missing_nodes:
            raise ValueError(f"Manifest nodes missing assignments for: {missing_nodes}")
        return cls(
            manifest_schema_version=_MANIFEST_SCHEMA_VERSION,
            request_schema_version=_SCHEMA_VERSION,
            run_id=plan.run_id,
            tool_key=plan.tool_key,
            bucket_size=bucket_size,
            input_fields=plan.input_fields,
            iterable_output_field=plan.iterable_output_field,
            local_results_dir=str(Path(local_results_dir)),
            nodes=tuple(nodes),
            requests=tuple(
                RemoteRunRequest(
                    node_name=assignment.node.name,
                    bucket_id=assignment.bucket.bucket_id,
                    item_indices=assignment.bucket.item_indices,
                    total_cost=assignment.bucket.total_cost,
                    request=plan.request_for(assignment),
                )
                for assignment in plan.assignments
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RemoteRunManifest":
        """Create a manifest from JSON-compatible data."""
        manifest_version = int(data["manifest_schema_version"])
        request_version = int(data["request_schema_version"])
        if manifest_version != _MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported remote run manifest_schema_version: {manifest_version!r}")
        if request_version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported remote run request_schema_version: {request_version!r}")
        return cls(
            manifest_schema_version=manifest_version,
            request_schema_version=request_version,
            run_id=str(data["run_id"]),
            tool_key=str(data["tool_key"]),
            bucket_size=int(data["bucket_size"]),
            input_fields=tuple(str(field) for field in data["input_fields"]),
            iterable_output_field=str(data["iterable_output_field"]),
            local_results_dir=str(data["local_results_dir"]),
            nodes=tuple(RemoteNode.from_dict(node) for node in data["nodes"]),
            requests=tuple(RemoteRunRequest.from_dict(request) for request in data["requests"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to JSON-compatible data."""
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "request_schema_version": self.request_schema_version,
            "run_id": self.run_id,
            "tool_key": self.tool_key,
            "bucket_size": self.bucket_size,
            "input_fields": list(self.input_fields),
            "iterable_output_field": self.iterable_output_field,
            "local_results_dir": self.local_results_dir,
            "nodes": [node.to_dict() for node in self.nodes],
            "requests": [request.to_dict() for request in self.requests],
        }

    def node_for_request(self, request: RemoteRunRequest) -> RemoteNode:
        """Return the node assigned to one manifest request."""
        nodes_by_name = {node.name: node for node in self.nodes}
        try:
            return nodes_by_name[request.node_name]
        except KeyError as exc:
            raise ValueError(f"Remote run manifest references unknown node {request.node_name!r}") from exc

    def local_result_path(self, request: RemoteRunRequest) -> Path:
        """Return the local result path expected after rsync mirroring."""
        return Path(self.local_results_dir) / request.node_name / self.run_id / f"{request.bucket_id}.json"


@dataclass(frozen=True)
class RemoteLaunchResult:
    """Summary of a manifest-backed remote launch."""

    run_id: str
    manifest_path: str
    output_path: str
    local_results_dir: str
    bucket_count: int
    staged_count: int
    preflight: dict[str, Any]
    managers: dict[str, str]
    rsync_status: str | None = None
    diagnostics: dict[str, Any] | None = None
    initial_node_loads: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize launch summary to JSON-compatible data."""
        payload = {
            "run_id": self.run_id,
            "manifest": self.manifest_path,
            "output": self.output_path,
            "local_results_dir": self.local_results_dir,
            "bucket_count": self.bucket_count,
            "staged_count": self.staged_count,
            "preflight": self.preflight,
            "managers": self.managers,
            "rsync_status": self.rsync_status,
        }
        if self.diagnostics is not None:
            payload["diagnostics"] = self.diagnostics
        if self.initial_node_loads is not None:
            payload["initial_node_loads"] = self.initial_node_loads
        return payload


@dataclass(frozen=True)
class ExistingRemoteBatchLaunchResult:
    """Summary of a completed manager-backed existing-command batch."""

    run_id: str
    task_count: int
    assignment: dict[str, int]
    manifest: str
    local_results_dir: str
    local_artifacts_dir: str
    summary: str
    preflight: dict[str, Any]
    managers: dict[str, str]
    admission: dict[str, dict[str, Any]]
    rebalances: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the user-facing launch result."""
        return {
            "run_id": self.run_id,
            "task_count": self.task_count,
            "assignment": self.assignment,
            "manifest": self.manifest,
            "local_results_dir": self.local_results_dir,
            "local_artifacts_dir": self.local_artifacts_dir,
            "summary": self.summary,
            "preflight": self.preflight,
            "managers": self.managers,
            "admission": self.admission,
            "rebalances": list(self.rebalances),
        }


def write_remote_compute_receipt(payload: dict[str, Any], path: Path | str) -> None:
    """Persist the terminal high-level command result for later audit."""
    _write_json_atomic(Path(path), payload, overwrite=True)


class RemoteDispatchPlanner:
    """Plan bucketed remote execution across heterogeneous nodes."""

    def __init__(
        self,
        nodes: list[RemoteNode],
        bucket_size: int,
        *,
        initial_node_loads: dict[str, float] | None = None,
    ):
        """Initialize a planner with node capabilities and bucket size."""
        if not nodes:
            raise ValueError("RemoteDispatchPlanner requires at least one node")
        if bucket_size < 1:
            raise ValueError(f"bucket_size must be >= 1, got {bucket_size}")
        names = [node.name for node in nodes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate remote node names: {duplicates}")
        load_names = set(initial_node_loads or {})
        unknown_load_names = sorted(load_names - set(names))
        if unknown_load_names:
            raise ValueError(f"Initial node loads reference unknown node(s): {unknown_load_names}")
        for node_name, load in (initial_node_loads or {}).items():
            if load < 0:
                raise ValueError(f"Initial node load for {node_name!r} must be >= 0, got {load}")
        self.nodes = tuple(nodes)
        self.bucket_size = bucket_size
        self.initial_node_loads = dict(initial_node_loads or {})

    def build_plan(
        self,
        tool_key: str,
        inputs: BaseToolInput | dict[str, Any],
        config: BaseConfig | dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
    ) -> RemoteDispatchPlan:
        """Build a remote dispatch plan for an iterable-input tool."""
        spec = ToolRegistry.get(tool_key)
        if spec.iterable_input_fields is None or spec.iterable_output_field is None:
            raise ValueError(f"Tool {tool_key!r} does not declare iterable inputs for remote bucketing")

        typed_inputs = inputs if isinstance(inputs, spec.input_model) else spec.input_model.model_validate(inputs)
        typed_config: BaseConfig
        if config is None:
            typed_config = cast(BaseConfig, spec.config_model())
        elif isinstance(config, spec.config_model):
            typed_config = config
        else:
            typed_config = cast(BaseConfig, spec.config_model.model_validate(config))

        active_fields = tuple(
            field for field in spec.iterable_input_fields if getattr(typed_inputs, field, None) is not None
        )
        if not active_fields:
            raise ValueError(f"Tool {tool_key!r} has no populated iterable input fields")
        primary_field = active_fields[0]
        primary_items = list(getattr(typed_inputs, primary_field))
        for field in active_fields[1:]:
            values = getattr(typed_inputs, field)
            if len(values) != len(primary_items):
                raise ValueError(
                    f"Remote iterable field {field!r} (len {len(values)}) is not aligned "
                    f"with primary {primary_field!r} (len {len(primary_items)})"
                )

        buckets = self._make_buckets(type(typed_inputs), primary_items)
        assignments = self._assign_buckets(buckets)
        return RemoteDispatchPlan(
            tool_key=tool_key,
            run_id=run_id or f"{tool_key}-{uuid.uuid4().hex}",
            input_fields=active_fields,
            iterable_output_field=spec.iterable_output_field,
            inputs=typed_inputs,
            config=typed_config,
            assignments=assignments,
        )

    def _make_buckets(self, input_model: type[BaseToolInput], items: list[Any]) -> tuple[RemoteBucket, ...]:
        """Split input rows into contiguous buckets."""
        buckets: list[RemoteBucket] = []
        for bucket_index, start in enumerate(range(0, len(items), self.bucket_size)):
            end = min(start + self.bucket_size, len(items))
            indices = tuple(range(start, end))
            total_cost = sum(float(input_model.item_cost(items[idx])) for idx in indices)
            buckets.append(
                RemoteBucket(bucket_id=f"bucket-{bucket_index:05d}", item_indices=indices, total_cost=total_cost)
            )
        return tuple(buckets)

    def _assign_buckets(self, buckets: tuple[RemoteBucket, ...]) -> tuple[RemoteAssignment, ...]:
        """Assign buckets by LPT using node weight and concurrency as capacity."""
        loads = {node.name: self.initial_node_loads.get(node.name, 0.0) for node in self.nodes}
        nodes_by_name = {node.name: node for node in self.nodes}
        assignments: list[RemoteAssignment] = []
        for bucket in sorted(buckets, key=lambda item: item.total_cost, reverse=True):
            node_name = min(
                loads,
                key=lambda name: (
                    loads[name] / (nodes_by_name[name].weight * nodes_by_name[name].max_concurrent_buckets)
                ),
            )
            node = nodes_by_name[node_name]
            assignments.append(RemoteAssignment(node=node, bucket=bucket))
            loads[node.name] += bucket.total_cost
        return tuple(
            sorted(assignments, key=lambda item: item.bucket.item_indices[0] if item.bucket.item_indices else -1)
        )


class RemoteSubmissionClient:
    """Stage planned bucket requests into node-side manager queues over SSH."""

    _STAGE_MANY_PY = (
        "import json, os, pathlib, sys\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "emit_summary = len(sys.argv) > 2 and sys.argv[2] == '--summary'\n"
        "payload = json.load(sys.stdin)\n"
        "state_root = root.parent.parent\n"
        "root.mkdir(parents=True, exist_ok=True)\n"
        "summary = {'staged': [], 'skipped': []}\n"
        "def same_request(path, request):\n"
        "    try:\n"
        "        return json.loads(path.read_text()) == request\n"
        "    except Exception as exc:\n"
        "        raise SystemExit(f'unreadable existing request at {path}: {type(exc).__name__}: {exc}')\n"
        "def same_completed_result(path, request):\n"
        "    try:\n"
        "        envelope = json.loads(path.read_text())\n"
        "    except Exception as exc:\n"
        "        raise SystemExit(f'unreadable existing result at {path}: {type(exc).__name__}: {exc}')\n"
        "    expected = {\n"
        "        'run_id': request.get('run_id'),\n"
        "        'bucket_id': request.get('bucket_id'),\n"
        "        'tool_key': request.get('tool_key'),\n"
        "        'item_indices': request.get('item_indices', []),\n"
        "    }\n"
        "    actual = {key: envelope.get(key) for key in expected}\n"
        "    if actual != expected:\n"
        "        raise SystemExit(f'existing result differs from manifest at {path}')\n"
        "    if envelope.get('status') == 'failed':\n"
        "        raise SystemExit(f'request previously failed at {path}; use a new run_id or clear it after review')\n"
        "    if envelope.get('status') == 'completed' and actual == expected:\n"
        "        return True\n"
        "    raise SystemExit(f'existing non-completed result at {path}; clear it after review')\n"
        "for item in payload['requests']:\n"
        "    name = item['filename']\n"
        "    if '/' in name or name.startswith('.'):\n"
        "        raise SystemExit(f'invalid request filename: {name!r}')\n"
        "    request = item['request']\n"
        "    relative = pathlib.Path(root.name) / name\n"
        "    protected = {\n"
        "        state: state_root / state / relative\n"
        "        for state in ('queue', 'running', 'submitted', 'done', 'failed')\n"
        "    }\n"
        "    result_path = state_root / 'results' / root.name / name\n"
        "    for state, path in protected.items():\n"
        "        if not path.exists():\n"
        "            continue\n"
        "        if state == 'running':\n"
        "            if not same_request(path, request):\n"
        "                raise SystemExit(\n"
        "                    f'existing running request differs from manifest at {path}; use a new run_id or clear it after review'\n"
        "                )\n"
        "            if result_path.exists() and same_completed_result(result_path, request):\n"
        "                summary['skipped'].append({'filename': name, 'state': 'running_result', 'path': str(result_path)})\n"
        "                break\n"
        "            summary['skipped'].append({'filename': name, 'state': 'running', 'path': str(path)})\n"
        "            break\n"
        "        if state == 'failed':\n"
        "            raise SystemExit(\n"
        "                f'request previously failed at {path}; use a new run_id or clear it after review'\n"
        "            )\n"
        "        if same_request(path, request):\n"
        "            summary['skipped'].append({'filename': name, 'state': state, 'path': str(path)})\n"
        "            break\n"
        "        raise SystemExit(\n"
        "            f'existing {state} request differs from manifest at {path}; use a new run_id or clear it after review'\n"
        "        )\n"
        "    else:\n"
        "        if result_path.exists() and same_completed_result(result_path, request):\n"
        "            summary['skipped'].append({'filename': name, 'state': 'result', 'path': str(result_path)})\n"
        "            continue\n"
        "        final = root / name\n"
        "        tmp = root / ('.' + name + '.tmp')\n"
        "        tmp.write_text(json.dumps(request, sort_keys=True) + '\\n')\n"
        "        os.replace(tmp, final)\n"
        "        summary['staged'].append({'filename': name, 'path': str(final)})\n"
        "        continue\n"
        "    continue\n"
        "if emit_summary:\n"
        "    print(json.dumps(summary, sort_keys=True))\n"
    )

    def render_stage_script(self, node: RemoteNode, run_id: str, bucket_id: str) -> str:
        """Return the remote shell script used to atomically stage one request."""
        _validate_path_segment("run_id", run_id)
        _validate_path_segment("bucket_id", bucket_id)
        queue_dir = node.queue_root / run_id
        final_path = queue_dir / f"{bucket_id}.json"
        tmp_path = queue_dir / f".{bucket_id}.json.tmp"
        return (
            f"mkdir -p {shlex.quote(str(queue_dir))} && "
            f"umask 077 && "
            f"cat > {shlex.quote(str(tmp_path))} && "
            f"mv {shlex.quote(str(tmp_path))} {shlex.quote(str(final_path))}"
        )

    def render_stage_many_script(self, node: RemoteNode, run_id: str, *, summary: bool = False) -> str:
        """Return the remote shell script used to stage many requests at once."""
        _validate_path_segment("run_id", run_id)
        queue_dir = node.queue_root / run_id
        summary_arg = " --summary" if summary else ""
        return (
            "umask 077 && "
            f"{shlex.quote(node.python)} -c {shlex.quote(self._STAGE_MANY_PY)} "
            f"{shlex.quote(str(queue_dir))}{summary_arg}"
        )

    def submit_assignment(
        self,
        plan: RemoteDispatchPlan,
        assignment: RemoteAssignment,
        *,
        runner: Any = subprocess.run,
    ) -> PurePosixPath:
        """Stage one assignment request over SSH and return its remote queue path."""
        payload = json.dumps(plan.request_for(assignment), sort_keys=True) + "\n"
        script = self.render_stage_script(assignment.node, plan.run_id, assignment.bucket.bucket_id)
        cmd = ["ssh", *assignment.node.ssh_args, assignment.node.host, script]
        runner(cmd, input=payload, text=True, check=True)
        return assignment.node.queue_root / plan.run_id / f"{assignment.bucket.bucket_id}.json"

    def submit_node_assignments(
        self,
        plan: RemoteDispatchPlan,
        node: RemoteNode,
        assignments: list[RemoteAssignment],
        *,
        runner: Any = subprocess.run,
    ) -> list[PurePosixPath]:
        """Stage all assignments for one node through one SSH command."""
        if not assignments:
            return []
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": plan.run_id,
            "requests": [
                {
                    "filename": f"{assignment.bucket.bucket_id}.json",
                    "request": plan.request_for(assignment),
                }
                for assignment in assignments
            ],
        }
        script = self.render_stage_many_script(node, plan.run_id)
        cmd = ["ssh", *node.ssh_args, node.host, script]
        runner(cmd, input=json.dumps(payload, sort_keys=True) + "\n", text=True, check=True)
        return [node.queue_root / plan.run_id / f"{assignment.bucket.bucket_id}.json" for assignment in assignments]

    def submit_plan(
        self,
        plan: RemoteDispatchPlan,
        *,
        runner: Any = subprocess.run,
    ) -> list[PurePosixPath]:
        """Stage every bucket request in a plan."""
        assignments_by_node: dict[str, list[RemoteAssignment]] = {}
        nodes_by_name: dict[str, RemoteNode] = {}
        for assignment in plan.assignments:
            assignments_by_node.setdefault(assignment.node.name, []).append(assignment)
            nodes_by_name[assignment.node.name] = assignment.node

        paths_by_bucket: dict[str, PurePosixPath] = {}
        for node_name, assignments in assignments_by_node.items():
            node_paths = self.submit_node_assignments(
                plan,
                nodes_by_name[node_name],
                assignments,
                runner=runner,
            )
            for assignment, path in zip(assignments, node_paths, strict=True):
                paths_by_bucket[assignment.bucket.bucket_id] = path
        return [paths_by_bucket[assignment.bucket.bucket_id] for assignment in plan.assignments]


def load_remote_run_manifest(path: Path | str) -> RemoteRunManifest:
    """Load a remote run manifest from disk."""
    return RemoteRunManifest.from_dict(json.loads(Path(path).read_text()))


def write_remote_run_manifest(
    manifest: RemoteRunManifest,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    """Write a remote run manifest atomically."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Remote run manifest already exists: {output_path}")
    _write_json_atomic(output_path, manifest.to_dict())


def plan_existing_remote_command_batch(
    nodes: list[ExistingRemoteBatchNode],
    *,
    input_fasta: Path | str,
    run_id: str,
    bucket_size: int,
    local_results_dir: Path | str,
    software: str,
    initial_node_loads: dict[str, float] | None = None,
) -> RemoteRunManifest:
    """Freeze a FASTA batch as node-specific existing-command manager requests."""
    _validate_path_segment("run_id", run_id)
    _validate_path_segment("software", software)
    if not nodes:
        raise ValueError("Existing remote command batch requires at least one node")
    manager_nodes = [node.manager for node in nodes]
    _raise_for_remote_profile_errors(manager_nodes)
    command_by_name = {node.command.name: node.command for node in nodes}
    if len(command_by_name) != len(nodes):
        raise ValueError("Existing remote command batch has duplicate node names")
    for node in nodes:
        command = node.command
        if command.status != "ready":
            raise ValueError(
                f"Existing remote command profile {command.name!r} has status={command.status!r}; ready is required"
            )
        if command.software != software:
            raise ValueError(
                f"Existing remote command profile {command.name!r} software={command.software!r}, "
                f"expected {software!r}"
            )
        if not command.required_artifacts:
            raise ValueError(f"Existing remote command profile {command.name!r} has no required_artifacts")
    _validate_existing_batch_gpu_contract(nodes, software)

    records = read_fasta_records(input_fasta)
    buckets: list[RemoteBucket] = []
    for bucket_index, start in enumerate(range(0, len(records), bucket_size)):
        end = min(start + bucket_size, len(records))
        indices = tuple(range(start, end))
        buckets.append(
            RemoteBucket(
                bucket_id=f"bucket-{bucket_index:05d}",
                item_indices=indices,
                total_cost=float(sum(len(records[index].sequence) for index in indices)),
            )
        )
    planner = RemoteDispatchPlanner(
        manager_nodes,
        bucket_size,
        initial_node_loads=initial_node_loads,
    )
    assignments = planner._assign_buckets(tuple(buckets))
    requests: list[RemoteRunRequest] = []
    artifact_destinations: set[str] = set()
    for assignment in assignments:
        manager = assignment.node
        command = command_by_name[manager.name]
        resource_admission: dict[str, Any] | None = None
        if command.gpu_admission is not None:
            gpu_id = (command.env or {})["CUDA_VISIBLE_DEVICES"].split(",", 1)[0].strip()
            if not gpu_id:
                raise ValueError(
                    f"Existing remote command profile {command.name!r} has an empty CUDA_VISIBLE_DEVICES"
                )
            resource_admission = {
                "kind": "cuda_idle",
                "gpu_id": gpu_id,
                **command.gpu_admission,
            }
        artifact_root = PurePosixPath(command.work_dir) / run_id / "artifacts"
        tasks: list[dict[str, Any]] = []
        for index in assignment.bucket.item_indices:
            task_id = f"task_{index:05d}"
            artifact_dir = artifact_root / task_id / f"attempt_{manager.name}"
            artifact_text = str(artifact_dir)
            if artifact_text in artifact_destinations:
                raise ValueError(f"Duplicate existing command artifact destination: {artifact_text}")
            artifact_destinations.add(artifact_text)
            record = records[index]
            tasks.append(
                {
                    "task_id": task_id,
                    "header": record.header,
                    "sequence_length": len(record.sequence),
                    "input_filename": f"{task_id}.fasta",
                    "input_text": record.to_text(),
                    "artifact_dir": artifact_text,
                    "required_artifacts": list(command.required_artifacts),
                }
            )
        existing_command = {
            "software": software,
            "node_name": manager.name,
            "artifact_root": str(artifact_root),
            "command_argv": list(command.command_argv),
            "env": command.env or {},
            "required_stdout_patterns": list(command.required_stdout_patterns),
            "tasks": tasks,
        }
        if resource_admission is not None:
            existing_command["resource_admission"] = resource_admission
        request = {
            "schema_version": _SCHEMA_VERSION,
            "request_kind": "existing_command",
            "tool_key": f"existing-{software}",
            "run_id": run_id,
            "bucket_id": assignment.bucket.bucket_id,
            "item_indices": list(assignment.bucket.item_indices),
            "existing_command": existing_command,
        }
        requests.append(
            RemoteRunRequest(
                node_name=manager.name,
                bucket_id=assignment.bucket.bucket_id,
                item_indices=assignment.bucket.item_indices,
                total_cost=assignment.bucket.total_cost,
                request=request,
            )
        )
    return RemoteRunManifest(
        manifest_schema_version=_MANIFEST_SCHEMA_VERSION,
        request_schema_version=_SCHEMA_VERSION,
        run_id=run_id,
        tool_key=f"existing-{software}",
        bucket_size=bucket_size,
        input_fields=("fasta_records",),
        iterable_output_field="existing_command_results",
        local_results_dir=str(Path(local_results_dir).resolve()),
        nodes=tuple(manager_nodes),
        requests=tuple(requests),
    )


def render_existing_artifact_pull_commands(
    manifest: RemoteRunManifest,
    local_artifacts_dir: Path | str,
) -> list[list[str]]:
    """Render one rsync pull per assigned node for existing-command artifacts."""
    local_root = Path(local_artifacts_dir)
    requests_by_node: dict[str, list[RemoteRunRequest]] = {}
    for request in manifest.requests:
        requests_by_node.setdefault(request.node_name, []).append(request)
    commands: list[list[str]] = []
    for node in manifest.nodes:
        requests = requests_by_node.get(node.name, [])
        if not requests:
            continue
        roots = {
            str(request.request["existing_command"]["artifact_root"])
            for request in requests
        }
        if len(roots) != 1:
            raise ValueError(f"Existing command manifest has inconsistent artifact roots for node {node.name!r}")
        remote_root = roots.pop()
        cmd = ["rsync", "-a"]
        if node.sync_ssh_args:
            cmd.extend(["-e", shlex.join(["ssh", *node.sync_ssh_args])])
        cmd.extend(
            [
                f"{node.sync_host}:{remote_root}/",
                str(local_root / node.name) + "/",
            ]
        )
        commands.append(cmd)
    return commands


def collect_existing_remote_command_manifest(
    manifest: RemoteRunManifest,
    local_artifacts_dir: Path | str,
) -> dict[str, Any]:
    """Validate completed existing-command envelopes and their mirrored native artifacts."""
    local_root = Path(local_artifacts_dir)
    rows_by_index: dict[int, dict[str, Any]] = {}
    for node, request, result_path in _manifest_result_entries(manifest):
        envelope = cast(dict[str, Any], json.loads(result_path.read_text()))
        _validate_manifest_result_envelope(manifest, node, request, result_path, envelope)
        command_results = envelope.get("existing_command_results")
        tasks = request.request.get("existing_command", {}).get("tasks")
        if not isinstance(command_results, list) or not isinstance(tasks, list):
            raise RuntimeError(f"Existing command result is missing task rows: {node.name}/{request.bucket_id}")
        if len(command_results) != len(request.item_indices) or len(tasks) != len(request.item_indices):
            raise RuntimeError(
                f"Existing command result count mismatch for {node.name}/{request.bucket_id}: "
                f"results={len(command_results)} tasks={len(tasks)} items={len(request.item_indices)}"
            )
        result_by_task = {str(result.get("task_id")): result for result in command_results}
        for index, task in zip(request.item_indices, tasks, strict=True):
            task_id = str(task["task_id"])
            result = result_by_task.get(task_id)
            if result is None:
                raise RuntimeError(f"Existing command result missing {task_id!r} for {node.name}/{request.bucket_id}")
            expected_stdout_patterns = request.request["existing_command"].get(
                "required_stdout_patterns", []
            )
            validated_stdout_patterns = result.get("validated_stdout_patterns", [])
            if validated_stdout_patterns != expected_stdout_patterns:
                raise RuntimeError(
                    f"Existing command stdout evidence mismatch for {task_id!r}: "
                    f"expected={expected_stdout_patterns!r} validated={validated_stdout_patterns!r}"
                )
            local_task_dir = local_root / node.name / task_id / f"attempt_{node.name}"
            required = task.get("required_artifacts")
            if not isinstance(required, list):
                raise RuntimeError(f"Existing command manifest task {task_id!r} has no required_artifacts")
            artifacts = _required_existing_artifacts(local_task_dir / "output", required)
            envelope_artifacts = sorted(str(path) for path in result.get("artifacts", []))
            if artifacts != envelope_artifacts:
                raise RuntimeError(
                    f"Existing command artifact inventory mismatch for {task_id!r}: "
                    f"envelope={envelope_artifacts!r} local={artifacts!r}"
                )
            rows_by_index[index] = {
                "item_index": index,
                "task_id": task_id,
                "header": str(task["header"]),
                "sequence_length": int(task["sequence_length"]),
                "node": node.name,
                "local_artifact_dir": str(local_task_dir),
                "artifacts": artifacts,
            }
            if validated_stdout_patterns:
                rows_by_index[index]["validated_stdout_patterns"] = list(validated_stdout_patterns)
    expected = {index for request in manifest.requests for index in request.item_indices}
    if set(rows_by_index) != expected:
        raise RuntimeError(
            f"Existing command collected task indices differ: expected={sorted(expected)} actual={sorted(rows_by_index)}"
        )
    tasks = [rows_by_index[index] for index in sorted(rows_by_index)]
    return {
        "run_id": manifest.run_id,
        "tool_key": manifest.tool_key,
        "task_count": len(tasks),
        "tasks": tasks,
    }


def ensure_remote_run_output_path_available(path: Path | str, *, overwrite: bool = False) -> None:
    """Raise when a remote run output path would be overwritten."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Remote run output already exists: {output_path}")


def write_remote_run_output(
    output: BaseToolOutput,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    """Write a collected remote run output atomically."""
    output_path = Path(path)
    _write_text_atomic(
        output_path,
        str(output.model_dump_json(indent=2)) + "\n",
        overwrite=overwrite,
        exists_message=f"Remote run output already exists: {output_path}",
    )


def submit_remote_run_manifest(
    manifest: RemoteRunManifest,
    *,
    bucket_ids: set[str] | None = None,
    runner: Any = subprocess.run,
) -> list[PurePosixPath]:
    """Stage all or a selected set of bucket requests recorded in a manifest."""
    submitter = RemoteSubmissionClient()
    known_bucket_ids = {request.bucket_id for request in manifest.requests}
    if bucket_ids is not None and not bucket_ids <= known_bucket_ids:
        raise ValueError(f"Unknown manifest bucket ids: {sorted(bucket_ids - known_bucket_ids)}")
    selected_requests = [
        request
        for request in manifest.requests
        if bucket_ids is None or request.bucket_id in bucket_ids
    ]
    requests_by_node: dict[str, list[RemoteRunRequest]] = {}
    for request in selected_requests:
        requests_by_node.setdefault(request.node_name, []).append(request)

    paths_by_bucket: dict[str, PurePosixPath] = {}
    for node in manifest.nodes:
        node_requests = requests_by_node.get(node.name, [])
        if not node_requests:
            continue
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": manifest.run_id,
            "requests": [
                {
                    "filename": f"{request.bucket_id}.json",
                    "request": request.request,
                }
                for request in node_requests
            ],
        }
        script = submitter.render_stage_many_script(node, manifest.run_id)
        cmd = ["ssh", *node.ssh_args, node.host, script]
        runner(cmd, input=json.dumps(payload, sort_keys=True) + "\n", text=True, check=True)
        for request in node_requests:
            paths_by_bucket[request.bucket_id] = node.queue_root / manifest.run_id / f"{request.bucket_id}.json"

    return [paths_by_bucket[request.bucket_id] for request in selected_requests]


def _manifest_result_entries(manifest: RemoteRunManifest) -> list[tuple[RemoteNode, RemoteRunRequest, Path]]:
    """Return expected local result entries for one manifest."""
    return [
        (
            manifest.node_for_request(request),
            request,
            manifest.local_result_path(request),
        )
        for request in manifest.requests
    ]


def _validate_manifest_result_envelope(
    manifest: RemoteRunManifest,
    node: RemoteNode,
    request: RemoteRunRequest,
    path: Path,
    envelope: dict[str, Any],
) -> None:
    """Raise when a mirrored manifest result is not the exact completed bucket."""
    identity_checks = (
        ("run_id", manifest.run_id, envelope.get("run_id")),
        ("bucket_id", request.bucket_id, envelope.get("bucket_id")),
        ("tool_key", manifest.tool_key, envelope.get("tool_key")),
        ("item_indices", list(request.item_indices), envelope.get("item_indices")),
    )
    mismatches = [
        f"{field} expected {expected!r} got {actual!r}"
        for field, expected, actual in identity_checks
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError(
            f"Remote run {manifest.run_id} got result for wrong identity at "
            f"{node.name}/{request.bucket_id}: {'; '.join(mismatches)} (result_path={path})"
        )

    status = envelope.get("status")
    if status == "failed":
        trace_tail = str(envelope.get("traceback", "")).splitlines()[-1:] or [""]
        raise RuntimeError(
            f"Remote run {manifest.run_id} failed: {node.name}/{request.bucket_id}: "
            f"{envelope.get('error', '<no error>')} (result_path={path}; trace_tail={trace_tail[0]})"
        )
    if status != "completed":
        raise RuntimeError(
            f"Remote run {manifest.run_id} has non-completed result for "
            f"{node.name}/{request.bucket_id}: status={status!r} (result_path={path})"
        )


def _missing_or_raise_for_invalid_manifest_results(manifest: RemoteRunManifest) -> list[Path]:
    """Return missing manifest result paths after validating every visible envelope."""
    missing: list[Path] = []
    for node, request, path in _manifest_result_entries(manifest):
        if not path.exists():
            missing.append(path)
            continue
        try:
            envelope = cast(dict[str, Any], json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Remote run {manifest.run_id} has unreadable result for "
                f"{node.name}/{request.bucket_id}: {exc} (result_path={path})"
            ) from exc
        _validate_manifest_result_envelope(manifest, node, request, path, envelope)
    return missing


def _tail_text(path: Path | str, line_count: int) -> str:
    """Return the last ``line_count`` lines from a local text file."""
    text_path = Path(path)
    if not text_path.exists():
        return ""
    return "\n".join(text_path.read_text(errors="replace").splitlines()[-line_count:])


def _raise_if_background_rsync_inactive(
    manifest: RemoteRunManifest,
    missing: list[Path],
    *,
    rsync_pid_path: Path | str | None,
    rsync_log_path: Path | str | None,
    rsync_client: "LocalRsyncPullClient | None",
    rsync_log_tail_lines: int,
    runner: Any,
) -> None:
    """Raise when background rsync health is requested but its pid is not running."""
    if rsync_pid_path is None or not missing:
        return
    status = (rsync_client or LocalRsyncPullClient()).status(rsync_pid_path, runner=runner).strip()
    if status.startswith("running "):
        return
    preview = ", ".join(str(path) for path in missing[:3])
    message = (
        f"Remote run {manifest.run_id} rsync pull loop is not running: status={status!r}; "
        f"pid_path={rsync_pid_path}; waiting for {len(missing)} result(s): {preview}"
    )
    if rsync_log_path is not None:
        tail = _tail_text(rsync_log_path, rsync_log_tail_lines)
        message += f"; log_path={rsync_log_path}; log_tail={tail or '<empty or missing>'}"
    raise RuntimeError(message)


def _run_inline_rsync_pull(command: list[str], *, runner: Any) -> str | None:
    """Run one inline rsync command and return a retryable transport error."""
    try:
        runner(command, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = str(getattr(exc, "stderr", "") or "").strip()
        return f"{exc}; stderr={stderr or '<empty>'}"
    return None


def wait_remote_run_manifest(
    manifest: RemoteRunManifest,
    *,
    poll_interval: float = 5.0,
    timeout_sec: float | None = None,
    pull_results: bool = True,
    rsync_pid_path: Path | str | None = None,
    rsync_log_path: Path | str | None = None,
    rsync_client: "LocalRsyncPullClient | None" = None,
    rsync_log_tail_lines: int = 40,
    poll_hook: Callable[[RemoteRunManifest], RemoteRunManifest] | None = None,
    runner: Any = subprocess.run,
) -> RemoteRunManifest:
    """Wait until every manifest result exists locally."""
    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
    if timeout_sec is not None and timeout_sec <= 0:
        raise ValueError(f"timeout_sec must be > 0, got {timeout_sec}")
    if rsync_log_tail_lines < 1:
        raise ValueError(f"rsync_log_tail_lines must be >= 1, got {rsync_log_tail_lines}")
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
    last_pull_error: str | None = None
    while True:
        if poll_hook is not None:
            manifest = poll_hook(manifest)
        missing = _missing_or_raise_for_invalid_manifest_results(manifest)
        if not missing:
            return manifest
        if pull_results:
            pull_failed = False
            for cmd in render_rsync_pull_commands(list(manifest.nodes), manifest.local_results_dir):
                pull_error = _run_inline_rsync_pull(cmd, runner=runner)
                if pull_error is not None:
                    pull_failed = True
                    last_pull_error = pull_error
                    logger.warning(
                        "Remote run %s rsync pull attempt failed; will retry within the wait timeout: %s",
                        manifest.run_id,
                        last_pull_error,
                    )
            if not pull_failed:
                last_pull_error = None
            missing = _missing_or_raise_for_invalid_manifest_results(manifest)
            if not missing:
                return manifest
        _raise_if_background_rsync_inactive(
            manifest,
            missing,
            rsync_pid_path=rsync_pid_path,
            rsync_log_path=rsync_log_path,
            rsync_client=rsync_client,
            rsync_log_tail_lines=rsync_log_tail_lines,
            runner=runner,
        )
        if deadline is not None and time.monotonic() >= deadline:
            preview = ", ".join(str(path) for path in missing[:3])
            message = f"Remote run {manifest.run_id} timed out waiting for {len(missing)} result(s): {preview}"
            if last_pull_error is not None:
                message += f"; last rsync pull error: {last_pull_error}"
            raise TimeoutError(message)
        time.sleep(poll_interval)


def collect_remote_run_manifest(manifest: RemoteRunManifest) -> BaseToolOutput:
    """Merge completed manifest result files into the tool's output model."""
    spec = ToolRegistry.get(manifest.tool_key)
    output_items: dict[int, Any] = {}
    warnings: list[str] = []
    errors: list[str] = []
    metadata: dict[str, Any] = {"remote_dispatch": {"run_id": manifest.run_id, "buckets": len(manifest.requests)}}

    for node, request, result_path in _manifest_result_entries(manifest):
        envelope = cast(dict[str, Any], json.loads(result_path.read_text()))
        _validate_manifest_result_envelope(manifest, node, request, result_path, envelope)
        output = spec.output_model.model_validate(envelope["output"])
        bucket_outputs = list(getattr(output, manifest.iterable_output_field))
        if len(bucket_outputs) != len(request.item_indices):
            raise RuntimeError(
                f"Remote bucket {request.bucket_id} returned {len(bucket_outputs)} "
                f"{manifest.iterable_output_field} for {len(request.item_indices)} input item(s)"
            )
        output_items.update(dict(zip(request.item_indices, bucket_outputs, strict=True)))
        warnings.extend(output.warnings)
        errors.extend(output.errors)

    expected = {index for request in manifest.requests for index in request.item_indices}
    missing = sorted(expected - set(output_items))
    if missing:
        raise RuntimeError(f"Missing remote output item(s): {missing}")
    merged = [output_items[index] for index in sorted(output_items)]
    output_payload: dict[str, Any] = {
        manifest.iterable_output_field: merged,
        "warnings": warnings,
        "errors": errors,
        "metadata": metadata,
    }
    return cast(BaseToolOutput, spec.output_model.model_validate(output_payload))


def _raise_for_remote_profile_errors(nodes: list[RemoteNode]) -> None:
    """Raise when static remote profile validation reports errors."""
    errors = [issue for issue in validate_remote_nodes(nodes) if issue.level == "error"]
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid remote profile: {first.node_name or '-'} {first.field}: {first.message}")


def _raise_for_preflight_failures(preflight: dict[str, Any]) -> None:
    """Raise when any remote preflight reports ``ok=false``."""
    failed = {name: payload for name, payload in preflight.items() if not payload.get("ok")}
    if failed:
        raise ValueError(f"Remote preflight failed: {json.dumps(failed, sort_keys=True)}")


def _pending_bucket_count_from_diagnostics(payload: dict[str, Any]) -> int:
    """Return pending bucket count from manager diagnostics."""
    counts = payload.get("counts") or {}
    return sum(int(counts.get(key, 0) or 0) for key in ("queue", "running", "submitted"))


def remote_run_diagnostics_backlog_loads(
    nodes: list[RemoteNode],
    *,
    manager_client: "RemoteManagerClient | None" = None,
    timeout_sec: float | None = None,
    runner: Any = subprocess.run,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Read manager diagnostics and convert visible backlog into planner loads."""
    manager = manager_client or RemoteManagerClient()
    diagnostics: dict[str, Any] = {}
    for node in nodes:
        if timeout_sec is None:
            diagnostics[node.name] = manager.diagnostics_node(node, runner=runner)
        else:
            diagnostics[node.name] = manager.diagnostics_node(node, timeout_sec=timeout_sec, runner=runner)
    initial_node_loads = {
        node.name: float(_pending_bucket_count_from_diagnostics(diagnostics[node.name])) for node in nodes
    }
    return diagnostics, initial_node_loads


def _manager_preflight_node(
    manager: "RemoteManagerClient",
    node: RemoteNode,
    *,
    timeout_sec: float | None,
    runner: Any,
) -> dict[str, Any]:
    """Run preflight with an optional timeout without changing narrow test fakes."""
    if timeout_sec is None:
        return manager.preflight_node(node, runner=runner)
    return manager.preflight_node(node, timeout_sec=timeout_sec, runner=runner)


def _manager_start_node(
    manager: "RemoteManagerClient",
    node: RemoteNode,
    *,
    poll_interval: float,
    slurm_group_size: int,
    timeout_sec: float | None,
    runner: Any,
) -> str:
    """Start a manager with an optional timeout without changing narrow test fakes."""
    if timeout_sec is None:
        return manager.start_node(
            node,
            poll_interval=poll_interval,
            slurm_group_size=slurm_group_size,
            runner=runner,
        )
    return manager.start_node(
        node,
        poll_interval=poll_interval,
        slurm_group_size=slurm_group_size,
        timeout_sec=timeout_sec,
        runner=runner,
    )


def _validate_remote_run_rsync_mode(
    rsync_mode: RemoteRsyncMode,
    *,
    rsync_pid_path: Path | str | None,
    rsync_log_path: Path | str | None,
) -> None:
    """Validate rsync mode arguments for manifest-backed runs."""
    if rsync_mode not in ("inline", "background"):
        raise ValueError(f"Unsupported rsync_mode {rsync_mode!r}")
    if rsync_mode == "background" and (rsync_pid_path is None or rsync_log_path is None):
        raise ValueError("background rsync_mode requires rsync_pid_path and rsync_log_path")


def _submit_wait_collect_remote_run_manifest(
    manifest: RemoteRunManifest,
    *,
    manifest_path: Path | str,
    output_path: Path | str,
    preflight: dict[str, Any],
    managers: dict[str, str],
    diagnostics: dict[str, Any] | None,
    initial_node_loads: dict[str, float] | None,
    wait_poll_interval: float,
    timeout_sec: float | None,
    rsync_mode: RemoteRsyncMode,
    rsync_pid_path: Path | str | None,
    rsync_log_path: Path | str | None,
    rsync_interval_sec: int,
    rsync_client: "LocalRsyncPullClient | None",
    overwrite_output: bool,
    runner: Any,
) -> RemoteLaunchResult:
    """Submit, wait for, collect, and summarize one manifest-backed run."""
    _validate_remote_run_rsync_mode(
        rsync_mode,
        rsync_pid_path=rsync_pid_path,
        rsync_log_path=rsync_log_path,
    )
    ensure_remote_run_output_path_available(output_path, overwrite=overwrite_output)
    nodes = list(manifest.nodes)
    local_results = Path(manifest.local_results_dir)
    rsync_status: str | None = None
    active_rsync: LocalRsyncPullClient | None = None
    if rsync_mode == "background":
        active_rsync = rsync_client or LocalRsyncPullClient()
        rsync_status = active_rsync.start(
            nodes,
            local_results,
            pid_path=cast(Path | str, rsync_pid_path),
            log_path=cast(Path | str, rsync_log_path),
            interval_sec=rsync_interval_sec,
            runner=runner,
        )

    staged = submit_remote_run_manifest(manifest, runner=runner)
    wait_remote_run_manifest(
        manifest,
        poll_interval=wait_poll_interval,
        timeout_sec=timeout_sec,
        pull_results=rsync_mode == "inline",
        rsync_pid_path=rsync_pid_path if rsync_mode == "background" else None,
        rsync_log_path=rsync_log_path if rsync_mode == "background" else None,
        rsync_client=active_rsync if rsync_mode == "background" else None,
        runner=runner,
    )
    output = collect_remote_run_manifest(manifest)
    write_remote_run_output(output, output_path, overwrite=overwrite_output)
    return RemoteLaunchResult(
        run_id=manifest.run_id,
        manifest_path=str(manifest_path),
        output_path=str(output_path),
        local_results_dir=manifest.local_results_dir,
        bucket_count=len(manifest.requests),
        staged_count=len(staged),
        preflight=preflight,
        managers=managers,
        rsync_status=rsync_status,
        diagnostics=diagnostics,
        initial_node_loads=initial_node_loads,
    )


def resume_remote_run_manifest(
    manifest: RemoteRunManifest,
    *,
    manifest_path: Path | str,
    output_path: Path | str,
    manager_poll_interval: float = 5.0,
    slurm_group_size: int = 10,
    wait_poll_interval: float = 5.0,
    timeout_sec: float | None = None,
    rsync_mode: RemoteRsyncMode = "inline",
    rsync_pid_path: Path | str | None = None,
    rsync_log_path: Path | str | None = None,
    rsync_interval_sec: int = 10,
    overwrite_output: bool = False,
    remote_check_timeout_sec: float | None = None,
    manager_client: "RemoteManagerClient | None" = None,
    rsync_client: "LocalRsyncPullClient | None" = None,
    runner: Any = subprocess.run,
) -> RemoteLaunchResult:
    """Resume an existing manifest by starting managers, submitting, waiting, and collecting."""
    ensure_remote_run_output_path_available(output_path, overwrite=overwrite_output)
    nodes = list(manifest.nodes)
    _raise_for_remote_profile_errors(nodes)
    manager = manager_client or RemoteManagerClient()
    preflight = {
        node.name: _manager_preflight_node(
            manager,
            node,
            timeout_sec=remote_check_timeout_sec,
            runner=runner,
        )
        for node in nodes
    }
    _raise_for_preflight_failures(preflight)
    managers = {
        node.name: _manager_start_node(
            manager,
            node,
            poll_interval=manager_poll_interval,
            slurm_group_size=slurm_group_size,
            timeout_sec=remote_check_timeout_sec,
            runner=runner,
        )
        for node in nodes
    }
    return _submit_wait_collect_remote_run_manifest(
        manifest,
        manifest_path=manifest_path,
        output_path=output_path,
        preflight=preflight,
        managers=managers,
        diagnostics=None,
        initial_node_loads=None,
        wait_poll_interval=wait_poll_interval,
        timeout_sec=timeout_sec,
        rsync_mode=rsync_mode,
        rsync_pid_path=rsync_pid_path,
        rsync_log_path=rsync_log_path,
        rsync_interval_sec=rsync_interval_sec,
        rsync_client=rsync_client,
        overwrite_output=overwrite_output,
        runner=runner,
    )


def _validate_existing_batch_gpu_contract(nodes: list[ExistingRemoteBatchNode], software: str) -> None:
    """Reject CPU demo profiles before GPU scientific work reaches a manager."""
    if software.lower() not in {"af3", "alphafold3"}:
        return
    for node in nodes:
        manager = node.manager
        if not manager.worker_device.startswith("cuda"):
            raise ValueError(
                f"AlphaFold3 existing batch node {manager.name!r} must use a CUDA worker_device, "
                f"got {manager.worker_device!r}"
            )
        if manager.scheduler == "direct" and not (node.command.env or {}).get("CUDA_VISIBLE_DEVICES"):
            raise ValueError(
                f"AlphaFold3 direct node {manager.name!r} must set CUDA_VISIBLE_DEVICES in command env"
            )
        if manager.scheduler == "direct" and node.command.gpu_admission is None:
            raise ValueError(
                f"AlphaFold3 direct node {manager.name!r} must configure gpu_admission so busy GPUs stay queued"
            )
        if manager.scheduler == "slurm" and not any("gpu" in arg for arg in manager.sbatch_args):
            raise ValueError(f"AlphaFold3 Slurm node {manager.name!r} must request a GPU in sbatch_args")
        if "CudaDevice" not in node.command.required_stdout_patterns:
            raise ValueError(
                f"AlphaFold3 existing batch node {manager.name!r} required_stdout_patterns "
                "must include 'CudaDevice' so CPU execution cannot pass artifact validation"
            )


def _existing_batch_runtime_preflight_command(node: ExistingRemoteBatchNode) -> list[str]:
    """Render a no-inference path and GPU visibility preflight for one existing command."""
    command = node.command
    executable = command.command_argv[0]
    if not PurePosixPath(executable).is_absolute():
        raise ValueError(f"Existing batch command executable must be absolute: {executable!r}")
    paths: list[str] = []
    for token in command.command_argv[1:]:
        candidate = token.rsplit("=", 1)[-1]
        if "{" in candidate or not PurePosixPath(candidate).is_absolute():
            continue
        paths.append(candidate)
    checks = [
        "set -euo pipefail",
        f"test -x {shlex.quote(executable)}",
        *(f"test -e {shlex.quote(path)}" for path in dict.fromkeys(paths)),
    ]
    if node.manager.scheduler == "direct":
        gpu_id = (command.env or {}).get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0]
        if not gpu_id:
            raise ValueError(f"Direct existing batch node {node.manager.name!r} has no CUDA_VISIBLE_DEVICES")
        checks.append(f"nvidia-smi -i {shlex.quote(gpu_id)} --query-gpu=index --format=csv,noheader >/dev/null")
    else:
        checks.extend(["command -v sbatch >/dev/null", "command -v sacct >/dev/null"])
    checks.append("printf 'runtime-ready\\n'")
    script = "; ".join(checks)
    return ["ssh", *node.manager.ssh_args, node.manager.host, "bash", "-lc", shlex.quote(script)]


def _ensure_existing_batch_manager_state(
    node: RemoteNode,
    *,
    timeout_sec: float | None,
    runner: Any,
) -> None:
    """Create only the manager state directories authorized by the profile."""
    root = PurePosixPath(node.work_dir)
    paths = [
        root,
        node.queue_root,
        node.results_root,
        node.logs_root,
        *(root / name for name in ("running", "done", "failed", "submitted", "cancelled")),
    ]
    remote_command = shlex.join(["mkdir", "-p", *(str(path) for path in paths)])
    kwargs: dict[str, Any] = {"check": True, "text": True, "capture_output": True}
    if timeout_sec is not None:
        kwargs["timeout"] = timeout_sec
    runner(["ssh", *node.ssh_args, node.host, remote_command], **kwargs)


def _existing_batch_admission_probe_command(node: ExistingRemoteBatchNode) -> list[str]:
    """Render one direct-GPU or Slurm scheduling admission probe."""
    if node.manager.scheduler == "slurm":
        sbatch_args = [arg for arg in node.manager.sbatch_args if arg != "--parsable"]
        remote_command = shlex.join(["sbatch", "--test-only", *sbatch_args, "--wrap=true"])
        return ["ssh", *node.manager.ssh_args, node.manager.host, remote_command]
    gpu_id = (node.command.env or {}).get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    if not gpu_id:
        raise ValueError(f"Direct existing batch node {node.manager.name!r} has no CUDA_VISIBLE_DEVICES")
    return [
        "ssh",
        *node.manager.ssh_args,
        node.manager.host,
        "nvidia-smi",
        "-i",
        gpu_id,
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]


def _probe_existing_batch_admission(
    node: ExistingRemoteBatchNode,
    *,
    timeout_sec: float | None,
    runner: Any,
) -> dict[str, Any]:
    """Return whether one node can accept newly planned work now."""
    kwargs: dict[str, Any] = {"check": True, "text": True, "capture_output": True}
    if timeout_sec is not None:
        kwargs["timeout"] = timeout_sec
    command = _existing_batch_admission_probe_command(node)
    completed = runner(command, **kwargs)
    if node.manager.scheduler == "slurm":
        stdout = str(getattr(completed, "stdout", "") or "").strip()
        stderr = str(getattr(completed, "stderr", "") or "").strip()
        output = "\n".join(part for part in (stdout, stderr) if part)
        match = re.search(r"\bto start at ([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8})\b", output)
        if match is None:
            raise ValueError(
                f"Existing batch Slurm admission probe for node {node.manager.name!r} returned "
                f"unexpected output: stdout={stdout!r}; stderr={stderr!r}"
            )
        start_at = match.group(1)
        start_delay_sec = max(0.0, datetime.fromisoformat(start_at).timestamp() - time.time())
        return {
            "kind": "slurm_test_only",
            "ready": start_delay_sec <= _SLURM_ADMISSION_MAX_START_DELAY_SEC,
            "start_at": start_at,
            "start_delay_sec": start_delay_sec,
            "max_start_delay_sec": _SLURM_ADMISSION_MAX_START_DELAY_SEC,
        }
    stdout = str(getattr(completed, "stdout", "") or "").strip()
    fields = [field.strip() for field in stdout.splitlines()[-1].split(",")] if stdout else []
    if len(fields) != 2:
        raise ValueError(
            f"Existing batch admission probe for node {node.manager.name!r} returned "
            f"unexpected output: stdout={stdout!r}; stderr={str(getattr(completed, 'stderr', '') or '')!r}"
        )
    utilization, memory = (int(field) for field in fields)
    thresholds = node.command.gpu_admission or {}
    max_utilization = int(thresholds["max_utilization_percent"])
    max_memory = int(thresholds["max_memory_used_mib"])
    return {
        "kind": "cuda_idle",
        "ready": utilization <= max_utilization and memory <= max_memory,
        "gpu_id": (node.command.env or {})["CUDA_VISIBLE_DEVICES"].split(",", 1)[0].strip(),
        "utilization_percent": utilization,
        "memory_used_mib": memory,
        "max_utilization_percent": max_utilization,
        "max_memory_used_mib": max_memory,
    }


def _select_existing_batch_planning_nodes(
    nodes: list[ExistingRemoteBatchNode],
    admission: dict[str, dict[str, Any]],
) -> list[ExistingRemoteBatchNode]:
    """Prefer admitted nodes, using direct queues as the holding pool when all are busy."""
    ready = [node for node in nodes if admission[node.manager.name]["ready"]]
    direct = [node for node in nodes if node.manager.scheduler == "direct"]
    return ready or direct or nodes


_WITHDRAW_QUEUED_REQUEST_PY = """import json, os, pathlib, sys
queue_path = pathlib.Path(sys.argv[1])
cancelled_path = pathlib.Path(sys.argv[2])
expected = json.load(sys.stdin)
def same(path):
    return json.loads(path.read_text()) == expected
if cancelled_path.exists():
    if not same(cancelled_path):
        raise SystemExit(f'cancelled request differs at {cancelled_path}')
    status = 'already_cancelled'
elif queue_path.exists():
    try:
        if not same(queue_path):
            raise SystemExit(f'queued request differs at {queue_path}')
        cancelled_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(queue_path, cancelled_path)
        status = 'withdrawn'
    except FileNotFoundError:
        status = 'not_queued'
else:
    status = 'not_queued'
print(json.dumps({'status': status, 'queue_path': str(queue_path), 'cancelled_path': str(cancelled_path)}, sort_keys=True))
"""


def _withdraw_queued_manifest_request(
    node: RemoteNode,
    request: RemoteRunRequest,
    *,
    cancellation_id: str,
    timeout_sec: float | None = None,
    retry_interval: float = 1.0,
    runner: Any,
) -> dict[str, Any]:
    """Atomically preserve and withdraw one exact unclaimed remote request."""
    _validate_path_segment("cancellation_id", cancellation_id)
    queue_path = node.queue_root / request.request["run_id"] / f"{request.bucket_id}.json"
    cancelled_path = (
        PurePosixPath(node.work_dir)
        / "cancelled"
        / request.request["run_id"]
        / request.bucket_id
        / f"{cancellation_id}.json"
    )
    remote_command = shlex.join(
        [
            node.python,
            "-c",
            _WITHDRAW_QUEUED_REQUEST_PY,
            str(queue_path),
            str(cancelled_path),
        ]
    )
    command = [
        "ssh",
        *node.ssh_args,
        node.host,
        remote_command,
    ]
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
    while True:
        kwargs: dict[str, Any] = {
            "input": json.dumps(request.request, sort_keys=True) + "\n",
            "check": True,
            "text": True,
            "capture_output": True,
        }
        if deadline is not None:
            kwargs["timeout"] = max(0.001, deadline - time.monotonic())
        try:
            completed = runner(command, **kwargs)
            break
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 255 or deadline is None or time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Remote queue withdrawal failed for {node.name}/{request.bucket_id}: "
                    f"stdout={str(exc.stdout or '')!r}; stderr={str(exc.stderr or '')!r}"
                ) from exc
            logger.warning(
                "Retrying SSH queue withdrawal for %s/%s after transport failure: %s",
                node.name,
                request.bucket_id,
                str(exc.stderr or exc),
            )
            time.sleep(min(retry_interval, max(0.0, deadline - time.monotonic())))
    return _parse_remote_manager_json_result(node, "queue withdrawal", completed)


def _manifest_replacing_request(
    manifest: RemoteRunManifest,
    replacement: RemoteRunRequest,
    *,
    batch_nodes: list[ExistingRemoteBatchNode],
) -> RemoteRunManifest:
    """Return a manifest with one bucket replacement and only referenced node snapshots."""
    requests = tuple(
        replacement if request.bucket_id == replacement.bucket_id else request
        for request in manifest.requests
    )
    referenced = {request.node_name for request in requests}
    managers = {node.manager.name: node.manager for node in batch_nodes}
    missing = sorted(referenced - set(managers))
    if missing:
        raise ValueError(f"Rebalanced manifest references unavailable profile node(s): {missing}")
    return RemoteRunManifest(
        manifest_schema_version=manifest.manifest_schema_version,
        request_schema_version=manifest.request_schema_version,
        run_id=manifest.run_id,
        tool_key=manifest.tool_key,
        bucket_size=manifest.bucket_size,
        input_fields=manifest.input_fields,
        iterable_output_field=manifest.iterable_output_field,
        local_results_dir=manifest.local_results_dir,
        nodes=tuple(node.manager for node in batch_nodes if node.manager.name in referenced),
        requests=requests,
    )


def _validate_existing_batch_resume_manifest(
    manifest: RemoteRunManifest,
    *,
    batch_nodes: list[ExistingRemoteBatchNode],
    input_fasta: Path | str,
    software: str,
) -> None:
    """Reject changed work while allowing journaled node reassignments."""
    nodes_by_name = {node.manager.name: node for node in batch_nodes}
    referenced = {request.node_name for request in manifest.requests}
    missing = sorted(referenced - set(nodes_by_name))
    if missing:
        raise ValueError(f"Existing batch profile is missing manifest node(s): {missing}")
    manifest_nodes = {node.name: node.to_dict() for node in manifest.nodes}
    expected_nodes = {
        name: nodes_by_name[name].manager.to_dict()
        for name in referenced
    }
    if manifest_nodes != expected_nodes:
        raise ValueError("Existing batch manifest node snapshots differ from the current profile")

    expected_by_node: dict[str, dict[str, RemoteRunRequest]] = {}
    for name in referenced:
        expected = plan_existing_remote_command_batch(
            [nodes_by_name[name]],
            input_fasta=input_fasta,
            run_id=manifest.run_id,
            bucket_size=manifest.bucket_size,
            local_results_dir=manifest.local_results_dir,
            software=software,
        )
        expected_by_node[name] = {request.bucket_id: request for request in expected.requests}
    expected_bucket_ids = set(next(iter(expected_by_node.values())))
    actual_bucket_ids = {request.bucket_id for request in manifest.requests}
    if len(actual_bucket_ids) != len(manifest.requests) or actual_bucket_ids != expected_bucket_ids:
        raise ValueError("Existing batch manifest buckets differ from the requested FASTA batch")
    mismatches = [
        request.bucket_id
        for request in manifest.requests
        if request.to_dict()
        != expected_by_node[request.node_name][request.bucket_id].to_dict()
    ]
    if mismatches:
        raise ValueError(
            "Requested existing batch does not match the existing manifest bucket(s): "
            + ", ".join(sorted(mismatches))
        )


def _rebalance_unclaimed_existing_batch_manifest(
    manifest: RemoteRunManifest,
    *,
    batch_nodes: list[ExistingRemoteBatchNode],
    input_fasta: Path | str,
    software: str,
    admission: dict[str, dict[str, Any]],
    manifest_path: Path | str,
    withdraw_timeout_sec: float | None = None,
    withdrawer: Any = _withdraw_queued_manifest_request,
    runner: Any = subprocess.run,
) -> tuple[RemoteRunManifest, list[dict[str, str]]]:
    """Move missing, unclaimed requests from unavailable nodes to admitted nodes."""
    _missing_or_raise_for_invalid_manifest_results(manifest)
    ready_nodes = [node for node in batch_nodes if admission[node.manager.name]["ready"]]
    incomplete_by_node = {
        node.manager.name: sum(
            1
            for request in manifest.requests
            if request.node_name == node.manager.name
            and not manifest.local_result_path(request).exists()
        )
        for node in ready_nodes
    }
    target_capacity = {
        node.manager.name: max(
            0,
            (
                1
                if node.manager.scheduler == "direct"
                else cast(int, node.manager.max_active_slurm_jobs)
            )
            - incomplete_by_node[node.manager.name],
        )
        for node in ready_nodes
    }
    target_slots = [
        node
        for slot_index in range(max(target_capacity.values(), default=0))
        for node in ready_nodes
        if target_capacity[node.manager.name] > slot_index
    ]
    candidates = [
        request
        for request in manifest.requests
        if not manifest.local_result_path(request).exists()
        and not admission[request.node_name]["ready"]
    ][: len(target_slots)]
    if not candidates or not target_slots:
        return manifest, []

    target_by_bucket = {
        request.bucket_id: target_slots[index].manager.name
        for index, request in enumerate(candidates)
    }
    selected_target_names = set(target_by_bucket.values())
    target_manifests = {
        node.manager.name: plan_existing_remote_command_batch(
            [node],
            input_fasta=input_fasta,
            run_id=manifest.run_id,
            bucket_size=manifest.bucket_size,
            local_results_dir=manifest.local_results_dir,
            software=software,
        )
        for node in ready_nodes
        if node.manager.name in selected_target_names
    }
    target_requests = {
        node_name: {request.bucket_id: request for request in target.requests}
        for node_name, target in target_manifests.items()
    }
    nodes_by_name = {node.manager.name: node.manager for node in batch_nodes}
    journal_root = Path(manifest_path).parent / "rebalances"
    moves: list[dict[str, str]] = []
    updated = manifest
    for candidate in candidates:
        target_name = target_by_bucket[candidate.bucket_id]
        replacement = target_requests[target_name][candidate.bucket_id]
        journal_identity = {
            "bucket_id": candidate.bucket_id,
            "from_node": candidate.node_name,
            "to_node": target_name,
            "old_request": candidate.to_dict(),
            "new_request": replacement.to_dict(),
        }
        prior_indices: list[int] = []
        incomplete: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(journal_root.glob(f"{candidate.bucket_id}.move-*.json")):
            match = re.search(r"\.move-([0-9]+)\.", path.name)
            if match is not None:
                prior_indices.append(int(match.group(1)))
            payload = cast(dict[str, Any], json.loads(path.read_text()))
            if payload.get("status") not in {"planned", "not_moved"}:
                continue
            if all(payload.get(key) == value for key, value in journal_identity.items()):
                incomplete.append((path, payload))
        if len(incomplete) > 1:
            raise RuntimeError(
                f"Multiple incomplete rebalance journals for {candidate.bucket_id}: "
                + ", ".join(str(path) for path, _ in incomplete)
            )
        if incomplete:
            journal_path, journal = incomplete[0]
            cancellation_id = str(journal["cancellation_id"])
            journal["status"] = "planned"
            journal.pop("withdrawal", None)
            _write_json_atomic(journal_path, journal, overwrite=True)
        else:
            move_index = max(prior_indices, default=0) + 1
            cancellation_id = (
                f"move-{move_index:05d}-{candidate.node_name}-to-{target_name}"
            )
            journal_path = journal_root / (
                f"{candidate.bucket_id}.move-{move_index:05d}."
                f"{candidate.node_name}-to-{target_name}.json"
            )
            journal = {
                "schema_version": 1,
                "status": "planned",
                "cancellation_id": cancellation_id,
                **journal_identity,
            }
            _write_json_atomic(journal_path, journal)

        withdrawal = withdrawer(
            nodes_by_name[candidate.node_name],
            candidate,
            cancellation_id=cancellation_id,
            timeout_sec=withdraw_timeout_sec,
            runner=runner,
        )
        if withdrawal.get("status") not in {"withdrawn", "already_cancelled"}:
            journal["status"] = "not_moved"
            journal["withdrawal"] = withdrawal
            _write_json_atomic(journal_path, journal, overwrite=True)
            continue
        updated = _manifest_replacing_request(updated, replacement, batch_nodes=batch_nodes)
        write_remote_run_manifest(updated, manifest_path, overwrite=True)
        journal["status"] = "completed"
        journal["withdrawal"] = withdrawal
        _write_json_atomic(journal_path, journal, overwrite=True)
        moves.append(
            {
                "bucket_id": candidate.bucket_id,
                "from_node": candidate.node_name,
                "to_node": target_name,
            }
        )
    return updated, moves


def launch_existing_remote_command_batch(
    nodes: list[ExistingRemoteBatchNode],
    *,
    input_fasta: Path | str,
    software: str,
    run_dir: Path | str,
    bucket_size: int = 1,
    slurm_group_size: int = 10,
    wait_poll_interval: float = 5.0,
    timeout_sec: float | None = None,
    remote_check_timeout_sec: float | None = None,
    manager_client: "RemoteManagerClient | None" = None,
    runner: Any = subprocess.run,
) -> ExistingRemoteBatchLaunchResult:
    """Plan, launch, wait for, mirror, and validate an existing-command FASTA batch."""
    run_root = Path(run_dir).resolve()
    run_id = _validate_path_segment("run_dir name", run_root.name)
    manifest_path = run_root / "run.manifest.json"
    local_results = run_root / "remote_results"
    local_artifacts = run_root / "artifacts"
    summary_path = run_root / "summary.json"
    _validate_existing_batch_gpu_contract(nodes, software)
    resume_existing = manifest_path.is_file()
    if run_root.exists() and not resume_existing:
        existing_files = [path for path in run_root.rglob("*") if path.is_file() or path.is_symlink()]
        if existing_files:
            raise FileExistsError(
                f"Existing batch run directory has no resumable manifest and is not empty: {run_root}"
            )
    manifest: RemoteRunManifest | None = None
    if resume_existing:
        manifest = load_remote_run_manifest(manifest_path)
        _validate_existing_batch_resume_manifest(
            manifest,
            batch_nodes=nodes,
            input_fasta=input_fasta,
            software=software,
        )

    runtime_preflight: dict[str, Any] = {}
    admission: dict[str, dict[str, Any]] = {}
    for batch_node in nodes:
        kwargs: dict[str, Any] = {"check": True, "text": True, "capture_output": True}
        if remote_check_timeout_sec is not None:
            kwargs["timeout"] = remote_check_timeout_sec
        completed = runner(_existing_batch_runtime_preflight_command(batch_node), **kwargs)
        runtime_preflight[batch_node.manager.name] = {
            "ok": True,
            "stdout": str(getattr(completed, "stdout", "") or ""),
            "stderr": str(getattr(completed, "stderr", "") or ""),
        }
        admission[batch_node.manager.name] = _probe_existing_batch_admission(
            batch_node,
            timeout_sec=remote_check_timeout_sec,
            runner=runner,
        )
    initial_admission = {name: dict(value) for name, value in admission.items()}

    rebalances: list[dict[str, str]] = []
    if resume_existing:
        manifest = cast(RemoteRunManifest, manifest)
        manifest, rebalances = _rebalance_unclaimed_existing_batch_manifest(
            manifest,
            batch_nodes=nodes,
            input_fasta=input_fasta,
            software=software,
            admission=admission,
            manifest_path=manifest_path,
            withdraw_timeout_sec=remote_check_timeout_sec,
            runner=runner,
        )
    else:
        planning_nodes = _select_existing_batch_planning_nodes(nodes, admission)
        manifest = plan_existing_remote_command_batch(
            planning_nodes,
            input_fasta=input_fasta,
            run_id=run_id,
            bucket_size=bucket_size,
            local_results_dir=local_results,
            software=software,
        )
    if not resume_existing:
        write_remote_run_manifest(manifest, manifest_path)

    manager = manager_client or RemoteManagerClient()
    manager_preflight: dict[str, dict[str, Any]] = {}
    managers: dict[str, str] = {}
    local_results.mkdir(parents=True, exist_ok=True)

    def activate_manifest_nodes(active_manifest: RemoteRunManifest) -> None:
        for manager_node in active_manifest.nodes:
            if manager_node.name not in manager_preflight:
                _ensure_existing_batch_manager_state(
                    manager_node,
                    timeout_sec=remote_check_timeout_sec,
                    runner=runner,
                )
                result = _manager_preflight_node(
                    manager,
                    manager_node,
                    timeout_sec=remote_check_timeout_sec,
                    runner=runner,
                )
                _raise_for_preflight_failures({manager_node.name: result})
                manager_preflight[manager_node.name] = result
            if manager_node.name not in managers:
                managers[manager_node.name] = _manager_start_node(
                    manager,
                    manager_node,
                    poll_interval=5.0,
                    slurm_group_size=slurm_group_size,
                    timeout_sec=remote_check_timeout_sec,
                    runner=runner,
                )
            (local_results / manager_node.name).mkdir(exist_ok=True)

    activate_manifest_nodes(manifest)
    submit_remote_run_manifest(manifest, runner=runner)

    last_rebalance_probe = float("-inf")

    def rebalance_poll_hook(current: RemoteRunManifest) -> RemoteRunManifest:
        nonlocal last_rebalance_probe
        now = time.monotonic()
        if now - last_rebalance_probe < 30.0:
            return current
        last_rebalance_probe = now
        for batch_node in nodes:
            admission[batch_node.manager.name] = _probe_existing_batch_admission(
                batch_node,
                timeout_sec=remote_check_timeout_sec,
                runner=runner,
            )
        updated, moves = _rebalance_unclaimed_existing_batch_manifest(
            current,
            batch_nodes=nodes,
            input_fasta=input_fasta,
            software=software,
            admission=admission,
            manifest_path=manifest_path,
            withdraw_timeout_sec=remote_check_timeout_sec,
            runner=runner,
        )
        if not moves:
            return current
        rebalances.extend(moves)
        activate_manifest_nodes(updated)
        submit_remote_run_manifest(
            updated,
            bucket_ids={move["bucket_id"] for move in moves},
            runner=runner,
        )
        return updated

    manifest = wait_remote_run_manifest(
        manifest,
        poll_interval=wait_poll_interval,
        timeout_sec=timeout_sec,
        pull_results=True,
        poll_hook=rebalance_poll_hook,
        runner=runner,
    )
    manager_nodes = list(manifest.nodes)
    local_artifacts.mkdir(parents=True, exist_ok=True)
    for manager_node in manager_nodes:
        (local_artifacts / manager_node.name).mkdir(exist_ok=True)
    for command in render_existing_artifact_pull_commands(manifest, local_artifacts):
        runner(command, check=True)
    summary = collect_existing_remote_command_manifest(manifest, local_artifacts)
    _write_json_atomic(summary_path, summary)
    assignment = {
        manager_node.name: sum(request.node_name == manager_node.name for request in manifest.requests)
        for manager_node in manager_nodes
    }
    return ExistingRemoteBatchLaunchResult(
        run_id=run_id,
        task_count=int(summary["task_count"]),
        assignment=assignment,
        manifest=str(manifest_path),
        local_results_dir=str(local_results),
        local_artifacts_dir=str(local_artifacts),
        summary=str(summary_path),
        preflight={
            name: {
                "runtime": runtime_preflight[name],
                "admission": initial_admission[name],
                "manager": manager_preflight.get(name),
            }
            for name in runtime_preflight
        },
        managers=managers,
        admission=admission,
        rebalances=tuple(rebalances),
    )


def launch_remote_run(
    tool_key: str,
    nodes: list[RemoteNode],
    inputs: BaseToolInput | dict[str, Any],
    config: BaseConfig | dict[str, Any] | None,
    *,
    bucket_size: int,
    local_results_dir: Path | str,
    manifest_path: Path | str,
    output_path: Path | str,
    run_id: str | None = None,
    overwrite: bool = False,
    manager_poll_interval: float = 5.0,
    slurm_group_size: int = 10,
    wait_poll_interval: float = 5.0,
    timeout_sec: float | None = None,
    rsync_mode: RemoteRsyncMode = "inline",
    rsync_pid_path: Path | str | None = None,
    rsync_log_path: Path | str | None = None,
    rsync_interval_sec: int = 10,
    use_diagnostics_backlog: bool = False,
    overwrite_output: bool = False,
    remote_check_timeout_sec: float | None = None,
    manager_client: "RemoteManagerClient | None" = None,
    rsync_client: "LocalRsyncPullClient | None" = None,
    runner: Any = subprocess.run,
) -> RemoteLaunchResult:
    """Preflight managers, submit a manifest run, wait, and collect output."""
    ensure_remote_run_output_path_available(output_path, overwrite=overwrite_output)
    _validate_remote_run_rsync_mode(
        rsync_mode,
        rsync_pid_path=rsync_pid_path,
        rsync_log_path=rsync_log_path,
    )

    _raise_for_remote_profile_errors(nodes)
    manager = manager_client or RemoteManagerClient()
    preflight = {
        node.name: _manager_preflight_node(
            manager,
            node,
            timeout_sec=remote_check_timeout_sec,
            runner=runner,
        )
        for node in nodes
    }
    _raise_for_preflight_failures(preflight)
    managers = {
        node.name: _manager_start_node(
            manager,
            node,
            poll_interval=manager_poll_interval,
            slurm_group_size=slurm_group_size,
            timeout_sec=remote_check_timeout_sec,
            runner=runner,
        )
        for node in nodes
    }
    diagnostics: dict[str, Any] | None = None
    initial_node_loads: dict[str, float] | None = None
    if use_diagnostics_backlog:
        diagnostics, initial_node_loads = remote_run_diagnostics_backlog_loads(
            nodes,
            manager_client=manager,
            timeout_sec=remote_check_timeout_sec,
            runner=runner,
        )

    local_results = Path(local_results_dir).resolve()
    plan = RemoteDispatchPlanner(nodes, bucket_size, initial_node_loads=initial_node_loads).build_plan(
        tool_key,
        inputs,
        config,
        run_id=run_id,
    )
    manifest = RemoteRunManifest.from_plan(
        plan,
        nodes=nodes,
        bucket_size=bucket_size,
        local_results_dir=local_results,
    )
    write_remote_run_manifest(manifest, manifest_path, overwrite=overwrite)

    return _submit_wait_collect_remote_run_manifest(
        manifest,
        manifest_path=manifest_path,
        output_path=output_path,
        preflight=preflight,
        managers=managers,
        diagnostics=diagnostics,
        initial_node_loads=initial_node_loads,
        wait_poll_interval=wait_poll_interval,
        timeout_sec=timeout_sec,
        rsync_mode=rsync_mode,
        rsync_pid_path=rsync_pid_path,
        rsync_log_path=rsync_log_path,
        rsync_interval_sec=rsync_interval_sec,
        rsync_client=rsync_client,
        overwrite_output=overwrite_output,
        runner=runner,
    )


class RemoteToolDispatcher:
    """ToolRegistry dispatch backend for explicit ``device="remote"`` calls."""

    def __init__(
        self,
        nodes: list[RemoteNode],
        *,
        bucket_size: int,
        local_results_dir: Path | str,
        submission_client: RemoteSubmissionClient | None = None,
        poll_interval: float = 5.0,
        timeout_sec: float | None = None,
        pull_results: bool = True,
        runner: Any = subprocess.run,
    ):
        """Initialize a synchronous remote dispatcher."""
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout_sec is not None and timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be > 0, got {timeout_sec}")
        self.nodes = list(nodes)
        self.planner = RemoteDispatchPlanner(self.nodes, bucket_size)
        self.local_results_dir = Path(local_results_dir)
        self.submission_client = submission_client or RemoteSubmissionClient()
        self.poll_interval = poll_interval
        self.timeout_sec = timeout_sec
        self.pull_results = pull_results
        self.runner = runner
        self.last_plan: RemoteDispatchPlan | None = None
        self._previous_backend: Any = None

    def __enter__(self) -> "RemoteToolDispatcher":
        """Install this object as the active ToolRegistry dispatch backend."""
        self._previous_backend = ToolRegistry._dispatch_backend
        ToolRegistry.configure_dispatch_backend(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Restore the previous ToolRegistry dispatch backend."""
        ToolRegistry.configure_dispatch_backend(self._previous_backend)
        self._previous_backend = None
        return False

    def __call__(self, tool_key: str, inputs: BaseToolInput, config: BaseConfig) -> BaseToolOutput | None:
        """Route only explicit ``device="remote"`` tool calls."""
        if getattr(config, "device", None) != "remote":
            return None
        plan = self.planner.build_plan(tool_key, inputs, config)
        self.last_plan = plan
        self.submission_client.submit_plan(plan, runner=self.runner)
        self.wait_for_plan(plan)
        return collect_plan_outputs(plan, self.local_results_dir)

    def wait_for_plan(self, plan: RemoteDispatchPlan) -> None:
        """Wait until every bucket result for ``plan`` exists locally."""
        deadline = None if self.timeout_sec is None else time.monotonic() + self.timeout_sec
        while True:
            self._raise_for_failed_results(plan)
            missing = [path for path in self._local_result_paths(plan) if not path.exists()]
            if not missing:
                return
            if self.pull_results:
                self.pull_once()
                self._raise_for_failed_results(plan)
                missing = [path for path in self._local_result_paths(plan) if not path.exists()]
                if not missing:
                    return
            if deadline is not None and time.monotonic() >= deadline:
                preview = ", ".join(str(path) for path in missing[:3])
                raise TimeoutError(
                    f"Remote dispatch {plan.run_id} timed out waiting for {len(missing)} result(s): {preview}"
                )
            time.sleep(self.poll_interval)

    def pull_once(self) -> None:
        """Run one rsync pull round for every configured node."""
        for cmd in render_rsync_pull_commands(self.nodes, self.local_results_dir):
            self.runner(cmd, check=True)

    def _local_result_paths(self, plan: RemoteDispatchPlan) -> list[Path]:
        """Return local result paths expected after rsync mirroring."""
        return [path for _, _, path in self._local_result_entries(plan)]

    def _local_result_entries(self, plan: RemoteDispatchPlan) -> list[tuple[RemoteNode, RemoteBucket, Path]]:
        """Return expected local result entries with node and bucket metadata."""
        return [
            (
                assignment.node,
                assignment.bucket,
                self.local_results_dir / assignment.node.name / plan.run_id / f"{assignment.bucket.bucket_id}.json",
            )
            for assignment in plan.assignments
        ]

    def _raise_for_failed_results(self, plan: RemoteDispatchPlan) -> None:
        """Raise as soon as any mirrored result envelope reports failure."""
        failures: list[str] = []
        for node, bucket, path in self._local_result_entries(plan):
            if not path.exists():
                continue
            try:
                envelope = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                failures.append(f"{node.name}/{bucket.bucket_id}: unreadable {path}: {exc}")
                continue
            if envelope.get("status") == "failed":
                failures.append(
                    f"{node.name}/{bucket.bucket_id}: {envelope.get('error', '<no error>')} (result_path={path})"
                )
        if failures:
            raise RuntimeError(f"Remote dispatch {plan.run_id} failed: {'; '.join(failures)}")


def _parse_remote_manager_json_result(
    node: RemoteNode,
    operation: str,
    result: Any,
) -> dict[str, Any]:
    """Parse a final one-line manager JSON payload and surface preceding stdout noise."""
    stdout = str(getattr(result, "stdout", "") or "{}").strip()
    stderr = str(getattr(result, "stderr", "") or "")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        lines = stdout.splitlines()
        try:
            payload = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as final_exc:
            raise ValueError(
                f"Remote manager {operation} for node {node.name!r} returned invalid JSON: "
                f"stdout={stdout!r}; stderr={stderr!r}"
            ) from final_exc
        prefix = "\n".join(lines[:-1]).strip()
        if prefix:
            logger.warning(
                "Remote manager %s for node %r wrote non-JSON stdout before its payload: %s",
                operation,
                node.name,
                prefix,
            )
    if not isinstance(payload, dict):
        raise ValueError(
            f"Remote manager {operation} for node {node.name!r} returned JSON "
            f"{type(payload).__name__}, expected object: stdout={stdout!r}; stderr={stderr!r}"
        )
    return cast(dict[str, Any], payload)


class RemoteManagerClient:
    """Start and inspect node-side managers over SSH."""

    def start_node(
        self,
        node: RemoteNode,
        *,
        poll_interval: float = 5.0,
        slurm_group_size: int = 10,
        timeout_sec: float | None = None,
        runner: Any = subprocess.run,
    ) -> str:
        """Start one node manager in the background and return remote stdout."""
        script = render_start_manager_script(
            node,
            poll_interval=poll_interval,
            slurm_group_size=slurm_group_size,
        )
        kwargs: dict[str, Any] = {"check": True, "text": True, "capture_output": True}
        if timeout_sec is not None:
            kwargs["timeout"] = timeout_sec
        result = runner(_remote_bash_command(node, script), **kwargs)
        return str(getattr(result, "stdout", "")).strip()

    def status_node(
        self,
        node: RemoteNode,
        *,
        timeout_sec: float | None = None,
        runner: Any = subprocess.run,
    ) -> str:
        """Return one node manager status string."""
        kwargs: dict[str, Any] = {"check": True, "text": True, "capture_output": True}
        if timeout_sec is not None:
            kwargs["timeout"] = timeout_sec
        result = runner(
            _remote_bash_command(node, render_manager_status_script(node)),
            **kwargs,
        )
        return str(getattr(result, "stdout", "")).strip()

    def preflight_node(
        self,
        node: RemoteNode,
        *,
        timeout_sec: float | None = None,
        runner: Any = subprocess.run,
    ) -> dict[str, Any]:
        """Run preflight checks on one node and return parsed JSON."""
        kwargs: dict[str, Any] = {"check": True, "text": True, "capture_output": True}
        if timeout_sec is not None:
            kwargs["timeout"] = timeout_sec
        result = runner(
            _remote_bash_command(node, render_preflight_script(node)),
            **kwargs,
        )
        return _parse_remote_manager_json_result(node, "preflight", result)

    def diagnostics_node(
        self,
        node: RemoteNode,
        *,
        timeout_sec: float | None = None,
        runner: Any = subprocess.run,
    ) -> dict[str, Any]:
        """Return queue/result/log diagnostics for one node."""
        kwargs: dict[str, Any] = {"check": True, "text": True, "capture_output": True}
        if timeout_sec is not None:
            kwargs["timeout"] = timeout_sec
        result = runner(
            _remote_bash_command(node, render_manager_diagnostics_script(node)),
            **kwargs,
        )
        return _parse_remote_manager_json_result(node, "diagnostics", result)


class RemoteDeploymentClient:
    """Run explicit remote deployment commands over SSH/rsync."""

    def deploy_node(
        self,
        node: RemoteNode,
        local_repo_dir: Path | str,
        *,
        runner: Any = subprocess.run,
    ) -> list[Any]:
        """Execute rendered deployment commands for one node."""
        return [
            runner(cmd, check=True, text=True, capture_output=True)
            for cmd in render_deploy_commands(node, local_repo_dir)
        ]


def render_rsync_pull_commands(nodes: list[RemoteNode], local_results_dir: Path | str) -> list[list[str]]:
    """Render one rsync pull command per node result root."""
    local_root = Path(local_results_dir)
    commands: list[list[str]] = []
    for node in nodes:
        _validate_path_segment("RemoteNode.name", node.name)
        cmd = [
            "rsync",
            "-az",
            "--partial",
        ]
        if node.sync_ssh_args:
            cmd.extend(["-e", shlex.join(["ssh", *node.sync_ssh_args])])
        cmd.extend(
            [
                f"{node.sync_host}:{node.results_root}/",
                str(local_root / node.name) + "/",
            ]
        )
        commands.append(cmd)
    return commands


def _remote_error_text(value: Any) -> str:
    """Render subprocess output values as JSON-safe text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _remote_check_error(exc: subprocess.CalledProcessError | subprocess.TimeoutExpired) -> dict[str, Any]:
    """Serialize a failed remote smoke check without hiding the command failure."""
    if isinstance(exc, subprocess.CalledProcessError):
        return {
            "type": type(exc).__name__,
            "returncode": exc.returncode,
            "cmd": exc.cmd,
            "stdout": _remote_error_text(getattr(exc, "stdout", None)),
            "stderr": _remote_error_text(exc.stderr),
        }
    return {
        "type": type(exc).__name__,
        "cmd": exc.cmd,
        "timeout": exc.timeout,
        "stdout": _remote_error_text(exc.stdout),
        "stderr": _remote_error_text(exc.stderr),
    }


def _failed_remote_check_payload(
    node: RemoteNode,
    stage: str,
    exc: subprocess.CalledProcessError | subprocess.TimeoutExpired,
) -> dict[str, Any]:
    """Build a visible failed-check payload for a node smoke stage."""
    error = _remote_check_error(exc)
    detail = error.get("stderr") or error.get("stdout") or error.get("type")
    return {
        "node": node.name,
        "ok": False,
        "checks": [{"name": stage, "ok": False, "detail": detail}],
        "error": error,
    }


def _remote_status_failure(exc: subprocess.CalledProcessError | subprocess.TimeoutExpired) -> str:
    """Build a compact visible status string for failed manager status checks."""
    error = _remote_check_error(exc)
    detail = error.get("stderr") or error.get("stdout") or error.get("type")
    return f"check-failed: {detail}"


def smoke_remote_nodes(
    nodes: list[RemoteNode],
    *,
    local_results_dir: Path | str,
    timeout_sec: float | None = None,
    manager_client: RemoteManagerClient | None = None,
    runner: Any = subprocess.run,
) -> RemoteSmokeReport:
    """Run no-submit static and remote preflight checks for remote nodes."""
    profile_issues = tuple(validate_remote_nodes(nodes))
    profile_errors = [issue for issue in profile_issues if issue.level == "error"]
    if profile_errors:
        return RemoteSmokeReport(
            ok=False,
            profile_issues=profile_issues,
            preflight={},
            manager_status={},
            diagnostics={},
            rsync_pull_commands=(),
        )

    manager = manager_client or RemoteManagerClient()

    def run_preflight(node: RemoteNode) -> tuple[dict[str, Any], bool]:
        try:
            return manager.preflight_node(node, timeout_sec=timeout_sec, runner=runner), False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return _failed_remote_check_payload(node, "preflight", exc), True

    def run_status(node: RemoteNode) -> tuple[str, bool]:
        try:
            return manager.status_node(node, timeout_sec=timeout_sec, runner=runner), False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return _remote_status_failure(exc), True

    def run_diagnostics(node: RemoteNode) -> tuple[dict[str, Any], bool]:
        try:
            return manager.diagnostics_node(node, timeout_sec=timeout_sec, runner=runner), False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return _failed_remote_check_payload(node, "diagnostics", exc), True

    preflight_results = {node.name: run_preflight(node) for node in nodes}
    status_results = {node.name: run_status(node) for node in nodes}
    diagnostics_results = {node.name: run_diagnostics(node) for node in nodes}
    preflight = {name: result for name, (result, _) in preflight_results.items()}
    manager_status = {name: result for name, (result, _) in status_results.items()}
    diagnostics = {name: result for name, (result, _) in diagnostics_results.items()}
    remote_check_failed = any(failed for _, failed in preflight_results.values())
    remote_check_failed = remote_check_failed or any(failed for _, failed in status_results.values())
    remote_check_failed = remote_check_failed or any(failed for _, failed in diagnostics_results.values())
    rsync_commands = tuple(tuple(cmd) for cmd in render_rsync_pull_commands(nodes, local_results_dir))
    ok = all(payload.get("ok") for payload in preflight.values()) and not remote_check_failed
    return RemoteSmokeReport(
        ok=ok,
        profile_issues=profile_issues,
        preflight=preflight,
        manager_status=manager_status,
        diagnostics=diagnostics,
        rsync_pull_commands=rsync_commands,
    )


def _shell_command_line(cmd: list[str]) -> str:
    """Render a shell-safe command line."""
    return " ".join(shlex.quote(part) for part in cmd)


def render_continuous_rsync_script(
    nodes: list[RemoteNode],
    local_results_dir: Path | str,
    *,
    interval_sec: int = 10,
) -> str:
    """Render a loop that continuously pulls result roots in parallel."""
    if not nodes:
        raise ValueError("render_continuous_rsync_script requires at least one node")
    if interval_sec < 1:
        raise ValueError(f"interval_sec must be >= 1, got {interval_sec}")
    lines = ["while true; do", "  pids=()"]
    for cmd in render_rsync_pull_commands(nodes, local_results_dir):
        lines.append(f"  {_shell_command_line(cmd)} &")
        lines.append('  pids+=("$!")')
    lines.extend(
        [
            "  status=0",
            '  for pid in "${pids[@]}"; do',
            '    wait "$pid" || status=$?',
            "  done",
            '  if [ "$status" -ne 0 ]; then',
            '    exit "$status"',
            "  fi",
            f"  sleep {interval_sec}",
            "done",
        ]
    )
    return "\n".join(lines)


def render_start_rsync_pull_script(
    nodes: list[RemoteNode],
    local_results_dir: Path | str,
    *,
    pid_path: Path | str,
    log_path: Path | str,
    interval_sec: int = 10,
) -> str:
    """Render a local shell script that starts the background rsync pull loop."""
    loop_script = render_continuous_rsync_script(nodes, local_results_dir, interval_sec=interval_sec)
    pid = shlex.quote(str(pid_path))
    log = shlex.quote(str(log_path))
    pid_parent = shlex.quote(str(Path(pid_path).parent))
    log_parent = shlex.quote(str(Path(log_path).parent))
    local_root = shlex.quote(str(local_results_dir))
    return (
        "set -euo pipefail; "
        f"mkdir -p {pid_parent} {log_parent} {local_root}; "
        f'if [ -s {pid} ] && kill -0 "$(cat {pid})" 2>/dev/null; then '
        f'echo running "$(cat {pid})"; '
        "else "
        f"nohup bash -lc {shlex.quote(loop_script)} >> {log} 2>&1 < /dev/null & "
        f"echo $! > {pid}; "
        "sleep 0.2; "
        f'if kill -0 "$(cat {pid})" 2>/dev/null; then '
        f'echo started "$(cat {pid})"; '
        "else "
        "echo failed; "
        f"if [ -f {log} ]; then tail -n 40 {log}; fi; "
        "exit 1; "
        "fi; "
        "fi"
    )


def render_rsync_pull_status_script(pid_path: Path | str) -> str:
    """Render a local shell script that reports the rsync pull loop state."""
    pid = shlex.quote(str(pid_path))
    return (
        "set -euo pipefail; "
        f'if [ -s {pid} ] && kill -0 "$(cat {pid})" 2>/dev/null; then '
        f'echo running "$(cat {pid})"; '
        f"elif [ -s {pid} ]; then "
        f'echo stale "$(cat {pid})"; '
        "else "
        "echo stopped; "
        "fi"
    )


def render_stop_rsync_pull_script(pid_path: Path | str) -> str:
    """Render a local shell script that stops the rsync pull loop."""
    pid = shlex.quote(str(pid_path))
    return (
        "set -euo pipefail; "
        f'if [ -s {pid} ] && kill -0 "$(cat {pid})" 2>/dev/null; then '
        f'kill "$(cat {pid})"; '
        f"rm -f {pid}; "
        "echo stopped; "
        f"elif [ -s {pid} ]; then "
        f"rm -f {pid}; "
        "echo stale-removed; "
        "else "
        "echo stopped; "
        "fi"
    )


class LocalRsyncPullClient:
    """Manage the local background rsync pull loop."""

    def start(
        self,
        nodes: list[RemoteNode],
        local_results_dir: Path | str,
        *,
        pid_path: Path | str,
        log_path: Path | str,
        interval_sec: int = 10,
        runner: Any = subprocess.run,
    ) -> str:
        """Start the pull loop locally and return its stdout."""
        result = runner(
            [
                "bash",
                "-lc",
                render_start_rsync_pull_script(
                    nodes,
                    local_results_dir,
                    pid_path=pid_path,
                    log_path=log_path,
                    interval_sec=interval_sec,
                ),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        return str(getattr(result, "stdout", "")).strip()

    def status(
        self,
        pid_path: Path | str,
        *,
        runner: Any = subprocess.run,
    ) -> str:
        """Return local pull loop status."""
        result = runner(
            ["bash", "-lc", render_rsync_pull_status_script(pid_path)],
            check=True,
            text=True,
            capture_output=True,
        )
        return str(getattr(result, "stdout", "")).strip()

    def stop(
        self,
        pid_path: Path | str,
        *,
        runner: Any = subprocess.run,
    ) -> str:
        """Stop the local pull loop and return its stdout."""
        result = runner(
            ["bash", "-lc", render_stop_rsync_pull_script(pid_path)],
            check=True,
            text=True,
            capture_output=True,
        )
        return str(getattr(result, "stdout", "")).strip()


def render_manager_loop_command(
    node: RemoteNode,
    *,
    poll_interval: float = 5.0,
    once: bool = False,
    slurm_group_size: int = 10,
) -> list[str]:
    """Render the manager-loop command for a remote node."""
    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
    if slurm_group_size < 1:
        raise ValueError(f"slurm_group_size must be >= 1, got {slurm_group_size}")
    if node.scheduler == "slurm" and node.max_active_slurm_jobs is None:
        raise ValueError(f"RemoteNode {node.name!r}: Slurm manager requires max_active_slurm_jobs")
    cmd = [
        node.python,
        "-m",
        "proto_tools.utils.remote_execution",
        "manager-loop",
        "--queue-dir",
        str(node.queue_root),
        "--results-dir",
        str(node.results_root),
        "--scheduler",
        node.scheduler,
        "--poll-interval",
        str(poll_interval),
        "--python",
        node.python,
        "--slurm-group-size",
        str(slurm_group_size),
    ]
    if node.max_active_slurm_jobs is not None:
        cmd.extend(["--max-active-slurm-jobs", str(node.max_active_slurm_jobs)])
    cmd.extend(f"--sbatch-arg={arg}" for arg in node.sbatch_args)
    if once:
        cmd.append("--once")
    return cmd


def render_start_manager_script(
    node: RemoteNode,
    *,
    poll_interval: float = 5.0,
    slurm_group_size: int = 10,
) -> str:
    """Render a remote shell script that starts a manager if it is not already running."""
    manager_cmd = _remote_env_prefix(node) + shlex.join(
        render_manager_loop_command(
            node,
            poll_interval=poll_interval,
            slurm_group_size=slurm_group_size,
        )
    )
    pid_path = shlex.quote(str(node.manager_pid_path))
    log_path = shlex.quote(str(node.manager_log_path))
    return (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(str(node.queue_root))} "
        f"{shlex.quote(str(node.results_root))} "
        f"{shlex.quote(str(node.logs_root))}; "
        f'if [ -s {pid_path} ] && kill -0 "$(cat {pid_path})" 2>/dev/null; then '
        f'echo running "$(cat {pid_path})"; '
        "else "
        f"nohup bash -lc {shlex.quote(manager_cmd)} >> {log_path} 2>&1 < /dev/null & "
        f"echo $! > {pid_path}; "
        "sleep 0.2; "
        f'if kill -0 "$(cat {pid_path})" 2>/dev/null; then '
        f'echo started "$(cat {pid_path})"; '
        "else "
        "echo failed; "
        f"tail -n 40 {log_path} 2>/dev/null || true; "
        "exit 1; "
        "fi; "
        "fi"
    )


def render_manager_status_script(node: RemoteNode) -> str:
    """Render a remote shell script that reports manager pid state."""
    pid_path = shlex.quote(str(node.manager_pid_path))
    return (
        "set -euo pipefail; "
        f'if [ -s {pid_path} ] && kill -0 "$(cat {pid_path})" 2>/dev/null; then '
        f'echo running "$(cat {pid_path})"; '
        f"elif [ -s {pid_path} ]; then "
        f'echo stale "$(cat {pid_path})"; '
        "else "
        "echo stopped; "
        "fi"
    )


def render_preflight_script(node: RemoteNode) -> str:
    """Render a JSON-emitting remote preflight script."""
    repo_dir = json.dumps(node.repo_dir)
    work_dir = json.dumps(node.work_dir)
    queue_dir = json.dumps(str(node.queue_root))
    results_dir = json.dumps(str(node.results_root))
    logs_dir = json.dumps(str(node.logs_root))
    scheduler = json.dumps(node.scheduler)
    node_name = json.dumps(node.name)
    script = f"""import importlib.util
import json
import os
import pathlib
import shutil
import sys

checks = []

def add(name, ok, detail):
    checks.append({{"name": name, "ok": bool(ok), "detail": str(detail)}})

add("python>=3.10", sys.version_info >= (3, 10), sys.version.split()[0])
add("import proto_tools", importlib.util.find_spec("proto_tools") is not None, "proto_tools")

repo_dir = {repo_dir}
if repo_dir is not None:
    repo_path = pathlib.Path(repo_dir)
    add("repo_dir exists", repo_path.is_dir(), repo_dir)

for label, raw_path in [
    ("work_dir", {work_dir}),
    ("queue_dir", {queue_dir}),
    ("results_dir", {results_dir}),
    ("logs_dir", {logs_dir}),
]:
    path = pathlib.Path(raw_path)
    add(label + " exists", path.is_dir(), raw_path)
    add(label + " writable", path.is_dir() and os.access(path, os.W_OK), raw_path)

add("rsync available", shutil.which("rsync") is not None, shutil.which("rsync"))
if {scheduler} == "slurm":
    add("sbatch available", shutil.which("sbatch") is not None, shutil.which("sbatch"))
    add("sacct available", shutil.which("sacct") is not None, shutil.which("sacct"))

print(json.dumps({{"node": {node_name}, "ok": all(item["ok"] for item in checks), "checks": checks}}, sort_keys=True))
"""
    return "set -euo pipefail; " + _remote_env_prefix(node) + f"{shlex.quote(node.python)} - <<'PY'\n{script}PY"


def render_manager_diagnostics_script(node: RemoteNode) -> str:
    """Render a JSON-emitting remote manager diagnostics script."""
    paths_payload = json.dumps(
        {
            "queue": str(node.queue_root),
            "running": str(PurePosixPath(node.work_dir) / "running"),
            "submitted": str(PurePosixPath(node.work_dir) / "submitted"),
            "done": str(PurePosixPath(node.work_dir) / "done"),
            "failed": str(PurePosixPath(node.work_dir) / "failed"),
            "results": str(node.results_root),
        }
    )
    pid_path = json.dumps(str(node.manager_pid_path))
    log_path = json.dumps(str(node.manager_log_path))
    node_name = json.dumps(node.name)
    script = f"""import json
import os
import pathlib
import time

paths = {paths_payload}
pid_path = pathlib.Path({pid_path})
log_path = pathlib.Path({log_path})
now = time.time()

def is_sidecar(path):
    return str(path).endswith(".json.slurm.json")

def is_tmp(path):
    return str(path).endswith(".tmp")

def logical_json_files(root):
    if not root.exists():
        return []
    return [path for path in root.rglob("*.json") if not is_sidecar(path) and not is_tmp(path)]

def age_sec(path):
    try:
        return max(0.0, round(now - path.stat().st_mtime, 3))
    except OSError:
        return None

def request_summary(path):
    item = {{"path": str(path), "age_sec": age_sec(path)}}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        item["error"] = f"unreadable: {{type(exc).__name__}}: {{exc}}"
    else:
        item["run_id"] = payload.get("run_id")
        item["bucket_id"] = payload.get("bucket_id")
        item["tool_key"] = payload.get("tool_key")
    return item

counts = {{}}
for key, raw_path in paths.items():
    counts[key] = len(logical_json_files(pathlib.Path(raw_path)))

sidecar_counts = {{}}
submitted_root = pathlib.Path(paths["submitted"])
sidecar_counts["submitted"] = (
    len(list(submitted_root.rglob("*.json.slurm.json"))) if submitted_root.exists() else 0
)

if pid_path.exists():
    pid = pid_path.read_text().strip()
    running = pid.isdigit() and pathlib.Path("/proc", pid).exists()
    manager = {{"status": "running" if running else "stale", "pid": pid}}
else:
    manager = {{"status": "stopped", "pid": None}}

log_tail = ""
if log_path.exists():
    log_tail = "\\n".join(log_path.read_text(errors="replace").splitlines()[-20:])

recent_failed = []
failed_root = pathlib.Path(paths["failed"])
if failed_root.exists():
    for path in sorted(logical_json_files(failed_root))[:5]:
        item = {{"path": str(path), "age_sec": age_sec(path)}}
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            item["error"] = f"unreadable: {{type(exc).__name__}}: {{exc}}"
        else:
            item["error"] = payload.get("error")
            item["run_id"] = payload.get("run_id")
            item["bucket_id"] = payload.get("bucket_id")
            item["tool_key"] = payload.get("tool_key")
        recent_failed.append(item)

recent_running = []
running_root = pathlib.Path(paths["running"])
if running_root.exists():
    for path in sorted(logical_json_files(running_root))[:5]:
        recent_running.append(request_summary(path))

recent_submitted = []
if submitted_root.exists():
    for path in sorted(submitted_root.rglob("*.json.slurm.json"))[:5]:
        request_path = pathlib.Path(str(path)[:-len(".slurm.json")])
        item = {{
            "path": str(path),
            "sidecar_path": str(path),
            "request_path": str(request_path),
            "request_exists": request_path.exists(),
            "age_sec": age_sec(path),
        }}
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            item["error"] = f"unreadable: {{type(exc).__name__}}: {{exc}}"
        else:
            item["slurm_job_id"] = payload.get("slurm_job_id")
            item["bucket_id"] = payload.get("bucket_id")
            item["result_path"] = payload.get("result_path")
        if request_path.exists():
            try:
                request_payload = json.loads(request_path.read_text())
            except Exception as exc:
                item["request_error"] = f"unreadable: {{type(exc).__name__}}: {{exc}}"
            else:
                item["run_id"] = request_payload.get("run_id")
                item["bucket_id"] = item.get("bucket_id") or request_payload.get("bucket_id")
                item["tool_key"] = request_payload.get("tool_key")
        recent_submitted.append(item)

recent_tmp = []
for key, raw_path in paths.items():
    root = pathlib.Path(raw_path)
    if root.exists():
        for path in sorted(root.rglob("*.tmp"))[:5]:
            recent_tmp.append({{"root": key, "path": str(path), "age_sec": age_sec(path)}})

recent_failed_results = []
failed_results_root = pathlib.Path(paths["results"]) / "failed"
if failed_results_root.exists():
    for path in sorted(logical_json_files(failed_results_root))[:5]:
        item = request_summary(path)
        try:
            payload = json.loads(path.read_text())
        except Exception:
            pass
        else:
            item["status"] = payload.get("status")
            item["error"] = payload.get("error")
        recent_failed_results.append(item)

print(json.dumps({{"node": {node_name}, "manager": manager, "counts": counts, "sidecar_counts": sidecar_counts, "log_tail": log_tail, "recent_failed": recent_failed, "recent_running": recent_running, "recent_submitted": recent_submitted, "recent_tmp": recent_tmp, "recent_failed_results": recent_failed_results}}, sort_keys=True))
"""
    return "set -euo pipefail; " + f"{shlex.quote(node.python)} - <<'PY'\n{script}PY"


def _existing_command_argv(template: list[Any], *, input_path: Path, output_dir: Path) -> list[str]:
    """Render one existing-command argv template without invoking a shell."""
    if not template or not all(isinstance(part, str) and part for part in template):
        raise ValueError("existing_command.command_argv must be a non-empty list of non-empty strings")
    return [
        part.replace("{input}", str(input_path)).replace("{output}", str(output_dir))
        for part in template
    ]


def _required_existing_artifacts(output_dir: Path, patterns: list[Any]) -> list[str]:
    """Validate required artifact globs and return the complete output inventory."""
    if not patterns or not all(isinstance(pattern, str) and pattern for pattern in patterns):
        raise ValueError("existing command task required_artifacts must be a non-empty list")
    for pattern in patterns:
        pattern_path = PurePosixPath(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise ValueError(f"existing command required artifact pattern must stay relative: {pattern!r}")
        matches = sorted(path for path in output_dir.glob(pattern) if path.is_file() and path.stat().st_size > 0)
        if not matches:
            raise FileNotFoundError(f"required artifact missing or empty: {pattern} under {output_dir}")
    return sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.stat().st_size > 0
    )


def _execute_existing_command_request(request: dict[str, Any]) -> dict[str, Any]:
    """Execute node-resolved existing commands and preserve their native artifacts."""
    payload = request.get("existing_command")
    if not isinstance(payload, dict):
        raise ValueError("existing_command request must contain an existing_command object")
    command_argv = payload.get("command_argv")
    if not isinstance(command_argv, list):
        raise ValueError("existing_command.command_argv must be a list")
    raw_env = payload.get("env") or {}
    if not isinstance(raw_env, dict):
        raise ValueError("existing_command.env must be an object")
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid existing command environment variable name: {key!r}")
        if not isinstance(value, str):
            raise ValueError(f"existing command environment value for {key!r} must be a string")
        env[key] = value
    required_stdout_patterns = payload.get("required_stdout_patterns") or []
    if not isinstance(required_stdout_patterns, list) or not all(
        isinstance(pattern, str) and pattern for pattern in required_stdout_patterns
    ):
        raise ValueError("existing_command.required_stdout_patterns must be a list of non-empty strings")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("existing_command.tasks must be a non-empty list")
    if len(tasks) != len(request.get("item_indices", [])):
        raise ValueError("existing_command.tasks must align one-to-one with item_indices")

    command_results: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each existing command task must be an object")
        task_id = _validate_path_segment("existing command task_id", str(task.get("task_id", "")))
        input_filename = _validate_path_segment(
            "existing command input_filename", str(task.get("input_filename", ""))
        )
        input_text = task.get("input_text")
        if not isinstance(input_text, str) or not input_text:
            raise ValueError(f"existing command task {task_id!r} input_text must be non-empty")
        artifact_dir = Path(str(task.get("artifact_dir", "")))
        if not artifact_dir.is_absolute():
            raise ValueError(f"existing command task {task_id!r} artifact_dir must be absolute")
        artifact_dir.mkdir(parents=True, exist_ok=False)
        input_path = artifact_dir / "input" / input_filename
        output_dir = artifact_dir / "output"
        log_dir = artifact_dir / "logs"
        output_dir.mkdir(parents=True)
        log_dir.mkdir(parents=True)
        _write_text_atomic(input_path, input_text, overwrite=False)

        argv = _existing_command_argv(command_argv, input_path=input_path, output_dir=output_dir)
        completed = subprocess.run(  # noqa: S603 -- operator-owned profile freezes argv in the manifest
            argv,
            check=False,
            text=True,
            capture_output=True,
            env={**os.environ, **env},
        )
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        _write_text_atomic(stdout_path, completed.stdout or "", overwrite=False)
        _write_text_atomic(stderr_path, completed.stderr or "", overwrite=False)
        if completed.returncode != 0:
            stderr_tail = "\n".join((completed.stderr or "").splitlines()[-20:])
            raise RuntimeError(
                f"existing command task {task_id!r} exited {completed.returncode}; stderr_tail={stderr_tail!r}"
            )
        missing_stdout_patterns = [
            pattern for pattern in required_stdout_patterns if pattern not in (completed.stdout or "")
        ]
        if missing_stdout_patterns:
            raise RuntimeError(
                f"existing command task {task_id!r} missing required stdout pattern(s): "
                f"{missing_stdout_patterns!r}; stdout_path={stdout_path}"
            )
        required = task.get("required_artifacts")
        if not isinstance(required, list):
            raise ValueError(f"existing command task {task_id!r} required_artifacts must be a list")
        artifacts = _required_existing_artifacts(output_dir, required)
        result_row = {
            "task_id": task_id,
            "artifact_dir": str(artifact_dir),
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "artifacts": artifacts,
        }
        if required_stdout_patterns:
            result_row["validated_stdout_patterns"] = list(required_stdout_patterns)
        command_results.append(result_row)

    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "completed",
        "request_kind": "existing_command",
        "tool_key": request["tool_key"],
        "run_id": request["run_id"],
        "bucket_id": request["bucket_id"],
        "item_indices": request["item_indices"],
        "existing_command_results": command_results,
    }


def execute_request(request: dict[str, Any]) -> dict[str, Any]:
    """Execute one remote request in the current process and return a result envelope."""
    if request.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"Unsupported remote request schema_version: {request.get('schema_version')!r}")
    _request_path_segment(request, "run_id")
    _request_path_segment(request, "bucket_id")
    request_kind = request.get("request_kind", "tool")
    if request_kind == "existing_command":
        return _execute_existing_command_request(request)
    if request_kind != "tool":
        raise ValueError(f"Unsupported remote request kind: {request_kind!r}")
    tool_key = str(request["tool_key"])
    spec = ToolRegistry.get(tool_key)
    inputs = spec.input_model.model_validate(request["inputs"])
    config = spec.config_model.model_validate(request.get("config") or {})
    output = spec.function(inputs, config)
    if not isinstance(output, BaseToolOutput):
        raise TypeError(f"Tool {tool_key!r} returned {type(output).__name__}, expected BaseToolOutput")
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "completed",
        "tool_key": tool_key,
        "run_id": request["run_id"],
        "bucket_id": request["bucket_id"],
        "item_indices": request["item_indices"],
        "output": output.model_dump(mode="json", exclude_none=True),
    }


def _write_text_atomic(
    path: Path,
    text: str,
    *,
    overwrite: bool = True,
    exists_message: str | None = None,
) -> None:
    """Write text atomically through a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(exists_message or f"File already exists: {path}")
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        if overwrite:
            tmp_path.replace(path)
        else:
            try:
                os.link(tmp_path, path)
            except FileExistsError as exc:
                raise FileExistsError(exists_message or f"File already exists: {path}") from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_json_atomic(path: Path, payload: dict[str, Any], *, overwrite: bool = True) -> None:
    """Write JSON atomically by replacing a sibling temporary file."""
    _write_text_atomic(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )


def _failure_envelope(request: dict[str, Any] | None, exc: BaseException) -> dict[str, Any]:
    """Build a failed result envelope from a request and exception."""
    request = request or {}
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "failed",
        "request_kind": request.get("request_kind", "tool"),
        "tool_key": request.get("tool_key"),
        "run_id": request.get("run_id"),
        "bucket_id": request.get("bucket_id"),
        "item_indices": request.get("item_indices", []),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def execute_request_file(request_path: Path | str, output_path: Path | str) -> dict[str, Any]:
    """Execute one request file and write the completed result envelope."""
    request: dict[str, Any] | None = None
    try:
        request = json.loads(Path(request_path).read_text())
        result = execute_request(request)
        _write_json_atomic(Path(output_path), result)
        return result
    except Exception as exc:
        _write_json_atomic(Path(output_path), _failure_envelope(request, exc))
        raise


def _request_result_path(results_root: Path, request: dict[str, Any]) -> Path:
    """Return the safe result path for one request payload."""
    run_id = _request_path_segment(request, "run_id")
    bucket_id = _request_path_segment(request, "bucket_id")
    return results_root / run_id / f"{bucket_id}.json"


def _invalid_request_result_path(results_root: Path, request_path: Path) -> Path:
    """Return an in-root failure path for a malformed request."""
    return results_root / "failed" / f"{request_path.name}.json"


def execute_request_batch(request_paths: list[Path | str], results_dir: Path | str) -> list[dict[str, Any]]:
    """Execute many request files and write each result envelope."""
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    results_root = Path(results_dir)
    for request_path in request_paths:
        request: dict[str, Any] | None = None
        result_path: Path | None = None
        try:
            path = Path(request_path)
            request = json.loads(path.read_text())
            result_path = _request_result_path(results_root, request)
            results.append(execute_request_file(request_path, result_path))
        except Exception as exc:
            failures.append(f"{request_path}: {type(exc).__name__}: {exc}")
            fallback_path = result_path or _invalid_request_result_path(results_root, Path(request_path))
            _write_json_atomic(fallback_path, _failure_envelope(request, exc))
    if failures:
        raise RuntimeError(f"{len(failures)} remote request(s) failed: {'; '.join(failures[:3])}")
    return results


@dataclass(frozen=True)
class ManagerEvent:
    """A manager action taken for one request file."""

    status: Literal["completed", "submitted", "failed"]
    request_path: Path
    result_path: Path
    slurm_job_id: str | None = None


def _move_request(src: Path, root: Path, relative: Path) -> Path:
    """Move a request file under a state root while preserving its relative path."""
    dst = root / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dst)
    return dst


def _terminal_result_status_for_request(
    envelope: dict[str, Any],
    request: dict[str, Any],
    result_path: Path,
) -> Literal["completed", "failed"]:
    """Return terminal result status after checking it belongs to a request."""
    identity_checks = (
        ("run_id", request.get("run_id"), envelope.get("run_id")),
        ("bucket_id", request.get("bucket_id"), envelope.get("bucket_id")),
        ("tool_key", request.get("tool_key"), envelope.get("tool_key")),
        ("item_indices", list(request.get("item_indices", [])), envelope.get("item_indices")),
    )
    mismatches = [
        f"{field} expected {expected!r} got {actual!r}"
        for field, expected, actual in identity_checks
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError(f"Result envelope identity mismatch at {result_path}: {'; '.join(mismatches)}")
    status = envelope.get("status")
    if status not in ("completed", "failed"):
        raise RuntimeError(f"Result envelope at {result_path} is not terminal: status={status!r}")
    return cast(Literal["completed", "failed"], status)


def reconcile_running_requests(
    running_dir: Path | str,
    results_dir: Path | str,
    *,
    done_dir: Path | str | None = None,
    failed_dir: Path | str | None = None,
) -> list[ManagerEvent]:
    """Move running requests with visible terminal result envelopes to final state roots."""
    running_root = Path(running_dir)
    results_root = Path(results_dir)
    state_root = running_root.parent
    done_root = Path(done_dir) if done_dir is not None else state_root / "done"
    failed_root = Path(failed_dir) if failed_dir is not None else state_root / "failed"
    events: list[ManagerEvent] = []
    for request_path in sorted(running_root.rglob("*.json")):
        if str(request_path).endswith(".json.slurm.json"):
            continue
        relative = request_path.relative_to(running_root)
        request = cast(dict[str, Any], json.loads(request_path.read_text()))
        result_path = _request_result_path(results_root, request)
        if not result_path.exists():
            continue
        envelope = cast(dict[str, Any], json.loads(result_path.read_text()))
        status = _terminal_result_status_for_request(envelope, request, result_path)
        if status == "completed":
            final_request = _move_request(request_path, done_root, relative)
            events.append(ManagerEvent(status="completed", request_path=final_request, result_path=result_path))
        else:
            failed_request = _move_request(request_path, failed_root, relative)
            events.append(ManagerEvent(status="failed", request_path=failed_request, result_path=result_path))
    return events


def _slurm_submit_command(
    python: str,
    request_paths: list[Path],
    results_dir: Path,
    sbatch_args: tuple[str, ...],
) -> list[str]:
    """Build an sbatch command that executes one group of request files."""
    if not request_paths:
        raise ValueError("request_paths must be non-empty")
    request_args = " ".join(f"--request {shlex.quote(str(path))}" for path in request_paths)
    worker_cmd = (
        f"{shlex.quote(python)} -m proto_tools.utils.remote_execution run-request-batch "
        f"--results-dir {shlex.quote(str(results_dir))} "
        f"{request_args}"
    )
    cmd = ["sbatch"]
    if "--parsable" not in sbatch_args:
        cmd.append("--parsable")
    cmd.extend(sbatch_args)
    cmd.extend(["--wrap", worker_cmd])
    return cmd


def _parse_slurm_job_id(result: Any) -> str | None:
    """Parse the first line of sbatch stdout as the job id."""
    stdout = str(getattr(result, "stdout", "") or "").strip()
    return stdout.splitlines()[0] if stdout else None


def _submitted_at_epoch(sidecar_path: Path, sidecar_payload: dict[str, Any]) -> float:
    """Return sidecar submission time, using mtime for pre-timestamp sidecars."""
    raw_value = sidecar_payload.get("submitted_at_epoch")
    if raw_value is None:
        return sidecar_path.stat().st_mtime
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid submitted_at_epoch in {sidecar_path}: {raw_value!r}") from exc


def _slurm_state_base(state: str | None) -> str:
    """Return the normalized top-level Slurm state token."""
    return str(state or "UNKNOWN").split()[0].split("+")[0].upper()


def _query_sacct_states(job_ids: set[str], *, runner: Any = subprocess.run) -> dict[str, str]:
    """Query Slurm accounting states for submitted job ids."""
    if not job_ids:
        return {}
    cmd = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State",
        "--jobs",
        ",".join(sorted(job_ids)),
    ]
    result = runner(cmd, check=True, text=True, capture_output=True)
    states_by_job: dict[str, list[str]] = {job_id: [] for job_id in job_ids}
    for raw_line in str(getattr(result, "stdout", "") or "").splitlines():
        if "|" not in raw_line:
            continue
        raw_job_id, raw_state = raw_line.split("|", 1)
        state = _slurm_state_base(raw_state)
        for job_id in job_ids:
            if raw_job_id == job_id or raw_job_id.startswith(f"{job_id}."):
                states_by_job[job_id].append(state)

    collapsed: dict[str, str] = {}
    for job_id, states in states_by_job.items():
        if any(state in _SLURM_FAILED_STATES for state in states):
            collapsed[job_id] = next(state for state in states if state in _SLURM_FAILED_STATES)
        elif states and all(state in _SLURM_COMPLETED_STATES for state in states):
            collapsed[job_id] = "COMPLETED"
        elif any(state in _SLURM_ACTIVE_STATES for state in states):
            collapsed[job_id] = next(state for state in states if state in _SLURM_ACTIVE_STATES)
        elif states:
            collapsed[job_id] = states[0]
    return collapsed


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    """Split items into non-empty chunks."""
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _queued_request_resource_ready(
    request: dict[str, Any],
    *,
    runner: Any = subprocess.run,
) -> bool:
    """Return whether a queued direct request passes its explicit resource admission gate."""
    if request.get("request_kind", "tool") != "existing_command":
        return True
    existing_command = request.get("existing_command")
    if not isinstance(existing_command, dict):
        return True
    admission = existing_command.get("resource_admission")
    if admission is None:
        return True
    if not isinstance(admission, dict):
        raise ValueError("existing_command.resource_admission must be an object")
    if admission.get("kind") != "cuda_idle":
        raise ValueError(
            f"unsupported existing command resource admission kind: {admission.get('kind')!r}"
        )
    gpu_id = str(admission.get("gpu_id", "")).strip()
    if not gpu_id:
        raise ValueError("cuda_idle resource admission requires gpu_id")
    max_utilization = int(admission["max_utilization_percent"])
    max_memory = int(admission["max_memory_used_mib"])
    if not 0 <= max_utilization <= 100:
        raise ValueError("cuda_idle max_utilization_percent must be in [0, 100]")
    if max_memory < 0:
        raise ValueError("cuda_idle max_memory_used_mib must be >= 0")
    command = [
        "nvidia-smi",
        "-i",
        gpu_id,
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    completed = runner(command, check=True, text=True, capture_output=True)
    output = str(getattr(completed, "stdout", "") or "").strip()
    fields = [field.strip() for field in output.splitlines()[0].split(",")] if output else []
    if len(fields) != 2:
        raise ValueError(f"unexpected nvidia-smi admission output: {output!r}")
    utilization, memory = (int(field) for field in fields)
    return utilization <= max_utilization and memory <= max_memory


def _submit_slurm_group(
    group: list[tuple[Path, Path, dict[str, Any], Path]],
    *,
    results_root: Path,
    submitted_root: Path,
    failed_root: Path,
    python: str,
    sbatch_args: tuple[str, ...],
    runner: Any,
) -> list[ManagerEvent]:
    """Submit one Slurm group and return per-request manager events."""
    events: list[ManagerEvent] = []
    submitted_by_relative: dict[Path, Path] = {}
    try:
        for claimed, relative, _, _ in group:
            submitted_by_relative[relative] = _move_request(claimed, submitted_root, relative)
        cmd = _slurm_submit_command(
            python,
            [submitted_by_relative[relative] for _, relative, _, _ in group],
            results_root,
            sbatch_args,
        )
        result = runner(cmd, check=True, text=True, capture_output=True)
        slurm_job_id = _parse_slurm_job_id(result)
        if not slurm_job_id:
            raise RuntimeError("sbatch did not return a Slurm job id")
        submitted_at_epoch = time.time()
        for _, relative, request, result_path in group:
            final_request = submitted_by_relative[relative]
            _write_json_atomic(
                final_request.with_name(f"{final_request.name}.slurm.json"),
                {
                    "schema_version": _SCHEMA_VERSION,
                    "status": "submitted",
                    "tool_key": request.get("tool_key"),
                    "run_id": request.get("run_id"),
                    "bucket_id": request.get("bucket_id"),
                    "item_indices": request.get("item_indices", []),
                    "slurm_job_id": slurm_job_id,
                    "submitted_at_epoch": submitted_at_epoch,
                    "result_path": str(result_path),
                },
            )
            events.append(
                ManagerEvent(
                    status="submitted",
                    request_path=final_request,
                    result_path=result_path,
                    slurm_job_id=slurm_job_id,
                )
            )
    except Exception as exc:
        for claimed, relative, request, result_path in group:
            _write_json_atomic(result_path, _failure_envelope(request, exc))
            current_request = submitted_by_relative.get(relative, claimed)
            failed_request = _move_request(current_request, failed_root, relative)
            sidecar = current_request.with_name(f"{current_request.name}.slurm.json")
            if sidecar.exists():
                _move_slurm_sidecar(sidecar, failed_root, relative)
            events.append(ManagerEvent(status="failed", request_path=failed_request, result_path=result_path))
    return events


def _request_path_for_slurm_sidecar(sidecar_path: Path) -> Path:
    """Return the request path paired with a ``.slurm.json`` sidecar."""
    if not sidecar_path.name.endswith(".slurm.json"):
        raise ValueError(f"Not a Slurm sidecar path: {sidecar_path}")
    return sidecar_path.with_name(sidecar_path.name[: -len(".slurm.json")])


def _move_slurm_sidecar(sidecar_path: Path, state_root: Path, relative_request: Path) -> Path:
    """Move a Slurm sidecar next to a request in a state directory."""
    dst = state_root / relative_request.parent / sidecar_path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.replace(dst)
    return dst


def _slurm_failed_envelope(request: dict[str, Any], job_id: str | None, state: str, result_path: Path) -> dict[str, Any]:
    """Build a failed envelope for a terminal Slurm job that wrote no result."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "failed",
        "tool_key": request.get("tool_key"),
        "run_id": request.get("run_id"),
        "bucket_id": request.get("bucket_id"),
        "item_indices": request.get("item_indices", []),
        "error": f"Slurm job {job_id or '<unknown>'} ended with state {state} before writing {result_path}",
        "traceback": "",
    }


def reconcile_submitted_slurm_jobs(
    submitted_dir: Path | str,
    results_dir: Path | str,
    *,
    done_dir: Path | str | None = None,
    failed_dir: Path | str | None = None,
    slurm_accounting_grace_sec: float = _SLURM_ACCOUNTING_GRACE_SEC,
    runner: Any = subprocess.run,
) -> list[ManagerEvent]:
    """Reconcile submitted Slurm requests against result files and accounting state."""
    if slurm_accounting_grace_sec < 0:
        raise ValueError(f"slurm_accounting_grace_sec must be >= 0, got {slurm_accounting_grace_sec}")
    submitted_root = Path(submitted_dir)
    results_root = Path(results_dir)
    state_root = submitted_root.parent
    done_root = Path(done_dir) if done_dir is not None else state_root / "done"
    failed_root = Path(failed_dir) if failed_dir is not None else state_root / "failed"

    events: list[ManagerEvent] = []
    missing_result_entries: list[tuple[Path, Path, Path, dict[str, Any], str | None, float]] = []
    sidecars = sorted(submitted_root.rglob("*.json.slurm.json"))
    for sidecar in sidecars:
        request_path = _request_path_for_slurm_sidecar(sidecar)
        relative = request_path.relative_to(submitted_root)
        request: dict[str, Any] | None = None
        result_path: Path | None = None
        try:
            sidecar_payload = json.loads(sidecar.read_text())
            request = json.loads(request_path.read_text())
            result_path = _request_result_path(results_root, request)
            recorded_result_path = sidecar_payload.get("result_path")
            if recorded_result_path is not None and Path(str(recorded_result_path)) != result_path:
                raise ValueError(
                    f"Slurm sidecar result_path mismatch for {request_path}: "
                    f"expected {result_path}, got {recorded_result_path}"
                )
            job_id = None if sidecar_payload.get("slurm_job_id") is None else str(sidecar_payload["slurm_job_id"])
            submitted_at_epoch = _submitted_at_epoch(sidecar, sidecar_payload)
        except Exception as exc:
            result_path = result_path or _invalid_request_result_path(results_root, request_path)
            _write_json_atomic(result_path, _failure_envelope(request, exc))
            failed_request = _move_request(request_path, failed_root, relative)
            _move_slurm_sidecar(sidecar, failed_root, relative)
            events.append(ManagerEvent(status="failed", request_path=failed_request, result_path=result_path))
            continue

        if result_path.exists():
            envelope = json.loads(result_path.read_text())
            if envelope.get("status") == "completed":
                final_request = _move_request(request_path, done_root, relative)
                _move_slurm_sidecar(sidecar, done_root, relative)
                events.append(ManagerEvent(status="completed", request_path=final_request, result_path=result_path))
            elif envelope.get("status") == "failed":
                failed_request = _move_request(request_path, failed_root, relative)
                _move_slurm_sidecar(sidecar, failed_root, relative)
                events.append(ManagerEvent(status="failed", request_path=failed_request, result_path=result_path))
            continue

        missing_result_entries.append((sidecar, request_path, result_path, request, job_id, submitted_at_epoch))

    job_ids = {job_id for _, _, _, _, job_id, _ in missing_result_entries if job_id}
    states = _query_sacct_states(job_ids, runner=runner)
    now = time.time()

    for sidecar, request_path, result_path, request, job_id, submitted_at_epoch in missing_result_entries:
        relative = request_path.relative_to(submitted_root)
        state = states.get(job_id or "")
        if state in _SLURM_FAILED_STATES:
            _write_json_atomic(result_path, _slurm_failed_envelope(request, job_id, state, result_path))
            failed_request = _move_request(request_path, failed_root, relative)
            _move_slurm_sidecar(sidecar, failed_root, relative)
            events.append(
                ManagerEvent(
                    status="failed",
                    request_path=failed_request,
                    result_path=result_path,
                    slurm_job_id=job_id,
                )
            )
        elif state in _SLURM_COMPLETED_STATES:
            _write_json_atomic(result_path, _slurm_failed_envelope(request, job_id, "COMPLETED_WITHOUT_RESULT", result_path))
            failed_request = _move_request(request_path, failed_root, relative)
            _move_slurm_sidecar(sidecar, failed_root, relative)
            events.append(
                ManagerEvent(
                    status="failed",
                    request_path=failed_request,
                    result_path=result_path,
                    slurm_job_id=job_id,
                )
            )
        elif job_id is None:
            _write_json_atomic(result_path, _slurm_failed_envelope(request, job_id, "SLURM_JOB_ID_MISSING", result_path))
            failed_request = _move_request(request_path, failed_root, relative)
            _move_slurm_sidecar(sidecar, failed_root, relative)
            events.append(
                ManagerEvent(
                    status="failed",
                    request_path=failed_request,
                    result_path=result_path,
                    slurm_job_id=job_id,
                )
            )
        elif state not in _SLURM_ACTIVE_STATES and now - submitted_at_epoch >= slurm_accounting_grace_sec:
            stale_state = "SLURM_JOB_NOT_FOUND_OR_STALE" if state is None else f"SLURM_JOB_NOT_FOUND_OR_STALE_{state}"
            _write_json_atomic(result_path, _slurm_failed_envelope(request, job_id, stale_state, result_path))
            failed_request = _move_request(request_path, failed_root, relative)
            _move_slurm_sidecar(sidecar, failed_root, relative)
            events.append(
                ManagerEvent(
                    status="failed",
                    request_path=failed_request,
                    result_path=result_path,
                    slurm_job_id=job_id,
                )
            )
    return events


def _active_slurm_job_count(submitted_root: Path) -> int:
    """Count submitted Slurm jobs still tracked after reconciliation."""
    active: set[str] = set()
    for sidecar in sorted(submitted_root.rglob("*.json.slurm.json")):
        payload = json.loads(sidecar.read_text())
        job_id = payload.get("slurm_job_id")
        active.add(str(job_id) if job_id else str(sidecar))
    return len(active)


def run_manager_once(
    queue_dir: Path | str,
    results_dir: Path | str,
    *,
    scheduler: RemoteScheduler = "direct",
    running_dir: Path | str | None = None,
    done_dir: Path | str | None = None,
    failed_dir: Path | str | None = None,
    submitted_dir: Path | str | None = None,
    python: str = sys.executable,
    sbatch_args: tuple[str, ...] = (),
    slurm_group_size: int = 10,
    max_active_slurm_jobs: int | None = None,
    slurm_accounting_grace_sec: float = _SLURM_ACCOUNTING_GRACE_SEC,
    runner: Any = subprocess.run,
) -> list[ManagerEvent]:
    """Consume currently queued requests once."""
    if scheduler not in ("direct", "slurm"):
        raise ValueError(f"Unsupported scheduler {scheduler!r}")
    if slurm_group_size < 1:
        raise ValueError(f"slurm_group_size must be >= 1, got {slurm_group_size}")
    if scheduler == "slurm" and max_active_slurm_jobs is None:
        raise ValueError("max_active_slurm_jobs is required when scheduler='slurm'")
    if max_active_slurm_jobs is not None and max_active_slurm_jobs < 1:
        raise ValueError(f"max_active_slurm_jobs must be >= 1, got {max_active_slurm_jobs}")
    queue_root = Path(queue_dir)
    results_root = Path(results_dir)
    state_root = queue_root.parent
    running_root = Path(running_dir) if running_dir is not None else state_root / "running"
    done_root = Path(done_dir) if done_dir is not None else state_root / "done"
    failed_root = Path(failed_dir) if failed_dir is not None else state_root / "failed"
    submitted_root = Path(submitted_dir) if submitted_dir is not None else state_root / "submitted"

    events: list[ManagerEvent] = []
    events.extend(
        reconcile_running_requests(
            running_root,
            results_root,
            done_dir=done_root,
            failed_dir=failed_root,
        )
    )
    if scheduler == "slurm":
        events.extend(
            reconcile_submitted_slurm_jobs(
                submitted_root,
                results_root,
                done_dir=done_root,
                failed_dir=failed_root,
                slurm_accounting_grace_sec=slurm_accounting_grace_sec,
                runner=runner,
            )
        )

    request_files = sorted(queue_root.rglob("*.json"))
    if scheduler == "slurm" and max_active_slurm_jobs is not None:
        active_jobs = _active_slurm_job_count(submitted_root)
        available_jobs = max_active_slurm_jobs - active_jobs
        if available_jobs <= 0:
            return events
        request_files = request_files[: available_jobs * slurm_group_size]

    slurm_pending: list[tuple[Path, Path, dict[str, Any], Path]] = []
    for request_file in request_files:
        queued_request: dict[str, Any] | None = None
        admission_error: Exception | None = None
        if scheduler == "direct":
            try:
                queued_request = cast(dict[str, Any], json.loads(request_file.read_text()))
                if not _queued_request_resource_ready(queued_request, runner=runner):
                    continue
            except Exception as exc:
                admission_error = exc
        relative = request_file.relative_to(queue_root)
        claimed = _move_request(request_file, running_root, relative)
        request: dict[str, Any] | None = queued_request
        result_path: Path | None = None
        try:
            if admission_error is not None:
                raise admission_error
            if request is None:
                request = json.loads(claimed.read_text())
            result_path = _request_result_path(results_root, request)
            if scheduler == "direct":
                result = execute_request(request)
                _write_json_atomic(result_path, result)
                final_request = _move_request(claimed, done_root, relative)
                events.append(ManagerEvent(status="completed", request_path=final_request, result_path=result_path))
            elif scheduler == "slurm":
                slurm_pending.append((claimed, relative, request, result_path))
        except Exception as exc:
            result_path = result_path or _invalid_request_result_path(results_root, claimed)
            _write_json_atomic(result_path, _failure_envelope(request, exc))
            failed_request = _move_request(claimed, failed_root, relative)
            events.append(ManagerEvent(status="failed", request_path=failed_request, result_path=result_path))

    for group in _chunks(slurm_pending, slurm_group_size):
        events.extend(
            _submit_slurm_group(
                group,
                results_root=results_root,
                submitted_root=submitted_root,
                failed_root=failed_root,
                python=python,
                sbatch_args=sbatch_args,
                runner=runner,
            )
        )
    return events


def run_manager_loop(
    queue_dir: Path | str,
    results_dir: Path | str,
    *,
    scheduler: RemoteScheduler = "direct",
    poll_interval: float = 5.0,
    once: bool = False,
    python: str = sys.executable,
    sbatch_args: tuple[str, ...] = (),
    slurm_group_size: int = 10,
    max_active_slurm_jobs: int | None = None,
    slurm_accounting_grace_sec: float = _SLURM_ACCOUNTING_GRACE_SEC,
    runner: Any = subprocess.run,
) -> None:
    """Run a persistent node manager loop."""
    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
    if slurm_group_size < 1:
        raise ValueError(f"slurm_group_size must be >= 1, got {slurm_group_size}")
    if scheduler == "slurm" and max_active_slurm_jobs is None:
        raise ValueError("max_active_slurm_jobs is required when scheduler='slurm'")
    if max_active_slurm_jobs is not None and max_active_slurm_jobs < 1:
        raise ValueError(f"max_active_slurm_jobs must be >= 1, got {max_active_slurm_jobs}")

    persist_ctx = ToolInstance.persist() if scheduler == "direct" else None
    if persist_ctx is not None:
        persist_ctx.__enter__()
    try:
        while True:
            try:
                run_manager_once(
                    queue_dir,
                    results_dir,
                    scheduler=scheduler,
                    python=python,
                    sbatch_args=sbatch_args,
                    slurm_group_size=slurm_group_size,
                    max_active_slurm_jobs=max_active_slurm_jobs,
                    slurm_accounting_grace_sec=slurm_accounting_grace_sec,
                    runner=runner,
                )
            except Exception:
                traceback.print_exc()
                if once:
                    raise
            if once:
                return
            time.sleep(poll_interval)
    finally:
        if persist_ctx is not None:
            persist_ctx.__exit__(None, None, None)


def collect_plan_outputs(plan: RemoteDispatchPlan, results_dir: Path | str) -> BaseToolOutput:
    """Merge completed bucket result files into the tool's output model."""
    spec = ToolRegistry.get(plan.tool_key)
    output_items: dict[int, Any] = {}
    warnings: list[str] = []
    errors: list[str] = []
    metadata: dict[str, Any] = {"remote_dispatch": {"run_id": plan.run_id, "buckets": len(plan.assignments)}}

    for assignment in plan.assignments:
        result_path = Path(results_dir) / assignment.node.name / plan.run_id / f"{assignment.bucket.bucket_id}.json"
        envelope = json.loads(result_path.read_text())
        if envelope.get("status") != "completed":
            trace_tail = str(envelope.get("traceback", "")).splitlines()[-1:] or [""]
            raise RuntimeError(
                f"Remote bucket {assignment.node.name}/{assignment.bucket.bucket_id} did not complete: "
                f"{envelope.get('error')} (result_path={result_path}; trace_tail={trace_tail[0]})"
            )
        output = spec.output_model.model_validate(envelope["output"])
        bucket_outputs = list(getattr(output, plan.iterable_output_field))
        if len(bucket_outputs) != len(assignment.bucket.item_indices):
            raise RuntimeError(
                f"Remote bucket {assignment.bucket.bucket_id} returned {len(bucket_outputs)} "
                f"{plan.iterable_output_field} for {len(assignment.bucket.item_indices)} input item(s)"
            )
        output_items.update(dict(zip(assignment.bucket.item_indices, bucket_outputs, strict=True)))
        warnings.extend(output.warnings)
        errors.extend(output.errors)

    expected = set(range(sum(len(assignment.bucket.item_indices) for assignment in plan.assignments)))
    missing = sorted(expected - set(output_items))
    if missing:
        raise RuntimeError(f"Missing remote output item(s): {missing}")
    merged = [output_items[index] for index in sorted(output_items)]
    output_payload: dict[str, Any] = {
        plan.iterable_output_field: merged,
        "warnings": warnings,
        "errors": errors,
        "metadata": metadata,
    }
    return cast(BaseToolOutput, spec.output_model.model_validate(output_payload))


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the remote manager CLI parser."""
    parser = argparse.ArgumentParser(prog="python -m proto_tools.utils.remote_execution")
    sub = parser.add_subparsers(dest="command", required=True)

    run_request = sub.add_parser("run-request")
    run_request.add_argument("--request", required=True)
    run_request.add_argument("--output", required=True)

    run_request_batch = sub.add_parser("run-request-batch")
    run_request_batch.add_argument("--request", action="append", required=True)
    run_request_batch.add_argument("--results-dir", required=True)

    loop = sub.add_parser("manager-loop")
    loop.add_argument("--queue-dir", required=True)
    loop.add_argument("--results-dir", required=True)
    loop.add_argument("--scheduler", choices=("direct", "slurm"), default="direct")
    loop.add_argument("--poll-interval", type=float, default=5.0)
    loop.add_argument("--once", action="store_true")
    loop.add_argument("--python", default=sys.executable)
    loop.add_argument("--sbatch-arg", action="append", default=[])
    loop.add_argument("--slurm-group-size", type=int, default=10)
    loop.add_argument(
        "--slurm-accounting-grace-sec",
        type=float,
        default=_SLURM_ACCOUNTING_GRACE_SEC,
        help="Seconds to wait before failing submitted Slurm jobs missing from accounting.",
    )
    loop.add_argument(
        "--max-active-slurm-jobs",
        type=int,
        help="Required with --scheduler slurm; caps active submitted Slurm jobs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for node-side request execution and manager loops."""
    args = _build_arg_parser().parse_args(argv)
    if args.command == "run-request":
        execute_request_file(args.request, args.output)
        return 0
    if args.command == "run-request-batch":
        execute_request_batch(args.request, args.results_dir)
        return 0
    if args.command == "manager-loop":
        run_manager_loop(
            args.queue_dir,
            args.results_dir,
            scheduler=args.scheduler,
            poll_interval=args.poll_interval,
            once=args.once,
            python=args.python,
            sbatch_args=tuple(args.sbatch_arg),
            slurm_group_size=args.slurm_group_size,
            max_active_slurm_jobs=args.max_active_slurm_jobs,
            slurm_accounting_grace_sec=args.slurm_accounting_grace_sec,
        )
        return 0
    raise ValueError(f"Unsupported command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())

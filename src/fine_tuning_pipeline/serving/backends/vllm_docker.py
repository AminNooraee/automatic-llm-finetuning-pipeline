"""Safe local-Docker launcher for one vLLM process with one static LoRA."""

from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence

from ..contracts import (
    ServingConflictError,
    ServingError,
    ServingLaunchRequest,
    ServingLaunchResult,
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class SubprocessCommandRunner:
    def run(self, command: Sequence[str]) -> CommandResult:
        try:
            result = subprocess.run(
                list(command), capture_output=True, text=True, check=False, timeout=60
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ServingError(f"Docker command could not be executed: {error}") from error
        return CommandResult(result.returncode, result.stdout, result.stderr)


def port_is_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
        return True
    except OSError:
        return False


def build_vllm_docker_command(request: ServingLaunchRequest) -> tuple[str, ...]:
    config = request.config
    if config.vllm is None:
        raise ServingError("Internal serving error: vLLM configuration is missing")
    adapter = request.adapter_path.resolve()
    publish_host = (
        f"[{config.bind_host}]" if ":" in config.bind_host else config.bind_host
    )
    command = [
        "docker", "run", "--detach",
        "--name", config.container_name,
        "--restart", config.restart_policy,
        "--label", "fine-tuning-pipeline.managed=true",
        "--label", f"fine-tuning-pipeline.run-id={request.run_id}",
        "--label", f"fine-tuning-pipeline.execution-id={request.execution_id or request.run_id}",
        "--gpus", f"device={config.vllm.gpu_devices}",
        "--publish", f"{publish_host}:{config.port}:8000",
    ]
    for resolver in request.runtime_dns:
        command.extend(("--dns", resolver))
    command.extend(("--volume", f"{adapter}:/adapters/fine-tuned:ro"))
    if request.host_hf_cache_path is not None:
        command.extend((
            "--volume", f"{request.host_hf_cache_path}:/root/.cache/huggingface",
        ))
    command.extend([
        config.vllm.image,
        request.base_model,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", config.base_model_name,
        "--enable-lora",
        "--lora-modules", f"{config.fine_tuned_model_name}=/adapters/fine-tuned",
        "--max-lora-rank", str(request.lora_rank),
        "--gpu-memory-utilization", str(config.vllm.gpu_memory_utilization),
        "--max-model-len", str(config.vllm.max_model_len),
        "--max-num-seqs", str(config.vllm.max_num_seqs),
    ])
    if request.resolved_revision:
        command.extend(("--revision", request.resolved_revision))
    return tuple(command)


class VllmDockerBackend:
    """Inspect conflicts, then create exactly one current-run-owned container."""

    def __init__(
        self,
        *,
        runner: SubprocessCommandRunner | None = None,
        port_checker: Callable[[str, int], bool] = port_is_available,
    ):
        self.runner = runner or SubprocessCommandRunner()
        self.port_checker = port_checker

    def preflight(self, request: ServingLaunchRequest) -> None:
        version = self.runner.run(("docker", "version", "--format", "{{.Server.Version}}"))
        if version.returncode != 0:
            raise ServingError(
                "Docker is unavailable or its daemon cannot be reached. No Docker resource was modified."
            )
        if not self.port_checker(request.config.bind_host, request.config.port):
            raise ServingConflictError(
                f"Serving conflict detected. Port {request.config.port} is already in use. "
                "No existing service was modified."
            )
        inspected = self.runner.run(
            ("docker", "container", "ls", "--all", "--format", "{{.Names}}")
        )
        if inspected.returncode != 0:
            raise ServingError(
                "Docker container-name inspection failed. No Docker resource was modified."
            )
        names = {line.strip() for line in inspected.stdout.splitlines() if line.strip()}
        if request.config.container_name in names:
            raise ServingConflictError(
                f"Serving conflict detected. Container name {request.config.container_name!r} "
                "already exists. No existing container was modified."
            )

    def launch(self, request: ServingLaunchRequest) -> ServingLaunchResult:
        command = build_vllm_docker_command(request)
        result = self.runner.run(command)
        container_id = result.stdout.strip() if result.returncode == 0 else ""
        if not container_id:
            container_id = self._owned_container_id(request) or ""
        if result.returncode != 0 and not container_id:
            diagnostic = " ".join((result.stderr or result.stdout).split())[:500]
            raise ServingError(
                "vLLM Docker launch failed; no existing resource was stopped, removed, or replaced"
                + (f": {diagnostic}" if diagnostic else "")
            )
        if not container_id:
            raise ServingError("Docker reported success but returned no vLLM container ID")
        image = self.runner.run((
            "docker", "image", "inspect", "--format", "{{.Id}}", request.config.vllm.image
        ))
        return ServingLaunchResult(
            container_id=container_id,
            command=command,
            image_id=image.stdout.strip() if image.returncode == 0 else None,
        )

    def _owned_container_id(self, request: ServingLaunchRequest) -> str | None:
        inspected = self.runner.run((
            "docker", "container", "inspect", "--format",
            "{{.Id}}|{{index .Config.Labels \"fine-tuning-pipeline.managed\"}}|"
            "{{index .Config.Labels \"fine-tuning-pipeline.run-id\"}}|"
            "{{index .Config.Labels \"fine-tuning-pipeline.execution-id\"}}",
            request.config.container_name,
        ))
        if inspected.returncode != 0:
            return None
        parts = inspected.stdout.strip().split("|")
        expected_execution = request.execution_id or request.run_id
        if len(parts) == 4 and parts[1:] == ["true", request.run_id, expected_execution]:
            return parts[0] or None
        return None

    def remove_if_owned(self, request: ServingLaunchRequest, container_id: str) -> bool:
        """Remove only the exact container carrying this execution's labels."""
        inspected = self.runner.run((
            "docker", "container", "inspect", "--format",
            "{{.Id}}|{{index .Config.Labels \"fine-tuning-pipeline.managed\"}}|"
            "{{index .Config.Labels \"fine-tuning-pipeline.run-id\"}}|"
            "{{index .Config.Labels \"fine-tuning-pipeline.execution-id\"}}",
            request.config.container_name,
        ))
        execution_id = request.execution_id or request.run_id
        expected = f"{container_id}|true|{request.run_id}|{execution_id}"
        if inspected.returncode != 0 or inspected.stdout.strip() != expected:
            return False
        removed = self.runner.run(("docker", "container", "rm", "--force", container_id))
        if removed.returncode != 0:
            raise ServingError("Owned serving container cleanup failed")
        return True

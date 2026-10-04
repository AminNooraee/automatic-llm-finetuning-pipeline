"""Trusted Docker-socket controller for the one-command deployment profile."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import yaml

from ..artifact_validator import read_lora_adapter_metadata, validate_model_artifacts
from ..gateway.config import resolve_gateway_config
from ..gateway.manager import GatewayManager
from ..security import redact_data, redact_text
from ..serving.config import resolve_serving_config
from ..serving.manager import ServingManager
from ..training_config import resolve_training_config
from .config import resolve_orchestration_config


class ControllerError(RuntimeError):
    pass


@dataclass(frozen=True)
class DockerResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class DockerRunner:
    def run(self, command: Sequence[str], *, timeout: float | None = None) -> DockerResult:
        try:
            result = subprocess.run(
                list(command), capture_output=True, text=True, check=False, timeout=timeout
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ControllerError(f"Docker operation could not be executed: {type(error).__name__}") from None
        return DockerResult(result.returncode, result.stdout, result.stderr)


def _safe_docker_error_reason(
    result: DockerResult, *, secrets: Sequence[str] = (), limit: int = 320
) -> str:
    """Return one bounded, redacted diagnostic line from a failed Docker call."""
    raw = result.stderr.strip() or result.stdout.strip()
    if not raw:
        return f"Docker exited with status {result.returncode}"
    reason = " ".join(redact_text(raw, list(secrets)).split())
    if len(reason) > limit:
        reason = reason[: limit - 3].rstrip() + "..."
    return reason


def _validate_runtime_dns(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            raise ControllerError(
                "PIPELINE_DOCKER_RUNTIME_DNS contains an invalid resolver address"
            ) from None
        mapped = getattr(address, "ipv4_mapped", None)
        if address.is_loopback or address.is_unspecified or (
            mapped is not None and mapped.is_loopback
        ):
            raise ControllerError(
                "PIPELINE_DOCKER_RUNTIME_DNS contains an unusable resolver address"
            )
        canonical = str(address)
        if canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized)


def _runtime_dns_from_environment() -> tuple[str, ...]:
    raw = os.environ.get("PIPELINE_DOCKER_RUNTIME_DNS", "")
    values = raw.split()
    if not values:
        raise ControllerError("PIPELINE_DOCKER_RUNTIME_DNS is missing or empty")
    return _validate_runtime_dns(values)


def _validate_selected_gpu(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
        raise ControllerError(
            "PIPELINE_SELECTED_GPU must be exactly one non-negative integer GPU index"
        )
    return value.lstrip("0") or "0"


def _selected_gpu_from_environment() -> str | None:
    if "PIPELINE_SELECTED_GPU" not in os.environ:
        return None
    return _validate_selected_gpu(os.environ["PIPELINE_SELECTED_GPU"])


@dataclass(frozen=True)
class RuntimeLayout:
    host_project: Path
    host_runs: Path
    host_cache: Path
    host_state: Path
    host_secrets: Path
    controller_project: Path = Path("/workspace/project")
    controller_runs: Path = Path("/workspace/runs")
    controller_state: Path = Path("/workspace/state")
    worker_uid: int | None = None
    worker_gid: int | None = None
    host_datasets: Path | None = None
    controller_datasets: Path = Path("/workspace/datasets")


def _safe_relative(value: str, label: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value or any(ord(char) < 32 for char in value):
        raise ControllerError(f"{label} must be a portable relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ControllerError(f"{label} must be a normalized relative path without traversal")
    return candidate


def _beneath(root: Path, relative: str, label: str, *, must_exist: bool = True) -> Path:
    rel = _safe_relative(relative, label)
    root_resolved = root.resolve(strict=True)
    target = root_resolved.joinpath(*rel.parts).resolve(strict=must_exist)
    try:
        target.relative_to(root_resolved)
    except ValueError:
        raise ControllerError(f"{label} escapes its configured shared root") from None
    return target


def _host_join(root: Path, relative: str, label: str) -> Path:
    """Map an already-validated relative path onto a daemon-visible host root."""
    rel = _safe_relative(relative, label)
    if not root.is_absolute():
        raise ControllerError(f"{label} host root must be absolute")
    return root.joinpath(*rel.parts)


def _atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(redact_data(value), indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_secret(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").rstrip("\r\n")
    return value or None


def _worker_config(raw: dict, config_path: Path, layout: RuntimeLayout) -> dict:
    """Translate a local dataset into the trainer's read-only dataset mount."""
    value = redact_data(raw)
    dataset = value.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("path"), str):
        raise ControllerError("dataset.path must be a non-empty string")
    raw_path = dataset["path"].strip()
    if not raw_path:
        raise ControllerError("dataset.path must be a non-empty string")
    host_root = layout.host_datasets or layout.host_project
    controller_root = layout.controller_datasets
    candidate: Path | None = None
    relative: Path | None = None
    configured = Path(raw_path)
    if configured.is_absolute():
        try:
            relative = configured.relative_to(host_root)
        except ValueError:
            raise ControllerError(
                "Absolute dataset.path must remain beneath PIPELINE_DATASETS_DIR"
            ) from None
        candidate = controller_root / relative
    else:
        project_candidate = (config_path.parent / configured).resolve()
        try:
            project_relative = project_candidate.relative_to(layout.controller_project.resolve())
        except ValueError:
            project_relative = None
        if project_relative is not None and host_root == layout.host_project:
            relative = project_relative
            candidate = controller_root / relative
        else:
            relative = configured
            candidate = controller_root / relative
    if candidate.exists():
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(controller_root.resolve(strict=True))
        except ValueError:
            raise ControllerError("dataset.path escapes PIPELINE_DATASETS_DIR") from None
        dataset["path"] = f"/workspace/datasets/{relative.as_posix()}"
    elif configured.is_absolute() or raw_path.startswith((".", "..")):
        raise ControllerError("Configured local dataset path does not exist in PIPELINE_DATASETS_DIR")
    return value


def _record_phase(metadata_path: Path, phase: str, status: str, **details) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    phases = metadata.setdefault("phases", {})
    phases[phase] = {"status": status, **redact_data(details)}
    _atomic_json(metadata_path, metadata)


class PipelineController:
    def __init__(
        self,
        *,
        docker: DockerRunner | None = None,
        serving_manager_factory=ServingManager,
        gateway_manager_factory=GatewayManager,
    ):
        self.docker = docker or DockerRunner()
        self.serving_manager_factory = serving_manager_factory
        self.gateway_manager_factory = gateway_manager_factory

    def deploy(
        self,
        *,
        execution_id: str,
        config_relative: str,
        layout: RuntimeLayout,
        training_image: str,
        training_image_id: str,
        source_revision: str,
        source_identity: str,
        runtime_dns: Sequence[str] = (),
        selected_gpu: str | None = None,
    ) -> Path:
        selected_gpu = _validate_selected_gpu(selected_gpu)
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{7,63}", execution_id):
            raise ControllerError("Invalid execution ID")
        runtime_dns = _validate_runtime_dns(runtime_dns)
        config_rel = _safe_relative(config_relative, "Configuration path")
        controller_config = _beneath(
            layout.controller_project, config_rel.as_posix(), "Configuration path"
        )
        raw = yaml.safe_load(controller_config.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ControllerError("Configuration must be a YAML mapping")
        orchestration = resolve_orchestration_config(raw.get("orchestration"))
        if not orchestration.enabled:
            raise ControllerError("The one-command controller requires orchestration.enabled=true")
        configured_training_gpu_devices = orchestration.training_gpu_devices
        effective_training_gpu_devices = selected_gpu or configured_training_gpu_devices
        if selected_gpu is not None:
            print(
                "GPU admission override: "
                f"configured training devices={configured_training_gpu_devices}; "
                f"effective device={effective_training_gpu_devices}"
            )

        gateway_key = _read_secret(Path("/run/pipeline-secrets/litellm_api_key"))
        if gateway_key:
            os.environ["LITELLM_API_KEY"] = gateway_key

        execution_dir = layout.controller_state / "executions" / execution_id
        execution_dir.mkdir(parents=True, exist_ok=False)
        if layout.worker_uid is not None and layout.worker_gid is not None:
            os.chown(execution_dir, layout.worker_uid, layout.worker_gid)
            execution_dir.chmod(0o700)
        worker_config_path = execution_dir / "worker_config.yaml"
        worker_config_path.write_text(
            yaml.safe_dump(_worker_config(raw, controller_config, layout), sort_keys=False),
            encoding="utf-8",
        )
        result_path = execution_dir / "training_result.json"
        host_execution_dir = layout.host_state / "executions" / execution_id
        training_name = f"ft-training-{execution_id[:24]}"
        labels = (
            "--label", "fine-tuning-pipeline.managed=true",
            "--label", "fine-tuning-pipeline.role=training",
            "--label", f"fine-tuning-pipeline.execution-id={execution_id}",
        )
        command = [
            "docker", "container", "create", "--name", training_name,
            *labels,
            "--gpus", f"device={effective_training_gpu_devices}",
            "--mount", f"type=bind,src={layout.host_project},dst=/workspace/project,readonly",
            "--mount", (
                f"type=bind,src={layout.host_datasets or layout.host_project},"
                "dst=/workspace/datasets,readonly"
            ),
            "--mount", f"type=bind,src={layout.host_runs},dst=/workspace/runs",
            "--mount", f"type=bind,src={layout.host_cache},dst=/cache/huggingface",
            "--mount", f"type=bind,src={host_execution_dir},dst=/workspace/state/executions/{execution_id}",
        ]
        for resolver in runtime_dns:
            command.extend(("--dns", resolver))
        hf_secret = layout.host_secrets / "hf_token"
        if Path("/run/pipeline-secrets/hf_token").is_file():
            command.extend((
                "--mount", f"type=bind,src={hf_secret},dst=/run/secrets/hf_token,readonly",
                "--env", "HF_TOKEN_FILE=/run/secrets/hf_token",
            ))
        command.extend((
            "--env", f"PIPELINE_TRAINING_IMAGE={training_image}",
            "--env", f"PIPELINE_TRAINING_IMAGE_ID={training_image_id}",
            training_image,
            "python", "-m", "fine_tuning_pipeline.orchestration.training_worker",
            "--config", f"/workspace/state/executions/{execution_id}/worker_config.yaml",
            "--runs-root", "/workspace/runs",
            "--result", f"/workspace/state/executions/{execution_id}/training_result.json",
            "--execution-id", execution_id,
        ))
        _atomic_json(execution_dir / "ownership.json", {
            "schema_version": 1,
            "execution_id": execution_id,
            "training_container": {"name": training_name, "id": None, "status": "planned"},
        })
        created = self.docker.run(command)
        if created.returncode != 0 or not created.stdout.strip():
            self._remove_planned_training_if_owned(training_name, execution_id)
            reason = _safe_docker_error_reason(
                created, secrets=(gateway_key,) if gateway_key else ()
            )
            raise ControllerError(
                f"Training container creation failed: {reason}; "
                "no serving or gateway mutation occurred"
            )
        training_id = created.stdout.strip()
        try:
            _atomic_json(execution_dir / "ownership.json", {
                "schema_version": 1,
                "execution_id": execution_id,
                "training_container": {"name": training_name, "id": training_id, "status": "created"},
            })
        except Exception:
            self._remove_training_if_owned(training_name, training_id, execution_id)
            raise
        try:
            started = self.docker.run(("docker", "container", "start", training_id))
            if started.returncode != 0:
                raise ControllerError("Training container could not be started")
            waited = self.docker.run(("docker", "container", "wait", training_id))
            logs = self.docker.run(("docker", "container", "logs", training_id))
            if logs.stdout:
                print(logs.stdout, end="")
            try:
                exit_code = int(waited.stdout.strip()) if waited.returncode == 0 else -1
            except ValueError:
                exit_code = -1
        finally:
            self._remove_training_if_owned(training_name, training_id, execution_id)
        if exit_code != 0:
            raise ControllerError("Training failed; serving and gateway stages were not started")
        if not result_path.is_file():
            raise ControllerError("Training succeeded without the required result descriptor")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self._validate_result(result, execution_id)
        run_root = _beneath(layout.controller_runs, result["run_directory"], "Training run directory")
        model_dir = _beneath(layout.controller_runs, result["model_directory"], "Model directory")
        metadata_path = _beneath(layout.controller_runs, result["metadata_path"], "Metadata path")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_id") != result["run_id"]:
            raise ControllerError("Training result and metadata run IDs do not match")
        _record_phase(
            metadata_path, "training", "success", execution_id=execution_id,
            image={"reference": training_image, "id": training_image_id},
            gpu_devices={
                "configured": configured_training_gpu_devices,
                "effective": effective_training_gpu_devices,
                "admission_override": selected_gpu,
            },
        )
        training_config = resolve_training_config(raw["training"])
        validate_model_artifacts(
            model_dir,
            training_config.method,
            expected_base_model=raw["model"]["name"],
            expected_base_revisions=(
                metadata.get("model", {}).get("requested_revision"),
                metadata.get("model", {}).get("resolved_revision"),
            ),
        )
        adapter = read_lora_adapter_metadata(
            model_dir,
            expected_base_model=raw["model"]["name"],
            expected_base_revisions=(
                metadata.get("model", {}).get("requested_revision"),
                metadata.get("model", {}).get("resolved_revision"),
            ),
        )
        serving = resolve_serving_config(
            raw.get("serving"), run_id=result["run_id"],
            base_model=raw["model"]["name"], training_method=training_config.method,
        )
        if not serving.enabled:
            raise ControllerError("The full deployment controller requires serving.enabled=true")
        configured_serving_gpu_devices = (
            serving.vllm.gpu_devices if serving.vllm is not None else None
        )
        if selected_gpu is not None and serving.vllm is not None:
            serving = replace(
                serving,
                vllm=replace(serving.vllm, gpu_devices=selected_gpu),
            )
            print(
                "GPU admission override: "
                f"configured serving devices={configured_serving_gpu_devices}; "
                f"effective device={selected_gpu}"
            )
        gateway = resolve_gateway_config(raw.get("gateway"), serving=serving, run_id=result["run_id"])
        host_model_dir = _host_join(layout.host_runs, result["model_directory"], "Host model directory")
        serving_manager = self.serving_manager_factory()
        serving_result = None
        current_phase = "serving"
        _record_phase(
            metadata_path, "serving", "starting", backend=serving.backend,
            runtime=serving.runtime,
        )
        try:
            serving_result = serving_manager.start_and_verify(
                run_id=result["run_id"], run_root=run_root, config=serving,
                base_model=raw["model"]["name"],
                resolved_revision=(metadata.get("model", {}).get("resolved_revision")
                                   or metadata.get("model", {}).get("requested_revision")),
                adapter_path=host_model_dir, lora_rank=adapter.rank,
                execution_id=execution_id, host_hf_cache_path=layout.host_cache,
                runtime_dns=tuple(runtime_dns),
                cleanup_on_failure=orchestration.cleanup_on_failure,
            )
            _record_phase(
                metadata_path, "serving", "ready",
                manifest=serving_result["manifest_path"],
                health_check=serving_result["health_path"],
                container_id=serving_result["container_id"],
            )
            gateway_result = None
            if gateway.enabled:
                current_phase = "gateway"
                _record_phase(metadata_path, "gateway", "starting", provider=gateway.provider)
                gateway_result = self.gateway_manager_factory(gateway).register_and_verify(
                    run_id=result["run_id"], run_root=run_root, serving=serving,
                    cleanup_on_failure=orchestration.cleanup_on_failure,
                )
                _record_phase(
                    metadata_path, "gateway", "ready",
                    manifest=gateway_result["manifest_path"],
                    registration=gateway_result["registration_path"],
                )
        except Exception as error:
            serving_rolled_back = False
            serving_rollback_error = None
            if serving_result is not None and orchestration.cleanup_on_failure:
                try:
                    serving_rolled_back = serving_manager.cleanup_owned(
                        run_id=result["run_id"], execution_id=execution_id,
                        container_id=serving_result["container_id"], config=serving,
                        base_model=raw["model"]["name"], adapter_path=host_model_dir,
                        lora_rank=adapter.rank, host_hf_cache_path=layout.host_cache,
                        runtime_dns=tuple(runtime_dns),
                    )
                except Exception as cleanup_error:
                    serving_rollback_error = type(cleanup_error).__name__
                serving_result = {**serving_result, "rolled_back": serving_rolled_back}
            _record_phase(
                metadata_path, current_phase, "failed",
                error={"type": type(error).__name__},
                serving_rolled_back=serving_rolled_back,
                serving_rollback_error=serving_rollback_error,
            )
            self._write_deployment_manifest(
                run_root, execution_id, result, source_revision, source_identity,
                training_image, training_image_id, serving, serving_result, gateway, None, "failed",
                configured_training_gpu_devices=configured_training_gpu_devices,
                effective_training_gpu_devices=effective_training_gpu_devices,
                configured_serving_gpu_devices=configured_serving_gpu_devices,
                admitted_gpu_override=selected_gpu,
            )
            raise
        manifest = self._write_deployment_manifest(
            run_root, execution_id, result, source_revision, source_identity,
            training_image, training_image_id, serving, serving_result, gateway, gateway_result, "success",
            configured_training_gpu_devices=configured_training_gpu_devices,
            effective_training_gpu_devices=effective_training_gpu_devices,
            configured_serving_gpu_devices=configured_serving_gpu_devices,
            admitted_gpu_override=selected_gpu,
        )
        self._print_success(result["run_id"], serving, gateway, manifest)
        return manifest

    def _remove_training_if_owned(self, name: str, container_id: str, execution_id: str) -> None:
        inspected = self.docker.run((
            "docker", "container", "inspect", "--format",
            "{{.Id}}|{{index .Config.Labels \"fine-tuning-pipeline.managed\"}}|"
            "{{index .Config.Labels \"fine-tuning-pipeline.role\"}}|"
            "{{index .Config.Labels \"fine-tuning-pipeline.execution-id\"}}", name,
        ))
        expected = f"{container_id}|true|training|{execution_id}"
        if inspected.returncode == 0 and inspected.stdout.strip() == expected:
            removed = self.docker.run(("docker", "container", "rm", container_id))
            if removed.returncode != 0:
                raise ControllerError("Owned training container cleanup failed")
            return
        raise ControllerError(
            "Training container ownership could not be verified; serving was not started"
        )

    def _remove_planned_training_if_owned(self, name: str, execution_id: str) -> None:
        inspected = self.docker.run((
            "docker", "container", "inspect", "--format",
            "{{.Id}}|{{index .Config.Labels \"fine-tuning-pipeline.managed\"}}|"
            "{{index .Config.Labels \"fine-tuning-pipeline.role\"}}|"
            "{{index .Config.Labels \"fine-tuning-pipeline.execution-id\"}}", name,
        ))
        if inspected.returncode != 0:
            return
        parts = inspected.stdout.strip().split("|")
        if len(parts) == 4 and parts[1:] == ["true", "training", execution_id] and parts[0]:
            removed = self.docker.run(("docker", "container", "rm", "--force", parts[0]))
            if removed.returncode != 0:
                raise ControllerError("Ambiguous owned training-container cleanup failed")

    @staticmethod
    def _validate_result(result: object, execution_id: str) -> None:
        if not isinstance(result, dict) or result.get("status") != "success":
            raise ControllerError("Training result does not report success")
        if result.get("execution_id") != execution_id:
            raise ControllerError("Training result belongs to another execution")
        for field in ("run_id", "run_directory", "model_directory", "metadata_path"):
            if not isinstance(result.get(field), str) or not result[field]:
                raise ControllerError(f"Training result field {field!r} is missing")
        _safe_relative(result["run_directory"], "Training run directory")
        _safe_relative(result["model_directory"], "Model directory")
        _safe_relative(result["metadata_path"], "Metadata path")
        run_parts = _safe_relative(result["run_directory"], "Training run directory").parts
        for field in ("model_directory", "metadata_path"):
            parts = _safe_relative(result[field], field).parts
            if parts[:len(run_parts)] != run_parts:
                raise ControllerError(f"Training result field {field!r} is outside its run directory")

    @staticmethod
    def _write_deployment_manifest(
        run_root: Path, execution_id: str, training: dict,
        source_revision: str, source_identity: str,
        training_image: str, training_image_id: str,
        serving, serving_result, gateway, gateway_result, status: str,
        *,
        configured_training_gpu_devices: str,
        effective_training_gpu_devices: str,
        configured_serving_gpu_devices: str | None,
        admitted_gpu_override: str | None,
    ) -> Path:
        path = run_root / "deployment_manifest.json"
        value = {
            "schema_version": 1,
            "execution_id": execution_id,
            "run_id": training["run_id"],
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": {"revision": source_revision, "identity": source_identity},
            "training": {
                "status": "success", "base_model": training.get("base_model"),
                "resolved_revision": training.get("resolved_revision"),
                "method": training.get("training_method"),
                "artifact_type": training.get("adapter_type"),
                "model_directory": training.get("model_directory"),
                "artifact_relationship": "base_model_plus_lora_adapter",
                "image": {"reference": training_image, "id": training_image_id},
                "gpu_devices": {
                    "configured": configured_training_gpu_devices,
                    "effective": effective_training_gpu_devices,
                    "admission_override": admitted_gpu_override,
                },
            },
            "serving": {
                "provider": "vllm",
                "status": (
                    "rolled_back" if serving_result and serving_result.get("rolled_back")
                    else "ready" if serving_result else "failed"
                ),
                "base_url": serving.base_url,
                "base_alias": serving.base_model_name,
                "fine_tuned_alias": serving.fine_tuned_model_name,
                "container_name": serving.container_name,
                "container_id": serving_result.get("container_id") if serving_result else None,
                "image": serving.vllm.image if serving.vllm else None,
                "image_id": serving_result.get("image_id") if serving_result else None,
                "restart_policy": serving.restart_policy,
                "port": serving.port,
                "gpu_devices": serving.vllm.gpu_devices if serving.vllm else None,
                "configured_gpu_devices": configured_serving_gpu_devices,
                "admission_override": admitted_gpu_override,
                "manifest": serving_result.get("manifest_path") if serving_result else None,
                "health_check": serving_result.get("health_path") if serving_result else None,
                "operation": (
                    serving_result.get("operation_path") if serving_result
                    else "serving/operation.json"
                    if (run_root / "serving" / "operation.json").is_file() else None
                ),
                "log": serving_result.get("log_path") if serving_result else "serving/serving.log",
                "cache_strategy": "shared_complete_huggingface_cache_root",
            },
            "gateway": {
                "provider": "litellm",
                "status": "ready" if gateway_result else ("failed" if gateway.enabled else "disabled"),
                "manifest": gateway_result.get("manifest_path") if gateway_result else None,
                "registration": (
                    gateway_result.get("registration_path") if gateway_result
                    else "gateway/registration.json"
                    if (run_root / "gateway" / "registration.json").is_file() else None
                ),
                "health_check": gateway_result.get("health_path") if gateway_result else None,
                "registrations": gateway_result.get("registrations", []) if gateway_result else [],
                "base_url": gateway.base_url if gateway.enabled else None,
                "base_alias": (
                    gateway.registration.base_model_name
                    if gateway.enabled and gateway.registration else None
                ),
                "fine_tuned_alias": (
                    gateway.registration.fine_tuned_model_name
                    if gateway.enabled and gateway.registration else None
                ),
            },
            "references": {"metadata": "metadata.json"},
        }
        _atomic_json(path, value)
        return path

    @staticmethod
    def _print_success(run_id, serving, gateway, manifest: Path) -> None:
        print("=" * 60)
        print("DEPLOYMENT SUCCESS")
        print("=" * 60)
        print(f"Run ID: {run_id}\nTraining: success")
        print(f"\nDirect vLLM:\nBase URL: {serving.base_url}")
        print(f"Base model: {serving.base_model_name}")
        print(f"Fine-tuned model: {serving.fine_tuned_model_name}")
        if gateway.enabled and gateway.registration:
            print(f"\nLiteLLM:\nGateway URL: {gateway.base_url}")
            print(f"Base model alias: {gateway.registration.base_model_name}")
            print(f"Fine-tuned model alias: {gateway.registration.fine_tuned_model_name}")
        print(f"\nServing container: {serving.container_name}")
        print(f"Deployment manifest: {manifest}")
        print("=" * 60)


def _layout_from_environment() -> RuntimeLayout:
    def required(name: str) -> Path:
        value = os.environ.get(name)
        if not value:
            raise ControllerError(f"Required controller environment variable {name} is missing")
        path = Path(value)
        if not path.is_absolute():
            raise ControllerError(f"{name} must be an absolute host-visible path")
        if any(character in value for character in (",", "\n", "\r", "\x00")):
            raise ControllerError(f"{name} contains a character unsafe for Docker bind mounts")
        return path
    return RuntimeLayout(
        required("PIPELINE_HOST_PROJECT_DIR"), required("PIPELINE_HOST_RUNS_DIR"),
        required("PIPELINE_HOST_HF_CACHE_DIR"), required("PIPELINE_HOST_STATE_DIR"),
        required("PIPELINE_HOST_SECRETS_DIR"),
        worker_uid=int(os.environ["PIPELINE_WORKER_UID"]),
        worker_gid=int(os.environ["PIPELINE_WORKER_GID"]),
        host_datasets=required("PIPELINE_HOST_DATASETS_DIR"),
    )


def main() -> int:
    try:
        controller = PipelineController()
        manifest = controller.deploy(
            execution_id=os.environ.get("PIPELINE_EXECUTION_ID") or uuid.uuid4().hex,
            config_relative=os.environ.get("PIPELINE_CONFIG_RELATIVE", "configs/config.yaml"),
            layout=_layout_from_environment(),
            training_image=os.environ["PIPELINE_TRAINING_IMAGE"],
            training_image_id=os.environ["PIPELINE_TRAINING_IMAGE_ID"],
            source_revision=os.environ.get("PIPELINE_SOURCE_REVISION", "unknown"),
            source_identity=os.environ.get("PIPELINE_SOURCE_IDENTITY", "unknown"),
            runtime_dns=_runtime_dns_from_environment(),
            selected_gpu=_selected_gpu_from_environment(),
        )
        return 0 if manifest else 1
    except Exception as error:
        print(f"DEPLOYMENT FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

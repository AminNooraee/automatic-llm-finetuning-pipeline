"""Strict public configuration for the qualified vLLM Docker path."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping
from urllib.parse import urlsplit


class ServingConfigError(ValueError):
    """Raised when automatic serving cannot be configured safely."""


@dataclass(frozen=True)
class VllmConfig:
    image: str
    gpu_devices: str
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int


@dataclass(frozen=True)
class HealthCheckConfig:
    enabled: bool
    timeout_seconds: float
    interval_seconds: float


@dataclass(frozen=True)
class ServingConfig:
    enabled: bool
    backend: str = "vllm"
    runtime: str = "docker"
    bind_host: str = "0.0.0.0"
    advertise_host: str = "localhost"
    port: int = 8101
    base_model_name: str = ""
    fine_tuned_model_name: str = ""
    container_name: str = ""
    restart_policy: str = "no"
    vllm: VllmConfig | None = None
    health_check: HealthCheckConfig | None = None

    @property
    def base_url(self) -> str:
        host = self.advertise_host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}/v1"


_TOP_FIELDS = {
    "enabled", "backend", "runtime", "bind_host", "advertise_host", "port",
    "base_model_name", "fine_tuned_model_name", "container_name", "restart_policy", "vllm", "health_check",
}
_VLLM_FIELDS = {
    "image", "gpu_devices", "gpu_memory_utilization", "max_model_len", "max_num_seqs"
}
_HEALTH_FIELDS = {"enabled", "timeout_seconds", "interval_seconds"}
_PUBLIC_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_GPU_DEVICES = re.compile(r"^[0-9]+(?:,[0-9]+)*$")


def safe_identifier(value: str, *, fallback: str, limit: int = 63) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-._")
    normalized = re.sub(r"[-._]{2,}", "-", normalized)
    return (normalized or fallback)[:limit].rstrip("-._") or fallback


def resolve_serving_config(
    raw: Any,
    *,
    run_id: str,
    base_model: str,
    training_method: str,
) -> ServingConfig:
    if raw is None:
        return ServingConfig(enabled=False)
    section = _mapping(raw, "serving")
    _unknown(section, _TOP_FIELDS, "serving")
    enabled = _boolean(section.get("enabled", False), "serving.enabled")
    if not enabled:
        return ServingConfig(enabled=False)
    if training_method != "lora":
        raise ServingConfigError(
            "Automatic serving is currently qualified only for training.method 'lora'; "
            f"training.method {training_method!r} is unsupported"
        )

    backend = _choice(section.get("backend", "vllm"), "serving.backend", {"vllm"})
    runtime = _choice(section.get("runtime", "docker"), "serving.runtime", {"docker"})
    bind_host = _host(section.get("bind_host", "0.0.0.0"), "serving.bind_host", wildcard=True)
    advertise_host = _host(
        section.get("advertise_host", "localhost"), "serving.advertise_host", wildcard=False
    )
    port = _integer(section.get("port", 8101), "serving.port", minimum=1, maximum=65535)

    trace = safe_identifier(run_id, fallback="run", limit=40)
    base_fallback = f"{safe_identifier(base_model.rsplit('/', 1)[-1], fallback='base', limit=72)}-{trace}"
    base_name = _name(section.get("base_model_name", "auto"), "serving.base_model_name", base_fallback)
    fine_name = _name(
        section.get("fine_tuned_model_name", "auto"),
        "serving.fine_tuned_model_name",
        f"{safe_identifier(base_name, fallback='model', limit=80)}-finetuned",
    )
    if base_name == fine_name:
        raise ServingConfigError("serving base and fine-tuned model names must be different")
    container = _container_name(
        section.get("container_name", "auto"), f"ft-serving-{trace}"
    )
    restart_policy = _choice(
        section.get("restart_policy", "no"),
        "serving.restart_policy",
        {"no", "unless-stopped", "on-failure"},
    )

    raw_vllm = _mapping(section.get("vllm", {}), "serving.vllm")
    _unknown(raw_vllm, _VLLM_FIELDS, "serving.vllm")
    image = _nonempty(raw_vllm.get("image", "vllm/vllm-openai:v0.11.0"), "serving.vllm.image")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", image):
        raise ServingConfigError("serving.vllm.image is not a valid Docker image reference")
    gpu_devices = _nonempty(raw_vllm.get("gpu_devices", "0"), "serving.vllm.gpu_devices")
    if not _GPU_DEVICES.fullmatch(gpu_devices):
        raise ServingConfigError(
            "serving.vllm.gpu_devices must be a comma-separated list of non-negative integers"
        )
    if len(set(gpu_devices.split(","))) != len(gpu_devices.split(",")):
        raise ServingConfigError("serving.vllm.gpu_devices must not contain duplicates")
    memory = _number(
        raw_vllm.get("gpu_memory_utilization", 0.15),
        "serving.vllm.gpu_memory_utilization",
    )
    if not 0 < memory <= 1:
        raise ServingConfigError("serving.vllm.gpu_memory_utilization must be greater than 0 and at most 1")
    vllm = VllmConfig(
        image=image,
        gpu_devices=gpu_devices,
        gpu_memory_utilization=memory,
        max_model_len=_integer(raw_vllm.get("max_model_len", 4096), "serving.vllm.max_model_len", minimum=1),
        max_num_seqs=_integer(raw_vllm.get("max_num_seqs", 2), "serving.vllm.max_num_seqs", minimum=1),
    )

    raw_health = _mapping(section.get("health_check", {}), "serving.health_check")
    _unknown(raw_health, _HEALTH_FIELDS, "serving.health_check")
    health = HealthCheckConfig(
        enabled=_boolean(raw_health.get("enabled", True), "serving.health_check.enabled"),
        timeout_seconds=_positive_number(
            raw_health.get("timeout_seconds", 180), "serving.health_check.timeout_seconds"
        ),
        interval_seconds=_positive_number(
            raw_health.get("interval_seconds", 2), "serving.health_check.interval_seconds"
        ),
    )
    if not health.enabled:
        raise ServingConfigError(
            "The qualified serving workflow requires health_check.enabled=true"
        )
    if health.interval_seconds > health.timeout_seconds:
        raise ServingConfigError(
            "serving.health_check.interval_seconds must not exceed timeout_seconds"
        )
    return ServingConfig(
        enabled=True,
        backend=backend,
        runtime=runtime,
        bind_host=bind_host,
        advertise_host=advertise_host,
        port=port,
        base_model_name=base_name,
        fine_tuned_model_name=fine_name,
        container_name=container,
        restart_policy=restart_policy,
        vllm=vllm,
        health_check=health,
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ServingConfigError(f"{label} must be a YAML mapping")
    return value


def _unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ServingConfigError(f"Unsupported {label} parameter(s): {', '.join(unknown)}")


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ServingConfigError(f"{label} must be true or false")
    return value


def _choice(value: Any, label: str, choices: set[str]) -> str:
    normalized = _nonempty(value, label).lower()
    if normalized not in choices:
        raise ServingConfigError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return normalized


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServingConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: Any, label: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ServingConfigError(f"{label} must be an integer of at least {minimum}")
    if maximum is not None and value > maximum:
        raise ServingConfigError(f"{label} must be at most {maximum}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ServingConfigError(f"{label} must be a finite number")
    return float(value)


def _positive_number(value: Any, label: str) -> float:
    number = _number(value, label)
    if number <= 0:
        raise ServingConfigError(f"{label} must be greater than 0")
    return number


def _name(value: Any, label: str, generated: str) -> str:
    selected = generated if value == "auto" else _nonempty(value, label)
    if not _PUBLIC_NAME.fullmatch(selected):
        raise ServingConfigError(
            f"{label} must start with an alphanumeric character and contain only letters, "
            "numbers, dot, underscore, and hyphen (maximum 128 characters)"
        )
    return selected


def _container_name(value: Any, generated: str) -> str:
    selected = generated if value == "auto" else _nonempty(value, "serving.container_name")
    if not _CONTAINER_NAME.fullmatch(selected):
        raise ServingConfigError(
            "serving.container_name must use Docker-safe letters, numbers, dot, underscore, "
            "or hyphen and be at most 63 characters"
        )
    return selected


def _host(value: Any, label: str, *, wildcard: bool) -> str:
    host = _nonempty(value, label)
    if any(character.isspace() for character in host) or any(item in host for item in ("/", "?", "#", "@")):
        raise ServingConfigError(f"{label} must be a hostname or IP address without a URL scheme or path")
    candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        parsed = urlsplit(f"//{host}")
        embedded_port = parsed.port
    except ValueError:
        raise ServingConfigError(f"{label} must be a valid hostname or IP address") from None
    if parsed.hostname is None or embedded_port is not None:
        raise ServingConfigError(f"{label} must not include a port")
    if candidate == "*":
        raise ServingConfigError(f"{label} must be a hostname or IP address")
    if not wildcard and candidate in {"0.0.0.0", "::", "*"}:
        raise ServingConfigError(f"{label} must be reachable and cannot be a wildcard address")
    return candidate

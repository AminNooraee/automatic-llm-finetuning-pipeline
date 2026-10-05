"""Deterministic resource estimation and serving-port admission."""

from __future__ import annotations

import argparse
import math
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .gateway.config import resolve_gateway_config
from .orchestration.config import resolve_orchestration_config
from .serving.config import ServingConfig, resolve_serving_config
from .training_config import TrainingConfig, resolve_training_config


class ResourcePreflightError(ValueError):
    """Raised when resource admission cannot be calculated safely."""


@dataclass(frozen=True)
class ResourcePolicy:
    enabled: bool
    model_parameters: int = 0
    activation_bytes_per_token: int = 0
    safety_margin_percent: int = 0


@dataclass(frozen=True)
class ResourceEstimate:
    required_training_mib: int
    serving_memory_utilization_bps: int
    subtotal_training_mib: int
    safety_margin_percent: int
    label: str = "estimate"


_POLICY_FIELDS = {
    "enabled",
    "model_parameter_estimate",
    "activation_bytes_per_token",
    "safety_margin_percent",
}


def resolve_resource_policy(raw: Any) -> ResourcePolicy:
    if raw is None:
        return ResourcePolicy(enabled=False)
    if not isinstance(raw, Mapping):
        raise ResourcePreflightError("resource_preflight must be a YAML mapping")
    unknown = sorted(str(key) for key in raw if key not in _POLICY_FIELDS)
    if unknown:
        raise ResourcePreflightError(
            f"Unsupported resource_preflight parameter(s): {', '.join(unknown)}"
        )
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ResourcePreflightError("resource_preflight.enabled must be true or false")
    if not enabled:
        if set(raw) != {"enabled"}:
            raise ResourcePreflightError(
                "Disabled resource_preflight must not contain estimation parameters"
            )
        return ResourcePolicy(enabled=False)
    return ResourcePolicy(
        enabled=True,
        model_parameters=_positive_integer(
            raw.get("model_parameter_estimate"),
            "resource_preflight.model_parameter_estimate",
        ),
        activation_bytes_per_token=_positive_integer(
            raw.get("activation_bytes_per_token"),
            "resource_preflight.activation_bytes_per_token",
        ),
        safety_margin_percent=_bounded_integer(
            raw.get("safety_margin_percent"),
            "resource_preflight.safety_margin_percent",
            minimum=10,
            maximum=100,
        ),
    )


def estimate_required_vram(
    policy: ResourcePolicy,
    training: TrainingConfig,
    serving: ServingConfig,
) -> ResourceEstimate:
    if not policy.enabled:
        raise ResourcePreflightError("resource_preflight is not enabled")
    if serving.vllm is None:
        raise ResourcePreflightError("resource preflight requires serving.vllm configuration")

    precision_bytes = 4 if training.precision == "fp32" else 2
    if training.method == "lora":
        # Frozen weights plus a conservative full-model-equivalent backward/workspace allowance.
        parameter_bytes = policy.model_parameters * (precision_bytes + 2)
    elif training.method == "full":
        # Weights, gradients, and conservative FP32 master/optimizer state allowance.
        parameter_bytes = policy.model_parameters * (precision_bytes * 2 + 12)
    else:  # Defensive: resolve_training_config currently prevents this branch.
        raise ResourcePreflightError(
            f"No VRAM estimation policy exists for training method {training.method!r}"
        )

    activation_bytes = (
        training.batch_size
        * training.cutoff_len
        * policy.activation_bytes_per_token
    )
    if training.gradient_checkpointing:
        activation_bytes = (activation_bytes + 1) // 2
    subtotal_bytes = parameter_bytes + activation_bytes
    required_bytes = (
        subtotal_bytes * (100 + policy.safety_margin_percent) + 99
    ) // 100
    mib = 1024 * 1024
    utilization_bps = math.ceil(serving.vllm.gpu_memory_utilization * 10_000)
    return ResourceEstimate(
        required_training_mib=math.ceil(required_bytes / mib),
        serving_memory_utilization_bps=utilization_bps,
        subtotal_training_mib=math.ceil(subtotal_bytes / mib),
        safety_margin_percent=policy.safety_margin_percent,
    )


def select_available_port(
    bind_host: str,
    start: int,
    end: int,
    *,
    checker: Callable[[str, int], bool] | None = None,
) -> int:
    if start > end:
        raise ResourcePreflightError("serving.port_range.start must not exceed end")
    available = checker or port_is_available
    for port in range(start, end + 1):
        if available(bind_host, port):
            return port
    raise ResourcePreflightError(
        f"No available serving port exists in configured range {start}-{end}; "
        "no existing service was modified"
    )


def port_is_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
        return True
    except OSError:
        return False


def load_and_estimate(
    config_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[ResourcePolicy, ResourceEstimate | None, ServingConfig | None]:
    raw = _load_config(config_path)
    policy = resolve_resource_policy(raw.get("resource_preflight"))
    if not policy.enabled:
        return policy, None, None

    model_name = _validate_pipeline_shape(raw)
    training = resolve_training_config(raw.get("training"))
    orchestration = resolve_orchestration_config(raw.get("orchestration"))
    if not orchestration.enabled:
        raise ResourcePreflightError(
            "resource_preflight requires orchestration.enabled=true"
        )
    serving_raw = raw.get("serving")
    selected_environment = dict(os.environ if environment is None else environment)
    selected_environment.setdefault(
        "PIPELINE_SELECTED_PORT", str(_auto_port_range(serving_raw)[0])
    )
    serving = resolve_serving_config(
        serving_raw,
        run_id="resource-preflight",
        base_model=model_name,
        training_method=training.method,
        environment=selected_environment,
    )
    if not serving.enabled:
        raise ResourcePreflightError(
            "resource_preflight requires serving.enabled=true"
        )
    resolve_gateway_config(
        raw.get("gateway"),
        serving=serving,
        run_id="resource-preflight",
        environment=selected_environment,
    )
    return policy, estimate_required_vram(policy, training, serving), serving


def select_configured_port(
    config_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    checker: Callable[[str, int], bool] | None = None,
) -> int | None:
    raw = _load_config(config_path)
    policy = resolve_resource_policy(raw.get("resource_preflight"))
    if not policy.enabled:
        return None
    serving_raw = raw.get("serving")
    start, end = _auto_port_range(serving_raw)
    model = raw.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("name"), str):
        raise ResourcePreflightError("model.name must be a non-empty string")
    training = resolve_training_config(raw.get("training"))
    selected_environment = dict(os.environ if environment is None else environment)
    selected_environment["PIPELINE_SELECTED_PORT"] = str(start)
    serving = resolve_serving_config(
        serving_raw,
        run_id="resource-preflight",
        base_model=model["name"],
        training_method=training.method,
        environment=selected_environment,
    )
    return select_available_port(serving.bind_host, start, end, checker=checker)


def _auto_port_range(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, Mapping):
        raise ResourcePreflightError("serving must be a YAML mapping")
    if raw.get("port", 8101) != "auto":
        raise ResourcePreflightError(
            "Enabled resource_preflight requires serving.port: auto"
        )
    value = raw.get("port_range")
    if not isinstance(value, Mapping):
        raise ResourcePreflightError(
            "serving.port_range must be configured when serving.port is auto"
        )
    unknown = sorted(str(key) for key in value if key not in {"start", "end"})
    if unknown:
        raise ResourcePreflightError(
            f"Unsupported serving.port_range parameter(s): {', '.join(unknown)}"
        )
    start = _bounded_integer(value.get("start"), "serving.port_range.start", 1, 65535)
    end = _bounded_integer(value.get("end"), "serving.port_range.end", 1, 65535)
    if start > end:
        raise ResourcePreflightError("serving.port_range.start must not exceed end")
    return start, end


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ResourcePreflightError(
            f"Could not read pipeline configuration: {type(error).__name__}"
        ) from None
    if not isinstance(raw, dict):
        raise ResourcePreflightError("Configuration must be a YAML mapping")
    return raw


def _validate_pipeline_shape(raw: Mapping[str, Any]) -> str:
    """Validate config fields needed by the real worker before resource admission."""
    for section_name in ("model", "dataset", "training"):
        if not isinstance(raw.get(section_name), Mapping):
            raise ResourcePreflightError(
                f"Training config needs a {section_name} section"
            )
    model_name = _required_text(raw["model"].get("name"), "model.name")
    dataset_name = _required_text(raw["dataset"].get("name"), "dataset.name")
    if "," in dataset_name:
        raise ResourcePreflightError("dataset.name must name a single dataset")
    _required_text(raw["dataset"].get("path"), "dataset.path")
    for optional_mapping in ("output", "observability"):
        if optional_mapping in raw and not isinstance(raw[optional_mapping], Mapping):
            raise ResourcePreflightError(
                f"Training config {optional_mapping} section must be a YAML mapping"
            )
    return model_name


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResourcePreflightError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_integer(value: Any, label: str) -> int:
    return _bounded_integer(value, label, 1, 10**15)


def _bounded_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ResourcePreflightError(
            f"{label} must be an integer between {minimum} and {maximum}"
        )
    return value


def _write_env(path: Path, values: Mapping[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            for name, value in values.items():
                file.write(f"{name}={value}\n")
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("estimate", "select-port"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "estimate":
            environment = dict(os.environ)
            secret_path = Path("/run/pipeline-secrets/litellm_api_key")
            if secret_path.is_file():
                secret = secret_path.read_text(encoding="utf-8").rstrip("\r\n")
                if secret:
                    environment["LITELLM_API_KEY"] = secret
            policy, estimate, _serving = load_and_estimate(
                args.config, environment=environment
            )
            if not policy.enabled:
                _write_env(args.output, {"PIPELINE_RESOURCE_PREFLIGHT_ENABLED": 0})
                return 0
            assert estimate is not None
            _write_env(
                args.output,
                {
                    "PIPELINE_RESOURCE_PREFLIGHT_ENABLED": 1,
                    "GPU_ESTIMATED_TRAINING_MIB": estimate.required_training_mib,
                    "GPU_SERVING_MEMORY_UTILIZATION_BPS": (
                        estimate.serving_memory_utilization_bps
                    ),
                },
            )
            print(
                "Estimated training VRAM: "
                f"{estimate.required_training_mib} MiB "
                f"({estimate.safety_margin_percent}% safety margin; estimate, not guarantee)"
            )
        else:
            selected = select_configured_port(args.config)
            if selected is None:
                raise ResourcePreflightError("resource_preflight is not enabled")
            _write_env(args.output, {"PIPELINE_SELECTED_PORT": selected})
            print(f"Selected serving port: {selected}")
        return 0
    except (ResourcePreflightError, ValueError) as error:
        print(f"Resource preflight failed: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

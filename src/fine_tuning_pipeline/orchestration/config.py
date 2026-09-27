"""Strict additive configuration for the containerized deployment controller."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


class OrchestrationConfigError(ValueError):
    pass


@dataclass(frozen=True)
class OrchestrationConfig:
    enabled: bool = False
    training_gpu_devices: str = "0"
    cleanup_on_failure: bool = True


_GPU_DEVICES = re.compile(r"^[0-9]+(?:,[0-9]+)*$")


def resolve_orchestration_config(raw: Any) -> OrchestrationConfig:
    if raw is None:
        return OrchestrationConfig()
    if not isinstance(raw, Mapping):
        raise OrchestrationConfigError("orchestration must be a YAML mapping")
    unknown = sorted(set(raw) - {"enabled", "training", "cleanup_on_failure"})
    if unknown:
        raise OrchestrationConfigError(
            "Unsupported orchestration parameter(s): " + ", ".join(unknown)
        )
    enabled = raw.get("enabled", False)
    cleanup = raw.get("cleanup_on_failure", True)
    if not isinstance(enabled, bool) or not isinstance(cleanup, bool):
        raise OrchestrationConfigError(
            "orchestration.enabled and cleanup_on_failure must be true or false"
        )
    training = raw.get("training", {})
    if not isinstance(training, Mapping):
        raise OrchestrationConfigError("orchestration.training must be a YAML mapping")
    training_unknown = sorted(set(training) - {"gpu_devices"})
    if training_unknown:
        raise OrchestrationConfigError(
            "Unsupported orchestration.training parameter(s): "
            + ", ".join(training_unknown)
        )
    gpu_devices = training.get("gpu_devices", "0")
    if not isinstance(gpu_devices, str) or not _GPU_DEVICES.fullmatch(gpu_devices):
        raise OrchestrationConfigError(
            "orchestration.training.gpu_devices must be comma-separated non-negative integers"
        )
    if len(set(gpu_devices.split(","))) != len(gpu_devices.split(",")):
        raise OrchestrationConfigError(
            "orchestration.training.gpu_devices must not contain duplicates"
        )
    return OrchestrationConfig(enabled, gpu_devices, cleanup)


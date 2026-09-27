"""Provider-neutral serving artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..security import redact_data
from .config import ServingConfig
from .contracts import EndpointHealth, ServingLaunchResult


def endpoint_manifest(config: ServingConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ready",
        "api": "openai-compatible",
        "base_url": config.base_url,
        "container_name": config.container_name,
        "restart_policy": config.restart_policy,
        "models": {
            "base": {"name": config.base_model_name},
            "fine_tuned": {"name": config.fine_tuned_model_name},
        },
    }


def health_artifact(health: EndpointHealth) -> dict[str, Any]:
    return {
        "status": health.status,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "discovered_models": list(health.discovered_models),
        "base_inference": health.base_inference,
        "fine_tuned_inference": health.fine_tuned_inference,
        "attempts": health.attempts,
    }


def serving_operation(
    *,
    run_id: str,
    container_name: str,
    container_id: str,
    status: str,
    error: str | None = None,
    execution_id: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "container_name": container_name,
        "container_id": container_id,
        "ownership_label": f"fine-tuning-pipeline.run-id={run_id}",
        "ownership_labels": {
            "fine-tuning-pipeline.managed": "true",
            "fine-tuning-pipeline.run-id": run_id,
            "fine-tuning-pipeline.execution-id": execution_id or run_id,
        },
        "status": status,
    }
    if error is not None:
        value["error"] = error
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(redact_data(value), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)

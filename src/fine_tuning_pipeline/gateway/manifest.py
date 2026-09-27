"""Provider-neutral gateway handoff and non-secret registration artifacts."""

from __future__ import annotations

from typing import Any, Iterable

from ..serving.contracts import EndpointHealth
from ..serving.manifest import health_artifact
from .config import GatewayConfig
from .contracts import RegistrationAttempt, RegistrationRecord


def gateway_manifest(config: GatewayConfig) -> dict[str, Any]:
    assert config.registration is not None
    return {
        "schema_version": 1,
        "status": "ready",
        "provider": config.provider,
        "api": "openai-compatible",
        "base_url": config.base_url,
        "models": {
            "base": {"name": config.registration.base_model_name},
            "fine_tuned": {"name": config.registration.fine_tuned_model_name},
        },
    }


def registration_artifact(
    records: Iterable[RegistrationRecord],
    *,
    run_id: str,
    status: str,
    attempts: Iterable[RegistrationAttempt] = (),
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provider": "litellm",
        "mode": "dynamic_db",
        "status": status,
        "ownership": {
            "created_by_run": run_id,
            "basis": "registration response for a request labeled with this pipeline run ID",
        },
        "registrations": [
            {
                "role": record.role,
                "alias": record.alias,
                "backend_model": record.backend_model,
                "api_base": record.api_base,
                "registration_id": record.registration_id,
                "status": record.status,
            }
            for record in records
        ],
        "attempts": [
            {
                "role": attempt.role,
                "alias": attempt.alias,
                "outcome": attempt.outcome,
                "registration_id": attempt.registration_id,
            }
            for attempt in attempts
        ],
        "automatic_rollback_performed": False,
    }


__all__ = ["EndpointHealth", "gateway_manifest", "health_artifact", "registration_artifact"]

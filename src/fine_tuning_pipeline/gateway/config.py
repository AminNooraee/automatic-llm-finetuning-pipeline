"""Strict configuration for externally managed LiteLLM registration."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from math import isfinite
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from ..security import SecretValue, resolve_environment_value, resolve_secret_reference
from ..serving.config import ServingConfig, safe_identifier


class GatewayConfigError(ValueError):
    """Raised when gateway registration cannot be configured safely."""


@dataclass(frozen=True)
class GatewayHealthConfig:
    enabled: bool
    verify_models: bool
    verify_inference: bool


@dataclass(frozen=True)
class GatewayRegistrationConfig:
    mode: str
    base_model_name: str
    fine_tuned_model_name: str


@dataclass(frozen=True)
class GatewayConfig:
    enabled: bool
    provider: str = "litellm"
    base_url: str = ""
    api_key: SecretValue | None = None
    registration: GatewayRegistrationConfig | None = None
    timeout_seconds: float = 60
    health_check: GatewayHealthConfig | None = None
    allow_local_backend: bool = False
    allow_insecure_http: bool = False


_TOP_FIELDS = {
    "enabled", "provider", "base_url", "api_key", "registration", "timeout_seconds",
    "health_check", "allow_local_backend", "allow_insecure_http",
}
_REGISTRATION_FIELDS = {"mode", "base_model_name", "fine_tuned_model_name"}
_HEALTH_FIELDS = {"enabled", "verify_models", "verify_inference"}


def resolve_gateway_config(
    raw: Any,
    *,
    serving: ServingConfig,
    run_id: str,
    environment: Mapping[str, str] | None = None,
) -> GatewayConfig:
    if raw is None:
        return GatewayConfig(enabled=False)
    section = _mapping(raw, "gateway")
    _unknown(section, _TOP_FIELDS, "gateway")
    enabled = _boolean(section.get("enabled", False), "gateway.enabled")
    if not enabled:
        return GatewayConfig(enabled=False)
    if not serving.enabled:
        raise GatewayConfigError("gateway.enabled=true requires serving.enabled=true")
    allow_local_backend = _boolean(
        section.get("allow_local_backend", False), "gateway.allow_local_backend"
    )
    allow_insecure_http = _boolean(
        section.get("allow_insecure_http", False), "gateway.allow_insecure_http"
    )
    if _is_loopback_host(serving.advertise_host) and not allow_local_backend:
        raise GatewayConfigError(
            "gateway.enabled=true requires an explicitly gateway-reachable "
            "serving.advertise_host; set gateway.allow_local_backend=true only "
            "when LiteLLM is intentionally co-located for local development"
        )
    provider = _choice(section.get("provider", "litellm"), "gateway.provider", {"litellm"})
    try:
        base_url = _url(
            resolve_environment_value(section.get("base_url"), "gateway.base_url", environment=environment),
            allow_insecure_http=allow_insecure_http,
        )
        api_key = resolve_secret_reference(
            section.get("api_key"), "gateway.api_key", environment=environment
        )
    except ValueError as error:
        raise GatewayConfigError(str(error)) from None

    raw_registration = _mapping(section.get("registration", {}), "gateway.registration")
    _unknown(raw_registration, _REGISTRATION_FIELDS, "gateway.registration")
    mode = _choice(
        raw_registration.get("mode", "dynamic_db"),
        "gateway.registration.mode",
        {"dynamic_db"},
    )
    trace = safe_identifier(run_id, fallback="run", limit=32)
    base_alias = _alias(
        raw_registration.get("base_model_name", "auto"),
        "gateway.registration.base_model_name",
        f"{safe_identifier(serving.base_model_name, fallback='base', limit=88)}-{trace}",
    )
    fine_alias = _alias(
        raw_registration.get("fine_tuned_model_name", "auto"),
        "gateway.registration.fine_tuned_model_name",
        f"{safe_identifier(serving.fine_tuned_model_name, fallback='finetuned', limit=88)}-{trace}",
    )
    if base_alias == fine_alias:
        raise GatewayConfigError("gateway base and fine-tuned model names must be different")

    timeout = _positive_number(section.get("timeout_seconds", 60), "gateway.timeout_seconds")
    raw_health = _mapping(section.get("health_check", {}), "gateway.health_check")
    _unknown(raw_health, _HEALTH_FIELDS, "gateway.health_check")
    health = GatewayHealthConfig(
        enabled=_boolean(raw_health.get("enabled", True), "gateway.health_check.enabled"),
        verify_models=_boolean(
            raw_health.get("verify_models", True), "gateway.health_check.verify_models"
        ),
        verify_inference=_boolean(
            raw_health.get("verify_inference", True), "gateway.health_check.verify_inference"
        ),
    )
    if not (health.enabled and health.verify_models and health.verify_inference):
        raise GatewayConfigError(
            "The qualified gateway workflow requires health_check.enabled, verify_models, "
            "and verify_inference to all be true"
        )
    return GatewayConfig(
        enabled=True,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        registration=GatewayRegistrationConfig(mode, base_alias, fine_alias),
        timeout_seconds=timeout,
        health_check=health,
        allow_local_backend=allow_local_backend,
        allow_insecure_http=allow_insecure_http,
    )


def _url(value: str, *, allow_insecure_http: bool) -> str:
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError:
        raise GatewayConfigError("gateway.base_url must contain a valid host and port") from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise GatewayConfigError("gateway.base_url must be an absolute http or https URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise GatewayConfigError(
            "gateway.base_url must not contain credentials, query parameters, or a fragment"
        )
    if (
        parsed.scheme == "http"
        and not _is_loopback_host(parsed.hostname)
        and not allow_insecure_http
    ):
        raise GatewayConfigError(
            "Remote LiteLLM HTTP endpoints are disabled by default because gateway "
            "credentials would be transmitted without TLS. Use HTTPS or explicitly set "
            "gateway.allow_insecure_http=true for a trusted private/local network."
        )
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]").rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GatewayConfigError(f"{label} must be a YAML mapping")
    return value


def _unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise GatewayConfigError(f"Unsupported {label} parameter(s): {', '.join(unknown)}")


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise GatewayConfigError(f"{label} must be true or false")
    return value


def _choice(value: Any, label: str, choices: set[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GatewayConfigError(f"{label} must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise GatewayConfigError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return normalized


def _alias(value: Any, label: str, generated: str) -> str:
    selected = generated if value == "auto" else value
    if not isinstance(selected, str) or not selected.strip():
        raise GatewayConfigError(f"{label} must be a non-empty string")
    selected = selected.strip()
    if safe_identifier(selected, fallback="invalid", limit=128) != selected.lower():
        raise GatewayConfigError(
            f"{label} must contain only lowercase letters, numbers, dot, underscore, and hyphen"
        )
    return selected


def _positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise GatewayConfigError(f"{label} must be a positive finite number")
    return float(value)

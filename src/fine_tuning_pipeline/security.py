"""Narrow secret resolution and redaction helpers for operational stages."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping


_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_SENSITIVE_NAMES = {
    "api_key", "apikey", "token", "access_token", "auth_token", "refresh_token",
    "id_token", "hf_token", "bearer_token", "secret", "secret_key", "client_secret", "master_key", "password",
    "passwd", "authorization", "credential", "credentials", "private_key",
}
_SENSITIVE_SUFFIXES = (
    "_api_key", "_access_token", "_auth_token", "_refresh_token", "_id_token",
    "_bearer_token", "_secret", "_secret_key", "_password", "_credential",
    "_credentials", "_private_key",
)


class SecretResolutionError(ValueError):
    """Raised when a required environment-backed secret is unavailable."""


class SecretValue:
    """Keep a secret's persisted reference separate from its runtime value."""

    __slots__ = ("reference", "__value")

    def __init__(self, reference: str, _value: str):
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "_SecretValue__value", _value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("SecretValue is immutable")

    def __deepcopy__(self, memo: dict[int, Any]) -> "SecretValue":
        return self

    def reveal(self) -> str:
        return self.__value

    def __repr__(self) -> str:
        return f"SecretValue(reference={self.reference!r}, value=<redacted>)"

    def __str__(self) -> str:
        return "<redacted>"


def resolve_secret_reference(
    value: Any,
    label: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> SecretValue:
    """Resolve exactly ``${NAME}``; literal credentials are intentionally rejected."""
    if not isinstance(value, str):
        raise SecretResolutionError(f"{label} must use an environment reference such as ${{NAME}}")
    match = _ENV_REFERENCE.fullmatch(value.strip())
    if match is None:
        raise SecretResolutionError(
            f"{label} must be an environment reference such as ${{LITELLM_API_KEY}}; "
            "literal credentials are not accepted"
        )
    source = os.environ if environment is None else environment
    resolved = source.get(match.group(1))
    if not isinstance(resolved, str) or not resolved:
        raise SecretResolutionError(
            f"Required environment variable {match.group(1)} referenced by {label} is not set"
        )
    return SecretValue(reference=value.strip(), _value=resolved)


def resolve_environment_value(
    value: Any,
    label: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve an optional environment reference for a non-secret string value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    stripped = value.strip()
    match = _ENV_REFERENCE.fullmatch(stripped)
    if match is None:
        return stripped
    source = os.environ if environment is None else environment
    resolved = source.get(match.group(1))
    if not isinstance(resolved, str) or not resolved:
        raise ValueError(
            f"Required environment variable {match.group(1)} referenced by {label} is not set"
        )
    return resolved


def redact_data(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON/YAML-safe copy with values under sensitive keys removed."""
    if key is not None and _is_sensitive_key(key):
        if isinstance(value, str) and _ENV_REFERENCE.fullmatch(value.strip()):
            return value
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(item_key): redact_data(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return [redact_data(item) for item in value]
    if isinstance(value, SecretValue):
        return value.reference
    return value


def _is_sensitive_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    return normalized in _SENSITIVE_NAMES or normalized.endswith(_SENSITIVE_SUFFIXES)


def redact_text(value: Any, secrets: tuple[str, ...] | list[str] = ()) -> str:
    """Remove known values and common authorization-shaped values from text."""
    text = str(value)
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)((?:api[_-]?key|token|secret|password)\s*[:=]\s*)['\"]?[^\s,'\";}]+",
        r"\1<redacted>",
        text,
    )
    return text

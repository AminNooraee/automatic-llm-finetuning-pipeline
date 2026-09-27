"""Interfaces and operational records for gateway providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class GatewayError(RuntimeError):
    """Base error for gateway preflight, registration, and verification."""


class GatewayConflictError(GatewayError):
    """Raised instead of overwriting an existing alias."""


class GatewayCapabilityError(GatewayError):
    """Raised when dynamic registration is not exposed by the target."""


class GatewayRegistrationOutcomeError(GatewayError):
    """A mutation failed and reconciliation classified its external state."""

    def __init__(self, message: str, *, alias: str, outcome: str):
        super().__init__(message)
        self.alias = alias
        self.outcome = outcome


@dataclass(frozen=True)
class RegistrationRecord:
    role: str
    alias: str
    backend_model: str
    api_base: str
    registration_id: str | None
    status: str = "created"


@dataclass(frozen=True)
class RegistrationAttempt:
    role: str
    alias: str
    outcome: str
    registration_id: str | None = None


class GatewayProvider(Protocol):
    def preflight(self) -> None: ...

    def aliases(self) -> set[str]: ...

    def management_aliases(self) -> set[str]: ...

    def register(
        self,
        *,
        role: str,
        alias: str,
        served_model: str,
        api_base: str,
        run_id: str,
    ) -> RegistrationRecord: ...

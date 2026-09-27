"""Interfaces and data contracts for serving backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .config import ServingConfig


class ServingError(RuntimeError):
    """Base error for actionable serving-stage failures."""


class ServingConflictError(ServingError):
    """Raised when launch would collide with an existing resource."""


class ServingVerificationError(ServingError):
    """Raised when the OpenAI-compatible endpoint is not actually usable."""


@dataclass(frozen=True)
class ServingLaunchRequest:
    run_id: str
    base_model: str
    resolved_revision: str | None
    adapter_path: Path
    lora_rank: int
    config: ServingConfig
    execution_id: str | None = None
    host_hf_cache_path: Path | None = None


@dataclass(frozen=True)
class ServingLaunchResult:
    container_id: str
    command: tuple[str, ...]
    image_id: str | None = None


@dataclass(frozen=True)
class EndpointHealth:
    status: str
    discovered_models: tuple[str, ...]
    base_inference: str
    fine_tuned_inference: str
    attempts: int


class ServingBackend(Protocol):
    def preflight(self, request: ServingLaunchRequest) -> None: ...

    def launch(self, request: ServingLaunchRequest) -> ServingLaunchResult: ...


class CommandRunner(Protocol):
    def run(self, command: Sequence[str]): ...

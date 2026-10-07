"""LiteLLM dynamic model registration through supported HTTP APIs only."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from ...http_client import JsonTransport, UrllibJsonTransport
from ...security import redact_text
from ..config import GatewayConfig
from ..contracts import (
    GatewayCapabilityError,
    GatewayError,
    GatewayRegistrationOutcomeError,
    RegistrationRecord,
)


BACKEND_API_KEY_PLACEHOLDER = "not-required"
RECONCILIATION_INTERVAL_SECONDS = 2.0
RECONCILIATION_GRACE_SECONDS = 30.0


class LiteLLMProvider:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        transport: JsonTransport | None = None,
        reconciliation_interval_seconds: float = RECONCILIATION_INTERVAL_SECONDS,
        reconciliation_grace_seconds: float = RECONCILIATION_GRACE_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if config.api_key is None:
            raise GatewayError("Internal gateway error: API credential is unavailable")
        self.config = config
        self.transport = transport or UrllibJsonTransport()
        if reconciliation_interval_seconds <= 0 or reconciliation_grace_seconds < 0:
            raise ValueError("LiteLLM reconciliation timing must be non-negative")
        self.reconciliation_interval_seconds = reconciliation_interval_seconds
        self.reconciliation_grace_seconds = reconciliation_grace_seconds
        self._sleep = sleep

    @property
    def _headers(self) -> dict[str, str]:
        assert self.config.api_key is not None
        return {"Authorization": f"Bearer {self.config.api_key.reveal()}"}

    @property
    def _management_root(self) -> str:
        return self.config.base_url[:-3] if self.config.base_url.endswith("/v1") else self.config.base_url

    def preflight(self) -> None:
        self.management_models()

    def management_models(self) -> tuple[Mapping[str, Any], ...]:
        response = self._request("GET", f"{self._management_root}/model/info")
        if response.status in {401, 403}:
            raise GatewayError("LiteLLM authentication failed during dynamic-registration preflight")
        if response.status in {404, 405}:
            raise GatewayCapabilityError(
                "LiteLLM does not expose the required dynamic model-management capability "
                "(/model/info and /model/new)"
            )
        if response.status >= 400:
            raise GatewayError(f"LiteLLM capability preflight failed with HTTP {response.status}")
        if not isinstance(response.data, Mapping) or not isinstance(response.data.get("data"), list):
            raise GatewayError("LiteLLM /model/info returned an invalid response")
        return tuple(item for item in response.data["data"] if isinstance(item, Mapping))

    def management_aliases(self) -> set[str]:
        return {
            item["model_name"]
            for item in self.management_models()
            if isinstance(item.get("model_name"), str)
        }

    def aliases(self) -> set[str]:
        response = self._request("GET", f"{self.config.base_url}/models")
        if response.status in {401, 403}:
            raise GatewayError("LiteLLM authentication failed while checking model aliases")
        if response.status != 200:
            raise GatewayError(f"LiteLLM alias query failed with HTTP {response.status}")
        if not isinstance(response.data, Mapping) or not isinstance(response.data.get("data"), list):
            raise GatewayError("LiteLLM /v1/models returned an invalid response")
        return {
            item["id"]
            for item in response.data["data"]
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }

    def register(
        self,
        *,
        role: str,
        alias: str,
        served_model: str,
        api_base: str,
        run_id: str,
    ) -> RegistrationRecord:
        backend_model = f"openai/{served_model}"
        payload = {
            "model_name": alias,
            "litellm_params": {
                "model": backend_model,
                "api_base": api_base,
                "api_key": BACKEND_API_KEY_PLACEHOLDER,
            },
            "model_info": {"pipeline_run_id": run_id, "pipeline_role": role},
        }
        try:
            response = self._request(
                "POST", f"{self._management_root}/model/new", payload=payload
            )
        except GatewayError as error:
            return self._reconcile_registration(
                role=role,
                alias=alias,
                backend_model=backend_model,
                api_base=api_base,
                run_id=run_id,
                cause=str(error),
                poll=True,
            )
        if response.status not in {200, 201}:
            return self._reconcile_registration(
                role=role,
                alias=alias,
                backend_model=backend_model,
                api_base=api_base,
                run_id=run_id,
                cause=f"HTTP {response.status}",
                poll=False,
            )
        registration_id = _registration_id(response.data)
        return RegistrationRecord(role, alias, backend_model, api_base, registration_id)

    def _reconcile_registration(
        self,
        *,
        role: str,
        alias: str,
        backend_model: str,
        api_base: str,
        run_id: str,
        cause: str,
        poll: bool,
    ) -> RegistrationRecord:
        attempts = self._poll_attempts() if poll else 1
        last_poll_available = False
        for attempt in range(attempts):
            try:
                models = self.management_models()
            except GatewayError:
                last_poll_available = False
            else:
                last_poll_available = True
                owned, foreign = _alias_ownership_matches(
                    models, alias=alias, run_id=run_id, role=role
                )
                if foreign:
                    raise GatewayRegistrationOutcomeError(
                        f"LiteLLM registration for alias {alias!r} conflicts with an existing "
                        f"registration after {cause}; no retry was attempted",
                        alias=alias,
                        outcome="conflict",
                    ) from None
                if len(owned) == 1:
                    return RegistrationRecord(
                        role,
                        alias,
                        backend_model,
                        api_base,
                        _registration_id(owned[0]),
                        status="created_after_ambiguous_response",
                    )
                if len(owned) > 1:
                    raise GatewayRegistrationOutcomeError(
                        f"LiteLLM registration outcome for alias {alias!r} is indeterminate "
                        f"after {cause}; multiple current-run records were found and no retry "
                        "was attempted",
                        alias=alias,
                        outcome="indeterminate",
                    ) from None
            if attempt + 1 < attempts:
                self._sleep(self.reconciliation_interval_seconds)

        if not last_poll_available:
            raise GatewayRegistrationOutcomeError(
                f"LiteLLM registration outcome for alias {alias!r} is indeterminate after {cause}; "
                "authoritative reconciliation was unavailable and no retry was attempted",
                alias=alias,
                outcome="indeterminate",
            ) from None
        raise GatewayRegistrationOutcomeError(
            f"LiteLLM registration for alias {alias!r} was absent after the bounded "
            f"reconciliation grace period following {cause}; no retry was attempted",
            alias=alias,
            outcome="absent",
        ) from None

    def _poll_attempts(self) -> int:
        return int(
            self.reconciliation_grace_seconds // self.reconciliation_interval_seconds
        ) + 1

    def delete_owned(self, record: RegistrationRecord, *, run_id: str) -> bool:
        """Delete one exact registration only after authoritative ownership proof."""
        if not record.registration_id:
            return False
        matches = []
        for item in self.management_models():
            info = item.get("model_info")
            if (
                item.get("model_name") == record.alias
                and _registration_id(item) == record.registration_id
                and isinstance(info, Mapping)
                and info.get("pipeline_run_id") == run_id
                and info.get("pipeline_role") == record.role
            ):
                matches.append(item)
        if len(matches) != 1:
            return False
        response = self._request(
            "POST",
            f"{self._management_root}/model/delete",
            payload={"id": record.registration_id},
        )
        if response.status not in {200, 201, 204}:
            raise GatewayError(
                f"LiteLLM owned-registration rollback failed with HTTP {response.status}"
            )
        remaining = [
            item for item in self.management_models()
            if _registration_id(item) == record.registration_id
        ]
        if remaining:
            raise GatewayError("LiteLLM owned-registration rollback could not be verified")
        return True

    def delete_owned_by_identity(
        self, *, role: str, alias: str, run_id: str
    ) -> tuple[str, ...]:
        """Find and delete late registrations carrying this run's exact ownership labels."""
        deleted: list[str] = []
        attempts = self._poll_attempts()
        for attempt in range(attempts):
            try:
                models = self.management_models()
            except GatewayError as error:
                raise GatewayError(
                    f"LiteLLM cleanup for alias {alias!r} could not verify authoritative state"
                ) from error
            owned, _foreign = _alias_ownership_matches(
                models, alias=alias, run_id=run_id, role=role
            )
            for item in owned:
                registration_id = _registration_id(item)
                if registration_id is None:
                    raise GatewayError(
                        f"LiteLLM cleanup for alias {alias!r} found an owned registration "
                        "without an authoritative registration ID"
                    )
                record = RegistrationRecord(role, alias, "", "", registration_id)
                if self.delete_owned(record, run_id=run_id):
                    deleted.append(registration_id)
            if attempt + 1 < attempts:
                self._sleep(self.reconciliation_interval_seconds)

        remaining_models = self.management_models()
        remaining, _foreign = _alias_ownership_matches(
            remaining_models, alias=alias, run_id=run_id, role=role
        )
        if remaining:
            raise GatewayError(
                f"LiteLLM cleanup for alias {alias!r} could not verify that owned "
                "registrations were removed"
            )
        return tuple(dict.fromkeys(deleted))

    def _request(self, method: str, url: str, *, payload: Mapping[str, Any] | None = None):
        try:
            return self.transport.request(
                method,
                url,
                headers=self._headers,
                payload=payload,
                timeout=self.config.timeout_seconds,
            )
        except Exception as error:
            secret = self.config.api_key.reveal() if self.config.api_key else ""
            raise GatewayError(
                f"LiteLLM connection failed: {redact_text(error, [secret])}"
            ) from None


def _registration_id(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("model_id", "id"):
        if isinstance(value.get(key), str):
            return value[key]
    model_info = value.get("model_info")
    if isinstance(model_info, Mapping) and isinstance(model_info.get("id"), str):
        return model_info["id"]
    return None


def _alias_ownership_matches(
    models: tuple[Mapping[str, Any], ...],
    *,
    alias: str,
    run_id: str,
    role: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    owned: list[Mapping[str, Any]] = []
    foreign: list[Mapping[str, Any]] = []
    for item in models:
        if item.get("model_name") != alias:
            continue
        info = item.get("model_info")
        if (
            isinstance(info, Mapping)
            and info.get("pipeline_run_id") == run_id
            and info.get("pipeline_role") == role
        ):
            owned.append(item)
        else:
            foreign.append(item)
    return owned, foreign

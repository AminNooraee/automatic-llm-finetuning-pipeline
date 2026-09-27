"""LiteLLM dynamic model registration through supported HTTP APIs only."""

from __future__ import annotations

from typing import Any, Mapping

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


class LiteLLMProvider:
    def __init__(self, config: GatewayConfig, *, transport: JsonTransport | None = None):
        if config.api_key is None:
            raise GatewayError("Internal gateway error: API credential is unavailable")
        self.config = config
        self.transport = transport or UrllibJsonTransport()

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
            )
        if response.status not in {200, 201}:
            return self._reconcile_registration(
                role=role,
                alias=alias,
                backend_model=backend_model,
                api_base=api_base,
                run_id=run_id,
                cause=f"HTTP {response.status}",
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
    ) -> RegistrationRecord:
        try:
            models = self.management_models()
        except GatewayError:
            raise GatewayRegistrationOutcomeError(
                f"LiteLLM registration outcome for alias {alias!r} is indeterminate after {cause}; "
                "authoritative reconciliation was unavailable and no retry was attempted",
                alias=alias,
                outcome="indeterminate",
            ) from None
        alias_matches = [item for item in models if item.get("model_name") == alias]
        matches = []
        for item in alias_matches:
            model_info = item.get("model_info")
            if (
                isinstance(model_info, Mapping)
                and model_info.get("pipeline_run_id") == run_id
                and model_info.get("pipeline_role") == role
            ):
                matches.append(item)
        if len(matches) == 1:
            return RegistrationRecord(
                role,
                alias,
                backend_model,
                api_base,
                _registration_id(matches[0]),
                status="created_after_ambiguous_response",
            )
        if alias_matches:
            raise GatewayRegistrationOutcomeError(
                f"LiteLLM registration for alias {alias!r} conflicts with an existing "
                f"registration after {cause}; no retry was attempted",
                alias=alias,
                outcome="conflict",
            ) from None
        if not matches:
            raise GatewayRegistrationOutcomeError(
                f"LiteLLM registration for alias {alias!r} was absent after {cause}; no retry was attempted",
                alias=alias,
                outcome="absent",
            ) from None
        raise GatewayRegistrationOutcomeError(
            f"LiteLLM registration outcome for alias {alias!r} is indeterminate after {cause}; "
            "multiple current-run records were found and no retry was attempted",
            alias=alias,
            outcome="indeterminate",
        ) from None

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

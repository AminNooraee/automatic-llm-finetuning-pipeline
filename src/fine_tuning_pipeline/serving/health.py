"""OpenAI-compatible model discovery and inference verification."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from ..http_client import JsonTransport, UrllibJsonTransport
from ..security import redact_text
from .contracts import EndpointHealth, ServingVerificationError


class OpenAIEndpointVerifier:
    def __init__(
        self,
        *,
        transport: JsonTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.transport = transport or UrllibJsonTransport()
        self.monotonic = monotonic
        self.sleep = sleep

    def verify(
        self,
        *,
        base_url: str,
        base_model: str,
        fine_tuned_model: str,
        timeout_seconds: float,
        interval_seconds: float,
        wait_for_readiness: bool = True,
        api_key: str | None = None,
    ) -> EndpointHealth:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        deadline = self.monotonic() + timeout_seconds
        attempts = 0
        last_error = "endpoint did not respond"
        discovered: tuple[str, ...] = ()
        while True:
            attempts += 1
            try:
                response = self.transport.request(
                    "GET", f"{base_url.rstrip('/')}/models", headers=headers, timeout=min(timeout_seconds, 30)
                )
                if response.status in {401, 403}:
                    raise ServingVerificationError("OpenAI-compatible endpoint authentication failed")
                if response.status == 200:
                    discovered = _model_ids(response.data)
                    missing = [name for name in (base_model, fine_tuned_model) if name not in discovered]
                    if not missing:
                        break
                    last_error = f"model discovery is missing alias(es): {', '.join(missing)}"
                else:
                    last_error = f"model discovery returned HTTP {response.status}"
            except ServingVerificationError:
                raise
            except Exception as error:
                last_error = redact_text(error, [api_key] if api_key else [])
            if not wait_for_readiness or self.monotonic() >= deadline:
                raise ServingVerificationError(
                    f"Endpoint readiness timed out after {attempts} attempt(s): {last_error}"
                )
            self.sleep(min(interval_seconds, max(0, deadline - self.monotonic())))

        self._verify_inference(base_url, base_model, headers, timeout_seconds, api_key)
        self._verify_inference(base_url, fine_tuned_model, headers, timeout_seconds, api_key)
        return EndpointHealth(
            status="ready",
            discovered_models=discovered,
            base_inference="passed",
            fine_tuned_inference="passed",
            attempts=attempts,
        )

    def _verify_inference(
        self,
        base_url: str,
        model: str,
        headers: Mapping[str, str],
        timeout: float,
        api_key: str | None,
    ) -> None:
        payload: Mapping[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 2,
            "temperature": 0,
            "stream": False,
        }
        try:
            response = self.transport.request(
                "POST",
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                payload=payload,
                timeout=timeout,
            )
        except Exception as error:
            raise ServingVerificationError(
                f"Inference verification failed for model {model!r}: "
                f"{redact_text(error, [api_key] if api_key else [])}"
            ) from None
        if response.status in {401, 403}:
            raise ServingVerificationError("OpenAI-compatible endpoint authentication failed")
        if response.status != 200 or not _has_completion(response.data):
            raise ServingVerificationError(
                f"Inference verification failed for model {model!r}: HTTP {response.status} "
                "or invalid Chat Completions response"
            )


def _model_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping) or not isinstance(value.get("data"), list):
        return ()
    return tuple(
        item["id"]
        for item in value["data"]
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    )


def _has_completion(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    choices = value.get("choices")
    return isinstance(choices, list) and bool(choices) and isinstance(choices[0], Mapping)

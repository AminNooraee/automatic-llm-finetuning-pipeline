"""Small injectable JSON HTTP transport built only on the standard library."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class JsonResponse:
    status: int
    data: Any


class JsonTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        payload: Mapping[str, Any] | None = None,
        timeout: float = 60,
    ) -> JsonResponse: ...


class UrllibJsonTransport:
    """Default transport; tests can supply an in-memory implementation."""

    def __init__(self, *, opener: Any | None = None):
        # Authentication headers added to urllib Requests are otherwise copied
        # to redirects, including redirects to another origin.
        self.opener = opener or urllib.request.build_opener(_NoRedirectHandler())

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        payload: Mapping[str, Any] | None = None,
        timeout: float = 60,
    ) -> JsonResponse:
        body = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
        request_headers = {"Accept": "application/json", **dict(headers or {})}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return JsonResponse(int(response.status), _decode_json(response.read()))
        except urllib.error.HTTPError as error:
            return JsonResponse(int(error.code), _decode_json(error.read()))


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _decode_json(body: bytes) -> Any:
    if not body:
        return None
    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"message": text[:2000]}

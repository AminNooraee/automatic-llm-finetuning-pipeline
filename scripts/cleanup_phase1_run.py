from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def fail(message: str) -> None:
    print(f"CLEANUP FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def request(method: str, url: str, api_key: str, payload=None):
    body = None
    headers = {"Authorization": f"Bearer {api_key}"}

    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            data = json.loads(raw) if raw else None
            return response.status, data
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            data = json.loads(raw) if raw else None
        except Exception:
            data = None
        return exc.code, data


def registration_id(item: dict) -> str | None:
    for key in ("model_id", "id"):
        if isinstance(item.get(key), str):
            return item[key]

    info = item.get("model_info")
    if isinstance(info, dict) and isinstance(info.get("id"), str):
        return info["id"]

    return None


def get_models(root: str, api_key: str) -> list[dict]:
    status, data = request("GET", f"{root}/model/info", api_key)

    if status != 200:
        fail(f"LiteLLM /model/info returned HTTP {status}")

    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        fail("LiteLLM /model/info returned invalid data")

    return [x for x in data["data"] if isinstance(x, dict)]


def cleanup_gateway(
    *,
    root: str,
    api_key: str,
    run_id: str,
    registration_path: Path,
) -> None:

    artifact = json.loads(registration_path.read_text())

    ownership = artifact.get("ownership", {})
    if ownership.get("created_by_run") != run_id:
        fail("registration artifact does not belong to requested run")

    registrations = artifact.get("registrations")
    if not isinstance(registrations, list) or not registrations:
        fail("registration artifact has no registrations")

    for record in registrations:
        role = record.get("role")
        alias = record.get("alias")
        rid = record.get("registration_id")

        if not all(isinstance(x, str) and x for x in (role, alias, rid)):
            fail("invalid registration record")

        models = get_models(root, api_key)

        matches = [m for m in models if registration_id(m) == rid]

        if not matches:
            print(f"LiteLLM registration already absent: {alias}")
            continue

        if len(matches) != 1:
            fail(f"registration ID {rid} is not unique")

        item = matches[0]
        info = item.get("model_info")

        if (
            item.get("model_name") != alias
            or not isinstance(info, dict)
            or info.get("pipeline_run_id") != run_id
            or info.get("pipeline_role") != role
        ):
            fail(f"ownership verification failed for alias {alias}")

        status, _ = request(
            "POST",
            f"{root}/model/delete",
            api_key,
            {"id": rid},
        )

        if status not in (200, 201, 204):
            fail(f"LiteLLM deletion failed for {alias}: HTTP {status}")

        remaining = [
            m for m in get_models(root, api_key)
            if registration_id(m) == rid
        ]

        if remaining:
            fail(f"LiteLLM deletion could not be verified for {alias}")

        print(f"Deleted LiteLLM registration: {alias}")


def cleanup_serving(
    *,
    run_id: str,
    operation_path: Path,
) -> None:

    operation = json.loads(operation_path.read_text())

    if operation.get("run_id") != run_id:
        fail("serving artifact does not belong to requested run")

    container_id = operation.get("container_id")
    container_name = operation.get("container_name")

    if not isinstance(container_id, str) or not container_id:
        fail("serving artifact has no container ID")

    result = subprocess.run(
        ["docker", "container", "inspect", container_id],
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print(f"Serving container already absent: {container_name}")
        return

    data = json.loads(result.stdout)

    if not isinstance(data, list) or len(data) != 1:
        fail("unexpected docker inspect response")

    item = data[0]

    if item.get("Id") != container_id:
        fail("serving container ID mismatch")

    labels = item.get("Config", {}).get("Labels", {})

    expected = operation.get("ownership_labels", {})

    if not isinstance(expected, dict):
        fail("serving artifact ownership labels are invalid")

    for key, value in expected.items():
        if labels.get(key) != value:
            fail(f"serving ownership verification failed for label {key}")

    if labels.get("fine-tuning-pipeline.managed") != "true":
        fail("container is not pipeline-managed")

    if labels.get("fine-tuning-pipeline.run-id") != run_id:
        fail("container run ownership mismatch")

    removed = subprocess.run(
        ["docker", "container", "rm", "--force", container_id],
        text=True,
        capture_output=True,
    )

    if removed.returncode != 0:
        fail(f"could not remove serving container: {removed.stderr.strip()}")

    print(f"Removed serving container: {container_name}")


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--serving-operation", type=Path, required=True)

    args = parser.parse_args()

    api_key = os.environ.get("LITELLM_API_KEY")
    base_url = os.environ.get("LITELLM_BASE_URL")

    if not api_key:
        fail("LITELLM_API_KEY is missing")

    if not base_url:
        fail("LITELLM_BASE_URL is missing")

    root = base_url.rstrip("/")

    if root.endswith("/v1"):
        root = root[:-3]

    cleanup_gateway(
        root=root,
        api_key=api_key,
        run_id=args.run_id,
        registration_path=args.registration,
    )

    cleanup_serving(
        run_id=args.run_id,
        operation_path=args.serving_operation,
    )

    print("CLEANUP SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

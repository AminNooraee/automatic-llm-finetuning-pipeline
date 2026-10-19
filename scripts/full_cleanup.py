#!/usr/bin/env python3
"""Explicit, ownership-verified cleanup for one Project 1 pipeline/run."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = "automatic-llm-finetuning-pipeline"
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def fail(message: str) -> None:
    raise SystemExit(f"FULL CLEANUP FAILED: {message}")


def valid_id(value: str | None, label: str) -> str | None:
    if value is not None and (not ID_RE.fullmatch(value) or value in {".", ".."}):
        fail(f"invalid {label}")
    return value


def json_file(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def child(root: Path, name: str) -> Path:
    valid_id(name, "resource ID")
    base = root.resolve(strict=True)
    target = base / name
    if target.is_symlink():
        fail(f"refusing symlink target: {target}")
    if target.resolve(strict=False).parent != base:
        fail(f"target escapes configured root: {target}")
    return target


def tree_size(path: Path) -> int:
    if not path.exists() or path.is_symlink():
        return 0
    total = 0
    try:
        for root, _, files in os.walk(path):
            for filename in files:
                filepath = Path(root) / filename
                try:
                    if filepath.is_file() and not filepath.is_symlink():
                        total += filepath.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def find_trusted_helper_image(docker: Docker) -> str | None:
    for identity in docker.ids(
        "image", "label=fine-tuning-pipeline.image-role=controller"
    ):
        item = docker.inspect("image", identity)
        labels = item.get("Config", {}).get("Labels", {}) if item else {}
        if labels.get("fine-tuning-pipeline.image-role") == "controller":
            return identity
    for identity in docker.ids(
        "image", "label=fine-tuning-pipeline.image-role"
    ):
        item = docker.inspect("image", identity)
        labels = item.get("Config", {}).get("Labels", {}) if item else {}
        if labels.get("fine-tuning-pipeline.image-role") in {
            "controller",
            "trainer",
            "job-preparer",
        }:
            return identity
    return None


def safe_rmtree(target: Path, docker: Docker) -> None:
    if not target.exists():
        return
    if target.is_symlink():
        fail(f"refusing symlink target: {target}")
    try:
        shutil.rmtree(target)
        return
    except PermissionError:
        pass
    except OSError as error:
        import errno

        if error.errno not in {errno.EACCES, errno.EPERM}:
            raise

    helper_image = find_trusted_helper_image(docker)
    if not helper_image:
        fail(
            f"cannot remove root-owned directory {target}: no trusted Project 1 helper image available"
        )

    resolved_target = target.resolve()
    if resolved_target in {
        Path("/"),
        Path("/home"),
        Path("/srv"),
        Path("/tmp"),
        Path(os.environ.get("TMPDIR", "/tmp")),
    }:
        fail(f"refusing to delete dangerous path: {resolved_target}")

    result = docker.run(
        "run",
        "--rm",
        "--network",
        "none",
        "--mount",
        f"type=bind,src={resolved_target},dst=/target",
        helper_image,
        "sh",
        "-c",
        "rm -rf /target/* /target/.[!.]* /target/..?* 2>/dev/null || true; find /target -mindepth 1 -delete 2>/dev/null || true",
    )
    if result.returncode != 0:
        fail(f"Docker helper deletion failed for {target}: {result.stderr.strip()}")

    try:
        target.rmdir()
    except (PermissionError, OSError) as error:
        if target.exists():
            fail(f"could not remove directory {target} after helper cleanup: {error}")


class Docker:
    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *args], text=True, capture_output=True, check=False
        )

    def ids(self, kind: str, *filters: str) -> list[str]:
        command = [kind, "ls", "-aq"] if kind == "container" else ["image", "ls", "-q"]
        for value in filters:
            command.extend(("--filter", value))
        result = self.run(*command)
        if result.returncode != 0:
            fail(f"Docker {kind} discovery failed: {result.stderr.strip()}")
        return list(dict.fromkeys(result.stdout.split()))

    def inspect(self, kind: str, identity: str) -> dict | None:
        result = self.run(kind, "inspect", identity)
        if result.returncode != 0:
            return None
        try:
            value = json.loads(result.stdout)
        except ValueError:
            fail(f"invalid Docker inspect response for {identity}")
        return value[0] if isinstance(value, list) and len(value) == 1 else None


def discover_targets(
    runs_root: Path,
    state_root: Path,
    *,
    run_id: str | None,
    execution_id: str | None,
    pipeline_id: str | None,
) -> tuple[set[str], set[str]]:
    runs = {run_id} if run_id else set()
    executions = {execution_id} if execution_id else set()
    claims_root = state_root / "claims"
    if claims_root.is_dir():
        for claim_path in claims_root.glob("*.json"):
            claim = json_file(claim_path)
            claimed_execution = claim.get("execution_id")
            if isinstance(claimed_execution, str) and (
                execution_id == claimed_execution
                or (pipeline_id and str(claim.get("pipeline_id")) == pipeline_id)
            ):
                executions.add(claimed_execution)
    execution_root = state_root / "executions"
    if execution_root.is_dir():
        for entry in execution_root.iterdir():
            if (
                not entry.is_dir()
                or entry.is_symlink()
                or not ID_RE.fullmatch(entry.name)
            ):
                continue
            claim = json_file(entry / "claim.json")
            if execution_id == entry.name or (
                pipeline_id and str(claim.get("pipeline_id")) == pipeline_id
            ):
                executions.add(entry.name)
                result = json_file(entry / "training_result.json")
                if isinstance(result.get("run_id"), str):
                    runs.add(result["run_id"])
    if runs_root.is_dir():
        for entry in runs_root.iterdir():
            if (
                not entry.is_dir()
                or entry.is_symlink()
                or not ID_RE.fullmatch(entry.name)
            ):
                continue
            owner = json_file(entry / "ownership.json")
            if (execution_id and owner.get("execution_id") == execution_id) or (
                pipeline_id and str(owner.get("pipeline_id")) == pipeline_id
            ):
                runs.add(entry.name)
                if isinstance(owner.get("execution_id"), str):
                    executions.add(owner["execution_id"])
    return runs, executions


def lite_root(value: str) -> str:
    root = value.rstrip("/")
    return root.removesuffix("/v1")


def api(
    method: str, url: str, key: str, payload: dict | None = None
) -> tuple[int, object]:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": f"Bearer {key}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        return error.code, None


def cleanup_litellm(run_ids: set[str], base_url: str, key: str) -> int:
    root = lite_root(base_url)
    status, response = api("GET", f"{root}/model/info", key)
    if (
        status != 200
        or not isinstance(response, dict)
        or not isinstance(response.get("data"), list)
    ):
        fail(f"LiteLLM ownership discovery failed with HTTP {status}")
    removed = 0
    for item in response["data"]:
        info = item.get("model_info") if isinstance(item, dict) else None
        if not isinstance(info, dict) or info.get("pipeline_run_id") not in run_ids:
            continue
        registration_id = item.get("model_id") or item.get("id") or info.get("id")
        if not isinstance(registration_id, str) or not registration_id:
            fail("owned LiteLLM record has no authoritative ID")
        delete_status, _ = api(
            "POST", f"{root}/model/delete", key, {"id": registration_id}
        )
        if delete_status not in {200, 201, 204, 404}:
            fail(f"LiteLLM deletion failed with HTTP {delete_status}")
        removed += delete_status != 404
    status, response = api("GET", f"{root}/model/info", key)
    remaining = (
        [
            item
            for item in response.get("data", [])
            if isinstance(item, dict)
            and isinstance(item.get("model_info"), dict)
            and item["model_info"].get("pipeline_run_id") in run_ids
        ]
        if status == 200 and isinstance(response, dict)
        else ["unverified"]
    )
    if remaining:
        fail("owned LiteLLM registrations remain or verification was unavailable")
    return removed


def matching_containers(
    docker: Docker, runs: set[str], executions: set[str], pipeline_id: str | None
) -> list[str]:
    matches: list[str] = []
    for identity in docker.ids("container", "label=fine-tuning-pipeline.managed=true"):
        item = docker.inspect("container", identity)
        labels = item.get("Config", {}).get("Labels", {}) if item else {}
        if (
            labels.get("fine-tuning-pipeline.run-id") in runs
            or labels.get("fine-tuning-pipeline.execution-id") in executions
            or (
                pipeline_id
                and labels.get("fine-tuning-pipeline.pipeline-id") == pipeline_id
            )
        ):
            matches.append(identity)
    return matches


def cleanup_containers(docker: Docker, identities: list[str]) -> int:
    removed = 0
    for identity in identities:
        item = docker.inspect("container", identity)
        if item is None:
            continue
        labels = item.get("Config", {}).get("Labels", {})
        if labels.get("fine-tuning-pipeline.managed") != "true":
            fail(f"container ownership changed: {identity}")
        result = docker.run("container", "rm", "--force", identity)
        if result.returncode != 0 and docker.inspect("container", identity) is not None:
            fail(
                f"could not remove owned container {identity}: {result.stderr.strip()}"
            )
        removed += result.returncode == 0
    return removed


def active_project_containers(docker: Docker) -> list[str]:
    result = docker.run(
        "container", "ls", "-q", "--filter", "label=fine-tuning-pipeline.managed=true"
    )
    if result.returncode != 0:
        fail("could not verify active Project 1 containers")
    return result.stdout.split()


def cleanup_images(docker: Docker) -> tuple[int, int]:
    if active_project_containers(docker):
        print("Project 1 images preserved: another active Project 1 container exists")
        return 0, 0
    image_ids = docker.ids("image", "label=fine-tuning-pipeline.image-role")
    before = 0
    removed = 0
    for identity in image_ids:
        item = docker.inspect("image", identity)
        labels = item.get("Config", {}).get("Labels", {}) if item else {}
        if labels.get("fine-tuning-pipeline.image-role") not in {
            "trainer",
            "controller",
            "job-preparer",
        }:
            continue
        before += int(item.get("Size", 0))
        result = docker.run("image", "rm", identity)
        if result.returncode == 0 or docker.inspect("image", identity) is None:
            removed += 1
    # Docker build cache has no reliable project provenance in the classic
    # builder. Deliberately do not issue a broad builder/system prune.
    return removed, before


def clear_dedicated_cache(cache_root: Path, docker: Docker, dedicated: bool) -> int:
    if not dedicated:
        print("HuggingFace cache preserved: not explicitly configured as dedicated")
        return 0
    if active_project_containers(docker):
        print("HuggingFace cache preserved: active Project 1 container exists")
        return 0
    root = cache_root.resolve(strict=True)
    if cache_root.is_symlink() or root in {Path("/"), Path("/home"), Path("/srv")}:
        fail("unsafe HuggingFace cache root")
    reclaimed = tree_size(root)
    for entry in root.iterdir():
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry)
    print(
        "Dedicated HuggingFace cache cleared; the next run will download models again"
    )
    return reclaimed


def disk_free(path: Path) -> int:
    return shutil.disk_usage(path).free


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--execution-id")
    parser.add_argument("--pipeline-id")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--hf-cache-root", type=Path, required=True)
    parser.add_argument("--hf-cache-dedicated", action="store_true")
    parser.add_argument(
        "--tmp-root", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp"))
    )
    args = parser.parse_args(argv)
    run_id = valid_id(args.run_id, "run ID")
    execution_id = valid_id(args.execution_id, "execution ID")
    pipeline_id = valid_id(args.pipeline_id, "pipeline ID")
    if not any((run_id, execution_id, pipeline_id)):
        fail("one of --run-id, --execution-id, or --pipeline-id is required")
    for root in (args.runs_root, args.state_root, args.hf_cache_root):
        if not root.is_dir() or root.is_symlink():
            fail(f"configured root is missing or unsafe: {root}")

    before = disk_free(args.runs_root)
    runs, executions = discover_targets(
        args.runs_root,
        args.state_root,
        run_id=run_id,
        execution_id=execution_id,
        pipeline_id=pipeline_id,
    )
    docker = Docker()
    if runs:
        base_url = os.environ.get("LITELLM_BASE_URL")
        key = os.environ.get("LITELLM_API_KEY")
        if not base_url or not key:
            fail(
                "LITELLM_BASE_URL and LITELLM_API_KEY are required to verify registration cleanup"
            )
        cleanup_litellm(runs, base_url, key)
    cleanup_containers(
        docker, matching_containers(docker, runs, executions, pipeline_id)
    )

    run_bytes = 0
    for identity in runs:
        target = child(args.runs_root, identity)
        run_bytes += tree_size(target)
        if target.exists():
            safe_rmtree(target, docker)
    state_entries = 0
    execution_root = args.state_root / "executions"
    if execution_root.is_dir():
        for identity in executions:
            target = child(execution_root, identity)
            if target.exists():
                safe_rmtree(target, docker)
                state_entries += 1
    claims_root = args.state_root / "claims"
    if claims_root.is_dir():
        for identity in executions:
            claim = claims_root / f"{identity}.json"
            if claim.is_file() and not claim.is_symlink():
                try:
                    claim.unlink()
                    state_entries += 1
                except PermissionError:
                    try:
                        claim.chmod(0o644)
                        claim.unlink()
                        state_entries += 1
                    except OSError:
                        pass
                except OSError:
                    pass

    # Only execution-qualified temp names are eligible; never sweep /tmp broadly.
    temporary = 0
    for identity in executions:
        for target in args.tmp_root.glob(f"automatic-llm-*{identity}*"):
            if target.is_symlink() or target.is_file():
                target.unlink(missing_ok=True)
            elif target.is_dir():
                safe_rmtree(target, docker)
            temporary += 1

    images, image_bytes = cleanup_images(docker)
    cache_bytes = clear_dedicated_cache(
        args.hf_cache_root, docker, args.hf_cache_dedicated
    )
    remaining = matching_containers(docker, runs, executions, pipeline_id)
    if remaining:
        fail(f"{len(remaining)} owned containers remain")
    after = disk_free(args.runs_root)
    print("\nFULL CLEANUP SUCCESS")
    print("Project 1:")
    print("  LiteLLM registrations remaining for run: 0")
    print("  serving/training containers remaining for run: 0")
    print(
        f"  run directories remaining: {sum(child(args.runs_root, x).exists() for x in runs)}"
    )
    print(f"  state entries removed: {state_entries}")
    print(f"  temporary files removed: {temporary}")
    print(f"  stale owned Docker images removed: {images}")
    print(
        "  attributable Docker build cache removed: 0 (provenance unavailable; safely preserved)"
    )
    print(f"  Docker image space eligible/reclaimed: {image_bytes} bytes")
    print(f"  run/cache disk space reclaimed: {run_bytes + cache_bytes} bytes")
    print(f"disk before cleanup (free): {before} bytes")
    print(f"disk after cleanup (free): {after} bytes")
    print(f"total filesystem space reclaimed: {max(0, after - before)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

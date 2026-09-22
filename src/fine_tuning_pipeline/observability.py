"""Best-effort reproducibility, provenance, metrics, and resource collection."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    """Hash the exact bytes of a run-owned artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:
            return None
    return None


def collect_environment() -> dict[str, Any]:
    """Collect only non-secret environment details; every probe is optional."""
    snapshot: dict[str, Any] = {
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "os": {"system": platform.system(), "release": platform.release()},
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "accelerate": _package_version("accelerate"),
            "llamafactory": _package_version("llamafactory", "llama-factory"),
            "datasets": _package_version("datasets"),
            "trl": _package_version("trl"),
            "safetensors": _package_version("safetensors"),
        },
        "torch_cuda_build": None,
        "cuda_available": None,
        "cuda_runtime_visible_to_torch": None,
        "visible_gpu_count": None,
        "visible_gpus": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "collection_warnings": [],
    }
    try:
        import torch

        snapshot["torch_cuda_build"] = getattr(torch.version, "cuda", None)
        available = bool(torch.cuda.is_available())
        snapshot["cuda_available"] = available
        count = int(torch.cuda.device_count()) if available else 0
        snapshot["visible_gpu_count"] = count
        snapshot["visible_gpus"] = [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(count)
        ]
        if available:
            try:
                cudart = torch.cuda.cudart()
                error_code, runtime_version = cudart.cudaRuntimeGetVersion()
                if error_code == 0:
                    snapshot["cuda_runtime_visible_to_torch"] = runtime_version
            except Exception:
                snapshot["cuda_runtime_visible_to_torch"] = None
    except Exception as error:
        snapshot["collection_warnings"].append(
            f"PyTorch runtime inspection unavailable: {_short_error(error)}"
        )
    return snapshot


def collect_container_provenance(
    configured: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect container execution and accept launcher-supplied image identity."""
    configured = configured or {}
    markers: list[str] = []
    try:
        if Path("/.dockerenv").exists():
            markers.append("/.dockerenv")
        cgroup = Path("/proc/1/cgroup")
        if cgroup.is_file():
            text = cgroup.read_text(encoding="utf-8", errors="replace").lower()
            if any(marker in text for marker in ("docker", "containerd", "kubepods", "podman")):
                markers.append("/proc/1/cgroup")
    except Exception:
        pass

    explicit_container = os.environ.get("PIPELINE_CONTAINERIZED")
    if explicit_container is not None:
        appears_containerized = explicit_container.strip().lower() in {"1", "true", "yes"}
        detection = "launcher_environment"
    else:
        appears_containerized = bool(markers)
        detection = "runtime_markers" if markers else "not_detected"

    return {
        "appears_containerized": appears_containerized,
        "detection": detection,
        "runtime_markers": markers,
        "runtime": os.environ.get("PIPELINE_CONTAINER_RUNTIME") or configured.get("runtime"),
        "image": os.environ.get("PIPELINE_CONTAINER_IMAGE") or configured.get("image"),
        "image_id": os.environ.get("PIPELINE_CONTAINER_IMAGE_ID") or configured.get("image_id"),
        "image_digest": os.environ.get("PIPELINE_CONTAINER_IMAGE_DIGEST")
        or configured.get("image_digest"),
        "container_id": os.environ.get("PIPELINE_CONTAINER_ID"),
    }


def collect_resource_baseline(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Describe safe process-oriented metrics and explicit unavailable values."""
    return {
        "visible_gpu_count": environment.get("visible_gpu_count"),
        "visible_gpus": environment.get("visible_gpus", []),
        "cuda_visible_devices": environment.get("cuda_visible_devices"),
        "process_cuda_memory": {
            "initial_allocated_bytes": None,
            "initial_reserved_bytes": None,
            "final_allocated_bytes": None,
            "final_reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "availability": (
                "unavailable: training runs in a child process and PyTorch CUDA "
                "allocator counters are process-local"
            ),
        },
        "training_wall_clock_seconds": None,
    }


def extract_training_metrics(model_dir: str | Path) -> dict[str, Any]:
    """Normalize final Trainer metrics without copying per-step log history."""
    output = Path(model_dir)
    files = ("train_results.json", "all_results.json", "trainer_state.json")
    loaded: dict[str, Mapping[str, Any]] = {}
    warnings: list[str] = []
    for name in files:
        path = output / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                loaded[name] = value
            else:
                warnings.append(f"{name} did not contain a JSON object")
        except Exception as error:
            warnings.append(f"Could not parse {name}: {_short_error(error)}")

    aliases = {
        "train_loss": ("train_loss",),
        "train_runtime": ("train_runtime",),
        "train_samples_per_second": ("train_samples_per_second",),
        "train_steps_per_second": ("train_steps_per_second",),
        "final_epoch": ("epoch",),
        "global_step": ("global_step",),
    }
    normalized: dict[str, Any] = {}
    # Result summaries win; trainer_state supplies final epoch/global step fallback.
    ordered = [loaded[name] for name in files if name in loaded]
    for target, source_keys in aliases.items():
        for document in ordered:
            value = next((document[key] for key in source_keys if key in document), None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized[target] = value
                break

    state = loaded.get("trainer_state.json")
    if state:
        for target in ("final_epoch", "global_step"):
            key = "epoch" if target == "final_epoch" else target
            value = state.get(key)
            if target not in normalized and isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized[target] = value

    return {
        **normalized,
        "source_artifacts": [name for name in files if name in loaded],
        "log_history_artifact": "trainer_state.json"
        if isinstance(state, Mapping) and isinstance(state.get("log_history"), list)
        else None,
        "collection_warnings": warnings,
    }


def _short_error(error: BaseException) -> str:
    return (" ".join(str(error).split()) or type(error).__name__)[:240]

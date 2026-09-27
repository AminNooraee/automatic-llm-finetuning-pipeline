"""Post-training validation for expected model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ArtifactValidationError(RuntimeError):
    """Raised when a training process exits without usable model artifacts."""


@dataclass(frozen=True)
class LoraAdapterMetadata:
    rank: int
    base_model_name_or_path: str
    revision: str | None


def _require_non_empty_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ArtifactValidationError(f"Missing {label}: {path}")
    if path.stat().st_size == 0:
        raise ArtifactValidationError(f"Empty {label}: {path}")
    return path


def _full_model_files(model_dir: Path) -> list[Path]:
    patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    )
    files: dict[Path, None] = {}
    for pattern in patterns:
        for path in model_dir.glob(pattern):
            if path.is_file() and path.stat().st_size > 0:
                files[path] = None
    return sorted(files)


def _normalize_model_reference(value: str) -> tuple[str, str | None]:
    """Normalize only unambiguous Hugging Face identifier spelling.

    Comparison remains case-sensitive. We trim whitespace/trailing separators,
    accept the canonical Hugging Face URL form, and recognize an optional
    ``@revision`` suffix without guessing relationships from local cache paths.
    """
    normalized = value.strip().replace("\\", "/").rstrip("/")
    prefix = "https://huggingface.co/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix):].rstrip("/")
    revision = None
    if "/tree/" in normalized:
        normalized, revision = normalized.split("/tree/", 1)
        normalized = normalized.rstrip("/")
        revision = revision.strip("/") or None
    elif "@" in normalized:
        candidate, suffix = normalized.rsplit("@", 1)
        if candidate and suffix:
            normalized, revision = candidate, suffix
    return normalized, revision


def read_lora_adapter_metadata(
    model_dir: str | Path,
    *,
    expected_base_model: str | None = None,
    expected_base_revisions: Sequence[str | None] = (),
) -> LoraAdapterMetadata:
    config_path = _require_non_empty_file(
        Path(model_dir).resolve() / "adapter_config.json", "LoRA adapter configuration"
    )
    try:
        value: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(
            f"Invalid LoRA adapter configuration JSON: {config_path}: {error}"
        ) from None
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(
            f"LoRA adapter configuration must be a JSON object: {config_path}"
        )
    rank = value.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ArtifactValidationError(
            "LoRA adapter configuration field 'r' must be a positive integer"
        )
    declared_base = value.get("base_model_name_or_path")
    if not isinstance(declared_base, str) or not declared_base.strip():
        raise ArtifactValidationError(
            "LoRA adapter configuration field 'base_model_name_or_path' must be a non-empty string"
        )
    declared_revision = value.get("revision")
    if declared_revision is not None and (
        not isinstance(declared_revision, str) or not declared_revision.strip()
    ):
        raise ArtifactValidationError(
            "LoRA adapter configuration field 'revision' must be a non-empty string when present"
        )
    normalized_declared, suffix_revision = _normalize_model_reference(declared_base)
    effective_revision = (
        declared_revision.strip() if isinstance(declared_revision, str) else suffix_revision
    )
    if expected_base_model is not None:
        normalized_expected, expected_suffix = _normalize_model_reference(expected_base_model)
        if normalized_declared != normalized_expected:
            raise ArtifactValidationError(
                "LoRA adapter base model does not match the training run: "
                f"declared {declared_base!r}, expected {expected_base_model!r}"
            )
        accepted_revisions = {
            item.strip() for item in (*expected_base_revisions, expected_suffix)
            if isinstance(item, str) and item.strip()
        }
        if effective_revision is not None and accepted_revisions and effective_revision not in accepted_revisions:
            raise ArtifactValidationError(
                "LoRA adapter revision does not match the training run: "
                f"declared {effective_revision!r}"
            )
    return LoraAdapterMetadata(rank, normalized_declared, effective_revision)


def validate_model_artifacts(
    model_dir: str | Path,
    method: str,
    *,
    expected_base_model: str | None = None,
    expected_base_revisions: Sequence[str | None] = (),
) -> list[Path]:
    """Return verified artifact paths or raise a precise validation error."""
    output = Path(model_dir).resolve()
    if not output.is_dir():
        raise ArtifactValidationError(f"Model output directory was not created: {output}")

    normalized_method = method.strip().lower() if isinstance(method, str) else ""
    if normalized_method == "lora":
        read_lora_adapter_metadata(
            output,
            expected_base_model=expected_base_model,
            expected_base_revisions=expected_base_revisions,
        )
        return [
            output / "adapter_config.json",
            _require_non_empty_file(
                output / "adapter_model.safetensors", "LoRA adapter weights"
            ),
        ]

    if normalized_method == "full":
        config_file = _require_non_empty_file(
            output / "config.json", "full-model configuration"
        )
        model_files = _full_model_files(output)
        if not model_files:
            raise ArtifactValidationError(
                "Full fine-tuning finished without model weights. Expected "
                f"model.safetensors, model-*.safetensors, pytorch_model.bin, or "
                f"pytorch_model-*.bin under {output}"
            )
        return [config_file, *model_files]

    raise ArtifactValidationError(
        f"Artifact validation is not implemented for training method {method!r}"
    )

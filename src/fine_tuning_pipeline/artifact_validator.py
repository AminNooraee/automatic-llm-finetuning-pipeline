"""Post-training validation for expected model artifacts."""

from __future__ import annotations

from pathlib import Path


class ArtifactValidationError(RuntimeError):
    """Raised when a training process exits without usable model artifacts."""


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


def validate_model_artifacts(model_dir: str | Path, method: str) -> list[Path]:
    """Return verified artifact paths or raise a precise validation error."""
    output = Path(model_dir).resolve()
    if not output.is_dir():
        raise ArtifactValidationError(f"Model output directory was not created: {output}")

    normalized_method = method.strip().lower() if isinstance(method, str) else ""
    if normalized_method == "lora":
        return [
            _require_non_empty_file(
                output / "adapter_config.json", "LoRA adapter configuration"
            ),
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

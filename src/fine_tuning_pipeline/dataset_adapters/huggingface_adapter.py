"""HuggingFace Hub/local builder source adapter with lazy imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .base import (
    AdapterContext,
    AdapterStage,
    BaseDatasetAdapter,
    DatasetValidationError,
    DetectionResult,
    OptionalDependencyError,
    resolve_source_path,
)


_LOCAL_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".parquet", ".pq", ".txt"}


def _looks_like_local_path(source: str, context: AdapterContext) -> bool:
    candidate = Path(source)
    if candidate.is_absolute() or source.startswith((".", "\\", "/")):
        return True
    if candidate.suffix.lower() in _LOCAL_SUFFIXES:
        return True
    return resolve_source_path(source, context).exists()


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise OptionalDependencyError(
            "HuggingFace dataset input requires the optional 'datasets' package. "
            "Install it only when HuggingFace sources are needed."
        ) from error
    return load_dataset(*args, **kwargs)


class HuggingFaceAdapter(BaseDatasetAdapter):
    name = "huggingface"
    aliases = ("hf", "huggingface_dataset")
    stage = AdapterStage.SOURCE
    priority = 0

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        if not isinstance(data, str) or not data.strip():
            return DetectionResult(0.0, False, ("source is not a dataset identifier",))
        if _looks_like_local_path(data, context):
            return DetectionResult(0.0, False, ("source looks like a local file path",))
        return DetectionResult(
            0.88,
            True,
            ("non-file string is a possible HuggingFace dataset identifier",),
        )

    def validate(self, data: Any, context: AdapterContext) -> None:
        if not isinstance(data, str) or not data.strip():
            raise DatasetValidationError(
                "huggingface: dataset path must be a non-empty dataset identifier"
            )
        load_kwargs = context.options.get("load_kwargs", {})
        if not isinstance(load_kwargs, Mapping):
            raise DatasetValidationError("dataset.load_kwargs must be a mapping")

    def convert(self, data: Any, context: AdapterContext) -> list[dict[str, Any]]:
        subset = context.options.get("subset")
        split = context.options.get("split", "train")
        load_kwargs = dict(context.options.get("load_kwargs", {}))
        if "split" in load_kwargs or "name" in load_kwargs:
            raise DatasetValidationError(
                "Put HuggingFace 'split' and 'subset' directly under dataset, "
                "not in dataset.load_kwargs"
            )
        try:
            loaded = _load_dataset(data, subset, split=split, **load_kwargs)
            rows = [dict(row) for row in loaded]
        except OptionalDependencyError:
            raise
        except Exception as error:
            raise DatasetValidationError(
                f"Unable to load HuggingFace dataset {data!r} "
                f"(subset={subset!r}, split={split!r}): {error}"
            ) from error
        return rows

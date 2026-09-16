"""Backward-compatible facade for the modular dataset adapter framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .dataset_adapters import DEFAULT_FRAMEWORK, DatasetConversionResult


SUPPORTED_FORMATS = {"auto", *DEFAULT_FRAMEWORK.schema_registry.supported_formats}
SUPPORTED_SOURCE_FORMATS = {"auto", *DEFAULT_FRAMEWORK.source_registry.supported_formats}


def load_raw_dataset(
    dataset_path: str | Path,
    source_format: str = "auto",
    *,
    base_dir: str | Path | None = None,
    options: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Load a supported source while preserving the legacy list return type."""
    rows, _ = DEFAULT_FRAMEWORK.load_rows(
        dataset_path,
        source_format=source_format,
        base_dir=base_dir,
        options=options,
    )
    return rows


def adapt_dataset(
    rows: list[Any],
    dataset_format: str = "auto",
    *,
    options: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Normalize in-memory records to LLaMA-Factory-compatible SFT rows."""
    return DEFAULT_FRAMEWORK.convert_rows(
        rows,
        dataset_format=dataset_format,
        options=options,
        require_task="sft",
    ).records


def adapt_dataset_source(
    source: str | Path,
    dataset_format: str = "auto",
    *,
    source_format: str = "auto",
    base_dir: str | Path | None = None,
    options: Mapping[str, Any] | None = None,
) -> DatasetConversionResult:
    """Load, detect, validate, and normalize one configured dataset source."""
    return DEFAULT_FRAMEWORK.convert_dataset(
        source,
        dataset_format=dataset_format,
        source_format=source_format,
        base_dir=base_dir,
        options=options,
        require_task="sft",
    )


def detect_dataset_format(rows: list[Any]):
    """Expose ranked confidence scores for diagnostics and tooling."""
    from .dataset_adapters import AdapterContext

    return DEFAULT_FRAMEWORK.schema_registry.score(rows, AdapterContext())

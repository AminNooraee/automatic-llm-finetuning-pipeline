"""Parquet dataset source adapter with a lazy optional dependency."""

from __future__ import annotations

from typing import Any

from .base import (
    AdapterContext,
    AdapterStage,
    BaseDatasetAdapter,
    DatasetValidationError,
    DetectionResult,
    OptionalDependencyError,
    resolve_source_path,
)


class ParquetAdapter(BaseDatasetAdapter):
    name = "parquet"
    stage = AdapterStage.SOURCE
    priority = 30

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        path = resolve_source_path(data, context)
        extension_match = path.suffix.lower() in {".parquet", ".pq"}
        if not path.is_file():
            return DetectionResult(
                0.75 if extension_match else 0.0,
                extension_match,
                ("Parquet file extension" if extension_match else "not a Parquet path",),
            )
        try:
            with path.open("rb") as file:
                magic = file.read(4)
        except OSError as error:
            return DetectionResult(0.0, False, (f"cannot inspect file: {error}",))
        if magic == b"PAR1":
            return DetectionResult(1.0, True, ("Parquet magic bytes",))
        return DetectionResult(
            0.40 if extension_match else 0.0,
            False,
            ("Parquet magic bytes are missing",),
        )

    def validate(self, data: Any, context: AdapterContext) -> None:
        path = resolve_source_path(data, context)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

    def convert(self, data: Any, context: AdapterContext) -> list[dict[str, Any]]:
        path = resolve_source_path(data, context)
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise OptionalDependencyError(
                "Parquet input requires the optional 'pyarrow' package. "
                "Install it only when Parquet support is needed."
            ) from error
        try:
            # Passing an owned stream avoids a lingering Windows file handle from
            # the dataset API's internal Parquet source discovery/cache.
            with path.open("rb") as stream:
                table = parquet.read_table(stream)
                rows = table.to_pylist()
            del table
            return rows
        except Exception as error:
            raise DatasetValidationError(
                f"Unable to read Parquet dataset {path}: {error}"
            ) from error

"""CSV dataset source adapter."""

from __future__ import annotations

import csv
from typing import Any

from .base import (
    AdapterContext,
    AdapterStage,
    BaseDatasetAdapter,
    DatasetValidationError,
    DetectionResult,
    resolve_source_path,
)


class CsvAdapter(BaseDatasetAdapter):
    name = "csv"
    stage = AdapterStage.SOURCE
    priority = 10

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        path = resolve_source_path(data, context)
        extension_match = path.suffix.lower() == ".csv"
        if not path.is_file():
            return DetectionResult(
                0.70 if extension_match else 0.0,
                extension_match,
                (".csv file extension" if extension_match else "not a CSV path",),
            )
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                sample = file.read(8192)
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
        except (OSError, UnicodeError, csv.Error) as error:
            return DetectionResult(
                0.45 if extension_match else 0.0,
                False,
                (f"CSV probe failed: {error}",),
            )
        if sample and (extension_match or has_header):
            score = 0.98 if extension_match else 0.82
            return DetectionResult(
                score,
                True,
                (f"delimited rows detected with {dialect.delimiter!r} separator",),
            )
        return DetectionResult(0.0, False, ("not recognized as CSV",))

    def validate(self, data: Any, context: AdapterContext) -> None:
        path = resolve_source_path(data, context)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

    def convert(self, data: Any, context: AdapterContext) -> list[dict[str, Any]]:
        path = resolve_source_path(data, context)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                sample = file.read(8192)
                file.seek(0)
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(file, dialect=dialect)
                if not reader.fieldnames or any(not field for field in reader.fieldnames):
                    raise DatasetValidationError(
                        f"CSV dataset {path} must have a non-empty header row"
                    )
                return [dict(row) for row in reader]
        except csv.Error as error:
            raise DatasetValidationError(f"Invalid CSV dataset {path}: {error}") from error

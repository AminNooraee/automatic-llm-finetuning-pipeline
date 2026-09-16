"""JSON Lines and NDJSON dataset source adapter."""

from __future__ import annotations

import json
from typing import Any

from .base import (
    AdapterContext,
    AdapterStage,
    BaseDatasetAdapter,
    DatasetValidationError,
    DetectionResult,
    resolve_source_path,
)


class JsonlAdapter(BaseDatasetAdapter):
    name = "jsonl"
    aliases = ("ndjson",)
    stage = AdapterStage.SOURCE
    priority = 20

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        path = resolve_source_path(data, context)
        extension_match = path.suffix.lower() in {".jsonl", ".ndjson"}
        if not path.is_file():
            return DetectionResult(
                0.70 if extension_match else 0.0,
                extension_match,
                ("JSONL/NDJSON file extension" if extension_match else "not a JSONL path",),
            )

        inspected = 0
        try:
            with path.open("r", encoding="utf-8-sig") as file:
                for line in file:
                    if not line.strip():
                        continue
                    inspected += 1
                    json.loads(line)
                    if inspected == 5:
                        break
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            return DetectionResult(
                0.35 if extension_match else 0.0,
                False,
                (f"line-oriented JSON probe failed: {error}",),
            )

        if inspected:
            score = 0.99 if extension_match else 0.90
            return DetectionResult(score, True, (f"{inspected} JSON line(s) parsed",))
        return DetectionResult(
            0.30 if extension_match else 0.0,
            False,
            ("file contains no JSON records",),
        )

    def validate(self, data: Any, context: AdapterContext) -> None:
        path = resolve_source_path(data, context)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

    def convert(self, data: Any, context: AdapterContext) -> list[Any]:
        path = resolve_source_path(data, context)
        rows: list[Any] = []
        with path.open("r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise DatasetValidationError(
                        f"Invalid JSONL dataset {path} at line {line_number}, "
                        f"column {error.colno}: {error.msg}"
                    ) from error
        return rows

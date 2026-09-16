"""JSON-array dataset source adapter."""

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


class JsonArrayAdapter(BaseDatasetAdapter):
    name = "json"
    aliases = ("json_array",)
    stage = AdapterStage.SOURCE
    priority = 20

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        path = resolve_source_path(data, context)
        extension_match = path.suffix.lower() == ".json"
        if not path.is_file():
            return DetectionResult(
                0.65 if extension_match else 0.0,
                extension_match,
                (".json file extension" if extension_match else "not a JSON path",),
            )
        try:
            with path.open("r", encoding="utf-8-sig") as file:
                first_character = next(
                    (character for character in iter(lambda: file.read(1), "") if not character.isspace()),
                    "",
                )
        except (OSError, UnicodeError) as error:
            return DetectionResult(0.0, False, (f"cannot inspect file: {error}",))

        if first_character == "[":
            score = 0.99 if extension_match else 0.90
            return DetectionResult(score, True, ("top-level JSON array marker",))
        if extension_match:
            return DetectionResult(
                0.45,
                False,
                (".json extension found, but content is not a JSON array",),
            )
        return DetectionResult(0.0, False, ("not a JSON array",))

    def validate(self, data: Any, context: AdapterContext) -> None:
        path = resolve_source_path(data, context)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

    def convert(self, data: Any, context: AdapterContext) -> list[Any]:
        path = resolve_source_path(data, context)
        try:
            with path.open("r", encoding="utf-8-sig") as file:
                rows = json.load(file)
        except json.JSONDecodeError as error:
            raise DatasetValidationError(
                f"Invalid JSON dataset {path} at line {error.lineno}, "
                f"column {error.colno}: {error.msg}"
            ) from error
        if not isinstance(rows, list):
            raise DatasetValidationError(
                f"JSON dataset {path} must contain a top-level array"
            )
        return rows

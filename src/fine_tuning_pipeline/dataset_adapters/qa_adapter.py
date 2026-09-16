"""Question/answer schema adapter, including the legacy prompt/response variant."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import (
    AdapterContext,
    AdapterStage,
    BaseDatasetAdapter,
    DatasetValidationError,
    DetectionResult,
    detected_fields,
    require_non_empty_rows,
)


class QAAdapter(BaseDatasetAdapter):
    name = "qa_json"
    aliases = ("qa", "question_answer")
    stage = AdapterStage.SCHEMA
    variants = (("question", "answer"), ("prompt", "response"))

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        fields = detected_fields(data)
        if not isinstance(data, (list, tuple)) or not data:
            return DetectionResult(0.0, False, ("dataset is empty",), fields)
        sample = data[:25]
        recognized = sum(
            isinstance(row, Mapping) and self._variant(row) is not None
            for row in sample
        )
        matched = recognized == len(sample)
        if matched:
            score = 0.98
        else:
            field_set = set(fields)
            partial = max(
                len(field_set.intersection(variant)) / len(variant)
                for variant in self.variants
            )
            score = round(0.6 * partial, 3)
        if "context" in fields and matched:
            score -= 0.12
        return DetectionResult(
            score,
            matched,
            (
                "rows contain question/answer or prompt/response fields"
                if matched
                else "some rows lack a supported Q&A field pair",
            ),
            fields,
        )

    def validate(self, data: Any, context: AdapterContext) -> None:
        rows = require_non_empty_rows(data, self.name)
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise DatasetValidationError(f"{self.name}: row {index} must be an object")
            variant = self._variant(row)
            if variant is None:
                raise DatasetValidationError(
                    f"{self.name}: row {index} needs question/answer or prompt/response "
                    f"fields; detected fields: {sorted(str(key) for key in row)}"
                )
            for field in variant:
                value = row[field]
                if not isinstance(value, str) or not value.strip():
                    raise DatasetValidationError(
                        f"{self.name}: row {index} needs a non-empty {field} string"
                    )
            context_value = row.get("context", "")
            if not isinstance(context_value, str):
                raise DatasetValidationError(
                    f"{self.name}: row {index} field 'context' must be a string"
                )

    def convert(self, data: Any, context: AdapterContext) -> list[dict[str, str]]:
        converted = []
        for row in data:
            variant = self._variant(row)
            if variant is None:  # validate() provides the user-facing error.
                raise DatasetValidationError(f"{self.name}: unsupported row structure")
            question_field, answer_field = variant
            converted.append(
                {
                    "instruction": row[question_field],
                    "input": row.get("context", ""),
                    "output": row[answer_field],
                }
            )
        return converted

    def _variant(self, row: Mapping[str, Any]) -> tuple[str, str] | None:
        return next(
            (variant for variant in self.variants if all(field in row for field in variant)),
            None,
        )

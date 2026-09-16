"""Core types and reusable behavior for dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class AdapterStage(str, Enum):
    SOURCE = "source"
    SCHEMA = "schema"


class DatasetAdapterError(ValueError):
    """Base error raised by the dataset adapter framework."""


class DatasetFormatDetectionError(DatasetAdapterError):
    """Raised when automatic detection is unsuccessful or ambiguous."""


class DatasetValidationError(DatasetAdapterError):
    """Raised when a selected adapter cannot validate its input."""


class OptionalDependencyError(DatasetAdapterError):
    """Raised when an optional source dependency is needed but unavailable."""


class UnsupportedDatasetTaskError(DatasetAdapterError):
    """Raised when a valid dataset targets a task the pipeline cannot train."""


@dataclass(frozen=True)
class DetectionResult:
    """An adapter's confidence and evidence for a detection decision."""

    confidence: float
    matched: bool
    reasons: tuple[str, ...] = ()
    detected_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1")


@dataclass(frozen=True)
class AdapterContext:
    """Configuration and path context shared with adapters."""

    base_dir: Path = field(default_factory=Path.cwd)
    source_name: str = "<memory>"
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionCandidate:
    adapter_name: str
    result: DetectionResult


@dataclass(frozen=True)
class DatasetConversionResult:
    records: list[dict[str, Any]]
    source_format: str
    dataset_format: str
    confidence: float
    task_type: str
    candidates: tuple[DetectionCandidate, ...] = ()


class BaseDatasetAdapter(ABC):
    """Common lifecycle for source and schema adapters."""

    name: str
    aliases: tuple[str, ...] = ()
    stage: AdapterStage
    priority: int = 0
    task_type: str = "sft"

    @abstractmethod
    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        """Return confidence and evidence without mutating the input."""

    @abstractmethod
    def validate(self, data: Any, context: AdapterContext) -> None:
        """Raise a detailed error when data is invalid for this adapter."""

    @abstractmethod
    def convert(self, data: Any, context: AdapterContext) -> Any:
        """Convert data to the next representation in the pipeline."""

    @property
    def format_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


def normalize_format_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatDetectionError("Dataset format must be a non-empty string")
    return value.strip().lower().replace("-", "_")


def resolve_source_path(data: Any, context: AdapterContext) -> Path:
    path = Path(data)
    if not path.is_absolute():
        path = context.base_dir / path
    return path.resolve()


def require_non_empty_rows(rows: Any, adapter_name: str) -> Sequence[Any]:
    if not isinstance(rows, (list, tuple)) or not rows:
        raise DatasetValidationError(
            f"{adapter_name}: dataset must contain at least one example"
        )
    return rows


def detected_fields(rows: Any, sample_size: int = 25) -> tuple[str, ...]:
    if not isinstance(rows, (list, tuple)):
        return ()
    fields: set[str] = set()
    for row in rows[:sample_size]:
        if isinstance(row, Mapping):
            fields.update(str(key) for key in row)
    return tuple(sorted(fields))


class FieldMappingAdapter(BaseDatasetAdapter):
    """Reusable schema adapter for formats defined by named string fields."""

    stage = AdapterStage.SCHEMA
    required_fields: tuple[str, ...] = ()
    output_mapping: Mapping[str, str] = {}
    output_defaults: Mapping[str, Any] = {}
    detection_confidence: float = 0.98
    confidence_penalty_fields: tuple[str, ...] = ()

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        fields = detected_fields(data)
        if (
            not isinstance(data, (list, tuple))
            or not data
            or not fields
            or not self.required_fields
        ):
            return DetectionResult(0.0, False, ("no recognizable object fields",), fields)

        field_set = set(fields)
        sample = data[:25]
        complete_rows = sum(
            isinstance(row, Mapping)
            and all(field in row for field in self.required_fields)
            for row in sample
        )
        if complete_rows != len(sample):
            present_fields = sum(
                sum(field in row for field in self.required_fields)
                if isinstance(row, Mapping)
                else 0
                for row in sample
            )
            total_fields = len(sample) * len(self.required_fields)
            score = round(0.6 * present_fields / total_fields, 3)
            reason = (
                f"required fields are complete in {complete_rows}/{len(sample)} "
                "sampled rows"
            )
            return DetectionResult(score, False, (reason,), fields)

        score = self.detection_confidence
        penalties = [field for field in self.confidence_penalty_fields if field in field_set]
        if penalties:
            score = max(0.0, score - 0.12)
        reasons = [f"required fields present: {list(self.required_fields)}"]
        if penalties:
            reasons.append(f"also contains fields preferred by a more specific format: {penalties}")
        return DetectionResult(round(score, 3), True, tuple(reasons), fields)

    def validate(self, data: Any, context: AdapterContext) -> None:
        rows = require_non_empty_rows(data, self.name)
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise DatasetValidationError(
                    f"{self.name}: row {index} must be an object, got {type(row).__name__}"
                )
            missing = [field for field in self.required_fields if field not in row]
            if missing:
                raise DatasetValidationError(
                    f"{self.name}: row {index} is missing required fields {missing}; "
                    f"detected fields: {sorted(str(key) for key in row)}"
                )
            for field in self.required_fields:
                value = row[field]
                if not isinstance(value, str) or not value.strip():
                    raise DatasetValidationError(
                        f"{self.name}: row {index} needs a non-empty {field} string"
                    )

            for output_field, source_field in self.output_mapping.items():
                if source_field in row and not isinstance(row[source_field], str):
                    raise DatasetValidationError(
                        f"{self.name}: row {index} field {source_field!r} must be a string"
                    )

    def convert(self, data: Any, context: AdapterContext) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for row in data:
            converted.append(
                {
                    output_field: row.get(source_field, self.output_defaults.get(output_field))
                    for output_field, source_field in self.output_mapping.items()
                }
            )
        return converted

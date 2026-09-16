"""Adapter registration, scoring, and dataset conversion orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .base import (
    AdapterContext,
    AdapterStage,
    BaseDatasetAdapter,
    DatasetConversionResult,
    DatasetFormatDetectionError,
    DatasetValidationError,
    DetectionCandidate,
    DetectionResult,
    UnsupportedDatasetTaskError,
    detected_fields,
    normalize_format_name,
)


class AdapterRegistry:
    """A stage-specific collection of independently implemented adapters."""

    def __init__(self, stage: AdapterStage, ambiguity_margin: float = 0.03):
        self.stage = stage
        self.ambiguity_margin = ambiguity_margin
        self._adapters: list[BaseDatasetAdapter] = []
        self._names: dict[str, BaseDatasetAdapter] = {}

    @property
    def adapters(self) -> tuple[BaseDatasetAdapter, ...]:
        return tuple(self._adapters)

    @property
    def supported_formats(self) -> tuple[str, ...]:
        return tuple(sorted(adapter.name for adapter in self._adapters))

    def register(self, adapter: BaseDatasetAdapter) -> None:
        if adapter.stage != self.stage:
            raise ValueError(
                f"Cannot register {adapter.name!r} {adapter.stage.value} adapter "
                f"in the {self.stage.value} registry"
            )
        normalized_names = [normalize_format_name(name) for name in adapter.format_names]
        duplicate = next((name for name in normalized_names if name in self._names), None)
        if duplicate is not None:
            raise ValueError(f"Adapter format name already registered: {duplicate}")
        self._adapters.append(adapter)
        for name in normalized_names:
            self._names[name] = adapter

    def get(self, name: str) -> BaseDatasetAdapter:
        normalized = normalize_format_name(name)
        try:
            return self._names[normalized]
        except KeyError as error:
            choices = ", ".join(self.supported_formats)
            raise DatasetFormatDetectionError(
                f"Unsupported {self.stage.value} format {name!r}. "
                f"Supported formats: auto, {choices}"
            ) from error

    def score(
        self, data: Any, context: AdapterContext
    ) -> tuple[DetectionCandidate, ...]:
        candidates = [
            DetectionCandidate(adapter.name, adapter.detect(data, context))
            for adapter in self._adapters
        ]
        candidates.sort(
            key=lambda candidate: (
                candidate.result.confidence,
                self.get(candidate.adapter_name).priority,
                candidate.adapter_name,
            ),
            reverse=True,
        )
        return tuple(candidates)

    def select(
        self,
        data: Any,
        requested_format: str,
        context: AdapterContext,
    ) -> tuple[BaseDatasetAdapter, DetectionResult, tuple[DetectionCandidate, ...]]:
        requested = normalize_format_name(requested_format or "auto")
        candidates = self.score(data, context)
        if requested != "auto":
            adapter = self.get(requested)
            result = next(
                candidate.result
                for candidate in candidates
                if candidate.adapter_name == adapter.name
            )
            return adapter, result, candidates

        matches = [candidate for candidate in candidates if candidate.result.matched]
        if not matches:
            raise DatasetFormatDetectionError(
                self._format_detection_error("Dataset format not recognized.", data, candidates)
            )

        best = matches[0]
        if len(matches) > 1:
            second = matches[1]
            if best.result.confidence - second.result.confidence <= self.ambiguity_margin:
                raise DatasetFormatDetectionError(
                    self._format_detection_error(
                        "Dataset format is ambiguous.", data, tuple(matches)
                    )
                )

        return self.get(best.adapter_name), best.result, candidates

    def _format_detection_error(
        self,
        heading: str,
        data: Any,
        candidates: Iterable[DetectionCandidate],
    ) -> str:
        fields = list(detected_fields(data))
        ranked = [candidate for candidate in candidates if candidate.result.confidence > 0]
        lines = [heading, f"Detected fields: {fields}", "", "Candidate formats:"]
        if ranked:
            for candidate in list(ranked)[:5]:
                reasons = "; ".join(candidate.result.reasons) or "no matching evidence"
                lines.append(
                    f"- {candidate.adapter_name}: confidence "
                    f"{candidate.result.confidence:.2f} ({reasons})"
                )
        else:
            lines.append("- none")
        config_key = "source_format" if self.stage == AdapterStage.SOURCE else "format"
        lines.extend(
            ["", f"Set dataset.{config_key} explicitly to override detection."]
        )
        return "\n".join(lines)


class DatasetAdapterFramework:
    """Run source and schema adapters without coupling either registry."""

    def __init__(
        self,
        source_registry: AdapterRegistry,
        schema_registry: AdapterRegistry,
    ):
        self.source_registry = source_registry
        self.schema_registry = schema_registry

    def load_rows(
        self,
        source: Any,
        *,
        source_format: str = "auto",
        base_dir: str | Path | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[list[Any], str]:
        context = AdapterContext(
            base_dir=Path(base_dir or Path.cwd()).resolve(),
            source_name=str(source),
            options=options or {},
        )
        adapter, _, _ = self.source_registry.select(source, source_format, context)
        adapter.validate(source, context)
        rows = adapter.convert(source, context)
        if not isinstance(rows, list) or not rows:
            raise DatasetValidationError(
                f"{adapter.name}: dataset must contain at least one example"
            )
        return rows, adapter.name

    def convert_rows(
        self,
        rows: list[Any],
        *,
        dataset_format: str = "auto",
        source_name: str = "<memory>",
        options: Mapping[str, Any] | None = None,
        require_task: str | None = "sft",
    ) -> DatasetConversionResult:
        context = AdapterContext(source_name=source_name, options=options or {})
        mapped_rows = self._apply_column_mapping(rows, context.options.get("columns"))
        adapter, detection, candidates = self.schema_registry.select(
            mapped_rows, dataset_format, context
        )
        adapter.validate(mapped_rows, context)
        converted = adapter.convert(mapped_rows, context)
        if require_task is not None and adapter.task_type != require_task:
            raise UnsupportedDatasetTaskError(
                f"Detected {adapter.name!r} dataset format, but the current pipeline "
                f"supports {require_task.upper()} training only. Preference/DPO training "
                "has not been enabled."
            )
        return DatasetConversionResult(
            records=converted,
            source_format="memory",
            dataset_format=adapter.name,
            confidence=detection.confidence,
            task_type=adapter.task_type,
            candidates=candidates,
        )

    def convert_dataset(
        self,
        source: Any,
        *,
        dataset_format: str = "auto",
        source_format: str = "auto",
        base_dir: str | Path | None = None,
        options: Mapping[str, Any] | None = None,
        require_task: str | None = "sft",
    ) -> DatasetConversionResult:
        rows, selected_source = self.load_rows(
            source,
            source_format=source_format,
            base_dir=base_dir,
            options=options,
        )
        result = self.convert_rows(
            rows,
            dataset_format=dataset_format,
            source_name=str(source),
            options=options,
            require_task=require_task,
        )
        return DatasetConversionResult(
            records=result.records,
            source_format=selected_source,
            dataset_format=result.dataset_format,
            confidence=result.confidence,
            task_type=result.task_type,
            candidates=result.candidates,
        )

    @staticmethod
    def _apply_column_mapping(
        rows: list[Any], mapping: Any
    ) -> list[Any]:
        if mapping is None:
            return rows
        if not isinstance(mapping, Mapping) or not mapping:
            raise DatasetValidationError("dataset.columns must be a non-empty mapping")
        for target, source in mapping.items():
            if (
                not isinstance(target, str)
                or not target.strip()
                or not isinstance(source, str)
                or not source.strip()
            ):
                raise DatasetValidationError(
                    "dataset.columns keys and values must be non-empty strings"
                )

        remapped: list[Any] = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise DatasetValidationError(
                    f"Column mapping requires object rows; row {index} is "
                    f"{type(row).__name__}"
                )
            updated = dict(row)
            for target, source in mapping.items():
                if source not in row:
                    raise DatasetValidationError(
                        f"dataset.columns maps {target!r} to missing source column "
                        f"{source!r} at row {index}"
                    )
                updated[target] = row[source]
            remapped.append(updated)
        return remapped

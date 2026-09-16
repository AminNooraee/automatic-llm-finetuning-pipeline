"""ChatML schema adapter and optional raw-text source adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .base import (
    AdapterContext,
    AdapterStage,
    BaseDatasetAdapter,
    DatasetValidationError,
    DetectionResult,
    detected_fields,
    resolve_source_path,
)
from .conversation import ConversationAdapter


CHATML_PATTERN = re.compile(
    r"<\|im_start\|>\s*(system|user|assistant)\s*\r?\n"
    r"(.*?)<\|im_end\|>",
    re.DOTALL,
)


def _chatml_text(row: Any) -> str | None:
    if isinstance(row, str):
        return row
    if isinstance(row, Mapping) and isinstance(row.get("text"), str):
        return row["text"]
    return None


class ChatMLTextSourceAdapter(BaseDatasetAdapter):
    name = "chatml_text"
    aliases = ("text",)
    stage = AdapterStage.SOURCE
    priority = 40

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        path = resolve_source_path(data, context)
        if not path.is_file():
            extension_match = path.suffix.lower() in {".txt", ".chatml"}
            return DetectionResult(
                0.60 if extension_match else 0.0,
                extension_match,
                ("ChatML text file extension" if extension_match else "not ChatML text",),
            )
        try:
            with path.open("r", encoding="utf-8-sig") as file:
                sample = file.read(16384)
        except (OSError, UnicodeError) as error:
            return DetectionResult(0.0, False, (f"cannot inspect file: {error}",))
        markers = "<|im_start|>" in sample and "<|im_end|>" in sample
        return DetectionResult(
            0.99 if markers else 0.0,
            markers,
            ("ChatML start/end markers" if markers else "ChatML markers not found",),
        )

    def validate(self, data: Any, context: AdapterContext) -> None:
        path = resolve_source_path(data, context)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

    def convert(self, data: Any, context: AdapterContext) -> list[str]:
        path = resolve_source_path(data, context)
        return [path.read_text(encoding="utf-8-sig")]


class ChatMLAdapter(ConversationAdapter):
    name = "chatml"
    priority = 20

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        fields = detected_fields(data)
        rows = data[:25] if isinstance(data, (list, tuple)) else []
        texts = [_chatml_text(row) for row in rows]
        matched = bool(texts) and all(
            text is not None
            and "<|im_start|>" in text
            and "<|im_end|>" in text
            for text in texts
        )
        return DetectionResult(
            1.0 if matched else 0.0,
            matched,
            ("ChatML start/end markers" if matched else "ChatML markers not found",),
            fields,
        )

    def extract_messages(self, row: Any, row_index: int) -> list[dict[str, str]]:
        text = _chatml_text(row)
        if text is None:
            raise DatasetValidationError(
                f"{self.name}: row {row_index} must be a ChatML string or have a text field"
            )
        matches = list(CHATML_PATTERN.finditer(text))
        remainder = CHATML_PATTERN.sub("", text).strip()
        if not matches or remainder:
            raise DatasetValidationError(
                f"{self.name}: row {row_index} contains malformed or unsupported ChatML"
            )
        return [
            {"role": match.group(1), "content": match.group(2).strip()}
            for match in matches
        ]

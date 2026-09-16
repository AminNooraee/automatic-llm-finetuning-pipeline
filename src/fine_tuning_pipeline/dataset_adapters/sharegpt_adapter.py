"""ShareGPT conversations schema adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import AdapterContext, DatasetValidationError, DetectionResult, detected_fields
from .conversation import ConversationAdapter


ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


class ShareGPTAdapter(ConversationAdapter):
    name = "sharegpt"
    priority = 20

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        fields = detected_fields(data)
        if "conversations" not in fields:
            return DetectionResult(
                0.0, False, ("missing conversations field",), fields
            )
        rows = data[:25] if isinstance(data, (list, tuple)) else []
        valid_structure = bool(rows) and all(
            isinstance(row, Mapping)
            and isinstance(row.get("conversations"), list)
            and bool(row["conversations"])
            and all(
                isinstance(message, Mapping)
                and "from" in message
                and "value" in message
                for message in row["conversations"]
            )
            for row in rows
        )
        return DetectionResult(
            1.0 if valid_structure else 0.55,
            valid_structure,
            (
                "conversations contain from/value objects"
                if valid_structure
                else "conversations field does not contain from/value objects"
            ,),
            fields,
        )

    def extract_messages(self, row: Any, row_index: int) -> list[dict[str, str]]:
        if not isinstance(row, Mapping) or "conversations" not in row:
            fields = sorted(str(key) for key in row) if isinstance(row, Mapping) else []
            raise DatasetValidationError(
                f"{self.name}: row {row_index} is missing required field "
                f"'conversations'; detected fields: {fields}"
            )
        conversations = row["conversations"]
        if not isinstance(conversations, list):
            raise DatasetValidationError(
                f"{self.name}: row {row_index} conversations must be a list"
            )
        messages: list[dict[str, str]] = []
        for turn_index, message in enumerate(conversations, start=1):
            if not isinstance(message, Mapping):
                raise DatasetValidationError(
                    f"{self.name}: row {row_index}, turn {turn_index} must be an object"
                )
            source_role = message.get("from")
            role = ROLE_MAP.get(source_role, source_role)
            messages.append({"role": role, "content": message.get("value")})
        return messages

"""OpenAI messages schema adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import AdapterContext, DatasetValidationError, DetectionResult, detected_fields
from .conversation import ConversationAdapter


class OpenAIChatAdapter(ConversationAdapter):
    name = "openai_chat"
    aliases = ("openai", "messages")
    priority = 20

    def detect(self, data: Any, context: AdapterContext) -> DetectionResult:
        fields = detected_fields(data)
        if "messages" not in fields:
            return DetectionResult(
                0.0, False, ("missing messages field",), fields
            )
        rows = data[:25] if isinstance(data, (list, tuple)) else []
        valid_structure = bool(rows) and all(
            isinstance(row, Mapping)
            and isinstance(row.get("messages"), list)
            and bool(row["messages"])
            and all(
                isinstance(message, Mapping)
                and "role" in message
                and "content" in message
                for message in row["messages"]
            )
            for row in rows
        )
        return DetectionResult(
            1.0 if valid_structure else 0.55,
            valid_structure,
            (
                "messages contain role/content objects"
                if valid_structure
                else "messages field does not contain role/content objects"
            ,),
            fields,
        )

    def extract_messages(self, row: Any, row_index: int) -> list[dict[str, str]]:
        if not isinstance(row, Mapping) or "messages" not in row:
            fields = sorted(str(key) for key in row) if isinstance(row, Mapping) else []
            raise DatasetValidationError(
                f"{self.name}: row {row_index} is missing required field 'messages'; "
                f"detected fields: {fields}"
            )
        messages = row["messages"]
        if not isinstance(messages, list):
            raise DatasetValidationError(
                f"{self.name}: row {row_index} messages must be a list"
            )
        return [dict(message) if isinstance(message, Mapping) else message for message in messages]

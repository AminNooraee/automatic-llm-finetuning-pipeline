"""Shared validation and SFT conversion for conversation-shaped schemas."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import (
    AdapterContext,
    AdapterStage,
    BaseDatasetAdapter,
    DatasetValidationError,
    require_non_empty_rows,
)


ALLOWED_ROLES = {"system", "user", "assistant"}


def validate_messages(messages: Any, adapter_name: str, row_index: int) -> None:
    if not isinstance(messages, list) or not messages:
        raise DatasetValidationError(
            f"{adapter_name}: row {row_index} must contain a non-empty conversation"
        )
    seen_user = False
    assistant_count = 0
    awaiting_assistant = False
    for turn_index, message in enumerate(messages, start=1):
        if not isinstance(message, Mapping):
            raise DatasetValidationError(
                f"{adapter_name}: row {row_index}, turn {turn_index} must be an object"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise DatasetValidationError(
                f"{adapter_name}: row {row_index}, turn {turn_index} has unsupported "
                f"role {role!r}; expected system, user, or assistant"
            )
        if not isinstance(content, str) or not content.strip():
            raise DatasetValidationError(
                f"{adapter_name}: row {row_index}, turn {turn_index} needs non-empty content"
            )
        if role == "system" and (seen_user or assistant_count):
            raise DatasetValidationError(
                f"{adapter_name}: row {row_index}, turn {turn_index} places a system "
                "message after the conversation started"
            )
        if role == "user":
            if awaiting_assistant:
                raise DatasetValidationError(
                    f"{adapter_name}: row {row_index}, turn {turn_index} has consecutive "
                    "user messages"
                )
            seen_user = True
            awaiting_assistant = True
        elif role == "assistant":
            if not seen_user or not awaiting_assistant:
                raise DatasetValidationError(
                    f"{adapter_name}: row {row_index}, turn {turn_index} has an "
                    "assistant response without a preceding user message"
                )
            assistant_count += 1
            awaiting_assistant = False
    if assistant_count == 0:
        raise DatasetValidationError(
            f"{adapter_name}: row {row_index} must contain at least one assistant response"
        )
    if awaiting_assistant:
        raise DatasetValidationError(
            f"{adapter_name}: row {row_index} ends with a user message without a response"
        )


def messages_to_sft(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    history: list[dict[str, str]] = []
    current_user: dict[str, str] | None = None
    current_user_history_index = 0
    for message in messages:
        role = message["role"]
        if role == "user":
            current_user = message
            current_user_history_index = len(history)
        elif role == "assistant" and current_user is not None:
            context = "\n".join(
                f"{prior['role'].capitalize()}: {prior['content']}"
                for prior in history[:current_user_history_index]
            )
            rows.append(
                {
                    "instruction": current_user["content"],
                    "input": context,
                    "output": message["content"],
                }
            )
            current_user = None
        history.append(message)
    return rows


class ConversationAdapter(BaseDatasetAdapter):
    """Base class sharing conversation validation and turn expansion."""

    stage = AdapterStage.SCHEMA

    def extract_messages(self, row: Any, row_index: int) -> list[dict[str, str]]:
        raise NotImplementedError

    def validate(self, data: Any, context: AdapterContext) -> None:
        rows = require_non_empty_rows(data, self.name)
        for row_index, row in enumerate(rows, start=1):
            messages = self.extract_messages(row, row_index)
            validate_messages(messages, self.name, row_index)

    def convert(self, data: Any, context: AdapterContext) -> list[dict[str, str]]:
        converted: list[dict[str, str]] = []
        for row_index, row in enumerate(data, start=1):
            converted.extend(messages_to_sft(self.extract_messages(row, row_index)))
        return converted

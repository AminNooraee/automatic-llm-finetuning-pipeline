"""Prompt/completion schema adapter."""

from .base import FieldMappingAdapter


class PromptCompletionAdapter(FieldMappingAdapter):
    name = "prompt_completion"
    aliases = ("prompt-completion",)
    required_fields = ("prompt", "completion")
    output_mapping = {
        "instruction": "prompt",
        "input": "input",
        "output": "completion",
    }
    output_defaults = {"input": ""}
    detection_confidence = 0.98

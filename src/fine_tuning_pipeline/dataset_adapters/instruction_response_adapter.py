"""Instruction/response schema adapter."""

from .base import FieldMappingAdapter


class InstructionResponseAdapter(FieldMappingAdapter):
    name = "instruction_response"
    aliases = ("instruction-response",)
    required_fields = ("instruction", "response")
    output_mapping = {
        "instruction": "instruction",
        "input": "input",
        "output": "response",
    }
    output_defaults = {"input": ""}
    detection_confidence = 0.98

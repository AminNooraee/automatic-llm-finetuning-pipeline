"""Alpaca instruction/input/output schema adapter."""

from .base import FieldMappingAdapter


class AlpacaAdapter(FieldMappingAdapter):
    name = "alpaca"
    required_fields = ("instruction", "output")
    output_mapping = {
        "instruction": "instruction",
        "input": "input",
        "output": "output",
    }
    output_defaults = {"input": ""}
    detection_confidence = 0.99

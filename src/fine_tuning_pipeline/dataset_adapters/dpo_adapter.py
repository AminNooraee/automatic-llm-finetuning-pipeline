"""DPO preference schema adapter; training integration is intentionally absent."""

from .base import FieldMappingAdapter


class DPOAdapter(FieldMappingAdapter):
    name = "dpo"
    aliases = ("preference",)
    task_type = "preference"
    required_fields = ("prompt", "chosen", "rejected")
    output_mapping = {
        "prompt": "prompt",
        "chosen": "chosen",
        "rejected": "rejected",
    }
    detection_confidence = 1.0

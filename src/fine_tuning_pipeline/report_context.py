"""Sanitized, provider-neutral report context for downstream benchmark reporting.

This artifact carries only user-authored, non-secret training facts so that a
downstream benchmark report can attribute its results to the exact training that
produced the model. It deliberately excludes endpoints, aliases, credentials,
environment data, and infrastructure details: those stay in the serving and
gateway manifests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .training_config import TrainingConfig


def build_report_context(
    *,
    model_name: str,
    training_config: TrainingConfig,
    training_duration_seconds: float | None,
) -> dict[str, Any]:
    """Build a fixed-shape, non-secret report context from training inputs.

    Values are copied by type from validated ``TrainingConfig`` and the
    user-supplied base model name. No endpoints, aliases, secrets, or
    environment facts are included.
    """

    duration = _clean_duration(training_duration_seconds)
    training: dict[str, Any] = {
        "model_name": str(model_name),
        "epochs": float(training_config.epochs),
        "learning_rate": float(training_config.learning_rate),
        "batch_size": int(training_config.batch_size),
        "cutoff_len": int(training_config.cutoff_len),
    }
    if training_config.lora is not None:
        training["lora"] = {
            "rank": int(training_config.lora.rank),
            "alpha": int(training_config.lora.alpha),
            "dropout": float(training_config.lora.dropout),
        }
    return {
        "schema_version": 1,
        "training_duration_seconds": duration,
        "training": training,
    }


def write_report_context(
    run_root: Path, context: Mapping[str, Any]
) -> Path:
    """Persist the report context next to the run's other artifacts."""

    destination = Path(run_root) / "report_context.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(context), indent=2, ensure_ascii=False, sort_keys=True)
    destination.write_text(payload + "\n", encoding="utf-8")
    return destination


def _clean_duration(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number

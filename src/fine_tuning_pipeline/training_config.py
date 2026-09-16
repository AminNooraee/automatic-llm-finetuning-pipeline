"""Validation and LLaMA-Factory translation for training configuration."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


class TrainingConfigError(ValueError):
    """Raised when the user-provided training configuration is invalid."""


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float


@dataclass(frozen=True)
class _MethodSpec:
    finetuning_type: str
    uses_lora: bool


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    finetuning_type: str
    epochs: float
    learning_rate: float
    batch_size: int
    precision: str
    gradient_checkpointing: bool
    cutoff_len: int
    save_steps: int
    logging_steps: int
    lora: LoraConfig | None = None

    def to_llamafactory_args(self) -> dict[str, Any]:
        """Return only validated arguments understood by LLaMA-Factory."""
        args: dict[str, Any] = {
            "finetuning_type": self.finetuning_type,
            "num_train_epochs": self.epochs,
            "per_device_train_batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "fp16": self.precision == "fp16",
            "bf16": self.precision == "bf16",
            "gradient_checkpointing": self.gradient_checkpointing,
            "cutoff_len": self.cutoff_len,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
        }
        if self.lora is not None:
            args.update(
                {
                    "lora_rank": self.lora.rank,
                    "lora_alpha": self.lora.alpha,
                    "lora_dropout": self.lora.dropout,
                }
            )
        return args


_COMMON_FIELDS = {
    "method",
    "epochs",
    "learning_rate",
    "batch_size",
    "precision",
    "gradient_checkpointing",
    "cutoff_len",
    "save_steps",
    "logging_steps",
    "lora",
}
_REQUIRED_FIELDS = _COMMON_FIELDS - {"lora"}
_LORA_FIELDS = {"rank", "alpha", "dropout"}
_METHOD_SPECS = {
    "lora": _MethodSpec(finetuning_type="lora", uses_lora=True),
    "full": _MethodSpec(finetuning_type="full", uses_lora=False),
}
_SUPPORTED_PRECISIONS = {"fp32", "fp16", "bf16"}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingConfigError(f"{label} must be a YAML mapping")
    return value


def _reject_unknown_fields(values: Mapping[str, Any], allowed: set[str], label: str) -> None:
    invalid_keys = [repr(key) for key in values if not isinstance(key, str)]
    if invalid_keys:
        raise TrainingConfigError(
            f"Unsupported {label} parameter key(s): {', '.join(invalid_keys)}. "
            "Parameter names must be strings"
        )
    unknown = sorted(set(values) - allowed)
    if unknown:
        supported = ", ".join(sorted(allowed))
        raise TrainingConfigError(
            f"Unsupported {label} parameter(s): {', '.join(unknown)}. "
            f"Supported parameters: {supported}"
        )


def _require_fields(values: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(values))
    if missing:
        raise TrainingConfigError(
            f"Missing required {label} parameter(s): {', '.join(missing)}"
        )


def _require_choice(value: Any, label: str, choices: set[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError(f"{label} must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in choices:
        supported = ", ".join(sorted(choices))
        if label == "training.method" and normalized == "qlora":
            raise TrainingConfigError(
                "training.method 'qlora' is reserved for future support but is not enabled yet; "
                "use 'lora' or 'full'"
            )
        raise TrainingConfigError(f"{label} must be one of: {supported}")
    return normalized


def _require_positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise TrainingConfigError(f"{label} must be a positive number")
    return value


def _require_positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingConfigError(f"{label} must be a positive integer")
    return value


def _require_boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingConfigError(f"{label} must be true or false")
    return value


def _parse_lora_config(value: Any) -> LoraConfig:
    raw_lora = _require_mapping(value, "training.lora")
    _reject_unknown_fields(raw_lora, _LORA_FIELDS, "training.lora")
    _require_fields(raw_lora, _LORA_FIELDS, "training.lora")

    dropout = raw_lora["dropout"]
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not isfinite(dropout)
        or not 0 <= dropout < 1
    ):
        raise TrainingConfigError(
            "training.lora.dropout must be a number greater than or equal to 0 and less than 1"
        )

    return LoraConfig(
        rank=_require_positive_integer(raw_lora["rank"], "training.lora.rank"),
        alpha=_require_positive_integer(raw_lora["alpha"], "training.lora.alpha"),
        dropout=float(dropout),
    )


def resolve_training_config(raw_training: Any) -> TrainingConfig:
    """Validate the public config schema and return a typed configuration."""
    training = _require_mapping(raw_training, "training")
    _reject_unknown_fields(training, _COMMON_FIELDS, "training")
    _require_fields(training, _REQUIRED_FIELDS, "training")

    method = _require_choice(
        training["method"], "training.method", set(_METHOD_SPECS)
    )
    method_spec = _METHOD_SPECS[method]
    precision = _require_choice(
        training["precision"], "training.precision", _SUPPORTED_PRECISIONS
    )

    raw_lora = training.get("lora")
    if method_spec.uses_lora:
        if raw_lora is None:
            raise TrainingConfigError(
                "training.lora is required when training.method is 'lora'"
            )
        lora = _parse_lora_config(raw_lora)
    else:
        if raw_lora is not None:
            raise TrainingConfigError(
                "training.lora is only supported when training.method is 'lora'"
            )
        lora = None

    return TrainingConfig(
        method=method,
        finetuning_type=method_spec.finetuning_type,
        epochs=_require_positive_number(training["epochs"], "training.epochs"),
        learning_rate=_require_positive_number(
            training["learning_rate"], "training.learning_rate"
        ),
        batch_size=_require_positive_integer(
            training["batch_size"], "training.batch_size"
        ),
        precision=precision,
        gradient_checkpointing=_require_boolean(
            training["gradient_checkpointing"], "training.gradient_checkpointing"
        ),
        cutoff_len=_require_positive_integer(
            training["cutoff_len"], "training.cutoff_len"
        ),
        save_steps=_require_positive_integer(
            training["save_steps"], "training.save_steps"
        ),
        logging_steps=_require_positive_integer(
            training["logging_steps"], "training.logging_steps"
        ),
        lora=lora,
    )

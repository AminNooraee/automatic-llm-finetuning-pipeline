"""Streaming console presentation for raw LLaMA-Factory output."""

from __future__ import annotations

import ast
import re
import sys
from typing import Any, Mapping, TextIO


CONSOLE_VERBOSITIES = {"quiet", "concise", "full"}
DEFAULT_CONSOLE_VERBOSITY = "concise"

_PROGRESS_RE = re.compile(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)")
_IMPORTANT_RE = re.compile(r"\b(?:ERROR|CRITICAL)\b", re.IGNORECASE)
_WARNING_RE = re.compile(r"\bWARNING\b", re.IGNORECASE)
_TRACEBACK_END_RE = re.compile(r"^[\w.]+(?:Error|Exception):")
_TRAINING_DETAIL_RE = re.compile(
    r"(?:Num examples|Num Epochs|Total optimization steps)\s*=",
    re.IGNORECASE,
)
_FINAL_METRIC_RE = re.compile(
    r"(?:epoch|train_loss|train_runtime|train_samples_per_second|"
    r"train_steps_per_second)\s*=",
    re.IGNORECASE,
)
_METRIC_KEYS = {
    "loss",
    "grad_norm",
    "learning_rate",
    "epoch",
    "train_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
}


class ConsoleVerbosityError(ValueError):
    """Raised when the public observability configuration is invalid."""


def resolve_console_verbosity(config: Mapping[str, Any]) -> str:
    """Validate the bounded observability console configuration."""
    section = config.get("observability")
    if section is None:
        return DEFAULT_CONSOLE_VERBOSITY
    if not isinstance(section, Mapping):
        raise ConsoleVerbosityError("observability must be a YAML mapping")
    unknown = sorted(str(key) for key in section if key != "console_verbosity")
    if unknown:
        raise ConsoleVerbosityError(
            "Unsupported observability parameter(s): "
            f"{', '.join(unknown)}. Supported parameters: console_verbosity"
        )
    value = section.get("console_verbosity", DEFAULT_CONSOLE_VERBOSITY)
    if not isinstance(value, str) or value.strip().lower() not in CONSOLE_VERBOSITIES:
        raise ConsoleVerbosityError(
            "observability.console_verbosity must be one of: concise, full, quiet"
        )
    return value.strip().lower()


class ConsoleOutputFormatter:
    """Incrementally filter backend output without buffering the complete log."""

    def __init__(self, verbosity: str, stream: TextIO | None = None):
        if verbosity not in CONSOLE_VERBOSITIES:
            raise ConsoleVerbosityError(f"Unsupported console verbosity: {verbosity!r}")
        self.verbosity = verbosity
        self.stream = stream or sys.stdout
        self._buffer = ""
        self._in_traceback = False
        self._current_step: int | None = None
        self._total_steps: int | None = None
        self._last_progress_step = 0

    def feed(self, text: str) -> None:
        if not text:
            return
        if self.verbosity == "full":
            self.stream.write(text)
            self.stream.flush()
            return

        self._buffer += text
        start = 0
        for index, character in enumerate(self._buffer):
            if character not in "\r\n":
                continue
            record = self._buffer[start:index]
            if record or character == "\n":
                self._handle_record(record, carriage_return=character == "\r")
            start = index + 1
        self._buffer = self._buffer[start:]

    def close(self) -> None:
        if self.verbosity == "full":
            return
        if self._buffer:
            self._handle_record(self._buffer, carriage_return=False)
            self._buffer = ""

    def _handle_record(self, record: str, *, carriage_return: bool) -> None:
        line = _strip_ansi(record).strip()
        if not line:
            if self._in_traceback and not carriage_return:
                self._in_traceback = False
            return

        if line.startswith("Traceback (most recent call last):"):
            self._in_traceback = True
        if self._in_traceback or _IMPORTANT_RE.search(line):
            self._emit(line)
            if self._in_traceback and _TRACEBACK_END_RE.match(line):
                self._in_traceback = False
            return
        if self.verbosity == "quiet":
            return
        if _WARNING_RE.search(line):
            self._emit(line)
            return

        progress = _PROGRESS_RE.search(line) if ("%" in line or "it/s" in line) else None
        if progress:
            self._current_step = int(progress.group("current"))
            self._total_steps = int(progress.group("total"))
            threshold = max(1, self._total_steps // 20)
            if (
                self._current_step == self._total_steps
                or self._current_step >= self._last_progress_step + threshold
            ):
                percent = 100 * self._current_step / self._total_steps
                self._emit(
                    f"Progress: {self._current_step}/{self._total_steps} ({percent:.0f}%)"
                )
                self._last_progress_step = self._current_step
            return

        metrics = _parse_metric_mapping(line)
        if metrics:
            self._emit(_format_metrics(metrics, self._current_step, self._total_steps))
            return

        if (
            "***** Running training *****" in line
            or "***** train metrics *****" in line
            or _TRAINING_DETAIL_RE.search(line)
            or _FINAL_METRIC_RE.search(line)
        ):
            self._emit(line)

    def _emit(self, text: str) -> None:
        self.stream.write(text + "\n")
        self.stream.flush()


def _parse_metric_mapping(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = ast.literal_eval(line[start : end + 1])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, dict) or not (_METRIC_KEYS & set(value)):
        return None
    return {key: value[key] for key in _METRIC_KEYS if key in value}


def _format_metrics(
    metrics: Mapping[str, Any], current_step: int | None, total_steps: int | None
) -> str:
    parts: list[str] = []
    if current_step is not None and total_steps is not None:
        parts.append(f"Step {current_step}/{total_steps}")
    labels = (
        ("epoch", "Epoch"),
        ("loss", "Loss"),
        ("grad_norm", "GradNorm"),
        ("learning_rate", "LR"),
        ("train_loss", "Train loss"),
        ("train_runtime", "Runtime"),
        ("train_samples_per_second", "Samples/sec"),
        ("train_steps_per_second", "Steps/sec"),
    )
    for key, label in labels:
        if key in metrics:
            parts.append(f"{label} {_display_value(metrics[key], key)}")
    return " | ".join(parts)


def _display_value(value: Any, key: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if key == "learning_rate":
        return f"{number:.3g}"
    if key == "train_runtime":
        return f"{number:.1f}s"
    return f"{number:.4g}"


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)

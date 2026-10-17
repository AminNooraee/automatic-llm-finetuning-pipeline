"""Focused contract tests for the sanitized downstream report context."""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fine_tuning_pipeline.report_context import (
    build_report_context,
    write_report_context,
)
from fine_tuning_pipeline.train_pipeline import _write_training_report_context
from fine_tuning_pipeline.training_config import LoraConfig, TrainingConfig


class ReportContextTests(unittest.TestCase):
    def setUp(self):
        self.training = TrainingConfig(
            method="lora",
            finetuning_type="lora",
            epochs=3,
            learning_rate=0.0002,
            batch_size=4,
            precision="bf16",
            gradient_checkpointing=True,
            cutoff_len=2048,
            save_steps=10,
            logging_steps=2,
            lora=LoraConfig(rank=8, alpha=16, dropout=0.05),
            attention_backend="fa2",
        )

    def test_shape_contains_only_user_editable_fields_and_duration(self):
        context = build_report_context(
            model_name="org/model",
            training_config=self.training,
            training_duration_seconds=12.75,
        )

        self.assertEqual(
            context,
            {
                "schema_version": 1,
                "training_duration_seconds": 12.75,
                "training": {
                    "model_name": "org/model",
                    "epochs": 3.0,
                    "learning_rate": 0.0002,
                    "batch_size": 4,
                    "cutoff_len": 2048,
                    "lora": {"rank": 8, "alpha": 16, "dropout": 0.05},
                },
            },
        )
        serialized = json.dumps(context)
        for forbidden in (
            "precision",
            "gradient_checkpointing",
            "save_steps",
            "logging_steps",
            "attention_backend",
            "api_key",
            "Authorization",
            "endpoint",
            "environment",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_missing_or_invalid_duration_is_not_changed_to_zero(self):
        for value in (None, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                context = build_report_context(
                    model_name="org/model",
                    training_config=self.training,
                    training_duration_seconds=value,
                )
                self.assertIsNone(context["training_duration_seconds"])

    def test_written_json_matches_sanitized_context(self):
        context = build_report_context(
            model_name="org/model",
            training_config=self.training,
            training_duration_seconds=9.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_report_context(Path(directory), context)
            self.assertEqual(path.name, "report_context.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), context)

    def test_pipeline_propagates_recorded_training_wall_clock_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = Mock()

            class FakeRun:
                metadata = {
                    "resources": {"training_wall_clock_seconds": 17.25}
                }
                paths = SimpleNamespace(root=Path(directory))

                @contextmanager
                def logging(self):
                    yield logger

            prepared = SimpleNamespace(
                config={"model": {"name": "org/model"}},
                training_config=self.training,
            )
            _write_training_report_context(FakeRun(), prepared)
            payload = json.loads(
                (Path(directory) / "report_context.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["training_duration_seconds"], 17.25)
            logger.info.assert_called_once()


if __name__ == "__main__":
    unittest.main()

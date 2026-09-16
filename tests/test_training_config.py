import tempfile
import unittest
from pathlib import Path

import yaml

from fine_tuning_pipeline.train_pipeline import PIPELINE_DIR
from fine_tuning_pipeline.training_config import TrainingConfigError, resolve_training_config
from fine_tuning_pipeline.yaml_generator import build_llamafactory_config, generate_training_yaml


def valid_training_config():
    return {
        "method": "lora",
        "epochs": 3,
        "learning_rate": 0.0002,
        "batch_size": 4,
        "precision": "bf16",
        "gradient_checkpointing": True,
        "cutoff_len": 2048,
        "save_steps": 100,
        "logging_steps": 10,
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
        },
    }


class TrainingConfigurationTests(unittest.TestCase):
    def test_all_config_values_propagate_to_llamafactory_arguments(self):
        resolved = resolve_training_config(valid_training_config())

        self.assertEqual(
            resolved.to_llamafactory_args(),
            {
                "finetuning_type": "lora",
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "learning_rate": 0.0002,
                "fp16": False,
                "bf16": True,
                "gradient_checkpointing": True,
                "cutoff_len": 2048,
                "save_steps": 100,
                "logging_steps": 10,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
            },
        )

    def test_invalid_and_unsupported_parameters_are_rejected(self):
        cases = []

        unknown = valid_training_config()
        unknown["warmup_steps"] = 5
        cases.append((unknown, "Unsupported training parameter.*warmup_steps"))

        legacy_flat_lora = valid_training_config()
        legacy_flat_lora["lora_rank"] = 16
        cases.append((legacy_flat_lora, "Unsupported training parameter.*lora_rank"))

        invalid_precision = valid_training_config()
        invalid_precision["precision"] = "tf32"
        cases.append((invalid_precision, "training.precision must be one of"))

        invalid_boolean = valid_training_config()
        invalid_boolean["gradient_checkpointing"] = "true"
        cases.append((invalid_boolean, "training.gradient_checkpointing must be true or false"))

        invalid_length = valid_training_config()
        invalid_length["cutoff_len"] = 0
        cases.append((invalid_length, "training.cutoff_len must be a positive integer"))

        non_finite_rate = valid_training_config()
        non_finite_rate["learning_rate"] = float("nan")
        cases.append((non_finite_rate, "training.learning_rate must be a positive number"))

        invalid_lora = valid_training_config()
        invalid_lora["lora"]["dropout"] = 1.0
        cases.append((invalid_lora, "training.lora.dropout"))

        missing = valid_training_config()
        del missing["logging_steps"]
        cases.append((missing, "Missing required training parameter.*logging_steps"))

        for raw_config, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(TrainingConfigError, expected_error):
                    resolve_training_config(raw_config)

    def test_lora_section_is_method_specific(self):
        missing_lora = valid_training_config()
        del missing_lora["lora"]
        with self.assertRaisesRegex(TrainingConfigError, "training.lora is required"):
            resolve_training_config(missing_lora)

        full = valid_training_config()
        full["method"] = "full"
        with self.assertRaisesRegex(TrainingConfigError, "only supported.*'lora'"):
            resolve_training_config(full)

        del full["lora"]
        generated = resolve_training_config(full).to_llamafactory_args()
        self.assertEqual(generated["finetuning_type"], "full")
        self.assertNotIn("lora_rank", generated)
        self.assertNotIn("lora_alpha", generated)
        self.assertNotIn("lora_dropout", generated)

    def test_qlora_is_reserved_until_quantization_is_configurable(self):
        config = valid_training_config()
        config["method"] = "qlora"
        with self.assertRaisesRegex(TrainingConfigError, "reserved for future support"):
            resolve_training_config(config)

    def test_generated_yaml_is_valid_and_contains_only_translated_fields(self):
        config = {
            "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "template": "qwen"},
            "dataset": {"name": "example"},
            "training": valid_training_config(),
            "output": {"path": "../models", "model_name": "example-model"},
        }

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary_dir:
            output_file = Path(temporary_dir) / "training.yaml"
            generated_file = generate_training_yaml(
                config,
                dataset_dir=Path(temporary_dir) / "data",
                output_dir=Path(temporary_dir) / "model",
                output_file=output_file,
            )
            loaded = yaml.safe_load(Path(generated_file).read_text(encoding="utf-8"))

        self.assertIsInstance(loaded, dict)
        self.assertEqual(loaded["stage"], "sft")
        self.assertIs(loaded["do_train"], True)
        self.assertEqual(loaded["template"], "qwen")
        self.assertEqual(loaded["num_train_epochs"], 3)
        self.assertEqual(loaded["per_device_train_batch_size"], 4)
        self.assertEqual(loaded["learning_rate"], 0.0002)
        self.assertIs(loaded["bf16"], True)
        self.assertIs(loaded["fp16"], False)
        self.assertIs(loaded["disable_gradient_checkpointing"], False)
        self.assertNotIn("gradient_checkpointing", loaded)
        self.assertEqual(loaded["cutoff_len"], 2048)
        self.assertEqual(loaded["save_steps"], 100)
        self.assertEqual(loaded["logging_steps"], 10)
        self.assertEqual(loaded["lora_rank"], 8)
        self.assertEqual(loaded["lora_alpha"], 16)
        self.assertEqual(loaded["lora_dropout"], 0.05)
        self.assertNotIn("rank", loaded)
        self.assertNotIn("alpha", loaded)
        self.assertNotIn("dropout", loaded)

    def test_checkpointing_uses_llamafactory_loader_flag(self):
        for requested in (False, True):
            with self.subTest(requested=requested):
                training = valid_training_config()
                training["gradient_checkpointing"] = requested
                config = {
                    "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "template": "qwen"},
                    "dataset": {"name": "example"},
                    "training": training,
                }
                generated = build_llamafactory_config(
                    config,
                    dataset_dir=Path(__file__).resolve().parent / "dataset",
                    output_dir=Path(__file__).resolve().parent / "model",
                )
                self.assertIs(generated["disable_gradient_checkpointing"], not requested)
                self.assertNotIn("gradient_checkpointing", generated)


if __name__ == "__main__":
    unittest.main()

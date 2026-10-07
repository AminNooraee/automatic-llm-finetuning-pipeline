"""Relocation checks for the editable source-checkout layout."""

import importlib
import os
import pkgutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import fine_tuning_pipeline
from fine_tuning_pipeline.model_manager import resolve_model_compatibility
from fine_tuning_pipeline.train_pipeline import (
    DEFAULT_CONFIG_PATH,
    REPOSITORY_ROOT,
    load_config,
    main,
    resolve_config_path,
)
from fine_tuning_pipeline.training_config import resolve_training_config
from fine_tuning_pipeline.yaml_generator import build_llamafactory_config


class RepositoryLayoutTests(unittest.TestCase):
    def test_all_source_modules_import_from_package(self):
        modules = list(pkgutil.walk_packages(
            fine_tuning_pipeline.__path__, fine_tuning_pipeline.__name__ + "."
        ))
        self.assertEqual(len(modules), 53)
        for module in modules:
            with self.subTest(module=module.name):
                imported = importlib.import_module(module.name)
                self.assertTrue(
                    Path(imported.__file__).resolve().is_relative_to(REPOSITORY_ROOT / "src")
                )

    def test_default_and_named_configs_resolve_checkout_paths(self):
        self.assertEqual(DEFAULT_CONFIG_PATH, REPOSITORY_ROOT / "configs" / "config.yaml")
        names = (
            "config.yaml", "qwen_example.yaml", "llama_example.yaml",
            "huggingface_dataset_example.yaml", "serving_gateway_example.yaml",
            "full_pipeline_example.yaml", "ci_phase1.yaml",
        )
        for name in names:
            with self.subTest(config=name):
                path = REPOSITORY_ROOT / "configs" / name
                config = load_config(path)
                self.assertEqual(resolve_training_config(config["training"]).method, "lora")
                self.assertEqual(
                    resolve_config_path(config["output"]["runs_path"], path.parent),
                    REPOSITORY_ROOT / "runs",
                )
                if name == "huggingface_dataset_example.yaml":
                    self.assertEqual(config["dataset"]["path"], "lhoestq/demo1")
                    self.assertEqual(config["dataset"]["split"], "train[4:5]")
                elif name == "ci_phase1.yaml":
                    self.assertEqual(config["dataset"]["path"], "phase1_train.json")
                else:
                    dataset = resolve_config_path(config["dataset"]["path"], path.parent)
                    self.assertEqual(dataset, REPOSITORY_ROOT / "examples" / "datasets" / "alpaca_demo.json")
                    self.assertTrue(dataset.is_file())

    def test_phase1_ci_config_uses_only_portable_environment_references(self):
        config = load_config(REPOSITORY_ROOT / "configs" / "ci_phase1.yaml")
        self.assertEqual(config["model"]["name"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(config["training"]["method"], "lora")
        self.assertEqual(config["training"]["epochs"], 1)
        resolved_training = resolve_training_config(config["training"])
        self.assertEqual(resolved_training.attention_backend, "eager")
        model_compatibility = resolve_model_compatibility(
            config["model"]["name"],
            config_loader=lambda _model_name: {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
            },
        )
        config["model"]["template"] = model_compatibility.template
        generated_training = build_llamafactory_config(
            config,
            dataset_dir=REPOSITORY_ROOT / "datasets",
            output_dir=REPOSITORY_ROOT / "runs" / "phase1" / "model",
            training_config=resolved_training,
        )
        self.assertEqual(generated_training["flash_attn"], "disabled")
        self.assertTrue(config["orchestration"]["enabled"])
        self.assertTrue(config["resource_preflight"]["enabled"])
        self.assertEqual(
            config["resource_preflight"]["model_parameter_estimate"], 490000000
        )
        self.assertEqual(config["serving"]["port"], "auto")
        self.assertEqual(config["serving"]["port_range"], {"start": 8101, "end": 8199})
        self.assertEqual(config["serving"]["advertise_host"], "${SERVING_ADVERTISE_HOST}")
        self.assertEqual(config["gateway"]["base_url"], "${LITELLM_BASE_URL}")
        self.assertEqual(config["gateway"]["api_key"], "${LITELLM_API_KEY}")
        self.assertTrue(config["gateway"]["allow_insecure_http"])

        ci = yaml.safe_load((REPOSITORY_ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8"))
        phase = ci["phase1-finetune"]
        self.assertEqual(phase["when"], "manual")
        self.assertEqual(phase["variables"]["PIPELINE_RESOURCE_PREFLIGHT"], "1")
        self.assertEqual(phase["variables"]["GPU_DEVICE"], "auto")
        self.assertNotIn("GPU_MIN_FREE_MIB", phase["variables"])

    def test_installed_import_and_default_config_ignore_working_directory(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            result = subprocess.run(
                [sys.executable, "-c", (
                    "from fine_tuning_pipeline.train_pipeline import load_config; "
                    "assert load_config()['model']['name'] == 'Qwen/Qwen2.5-0.5B-Instruct'; "
                    "print('DEFAULT_CONFIG_OK')"
                )],
                cwd=temporary, env=environment, capture_output=True, text=True,
                check=True, timeout=60,
            )
        self.assertIn("DEFAULT_CONFIG_OK", result.stdout)

    def test_module_entry_point_uses_existing_training_workflow(self):
        with patch("fine_tuning_pipeline.train_pipeline.execute_training") as execute:
            main()
        execute.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

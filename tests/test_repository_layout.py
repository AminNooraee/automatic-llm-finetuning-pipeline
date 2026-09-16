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

import fine_tuning_pipeline
from fine_tuning_pipeline.train_pipeline import (
    DEFAULT_CONFIG_PATH,
    REPOSITORY_ROOT,
    load_config,
    main,
    resolve_config_path,
)
from fine_tuning_pipeline.training_config import resolve_training_config


class RepositoryLayoutTests(unittest.TestCase):
    def test_all_source_modules_import_from_package(self):
        modules = list(pkgutil.walk_packages(
            fine_tuning_pipeline.__path__, fine_tuning_pipeline.__name__ + "."
        ))
        self.assertEqual(len(modules), 28)
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
            "huggingface_dataset_example.yaml",
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
                else:
                    dataset = resolve_config_path(config["dataset"]["path"], path.parent)
                    self.assertEqual(dataset, REPOSITORY_ROOT / "examples" / "datasets" / "alpaca_demo.json")
                    self.assertTrue(dataset.is_file())

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

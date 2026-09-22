import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import yaml

from fine_tuning_pipeline.artifact_validator import ArtifactValidationError, validate_model_artifacts
from fine_tuning_pipeline.run_manager import RunManager
from fine_tuning_pipeline.train_pipeline import execute_training


TESTS_DIR = Path(__file__).resolve().parent


class RunManagerTests(unittest.TestCase):
    def temporary_project_dir(self):
        temporary = tempfile.TemporaryDirectory(dir=TESTS_DIR)
        path = Path(temporary.name).resolve()
        self.addCleanup(temporary.cleanup)
        return path

    def test_run_folder_creation_and_no_overwrite(self):
        root = self.temporary_project_dir()
        input_config = root / "config.yaml"
        input_config.write_text("model: {}\n", encoding="utf-8")
        fixed_time = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)

        first = RunManager.create(
            root / "runs",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_name="demo",
            input_config_path=input_config,
            created_at=fixed_time,
        )
        second = RunManager.create(
            root / "runs",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_name="demo",
            input_config_path=input_config,
            created_at=fixed_time,
        )

        self.assertEqual(
            first.run_id,
            "20260916_120000_qwen2-5-0-5b-instruct_demo",
        )
        self.assertEqual(second.run_id, f"{first.run_id}_02")
        self.assertNotEqual(first.paths.root, second.paths.root)
        for run in (first, second):
            self.assertTrue(run.paths.config.is_dir())
            self.assertTrue(run.paths.dataset.is_dir())
            self.assertTrue(run.paths.logs.is_dir())
            self.assertTrue(run.paths.model.is_dir())
            self.assertTrue(run.paths.input_config.is_file())
            self.assertTrue(run.paths.metadata.is_file())

    def test_missing_lora_artifact_is_rejected(self):
        root = self.temporary_project_dir()
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(
            ArtifactValidationError, "Missing LoRA adapter weights"
        ):
            validate_model_artifacts(model_dir, "lora")

    def test_successful_lora_artifacts_are_returned(self):
        root = self.temporary_project_dir()
        model_dir = root / "model"
        model_dir.mkdir()
        config_file = model_dir / "adapter_config.json"
        weights_file = model_dir / "adapter_model.safetensors"
        config_file.write_text("{}", encoding="utf-8")
        weights_file.write_bytes(b"weights")

        self.assertEqual(
            validate_model_artifacts(model_dir, "lora"),
            [config_file, weights_file],
        )

    def test_successful_full_model_artifacts_are_returned(self):
        root = self.temporary_project_dir()
        model_dir = root / "model"
        model_dir.mkdir()
        config_file = model_dir / "config.json"
        weights_file = model_dir / "model-00001-of-00002.safetensors"
        config_file.write_text("{}", encoding="utf-8")
        weights_file.write_bytes(b"weights")

        self.assertEqual(
            validate_model_artifacts(model_dir, "full"),
            [config_file, weights_file],
        )


class RunPipelineLifecycleTests(unittest.TestCase):
    def setUp(self):
        model_config_patcher = patch(
            "fine_tuning_pipeline.model_manager._load_huggingface_config",
            return_value={
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
            },
        )
        model_config_patcher.start()
        self.addCleanup(model_config_patcher.stop)

    def temporary_project_dir(self):
        temporary = tempfile.TemporaryDirectory(dir=TESTS_DIR)
        path = Path(temporary.name).resolve()
        self.addCleanup(temporary.cleanup)
        return path

    def test_successful_run_writes_metadata_and_verifies_model(self):
        root = self.temporary_project_dir()

        def create_adapter(yaml_file, log_file=None, **_kwargs):
            arguments = yaml.safe_load(Path(yaml_file).read_text(encoding="utf-8"))
            model_dir = Path(arguments["output_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "adapter_model.safetensors").write_bytes(b"weights")
            (model_dir / "train_results.json").write_text(
                json.dumps({"train_loss": 0.75, "train_runtime": 4.0}),
                encoding="utf-8",
            )
            (model_dir / "trainer_state.json").write_text(
                json.dumps({"epoch": 1.0, "global_step": 2, "log_history": []}),
                encoding="utf-8",
            )
            if log_file is not None:
                with Path(log_file).open("a", encoding="utf-8") as file:
                    file.write("mock training output\n")

        with patch("fine_tuning_pipeline.train_pipeline.run_training", side_effect=create_adapter):
            run_dir = execute_training(runs_root=root / "runs")

        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["status"], "success")
        self.assertEqual(metadata["model"]["family"], "qwen")
        self.assertEqual(metadata["model"]["template"], "qwen")
        self.assertEqual(metadata["dataset"]["format"], "alpaca")
        self.assertEqual(metadata["dataset"]["num_samples"], 2)
        self.assertEqual(metadata["training"]["method"], "lora")
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["training_metrics"]["train_loss"], 0.75)
        self.assertEqual(metadata["training_metrics"]["global_step"], 2)
        self.assertGreaterEqual(metadata["resources"]["training_wall_clock_seconds"], 0)
        self.assertEqual(metadata["output"]["model_dir"], str(run_dir / "model"))
        self.assertEqual(
            metadata["output"]["verified_artifacts"],
            ["model/adapter_config.json", "model/adapter_model.safetensors"],
        )
        self.assertTrue((run_dir / "config" / "input_config.yaml").is_file())
        self.assertTrue((run_dir / "config" / "resolved_config.yaml").is_file())
        self.assertTrue((run_dir / "config" / "training.yaml").is_file())
        self.assertTrue((run_dir / "dataset" / "original_dataset.json").is_file())
        self.assertTrue((run_dir / "dataset" / "normalized_dataset.json").is_file())
        log = (run_dir / "logs" / "train.log").read_text(encoding="utf-8")
        self.assertIn("Resolving model compatibility", log)
        self.assertIn("Detecting, validating, and normalizing dataset", log)
        self.assertIn("Starting LLaMA-Factory training", log)
        self.assertIn("mock training output", log)
        self.assertIn("Run completed successfully", log)

    def test_pipeline_failure_marks_run_and_preserves_error_log(self):
        root = self.temporary_project_dir()
        with patch(
            "fine_tuning_pipeline.train_pipeline.run_training",
            side_effect=RuntimeError("training process failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "training process failed"):
                execute_training(runs_root=root / "runs")

        runs = list((root / "runs").iterdir())
        self.assertEqual(len(runs), 1)
        metadata = json.loads(
            (runs[0] / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error"]["type"], "RuntimeError")
        self.assertEqual(metadata["error"]["message"], "training process failed")
        log = (runs[0] / "logs" / "train.log").read_text(encoding="utf-8")
        self.assertIn("Training run failed", log)
        self.assertIn("training process failed", log)

    def test_missing_model_artifacts_mark_run_failed(self):
        root = self.temporary_project_dir()
        with patch("fine_tuning_pipeline.train_pipeline.run_training", return_value=None):
            with self.assertRaisesRegex(
                ArtifactValidationError, "Missing LoRA adapter configuration"
            ):
                execute_training(runs_root=root / "runs")

        run_dir = next((root / "runs").iterdir())
        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error"]["type"], "ArtifactValidationError")
        self.assertIn("adapter_config.json", metadata["error"]["message"])


if __name__ == "__main__":
    unittest.main()

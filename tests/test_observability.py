import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from fine_tuning_pipeline.model_manager import resolve_model_compatibility
from fine_tuning_pipeline.observability import (
    collect_container_provenance,
    collect_environment,
    collect_resource_baseline,
    extract_training_metrics,
    sha256_file,
)
from fine_tuning_pipeline.train_pipeline import DEFAULT_CONFIG_PATH, prepare_training
from fine_tuning_pipeline.trainer import run_training


TESTS_DIR = Path(__file__).resolve().parent


class ObservabilityUnitTests(unittest.TestCase):
    def temporary_dir(self):
        temporary = tempfile.TemporaryDirectory(dir=TESTS_DIR)
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name).resolve()

    def test_dataset_hash_is_deterministic_and_byte_exact(self):
        path = self.temporary_dir() / "dataset.json"
        path.write_bytes(b'[{"instruction":"x"}]\n')
        first = sha256_file(path)
        self.assertEqual(first, sha256_file(path))
        path.write_bytes(b'[{"instruction":"x"}]\r\n')
        self.assertNotEqual(first, sha256_file(path))

    def test_environment_contains_version_and_gpu_fallback_fields(self):
        environment = collect_environment()
        self.assertIn("python", environment)
        self.assertIn("platform", environment)
        self.assertIn("torch", environment["packages"])
        self.assertIn("transformers", environment["packages"])
        self.assertIn("visible_gpus", environment)
        resources = collect_resource_baseline(environment)
        self.assertIsNone(resources["process_cuda_memory"]["peak_allocated_bytes"])
        self.assertIn("process-local", resources["process_cuda_memory"]["availability"])

    def test_optional_container_metadata_uses_only_explicit_provenance(self):
        values = {
            "PIPELINE_CONTAINERIZED": "true",
            "PIPELINE_CONTAINER_IMAGE": "example/trainer:h100",
            "PIPELINE_CONTAINER_IMAGE_ID": "sha256:image",
            "PIPELINE_CONTAINER_IMAGE_DIGEST": "sha256:digest",
            "PIPELINE_CONTAINER_RUNTIME": "docker",
        }
        with patch.dict(os.environ, values, clear=False):
            provenance = collect_container_provenance()
        self.assertTrue(provenance["appears_containerized"])
        self.assertEqual(provenance["image"], "example/trainer:h100")
        self.assertEqual(provenance["image_digest"], "sha256:digest")
        self.assertEqual(provenance["runtime"], "docker")

    def test_metrics_are_normalized_without_copying_log_history(self):
        model_dir = self.temporary_dir()
        (model_dir / "train_results.json").write_text(
            json.dumps(
                {
                    "train_loss": 1.25,
                    "train_runtime": 12.5,
                    "train_samples_per_second": 3.0,
                    "train_steps_per_second": 1.5,
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "trainer_state.json").write_text(
            json.dumps({"epoch": 2.0, "global_step": 8, "log_history": [{"loss": 2.0}]}),
            encoding="utf-8",
        )
        metrics = extract_training_metrics(model_dir)
        self.assertEqual(metrics["train_loss"], 1.25)
        self.assertEqual(metrics["final_epoch"], 2.0)
        self.assertEqual(metrics["global_step"], 8)
        self.assertEqual(metrics["log_history_artifact"], "trainer_state.json")
        self.assertNotIn("log_history", metrics)

    def test_revision_resolution_and_unavailable_reason(self):
        resolved = resolve_model_compatibility(
            "Qwen/Qwen2.5-0.5B-Instruct",
            requested_revision="main",
            config_loader=lambda _name, revision=None: {
                "model_type": "qwen2",
                "_commit_hash": "a" * 40,
            },
        )
        self.assertEqual(resolved.requested_revision, "main")
        self.assertEqual(resolved.resolved_revision, "a" * 40)
        self.assertEqual(resolved.revision_status, "resolved")

        unavailable = resolve_model_compatibility(
            "Qwen/Qwen2.5-0.5B-Instruct",
            config_loader=lambda _name: (_ for _ in ()).throw(OSError("offline")),
        )
        self.assertIsNone(unavailable.resolved_revision)
        self.assertEqual(unavailable.revision_status, "unavailable")
        self.assertIn("offline", unavailable.revision_unavailable_reason)


class LiveTrainerTests(unittest.TestCase):
    def temporary_dir(self):
        temporary = tempfile.TemporaryDirectory(dir=TESTS_DIR)
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name).resolve()

    def _process(self, output, returncode):
        return type(
            "Process",
            (),
            {"stdout": StringIO(output), "wait": lambda self: returncode},
        )()

    def test_success_output_is_teed_live_and_persisted(self):
        log = self.temporary_dir() / "train.log"
        terminal = StringIO()
        with patch(
            "fine_tuning_pipeline.trainer.subprocess.Popen",
            return_value=self._process("{'loss': 1.2, 'epoch': 1.0}\n", 0),
        ):
            with redirect_stdout(terminal):
                result = run_training("training.yaml", log_file=log)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Epoch 1 | Loss 1.2", terminal.getvalue())
        self.assertIn("'loss': 1.2", log.read_text(encoding="utf-8"))

    def test_failed_output_is_preserved_before_exit_error(self):
        log = self.temporary_dir() / "train.log"
        terminal = StringIO()
        with patch(
            "fine_tuning_pipeline.trainer.subprocess.Popen",
            return_value=self._process("fatal backend error\n", 7),
        ):
            with redirect_stdout(terminal):
                with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                    run_training("training.yaml", log_file=log)
        self.assertIn("fatal backend error", terminal.getvalue())
        self.assertIn("fatal backend error", log.read_text(encoding="utf-8"))


class MetadataV2IntegrationTests(unittest.TestCase):
    def temporary_dir(self):
        temporary = tempfile.TemporaryDirectory(dir=TESTS_DIR)
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name).resolve()

    def test_prepared_run_has_v2_reproducibility_and_relationship_metadata(self):
        root = self.temporary_dir()
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["model"]["revision"] = "release-1"
        config["dataset"]["path"] = str(
            DEFAULT_CONFIG_PATH.parent.parent / "examples" / "datasets" / "alpaca_demo.json"
        )
        custom = root / "config.yaml"
        custom.write_text(yaml.safe_dump(config), encoding="utf-8")
        model_config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "_commit_hash": "b" * 40,
        }
        with patch(
            "fine_tuning_pipeline.model_manager._load_huggingface_config",
            return_value=model_config,
        ):
            _, yaml_file = prepare_training(custom, artifact_root=root / "runs")

        run_dir = Path(yaml_file).parents[1]
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        training_yaml = yaml.safe_load(Path(yaml_file).read_text(encoding="utf-8"))
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["model"]["name"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(metadata["model"]["requested_revision"], "release-1")
        self.assertEqual(metadata["model"]["resolved_revision"], "b" * 40)
        self.assertEqual(training_yaml["model_revision"], "release-1")
        normalized = run_dir / metadata["dataset"]["normalized_path"]
        self.assertEqual(metadata["dataset"]["sha256"], sha256_file(normalized))
        self.assertEqual(metadata["dataset"]["source_format"], "json")
        self.assertEqual(metadata["dataset"]["format"], "alpaca")
        self.assertEqual(metadata["artifact"]["type"], "lora_adapter")
        self.assertFalse(metadata["artifact"]["merged"])
        self.assertEqual(metadata["serving"]["adapter_path"], str(run_dir / "model"))
        self.assertFalse(metadata["serving"]["endpoint_configured"])
        self.assertEqual(metadata["environment_artifact"], "environment.json")
        self.assertTrue((run_dir / "environment.json").is_file())
        # Existing v1-compatible fields remain available.
        for key in ("run_id", "status", "created_at", "model", "dataset", "training", "output"):
            self.assertIn(key, metadata)

    def test_optional_environment_failure_does_not_fail_preparation(self):
        root = self.temporary_dir()
        with patch(
            "fine_tuning_pipeline.model_manager._load_huggingface_config",
            return_value={"model_type": "qwen2"},
        ), patch(
            "fine_tuning_pipeline.train_pipeline.collect_environment",
            side_effect=RuntimeError("optional probe failed"),
        ):
            _, yaml_file = prepare_training(artifact_root=root / "runs")
        metadata = json.loads(
            (Path(yaml_file).parents[1] / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["status"], "prepared")


if __name__ == "__main__":
    unittest.main()

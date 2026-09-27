import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning_pipeline.orchestration.config import (
    OrchestrationConfigError,
    resolve_orchestration_config,
)
from fine_tuning_pipeline.orchestration.controller import (
    ControllerError,
    DockerResult,
    PipelineController,
    RuntimeLayout,
    _safe_relative,
    _worker_config,
)
from fine_tuning_pipeline.orchestration.training_worker import run_worker


TESTS_DIR = Path(__file__).resolve().parent


def config_value():
    return {
        "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
        "dataset": {"path": "data.json", "name": "demo"},
        "training": {
            "method": "lora", "epochs": 1, "learning_rate": 0.0002,
            "batch_size": 1, "precision": "bf16", "cutoff_len": 128,
            "gradient_checkpointing": False, "save_steps": 10, "logging_steps": 1,
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.0},
        },
        "orchestration": {
            "enabled": True, "training": {"gpu_devices": "1"},
            "cleanup_on_failure": True,
        },
        "serving": {
            "enabled": True, "advertise_host": "model-server.example",
            "restart_policy": "unless-stopped",
        },
        "gateway": {"enabled": False},
    }


class FakeDocker:
    def __init__(self, root: Path, execution_id: str, *, exit_code=0, result_override=None):
        self.root = root
        self.execution_id = execution_id
        self.exit_code = exit_code
        self.result_override = result_override
        self.calls = []

    def run(self, command, **_kwargs):
        command = tuple(str(item) for item in command)
        self.calls.append(command)
        if command[:3] == ("docker", "container", "create"):
            return DockerResult(0, "train-id\n", "")
        if command[:3] == ("docker", "container", "start"):
            return DockerResult(0, "train-id\n", "")
        if command[:3] == ("docker", "container", "wait"):
            if self.exit_code == 0:
                self._write_result()
            return DockerResult(0, f"{self.exit_code}\n", "")
        if command[:3] == ("docker", "container", "logs"):
            return DockerResult(0, "training log\n", "")
        if command[:3] == ("docker", "container", "inspect"):
            return DockerResult(0, f"train-id|true|training|{self.execution_id}\n", "")
        if command[:3] == ("docker", "container", "rm"):
            return DockerResult(0, "train-id\n", "")
        raise AssertionError(command)

    def _write_result(self):
        run = self.root / "runs" / "run-1"
        model = run / "model"
        model.mkdir(parents=True)
        (model / "adapter_config.json").write_text(json.dumps({
            "r": 8, "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"
        }), encoding="utf-8")
        (model / "adapter_model.safetensors").write_bytes(b"fake")
        metadata = {
            "run_id": "run-1",
            "model": {"requested_revision": None, "resolved_revision": "abc123"},
        }
        (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        result = {
            "schema_version": 1, "execution_id": self.execution_id,
            "run_id": "run-1", "status": "success", "run_directory": "run-1",
            "model_directory": "run-1/model", "metadata_path": "run-1/metadata.json",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "resolved_revision": "abc123",
            "training_method": "lora", "adapter_type": "lora_adapter",
        }
        if self.result_override:
            result.update(self.result_override)
        path = self.root / "state" / "executions" / self.execution_id / "training_result.json"
        path.write_text(json.dumps(result), encoding="utf-8")


class FakeServingManager:
    instances = []

    def __init__(self):
        self.calls = []
        self.cleaned = []
        self.__class__.instances.append(self)

    def start_and_verify(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "container_id": "serving-id", "manifest_path": "serving/endpoint_manifest.json",
            "health_path": "serving/health_check.json", "operation_path": "serving/operation.json",
            "log_path": "serving/serving.log",
        }

    def cleanup_owned(self, **kwargs):
        self.cleaned.append(kwargs)
        return True


class OrchestrationConfigTests(unittest.TestCase):
    def test_disabled_default_and_strict_gpu_configuration(self):
        self.assertFalse(resolve_orchestration_config(None).enabled)
        resolved = resolve_orchestration_config({
            "enabled": True, "training": {"gpu_devices": "0,2"},
            "cleanup_on_failure": False,
        })
        self.assertEqual(resolved.training_gpu_devices, "0,2")
        for value in ("all", "0,0", "-1", 1):
            with self.subTest(value=value), self.assertRaises(OrchestrationConfigError):
                resolve_orchestration_config({"enabled": True, "training": {"gpu_devices": value}})

    def test_relative_contract_rejects_absolute_and_traversal_paths(self):
        for value in ("../run", "/runs/run", "run/../other", ""):
            with self.subTest(value=value), self.assertRaises(ControllerError):
                _safe_relative(value, "result")

    def test_external_dataset_root_is_translated_and_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            project = root / "project"
            datasets = root / "datasets"
            project.mkdir()
            datasets.mkdir()
            (datasets / "sample.json").write_text("[]", encoding="utf-8")
            config_path = project / "config.yaml"
            layout = RuntimeLayout(
                project, root / "runs", root / "cache", root / "state", root / "secrets",
                controller_project=project, host_datasets=datasets,
                controller_datasets=datasets,
            )
            raw = config_value()
            raw["dataset"]["path"] = "sample.json"
            translated = _worker_config(raw, config_path, layout)
            self.assertEqual(translated["dataset"]["path"], "/workspace/datasets/sample.json")
            raw["dataset"]["path"] = "../outside.json"
            with self.assertRaises(ControllerError):
                _worker_config(raw, config_path, layout)


class TrainingWorkerTests(unittest.TestCase):
    def test_worker_writes_versioned_relative_secret_free_result(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            runs = root / "runs"
            run = runs / "run-1"
            run.mkdir(parents=True)
            (run / "metadata.json").write_text(json.dumps({
                "run_id": "run-1",
                "model": {"name": "Org/Base", "resolved_revision": "abc"},
                "training": {"method": "lora", "lora": {"rank": 8}},
                "artifact": {"type": "lora_adapter"},
            }), encoding="utf-8")
            result = root / "training_result.json"
            with patch(
                "fine_tuning_pipeline.orchestration.training_worker.execute_training",
                return_value=run,
            ):
                run_worker(
                    config_path=root / "config.yaml", runs_root=runs,
                    result_path=result, execution_id="exec-12345678",
                )
            value = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(value["run_directory"], "run-1")
            self.assertEqual(value["model_directory"], "run-1/model")
            self.assertEqual(value["lora_rank"], 8)
            self.assertNotIn(str(root), result.read_text(encoding="utf-8"))

    def test_worker_failure_does_not_persist_exception_message(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            result = root / "training_result.json"
            with patch(
                "fine_tuning_pipeline.orchestration.training_worker.execute_training",
                side_effect=RuntimeError("fake-secret-in-error"),
            ), self.assertRaises(RuntimeError):
                run_worker(
                    config_path=root / "config.yaml", runs_root=root / "runs",
                    result_path=result, execution_id="exec-12345678",
                )
            text = result.read_text(encoding="utf-8")
            self.assertNotIn("fake-secret-in-error", text)
            self.assertIn("RuntimeError", text)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        FakeServingManager.instances.clear()

    def _layout(self, root):
        for name in ("project", "runs", "cache", "state", "secrets"):
            (root / name).mkdir()
        (root / "project" / "config.yaml").write_text(
            json.dumps(config_value()), encoding="utf-8"
        )
        return RuntimeLayout(
            root / "project", root / "runs", root / "cache", root / "state",
            root / "secrets", root / "project", root / "runs", root / "state",
        )

    def test_training_isolated_then_serving_persists_with_expected_mounts(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            layout = self._layout(root)
            execution = "exec-12345678"
            docker = FakeDocker(root, execution)
            manifest = PipelineController(
                docker=docker, serving_manager_factory=FakeServingManager
            ).deploy(
                execution_id=execution, config_relative="config.yaml", layout=layout,
                training_image="trainer:test", training_image_id="sha256:trainer",
                source_revision="revision", source_identity="revision-dirty-hash",
            )
            create = docker.calls[0]
            self.assertIn("device=1", create)
            self.assertFalse(any("docker.sock" in item for item in create))
            self.assertTrue(any("/workspace/project,readonly" in item for item in create))
            self.assertTrue(any("/cache/huggingface" in item for item in create))
            self.assertLess(
                next(i for i, call in enumerate(docker.calls) if call[2] == "wait"),
                len(docker.calls),
            )
            serving = FakeServingManager.instances[0]
            self.assertEqual(serving.calls[0]["execution_id"], execution)
            self.assertEqual(serving.calls[0]["config"].restart_policy, "unless-stopped")
            self.assertEqual(serving.cleaned, [])
            value = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "success")
            self.assertEqual(value["serving"]["container_id"], "serving-id")

    def test_failed_training_prevents_serving_and_removes_only_owned_trainer(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            layout = self._layout(root)
            execution = "exec-12345678"
            docker = FakeDocker(root, execution, exit_code=7)
            with self.assertRaisesRegex(ControllerError, "Training failed"):
                PipelineController(
                    docker=docker, serving_manager_factory=FakeServingManager
                ).deploy(
                    execution_id=execution, config_relative="config.yaml", layout=layout,
                    training_image="trainer:test", training_image_id="sha256:trainer",
                    source_revision="revision", source_identity="revision",
                )
            self.assertEqual(FakeServingManager.instances, [])
            self.assertEqual(docker.calls[-1][:3], ("docker", "container", "rm"))

    def test_result_path_outside_its_run_is_rejected_before_serving(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            layout = self._layout(root)
            execution = "exec-12345678"
            docker = FakeDocker(root, execution, result_override={"model_directory": "other/model"})
            with self.assertRaisesRegex(ControllerError, "outside its run"):
                PipelineController(
                    docker=docker, serving_manager_factory=FakeServingManager
                ).deploy(
                    execution_id=execution, config_relative="config.yaml", layout=layout,
                    training_image="trainer:test", training_image_id="sha256:trainer",
                    source_revision="revision", source_identity="revision",
                )
            self.assertEqual(FakeServingManager.instances, [])


class LauncherStaticTests(unittest.TestCase):
    def test_launcher_requires_no_host_python_and_keeps_secrets_out_of_arguments(self):
        text = (TESTS_DIR.parent / "scripts" / "run_pipeline.sh").read_text(encoding="utf-8")
        self.assertIn("docker info", text)
        self.assertIn("fine-tuning-pipeline.source-identity", text)
        self.assertIn("--rebuild", text)
        self.assertNotIn("python ", text.lower())
        self.assertNotIn('--env "LITELLM_API_KEY=', text)
        self.assertIn("dst=/run/pipeline-secrets,readonly", text)
        self.assertIn("/var/run/docker.sock", text)

    def test_controller_image_is_minimal_and_trainer_never_receives_socket(self):
        dockerfile = (
            TESTS_DIR.parent / "docker" / "Dockerfile.controller"
        ).read_text(encoding="utf-8")
        self.assertIn("docker.io", dockerfile)
        self.assertIn("fine_tuning_pipeline.orchestration.controller", dockerfile)
        for forbidden in ("torch", "cuda", "vllm", "llamafactory"):
            self.assertNotIn(forbidden, dockerfile.lower())
        controller = (
            TESTS_DIR.parent / "src" / "fine_tuning_pipeline" / "orchestration" / "controller.py"
        ).read_text(encoding="utf-8")
        create_section = controller.split('"docker", "container", "create"', 1)[1].split(
            "command.extend", 1
        )[0]
        self.assertNotIn("docker.sock", create_section)


if __name__ == "__main__":
    unittest.main()

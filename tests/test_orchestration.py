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
    _safe_docker_error_reason,
    _safe_relative,
    _runtime_dns_from_environment,
    _selected_gpu_from_environment,
    _validate_selected_gpu,
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
                runtime_dns=("192.0.2.53", "2001:db8::53"),
            )
            create = docker.calls[0]
            self.assertIn("device=1", create)
            self.assertFalse(any("docker.sock" in item for item in create))
            self.assertTrue(any("/workspace/project,readonly" in item for item in create))
            self.assertTrue(any("/cache/huggingface" in item for item in create))
            dns_pairs = [
                create[index + 1] for index, item in enumerate(create) if item == "--dns"
            ]
            self.assertEqual(dns_pairs, ["192.0.2.53", "2001:db8::53"])
            self.assertLess(
                next(i for i, call in enumerate(docker.calls) if call[2] == "wait"),
                len(docker.calls),
            )
            serving = FakeServingManager.instances[0]
            self.assertEqual(serving.calls[0]["execution_id"], execution)
            self.assertEqual(
                serving.calls[0]["runtime_dns"], ("192.0.2.53", "2001:db8::53")
            )
            self.assertEqual(serving.calls[0]["config"].restart_policy, "unless-stopped")
            self.assertEqual(serving.calls[0]["config"].vllm.gpu_devices, "0")
            self.assertEqual(serving.cleaned, [])
            value = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "success")
            self.assertEqual(value["serving"]["container_id"], "serving-id")
            self.assertEqual(value["training"]["gpu_devices"], {
                "configured": "1", "effective": "1", "admission_override": None,
            })
            self.assertEqual(value["serving"]["configured_gpu_devices"], "0")
            self.assertIsNone(value["serving"]["admission_override"])

    def test_selected_gpu_overrides_training_and_serving_with_provenance(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            layout = self._layout(root)
            raw = config_value()
            raw["orchestration"]["training"]["gpu_devices"] = "0,2"
            raw["serving"]["vllm"] = {"gpu_devices": "2,3"}
            (root / "project" / "config.yaml").write_text(
                json.dumps(raw), encoding="utf-8"
            )
            execution = "exec-12345678"
            docker = FakeDocker(root, execution)
            manifest = PipelineController(
                docker=docker, serving_manager_factory=FakeServingManager
            ).deploy(
                execution_id=execution, config_relative="config.yaml", layout=layout,
                training_image="trainer:test", training_image_id="sha256:trainer",
                source_revision="revision", source_identity="revision", selected_gpu="1",
            )
            create = docker.calls[0]
            self.assertEqual(create[create.index("--gpus") + 1], "device=1")
            self.assertFalse(any("CUDA_VISIBLE_DEVICES" in item for item in create))
            serving_config = FakeServingManager.instances[0].calls[0]["config"]
            self.assertEqual(serving_config.vllm.gpu_devices, "1")
            value = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(value["training"]["gpu_devices"], {
                "configured": "0,2", "effective": "1", "admission_override": "1",
            })
            self.assertEqual(value["serving"]["configured_gpu_devices"], "2,3")
            self.assertEqual(value["serving"]["gpu_devices"], "1")
            self.assertEqual(value["serving"]["admission_override"], "1")

    def test_legacy_multi_gpu_config_is_preserved_without_override(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            layout = self._layout(root)
            raw = config_value()
            raw["orchestration"]["training"]["gpu_devices"] = "0,2"
            raw["serving"]["vllm"] = {"gpu_devices": "1,3"}
            (root / "project" / "config.yaml").write_text(
                json.dumps(raw), encoding="utf-8"
            )
            execution = "exec-12345678"
            docker = FakeDocker(root, execution)
            PipelineController(
                docker=docker, serving_manager_factory=FakeServingManager
            ).deploy(
                execution_id=execution, config_relative="config.yaml", layout=layout,
                training_image="trainer:test", training_image_id="sha256:trainer",
                source_revision="revision", source_identity="revision",
            )
            create = docker.calls[0]
            self.assertEqual(create[create.index("--gpus") + 1], "device=0,2")
            self.assertEqual(
                FakeServingManager.instances[0].calls[0]["config"].vllm.gpu_devices,
                "1,3",
            )

    def test_invalid_selected_gpu_fails_before_any_docker_mutation(self):
        for value in ("", "-1", "0,1", " 1", "1 ", "gpu1", "$(id)"):
            with self.subTest(value=value), tempfile.TemporaryDirectory(
                dir=TESTS_DIR
            ) as temporary:
                root = Path(temporary)
                layout = self._layout(root)
                docker = FakeDocker(root, "exec-12345678")
                with self.assertRaisesRegex(ControllerError, "PIPELINE_SELECTED_GPU"):
                    PipelineController(
                        docker=docker, serving_manager_factory=FakeServingManager
                    ).deploy(
                        execution_id="exec-12345678", config_relative="config.yaml",
                        layout=layout, training_image="trainer:test",
                        training_image_id="sha256:trainer", source_revision="revision",
                        source_identity="revision", selected_gpu=value,
                    )
                self.assertEqual(docker.calls, [])

    def test_unavailable_selected_port_fails_before_training_container_mutation(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            layout = self._layout(root)
            raw = config_value()
            raw["serving"].update({
                "port": "auto", "port_range": {"start": 8101, "end": 8199}
            })
            (root / "project" / "config.yaml").write_text(
                json.dumps(raw), encoding="utf-8"
            )
            docker = FakeDocker(root, "exec-12345678")
            with patch.dict(
                "os.environ", {"PIPELINE_SELECTED_PORT": "8107"}, clear=True
            ), patch(
                "fine_tuning_pipeline.orchestration.controller.port_is_available",
                return_value=False,
            ), self.assertRaisesRegex(ControllerError, "training was not started"):
                PipelineController(
                    docker=docker, serving_manager_factory=FakeServingManager
                ).deploy(
                    execution_id="exec-12345678", config_relative="config.yaml",
                    layout=layout, training_image="trainer:test",
                    training_image_id="sha256:trainer", source_revision="revision",
                    source_identity="revision",
                )
            self.assertEqual(docker.calls, [])
            self.assertEqual(FakeServingManager.instances, [])

    def test_fixed_port_ignores_ambient_selected_port_for_admission_recheck(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            layout = self._layout(root)
            execution = "exec-12345678"
            docker = FakeDocker(root, execution)
            with patch.dict(
                "os.environ", {"PIPELINE_SELECTED_PORT": "8199"}, clear=True
            ), patch(
                "fine_tuning_pipeline.orchestration.controller.port_is_available",
                return_value=False,
            ) as checker:
                PipelineController(
                    docker=docker, serving_manager_factory=FakeServingManager
                ).deploy(
                    execution_id=execution,
                    config_relative="config.yaml",
                    layout=layout,
                    training_image="trainer:test",
                    training_image_id="sha256:trainer",
                    source_revision="revision",
                    source_identity="revision",
                )
            checker.assert_not_called()
            serving = FakeServingManager.instances[0].calls[0]["config"]
            self.assertEqual(serving.port, 8101)
            self.assertIsNone(serving.port_range)

    def test_selected_port_reaches_serving_and_deployment_manifest(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            layout = self._layout(root)
            raw = config_value()
            raw["serving"].update({
                "port": "auto", "port_range": {"start": 8101, "end": 8199}
            })
            (root / "project" / "config.yaml").write_text(
                json.dumps(raw), encoding="utf-8"
            )
            execution = "exec-12345678"
            docker = FakeDocker(root, execution)
            with patch.dict(
                "os.environ", {"PIPELINE_SELECTED_PORT": "8107"}, clear=True
            ), patch(
                "fine_tuning_pipeline.orchestration.controller.port_is_available",
                return_value=True,
            ):
                manifest = PipelineController(
                    docker=docker, serving_manager_factory=FakeServingManager
                ).deploy(
                    execution_id=execution, config_relative="config.yaml", layout=layout,
                    training_image="trainer:test", training_image_id="sha256:trainer",
                    source_revision="revision", source_identity="revision",
                )
            serving = FakeServingManager.instances[0].calls[0]["config"]
            self.assertEqual(serving.port, 8107)
            self.assertEqual(serving.base_url, "http://model-server.example:8107/v1")
            value = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(value["serving"]["base_url"], serving.base_url)

    def test_selected_gpu_environment_is_optional_normalized_and_strict(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(_selected_gpu_from_environment())
        with patch.dict("os.environ", {"PIPELINE_SELECTED_GPU": "001"}, clear=True):
            self.assertEqual(_selected_gpu_from_environment(), "1")
        self.assertEqual(_validate_selected_gpu("0"), "0")
        self.assertEqual(_validate_selected_gpu("0" * 5000 + "1"), "1")
        for value in ("", "-1", "0,1", " 1", "1 ", "gpu"):
            with self.subTest(value=value), self.assertRaises(ControllerError):
                _validate_selected_gpu(value)

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

    def test_create_failure_includes_redacted_bounded_docker_reason(self):
        secret = "controller-test-secret"
        api_error = (
            "Error response from daemon: client version 1.41 is too old. "
            "Minimum supported API version is 1.44; api_key=" + secret
        )
        reason = _safe_docker_error_reason(
            DockerResult(1, "", api_error), secrets=(secret,)
        )
        self.assertIn("client version 1.41 is too old", reason)
        self.assertIn("Minimum supported API version is 1.44", reason)
        self.assertNotIn(secret, reason)
        self.assertIn("<redacted>", reason)
        self.assertLessEqual(len(reason), 320)

        class CreateFailureDocker:
            def __init__(self):
                self.calls = []

            def run(self, command, **_kwargs):
                command = tuple(str(item) for item in command)
                self.calls.append(command)
                if command[:3] == ("docker", "container", "create"):
                    return DockerResult(1, "", api_error)
                if command[:3] == ("docker", "container", "inspect"):
                    return DockerResult(1, "", "not found")
                raise AssertionError(command)

        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            layout = self._layout(root)
            docker = CreateFailureDocker()
            with self.assertRaises(ControllerError) as raised:
                PipelineController(
                    docker=docker, serving_manager_factory=FakeServingManager
                ).deploy(
                    execution_id="exec-12345678", config_relative="config.yaml",
                    layout=layout, training_image="trainer:test",
                    training_image_id="sha256:trainer",
                    source_revision="revision", source_identity="revision",
                )
            message = str(raised.exception)
            self.assertIn("client version 1.41 is too old", message)
            self.assertIn("no serving or gateway mutation occurred", message)
            self.assertNotIn(secret, message)
            self.assertEqual(FakeServingManager.instances, [])

    def test_runtime_dns_environment_is_revalidated_without_shell_interpretation(self):
        with patch.dict(
            "os.environ",
            {"PIPELINE_DOCKER_RUNTIME_DNS": "192.0.2.53 2001:DB8::53 192.0.2.53"},
            clear=True,
        ):
            self.assertEqual(
                _runtime_dns_from_environment(), ("192.0.2.53", "2001:db8::53")
            )
        for value in (
            "", "127.0.0.53", "::1", "::ffff:127.0.0.1",
            "192.0.2.53;touch", "resolver.example",
        ):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"PIPELINE_DOCKER_RUNTIME_DNS": value}, clear=True
            ), self.assertRaises(ControllerError):
                _runtime_dns_from_environment()


class LauncherStaticTests(unittest.TestCase):
    def test_unified_preflight_order_precedes_controller_and_training(self):
        text = (TESTS_DIR.parent / "scripts" / "run_pipeline.sh").read_text(
            encoding="utf-8"
        )
        estimate = text.index(
            "    run_resource_preflight_helper estimate none estimate.env"
        )
        gpu = text.index(
            '    GPU_PREFLIGHT_ENV_FILE=$selected_gpu_env sh "$PROJECT_DIR/scripts/gpu_preflight.sh"',
            estimate,
        )
        port = text.index(
            "    run_resource_preflight_helper select-port host port.env", gpu
        )
        controller = text.index(
            "set -- docker run --rm --read-only --network host \\", port
        )
        self.assertLess(estimate, gpu)
        self.assertLess(gpu, port)
        self.assertLess(port, controller)

    def test_launcher_requires_no_host_python_and_keeps_secrets_out_of_arguments(self):
        text = (TESTS_DIR.parent / "scripts" / "run_pipeline.sh").read_text(encoding="utf-8")
        self.assertIn("docker info", text)
        self.assertIn("fine-tuning-pipeline.source-identity", text)
        self.assertIn("--rebuild", text)
        self.assertNotIn("command -v python", text.lower())
        self.assertIn(
            '"$CONTROLLER_IMAGE" python -m fine_tuning_pipeline.resource_preflight',
            text,
        )
        self.assertNotIn('--env "LITELLM_API_KEY=', text)
        self.assertIn("dst=/run/pipeline-secrets,readonly", text)
        self.assertIn(
            "--mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock",
            text,
        )

    def test_controller_image_is_minimal_and_trainer_never_receives_socket(self):
        dockerfiles = [
            (TESTS_DIR.parent / "docker" / name).read_text(encoding="utf-8")
            for name in ("Dockerfile.controller", "Dockerfile.controller.host-dns")
        ]
        for dockerfile in dockerfiles:
            self.assertNotIn("docker.io", dockerfile)
            self.assertIn(
                "ARG DOCKER_CLI_IMAGE=docker:27.5.1-cli@sha256:"
                "851f91d241214e7c6db86513b270d58776379aacc5eb9c4a87e5b47115e3065c",
                dockerfile,
            )
            self.assertIn(
                "COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker",
                dockerfile,
            )
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

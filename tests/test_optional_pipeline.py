import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

from fine_tuning_pipeline.gateway.config import GatewayConfigError
from fine_tuning_pipeline.serving.config import ServingConfigError
from fine_tuning_pipeline.train_pipeline import DEFAULT_CONFIG_PATH, execute_training, prepare_training


TESTS_DIR = Path(__file__).resolve().parent
FAKE_SECRET = "pipeline-fake-secret-never-write"


def create_adapter(yaml_file, log_file=None, **_kwargs):
    arguments = yaml.safe_load(Path(yaml_file).read_text(encoding="utf-8"))
    model_dir = Path(arguments["output_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "adapter_config.json").write_text(
        json.dumps({"r": 8, "base_model_name_or_path": arguments["model_name_or_path"]}),
        encoding="utf-8",
    )
    (model_dir / "adapter_model.safetensors").write_bytes(b"weights")


class SuccessfulServing:
    def start_and_verify(self, *, run_root, **_kwargs):
        directory = run_root / "serving"
        directory.mkdir(exist_ok=True)
        (directory / "endpoint_manifest.json").write_text("{}", encoding="utf-8")
        (directory / "health_check.json").write_text("{}", encoding="utf-8")
        (directory / "operation.json").write_text("{}", encoding="utf-8")
        (directory / "serving.log").write_text("ready", encoding="utf-8")
        return {
            "manifest_path": "serving/endpoint_manifest.json",
            "health_path": "serving/health_check.json",
            "operation_path": "serving/operation.json",
            "log_path": "serving/serving.log",
        }


class SuccessfulGateway:
    def __init__(self, config): self.config = config
    def register_and_verify(self, *, run_root, **_kwargs):
        directory = run_root / "gateway"
        directory.mkdir(exist_ok=True)
        for name in ("gateway_manifest.json", "registration.json", "health_check.json"):
            (directory / name).write_text("{}", encoding="utf-8")
        (directory / "gateway.log").write_text("ready", encoding="utf-8")
        return {
            "manifest_path": "gateway/gateway_manifest.json",
            "registration_path": "gateway/registration.json",
            "health_path": "gateway/health_check.json",
            "log_path": "gateway/gateway.log",
        }


class OptionalPipelineTests(unittest.TestCase):
    def setUp(self):
        model_patcher = patch(
            "fine_tuning_pipeline.model_manager._load_huggingface_config",
            return_value={"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
        )
        model_patcher.start()
        self.addCleanup(model_patcher.stop)

    def temporary_project(self):
        temporary = tempfile.TemporaryDirectory(dir=TESTS_DIR)
        root = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["dataset"]["path"] = str(
            DEFAULT_CONFIG_PATH.parents[1] / "examples" / "datasets" / "alpaca_demo.json"
        )
        return root, config

    def write_config(self, root, config):
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_missing_and_disabled_serving_never_construct_manager(self):
        for serving in (None, {"enabled": False}):
            with self.subTest(serving=serving):
                root, config = self.temporary_project()
                if serving is not None:
                    config["serving"] = serving
                config_path = self.write_config(root, config)
                with patch(
                    "fine_tuning_pipeline.train_pipeline.run_training", side_effect=create_adapter
                ):
                    run_dir = execute_training(
                        config_path,
                        runs_root=root / "runs",
                        serving_manager_factory=lambda: self.fail("serving manager constructed"),
                    )
                metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "success")
                self.assertNotIn("endpoint_handoff", metadata["output"])
                self.assertFalse((run_dir / "serving").exists())

    def test_all_three_modes_and_preferred_handoff(self):
        root, config = self.temporary_project()
        original_training = dict(config["training"])
        config["serving"] = {
            "enabled": True,
            "advertise_host": "model-server.example",
            "base_model_name": "served-base",
            "fine_tuned_model_name": "served-fine",
        }
        serving_only = self.write_config(root, config)
        with patch("fine_tuning_pipeline.train_pipeline.run_training", side_effect=create_adapter):
            first = execute_training(
                serving_only,
                runs_root=root / "serving-runs",
                serving_manager_factory=SuccessfulServing,
                gateway_manager_factory=lambda _config: self.fail("gateway manager constructed"),
            )
        first_metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(first_metadata["phases"]["training"]["status"], "success")
        self.assertEqual(first_metadata["phases"]["serving"]["status"], "ready")
        self.assertNotIn("status", first_metadata["training"])
        self.assertEqual(first_metadata["training"], original_training)
        self.assertEqual(first_metadata["output"]["endpoint_handoff"]["source"], "serving")

        config["gateway"] = {
            "enabled": True,
            "provider": "litellm",
            "base_url": "${LITELLM_BASE_URL}",
            "api_key": "${LITELLM_API_KEY}",
            "registration": {
                "mode": "dynamic_db",
                "base_model_name": "gateway-base",
                "fine_tuned_model_name": "gateway-fine",
            },
        }
        gateway_path = self.write_config(root, config)
        with patch.dict(
            "os.environ",
            {"LITELLM_BASE_URL": "https://gateway.example", "LITELLM_API_KEY": FAKE_SECRET},
            clear=False,
        ), patch("fine_tuning_pipeline.train_pipeline.run_training", side_effect=create_adapter):
            second = execute_training(
                gateway_path,
                runs_root=root / "gateway-runs",
                serving_manager_factory=SuccessfulServing,
                gateway_manager_factory=SuccessfulGateway,
            )
        second_metadata = json.loads((second / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(second_metadata["phases"]["gateway"]["status"], "ready")
        self.assertEqual(second_metadata["output"]["endpoint_handoff"]["source"], "gateway")
        all_text = "".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in second.rglob("*") if path.is_file()
        )
        self.assertNotIn(FAKE_SECRET, all_text)
        self.assertIn("${LITELLM_API_KEY}", all_text)

    def test_container_training_only_does_not_resolve_gateway_secret_or_start_managers(self):
        root, config = self.temporary_project()
        config["serving"] = {
            "enabled": True, "advertise_host": "model-server.example"
        }
        config["gateway"] = {
            "enabled": True, "base_url": "https://gateway.example",
            "api_key": "${LITELLM_API_KEY}",
        }
        with patch.dict("os.environ", {}, clear=True), patch(
            "fine_tuning_pipeline.train_pipeline.run_training", side_effect=create_adapter
        ):
            run_dir = execute_training(
                self.write_config(root, config), runs_root=root / "runs",
                enable_optional_phases=False,
                serving_manager_factory=lambda: self.fail("serving manager constructed"),
                gateway_manager_factory=lambda _config: self.fail("gateway manager constructed"),
            )
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "success")
        self.assertFalse((run_dir / "serving").exists())

    def test_serving_and_gateway_failures_preserve_successful_training_phase(self):
        root, config = self.temporary_project()
        config["serving"] = {"enabled": True, "advertise_host": "model-server.example"}
        config_path = self.write_config(root, config)

        class FailedServing:
            def start_and_verify(self, **_kwargs): raise RuntimeError("readiness timeout")

        serving_console = io.StringIO()
        with patch("fine_tuning_pipeline.train_pipeline.run_training", side_effect=create_adapter), redirect_stdout(serving_console):
            with self.assertRaisesRegex(RuntimeError, "readiness timeout"):
                execute_training(
                    config_path, runs_root=root / "serving-failed", serving_manager_factory=FailedServing
                )
        self.assertIn("Training succeeded; serving failed.", serving_console.getvalue())
        self.assertNotIn("=== Training Failed ===", serving_console.getvalue())
        run_dir = next((root / "serving-failed").iterdir())
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["phases"]["training"]["status"], "success")
        self.assertEqual(metadata["phases"]["serving"]["status"], "failed")
        self.assertNotIn("status", metadata["training"])
        self.assertTrue((run_dir / "model" / "adapter_model.safetensors").is_file())

        config["gateway"] = {
            "enabled": True,
            "base_url": "https://gateway.example",
            "api_key": "${LITELLM_API_KEY}",
        }
        config_path = self.write_config(root, config)

        class FailedGateway:
            def __init__(self, _config): pass
            def register_and_verify(self, **_kwargs): raise RuntimeError("gateway verification failed")

        gateway_console = io.StringIO()
        with patch.dict("os.environ", {"LITELLM_API_KEY": FAKE_SECRET}, clear=False), patch(
            "fine_tuning_pipeline.train_pipeline.run_training", side_effect=create_adapter
        ), redirect_stdout(gateway_console):
            with self.assertRaisesRegex(RuntimeError, "gateway verification failed"):
                execute_training(
                    config_path,
                    runs_root=root / "gateway-failed",
                    serving_manager_factory=SuccessfulServing,
                    gateway_manager_factory=FailedGateway,
                )
        self.assertIn(
            "Training and serving succeeded; gateway failed.", gateway_console.getvalue()
        )
        self.assertNotIn("=== Training Failed ===", gateway_console.getvalue())
        run_dir = next((root / "gateway-failed").iterdir())
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["phases"]["training"]["status"], "success")
        self.assertEqual(metadata["phases"]["serving"]["status"], "ready")
        self.assertEqual(metadata["phases"]["gateway"]["status"], "failed")

    def test_unsupported_full_serving_and_gateway_without_serving_fail_during_preparation(self):
        root, config = self.temporary_project()
        config["training"]["method"] = "full"
        del config["training"]["lora"]
        config["serving"] = {"enabled": True}
        with self.assertRaises(ServingConfigError):
            prepare_training(self.write_config(root, config), artifact_root=root / "full-runs")

        root, config = self.temporary_project()
        config["gateway"] = {"enabled": True}
        with self.assertRaises(GatewayConfigError):
            prepare_training(self.write_config(root, config), artifact_root=root / "gateway-runs")

    def test_literal_secret_is_redacted_even_from_failed_input_snapshot_and_exception(self):
        root, config = self.temporary_project()
        literal = "literal-secret-must-disappear"
        config["serving"] = {
            "enabled": True,
            "advertise_host": "model-server.example",
        }
        config["gateway"] = {
            "enabled": True,
            "base_url": "https://gateway.example",
            "api_key": literal,
        }
        with self.assertRaises(GatewayConfigError) as caught:
            prepare_training(self.write_config(root, config), artifact_root=root / "runs")
        self.assertNotIn(literal, str(caught.exception))
        run_dir = next((root / "runs").iterdir())
        all_text = "".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in run_dir.rglob("*") if path.is_file()
        )
        self.assertNotIn(literal, all_text)

    def test_explicit_insecure_gateway_warns_without_exposing_credentials(self):
        root, config = self.temporary_project()
        config["serving"] = {
            "enabled": True,
            "advertise_host": "model-server.internal",
        }
        config["gateway"] = {
            "enabled": True,
            "base_url": "http://litellm.internal:4000",
            "api_key": "${LITELLM_API_KEY}",
            "allow_insecure_http": True,
        }
        with patch.dict("os.environ", {"LITELLM_API_KEY": FAKE_SECRET}, clear=False):
            prepare_training(
                self.write_config(root, config), artifact_root=root / "runs"
            )
        run_dir = next((root / "runs").iterdir())
        combined = "".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in run_dir.rglob("*") if path.is_file()
        )
        self.assertIn("explicitly permitted insecure HTTP", combined)
        self.assertNotIn(FAKE_SECRET, combined)
        self.assertIn("${LITELLM_API_KEY}", combined)


if __name__ == "__main__":
    unittest.main()

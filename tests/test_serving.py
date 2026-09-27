import json
import tempfile
import unittest
from pathlib import Path

from fine_tuning_pipeline.http_client import JsonResponse
from fine_tuning_pipeline.serving.backends.vllm_docker import (
    CommandResult,
    VllmDockerBackend,
    build_vllm_docker_command,
)
from fine_tuning_pipeline.serving.config import ServingConfigError, resolve_serving_config
from fine_tuning_pipeline.serving.contracts import (
    ServingConflictError,
    ServingLaunchRequest,
    ServingLaunchResult,
    ServingVerificationError,
)
from fine_tuning_pipeline.serving.health import OpenAIEndpointVerifier
from fine_tuning_pipeline.serving.manager import ServingManager


TESTS_DIR = Path(__file__).resolve().parent


def serving_config(**updates):
    raw = {
        "enabled": True,
        "backend": "vllm",
        "runtime": "docker",
        "bind_host": "0.0.0.0",
        "advertise_host": "model.example",
        "port": 8101,
        "base_model_name": "example-base",
        "fine_tuned_model_name": "example-finetuned",
        "container_name": "example-serving",
        "vllm": {
            "image": "vllm/vllm-openai:v0.11.0",
            "gpu_devices": "0",
            "gpu_memory_utilization": 0.15,
            "max_model_len": 4096,
            "max_num_seqs": 2,
        },
        "health_check": {"enabled": True, "timeout_seconds": 10, "interval_seconds": 1},
    }
    raw.update(updates)
    return resolve_serving_config(
        raw, run_id="20260927_120000_example", base_model="Org/Base-Model", training_method="lora"
    )


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ServingConfigTests(unittest.TestCase):
    def test_missing_and_disabled_sections_preserve_train_only_mode(self):
        missing = resolve_serving_config(
            None, run_id="run", base_model="Org/Model", training_method="full"
        )
        disabled = resolve_serving_config(
            {"enabled": False}, run_id="run", base_model="Org/Model", training_method="full"
        )
        self.assertFalse(missing.enabled)
        self.assertFalse(disabled.enabled)

    def test_auto_names_are_safe_deterministic_and_traceable(self):
        raw = {"enabled": True}
        first = resolve_serving_config(
            raw, run_id="Run ID / Unsafe", base_model="Org/Base Model!", training_method="lora"
        )
        second = resolve_serving_config(
            raw, run_id="Run ID / Unsafe", base_model="Org/Base Model!", training_method="lora"
        )
        self.assertEqual(first, second)
        for value in (first.base_model_name, first.fine_tuned_model_name, first.container_name):
            self.assertRegex(value, r"^[a-z0-9][a-z0-9._-]+$")
        self.assertIn("run-id-unsafe", first.base_model_name)

    def test_strict_validation_rejects_unsupported_and_invalid_values(self):
        cases = (
            ({"backend": "tgi"}, "serving.backend"),
            ({"runtime": "kubernetes"}, "serving.runtime"),
            ({"port": 70000}, "serving.port"),
            ({"vllm": {"gpu_devices": "all"}}, "gpu_devices"),
            ({"vllm": {"gpu_devices": "0,0"}}, "duplicates"),
            ({"vllm": {"gpu_memory_utilization": 0}}, "gpu_memory_utilization"),
            ({"vllm": {"max_model_len": 0}}, "max_model_len"),
            ({"vllm": {"max_num_seqs": 0}}, "max_num_seqs"),
            ({"container_name": "bad name"}, "container_name"),
            ({"health_check": {"enabled": False}}, "requires health_check.enabled"),
            ({"unexpected": True}, "Unsupported serving"),
        )
        for update, message in cases:
            with self.subTest(update=update):
                raw = {"enabled": True, **update}
                with self.assertRaisesRegex(ServingConfigError, message):
                    resolve_serving_config(
                        raw, run_id="run", base_model="Org/Model", training_method="lora"
                    )

    def test_full_training_auto_serving_is_rejected(self):
        with self.assertRaisesRegex(ServingConfigError, "qualified only.*lora"):
            resolve_serving_config(
                {"enabled": True}, run_id="run", base_model="Org/Model", training_method="full"
            )

    def test_advertise_host_builds_provider_neutral_url(self):
        config = serving_config(advertise_host="10.20.30.40", port=9000)
        self.assertEqual(config.base_url, "http://10.20.30.40:9000/v1")


class DockerBackendTests(unittest.TestCase):
    def request(self, revision="abc123"):
        config = serving_config()
        return ServingLaunchRequest(
            "run-1", "Org/Base-Model", revision, Path("adapter"), 64, config
        )

    def test_command_loads_base_once_and_static_lora_in_one_process(self):
        command = build_vllm_docker_command(self.request())
        self.assertEqual(command.count("docker"), 1)
        self.assertEqual(command.count("run"), 1)
        self.assertEqual(command.count("Org/Base-Model"), 1)
        self.assertIn("--served-model-name", command)
        self.assertIn("example-base", command)
        self.assertIn("--enable-lora", command)
        self.assertIn("example-finetuned=/adapters/fine-tuned", command)
        self.assertEqual(command[command.index("--max-lora-rank") + 1], "64")
        self.assertIn("--revision", command)
        self.assertIn("abc123", command)
        self.assertTrue(any("adapter" in item and ":/adapters/fine-tuned:ro" in item for item in command))
        self.assertFalse(any(item in {"stop", "rm", "kill"} for item in command))
        self.assertEqual(command[command.index("--restart") + 1], "no")
        self.assertIn("fine-tuning-pipeline.managed=true", command)

    def test_command_omits_revision_only_when_resolution_is_unavailable(self):
        self.assertNotIn("--revision", build_vllm_docker_command(self.request(None)))

    def test_occupied_port_fails_before_docker_mutation(self):
        class Runner:
            def __init__(self): self.commands = []
            def run(self, command):
                self.commands.append(tuple(command))
                return CommandResult(0, "1", "")

        runner = Runner()
        backend = VllmDockerBackend(runner=runner, port_checker=lambda *_: False)
        with self.assertRaisesRegex(ServingConflictError, "Port 8101"):
            backend.preflight(self.request())
        self.assertEqual(len(runner.commands), 1)
        self.assertNotIn("run", runner.commands[0])

    def test_existing_container_name_fails_without_stop_or_remove(self):
        class Runner:
            def __init__(self): self.commands = []
            def run(self, command):
                self.commands.append(tuple(command))
                return CommandResult(0, "example-serving\n", "")

        runner = Runner()
        backend = VllmDockerBackend(runner=runner, port_checker=lambda *_: True)
        with self.assertRaisesRegex(ServingConflictError, "already exists"):
            backend.preflight(self.request())
        flattened = " ".join(" ".join(command) for command in runner.commands)
        self.assertNotRegex(flattened, r"\b(stop|rm|restart|kill)\b")

    def test_docker_unavailable_is_actionable(self):
        class Runner:
            def run(self, command): return CommandResult(1, "", "daemon unavailable")

        with self.assertRaisesRegex(Exception, "Docker is unavailable"):
            VllmDockerBackend(runner=Runner()).preflight(self.request())

    def test_cache_restart_and_execution_ownership_are_passed_without_shell_quoting(self):
        request = ServingLaunchRequest(
            "run-1", "Org/Base-Model", "abc123", Path("adapter path"), 8,
            serving_config(restart_policy="unless-stopped"),
            execution_id="exec-1", host_hf_cache_path=Path("cache path"),
        )
        command = build_vllm_docker_command(request)
        self.assertEqual(command[command.index("--restart") + 1], "unless-stopped")
        self.assertIn("fine-tuning-pipeline.execution-id=exec-1", command)
        self.assertTrue(any("cache path" in item and ":/root/.cache/huggingface" in item for item in command))
        self.assertTrue(any("adapter path" in item for item in command))

    def test_cleanup_requires_exact_id_and_all_ownership_labels(self):
        request = ServingLaunchRequest(
            "run-1", "Org/Base", None, Path("adapter"), 8, serving_config(),
            execution_id="exec-1",
        )

        class Runner:
            def __init__(self, inspection): self.inspection = inspection; self.commands = []
            def run(self, command):
                self.commands.append(tuple(command))
                if "inspect" in command:
                    return CommandResult(0, self.inspection, "")
                return CommandResult(0, "container-id", "")

        wrong = Runner("container-id|true|other-run|exec-1\n")
        self.assertFalse(VllmDockerBackend(runner=wrong).remove_if_owned(request, "container-id"))
        self.assertFalse(any("rm" in command for command in wrong.commands))
        owned = Runner("container-id|true|run-1|exec-1\n")
        self.assertTrue(VllmDockerBackend(runner=owned).remove_if_owned(request, "container-id"))
        self.assertEqual(owned.commands[-1][:3], ("docker", "container", "rm"))

    def test_ambiguous_run_response_reconciles_only_exact_owned_container(self):
        request = ServingLaunchRequest(
            "run-1", "Org/Base", None, Path("adapter"), 8, serving_config(),
            execution_id="exec-1",
        )

        class Runner:
            def __init__(self): self.commands = []
            def run(self, command):
                self.commands.append(tuple(command))
                if command[:2] == ("docker", "run"):
                    return CommandResult(1, "", "connection reset")
                if "container" in command and "inspect" in command:
                    return CommandResult(0, "owned-id|true|run-1|exec-1\n", "")
                if "image" in command and "inspect" in command:
                    return CommandResult(0, "sha256:image\n", "")
                raise AssertionError(command)

        result = VllmDockerBackend(runner=Runner()).launch(request)
        self.assertEqual(result.container_id, "owned-id")
        self.assertEqual(result.image_id, "sha256:image")


class ServingHealthTests(unittest.TestCase):
    def success_responses(self, base="base", fine="fine"):
        models = JsonResponse(200, {"data": [{"id": base}, {"id": fine}]})
        completion = JsonResponse(200, {"choices": [{"message": {"content": "OK"}}]})
        return [models, completion, completion]

    def test_models_and_both_inference_aliases_are_verified(self):
        transport = FakeTransport(self.success_responses())
        result = OpenAIEndpointVerifier(transport=transport).verify(
            base_url="http://server/v1", base_model="base", fine_tuned_model="fine",
            timeout_seconds=1, interval_seconds=0.01, wait_for_readiness=False,
        )
        self.assertEqual(result.status, "ready")
        payload_models = [call[2]["payload"]["model"] for call in transport.calls[1:]]
        self.assertEqual(payload_models, ["base", "fine"])

    def test_missing_alias_and_bad_inference_are_distinct_failures(self):
        with self.assertRaisesRegex(ServingVerificationError, "missing alias.*fine"):
            OpenAIEndpointVerifier(
                transport=FakeTransport([JsonResponse(200, {"data": [{"id": "base"}]})])
            ).verify(
                base_url="http://server/v1", base_model="base", fine_tuned_model="fine",
                timeout_seconds=1, interval_seconds=1, wait_for_readiness=False,
            )
        with self.assertRaisesRegex(ServingVerificationError, "model 'fine'"):
            responses = self.success_responses()
            responses[-1] = JsonResponse(500, {"error": "failed"})
            OpenAIEndpointVerifier(transport=FakeTransport(responses)).verify(
                base_url="http://server/v1", base_model="base", fine_tuned_model="fine",
                timeout_seconds=1, interval_seconds=1, wait_for_readiness=False,
            )

    def test_readiness_timeout_is_actionable(self):
        transport = FakeTransport([JsonResponse(503, {})])
        with self.assertRaisesRegex(ServingVerificationError, "readiness timed out"):
            OpenAIEndpointVerifier(transport=transport).verify(
                base_url="http://server/v1", base_model="base", fine_tuned_model="fine",
                timeout_seconds=1, interval_seconds=1, wait_for_readiness=False,
            )

    def test_manager_writes_secret_free_provider_neutral_artifacts(self):
        class Backend:
            def preflight(self, request): self.request = request
            def launch(self, request):
                return ServingLaunchResult("container-id", ("docker", "run"))

        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            result = ServingManager(
                backend=Backend(),
                verifier=OpenAIEndpointVerifier(
                    transport=FakeTransport(
                        self.success_responses("example-base", "example-finetuned")
                    )
                ),
            ).start_and_verify(
                run_id="run", run_root=root, config=serving_config(), base_model="Org/Base",
                resolved_revision="revision", adapter_path=root / "model", lora_rank=64,
            )
            manifest = json.loads((root / result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["base_url"], "http://model.example:8101/v1")
            self.assertEqual(manifest["models"]["fine_tuned"]["name"], "example-finetuned")
            operation = json.loads((root / "serving" / "operation.json").read_text(encoding="utf-8"))
            self.assertEqual(operation["status"], "ready")
            self.assertEqual(operation["container_id"], "container-id")
            combined = "".join(path.read_text(encoding="utf-8") for path in (root / "serving").iterdir())
            self.assertNotIn("api_key", combined.lower())

    def test_manager_persists_owned_container_when_readiness_fails(self):
        class Backend:
            def preflight(self, request): pass
            def launch(self, request):
                return ServingLaunchResult("owned-container-id", ("docker", "run"))

        class FailedVerifier:
            def verify(self, **_kwargs):
                raise ServingVerificationError("readiness failed")

        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ServingVerificationError, "readiness failed"):
                ServingManager(backend=Backend(), verifier=FailedVerifier()).start_and_verify(
                    run_id="run-owned",
                    run_root=root,
                    config=serving_config(),
                    base_model="Org/Base",
                    resolved_revision="revision",
                    adapter_path=root / "model",
                    lora_rank=32,
                )
            operation = json.loads(
                (root / "serving" / "operation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(operation["status"], "failed")
            self.assertEqual(operation["container_id"], "owned-container-id")
            self.assertEqual(
                operation["ownership_label"], "fine-tuning-pipeline.run-id=run-owned"
            )


if __name__ == "__main__":
    unittest.main()

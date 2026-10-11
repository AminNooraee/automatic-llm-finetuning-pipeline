import tempfile
from dataclasses import replace
import unittest
from pathlib import Path

import yaml

from fine_tuning_pipeline.resource_preflight import (
    ResourcePolicy,
    ModelArchitecture,
    ServingMemoryPolicy,
    estimate_serving_memory,
    ResourcePreflightError,
    estimate_required_vram,
    load_and_estimate,
    select_available_port,
)
from fine_tuning_pipeline.serving.config import resolve_serving_config
from fine_tuning_pipeline.training_config import resolve_training_config


TESTS_DIR = Path(__file__).resolve().parent


def training_config(**updates):
    raw = {
        "method": "lora",
        "epochs": 1,
        "learning_rate": 0.0002,
        "batch_size": 1,
        "precision": "bf16",
        "gradient_checkpointing": False,
        "cutoff_len": 2048,
        "save_steps": 10,
        "logging_steps": 1,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0},
    }
    raw.update(updates)
    if raw["method"] == "full":
        raw.pop("lora", None)
    return resolve_training_config(raw)


def serving_config(utilization=0.15):
    return resolve_serving_config(
        {
            "enabled": True,
            "advertise_host": "model.example",
            "port": 8101,
            "vllm": {"gpu_memory_utilization": utilization},
        },
        run_id="estimate",
        base_model="Org/Model",
        training_method="lora",
    )


class ResourceEstimatorTests(unittest.TestCase):
    policy = ResourcePolicy(True, 490_000_000, 1_048_576, 25)

    def test_vram_estimate_is_deterministic_and_explicitly_labeled(self):
        first = estimate_required_vram(
            self.policy, training_config(), serving_config()
        )
        second = estimate_required_vram(
            self.policy, training_config(), serving_config()
        )
        self.assertEqual(first, second)
        self.assertEqual(first.subtotal_training_mib, 3918)
        self.assertEqual(first.required_training_mib, 4897)
        self.assertIsNone(first.serving)
        self.assertEqual(first.safety_margin_percent, 25)
        self.assertEqual(first.label, "estimate")

    def test_estimate_considers_training_shape_method_and_checkpointing(self):
        baseline = estimate_required_vram(
            self.policy, training_config(), serving_config()
        )
        larger_batch = estimate_required_vram(
            self.policy, training_config(batch_size=2), serving_config()
        )
        longer_sequence = estimate_required_vram(
            self.policy, training_config(cutoff_len=4096), serving_config()
        )
        checkpointed = estimate_required_vram(
            self.policy,
            training_config(gradient_checkpointing=True),
            serving_config(),
        )
        full = estimate_required_vram(
            self.policy, training_config(method="full"), serving_config()
        )
        fp32 = estimate_required_vram(
            self.policy, training_config(precision="fp32"), serving_config()
        )
        self.assertGreater(larger_batch.required_training_mib, baseline.required_training_mib)
        self.assertGreater(longer_sequence.required_training_mib, baseline.required_training_mib)
        self.assertLess(checkpointed.required_training_mib, baseline.required_training_mib)
        self.assertGreater(full.required_training_mib, baseline.required_training_mib)
        self.assertGreater(fp32.required_training_mib, baseline.required_training_mib)

    def test_serving_formula_and_architecture_dimensions(self):
        architecture = ModelArchitecture("test", 16, 2, 4, 1, 8, 100, None)
        memory = ServingMemoryPolicy(True, 10, 2, 15, 1000, 9000)
        policy = ResourcePolicy(True, 1000, 1, 25, architecture, memory)
        serving = resolve_serving_config(
            {"enabled": True, "vllm": {
                "gpu_memory_utilization": "auto", "max_model_len": 10, "max_num_seqs": 2,
            }},
            run_id="test", base_model="test", training_method="lora",
        )
        baseline = estimate_serving_memory(policy, training_config(), serving)
        self.assertEqual(baseline.kv_cache_mib, 1)
        self.assertEqual(baseline.kv_cache_bytes, 1280)
        # Exact bytes: 2 K/V * 2 layers * 1 KV head * 8 head dim
        # * 10 tokens * 2 sequences * 2 bf16 bytes = 1280.
        self.assertEqual(
            2 * architecture.num_hidden_layers * architecture.num_key_value_heads
            * architecture.head_dim * serving.vllm.max_model_len
            * serving.vllm.max_num_seqs * 2, 1280,
        )
        self.assertEqual(baseline.adapter_mib, 1)
        self.assertEqual(baseline.runtime_overhead_mib, 10)
        self.assertGreater(baseline.required_serving_mib, baseline.subtotal_mib)
        self.assertEqual(baseline, estimate_serving_memory(policy, training_config(), serving))
        self.assertGreater(
            estimate_serving_memory(policy, training_config(), replace(
                serving, vllm=replace(serving.vllm, max_model_len=20)
            )).kv_cache_bytes, baseline.kv_cache_bytes,
        )
        self.assertGreater(
            estimate_serving_memory(policy, training_config(), replace(
                serving, vllm=replace(serving.vllm, max_num_seqs=4)
            )).kv_cache_bytes, baseline.kv_cache_bytes,
        )
        for field in ("num_hidden_layers", "num_key_value_heads", "head_dim"):
            larger = replace(architecture, **{field: getattr(architecture, field) + 1})
            # Test bytes through larger settings where MiB rounding cannot mask change.
            large_serving = replace(serving, vllm=replace(serving.vllm, max_model_len=100, max_num_seqs=10000))
            self.assertGreater(
                estimate_serving_memory(replace(policy, architecture=larger), training_config(), large_serving).kv_cache_mib,
                estimate_serving_memory(policy, training_config(), large_serving).kv_cache_mib,
            )
        self.assertEqual(
            baseline.kv_cache_mib,
            estimate_serving_memory(replace(policy, architecture=replace(architecture, sliding_window=5)), training_config(), serving).kv_cache_mib,
        )
        with self.assertRaisesRegex(ResourcePreflightError, "exceeds model context"):
            estimate_serving_memory(policy, training_config(), replace(
                serving, vllm=replace(serving.vllm, max_model_len=101)
            ))

    def test_real_catalog_serving_estimates_scale(self):
        catalog = yaml.safe_load((TESTS_DIR.parent / "configs/model_catalog.yaml").read_text())["models"]
        memory = ServingMemoryPolicy(True, 2048, 2, 15, 1000, 9000)
        serving = resolve_serving_config(
            {"enabled": True, "vllm": {"gpu_memory_utilization": "auto"}},
            run_id="test", base_model="test", training_method="lora",
        )
        estimates = []
        for model in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B"):
            entry = catalog[model]
            policy = ResourcePolicy(True, entry["parameter_count"], 1, 25, ModelArchitecture(**entry["architecture"]), memory)
            estimates.append(estimate_serving_memory(policy, training_config(precision=entry["precision"]), serving))
        self.assertLess(estimates[0].required_serving_mib, estimates[1].required_serving_mib)
        self.assertLess(estimates[1].required_serving_mib, estimates[2].required_serving_mib)
        policy = ResourcePolicy(True, 1_000_000_000, 1, 25, ModelArchitecture(**catalog["Qwen/Qwen2.5-7B-Instruct"]["architecture"]), memory)
        self.assertGreater(
            estimate_serving_memory(policy, training_config(precision="fp32"), serving).weight_mib,
            estimate_serving_memory(policy, training_config(precision="bf16"), serving).weight_mib,
        )

    def test_invalid_config_fails_before_any_admission_callback(self):
        raw = {
            "model": {"name": "Org/Model"},
            "training": {},
            "orchestration": {"enabled": True, "training": {"gpu_devices": "0"}},
            "serving": {
                "enabled": True,
                "advertise_host": "model.example",
                "port": "auto",
                "port_range": {"start": 8101, "end": 8199},
            },
            "resource_preflight": {
                "enabled": True,
                "model_parameter_estimate": 490_000_000,
                "activation_bytes_per_token": 1_048_576,
                "safety_margin_percent": 25,
            },
        }
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_and_estimate(path, environment={})

    def test_invalid_dataset_config_fails_before_resource_admission(self):
        raw = {
            "model": {"name": "Org/Model"},
            "dataset": {"path": "data.json", "name": ""},
            "training": {},
            "resource_preflight": {
                "enabled": True,
                "model_parameter_estimate": 490_000_000,
                "activation_bytes_per_token": 1_048_576,
                "safety_margin_percent": 25,
            },
        }
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ResourcePreflightError, "dataset.name"):
                load_and_estimate(path, environment={})

    def test_phase1_config_produces_expected_estimate(self):
        config = TESTS_DIR.parent / "configs" / "ci_phase1.yaml"
        policy, estimate, serving = load_and_estimate(
            config,
            environment={
                "SERVING_ADVERTISE_HOST": "model-worker.example",
                "PIPELINE_SELECTED_PORT": "8101",
                "LITELLM_BASE_URL": "http://litellm.internal:4000",
                "LITELLM_API_KEY": "fake-test-key",
            },
        )
        self.assertTrue(policy.enabled)
        self.assertEqual(estimate.required_training_mib, 6065)
        self.assertGreater(estimate.serving.required_serving_mib, 0)
        self.assertEqual(serving.port_range, (8101, 8199))


class PortSelectionTests(unittest.TestCase):
    def test_auto_port_picks_first_available_and_skips_occupied_ports(self):
        checked = []

        def checker(_host, port):
            checked.append(port)
            return port == 8103

        self.assertEqual(
            select_available_port("0.0.0.0", 8101, 8104, checker=checker),
            8103,
        )
        self.assertEqual(checked, [8101, 8102, 8103])

    def test_exhausted_port_range_fails_without_mutation(self):
        checked = []

        def checker(_host, port):
            checked.append(port)
            return False

        with self.assertRaisesRegex(
            ResourcePreflightError, "No available serving port.*no existing service"
        ):
            select_available_port("0.0.0.0", 8101, 8102, checker=checker)
        self.assertEqual(checked, [8101, 8102])


if __name__ == "__main__":
    unittest.main()

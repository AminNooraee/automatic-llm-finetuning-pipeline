import tempfile
import unittest
from pathlib import Path

import yaml

from fine_tuning_pipeline.resource_preflight import (
    ResourcePolicy,
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
        self.assertEqual(first.serving_memory_utilization_bps, 1500)
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
        self.assertEqual(estimate.required_training_mib, 4897)
        self.assertEqual(estimate.serving_memory_utilization_bps, 1500)
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

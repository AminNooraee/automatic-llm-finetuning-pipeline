"""User job validation and prepared-artifact contract."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.prepare_ci_job import (
    BENCHMARK_EXTENSIONS,
    TRAIN_EXTENSIONS,
    JobConfigError,
    prepare_job,
)
from scripts.validate_prepared_train import training_path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class UserJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job_dir = self.root / "job"
        self.job_dir.mkdir()
        self.output = self.root / "prepared"
        self.job = {
            "schema_version": 1,
            "job": {"name": "example-job"},
            "model": {"name": MODEL},
            "datasets": {
                "train": {"path": "train.json", "source_format": "auto", "format": "auto"},
                "benchmark": {"path": "benchmark.json", "format": "auto"},
            },
            "training": {
                "epochs": 1, "learning_rate": 0.00005, "batch_size": 1,
                "cutoff_len": 2048, "lora": {"rank": 8, "alpha": 16, "dropout": 0},
            },
        }
        (self.job_dir / "train.json").write_text('[{"instruction":"Hi","output":"Hello"}]')
        (self.job_dir / "benchmark.json").write_text('[{"prompt":"Hi"}]')

    def prepare(self):
        job_path = self.job_dir / "job.yaml"
        job_path.write_text(yaml.safe_dump(self.job))
        prepare_job(
            job_path,
            ROOT / "configs/model_catalog.yaml",
            ROOT / "configs/ci_infrastructure.yaml",
            self.output,
        )
        return json.loads((self.output / "job_manifest.json").read_text())

    def test_json_manifest_and_catalog_controlled_config(self):
        manifest = self.prepare()
        config = yaml.safe_load((self.output / "finetune.yaml").read_text())
        self.assertEqual(manifest["job_name"], "example-job")
        self.assertEqual(manifest["model"], {"name": MODEL, "revision": None})
        self.assertEqual(manifest["generated_config"], "finetune.yaml")
        self.assertEqual(config["dataset"]["path"], "train.json")
        self.assertEqual(config["training"]["precision"], "fp32")
        self.assertEqual(config["training"]["attention_backend"], "eager")
        self.assertEqual(config["resource_preflight"]["model_parameter_estimate"], 490000000)
        for field, filename in (
            ("train_dataset", "train.json"),
            ("benchmark_dataset", "benchmark.json"),
        ):
            self.assertEqual(manifest[field]["file"], filename)
            digest = hashlib.sha256((self.output / filename).read_bytes()).hexdigest()
            self.assertEqual(manifest[field]["sha256"], digest)
        self.assertEqual(training_path(self.output), (self.output / "train.json").resolve())

    def test_supported_training_extensions(self):
        for suffix in sorted(TRAIN_EXTENSIONS):
            with self.subTest(suffix=suffix):
                name = "train" + suffix
                (self.job_dir / name).write_bytes(b"sample")
                self.job["datasets"]["train"]["path"] = name
                manifest = self.prepare()
                self.assertEqual(manifest["train_dataset"]["file"], name)
                self.assertEqual(training_path(self.output).name, name)

    def test_benchmark_extensions(self):
        for suffix in sorted(BENCHMARK_EXTENSIONS):
            with self.subTest(suffix=suffix):
                name = "benchmark" + suffix
                (self.job_dir / name).write_bytes(b"sample")
                self.job["datasets"]["benchmark"]["path"] = name
                self.assertEqual(self.prepare()["benchmark_dataset"]["file"], name)

    def test_missing_and_unsafe_paths(self):
        for field, path in (
            ("train", "missing.json"), ("benchmark", "missing.csv"),
            ("train", "../train.json"), ("benchmark", "/tmp/benchmark.json"),
            ("train", "sub/../train.json"), ("benchmark", "./benchmark.json"),
            ("train", "C:/train.json"),
        ):
            with self.subTest(field=field, path=path):
                original = self.job["datasets"][field]["path"]
                self.job["datasets"][field]["path"] = path
                with self.assertRaises(JobConfigError):
                    self.prepare()
                self.job["datasets"][field]["path"] = original

    def test_unsupported_extensions_and_same_file(self):
        (self.job_dir / "train.exe").write_bytes(b"bad")
        self.job["datasets"]["train"]["path"] = "train.exe"
        with self.assertRaisesRegex(JobConfigError, "unsupported extension"):
            self.prepare()
        self.job["datasets"]["train"]["path"] = "train.json"
        (self.job_dir / "benchmark.txt").write_bytes(b"bad")
        self.job["datasets"]["benchmark"]["path"] = "benchmark.txt"
        with self.assertRaisesRegex(JobConfigError, "unsupported extension"):
            self.prepare()
        self.job["datasets"]["benchmark"]["path"] = "train.json"
        with self.assertRaisesRegex(JobConfigError, "different files"):
            self.prepare()

    def test_schema_unknown_fields_names_and_model(self):
        cases = [
            (lambda: self.job.update(schema_version=True), "schema_version"),
            (lambda: self.job.update(gateway={}), "Unsupported job"),
            (lambda: self.job.update({42: "unknown"}), "Unsupported job"),
            (lambda: self.job["model"].update(precision="bf16"), "Unsupported model"),
            (lambda: self.job["job"].update(name="../unsafe"), "job.name"),
            (lambda: self.job["model"].update(name="unapproved/model"), "MODEL_NOT_APPROVED"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                original = json.loads(json.dumps(self.job))
                mutate()
                with self.assertRaisesRegex(JobConfigError, message):
                    self.prepare()
                self.job = original

    def test_hyperparameter_bounds_and_types(self):
        for key, value in (
            ("epochs", 0), ("epochs", True), ("epochs", float("inf")),
            ("learning_rate", 0), ("learning_rate", True),
            ("batch_size", 0), ("batch_size", True),
            ("cutoff_len", 0), ("cutoff_len", True),
        ):
            with self.subTest(key=key, value=value):
                original = self.job["training"][key]
                self.job["training"][key] = value
                with self.assertRaises(JobConfigError):
                    self.prepare()
                self.job["training"][key] = original
        for key, value in (
            ("rank", 0), ("rank", True), ("alpha", 0), ("alpha", True),
            ("dropout", -0.1), ("dropout", 1), ("dropout", True),
            ("dropout", float("nan")),
        ):
            with self.subTest(key=key, value=value):
                original = self.job["training"]["lora"][key]
                self.job["training"]["lora"][key] = value
                with self.assertRaises(JobConfigError):
                    self.prepare()
                self.job["training"]["lora"][key] = original

    def test_manifest_filename_rejection(self):
        self.prepare()
        path = self.output / "job_manifest.json"
        for filename in (
            "../train.json", "/tmp/train.json", "sub/train.json",
            "train.exe", "benchmark.json",
        ):
            with self.subTest(filename=filename):
                manifest = json.loads(path.read_text())
                manifest["train_dataset"]["file"] = filename
                path.write_text(json.dumps(manifest))
                with self.assertRaises((ValueError, OSError)):
                    training_path(self.output)

    def test_ci_artifact_and_gitlab_contract(self):
        ci_text = (ROOT / ".gitlab-ci.yml").read_text()
        ci = yaml.safe_load(ci_text)
        prepare = ci["prepare-user-job"]
        phase = ci["phase1-finetune"]
        self.assertIn("ci_artifacts/job/", prepare["artifacts"]["paths"])
        self.assertIn("validate_prepared_train.py", str(phase["script"]))
        self.assertNotIn('PIPELINE_DATASETS_DIR/train.json', str(phase["script"]))
        for artifact in (
            "ci_artifacts/project1/gateway_manifest.json",
            "ci_artifacts/project1/upstream.env",
            "ci_artifacts/job/benchmark.*",
            "ci_artifacts/job/job_manifest.json",
        ):
            self.assertIn(artifact, phase["artifacts"]["paths"])
        for key in (
            "PROJECT1_UPSTREAM_PROJECT_ID", "PROJECT1_UPSTREAM_PIPELINE_ID",
            "PROJECT1_UPSTREAM_JOB_ID", "PROJECT1_UPSTREAM_COMMIT_SHA",
            "PROJECT1_BENCHMARK_FILE",
        ):
            self.assertIn(key, ci_text)
        self.assertNotIn("strategy: mirror", ci_text)
        self.assertNotIn("needs:project", ci_text)
        self.assertEqual(ci["trigger-phase1-benchmark"]["trigger"]["strategy"], "depend")
        self.assertIn("--network=host", str(prepare["script"]))
        self.assertIn("Dockerfile.controller.host-dns", str(prepare["script"]))


if __name__ == "__main__":
    unittest.main()

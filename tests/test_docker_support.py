"""Static Docker deployment contracts; real image validation needs Docker."""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DockerSupportTests(unittest.TestCase):
    def setUp(self):
        self.dockerfile = (REPOSITORY_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        self.instructions = re.sub(r"\\\s*\n\s*", " ", self.dockerfile)
        dependency_script = (
            REPOSITORY_ROOT / "docker" / "install-training-dependencies.sh"
        ).read_text(encoding="utf-8")
        self.dependency_instructions = re.sub(r"\\\s*\n\s*", " ", dependency_script)
        self.ignore_rules = [
            line.strip() for line in
            (REPOSITORY_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]

    def test_cpu_default_and_digest_pinned_cuda_design(self):
        self.assertIn("ARG RUNTIME=cpu", self.dockerfile)
        self.assertRegex(self.dockerfile, r"ARG CPU_IMAGE=python:3\.12\.7-slim-bookworm@sha256:[a-f0-9]{64}")
        self.assertRegex(self.dockerfile, r"ARG CUDA_IMAGE=nvidia/cuda:12\.8\.1-runtime-ubuntu24\.04@sha256:[a-f0-9]{64}")
        self.assertIn("FROM ${RUNTIME}-base AS application", self.dockerfile)
        self.assertIn("https://download.pytorch.org/whl/cpu", self.dockerfile)
        self.assertIn("https://download.pytorch.org/whl/cu128", self.dockerfile)
        self.assertNotIn("torch.cuda.is_available()", self.dockerfile)
        pins = (REPOSITORY_ROOT / "docker" / "pytorch-requirements.txt").read_text(encoding="utf-8")
        for pin in ("torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0"):
            self.assertIn(pin, pins)

    def test_build_dependencies_and_non_root_entry_point(self):
        self.assertIn("--constraint /opt/pytorch-constraints.txt", self.dependency_instructions)
        self.assertIn("for name in ('torch', 'torchvision', 'torchaudio')", self.dependency_instructions)
        self.assertNotIn("pip freeze > /opt/pytorch-constraints.txt", self.dependency_instructions)
        self.assertIn("-r requirements.txt", self.dependency_instructions)
        self.assertIn("python -m pip check", self.dependency_instructions)
        self.assertIn("RUN python -m unittest discover -s tests -v", self.instructions)
        self.assertIn("USER ${APP_UID}:${APP_GID}", self.instructions)
        command = re.search(r"^CMD (.+)$", self.instructions, re.MULTILINE).group(1)
        entrypoint = re.search(r"^ENTRYPOINT (.+)$", self.instructions, re.MULTILINE).group(1)
        self.assertEqual(json.loads(command), ["python", "-m", "fine_tuning_pipeline.train_pipeline"])
        self.assertEqual(json.loads(entrypoint), ["/usr/bin/tini", "-g", "--"])

    def test_copy_sources_exist_and_exclude_host_runtime_folders(self):
        sources = []
        for copy in re.findall(r"^COPY (.+)$", self.instructions, re.MULTILINE):
            sources.extend(copy.split()[:-1])
        self.assertTrue(sources)
        for source in sources:
            with self.subTest(source=source):
                self.assertNotIn(source, (".", "./", "runs/", "models/", "datasets/", "venv/", "LLaMA-Factory/"))
                self.assertTrue((REPOSITORY_ROOT / source).exists())
        self.assertIn("configs/", sources)
        self.assertIn("src/fine_tuning_pipeline/", sources)

    def test_build_context_defaults_to_deny_and_reopens_only_project_files(self):
        self.assertEqual(self.ignore_rules[0], "**")
        for rule in ("!requirements.txt", "!pyproject.toml", "!docker/Dockerfile",
                     "!docker/Dockerfile.host-dns",
                     "!docker/Dockerfile.controller.host-dns",
                     "!configs/config.yaml", "!configs/serving_gateway_example.yaml",
                     "!src/fine_tuning_pipeline/*.py", "!tests/*.py"):
            self.assertIn(rule, self.ignore_rules)
        for directory in ("docker", "src", "src/fine_tuning_pipeline",
                          "src/fine_tuning_pipeline/dataset_adapters",
                          "src/fine_tuning_pipeline/serving",
                          "src/fine_tuning_pipeline/serving/backends",
                          "src/fine_tuning_pipeline/gateway",
                          "src/fine_tuning_pipeline/gateway/providers", "configs",
                          "examples", "examples/datasets", "tests"):
            self.assertIn(directory + "/**", self.ignore_rules)
        for directory in ("runs", "models", "checkpoints", "datasets", "venv", ".venv",
                          "fine_tuning_pipeline", "LLaMA-Factory", "benchmark_pipeline", "serving", "evaluation"):
            self.assertFalse(any(rule.startswith("!" + directory + "/") for rule in self.ignore_rules))
        for rule in ("**/__pycache__", "**/tmp*", "**/.env.*", "**/*.safetensors", "**/*.log"):
            self.assertIn(rule, self.ignore_rules)
        self.assertNotIn("!docker/configure-build-dns.sh", self.ignore_rules)

    def test_user_job_modules_are_in_build_context_allowlist(self):
        required = ("prepare_ci_job.py", "validate_prepared_train.py")
        deny_index = self.ignore_rules.index("scripts/**")
        for name in required:
            rule = f"!scripts/{name}"
            self.assertIn(rule, self.ignore_rules)
            self.assertGreater(self.ignore_rules.index(rule), deny_index)
        self.assertNotIn("!scripts/*.py", self.ignore_rules)
        self.assertIn("!tests/*.py", self.ignore_rules)
        self.assertIn("COPY scripts/ ./scripts/", self.dockerfile)
        host_dns_dockerfile = (
            REPOSITORY_ROOT / "docker" / "Dockerfile.host-dns"
        ).read_text(encoding="utf-8")
        self.assertIn("COPY scripts/ ./scripts/", host_dns_dockerfile)

        with tempfile.TemporaryDirectory() as temporary:
            context = Path(temporary)
            (context / "scripts").mkdir()
            (context / "tests").mkdir()
            for name in required:
                shutil.copy2(REPOSITORY_ROOT / "scripts" / name, context / "scripts" / name)
            for name in ("__init__.py", "test_user_job.py"):
                shutil.copy2(REPOSITORY_ROOT / "tests" / name, context / "tests" / name)
            code = (
                "import pathlib, sys; sys.path.insert(0, '.'); "
                "import tests.test_user_job as test; "
                "import scripts.prepare_ci_job as prepare; "
                "import scripts.validate_prepared_train as validate; "
                "root = pathlib.Path.cwd().resolve(); "
                "assert all(pathlib.Path(module.__file__).resolve().is_relative_to(root) "
                "for module in (test, prepare, validate))"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=context,
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

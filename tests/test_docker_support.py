"""Static Docker deployment contracts; real image validation needs Docker."""

import json
import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DockerSupportTests(unittest.TestCase):
    def setUp(self):
        self.dockerfile = (REPOSITORY_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        self.instructions = re.sub(r"\\\s*\n\s*", " ", self.dockerfile)
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
        self.assertIn("--constraint /opt/pytorch-constraints.txt", self.instructions)
        self.assertIn("for name in ('torch', 'torchvision', 'torchaudio')", self.instructions)
        self.assertNotIn("pip freeze > /opt/pytorch-constraints.txt", self.instructions)
        self.assertIn("-r requirements.txt", self.instructions)
        self.assertIn("python -m pip check", self.instructions)
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
                     "!configs/config.yaml", "!src/fine_tuning_pipeline/*.py", "!tests/*.py"):
            self.assertIn(rule, self.ignore_rules)
        for directory in ("docker", "src", "src/fine_tuning_pipeline",
                          "src/fine_tuning_pipeline/dataset_adapters", "configs",
                          "examples", "examples/datasets", "tests"):
            self.assertIn(directory + "/**", self.ignore_rules)
        for directory in ("runs", "models", "checkpoints", "datasets", "venv", ".venv",
                          "fine_tuning_pipeline", "LLaMA-Factory", "benchmark_pipeline", "serving", "evaluation"):
            self.assertFalse(any(rule.startswith("!" + directory + "/") for rule in self.ignore_rules))
        for rule in ("**/__pycache__", "**/tmp*", "**/.env.*", "**/*.safetensors", "**/*.log"):
            self.assertIn(rule, self.ignore_rules)


if __name__ == "__main__":
    unittest.main()

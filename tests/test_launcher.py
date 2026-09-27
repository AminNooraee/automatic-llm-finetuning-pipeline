import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _shell() -> str | None:
    found = shutil.which("sh") or shutil.which("bash")
    if found:
        return found
    candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"
    return str(candidate) if candidate.is_file() else None


SHELL = _shell()


@unittest.skipUnless(SHELL, "A POSIX-compatible shell is required for launcher tests")
class DockerBuildNetworkLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tests")
        self.root = Path(self.temporary.name)
        for directory in ("scripts", "configs", "src", "docker", "fake-bin"):
            (self.root / directory).mkdir()
        shutil.copy2(REPOSITORY_ROOT / "scripts/run_pipeline.sh", self.root / "scripts/run_pipeline.sh")
        (self.root / "configs/test.yaml").write_text("orchestration:\n  enabled: true\n", encoding="utf-8")
        (self.root / "pyproject.toml").write_text("[project]\nname='launcher-test'\n", encoding="utf-8")
        (self.root / "requirements.txt").write_text("", encoding="utf-8")
        self.log = self.root / "docker.log"
        fake = self.root / "fake-bin/docker"
        fake.write_text(
            """#!/bin/sh
printf '%s\\n' \"$*\" >> \"$FAKE_DOCKER_LOG\"
if [ \"${1-}\" = info ]; then exit 0; fi
if [ \"${1-} ${2-}\" = \"image inspect\" ]; then
    case \"$*\" in
        *source-identity*)
            if [ \"${FAKE_REUSE-0}\" = 1 ]; then git -C \"$FAKE_PROJECT\" rev-parse HEAD; exit 0; fi
            exit 1
            ;;
        *) echo sha256:training-image; exit 0 ;;
    esac
fi
if [ \"${1-}\" = build ]; then
    case \"${FAKE_BUILD_SCENARIO-success}: $*\" in
        dns_then_success:*--network=host*) exit 0 ;;
        dns_then_success:*) echo 'Temporary failure resolving deb.debian.org' >&2; exit 7 ;;
        dns_host_fail:*--network=host*) echo 'host retry failed' >&2; exit 9 ;;
        dns_host_fail:*) echo 'Could not resolve deb.debian.org' >&2; exit 7 ;;
        dns:*) echo 'DNS resolution failed' >&2; exit 7 ;;
        unrelated:*) echo "${FAKE_FAILURE_MESSAGE-Dockerfile syntax error}" >&2; exit 6 ;;
        *) exit 0 ;;
    esac
fi
if [ \"${1-}\" = run ]; then exit 0; fi
exit 1
""",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        subprocess.run(("git", "init", "-q", str(self.root)), check=True)
        subprocess.run(("git", "-C", str(self.root), "config", "user.email", "test@example.invalid"), check=True)
        subprocess.run(("git", "-C", str(self.root), "config", "user.name", "Launcher Test"), check=True)
        subprocess.run(("git", "-C", str(self.root), "add", "."), check=True)
        subprocess.run(("git", "-C", str(self.root), "commit", "-qm", "fixture"), check=True)

    def tearDown(self):
        self.temporary.cleanup()

    def run_launcher(
        self, *, mode=None, scenario="success", reuse=False, rebuild=False,
        failure_message=None,
    ):
        self.log.unlink(missing_ok=True)
        environment = os.environ.copy()
        environment.update({
            "PATH": f"{self.root / 'fake-bin'}{os.pathsep}{environment['PATH']}",
            "FAKE_DOCKER_LOG": self.log.as_posix(),
            "FAKE_PROJECT": self.root.as_posix(),
            "FAKE_BUILD_SCENARIO": scenario,
            "FAKE_REUSE": "1" if reuse else "0",
            "PIPELINE_RUNS_DIR": (self.root / "runtime/runs").as_posix(),
            "PIPELINE_HF_CACHE_DIR": (self.root / "runtime/cache").as_posix(),
            "PIPELINE_STATE_DIR": (self.root / "runtime/state").as_posix(),
        })
        if mode is not None:
            environment["PIPELINE_DOCKER_BUILD_NETWORK"] = mode
        if failure_message is not None:
            environment["FAKE_FAILURE_MESSAGE"] = failure_message
        command = [SHELL, (self.root / "scripts/run_pipeline.sh").as_posix()]
        if rebuild:
            command.append("--rebuild")
        command.append("configs/test.yaml")
        return subprocess.run(command, cwd=self.root, env=environment, text=True, capture_output=True)

    def commands(self):
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def builds(self):
        return [line for line in self.commands() if line.startswith("build ")]

    def test_default_environment_is_auto_and_success_has_no_retry(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.builds()), 2)
        self.assertTrue(all("--network=host" not in line for line in self.builds()))
        self.assertNotIn("Retrying once", result.stderr)

    def test_invalid_mode_is_rejected_before_docker_access(self):
        result = self.run_launcher(mode="bridge")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be one of: auto, default, host", result.stderr)
        self.assertEqual(self.commands(), [])

    def test_auto_dns_failure_retries_each_image_once_and_continues(self):
        result = self.run_launcher(scenario="dns_then_success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.builds()), 4)
        self.assertEqual(sum("--network=host" in line for line in self.builds()), 2)
        self.assertEqual(result.stderr.count("Retrying once with host build networking"), 2)

    def test_auto_failed_host_retry_returns_failure(self):
        result = self.run_launcher(scenario="dns_host_fail")
        self.assertEqual(result.returncode, 9)
        self.assertEqual(len(self.builds()), 2)
        self.assertIn("--network=host", self.builds()[1])
        self.assertFalse(any(line.startswith("run ") for line in self.commands()))

    def test_auto_unrelated_failure_does_not_retry(self):
        failures = (
            "Dockerfile syntax error",
            "COPY failed: file not found",
            "ResolutionImpossible: dependency conflict",
            "test command returned a non-zero code",
            "invalid build argument",
            "permission denied",
        )
        for message in failures:
            with self.subTest(message=message):
                result = self.run_launcher(
                    scenario="unrelated", failure_message=message
                )
                self.assertEqual(result.returncode, 6)
                self.assertEqual(len(self.builds()), 1)
                self.assertNotIn("Retrying once", result.stderr)

    def test_default_mode_never_falls_back(self):
        result = self.run_launcher(mode="default", scenario="dns")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(self.builds()), 1)
        self.assertNotIn("--network=host", self.builds()[0])

    def test_host_mode_uses_host_network_for_both_builds(self):
        result = self.run_launcher(mode="host")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.builds()), 2)
        self.assertTrue(all("--network=host" in line for line in self.builds()))
        self.assertIn("explicitly configured host networking", result.stdout)

    def test_matching_images_are_reused_without_building(self):
        result = self.run_launcher(reuse=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.builds(), [])
        self.assertEqual(result.stdout.count("Reusing image"), 2)

    def test_rebuild_overrides_matching_image_reuse(self):
        result = self.run_launcher(reuse=True, rebuild=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.builds()), 2)

    def test_runtime_network_arguments_are_not_changed_by_build_mode(self):
        result = self.run_launcher(mode="host")
        self.assertEqual(result.returncode, 0, result.stderr)
        runtime = [line for line in self.commands() if line.startswith("run ")]
        self.assertEqual(len(runtime), 1)
        self.assertIn("--read-only --network host", runtime[0])
        self.assertNotIn("--network=host", runtime[0])


if __name__ == "__main__":
    unittest.main()

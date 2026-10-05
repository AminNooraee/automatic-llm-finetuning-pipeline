import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = REPOSITORY_ROOT / "scripts" / "gpu_preflight.sh"


def _shell() -> str | None:
    found = shutil.which("sh") or shutil.which("bash")
    if found:
        return found
    candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"
    return str(candidate) if candidate.is_file() else None


SHELL = _shell()


@unittest.skipUnless(SHELL, "A POSIX-compatible shell is required for GPU preflight tests")
class GpuPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tests")
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.docker_log = self.root / "docker.log"

    def tearDown(self):
        self.temporary.cleanup()

    def _write_executable(self, name: str, content: str) -> None:
        path = self.bin / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    def _docker(self, *, daemon_ok: bool = True, nvidia_runtime: bool = True) -> None:
        self._write_executable(
            "docker",
            "#!/bin/sh\n"
            'printf \'%s\\n\' "$*" >> "$FAKE_DOCKER_LOG"\n'
            + ("exit 9\n" if not daemon_ok else "")
            + (
                "printf '%s\\n' '{\"nvidia\":{\"path\":\"nvidia-container-runtime\"},"
                "\"runc\":{\"path\":\"runc\"}}'\n"
                if nvidia_runtime
                else "printf '%s\\n' '{\"runc\":{\"path\":\"runc\"}}'\n"
            ),
        )

    def _nvidia(self, *, works: bool = True) -> None:
        body = "#!/bin/sh\n"
        body += "printf '%s' \"$FAKE_NVIDIA_OUTPUT\"\n" if works else "exit 8\n"
        self._write_executable("nvidia-smi", body)

    def run_preflight(
        self, *, minimum: str | None = "1000", device: str | None = None,
        nvidia: bool = True, docker: bool = True, daemon_ok: bool = True,
        nvidia_runtime: bool = True, gpu_output: str = "0, 12000\n",
        nvidia_works: bool = True, env_file: Path | None = None,
        isolated_path: bool = False,
        estimated: str | None = None, serving_utilization_bps: str = "1500",
    ) -> subprocess.CompletedProcess[str]:
        if nvidia:
            self._nvidia(works=nvidia_works)
        if docker:
            self._docker(daemon_ok=daemon_ok, nvidia_runtime=nvidia_runtime)
        environment = os.environ.copy()
        environment.update({
            "PATH": str(self.bin) if isolated_path else f"{self.bin}{os.pathsep}{environment['PATH']}",
            "FAKE_DOCKER_LOG": str(self.docker_log),
            "FAKE_NVIDIA_OUTPUT": gpu_output,
        })
        for key in (
            "GPU_MIN_FREE_MIB", "GPU_DEVICE", "GPU_PREFLIGHT_ENV_FILE",
            "GPU_ESTIMATED_TRAINING_MIB", "GPU_SERVING_MEMORY_UTILIZATION_BPS",
        ):
            environment.pop(key, None)
        if estimated is not None:
            environment["GPU_ESTIMATED_TRAINING_MIB"] = estimated
            environment["GPU_SERVING_MEMORY_UTILIZATION_BPS"] = serving_utilization_bps
        elif minimum is not None:
            environment["GPU_MIN_FREE_MIB"] = minimum
        if device is not None:
            environment["GPU_DEVICE"] = device
        if env_file is not None:
            environment["GPU_PREFLIGHT_ENV_FILE"] = env_file.as_posix()
        return subprocess.run(
            (SHELL, PREFLIGHT.as_posix()), cwd=self.root, env=environment,
            text=True, capture_output=True,
        )

    def assert_failed_with(self, result, message):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)

    def test_missing_nvidia_smi(self):
        self._docker()
        result = self.run_preflight(nvidia=False, docker=False, isolated_path=True)
        self.assert_failed_with(result, "nvidia-smi is required")

    def test_missing_docker_cli(self):
        self._nvidia()
        result = self.run_preflight(nvidia=False, docker=False, isolated_path=True)
        self.assert_failed_with(result, "Docker CLI is required")

    def test_docker_permission_or_daemon_failure(self):
        result = self.run_preflight(daemon_ok=False)
        self.assert_failed_with(result, "daemon is unavailable or permission was denied")

    def test_missing_nvidia_runtime(self):
        result = self.run_preflight(nvidia_runtime=False)
        self.assert_failed_with(result, "NVIDIA container runtime is not configured")

    def test_nvidia_smi_query_failure(self):
        result = self.run_preflight(nvidia_works=False)
        self.assert_failed_with(result, "could not query the NVIDIA driver")

    def test_no_gpus_detected(self):
        result = self.run_preflight(gpu_output="")
        self.assert_failed_with(result, "No NVIDIA GPUs were detected")

    def test_missing_gpu_min_free_mib(self):
        result = self.run_preflight(minimum=None)
        self.assert_failed_with(result, "must be explicitly configured")

    def test_invalid_gpu_min_free_mib(self):
        for value in ("", "-1", "1.5", "1 MiB", "+1"):
            with self.subTest(value=value):
                self.assert_failed_with(
                    self.run_preflight(minimum=value), "must be a non-negative integer"
                )

    def test_invalid_gpu_device(self):
        for value in ("", "AUTO", "-1", "0.0", "0 1"):
            with self.subTest(value=value):
                self.assert_failed_with(
                    self.run_preflight(device=value),
                    "must be 'auto' or a non-negative numeric GPU index",
                )

    def test_requested_gpu_does_not_exist(self):
        result = self.run_preflight(device="4", gpu_output="0, 1000\n2, 9000\n")
        self.assert_failed_with(result, "Requested GPU index 4 does not exist")

    def test_auto_chooses_gpu_with_most_free_memory(self):
        result = self.run_preflight(gpu_output="0, 4000\n1, 16000\n2, 8000\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Detected GPUs:\n  GPU 0: 4000 MiB free", result.stdout)
        self.assertIn("  GPU 1: 16000 MiB free", result.stdout)
        self.assertIn("Selected GPU: 1", result.stdout)
        self.assertIn("Free VRAM: 16000 MiB", result.stdout)

    def test_explicit_gpu_selection_works(self):
        result = self.run_preflight(device="2", gpu_output="0, 20000\n2, 3000\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Selected GPU: 2", result.stdout)
        self.assertIn("Free VRAM: 3000 MiB", result.stdout)

    def test_insufficient_free_vram_fails_closed(self):
        result = self.run_preflight(minimum="12001", gpu_output="0, 12000\n")
        self.assert_failed_with(result, "has 12000 MiB free; 12001 MiB is required")

    def test_sufficient_free_vram_succeeds_at_boundary(self):
        result = self.run_preflight(minimum="12000", gpu_output="0, 12000\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Required VRAM: 12000 MiB", result.stdout)
        self.assertIn("GPU preflight passed", result.stdout)

    def test_estimated_admission_selects_best_eligible_gpu(self):
        result = self.run_preflight(
            estimated="5000",
            gpu_output="0, 11000, 80000\n1, 8000, 24000\n2, 10000, 24000\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Selected GPU: 2", result.stdout)
        self.assertIn("Estimated required VRAM: 5000 MiB", result.stdout)

    def test_estimated_admission_failure_lists_every_gpu_and_reason(self):
        result = self.run_preflight(
            estimated="5000",
            gpu_output="0, 11000, 80000\n1, 4000, 24000\n",
        )
        self.assert_failed_with(
            result, "No GPU satisfies the estimated training/serving VRAM requirements."
        )
        self.assertIn("GPU 0: 11000 MiB free", result.stderr)
        self.assertIn(
            "GPU 0 rejected: 11000 MiB free; 12000 MiB estimated required",
            result.stderr,
        )
        self.assertIn(
            "GPU 1 rejected: 4000 MiB free; 5000 MiB estimated required",
            result.stderr,
        )

    def test_estimated_admission_env_records_effective_requirement(self):
        env_file = self.root / "estimated.env"
        result = self.run_preflight(
            estimated="5000", gpu_output="3, 9000, 24000\n", env_file=env_file
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            env_file.read_text(encoding="utf-8"),
            "PIPELINE_SELECTED_GPU=3\nCUDA_VISIBLE_DEVICES=3\n"
            "PIPELINE_ESTIMATED_REQUIRED_VRAM_MIB=5000\n",
        )

    def test_env_output_contains_selected_gpu_and_has_safe_permissions(self):
        env_file = self.root / "admission.env"
        result = self.run_preflight(gpu_output="0, 4000\n3, 9000\n", env_file=env_file)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            env_file.read_text(encoding="utf-8"),
            "PIPELINE_SELECTED_GPU=3\nCUDA_VISIBLE_DEVICES=3\n",
        )
        if os.name == "nt":
            # NTFS does not expose POSIX mode bits faithfully through Git Bash.
            self.assertIn('chmod 600 "$env_file"', PREFLIGHT.read_text(encoding="utf-8"))
        else:
            self.assertEqual(stat.S_IMODE(env_file.stat().st_mode), 0o600)

    def test_preflight_is_read_only_for_docker(self):
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.docker_log.read_text(encoding="utf-8").splitlines(),
            ["info --format {{json .Runtimes}}"],
        )
        source = PREFLIGHT.read_text(encoding="utf-8")
        for command in ("docker stop", "docker kill", "docker rm", "systemctl", "nvidia-smi --gpu-reset"):
            with self.subTest(command=command):
                self.assertNotIn(command, source)


if __name__ == "__main__":
    unittest.main()

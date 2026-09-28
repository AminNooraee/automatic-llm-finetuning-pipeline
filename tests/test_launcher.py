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
class ResolverParsingTests(unittest.TestCase):
    helper = REPOSITORY_ROOT / "scripts/build_dns.sh"

    def normalize(self, value, mode="strict"):
        command = '. "$1"; normalize_build_dns "$2" "$3"'
        return subprocess.run(
            (SHELL, "-c", command, "resolver-test", self.helper.as_posix(), value, mode),
            text=True, capture_output=True,
        )

    def discover(self, content):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tests") as temporary:
            path = Path(temporary) / "resolv.conf"
            path.write_text(content, encoding="utf-8")
            command = '. "$1"; discover_build_dns "$2"'
            return subprocess.run(
                (SHELL, "-c", command, "resolver-test", self.helper.as_posix(), path.as_posix()),
                text=True, capture_output=True,
            )

    def test_resolv_conf_preserves_order_deduplicates_and_filters_invalid_stubs(self):
        result = self.discover(
            "nameserver 127.0.0.53\n"
            "nameserver 192.0.2.53\n"
            "nameserver malformed\n"
            "nameserver 2001:db8::53\n"
            "nameserver 192.0.2.53\n"
            "nameserver ::1\n"
            "nameserver 127.0.0.1\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "192.0.2.53 2001:db8::53")

    def test_empty_usable_resolver_set_fails(self):
        result = self.discover(
            "nameserver 127.0.0.1\nnameserver 127.0.0.53\n"
            "nameserver ::1\nnameserver not-an-ip\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_explicit_values_accept_ip_literals_and_reject_malformed_or_shell_text(self):
        accepted = self.normalize("192.0.2.53, 2001:db8::53 192.0.2.53")
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(accepted.stdout.strip(), "192.0.2.53 2001:db8::53")
        for value in (
            "0.0.0.0", "::", "127.0.0.1", "127.0.0.53", "::1",
            "::ffff:127.0.0.1", "999.1.1.1", "2001::db8::1",
            "192.0.2.53;touch /tmp/owned", "resolver.example",
        ):
            with self.subTest(value=value):
                self.assertNotEqual(self.normalize(value).returncode, 0)


@unittest.skipUnless(SHELL, "A POSIX-compatible shell is required for launcher tests")
class DockerBuildNetworkLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tests")
        self.root = Path(self.temporary.name)
        for directory in ("scripts", "configs", "src", "docker", "fake-bin"):
            (self.root / directory).mkdir()
        shutil.copy2(REPOSITORY_ROOT / "scripts/run_pipeline.sh", self.root / "scripts/run_pipeline.sh")
        shutil.copy2(REPOSITORY_ROOT / "scripts/build_dns.sh", self.root / "scripts/build_dns.sh")
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
        dns_third_success:*PIPELINE_BUILD_DNS=*) exit 0 ;;
        dns_third_success:*--network=host*) echo 'Name or service not known' >&2; exit 8 ;;
        dns_third_success:*) echo 'Temporary failure resolving deb.debian.org' >&2; exit 7 ;;
        dns_third_fail:*PIPELINE_BUILD_DNS=*) echo 'DNS resolution still failed' >&2; exit 11 ;;
        dns_third_fail:*--network=host*) echo 'Could not resolve deb.debian.org' >&2; exit 8 ;;
        dns_third_fail:*) echo 'Temporary failure resolving deb.debian.org' >&2; exit 7 ;;
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
        failure_message=None, build_dns=None,
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
        if build_dns is not None:
            environment["PIPELINE_DOCKER_BUILD_DNS"] = build_dns
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

    def test_auto_host_dns_failure_retries_with_resolvers_for_both_images(self):
        result = self.run_launcher(
            scenario="dns_third_success", build_dns="192.0.2.53,2001:db8::53"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.builds()), 6)
        injected = [line for line in self.builds() if "PIPELINE_BUILD_DNS=" in line]
        self.assertEqual(len(injected), 2)
        self.assertTrue(all("--network=host" in line for line in injected))
        self.assertTrue(all(
            "PIPELINE_BUILD_DNS=192.0.2.53 2001:db8::53" in line
            for line in injected
        ))
        self.assertEqual(result.stderr.count("validated host DNS resolvers"), 2)

    def test_resolver_injection_failure_returns_real_status(self):
        result = self.run_launcher(
            scenario="dns_third_fail", build_dns="192.0.2.53"
        )
        self.assertEqual(result.returncode, 11)
        self.assertEqual(len(self.builds()), 3)
        self.assertIn("PIPELINE_BUILD_DNS=192.0.2.53", self.builds()[2])
        self.assertFalse(any(line.startswith("run ") for line in self.commands()))

    def test_invalid_explicit_dns_is_rejected_without_build_or_shell_execution(self):
        marker = self.root / "owned"
        result = self.run_launcher(build_dns=f"192.0.2.53;touch {marker.as_posix()}")
        self.assertEqual(result.returncode, 2)
        self.assertIn("valid, non-loopback IPv4/IPv6", result.stderr)
        self.assertEqual(self.builds(), [])
        self.assertFalse(marker.exists())

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

    def test_dockerfiles_apply_override_to_every_networked_build_step(self):
        controller = (REPOSITORY_ROOT / "docker/Dockerfile.controller").read_text(encoding="utf-8")
        trainer = (REPOSITORY_ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
        helper_call = '/bin/sh /usr/local/bin/configure-build-dns "$PIPELINE_BUILD_DNS"'
        self.assertEqual(controller.count(helper_call), 2)
        self.assertEqual(trainer.count(helper_call), 4)
        labels = "\n".join(
            line for line in controller.splitlines() + trainer.splitlines()
            if line.lstrip().startswith("LABEL")
        )
        self.assertNotIn("PIPELINE_BUILD_DNS", labels)


if __name__ == "__main__":
    unittest.main()

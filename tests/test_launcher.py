import os
import re
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

    def test_discovery_preserves_order_deduplicates_and_filters_invalid_stubs(self):
        result = self.discover(
            "nameserver 127.0.0.53\n"
            "nameserver 192.0.2.53\n"
            "nameserver malformed\n"
            "nameserver 2001:db8::53\n"
            "nameserver 192.0.2.53\n"
            "nameserver ::1\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "192.0.2.53 2001:db8::53")

    def test_discovery_reads_only_nameserver_entries(self):
        result = self.discover(
            "search private.invalid\noptions rotate\nnameserver 192.0.2.53 # comment\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "192.0.2.53")

    def test_empty_usable_resolver_set_fails(self):
        result = self.discover(
            "nameserver 127.0.0.1\nnameserver 127.0.0.53\n"
            "nameserver ::1\nnameserver not-an-ip\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_explicit_values_accept_normalize_and_deduplicate_literals(self):
        result = self.normalize("192.0.2.53, 2001:DB8::53 192.0.2.53 1::")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "192.0.2.53 2001:db8::53 1::")

    def test_explicit_values_reject_malformed_loopback_and_shell_text(self):
        for value in (
            "0.0.0.0", "::", "127.0.0.1", "127.0.0.53", "::1",
            "::ffff:127.0.0.1", "::ffff:7f00:1", "::ffff:7fff:ffff",
            "0:0:0:0:0:0:0:0", "0:0:0:0:0:0:0:1",
            "999.1.1.1", "2001::db8::1",
            "192.0.2.53;touch /tmp/owned", "resolver.example", "01.2.3.4",
        ):
            with self.subTest(value=value):
                self.assertNotEqual(self.normalize(value).returncode, 0)


@unittest.skipUnless(SHELL, "A POSIX-compatible shell is required for launcher tests")
class DockerBuildNetworkLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tests")
        self.external_temporary = tempfile.TemporaryDirectory(prefix="pipeline dns context ")
        self.root = Path(self.temporary.name)
        self.external_tmp = Path(self.external_temporary.name)
        for directory in ("scripts", "configs", "src", "docker", "fake-bin"):
            (self.root / directory).mkdir()
        shutil.copy2(REPOSITORY_ROOT / "scripts/run_pipeline.sh", self.root / "scripts/run_pipeline.sh")
        shutil.copy2(REPOSITORY_ROOT / "scripts/build_dns.sh", self.root / "scripts/build_dns.sh")
        (self.root / "configs/test.yaml").write_text(
            "orchestration:\n  enabled: true\n", encoding="utf-8"
        )
        (self.root / "pyproject.toml").write_text(
            "[project]\nname='launcher-test'\n", encoding="utf-8"
        )
        (self.root / "requirements.txt").write_text("", encoding="utf-8")
        self.log = self.root / "docker.log"
        self.context_log = self.root / "dns-contexts.log"
        self.evidence = self.root / "dns-evidence.log"
        fake = self.root / "fake-bin/docker"
        fake.write_text(
            """#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
if [ "${1-}" = info ]; then exit 0; fi
if [ "${1-} ${2-}" = "image inspect" ]; then
    case "$*" in
        *source-identity*)
            if [ "${FAKE_REUSE-0}" = 1 ]; then git -C "$FAKE_PROJECT" rev-parse HEAD; exit 0; fi
            exit 1 ;;
        *) echo sha256:training-image; exit 0 ;;
    esac
fi
if [ "${1-}" = build ]; then
    dns_context=
    previous=
    for argument in "$@"; do
        if [ "$previous" = --build-context ]; then
            case "$argument" in pipeline_dns=*) dns_context=${argument#pipeline_dns=} ;; esac
        fi
        previous=$argument
    done
    if [ -n "$dns_context" ]; then
        printf '%s\\n' "$dns_context" >> "$FAKE_CONTEXT_LOG"
        printf 'dir-mode=%s file-mode=%s\\n' \
            "$(stat -c '%a' "$dns_context")" \
            "$(stat -c '%a' "$dns_context/resolv.conf")" >> "$FAKE_DNS_EVIDENCE"
        sed 's/^/content=/' "$dns_context/resolv.conf" >> "$FAKE_DNS_EVIDENCE"
    fi
    case "${FAKE_BUILD_SCENARIO-success}: $*" in
        signal_context:*--build-context*) kill -TERM "$PPID"; sleep 1; exit 15 ;;
        dns_third_success:*--build-context*) exit 0 ;;
        dns_third_success:*--network=host*) echo 'Name or service not known' >&2; exit 8 ;;
        dns_third_success:*) echo 'Temporary failure resolving deb.debian.org' >&2; exit 7 ;;
        dns_third_fail:*--build-context*) echo 'DNS resolution still failed' >&2; exit 11 ;;
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
if [ "${1-}" = run ]; then exit 0; fi
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
        self.external_temporary.cleanup()

    def run_launcher(
        self, *, mode=None, scenario="success", reuse=False, rebuild=False,
        failure_message=None, build_dns=None,
    ):
        for path in (self.log, self.context_log, self.evidence):
            path.unlink(missing_ok=True)
        environment = os.environ.copy()
        environment.update({
            "PATH": f"{self.root / 'fake-bin'}{os.pathsep}{environment['PATH']}",
            "FAKE_DOCKER_LOG": self.log.as_posix(),
            "FAKE_CONTEXT_LOG": self.context_log.as_posix(),
            "FAKE_DNS_EVIDENCE": self.evidence.as_posix(),
            "FAKE_PROJECT": self.root.as_posix(),
            "FAKE_BUILD_SCENARIO": scenario,
            "FAKE_REUSE": "1" if reuse else "0",
            "PIPELINE_RUNS_DIR": (self.root / "runtime/runs").as_posix(),
            "PIPELINE_HF_CACHE_DIR": (self.root / "runtime/cache").as_posix(),
            "PIPELINE_STATE_DIR": (self.root / "runtime/state").as_posix(),
            "TMPDIR": self.external_tmp.as_posix(),
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

    def contexts(self):
        return self.context_log.read_text(encoding="utf-8").splitlines() if self.context_log.exists() else []

    def test_default_environment_is_auto_and_success_has_no_retry(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.builds()), 2)
        self.assertTrue(all("--network=host" not in line for line in self.builds()))
        self.assertNotIn("Retrying once", result.stderr)

    def test_accepted_modes_are_exact(self):
        for mode in ("auto", "default", "host", "host-dns"):
            with self.subTest(mode=mode):
                result = self.run_launcher(mode=mode, build_dns="192.0.2.53")
                self.assertEqual(result.returncode, 0, result.stderr)
        for mode in ("bridge", "HOST", "host_dns", "auto "):
            with self.subTest(mode=mode):
                result = self.run_launcher(mode=mode)
                self.assertEqual(result.returncode, 2)

    def test_invalid_mode_is_rejected_before_docker_access(self):
        result = self.run_launcher(mode="bridge")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be one of: auto, default, host, host-dns", result.stderr)
        self.assertEqual(self.commands(), [])

    def test_auto_dns_failure_retries_each_image_once_and_continues(self):
        result = self.run_launcher(scenario="dns_then_success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.builds()), 4)
        self.assertEqual(sum("--network=host" in line for line in self.builds()), 2)
        self.assertEqual(self.contexts(), [])

    def test_auto_failed_host_retry_with_unrelated_error_stops(self):
        result = self.run_launcher(scenario="dns_host_fail")
        self.assertEqual(result.returncode, 9)
        self.assertEqual(len(self.builds()), 2)
        self.assertEqual(self.contexts(), [])

    def test_auto_host_dns_failure_uses_exactly_three_attempts_per_image(self):
        result = self.run_launcher(
            scenario="dns_third_success", build_dns="192.0.2.53,2001:db8::53"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.builds()), 6)
        injected = [line for line in self.builds() if "--build-context pipeline_dns=" in line]
        self.assertEqual(len(injected), 2)
        self.assertTrue(all("--network=host" in line and ".host-dns" in line for line in injected))

    def test_dns_context_build_failure_returns_real_status(self):
        result = self.run_launcher(scenario="dns_third_fail", build_dns="192.0.2.53")
        self.assertEqual(result.returncode, 11)
        self.assertEqual(len(self.builds()), 3)
        self.assertFalse(any(line.startswith("run ") for line in self.commands()))

    def test_invalid_explicit_dns_is_rejected_without_build_or_shell_execution(self):
        marker = self.root / "owned"
        result = self.run_launcher(build_dns=f"192.0.2.53;touch {marker.as_posix()}")
        self.assertEqual(result.returncode, 2)
        self.assertIn("valid, non-loopback IPv4/IPv6", result.stderr)
        self.assertEqual(self.builds(), [])
        self.assertFalse(marker.exists())

    def test_host_dns_no_usable_explicit_dns_fails_immediately(self):
        result = self.run_launcher(mode="host-dns", build_dns="127.0.0.53")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.builds(), [])

    def test_auto_unrelated_default_failure_does_not_retry(self):
        for message in (
            "Dockerfile syntax error", "COPY failed: file not found",
            "ResolutionImpossible: dependency conflict", "permission denied",
        ):
            with self.subTest(message=message):
                result = self.run_launcher(scenario="unrelated", failure_message=message)
                self.assertEqual(result.returncode, 6)
                self.assertEqual(len(self.builds()), 1)

    def test_default_mode_is_exactly_one_normal_attempt(self):
        result = self.run_launcher(mode="default", scenario="dns")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(self.builds()), 1)
        self.assertNotIn("--network=host", self.builds()[0])
        self.assertNotIn("--build-context", self.builds()[0])

    def test_host_mode_is_exactly_one_host_attempt_per_image(self):
        result = self.run_launcher(mode="host")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.builds()), 2)
        self.assertTrue(all("--network=host" in line for line in self.builds()))
        self.assertTrue(all("--build-context" not in line for line in self.builds()))

    def test_host_mode_dns_failure_never_uses_context_fallback(self):
        result = self.run_launcher(mode="host", scenario="dns")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(self.builds()), 1)
        self.assertIn("--network=host", self.builds()[0])
        self.assertNotIn("--build-context", self.builds()[0])

    def test_host_dns_directly_uses_context_without_probe(self):
        result = self.run_launcher(mode="host-dns", build_dns="192.0.2.53")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.builds()), 2)
        self.assertTrue(all("--network=host" in line for line in self.builds()))
        self.assertTrue(all("--build-context pipeline_dns=" in line for line in self.builds()))
        self.assertTrue(all(".host-dns" in line for line in self.builds()))
        self.assertNotIn("Retrying once", result.stderr)

    def test_host_dns_directly_discovers_dns_when_no_override_is_set(self):
        with (self.root / "scripts/build_dns.sh").open("a", encoding="utf-8") as helper:
            helper.write("\ndiscover_build_dns() { printf '%s\\n' '192.0.2.54 2001:db8::54'; }\n")
        result = self.run_launcher(mode="host-dns")
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = self.evidence.read_text(encoding="utf-8")
        self.assertIn("content=nameserver 192.0.2.54", evidence)
        self.assertIn("content=nameserver 2001:db8::54", evidence)
        self.assertEqual(len(self.builds()), 2)

    def test_host_dns_without_a_usable_discovered_resolver_fails_before_build(self):
        with (self.root / "scripts/build_dns.sh").open("a", encoding="utf-8") as helper:
            helper.write("\ndiscover_build_dns() { return 2; }\n")
        result = self.run_launcher(mode="host-dns")
        self.assertEqual(result.returncode, 2)
        self.assertIn("No usable non-loopback DNS resolver", result.stderr)
        self.assertEqual(self.builds(), [])

    def test_auto_preserves_host_failure_status_when_no_resolver_is_discovered(self):
        with (self.root / "scripts/build_dns.sh").open("a", encoding="utf-8") as helper:
            helper.write("\ndiscover_build_dns() { return 2; }\n")
        result = self.run_launcher(scenario="dns_third_success")
        self.assertEqual(result.returncode, 8)
        self.assertEqual(len(self.builds()), 2)
        self.assertEqual(self.contexts(), [])

    def test_host_dns_generates_normalized_nameserver_only_context(self):
        result = self.run_launcher(
            mode="host-dns", build_dns="2001:DB8::53, 192.0.2.53 2001:db8::53"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = self.evidence.read_text(encoding="utf-8").splitlines()
        self.assertEqual(evidence.count("content=nameserver 2001:db8::53"), 2)
        self.assertEqual(evidence.count("content=nameserver 192.0.2.53"), 2)
        self.assertTrue(all(line.startswith(("dir-mode=", "content=nameserver ")) for line in evidence))

    def test_host_dns_context_permissions_are_private(self):
        result = self.run_launcher(mode="host-dns", build_dns="192.0.2.53")
        self.assertEqual(result.returncode, 0, result.stderr)
        permissions = [line for line in self.evidence.read_text().splitlines() if line.startswith("dir-mode=")]
        self.assertEqual(permissions, ["dir-mode=700 file-mode=600"] * 2)

    def test_host_dns_context_is_removed_after_success(self):
        result = self.run_launcher(mode="host-dns", build_dns="192.0.2.53")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.contexts())
        self.assertTrue(all(Path(path).parent == self.external_tmp for path in self.contexts()))
        self.assertTrue(all(not Path(path).is_relative_to(self.root) for path in self.contexts()))
        self.assertTrue(all(not Path(path).exists() for path in self.contexts()))

    def test_host_dns_context_is_removed_after_failure(self):
        result = self.run_launcher(mode="host-dns", scenario="dns_third_fail", build_dns="192.0.2.53")
        self.assertEqual(result.returncode, 11)
        self.assertEqual(len(self.contexts()), 1)
        self.assertFalse(Path(self.contexts()[0]).exists())

    def test_signal_cleanup_removes_host_dns_context(self):
        result = self.run_launcher(mode="host-dns", scenario="signal_context", build_dns="192.0.2.53")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.contexts()), 1)
        self.assertFalse(Path(self.contexts()[0]).exists())

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
        result = self.run_launcher(mode="host-dns", build_dns="192.0.2.53")
        self.assertEqual(result.returncode, 0, result.stderr)
        runtime = [line for line in self.commands() if line.startswith("run ")]
        self.assertEqual(len(runtime), 1)
        self.assertIn("--read-only --network host", runtime[0])
        self.assertNotIn("--network=host", runtime[0])

    def test_recovery_dockerfiles_cover_every_networked_step(self):
        normal_controller = (REPOSITORY_ROOT / "docker/Dockerfile.controller").read_text()
        normal_trainer = (REPOSITORY_ROOT / "docker/Dockerfile").read_text()
        recovery_controller = (REPOSITORY_ROOT / "docker/Dockerfile.controller.host-dns").read_text()
        recovery_trainer = (REPOSITORY_ROOT / "docker/Dockerfile.host-dns").read_text()
        mount = "RUN --mount=type=bind,from=pipeline_dns,source=resolv.conf,target=/etc/resolv.conf"
        self.assertEqual(recovery_controller.count(mount), 2)
        self.assertEqual(recovery_trainer.count(mount), 4)
        self.assertEqual(recovery_controller.count("apt-get update"), 1)
        self.assertEqual(recovery_controller.count("pip install"), 1)
        self.assertEqual(recovery_trainer.count("apt-get update"), 2)
        self.assertEqual(recovery_trainer.count("pip install pip==24.2"), 1)
        self.assertEqual(recovery_trainer.count("install-training-dependencies.sh"), 1)
        self.assertNotIn("pipeline_dns", normal_controller + normal_trainer)

    def test_recovery_dockerfiles_otherwise_match_ordinary_build_contracts(self):
        def canonicalize(contents):
            contents = re.sub(
                r"RUN --mount=type=bind,from=pipeline_dns,source=resolv\.conf,target=/etc/resolv\.conf \\\n\s*",
                "RUN ", contents,
            )
            return "\n".join(
                line for line in contents.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )

        pairs = (
            ("Dockerfile.controller", "Dockerfile.controller.host-dns"),
            ("Dockerfile", "Dockerfile.host-dns"),
        )
        for ordinary, recovery in pairs:
            with self.subTest(ordinary=ordinary):
                ordinary_text = (REPOSITORY_ROOT / "docker" / ordinary).read_text()
                recovery_text = (REPOSITORY_ROOT / "docker" / recovery).read_text()
                self.assertEqual(canonicalize(recovery_text), canonicalize(ordinary_text))

    def test_source_identity_excludes_runtime_resolver_context(self):
        launcher = (REPOSITORY_ROOT / "scripts/run_pipeline.sh").read_text()
        source_line = next(line for line in launcher.splitlines() if line.startswith("SOURCE_PATHS="))
        self.assertNotIn("DNS_CONTEXT", source_line)
        self.assertNotIn("resolv.conf", source_line)

    def test_failed_resolv_conf_rewriter_is_removed(self):
        self.assertFalse((REPOSITORY_ROOT / "docker/configure-build-dns.sh").exists())
        paths = (
            REPOSITORY_ROOT / "scripts/run_pipeline.sh",
            REPOSITORY_ROOT / "docker/Dockerfile",
            REPOSITORY_ROOT / "docker/Dockerfile.controller",
            REPOSITORY_ROOT / "docker/Dockerfile.host-dns",
            REPOSITORY_ROOT / "docker/Dockerfile.controller.host-dns",
        )
        combined = "\n".join(path.read_text() for path in paths)
        self.assertNotIn("configure-build-dns", combined)
        self.assertNotIn("PIPELINE_BUILD_DNS", combined)

    def test_production_dns_recovery_files_have_no_private_network_or_windows_paths(self):
        paths = (
            REPOSITORY_ROOT / "scripts/build_dns.sh",
            REPOSITORY_ROOT / "scripts/run_pipeline.sh",
            REPOSITORY_ROOT / "docker/Dockerfile.host-dns",
            REPOSITORY_ROOT / "docker/Dockerfile.controller.host-dns",
        )
        combined = "\n".join(path.read_text() for path in paths)
        self.assertNotRegex(combined, r"\b10\.\d+\.\d+\.\d+\b")
        self.assertNotRegex(combined, r"\b192\.168\.\d+\.\d+\b")
        self.assertNotRegex(combined, r"\b172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+\b")
        self.assertNotRegex(combined, r"(?i)C:\\Users\\[^\\\s]+")


if __name__ == "__main__":
    unittest.main()

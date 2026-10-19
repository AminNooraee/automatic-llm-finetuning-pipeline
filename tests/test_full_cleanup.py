from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "project1_full_cleanup", ROOT / "scripts" / "full_cleanup.py"
)
assert SPEC and SPEC.loader
CLEANUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLEANUP)


class FullCleanupTests(unittest.TestCase):
    def roots(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        runs, state, cache = base / "runs", base / "state", base / "cache"
        for path in (runs, state / "executions", cache):
            path.mkdir(parents=True)
        return base, runs, state, cache

    def test_discovers_run_from_early_state_without_success_artifacts(self):
        _, runs, state, _ = self.roots()
        claims = state / "claims"
        claims.mkdir()
        (claims / "exec-owned.json").write_text(
            json.dumps({"execution_id": "exec-owned", "pipeline_id": "101"}),
            encoding="utf-8",
        )
        run = runs / "run-owned"
        run.mkdir()
        (run / "ownership.json").write_text(
            json.dumps(
                {
                    "run_id": "run-owned",
                    "execution_id": "exec-owned",
                    "pipeline_id": "101",
                }
            ),
            encoding="utf-8",
        )
        foreign = runs / "run-foreign"
        foreign.mkdir()
        (foreign / "ownership.json").write_text(
            json.dumps({"run_id": "run-foreign", "pipeline_id": "202"}),
            encoding="utf-8",
        )

        found_runs, found_executions = CLEANUP.discover_targets(
            runs, state, run_id=None, execution_id=None, pipeline_id="101"
        )
        self.assertEqual(found_runs, {"run-owned"})
        self.assertEqual(found_executions, {"exec-owned"})

    def test_run_and_state_targets_are_exact_and_idempotent(self):
        _, runs, state, _ = self.roots()
        target = runs / "run-owned"
        target.mkdir()
        (target / "checkpoint-1").mkdir()
        (target / "checkpoint-1" / "state").write_text("x", encoding="utf-8")
        selected = CLEANUP.child(runs, "run-owned")
        self.assertGreater(CLEANUP.tree_size(selected), 0)
        import shutil

        shutil.rmtree(selected)
        self.assertFalse(selected.exists())
        self.assertEqual(CLEANUP.tree_size(selected), 0)
        self.assertTrue((state / "executions").exists())

    def test_dedicated_cache_is_cleared_but_shared_cache_is_preserved(self):
        _, _, _, cache = self.roots()
        (cache / "model.bin").write_bytes(b"model")

        class Docker:
            @staticmethod
            def run(*_args):
                from subprocess import CompletedProcess

                return CompletedProcess([], 0, "", "")

        self.assertEqual(CLEANUP.clear_dedicated_cache(cache, Docker(), False), 0)
        self.assertTrue((cache / "model.bin").exists())
        self.assertEqual(CLEANUP.clear_dedicated_cache(cache, Docker(), True), 5)
        self.assertFalse((cache / "model.bin").exists())

    def test_active_run_protects_images_and_cache(self):
        _, _, _, cache = self.roots()
        (cache / "model.bin").write_bytes(b"model")

        class Docker:
            @staticmethod
            def run(*_args):
                from subprocess import CompletedProcess

                return CompletedProcess([], 0, "active\n", "")

        self.assertEqual(CLEANUP.cleanup_images(Docker()), (0, 0))
        self.assertEqual(CLEANUP.clear_dedicated_cache(cache, Docker(), True), 0)
        self.assertTrue((cache / "model.bin").exists())

    def test_cleanup_has_no_broad_or_privileged_commands(self):
        text = (ROOT / "scripts" / "full_cleanup.py").read_text(encoding="utf-8")
        for forbidden in (
            "docker system prune",
            "volume prune",
            "chmod 777",
            "sudo",
            "rm -rf /home",
            "rm -rf /srv",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIn("provenance unavailable; safely preserved", text)

    def test_safe_rmtree_normal_deletion(self):
        _, runs, _, _ = self.roots()
        target = runs / "run-normal"
        target.mkdir()
        (target / "file.txt").write_text("hello", encoding="utf-8")
        docker = CLEANUP.Docker()
        CLEANUP.safe_rmtree(target, docker)
        self.assertFalse(target.exists())

    def test_safe_rmtree_permission_error_triggers_docker_fallback(self):
        _, runs, _, _ = self.roots()
        target = runs / "run-perm"
        target.mkdir()
        (target / "file.txt").write_text("root-owned", encoding="utf-8")

        class MockDocker:
            def __init__(self):
                self.commands = []

            def ids(self, kind, *filters):
                if kind == "image":
                    return ["controller-img-123"]
                return []

            def inspect(self, kind, identity):
                if identity == "controller-img-123":
                    return {
                        "Config": {
                            "Labels": {"fine-tuning-pipeline.image-role": "controller"}
                        }
                    }
                return None

            def run(self, *args):
                from subprocess import CompletedProcess

                self.commands.append(list(args))
                for item in list(target.iterdir()):
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                return CompletedProcess(list(args), 0, "", "")

        mock_docker = MockDocker()
        import unittest.mock

        with unittest.mock.patch(
            "shutil.rmtree", side_effect=PermissionError("Permission denied")
        ):
            CLEANUP.safe_rmtree(target, mock_docker)

        self.assertFalse(target.exists())
        self.assertEqual(len(mock_docker.commands), 1)
        cmd = mock_docker.commands[0]
        self.assertIn("run", cmd)
        self.assertIn("--rm", cmd)
        self.assertIn("--network", cmd)
        self.assertIn("none", cmd)
        self.assertIn(
            f"type=bind,src={target.resolve()},dst=/target",
            cmd[cmd.index("--mount") + 1],
        )
        self.assertIn("controller-img-123", cmd)
        # Isolation: no socket, no runs root broadly, no secrets
        cmd_str = " ".join(cmd)
        self.assertNotIn("docker.sock", cmd_str)
        self.assertNotIn("pipeline-secrets", cmd_str)
        self.assertNotIn(f"src={runs.resolve()},", cmd_str)

    def test_legacy_root_owned_state_directory_uses_same_safe_fallback(self):
        _, _, state, _ = self.roots()
        target = state / "executions" / "exec-legacy"
        target.mkdir(parents=True)
        (target / "claim.json").write_text("{}", encoding="utf-8")

        class MockDocker:
            def __init__(self):
                self.commands = []

            def ids(self, kind, *filters):
                return ["controller-img-456"] if kind == "image" else []

            def inspect(self, kind, identity):
                return {
                    "Config": {
                        "Labels": {"fine-tuning-pipeline.image-role": "controller"}
                    }
                }

            def run(self, *args):
                from subprocess import CompletedProcess

                self.commands.append(list(args))
                for item in list(target.iterdir()):
                    item.unlink()
                return CompletedProcess(list(args), 0, "", "")

        mock_docker = MockDocker()
        import unittest.mock

        with unittest.mock.patch(
            "shutil.rmtree", side_effect=PermissionError("Permission denied")
        ):
            CLEANUP.safe_rmtree(target, mock_docker)

        self.assertFalse(target.exists())
        self.assertEqual(len(mock_docker.commands), 1)
        self.assertIn(
            f"type=bind,src={target.resolve()},dst=/target", mock_docker.commands[0]
        )

    def test_safe_rmtree_fails_safely_if_no_trusted_helper_image(self):
        _, runs, _, _ = self.roots()
        target = runs / "run-no-helper"
        target.mkdir()

        class MockDockerEmpty:
            def ids(self, kind, *filters):
                return []

            def inspect(self, kind, identity):
                return None

            def run(self, *args):
                from subprocess import CompletedProcess

                return CompletedProcess([], 0, "", "")

        import unittest.mock

        with unittest.mock.patch(
            "shutil.rmtree", side_effect=PermissionError("Permission denied")
        ):
            with self.assertRaises(SystemExit) as ctx:
                CLEANUP.safe_rmtree(target, MockDockerEmpty())
            self.assertIn(
                "no trusted Project 1 helper image available", str(ctx.exception)
            )

    def test_child_path_validation_and_symlink_rejection(self):
        _, runs, _, _ = self.roots()
        # Invalid IDs
        for bad in ("..", ".", "../etc", "a/b", "/root"):
            with self.assertRaises(SystemExit):
                CLEANUP.child(runs, bad)
        # Symlink target
        real_dir = runs / "real_dir"
        real_dir.mkdir()
        symlink_dir = runs / "symlink_dir"
        try:
            symlink_dir.symlink_to(real_dir)
            with self.assertRaises(SystemExit):
                CLEANUP.child(runs, "symlink_dir")
            with self.assertRaises(SystemExit):
                CLEANUP.safe_rmtree(symlink_dir, CLEANUP.Docker())
        except (OSError, NotImplementedError):
            pass  # Windows may not allow unprivileged symlinks

    def test_missing_litellm_and_container_is_idempotent(self):
        class MockDocker:
            def ids(self, kind, *filters):
                return []

            def inspect(self, kind, identity):
                return None

            def run(self, *args):
                from subprocess import CompletedProcess

                return CompletedProcess([], 0, "", "")

        # Container cleanup on empty list
        self.assertEqual(CLEANUP.cleanup_containers(MockDocker(), []), 0)
        # Container cleanup on nonexistent container ID
        self.assertEqual(
            CLEANUP.cleanup_containers(MockDocker(), ["missing-container"]), 0
        )

        # LiteLLM cleanup on empty data
        import unittest.mock

        with unittest.mock.patch.object(
            CLEANUP, "api", return_value=(200, {"data": []})
        ):
            self.assertEqual(
                CLEANUP.cleanup_litellm({"run-1"}, "http://localhost:4000", "key"), 0
            )

    def test_already_deleted_target_is_idempotent(self):
        _, runs, _, _ = self.roots()
        target = runs / "already-deleted"
        docker = CLEANUP.Docker()
        # Does not raise
        CLEANUP.safe_rmtree(target, docker)
        self.assertFalse(target.exists())

    def test_realistic_legacy_mixed_ownership_simulation(self):
        _, runs, _, _ = self.roots()
        run_id = "20261019_091805_qwen2-5-0-5b-instruct_example-qwen-job-train"
        target_run = runs / run_id
        target_run.mkdir()
        (target_run / "model").mkdir()
        (target_run / "model" / "adapter_model.safetensors").write_bytes(b"adapter")
        (target_run / "gateway").mkdir()
        (target_run / "gateway" / "gateway_manifest.json").write_text(
            "{}", encoding="utf-8"
        )
        (target_run / "serving").mkdir()
        (target_run / "serving" / "operation.json").write_text("{}", encoding="utf-8")
        (target_run / "deployment_manifest.json").write_text("{}", encoding="utf-8")
        (target_run / "ownership.json").write_text(
            json.dumps(
                {"run_id": run_id, "execution_id": "exec-sim", "pipeline_id": "999"}
            ),
            encoding="utf-8",
        )

        foreign_run = runs / "foreign_run_123"
        foreign_run.mkdir()
        (foreign_run / "ownership.json").write_text(
            json.dumps({"run_id": "foreign_run_123", "pipeline_id": "foreign"}),
            encoding="utf-8",
        )

        class MockDocker:
            def __init__(self):
                self.commands = []

            def ids(self, kind, *filters):
                return ["controller-img-sim"] if kind == "image" else []

            def inspect(self, kind, identity):
                return {
                    "Config": {
                        "Labels": {"fine-tuning-pipeline.image-role": "controller"}
                    },
                    "Size": 5000,
                }

            def run(self, *args):
                from subprocess import CompletedProcess

                self.commands.append(list(args))
                if args and args[0] == "run":
                    import shutil

                    for item in list(target_run.iterdir()):
                        if item.is_file() or item.is_symlink():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                return CompletedProcess(list(args), 0, "", "")

        mock_docker = MockDocker()
        import unittest.mock

        original_rmtree = shutil.rmtree

        def failing_rmtree(path, *args, **kwargs):
            if Path(path) == target_run:
                raise PermissionError(
                    "[Errno 13] Permission denied: 'gateway_manifest.json'"
                )
            return original_rmtree(path, *args, **kwargs)

        with unittest.mock.patch("shutil.rmtree", side_effect=failing_rmtree):
            CLEANUP.safe_rmtree(target_run, mock_docker)

        # Target run was removed via fallback
        self.assertFalse(target_run.exists())
        # Foreign run was NOT touched
        self.assertTrue(foreign_run.exists())
        self.assertTrue((foreign_run / "ownership.json").exists())
        # Rerunning is safe and idempotent
        CLEANUP.safe_rmtree(target_run, mock_docker)
        self.assertFalse(target_run.exists())


if __name__ == "__main__":
    unittest.main()

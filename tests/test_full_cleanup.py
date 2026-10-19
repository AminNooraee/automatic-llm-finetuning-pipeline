from __future__ import annotations

import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()

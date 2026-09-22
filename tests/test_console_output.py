import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

from fine_tuning_pipeline.console_output import (
    ConsoleOutputFormatter,
    ConsoleVerbosityError,
    resolve_console_verbosity,
)
from fine_tuning_pipeline.train_pipeline import (
    DEFAULT_CONFIG_PATH,
    execute_training,
    prepare_training,
)
from fine_tuning_pipeline.trainer import run_training


TESTS_DIR = Path(__file__).resolve().parent
RAW_OUTPUT = (
    "loading configuration file config.json from cache...\n"
    "Qwen2Config {\n  hidden_config: true\n}\n"
    "***** Running training *****\n"
    "Num examples = 1200\n"
    "Num Epochs = 3\n"
    " 50%|#####     | 75/150 [02:21<02:19, 1.86s/it]\r"
    "{'loss': '1.234', 'grad_norm': '0.91', "
    "'learning_rate': '0.0001', 'epoch': '0.5'}\n"
    "WARNING important backend warning\n"
    "input_ids: [1, 2, 3, 4]\n"
    "{'train_runtime': '288.1', 'train_loss': '0.75', "
    "'train_samples_per_second': '4.16', 'train_steps_per_second': '0.52'}\n"
)


class ConsoleOutputFormatterTests(unittest.TestCase):
    def render(self, mode, raw=RAW_OUTPUT):
        stream = io.StringIO()
        formatter = ConsoleOutputFormatter(mode, stream)
        for index in range(0, len(raw), 17):
            formatter.feed(raw[index : index + 17])
        formatter.close()
        return stream.getvalue()

    def test_default_verbosity_is_concise_and_legacy_config_works(self):
        self.assertEqual(resolve_console_verbosity({}), "concise")
        self.assertEqual(
            resolve_console_verbosity(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())),
            "concise",
        )

    def test_invalid_or_unbounded_observability_config_is_rejected(self):
        with self.assertRaisesRegex(ConsoleVerbosityError, "must be one of"):
            resolve_console_verbosity(
                {"observability": {"console_verbosity": "verbose"}}
            )
        with self.assertRaisesRegex(ConsoleVerbosityError, "Unsupported observability"):
            resolve_console_verbosity({"observability": {"debug_dump": True}})

    def test_quiet_hides_backend_progress_but_surfaces_errors(self):
        output = self.render(
            "quiet",
            "INFO loading cache\nERROR CUDA unavailable\nTraceback (most recent call last):\n"
            "  File 'trainer.py', line 1\nRuntimeError: failed\n",
        )
        self.assertNotIn("loading cache", output)
        self.assertIn("ERROR CUDA unavailable", output)
        self.assertIn("RuntimeError: failed", output)

    def test_concise_shows_metrics_and_warning_but_hides_noise(self):
        output = self.render("concise")
        for expected in (
            "Running training",
            "Num examples = 1200",
            "Step 75/150",
            "Epoch 0.5",
            "Loss 1.234",
            "GradNorm 0.91",
            "LR 0.0001",
            "Train loss 0.75",
            "Runtime 288.1s",
            "WARNING important backend warning",
        ):
            self.assertIn(expected, output)
        for hidden in ("loading configuration", "Qwen2Config", "hidden_config", "input_ids"):
            self.assertNotIn(hidden, output)

    def test_full_is_exactly_the_raw_stream(self):
        self.assertEqual(self.render("full"), RAW_OUTPUT)

    def test_carriage_return_progress_is_throttled(self):
        raw = "".join(
            f" {step}%|bar| {step}/100 [00:01<00:01, 2.0it/s]\r"
            for step in range(1, 101)
        )
        output = self.render("concise", raw)
        progress_lines = output.splitlines()
        self.assertLessEqual(len(progress_lines), 21)
        self.assertIn("Progress: 100/100 (100%)", output)


class TrainerConsoleModeTests(unittest.TestCase):
    def temporary_dir(self):
        temporary = tempfile.TemporaryDirectory(dir=TESTS_DIR)
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name).resolve()

    @staticmethod
    def process(output=RAW_OUTPUT, returncode=0):
        return type(
            "Process",
            (),
            {
                "stdout": io.BytesIO(output.encode("utf-8")),
                "wait": lambda self: returncode,
            },
        )()

    def run_mode(self, mode, *, returncode=0, output=RAW_OUTPUT):
        log = self.temporary_dir() / f"{mode}.log"
        terminal = io.StringIO()
        with patch(
            "fine_tuning_pipeline.trainer.subprocess.Popen",
            return_value=self.process(output, returncode),
        ):
            with redirect_stdout(terminal):
                if returncode:
                    with self.assertRaisesRegex(RuntimeError, f"exit code {returncode}"):
                        run_training("training.yaml", log_file=log, console_verbosity=mode)
                else:
                    run_training("training.yaml", log_file=log, console_verbosity=mode)
        return terminal.getvalue(), log.read_bytes().decode("utf-8")

    def test_raw_log_is_complete_in_every_console_mode(self):
        for mode in ("quiet", "concise", "full"):
            with self.subTest(mode=mode):
                _terminal, log = self.run_mode(mode)
                self.assertIn(RAW_OUTPUT, log)

    def test_full_mode_streams_all_raw_output(self):
        terminal, _log = self.run_mode("full")
        self.assertIn(RAW_OUTPUT, terminal)

    def test_failure_output_is_logged_visible_and_propagated(self):
        terminal, log = self.run_mode(
            "concise", returncode=9, output="ERROR backend exploded\n"
        )
        self.assertIn("ERROR backend exploded", terminal)
        self.assertIn("ERROR backend exploded", log)

    def test_invalid_config_fails_before_run_allocation(self):
        root = self.temporary_dir()
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["dataset"]["path"] = str(
            DEFAULT_CONFIG_PATH.parent.parent / "examples" / "datasets" / "alpaca_demo.json"
        )
        config["observability"] = {"console_verbosity": "everything"}
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        runs = root / "runs"
        with self.assertRaisesRegex(ConsoleVerbosityError, "must be one of"):
            prepare_training(config_path, artifact_root=runs)
        self.assertFalse(runs.exists())

    def test_quiet_pipeline_shows_lifecycle_without_preparation_chatter(self):
        root = self.temporary_dir()
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["dataset"]["path"] = str(
            DEFAULT_CONFIG_PATH.parent.parent / "examples" / "datasets" / "alpaca_demo.json"
        )
        config["observability"] = {"console_verbosity": "quiet"}
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        def create_adapter(yaml_file, log_file=None, **kwargs):
            arguments = yaml.safe_load(Path(yaml_file).read_text(encoding="utf-8"))
            model_dir = Path(arguments["output_dir"])
            (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "adapter_model.safetensors").write_bytes(b"weights")
            self.assertEqual(kwargs["console_verbosity"], "quiet")

        terminal = io.StringIO()
        with patch(
            "fine_tuning_pipeline.model_manager._load_huggingface_config",
            return_value={"model_type": "qwen2"},
        ), patch(
            "fine_tuning_pipeline.train_pipeline.run_training",
            side_effect=create_adapter,
        ), redirect_stdout(terminal):
            run_dir = execute_training(config_path, runs_root=root / "runs")

        output = terminal.getvalue()
        self.assertIn("Created run", output)
        self.assertIn("=== Training Started ===", output)
        self.assertIn("=== Training Complete ===", output)
        self.assertIn("Status: SUCCESS", output)
        self.assertNotIn("Resolving model compatibility", output)
        self.assertNotIn("dataset_info.json generated", output)
        self.assertNotIn("LLaMA-Factory config created", output)
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "success")


if __name__ == "__main__":
    unittest.main()

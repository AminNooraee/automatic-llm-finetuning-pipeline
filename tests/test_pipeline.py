import json
import os
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from fine_tuning_pipeline.dataset_adapter import adapt_dataset, load_raw_dataset
from fine_tuning_pipeline.dataset_loader import validate_unified_dataset
from fine_tuning_pipeline.train_pipeline import DEFAULT_CONFIG_PATH, REPOSITORY_ROOT, prepare_training
from fine_tuning_pipeline.trainer import run_training


TESTS_DIR = Path(__file__).resolve().parent


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self):
        model_config_patcher = patch(
            "fine_tuning_pipeline.model_manager._load_huggingface_config",
            return_value={
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
            },
        )
        model_config_patcher.start()
        self.addCleanup(model_config_patcher.stop)

    def temporary_project_dir(self):
        temporary = tempfile.TemporaryDirectory(dir=TESTS_DIR)
        path = Path(temporary.name).resolve()
        self.assertTrue(path.is_relative_to(TESTS_DIR))
        self.addCleanup(temporary.cleanup)
        return path

    def single_run(self, runs_root):
        runs = [path for path in runs_root.iterdir() if path.is_dir()]
        self.assertEqual(len(runs), 1)
        return runs[0]

    def test_current_alpaca_dataset_keeps_its_training_fields(self):
        raw = load_raw_dataset(REPOSITORY_ROOT / "examples" / "datasets" / "alpaca_demo.json")
        unified = validate_unified_dataset(adapt_dataset(raw))
        self.assertEqual(unified, raw)

    def test_qa_variants_normalize_to_alpaca_fields(self):
        raw = [
            {"question": "What is AI?", "answer": "Intelligent computing", "context": "briefly"},
            {"prompt": "What is ML?", "response": "Learning from data"},
        ]
        unified = validate_unified_dataset(adapt_dataset(raw, "qa_json"))
        self.assertEqual(
            unified,
            [
                {"instruction": "What is AI?", "input": "briefly", "output": "Intelligent computing"},
                {"instruction": "What is ML?", "input": "", "output": "Learning from data"},
            ],
        )

    def test_empty_answer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty answer"):
            adapt_dataset([{"question": "What is AI?", "answer": "   "}])

    def test_qwen_artifacts_work_from_project_root(self):
        root = self.temporary_project_dir()
        vendor_registry = REPOSITORY_ROOT / "LLaMA-Factory" / "data" / "dataset_info.json"
        vendor_before = vendor_registry.read_bytes() if vendor_registry.is_file() else None
        previous_cwd = Path.cwd()
        try:
            os.chdir(REPOSITORY_ROOT)
            resolved_config, yaml_file = prepare_training(artifact_root=root / "artifacts")
        finally:
            os.chdir(previous_cwd)

        run_dir = Path(yaml_file).parents[1]
        args = yaml.safe_load(Path(yaml_file).read_text(encoding="utf-8"))
        registry = json.loads(
            (run_dir / "dataset" / "dataset_info.json").read_text(encoding="utf-8")
        )
        dataset = Path(args["dataset_dir"]) / registry["demo_dataset"]["file_name"]
        unified = json.loads(dataset.read_text(encoding="utf-8"))
        self.assertTrue(dataset.is_file())
        self.assertEqual(len(unified), 2)
        self.assertEqual(args["model_name_or_path"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(args["template"], "qwen")
        self.assertEqual(resolved_config["model"]["family"], "qwen")
        self.assertEqual(resolved_config["model"]["template"], "qwen")
        self.assertEqual(args["finetuning_type"], "lora")
        self.assertEqual(args["num_train_epochs"], 1)
        self.assertEqual(args["per_device_train_batch_size"], 1)
        self.assertEqual(args["learning_rate"], 0.0002)
        self.assertIs(args["fp16"], False)
        self.assertIs(args["bf16"], False)
        self.assertIs(args["disable_gradient_checkpointing"], True)
        self.assertEqual(args["cutoff_len"], 2048)
        self.assertEqual(args["save_steps"], 10)
        self.assertEqual(args["logging_steps"], 1)
        self.assertEqual(args["lora_rank"], 8)
        self.assertEqual(args["lora_alpha"], 16)
        self.assertEqual(args["lora_dropout"], 0.0)
        self.assertEqual(args["output_dir"], str(run_dir / "model"))
        self.assertEqual(Path(yaml_file), run_dir / "config" / "training.yaml")
        self.assertTrue((run_dir / "config" / "input_config.yaml").is_file())
        self.assertTrue((run_dir / "config" / "resolved_config.yaml").is_file())
        self.assertTrue((run_dir / "dataset" / "original_dataset.json").is_file())
        self.assertTrue((run_dir / "logs" / "train.log").is_file())
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "prepared")
        self.assertEqual(metadata["dataset"]["num_samples"], 2)
        self.assertNotEqual(Path(args["dataset_dir"]), vendor_registry.parent)
        if vendor_before is not None:
            self.assertEqual(vendor_registry.read_bytes(), vendor_before)

    def test_invalid_training_config_stops_before_artifacts(self):
        root = self.temporary_project_dir()
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["training"]["unsupported_option"] = True
        custom_config = root / "config.yaml"
        custom_config.write_text(yaml.safe_dump(config), encoding="utf-8")

        artifact_root = root / "artifacts"
        with self.assertRaisesRegex(ValueError, "Unsupported training parameter"):
            prepare_training(custom_config, artifact_root=artifact_root)
        run_dir = self.single_run(artifact_root)
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "failed")
        self.assertIn("Unsupported training parameter", metadata["error"]["message"])

    def test_raw_qa_file_flows_into_local_training_configs(self):
        root = self.temporary_project_dir()
        raw = [{"question": "What is AI?", "answer": "Intelligent computing"}]
        (root / "qa.json").write_text(json.dumps(raw), encoding="utf-8")
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["dataset"] = {"path": "qa.json", "name": "qa_example", "format": "qa_json"}
        config["output"] = {"path": str(root / "models"), "model_name": "qa_example"}
        custom_config = root / "config.yaml"
        custom_config.write_text(yaml.safe_dump(config), encoding="utf-8")

        _, yaml_file = prepare_training(custom_config, artifact_root=root / "artifacts")
        run_dir = Path(yaml_file).parents[1]
        registry = json.loads(
            (run_dir / "dataset" / "dataset_info.json").read_text(encoding="utf-8")
        )
        unified = json.loads(
            (run_dir / "dataset" / "normalized_dataset.json").read_text(encoding="utf-8")
        )
        self.assertEqual(registry["qa_example"]["file_name"], "normalized_dataset.json")
        self.assertEqual(unified, [{"instruction": "What is AI?", "input": "", "output": "Intelligent computing"}])

    def test_invalid_data_does_not_create_training_artifacts(self):
        root = self.temporary_project_dir()
        (root / "qa.json").write_text(json.dumps([{"question": "What?", "answer": ""}]), encoding="utf-8")
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["dataset"] = {"path": "qa.json", "name": "qa_example", "format": "qa_json"}
        custom_config = root / "config.yaml"
        custom_config.write_text(yaml.safe_dump(config), encoding="utf-8")

        artifact_root = root / "artifacts"
        with self.assertRaisesRegex(ValueError, "non-empty answer"):
            prepare_training(custom_config, artifact_root=artifact_root)
        run_dir = self.single_run(artifact_root)
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "failed")

    def test_dpo_data_is_recognized_without_enabling_dpo_training(self):
        root = self.temporary_project_dir()
        raw = [{"prompt": "Question", "chosen": "Good", "rejected": "Bad"}]
        (root / "dpo.json").write_text(json.dumps(raw), encoding="utf-8")
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["dataset"] = {"path": "dpo.json", "name": "dpo_example", "format": "auto"}
        custom_config = root / "config.yaml"
        custom_config.write_text(yaml.safe_dump(config), encoding="utf-8")

        artifact_root = root / "artifacts"
        with self.assertRaisesRegex(ValueError, "SFT training only"):
            prepare_training(custom_config, artifact_root=artifact_root)
        run_dir = self.single_run(artifact_root)
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "failed")

    def test_incompatible_model_template_stops_before_artifacts(self):
        root = self.temporary_project_dir()
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["model"]["template"] = "mistral"
        custom_config = root / "config.yaml"
        custom_config.write_text(yaml.safe_dump(config), encoding="utf-8")

        artifact_root = root / "artifacts"
        with self.assertRaisesRegex(ValueError, "resolves to the 'qwen'"):
            prepare_training(custom_config, artifact_root=artifact_root)
        run_dir = self.single_run(artifact_root)
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "failed")

    def test_llama_template_flows_into_generated_training_yaml(self):
        root = self.temporary_project_dir()
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["model"] = {"name": "meta-llama/Meta-Llama-3-8B-Instruct"}
        config["dataset"]["path"] = str(REPOSITORY_ROOT / "examples" / "datasets" / "alpaca_demo.json")
        config["output"] = {"path": str(root / "models"), "model_name": "llama_example"}
        custom_config = root / "config.yaml"
        custom_config.write_text(yaml.safe_dump(config), encoding="utf-8")

        with patch(
            "fine_tuning_pipeline.model_manager._load_huggingface_config",
            return_value={"model_type": "llama", "vocab_size": 128256},
        ):
            resolved_config, yaml_file = prepare_training(
                custom_config,
                artifact_root=root / "artifacts",
            )

        args = yaml.safe_load(Path(yaml_file).read_text(encoding="utf-8"))
        self.assertEqual(resolved_config["model"]["family"], "llama")
        self.assertEqual(args["template"], "llama3")

    def test_training_uses_current_interpreter_and_propagates_failure(self):
        with patch("fine_tuning_pipeline.trainer.subprocess.run") as command_runner:
            command_runner.return_value.returncode = 0
            run_training("train.yaml")
            command = command_runner.call_args.args[0]
            self.assertEqual(command[:4], [sys.executable, "-m", "llamafactory.cli", "train"])
            self.assertEqual(command[4], str(Path("train.yaml").resolve()))

            command_runner.return_value.returncode = 3
            with self.assertRaisesRegex(RuntimeError, "exit code 3"):
                run_training("train.yaml")

    def test_training_subprocess_output_is_captured_in_run_log(self):
        root = self.temporary_project_dir()
        log_file = root / "logs" / "train.log"

        process = type(
            "Process",
            (),
            {"stdout": StringIO("llamafactory output\n"), "wait": lambda self: 0},
        )()

        with patch("fine_tuning_pipeline.trainer.subprocess.Popen", return_value=process) as runner:
            run_training("train.yaml", log_file=log_file)

        self.assertEqual(runner.call_args.kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(runner.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertIn("Launching command:", log_file.read_text(encoding="utf-8"))
        self.assertIn("llamafactory output", log_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

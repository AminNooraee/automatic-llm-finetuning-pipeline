import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning_pipeline.dataset_adapters import (
    DEFAULT_FRAMEWORK,
    DatasetValidationError,
    OptionalDependencyError,
)


class DatasetSourceAdapterTests(unittest.TestCase):
    def temporary_directory(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        path = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        return path

    def test_json_array_detection_and_conversion(self):
        root = self.temporary_directory()
        path = root / "data.json"
        rows = [{"instruction": "Question", "output": "Answer"}]
        path.write_text(json.dumps(rows), encoding="utf-8")
        result = DEFAULT_FRAMEWORK.convert_dataset(path)
        self.assertEqual(result.source_format, "json")
        self.assertEqual(result.dataset_format, "alpaca")
        self.assertEqual(result.records[0]["input"], "")

    def test_jsonl_detection_and_conversion(self):
        root = self.temporary_directory()
        path = root / "data.jsonl"
        path.write_text(
            json.dumps({"prompt": "Question", "completion": "Answer"}) + "\n",
            encoding="utf-8",
        )
        result = DEFAULT_FRAMEWORK.convert_dataset(path)
        self.assertEqual(result.source_format, "jsonl")
        self.assertEqual(result.dataset_format, "prompt_completion")

    def test_csv_detection_and_conversion(self):
        root = self.temporary_directory()
        path = root / "data.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerow({"question": "Question", "answer": "Answer"})
        result = DEFAULT_FRAMEWORK.convert_dataset(path)
        self.assertEqual(result.source_format, "csv")
        self.assertEqual(result.dataset_format, "qa_json")

    def test_csv_dialect_is_detected(self):
        root = self.temporary_directory()
        path = root / "data.csv"
        path.write_text("question;answer\nQuestion;Answer\n", encoding="utf-8")
        result = DEFAULT_FRAMEWORK.convert_dataset(path)
        self.assertEqual(
            result.records,
            [{"instruction": "Question", "input": "", "output": "Answer"}],
        )

    def test_parquet_detection_and_conversion(self):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is not installed")
        root = self.temporary_directory()
        path = root / "data.parquet"
        parquet.write_table(
            arrow.Table.from_pylist([{"instruction": "Question", "response": "Answer"}]),
            path,
        )
        result = DEFAULT_FRAMEWORK.convert_dataset(path)
        self.assertEqual(result.source_format, "parquet")
        self.assertEqual(result.dataset_format, "instruction_response")

    def test_raw_chatml_text_detection_and_conversion(self):
        root = self.temporary_directory()
        path = root / "conversation.txt"
        path.write_text(
            "<|im_start|>user\nQuestion<|im_end|>\n"
            "<|im_start|>assistant\nAnswer<|im_end|>",
            encoding="utf-8",
        )
        result = DEFAULT_FRAMEWORK.convert_dataset(path)
        self.assertEqual(result.source_format, "chatml_text")
        self.assertEqual(result.dataset_format, "chatml")

    def test_huggingface_loader_is_lazy_and_uses_configured_split(self):
        loaded = [{"question": "Question", "answer": "Answer"}]
        with patch("fine_tuning_pipeline.dataset_adapters.huggingface_adapter._load_dataset", return_value=loaded) as loader:
            result = DEFAULT_FRAMEWORK.convert_dataset(
                "organization/dataset",
                options={"subset": "english", "split": "validation"},
            )
        loader.assert_called_once_with(
            "organization/dataset", "english", split="validation"
        )
        self.assertEqual(result.source_format, "huggingface")
        self.assertEqual(result.dataset_format, "qa_json")

    def test_missing_huggingface_dependency_is_actionable(self):
        with patch(
            "fine_tuning_pipeline.dataset_adapters.huggingface_adapter._load_dataset",
            side_effect=OptionalDependencyError("install datasets"),
        ):
            with self.assertRaisesRegex(OptionalDependencyError, "install datasets"):
                DEFAULT_FRAMEWORK.convert_dataset("organization/dataset")

    def test_invalid_jsonl_reports_line_number(self):
        root = self.temporary_directory()
        path = root / "bad.jsonl"
        path.write_text('{"question": "valid", "answer": "yes"}\n{bad}\n', encoding="utf-8")
        with self.assertRaisesRegex(DatasetValidationError, "line 2"):
            DEFAULT_FRAMEWORK.convert_dataset(path, source_format="jsonl")

    def test_invalid_json_array_is_rejected(self):
        root = self.temporary_directory()
        path = root / "bad.json"
        path.write_text('{"instruction": "not an array"}', encoding="utf-8")
        with self.assertRaisesRegex(DatasetValidationError, "top-level array"):
            DEFAULT_FRAMEWORK.convert_dataset(path, source_format="json")

    def test_invalid_csv_header_is_rejected(self):
        root = self.temporary_directory()
        path = root / "bad.csv"
        path.write_text(",answer\nQuestion,Answer\n", encoding="utf-8")
        with self.assertRaisesRegex(DatasetValidationError, "header"):
            DEFAULT_FRAMEWORK.convert_dataset(path, source_format="csv")

    def test_invalid_parquet_is_rejected(self):
        root = self.temporary_directory()
        path = root / "bad.parquet"
        path.write_bytes(b"PAR1not-a-valid-parquet-file")
        with self.assertRaisesRegex(DatasetValidationError, "Unable to read Parquet"):
            DEFAULT_FRAMEWORK.convert_dataset(path, source_format="parquet")

    def test_invalid_chatml_text_is_rejected(self):
        root = self.temporary_directory()
        path = root / "bad.txt"
        path.write_text("plain text", encoding="utf-8")
        with self.assertRaisesRegex(DatasetValidationError, "malformed"):
            DEFAULT_FRAMEWORK.convert_dataset(
                path,
                source_format="chatml_text",
                dataset_format="chatml",
            )

    def test_missing_local_source_is_not_treated_as_huggingface(self):
        root = self.temporary_directory()
        with self.assertRaises(FileNotFoundError):
            DEFAULT_FRAMEWORK.convert_dataset(root / "missing.json")


if __name__ == "__main__":
    unittest.main()

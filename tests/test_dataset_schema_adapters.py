import unittest

from fine_tuning_pipeline.dataset_adapters import (
    AdapterContext,
    DEFAULT_FRAMEWORK,
    DatasetFormatDetectionError,
    DatasetValidationError,
    UnsupportedDatasetTaskError,
)


class DatasetSchemaAdapterTests(unittest.TestCase):
    def convert(self, rows, dataset_format="auto", require_task="sft"):
        return DEFAULT_FRAMEWORK.convert_rows(
            rows,
            dataset_format=dataset_format,
            require_task=require_task,
        )

    def test_field_mapping_formats_detect_and_convert(self):
        cases = [
            (
                "alpaca",
                {"instruction": "Explain AI", "input": "Briefly", "output": "Answer"},
                {"instruction": "Explain AI", "input": "Briefly", "output": "Answer"},
            ),
            (
                "instruction_response",
                {"instruction": "Explain AI", "response": "Answer"},
                {"instruction": "Explain AI", "input": "", "output": "Answer"},
            ),
            (
                "prompt_completion",
                {"prompt": "Explain AI", "completion": "Answer"},
                {"instruction": "Explain AI", "input": "", "output": "Answer"},
            ),
            (
                "qa_json",
                {"question": "What is AI?", "answer": "Answer"},
                {"instruction": "What is AI?", "input": "", "output": "Answer"},
            ),
            (
                "rag_qa",
                {"context": "Reference", "question": "What is AI?", "answer": "Answer"},
                {"instruction": "What is AI?", "input": "Reference", "output": "Answer"},
            ),
        ]
        for expected_format, raw, expected in cases:
            with self.subTest(expected_format=expected_format):
                result = self.convert([raw])
                self.assertEqual(result.dataset_format, expected_format)
                self.assertGreaterEqual(result.confidence, 0.86)
                self.assertEqual(result.records, [expected])

    def test_openai_chat_detects_and_expands_assistant_turns(self):
        raw = [{
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
                {"role": "assistant", "content": "Second answer"},
            ]
        }]
        result = self.convert(raw)
        self.assertEqual(result.dataset_format, "openai_chat")
        self.assertEqual(
            result.records,
            [
                {
                    "instruction": "First question",
                    "input": "System: Be concise",
                    "output": "First answer",
                },
                {
                    "instruction": "Second question",
                    "input": (
                        "System: Be concise\nUser: First question\n"
                        "Assistant: First answer"
                    ),
                    "output": "Second answer",
                },
            ],
        )

    def test_sharegpt_detects_and_converts(self):
        raw = [{
            "conversations": [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Answer"},
            ]
        }]
        result = self.convert(raw)
        self.assertEqual(result.dataset_format, "sharegpt")
        self.assertEqual(
            result.records,
            [{"instruction": "Question", "input": "", "output": "Answer"}],
        )

    def test_chatml_detects_strings_and_text_fields(self):
        conversation = (
            "<|im_start|>user\nQuestion<|im_end|>\n"
            "<|im_start|>assistant\nAnswer<|im_end|>"
        )
        for raw in ([conversation], [{"text": conversation}]):
            with self.subTest(raw=raw):
                result = self.convert(raw)
                self.assertEqual(result.dataset_format, "chatml")
                self.assertEqual(
                    result.records,
                    [{"instruction": "Question", "input": "", "output": "Answer"}],
                )

    def test_dpo_interface_converts_but_sft_pipeline_rejects_it(self):
        raw = [{"prompt": "Question", "chosen": "Good", "rejected": "Bad"}]
        result = self.convert(raw, require_task=None)
        self.assertEqual(result.dataset_format, "dpo")
        self.assertEqual(result.task_type, "preference")
        self.assertEqual(result.records, raw)
        with self.assertRaisesRegex(UnsupportedDatasetTaskError, "SFT training only"):
            self.convert(raw)

    def test_manual_override_is_forced_and_validated(self):
        raw = [{"question": "Question", "answer": "Answer"}]
        with self.assertRaisesRegex(DatasetValidationError, "missing required field"):
            self.convert(raw, dataset_format="sharegpt")

    def test_ambiguous_schema_reports_scored_candidates(self):
        raw = [{
            "instruction": "Instruction",
            "output": "Output",
            "prompt": "Prompt",
            "completion": "Completion",
        }]
        with self.assertRaises(DatasetFormatDetectionError) as raised:
            self.convert(raw)
        message = str(raised.exception)
        self.assertIn("Dataset format is ambiguous", message)
        self.assertIn("alpaca: confidence", message)
        self.assertIn("prompt_completion: confidence", message)

    def test_unknown_schema_reports_fields_and_candidates(self):
        with self.assertRaises(DatasetFormatDetectionError) as raised:
            self.convert([{"question": "Question", "unknown": "value"}])
        message = str(raised.exception)
        self.assertIn("Dataset format not recognized", message)
        self.assertIn("'question'", message)
        self.assertIn("qa_json: confidence", message)

    def test_detection_does_not_combine_fields_from_different_rows(self):
        rows = [{"instruction": "Question"}, {"output": "Answer"}]
        with self.assertRaises(DatasetFormatDetectionError) as raised:
            self.convert(rows)
        self.assertIn("complete in 0/2 sampled rows", str(raised.exception))

    def test_each_schema_adapter_rejects_invalid_input(self):
        invalid_by_format = {
            "alpaca": [{"instruction": "Question", "output": ""}],
            "instruction_response": [{"instruction": "Question", "response": ""}],
            "prompt_completion": [{"prompt": "Question", "completion": ""}],
            "qa_json": [{"question": "Question", "answer": ""}],
            "rag_qa": [{"context": "", "question": "Question", "answer": "Answer"}],
            "openai_chat": [{"messages": [{"role": "assistant", "content": "Answer"}]}],
            "sharegpt": [{"conversations": [{"from": "gpt", "value": "Answer"}]}],
            "chatml": ["<|im_start|>user\nQuestion"],
            "dpo": [{"prompt": "Question", "chosen": "", "rejected": "Bad"}],
        }
        for dataset_format, rows in invalid_by_format.items():
            with self.subTest(dataset_format=dataset_format):
                with self.assertRaises(DatasetValidationError):
                    self.convert(
                        rows,
                        dataset_format=dataset_format,
                        require_task=None,
                    )

    def test_registry_exposes_detection_scores(self):
        candidates = DEFAULT_FRAMEWORK.schema_registry.score(
            [{"question": "Question", "answer": "Answer"}],
            AdapterContext(),
        )
        scores = {candidate.adapter_name: candidate.result.confidence for candidate in candidates}
        self.assertEqual(scores["qa_json"], 0.98)
        self.assertGreater(scores["qa_json"], scores["rag_qa"])

    def test_column_mapping_enables_nonstandard_columns(self):
        result = DEFAULT_FRAMEWORK.convert_rows(
            [{"prompt_text": "Question", "answer_text": "Answer"}],
            options={
                "columns": {
                    "instruction": "prompt_text",
                    "output": "answer_text",
                }
            },
        )
        self.assertEqual(result.dataset_format, "alpaca")
        self.assertEqual(
            result.records,
            [{"instruction": "Question", "input": "", "output": "Answer"}],
        )


if __name__ == "__main__":
    unittest.main()

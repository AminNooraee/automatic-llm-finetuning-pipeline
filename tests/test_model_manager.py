import unittest

from fine_tuning_pipeline.model_manager import ModelCompatibilityError, resolve_model_compatibility


class ModelManagerTests(unittest.TestCase):
    def resolve(self, model_name, model_config, requested_template=None):
        return resolve_model_compatibility(
            model_name,
            requested_template=requested_template,
            config_loader=lambda _: model_config,
        )

    def test_qwen_resolves_qwen_template(self):
        result = self.resolve(
            "Qwen/Qwen2.5-0.5B-Instruct",
            {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
        )
        self.assertEqual(result.family, "qwen")
        self.assertEqual(result.variant, "qwen2.5")
        self.assertEqual(result.template, "qwen")

    def test_llama3_resolves_llama3_template(self):
        result = self.resolve(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            {"model_type": "llama", "vocab_size": 128256},
        )
        self.assertEqual(result.family, "llama")
        self.assertEqual(result.variant, "llama3")
        self.assertEqual(result.template, "llama3")

    def test_llama2_resolves_llama2_template(self):
        result = self.resolve(
            "meta-llama/Llama-2-7b-chat-hf",
            {"model_type": "llama", "vocab_size": 32000},
        )
        self.assertEqual(result.family, "llama")
        self.assertEqual(result.variant, "llama2")
        self.assertEqual(result.template, "llama2")

    def test_mistral_resolves_mistral_template(self):
        result = self.resolve(
            "mistralai/Mistral-7B-Instruct-v0.3",
            {"model_type": "mistral", "architectures": ["MistralForCausalLM"]},
        )
        self.assertEqual(result.family, "mistral")
        self.assertEqual(result.template, "mistral")

    def test_gemma_resolves_gemma_template(self):
        result = self.resolve(
            "google/gemma-2b-it",
            {"model_type": "gemma", "architectures": ["GemmaForCausalLM"]},
        )
        self.assertEqual(result.family, "gemma")
        self.assertEqual(result.template, "gemma")

    def test_model_family_can_be_detected_from_config(self):
        result = self.resolve(
            "organization/custom-instruct-model",
            {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
        )
        self.assertEqual(result.family, "qwen")
        self.assertEqual(result.template, "qwen")
        self.assertEqual(result.detection_source, "config")

    def test_name_fallback_reports_config_warning(self):
        def unavailable(_):
            raise OSError("offline")

        result = resolve_model_compatibility(
            "Qwen/Qwen2.5-0.5B-Instruct",
            config_loader=unavailable,
        )
        self.assertEqual(result.template, "qwen")
        self.assertEqual(result.detection_source, "name")
        self.assertTrue(any("name only" in warning for warning in result.warnings))

    def test_model_name_and_config_conflict_is_rejected(self):
        with self.assertRaisesRegex(ModelCompatibilityError, "looks like qwen"):
            self.resolve(
                "Qwen/Qwen2.5-0.5B-Instruct",
                {"model_type": "mistral"},
            )

    def test_incompatible_template_override_is_rejected(self):
        with self.assertRaisesRegex(ModelCompatibilityError, "resolves to the 'qwen'"):
            self.resolve(
                "Qwen/Qwen2.5-0.5B-Instruct",
                {"model_type": "qwen2"},
                requested_template="mistral",
            )

    def test_unsupported_model_is_rejected(self):
        with self.assertRaisesRegex(ModelCompatibilityError, "Unsupported model"):
            self.resolve(
                "google/bert-base-uncased",
                {"model_type": "bert", "architectures": ["BertModel"]},
            )

    def test_base_checkpoint_emits_warning(self):
        result = self.resolve(
            "mistralai/Mistral-7B-v0.1",
            {"model_type": "mistral"},
        )
        self.assertTrue(any("base checkpoint" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()

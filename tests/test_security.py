import json
import tempfile
import unittest
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from fine_tuning_pipeline.run_manager import RunManager
from fine_tuning_pipeline.http_client import _NoRedirectHandler
from fine_tuning_pipeline.security import SecretValue, redact_data, redact_text


TESTS_DIR = Path(__file__).resolve().parent
FAKE_SECRET = "security-test-secret-value"


class SecurityBoundaryTests(unittest.TestCase):
    def test_authenticated_cross_origin_redirect_is_not_followed(self):
        original = urllib.request.Request(
            "https://gateway.example/model/info",
            headers={"Authorization": f"Bearer {FAKE_SECRET}"},
        )
        redirected = _NoRedirectHandler().redirect_request(
            original,
            None,
            302,
            "Found",
            {"Location": "https://attacker.example/capture"},
            "https://attacker.example/capture",
        )
        self.assertIsNone(redirected)

    def test_secret_value_repr_str_and_recursive_data_are_safe(self):
        secret = SecretValue("${TEST_KEY}", FAKE_SECRET)
        self.assertNotIn(FAKE_SECRET, repr(secret))
        self.assertNotIn(FAKE_SECRET, str(secret))

        @dataclass
        class Snapshot:
            credential: SecretValue

        self.assertNotIn(FAKE_SECRET, repr(asdict(Snapshot(secret))))
        redacted = redact_data(
            {
                "api_key": FAKE_SECRET,
                "nested": {"password": FAKE_SECRET, "token": "${TOKEN}"},
                "object": secret,
            }
        )
        serialized = json.dumps(redacted)
        self.assertNotIn(FAKE_SECRET, serialized)
        self.assertIn("${TOKEN}", serialized)
        self.assertIn("${TEST_KEY}", serialized)

    def test_benign_token_fields_are_preserved_and_secret_names_are_redacted(self):
        value = {
            "tokenizer": "Org/Tokenizer",
            "tokenizer_name": "Org/Tokenizer",
            "max_tokens": 128,
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "access_token": FAKE_SECRET,
            "bearer_token": FAKE_SECRET,
            "signing_secret_key": FAKE_SECRET,
            "service_api_key": FAKE_SECRET,
        }
        redacted = redact_data(value)
        self.assertEqual(redacted["tokenizer"], "Org/Tokenizer")
        self.assertEqual(redacted["tokenizer_name"], "Org/Tokenizer")
        self.assertEqual(redacted["max_tokens"], 128)
        self.assertEqual(redacted["eos_token"], "</s>")
        self.assertEqual(redacted["pad_token"], "<pad>")
        self.assertEqual(redacted["access_token"], "<redacted>")
        self.assertEqual(redacted["bearer_token"], "<redacted>")
        self.assertEqual(redacted["signing_secret_key"], "<redacted>")
        self.assertEqual(redacted["service_api_key"], "<redacted>")

    def test_text_redaction_removes_known_and_header_shaped_values(self):
        text = redact_text(
            f"upstream echoed {FAKE_SECRET}; Authorization: Bearer another-secret; api_key=third",
            [FAKE_SECRET],
        )
        for value in (FAKE_SECRET, "another-secret", "third"):
            self.assertNotIn(value, text)

    def test_run_snapshots_metadata_logs_and_exceptions_never_persist_secret(self):
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            input_config = root / "config.yaml"
            input_config.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "Org/Model"},
                        "gateway": {
                            "api_key": FAKE_SECRET,
                            "safe_reference": "${LITELLM_API_KEY}",
                        },
                    }
                ),
                encoding="utf-8",
            )
            run = RunManager.create(
                root / "runs",
                model_name="Org/Model",
                dataset_name="dataset",
                input_config_path=input_config,
            )
            run.register_secret(FAKE_SECRET)
            run.write_resolved_config(
                {"gateway": {"api_key": FAKE_SECRET, "reference": "${LITELLM_API_KEY}"}}
            )
            with run.logging() as logger:
                logger.error("authorization failed for %s", FAKE_SECRET)
            run.record_phase_status(
                "gateway", "failed", error=RuntimeError(f"upstream echoed {FAKE_SECRET}")
            )
            run.mark_failed(RuntimeError(f"fatal api_key={FAKE_SECRET}"))
            combined = "".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in run.paths.root.rglob("*") if path.is_file()
            )
            self.assertNotIn(FAKE_SECRET, combined)
            self.assertIn("${LITELLM_API_KEY}", combined)


if __name__ == "__main__":
    unittest.main()

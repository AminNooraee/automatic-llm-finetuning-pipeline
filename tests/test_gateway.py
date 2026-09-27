import json
import tempfile
import unittest
from pathlib import Path

from fine_tuning_pipeline.gateway.config import GatewayConfigError, resolve_gateway_config
from fine_tuning_pipeline.gateway.contracts import (
    GatewayCapabilityError,
    GatewayConflictError,
    GatewayError,
    GatewayRegistrationOutcomeError,
    RegistrationRecord,
)
from fine_tuning_pipeline.gateway.manager import GatewayManager
from fine_tuning_pipeline.gateway.providers.litellm import (
    BACKEND_API_KEY_PLACEHOLDER,
    LiteLLMProvider,
)
from fine_tuning_pipeline.http_client import JsonResponse
from fine_tuning_pipeline.serving.config import resolve_serving_config
from fine_tuning_pipeline.serving.contracts import EndpointHealth


TESTS_DIR = Path(__file__).resolve().parent
FAKE_SECRET = "fake-master-key-never-persist"


def serving_config(advertise_host="model-server.example"):
    return resolve_serving_config(
        {
            "enabled": True,
            "advertise_host": advertise_host,
            "port": 8101,
            "base_model_name": "served-base",
            "fine_tuned_model_name": "served-fine",
        },
        run_id="run-1",
        base_model="Org/Base",
        training_method="lora",
    )


def gateway_config(raw=None, environment=None):
    value = {
        "enabled": True,
        "provider": "litellm",
        "base_url": "${LITELLM_BASE_URL}",
        "api_key": "${LITELLM_API_KEY}",
        "registration": {
            "mode": "dynamic_db",
            "base_model_name": "gateway-base",
            "fine_tuned_model_name": "gateway-fine",
        },
        "timeout_seconds": 10,
        "health_check": {"enabled": True, "verify_models": True, "verify_inference": True},
    }
    if raw:
        value.update(raw)
    return resolve_gateway_config(
        value,
        serving=serving_config(),
        run_id="run-1",
        environment=environment or {
            "LITELLM_BASE_URL": "https://gateway.example",
            "LITELLM_API_KEY": FAKE_SECRET,
        },
    )


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class GatewayConfigTests(unittest.TestCase):
    def test_missing_and_disabled_sections_make_no_secret_lookup(self):
        serving = resolve_serving_config(
            None, run_id="run", base_model="Org/Model", training_method="lora"
        )
        self.assertFalse(
            resolve_gateway_config(None, serving=serving, run_id="run", environment={}).enabled
        )
        self.assertFalse(
            resolve_gateway_config(
                {"enabled": False}, serving=serving, run_id="run", environment={}
            ).enabled
        )

    def test_enabled_gateway_requires_serving(self):
        disabled = resolve_serving_config(
            None, run_id="run", base_model="Org/Model", training_method="lora"
        )
        with self.assertRaisesRegex(GatewayConfigError, "requires serving.enabled"):
            resolve_gateway_config(
                {"enabled": True}, serving=disabled, run_id="run", environment={}
            )

    def test_environment_credential_and_url_are_resolved_only_in_memory(self):
        config = gateway_config()
        self.assertEqual(config.base_url, "https://gateway.example/v1")
        self.assertEqual(config.api_key.reveal(), FAKE_SECRET)
        self.assertNotIn(FAKE_SECRET, repr(config.api_key))
        self.assertEqual(config.api_key.reference, "${LITELLM_API_KEY}")

    def test_missing_or_literal_credential_is_rejected_without_echoing_value(self):
        with self.assertRaisesRegex(GatewayConfigError, "LITELLM_API_KEY.*not set"):
            gateway_config(environment={"LITELLM_BASE_URL": "https://gateway.example"})
        literal = "literal-do-not-echo"
        with self.assertRaises(GatewayConfigError) as caught:
            gateway_config({"api_key": literal})
        self.assertNotIn(literal, str(caught.exception))

    def test_provider_url_and_health_validation_are_strict(self):
        cases = (
            ({"provider": "other"}, "gateway.provider"),
            ({"base_url": "ftp://gateway.example"}, "absolute http"),
            ({"base_url": "http://gateway.example"}, "must use HTTPS"),
            ({"base_url": "https://user:pass@gateway.example"}, "must not contain credentials"),
            ({"health_check": {"enabled": True, "verify_models": False, "verify_inference": True}}, "requires health_check"),
            ({"timeout_seconds": 0}, "positive finite"),
            ({"unexpected": True}, "Unsupported gateway"),
        )
        for raw, message in cases:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(GatewayConfigError, message):
                    gateway_config(raw)

    def test_http_is_limited_to_explicit_loopback_development(self):
        config = gateway_config(
            {"base_url": "http://127.0.0.1:4000", "allow_local_backend": True}
        )
        self.assertEqual(config.base_url, "http://127.0.0.1:4000/v1")

    def test_loopback_serving_backend_requires_explicit_opt_in(self):
        raw = {
            "enabled": True,
            "base_url": "https://gateway.example",
            "api_key": "${LITELLM_API_KEY}",
        }
        with self.assertRaisesRegex(GatewayConfigError, "gateway-reachable"):
            resolve_gateway_config(
                raw,
                serving=serving_config("localhost"),
                run_id="run-1",
                environment={"LITELLM_API_KEY": FAKE_SECRET},
            )
        raw["allow_local_backend"] = True
        self.assertTrue(
            resolve_gateway_config(
                raw,
                serving=serving_config("127.0.0.1"),
                run_id="run-1",
                environment={"LITELLM_API_KEY": FAKE_SECRET},
            ).allow_local_backend
        )


class LiteLLMProviderTests(unittest.TestCase):
    def test_preflight_alias_query_and_both_registration_mappings(self):
        transport = FakeTransport([
            JsonResponse(200, {"data": []}),
            JsonResponse(200, {"data": []}),
            JsonResponse(201, {"model_id": "id-base"}),
            JsonResponse(201, {"model_id": "id-fine"}),
        ])
        provider = LiteLLMProvider(gateway_config(), transport=transport)
        provider.preflight()
        self.assertEqual(provider.aliases(), set())
        base = provider.register(
            role="base", alias="gateway-base", served_model="served-base",
            api_base="http://model-server.example:8101/v1", run_id="run-1",
        )
        fine = provider.register(
            role="fine_tuned", alias="gateway-fine", served_model="served-fine",
            api_base="http://model-server.example:8101/v1", run_id="run-1",
        )
        self.assertEqual(base.backend_model, "openai/served-base")
        self.assertEqual(fine.backend_model, "openai/served-fine")
        payloads = [call[2]["payload"] for call in transport.calls[2:]]
        self.assertEqual(payloads[0]["litellm_params"]["api_base"], "http://model-server.example:8101/v1")
        self.assertEqual(payloads[1]["litellm_params"]["model"], "openai/served-fine")
        self.assertEqual(
            payloads[0]["litellm_params"],
            {
                "model": "openai/served-base",
                "api_base": "http://model-server.example:8101/v1",
                "api_key": BACKEND_API_KEY_PLACEHOLDER,
            },
        )
        self.assertEqual(payloads[0]["model_info"]["pipeline_run_id"], "run-1")
        self.assertNotIn(FAKE_SECRET, json.dumps(payloads))

    def test_capability_auth_connection_and_registration_failures_are_actionable(self):
        with self.assertRaises(GatewayCapabilityError):
            LiteLLMProvider(
                gateway_config(), transport=FakeTransport([JsonResponse(404, {})])
            ).preflight()
        with self.assertRaisesRegex(GatewayError, "authentication failed"):
            LiteLLMProvider(
                gateway_config(), transport=FakeTransport([JsonResponse(401, {})])
            ).preflight()
        with self.assertRaises(GatewayError) as caught:
            LiteLLMProvider(
                gateway_config(), transport=FakeTransport([RuntimeError(FAKE_SECRET)])
            ).preflight()
        self.assertNotIn(FAKE_SECRET, str(caught.exception))
        with self.assertRaisesRegex(GatewayRegistrationOutcomeError, "was absent") as caught:
            LiteLLMProvider(
                gateway_config(),
                transport=FakeTransport(
                    [JsonResponse(500, {}), JsonResponse(200, {"data": []})]
                ),
            ).register(
                role="base", alias="gateway-base", served_model="served-base",
                api_base="http://server/v1", run_id="run-1",
            )
        self.assertEqual(caught.exception.outcome, "absent")

    def test_ambiguous_registration_is_reconciled_without_retry(self):
        owned = {
            "model_name": "gateway-base",
            "litellm_params": {"model": "openai/served-base"},
            "model_info": {
                "id": "persisted-id",
                "pipeline_run_id": "run-1",
                "pipeline_role": "base",
            },
        }
        transport = FakeTransport(
            [RuntimeError("connection reset"), JsonResponse(200, {"data": [owned]})]
        )
        record = LiteLLMProvider(gateway_config(), transport=transport).register(
            role="base",
            alias="gateway-base",
            served_model="served-base",
            api_base="http://server/v1",
            run_id="run-1",
        )
        self.assertEqual(record.status, "created_after_ambiguous_response")
        self.assertEqual(record.registration_id, "persisted-id")
        self.assertEqual([call[0] for call in transport.calls], ["POST", "GET"])

    def test_unavailable_reconciliation_is_indeterminate_and_not_retried(self):
        transport = FakeTransport([RuntimeError("timeout"), RuntimeError("still unavailable")])
        with self.assertRaises(GatewayRegistrationOutcomeError) as caught:
            LiteLLMProvider(gateway_config(), transport=transport).register(
                role="base",
                alias="gateway-base",
                served_model="served-base",
                api_base="http://server/v1",
                run_id="run-1",
            )
        self.assertEqual(caught.exception.outcome, "indeterminate")
        self.assertEqual([call[0] for call in transport.calls], ["POST", "GET"])

    def test_reconciliation_detects_concurrent_authoritative_alias_conflict(self):
        conflicting = {
            "model_name": "gateway-base",
            "model_info": {"pipeline_run_id": "another-run", "pipeline_role": "base"},
        }
        transport = FakeTransport(
            [JsonResponse(409, {}), JsonResponse(200, {"data": [conflicting]})]
        )
        with self.assertRaises(GatewayRegistrationOutcomeError) as caught:
            LiteLLMProvider(gateway_config(), transport=transport).register(
                role="base", alias="gateway-base", served_model="served-base",
                api_base="http://server/v1", run_id="run-1",
            )
        self.assertEqual(caught.exception.outcome, "conflict")
        self.assertEqual([call[0] for call in transport.calls], ["POST", "GET"])


class GatewayManagerTests(unittest.TestCase):
    class Verifier:
        def __init__(self): self.calls = []
        def verify(self, **kwargs):
            self.calls.append(kwargs)
            return EndpointHealth("ready", ("gateway-base", "gateway-fine"), "passed", "passed", 1)

    class Provider:
        def __init__(self, existing=(), fail_role=None, client_existing=()):
            self.existing = set(existing)
            self.client_existing = set(client_existing)
            self.fail_role = fail_role
            self.created = []
        def preflight(self): pass
        def aliases(self): return self.client_existing
        def management_aliases(self):
            return self.existing | {record.alias for record in self.created}
        def register(self, *, role, alias, served_model, api_base, run_id):
            if role == self.fail_role:
                raise GatewayRegistrationOutcomeError(
                    "registration rejected", alias=alias, outcome="absent"
                )
            record = RegistrationRecord(role, alias, f"openai/{served_model}", api_base, f"id-{role}")
            self.created.append(record)
            return record

    def test_existing_alias_fails_before_any_registration(self):
        provider = self.Provider(existing={"gateway-fine"})
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            with self.assertRaisesRegex(GatewayConflictError, "already exists"):
                GatewayManager(gateway_config(), provider=provider, verifier=self.Verifier()).register_and_verify(
                    run_id="run-1", run_root=Path(temporary), serving=serving_config()
                )
        self.assertEqual(provider.created, [])

    def test_management_conflict_is_detected_even_when_hidden_from_v1_models(self):
        provider = self.Provider(existing={"gateway-base"}, client_existing=())
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            with self.assertRaisesRegex(GatewayConflictError, "gateway-base"):
                GatewayManager(
                    gateway_config(), provider=provider, verifier=self.Verifier()
                ).register_and_verify(
                    run_id="run-1", run_root=Path(temporary), serving=serving_config()
                )
        self.assertEqual(provider.created, [])

    def test_partial_registration_is_recorded_and_never_deleted(self):
        provider = self.Provider(fail_role="fine_tuned")
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(GatewayError, "registration rejected"):
                GatewayManager(gateway_config(), provider=provider, verifier=self.Verifier()).register_and_verify(
                    run_id="run-1", run_root=root, serving=serving_config()
                )
            artifact = json.loads((root / "gateway" / "registration.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["status"], "partial_failed")
            self.assertEqual(len(artifact["registrations"]), 1)
            self.assertFalse(artifact["automatic_rollback_performed"])
            self.assertFalse(hasattr(provider, "delete"))

    def test_failed_first_registration_is_not_reported_as_partial(self):
        provider = self.Provider(fail_role="base")
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(GatewayError, "registration rejected"):
                GatewayManager(
                    gateway_config(), provider=provider, verifier=self.Verifier()
                ).register_and_verify(
                    run_id="run-1", run_root=root, serving=serving_config()
                )
            artifact = json.loads(
                (root / "gateway" / "registration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(artifact["status"], "registration_failed")
            self.assertEqual(artifact["registrations"], [])
            self.assertEqual(artifact["attempts"][0]["outcome"], "absent")

    def test_success_verifies_both_aliases_and_writes_secret_free_handoff(self):
        provider = self.Provider()
        verifier = self.Verifier()
        with tempfile.TemporaryDirectory(dir=TESTS_DIR) as temporary:
            root = Path(temporary)
            result = GatewayManager(
                gateway_config(), provider=provider, verifier=verifier
            ).register_and_verify(run_id="run-1", run_root=root, serving=serving_config())
            manifest = json.loads((root / result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["base_url"], "https://gateway.example/v1")
            self.assertEqual(verifier.calls[0]["base_model"], "gateway-base")
            self.assertEqual(verifier.calls[0]["fine_tuned_model"], "gateway-fine")
            self.assertEqual(provider.created[0].api_base, "http://model-server.example:8101/v1")
            combined = "".join(
                path.read_text(encoding="utf-8") for path in (root / "gateway").iterdir()
            )
            self.assertNotIn(FAKE_SECRET, combined)
            self.assertNotIn("Authorization", combined)


if __name__ == "__main__":
    unittest.main()

"""Ownership-aware LiteLLM registration and end-to-end verification."""

from __future__ import annotations

import logging
from pathlib import Path

from ..security import redact_text
from ..serving.config import ServingConfig
from ..serving.health import OpenAIEndpointVerifier
from ..serving.manifest import health_artifact, write_json
from .config import GatewayConfig
from .contracts import (
    GatewayConflictError,
    GatewayProvider,
    GatewayRegistrationOutcomeError,
    RegistrationAttempt,
    RegistrationRecord,
)
from .manifest import gateway_manifest, registration_artifact
from .providers.litellm import LiteLLMProvider


class GatewayManager:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        provider: GatewayProvider | None = None,
        verifier: OpenAIEndpointVerifier | None = None,
    ):
        self.config = config
        self.provider = provider or LiteLLMProvider(config)
        self.verifier = verifier or OpenAIEndpointVerifier()

    def register_and_verify(
        self,
        *,
        run_id: str,
        run_root: Path,
        serving: ServingConfig,
    ) -> dict:
        directory = run_root / "gateway"
        directory.mkdir(parents=True, exist_ok=True)
        logger = _file_logger(run_id, directory / "gateway.log")
        created: list[RegistrationRecord] = []
        attempts: list[RegistrationAttempt] = []
        try:
            registration = self.config.registration
            api_key = self.config.api_key
            if registration is None or api_key is None:
                raise RuntimeError("Internal gateway error: registration configuration is missing")
            logger.info("Gateway preflight provider=%s mode=%s", self.config.provider, registration.mode)
            self.provider.preflight()
            existing = self.provider.management_aliases() | self.provider.aliases()
            conflicts = [
                alias for alias in (
                    registration.base_model_name, registration.fine_tuned_model_name
                ) if alias in existing
            ]
            if conflicts:
                raise GatewayConflictError(
                    "Gateway model alias already exists: " + ", ".join(conflicts)
                    + ". Existing registration was not modified. Choose another alias."
                )

            pairs = (
                ("base", registration.base_model_name, serving.base_model_name),
                ("fine_tuned", registration.fine_tuned_model_name, serving.fine_tuned_model_name),
            )
            for role, alias, served_model in pairs:
                if alias in self.provider.management_aliases():
                    raise GatewayConflictError(
                        f"Gateway model alias already exists: {alias}. Existing registration "
                        "was not modified. Choose another alias."
                    )
                attempts.append(RegistrationAttempt(role, alias, "attempting"))
                write_json(
                    directory / "registration.json",
                    registration_artifact(
                        created, run_id=run_id, status="mutation_in_progress", attempts=attempts
                    ),
                )
                try:
                    record = self.provider.register(
                        role=role,
                        alias=alias,
                        served_model=served_model,
                        api_base=serving.base_url,
                        run_id=run_id,
                    )
                except GatewayRegistrationOutcomeError as error:
                    attempts[-1] = RegistrationAttempt(role, alias, error.outcome)
                    raise
                except Exception:
                    attempts[-1] = RegistrationAttempt(role, alias, "indeterminate")
                    raise
                created.append(record)
                attempts[-1] = RegistrationAttempt(
                    role, alias, record.status, record.registration_id
                )
                write_json(
                    directory / "registration.json",
                    registration_artifact(
                        created, run_id=run_id, status="partial", attempts=attempts
                    ),
                )
                logger.info("Registered gateway alias=%s role=%s", alias, role)

            health = self.verifier.verify(
                base_url=self.config.base_url,
                base_model=registration.base_model_name,
                fine_tuned_model=registration.fine_tuned_model_name,
                timeout_seconds=self.config.timeout_seconds,
                interval_seconds=min(2, self.config.timeout_seconds),
                wait_for_readiness=True,
                api_key=api_key.reveal(),
            )
            manifest = gateway_manifest(self.config)
            write_json(directory / "gateway_manifest.json", manifest)
            write_json(directory / "health_check.json", health_artifact(health))
            write_json(
                directory / "registration.json",
                registration_artifact(
                    created, run_id=run_id, status="ready", attempts=attempts
                ),
            )
            logger.info("Gateway verification passed for both model aliases")
            return {
                "manifest": manifest,
                "manifest_path": "gateway/gateway_manifest.json",
                "registration_path": "gateway/registration.json",
                "health_path": "gateway/health_check.json",
                "log_path": "gateway/gateway.log",
            }
        except Exception as error:
            if attempts:
                indeterminate = any(item.outcome == "indeterminate" for item in attempts)
                write_json(
                    directory / "registration.json",
                    registration_artifact(
                        created,
                        run_id=run_id,
                        status=(
                            "indeterminate_failed"
                            if indeterminate
                            else "registration_failed"
                            if not created
                            else "partial_failed"
                            if len(created) < 2
                            else "verification_failed"
                        ),
                        attempts=attempts,
                    ),
                )
            secret = self.config.api_key.reveal() if self.config.api_key else ""
            safe_error = redact_text(error, [secret])
            logger.error("Gateway setup failed: %s", safe_error)
            if safe_error != str(error):
                from .contracts import GatewayError

                raise GatewayError(safe_error) from None
            raise
        finally:
            for handler in tuple(logger.handlers):
                logger.removeHandler(handler)
                handler.close()


def _file_logger(run_id: str, path: Path) -> logging.Logger:
    logger = logging.getLogger(f"fine_tuning_pipeline.gateway.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger

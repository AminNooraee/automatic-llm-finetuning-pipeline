"""Serving-stage orchestration independent of the training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from .backends.vllm_docker import VllmDockerBackend
from .config import ServingConfig
from .contracts import ServingBackend, ServingLaunchRequest
from .health import OpenAIEndpointVerifier
from .manifest import endpoint_manifest, health_artifact, serving_operation, write_json


class ServingManager:
    def __init__(
        self,
        *,
        backend: ServingBackend | None = None,
        verifier: OpenAIEndpointVerifier | None = None,
    ):
        self.backend = backend or VllmDockerBackend()
        self.verifier = verifier or OpenAIEndpointVerifier()

    def start_and_verify(
        self,
        *,
        run_id: str,
        run_root: Path,
        config: ServingConfig,
        base_model: str,
        resolved_revision: str | None,
        adapter_path: Path,
        lora_rank: int,
    ) -> dict:
        directory = run_root / "serving"
        directory.mkdir(parents=True, exist_ok=True)
        logger = _file_logger(run_id, directory / "serving.log")
        request = ServingLaunchRequest(
            run_id=run_id,
            base_model=base_model,
            resolved_revision=resolved_revision,
            adapter_path=adapter_path,
            lora_rank=lora_rank,
            config=config,
        )
        launch = None
        try:
            logger.info("Serving preflight backend=%s runtime=%s", config.backend, config.runtime)
            self.backend.preflight(request)
            launch = self.backend.launch(request)
            write_json(
                directory / "operation.json",
                serving_operation(
                    run_id=run_id,
                    container_name=config.container_name,
                    container_id=launch.container_id,
                    status="starting",
                ),
            )
            logger.info(
                "vLLM container launched for base_alias=%s fine_tuned_alias=%s",
                config.base_model_name,
                config.fine_tuned_model_name,
            )
            health_config = config.health_check
            if health_config is None:
                raise RuntimeError("Internal serving error: health-check configuration is missing")
            health = self.verifier.verify(
                base_url=config.base_url,
                base_model=config.base_model_name,
                fine_tuned_model=config.fine_tuned_model_name,
                timeout_seconds=health_config.timeout_seconds,
                interval_seconds=health_config.interval_seconds,
                wait_for_readiness=health_config.enabled,
            )
            manifest = endpoint_manifest(config)
            write_json(directory / "endpoint_manifest.json", manifest)
            write_json(directory / "health_check.json", health_artifact(health))
            write_json(
                directory / "operation.json",
                serving_operation(
                    run_id=run_id,
                    container_name=config.container_name,
                    container_id=launch.container_id,
                    status="ready",
                ),
            )
            logger.info("Direct endpoint verification passed for both model aliases")
            return {
                "manifest": manifest,
                "container_id": launch.container_id,
                "manifest_path": "serving/endpoint_manifest.json",
                "health_path": "serving/health_check.json",
                "operation_path": "serving/operation.json",
                "log_path": "serving/serving.log",
            }
        except Exception as error:
            if launch is not None:
                write_json(
                    directory / "operation.json",
                    serving_operation(
                        run_id=run_id,
                        container_name=config.container_name,
                        container_id=launch.container_id,
                        status="failed",
                        error=str(error),
                    ),
                )
            logger.exception("Serving setup failed")
            raise
        finally:
            for handler in tuple(logger.handlers):
                logger.removeHandler(handler)
                handler.close()


def _file_logger(run_id: str, path: Path) -> logging.Logger:
    logger = logging.getLogger(f"fine_tuning_pipeline.serving.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger

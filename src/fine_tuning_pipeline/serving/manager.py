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
        execution_id: str | None = None,
        host_hf_cache_path: Path | None = None,
        cleanup_on_failure: bool = False,
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
            execution_id=execution_id,
            host_hf_cache_path=host_hf_cache_path,
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
                    execution_id=execution_id,
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
                    execution_id=execution_id,
                ),
            )
            logger.info("Direct endpoint verification passed for both model aliases")
            return {
                "manifest": manifest,
                "container_id": launch.container_id,
                "image_id": launch.image_id,
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
                        execution_id=execution_id,
                    ),
                )
                if cleanup_on_failure and hasattr(self.backend, "remove_if_owned"):
                    removed = self.backend.remove_if_owned(request, launch.container_id)
                    logger.info("Owned failed serving container cleanup performed=%s", removed)
            logger.exception("Serving setup failed")
            raise
        finally:
            for handler in tuple(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

    def cleanup_owned(
        self,
        *,
        run_id: str,
        execution_id: str,
        container_id: str,
        config: ServingConfig,
        base_model: str,
        adapter_path: Path,
        lora_rank: int,
        host_hf_cache_path: Path | None = None,
    ) -> bool:
        request = ServingLaunchRequest(
            run_id=run_id,
            base_model=base_model,
            resolved_revision=None,
            adapter_path=adapter_path,
            lora_rank=lora_rank,
            config=config,
            execution_id=execution_id,
            host_hf_cache_path=host_hf_cache_path,
        )
        if not hasattr(self.backend, "remove_if_owned"):
            return False
        return self.backend.remove_if_owned(request, container_id)


def _file_logger(run_id: str, path: Path) -> logging.Logger:
    logger = logging.getLogger(f"fine_tuning_pipeline.serving.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger

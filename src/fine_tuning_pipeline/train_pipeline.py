import json
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .artifact_validator import validate_model_artifacts
from .console_output import resolve_console_verbosity
from .dataset_adapter import adapt_dataset_source
from .dataset_config_generator import generate_dataset_info
from .dataset_loader import validate_unified_dataset
from .model_manager import resolve_model_compatibility
from .observability import (
    collect_container_provenance,
    collect_environment,
    collect_resource_baseline,
    extract_training_metrics,
    sha256_file,
)
from .run_manager import RunManager
from .trainer import run_training
from .training_config import TrainingConfig, resolve_training_config
from .yaml_generator import generate_training_yaml


PIPELINE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PIPELINE_DIR.parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class PreparedRun:
    config: dict
    training_config: TrainingConfig
    yaml_file: str
    run: RunManager
    console_verbosity: str


def load_config(config_path=None):
    config_file = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    for section in ("model", "dataset", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Training config needs a {section} section")
    if "output" not in config:
        config["output"] = {}
    elif not isinstance(config["output"], dict):
        raise ValueError("Training config output section must be a YAML mapping")
    return config


def require_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def resolve_config_path(raw_path, config_dir):
    path = Path(require_text(raw_path, "Path"))
    return (config_dir / path).resolve()


def _resolve_runs_root(config, config_file, runs_root=None, artifact_root=None):
    if runs_root is not None and artifact_root is not None:
        raise ValueError("Use runs_root or artifact_root, not both")
    explicit_root = runs_root if runs_root is not None else artifact_root
    if explicit_root is not None:
        return Path(explicit_root).resolve()
    configured_root = config["output"].get("runs_path", "../runs")
    return resolve_config_path(configured_root, config_file.parent)


def _resolved_dataset_source(source, config_dir):
    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    candidate = candidate.resolve()
    return str(candidate) if candidate.exists() else str(source)


def _prepare_run(config_path=None, runs_root=None, artifact_root=None):
    config_file = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    config_file = config_file.resolve()
    config = load_config(config_file)
    console_verbosity = resolve_console_verbosity(config)

    configured_model_name = require_text(config["model"].get("name"), "model.name")
    dataset_name = require_text(config["dataset"].get("name"), "dataset.name")
    if "," in dataset_name:
        raise ValueError("dataset.name must name a single dataset")
    dataset_source = require_text(config["dataset"].get("path"), "dataset.path")
    selected_runs_root = _resolve_runs_root(
        config,
        config_file,
        runs_root=runs_root,
        artifact_root=artifact_root,
    )
    run = RunManager.create(
        selected_runs_root,
        model_name=configured_model_name,
        dataset_name=dataset_name,
        input_config_path=config_file,
        console_verbosity=console_verbosity,
    )

    with run.logging() as logger:
        logger.info("Created run %s at %s", run.run_id, run.paths.root)
        try:
            try:
                environment = collect_environment()
            except Exception as error:
                logger.warning("Optional environment collection failed: %s", error)
                environment = {"collection_error": str(error)}
            try:
                container_config = config.get("provenance", {}).get("container", {})
                container = collect_container_provenance(container_config)
            except Exception as error:
                logger.warning("Optional container provenance collection failed: %s", error)
                container = {"appears_containerized": None, "collection_error": str(error)}
            run.record_environment(environment)
            run.record_observability(
                container=container,
                resources=collect_resource_baseline(environment),
            )

            logger.info("Resolving model compatibility for %s", configured_model_name)
            model_compatibility = resolve_model_compatibility(
                configured_model_name,
                requested_template=config["model"].get("template"),
                requested_revision=config["model"].get("revision"),
            )
            config["model"]["family"] = model_compatibility.family
            config["model"]["template"] = model_compatibility.template
            run.record_model(
                name=configured_model_name,
                family=model_compatibility.family,
                template=model_compatibility.template,
                requested_revision=model_compatibility.requested_revision,
                resolved_revision=model_compatibility.resolved_revision,
                revision_status=model_compatibility.revision_status,
                revision_unavailable_reason=model_compatibility.revision_unavailable_reason,
            )
            logger.info(
                "Resolved model family=%s variant=%s template=%s source=%s",
                model_compatibility.family,
                model_compatibility.variant,
                model_compatibility.template,
                model_compatibility.detection_source,
            )
            for warning in model_compatibility.warnings:
                logger.warning("Model compatibility: %s", warning)

            logger.info("Validating training configuration")
            training_config = resolve_training_config(config["training"])
            run.record_training(config["training"])
            run.record_artifact_relationship(
                method=training_config.method,
                base_model=configured_model_name,
                requested_base_revision=model_compatibility.requested_revision,
                base_revision=model_compatibility.resolved_revision,
                template=model_compatibility.template,
            )

            logger.info("Snapshotting dataset source %s", dataset_source)
            snapshot = run.snapshot_dataset(
                dataset_source,
                base_dir=config_file.parent,
                options=config["dataset"],
            )
            logger.info("Detecting, validating, and normalizing dataset")
            conversion = adapt_dataset_source(
                dataset_source,
                config["dataset"].get("format", "auto"),
                source_format=config["dataset"].get("source_format", "auto"),
                base_dir=config_file.parent,
                options=config["dataset"],
            )
            unified_rows = validate_unified_dataset(conversion.records)
            resolved_source = _resolved_dataset_source(dataset_source, config_file.parent)
            config["dataset"]["path"] = resolved_source
            config["dataset"]["source_format"] = conversion.source_format
            config["dataset"]["format"] = conversion.dataset_format
            config["dataset"]["snapshot"] = snapshot.relative_to(run.paths.root).as_posix()
            logger.info(
                "Prepared %s samples as %s from %s (confidence %.2f)",
                len(unified_rows),
                conversion.dataset_format,
                conversion.source_format,
                conversion.confidence,
            )

            normalized_path = run.paths.dataset / "normalized_dataset.json"
            with normalized_path.open("w", encoding="utf-8") as file:
                json.dump(unified_rows, file, ensure_ascii=False, indent=2)
            run.record_dataset(
                source=resolved_source,
                name=dataset_name,
                source_format=conversion.source_format,
                dataset_format=conversion.dataset_format,
                num_samples=len(unified_rows),
                snapshot=snapshot,
                normalized_path=normalized_path,
                sha256=sha256_file(normalized_path),
            )
            generate_dataset_info(
                dataset_name,
                normalized_path,
                run.paths.dataset,
                announce=console_verbosity != "quiet",
            )

            config["output"] = {
                "runs_path": str(selected_runs_root),
                "run_id": run.run_id,
                "run_dir": str(run.paths.root),
                "model_dir": str(run.paths.model),
            }
            run.write_resolved_config(config)
            yaml_file = generate_training_yaml(
                config,
                dataset_dir=run.paths.dataset,
                output_dir=run.paths.model,
                output_file=run.paths.training_yaml,
                training_config=training_config,
                announce=console_verbosity != "quiet",
            )
            run.set_status("prepared")
            logger.info("Training preparation completed: %s", yaml_file)
            return PreparedRun(
                config, training_config, yaml_file, run, console_verbosity
            )
        except Exception as error:
            run.mark_failed(error)
            logger.exception("Run preparation failed: %s", error)
            _print_run_summary(
                run, config, "failed", console_verbosity=console_verbosity
            )
            raise


def prepare_training(config_path=None, artifact_root=None, runs_root=None):
    """Prepare one isolated run while retaining the legacy tuple return value."""
    prepared = _prepare_run(
        config_path,
        runs_root=runs_root,
        artifact_root=artifact_root,
    )
    return prepared.config, prepared.yaml_file


def _print_training_header(prepared: PreparedRun) -> None:
    if prepared.console_verbosity == "full":
        return
    dataset = prepared.run.metadata.get("dataset", {})
    print("\n=== Training Started ===")
    print(f"Model: {prepared.config['model']['name']}")
    print(f"Method: {prepared.training_config.method.upper()}")
    print(f"Dataset: {dataset.get('name', prepared.config['dataset']['name'])}")
    if dataset.get("num_samples") is not None:
        print(f"Samples: {dataset['num_samples']}")
    print(f"Epochs: {prepared.training_config.epochs:g}")
    print(f"Batch size: {prepared.training_config.batch_size}\n")


def _print_run_summary(
    run,
    config,
    status,
    *,
    console_verbosity="concise",
    training_metrics=None,
):
    configured_method = config.get("training", {}).get("method")
    method = configured_method.strip().lower() if isinstance(configured_method, str) else ""
    if method == "lora":
        output_label = "LoRA adapter"
    elif method == "full":
        output_label = "full fine-tuned model"
    else:
        output_label = "training output"
    model_name = config["model"]["name"]
    dataset_path = config["dataset"]["path"]
    if console_verbosity != "full":
        heading = "Training Complete" if status == "success" else "Training Failed"
        print(f"\n=== {heading} ===")
        metrics = training_metrics or {}
        labels = (
            ("train_loss", "Train loss"),
            ("train_runtime", "Runtime"),
            ("train_samples_per_second", "Samples/sec"),
            ("train_steps_per_second", "Steps/sec"),
        )
        if console_verbosity == "concise":
            for key, label in labels:
                if key in metrics:
                    suffix = " s" if key == "train_runtime" else ""
                    print(f"{label}: {metrics[key]}{suffix}")
        print(f"Run: {run.paths.root}")
        print(f"Model: {model_name} ({output_label})")
        print(f"Dataset: {dataset_path}")
        print(f"Status: {status.upper()}")
        return

    print("\nTraining completed successfully.\n" if status == "success" else "\nTraining failed.\n")
    print(f"Run:\n{run.paths.root}\n")
    print(f"Model:\n{model_name} {output_label}\n")
    print(f"Dataset:\n{dataset_path}\n")
    print(f"Status:\n{status.upper()}")


def execute_training(config_path=None, runs_root=None):
    """Prepare, train, verify, and finalize one traceable run."""
    prepared = _prepare_run(config_path, runs_root=runs_root)
    run = prepared.run

    with run.logging() as logger:
        run.set_status("training")
        logger.info("Starting LLaMA-Factory training")
    _print_training_header(prepared)

    training_started = time.monotonic()
    try:
        training_result = run_training(
            prepared.yaml_file,
            log_file=run.paths.train_log,
            console_verbosity=prepared.console_verbosity,
        )
        duration = getattr(training_result, "wall_clock_seconds", None)
        run.record_training_duration(
            float(duration) if isinstance(duration, (int, float)) else time.monotonic() - training_started
        )
        with run.logging() as logger:
            run.set_status("verifying")
            logger.info("Training process exited successfully; verifying model artifacts")
            artifacts = validate_model_artifacts(
                run.paths.model,
                prepared.training_config.method,
            )
            run.record_verified_artifacts(artifacts)
            metrics = {}
            try:
                metrics = extract_training_metrics(run.paths.model)
                run.record_training_metrics(metrics)
            except Exception as error:
                logger.warning("Optional final training metric collection failed: %s", error)
            run.set_status("success")
            logger.info(
                "Run completed successfully with verified artifacts: %s",
                ", ".join(path.name for path in artifacts),
            )
    except Exception as error:
        try:
            run.record_training_duration(time.monotonic() - training_started)
        except Exception:
            pass
        run.mark_failed(error)
        with run.logging() as logger:
            logger.exception("Training run failed: %s", error)
        _print_run_summary(
            run,
            prepared.config,
            "failed",
            console_verbosity=prepared.console_verbosity,
        )
        raise

    _print_run_summary(
        run,
        prepared.config,
        "success",
        console_verbosity=prepared.console_verbosity,
        training_metrics=metrics,
    )
    return run.paths.root


def main():
    print("=== Fine-tuning Pipeline Started ===")
    execute_training()


if __name__ == "__main__":
    main()

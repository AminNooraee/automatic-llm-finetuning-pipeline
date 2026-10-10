from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath

import yaml


class JobConfigError(ValueError):
    pass


def load_yaml(path: Path, label: str) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise JobConfigError(
            f"Could not read {label}: {type(exc).__name__}"
        ) from None

    if not isinstance(data, dict):
        raise JobConfigError(f"{label} must be a YAML mapping")

    return data


def require_mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise JobConfigError(f"{label} must be a YAML mapping")
    return value


def require_text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JobConfigError(f"{label} must be a non-empty string")
    return value.strip()


def require_positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JobConfigError(f"{label} must be a positive integer")
    return value


def require_positive_number(value, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise JobConfigError(f"{label} must be a positive number")
    return float(value)


def safe_relative_path(value, label: str) -> PurePosixPath:
    text = require_text(value, label)

    if "\\" in text:
        raise JobConfigError(
            f"{label} must use '/' instead of Windows backslashes"
        )

    path = PurePosixPath(text)

    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise JobConfigError(
            f"{label} must be a safe relative path"
        )

    return path


def resolve_job_file(job_dir: Path, value, label: str) -> Path:
    relative = safe_relative_path(value, label)

    root = job_dir.resolve()
    target = root.joinpath(*relative.parts).resolve()

    try:
        target.relative_to(root)
    except ValueError:
        raise JobConfigError(
            f"{label} escapes the job directory"
        ) from None

    if not target.is_file():
        raise JobConfigError(
            f"{label} does not exist: {target}"
        )

    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def reject_unknown(mapping: dict, allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)

    if unknown:
        raise JobConfigError(
            f"Unsupported {label} field(s): {', '.join(unknown)}"
        )


def prepare_job(
    job_path: Path,
    catalog_path: Path,
    infrastructure_path: Path,
    output_dir: Path,
) -> None:

    job_path = job_path.resolve()
    catalog_path = catalog_path.resolve()
    infrastructure_path = infrastructure_path.resolve()

    job = load_yaml(job_path, "job.yaml")

    reject_unknown(
        job,
        {"schema_version", "job", "model", "datasets", "training"},
        "job",
    )

    if job.get("schema_version") != 1:
        raise JobConfigError("schema_version must be 1")

    # ---------------------------------------------------------
    # Job
    # ---------------------------------------------------------

    job_section = require_mapping(job.get("job"), "job")

    reject_unknown(
        job_section,
        {"name"},
        "job",
    )

    job_name = require_text(
        job_section.get("name"),
        "job.name",
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = require_mapping(job.get("model"), "model")

    reject_unknown(
        model,
        {"name", "revision"},
        "model",
    )

    model_name = require_text(
        model.get("name"),
        "model.name",
    )

    revision = model.get("revision")

    if revision is not None:
        revision = require_text(
            revision,
            "model.revision",
        )

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    datasets = require_mapping(
        job.get("datasets"),
        "datasets",
    )

    reject_unknown(
        datasets,
        {"train", "benchmark"},
        "datasets",
    )

    train = require_mapping(
        datasets.get("train"),
        "datasets.train",
    )

    benchmark = require_mapping(
        datasets.get("benchmark"),
        "datasets.benchmark",
    )

    reject_unknown(
        train,
        {"path", "source_format", "format"},
        "datasets.train",
    )

    reject_unknown(
        benchmark,
        {"path", "format"},
        "datasets.benchmark",
    )

    job_dir = job_path.parent

    train_source = resolve_job_file(
        job_dir,
        train.get("path"),
        "datasets.train.path",
    )

    benchmark_source = resolve_job_file(
        job_dir,
        benchmark.get("path"),
        "datasets.benchmark.path",
    )

    if train_source == benchmark_source:
        raise JobConfigError(
            "Training and benchmark datasets must be different files"
        )

    # ---------------------------------------------------------
    # Training parameters
    # ---------------------------------------------------------

    training = require_mapping(
        job.get("training"),
        "training",
    )

    reject_unknown(
        training,
        {
            "epochs",
            "learning_rate",
            "batch_size",
            "cutoff_len",
            "lora",
        },
        "training",
    )

    epochs = require_positive_number(
        training.get("epochs"),
        "training.epochs",
    )

    learning_rate = require_positive_number(
        training.get("learning_rate"),
        "training.learning_rate",
    )

    batch_size = require_positive_int(
        training.get("batch_size"),
        "training.batch_size",
    )

    cutoff_len = require_positive_int(
        training.get("cutoff_len"),
        "training.cutoff_len",
    )

    lora = require_mapping(
        training.get("lora"),
        "training.lora",
    )

    reject_unknown(
        lora,
        {"rank", "alpha", "dropout"},
        "training.lora",
    )

    rank = require_positive_int(
        lora.get("rank"),
        "training.lora.rank",
    )

    alpha = require_positive_int(
        lora.get("alpha"),
        "training.lora.alpha",
    )

    dropout = lora.get("dropout")

    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0 <= float(dropout) < 1
    ):
        raise JobConfigError(
            "training.lora.dropout must be >= 0 and < 1"
        )

    dropout = float(dropout)

    # ---------------------------------------------------------
    # Internal model catalog
    # ---------------------------------------------------------

    catalog = load_yaml(
        catalog_path,
        "model catalog",
    )

    if catalog.get("schema_version") != 1:
        raise JobConfigError(
            "model catalog schema_version must be 1"
        )

    catalog_models = require_mapping(
        catalog.get("models"),
        "model catalog.models",
    )

    if model_name not in catalog_models:
        raise JobConfigError(
            f"MODEL_NOT_APPROVED: {model_name}"
        )

    policy = require_mapping(
        catalog_models[model_name],
        f"catalog entry for {model_name}",
    )

    parameter_count = require_positive_int(
        policy.get("parameter_count"),
        "parameter_count",
    )

    precision = require_text(
        policy.get("precision"),
        "precision",
    ).lower()

    attention_backend = require_text(
        policy.get("attention_backend"),
        "attention_backend",
    ).lower()

    if precision not in {
        "fp32",
        "fp16",
        "bf16",
    }:
        raise JobConfigError(
            "Unsupported model precision"
        )

    if attention_backend not in {
        "auto",
        "eager",
        "sdpa",
        "fa2",
        "fa3",
    }:
        raise JobConfigError(
            "Unsupported attention backend"
        )

    # ---------------------------------------------------------
    # Trusted company infrastructure
    # ---------------------------------------------------------

    infrastructure = load_yaml(
        infrastructure_path,
        "CI infrastructure config",
    )

    if infrastructure.get("schema_version") != 1:
        raise JobConfigError(
            "CI infrastructure schema_version must be 1"
        )

    training_defaults = require_mapping(
        infrastructure.get("training_defaults"),
        "training_defaults",
    )

    if training_defaults.get("method") != "lora":
        raise JobConfigError(
            "CI infrastructure currently supports only LoRA"
        )

    resource_preflight = dict(
        require_mapping(
            infrastructure.get("resource_preflight"),
            "resource_preflight",
        )
    )

    resource_preflight[
        "model_parameter_estimate"
    ] = parameter_count

    # ---------------------------------------------------------
    # Output package
    # ---------------------------------------------------------

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    train_suffix = train_source.suffix.lower()
    benchmark_suffix = benchmark_source.suffix.lower()

    train_destination = (
        output_dir / f"train{train_suffix}"
    )

    benchmark_destination = (
        output_dir / f"benchmark{benchmark_suffix}"
    )

    shutil.copy2(
        train_source,
        train_destination,
    )

    shutil.copy2(
        benchmark_source,
        benchmark_destination,
    )

    shutil.copy2(
        job_path,
        output_dir / "job.yaml",
    )

    generated_model = {
        "name": model_name,
    }

    if revision:
        generated_model[
            "revision"
        ] = revision

    generated_training = dict(
        training_defaults
    )

    generated_training.update(
        {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "precision": precision,
            "attention_backend": attention_backend,
            "cutoff_len": cutoff_len,
            "lora": {
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
            },
        }
    )

    generated_config = {
        "model": generated_model,

        "dataset": {
            "path": train_destination.name,
            "name": f"{job_name}-train",
            "source_format": train.get(
                "source_format",
                "auto",
            ),
            "format": train.get(
                "format",
                "auto",
            ),
        },

        "training": generated_training,

        "resource_preflight": resource_preflight,

        "orchestration": infrastructure.get(
            "orchestration"
        ),

        "serving": infrastructure.get(
            "serving"
        ),

        "gateway": infrastructure.get(
            "gateway"
        ),

        "output": infrastructure.get(
            "output"
        ),

        "observability": infrastructure.get(
            "observability"
        ),
    }

    config_output = (
        output_dir / "finetune.yaml"
    )

    config_output.write_text(
        yaml.safe_dump(
            generated_config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "job_name": job_name,
        "model": {
            "name": model_name,
            "revision": revision,
        },
        "train_dataset": {
            "file": train_destination.name,
            "sha256": sha256_file(
                train_destination
            ),
        },
        "benchmark_dataset": {
            "file": benchmark_destination.name,
            "format": benchmark.get(
                "format",
                "auto",
            ),
            "sha256": sha256_file(
                benchmark_destination
            ),
        },
        "generated_config": "finetune.yaml",
    }

    (
        output_dir / "job_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=== USER JOB PREPARED ===")
    print(f"Job: {job_name}")
    print(f"Model: {model_name}")
    print(
        f"Train dataset: {train_destination.name}"
    )
    print(
        f"Benchmark dataset: {benchmark_destination.name}"
    )
    print(
        f"Generated config: {config_output}"
    )


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--job",
        type=Path,
        default=Path("job/job.yaml"),
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "configs/model_catalog.yaml"
        ),
    )

    parser.add_argument(
        "--infrastructure",
        type=Path,
        default=Path(
            "configs/ci_infrastructure.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "ci_artifacts/job"
        ),
    )

    args = parser.parse_args()

    try:
        prepare_job(
            args.job,
            args.catalog,
            args.infrastructure,
            args.output_dir,
        )

        return 0

    except JobConfigError as exc:

        print(
            f"Job preparation failed: {exc}",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(main())
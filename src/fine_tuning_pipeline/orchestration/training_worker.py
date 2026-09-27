"""Training-container entry point with an explicit, secret-free result contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..security import redact_data
from ..train_pipeline import execute_training


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(redact_data(value), indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_secret_file(variable: str, file_variable: str) -> None:
    path = os.environ.get(file_variable)
    if not path:
        return
    value = Path(path).read_text(encoding="utf-8").rstrip("\r\n")
    if value:
        os.environ[variable] = value


def run_worker(
    *, config_path: Path, runs_root: Path, result_path: Path, execution_id: str
) -> Path:
    _load_secret_file("HF_TOKEN", "HF_TOKEN_FILE")
    try:
        run_root = execute_training(
            config_path, runs_root=runs_root, enable_optional_phases=False
        )
        resolved_runs = runs_root.resolve()
        resolved_run = run_root.resolve()
        relative_run = resolved_run.relative_to(resolved_runs)
        metadata = json.loads((resolved_run / "metadata.json").read_text(encoding="utf-8"))
        artifact = metadata.get("artifact", {})
        model = metadata.get("model", {})
        training = metadata.get("training", {})
        result = {
            "schema_version": 1,
            "execution_id": execution_id,
            "run_id": metadata["run_id"],
            "status": "success",
            "run_directory": relative_run.as_posix(),
            "model_directory": (relative_run / "model").as_posix(),
            "metadata_path": (relative_run / "metadata.json").as_posix(),
            "base_model": model.get("name"),
            "resolved_revision": model.get("resolved_revision"),
            "requested_revision": model.get("requested_revision"),
            "training_method": training.get("method"),
            "adapter_type": artifact.get("type"),
            "lora_rank": (
                training.get("lora", {}).get("rank")
                if isinstance(training.get("lora"), dict)
                else training.get("lora_rank")
            ),
            "artifact_validation": "passed",
            "training_image": {
                "reference": os.environ.get("PIPELINE_TRAINING_IMAGE"),
                "id": os.environ.get("PIPELINE_TRAINING_IMAGE_ID"),
            },
        }
        _atomic_json(result_path, result)
        return run_root
    except BaseException as error:
        _atomic_json(result_path, {
            "schema_version": 1,
            "execution_id": execution_id,
            "status": "failed",
            "error": {"type": type(error).__name__},
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--execution-id", required=True)
    args = parser.parse_args(argv)
    run_worker(
        config_path=args.config,
        runs_root=args.runs_root,
        result_path=args.result,
        execution_id=args.execution_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

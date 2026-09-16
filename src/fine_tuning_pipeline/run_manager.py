"""Run directory lifecycle, metadata, snapshots, and logging."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized or fallback)[:48]


@dataclass(frozen=True)
class RunPaths:
    root: Path
    config: Path
    dataset: Path
    logs: Path
    model: Path
    input_config: Path
    resolved_config: Path
    training_yaml: Path
    train_log: Path
    metadata: Path


class RunManager:
    """Own all files and lifecycle state for one independent training run."""

    def __init__(self, run_id: str, paths: RunPaths, metadata: dict[str, Any]):
        self.run_id = run_id
        self.paths = paths
        self._metadata = metadata

    @classmethod
    def create(
        cls,
        runs_root: str | Path,
        *,
        model_name: str,
        dataset_name: str,
        input_config_path: str | Path,
        created_at: datetime | None = None,
    ) -> "RunManager":
        timestamp = created_at or datetime.now(timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)
        base_id = "_".join(
            (
                timestamp.strftime("%Y%m%d_%H%M%S"),
                _slug(model_name.rsplit("/", 1)[-1], "model"),
                _slug(dataset_name, "dataset"),
            )
        )

        root = Path(runs_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while True:
            run_id = base_id if attempt == 1 else f"{base_id}_{attempt:02d}"
            run_root = root / run_id
            try:
                run_root.mkdir(exist_ok=False)
                break
            except FileExistsError:
                attempt += 1

        config_dir = run_root / "config"
        dataset_dir = run_root / "dataset"
        logs_dir = run_root / "logs"
        model_dir = run_root / "model"
        for directory in (config_dir, dataset_dir, logs_dir, model_dir):
            directory.mkdir()

        paths = RunPaths(
            root=run_root,
            config=config_dir,
            dataset=dataset_dir,
            logs=logs_dir,
            model=model_dir,
            input_config=config_dir / "input_config.yaml",
            resolved_config=config_dir / "resolved_config.yaml",
            training_yaml=config_dir / "training.yaml",
            train_log=logs_dir / "train.log",
            metadata=run_root / "metadata.json",
        )
        metadata = {
            "run_id": run_id,
            "status": "created",
            "created_at": _iso_utc(timestamp),
            "model": {"name": model_name},
            "dataset": {"name": dataset_name},
            "training": {},
            "output": {"model_dir": str(model_dir)},
        }
        manager = cls(run_id, paths, metadata)
        manager._write_metadata()
        try:
            shutil.copy2(Path(input_config_path).resolve(), paths.input_config)
        except Exception as error:
            manager.mark_failed(error)
            raise
        return manager

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @contextmanager
    def logging(self) -> Iterator[logging.Logger]:
        """Provide a terminal and per-run file logger without leaked handlers."""
        logger = logging.getLogger(f"fine_tuning_pipeline.run.{self.run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        file_handler = logging.FileHandler(self.paths.train_log, encoding="utf-8")
        stream_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        try:
            yield logger
        finally:
            for handler in (file_handler, stream_handler):
                logger.removeHandler(handler)
                handler.close()

    def snapshot_dataset(
        self,
        source: str | Path,
        *,
        base_dir: str | Path,
        options: Mapping[str, Any],
    ) -> Path:
        """Copy a local source or record a descriptor for a remote source."""
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = Path(base_dir) / source_path
        source_path = source_path.resolve()

        if source_path.is_file():
            suffix = "".join(source_path.suffixes) or ".data"
            destination = self.paths.dataset / f"original_dataset{suffix}"
            shutil.copy2(source_path, destination)
            return destination

        destination = self.paths.dataset / "original_dataset.json"
        descriptor = {
            "source": str(source),
            "source_format": options.get("source_format", "auto"),
            "subset": options.get("subset"),
            "split": options.get("split", "train"),
        }
        with destination.open("w", encoding="utf-8") as file:
            json.dump(descriptor, file, indent=2, ensure_ascii=False)
        return destination

    def write_resolved_config(self, config: Mapping[str, Any]) -> None:
        with self.paths.resolved_config.open("w", encoding="utf-8") as file:
            yaml.safe_dump(dict(config), file, sort_keys=False)

    def record_model(self, *, name: str, family: str, template: str) -> None:
        self._metadata["model"] = {
            "name": name,
            "family": family,
            "template": template,
        }
        self._write_metadata()

    def record_dataset(
        self,
        *,
        source: str,
        name: str,
        source_format: str,
        dataset_format: str,
        num_samples: int,
        snapshot: Path,
    ) -> None:
        self._metadata["dataset"] = {
            "path": source,
            "name": name,
            "source_format": source_format,
            "format": dataset_format,
            "num_samples": num_samples,
            "snapshot": snapshot.relative_to(self.paths.root).as_posix(),
        }
        self._write_metadata()

    def record_training(self, training: Mapping[str, Any]) -> None:
        self._metadata["training"] = dict(training)
        self._write_metadata()

    def set_status(self, status: str) -> None:
        self._metadata["status"] = status
        if status in {"success", "failed"}:
            self._metadata["completed_at"] = _iso_utc(datetime.now(timezone.utc))
        self._write_metadata()

    def mark_failed(self, error: BaseException) -> None:
        self._metadata["status"] = "failed"
        self._metadata["completed_at"] = _iso_utc(datetime.now(timezone.utc))
        self._metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self._write_metadata()

    def record_verified_artifacts(self, artifacts: list[Path]) -> None:
        self._metadata["output"]["verified_artifacts"] = [
            artifact.relative_to(self.paths.root).as_posix() for artifact in artifacts
        ]
        self._write_metadata()

    def _write_metadata(self) -> None:
        temporary = self.paths.metadata.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(self._metadata, file, indent=2, ensure_ascii=False)
        temporary.replace(self.paths.metadata)

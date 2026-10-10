"""Return the validated training dataset path from a prepared job manifest."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


TRAIN_FILENAME = re.compile(r"train\.(?:json|jsonl|ndjson|csv|parquet|pq|txt|chatml)\Z")


def training_path(prepared_dir: Path) -> Path:
    root = prepared_dir.resolve(strict=True)
    manifest = json.loads((root / "job_manifest.json").read_text(encoding="utf-8"))
    filename = manifest["train_dataset"]["file"]
    if not isinstance(filename, str) or not TRAIN_FILENAME.fullmatch(filename):
        raise ValueError("Manifest has an invalid training filename")
    target = (root / filename).resolve(strict=True)
    if target.parent != root or not target.is_file():
        raise ValueError("Training dataset must be a file in the prepared job directory")
    return target


if __name__ == "__main__":
    try:
        print(training_path(Path(sys.argv[1])))
    except (IndexError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Invalid prepared training dataset: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

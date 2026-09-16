import json
import os
from pathlib import Path


def generate_dataset_info(
        name,
        file_path,
        registry_dir
):

    dataset_path = Path(file_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_path = Path(registry_dir).resolve() / "dataset_info.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info = {}

    relative_path = os.path.relpath(dataset_path, output_path.parent).replace(os.sep, "/")

    dataset_info[name] = {
        "file_name": relative_path,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            dataset_info,
            f,
            indent=2,
            ensure_ascii=False
        )


    print("dataset_info.json generated")
    return output_path

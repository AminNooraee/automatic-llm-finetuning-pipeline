import subprocess
import sys
from pathlib import Path


def run_training(yaml_file, log_file=None):

    print("Starting LLaMA-Factory training...")

    command = [
        sys.executable,
        "-m",
        "llamafactory.cli",
        "train",
        str(Path(yaml_file).resolve())
    ]

    if log_file is None:
        result = subprocess.run(command)
    else:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"Launching command: {' '.join(command)}\n")
            stream.flush()
            result = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )

    if result.returncode != 0:
        raise RuntimeError(f"Training failed (exit code {result.returncode})")
    return result

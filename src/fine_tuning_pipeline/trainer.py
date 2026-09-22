import codecs
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .console_output import ConsoleOutputFormatter


@dataclass(frozen=True)
class TrainingProcessResult:
    returncode: int
    wall_clock_seconds: float


def run_training(yaml_file, log_file=None, console_verbosity="concise"):

    print("Starting LLaMA-Factory training...")

    command = [
        sys.executable,
        "-m",
        "llamafactory.cli",
        "train",
        str(Path(yaml_file).resolve())
    ]

    started = time.monotonic()
    if log_file is None:
        result = subprocess.run(command)
    else:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" avoids Windows newline translation so backend carriage
        # returns/newlines remain faithful in the authoritative raw log.
        with log_path.open("a", encoding="utf-8", newline="") as stream:
            launch_line = f"Launching command: {' '.join(command)}\n"
            stream.write(launch_line)
            stream.flush()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            formatter = ConsoleOutputFormatter(console_verbosity)
            if process.stdout is not None:
                # Raw chunks preserve tqdm-style carriage-return progress; line
                # iteration can hide progress until a newline or process exit.
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    raw = process.stdout.read(4096)
                    if not raw:
                        break
                    chunk = raw if isinstance(raw, str) else decoder.decode(raw)
                    stream.write(chunk)
                    stream.flush()
                    formatter.feed(chunk)
                tail = decoder.decode(b"", final=True)
                if tail:
                    stream.write(tail)
                    stream.flush()
                    formatter.feed(tail)
            formatter.close()
            returncode = process.wait()
            result = TrainingProcessResult(
                returncode=returncode,
                wall_clock_seconds=time.monotonic() - started,
            )

    if result.returncode != 0:
        raise RuntimeError(f"Training failed (exit code {result.returncode})")
    if isinstance(result, TrainingProcessResult):
        return result
    return TrainingProcessResult(
        returncode=result.returncode,
        wall_clock_seconds=time.monotonic() - started,
    )

# Automatic LLM Fine-tuning Pipeline

Configuration-driven supervised fine-tuning with automatic dataset preparation,
model/template compatibility, and traceable training runs.

Project #1 is functionally validated for the supported SFT/LoRA workflow. This
repository uses an editable `src/` package, root tests, and documented example
configs. The project's original code is licensed under [Apache License 2.0](LICENSE).

Release version: `v1.0.0`. See [release notes](RELEASE_NOTES.md),
[changelog](CHANGELOG.md), [publishing checklist](RELEASE_CHECKLIST.md), and
[readiness audit](docs/release_readiness.md). The CUDA Docker workflow has since
completed a real LoRA run on an NVIDIA H100; this does not qualify every image,
driver, model, dataset, or full-fine-tuning combination.

## Overview

Provide a HuggingFace model identifier, a supported dataset source, and training
parameters in YAML. The pipeline normalizes data, generates dataset registration
and training configuration, launches training, and verifies its saved output.
Users do not need to edit Python or LLaMA-Factory dataset/configuration files for
supported inputs.

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is the training engine.
This project supplies the preparation, compatibility, configuration, and run
lifecycle around it; it does not implement a new trainer or replace the upstream
project. LoRA output is an adapter used together with its original base model.

## Architecture

```text
Dataset
  |
Dataset Adapter (source loading + schema detection)
  |
Normalization: instruction / input / output
  |
Model Compatibility Layer: family + template
  |
Training Configuration: validated YAML parameters
  |
LLaMA-Factory SFT
  |
LoRA Adapter (or full-method model output)
  |
Artifact Validation -> metadata success / failure
```

This is the conceptual data-to-model flow. At runtime model compatibility and
training configuration are validated before dataset preparation. Run management,
logging, and metadata span the whole execution. See [architecture](docs/architecture.md).

## Features

- Multi-model support with automatic family/template resolution.
- Multi-dataset support, automatic format conversion, confidence-scored detection,
  explicit overrides, and configuration-only column mapping.
- Validated, configuration-driven training and LoRA parameters.
- Unique run directories with input/resolved/training configuration snapshots.
- Persistent pipeline/training logs and metadata tracking.
- Versioned experiment metadata with exact normalized-dataset SHA-256 hashes,
  best-effort Hub revisions, environment/container provenance, resource context,
  and normalized final Trainer metrics.
- Live LLaMA-Factory stdout/stderr in the terminal while retaining the complete
  `logs/train.log` file.
- Artifact validation before a run is declared successful.
- Post-training adapter loading and inference validated during acceptance;
  inference is not an automatic extra step of every training execution.

## Supported models

| Family | Automatic LLaMA-Factory template | Validation |
| --- | --- | --- |
| Qwen / Qwen2 / Qwen2.5 | `qwen` | Qwen2.5-0.5B real training and inference; family unit coverage |
| Meta Llama 2 / 3 | `llama2` / `llama3` | Configuration and tokenizer smoke tests |
| Mistral | `mistral` | Configuration and tokenizer smoke tests |
| Gemma | `gemma`, `gemma2`, `gemma3` by generation | Gemma configuration/tokenizer smoke; later generations not separately trained |

Supported-family selection is not a promise that every checkpoint or architecture
on HuggingFace works. Gated models require approved access and authentication;
larger models require suitable hardware. See [model support](docs/supported_models.md).

## Supported datasets

Sources: JSON arrays, JSONL, CSV, Parquet, HuggingFace datasets, and raw ChatML text.

Schemas: Alpaca, instruction/response, prompt/completion, QA, RAG QA, ShareGPT,
ChatML, and OpenAI chat. DPO detection/interface exists; DPO training is not enabled.
Use `dataset.format: auto` by default. Unsupported or ambiguous structures fail
with diagnostic candidates instead of silently choosing a schema.
See [dataset support](docs/supported_datasets.md) and [examples](examples/README.md).

## Installation

Python 3.12.7 was validated; the inspected upstream backend requires Python 3.11+.
Create a separate virtual environment for a new installation:

```bash
python -m venv .venv
```

Activate it on your platform:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS shell
source .venv/bin/activate
```

Install matching PyTorch, torchvision, and torchaudio builds using the
[official platform selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The manifest pins the tested application stack and upstream source revision.
PyTorch selection is host-specific; transitive packages are not fully locked.
Fresh-environment and Docker/GPU reproduction have not been validated in this
relocation pass. The existing validated `venv/` received only the editable
application package; its training dependencies were not changed.
See [installation](docs/installation.md) for the recorded CPU versions, optional
adapter-only dependencies, Conda bootstrap, and installation caveats.

## Docker installation

Install [Docker Engine](https://docs.docker.com/engine/install/) on a Linux server
or [Docker Desktop](https://docs.docker.com/desktop/) with Linux containers on
Windows/macOS. Start Docker, clone the repository, and build from its root:

```bash
docker build -f docker/Dockerfile -t automatic-llm-finetuner .
```

The default image uses a digest-pinned Python base, a virtual environment, the
existing project/LLaMA-Factory requirements, and a matched CPU PyTorch stack.
It includes no host datasets, runs, weights, caches, or virtual environments.
The build runs dependency/import/config checks and the regression suite; no GPU
is required. A CUDA 12.8 variant is selectable without changing training code:

```bash
docker build -f docker/Dockerfile --build-arg RUNTIME=cuda \
  -t automatic-llm-finetuner:cuda .
```

Edit `configs/config.yaml`. The bundled synthetic dataset works unchanged. For
your own dataset, set `dataset.path: ../datasets/my_dataset.json`; keep
`output.runs_path: ../runs`. Mount input directories read-only and runs writable
(Linux/macOS shell):

```bash
mkdir -p datasets runs
docker run --rm \
  --mount "type=bind,source=$(pwd)/configs,target=/app/configs,readonly" \
  --mount "type=bind,source=$(pwd)/datasets,target=/app/datasets,readonly" \
  --mount "type=bind,source=$(pwd)/runs,target=/app/runs" \
  --mount type=volume,source=llm-hf-cache,target=/cache/huggingface \
  automatic-llm-finetuner
```

Outputs persist as `runs/<run_id>/` on the host. Hub inputs need no local dataset
mount; the optional named cache volume persists downloads. The container runs as
UID/GID 1000 by default, so `runs/` and cache mounts must be writable by that user.
For a GPU host, use the CUDA image and add `--gpus all`; an appropriate NVIDIA
driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
are required. Do not put Hub tokens in build arguments or image files.

See [Docker deployment](docs/docker.md) for PowerShell commands, UID/GID settings,
GPU checks, named configs, credentials, and deployment acceptance. Base images
and the core stack are pinned, not every OS/transitive package. Docker is not
installed on this Windows development host, but the CUDA Docker workflow has
subsequently completed a real LoRA training run on an NVIDIA H100.

## Usage

Edit [configs/config.yaml](configs/config.yaml), then
run from the repository root with your activated environment:

```bash
python -m fine_tuning_pipeline.train_pipeline
```

Install the package with `python -m pip install -r requirements.txt` first.
Module execution avoids depending on flat-script imports.

Example configuration:

```yaml
model:
  name: Qwen/Qwen2.5-0.5B-Instruct
  # Optional Hub branch, tag, or commit; resolved commit is recorded when available.
  # revision: main
dataset:
  path: ../examples/datasets/alpaca_demo.json
  name: demo_dataset
  source_format: auto
  format: auto
training:
  method: lora
  epochs: 1
  learning_rate: 0.0002
  batch_size: 1
  precision: fp32
  gradient_checkpointing: false
  cutoff_len: 128
  save_steps: 10
  logging_steps: 1
  lora:
    rank: 8
    alpha: 16
    dropout: 0.0
output:
  runs_path: ../runs
observability:
  console_verbosity: concise
```

Dataset/output paths are relative to the YAML file, not the shell's working
directory. The CLI reads the default config; it does not implement a `--config`
option. Named files in [configs/](configs/) use the existing Python API, as shown
in [usage](docs/usage.md). Do not copy a named config into a different folder
without adjusting its relative paths.

Each execution saves `runs/<run_id>/config/`, `dataset/`, `logs/`, `model/`, and
`metadata.json`, plus `environment.json`. Metadata schema v2 preserves the
original fields and adds revision resolution, the SHA-256 of the exact normalized
dataset consumed by training, package/GPU/container context, wall-clock duration,
final Trainer metrics, and an explicit base-model/adapter relationship. Success
still requires both process completion and expected nonempty artifacts.

Console output defaults to `concise`: lifecycle events, throttled progress,
emitted training metrics, warnings, errors, and final metrics. Use `quiet` for
lifecycle and errors only, or `full` for the complete unfiltered backend stream:

```yaml
observability:
  console_verbosity: quiet  # quiet | concise | full
```

This affects terminal presentation only. `logs/train.log` always remains the
authoritative complete raw LLaMA-Factory/Transformers stdout and stderr log,
including output hidden from `quiet` and `concise`. See
[console observability](docs/observability.md).

LoRA output is recorded as `lora_adapter`, not a standalone model. The
provider-neutral `serving` block identifies the base, adapter, template, and a
suggested unique ID for a future integration. This change does **not** implement
endpoint serving, vLLM launch, LiteLLM registration, adapter merging, or Project
#2 integration.

## Validated results

| Acceptance check | Result |
| --- | --- |
| End-to-end Qwen2.5-0.5B LoRA training, two local rows, one epoch | PASS |
| Real HuggingFace subset through preparation and LoRA training | PASS |
| Saved tokenizer + base model + adapter inference | PASS |
| Qwen, Llama 2/3, Mistral, Gemma compatibility and tokenizer tests | PASS: smoke scope, not full cross-family training |
| Run isolation, logging, metadata, and LoRA artifact verification | PASS |
| Original acceptance regression suite | 64 passed |
| Post-relocation regression suite | 68 passed (64 original + 4 layout checks) |
| Current suite, including console observability and Docker contract tests | 92 passed; host/static scope |

Validation ran on Windows with CPU-only PyTorch. See the portable
[acceptance report](docs/acceptance_report.md) for evidence, exact scope, and limitations.

## Limitations

- Runtime metadata reports visible GPU/container facts but does not qualify every
  driver, CUDA, image, or orchestrator combination.
- Full fine-tuning has configuration/unit coverage but no real full-method run.
- QLoRA is not implemented; DPO training is also intentionally unsupported.
- Large datasets are materialized in memory; 100k/1M-row scale was not validated.
- Abrupt power loss can leave stale run status; automatic recovery is not implemented.
- A LoRA adapter is not a standalone merged model; its base model is required.
- Child-process CUDA allocator peaks are explicitly unavailable; whole-device
  shared-GPU usage is never mislabeled as process-specific usage.
- Minimal training and inference tests prove mechanics, not model quality improvement.

## Developer documentation

Source is under `src/fine_tuning_pipeline/`, tests under `tests/`, and the default
input under `configs/config.yaml`. Root `configs/` and `examples/` contain public
examples; generated outputs, private datasets, and the upstream checkout are
excluded by [.gitignore](.gitignore).

See [repository layout and relocation scope](docs/repository_structure.md).
[Development documentation](docs/development.md) explains how
future contributors can register a dataset adapter, add a model-family mapping,
or extend validated training settings without bypassing the existing layers.

With the environment active, run the existing regression suite:

```bash
python -m unittest discover -s tests -v
```

The approved relocation changes import/path references, not training logic.
Release preparation does not change the trainer or qualify container deployment.

## License

Original project code is licensed under [Apache License 2.0](LICENSE).
LLaMA-Factory is a separate Apache-2.0 dependency; model weights, tokenizers,
datasets, and other dependencies retain their own licenses and access terms.

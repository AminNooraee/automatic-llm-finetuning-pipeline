# Automatic LLM Fine-tuning Pipeline

Configuration-driven supervised fine-tuning with automatic dataset preparation,
model/template compatibility, traceable training runs, and optional verified
OpenAI-compatible serving and gateway handoff.

Project #1 is functionally validated for the supported SFT/LoRA workflow. This
repository uses an editable `src/` package, root tests, and documented example
configs. The project's original code is licensed under [Apache License 2.0](LICENSE).

Latest tagged release: `v1.0.0`. Current `main` contains unreleased post-v1.0.0
improvements. See [current main status](docs/current_status.md),
[release notes](RELEASE_NOTES.md), [changelog](CHANGELOG.md),
[publishing checklist](RELEASE_CHECKLIST.md), and
[readiness audit](docs/release_readiness.md).

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
Artifact Validation -> verified training artifact / training failure
  |
  +-- train-only completion
  |
  +-- optional vLLM Docker serving
        |-- base model alias
        |-- fine-tuned LoRA alias
        `-- /v1/models + inference verification
              |
              +-- direct endpoint handoff
              `-- optional existing LiteLLM registration
                    `-- gateway discovery + inference verification
                          `-- gateway endpoint handoff
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
- Quiet, default concise, and full console modes while retaining the complete
  unfiltered backend stream in `logs/train.log`.
- Artifact validation before a run is declared successful.
- Optional safe vLLM Docker launch for the qualified LoRA path, loading one base
  model and one static adapter in a single process under separate aliases.
- Direct `/v1/models` plus real Chat Completions verification for both aliases.
- Optional dynamic registration with an externally managed LiteLLM gateway,
  conflict checks, end-to-end inference verification, and provider-neutral handoff.
- Environment-backed gateway credentials with recursive snapshot/artifact/log/error
  redaction; resolved credentials are never persisted.
- Post-training adapter loading and inference validated during historical
  acceptance; train-only execution still performs no automatic inference.

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
Clean host-native installation remains a separate qualification gate. A narrow
CUDA Docker/H100 BF16 LoRA smoke run has succeeded; it does not establish general
GPU, driver, CUDA, model, precision, or full-fine-tuning compatibility.
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
and the core stack are pinned, not every OS/transitive package. The historical
Windows acceptance host had no Docker daemon. The later CUDA Docker workflow
completed a real LoRA training run on one NVIDIA H100; see the
[current status](docs/current_status.md) for its exact scope.

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

Omit `serving` and `gateway` for the unchanged train-only workflow. For the full
sanitized operational example, see
[configs/serving_gateway_example.yaml](configs/serving_gateway_example.yaml),
[serving](docs/serving.md), and [gateway registration](docs/gateway.md).
Remote LiteLLM gateways require HTTPS by default. An operator may explicitly set
`gateway.allow_insecure_http: true` only for a trusted private/local network;
this sends the gateway credential without TLS and produces a runtime warning.
The project never enables this opt-in automatically.

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
requires both process completion and method-aware artifact validation. LoRA runs
also require a valid adapter configuration, positive rank, and a base-model
relationship consistent with the recorded training inputs.

Console output defaults to `concise`: lifecycle events, throttled progress,
emitted training metrics, warnings, errors, and final metrics. Use `quiet` for
lifecycle and errors only, or `full` for the complete unfiltered backend stream:

```yaml
observability:
  console_verbosity: quiet  # quiet | concise | full
```

This affects terminal presentation only. `logs/train.log` contains pipeline
lifecycle records plus the complete unfiltered LLaMA-Factory/Transformers stdout
and stderr stream, including output hidden from `quiet` and `concise`. See
[console observability](docs/observability.md).

LoRA output is recorded as `lora_adapter`, not a standalone model. When optional
serving succeeds, `serving/endpoint_manifest.json` is the preferred handoff. When
optional LiteLLM registration also succeeds, `gateway/gateway_manifest.json`
becomes preferred. Both contain only an OpenAI-compatible base URL and the base/
fine-tuned aliases; downstream consumers do not need Docker, vLLM, LoRA-path, or
LiteLLM routing details. No Project #2 source dependency is introduced.

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
| Current suite, including serving/gateway/security mocks and Docker contracts | 141 passed, 0 failed, 0 skipped |
| CUDA Docker build | PASS |
| H100 Docker smoke | PASS: Qwen2.5-0.5B-Instruct, LoRA, BF16, one visible H100, tiny synthetic dataset, one epoch |

The original acceptance ran on Windows with CPU-only PyTorch. Later Docker/H100
evidence supplements rather than rewrites that history. See the
[current status](docs/current_status.md) and portable
[acceptance report](docs/acceptance_report.md) for scope and limitations.

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
- The accepted H100 smoke does not qualify all models, GPUs, drivers, CUDA
  versions, FP16, full fine-tuning, larger models, large datasets, or production
  serving.
- Serving and gateway code is mock-validated but has not yet been runtime-qualified
  against a real GPU/vLLM v0.11.0/LiteLLM dynamic-database environment.
- Automatic serving is intentionally rejected for `training.method: full`; only
  LoRA + vLLM + Docker is implemented in this initial path.

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

Current documentation distinguishes the historical v1.0.0/relocation evidence
from later unreleased validation. See [current main status](docs/current_status.md).

## One-command fully containerized deployment

On a Linux model server, configure the sanitized
[`configs/full_pipeline_example.yaml`](configs/full_pipeline_example.yaml), export
runtime credentials, and run:

```bash
export LITELLM_BASE_URL=https://litellm.example.com
export LITELLM_API_KEY='runtime-value' # use a secret manager/shell history controls
export HF_TOKEN='runtime-value' # only when the model requires it
sh scripts/run_pipeline.sh
```

Image builds use `PIPELINE_DOCKER_BUILD_NETWORK=auto` by default: the normal
Docker build network is tried first, and a recognized DNS/network-resolution
failure is retried once with host build networking. Set it to `default` to
disable fallback or `host` to use host networking from the first build attempt.
This setting affects image builds only; host build networking reduces build-time
network isolation and does not change container runtime networking.

If BuildKit still reports DNS resolution failure with host build networking,
`auto` performs one final build using validated non-loopback resolver IPs read
from the host's `/etc/resolv.conf`. `PIPELINE_DOCKER_BUILD_DNS` can explicitly
supply a comma- or space-separated IPv4/IPv6 list for that final attempt. The
launcher never invents public resolvers and never edits host resolver or Docker
daemon configuration.

Local datasets inside the checkout work with the default mount. For datasets in
an external directory, export `PIPELINE_DATASETS_DIR` as that directory's
absolute host path and make `dataset.path` relative to that root. The launcher
mounts the root read-only and translates the path for the trainer container.

The user clones the repository, selects the model/dataset/settings, supplies
runtime secrets, and runs that command. The pipeline builds or safely reuses
source-identified controller and CUDA trainer images, runs training in an
ephemeral container, validates the adapter, starts one persistent vLLM process
for the base plus static LoRA aliases, registers both aliases in an existing
LiteLLM, verifies both direct and gateway inference paths, and writes
`deployment_manifest.json`.

The host needs Git, Docker Engine, NVIDIA driver, and NVIDIA Container Toolkit.
It does not need host Python, torch, Transformers, LLaMA-Factory, vLLM, or a
virtual environment. The trusted controller mounts `/var/run/docker.sock`; that
socket is root-equivalent host access. Never run an unreviewed controller image.
The trainer and vLLM containers never receive the socket.

## License

Original project code is licensed under [Apache License 2.0](LICENSE).
LLaMA-Factory is a separate Apache-2.0 dependency; model weights, tokenizers,
datasets, and other dependencies retain their own licenses and access terms.

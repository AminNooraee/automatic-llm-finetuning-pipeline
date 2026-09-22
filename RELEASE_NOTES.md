# Release notes: v1.0.0

> **Historical v1.0.0 release snapshot.** The `main` branch contains newer
> unreleased improvements and additional Docker/H100 validation. See the
> [changelog](CHANGELOG.md) and [current main status](docs/current_status.md).

Preparation date: September 16, 2026

Status: **source release authorized; deployment qualification remains incomplete**.
Package metadata is `1.0.0`; the release Git tag is `v1.0.0`. Docker image
publication and successful container execution are not claimed by this release.
Original project code is licensed under [Apache License 2.0](LICENSE).

## Overview

Project #1 provides a configuration-driven supervised fine-tuning workflow around
LLaMA-Factory. A user supplies a supported HuggingFace model identifier, a dataset
source, and training parameters. The pipeline prepares data, resolves model/template
compatibility, generates run-local dataset registration/training YAML, launches
fine-tuning, and checks output artifacts before declaring success.

The validated result is a LoRA adapter used with its matching base model.
"Multi-model" and "multi-dataset" mean the documented supported families and
text-only schemas, not every architecture or arbitrary Hub dataset.

## Features

- Automatic fine-tuning orchestration using the installed LLaMA-Factory backend;
  no manual edits to its dataset registry or training configuration are needed.
- Multi-model compatibility: Qwen/Qwen2/Qwen2.5, Llama 2/3, Mistral, and Gemma,
  with automatic template resolution and incompatible-override validation.
- Multi-dataset sources: JSON arrays, JSONL, CSV, Parquet, HuggingFace datasets,
  and raw ChatML text.
- Supported schemas: Alpaca, instruction/response, prompt/completion, QA, RAG QA,
  ShareGPT, OpenAI chat, and ChatML. Scored automatic detection, manual overrides,
  and supported column mappings normalize data to instruction/input/output.
- Config-driven LoRA parameters and full-method configuration/validation.
  DPO recognition exists, but preference training is not enabled.
- Run management: unique collision-safe IDs, isolated artifacts, input/resolved/
  training config snapshots, dataset snapshots/normalization, logs, and metadata.
- Artifact verification: a zero process exit is insufficient if required saved
  model artifacts are missing or empty.
- Docker deployment files: digest-pinned CPU and selectable CUDA 12.8 runtime
  bases, matched PyTorch wheels, non-root execution, persistent mounts, Tini,
  build-context exclusions, and build-time dependency/regression checks.
- Editable `src/` package, root tests, public configs/synthetic examples, and
  installation/configuration/extension/acceptance documentation.

## Validation results

| Validation | Result | Scope |
| --- | --- | --- |
| Real Qwen2.5-0.5B LoRA training | PASS | Windows/CPU, local JSON, 2 rows, 1 epoch |
| Real HuggingFace dataset training | PASS | `lhoestq/demo1`, 1 selected row, generated dataset registration |
| Base-model-plus-adapter inference | PASS | Trained Qwen adapter and saved tokenizer loaded and generated text |
| Model compatibility | PASS | Qwen, Llama 2/3, Mistral, Gemma config/tokenizer/YAML smoke checks; not cross-family full training |
| Run isolation, logs, metadata, artifacts | PASS | Actual LoRA runs plus lifecycle/negative-path tests |
| Original acceptance suite | PASS | 64 tests |
| Relocated suite | PASS | 68 tests: original 64 plus 4 layout checks |
| Current release-preparation suite | PASS | 72 host tests, including 4 static Docker checks |
| Dependency consistency | PASS | Existing environment `pip check`; application pins match installed versions |
| Docker image build/container training | PENDING | Docker unavailable on the current validation host |
| Clean-target installation | PENDING | Existing-environment editable installation is not clean reproduction |

Real training/inference was completed before packaging. Release preparation does
not restart it or claim additional hardware/model-quality qualification. See
[the acceptance report](docs/acceptance_report.md) and
[release readiness audit](docs/release_readiness.md) for evidence and limitations.

## Installation and execution

Install a compatible platform-specific PyTorch package set, then from the checkout:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
```

Edit `configs/config.yaml`, then run:

```bash
python -m fine_tuning_pipeline.train_pipeline
```

The default config is checkout-relative. Dataset/output paths are relative to
the input YAML. Each execution writes an isolated `runs/<run_id>/` directory.
Named configs use the existing Python API, not a `--config` flag.

Docker build recipes (deployment acceptance still pending):

```bash
docker build -f docker/Dockerfile -t automatic-llm-finetuner:1.0.0 .
docker build -f docker/Dockerfile --build-arg RUNTIME=cuda \
  -t automatic-llm-finetuner:1.0.0-cuda .
```

Use the [documented persistent mounts](docs/docker.md) for configs, datasets,
runs, and optional Hub cache. GPU access additionally requires host driver/toolkit
provisioning; neither build assumes a GPU exists.

## Known limitations and release gates

- Model, tokenizer, dataset, and dependency licenses/access terms remain separate
  from this project's Apache-2.0 license.
- Actual CPU/CUDA image builds, container training/inference, and a clean
  installation must be validated before claiming reproducible deployment.
- Release files must pass staged-tree review for sensitive data, private corpora,
  and generated artifacts; ignore rules alone do not certify content safety.
- Core application versions, the upstream source revision, and image bases are
  pinned; OS packages, build helpers, and transitive dependencies are not fully
  hash-locked. Conda is only a Python/pip bootstrap.
- GPU/H100, real full fine-tuning, larger-family training, and 100k/1M-row scale
  remain untested. Data is materialized in memory; inference is a separate check.
- QLoRA and DPO training are not implemented. LoRA is not an automatically merged,
  standalone model; full shard/integrity certification is not provided.
- Abrupt termination/power loss may leave stale metadata; automatic recovery is
  not implemented. Minimal acceptance proves mechanics, not model quality gains.
- Model/data revision provenance must be retained separately for reproducibility.
  Standalone wheel distribution with bundled configs/examples is not qualified.

See [the release checklist](RELEASE_CHECKLIST.md) for source-publication evidence
and the outstanding deployment checks. These notes do not claim a hosted GitHub
Release object, container image upload, or fresh-environment qualification.

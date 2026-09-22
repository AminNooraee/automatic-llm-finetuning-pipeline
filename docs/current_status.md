# Current main status

Status date: **2026-09-22**

Last runtime/H100 acceptance commit: `d0875235fff2b6df11a0965a5d5fe8c5da281696`

Latest tagged release: **v1.0.0**. The `main` branch contains unreleased
post-v1.0.0 improvements summarized here and in the [changelog](../CHANGELOG.md).
This page describes current evidence; the release notes and release checklist
remain historical v1.0.0 records.

## Current verification

- The regression suite passes **94 tests**.
- The canonical CUDA Docker image builds successfully. The tested image was
  `automatic-llm-finetuner:cuda-d087523`.
- A real one-epoch LoRA smoke run completed in the CUDA Docker workflow on one
  visible NVIDIA H100 using `Qwen/Qwen2.5-0.5B-Instruct`, BF16, and a tiny
  synthetic dataset. Accepted run:
  `20260922_075253_qwen2-5-0-5b-instruct_smoke-demo`.
- The accepted run verified adapter artifacts, metadata schema v2, resolved model
  revision, normalized-dataset SHA-256, environment/GPU/container provenance,
  final Trainer metrics, and complete training logging.
- Console modes are `quiet`, default `concise`, and `full`. Concise mode shows
  Trainer-phase progress and per-step loss/epoch/gradient-norm/learning-rate
  values while filtering preprocessing noise and duplicate final progress.

The H100 result is deliberately narrow. It does not qualify all GPUs, drivers,
CUDA versions, images, model families, model sizes, datasets, or training methods.
It does not qualify FP16, large-scale data, real full fine-tuning, or production
serving. Historical Windows/CPU training and adapter-inference evidence remains
documented separately in the [acceptance report](acceptance_report.md).

## Reproducibility and run contract

Metadata schema v2 records requested and resolved model revisions, the exact
normalized-dataset SHA-256, package/platform/GPU context, optional container
identity, duration, normalized final metrics, and output relationships.
`environment.json` stores the environment snapshot separately.

A LoRA output is explicitly related to its required base model and is not
presented as a standalone merged model. Provider-neutral serving handoff metadata
records the base, adapter, template, and suggested identifier for future use.
No endpoint, vLLM, LiteLLM, adapter merge, inference server, benchmark integration,
or Project #2 connection is implemented.

`logs/train.log` contains pipeline lifecycle records and the complete unfiltered
LLaMA-Factory/Transformers stdout/stderr stream. Console filtering never changes
the backend stream written to that file.

## Known limitations and open qualification

- QLoRA and DPO training are not implemented.
- Large-data scale and bounded-memory operation are not qualified.
- Real full-fine-tuning acceptance has not been performed.
- FP16 has not received separate acceptance.
- Other model families have not received equivalent H100 weight-level validation.
- Serving and endpoints are not implemented.
- Abrupt termination can leave stale state; automatic recovery/reconciliation is
  not implemented.
- A complete clean-target installation remains a separate qualification gate.
  The Docker CUDA build/run evidence does not establish host-native clean-install
  reproduction, universal GPU qualification, or standalone-wheel distribution.


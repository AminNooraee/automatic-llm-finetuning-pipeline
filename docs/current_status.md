# Current main status

Status date: **2026-09-27**

Last runtime/H100 acceptance commit: `d0875235fff2b6df11a0965a5d5fe8c5da281696`

Latest tagged release: **v1.0.0**. The `main` branch contains unreleased
post-v1.0.0 improvements summarized here and in the [changelog](../CHANGELOG.md).
This page describes current evidence; the release notes and release checklist
remain historical v1.0.0 records.

## Current verification

- The complete current regression suite passes **138 tests** with **0 failures**
  and **0 skips**. This includes 32 focused serving, gateway, lifecycle, and
  secret-redaction tests that use no Docker daemon, GPU, vLLM, LiteLLM, or network.
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

The H100 training result is deliberately narrow. It does not qualify all GPUs, drivers,
CUDA versions, images, model families, model sizes, datasets, or training methods.
It does not qualify FP16, large-scale data, real full fine-tuning, or production
serving. Optional vLLM/LiteLLM behavior is mock-validated only. Historical
Windows/CPU training and adapter-inference evidence remains
documented separately in the [acceptance report](acceptance_report.md).

## Reproducibility and run contract

Metadata schema v2 records requested and resolved model revisions, the exact
normalized-dataset SHA-256, package/platform/GPU context, optional container
identity, duration, normalized final metrics, and output relationships.
`environment.json` stores the environment snapshot separately.

A LoRA output is explicitly related to its required base model and is not
presented as a standalone merged model. Optional LoRA + vLLM Docker serving and
optional LiteLLM dynamic registration are implemented behind modular interfaces.
Both stages require model discovery and base/fine-tuned inference before ready,
then emit provider-neutral endpoint handoffs. Adapter merge, other backends,
full-model automatic serving, benchmark integration, and a Project #2 connection
remain unimplemented.

`logs/train.log` contains pipeline lifecycle records and the complete unfiltered
LLaMA-Factory/Transformers stdout/stderr stream. Console filtering never changes
the backend stream written to that file.

## Known limitations and open qualification

- QLoRA and DPO training are not implemented.
- Large-data scale and bounded-memory operation are not qualified.
- Real full-fine-tuning acceptance has not been performed.
- FP16 has not received separate acceptance.
- Other model families have not received equivalent H100 weight-level validation.
- Real GPU/vLLM and LiteLLM dynamic-database runtime acceptance has not been performed.
- Abrupt termination can leave stale state; automatic recovery/reconciliation is
  not implemented.
- A complete clean-target installation remains a separate qualification gate.
  The Docker CUDA build/run evidence does not establish host-native clean-install
  reproduction, universal GPU qualification, or standalone-wheel distribution.


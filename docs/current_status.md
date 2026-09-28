# Current main status

Status date: **2026-09-28**

Last runtime/H100 acceptance commit: `d0875235fff2b6df11a0965a5d5fe8c5da281696`

Latest tagged release: **v1.0.0**. The `main` branch contains unreleased
post-v1.0.0 improvements summarized here and in the [changelog](../CHANGELOG.md).
This page describes current evidence; the release notes and release checklist
remain historical v1.0.0 records.

## Current verification

- The complete current regression suite passes **141 tests** with **0 failures**
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

- On 2026-09-28, the complete one-command workflow was accepted on Server53
  (Linux, 2x NVIDIA H100 80GB) using GPU1 only, with the existing Server52
  LiteLLM gateway. Accepted run: `20260928_074935_qwen2-5-0-5b-instruct_server53-acceptance-demo`.
  The run used Qwen/Qwen2.5-0.5B-Instruct, LoRA rank 8/alpha 16, BF16, one
  epoch, two demo samples, and two optimization steps. Training artifacts,
  direct vLLM discovery/inference, dynamic registration of both aliases, and
  real LiteLLM Chat Completions for both aliases all passed. The persistent
  serving container remained Up and a deployment manifest was produced.

The H100 and Server53 end-to-end results are deliberately narrow. They do not
qualify all GPUs, drivers, CUDA versions, images, model families, model sizes,
datasets, or training methods. They do not qualify FP16, large-scale data, real
full fine-tuning, QLoRA, DPO, or general production operation. Historical
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
- The accepted runtime scope is one Server53 H100 environment, one vLLM
  `v0.11.0` image, one Qwen2.5-0.5B LoRA/BF16 run, and one trusted private-LAN
  LiteLLM gateway. It is not general production qualification.
- Abrupt termination can leave stale state; automatic recovery/reconciliation is
  not implemented.
- A complete clean-target installation remains a separate qualification gate.
  The Docker CUDA build/run evidence does not establish host-native clean-install
  reproduction, universal GPU qualification, or standalone-wheel distribution.

## Containerized deployment status

The one-command launcher/controller/trainer handoff, ownership checks,
persistence policy, direct/gateway verification ordering, rollback, path
validation, and secret-free artifacts are covered by tests and by the narrow
2026-09-28 Server53 runtime acceptance. That acceptance verified real vLLM
`v0.11.0` and LiteLLM dynamic registration/inference interoperability, without
claiming broader production qualification.


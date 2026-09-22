# Changelog

This file records Project #1 capabilities and packaging milestones, not an
invented historical Git release sequence. Dates below are preparation dates.

## Unreleased

- Added metadata schema v2 with requested/resolved model revision, normalized
  dataset SHA-256, environment/package/GPU snapshots, container provenance,
  training duration, normalized final Trainer metrics, and explicit
  base-model/adapter relationships.
- Added provider-neutral future serving handoff metadata. Endpoint serving, vLLM,
  LiteLLM, adapter merging, and Project #2 integration remain unimplemented.
- Retained the full unfiltered backend stdout/stderr stream in `logs/train.log`
  alongside pipeline lifecycle logging.
- Added `quiet`, default `concise`, and `full` console modes, including
  Trainer-phase progress filtering, final-progress deduplication, and compact
  per-step metrics.
- Added deterministic Parquet handle cleanup.
- Built the CUDA Docker image and completed a narrow real H100 BF16 LoRA smoke
  validation with Qwen2.5-0.5B-Instruct.
- Expanded the current regression suite to 94 passing tests.
- Added an authoritative [current-main status page](docs/current_status.md).

Clean-target installation and broader GPU/model/method qualification remain
separate gates. These changes are post-v1.0.0 and are not retroactively part of
the tagged release.

## v1.0.0 — 2026-09-16

### Added

- Automatic, YAML-driven supervised fine-tuning orchestration around LLaMA-Factory.
- Modular dataset source/schema registries, confidence-scored detection, manual
  overrides, supported column mapping, validation, and SFT normalization.
- JSON/JSONL/CSV/Parquet/HuggingFace loading; Alpaca, instruction/response,
  prompt/completion, QA/RAG QA, ShareGPT, OpenAI chat, and ChatML schemas.
- Qwen, Llama 2/3, Mistral, and Gemma family detection, automatic templates,
  warnings, and incompatible model/template rejection.
- Validated LoRA and full-method training configuration with explicit parameters.
- Independent timestamped runs, snapshots, logs, metadata, and artifact checks.
- CPU-default and selectable CUDA Dockerfiles using pinned bases, a matched
  PyTorch stack, non-root execution, persistent runtime mounts, and build checks.
- Professional documentation, public configs/synthetic examples, editable package
  metadata, release notes, a publishing checklist, and a readiness audit.
- 72 regression tests: 64 original, 4 relocation, and 4 static Docker checks.

### Packaging and compatibility

- Organized 28 original source files under `src/fine_tuning_pipeline/`, seven
  existing test modules under root `tests/`, and public inputs under configs/examples.
- Updated imports, mock targets, and checkout/config-relative paths without
  changing training behavior. Default launch is module execution.
- Aligned package metadata with the requested `1.0.0` release version.
- Licensed original project code under Apache License 2.0 at the owner's request.
- Added/strengthened Git and Docker exclusions for private artifacts, caches,
  environments, checkpoint directories, credentials, and machine-local state.

### Previously validated correction

- Translated the user checkpointing flag to LLaMA-Factory's inverse
  `disable_gradient_checkpointing` loader flag during functional acceptance.
  Release preparation makes no further training-logic change.

### Validated

- Real one-epoch Qwen LoRA training with local JSON and a small HuggingFace subset.
- Saved base-model-plus-adapter inference and run artifacts/logs/success metadata.
- Family config/tokenizer/template/YAML smoke tests and handled failure paths.
- Relocated imports/config loading and 72 host regression tests; `pip check` passes.

### Not yet qualified

- Actual Docker builds/container training, clean installation, GPU/H100,
  real full fine-tuning, and large-data deployment. No QLoRA or DPO training.
- Fully hash-locked offline reproduction or standalone wheel distribution.
- Source publication does not certify Docker/clean-target reproduction or remove
  the need for licensing/privacy review of user-supplied models and datasets.

See [release notes](RELEASE_NOTES.md) and [the checklist](RELEASE_CHECKLIST.md).

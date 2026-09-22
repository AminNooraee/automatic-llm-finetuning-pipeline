# Project #1 release readiness audit

Target version: **v1.0.0** (package metadata `1.0.0`)

Audit date: September 16, 2026

> **Historical v1.0.0 readiness audit.** The body and its 71-file/72-test
> evidence describe the pre-release state on the audit date. They are retained
> rather than rewritten as current-main evidence. See the addendum below,
> [current main status](current_status.md), and the [changelog](../CHANGELOG.md).

## Current-main addendum - 2026-09-22

- The current regression suite passes 94 tests.
- The CUDA Docker image built successfully.
- A narrow H100 acceptance run succeeded with Qwen2.5-0.5B-Instruct, LoRA, BF16,
  one visible H100, a tiny synthetic dataset, and one epoch.
- The current source tree includes `observability.py`, `console_output.py`,
  `test_observability.py`, `test_console_output.py`, and
  `docs/observability.md`; these intentionally do not appear in the historical
  71-file staged listing below.
- Clean-target host installation remains a separate qualification gate.
- Universal GPU/model/driver/CUDA qualification, FP16, cross-family H100
  training, and real full fine-tuning are still not claimed.

Decision: **source publication authorized; GitHub preparation in progress**.
Functional SFT/LoRA acceptance, source publication, and deployment qualification
are different scopes. The owner selected Apache-2.0 and requested a release commit,
tag, and public repository. Docker/clean-target qualification remains incomplete
and must not be represented as successful by source publication.

## Audit results

| Area | Status | Evidence / remaining work |
| --- | --- | --- |
| README completeness | PASS | Overview/LLaMA-Factory relationship, architecture, features, supported inputs, local/Docker installation, usage/mounts, results, limits, developer/test links |
| Documentation consistency | PASS | Current `src/`/root-test layout, `configs/config.yaml`, module command, config-relative paths, run/artifact contract, and scoped validation claims agree |
| Historical/current test counts | PASS | Original 64, relocation 68, current 72 are explicitly distinguished |
| Docker documentation | PASS (content) | Canonical `-f docker/Dockerfile`, CPU/CUDA builds, Bash/PowerShell mounts, UID/GID/cache/authentication, GPU prerequisites, and acceptance commands |
| Actual Docker qualification | UNVERIFIED | No Docker executable/daemon on the current host; neither image build nor container train/inference has been performed |
| Requirements consistency | PASS (existing host) | Application pins match installed versions; backend revision matches local HEAD; `pip check` succeeds; PyYAML package dependency agrees with requirements |
| Platform selection | PASS (design) | Python minimum allows documented 3.12; Conda is explicitly bootstrap-only; Docker separates matched PyTorch CPU/cu128 packages from platform-neutral requirements |
| Environment reproduction | PARTIAL / RELEASE GATE | Immutable backend/image bases and core pins exist; clean installation is untested and transitive/build/OS dependencies are not fully hash-locked |
| Git ignore safety | PASS (probed paths) | Outputs, weights, caches, environments, private data, temp files, checkpoints, and common credential paths excluded; public release files remain includable |
| Sensitive staged-tree/history review | PASS (review scope) | Exact 71-file staged set contains only curated text sources/docs/public inputs; forbidden-path and credential-pattern checks passed; no prior root history. This is not exhaustive secret certification |
| Version and release documents | PASS | Package `1.0.0`, `v1.0.0` notes/changelog/checklist; remote tag verification is required |
| License | PASS | Owner selected Apache License 2.0; full text and package/documentation references updated |
| Core preservation | PASS | Source, training requirements, Conda bootstrap, Docker runtime logic, public inputs, and prior artifacts unchanged by release preparation |

Supported models/datasets are the documented family/text-SFT subset, not all
HuggingFace architectures or schemas. Multi-family smoke success is not full
weight-level training qualification. LoRA inference requires the matching base.

## Final verification evidence

- Full host suite: **72 tests passed** (`unittest discover -s tests -v`).
- Editable package metadata reports `1.0.0`; default configuration loads.
- Eight application dependency pins match the existing environment; `pip check`
  reports no broken requirements. This is not a fresh-install qualification.
- Seventy-six relative documentation links resolve, with zero broken links.
- Eighteen required private/generated path probes are ignored; curated public
  release files remain includable.
- A limited scan of curated files found no matches for selected credential/private
  key patterns. This is not a comprehensive staged-tree/history secrets review.
- Thirty-four protected source/environment/Docker files retain their hashes;
  186 prior runtime/evidence files retain their recorded sizes and timestamps.
- The earlier release audit performed no Git writes. This publication pass
  initializes the root repository and prepares the explicitly requested commit/tag.
  No source/test relocation or training-logic change is involved.

## Source publication review

The exact staged set below contains 71 files. No model/checkpoint/cache/private
corpus/virtual environment is included, and all entries are ordinary text files.
The largest entry at review is `tests/test_pipeline.py` (12,508 bytes).
`git diff --cached --check` and the forbidden tracked-path check pass. Selected
Hub/GitHub/AWS credential and private-key patterns have no matches. Public YAML
and synthetic data were inspected separately. No previous root Git history exists.
The full 72-test suite, `pip check`, default-config loading, version/license metadata,
and protected-file hash comparison passed again during publication preparation.

These checks do not claim an exhaustive organizational secret-scanner review,
Docker execution, clean installation, GPU testing, or a hosted GitHub Release.
Actual commit/tag/remote verification is reported in the publication handoff.

```text
.dockerignore
.gitignore
CHANGELOG.md
LICENSE
README.md
RELEASE_CHECKLIST.md
RELEASE_NOTES.md
configs/config.yaml
configs/huggingface_dataset_example.yaml
configs/llama_example.yaml
configs/qwen_example.yaml
docker/Dockerfile
docker/pytorch-requirements.txt
docs/acceptance_report.md
docs/architecture.md
docs/configuration.md
docs/development.md
docs/docker.md
docs/installation.md
docs/pipeline_reference.md
docs/release_readiness.md
docs/repository_structure.md
docs/supported_datasets.md
docs/supported_models.md
docs/usage.md
environment.yml
examples/README.md
examples/datasets/alpaca_demo.json
examples/datasets/openai_chat.json
examples/datasets/rag_qa.json
pyproject.toml
requirements.txt
src/fine_tuning_pipeline/__init__.py
src/fine_tuning_pipeline/artifact_validator.py
src/fine_tuning_pipeline/dataset_adapter.py
src/fine_tuning_pipeline/dataset_adapters/__init__.py
src/fine_tuning_pipeline/dataset_adapters/alpaca_adapter.py
src/fine_tuning_pipeline/dataset_adapters/base.py
src/fine_tuning_pipeline/dataset_adapters/chatml_adapter.py
src/fine_tuning_pipeline/dataset_adapters/conversation.py
src/fine_tuning_pipeline/dataset_adapters/csv_adapter.py
src/fine_tuning_pipeline/dataset_adapters/dpo_adapter.py
src/fine_tuning_pipeline/dataset_adapters/huggingface_adapter.py
src/fine_tuning_pipeline/dataset_adapters/instruction_response_adapter.py
src/fine_tuning_pipeline/dataset_adapters/json_adapter.py
src/fine_tuning_pipeline/dataset_adapters/jsonl_adapter.py
src/fine_tuning_pipeline/dataset_adapters/openai_chat_adapter.py
src/fine_tuning_pipeline/dataset_adapters/parquet_adapter.py
src/fine_tuning_pipeline/dataset_adapters/prompt_completion_adapter.py
src/fine_tuning_pipeline/dataset_adapters/qa_adapter.py
src/fine_tuning_pipeline/dataset_adapters/rag_qa_adapter.py
src/fine_tuning_pipeline/dataset_adapters/registry.py
src/fine_tuning_pipeline/dataset_adapters/sharegpt_adapter.py
src/fine_tuning_pipeline/dataset_config_generator.py
src/fine_tuning_pipeline/dataset_loader.py
src/fine_tuning_pipeline/model_manager.py
src/fine_tuning_pipeline/run_manager.py
src/fine_tuning_pipeline/train_pipeline.py
src/fine_tuning_pipeline/trainer.py
src/fine_tuning_pipeline/training_config.py
src/fine_tuning_pipeline/yaml_generator.py
tests/__init__.py
tests/test_dataset_registry.py
tests/test_dataset_schema_adapters.py
tests/test_dataset_source_adapters.py
tests/test_docker_support.py
tests/test_model_manager.py
tests/test_pipeline.py
tests/test_repository_layout.py
tests/test_run_management.py
tests/test_training_config.py
```

## Files excluded from publication

See [the complete never-commit list](../RELEASE_CHECKLIST.md). In particular:
`runs/`, `models/`, checkpoint directories, Hub caches/tokens, `venv/`/`.venv/`,
real secrets/credential files, private datasets, raw/generated acceptance artifacts,
and the local upstream checkout must stay local. The legacy runtime directory is
retained and ignored, not moved/deleted. Benchmark/serving/evaluation is out of scope.

Release preparation strengthens Git ignore rules for nested named caches/envs,
standalone `checkpoint-*` directories, and common credentials. Ignore probes use
read-only Git operations. A new project index is reviewed during publication.
Ignore matches do not
untrack existing content or detect secret values in otherwise valid source files.
[Git ignore semantics](https://git-scm.com/docs/gitignore).

## Publication and deployment follow-up

1. Retain the owner-selected Apache-2.0 license; review necessary dependency/model/
   dataset notices and terms before distributing any third-party material.
2. On a Docker-enabled machine, build both variants and qualify minimal mounted
   CPU LoRA training/artifact verification/inference; record image/dependency evidence.
3. Test documented installation in a fresh target environment, including tests,
   configuration, small local/Hub data, and adapter usability. Retain input provenance
   and resolved environment records; stronger locking is needed for bit-identical claims.
4. Review/scan the actual candidate/staged tree before the authorized source push;
   confirm no private material is included. Repository setup/publication is authorized.
5. Record verified commit/tag/remote evidence in the publication summary and keep
   pending deployment checks open in the [checklist](../RELEASE_CHECKLIST.md).

## Improvements, not blockers for the stated CPU/LoRA scope

- GPU/H100/mixed precision, larger-family/full-method training, and large-data
  qualification (required before advertising those deployments as tested).
- Complete platform/hash locks and offline wheel/source archives.
- Stronger model/data revision tracking, bounded-memory loading, shard integrity,
  and power-loss reconciliation.
- QLoRA/DPO training and merged standalone-model/wheel distribution, if later approved.

No training logic, model/dataset framework, benchmarking, serving, or evaluation
change is part of this audit. See [release notes](../RELEASE_NOTES.md),
[changelog](../CHANGELOG.md), [functional acceptance](acceptance_report.md), and
[Docker qualification](docker.md).

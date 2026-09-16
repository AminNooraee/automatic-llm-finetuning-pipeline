# Repository layout and relocation record

The approved source/test organization is complete. Files were moved individually:
no runtime directory, environment, cache, generated output, or temporary folder
was moved. Import and path references were updated without changing training logic.

## Published source layout

```text
automatic-llm-finetuning-pipeline/
  README.md
  LICENSE                     Apache License 2.0
  RELEASE_NOTES.md            v1.0.0 release scope/results
  CHANGELOG.md                 Capabilities and packaging milestones
  RELEASE_CHECKLIST.md         Publishing gates and never-commit policy
  .gitignore
  .dockerignore               Private runtime state excluded from builds
  pyproject.toml              Editable src-package installation
  requirements.txt            Pinned training integration + editable checkout
  environment.yml             Optional Python/pip bootstrap
  docs/
    architecture.md
    installation.md
    usage.md
    configuration.md
    supported_models.md
    supported_datasets.md
    acceptance_report.md
    development.md
    repository_structure.md
    pipeline_reference.md
    docker.md
    release_readiness.md
  docker/
    Dockerfile                CPU default / selectable CUDA runtime
    pytorch-requirements.txt   Matched platform-selected PyTorch stack
  configs/
    config.yaml               Default input
    qwen_example.yaml
    llama_example.yaml
    huggingface_dataset_example.yaml
  src/
    fine_tuning_pipeline/
      __init__.py
      artifact_validator.py
      dataset_adapter.py
      dataset_config_generator.py
      dataset_loader.py
      model_manager.py
      run_manager.py
      trainer.py
      training_config.py
      train_pipeline.py
      yaml_generator.py
      dataset_adapters/        Existing 18-file framework, unchanged
  tests/
    __init__.py
    test_dataset_registry.py
    test_dataset_schema_adapters.py
    test_dataset_source_adapters.py
    test_docker_support.py
    test_model_manager.py
    test_pipeline.py
    test_repository_layout.py
    test_run_management.py
    test_training_config.py
  examples/
    README.md
    datasets/
      alpaca_demo.json
      rag_qa.json
      openai_chat.json
```

The workspace folder does not need renaming to publish under this project name.
Root `.dockerignore` and `docker/` now provide container build support; actual
image-build/container/GPU qualification is pending. See [Docker deployment](docker.md).

## Moved files

| Previous location | New location | Count / classification |
| --- | --- | --- |
| `fine_tuning_pipeline/*.py` (selected implementation modules) | `src/fine_tuning_pipeline/` | 10 source modules |
| `fine_tuning_pipeline/dataset_adapters/*.py` | `src/fine_tuning_pipeline/dataset_adapters/` | 18 framework modules |
| `fine_tuning_pipeline/tests/test_*.py` | `tests/` | 7 existing regression modules |
| `fine_tuning_pipeline/config.yaml` | `configs/config.yaml` | Default input config |
| `datasets/demo_dataset.json` | `examples/datasets/alpaca_demo.json` | Public synthetic two-row fixture |
| `fine_tuning_pipeline/README.md` | `docs/pipeline_reference.md` | Existing pipeline documentation |

Total: 38 moved files. New package/test initializers, `pyproject.toml`, and four
layout checks in `tests/test_repository_layout.py` support the approved relocation.

Hash comparison confirms 25 moved source files are byte-identical, including all
dataset adapters, `model_manager.py`, and `training_config.py`. Only
`train_pipeline.py`, `dataset_adapter.py`, and `yaml_generator.py` needed
package import changes; the entry point also anchors its default YAML to
`configs/config.yaml` in the source checkout.

## Local-only files retained in place

| Location | Treatment |
| --- | --- |
| `runs/`, `models/`, `checkpoints/` | Runtime outputs; ignored, never moved or packaged |
| Legacy `fine_tuning_pipeline/` remainder | Raw acceptance evidence, generated YAML/data, caches and temporary directories; ignored |
| Private `datasets/` remainder | Ignored; only the public synthetic demo was moved |
| `venv/`, `.venv/`, HuggingFace caches | Machine-specific environment/cache state, not distributed |
| `LLaMA-Factory/` | Separate upstream checkout with existing data edits; preserved and ignored |
| `benchmark_pipeline/`, serving/evaluation code | Outside this task; not moved or changed |

At relocation time, the checkout root was not its own Git repository. Publication
preparation initializes a separate project-root source repository; the upstream
checkout retains its own Git history and pre-existing data edits and is excluded.

## Commands after relocation

From the repository root, install into your selected environment:

```bash
python -m pip install -r requirements.txt
python -m fine_tuning_pipeline.train_pipeline
python -m unittest discover -s tests -v
```

Default execution uses `configs/config.yaml`, independent of the shell directory.
Named configs use the existing API, not an invented `--config` option:

```bash
python -c "from fine_tuning_pipeline.train_pipeline import execute_training; execute_training('configs/qwen_example.yaml')"
```

Dataset paths and `output.runs_path` keep their config-relative semantics.
All provided YAML files place outputs in the checkout's `runs/`. Model output
and dataset registration remain absolute run-specific paths for LLaMA-Factory.

## Relocation validation

- Full regression suite: 68 tests passed (64 original + 4 relocation checks).
- All 28 package submodules import; installed imports/default config work from
  another directory without `PYTHONPATH`.
- Default and all three named YAML files load and pass training schema validation.
- Pipeline lifecycle tests cover preparation, successful mocked launch, artifact
  verification, metadata/logging, and handled failures.
- Editable installation succeeds without dependency upgrades; `pip check` passes.
- Cached JSON and HuggingFace recipes prepare successfully with real model config
  resolution; both generated YAML files parse with the installed LLaMA-Factory.
- Both existing actual LoRA adapters pass the relocated artifact validator.
- All 186 recorded runtime/evidence files retain their original sizes and
  modification timestamps; all 38 old selected paths are absent and new paths exist.

Real training/inference acceptance remains the previously completed validation.
This relocation does not restart those jobs or claim a fresh complete environment
installation. See [acceptance](acceptance_report.md) for smoke scope and limits.
Historical run metadata and retained source snapshots are not rewritten to change
past paths. Cached or generated files left in the legacy directory remain ignored.

## Remaining release gates

1. Retain [Apache License 2.0](../LICENSE) and review separate model, tokenizer,
   dataset, and dependency terms before distributing third-party material.
2. Inspect eventual staged contents for corpora, credentials, weights, raw logs,
   machine paths, vendor nesting, and unrelated code. Ignore rules do not untrack
   files already committed elsewhere.
3. Reproduce the pinned training environment on the chosen clean release target,
   lock transitive dependencies/binaries, and repeat minimal training/inference.
4. Build and qualify the provided CPU/CUDA Docker variants on a Docker-enabled
   target; H100/GPU and real container training/inference remain untested.
5. Editable source-checkout usage is supported; standalone wheel distribution
   with bundled default configs/examples is not claimed by this organization pass.

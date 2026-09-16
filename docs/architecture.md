# Architecture

The repository wraps LLaMA-Factory SFT; it prepares artifacts and manages runs
without changing the upstream trainer or its dataset registry.

## Inputs and outputs

Inputs are YAML sections for a HuggingFace model name, dataset source and alias,
and supported training parameters. The normal LoRA result is an adapter plus
saved tokenizer files, not a complete standalone copy of the base model.

```text
Model name -> Model Compatibility Layer -> family/template ----+
Training parameters -> Training Configuration -> validated args+-> training.yaml
Dataset -> source adapter -> schema adapter -> normalization --+-> dataset_info.json
                                                                     |
                                                               LLaMA-Factory
                                                                     |
                                                               LoRA adapter
                                                                     |
                                                             Artifact validation
                                                                     |
                                                           success / failed metadata
```

Run management, metadata, and logging apply throughout. Full-method output is
supported by configuration/artifact checks but was not actually trained in acceptance.

## Actual execution order

1. Load and check the default or explicitly supplied input YAML.
2. Allocate a unique run directory and snapshot the input config.
3. Resolve model family/template from model name and HuggingFace config.
4. Validate supported training fields and method-specific settings.
5. Snapshot the dataset source and load it through the source adapter registry.
6. Detect/validate its schema, normalize it, and validate SFT-compatible rows.
7. Write normalized JSON, run-local `dataset_info.json`, resolved input config,
   and LLaMA-Factory training YAML.
8. Launch the upstream CLI with this interpreter and absolute YAML path, capturing
   its stdout/stderr to the run log.
9. On exit zero, verify expected nonempty model artifacts; only then mark success.
10. On handled errors, mark failure, preserve the error log, and raise the error.
    Print a final run summary for training completion/failure.

Model/config validation failures happen before dataset preparation/training but
can still leave a run directory, input snapshot, metadata, and error log.
Malformed input rejected before run allocation does not have run-scoped metadata.

## Module responsibilities

All modules below live under `src/fine_tuning_pipeline/` and are imported as
`fine_tuning_pipeline.<module>`. The default config is `configs/config.yaml`;
package installation is editable from this checkout. Relocation does not change
the runtime flow or the LLaMA-Factory artifact contract.

| Current module | Responsibility |
| --- | --- |
| `train_pipeline.py` | Config loading, preparation, execution, lifecycle orchestration |
| `dataset_adapter.py` | Stable facade over the modular dataset framework |
| `dataset_adapters/base.py`, `registry.py` | Adapter contracts, scoring, registry selection, conversion |
| Dataset source/schema modules | Format-specific loading, validation, normalization |
| `dataset_loader.py` | Unified SFT row validation |
| `dataset_config_generator.py` | Run-local LLaMA-Factory dataset registration |
| `model_manager.py` | Model-family detection, generation/template mapping, compatibility warnings/errors |
| `training_config.py` | Supported parameter validation and typed argument mapping |
| `yaml_generator.py` | Complete run-specific YAML generation and backend-specific checkpointing translation |
| `trainer.py` | Upstream subprocess launch and combined training log capture |
| `run_manager.py` | Unique paths, snapshots, metadata persistence, logging |
| `artifact_validator.py` | Expected nonempty artifact checks for LoRA/full |

## Adapter stages

The source registry handles file/container loading; the schema registry handles
record meaning. CSV and JSONL are therefore not separate copies of every schema
adapter. Named-field adapters share `FieldMappingAdapter`; conversation adapters
share validation and per-assistant-turn SFT expansion.

Automatic detection reports heuristic confidence, not a statistical guarantee.
The registry rejects similarly scored matches. Manual overrides force an adapter
but still validate the input; preference records are explicitly rejected for SFT.

## Run contract

```text
runs/<UTC timestamp>_<model>_<dataset>[_collision suffix]/
  config/input_config.yaml
  config/resolved_config.yaml
  config/training.yaml
  dataset/original_dataset.<extension>
  dataset/normalized_dataset.json
  dataset/dataset_info.json
  logs/train.log
  model/
  metadata.json
```

Local datasets are copied with their source extension. Hub input gets an
`original_dataset.json` source/options descriptor, not a full raw dataset export.
LLaMA-Factory receives the run's dataset and model directories as absolute paths.
Collision-safe allocation never overwrites a previous run.

LoRA validation requires nonempty `adapter_config.json` and
`adapter_model.safetensors`. Full-method checks require nonempty `config.json`
and matching safetensors/bin weight files. These are existence/size checks, not
complete shard/integrity certification. Real LoRA deserialization was separately
proved during acceptance; inference is not invoked automatically by this flow.

Handled preparation/training errors are recorded. Power loss can leave stale
status because no process survives to finalize it; automatic restart/reconciliation
is not implemented. See [acceptance limitations](acceptance_report.md).

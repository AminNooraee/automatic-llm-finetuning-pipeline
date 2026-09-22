# Configuration reference

The default input is `configs/config.yaml`. Named examples live in
[configs/](../configs/) and use the existing API described in [usage](usage.md).
All supported common training fields must be explicit; the generator does not
silently fill in training hyperparameters. Upstream defaults still apply to
LLaMA-Factory options the wrapper does not expose.

## Model

| Key | Required | Behavior |
| --- | --- | --- |
| `model.name` | Yes | Nonempty HuggingFace identifier for a supported family |
| `model.template` | No | Automatic by default; an incompatible explicit override fails |
| `model.revision` | No | Hub branch, tag, or commit passed as `model_revision`; resolved commit is recorded when available |

Do not supply `family` to select behavior. Family/template are resolved and
recorded automatically. Config inspection can fall back to a recognizable name
with a warning when fetching is unavailable. Name/config conflicts, unsupported
families, and uncertain Llama generation stop preparation before training.
Revision lookup is best-effort: cached/offline runs remain supported, and
metadata records why a resolved Hub commit was unavailable.

## Optional container provenance

An orchestrator may supply non-secret image identity under `provenance.container`:

```yaml
provenance:
  container:
    image: registry.example/fine-tuner:h100
    image_id: sha256:...
    image_digest: sha256:...
    runtime: docker
```

The launcher variables `PIPELINE_CONTAINER_IMAGE`,
`PIPELINE_CONTAINER_IMAGE_ID`, `PIPELINE_CONTAINER_IMAGE_DIGEST`,
`PIPELINE_CONTAINER_RUNTIME`, `PIPELINE_CONTAINER_ID`, and
`PIPELINE_CONTAINERIZED` take precedence. Never put credentials in these values.

## Dataset

| Key | Required / default | Behavior |
| --- | --- | --- |
| `dataset.path` | Required | Local file or HuggingFace dataset identifier |
| `dataset.name` | Required | Nonempty single dataset registration alias; no commas |
| `dataset.source_format` | `auto` | Storage/container override: JSON, JSONL, CSV, Parquet, raw ChatML, HF |
| `dataset.format` | `auto` | Record-schema override; see [supported datasets](supported_datasets.md) |
| `dataset.columns` | Optional | Canonical-field -> source-column mapping |
| `dataset.subset` | Optional | HuggingFace dataset configuration name |
| `dataset.split` | `train` for HF | Split or subset expression such as `train[4:5]` |
| `dataset.load_kwargs` | Optional mapping | Passed to `datasets.load_dataset()` |

Put HuggingFace split/subset directly under `dataset`, not as `split`/`name` inside
`load_kwargs`. Not every Hub schema is automatically understood: nested or unusual
records must match a supported schema or use supported column mapping. Missing
local files are not silently treated as Hub identifiers.

```yaml
dataset:
  path: lhoestq/demo1
  name: hub_demo
  source_format: auto
  format: auto
  split: train[4:5]
  columns:
    instruction: package_name
    output: review
```

The mapping affects schema recognition/conversion; it is not arbitrary nested
data transformation. Detection scores are heuristics. Ambiguous matches fail;
set a supported explicit schema only when that interpretation is intentional.

## Training

| Input key | Allowed type/value | Generated LLaMA-Factory field |
| --- | --- | --- |
| `method` | `lora` or `full` | `finetuning_type` |
| `epochs` | Positive finite number | `num_train_epochs` |
| `learning_rate` | Positive finite number | `learning_rate` |
| `batch_size` | Positive integer | `per_device_train_batch_size` |
| `precision` | `fp32`, `fp16`, `bf16` | Explicit `fp16` and `bf16` flags |
| `gradient_checkpointing` | Boolean | Inverse `disable_gradient_checkpointing` loader flag |
| `cutoff_len` | Positive integer | `cutoff_len` |
| `save_steps` | Positive integer | `save_steps` |
| `logging_steps` | Positive integer | `logging_steps` |

## LoRA

For `training.method: lora`, all these fields are required:

| Input key | Allowed type/value | Generated field |
| --- | --- | --- |
| `training.lora.rank` | Positive integer | `lora_rank` |
| `training.lora.alpha` | Positive integer | `lora_alpha` |
| `training.lora.dropout` | Finite number in `[0, 1)` | `lora_dropout` |

For `method: full`, omit the `lora` section. Full fine-tuning configuration is
implemented, but a real full-method training run has not been acceptance-tested.
QLoRA is reserved and rejected until method-specific quantization is implemented.

Example:

```yaml
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
```

Unknown training parameters, missing fields, booleans supplied instead of numbers,
invalid precision/method values, and incompatible method-specific sections fail
validation. This wrapper is not an unrestricted YAML pass-through. For example,
gradient accumulation, schedulers, and quantization are not configurable wrapper
fields today; do not add them to YAML and expect upstream pass-through.

`fp32` disables the upstream mixed-precision flags; it does not guarantee frozen
base weights are reloaded in float32. The backend may retain the model config's
weight dtype for LoRA. BF16 was validated in one narrow Qwen2.5-0.5B H100 Docker
LoRA smoke run. FP16 has not been separately accepted, and broader
model/hardware combinations remain host/backend-dependent.

## Output

```yaml
output:
  runs_path: ../runs
```

The run root resolves relative to the YAML file. Default fallback is `../runs`.
The pipeline allocates the unique run ID and model/dataset/config subdirectories;
do not attempt to force an existing run as the output model directory through
unrecognized input keys. Absolute run paths are stored in the resolved config.

## Observability console

```yaml
observability:
  console_verbosity: concise
```

The section and field are optional; the default is `concise`.

| Value | Terminal behavior |
| --- | --- |
| `quiet` | Run lifecycle, compact model/dataset summaries, final status, and important errors |
| `concise` | Quiet output plus throttled progress, emitted training metrics, warnings, and errors |
| `full` | Complete unfiltered LLaMA-Factory/Transformers stdout and stderr |

Unknown observability keys and invalid values fail before a run directory is
allocated. This is a bounded wrapper setting, not an unrestricted backend
pass-through, and it does not change training arguments or hyperparameters.

In every mode, `logs/train.log` contains pipeline lifecycle records plus the
complete unfiltered backend stdout/stderr stream. Console filtering never removes
backend content from that file.

# Fine-tuning pipeline

The pipeline loads `configs/config.yaml`, detects and adapts the source dataset,
validates its `instruction`/`input`/`output` rows, creates local LLaMA-Factory
files, and starts SFT training. Local paths in `config.yaml` are resolved
relative to that file.

Install the editable package from the repository root, then run with the
project virtual environment:

```powershell
.\venv\Scripts\python.exe -m fine_tuning_pipeline.train_pipeline
```

## Model compatibility

Only the HuggingFace model identifier is required:

```yaml
model:
  name: meta-llama/Meta-Llama-3-8B-Instruct
```

Before processing the dataset, the pipeline inspects the model name and its
HuggingFace config, detects the model family, and selects the compatible
LLaMA-Factory template:

- Qwen, Qwen2, and Qwen2.5: `qwen`
- Llama 2: `llama2`
- Llama 3: `llama3`
- Mistral: `mistral`
- Gemma, Gemma 2, and Gemma 3: `gemma`, `gemma2`, and `gemma3`

An unsupported model, ambiguous Llama generation, model-name/config conflict,
or incompatible manual `model.template` override stops the run before dataset
or training artifacts are created. Selecting a model name that does not look
like an Instruct/Chat checkpoint produces a warning because it may be an
intentional base-model SFT run.

## Dataset configuration

Automatic source and schema detection is the default:

```yaml
dataset:
  path: ../datasets/data.jsonl
  name: my_dataset
  source_format: auto
  format: auto
```

`source_format` describes how records are stored. Supported values are:

- `auto` (default)
- `json` (a top-level JSON array)
- `jsonl` / `ndjson`
- `csv`
- `parquet`
- `chatml_text`
- `huggingface`

`format` describes the record schema. Supported values are:

- `auto` (default)
- `alpaca`
- `instruction_response`
- `prompt_completion`
- `qa_json`
- `rag_qa`
- `openai_chat`
- `sharegpt`
- `chatml`
- `dpo` (detection and validation only; DPO training is not enabled)

Automatic schema detection examines column names, JSON keys, and nested message
structure and reports a confidence score. If similarly scored adapters match,
the dataset is rejected as ambiguous and the error lists the candidates. Set
`dataset.format` explicitly to resolve such a case.

Alpaca examples use `instruction`, optional `input`, and `output`:

```json
[{"instruction": "Explain AI", "input": "briefly", "output": "AI is ..."}]
```

Instruction/response and prompt/completion are also accepted:

```json
[{"instruction": "Explain AI", "response": "AI is ..."}]
```

```json
[{"prompt": "Explain AI", "completion": "AI is ..."}]
```

Simple Q&A examples use `question`/`answer`. The legacy `prompt`/`response`
variant remains supported by `qa_json`. A `context` field selects `rag_qa` and
becomes the unified `input` field:

```json
[{"question": "What is AI?", "context": "briefly", "answer": "AI is ..."}]
```

OpenAI chat uses `messages` with `role`/`content`; ShareGPT uses
`conversations` with `from`/`value`. ChatML can be supplied as strings or a
`text` field in a structured source, or directly as a `.txt`/`.chatml` file.
Each assistant response becomes an SFT row; earlier system and conversation
turns are preserved in `input`.

For nonstandard CSV, Parquet, or HuggingFace columns, provide a canonical-to-
source mapping. This avoids Python changes:

```yaml
dataset:
  path: data.csv
  name: custom_data
  format: auto
  columns:
    instruction: prompt_text
    input: context_text
    output: answer_text
```

## HuggingFace datasets

HuggingFace support is lazy and optional. The `datasets` package is imported
only when a HuggingFace source is selected:

```yaml
dataset:
  path: organization/dataset-name
  name: training_data
  source_format: huggingface
  format: auto
  subset: null
  split: train
```

Optional `load_kwargs` are passed to `datasets.load_dataset()`. If the package
is unavailable, the pipeline reports how to enable HuggingFace support; local
JSON, JSONL, CSV, ChatML, and Parquet sources remain usable independently.
Parquet loading similarly imports `pyarrow` only when a Parquet source is used.

## Training configuration

Training hyperparameters are read from the `training` section and translated
to their LLaMA-Factory names. No training defaults are supplied by the YAML
generator, so every supported value must be explicit:

```yaml
training:
  method: lora
  epochs: 3
  learning_rate: 0.0002
  batch_size: 4
  precision: bf16
  gradient_checkpointing: true
  cutoff_len: 2048
  save_steps: 100
  logging_steps: 10
  lora:
    rank: 8
    alpha: 16
    dropout: 0.05
```

Supported precision values are `fp32`, `fp16`, and `bf16`. The layer emits
explicit LLaMA-Factory `fp16` and `bf16` flags, including both as `false` for
`fp32`. LoRA settings are required under `training.lora` when `method: lora`.
The user-facing `gradient_checkpointing` setting is translated to
LLaMA-Factory's inverse `disable_gradient_checkpointing` model-loader flag.

`method: full` is also supported; omit the `lora` section for a full
fine-tuning run. `qlora` is reserved for a future method-specific
quantization section and is rejected for now rather than silently running a
different training mode. Unknown parameters, missing required fields, invalid
types, and incompatible method-specific sections stop the pipeline before
dataset or training artifacts are created.

## Run management and artifacts

Every execution receives a unique timestamped directory under the configured
run root. The default is `../runs`, relative to `config.yaml`:

```yaml
output:
  runs_path: ../runs
```

Each run is isolated as follows:

```text
runs/<run_id>/
  config/
    input_config.yaml
    resolved_config.yaml
    training.yaml
  dataset/
    original_dataset.<source extension>
    normalized_dataset.json
    dataset_info.json
  logs/
    train.log
  model/
  environment.json
  metadata.json
```

Local input datasets are copied into the run using their original extension.
For HuggingFace and other non-file sources, `original_dataset.json` records the
source identifier, subset, and split. LLaMA-Factory writes only to the run's
`model/` directory, so a new run cannot resume or overwrite an older run.

`metadata.json` schema v2 retains the model, dataset, training, output, timestamp,
and status fields and adds requested/resolved model revision, revision lookup
status, the exact normalized dataset path and SHA-256, environment/package/GPU
details, optional container provenance, wall-clock duration, normalized final
training metrics, and the base-to-adapter relationship. The environment snapshot
is also stored in `environment.json`. A run is marked `success` only after training exits successfully
and its expected artifacts have been verified. LoRA runs require
`adapter_config.json` and `adapter_model.safetensors`; full fine-tuning requires
`config.json` plus non-empty model weight files. Pipeline messages and combined
LLaMA-Factory stdout/stderr are streamed live to the terminal and saved to
`logs/train.log`, including progress/loss/learning-rate/grad-norm fields whenever
the upstream trainer emits them.

Final values are normalized from `train_results.json`, `all_results.json`, and
`trainer_state.json`; training loss is not labeled as accuracy. Per-step history
stays authoritative in `trainer_state.json`. CUDA allocator counters are
process-local, so the parent marks them unavailable instead of reporting
whole-device shared-GPU memory as process usage. Optional probes fail open without
weakening strict model-artifact validation.

The provider-neutral `serving` block is a future handoff contract only. It does
not provide endpoint serving, vLLM launch, LiteLLM registration, adapter merging,
or Project #2 integration.

The legacy `fine_tuning_pipeline/` folder contains only retained local artifacts,
caches, and validation evidence. It is ignored and is not the source package.
See [the approved repository layout](repository_structure.md).

Run the regression tests from the repository root:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

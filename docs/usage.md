# Usage

Commands use the approved `src/` layout. Install the editable package with
`python -m pip install -r requirements.txt` from the repository root. Use the
same activated Python environment for preparation, training, and tests.

## Default one-command training

Edit [configs/config.yaml](../configs/config.yaml) with
your supported model, dataset source/alias, and required training fields. From
the repository root:

```bash
python -m fine_tuning_pipeline.train_pipeline
```

Module execution also works from another directory after editable installation;
the default config remains anchored to this checkout. On the validated Windows
workspace, activation is unnecessary with:

```powershell
.\venv\Scripts\python.exe -m fine_tuning_pipeline.train_pipeline
```

Every training execution creates a new isolated run. The two-row Qwen default
is a mechanics smoke test; even tiny CPU training can take minutes. Large/gated
models require suitable hardware and access before launch.

## Named example configs without changing the default

The existing Python API accepts a config path; no CLI `--config` flag exists.
From the repository root with your environment active:

```bash
python -c "from fine_tuning_pipeline.train_pipeline import execute_training; execute_training('configs/qwen_example.yaml')"
```

Substitute `configs/huggingface_dataset_example.yaml` for the actual one-row
Hub mechanics recipe. `configs/llama_example.yaml` is an illustrative BF16
configuration only; it is not a validated Llama training run and needs approved
gated access and capable hardware.

Model detection, normalization, dataset registration, and YAML generation are
automatic. Do not edit upstream `dataset_info.json` or a previous run's config.

## Preparation only

For inspection without starting training, use the existing preparation API from
the repository root:

```bash
python -c "from fine_tuning_pipeline.train_pipeline import prepare_training; config, yaml_path = prepare_training('configs/qwen_example.yaml'); print(yaml_path)"
```

This still creates a run and may fetch model configuration/dataset data, but it
does not fetch model weights for training or launch optimization. A successful
preparation-only run has status `prepared`, not `success`.

## Paths and output

Relative dataset paths and `output.runs_path` resolve against the supplied YAML's
directory. New files under `configs/` explicitly set `runs_path: ../runs`. The
default run-root fallback is also `../runs` relative to the YAML; set it explicitly
if using a config elsewhere. Do not blindly copy configs between directories.

Inspect the printed run path:

```text
runs/<run_id>/
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

Hub snapshots are source/options descriptors. The normalized selected records
are persisted. LoRA's model directory contains adapter configuration/weights and
saved tokenizer files; retain the matching base model identifier/revision for reuse.

## Understand completion and failure

Check `metadata.json`, `logs/train.log`, and the model directory together.
`success` means the process exited zero and required nonempty artifacts passed
validation; it does not mean quality benchmarking passed. Missing artifacts or
handled preparation/training errors mark failure and raise an error to the caller.
Input validation errors before run creation have no run-scoped log.

For abrupt shutdown, first inspect metadata, process state, log tail, and any
checkpoints before deciding how to recover. Automatic power-loss recovery is not
implemented; do not assume a stale `training` status is proof of an active job.
Never delete prior runs to force a retry.

## Cached/offline CPU execution

The acceptance recovery used cached assets and a small CPU thread count. For
cached-only work on Windows PowerShell, these process environment settings can
avoid unnecessary Hub probes:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
```

They are invocation settings, not additional pipeline YAML parameters. Do not
enable offline mode on a new host before obtaining required assets. Keep tokens
in a local credential mechanism/environment, never in configs committed to Git.

## Post-training inference

Acceptance proved loading `AutoModelForCausalLM` for the recorded base model,
`AutoTokenizer` from the run's model directory, and `PeftModel.from_pretrained`
for its adapter, followed by chat-prompt generation. See [acceptance](acceptance_report.md).
The pipeline does not automatically run inference after every execution or merge
the adapter into a standalone model. No serving/evaluation behavior is changed here.

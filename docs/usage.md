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
  serving/endpoint_manifest.json  # only when serving is enabled and ready
  serving/health_check.json
  serving/serving.log
  gateway/gateway_manifest.json   # only when gateway is enabled and ready
  gateway/registration.json
  gateway/health_check.json
  gateway/gateway.log
  environment.json
  metadata.json
```

Hub snapshots are source/options descriptors. The normalized selected records
are persisted. Metadata schema v2 records revision/dataset/environment/container/
metric provenance and the output relationship. LoRA's model directory contains
adapter configuration/weights and saved tokenizer files; retain the matching
base model identifier/revision because the adapter is not standalone.
With optional phases enabled, the provider-neutral serving/gateway manifests are
machine-readable endpoint handoffs. See [serving](serving.md),
[gateway](gateway.md), and the
[sanitized complete config](../configs/serving_gateway_example.yaml).

## Optional operational modes

- Train only: omit both optional sections or set each `enabled: false`.
- Train + vLLM: enable serving. Success requires verified training artifacts,
  direct discovery of both aliases, and direct inference through both aliases.
- Train + vLLM + LiteLLM: also enable gateway. Success additionally requires
  conflict-free registration, gateway discovery, and gateway inference through
  both aliases.

Gateway cannot be enabled without serving. Automatic serving cannot be enabled
for full fine-tuning in this initial implementation. Set `LITELLM_BASE_URL` and
`LITELLM_API_KEY` in the process environment before using the example; do not
replace the API-key reference with a literal value.

Remote LiteLLM URLs require HTTPS unless the operator explicitly sets
`gateway.allow_insecure_http: true`. That opt-in sends authenticated traffic
without TLS, produces a warning, and should be used only on a trusted private/local
network. It does not imply `allow_local_backend`; backend reachability remains a
separate decision.

## Understand completion and failure

Check `metadata.json`, `environment.json`, logs, manifests, and the model directory
together. In train-only mode, `success` retains its original meaning: training
exited zero and required artifacts passed validation. With optional phases,
top-level `success` also requires every requested endpoint check. Additive phase
status distinguishes `training: success` plus `serving`/`gateway: failed` and
verified training artifacts remain intact. None of these states proves model
quality improvement. Missing artifacts or handled phase errors mark top-level
failure and raise an error to the caller.
Input validation errors before run creation have no run-scoped log.

For abrupt shutdown, first inspect metadata, process state, log tail, and any
checkpoints before deciding how to recover. Automatic power-loss recovery is not
implemented; do not assume a stale `training` status is proof of an active job.
Never delete prior runs to force a retry.

Terminal output defaults to concise Trainer-phase progress and per-step metrics.
Select `quiet`, `concise`, or `full` with
`observability.console_verbosity`. This only changes presentation:
`logs/train.log` retains pipeline lifecycle records and the complete unfiltered
backend stdout/stderr stream. See [console observability](observability.md).

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
Train-only execution does not automatically run inference or merge the adapter.
Enabled serving/gateway phases do perform their minimal endpoint inference checks;
these establish route mechanics, not quality. Adapter merging is not implemented.

## One-command fully containerized deployment

```bash
export LITELLM_BASE_URL=https://litellm.example.com
export LITELLM_API_KEY='runtime-value'
sh scripts/run_pipeline.sh
```

The default `PIPELINE_DOCKER_BUILD_NETWORK=auto` first uses normal Docker build
networking. Only a recognized DNS/network-resolution failure causes one retry
with host build networking. Set `default` to forbid fallback or `host` to request
host build networking immediately. The option affects builds only; host mode
reduces build-network isolation and does not alter runtime container networking.

If host build networking also fails specifically at DNS resolution, `auto` makes
one last attempt using validated non-loopback resolvers discovered from
`/etc/resolv.conf`. An operator can select that attempt's resolvers with, for
example, `PIPELINE_DOCKER_BUILD_DNS='192.0.2.53 2001:db8::53'`. No resolver is
invented, and neither host DNS files nor Docker daemon configuration are edited.

Local datasets inside the checkout use the default dataset root. To use an
external directory, export `PIPELINE_DATASETS_DIR` as its absolute host path and
set `dataset.path` relative to that root. The root is mounted read-only at
`/workspace/datasets`; paths outside it are rejected.

Pass a repository-local config path as the only positional argument or add
`--rebuild`. The launcher needs no host Python. It validates Git/Docker and daemon
access, canonicalizes/creates mount roots, detects relevant tracked and untracked
source changes, builds or reuses both images, and returns the controller exit
status. Inspect a completed deployment without Python using:

```bash
sh scripts/pipeline_status.sh <run-id>
```

The status command is read-only. No automatic undeploy command is provided.
`.env.example` is a variable-name template only; the launcher deliberately does
not source shell files. Export values through the operator's approved runtime
secret mechanism.

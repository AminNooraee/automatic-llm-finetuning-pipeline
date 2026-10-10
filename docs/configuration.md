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

## Optional serving

Omitting `serving`, or setting `serving.enabled: false`, leaves the train-only
workflow unchanged. The initial automatic path accepts only `training.method:
lora`, `backend: vllm`, and `runtime: docker`; enabled serving with full training
fails during preparation.

```yaml
serving:
  enabled: true
  backend: vllm
  runtime: docker
  bind_host: 0.0.0.0
  advertise_host: model-server.example
  port: 8101
  base_model_name: example-base
  fine_tuned_model_name: example-finetuned
  container_name: auto
  vllm:
    image: vllm/vllm-openai:v0.11.0
    gpu_devices: "0"
    gpu_memory_utilization: 0.15
    max_model_len: 4096
    max_num_seqs: 2
  health_check:
    enabled: true
    timeout_seconds: 180
    interval_seconds: 2
```

`serving.port` accepts a fixed integer for backward compatibility. Unified
resource admission can instead use a bounded automatic range:

```yaml
resource_preflight:
  enabled: true
  model_parameter_estimate: 490000000
  activation_bytes_per_token: 1048576
  safety_margin_percent: 25

serving:
  port: auto
  port_range:
    start: 8101
    end: 8199
```

The parameter count and activation allowance are explicit planning inputs, not
values inferred from the model name. The estimate is conservative but is not a
guarantee. `safety_margin_percent` must be between 10 and 100. Automatic ports
require `PIPELINE_SELECTED_PORT`, which the opted-in launcher preflight produces;
users do not choose the port manually.

`bind_host` controls the local published socket. `advertise_host` must be a
reachable hostname/IP without a scheme, path, or port and becomes the host in
the endpoint handoff and LiteLLM backend URL. It cannot be a wildcard. Port is
1–65535; GPU devices are a unique comma-separated integer list; utilization is
in `(0, 1]`; lengths/counts are positive integers. Explicit aliases and Docker
names use a bounded safe-character set. `auto` values are deterministic and
traceable to the run ID. `health_check.enabled` must be `true`: automatic
serving is never declared ready without discovery and inference verification.
See [serving](serving.md).

## Optional LiteLLM gateway

Gateway registration is allowed only with enabled, successfully verified serving.
The gateway is externally managed; only `provider: litellm` and `registration.mode:
dynamic_db` are implemented.

```yaml
gateway:
  enabled: true
  provider: litellm
  allow_local_backend: false
  allow_insecure_http: false
  base_url: ${LITELLM_BASE_URL}
  api_key: ${LITELLM_API_KEY}
  registration:
    mode: dynamic_db
    base_model_name: example-base
    fine_tuned_model_name: example-finetuned
  timeout_seconds: 60
  health_check:
    enabled: true
    verify_models: true
    verify_inference: true
```

## Full deployment fields

The additive `orchestration` section is ignored by legacy train-only APIs and is
required by the one-command controller:

```yaml
orchestration:
  enabled: true
  training:
    gpu_devices: "0"
  cleanup_on_failure: true

serving:
  restart_policy: unless-stopped  # no, on-failure, or unless-stopped
```

Training and serving GPU lists are independent. `cleanup_on_failure` permits
rollback only for exact execution-owned resources; it never frees ports, kills
GPU processes, replaces containers, clears cache, or deletes unrelated gateway
models. Host roots can be set with `PIPELINE_RUNS_DIR`,
`PIPELINE_HF_CACHE_DIR`, and `PIPELINE_STATE_DIR`. Gateway URL and credentials
remain `${LITELLM_BASE_URL}` and `${LITELLM_API_KEY}` references.

`PIPELINE_DATASETS_DIR` selects the read-only local dataset root (the checkout is
the default). Local absolute dataset paths must stay beneath it; external-root
paths may be written relative to that root. The controller translates the path
into `/workspace/datasets` in a redacted worker-only config. Remote HuggingFace
dataset identifiers remain unchanged. In orchestration mode the launcher root
variables determine shared storage; legacy `output.runs_path` remains unchanged
for direct Python execution.

`PIPELINE_DOCKER_BUILD_NETWORK` accepts only `host-dns` and defaults to that
mode when unset. Any other explicit value fails before Docker access. Controller
and trainer image builds always use host networking plus a validated DNS context
with the corresponding `.host-dns` Dockerfiles. This does not affect container
runtime networking.

By default the launcher takes valid, ordered, unique, non-loopback `nameserver`
IP literals from the host's `/etc/resolv.conf`. Set `PIPELINE_DOCKER_BUILD_DNS`
to an explicit comma- or space-separated IPv4/IPv6 list to use that validated
list instead. Malformed values, hostnames, shell text, and loopback/stub
addresses are rejected. No public DNS service is inferred or appended. The
launcher generates a private temporary context outside the repository containing
only normalized `nameserver` lines and bind-mounts it over `/etc/resolv.conf`
for networked build steps. It cleans the context on success, failure, and
signals. It does not modify the host resolver file or Docker daemon.

The resolver list is supplied only to image builds. It is not part of source
identity, an OCI label, runtime configuration, Docker daemon configuration, or
host resolver configuration.

Runtime container DNS is configured separately. By default the launcher reuses
the same resolver discovery and validation rules against the host's
`/etc/resolv.conf`; `PIPELINE_DOCKER_RUNTIME_DNS` may instead provide an explicit
comma- or space-separated list. The normalized, non-loopback IP literals are
passed into the controller and applied as per-container Docker `--dns` arguments
to both the owned trainer and owned vLLM container. The selected addresses are
reported as non-secret operational information. This does not edit Docker daemon
configuration, create a network, or add host networking to either container.

`api_key` must be an exact `${ENVIRONMENT_VARIABLE}` reference; literal secrets
are rejected. `base_url` is normalized to `/v1`. HTTPS is accepted by default,
as is HTTP on an explicit loopback development URL. Remote HTTP is rejected
unless the operator deliberately sets `allow_insecure_http: true`; this transmits
the gateway credential without TLS, is appropriate only on a trusted private/local
network, and emits a non-secret warning. The project never enables it automatically.
URLs containing credentials, query strings, or fragments are rejected.

`allow_insecure_http` controls Project #1's authenticated connection to LiteLLM.
It is separate from `allow_local_backend`, which controls whether LiteLLM may be
given a loopback model-server URL. Gateway mode rejects a loopback
`serving.advertise_host` unless `allow_local_backend: true` explicitly states that
LiteLLM is co-located for local development. Resolved credentials are held only
in memory. Input/
resolved config snapshots preserve environment references and redact literal
sensitive values. See [gateway registration](gateway.md) and the complete
[sanitized example](../configs/serving_gateway_example.yaml).

Trusted-LAN HTTP example:

```yaml
gateway:
  enabled: true
  provider: litellm
  base_url: http://192.168.10.20:4000
  api_key: ${LITELLM_API_KEY}
  allow_insecure_http: true
```

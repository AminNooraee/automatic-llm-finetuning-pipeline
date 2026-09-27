# Optional vLLM serving

Serving is an optional post-training phase. Missing configuration and
`serving.enabled: false` retain the existing train-only behavior. The initial
qualified code path is deliberately narrow: LoRA training, vLLM, local Docker,
one vLLM process, one base model load, and one static adapter.

## Lifecycle and readiness

After training exits successfully and the LoRA adapter files pass artifact
validation, the serving manager:

1. Uses the exact recorded base model and resolved revision when available.
2. Uses the verified current-run model directory as the adapter mount.
3. Checks Docker availability, the requested host port, and container-name conflict.
4. Starts one detached container labeled with the current run ID. It never uses
   stop, remove, restart, kill, replacement, daemon configuration, image deletion,
   volume deletion, or network deletion operations.
5. Passes `--served-model-name` for the base alias and `--enable-lora` plus a
   static `--lora-modules` mapping for the fine-tuned alias. The verified
   adapter rank is passed as `--max-lora-rank`.
6. Polls `/v1/models`, requires both aliases, and sends a minimal non-streaming
   Chat Completions request to each alias.

A running container is not readiness evidence. The phase reaches `ready` only
after discovery and both inference calls pass. Port/name conflicts fail without
modifying the existing resource. The implementation does not automatically clean
up a failed launch because ownership and operator intent must remain explicit.

## Output

Enabled serving creates:

```text
serving/
  endpoint_manifest.json
  health_check.json
  operation.json
  serving.log
```

The endpoint manifest is provider-neutral and includes only status, API type,
advertised `/v1` base URL, and the base/fine-tuned names. It contains no adapter
path, Docker internals, or credential. With no gateway, metadata points
`output.endpoint_handoff` to this manifest.

`operation.json` is written immediately after a successful detached launch. It
records the run ID, container name and ID, ownership label, and `starting`,
`ready`, or `failed` state. A readiness failure does not remove the run-owned
container. Adapter validation parses JSON, requires a positive rank, and
compares the case-sensitive Hugging Face base identifier after trimming. It
accepts canonical `https://huggingface.co/<namespace>/<repo>/tree/<revision>`
and optional `@revision` spellings without guessing from local cache paths.

## Scope and validation

The command builder, conflict behavior, timeout/model checks, both inference
paths, lifecycle metadata, manifests, and logging are covered with mocks and do
not require Docker, a GPU, vLLM, or network access. This implementation has not
yet been accepted against a real vLLM v0.11.0 GPU container. Full-fine-tuned
model serving, QLoRA, other serving backends, remote Docker/SSH, Kubernetes, and
multi-node scheduling are not implemented.

## Full deployment lifecycle

The full profile sets `restart_policy: unless-stopped`; legacy configs retain the
old `no` default. The generated `docker run` uses argv tokens (not a shell), one
base model argument, `--served-model-name` for its alias, `--enable-lora`, one
`name=/adapters/fine-tuned` mapping, and the rank parsed from the actual
`adapter_config.json` as `--max-lora-rank`. A resolved model revision is supplied
when available. The complete HuggingFace cache and adapter directory are separate
bind mounts, so spaces remain part of one argv value.

Managed, run, and execution labels are written on creation. Port/name conflicts
fail without replacement. Failed-container removal first compares the exact ID
and all ownership labels. Successful vLLM is never removed by the controller and
continues after trainer, controller, and launcher exit.

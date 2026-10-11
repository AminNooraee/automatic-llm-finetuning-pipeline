# GPU resource preflight

`scripts/gpu_preflight.sh` is a read-only admission guard for GPU workers. It
checks host NVIDIA access, Docker daemon access, Docker's NVIDIA runtime, GPU
availability, and free VRAM before an expensive training command is started. It
does not stop, reset, reconfigure, or otherwise modify workloads or services.

For opted-in configurations, `scripts/run_pipeline.sh` now runs a unified
resource preflight immediately before the controller that creates the training
container. Legacy direct use of the GPU-only guard remains supported.

## Configuration

In legacy mode, `GPU_MIN_FREE_MIB` is required and must be a non-negative integer
in MiB. No default capacity policy is embedded in the GPU-only interface.

In unified mode, the controller image validates the pipeline YAML and derives an
estimated training requirement from `resource_preflight` plus the validated
training settings. The policy requires an explicit `model_parameter_estimate`;
the implementation never infers parameter count from a model name. It accounts
for training method, precision, batch size, cutoff length, gradient
checkpointing, and an explicit activation-bytes-per-token planning assumption.
It then applies the configured safety margin. Phase 1 uses a 25% margin.

The catalog is an offline architecture metadata snapshot. Job preparation validates
the selected entry and copies its exact architecture into generated `finetune.yaml`.
Runtime admission uses that generated snapshot and makes no Hugging Face network
request. Training and serving VRAM are separate conservative planning estimates,
not guarantees. Serving uses parameter count and precision for weights, a 2%
adapter allowance, and a KV cache allowance based on layers, KV heads, head
dimension, model length, and maximum concurrent sequences. It intentionally does
not reduce the KV estimate for sliding window attention. A 2048 MiB runtime
allowance and 15% serving safety margin complete the policy.

Each GPU must have at least max(training estimate, serving estimate) free MiB.
After selecting the physical GPU, the launcher calculates
`ceil(serving_estimate_mib * 10000 / selected_gpu_total_mib)` basis points.
The 1000 basis point minimum is a floor; the 9000 basis point maximum is a hard
admission boundary. The controller converts the admitted basis points to a
fraction and passes it to vLLM. Oversized single GPU jobs fail before training.
Tensor parallel serving is outside the current qualified path.

`GPU_DEVICE` is optional and defaults to `auto`. In unified mode, `auto` filters
out GPUs that cannot meet their estimated requirement, then selects the eligible
GPU with the greatest free VRAM. A non-negative numeric index selects that GPU
explicitly. Failure diagnostics include every detected GPU and its rejection
reason. This is a point-in-time check; it does not reserve GPU memory, so the
scheduler should prevent competing jobs from racing admission.

Example:

```sh
GPU_MIN_FREE_MIB="$REQUIRED_MIB" GPU_DEVICE=auto sh scripts/gpu_preflight.sh
```

Set `GPU_PREFLIGHT_ENV_FILE` to request a shell-compatible result file:

```sh
GPU_MIN_FREE_MIB="$REQUIRED_MIB" \
GPU_PREFLIGHT_ENV_FILE="$PWD/gpu-preflight.env" \
sh scripts/gpu_preflight.sh
. "$PWD/gpu-preflight.env"
```

The file is atomically replaced with mode `600` and contains the selected numeric
index as `PIPELINE_SELECTED_GPU` and `CUDA_VISIBLE_DEVICES`. Unified mode also
records `PIPELINE_SELECTED_GPU_TOTAL_MIB`,
`PIPELINE_ESTIMATED_TRAINING_MIB`, `PIPELINE_ESTIMATED_SERVING_MIB`,
`PIPELINE_ESTIMATED_REQUIRED_VRAM_MIB`, and
`PIPELINE_VLLM_GPU_MEMORY_UTILIZATION_BPS`. Its parent directory must already
exist. Place it in a job-private directory, and treat the selection as local to
the worker on which the check ran.

## Automatic serving port

An opted-in configuration uses `serving.port: auto` with a bounded
`serving.port_range`. After GPU admission, a read-only host-network probe selects
the first bindable port and emits `PIPELINE_SELECTED_PORT`. The serving resolver
treats that value as authoritative, so Docker publishing, endpoint manifests,
LiteLLM registration, and health checks share one base URL. The controller
rechecks the port immediately before training-container creation, and the serving
backend checks again before Docker launch. Neither check stops or changes an
existing listener. Because a probe cannot reserve a port throughout training, an
unrelated process can still win the port later; that case fails closed at serving
startup.

The test suite uses fake `nvidia-smi` and `docker` executables; it requires
neither a GPU nor a Docker daemon:

```sh
python -m unittest tests.test_gpu_preflight -v
```

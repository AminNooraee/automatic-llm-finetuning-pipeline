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

The result is deliberately reported as an estimate, not a guarantee. For each
GPU, admission takes the larger of the training estimate and the vLLM reservation
computed from that GPU's total memory and `serving.vllm.gpu_memory_utilization`.

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
records `PIPELINE_ESTIMATED_REQUIRED_VRAM_MIB`. Its parent directory must already
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

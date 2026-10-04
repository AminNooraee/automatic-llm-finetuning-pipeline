# GPU resource preflight

`scripts/gpu_preflight.sh` is a read-only admission guard for GPU workers. It
checks host NVIDIA access, Docker daemon access, Docker's NVIDIA runtime, GPU
availability, and free VRAM before an expensive training command is started. It
does not stop, reset, reconfigure, or otherwise modify workloads or services.

The guard is intentionally not invoked by `scripts/run_pipeline.sh` or GitLab CI
yet. A separately reviewed integration step should run it immediately before
training and consume its output.

## Configuration

`GPU_MIN_FREE_MIB` is required and must be a non-negative integer in MiB. No
default capacity policy is embedded in the repository.

`GPU_DEVICE` is optional and defaults to `auto`. `auto` selects the GPU reporting
the greatest amount of free VRAM at check time. A non-negative numeric index
selects that GPU explicitly. Admission fails when the selected GPU does not meet
the configured minimum. This is a point-in-time check; it does not reserve GPU
memory, so the scheduler should prevent competing jobs from racing admission.

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

The file is atomically replaced with mode `600` and contains only the selected
numeric index as `PIPELINE_SELECTED_GPU` and `CUDA_VISIBLE_DEVICES`. Its parent
directory must already exist. Place it in a job-private directory, and treat the
selection as local to the worker on which the check ran.

The test suite uses fake `nvidia-smi` and `docker` executables; it requires
neither a GPU nor a Docker daemon:

```sh
python -m unittest tests.test_gpu_preflight -v
```

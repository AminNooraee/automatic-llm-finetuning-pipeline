# Docker deployment

Docker wraps the existing pipeline: training logic, dataset/model/configuration
layers, and run/artifact validation are unchanged. The image starts the same
`python -m fine_tuning_pipeline.train_pipeline` command used locally.

## Prerequisites and installation

Install [Docker Engine on Linux](https://docs.docker.com/engine/install/) or
[Docker Desktop](https://docs.docker.com/desktop/) on Windows/macOS. Use **Linux
containers**, with enough RAM/disk for the selected model and dependency layers.
Confirm the daemon is running:

```bash
docker version
docker info
```

Builds need access to Docker Hub, Debian/Ubuntu package repositories, PyPI,
the PyTorch wheel index, and the pinned upstream Git repository. Models and
datasets are obtained at runtime, not baked into the image. Gated assets require
approved access regardless of container use.

## Design and reproducibility

| Component | CPU default | CUDA option |
| --- | --- | --- |
| Base | Digest-pinned `python:3.12.7-slim-bookworm` | Digest-pinned `nvidia/cuda:12.8.1-runtime-ubuntu24.04` |
| Python | 3.12.7 from the image | Ubuntu's Python 3.12 packages |
| Environment | `/opt/venv` | `/opt/venv` |
| PyTorch wheels | CPU index | CUDA 12.8 (`cu128`) index |
| PyTorch set | torch 2.11.0, torchvision 0.26.0, torchaudio 2.11.0 | Same matched versions, CUDA builds |

The matched wheel set follows the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/).
It is a new Linux deployment target, not a copy of the observed Windows/CPU wheel
versions or a claim of new training acceptance. The CUDA base contains runtime
libraries, not `nvcc`; no FlashAttention, QLoRA, or source-built CUDA extensions
are added. Building either variant requires **no GPU**. Imports inspect the
PyTorch build metadata; they do not assert an available CUDA device.

[The Dockerfile](../docker/Dockerfile) installs Git, certificates, native build
tools, OpenMP runtime support, and Tini. It uses the existing
[requirements.txt](../requirements.txt), including the immutable LLaMA-Factory
revision and editable project installation. The three selected PyTorch packages
are frozen into `/opt/pytorch-constraints.txt` before resolving the application
requirements, preventing silent CPU/CUDA wheel replacement. A successful build
runs the resolver for shared dependencies against the complete requirements, then
runs `pip check`, import/default-config checks, and the full regression suite.
The installed environment is recorded in `/opt/deployment-requirements.txt`.

Both base references include verified manifest digests. The Dockerfile defaults
to CPU via `RUNTIME=cpu`; `RUNTIME=cuda` selects the CUDA base. BuildKit selects the
required branch. The documented deployment target is Linux/x86-64; ARM and other
architectures are not qualified. On an ARM host, use `--platform linux/amd64`
when building and running for emulated CPU testing; do not infer GPU support.

OS packages, build-time backend helpers, and remaining transitive dependencies
are not fully hash-locked. A freeze record describes an installed image, not an
offline wheel archive. For release reproduction, retain the built image digest,
its dependency record, and approved inputs, then qualify minimal training/inference
on the actual server. Review/update pinned base images for security maintenance;
revalidate rather than silently changing their versions.

## Clone and build (CPU)

After cloning your published repository, run from its root:

```bash
docker build -f docker/Dockerfile -t automatic-llm-finetuner .
```

`-f` is necessary because the canonical Dockerfile is under `docker/`. The final
`.` is the repository build context. Root [.dockerignore](../.dockerignore)
defaults to excluding files, then admits only source, tests, build manifests,
four public YAML files, and three synthetic datasets. Existing `runs/`, models,
checkpoints, private `datasets/`, caches, virtual environments, legacy generated
outputs, temporary files, credentials, and unrelated projects are not sent to
the builder. It is separate from Git ignore rules.

The image retains the source checkout under `/app`, preserving default config
resolution. It does not rely on a copied host virtual environment, a local
`LLaMA-Factory/` checkout, or a standalone wheel containing configs.

## Run with persistent mounts

Edit `configs/config.yaml`. Its bundled public demo path works unchanged:
`../examples/datasets/alpaca_demo.json`. For your own file under the mounted
dataset directory, use these input values:

```yaml
dataset:
  path: ../datasets/my_dataset.json
  name: my_dataset
  source_format: auto
  format: auto
output:
  runs_path: ../runs
```

Keep the existing model/training sections. Local dataset paths resolve relative
to `/app/configs/config.yaml`; host absolute paths such as `E:\...` cannot be
used unchanged inside Linux containers. A HuggingFace identifier is also supported
and requires no local dataset mount. Mounting `configs/` replaces the bundled
directory: it must contain a complete `config.yaml`.

### Linux/macOS shell

Create mount directories if needed; this does not alter existing contents:

```bash
mkdir -p datasets runs
docker run --rm \
  --mount "type=bind,source=$(pwd)/configs,target=/app/configs,readonly" \
  --mount "type=bind,source=$(pwd)/datasets,target=/app/datasets,readonly" \
  --mount "type=bind,source=$(pwd)/runs,target=/app/runs" \
  --mount type=volume,source=llm-hf-cache,target=/cache/huggingface \
  automatic-llm-finetuner
```

The image already uses Tini; no extra Docker `--init` flag is needed.

### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force -Path .\datasets, .\runs | Out-Null
docker run --rm `
  --mount "type=bind,source=$($PWD.Path)/configs,target=/app/configs,readonly" `
  --mount "type=bind,source=$($PWD.Path)/datasets,target=/app/datasets,readonly" `
  --mount "type=bind,source=$($PWD.Path)/runs,target=/app/runs" `
  --mount type=volume,source=llm-hf-cache,target=/cache/huggingface `
  automatic-llm-finetuner
```

Docker Desktop must be able to share the workspace drive. The named cache volume
persists model/dataset downloads without including a host HuggingFace cache in
the image. Do not remove it if you need cached/offline execution.

| Mount | Mode | Purpose |
| --- | --- | --- |
| `/app/configs` | Read-only | User input YAML |
| `/app/datasets` | Read-only | Private/local source files |
| `/app/runs` | Read/write | Run snapshots, normalized data, logs, metadata, adapters/models |
| `/cache/huggingface` | Read/write | Optional persistent Hub cache |

Always mount `runs/` for real work: `--rm` removes an unmounted container's outputs.
Each training execution still creates a new isolated run. Check host
`runs/<run_id>/metadata.json`, `logs/train.log`, and the verified `model/` output;
container completion alone does not substitute for pipeline artifact validation.

Docker does not reliably expose its own image tag or digest inside a generic
container. A launcher that needs exact provenance should pass non-secret values:

```bash
docker run --rm --gpus all \
  -e PIPELINE_CONTAINERIZED=true \
  -e PIPELINE_CONTAINER_RUNTIME=docker \
  -e PIPELINE_CONTAINER_IMAGE=automatic-llm-finetuner:cuda \
  -e PIPELINE_CONTAINER_IMAGE_DIGEST=sha256:... \
  ...
```

Unavailable values stay null and never fail training. Common container markers
are detected, but normal host execution has no hard-coded Docker dependency.

## Non-root permissions

The container defaults to numeric UID/GID 1000. Input mounts need read/traverse
permission; `runs/` and the cache need write permission. On Linux with another
UID/GID, build for the intended execution user:

```bash
docker build -f docker/Dockerfile \
  --build-arg APP_UID="$(id -u)" --build-arg APP_GID="$(id -g)" \
  -t automatic-llm-finetuner .
```

Use non-root IDs. Existing cache volumes retain their prior ownership across
image rebuilds; provision a writable cache for the chosen user, or use another
volume name. Bind mounts retain host ownership. Do not fix permission failures
by making corpora/output directories world-writable or by changing pipeline logic.

## CUDA build and future GPU execution

```bash
docker build -f docker/Dockerfile --build-arg RUNTIME=cuda \
  -t automatic-llm-finetuner:cuda .
```

A CPU-only build server can build this image. GPU execution is a separate step:
the host needs an NVIDIA GPU, a driver compatible with CUDA 12.8, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured for Docker. A container does not install the host kernel driver.
Docker Desktop GPU support is platform-specific; Linux server deployment is the
intended GPU target. First check visibility on the GPU host:

```bash
docker run --rm --gpus all automatic-llm-finetuner:cuda \
  python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'No usable GPU visible'; print(torch.cuda.get_device_name(0))"
```

Then run with the same mounts plus GPU access (Linux shell):

```bash
docker run --rm --gpus all --shm-size=2g \
  --mount "type=bind,source=$(pwd)/configs,target=/app/configs,readonly" \
  --mount "type=bind,source=$(pwd)/datasets,target=/app/datasets,readonly" \
  --mount "type=bind,source=$(pwd)/runs,target=/app/runs" \
  --mount type=volume,source=llm-hf-cache,target=/cache/huggingface \
  automatic-llm-finetuner:cuda
```

Select supported precision/batch settings in YAML for the actual hardware; the
container does not override training parameters. The CUDA image can run without
`--gpus` on CPU, but it is larger and is not a GPU availability guarantee. The
CPU image cannot gain CUDA support merely by adding `--gpus all`.

## Named configs, authentication, and verification

Tini is the entry point; Docker command arguments replace the default Python
command. Use the same mounts when selecting another existing API recipe:

```bash
docker run --rm \
  --mount "type=bind,source=$(pwd)/configs,target=/app/configs,readonly" \
  --mount "type=bind,source=$(pwd)/runs,target=/app/runs" \
  --mount type=volume,source=llm-hf-cache,target=/cache/huggingface \
  automatic-llm-finetuner \
  python -c "from fine_tuning_pipeline.train_pipeline import execute_training; execute_training('/app/configs/huggingface_dataset_example.yaml')"
```

Supply gated-asset credentials only at runtime, for example `--env HF_TOKEN`
with `HF_TOKEN` already set in the host environment, or an approved runtime
credential mechanism. Never put tokens in a Docker build argument, copied YAML,
Dockerfile, dependency record, or published image. Hub cache volumes are private
runtime state and may contain tokens if you authenticate within them; do not publish.

After an image builds, check it without optimization:

```bash
docker run --rm automatic-llm-finetuner python -m pip check
docker run --rm automatic-llm-finetuner python -m unittest discover -s tests -v
docker run --rm automatic-llm-finetuner \
  python -c "from fine_tuning_pipeline.train_pipeline import load_config; print(load_config()['model']['name'])"
docker run --rm automatic-llm-finetuner cat /opt/deployment-requirements.txt
docker image inspect automatic-llm-finetuner --format '{{.Id}}'
```

For deployment acceptance, follow these with a minimal one-epoch LoRA run using
the persistent mounts, verify run success/artifacts, and load base + adapter for
inference on that target. Build tests do not establish model quality, GPU/H100
qualification, or full fine-tuning acceptance. Tini forwards termination signals
to the process group; abrupt termination can still leave stale run metadata,
as already documented in [usage](usage.md).

## Validation status

- Current host regression suite, including console observability and four static
  Docker checks: **92 passed**.
- Docker Hub base-image tags/digests verified; matched PyTorch versions checked
  against official installation instructions.
- Docker is unavailable on the current Windows development host. Separately, the
  CUDA Docker workflow has completed a real LoRA training run on an NVIDIA H100.
  That run does not qualify every image revision, driver, model, dataset, precision,
  or full-fine-tuning combination.

The original [Project #1 acceptance](acceptance_report.md) remains valid for its
recorded Windows/CPU workflow. Docker qualification is a separate release gate.

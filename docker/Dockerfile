# syntax=docker/dockerfile:1
# Build from the repository root: docker build -f docker/Dockerfile .
ARG RUNTIME=cpu
ARG CPU_IMAGE=python:3.12.7-slim-bookworm@sha256:60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867

FROM ${CPU_IMAGE} AS cpu-base
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git build-essential pkg-config libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*
ENV PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# CUDA runtime libraries, not a development/toolkit image. No GPU or nvcc is
# needed to build: this project uses prebuilt wheels, not FlashAttention/QLoRA.
FROM ${CUDA_IMAGE} AS cuda-base
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        ca-certificates git build-essential pkg-config libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*
ENV PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

FROM ${RUNTIME}-base AS application
ARG APP_UID=1000
ARG APP_GID=1000

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    HOME=/home/finetuner

WORKDIR /app
RUN python3 -m venv /opt/venv \
    && python -m pip install pip==24.2

# An explicit COPY allowlist complements the root build-context allowlist.
# Editable installation preserves the existing /app/configs default contract.
COPY README.md LICENSE pyproject.toml requirements.txt ./
COPY .dockerignore ./
COPY docker/ ./docker/
COPY src/fine_tuning_pipeline/ ./src/fine_tuning_pipeline/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Freeze the three selected CPU/CUDA packages before the backend resolver runs.
# Shared dependencies (e.g. fsspec) must still resolve against datasets/backend.
RUN python -m pip install --index-url "${PYTORCH_INDEX_URL}" \
        -r docker/pytorch-requirements.txt \
    && python -c "from importlib.metadata import version; print('\n'.join(name + '==' + version(name) for name in ('torch', 'torchvision', 'torchaudio')))" > /opt/pytorch-constraints.txt \
    && python -m pip install --constraint /opt/pytorch-constraints.txt \
        -r requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/deployment-requirements.txt \
    && python -c "import torch, torchvision, torchaudio, llamafactory; from fine_tuning_pipeline.train_pipeline import load_config; assert load_config()['model']['name']; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)"

# Ubuntu images may already have uid/gid 1000: reuse numeric IDs safely.
RUN if ! getent group "${APP_GID}" >/dev/null; then \
        groupadd --gid "${APP_GID}" finetuner; \
    fi \
    && if ! getent passwd "${APP_UID}" >/dev/null; then \
        useradd --uid "${APP_UID}" --gid "${APP_GID}" \
            --home-dir /home/finetuner --no-create-home finetuner; \
    fi \
    && mkdir -p /app/datasets /app/runs /cache/huggingface /home/finetuner \
    && chown -R "${APP_UID}:${APP_GID}" \
        /app/runs /cache /home/finetuner /app/tests

USER ${APP_UID}:${APP_GID}

# Mocked lifecycle/schema tests only; no training, Hub downloads or GPU required.
RUN python -m unittest discover -s tests -v

# Forward stop signals to the entire Python/training subprocess group.
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["python", "-m", "fine_tuning_pipeline.train_pipeline"]

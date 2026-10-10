#!/bin/sh
# One-command host launcher. Requires Git and Docker, never host Python.
set -eu

REBUILD=0
CONFIG_PATH=configs/full_pipeline_example.yaml
RESOURCE_PREFLIGHT=${PIPELINE_RESOURCE_PREFLIGHT:-0}
case "$RESOURCE_PREFLIGHT" in
    0|1) ;;
    *) echo "PIPELINE_RESOURCE_PREFLIGHT must be 0 or 1." >&2; exit 2 ;;
esac
DOCKER_BUILD_NETWORK=${PIPELINE_DOCKER_BUILD_NETWORK-host-dns}
if [ "$DOCKER_BUILD_NETWORK" != host-dns ]; then
    echo "PIPELINE_DOCKER_BUILD_NETWORK must be host-dns." >&2
    exit 2
fi
while [ "$#" -gt 0 ]; do
    case "$1" in
        --rebuild) REBUILD=1 ;;
        --help|-h)
            echo "Usage: $0 [--rebuild] [path/to/config.yaml]"
            exit 0
            ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) CONFIG_PATH=$1 ;;
    esac
    shift
done

SELECTED_GPU_IS_SET=0
SELECTED_PORT_IS_SET=0
if [ "${PIPELINE_SELECTED_GPU+x}" = x ]; then
    case "$PIPELINE_SELECTED_GPU" in
        ''|*[!0-9]*)
            echo "PIPELINE_SELECTED_GPU must be exactly one non-negative integer GPU index." >&2
            exit 2
            ;;
    esac
    SELECTED_GPU_IS_SET=1
fi

command -v git >/dev/null 2>&1 || { echo "Git is required." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker CLI is required." >&2; exit 1; }
docker info >/dev/null 2>&1 || {
    echo "Docker daemon is unavailable or permission was denied." >&2
    exit 1
}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)
. "$PROJECT_DIR/scripts/build_dns.sh"
EXPLICIT_BUILD_DNS=
if [ -n "${PIPELINE_DOCKER_BUILD_DNS-}" ]; then
    if EXPLICIT_BUILD_DNS=$(normalize_build_dns "$PIPELINE_DOCKER_BUILD_DNS" strict); then
        :
    else
        echo "PIPELINE_DOCKER_BUILD_DNS must contain only valid, non-loopback IPv4/IPv6 resolver addresses." >&2
        exit 2
    fi
fi
if [ -n "${PIPELINE_DOCKER_RUNTIME_DNS-}" ]; then
    if RUNTIME_DNS_RESOLVERS=$(normalize_build_dns "$PIPELINE_DOCKER_RUNTIME_DNS" strict); then
        :
    else
        echo "PIPELINE_DOCKER_RUNTIME_DNS must contain only valid, non-loopback IPv4/IPv6 resolver addresses." >&2
        exit 2
    fi
elif RUNTIME_DNS_RESOLVERS=$(discover_build_dns /etc/resolv.conf); then
    :
else
    echo "No usable non-loopback runtime DNS resolver was found in /etc/resolv.conf; set PIPELINE_DOCKER_RUNTIME_DNS explicitly." >&2
    exit 2
fi
echo "Using validated runtime DNS resolvers: $RUNTIME_DNS_RESOLVERS"
case "$CONFIG_PATH" in
    /*) CONFIG_ABS=$CONFIG_PATH ;;
    *) CONFIG_ABS=$PROJECT_DIR/$CONFIG_PATH ;;
esac
CONFIG_DIR=$(CDPATH= cd -- "$(dirname -- "$CONFIG_ABS")" 2>/dev/null && pwd -P) || {
    echo "Configuration directory does not exist." >&2; exit 1;
}
CONFIG_ABS=$CONFIG_DIR/$(basename -- "$CONFIG_ABS")
[ -f "$CONFIG_ABS" ] || { echo "Configuration file does not exist: $CONFIG_ABS" >&2; exit 1; }
case "$CONFIG_ABS" in
    "$PROJECT_DIR"/*) CONFIG_REL=${CONFIG_ABS#"$PROJECT_DIR"/} ;;
    *) echo "Configuration must be inside the repository checkout." >&2; exit 1 ;;
esac

RUNS_DIR=${PIPELINE_RUNS_DIR:-$PROJECT_DIR/runs}
CACHE_DIR=${PIPELINE_HF_CACHE_DIR:-$PROJECT_DIR/.cache/huggingface}
STATE_DIR=${PIPELINE_STATE_DIR:-$PROJECT_DIR/.pipeline-state}
DATASETS_DIR=${PIPELINE_DATASETS_DIR:-$PROJECT_DIR}
[ -d "$DATASETS_DIR" ] || { echo "Dataset root does not exist: $DATASETS_DIR" >&2; exit 1; }
mkdir -p "$RUNS_DIR" "$CACHE_DIR" "$STATE_DIR"
RUNS_DIR=$(CDPATH= cd -- "$RUNS_DIR" && pwd -P)
CACHE_DIR=$(CDPATH= cd -- "$CACHE_DIR" && pwd -P)
STATE_DIR=$(CDPATH= cd -- "$STATE_DIR" && pwd -P)
DATASETS_DIR=$(CDPATH= cd -- "$DATASETS_DIR" && pwd -P)

SOURCE_REVISION=$(git -C "$PROJECT_DIR" rev-parse HEAD)
SOURCE_IDENTITY=$SOURCE_REVISION
SOURCE_PATHS="src docker scripts configs pyproject.toml requirements.txt"
SOURCE_STATUS=$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=normal -- $SOURCE_PATHS)
if [ -n "$SOURCE_STATUS" ]; then
    DIRTY_HASH=$(
        git -C "$PROJECT_DIR" diff --binary HEAD -- $SOURCE_PATHS
        git -C "$PROJECT_DIR" ls-files --others --exclude-standard -- $SOURCE_PATHS | while IFS= read -r path; do
            printf '%s ' "$path"
            git -C "$PROJECT_DIR" hash-object -- "$path"
        done
    )
    DIRTY_HASH=$(printf '%s' "$DIRTY_HASH" | git hash-object --stdin)
    SOURCE_IDENTITY=$SOURCE_REVISION-dirty-$DIRTY_HASH
fi
TAG_ID=$(printf '%s' "$SOURCE_IDENTITY" | git hash-object --stdin | cut -c1-16)
HOST_UID=$(id -u)
HOST_GID=$(id -g)
EXECUTION_ID=exec-$(printf '%s' "$(date -u +%Y%m%dT%H%M%SZ)-$$-$TAG_ID" | git hash-object --stdin | cut -c1-24)
CONTROLLER_IMAGE=automatic-llm-finetuner-controller:$TAG_ID
TRAINING_IMAGE=automatic-llm-finetuner:$TAG_ID-cuda-u${HOST_UID}g${HOST_GID}
BUILD_LOG=
SECRET_DIR=
PREFLIGHT_DIR=
DNS_CONTEXT_DIR=

cleanup_dns_context() {
    if [ -n "$DNS_CONTEXT_DIR" ]; then
        rm -f -- "$DNS_CONTEXT_DIR/resolv.conf"
        rmdir -- "$DNS_CONTEXT_DIR" 2>/dev/null || true
        DNS_CONTEXT_DIR=
    fi
}

cleanup_temporary_files() {
    cleanup_dns_context
    if [ -n "$BUILD_LOG" ]; then
        rm -f -- "$BUILD_LOG"
    fi
    if [ -n "$SECRET_DIR" ]; then
        if [ -n "$PREFLIGHT_DIR" ]; then
            rm -f -- "$PREFLIGHT_DIR/estimate.env" "$PREFLIGHT_DIR/gpu.env" \
                "$PREFLIGHT_DIR/port.env"
            rmdir -- "$PREFLIGHT_DIR" 2>/dev/null || true
        fi
        rm -f -- "$SECRET_DIR/litellm_api_key" "$SECRET_DIR/hf_token"
        rmdir -- "$SECRET_DIR" 2>/dev/null || true
    fi
}
trap cleanup_temporary_files EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

run_docker_build_attempt() {
    build_dockerfile=$1
    build_dns_context=$2
    shift 2
    BUILD_LOG=$(mktemp "${TMPDIR:-/tmp}/automatic-llm-build.XXXXXX")
    chmod 600 "$BUILD_LOG"
    build_status=0
    docker build --network=host \
        --build-context "pipeline_dns=$build_dns_context" \
        --file "$build_dockerfile" "$@" >"$BUILD_LOG" 2>&1 || build_status=$?
    cat "$BUILD_LOG"
    rm -f -- "$BUILD_LOG"
    BUILD_LOG=
    return "$build_status"
}

prepare_dns_context() {
    if [ -n "$EXPLICIT_BUILD_DNS" ]; then
        BUILD_DNS_RESOLVERS=$EXPLICIT_BUILD_DNS
    elif BUILD_DNS_RESOLVERS=$(discover_build_dns /etc/resolv.conf); then
        :
    else
        echo "No usable non-loopback DNS resolver was found in /etc/resolv.conf." >&2
        return 2
    fi

    dns_temp_root=${TMPDIR:-/tmp}
    dns_temp_root=$(CDPATH= cd -- "$dns_temp_root" 2>/dev/null && pwd -P) || {
        echo "Temporary directory is unavailable: $dns_temp_root" >&2
        return 2
    }
    case "$dns_temp_root" in
        "$PROJECT_DIR"|"$PROJECT_DIR"/*)
            dns_temp_root=$(CDPATH= cd -- /tmp 2>/dev/null && pwd -P) || {
                echo "No temporary directory outside the repository is available." >&2
                return 2
            }
            ;;
    esac
    case "$dns_temp_root" in
        "$PROJECT_DIR"|"$PROJECT_DIR"/*)
            echo "No temporary directory outside the repository is available." >&2
            return 2
            ;;
    esac
    if ! DNS_CONTEXT_DIR=$(mktemp -d "$dns_temp_root/automatic-llm-dns.XXXXXX"); then
        echo "Could not create the temporary DNS build context." >&2
        return 2
    fi
    if ! chmod 700 "$DNS_CONTEXT_DIR"; then
        cleanup_dns_context
        echo "Could not secure the temporary DNS build context." >&2
        return 2
    fi
    if ! (
        umask 077
        : > "$DNS_CONTEXT_DIR/resolv.conf"
        for resolver in $BUILD_DNS_RESOLVERS; do
            printf 'nameserver %s\n' "$resolver" >> "$DNS_CONTEXT_DIR/resolv.conf"
        done
    ); then
        cleanup_dns_context
        echo "Could not write the temporary DNS resolver file." >&2
        return 2
    fi
    if ! chmod 644 "$DNS_CONTEXT_DIR/resolv.conf"; then
        cleanup_dns_context
        echo "Could not secure the temporary DNS resolver file." >&2
        return 2
    fi
}

run_host_dns_build() {
    build_dockerfile=$1
    shift
    prepare_dns_context || return $?
    echo "Docker build mode: host-dns"
    echo "Using validated host DNS resolvers: $BUILD_DNS_RESOLVERS"
    dns_build_status=0
    run_docker_build_attempt "$build_dockerfile" "$DNS_CONTEXT_DIR" "$@" || dns_build_status=$?
    cleanup_dns_context
    return "$dns_build_status"
}

ensure_image() {
    image=$1
    dockerfile=$2
    runtime=$3
    existing=$(docker image inspect --format '{{index .Config.Labels "fine-tuning-pipeline.source-identity"}}' "$image" 2>/dev/null || true)
    if [ "$REBUILD" -eq 0 ] && [ "$existing" = "$SOURCE_IDENTITY" ]; then
        echo "Reusing image $image"
        return
    fi
    if [ "$REBUILD" -eq 0 ] && [ -n "$existing" ]; then
        echo "Image tag $image exists with mismatched provenance; use --rebuild explicitly." >&2
        exit 1
    fi
    echo "Building image $image"
    set -- --tag "$image"
    if [ -n "$runtime" ]; then
        set -- "$@" \
            --build-arg "RUNTIME=$runtime" \
            --build-arg "APP_UID=$HOST_UID" \
            --build-arg "APP_GID=$HOST_GID"
    fi
    set -- "$@" \
        --build-arg "SOURCE_REVISION=$SOURCE_REVISION" \
        --build-arg "SOURCE_IDENTITY=$SOURCE_IDENTITY" "$PROJECT_DIR"
    run_host_dns_build "$dockerfile" "$@"
}

ensure_image "$CONTROLLER_IMAGE" "$PROJECT_DIR/docker/Dockerfile.controller.host-dns" ""
ensure_image "$TRAINING_IMAGE" "$PROJECT_DIR/docker/Dockerfile.host-dns" cuda
TRAINING_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$TRAINING_IMAGE")

SECRET_DIR=$(mktemp -d "${TMPDIR:-/tmp}/automatic-llm-secrets.XXXXXX")
if [ -n "${LITELLM_API_KEY-}" ]; then
    (umask 077 && printf '%s' "$LITELLM_API_KEY" > "$SECRET_DIR/litellm_api_key")
fi
if [ -n "${HF_TOKEN-}" ]; then
    (umask 077 && printf '%s' "$HF_TOKEN" > "$SECRET_DIR/hf_token")
fi

run_resource_preflight_helper() {
    helper_mode=$1
    helper_network=$2
    helper_output=$3
    set -- docker run --rm --read-only --network "$helper_network" \
        --security-opt no-new-privileges \
        --tmpfs /tmp:rw,noexec,nosuid,size=16m \
        --user "$HOST_UID:$HOST_GID" \
        --label fine-tuning-pipeline.managed=true \
        --label fine-tuning-pipeline.role=resource-preflight \
        --label "fine-tuning-pipeline.execution-id=$EXECUTION_ID" \
        --mount "type=bind,src=$PROJECT_DIR,dst=/workspace/project,readonly" \
        --mount "type=bind,src=$SECRET_DIR,dst=/run/pipeline-secrets,readonly" \
        --mount "type=bind,src=$PREFLIGHT_DIR,dst=/run/resource-preflight"
    if [ -n "${SERVING_ADVERTISE_HOST-}" ]; then
        set -- "$@" --env SERVING_ADVERTISE_HOST
    fi
    if [ -n "${LITELLM_BASE_URL-}" ]; then
        set -- "$@" --env LITELLM_BASE_URL
    fi
    set -- "$@" "$CONTROLLER_IMAGE" python -m fine_tuning_pipeline.resource_preflight \
        "$helper_mode" --config "/workspace/project/$CONFIG_REL" \
        --output "/run/resource-preflight/$helper_output"
    "$@"
}

if [ "$RESOURCE_PREFLIGHT" -eq 1 ]; then
    PREFLIGHT_DIR=$SECRET_DIR/resource-preflight
    mkdir "$PREFLIGHT_DIR"
    chmod 700 "$PREFLIGHT_DIR"

    run_resource_preflight_helper estimate none estimate.env
    . "$PREFLIGHT_DIR/estimate.env"
    if [ "${PIPELINE_RESOURCE_PREFLIGHT_ENABLED-0}" -ne 1 ]; then
        echo "PIPELINE_RESOURCE_PREFLIGHT=1 requires resource_preflight.enabled=true in the config." >&2
        exit 2
    fi
    export GPU_ESTIMATED_TRAINING_MIB GPU_SERVING_MEMORY_UTILIZATION_BPS
    selected_gpu_env=${GPU_PREFLIGHT_ENV_FILE:-$PREFLIGHT_DIR/gpu.env}
    GPU_PREFLIGHT_ENV_FILE=$selected_gpu_env sh "$PROJECT_DIR/scripts/gpu_preflight.sh"
    . "$selected_gpu_env"
    export PIPELINE_SELECTED_GPU
    SELECTED_GPU_IS_SET=1

    run_resource_preflight_helper select-port host port.env
    . "$PREFLIGHT_DIR/port.env"
    export PIPELINE_SELECTED_PORT
    SELECTED_PORT_IS_SET=1
fi

set -- docker run --rm --read-only --network host \
    --security-opt no-new-privileges \
    --tmpfs /tmp:rw,noexec,nosuid,size=64m \
    --label fine-tuning-pipeline.managed=true \
    --label fine-tuning-pipeline.role=controller \
    --label "fine-tuning-pipeline.execution-id=$EXECUTION_ID" \
    --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
    --mount "type=bind,src=$PROJECT_DIR,dst=/workspace/project,readonly" \
    --mount "type=bind,src=$DATASETS_DIR,dst=/workspace/datasets,readonly" \
    --mount "type=bind,src=$RUNS_DIR,dst=/workspace/runs" \
    --mount "type=bind,src=$CACHE_DIR,dst=/cache/huggingface" \
    --mount "type=bind,src=$STATE_DIR,dst=/workspace/state" \
    --mount "type=bind,src=$SECRET_DIR,dst=/run/pipeline-secrets,readonly" \
    --env "PIPELINE_EXECUTION_ID=$EXECUTION_ID" \
    --env "PIPELINE_CONFIG_RELATIVE=$CONFIG_REL" \
    --env "PIPELINE_HOST_PROJECT_DIR=$PROJECT_DIR" \
    --env "PIPELINE_HOST_DATASETS_DIR=$DATASETS_DIR" \
    --env "PIPELINE_HOST_RUNS_DIR=$RUNS_DIR" \
    --env "PIPELINE_HOST_HF_CACHE_DIR=$CACHE_DIR" \
    --env "PIPELINE_HOST_STATE_DIR=$STATE_DIR" \
    --env "PIPELINE_HOST_SECRETS_DIR=$SECRET_DIR" \
    --env "PIPELINE_TRAINING_IMAGE=$TRAINING_IMAGE" \
    --env "PIPELINE_TRAINING_IMAGE_ID=$TRAINING_IMAGE_ID" \
    --env "PIPELINE_DOCKER_RUNTIME_DNS=$RUNTIME_DNS_RESOLVERS" \
    --env "PIPELINE_SOURCE_REVISION=$SOURCE_REVISION" \
    --env "PIPELINE_SOURCE_IDENTITY=$SOURCE_IDENTITY"
set -- "$@" --env "PIPELINE_WORKER_UID=$HOST_UID" --env "PIPELINE_WORKER_GID=$HOST_GID"
if [ "$SELECTED_GPU_IS_SET" -eq 1 ]; then
    set -- "$@" --env "PIPELINE_SELECTED_GPU=$PIPELINE_SELECTED_GPU"
fi
if [ "$SELECTED_PORT_IS_SET" -eq 1 ]; then
    set -- "$@" --env "PIPELINE_SELECTED_PORT=$PIPELINE_SELECTED_PORT"
fi
if [ -n "${SERVING_ADVERTISE_HOST-}" ]; then
    set -- "$@" --env SERVING_ADVERTISE_HOST
fi
if [ -n "${LITELLM_BASE_URL-}" ]; then
    set -- "$@" --env LITELLM_BASE_URL
fi
set -- "$@" "$CONTROLLER_IMAGE"
"$@"

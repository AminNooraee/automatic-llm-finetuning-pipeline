#!/bin/sh
# One-command host launcher. Requires Git and Docker, never host Python.
set -eu

REBUILD=0
CONFIG_PATH=configs/full_pipeline_example.yaml
DOCKER_BUILD_NETWORK=${PIPELINE_DOCKER_BUILD_NETWORK:-auto}
case "$DOCKER_BUILD_NETWORK" in
    auto|default|host) ;;
    *)
        echo "PIPELINE_DOCKER_BUILD_NETWORK must be one of: auto, default, host." >&2
        exit 2
        ;;
esac
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

cleanup_temporary_files() {
    if [ -n "$BUILD_LOG" ]; then
        rm -f -- "$BUILD_LOG"
    fi
    if [ -n "$SECRET_DIR" ]; then
        rm -f -- "$SECRET_DIR/litellm_api_key" "$SECRET_DIR/hf_token"
        rmdir -- "$SECRET_DIR" 2>/dev/null || true
    fi
}
trap cleanup_temporary_files EXIT HUP INT TERM

build_failure_is_network_resolution() {
    grep -Eiq \
        'Temporary failure resolving|Could not resolve|Name or service not known|Network is unreachable|failure resolving|DNS resolution' \
        "$1"
}

run_docker_build_attempt() {
    build_network=$1
    build_dns=$2
    shift 2
    BUILD_LOG=$(mktemp "${TMPDIR:-/tmp}/automatic-llm-build.XXXXXX")
    chmod 600 "$BUILD_LOG"
    build_status=0
    if [ "$build_network" = host ] && [ -n "$build_dns" ]; then
        docker build --network=host --build-arg "PIPELINE_BUILD_DNS=$build_dns" \
            "$@" >"$BUILD_LOG" 2>&1 || build_status=$?
    elif [ "$build_network" = host ]; then
        docker build --network=host "$@" >"$BUILD_LOG" 2>&1 || build_status=$?
    else
        docker build "$@" >"$BUILD_LOG" 2>&1 || build_status=$?
    fi
    cat "$BUILD_LOG"
    BUILD_FAILURE_WAS_NETWORK=0
    if [ "$build_status" -ne 0 ] && build_failure_is_network_resolution "$BUILD_LOG"; then
        BUILD_FAILURE_WAS_NETWORK=1
    fi
    rm -f -- "$BUILD_LOG"
    BUILD_LOG=
    return "$build_status"
}

run_docker_build() {
    case "$DOCKER_BUILD_NETWORK" in
        default)
            run_docker_build_attempt default "" "$@"
            ;;
        host)
            echo "Docker build is using explicitly configured host networking."
            run_docker_build_attempt host "" "$@"
            ;;
        auto)
            if run_docker_build_attempt default "" "$@"; then
                return 0
            else
                first_status=$?
            fi
            if [ "$BUILD_FAILURE_WAS_NETWORK" -ne 1 ]; then
                return "$first_status"
            fi
            echo "Docker build failed due to a network/DNS resolution error." >&2
            echo "Retrying once with host build networking..." >&2
            if run_docker_build_attempt host "" "$@"; then
                return 0
            else
                host_status=$?
            fi
            if [ "$BUILD_FAILURE_WAS_NETWORK" -ne 1 ]; then
                return "$host_status"
            fi
            if [ -n "$EXPLICIT_BUILD_DNS" ]; then
                build_dns=$EXPLICIT_BUILD_DNS
            elif build_dns=$(discover_build_dns /etc/resolv.conf); then
                :
            else
                echo "Host build networking still cannot resolve package repositories, and no usable non-loopback resolver was found in /etc/resolv.conf." >&2
                return "$host_status"
            fi
            echo "Host build networking still cannot resolve package repositories." >&2
            echo "Retrying once with validated host DNS resolvers: $build_dns" >&2
            run_docker_build_attempt host "$build_dns" "$@"
            ;;
    esac
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
    set -- --file "$dockerfile" --tag "$image"
    if [ -n "$runtime" ]; then
        set -- "$@" \
            --build-arg "RUNTIME=$runtime" \
            --build-arg "APP_UID=$HOST_UID" \
            --build-arg "APP_GID=$HOST_GID"
    fi
    set -- "$@" \
        --build-arg "SOURCE_REVISION=$SOURCE_REVISION" \
        --build-arg "SOURCE_IDENTITY=$SOURCE_IDENTITY" "$PROJECT_DIR"
    run_docker_build "$@"
}

ensure_image "$CONTROLLER_IMAGE" "$PROJECT_DIR/docker/Dockerfile.controller" ""
ensure_image "$TRAINING_IMAGE" "$PROJECT_DIR/docker/Dockerfile" cuda
TRAINING_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$TRAINING_IMAGE")

SECRET_DIR=$(mktemp -d "${TMPDIR:-/tmp}/automatic-llm-secrets.XXXXXX")
if [ -n "${LITELLM_API_KEY-}" ]; then
    (umask 077 && printf '%s' "$LITELLM_API_KEY" > "$SECRET_DIR/litellm_api_key")
fi
if [ -n "${HF_TOKEN-}" ]; then
    (umask 077 && printf '%s' "$HF_TOKEN" > "$SECRET_DIR/hf_token")
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
    --env "PIPELINE_SOURCE_REVISION=$SOURCE_REVISION" \
    --env "PIPELINE_SOURCE_IDENTITY=$SOURCE_IDENTITY"
set -- "$@" --env "PIPELINE_WORKER_UID=$HOST_UID" --env "PIPELINE_WORKER_GID=$HOST_GID"
if [ -n "${LITELLM_BASE_URL-}" ]; then
    set -- "$@" --env LITELLM_BASE_URL
fi
set -- "$@" "$CONTROLLER_IMAGE"
"$@"

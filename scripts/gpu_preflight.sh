#!/bin/sh
# Read-only GPU admission check for hosts that run the training pipeline.
set -eu

fail() {
    printf 'GPU preflight failed: %s\n' "$1" >&2
    exit 1
}

normalize_decimal() {
    decimal=$1
    while [ "$decimal" != 0 ] && [ "${decimal#0}" != "$decimal" ]; do
        decimal=${decimal#0}
    done
    printf '%s\n' "$decimal"
}

# Return success when the first normalized, non-negative decimal is less than
# the second. Comparing digit-by-digit avoids shell integer overflow.
decimal_is_less() {
    left=$1
    right=$2
    [ "${#left}" -lt "${#right}" ] && return 0
    [ "${#left}" -gt "${#right}" ] && return 1
    while [ -n "$left" ]; do
        left_digit=${left%"${left#?}"}
        right_digit=${right%"${right#?}"}
        [ "$left_digit" -lt "$right_digit" ] && return 0
        [ "$left_digit" -gt "$right_digit" ] && return 1
        left=${left#?}
        right=${right#?}
    done
    return 1
}

if [ "${GPU_MIN_FREE_MIB+x}" != x ]; then
    fail "GPU_MIN_FREE_MIB must be explicitly configured."
fi
case $GPU_MIN_FREE_MIB in
    ''|*[!0-9]*) fail "GPU_MIN_FREE_MIB must be a non-negative integer." ;;
esac
required_mib=$(normalize_decimal "$GPU_MIN_FREE_MIB")

gpu_device=${GPU_DEVICE-auto}
case $gpu_device in
    auto) ;;
    ''|*[!0-9]*) fail "GPU_DEVICE must be 'auto' or a non-negative numeric GPU index." ;;
    *) gpu_device=$(normalize_decimal "$gpu_device") ;;
esac

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required but was not found."
command -v docker >/dev/null 2>&1 || fail "Docker CLI is required but was not found."

if ! docker_runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null); then
    fail "Docker daemon is unavailable or permission was denied for the current user."
fi
case $docker_runtimes in
    *'"nvidia":'*) ;;
    *) fail "NVIDIA container runtime is not configured in Docker." ;;
esac

if ! gpu_query=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null); then
    fail "nvidia-smi could not query the NVIDIA driver."
fi
[ -n "$gpu_query" ] || fail "No NVIDIA GPUs were detected."

detected=
selected_index=
selected_free=
requested_found=0
while IFS=', ' read -r gpu_index gpu_free extra; do
    case $gpu_index in
        ''|*[!0-9]*) fail "nvidia-smi returned an invalid GPU index." ;;
    esac
    case $gpu_free in
        ''|*[!0-9]*) fail "nvidia-smi returned invalid free-memory data." ;;
    esac
    [ -z "${extra-}" ] || fail "nvidia-smi returned an unexpected GPU record."

    gpu_index=$(normalize_decimal "$gpu_index")
    gpu_free=$(normalize_decimal "$gpu_free")
    detected="${detected}  GPU ${gpu_index}: ${gpu_free} MiB free
"

    if [ "$gpu_device" = auto ]; then
        if [ -z "$selected_index" ] || decimal_is_less "$selected_free" "$gpu_free"; then
            selected_index=$gpu_index
            selected_free=$gpu_free
        fi
    elif [ "$gpu_index" = "$gpu_device" ]; then
        requested_found=1
        selected_index=$gpu_index
        selected_free=$gpu_free
    fi
done <<EOF
$gpu_query
EOF

if [ "$gpu_device" != auto ] && [ "$requested_found" -ne 1 ]; then
    fail "Requested GPU index $gpu_device does not exist."
fi
[ -n "$selected_index" ] || fail "No NVIDIA GPUs were detected."
if decimal_is_less "$selected_free" "$required_mib"; then
    fail "GPU $selected_index has $selected_free MiB free; $required_mib MiB is required."
fi

preflight_temp=
cleanup_preflight_temp() {
    if [ -n "$preflight_temp" ]; then
        rm -f "$preflight_temp"
    fi
}
trap cleanup_preflight_temp EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "${GPU_PREFLIGHT_ENV_FILE+x}" = x ]; then
    env_file=$GPU_PREFLIGHT_ENV_FILE
    [ -n "$env_file" ] || fail "GPU_PREFLIGHT_ENV_FILE must not be empty."
    [ ! -d "$env_file" ] || fail "GPU_PREFLIGHT_ENV_FILE must name a file, not a directory."
    case $env_file in
        */*) env_dir=${env_file%/*}; [ -n "$env_dir" ] || env_dir=/ ;;
        *) env_dir=.; env_file=./$env_file ;;
    esac
    [ -d "$env_dir" ] || fail "GPU_PREFLIGHT_ENV_FILE parent directory does not exist."
    preflight_temp=$(mktemp "$env_dir/.gpu-preflight.XXXXXX") || \
        fail "Could not create the GPU preflight environment file."
    chmod 600 "$preflight_temp" || fail "Could not secure the GPU preflight environment file."
    if ! (umask 077 && printf 'PIPELINE_SELECTED_GPU=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
        "$selected_index" "$selected_index" > "$preflight_temp"); then
        fail "Could not write the GPU preflight environment file."
    fi
    mv -f "$preflight_temp" "$env_file" || fail "Could not publish the GPU preflight environment file."
    preflight_temp=
    chmod 600 "$env_file" || fail "Could not secure the published GPU preflight environment file."
fi

printf 'Detected GPUs:\n%s' "$detected"
printf 'Selected GPU: %s\n' "$selected_index"
printf 'Free VRAM: %s MiB\n' "$selected_free"
printf 'Required VRAM: %s MiB\n' "$required_mib"
printf 'GPU preflight passed.\n'

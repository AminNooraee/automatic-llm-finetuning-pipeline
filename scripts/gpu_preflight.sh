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

estimate_mode=0
serving_utilization_bps=0
if [ "${GPU_ESTIMATED_TRAINING_MIB+x}" = x ]; then
    case $GPU_ESTIMATED_TRAINING_MIB in
        ''|*[!0-9]*) fail "GPU_ESTIMATED_TRAINING_MIB must be a non-negative integer." ;;
    esac
    case ${GPU_SERVING_MEMORY_UTILIZATION_BPS-} in
        ''|*[!0-9]*) fail "GPU_SERVING_MEMORY_UTILIZATION_BPS must be an integer from 1 to 10000." ;;
    esac
    required_mib=$(normalize_decimal "$GPU_ESTIMATED_TRAINING_MIB")
    serving_utilization_bps=$(normalize_decimal "$GPU_SERVING_MEMORY_UTILIZATION_BPS")
    if [ "$serving_utilization_bps" = 0 ] || decimal_is_less 10000 "$serving_utilization_bps"; then
        fail "GPU_SERVING_MEMORY_UTILIZATION_BPS must be an integer from 1 to 10000."
    fi
    estimate_mode=1
else
    if [ "${GPU_MIN_FREE_MIB+x}" != x ]; then
        fail "GPU_MIN_FREE_MIB must be explicitly configured."
    fi
    case $GPU_MIN_FREE_MIB in
        ''|*[!0-9]*) fail "GPU_MIN_FREE_MIB must be a non-negative integer." ;;
    esac
    required_mib=$(normalize_decimal "$GPU_MIN_FREE_MIB")
fi

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

if [ "$estimate_mode" -eq 1 ]; then
    query_fields=index,memory.free,memory.total
else
    query_fields=index,memory.free
fi
if ! gpu_query=$(nvidia-smi --query-gpu="$query_fields" --format=csv,noheader,nounits 2>/dev/null); then
    fail "nvidia-smi could not query the NVIDIA driver."
fi
[ -n "$gpu_query" ] || fail "No NVIDIA GPUs were detected."

detected=
selected_index=
selected_free=
selected_required=
requested_found=0
rejections=
while IFS=', ' read -r gpu_index gpu_free gpu_total extra; do
    case $gpu_index in
        ''|*[!0-9]*) fail "nvidia-smi returned an invalid GPU index." ;;
    esac
    case $gpu_free in
        ''|*[!0-9]*) fail "nvidia-smi returned invalid free-memory data." ;;
    esac
    if [ "$estimate_mode" -eq 1 ]; then
        case $gpu_total in
            ''|*[!0-9]*) fail "nvidia-smi returned invalid total-memory data." ;;
        esac
    else
        [ -z "${gpu_total-}" ] || fail "nvidia-smi returned an unexpected GPU record."
    fi
    [ -z "${extra-}" ] || fail "nvidia-smi returned an unexpected GPU record."

    gpu_index=$(normalize_decimal "$gpu_index")
    gpu_free=$(normalize_decimal "$gpu_free")
    gpu_required=$required_mib
    if [ "$estimate_mode" -eq 1 ]; then
        gpu_total=$(normalize_decimal "$gpu_total")
        serving_required=$(( (gpu_total * serving_utilization_bps + 9999) / 10000 ))
        if decimal_is_less "$gpu_required" "$serving_required"; then
            gpu_required=$serving_required
        fi
        detected="${detected}  GPU ${gpu_index}: ${gpu_free} MiB free / ${gpu_total} MiB total
"
    else
        detected="${detected}  GPU ${gpu_index}: ${gpu_free} MiB free
"
    fi
    eligible=1
    if decimal_is_less "$gpu_free" "$gpu_required"; then
        if [ "$estimate_mode" -eq 1 ]; then
            eligible=0
        fi
        rejections="${rejections}  GPU ${gpu_index} rejected: ${gpu_free} MiB free; ${gpu_required} MiB estimated required
"
    fi

    if [ "$gpu_device" = auto ]; then
        if [ "$eligible" -eq 1 ] && { [ -z "$selected_index" ] || decimal_is_less "$selected_free" "$gpu_free"; }; then
            selected_index=$gpu_index
            selected_free=$gpu_free
            selected_required=$gpu_required
        fi
    elif [ "$gpu_index" = "$gpu_device" ]; then
        requested_found=1
        selected_index=$gpu_index
        selected_free=$gpu_free
        selected_required=$gpu_required
    fi
done <<EOF
$gpu_query
EOF

if [ "$gpu_device" != auto ] && [ "$requested_found" -ne 1 ]; then
    fail "Requested GPU index $gpu_device does not exist."
fi
[ -n "$selected_index" ] || fail "No GPU satisfies the estimated training/serving VRAM requirements.
Detected GPUs:
${detected}Rejections:
${rejections}"
if decimal_is_less "$selected_free" "$selected_required"; then
    fail "GPU $selected_index has $selected_free MiB free; $selected_required MiB is required.
Detected GPUs:
${detected}Rejections:
${rejections}"
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
    if ! (umask 077 && {
        printf 'PIPELINE_SELECTED_GPU=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
            "$selected_index" "$selected_index"
        if [ "$estimate_mode" -eq 1 ]; then
            printf 'PIPELINE_ESTIMATED_REQUIRED_VRAM_MIB=%s\n' "$selected_required"
        fi
    } > "$preflight_temp"); then
        fail "Could not write the GPU preflight environment file."
    fi
    mv -f "$preflight_temp" "$env_file" || fail "Could not publish the GPU preflight environment file."
    preflight_temp=
    chmod 600 "$env_file" || fail "Could not secure the published GPU preflight environment file."
fi

printf 'Detected GPUs:\n%s' "$detected"
printf 'Selected GPU: %s\n' "$selected_index"
printf 'Free VRAM: %s MiB\n' "$selected_free"
if [ "$estimate_mode" -eq 1 ]; then
    printf 'Estimated required VRAM: %s MiB (estimate, not guarantee)\n' "$selected_required"
    printf 'Training estimate before per-GPU serving reservation: %s MiB\n' "$required_mib"
else
    printf 'Required VRAM: %s MiB\n' "$required_mib"
fi
printf 'GPU preflight passed.\n'

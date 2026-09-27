#!/bin/sh
set -eu
[ "$#" -eq 1 ] || { echo "Usage: $0 <run-id>" >&2; exit 2; }
case "$1" in *[!A-Za-z0-9_.-]*|""|.*) echo "Invalid run ID" >&2; exit 2 ;; esac
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
RUNS_DIR=${PIPELINE_RUNS_DIR:-$SCRIPT_DIR/../runs}
MANIFEST=$RUNS_DIR/$1/deployment_manifest.json
[ -f "$MANIFEST" ] || { echo "Deployment manifest not found: $MANIFEST" >&2; exit 1; }
cat "$MANIFEST"

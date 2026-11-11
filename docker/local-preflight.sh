#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
case "$mode" in rover|jepa|data) ;; *) echo 'Usage: local-preflight.sh rover|jepa|data [IMAGE]' >&2; exit 2 ;; esac
image="${2:-samkash/mars-runtime:uv3}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$project_dir/docker/check-image-size.sh" "$image"
output_dir="$project_dir/runs/preflight/$mode"
mkdir -p "$output_dir"
# Anonymous runtime storage is removed with the container; reports stay on disk.
docker run --rm --platform linux/amd64 \
    --mount type=volume,destination=/runtime \
    --mount "type=bind,source=$output_dir,destination=/data" \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e OMNI_KIT_ALLOW_ROOT=1 \
    "$image" /opt/cluster/preflight-run.sh "$mode"

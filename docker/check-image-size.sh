#!/usr/bin/env bash
set -euo pipefail
image="${1:?Usage: check-image-size.sh IMAGE}"
size="$(docker image inspect "$image" --format '{{.Size}}')"
if [[ ! "$size" =~ ^[0-9]+$ ]] || (( size >= 1200000000 )); then
    echo "Image must be less than 1.2 GB (1,200,000,000 bytes); got: $size" >&2
    exit 1
fi
echo "$image: $size bytes (below 1.2 GB, uncompressed)"

#!/usr/bin/env bash
set -euo pipefail
test -f /runtime/ready-jepa
run_dir="/data/runs/jepa/${HOSTNAME}"
mkdir -p "$run_dir" "$JEPAWM_DSET" "$JEPAWM_CKPT" "$JEPAWM_OSSCKPT" "$TORCH_HOME" "$HF_HOME"
exec > >(tee "$run_dir/console.log") 2>&1
cp /opt/upstream.json "$run_dir/"
printf '%s\n' "$@" > "$run_dir/arguments.txt"
cd /runtime/jepa-wms
python setup_macros.py
nvidia-smi
"$@"
touch "$run_dir/SUCCESS"

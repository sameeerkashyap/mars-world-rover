#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
test -f /runtime/ready-rover-jepa
run_dir="/data/runs/rover-jepa-final/$HOSTNAME"
mkdir -p "$run_dir"
exec > >(tee "$run_dir/console.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"' EXIT
data=(/data/datasets/rover-jepa-collection/seed-20-mars/*/rover_seed_20.hdf5
      /data/datasets/rover-jepa-collection/seed-21-debug/*/rover_seed_21.hdf5)
test "${#data[@]}" = 2
python -u /opt/train/evaluate_rover_jepa.py --run "$1" --data "${data[@]}" --output "$run_dir/evaluation"

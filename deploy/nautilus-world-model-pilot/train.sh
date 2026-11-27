#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/opt/train
test -f /runtime/ready-rover-pilot
run_dir="/data/runs/rover-world-model-pilot/$TRAINING_RUN_ID"
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/console.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"' EXIT
if [[ ! -d "$run_dir/source" ]]; then
    cp -rL /opt/train "$run_dir/source"
fi
nvidia-smi > "$run_dir/hardware.txt"
if [[ ! -f "$run_dir/split.json" ]]; then
    python /opt/train/prepare_split.py "$run_dir"
fi
mapfile -t training < "$run_dir/train-files.txt"
mapfile -t validation < "$run_dir/validation-files.txt"
test "${#training[@]}" = 9
test "${#validation[@]}" = 1
baseline=/data/runs/rover-jepa-scale/mars-rover-scale-20261120-v1
baseline_model=("$baseline"/baseline/*/model.pt)
test "${#baseline_model[@]}" = 1
test -s "${baseline_model[0]}"
resume=()
if [[ -d "$run_dir/training" ]]; then
    resume=(--resume)
fi
python -u -m scripts.train_world_model_pilot \
    --data "${training[@]}" --validation-data "${validation[@]}" \
    --encoder-source /runtime/dinov2 --upstream /opt/train/upstream \
    --motion-baseline "${baseline_model[0]}" --cache-dir /data/cache/rover-jepa-features \
    --residual-checkpoint /data/runs/rover-jepa/mars-rover-jepa-train-v12-dfj2t/training/best.pt \
    --output "$run_dir/training" "${resume[@]}" "$@"
test -f "$run_dir/training/FINISHED"
touch "$run_dir/FINISHED"

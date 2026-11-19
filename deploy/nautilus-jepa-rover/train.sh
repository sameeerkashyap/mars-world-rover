#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
test -f /runtime/ready-rover-jepa
run_dir="/data/runs/rover-jepa/$HOSTNAME"
mkdir -p "$run_dir"
exec > >(tee "$run_dir/console.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"' EXIT
cp -rL /opt/train "$run_dir/source"
export PYTHONPATH=/opt/train
nvidia-smi > "$run_dir/hardware.txt"
python /opt/train/prepare_split.py "$run_dir"
mapfile -t training < "$run_dir/train-files.txt"
mapfile -t validation < "$run_dir/validation-files.txt"
mapfile -t testing < "$run_dir/test-files.txt"
# The established baseline uses exactly the same recording split as the JEPA run.
python /opt/train/scripts/train_rover_baseline.py --data "${training[@]}" \
    --validation-data "${validation[@]}" --test-data "${testing[@]}" --output "$run_dir/baseline"
baseline_model=("$run_dir"/baseline/*/model.pt)
test "${#baseline_model[@]}" = 1
python -u /opt/train/scripts/train_rover_jepa.py \
    --data "${training[@]}" --validation-data "${validation[@]}" --test-data "${testing[@]}" \
    --encoder-source /runtime/dinov2 --predictor-source /opt/train/meta_vit.py \
    --motion-baseline "${baseline_model[0]}" \
    --output "$run_dir/training" "$@"
test -f "$run_dir/training/FINISHED"
touch "$run_dir/FINISHED"

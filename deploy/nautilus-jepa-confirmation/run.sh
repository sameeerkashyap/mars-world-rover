#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
test -f /runtime/ready-rover-forecast
run_dir="/data/runs/rover-jepa-confirmation/$HOSTNAME"
mkdir -p "$run_dir" /data/runs/rover/training
exec > >(tee "$run_dir/console.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"' EXIT
cp -rL /opt/navigation "$run_dir/source"
cd "$ROVER_LAB_PATH"
export PYTHONPATH="/opt/navigation:$ROVER_LAB_PATH${PYTHONPATH:+:$PYTHONPATH}"
export ISAAC_SIM_PATH="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("isaacsim").submodule_search_locations)))')"
nvidia-smi > "$run_dir/hardware.txt"
python tools/verify_stack.py > "$run_dir/stack.json"
for index in {0..19}; do
    terrain="test-rocks-$((601 + index))"
    output="$run_dir/fresh-$terrain"
    python -X faulthandler -u /opt/navigation/rover_navigate.py \
        --jepa-run /data/runs/rover-jepa/mars-rover-jepa-train-v12-dfj2t \
        --terrain-checkpoint /data/runs/rover-jepa-scale/mars-rover-scale-20261120-v1/terrain-cost-mars-rover-scale-20261120-v1-s9ckw/best.pt \
        --test-plan /opt/navigation/test-plan.json --phase fresh --terrain "$terrain" \
        --seeds "$((100 + index))" --output "$output" --viz none
    test -f "$output/$HOSTNAME/SUCCESS"
done
python /opt/navigation/verify_confirmation.py "$run_dir"
touch "$run_dir/SUCCESS"

#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
test -f /runtime/ready-rover
run_dir="/data/runs/rover-goal-hold/$HOSTNAME"
mkdir -p "$run_dir" /data/runs/rover/training
exec > >(tee "$run_dir/console.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"' EXIT
cp -rL /opt/navigation "$run_dir/source"
cd "$ROVER_LAB_PATH"
export PYTHONPATH="/opt/navigation:$ROVER_LAB_PATH${PYTHONPATH:+:$PYTHONPATH}"
export ISAAC_SIM_PATH="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("isaacsim").submodule_search_locations)))')"
nvidia-smi > "$run_dir/hardware.txt"
python tools/verify_stack.py > "$run_dir/stack.json"
python -X faulthandler -u /opt/navigation/rover_navigate.py \
    --replay-plan /opt/navigation/replay-plan.json --phase development \
    --terrain test-rocks-501 --seeds 30 31 32 --steps 750 --output "$run_dir/replay" --viz none
test -f "$run_dir/replay/$HOSTNAME/SUCCESS"
python /opt/navigation/verify_goal_hold.py "$run_dir/replay/$HOSTNAME"
touch "$run_dir/SUCCESS"

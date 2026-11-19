#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
test -f /runtime/ready-rover
run_dir="/data/runs/rover/${HOSTNAME}"
mkdir -p "$run_dir" /data/runs/rover/training
exec > >(tee "$run_dir/console.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"' EXIT
cp /opt/navigation/{rover_navigate.py,model.pt,run.sh} /opt/navigation/scripts/*.py "$run_dir/"
cp /opt/upstream.json "$run_dir/"
cd "$ROVER_LAB_PATH"
export PYTHONPATH="/opt/navigation:$ROVER_LAB_PATH${PYTHONPATH:+:$PYTHONPATH}"
export ISAAC_SIM_PATH="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("isaacsim").submodule_search_locations)))')"
nvidia-smi > "$run_dir/hardware.txt"
python tools/verify_stack.py > "$run_dir/stack.json"
timeout --signal=TERM --kill-after=30s 3600s python -X faulthandler -u /opt/navigation/rover_navigate.py "$@" --viz none
test -f "/data/runs/rover-navigation/${HOSTNAME}/SUCCESS"
touch "$run_dir/SUCCESS"
